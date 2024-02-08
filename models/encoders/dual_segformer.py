import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys
sys.path.append('..')
sys.path.append('...')
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import FeatureRectifyModule as FRM
from .fusion import iAFF
import math
import time
# from engine.logger import get_logger

# logger = get_logger()


class DWConv(nn.Module):
    """
    Depthwise convolution bloc: input: x with size(B N C); output size (B N C)
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous() # B N C -> B C N -> B C H W
        x = self.dwconv(x) 
        x = x.flatten(2).transpose(1, 2) # B C H W -> B N C

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        """
        MLP Block: 
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

## local_region_shape = 1 --> Full Scale Attention
class MultiScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, local_region_shape = [4, 8, 40]):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.local_region_shape = local_region_shape
        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.attn_fusion = iAFF(dim)
        # self.final_proj = nn.Linear(dim * (len(self.local_region_shape)+1), dim)
        self.final_proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def fuse_ms_attn_map(self, A, H, W):
        B, N, C = A[0].shape
        output = A[1].permute(0, 2, 1).contiguous().view(B, C, H, W)
        global_attn = A[0].permute(0, 2, 1).contiguous().view(B, C, H, W)
        #print('shapes: ', output.shape, global_attn.shape)
        for i in range(2,len(A)):
            output = self.attn_fusion(output, A[i].permute(0, 2, 1).contiguous().view(B, C, H, W))
        output = self.attn_fusion(output, global_attn)
        # print('final fused: ',output.shape)
        return output


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    """ arr.shape -> B x num_head x H x W x C """
    # create overlapping patches
    def patchify(self, arr, H, W, patch_size, overlap = False):
        if not overlap:
            #print('arr: ',arr.shape)
            arr = arr.view(arr.shape[0], arr.shape[1], H, W, arr.shape[3])
            #print('arr view: ',arr.shape)
            patches = arr.view(arr.shape[0], arr.shape[1], arr.shape[2] // patch_size, patch_size, arr.shape[3] // patch_size, patch_size, arr.shape[4])
            #print('patches: ',patches.shape)
            #B x num_head x H//ps x ps x W//ps x ps x C
            # ####print('patches shape: ', patches.shape)
            patches = patches.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
            # B x num_head x C x H//ps x W//ps x ps x ps
            #print('patches permute: ', patches.shape)
            patches = patches.view(patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3], -1, patches.shape[6])
            #print('patches reshape: ', patches.shape)
            #exit()
            return patches
        else:
            stride = patch_size//2
            arr = arr.view(arr.shape[0], arr.shape[1], H, W, arr.shape[3])
            arr = arr.permute(0, 1, 4, 2, 3)
            patches = arr.unfold(4, patch_size, stride).unfold(3, patch_size, stride).contiguous()
            ###print('patches: ',patches.shape)
            patches = patches.view(arr.shape[0], arr.shape[1], -1, patch_size, patch_size)
            ##print('patches: ',patches.shape)
            #exit()
            return patches


    def attention(self, q, k, v):
        #####print(self.scale)
        #print('q: ',q.size())
        #print('k: ',k.size())
        #print('v: ',v.size())
        attn = (q @ k.transpose(-2, -1)) * self.scale   # scaling needs to be fixed
        # ####print('attn: ', attn.shape)   
        attn = attn.softmax(dim=-1)      #  couldn't figure out yet
        attn = self.attn_drop(attn)
        # attn = attn.view(attn.shape[0], attn.shape[1], -1, attn.shape[4])
        #print('attn after reshape: ',attn.shape) 
        x = (attn @ v)
        return x


    def forward(self, x, H, W):
        ###print('!!!!!!!!!!!!attention head: ',self.num_heads, ' !!!!!!!!!!')
        A = []
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        

        # This reduces dimension of k and v
        # 120, 160 --Flatten--> 19200--FNN--> 300
        if self.sr_ratio > 1:       
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W) 
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) 
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        k_full, v_full = kv[0], kv[1]
        ##print(f'global q:{q.shape} k:{k_full.shape} v:{v_full.shape}')
        a_1 = self.attention(q, k_full, v_full)
        ##print(f'full scale attn:{a_1.shape}')
        a_1 = a_1.transpose(1, 2)
        a_1 = a_1.reshape(B, N, C)
        a_1 = self.proj(a_1)
        a_1 = self.proj_drop(a_1)

        A.append(a_1)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        ##print(f'new k:{k.shape} new v:{v.shape} q:{q.shape}')
        for rg_shp in self.local_region_shape:
            q_patch = self.patchify(q, H, W, rg_shp)
            k_patch = self.patchify(k, H, W, rg_shp)
            v_patch = self.patchify(v, H, W, rg_shp)
            ##print(f'patchified q:{q_patch.shape}, k:{k_patch.shape}, v:{v_patch.shape}')
            patched_attn = self.attention(q_patch, k_patch, v_patch)
            #print('patched attention output: ',patched_attn.shape)
            #exit()
            patched_attn = patched_attn.permute(0, 1, 5, 2, 3, 4).contiguous()
            #patched_attn = patched_attn.view(0, 1, 5, 2, 3, 4)

            a_1 = patched_attn.view(patched_attn.shape[0], -1, patched_attn.shape[3], patched_attn.shape[4], rg_shp, rg_shp)
            #print('patched attention: ',a_1.shape)
            a_1 = a_1.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, C, N)
            #a_1 = a_1.reshape(B, C, N)
            #exit()
            #a_1 = patched_attn.view(patched_attn.shape[0], patched_attn.shape[1], -1, patched_attn.shape[4])
            ##print('local attn: ',a_1.shape)
            a_1 = a_1.transpose(1, 2)
            #a_1 = a_1.reshape(B, N, C)
            ##print('local attn reshape: ',a_1.shape)
            a_1 = self.proj(a_1)
            a_1 = self.proj_drop(a_1)
            #print('local attn final: ',a_1.shape)
            #exit()
            A.append(a_1)


        # A = torch.cat(A, dim=2)
        # A = self.final_proj(A)

        attn_fused = self.fuse_ms_attn_map(A, H, W)
        attn_fused = attn_fused.reshape(B, C, N).transpose(1, 2)
        attn_fused = self.final_proj(attn_fused)
        

        return A
       
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # ####print('!!!!!!!!!!!!attention head: ',self.num_heads, ' !!!!!!!!!!')
        B, N, C = x.shape
        # ####print()
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        # ####print(f'reshape final q:{q.shape}')
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W) 
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) 
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        k, v = kv[0], kv[1]
        
        # ####print(f'k:{k.shape}')
        # ####print(f'v:{v.shape}')
        attn = (q @ k.transpose(-2, -1)) * self.scale   
        # ####print('attention: ',attn.shape) 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # ####print('attn after reshape: ',attn.shape)
        x = (attn @ v)
        # ####print('attn*v: ',x.shape)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Adding extra variable to class initialization.
# local_region_shape:list --> to patchify query and key
class Block(nn.Module):
    """
    Transformer Block: Self-Attention -> Mix FFN -> OverLap Patch Merging
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, local_region_shape=[5, 10, 20, 40]):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.local_region_shape = local_region_shape
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, local_region_shape=self.local_region_shape)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        print('input to block: ',x.shape)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        ######print('patch size: ',patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        ######print('num_patches: ',self.num_patches)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # B C H W
        ###print('forward --> overlap patch embedding')
        #####print('input x: ',x.shape)
        #####print('proj layer: ',self.proj)
        x = self.proj(x)
        
        _, _, H, W = x.shape
        ###print(f'x after proj:{x.shape}')
        #####print(f'after projection H:{H} W:{W}')
        x = x.flatten(2).transpose(1, 2)
        ######print(f'x flatten:{x.shape}')
        # B H*W/16 C
        x = self.norm(x)
        ###print(f'x final:{x.shape}, H:{H} W:{W}')
        ######print(f'final x:{x.shape}')

        return x, H, W


class RGBXTransformer(nn.Module):
    def __init__(self, img_size=(480, 640), patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512], 
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=(img_size[0]// 4,img_size[1]//4), patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=(img_size[0]// 8,img_size[1]//8), patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=(img_size[0]// 16,img_size[1]//16), patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.extra_patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size=(img_size[0]// 4,img_size[1]//4), patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size=(img_size[0]// 8,img_size[1]//8), patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size=(img_size[0]// 16,img_size[1]//16), patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], local_region_shape=[5, 10, 20, 40])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.extra_block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], local_region_shape=[5, 10, 20, 40])
            for i in range(depths[0])])
        self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], local_region_shape=[5, 10, 20])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.extra_block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], local_region_shape=[5, 10, 20])
            for i in range(depths[1])])
        self.extra_norm2 = norm_layer(embed_dims[1])

        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], local_region_shape=[5, 10])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.extra_block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], local_region_shape=[5, 10])
            for i in range(depths[2])])
        self.extra_norm3 = norm_layer(embed_dims[2])

        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3], local_region_shape=[5])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.extra_block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3], local_region_shape=[5])
            for i in range(depths[3])])
        self.extra_norm4 = norm_layer(embed_dims[3])

        cur += depths[3]

        self.FRMs = nn.ModuleList([
                    FRM(dim=embed_dims[0], reduction=1),
                    FRM(dim=embed_dims[1], reduction=1),
                    FRM(dim=embed_dims[2], reduction=1),
                    FRM(dim=embed_dims[3], reduction=1)])

        self.FFMs = nn.ModuleList([
                    FFM(dim=embed_dims[0], reduction=1, num_heads=num_heads[0], norm_layer=norm_fuse),
                    FFM(dim=embed_dims[1], reduction=1, num_heads=num_heads[1], norm_layer=norm_fuse),
                    FFM(dim=embed_dims[2], reduction=1, num_heads=num_heads[2], norm_layer=norm_fuse),
                    FFM(dim=embed_dims[3], reduction=1, num_heads=num_heads[3], norm_layer=norm_fuse)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained="../../Results/saved_models/NYUDV2_CMX+Segformer-B2.pth"):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B x N x H x W
        """
        ###print("initial x_rgb: ",x_rgb.size())
        ###print(f'input:::rgb:{x_rgb.shape} ir:{x_e.shape}')
        B = x_rgb.shape[0]
        outs = []
        outs_fused = []

        # stage 1
        ###print("####################Stage 1############################")
        ###print('patch embedding 1')
        x_rgb, H, W = self.patch_embed1(x_rgb)
        # B H*W/16 C
        #####print("s1 x_rgb: ",x_rgb.size())
        ####print('IR patch embedding 1')
        x_e, _, _ = self.extra_patch_embed1(x_e)
        ###print("$$$$$RGB patch Process$$$$$$")
        for i, blk in enumerate(self.block1):
            ####print(f'Block: {i}')
            x_rgb = blk(x_rgb, H, W)
        ###print("$$$$$IR patch Process$$$$$$")
        for i, blk in enumerate(self.extra_block1):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm1(x_rgb)
        x_e = self.extra_norm1(x_e)
        ####print(f'****** output after attention blocks:{x_rgb.shape}********')

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        ####print(f'output rgb:{x_rgb.shape} ir:{x_e.shape}')
        x_rgb, x_e = self.FRMs[0](x_rgb, x_e)
        #####print(f'output after FRM rgb:{x_rgb.shape} ir:{x_e.shape}')
        x_fused = self.FFMs[0](x_rgb, x_e)
        ###print(f'final output:{x_fused.shape}')
        outs.append(x_fused)
        

        # stage 2
        ###print("####################Stage 2############################")
        ###print('patch embedding 2')
        x_rgb, H, W = self.patch_embed2(x_rgb)
        #####print("s2 x_rgb: ",x_rgb.size())
        ####print('IR patch embedding 2')
        x_e, _, _ = self.extra_patch_embed2(x_e)
        ###print("$$$$$RGB patch Process$$$$$$")
        for i, blk in enumerate(self.block2):
            x_rgb = blk(x_rgb, H, W)
        ###print("$$$$$IR patch Process$$$$$$")
        for i, blk in enumerate(self.extra_block2):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm2(x_rgb)
        x_e = self.extra_norm2(x_e)
        ####print(f'****** output after attention blocks:{x_rgb.shape}********')

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        ####print(f'output rgb:{x_rgb.shape} ir:{x_e.shape}')
        x_rgb, x_e = self.FRMs[1](x_rgb, x_e)
        #####print(f'output after FRM rgb:{x_rgb.shape} ir:{x_e.shape}')
        x_fused = self.FFMs[1](x_rgb, x_e)
        ###print(f'final output:{x_fused.shape}')
        outs.append(x_fused)
        

        # stage 3
        ###print("####################Stage 3############################")
        ###print('patch embedding 3')
        x_rgb, H, W = self.patch_embed3(x_rgb)
        #####print("s3 x_rgb: ",x_rgb.size())
        ####print('IR patch embedding 3')
        x_e, _, _ = self.extra_patch_embed3(x_e)
        ###print("$$$$$RGB patch Process$$$$$$")
        for i, blk in enumerate(self.block3):
            x_rgb = blk(x_rgb, H, W)
        ###print("$$$$$IR patch Process$$$$$$")
        for i, blk in enumerate(self.extra_block3):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm3(x_rgb)
        x_e = self.extra_norm3(x_e)
        ####print(f'****** output after attention blocks:{x_rgb.shape}********')

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        ####print(f'output rgb:{x_rgb.shape} ir:{x_e.shape}')
        x_rgb, x_e = self.FRMs[2](x_rgb, x_e)
        #####print(f'output after FRM rgb:{x_rgb.shape} ir:{x_e.shape}')
        x_fused = self.FFMs[2](x_rgb, x_e)
        ###print(f'final output:{x_fused.shape}')
        outs.append(x_fused)
        

        # stage 4
        ###print("####################Stage 4############################")
        ###print('patch embedding 4')
        x_rgb, H, W = self.patch_embed4(x_rgb)
        #####print("s4 x_rgb: ",x_rgb.size())
        ####print('IR patch embedding  4')
        x_e, _, _ = self.extra_patch_embed4(x_e)
        ###print("$$$$$RGB patch Process$$$$$$")
        for i, blk in enumerate(self.block4):
            x_rgb = blk(x_rgb, H, W)
        ###print("$$$$$IR patch Process$$$$$$")
        for i, blk in enumerate(self.extra_block4):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm4(x_rgb)
        x_e = self.extra_norm4(x_e)
        ####print(f'****** output after attention blocks:{x_rgb.shape}********')

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        ####print(f'output rgb:{x_rgb.shape} ir:{x_e.shape}')
        x_rgb, x_e = self.FRMs[3](x_rgb, x_e)
        #####print(f'output after FRM rgb:{x_rgb.shape} ir:{x_e.shape}')
        x_fused = self.FFMs[3](x_rgb, x_e)
        ###print(f'final output:{x_fused.shape}')
        outs.append(x_fused)
        
        return outs

    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        return out


def load_dualpath_model(model, model_file):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        ##print("string file")
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        #raw_state_dict = torch.load(model_file)
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    
    state_dict = {}
    for k, v in raw_state_dict.items():
        # ##print("keys: ", k)
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
            state_dict[k.replace('patch_embed', 'extra_patch_embed')] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
            state_dict[k.replace('block', 'extra_block')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            state_dict[k.replace('norm', 'extra_norm')] = v

    t_ioend = time.time()

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    
    t_end = time.time()
    # logger.info(
    #     "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
    #         t_ioend - t_start, t_end - t_ioend))


class mit_b0(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


if __name__=="__main__":
    backbone = mit_b2(norm_layer = nn.BatchNorm2d)
    
    # ####print(backbone)
    B = 4
    C = 3
    H = 480
    W = 640
    device = 'cuda:0'
    ms_attention = MultiScaleAttention(32, num_heads=4, sr_ratio=8, local_region_shape=[5,10,20,40])
    ms_attention = ms_attention.to(device)
    # ms_attention = nn.DataParallel(ms_attention, device_ids = [0,1])
    # ms_attention.to(f'cuda:{ms_attention.device_ids[0]}', non_blocking=True)

    f = torch.randn(B, 19200, 32).to(device)

    # ####print(f'input to multiScaleAttention:{f.shape}')
    y = ms_attention(f, 120, 160)

    # print('attn output: ',y.shape)
    # # rgb = torch.randn(B, C, H, W)
    # # x = torch.randn(B, C, H, W)
    # # outputs = backbone(rgb, x)
    # for output in outputs:
    #     print(output.size())


