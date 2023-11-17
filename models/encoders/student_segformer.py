import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
sys.path.append(parent_dir)

model_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(model_dir)

from merge_attn import MultiScaleAttention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from net_utils import FeatureFusionModule as FFM
from net_utils import FeatureRectifyModule as FRM
from decoders.MLPDecoder import DecoderHead





import math
import time
from engine.logger import get_logger




logger = get_logger()


class DWConv(nn.Module):
    """
    Depthwise convolution bloc: input: x with size(B N C); output size (B N C)
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)
        self.dim = dim

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous() # B N C -> B C N -> B C H W
        self.input_shape = x.shape
        x = self.dwconv(x) 
        x = x.flatten(2).transpose(1, 2) # B C H W -> B N C

        return x
    # # FLOPs = (2 * input_channels * kernel_size^2 * output_channels * output_height * output_width) / stride^2
    # def flops(self, H, W):
    #     # Calculate FLOPs for the depthwise convolution operation
    #     flops_dwconv = H * W * self.dim * 3 * 3 * 2  # kernel_size=3, stride=1
    #     # 3x3 kernel, input channels = dim, output channels = dim, multiply by 2 (for multiply and add)

    #     return flops_dwconv
    
    def flops(self):
        # Correct calculation for output dimensions
        padding = (1,1) 
        kernel_size = (3,3)
        stride = 1
        groups = self.dim
        in_chans = self.dim
        out_chans = self.dim

        output_height = ((self.input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride) + 1
        output_width = ((self.input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride) + 1

        # Convolution layer FLOPs
        conv_flops = 2 * out_chans * output_height * output_width * kernel_size[0] * kernel_size[1] * in_chans / groups

        total_flops = conv_flops
        return total_flops


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
        # self.H = H
        # self.W = W
        print('input: MLP ',x.shape, H, W)
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def flops(self):
        # H, W = self.H, self.W
        flops_mlp = self.fc1.in_features * self.fc1.out_features * 2
        flops_mlp += self.dwconv.flops()
        flops_mlp += self.fc2.in_features * self.fc2.out_features * 2
        return flops_mlp
    


class Block(nn.Module):
    """
    Transformer Block: Self-Attention -> Mix FFN -> OverLap Patch Merging
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, local_region_shape=[5, 10, 20, 40], img_size=(1024, 1024)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.local_region_shape = local_region_shape
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, local_region_shape=self.local_region_shape, img_size=img_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)
        # print('====== block ======', dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer, sr_ratio, local_region_shape, img_size)

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
        print('+++++++++ block +++++ input: ',x.shape, H, W)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

    def flops(self):
        # FLOPs for MultiScaleAttention
        attn_flops = self.attn.flops()

        # FLOPs for Mlp
        mlp_flops = self.mlp.flops()

        total_flops = attn_flops + mlp_flops
        return total_flops


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.stride = stride
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.input_shape = None

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size
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
        self.input_shape = x.shape
        print('x before proj: ',x.shape)
        x = self.proj(x)
        print('x after proj: ',x.shape)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # B H*W/16 C
        x = self.norm(x)
        print('x after norm: !!!!!!!!!!!!!!!!!! ',x.shape)

        return x, H, W
    def flops(self):
        # Correct calculation for output dimensions
        padding = (self.patch_size[0] // 2, self.patch_size[1] // 2)
        output_height = ((self.input_shape[2] + 2 * padding[0] - self.patch_size[0]) // self.stride) + 1
        output_width = ((self.input_shape[3] + 2 * padding[1] - self.patch_size[1]) // self.stride) + 1

        # Convolution layer FLOPs
        conv_flops = 2 * self.embed_dim * output_height * output_width * self.patch_size[0] * self.patch_size[1] * self.in_chans

        # Layer normalization FLOPs
        norm_flops = 2 * self.embed_dim * output_height * output_width

        total_flops = conv_flops + norm_flops
        return total_flops


# How to apply multihead multiscale
class RGBXTransformer(nn.Module):
    def __init__(self, img_size=(1024, 1024), patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512], 
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], local_region_shape=[8, 16], img_size=(img_size[0]// 4,img_size[1]//4))
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], local_region_shape=[4, 8, 8, 16], img_size=(img_size[0]// 8,img_size[1]//8))
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        # 64x64
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], local_region_shape=[1 ,2, 2, 4, 4], img_size=(img_size[0]// 16,img_size[1]//16))
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        #32x32
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3], local_region_shape=[1, 1, 1, 1, 2, 2, 2, 2], img_size=(img_size[0]// 32,img_size[1]//32))
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        cur += depths[3]

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

    def init_weights(self, pretrained="../../Results/saved_models/segformer/mit_b2.pth"):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x_rgb):
        """
        x_rgb: B x N x H x W
        """
        print('input: ',x_rgb.shape)
        B = x_rgb.shape[0]
        outs = []
        outs_fused = []

        # stage 1
        x_rgb, H, W = self.patch_embed1(x_rgb)

        print('############### Stage 1 ##########################')
        print('tokenization: ',x_rgb.shape)

        # exit()
        # B H*W/16 C
       
        for i, blk in enumerate(self.block1):
            x_rgb = blk(x_rgb, H, W)
        x_rgb = self.norm1(x_rgb)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_rgb)
        print('output: ',x_rgb.shape)
        print("******** End Stage 1 **************")
        
        

        # stage 2
        print('############### Stage 2 ##########################')
        x_rgb, H, W = self.patch_embed2(x_rgb)
        print('tokenization: ',x_rgb.shape)
        
        for i, blk in enumerate(self.block2):
            x_rgb = blk(x_rgb, H, W)
        
        x_rgb = self.norm2(x_rgb)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_rgb)

        print('output: ',x_rgb.shape)
        print("******** End Stage 2 **************")
        

        # stage 3
        x_rgb, H, W = self.patch_embed3(x_rgb)
        print('############### Stage 3 ##########################')
        print('tokenization: ',x_rgb.shape)
        
        for i, blk in enumerate(self.block3):
            x_rgb = blk(x_rgb, H, W)
        
        x_rgb = self.norm3(x_rgb)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_rgb)

        print('output: ',x_rgb.shape)
        print("******** End Stage 3 **************")
        

        # stage 4
        x_rgb, H, W = self.patch_embed4(x_rgb)
        print('############### Stage 4 ##########################')
        print('tokenization: ',x_rgb.shape)
        
        for i, blk in enumerate(self.block4):
            x_rgb = blk(x_rgb, H, W)
        x_rgb = self.norm4(x_rgb)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_rgb)

        print('output: ',x_rgb.shape)
        print("******** End Stage 4 **************")
        
        return outs

    def forward(self, x_rgb):
        # print()
        out = self.forward_features(x_rgb)
        return out
    
    def flops(self):
        flops = 0
        flops += self.patch_embed1.flops()
        flops += self.patch_embed2.flops()
        flops += self.patch_embed3.flops()
        flops += self.patch_embed4.flops()

        for i, blk in enumerate(self.block1):
            flops += blk.flops()
        for i, blk in enumerate(self.block2):
            flops += blk.flops()
        for i, blk in enumerate(self.block3):
            flops += blk.flops()
        for i, blk in enumerate(self.block4):
            flops += blk.flops()
        
        return flops


def load_dualpath_model(model, model_file):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        #raw_state_dict = torch.load(model_file)
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    

    t_ioend = time.time()

    model.load_state_dict(raw_state_dict, strict=False)
    #del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))


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
            img_size=(1024, 1024), patch_size=4, embed_dims=[64, 128, 320, 512], 
            num_heads=[2, 4, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], 
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


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
    
    # #######print(backbone)
    B = 4
    C = 3
    H = 1024
    W = 1024
    device = 'cuda:1'
    rgb = torch.randn(B, C, H, W)
    x = torch.randn(B, C, H, W)
    outputs = backbone(rgb)
    for output in outputs:
        print(output.size())