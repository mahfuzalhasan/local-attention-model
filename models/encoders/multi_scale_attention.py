import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from stoken_attn_fusion import StokenAttention, StokenAttentionLayer
from relative_pe import RelPosEmb

from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from fusion import iAFF
import math
import time

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

        # self.attn_fusion = iAFF(dim)
        # self.attn_fusion = []
        self.attn_fusion = [StokenAttentionLayer(dim, n_iter=1, 
                            sp_stoken_size=(local_region_shape[0], local_region_shape[0]), lp_stoken_size=None)]
        self.rel_pos_shape_wise = []
        for block_size in local_region_shape:
            self.rel_pos_shape_wise.append(RelPosEmb(block_size, block_size, head_dim))
        self.rel_pos_shape_wise = nn.ModuleList(self.rel_pos_shape_wise)

        for i in range(len(local_region_shape)-1):
            sp_stoken_size = (local_region_shape[i], local_region_shape[i])
            lp_stoken_size = (local_region_shape[i+1], local_region_shape[i+1])
            self.attn_fusion.append(StokenAttentionLayer(dim, n_iter=1, 
                            sp_stoken_size=sp_stoken_size, lp_stoken_size=lp_stoken_size))
        self.attn_fusion = nn.ModuleList(self.attn_fusion)
        # self.global_fusion = iAFF(dim)
        self.final_proj = nn.Linear(dim, dim)
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


    """ arr.shape -> B x num_head x H x W x C """
    # create overlapping patches
    def patchify(self, arr, H, W, patch_size, overlap = False):
        arr = arr.view(arr.shape[0], arr.shape[1], H, W, arr.shape[3])
        # B, nh, H, W, Ch
        B, Nh, H, W, Ch = arr.shape
        
        patches = arr.view(arr.shape[0], arr.shape[1], arr.shape[2] // patch_size, patch_size, arr.shape[3] // patch_size, patch_size, arr.shape[4])
        #B, nh, H//p, p, W//p, p, Ch
        #print(patches.shape)
        # patches = patches.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        # #B, nh, H//p, W//p, p, p , ch

        patches = patches.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, Nh, patch_size**2, Ch)
        #B, nh, H//p, W//p, p, p , ch

        
        # patches = patches.view(patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3], -1, patches.shape[6])
        # #B, H//p, W//p, nh, p^2, ch

        return patches


    def attention(self, q_p, k_p, v_p, rel_pos):
        B_, Nh, local_shape, Ch = q_p.shape
        q, k, v = map(lambda t: rearrange(t, 'b h n d -> (b h) n d', h = Nh), (q_p, k_p, v_p))
        #print(q.shape, k.shape, v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale   # scaling needs to be fixed
        attn = attn.softmax(dim=-1)      #  couldn't figure out yet
        # #print('attn matrix: ',attn.shape)
        a = rel_pos(q)
        #print('rel_pos: ',a.shape)
        attn = self.attn_drop(attn) 
        attn += a
        x = (attn @ v)
        x = x.view(B_, Nh, local_shape, Ch)
        return x

    # with different stoken fusion for smallest token and discarding global attention
    def fuse_ms_attn_map(self, A, H, W):
        B, N, C = A[0].shape
        output_small_patched_attn = A[0].permute(0, 2, 1).contiguous().view(B, C, H, W)
        output_small_patched_attn = self.attn_fusion[0](output_small_patched_attn, output_small_patched_attn)
        for i in range(1,len(A)):
            output_small_patched_attn = self.attn_fusion[i](output_small_patched_attn, A[i].permute(0, 2, 1).contiguous().view(B, C, H, W))
        return output_small_patched_attn

    def forward(self, x, H, W):
        ##print('!!!!!!!!!!!!attention head: ',self.num_heads, ' !!!!!!!!!!')
        A = []
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        #print('initial: ',q.shape, k.shape)
        for idx, patch_size in enumerate(self.local_region_shape):
            q_patch = self.patchify(q, H, W, patch_size)
            k_patch = self.patchify(k, H, W, patch_size)
            v_patch = self.patchify(v, H, W, patch_size)
            #print('for attn: ',q_patch.shape, k_patch.shape)
            rel_pos = self.rel_pos_shape_wise[idx]
            patched_attn = self.attention(q_patch, k_patch, v_patch, rel_pos)
            ## B_, Nh, p^2, Ch
            # #print(patched_attn.shape)
            patched_attn = patched_attn.permute(0, 2, 1, 3).contiguous()
            # B_, p^2, Nh, Ch --> (B, H//p, W//p, p^2, C)
            #print('attn size: ',patched_attn.shape)
            # a_1 = patched_attn.view(patched_attn.shape[0], -1, patched_attn.shape[3], patched_attn.shape[4], patch_size, patch_size)
            a_1 = patched_attn.view(B, H//patch_size, W//patch_size, patch_size**2, self.num_heads, C // self.num_heads)
            a_1 = a_1.reshape(B, N, C)
            # #print('final attn size: ',a_1.shape)
            # a_1 = a_1.transpose(1, 2)
            a_1 = self.proj(a_1)
            a_1 = self.proj_drop(a_1)
            A.append(a_1)
        attn_fused = self.fuse_ms_attn_map(A, H, W)
        attn_fused = attn_fused.reshape(B, C, N).transpose(1, 2)        
        attn_fused = self.final_proj(attn_fused) 
        return attn_fused

if __name__=="__main__":
    # ########print(backbone)
    B = 4
    C = 3
    H = 480
    W = 640
    device = 'cuda:0'
    ms_attention = MultiScaleAttention(32, num_heads=4, sr_ratio=8, local_region_shape=[2,4,8])
    ms_attention = ms_attention.to(device)
    # ms_attention = nn.DataParallel(ms_attention, device_ids = [0,1])
    # ms_attention.to(f'cuda:{ms_attention.device_ids[0]}', non_blocking=True)

    f = torch.randn(B, 19200, 32).to(device)
    ###print(f'input to multiScaleAttention:{f.shape}')
    y = ms_attention(f, 120, 160)
    print('output: ',y.shape)