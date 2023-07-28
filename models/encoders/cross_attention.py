

import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


from fusion import iAFF
import math
import time


class CrossAttention(nn.Module):
    def __init__(self, input_dim, s_patch, l_patch, num_heads, attn_drop = 0.):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.smaller_patch_size = s_patch
        self.larger_patch_size = l_patch
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def attention(self, q, k, v):
        ######print(self.scale)
        print('q: ',q.size())
        print('k: ',k.size())
        print('v: ',v.size())
        attn = (q @ k.transpose(-2, -1)) * self.scale   # scaling needs to be fixed
        # #####print('attn: ', attn.shape)   
        attn = attn.softmax(dim=-1)      #  couldn't figure out yet
        attn = self.attn_drop(attn)
        # attn = attn.view(attn.shape[0], attn.shape[1], -1, attn.shape[4])
        print('attn after reshape: ',attn.shape) 
        x = (attn @ v)
        return x

    def patchify(self, arr, H, W, patch_size, overlap = False):
        B = arr.shape[0]
        Ch = arr.shape[-1]
        # print('arr: ',arr.shape)
        arr = arr.view(arr.shape[0], arr.shape[1], H, W, arr.shape[3])
        print('arr view: ',arr.shape)
        patches = arr.view(arr.shape[0], arr.shape[1], arr.shape[2] // patch_size, patch_size, arr.shape[3] // patch_size, patch_size, arr.shape[4])
        ##print('patches: ',patches.shape)
        #B x num_head x H//ps x ps x W//ps x ps x C
        print('patches shape: ', patches.shape)
        patches = patches.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        # B x num_head x C x H//ps x W//ps x ps x ps
        print('patches permute: ', patches.shape)
        patches = patches.view(patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3], -1, patches.shape[6])
        print('patches reshape: ', patches.shape)
        patches = patches.reshape(B, self.num_heads, -1, patch_size*patch_size, Ch)
        print('patches final: ', patches.shape)
        return patches
    
    """ x_s: smaller patched (say, 5x5) attention .. B x C x H x W 
        x_l: larger patched (say, 10x10) attention .. B x C x H x W   
    """
    def forward(self, x_s, x_l, H, W):  

        B, N, C = x_s.shape
        
        q = self.query(x_l).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # Bx(num_heads)x(H1*W1)xhead_dimxC
        k = self.key(x_s).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # Bx(num_heads)x(H1*W1)xhead_dimxC
        v = self.value(x_s).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # Bx(num_heads)x(H1*W1)xhead_dimxC

        q_patch = self.patchify(q, H, W, self.larger_patch_size)
        k_patch = self.patchify(k, H, W, self.smaller_patch_size)
        v_patch = self.patchify(v, H, W, self.smaller_patch_size)

        a_1 = self.attention(q_patch, k_patch, v_patch)

        print('attention output: ',a_1.shape)

        
        return a_1
if __name__=="__main__":
    B = 2
    x_s = torch.randn(B, 19200, 32)
    x_l = torch.randn(B, 19200, 32)
    
    cs_attention = CrossAttention(32, 5, 10, 4)

    y = cs_attention(x_s, x_l, 120, 160)