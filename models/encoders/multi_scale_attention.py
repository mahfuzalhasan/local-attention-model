import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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

        self.attn_fusion = iAFF(dim)
        # self.final_proj = nn.Linear(dim * (len(self.local_region_shape)+1), dim)
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
        if not overlap:
            ##print('arr: ',arr.shape)
            arr = arr.view(arr.shape[0], arr.shape[1], H, W, arr.shape[3])
            ##print('arr view: ',arr.shape)
            patches = arr.view(arr.shape[0], arr.shape[1], arr.shape[2] // patch_size, patch_size, arr.shape[3] // patch_size, patch_size, arr.shape[4])
            ##print('patches: ',patches.shape)
            #B x num_head x H//ps x ps x W//ps x ps x C
            # #####print('patches shape: ', patches.shape)
            patches = patches.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
            # B x num_head x C x H//ps x W//ps x ps x ps
            ##print('patches permute: ', patches.shape)
            patches = patches.view(patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3], -1, patches.shape[6])
            ##print('patches reshape: ', patches.shape)
            #exit()
            return patches
        else:
            stride = patch_size//2
            arr = arr.view(arr.shape[0], arr.shape[1], H, W, arr.shape[3])
            arr = arr.permute(0, 1, 4, 2, 3)
            patches = arr.unfold(4, patch_size, stride).unfold(3, patch_size, stride).contiguous()
            ####print('patches: ',patches.shape)
            patches = patches.view(arr.shape[0], arr.shape[1], -1, patch_size, patch_size)
            ###print('patches: ',patches.shape)
            #exit()
            return patches


    def attention(self, q, k, v):
        ######print(self.scale)
        ##print('q: ',q.size())
        ##print('k: ',k.size())
        ##print('v: ',v.size())
        attn = (q @ k.transpose(-2, -1)) * self.scale   # scaling needs to be fixed
        # #####print('attn: ', attn.shape)   
        attn = attn.softmax(dim=-1)      #  couldn't figure out yet
        attn = self.attn_drop(attn)
        # attn = attn.view(attn.shape[0], attn.shape[1], -1, attn.shape[4])
        ##print('attn after reshape: ',attn.shape) 
        x = (attn @ v)
        return x


    def fuse_ms_attn_map(self, A, H, W):
        B, N, C = A[0].shape
        output = A[1].permute(0, 2, 1).contiguous().view(B, C, H, W)
        global_attn = A[0].permute(0, 2, 1).contiguous().view(B, C, H, W)
        ##print('shapes: ', output.shape, global_attn.shape)
        for i in range(2,len(A)):
            output = self.attn_fusion(output, A[i].permute(0, 2, 1).contiguous().view(B, C, H, W))
        output = self.attn_fusion(output, global_attn)
        #print('final fused: ',output.shape)
        return output

    def forward(self, x, H, W):
        ####print('!!!!!!!!!!!!attention head: ',self.num_heads, ' !!!!!!!!!!')
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
        ###print(f'global q:{q.shape} k:{k_full.shape} v:{v_full.shape}')
        a_1 = self.attention(q, k_full, v_full)
        ###print(f'full scale attn:{a_1.shape}')
        a_1 = a_1.transpose(1, 2)
        a_1 = a_1.reshape(B, N, C)
        a_1 = self.proj(a_1)
        a_1 = self.proj_drop(a_1)

        A.append(a_1)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        ###print(f'new k:{k.shape} new v:{v.shape} q:{q.shape}')
        for rg_shp in self.local_region_shape:
            q_patch = self.patchify(q, H, W, rg_shp)
            k_patch = self.patchify(k, H, W, rg_shp)
            v_patch = self.patchify(v, H, W, rg_shp)
            #print(f'patchified q:{q_patch.shape}, k:{k_patch.shape}, v:{v_patch.shape}')
            patched_attn = self.attention(q_patch, k_patch, v_patch)
            #print('patched attention output: ',patched_attn.shape)
            #exit()
            patched_attn = patched_attn.permute(0, 1, 5, 2, 3, 4).contiguous()
            #patched_attn = patched_attn.view(0, 1, 5, 2, 3, 4)
            #print('patched attention permute: ',patched_attn.shape)
            a_1 = patched_attn.view(patched_attn.shape[0], -1, patched_attn.shape[3], patched_attn.shape[4], rg_shp, rg_shp)
            #print('patched attention reshape: ',a_1.shape)
            a_1 = a_1.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, C, N)
            #a_1 = a_1.reshape(B, C, N)
            #exit()
            #a_1 = patched_attn.view(patched_attn.shape[0], patched_attn.shape[1], -1, patched_attn.shape[4])
            ###print('local attn: ',a_1.shape)
            a_1 = a_1.transpose(1, 2)
            #a_1 = a_1.reshape(B, N, C)
            ###print('local attn reshape: ',a_1.shape)
            a_1 = self.proj(a_1)
            a_1 = self.proj_drop(a_1)
            ##print('local attn final: ',a_1.shape)
            #exit()
            A.append(a_1)

        #print('$$$$multi attention shapes$$$$')
        # for attn_o in A:
        #     #print(attn_o.shape)
        attn_fused = self.fuse_ms_attn_map(A, H, W)
        attn_fused = attn_fused.reshape(B, C, N).transpose(1, 2)
        
        attn_fused = self.final_proj(attn_fused) 
        return attn_fused
        

        return A

if __name__=="__main__":
    # #######print(backbone)
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
    ##print(f'input to multiScaleAttention:{f.shape}')
    y = ms_attention(f, 120, 160)
    ##print('output: ',y.shape)