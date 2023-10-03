import torch.nn as nn
import torch
import torch.nn.functional as F
## Github: 
class AttentionST(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
                
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
                
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
                
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads *3, N).chunk(3, dim=2) # (B, num_heads, head_dim, N)
        
        attn = (k.transpose(-1, -2) @ q) * self.scale
        
        attn = attn.softmax(dim=-2) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, c, h, w = x.shape
        #print(f'unfold::: input:{x.shape} weights:{self.weights.shape}')
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2)        
        #print(f'after conv:{x.shape}')
        return x.reshape(b, c*9, h*w)

class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x

class StokenAttention(nn.Module):
    def __init__(self, dim, sp_stoken_size, lp_stoken_size = None, n_iter=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.n_iter = n_iter
        self.sp_stoken_size = sp_stoken_size
        self.lp_stoken_size = lp_stoken_size
                
        self.scale = dim ** - 0.5
        
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        
        self.stoken_refine = AttentionST(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
       
    ## Attention from larger patch --> x_l
    # Attention from smaller patch --> x_s
    def sample_resizing(self, x, token_size):
        B, C, H0, W0 = x.shape
        h, w = token_size
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        
        return x, pad_r, pad_b
        
    def stoken_forward(self, x_s, x_l=None):
        '''
           x: (B, C, H, W)
        '''
        if x_l is not None:
            hs, ws = self.sp_stoken_size
            token_small_attn = F.avg_pool2d(x_s, (hs, ws))
            token_small_attn = F.avg_pool2d(token_small_attn, (2, 2))

            ### for the sake of carrying info from large token space to small token space
            B, C, H0, W0 = x_l.shape
            x_l, pad_r, pad_b = self.sample_resizing(x_l, self.lp_stoken_size)
            _, _, H, W = x_l.shape
            h, w = self.lp_stoken_size
            hh, ww = H//h, W//w
            token_large_attn = F.adaptive_avg_pool2d(x_l, (hh, ww)) # (B, C, hh, ww)
            stoken_features = token_small_attn + token_large_attn
            ##########################################################################
            pixel_features = x_l.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C)
        else:   # For first attention
            B, C, H0, W0 = x_s.shape
            x_s, pad_r, pad_b = self.sample_resizing(x_s, self.sp_stoken_size)
            _, _, H, W = x_s.shape
            h, w = self.sp_stoken_size
            hh, ww = H//h, W//w
            
            stoken_features = F.adaptive_avg_pool2d(x_s, (hh, ww))
            pixel_features = x_s.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C)

        

        #print(f'token features:{stoken_features.shape}  pixel f:{pixel_features.shape}')
        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
                
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9)
                #print(f'stoken features:{stoken_features.shape}')
                affinity_matrix = pixel_features @ stoken_features * self.scale # (B, hh*ww, h*w, 9)
                affinity_matrix = affinity_matrix.softmax(-1) # (B, hh*ww, h*w, 9)
                #print(f'Q=X * S.T=affinity matrix:{affinity_matrix.shape}')
                s = affinity_matrix.sum(2).transpose(1, 2)
                #print('s from am: ',s.shape)
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                #print(f'affinity_matrix_sum:{affinity_matrix_sum.shape}')
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                #print(f'affinity_matrix_sum fold:{affinity_matrix_sum.shape}')

                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
                    
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
                    
                    stoken_features = stoken_features/(affinity_matrix_sum + 1e-12) # (B, C, hh, ww)

        # s token attention 
        #print('s token feature ', stoken_features.shape)
        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
        #print(f'S = Q.T * X:{stoken_features.shape}')
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
        
        stoken_features = stoken_features/(affinity_matrix_sum.detach() + 1e-12) # (B, C, hh, ww)
        #print(f'S reshape:{stoken_features.shape}')
        stoken_features = self.stoken_refine(stoken_features)
        #print(f'stoken attention out:{stoken_features.shape}')
        
        
        #### Token upsampling
        stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
        #print(f'stoken features unfold:{stoken_features.shape} ')
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9) # (B, hh*ww, C, 9)
        #print(f'stoken features:{stoken_features.shape} ')
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2) # (B, hh*ww, C, h*w)
        #print(f'pixel_features:{pixel_features.shape} ')
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        #print(f'pixel_features reshape:{pixel_features.shape} ')
        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        
        return pixel_features

    def forward(self, xs, xl):
        # if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
        # return self.stoken_forward(xs, xl)
        if self.lp_stoken_size is not None:
            return self.stoken_forward(xs, xl)
        else:
            return self.stoken_forward(xs)

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
                
        self.shortcut = nn.Parameter(torch.eye(kernel_size).reshape(1, 1, kernel_size, kernel_size))
        self.shortcut.requires_grad = False
        
    def forward(self, x):
        return F.conv2d(x, self.conv.weight+self.shortcut, self.conv.bias, stride=1, padding=self.kernel_size//2, groups=self.dim) # equal to x + conv(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True, downsample=False, kernel_size=5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
               
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
        self.conv = ResDWC(hidden_features, 3)
        
    def forward(self, x):       
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)
        return x

class StokenAttentionLayer(nn.Module):
    def __init__(self, dim, n_iter, sp_stoken_size, lp_stoken_size,
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5):
        super().__init__()
                        
        self.layerscale = layerscale
        
        self.pos_embed = ResDWC(dim, 3)
                                        
        self.norm1 = LayerNorm2d(dim)
        self.attn = StokenAttention(dim, sp_stoken_size, lp_stoken_size, n_iter=n_iter,                                     
                                    num_heads=num_heads, qkv_bias=qkv_bias, 
                                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)   
                    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)
                
        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
        
    def forward(self, x_s, x_l):
        x = self.pos_embed(x_l)

        #print(f'x_l after pos embed:{x.shape} layerscale:{self.layerscale}')
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x_s), self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x))) 
        else:
            x = x + self.drop_path(self.attn(self.norm1(x_s), self.norm1(x)))
            #print(f'x after attn:{x.shape}')
            x = x + self.drop_path(self.mlp2(self.norm2(x)))  
            #print(f'x after mlp2:{x.shape}')      
        return x

if __name__=="__main__":
    B = 4
    C = 32
    H = 120
    W = 160
    x_s = torch.randn(B, C, H, W)
    x_l = torch.randn(B, C, H, W)

    attn_fusion = StokenAttention(dim=32, sp_stoken_size=(5,5), lp_stoken_size=(10,10))
    y = attn_fusion(x_s, x_l)
