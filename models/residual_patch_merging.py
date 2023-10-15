import torch
import torch.nn as nn

class ConvStem(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        stem_dim = embed_dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_dim, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      groups=stem_dim, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      groups=stem_dim, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      groups=stem_dim, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(stem_dim, embed_dim,
                              kernel_size=3,
                              stride=2, padding=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(self.stem(x))
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)
        
class ResidualMergePatch(nn.Module):
    def __init__(self, dim, out_dim, num_tokens=1):
        super().__init__()
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim, bias=False)
        # use MaxPool3d to avoid permutations
        self.maxp = nn.MaxPool3d((2, 2, 1), (2, 2, 1))
        self.res_proj = nn.Linear(dim, out_dim, bias=False)

    def forward(self, x, H, W):
        global_token, x = x[:, :self.num_tokens].contiguous(), x[:, self.num_tokens:].contiguous()
        B, L, C = x.shape

        print(f'x:{x.shape} global token:{global_token.shape}')

        x = x.view(B, H, W, C)
        res = self.res_proj(self.maxp(x).view(B, -1, C))
        print('maxp: ',self.maxp(x).shape)
        print('res: ',res.shape)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        print(f'x0:{x0.shape} x1:{x1.shape} x2:{x2.shape} x3:{x3.shape}')
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        print('x after concat: ',x.shape)
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        print('x after view: ',x.shape)

        x = self.norm(x)
        x = self.reduction(x)
        x = x + res
        global_token = self.proj(self.norm2(global_token))
        x = torch.cat([global_token, x], 1)
        return x, (H // 2, W // 2)

if __name__=='__main__':
    B = 3
    N = 19200
    H = 1024
    W = 1024
    N = H * W
    C = 3
    num_tokens = 8
    embed_dim = 64
    output_dim = 96
    patch_embed = ConvStem(img_size=H, patch_size=8, embed_dim=embed_dim)
    global_token = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
    print('global token initial: ',global_token.shape)
    x = torch.randn(B, C, H, W)
    x, (H, W) = patch_embed(x)
    global_token = global_token.expand(x.shape[0], -1, -1)
    print(f'x:{x.shape} H:{H} W:{W} G:{global_token.shape}')
    x = torch.cat((global_token, x), dim=1)
    print('after concat: ',x.shape)
    rpm = ResidualMergePatch(embed_dim, output_dim, num_tokens)
    # x = torch.randn(B, N, C)
    
    y, (H, W) = rpm(x, H, W)
    print(y.shape, H, W)