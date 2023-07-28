import torch
import torch.nn as nn
import torch.nn.functional as F



B = 8
C = 32
H = 120
W = 160

M = torch.randn(B, C, H, W)
N = torch.randn(B, C, H, W)

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # Bx(num_heads)x(H*W)xhead_dim
        k = self.key(x).view(B, self.num_heads, self.head_dim, -1)  # Bx(num_heads)xhead_dimx(H*W)
        v = self.value(x).view(B, self.num_heads, self.head_dim, -1)  # Bx(num_heads)xhead_dimx(H*W)

        energy = torch.matmul(q, k)  # Bx(num_heads)x(H*W)x(H*W)
        attention = F.softmax(energy, dim=-1)  # Bx(num_heads)x(H*W)x(H*W)

        out = torch.matmul(v, attention).permute(0, 2, 1, 3).contiguous()  # Bxhead_dimx(num_heads)x(H*W)
        out = out.view(B, C, H, W)  # BxCxHxW
        return out

class CrossAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x1, x2):
        B, C, _, H1, W1 = x1.shape
        _, _, _, H2, W2 = x2.shape

        q = self.query(x1).view(B, C, self.num_heads, self.head_dim, -1).permute(0, 2, 4, 3, 1)  # Bx(num_heads)x(H1*W1)xhead_dimxC
        k = self.key(x2).view(B, C, self.num_heads, self.head_dim, -1)  # Bx(num_heads)xhead_dimxCx(H2*W2)
        v = self.value(x2).view(B, C, self.num_heads, self.head_dim, -1)  # Bx(num_heads)xhead_dimxCx(H2*W2)

        energy = torch.matmul(q, k)  # Bx(num_heads)x(H1*W1)x(H2*W2)
        attention = F.softmax(energy, dim=-1)  # Bx(num_heads)x(H1*W1)x(H2*W2)

        out = torch.matmul(v, attention).permute(0, 2, 4, 3, 1).contiguous()  # Bxhead_dimxCx(num_heads)x(H2*W2)
        out = out.view(B, C, H1, W1)  # BxCxH1xW1
        return out

# Assuming M and N are your tensors of size BxCx120x160
B, C, H, W = M.shape

# Patch sizes
M_patch_size = 4
N_patch_size = 8

# Extract patches
M_patches = M.unfold(2, M_patch_size, M_patch_size).unfold(3, M_patch_size, M_patch_size)
N_patches = N.unfold(2, N_patch_size, N_patch_size).unfold(3, N_patch_size, N_patch_size)

# Initialize attention modules
self_attention = SelfAttention(C, num_heads=8)
cross_attention = CrossAttention(C, num_heads=8)

# Apply attention to M and N
# M_self_attention = self_attention(M)
# N_self_attention = self_attention(N)
M_cross_attention = cross_attention(M_patches, N_patches)
N_cross_attention = cross_attention(N_patches, M_patches)

# Reshape the attention outputs back to their original sizes
# M_self_attention = M_self_attention.contiguous().view(B, C, H, W)
# N_self_attention = N_self_attention.contiguous().view(B, C, H, W)
M_cross_attention = M_cross_attention.contiguous().view(B, C, H, W)
N_cross_attention = N_cross_attention.contiguous().view(B, C, H, W)

# # Combine attentions using element-wise addition or any other fusion method
# combined_M = M + M_self_attention + M_cross_attention
# combined_N = N + N_self_attention + N_cross_attention
