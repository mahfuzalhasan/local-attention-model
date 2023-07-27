import torch

# Assuming you have tensors M and N with sizes BxCx120x160
B = 8
C = 32
M = torch.randn(B, C, 120, 160)
N = torch.randn(B, C, 120, 160)


import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming you have tensors M and N with sizes BxCx120x160

# Size of patches in M
patch_size_m = 4

# Size of patches in N
patch_size_n = 8

# Calculate the number of patches in each dimension for M and N
num_patches_m = (M.size(2) // patch_size_m, M.size(3) // patch_size_m)
num_patches_n = (N.size(2) // patch_size_n, N.size(3) // patch_size_n)

# Extract patches from M with size B x C x num_patches_m[0] x patch_size_m x num_patches_m[1] x patch_size_m
M_patches = M.unfold(2, patch_size_m, patch_size_m).unfold(3, patch_size_m, patch_size_m)




# Extract patches from N with size B x C x num_patches_n[0] x patch_size_n x num_patches_n[1] x patch_size_n
N_patches = N.unfold(2, patch_size_n, patch_size_n).unfold(3, patch_size_n, patch_size_n)
print(f'patches M:{M_patches.shape} N:{N_patches.shape}')

patch_conv = nn.Conv2d(32, 32, 4, stride=4, padding=0)

M = patch_conv(M)
M = F.adaptive_avg_pool2d(M, (N_patches.shape[2], N_patches.shape[3]))
print(f' M:{M.shape}')

import torch

# Assuming M and N are your tensors of size BxCx120x160
# and BxCx120x160 respectively

# Patch sizes
M_patch_size = 4
N_patch_size = 8

# Reshape M and N into patches
M_patches = M.unfold(2, M_patch_size, M_patch_size).unfold(3, M_patch_size, M_patch_size)
N_patches = N.unfold(2, N_patch_size, N_patch_size).unfold(3, N_patch_size, N_patch_size)

# Get the shapes of M_patches and N_patches
B, C, M_patches_H, M_patches_W, _, _ = M_patches.shape
_, _, N_patches_H, N_patches_W, _, _ = N_patches.shape

# Expand the dimensions of M_patches to match the dimensions of N_patches for broadcasting
M_patches = M_patches.unsqueeze(4).unsqueeze(5)
M_patches = M_patches.expand(-1, -1, -1, -1, N_patches_H, N_patches_W, -1, -1)

# Expand the dimensions of N_patches to match the dimensions of M_patches for broadcasting
N_patches = N_patches.unsqueeze(2).unsqueeze(3)
N_patches = N_patches.expand(-1, -1, M_patches_H, M_patches_W, -1, -1, -1, -1)
print(M_patches.shape, N_patches.shape)
# Element-wise multiplication
result = M_patches * N_patches

# Reshape the result tensor back to BxCx120x160
result = result.contiguous().view(B, C, M_patches_H * N_patches_H, M_patches_W * N_patches_W)

# At this point, 'result' will contain the element-wise multiplication of each 4x4 patch of M
# with all 4x4 patches from the corresponding 8x8 patch of N.
