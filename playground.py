# import torch

# # Assuming you have tensors M and N with sizes BxCx120x160
# B = 8
# C = 32
# M = torch.randn(B, C, 120, 160)
# N = torch.randn(B, C, 120, 160)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Assuming you have tensors M and N with sizes BxCx120x160

# # Size of patches in M
# patch_size_m = 4

# # Size of patches in N
# patch_size_n = 8

# # Calculate the number of patches in each dimension for M and N
# num_patches_m = (M.size(2) // patch_size_m, M.size(3) // patch_size_m)
# num_patches_n = (N.size(2) // patch_size_n, N.size(3) // patch_size_n)

# # Extract patches from M with size B x C x num_patches_m[0] x patch_size_m x num_patches_m[1] x patch_size_m
# M_patches = M.unfold(2, patch_size_m, patch_size_m).unfold(3, patch_size_m, patch_size_m)




# # Extract patches from N with size B x C x num_patches_n[0] x patch_size_n x num_patches_n[1] x patch_size_n
# N_patches = N.unfold(2, patch_size_n, patch_size_n).unfold(3, patch_size_n, patch_size_n)
# print(f'patches M:{M_patches.shape} N:{N_patches.shape}')

# patch_conv = nn.Conv2d(32, 32, 4, stride=4, padding=0)

# M = patch_conv(M)
# M = F.adaptive_avg_pool2d(M, (N_patches.shape[2], N_patches.shape[3]))
# print(f' M:{M.shape}')

# import torch

# # Assuming M and N are your tensors of size BxCx120x160
# # and BxCx120x160 respectively

# # Patch sizes
# M_patch_size = 4
# N_patch_size = 8

# # Reshape M and N into patches
# M_patches = M.unfold(2, M_patch_size, M_patch_size).unfold(3, M_patch_size, M_patch_size)
# N_patches = N.unfold(2, N_patch_size, N_patch_size).unfold(3, N_patch_size, N_patch_size)

# # Get the shapes of M_patches and N_patches
# B, C, M_patches_H, M_patches_W, _, _ = M_patches.shape
# _, _, N_patches_H, N_patches_W, _, _ = N_patches.shape

# # Expand the dimensions of M_patches to match the dimensions of N_patches for broadcasting
# M_patches = M_patches.unsqueeze(4).unsqueeze(5)
# M_patches = M_patches.expand(-1, -1, -1, -1, N_patches_H, N_patches_W, -1, -1)

# # Expand the dimensions of N_patches to match the dimensions of M_patches for broadcasting
# N_patches = N_patches.unsqueeze(2).unsqueeze(3)
# N_patches = N_patches.expand(-1, -1, M_patches_H, M_patches_W, -1, -1, -1, -1)
# print(M_patches.shape, N_patches.shape)
# # Element-wise multiplication
# result = M_patches * N_patches

# # Reshape the result tensor back to BxCx120x160
# result = result.contiguous().view(B, C, M_patches_H * N_patches_H, M_patches_W * N_patches_W)

# # At this point, 'result' will contain the element-wise multiplication of each 4x4 patch of M
# # with all 4x4 patches from the corresponding 8x8 patch of N.
import numpy as np
# s_y = -40
# s_x = 320
# e_y = 600
# e_x = 800

# img_pad = np.zeros((600, 800, 3))

# img_sub = img_pad[s_y:e_y, s_x: e_x, :]

# print('img sub: ',img_sub.shape)
import torch
from torchmetrics import JaccardIndex


# H = 513
# W = 513
# B = 8
# C = 19
# jaccard = JaccardIndex(task="multiclass", num_classes=19)

# target = torch.randint(0, 19, (B, H, W))
# target[B-1, 10:20, 0:3] = 255
# pred = torch.randn(B, C, H, W)
# print(pred.shape, target.shape)

# miou = jaccard(pred, target)
# print(miou)


def cal_mean_iou(pred, target):
    score = torch.exp(pred) # B, C, H, W
    jaccard = JaccardIndex(task="multiclass", num_classes=score.shape[1], ignore_index = 255).to(score.get_device())    
    
    mean_iou = jaccard(score, target)
    print('mean iou: ',mean_iou)
    return mean_iou.detach().cpu().numpy()

import torch
import torch.nn as nn
import torch.nn.functional as F

height = 4
width = 4
channel = 1
q = torch.randint(0, 9, (channel, height, width))
k = torch.randint(0, 9, (channel, height, width))
v = torch.randint(0, 9, (channel, height, width))


kernel_size = 3

k = F.pad(k, [(kernel_size-1)//2, (kernel_size-1)-((kernel_size-1)//2), (kernel_size-1)//2, (kernel_size-1)-((kernel_size-1)//2)])
v = F.pad(v, [(kernel_size-1)//2, (kernel_size-1)-((kernel_size-1)//2), (kernel_size-1)//2, (kernel_size-1)-((kernel_size-1)//2)])
# print('$$$$$$$padded k $$$$$')
print('####q####' )
print(q, q.shape)
print('####padded k####')
print(k, k.shape)
k = k.unfold(1, kernel_size, 1).unfold(2, kernel_size, 1)
v = v.unfold(1, kernel_size, 1).unfold(2, kernel_size, 1)
print('######### After Folding k ##############')
print(k, k.shape)
# exit()

k = k.reshape(height, width, channel, -1)
v = v.reshape(height, width, channel, -1)
q = q.reshape(height, width, channel, 1)
print('@@@@@@@@ After Reshape @@@@@@@@@@@')
print('q, k ,v: ',q.shape, k.shape, v.shape)
print('####q####' )
print(q)
print('####k####' )
print(k)

qk = torch.matmul(q.transpose(2, 3), k)
print('qk initial: ',qk.shape)

qk = qk.reshape( height, width, kernel_size, kernel_size)
print('qk: ',qk.shape, qk)


