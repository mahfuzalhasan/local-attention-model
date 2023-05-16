import torch
import torch.nn as nn

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x)
        print("x1: ",x.shape)
        x = x.view(b, -1, new_h, new_w)
        print("view: ",x.size())
        x = x.permute(0, 2, 3, 1)
        print("p: ",x.size())
        
        x = self.linear(x)

        return x

import numpy as np

# Create a 12x12 array

def patchify(arr, patch_size):
    patches = arr.reshape(arr.shape[0], arr.shape[1] // patch_size, patch_size, arr.shape[2] // patch_size, patch_size)
    print('patches shape: ',patches.shape)
    patches = patches.swapaxes(2, 3)
    print('patches shape after swap axes: ',patches.shape)
    patches = patches.reshape(arr.shape[0], -1, patch_size, patch_size)
    print('patches reshape: ',patches.shape)
    return patches

def patchify_torch(arr, patch_size):
    # arr = torch.from_numpy(arr)
    # Reshape the tensor into patches
    patches = arr.view(arr.shape[0], arr.shape[1] // patch_size, patch_size, arr.shape[2] // patch_size, patch_size)
    patches = patches.permute(0, 1, 3, 2, 4).contiguous()
    patches = patches.view(arr.shape[0], -1, patch_size, patch_size)
    return patches



if __name__=="__main__":
    # x = torch.randn(2, 3, 8, 8)
    # patchMerging = PatchMerging(3, 24, 2)
    # y = patchMerging(x)
    # print(y.shape)

    # q = torch.randn(8, 38400, 16)
    # k = torch.randn(8, 38400, 16)

    # attn = q @ k.transpose(-2, -1)
    # print(atten)
    B = 1
    H = 6
    W = 6

    patch_size = 2
    print("###########numpy operations###########")
    Q = np.ones((B, H, W))
    Q[:, 0, 0] = 2
    Q[:, 1, 0] = 0
    Q[:, 1, 1] = -1

    Q[:, 0, 2] = 1
    Q[:, 0, 3] = -5
    Q[:, 1, 2] = 0

    Q[:, 0, 4] = 0
    Q[:, 0, 5] = 1
    Q[:, 1, 4] = -1
    Q[:, 1, 5] = 0
    Q[:, 5, :] = 0
    print('Q')
    print(Q)
    Q_p = patchify(Q, patch_size)
    print("Q patch:")
    print(Q_p)
    Q_s = Q_p.reshape(1, 3, 3, 2, 2)
    Q_s = Q_s.swapaxes(2, 3)
    print('Q_s: ',Q_s.shape)
    Q_r = Q_s.reshape(Q.shape[0], H, W)
    print("Q_r: ",Q_r.shape)
    print(Q_r)

    K = np.zeros((B, H, W))
    K[:, 0, 0] = 2
    K[:, 1, 0] = 2
    K[:, 1, 1] = 4

    K[:, 0, 2] = 3
    K[:, 0, 3] = 2
    K[:, 1, 2] = 1

    K[:, 0, 4] = -1
    K[:, 0, 5] = -2
    K[:, 1, 4] = 9
    K[:, 1, 5] = 4
    print('K')
    print(K)
    QK = Q @ K
    print('QK')
    print(QK)
    mat_QK = np.matmul(Q, K)
    torch_Q = torch.from_numpy(Q)
    torch_K = torch.from_numpy(K)
    

    # K = K.transpose(0, 2, 1)
    # print('Transpose')
    # print(K)
    
    K_p = patchify(K, patch_size)
    print("K patch:",K_p.shape)
    print(K_p)
    K_s = K_p.reshape(1, 3, 3, 2, 2)
    
    K_s = K_s.swapaxes(2, 3)
    print('K_s: ',K_s.shape)
    K_r = K_s.reshape(K.shape[0], H, W)
    print("K_r: ",K_r.shape)
    print(K_r)
    K_p = np.transpose(K_p, (0, 1, 3, 2))
    mul = Q_p @ K_p
    print("mul")
    print(mul, mul.shape)
    mul_s = mul.reshape(1, 3, 3, 2, 2)
    mul_s = mul_s.swapaxes(2, 3)
    print('mul_s: ',mul_s.shape)
    mul_r = mul_s.reshape(mul.shape[0], H, W)
    print('mul reshape')
    print(mul_r, mul_r.shape)
    print("####################################")

    print("###########pyTorch operations###########")
    Q = torch.randn((B, H, W))
    # Q[:, 0, 0] = 2
    # Q[:, 1, 0] = 0
    # Q[:, 1, 1] = -1

    # Q[:, 0, 2] = 1
    # Q[:, 0, 3] = -5
    # Q[:, 1, 2] = 0

    # Q[:, 0, 4] = 0
    # Q[:, 0, 5] = 1
    # Q[:, 1, 4] = -1
    # Q[:, 1, 5] = 0
    # Q[:, 5, :] = 0
    # print('Q')
    # print(Q)
    Q_p = patchify_torch(Q, patch_size)
    # print("Q patch:")
    # print(Q_p)

    K = torch.randn((B, H, W))
    # K[:, 0, 0] = 2
    # K[:, 1, 0] = 2
    # K[:, 1, 1] = 4

    # K[:, 0, 2] = 3
    # K[:, 0, 3] = 2
    # K[:, 1, 2] = 1

    # K[:, 0, 4] = -1
    # K[:, 0, 5] = -2
    # K[:, 1, 4] = 9
    # K[:, 1, 5] = 4
    # print('K')
    # print(K)

    K_p = patchify_torch(K, patch_size)
    # print("K patch:")
    # print(K_p)

    mul = Q_p @ K_p.transpose(-2, -1)

    print('mul')
    print(mul)
    # mul_s = mul.view(1, 3, 3, 2, 2)
    # mul_s = mul_s.permute(0, 1, 3, 2, 4)
    # print('mul_s: ',mul_s.shape)
    # mul_r = mul_s.reshape(mul.shape[0], H, W)
    # print('mul reshape')
    # print(mul_r, mul_r.shape)

    mul = mul.softmax(dim=3)      #  couldn't figure out yet
    print('mul Soft')
    print(mul)
    # attn = self.attn_drop(attn)
    print("####################################")






