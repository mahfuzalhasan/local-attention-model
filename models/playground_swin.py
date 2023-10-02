import torch
import torch.nn as nn
import numpy as np
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    # print('\n ^^^^^ Window Partition ^^^^^ \n')
    B, H, W, C = x.shape
    #print(x.shape)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5)
    print(f'x after windowing:{windows.shape}')
    windows = windows.contiguous().view(-1, window_size, window_size, C)
    return windows

H, W = 9, 9
B = 1
C = 1
img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
window_size = 3
shift_size = 2
h_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
w_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
print(f'hslices:################# \n{h_slices}')
cnt = 0
for h in h_slices:
    print('##########h:########## ',h)
    for w in w_slices:
        print('w: ',w)
        img_mask[h, w] = cnt
        print(img_mask)
        # if cnt > 0:
        #     exit()
        cnt += 1
print(img_mask, img_mask.shape)

mask_windows = window_partition(img_mask, window_size)

print('mask windows: ',mask_windows.shape)

mask_windows = mask_windows.view(-1, window_size * window_size)
print('mask windows view: ',mask_windows.shape)
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
print(f'$$$$$$attn mask:\n {attn_mask} \n shape:{attn_mask.shape}')
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
print(f'$$$$$$attn mask final:\n {attn_mask} \n shape:{attn_mask.shape}')


relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), 4))

print(relative_position_bias_table, relative_position_bias_table.shape)


coords_h = torch.arange(window_size)
coords_w = torch.arange(window_size)
print(coords_h, coords_w)
coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
print(coords)
coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
print(coords_flatten)
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
print('relative coords \n ', relative_coords)

relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
relative_coords[:, :, 1] += window_size - 1
relative_coords[:, :, 0] *= 2 * window_size - 1
relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
print(relative_coords)

print('index: \n',relative_position_index)
