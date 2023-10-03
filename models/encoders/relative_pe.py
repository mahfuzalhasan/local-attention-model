import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat


def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

class RelPosEmb(nn.Module):
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size       #(key and value patch size)
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def rel_to_abs(self, x):
        b, l, m = x.shape
        r = (m + 1) // 2

        col_pad = torch.zeros((b, l, 1), **to(x))
        x = torch.cat((x, col_pad), dim = 2)
        flat_x = rearrange(x, 'b l c -> b (l c)')
        flat_pad = torch.zeros((b, m - l), **to(x))
        flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
        final_x = flat_x_padded.reshape(b, l + 1, m)
        final_x = final_x[:, :l, -r:]
        return final_x

    def relative_logits_1d(self, q, rel_k):
        b, h, w, _ = q.shape
        r = (rel_k.shape[0] + 1) // 2
        # #print('r: ',r)
        # #print('rel_k: ',rel_k.shape)

        logits = einsum('b x y d, r d -> b x y r', q, rel_k)
        logits = rearrange(logits, 'b x y r -> (b x) y r')
        logits = self.rel_to_abs(logits)

        logits = logits.reshape(b, h, w, r)
        logits = expand_dim(logits, dim = 2, k = r)
        return logits

    def forward(self, q):
        #print(f'######## inside rel pos embed############## \n')
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        #print('q rearrange shape: ',q.shape)
        rel_logits_w = self.relative_logits_1d(q, self.rel_width)

        #print(f'rel_logits_w_reshape:{rel_logits_w.shape}')
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')
        #print(f'q:{q.shape} rel_logits_w_reshape:{rel_logits_w.shape}')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = self.relative_logits_1d(q, self.rel_height)
        #print(f'rel_logits_h_reshape:{rel_logits_h.shape}')
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        #print(f'q:{q.shape} rel_logits_h_reshape:{rel_logits_h.shape}')
        return rel_logits_w + rel_logits_h

if __name__=='__main__':
    rpe = RelPosEmb(2, 2, 64)
