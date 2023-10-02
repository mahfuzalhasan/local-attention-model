import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat


height = 4
width = 4
channel = 1
batch = 1
block = 2
kernel_size = 2
halo = 1
x = torch.randint(0, 9, (batch, channel, height, width)).float()
print(f'############### x:\n {x}')

q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = block, p2 = block)
print(f'############### q_inp:\n {q_inp}\n shape: {q_inp.shape}')

k = F.pad(x, [(kernel_size-1)//2, (kernel_size-1)-((kernel_size-1)//2), (kernel_size-1)//2, (kernel_size-1)-((kernel_size-1)//2)])
print(f'$$$$$$$$$$$ k:\n {k} \n shape:{k.shape} $$$$$$$$$$$$$')
k = k.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
print(f'$$$$$$$$$$$ k unfold:\n {k} \n shape:{k.shape} \n $$$$$$$$$$$$$')
print(f'#### \n {x} \n')
kv_inp = F.unfold(x, kernel_size = block + halo * 2, stride = block, padding = halo)
print(f'!!!!!!!!!!!!! kv halo: \n{kv_inp} \n shape:{kv_inp.shape}!!!!!!!!!!!!!!')
kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c = channel)
print(f'##########kv_inp reshape###########\n\n shape:{kv_inp.shape}')

q, k = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = 1), (q_inp, kv_inp))

print(q.shape, k.shape)
sim = einsum('b i d, b j d -> b i j', q, k)
print(sim, sim.shape)