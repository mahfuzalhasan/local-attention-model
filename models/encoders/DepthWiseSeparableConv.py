import torch.nn as nn

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 5, padding = 2, dilation = 1, num_heads = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        # padding = (kernel_size + (kernel_size-1)*(dilation-1) - 1)//2
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, dilation=dilation, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.num_heads = num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        out = self.depthwise(x)
        out = self.pointwise(out)
        # print('depthwise output: ',out.shape)
        out = out.reshape(B, C, N).reshape(B, self.num_heads, C // self.num_heads, N ).permute(0, 1, 3, 2)
        return out