import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

__all__ = ['myConvTranspose2d', 'myBatchNorm2d', 'EqualConv2d']

class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        # self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, x):
        in_channel = x.shape[1]
        #print('conv',x.shape[1])
        weight = self.weight
        bbias = self.bias

        if hasattr(self, 'first_k_oup') and self.first_k_oup is not None:
            weight = weight[:self.first_k_oup]
            if self.bias!= None:
                bbias = bbias[:self.first_k_oup]


        weight = weight[:, :in_channel].contiguous()  # index sub channels for inference

        out = F.conv2d(
            x,
            weight,
            bias=bbias,
            stride=self.stride,
            padding=self.padding,
        )
        
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

class myConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
            
    def _output_padding(self, input, output_size):
        if output_size is None:
            return self.output_padding

        output_size = list(output_size)
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[-2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                .format(k, k + 2, len(output_size)))

        def dim_size(d):
            return ((input.size(d + 2) - 1) * self.stride[d] -
                    2 * self.padding[d] + self.kernel_size[d])

        min_sizes = [dim_size(d) for d in range(k)]
        max_sizes = [min_sizes[d] + self.stride[d] - 1 for d in range(k)]
        for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
            if size < min_size or size > max_size:
                raise ValueError((
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})").format(
                        output_size, min_sizes, max_sizes, input.size()[2:]))

        return tuple([output_size[d] - min_sizes[d] for d in range(k)])

    def forward(self, input, output_size=None):
        # print(input.shape[1])
        in_channel = input.shape[1]
        output_padding = self._output_padding(input, output_size)
        weight = self.weight
        
        bbias = self.bias
        if hasattr(self, 'first_k_oup') and self.first_k_oup is not None:
            weight = weight[:,:self.first_k_oup,:,:]
            if self.bias!= None:
                bbias = bbias[:self.first_k_oup]
        weight = weight[:in_channel].contiguous()

        return F.conv_transpose2d(
            input, weight, bbias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class myBatchNorm2d(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.gamma = nn.Parameter(torch.ones(1,channel,1,1))
        self.beta = nn.Parameter(torch.zeros(1,channel,1,1))



    def forward(self, x):
        results = 0.
        eps = 1e-5
        gamma = self.gamma
        beta = self.beta
        if hasattr(self, 'first_k_oup') and self.first_k_oup is not None:
            gamma = self.gamma[:,:self.first_k_oup].contiguous()
            beta = self.beta[:,:self.first_k_oup].contiguous()
        else:
            gamma = gamma[:,:self.channel].contiguous()
            beta = beta[:,:self.channel].contiguous()
        

        x_mean = torch.mean(x, axis=(0,2,3), keepdims=True)
        x_var = torch.var(x, axis= (0,2,3), unbiased=False, keepdims=True)# not bayles var
        x_normalized = (x - x_mean) / torch.sqrt(x_var + eps)
        results = gamma * x_normalized + beta
        return results