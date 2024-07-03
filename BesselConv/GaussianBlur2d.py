"""
----
Description:
    Implementation of a 2-d Gaussian blur filter to avoid aliasing
    when using B-CNNs or similar models.

    Inspired from https://github.com/QUVA-Lab/e2cnn/blob/master/e2cnn/nn/modules/pooling/pointwise_avg.py#L123
----
           Author: Valentin Delchevalerie
         Creation: 28-06-2024
Last modification: 28-06-2024
----
"""
"""
from typing import List, Tuple, Any, Union

import torch
from torch import nn

class GaussianBlur2d(nn.Module):

    def __init__(self,
                 C_in: int,
                 sigma: float,
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]] = None,):
        
        super(GaussianBlur2d, self).__init__()
        
        assert sigma > 0.
        
        filter_size = 2*int(round(3*sigma))+1
        
        self.kernel_size = (filter_size, filter_size)
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

        if padding is None:
            padding = int((filter_size-1)//2)
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        # Build the Gaussian smoothing filter
        grid_x = torch.arange(filter_size).repeat(filter_size).view(filter_size, filter_size)
        grid_y = grid_x.t()
        grid = torch.stack([grid_x, grid_y], dim=-1)

        mean = (filter_size - 1) / 2.
        variance = sigma ** 2.

        # setting the dtype is needed, otherwise it becomes an integer tensor
        r = -torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.float32)

        # Build the gaussian kernel
        _filter = torch.exp(r / (2 * variance))

        # Normalize
        _filter /= torch.sum(_filter)

        # The filter needs to be reshaped to be used in 2d depthwise convolution
        self.filter = _filter.view(1, 1, filter_size, filter_size).repeat((C_in, 1, 1, 1)).to('cuda')

    def forward(self, x):

        output = nn.functional.conv2d(x, self.filter, stride=self.stride, padding=self.padding, groups=x.shape[1])

        return output
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianBlur2d(nn.Module):
    def __init__(self, sigma, C_in=1):
        super(GaussianBlur2d, self).__init__()
        
        self.sigma = sigma
        k = 2 * int(round(3 * sigma)) + 1
        self.C_in = C_in

        self.register_buffer('w', self._gaussian_kernel(k, self.sigma, C_in))
        
    def _gaussian_kernel(self, kernel_size, sigma, n_channels):
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        g = torch.exp(-(x**2) / (2 * sigma**2))
        g_norm2d = torch.sum(g)**2
        g_kernel = torch.outer(g, g) / g_norm2d
        g_kernel = g_kernel.unsqueeze(0).unsqueeze(0)
        return g_kernel.expand(n_channels, 1, -1, -1)
    
    def forward(self, inputs):
        batch_size, channels, height, width = inputs.shape
        a = F.conv2d(inputs, self.w, groups=channels, padding='same')
        return a

# Example usage:
# blur_layer = GaussianBlur2d(sigma=1.5, C_in=3)
# input_tensor = torch.randn(1, 3, 224, 224)  # Example input
# output = blur_layer(input_tensor)