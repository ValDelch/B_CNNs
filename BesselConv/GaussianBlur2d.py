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
        r = -torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.float32())

        # Build the gaussian kernel
        _filter = torch.exp(r / (2 * variance))

        # Normalize
        _filter /= torch.sum(_filter)

        # The filter needs to be reshaped to be used in 2d depthwise convolution
        self.filter = _filter.view(1, 1, filter_size, filter_size).repeat((C_in, 1, 1, 1)).to('cuda')

    def forward(self, x):

        print('before:', x.shape)
        output = nn.functional.conv2d(x, self.filter, stride=self.stride, padding=self.padding, groups=x.shape[1])
        print('after:', output.shape)

        return output