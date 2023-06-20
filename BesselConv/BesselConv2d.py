"""
----
Description:
    Implementation of a 2-d Bessel Convolutional Layer using PyTorch.
    This layer can be made S-O(2)-invariant thanks to the use of the
    Bessel functions of the first kind.
----
           Author: Valentin Delchevalerie
         Creation: 19-06-2023
Last modification: 19-06-2023
----
"""


import numpy as np
import torch
from torch import nn
import einops
from .getTransMat import getTransMat


class BesselConv2d(nn.Module):
    """
    Main class: define the BesseConv2d layer
    """
    def __init__(self, k, C_in, C_out, strides=1, padding='VALID', reflex_inv=False, scale_inv=False, 
                 scales=[-2,0,2], activation=None, TensorCorePad=True, name=None, **kwargs):
        """
        Initialization of the layer. Called only once, before any training.

        * k is the size of the sliding window used for the convolution
        * C_out is the number of filters in the layer
        * strides is the classic strides parameters used in convolution
        * k_max is the maximum value of k_mj considered for the Bessel
          decomposition of the image. If k_max is set to 'auto', then a 
          value is infered based on k
        * padding is the classic padding parameters used in convolution
        * reflex_inv can be set to True in order to add invariance to reflections
        * activation is the activation function used on the output of the layer
          Available activations are ['relu', 'sigmoid', 'tanh', None]
        * When TensorCorePad is set to True, then padding is used to make mod(m_max+1,8)
          and mod(j_max+1,8) equal to 0. It makes possible for TF to use Tensor Cores.
          Note that this is only possible if mixed precision is also used.
        """
        super().__init__()

        """
        Initialization
        """

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if k % 2 == 0:
            ValueError("Kernel size 'k' should be an odd number.")
        if not isinstance(C_in, int) or C_in < 1:
            ValueError("'C_in' should be an integer > 0")
        if not isinstance(C_out, int) or C_out < 1:
            ValueError("'C_out' should be an integer > 0")
        if not isinstance(strides, int) or strides < 1:
            ValueError("'strides' should be an integer > 0")
        if padding not in ['VALID', 'SAME']:
            ValueError("'padding' should be 'VALID' or 'SAME'")
        if reflex_inv not in [True, False]:
            ValueError("'reflex_inv' should be set to True or False")
        if scale_inv not in [True, False]:
            ValueError("'scale_inv' should be set to True or False")
        if activation not in ['relu', 'sigmoid', 'tanh', None]:
            ValueError("'activation' should be 'relu', 'sigmoid', 'tanh' or None")
        if TensorCorePad not in [True, False]:
            ValueError("'TensorCorePad' should be set to True or False")

        self.k = k
        self.C_in = C_in
        self.C_out = C_out
        self.strides = strides
        self.padding = padding
        self.reflex_inv = reflex_inv
        self.scale_inv = scale_inv
        self.scales = scales
        self.activation = activation
        self.TensorCorePad = TensorCorePad
        
        if self.scale_inv == False:
            self.scales = [0]

        """
        Building the layer
        """

        # Get the transformation matrices
        k_max = np.pi * (self.k / 2.)
        self.all_T = []
        for scale in self.scales:

            transMat, MASK = getTransMat(self.k+scale, k_max, self.TensorCorePad)
            self.m_max = MASK.shape[0] - 1
            self.j_max = MASK.shape[1] - 1

            # Add the transformation matrix to the list
            transMat = np.reshape(
                np.transpose(transMat, (2,0,1,3)),
                (self.m_max+1, (self.k+scale)**2, self.j_max+1)
            )
            self.all_T.append(torch.from_numpy(transMat).to(self.device))

        # Define the weights
        fan_in = self.C_in * (self.k**2)

        w_r_ini = np.random.normal(size=(self.m_max+1, self.j_max+1, self.C_in * self.C_out), loc=0., scale=np.sqrt(2./fan_in))
        # Remove parameters when k_mj > k_max
        for m in range(MASK.shape[0]):
            for j in range(MASK.shape[1]):
                if not MASK[m,j]:
                    # Value should not be used
                    for _i in range(w_r_ini.shape[-1]):
                        w_r_ini[m,j,_i] = 0.
                    continue

        self.w_r = nn.Parameter(
            torch.from_numpy(w_r_ini).type(torch.float32).to(self.device),
            requires_grad=True
        )

        w_i_ini = np.random.normal(size=(self.m_max+1, self.j_max+1, self.C_in * self.C_out), loc=0., scale=np.sqrt(2./fan_in))
        # Remove parameters when k_mj > k_max
        # For m = 0, no imaginary part
        for _i in range(w_i_ini.shape[1]):
            for _j in range(w_i_ini.shape[2]):
                w_i_ini[0,_i,_j] = 0.
        # For m > 0
        for m in range(1, MASK.shape[0]):
            for j in range(MASK.shape[1]):
                if not MASK[m,j]:
                    # Value should not be used
                    for _i in range(w_i_ini.shape[-1]):
                        w_i_ini[m,j,_i] = 0.
                    continue

        self.w_i = nn.Parameter(
            torch.from_numpy(w_i_ini).type(torch.float32).to(self.device),
            requires_grad=True
        )

        # Define the bias
        self.b = nn.Parameter(
            torch.Tensor(self.C_out).type(torch.float32).to(self.device),
            requires_grad=True
        )

        # Initialize the bias
        bound = 1. / np.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)

        # Get the number of parameters
        # For m = 0, no imaginary part
        n_zeros = (self.j_max+1)*(self.C_in*self.C_out)
        # Checking 0's for the real part
        for j in range(1, self.j_max+1):
            if not MASK[0,j]:
                n_zeros += self.C_in*self.C_out
        # For m > 0
        for m in range(1, self.m_max+1):
            if not MASK[m,0]:
                n_zeros += 2*self.C_in*self.C_out*(self.j_max+1)
                continue
            for j in range(1, self.j_max+1):
                if not MASK[m,j]:
                    n_zeros += 2*self.C_in*self.C_out

        self.n_params = 2*(self.m_max+1)*(self.j_max+1)*self.C_in*self.C_out - n_zeros + self.C_out

    
    def forward(self, x):
        """
        This function computes the activation of the layer given (a) particular input(s).
        inputs is of shape (N, C_in, W, H).
        """

        # ----
        # Updating filters using the transformation matrix
        # ----

        all_a = []
        for i, scale in enumerate(self.scales):

            # conv2d only takes 4-d tensors as input.
            # n_out, m_max, real and imag are then wrapped together before performing convolutions.
            # They will be unwrapped later.

            if not self.reflex_inv:

                w_r = torch.add(
                    torch.matmul(self.all_T[i].real, self.w_r),
                    -torch.matmul(self.all_T[i].imag, self.w_i)
                )
                w_i = torch.add(
                    torch.matmul(self.all_T[i].real, self.w_i),
                    torch.matmul(self.all_T[i].imag, self.w_r)
                )

                self.w = einops.rearrange(
                    [w_r, w_i], 
                    'c m (k1 k2) (b1 b2) -> (c m b2) b1 k1 k2', 
                    k1=self.k+scale, k2=self.k+scale, b1=self.C_in, c=2, m=self.m_max+1, b2=self.C_out
                )

                # ----
                # Computation of the activation.
                # ----

                if self.padding == 'VALID' and scale > 0:
                    pad = scale // 2
                    output = torch.square(
                        torch.nn.functional.conv2d(x[:,:,:,:], self.w[:,:,:,:], padding=(pad,pad), stride=self.strides)
                    )
                elif self.padding == 'VALID' and scale < 0:
                    pad = scale // 2
                    output = torch.square(
                        torch.nn.functional.conv2d(x[:,-pad:pad,-pad:pad,:], self.w[:,:,:,:], padding='valid', stride=self.strides)
                    )
                else:
                    output = torch.square(
                        torch.nn.functional.conv2d(x[:,:,:,:], self.w[:,:,:,:], padding='same', stride=self.strides)
                    )

            else:

                _w_r = torch.matmul(self.all_T[i].real, self.w_r)
                _w_i = torch.matmul(self.all_T[i].imag, self.w_r)

                w_r = einops.rearrange(
                    [_w_r, _w_i],
                    'c1 m (k1 k2) (b1 b2) -> k1 k2 b1 (c1 m b2)',
                    c1=2, k1=self.k+scale, k2=self.k+scale, b1=self.C_in, m=self.m_max+1, b2=self.C_out
                )

                _w_r = torch.matmul(self.all_T[i].real, self.w_i)
                _w_i = torch.matmul(self.all_T[i].imag, self.w_i)

                w_i = einops.rearrange(
                    [_w_r, _w_i],
                    'c1 m (k1 k2) (b1 b2) -> k1 k2 b1 (c1 m b2)',
                    c1=2, k1=self.k+scale, k2=self.k+scale, b1=self.C_in, m=self.m_max+1, b2=self.C_out
                )

                self.w = einops.rearrange(
                    [w_r,w_i], 
                    'c2 k1 k2 b1 (c1 m b2) -> (c1 m b2) b1 k1 k2 c2', 
                    c1=2, k1=self.k+scale, k2=self.k+scale, b1=self.C_in, c2=2, m=self.m_max+1, b2=self.C_out
                )

                # ----
                # Computation of the activation.
                # ----

                if self.padding == 'VALID' and scale > 0:
                    pad = scale // 2
                    output = torch.add(
                        torch.square(
                            torch.nn.functional.conv2d(x[:,:,:,:], self.w[:,:,:,:,0], padding=(pad,pad), stride=self.strides)
                        ),
                        torch.square(
                            torch.nn.functional.conv2d(x[:,:,:,:], self.w[:,:,:,:,1], padding=(pad,pad), stride=self.strides)
                        )
                    )
                elif self.padding == 'VALID' and scale < 0:
                    pad = scale // 2
                    output = torch.add(
                        torch.square(
                            torch.nn.functional.conv2d(x[:,-pad:pad,-pad:pad,:], self.w[:,:,:,:,0], padding='valid', stride=self.strides)
                        ),
                        torch.square(
                            torch.nn.functional.conv2d(x[:,-pad:pad,-pad:pad,:], self.w[:,:,:,:,1], padding='valid', stride=self.strides)
                        )
                    )
                else:
                    output = torch.add(
                        torch.square(
                            torch.nn.functional.conv2d(x[:,:,:,:], self.w[:,:,:,:,0], padding=self.padding.lower(), stride=self.strides)
                        ),
                        torch.square(
                            torch.nn.functional.conv2d(x[:,:,:,:], self.w[:,:,:,:,1], padding=self.padding.lower(), stride=self.strides)
                        )
                    )

            all_a.append(
                torch.add(
                    einops.reduce(
                        output, 'b (c m b1) w h -> b b1 w h', 'sum', 
                        w=output.shape[2], h=output.shape[3], c=2, m=self.m_max+1, b1=self.C_out
                    ),
                    self.b[None,:,None,None]
                )[:,:,:,:,None]
            )

        a = torch.cat(all_a, axis=-1)

        if self.scale_inv:
            idx = torch.argmax(torch.sum(a, dim=(1,2,3), keepdim=False), axis=-1)
            a = torch.gather(a, index=idx, dim=-1)
        else:
            a = a[:,:,:,:,0]

        if self.activation == 'relu':
            return nn.ReLU(a)
        elif self.activation == 'sigmoid':
            return nn.Sigmoid(a)
        elif self.activation == 'tanh':
            return nn.Tanh(a)
        else:
            return a