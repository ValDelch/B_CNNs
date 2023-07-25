import torch
from torch import nn


class AttentiveNorm2d(nn.BatchNorm2d):


    def __init__(self, num_features, n_mixtures=5, eps=1e-5, momentum=0.1, 
                 affine=True, track_running_stats=True, device=None, dtype=None):
        super(AttentiveNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

        self.n_mixtures = n_mixtures
        self.dense_layer = nn.Linear(num_features, self.n_mixtures)

        w = 0.1 * torch.normal(mean=torch.zeros(n_mixtures, num_features), std=torch.ones(n_mixtures, num_features)) + 1.
        b = 0.1 * torch.normal(mean=torch.zeros(n_mixtures, num_features), std=torch.ones(n_mixtures, num_features))
        self.learnable_weights = nn.Parameter(w, requires_grad=True)
        self.learnable_bias = nn.Parameter(b, requires_grad=True)


    def forward(self, x):
        attention = self.dense_layer(torch.mean(x, dim=(2, 3))) # N x K
        gamma_readjust = torch.matmul(attention, self.learnable_weights) # N x C
        beta_readjust = torch.matmul(attention, self.learnable_bias) # N x C

        out_BN = super(AttentiveNorm2d, self).forward(x) # N x C x H x W

        return out_BN * gamma_readjust[:, :, None, None] + beta_readjust[:, :, None, None]