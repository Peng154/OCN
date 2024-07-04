import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, target_dim=None):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()
        self.target_dim = target_dim
        if self.target_dim is None:
            self.target_dim = list(range(num_features))

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        dim_num_err = x.dim() - self.stdev.dim()
        if dim_num_err == 1:
            self.stdev = self.stdev.unsqueeze(1)
            self.mean = self.mean.unsqueeze(1)
        if self.affine:
            # print(self.affine_weight.shape, self.affine_bias.shape, self.target_dim)
            x = x - self.affine_bias[self.target_dim]
            x = x / (self.affine_weight[self.target_dim] + self.eps*self.eps)
        x = x * self.stdev[..., self.target_dim]
        x = x + self.mean[..., self.target_dim]
        return x