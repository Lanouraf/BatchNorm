import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, filters, epsilon=1e-5, dims='2D'):
        """
        Implementation of the Batch Normalization block from:

        Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network
        training by reducing internal covariate shift. In International conference on machine
        learning, pages 448–456. PMLR, 2015.

        :param filters: Number of input filters
        :param epsilon: Constant for numerical stability
        :param dims: '1D' (B, C) or '2D' (B, C, H, W) implementation
        """
        super().__init__()
        if dims == '2D':
            self.shape=(1, filters, 1, 1)
            self.stats_axes=(0, 2, 3)
        else:  # dims == '1D':
            self.shape=(1, filters)
            self.stats_axes=(0)

        self.filters = filters
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(self.shape))
        self.beta = nn.Parameter(torch.zeros(self.shape))
        self.mu = nn.Parameter(torch.zeros(self.shape), requires_grad=False)  # running average
        self.var = nn.Parameter(torch.ones(self.shape), requires_grad=False)  # running variance

    def forward(self, x):
        if self.training:
            # Compute batch statistics
            var, mu = torch.var_mean(x, self.stats_axes, unbiased=False)
            mu = torch.reshape(mu, self.shape)
            var = torch.reshape(var, self.shape)

            # Update running statistics
            m = x.numel() / x.shape[1]  # Number of elements in batch (taking into account extra dimensions for 2D)
            avg_factor = 0.1  # momentum for running stats
            self.mu.data = avg_factor * mu + (1-avg_factor) * self.mu.data
            self.var.data = avg_factor * m / (m-1) * var + (1-avg_factor) * self.var.data  # Update with unbiased version of variance

            # Compute normalized output
            x_norm = (x - mu) / (torch.sqrt(var+self.epsilon))
            y = self.gamma*x_norm + self.beta
            return y
        else:
            # On inference, use running stats instead
            return self.gamma / torch.sqrt(self.var+self.epsilon) * x + (self.beta - self.gamma * self.mu / torch.sqrt(self.var + self.epsilon))
