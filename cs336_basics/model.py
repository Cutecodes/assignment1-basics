
import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        """带有初始化功能的线性层

        Args:
            d_in: int
                The number of input features.
            d_out: int
                The number of output features.
        """
        
        super().__init__()

        if device is None:
            device = 'cpu'
        if dtype is None:
            dtype = torch.float32

        std = math.sqrt(2 / (d_in + d_out))
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_out, d_in, device=device, dtype=dtype), 
                std=std, 
                a=-3*std, 
                b=3*std
            ),
            requires_grad=True,

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        if device is None:
            device = 'cpu'
        if dtype is None:
            dtype = torch.float32

        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), 
                std=1, 
                a=-3, 
                b=3
            ),
            requires_grad=True,

        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Args:
        hidden_size: int
            Dimensionality of the input to normalize.
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.

    Returns:
        FloatTensor of same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: FloatTensor of shape `(batch_size, *)`.
                The input to apply root mean square layer normalization on.
        Returns:
            FloatTensor of same shape as input
        """
        in_dtype = x.dtype

        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms

        return (self.weight * x).to(in_dtype)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x / (1 + torch.exp(-x))

class SwiGLU(nn.Module):
    pass