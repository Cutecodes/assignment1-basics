
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

def silu(x):
    return x / (1 + torch.exp(-x))

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        position = torch.arange(max_seq_len).float()
        sinusoid = torch.outer(position, freq)
        self.register_buffer("cos", sinusoid.cos())
        self.register_buffer("sin", sinusoid.sin())

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2] 
        x2 = x[..., 1::2] 
        
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        output = torch.empty_like(x)
        output[..., 0::2] = x1_rot
        output[..., 1::2] = x2_rot
        
        return output

def safe_softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_safe = x - x_max
    
    exp_x = torch.exp(x_safe)

    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def attention(q, k, v, attn_mask):
    L, S = q.size(-2), k.size(-2)
    scale_factor = 1 / math.sqrt(q.size(-1))

    attn_bias = torch.zeros(q.shape[:-2] + (L, S), dtype=q.dtype, device=q.device)

    if attn_mask is not None:
        attn_bias = attn_bias.masked_fill(attn_mask == False, float("-inf"))
    
    attn_scores = q @ k.transpose(-2, -1) * scale_factor
    attn_scores += attn_bias
    attn_weights = torch.softmax(attn_scores, dim=-1)

    output = attn_weights @ v

    return output

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, positional_encoder=None):
        super().__init__()

        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.q_proj = Linear(self.d_model, self.num_heads * self.head_dim)
        self.k_proj = Linear(self.d_model, self.num_kv_heads * self.head_dim)
        self.v_proj = Linear(self.d_model, self.num_kv_heads * self.head_dim)
        self.o_proj = Linear(self.d_model, self.d_model)

        self.positional_encoder = positional_encoder
    
    def forward(self, x, token_positions = None):
        batch_size, seq_length, hidden_dim = x.size()
        if token_positions is None:
            token_positions = torch.arange(seq_length, device=x.device)
        q = self.q_proj(x) # [batch_size, seq_length, num_heads*head_dim]
        k = self.k_proj(x) # [batch_size, seq_length, num_key_values*head_dim]
        v = self.v_proj(x) # [batch_size, seq_length, num_key_values*head_dim]
        
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)       # [batch_size, seq_length, num_heads, head_dim]
        k = k.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)  # [batch_size, seq_length, num_key_values, head_dim]
        
        q = q.transpose(1, 2)                                                                   # [batch_size, num_heads, seq_length, head_dim]
        k = k.transpose(1, 2)                                                                  # [batch_size, num_key_values, seq_length, head_dim]
        if self.positional_encoder:
            q = self.positional_encoder(q, token_positions)
            k = self.positional_encoder(k, token_positions)

        v = v.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1,2)   # [batch_size, num_key_values, seq_length, head_dim]
        
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        
        causal = True if q.size(2) == k.size(2) else False # During decoding phase. The lenghth of q is usually 1. 
        
        attn_mask = None
        if causal:
            attn_mask = torch.ones((seq_length,seq_length), dtype=torch.bool, device=q.device).tril(diagonal=0)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        
        out = attention(q, k, v, attn_mask)
        out = out.transpose(1, 2).reshape(batch_size, seq_length, hidden_dim)

        return self.o_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, positional_encoder = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.positional_encoder = positional_encoder
        
        self.ln1 = RMSNorm(d_model)
        self.attn = Attention(d_model, num_heads, num_heads, positional_encoder)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, x):
        x_attn = self.attn(self.ln1(x))
        attn_sublayer_output = x + x_attn

        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int, 
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        positional_encoder = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.positional_encoder = positional_encoder

        self.token_embeddings = Embedding(vocab_size, d_model)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, positional_encoder)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.token_embeddings(x)

        for layer in self.transformer_layers:
            x = layer(x)
        
        x = self.ln_final(x)

        return self.lm_head(x)
    