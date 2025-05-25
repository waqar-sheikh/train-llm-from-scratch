import math
import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)
    
    def forward(self, x):
        return self.weight[x]


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = 2 / (in_features + out_features)
        torch.nn.init.trunc_normal_(self.weight, 0, std, -3*std, 3*std)

    def forward(self, x):
        return x @ self.weight.T


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x):
        rms = (torch.sum(torch.square(x), dim=-1, keepdim=True) / self.d_model) + self.eps
        return (x / torch.sqrt(rms)) * self.weight


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x):
        out = torch.nn.functional.silu(self.w1(x))
        out *= (self.w3(x))
        return self.w2(out)


def softmax(x, dim):
    maximums = torch.max(x, dim=dim, keepdim=True).values
    scores = torch.exp(x - maximums)
    sums = torch.sum(scores, dim=dim, keepdim=True)
    scores = scores / sums
    return scores


def attention(q, k, v, mask):
    scores = q @ k.transpose(-2, -1)
    scores /= math.sqrt(k.shape[-1])
    scores = torch.where(mask, scores, float('-inf'))
    scores = softmax(scores, -1)
    return scores @ v


def apply_rope(x, token_positions=None, theta=10000.0):
    return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, apply_rope=False, theta=10000.0, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.apply_rope = apply_rope
        self.theta = theta
        self.d_head = d_model // num_heads
        self.device = device
        self.dtype = dtype

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)

        if self.apply_rope:
            q = apply_rope(q, token_positions, self.theta)
            k = apply_rope(k, token_positions, self.theta)

        mask = torch.tril(torch.ones(L, L, device=self.device, dtype=self.dtype)).bool().unsqueeze(0).unsqueeze(0)
        out = attention(q, k, v, mask).transpose(1, 2).reshape(B, L, self.d_model)
        return self.output_proj(out)