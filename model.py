import torch
from layers import *

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, apply_rope=False, theta=10000.0, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, num_heads, apply_rope=apply_rope, theta=theta, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, input):
        x1 = self.ln1(input)
        x2 = self.attn(x1) + input
        x3 = self.ln2(x2)
        return self.ffn(x3) + x2


class TransformerModel(torch.nn.Module):
    def __init__(self, vocab_size, context_length, d_model, d_ff, num_layers, num_heads, apply_rope=False, rope_theta=10000.0, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.position_embeddings = Embedding(context_length, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, apply_rope, rope_theta, device=device, dtype=dtype) for i in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, input):
        B, L = input.shape
        x = self.token_embeddings(input)
        x += self.position_embeddings(torch.arange(L))
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)