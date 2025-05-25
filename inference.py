import torch
from layers import softmax

class Generator():
    def __init__(self, model, tokenizer, context_length, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.device = device

    def generate(self, prompt, max_tokens, temperature=1, top_k=5):
        token_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        for i in range(max_tokens):
            logits = self.model(token_ids)
            last_logits = logits[:, -1, :] / temperature
            probs = softmax(last_logits, dim=-1)
            topk_values, topk_indices = torch.topk(probs, top_k)
            sampled_idx = torch.multinomial(topk_values, 1)
            sample = torch.gather(topk_indices, index=sampled_idx, dim=-1)
            token_ids = torch.cat([token_ids, sample], dim=-1)[:, -self.context_length:]
        
        return token_ids