import torch

def cross_entropy_loss(logits, targets):
    adjusted_logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    exp_logits = torch.exp(adjusted_logits)
    log_sum = torch.log(torch.sum(exp_logits, dim=-1))
    correct_logits = adjusted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum - correct_logits).mean()