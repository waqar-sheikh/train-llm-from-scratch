import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)


    def step(self, closure=None):

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data

                if t == 1:
                    state["first_moment"] = torch.zeros_like(grad)
                    state["second_moment"] = torch.zeros_like(grad)
                
                m = state["first_moment"]
                v = state["second_moment"]

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad

                lr_t = lr * ((1 - (beta2 ** t)) ** 0.5)
                lr_t /= 1 - (beta1 ** t)

                p.data -= (lr_t * m) / ((v ** 0.5) + eps)
                p.data -= lr * weight_decay * p.data

                state["first_moment"] = m
                state["second_moment"] = v
                state["t"] = t + 1


    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero_()


def cosine_lr_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return alpha_max * t / T_w
    elif t <= T_c:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w)))
        return alpha_min + (alpha_max - alpha_min) * cosine_decay
    else:
        return alpha_min
    

def gradient_clipping(params, max_norm, eps=1e-6):
    grads = [p.grad for p in params if p.grad is not None]

    total_norm = 0
    for grad in grads:
        total_norm += torch.sum(grad ** 2)
    total_norm = torch.sqrt(total_norm)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)