"""AdamW, the learning-rate schedule and gradient clipping, written by hand."""

import math

import torch


class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    For each parameter p with gradient g, at step t:
        m = beta1 * m + (1 - beta1) * g          # moving average of the gradient (momentum)
        v = beta2 * v + (1 - beta2) * g^2        # moving average of its square (how "noisy" g is)
        m_hat = m / (1 - beta1^t)                # correction: m and v start at 0, so they are underestimated
        v_hat = v / (1 - beta2^t)
        p = p - lr * wd * p                      # weight decay applied directly, not through the gradient
        p = p - lr * m_hat / (sqrt(v_hat) + eps) # adaptive step: small where g varies a lot, large where it is stable
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t, m, v = state["step"], state["m"], state["v"]

                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return loss


def lr_at(step, cfg):
    """Linear warmup, then a cosine-shaped decay down to min_lr.

    Warmup: at the start, Adam's m and v are poor estimates and the gradients are large;
    big steps now can wreck the initialization. Cosine: large steps while we are far from the
    minimum, small steps at the end so the model can "settle" into it.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.schedule == "constant":
        return cfg.lr
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def clip_grad_norm(params, max_norm):
    """If the total gradient norm exceeds max_norm, scale the gradient down (the direction stays the same)."""
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
    scale = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(scale)
    return total
