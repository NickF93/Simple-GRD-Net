"""Learning-rate scheduler implementations."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class GammaCosineAnnealingWarmRestarts(LRScheduler):
    """Cosine warm restarts with multiplicative max-LR decay at each restart."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        first_restart_steps: int,
        restart_t_mult: float,
        restart_gamma: float,
        last_epoch: int = -1,
    ) -> None:
        if first_restart_steps < 1:
            raise ValueError("first_restart_steps must be >= 1")
        if restart_t_mult < 1.0:
            raise ValueError("restart_t_mult must be >= 1.0")
        if restart_gamma <= 0.0:
            raise ValueError("restart_gamma must be > 0.0")

        self.first_restart_steps = first_restart_steps
        self.restart_t_mult = restart_t_mult
        self.restart_gamma = restart_gamma
        super().__init__(optimizer, last_epoch=last_epoch)

    def _cycle_state(self, step: int) -> tuple[int, int, int]:
        t_i = self.first_restart_steps
        cycle = 0
        t_cur = step

        while t_cur >= t_i:
            t_cur -= t_i
            cycle += 1
            t_i = max(1, int(round(t_i * self.restart_t_mult)))

        return cycle, t_cur, t_i

    def get_lr(self) -> list[float]:
        """Return current learning rates for each optimizer param-group."""
        step = max(self.last_epoch, 0)
        cycle, t_cur, t_i = self._cycle_state(step)
        amplitude = self.restart_gamma**cycle
        cosine = 0.5 * (1.0 + math.cos(math.pi * (t_cur / t_i)))
        return [base_lr * amplitude * cosine for base_lr in self.base_lrs]
