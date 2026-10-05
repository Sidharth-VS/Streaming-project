"""Controller strategies (plan Phase 6.2).

All policies map an observed relative standard error (the online signal from
``error_estimation``) and the user's target to a new sampling fraction.  Because
the standard error of a weighted sum scales as ``1 / sqrt(n)``, the *variance
rule* ``f_new = f * (observed / target)^2`` is the model-based step; the
proportional and PID policies apply it (or its log-domain equivalent) with
damping, while the bandit learns it from rewards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class ControllerPolicy(Protocol):
    name: str

    def update(self, observed_error: float, target: float, current_fraction: float) -> float:
        """Return the next sampling fraction (unclamped)."""
        ...

    def reset(self) -> None:  # pragma: no cover - trivial
        ...


@dataclass
class ProportionalPolicy:
    """Policy A — P-controller on the error gap (plan Phase 6.2).

    ``f_{k+1} = f_k * (1 + Kp * gap)`` with ``gap = (observed - target) / target``.
    When the observed error overshoots the target the fraction grows; when it
    is comfortably below, the fraction shrinks to save resources.  The step is
    capped to ``max_step`` per update to damp oscillation.
    """

    kp: float = 1.0
    max_step: float = 2.0
    name: str = "proportional"

    def update(self, observed_error: float, target: float, current_fraction: float) -> float:
        if target <= 0:
            return current_fraction
        # observed_error == 0 (census) is a *full* deficit, not "no signal":
        # the fraction must be allowed to shrink to save resources.
        gap = (observed_error - target) / target
        multiplier = 1.0 + self.kp * gap
        multiplier = min(max(multiplier, 1.0 / self.max_step), self.max_step)
        return current_fraction * multiplier

    def reset(self) -> None:
        pass


@dataclass
class PIDPolicy:
    """Policy B — PID controller in log-fraction space (plan Phase 6.2).

    Working in the log domain makes the multiplicative loop additive and
    symmetric; the integral term removes steady-state offset under persistent
    rate shifts and the derivative term damps oscillation.  Includes
    anti-windup clamping of the integral accumulator.
    """

    kp: float = 0.9
    ki: float = 0.15
    kd: float = 0.3
    max_step: float = 2.0
    integral_limit: float = 4.0
    integral: float = field(default=0.0, init=False)
    prev_error: float | None = field(default=None, init=False)
    name: str = "pid"

    def update(self, observed_error: float, target: float, current_fraction: float) -> float:
        if target <= 0:
            return current_fraction
        # Bounded log domain: treat the observed error as at most 1000x below
        # or above the target.  In particular census (observed == 0) maps to a
        # large-but-finite deficit instead of a "no signal" early return.
        e = math.log(max(observed_error, target * 1e-3) / target)
        self.integral = min(max(self.integral + e, -self.integral_limit), self.integral_limit)
        d = 0.0 if self.prev_error is None else e - self.prev_error
        self.prev_error = e
        u = self.kp * e + self.ki * self.integral + self.kd * d
        u = min(max(u, -math.log(self.max_step)), math.log(self.max_step))
        return current_fraction * math.exp(u)

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None


@dataclass
class BanditPolicy:
    """Policy C (stretch) — multi-armed bandit over discrete fractions.

    Treats each candidate fraction as an arm; the reward for acting with arm
    ``f`` is ``-( |observed/target - 1| + lambda * f )`` — accuracy tracked to
    the target minus a resource-cost penalty.  epsilon-greedy with a decaying
    exploration rate and per-arm running means.
    """

    arms: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
    resource_cost: float = 0.5  # lambda
    epsilon0: float = 0.3
    epsilon_decay: float = 0.95
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    means: dict[float, float] = field(default_factory=dict, init=False)
    counts: dict[float, int] = field(default_factory=dict, init=False)
    epsilon: float = field(default=0.0, init=False)
    _last_arm: float | None = field(default=None, init=False)
    name: str = "bandit"

    def __post_init__(self) -> None:
        self.epsilon = self.epsilon0
        for arm in self.arms:
            self.means.setdefault(arm, 0.0)
            self.counts.setdefault(arm, 0)

    def update(self, observed_error: float, target: float, current_fraction: float) -> float:
        # 1) learn from the reward of the arm acted upon since the last update.
        if self._last_arm is not None and target > 0:
            reward = -(abs(observed_error / target - 1.0) + self.resource_cost * self._last_arm)
            n = self.counts[self._last_arm]
            self.means[self._last_arm] = (self.means[self._last_arm] * n + reward) / (n + 1)
            self.counts[self._last_arm] = n + 1
            self.epsilon = max(self.epsilon * self.epsilon_decay, 0.02)

        # 2) act: epsilon-greedy over estimated means.
        if self.rng.random() < self.epsilon:
            arm = float(self.rng.choice(list(self.arms)))
        else:
            arm = max(self.arms, key=lambda a: self.means[a])
        self._last_arm = arm
        return arm

    def reset(self) -> None:
        self.__post_init__()


def build_policy(name: str, **kwargs) -> ControllerPolicy:
    name = name.lower()
    if name in {"p", "prop", "proportional"}:
        return ProportionalPolicy(**kwargs)
    if name == "pid":
        return PIDPolicy(**kwargs)
    if name in {"bandit", "rl"}:
        return BanditPolicy(**kwargs)
    raise ValueError(f"unknown policy {name!r} (use 'proportional', 'pid' or 'bandit')")
