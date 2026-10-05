"""Closed-loop adaptive sampling-budget controller (plan Phase 6 — the novelty).

Removes the paper's manual sampling-fraction tuning (the original lists an
adaptive feedback mechanism as an unrealized extension, §IV-B / Fig. 5) by
closing the loop between the error-estimation module and every node's budget:

    every ``cadence_windows`` windows at the root:
        observed <- relative standard error of the last result  (online signal,
                    no ground truth needed)
        if |observed - target| / target <= dead_band:  hold     (stability)
        elif windows_since_last_change < min_dwell:    hold     (stability)
        else: f_new = clamp(policy.update(...), f_min, f_max)
              publish ControlMessage(f_new) on the control topic

Propagation is a real design decision (plan Phase 6.3): budget updates travel
on a **dedicated lightweight control topic** from the root to every sampling
node, arrive after one control-plane hop, and take effect from the next window
onwards.  The data plane is never repurposed for control.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from approxiot.controller.policies import build_policy
from approxiot.kafka_io.transport import ControlMessage


@dataclass
class ControllerConfig:
    #: User-specified error target (relative), e.g. 0.001 = 0.1% accuracy loss.
    target_error: float = 0.001
    #: "proportional" (A), "pid" (B) or "bandit" (C).
    policy: str = "proportional"
    #: Evaluate the loop every K windows.
    cadence_windows: int = 2
    #: Dead-band: hold when |obs - target|/target <= dead_band (no thrashing).
    #: Deliberately wider than one would naively pick: the per-window standard
    #: error has a ~sqrt(2/(Y-1)) relative noise floor of its own at small
    #: sample counts, so tight bands make the loop chase estimation noise.
    dead_band: float = 0.3
    #: Length of the trailing median filter applied to the observed error
    #: signal (1 = raw).  A median is robust to both sampling noise and the
    #: census/no-signal bimodality of per-window stderr at replication scale
    #: (plan Phase 6.4 stability concern: avoid oscillation/thrashing).
    signal_median_window: int = 5
    #: Consecutive out-of-band observations required before actuating
    #: (confirmation-before-actuation).  Sample-count quantization makes any
    #: budget change jump the error signal, so reacting to a single excursion
    #: causes limit-cycling; requiring confirmation breaks that loop.
    confirmations_required: int = 2
    #: Default policy step cap forwarded when the caller leaves it unset.
    policy_kwargs: dict = field(default_factory=lambda: {"max_step": 1.35})
    #: Minimum windows between actual budget changes (dwell time).
    min_dwell_windows: int = 4
    #: Fraction bounds.
    f_min: float = 0.01
    f_max: float = 1.0
    #: Minimum relative change worth propagating (hysteresis).
    min_step: float = 0.05
    #: Policy kwargs forwarded to ``build_policy``.
    policy_kwargs: dict = field(default_factory=dict)


@dataclass
class ControllerEvent:
    window: int
    observed_error: float
    target: float
    old_fraction: float
    new_fraction: float
    action: str  # "hold_deadband" | "hold_dwell" | "hold_cadence" | "set"
    generation: int


class AdaptiveBudgetController:
    """Runs at the root; emits :class:`ControlMessage` budget updates."""

    def __init__(self, config: ControllerConfig, rng: np.random.Generator | None = None):
        self.config = config
        self.policy = build_policy(config.policy, **config.policy_kwargs)
        self._rng = rng or np.random.default_rng()
        self.fraction: float = 0.5
        self.generation: int = 0
        self.last_change_window: int = -10**9
        self._recent_errors: deque[float] = deque(maxlen=max(1, config.signal_median_window))
        self._confirmations: int = 0
        self.history: list[ControllerEvent] = []

    # ------------------------------------------------------------------ API
    def bind_fraction(self, fraction: float) -> None:
        """Initial fraction (the static/manual starting point)."""
        self.fraction = float(fraction)

    def observe(self, window: int, observed_error: float) -> ControlMessage | None:
        """Feed the loop one observation; returns a ControlMessage or None."""
        cfg = self.config
        observed_error = float(observed_error)
        # Trailing-median filter: don't chase per-window estimation noise.
        self._recent_errors.append(observed_error)
        observed_error = float(np.median(self._recent_errors))

        if window - self.last_change_window < cfg.cadence_windows:
            event = ControllerEvent(window, observed_error, cfg.target_error, self.fraction, self.fraction, "hold_cadence", self.generation)
            self.history.append(event)
            return None
        if window - self.last_change_window < cfg.min_dwell_windows:
            event = ControllerEvent(window, observed_error, cfg.target_error, self.fraction, self.fraction, "hold_dwell", self.generation)
            self.history.append(event)
            return None
        if cfg.target_error > 0 and abs(observed_error - cfg.target_error) / cfg.target_error <= cfg.dead_band:
            self._confirmations = 0
            event = ControllerEvent(window, observed_error, cfg.target_error, self.fraction, self.fraction, "hold_deadband", self.generation)
            self.history.append(event)
            return None

        # Confirmation-before-actuation: only move after the signal has been
        # out of band for several consecutive checks (anti-limit-cycling).
        self._confirmations += 1
        if self._confirmations < cfg.confirmations_required:
            event = ControllerEvent(window, observed_error, cfg.target_error, self.fraction, self.fraction, "hold_cadence", self.generation)
            self.history.append(event)
            return None

        raw = self.policy.update(observed_error, cfg.target_error, self.fraction)
        new_fraction = min(max(raw, cfg.f_min), cfg.f_max)

        if abs(new_fraction - self.fraction) / self.fraction < cfg.min_step:
            event = ControllerEvent(window, observed_error, cfg.target_error, self.fraction, self.fraction, "hold_cadence", self.generation)
            self.history.append(event)
            return None

        old = self.fraction
        self.fraction = new_fraction
        self.generation += 1
        self.last_change_window = window
        self._confirmations = 0
        self.history.append(
            ControllerEvent(window, observed_error, cfg.target_error, old, new_fraction, "set", self.generation)
        )
        return ControlMessage(
            fraction=new_fraction,
            generation=self.generation,
            issued_window=window,
            target=cfg.target_error,
            policy=self.policy.name,
        )

    # ------------------------------------------------------------- reporting
    @property
    def fraction_history(self) -> list[tuple[int, float]]:
        return [(e.window, e.new_fraction) for e in self.history if e.action == "set"]

    def oscillation(self) -> float:
        """Std-dev of the fraction over time (stability metric, plan Phase 7.2)."""
        fracs = [f for _, f in self.fraction_history]
        if len(fracs) < 2:
            return 0.0
        mean = sum(fracs) / len(fracs)
        return float(np.std(fracs))
