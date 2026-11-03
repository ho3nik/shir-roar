"""Wear-price controller (Section IV of the paper).

Units: wear in one interval is measured in multiples of that interval's budget, so w = 1 means
the region wore exactly on its linear service-life schedule. The price lam is in GPU-seconds per
interval-budget, so a block of wear cost c (in interval budgets) is written only if its value v
(GPU-seconds) exceeds lam * c.

Two update rules are provided. Both keep total wear within a constant of the schedule; this is
the standard virtual-queue / dual-ascent argument (Neely 2010; Mahdavi, Jin and Yang 2012) and
is not claimed as new.

  additive:        lam <- max(0, lam + eta * (w - 1))
      sum_{t<=T} w(t) <= T + rho_max / eta + w_max                      (if lam(1) <= rho_max)
  multiplicative:  lam <- max(lam_min, lam * exp(eta * (w - 1)))
      sum_{t<=T} w(t) <= T + ln(rho_max / lam(1)) / eta + w_max - 1     (if lam(1) <= rho_max)

rho_max is the largest value-to-cost ratio any candidate can have, w_max the largest wear one
interval can add. Proof sketch (multiplicative): max(lam_min, x) >= x gives
lam(T+1) >= lam(1) * exp(eta * sum (w - 1)). If lam(t) >= rho_max nothing passes, so w(t) = 0
and the price falls; otherwise lam(t+1) < rho_max * exp(eta * (w_max - 1)). Hence lam stays
below that level, and taking logs gives the bound.
"""
from __future__ import annotations

import math


class PriceController:
    def __init__(self, lam0: float, eta: float, mode: str = "mult", lam_min: float | None = None):
        if mode not in ("mult", "add"):
            raise ValueError("mode must be 'mult' or 'add'")
        if eta < 0:
            raise ValueError("eta must be >= 0")
        if mode == "mult" and lam0 <= 0:
            raise ValueError("multiplicative mode needs lam0 > 0")
        self.lam = float(lam0)
        self.lam0 = float(lam0)
        self.eta = float(eta)
        self.mode = mode
        self.lam_min = (lam0 * 1e-6 if lam_min is None else lam_min) if mode == "mult" else 0.0

    def update(self, w: float) -> float:
        """w = wear in the interval just ended, in units of that interval's budget."""
        if self.mode == "mult":
            self.lam = max(self.lam_min, self.lam * math.exp(self.eta * (w - 1.0)))
        else:
            self.lam = max(0.0, self.lam + self.eta * (w - 1.0))
        return self.lam

    def admits(self, value: float, cost: float) -> bool:
        return value - self.lam * cost > 0.0

    def overshoot_bound(self, rho_max: float, w_max: float) -> float:
        """Upper bound on sum_t (w(t) - 1), in interval budgets, valid when lam(1) <= rho_max."""
        if self.eta == 0:
            return math.inf
        if self.mode == "mult":
            return math.log(max(rho_max, self.lam0) / self.lam0) / self.eta + w_max - 1.0
        return rho_max / self.eta + w_max
