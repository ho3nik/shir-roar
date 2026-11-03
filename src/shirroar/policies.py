"""Flash write policies. Each policy is asked, when HBM evicts a block that is not already in
flash, whether to write it and to which region.

Baselines reimplemented from the literature (our code, not the authors'):
  WriteAll        - every evicted block is written (the 2026 HBF papers' default for KV).
  AdmitOnK        - write on the k-th reference (k = 2 is LARC's second-miss rule, 2013/2016;
                    SGLang's write_through_selective uses a hit-count threshold).
  RandomFraction  - write a fixed random share.
  PeriodQuota     - per-period write cap = remaining endurance / remaining periods, after the
                    Nimble Storage / HPE patent family (priority 2010); base rule selectable.
  StepwiseQuota   - quota per quantum with carried credit and progressively stricter admission
                    in a slack range, after EMC's erasure-quota throttling (priority 2015) and
                    Pannier (Middleware 2015). Thresholds here are our choice.
  FixedPrice      - value-minus-priced-wear rule with a constant price (a static write cost,
                    in the spirit of CacheSack, ATC 2022).
  OracleSet       - offline: writes a precomputed set chosen with knowledge of the future.
  AdaptiveK       - admit-on-k whose k follows the wear schedule: k rises while cumulative wear
                    leads the linear schedule by more than a band and falls while it lags (the
                    simplest budget-tracking frequency rule).
Proposed:
  ROAR            - value-minus-priced-wear rule whose per-region price follows the service-life
                    schedule through PriceController.
"""
from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Sequence

from .controller import PriceController


class Policy:
    name = "policy"
    uses_estimator = False

    def reset(self, sim) -> None:
        self.sim = sim

    def decide(self, b: int, pos: int, refs: int, parent_ok: bool, t: float, has_child: int = 1) -> Optional[int]:
        raise NotImplementedError

    def notify_write(self, r: int, t: float) -> None:
        pass

    def on_tick(self, t: float) -> None:
        pass

    def state(self) -> dict:
        return {}


class WriteAll(Policy):
    name = "write-all"

    def decide(self, b, pos, refs, parent_ok, t, has_child=1):
        return 0


class AdmitOnK(Policy):
    def __init__(self, k: int):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.name = f"admit-on-{k}"

    def decide(self, b, pos, refs, parent_ok, t, has_child=1):
        return 0 if refs >= self.k else None


class RandomFraction(Policy):
    def __init__(self, p: float, seed: int = 7):
        self.p = p
        self.seed = seed
        self.name = f"random-{p:.3f}"

    def reset(self, sim):
        super().reset(sim)
        self.rng = random.Random(self.seed)

    def decide(self, b, pos, refs, parent_ok, t, has_child=1):
        return 0 if self.rng.random() < self.p else None


FIVE_YEARS_S = 5 * 365.25 * 86400.0


class PeriodQuota(Policy):
    """Write while the period's quota lasts. quota = remaining endurance / remaining periods, computed at
    each period start (region 0 only). The horizon is the device's service life (five years by default,
    as the patent describes), so a period's unused quota is spread over the remaining life and is not
    repaid within the experiment; life_s=None uses the experiment's own horizon instead."""

    def __init__(self, period_s: float = 60.0, base_k: int = 1, life_s: Optional[float] = FIVE_YEARS_S):
        self.period = period_s
        self.base_k = base_k
        self.life_s = life_s
        self.name = f"period-quota(k={base_k})"

    def reset(self, sim):
        super().reset(sim)
        self.reg = sim.regions[0]
        self.period_end = sim.t_start
        self.quota = 0.0
        self.used = 0.0
        self._roll(sim.t_start)

    def _roll(self, t):
        sim = self.sim
        while t >= self.period_end:
            horizon = self.life_s if self.life_s else (sim.t_end - sim.t_start)
            total = self.reg.rate * horizon
            remaining = max(0.0, total - self.reg.bytes_written)
            left = max(1, int(-(-(sim.t_start + horizon - self.period_end) // self.period)))
            self.quota = remaining / left
            self.used = 0.0
            self.period_end += self.period

    def decide(self, b, pos, refs, parent_ok, t, has_child=1):
        self._roll(t)
        if refs < self.base_k:
            return None
        if self.used + self.reg.block_bytes > self.quota:
            return None
        return 0

    def notify_write(self, r, t):
        self.used += self.reg.block_bytes


class LoadProfile:
    """Expected load by hour of the run (any positive weights), as a load-aware quota would be given.
    cum(t) is the share of the run's expected load that falls before t (piecewise constant per hour)."""

    def __init__(self, hourly, t0: float = 0.0):
        self.w = [float(x) for x in hourly]
        self.t0 = t0
        tot = sum(self.w)
        self.cumw = [0.0]
        for x in self.w:
            self.cumw.append(self.cumw[-1] + x / tot)

    def cum(self, t: float) -> float:
        h = (t - self.t0) / 3600.0
        if h <= 0:
            return 0.0
        i = int(h)
        if i >= len(self.w):
            return 1.0
        return self.cumw[i] + (self.cumw[i + 1] - self.cumw[i]) * (h - i)


class ProfileQuota(PeriodQuota):
    """Period quota whose per-period share follows a load profile instead of being flat:
    quota = remaining budget x (expected load in this period / expected load in all remaining periods)."""

    def __init__(self, profile: LoadProfile, period_s: float = 60.0, base_k: int = 2,
                 life_s: Optional[float] = FIVE_YEARS_S):
        super().__init__(period_s, base_k, life_s)
        self.profile = profile
        self.name = f"load-prop quota(k={base_k})"

    def _roll(self, t):
        sim = self.sim
        while t >= self.period_end:
            run_budget = self.reg.rate * (sim.t_end - sim.t_start)
            a, b = self.profile.cum(self.period_end), self.profile.cum(self.period_end + self.period)
            if self.life_s:
                # this period's share of the profiled schedule, plus any lag spread over the rest of the life
                left = max(1, int(-(-(sim.t_start + self.life_s - self.period_end) // self.period)))
                lag = run_budget * a - self.reg.bytes_written
                self.quota = max(0.0, run_budget * (b - a) + lag / left)
            else:
                remaining = max(0.0, run_budget - self.reg.bytes_written)
                rest = 1.0 - a
                self.quota = remaining * (b - a) / rest if rest > 1e-12 else remaining
            self.used = 0.0
            self.period_end += self.period


class StepwiseQuota(Policy):
    """Quota per quantum q = schedule rate x quantum. Credit carries over. Within the allowance
    write every evicted block; then require 2, 3, 4 references in three slack steps of q each;
    then stop. A quantum that starts with negative credit writes nothing."""

    def __init__(self, quantum_s: float = 300.0, steps: Sequence[int] = (2, 3, 4), base_k: int = 1):
        self.quantum = quantum_s
        self.steps = tuple(steps)
        self.base_k = base_k        # references a block needs within the allowance (1: every evicted block)
        self.name = "stepwise-quota"

    def reset(self, sim):
        super().reset(sim)
        self.reg = sim.regions[0]
        self.q = self.reg.rate * self.quantum
        self.credit = 0.0
        self.used = 0.0
        self.end = sim.t_start
        self._roll(sim.t_start)

    def _roll(self, t):
        while t >= self.end:
            self.credit = self.credit - self.used + self.q
            self.used = 0.0
            self.end += self.quantum

    def decide(self, b, pos, refs, parent_ok, t, has_child=1):
        self._roll(t)
        if self.credit <= 0:
            return None
        w = self.used
        if w < self.credit:
            return 0 if refs >= self.base_k else None
        over = int((w - self.credit) // self.q)
        if over < len(self.steps):
            return 0 if refs >= self.steps[over] else None
        return None

    def notify_write(self, r, t):
        self.used += self.reg.block_bytes


class AdaptiveK(Policy):
    """Admit-on-k with k steered by the wear schedule (region 0 only).

    After every control interval, lead = bytes written - scheduled bytes. If lead > band, k rises by
    one; if lead < -band, k falls by one; k stays in [k_min, k_max]. A change is allowed at most once
    every `dwell_s` seconds. band is given in seconds of budget (band_s * schedule rate).
    dither: if True, k is a real number moved by +-step per interval and each decision rounds it at
    random (admit on floor(k) with probability 1 - frac(k)), so the write rate can sit between two
    integer rules instead of switching between them.
    parent_check: also refuse a block whose parent is in neither tier (a fixed-k rule never writes
    such a block, a varying k can)."""

    def __init__(self, band_s: float = 0.0, dwell_s: float = 0.0, k0: float = 3.0, k_min: int = 1, k_max: int = 12,
                 dither: bool = False, step: float = 0.01, parent_check: bool = True, seed: int = 13,
                 name: Optional[str] = None, profile: Optional["LoadProfile"] = None):
        self.profile = profile     # if given, the schedule follows the expected load instead of a straight line
        self.band_s = band_s
        self.dwell_s = dwell_s
        self.k0 = k0
        self.k_min = k_min
        self.k_max = k_max
        self.dither = dither
        self.step = step
        self.parent_check = parent_check
        self.seed = seed
        self.name = name or (f"adaptive-k(band={band_s:g}s,dwell={dwell_s:g}s{',dither' if dither else ''})")

    def reset(self, sim):
        super().reset(sim)
        self.reg = sim.regions[0]
        self.k = float(self.k0)
        self.last_change = -1e30
        self.rng = random.Random(self.seed)
        self.k_hist: List[float] = []

    def _k_now(self) -> int:
        if not self.dither:
            return int(round(self.k))
        base = int(math.floor(self.k))
        return base + (1 if self.rng.random() < self.k - base else 0)

    def decide(self, b, pos, refs, parent_ok, t, has_child=1):
        if self.parent_check and not parent_ok:
            return None
        return 0 if refs >= self._k_now() else None

    def on_tick(self, t):
        if self.profile is not None:
            sim = self.sim
            sched = self.reg.rate * (sim.t_end - sim.t_start) * self.profile.cum(t)
        else:
            sched = self.reg.scheduled_bytes(t)
        lead = self.reg.bytes_written - sched
        band = self.band_s * self.reg.rate
        if self.dither:
            if lead > band:
                self.k = min(self.k_max, self.k + self.step)
            elif lead < -band:
                self.k = max(self.k_min, self.k - self.step)
        elif t - self.last_change >= self.dwell_s:
            if lead > band and self.k < self.k_max:
                self.k += 1
                self.last_change = t
            elif lead < -band and self.k > self.k_min:
                self.k -= 1
                self.last_change = t
        self.k_hist.append(self.k)

    def state(self):
        return {"k": self.k}


class OracleSet(Policy):
    def __init__(self, selected: Iterable[int], name: str = "oracle"):
        self.selected = set(selected)
        self.name = name

    def decide(self, b, pos, refs, parent_ok, t, has_child=1):
        return 0 if b in self.selected else None


class ROAR(Policy):
    """Write block u to region r* = argmax_r v_u - lam_r * c_{u,r} if that is positive, else drop.

    v_u = E[future HBM-missing references | class] x (t_rec(depth) - t_read)   (GPU-seconds)
    c_{u,r} = block bytes / (region r's budget for one interval)                 (interval budgets)
    lam_r follows PriceController; eta = 0 gives a fixed price.
    parent_check: skip a block whose parent is neither in HBM nor in flash, because a stored
    child is useless to a prefix lookup without its parent.
    soft: if > 0, a block whose ratio v / (lam * c) lies within [1 - soft, 1 + soft] is written with
    probability rising linearly across that band, so a class of near-identical blocks is admitted
    in proportion instead of all at once. Nothing with ratio <= 1 - soft is written, so the wear
    bound holds with rho_max replaced by rho_max / (1 - soft).
    eta0, warm_s: optional gain schedule. For the first warm_s seconds the price moves with gain
    eta0 (to find its level quickly from a cold start), then with eta. The wear bound becomes the
    sum of the two phases' bounds, so it stays a constant.
    """
    uses_estimator = True

    def __init__(self, eta: float = 0.05, mode: str = "mult", lam0: Optional[float] = None,
                 parent_check: bool = True, name: Optional[str] = None, depth_weighted: bool = True,
                 soft: float = 0.0, seed: int = 11, eta0: Optional[float] = None, warm_s: float = 0.0,
                 value_mode: str = "est"):
        if value_mode not in ("est", "refs", "refs_depth"):
            raise ValueError("value_mode must be 'est', 'refs' or 'refs_depth'")
        self.value_mode = value_mode
        if not 0.0 <= soft < 1.0:
            raise ValueError("soft must be in [0, 1)")
        self.eta0 = eta0
        self.warm_s = warm_s
        self.eta = eta
        self.mode = mode
        self.lam0 = lam0
        self.parent_check = parent_check
        self.depth_weighted = depth_weighted
        self.soft = soft
        self.seed = seed
        self.name = name or (f"ROAR(eta={eta},{mode})" if eta > 0 else "fixed-price")

    def reset(self, sim):
        super().reset(sim)
        regs = sim.regions
        self.cost = [r.block_bytes / (r.rate * sim.tick_s) for r in regs]
        lam0 = self.lam0
        if lam0 is None:
            # Default start: a block is worth writing if it is expected to be read about once
            # at the shallowest depth, in the cheapest region.
            lam0 = sim.t_rec_pos(0) / min(self.cost)
        first = self.eta0 if self.eta0 is not None else self.eta
        self.ctrl: List[PriceController] = [PriceController(lam0, first, self.mode) for _ in regs]
        self.switch_at = sim.t_start + self.warm_s if self.eta0 is not None else None
        self.lam_switch: Optional[List[float]] = None
        self.win = [0.0] * len(regs)
        self.est = sim.estimator
        self.t_read = sim.t_read
        self.flat = sim.t_rec_pos(0)
        self.rng = random.Random(self.seed)
        self.max_rho = 0.0     # largest value / cost seen (for checking the wear bound)
        self.max_w = 0.0       # largest wear in one interval, in interval budgets

    def value(self, pos: int, refs: int, has_child: int = 1) -> float:
        save = (self.sim.t_rec_pos(pos) if self.depth_weighted else self.flat) - self.t_read
        if self.value_mode == "refs":          # ablation: plain reference count, no reuse model, no depth
            return float(refs) * (self.flat - self.t_read)
        if self.value_mode == "refs_depth":    # ablation: reference count times depth-aware savings
            return float(refs) * save
        return self.est.expect(refs, pos, has_child) * save

    def decide(self, b, pos, refs, parent_ok, t, has_child=1):
        if self.parent_check and not parent_ok:
            return None
        v = self.value(pos, refs, has_child)
        if v <= 0.0:
            return None
        rho = v / min(self.cost)
        if rho > self.max_rho:
            self.max_rho = rho
        d = self.soft
        best, best_score = None, -1.0
        for r, ctrl in enumerate(self.ctrl):
            price = ctrl.lam * self.cost[r]
            ratio = v / price if price > 0 else float("inf")
            if ratio > best_score:
                best, best_score = r, ratio
        if d == 0.0:
            return best if best_score > 1.0 else None
        if best_score <= 1.0 - d:
            return None
        if best_score >= 1.0 + d:
            return best
        return best if self.rng.random() < (best_score - (1.0 - d)) / (2.0 * d) else None

    def notify_write(self, r, t):
        self.win[r] += self.sim.regions[r].block_bytes

    def on_tick(self, t):
        sim = self.sim
        if self.switch_at is not None and self.lam_switch is None and t >= self.switch_at:
            self.lam_switch = [c.lam for c in self.ctrl]
            for c in self.ctrl:
                c.eta = self.eta
        for r, reg in enumerate(sim.regions):
            w = self.win[r] / (reg.rate * sim.tick_s)
            if w > self.max_w:
                self.max_w = w
            if self.ctrl[r].eta > 0:
                self.ctrl[r].update(w)
            self.win[r] = 0.0

    def state(self):
        return {f"lam{r}": c.lam for r, c in enumerate(self.ctrl)}

    def bound_intervals(self) -> float:
        """Proposition 1 bound on cumulative wear ahead of schedule, in interval budgets, using the
        largest value-to-cost ratio and interval wear actually observed (soft band included)."""
        import math
        rho = self.max_rho / (1.0 - self.soft) if self.soft else self.max_rho
        if self.eta0 is None or self.lam_switch is None:
            if self.eta0 is not None:      # never left the first phase
                return max(math.log(max(rho, c.lam0) / c.lam0) / self.eta0 + self.max_w - 1.0 for c in self.ctrl)
            return max(c.overshoot_bound(rho, self.max_w) for c in self.ctrl)
        out = []
        for c, ls in zip(self.ctrl, self.lam_switch):
            ph1 = math.log(max(rho, c.lam0) / c.lam0) / self.eta0 + self.max_w - 1.0
            ph2 = math.log(max(rho, ls) / ls) / self.eta + self.max_w - 1.0 if self.eta > 0 else math.inf
            out.append(ph1 + ph2)
        return max(out)


def FixedPrice(lam: float, **kw) -> ROAR:
    p = ROAR(eta=0.0, lam0=lam, **kw)
    p.name = kw.get("name") or "fixed-price"
    return p
