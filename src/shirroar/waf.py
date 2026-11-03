"""Closing the wear loop on the device's own write counter.

The admission controller counts host bytes, but the NAND wears by device bytes = host bytes x WAF, and the
device's WAF is not known in advance. Datacenter SSDs report the bytes written to the NAND (OCP Datacenter
NVMe SSD Specification v2.5, requirement SMART-1, "Physical Media Units Written"). WAFRegion models a region
whose true WAF is a function of time that the controller never sees; WAFLoop wraps a policy and, every
poll_s seconds, reads the device counter and the host counter, estimates WAF over a trailing window, and
sets the host-byte budget to the device budget divided by that estimate.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from .flash import Region
from .policies import Policy


class WAFRegion(Region):
    def __init__(self, name: str, block_bytes: float, device_endurance_bytes: float, life_s: float,
                 waf_fn: Callable[[float], float], cache_blocks: Optional[int] = None, t0: float = 0.0):
        super().__init__(name, block_bytes, device_endurance_bytes, life_s, cache_blocks, t0)
        self.dev_rate = device_endurance_bytes / life_s     # device bytes per second over the life
        self.waf_fn = waf_fn
        self.waf_hat = 1.0                                  # the controller's belief
        self.now = t0
        self.device_bytes = 0.0                             # what the OCP counter reports

    @property
    def rate(self) -> float:
        """Host-byte budget per second as the controller sees it."""
        return self.dev_rate / self.waf_hat

    def write(self, b: int) -> None:
        super().write(b)
        self.device_bytes += self.block_bytes * self.waf_fn(self.now)


class WAFLoop(Policy):
    """mode: 'naive' assumes WAF = 1 forever; 'known' is told the true WAF every tick (reference);
    'loop' polls the counters every poll_s and uses the trailing window_s ratio of device to host bytes."""

    def __init__(self, inner: Policy, mode: str = "loop", poll_s: float = 3600.0, window_s: float = 6 * 3600.0,
                 measure_from: float = 0.0):
        if mode not in ("naive", "known", "loop"):
            raise ValueError("mode must be naive, known or loop")
        self.inner, self.mode, self.poll_s, self.window_s = inner, mode, poll_s, window_s
        self.name = f"{inner.name} [{mode}]"
        self.uses_estimator = inner.uses_estimator
        self.measure_from = measure_from

    def reset(self, sim) -> None:
        super().reset(sim)
        self.inner.reset(sim)
        self.reg: WAFRegion = sim.regions[0]
        self.hist: List[Tuple[float, float, float]] = [(sim.t_start, 0.0, 0.0)]
        self.next_poll = sim.t_start + self.poll_s
        self.dev_at_measure: Optional[float] = None
        self.max_lead = 0.0          # device bytes ahead of the device schedule, max over ticks
        self.waf_trace: List[Tuple[float, float, float]] = []

    def _set(self, w: float) -> None:
        reg = self.reg
        if w <= 0:
            return
        reg.waf_hat = w
        if hasattr(self.inner, "cost"):          # ROAR prices in interval budgets: rescale them
            self.inner.cost = [r.block_bytes / (r.rate * self.sim.tick_s) for r in self.sim.regions]

    def on_tick(self, t: float) -> None:
        reg = self.reg
        reg.now = t
        if self.dev_at_measure is None and t >= self.measure_from:
            self.dev_at_measure = reg.device_bytes
        lead = reg.device_bytes - reg.dev_rate * (t - self.sim.t_start)
        if lead > self.max_lead:
            self.max_lead = lead
        if self.mode == "known":
            self._set(reg.waf_fn(t))
        elif self.mode == "loop" and t >= self.next_poll:
            self.next_poll += self.poll_s
            self.hist.append((t, reg.bytes_written, reg.device_bytes))
            t0, h0, d0 = next(x for x in self.hist if x[0] >= t - self.window_s - 1e-9)
            if reg.bytes_written > h0:
                self._set((reg.device_bytes - d0) / (reg.bytes_written - h0))
            self.waf_trace.append((t, reg.waf_hat, reg.waf_fn(t)))
        self.inner.on_tick(t)

    def decide(self, b, pos, refs, parent_ok, t, has_child=1):
        return self.inner.decide(b, pos, refs, parent_ok, t, has_child)

    def notify_write(self, r, t):
        self.inner.notify_write(r, t)

    def state(self):
        return {**self.inner.state(), "waf_hat": self.reg.waf_hat}


class SkipBlocks(Policy):
    """Never write the blocks in `skip` (for example the partial last block of each prompt); otherwise
    defer to the inner policy."""

    def __init__(self, inner: Policy, skip, label: str = "skip"):
        self.inner, self.skip = inner, skip
        self.name = f"{inner.name} [{label}]"
        self.uses_estimator = inner.uses_estimator

    def reset(self, sim) -> None:
        super().reset(sim)
        self.inner.reset(sim)

    def decide(self, b, pos, refs, parent_ok, t, has_child=1):
        if b in self.skip:
            return None
        return self.inner.decide(b, pos, refs, parent_ok, t, has_child)

    def notify_write(self, r, t):
        self.inner.notify_write(r, t)

    def on_tick(self, t):
        self.inner.on_tick(t)

    def state(self):
        return self.inner.state()


class ParentCheck(Policy):
    """Write only blocks a prefix lookup can reach (parent in HBM or flash); otherwise defer to inner."""

    def __init__(self, inner: Policy):
        self.inner = inner
        self.name = f"{inner.name} [parent check]"
        self.uses_estimator = inner.uses_estimator

    def reset(self, sim) -> None:
        super().reset(sim)
        self.inner.reset(sim)

    def decide(self, b, pos, refs, parent_ok, t, has_child=1):
        return self.inner.decide(b, pos, refs, parent_ok, t, has_child) if parent_ok else None

    def notify_write(self, r, t):
        self.inner.notify_write(r, t)

    def on_tick(self, t):
        self.inner.on_tick(t)

    def state(self):
        return self.inner.state()
