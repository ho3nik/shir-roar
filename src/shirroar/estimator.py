"""Online estimate of how many flash-servable reads a stored copy would get.

When HBM evicts a block, the estimator opens an observation window of `horizon_s` seconds and
counts the block's later references that miss in HBM (the references a flash copy could serve).
When the window closes, the count is added to the statistics of the block's class. Classes are
(reference count so far, capped at nmax) x (whether any request has continued past the block)
x (position bucket of the block in its prompt). A block no request has continued past is usually
the partial last block of a prompt, which is almost never reused. The estimate for a class is
shrunk toward the pooled estimate for the same (count, continued) pair, then the same count,
then the global mean, so sparse classes do not swing the decision. Only finished windows are
used, so the estimator never looks into the future.
"""
from __future__ import annotations

import bisect
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

POS_EDGES = (1, 2, 4, 8, 16)  # buckets: 0 | 1 | 2-3 | 4-7 | 8-15 | >=16


def pos_bucket(pos: int) -> int:
    return bisect.bisect_right(POS_EDGES, pos)


class ReuseEstimator:
    def __init__(self, horizon_s: float = 900.0, nmax: int = 6, shrink: float = 5.0, prior: float = 0.5):
        self.horizon = horizon_s
        self.nmax = nmax
        self.shrink = shrink
        self.prior = prior
        self.sum: Dict[Tuple[int, int, int], float] = defaultdict(float)
        self.cnt: Dict[Tuple[int, int, int], int] = defaultdict(int)
        self.sum_nc: Dict[Tuple[int, int], float] = defaultdict(float)
        self.cnt_nc: Dict[Tuple[int, int], int] = defaultdict(int)
        self.sum_n: Dict[int, float] = defaultdict(float)
        self.cnt_n: Dict[int, int] = defaultdict(int)
        self.sum_all = 0.0
        self.cnt_all = 0
        # open windows: per block a list of [count]; a FIFO of (close_time, block, cell, cls)
        self.open: Dict[int, List[List[int]]] = defaultdict(list)
        self.queue: Deque[Tuple[float, int, List[int], Tuple[int, int, int]]] = deque()
        self.frozen = False   # if True, no new windows open, so the statistics stop changing

    def state_dict(self) -> dict:
        return {"params": (self.horizon, self.nmax, self.shrink, self.prior),
                **{n: dict(getattr(self, n)) for n in ("sum", "cnt", "sum_nc", "cnt_nc", "sum_n", "cnt_n")},
                "sum_all": self.sum_all, "cnt_all": self.cnt_all}

    @classmethod
    def from_state(cls, st: dict, frozen: bool = False) -> "ReuseEstimator":
        e = cls(*st["params"])
        for n in ("sum", "cnt", "sum_nc", "cnt_nc", "sum_n", "cnt_n"):
            getattr(e, n).update(st[n])
        e.sum_all, e.cnt_all = st["sum_all"], st["cnt_all"]
        e.frozen = frozen
        return e

    def cls(self, refs: int, pos: int, has_child: int = 1) -> Tuple[int, int, int]:
        return (min(refs, self.nmax), has_child & 1, pos_bucket(pos))

    def expect(self, refs: int, pos: int, has_child: int = 1) -> float:
        n, hc, pb = self.cls(refs, pos, has_child)
        g = (self.sum_all + self.shrink * self.prior) / (self.cnt_all + self.shrink)
        pn = (self.sum_n[n] + self.shrink * g) / (self.cnt_n[n] + self.shrink)
        pnc = (self.sum_nc[(n, hc)] + self.shrink * pn) / (self.cnt_nc[(n, hc)] + self.shrink)
        c = self.cnt[(n, hc, pb)]
        return (self.sum[(n, hc, pb)] + self.shrink * pnc) / (c + self.shrink)

    def on_evict(self, b: int, t: float, refs: int, pos: int, has_child: int = 1) -> None:
        if self.frozen:
            return
        cell = [0]
        self.open[b].append(cell)
        self.queue.append((t + self.horizon, b, cell, self.cls(refs, pos, has_child)))

    def stats_copy(self) -> "ReuseEstimator":
        """A new estimator with this one's learned statistics and no open windows (warm start)."""
        e = ReuseEstimator(self.horizon, self.nmax, self.shrink, self.prior)
        for name in ("sum", "cnt", "sum_nc", "cnt_nc", "sum_n", "cnt_n"):
            getattr(e, name).update(getattr(self, name))
        e.sum_all, e.cnt_all = self.sum_all, self.cnt_all
        return e

    def on_hbm_miss_ref(self, b: int, t: float) -> None:
        cells = self.open.get(b)
        if cells:
            for cell in cells:
                cell[0] += 1

    def mature(self, t: float) -> None:
        q = self.queue
        while q and q[0][0] <= t:
            _, b, cell, c = q.popleft()
            cells = self.open[b]
            # Windows of one block open in time order and share one horizon, so they also
            # close in that order: the closing window is always the first one in the list.
            assert cells[0] is cell
            cells.pop(0)
            if not cells:
                del self.open[b]
            x = cell[0]
            self.sum[c] += x
            self.cnt[c] += 1
            self.sum_nc[c[:2]] += x
            self.cnt_nc[c[:2]] += 1
            self.sum_n[c[0]] += x
            self.cnt_n[c[0]] += 1
            self.sum_all += x
            self.cnt_all += 1


class SessionEstimator(ReuseEstimator):
    """ReuseEstimator with one more class dimension: the turn bucket of the last request that referenced
    the block (hbm.turn_bucket: turn 1, 2, 3-4, 5 or more), which the HBM pass stores in the upper bits of
    has_child. It is known online: a request's turn number is part of the request. The (count, continued,
    position, turn) estimate is shrunk toward the three-way estimate of the base class."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.sum4: Dict[Tuple[int, int, int, int], float] = defaultdict(float)
        self.cnt4: Dict[Tuple[int, int, int, int], int] = defaultdict(int)

    def cls4(self, refs: int, pos: int, has_child: int = 1):
        return self.cls(refs, pos, has_child) + (has_child >> 1,)

    def expect(self, refs: int, pos: int, has_child: int = 1) -> float:
        base = super().expect(refs, pos, has_child)
        c = self.cls4(refs, pos, has_child)
        return (self.sum4[c] + self.shrink * base) / (self.cnt4[c] + self.shrink)

    def on_evict(self, b: int, t: float, refs: int, pos: int, has_child: int = 1) -> None:
        if self.frozen:
            return
        cell = [0]
        self.open[b].append(cell)
        self.queue.append((t + self.horizon, b, cell, self.cls4(refs, pos, has_child)))

    def mature(self, t: float) -> None:
        q = self.queue
        while q and q[0][0] <= t:
            _, b, cell, c4 = q.popleft()
            cells = self.open[b]
            assert cells[0] is cell
            cells.pop(0)
            if not cells:
                del self.open[b]
            x = cell[0]
            c = c4[:3]
            self.sum4[c4] += x
            self.cnt4[c4] += 1
            self.sum[c] += x
            self.cnt[c] += 1
            self.sum_nc[c[:2]] += x
            self.cnt_nc[c[:2]] += 1
            self.sum_n[c[0]] += x
            self.cnt_n[c[0]] += 1
            self.sum_all += x
            self.cnt_all += 1
