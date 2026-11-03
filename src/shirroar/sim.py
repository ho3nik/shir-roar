"""Flash pass: replay an HBM event stream against flash regions and a write policy.

Prefix semantics follow vLLM and SGLang: a request uses cached blocks only up to its first
missing block; every later block is recomputed even if a copy exists somewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .estimator import ReuseEstimator
from .flash import Region
from .hbm import EVICT, REF, EventStream
from .model import B200, GPUSpec, ModelSpec
from .policies import Policy
from .trace import BLOCK_TOKENS


@dataclass
class Result:
    policy: str
    refs: int = 0
    hbm_hits: int = 0
    flash_hits: int = 0
    misses: int = 0
    repeat_refs: int = 0
    saved_hbm_s: float = 0.0
    saved_flash_s: float = 0.0
    bytes_written: List[float] = field(default_factory=list)
    blocks_written: List[int] = field(default_factory=list)
    endurance_bytes: List[float] = field(default_factory=list)
    scheduled_bytes: List[float] = field(default_factory=list)
    max_ahead_bytes: List[float] = field(default_factory=list)   # max over ticks of written - schedule
    series: Dict[str, list] = field(default_factory=dict)
    block_reads: Dict[int, int] = field(default_factory=dict)   # flash reads per block id (record=True only)

    def as_row(self) -> dict:
        row = {
            "policy": self.policy,
            "refs": self.refs, "hbm_hits": self.hbm_hits, "flash_hits": self.flash_hits,
            "misses": self.misses, "repeat_refs": self.repeat_refs,
            "saved_hbm_s": self.saved_hbm_s, "saved_flash_s": self.saved_flash_s,
        }
        for r, (bw, bl, en, sc, ma) in enumerate(zip(self.bytes_written, self.blocks_written, self.endurance_bytes,
                                                     self.scheduled_bytes, self.max_ahead_bytes)):
            row[f"r{r}_bytes"] = bw
            row[f"r{r}_blocks"] = bl
            row[f"r{r}_written_over_schedule"] = bw / sc if sc else float("nan")
            row[f"r{r}_max_ahead_over_total"] = ma / sc if sc else float("nan")
        return row


class Simulator:
    def __init__(self, stream: EventStream, model: ModelSpec, regions: Sequence[Region], policy: Policy,
                 gpu: GPUSpec = B200, tick_s: float = 30.0, estimator: Optional[ReuseEstimator] = None,
                 read_bw: float = 6.4e12, t_start: Optional[float] = None, t_end: Optional[float] = None,
                 record: bool = False, warmup_s: float = 0.0):
        self.stream = stream
        self.model = model
        self.regions = list(regions)
        self.policy = policy
        self.gpu = gpu
        self.tick_s = tick_s
        self.block_bytes = model.block_bytes()
        self.t_read = self.block_bytes / read_bw
        self.t_start = stream.t[0] if t_start is None else t_start
        self.t_end = stream.t[-1] if t_end is None else t_end
        if policy.uses_estimator and estimator is None:
            estimator = ReuseEstimator()
        self.estimator = estimator
        self.record = record
        self.warmup_s = warmup_s   # metrics count only events at or after t_start + warmup_s
        self._trec: Dict[int, float] = {}

    def t_rec_pos(self, pos: int) -> float:
        v = self._trec.get(pos)
        if v is None:
            v = self.model.recompute_seconds(pos * BLOCK_TOKENS, self.gpu.flops)
            self._trec[pos] = v
        return v

    def run(self) -> Result:
        S = self.stream
        kind, tt, blk, pos, req, hit, pin = S.kind, S.t, S.blk, S.pos, S.req, S.hbm_hit, S.parent_in_hbm
        hch = S.has_child
        parent = S.parent
        regions = self.regions
        pol = self.policy
        est = self.estimator
        for reg in regions:
            reg.t0 = self.t_start
        pol.reset(self)
        res = Result(pol.name)
        refs: Dict[int, int] = {}
        trec = self.t_rec_pos
        t_read = self.t_read
        tick = self.tick_s
        next_tick = self.t_start + tick
        cur_req, contiguous = -1, True
        ahead = [0.0] * len(regions)
        if self.record:
            res.series = {"t": [], "saved_flash_s": [], "flash_hits": []}
            for r in range(len(regions)):
                res.series[f"r{r}_bytes"] = []
                res.series[f"r{r}_sched"] = []

        def in_flash(b: int) -> bool:
            for reg in regions:
                if b in reg.resident:
                    return True
            return False

        COUNTERS = ("refs", "hbm_hits", "flash_hits", "misses", "repeat_refs", "saved_hbm_s", "saved_flash_s")
        t_meas = self.t_start + self.warmup_s
        snap = None if self.warmup_s > 0 else ({c: 0 for c in COUNTERS}, [0.0] * len(regions), [0] * len(regions))
        n = len(kind)
        for k in range(n):
            t = tt[k]
            if snap is None and t >= t_meas:
                snap = ({c: getattr(res, c) for c in COUNTERS}, [reg.bytes_written for reg in regions],
                        [reg.blocks_written for reg in regions])
            while t >= next_tick:
                pol.on_tick(next_tick)
                for r, reg in enumerate(regions):
                    a = reg.bytes_written - reg.scheduled_bytes(next_tick)
                    if a > ahead[r]:
                        ahead[r] = a
                if self.record:
                    res.series["t"].append(next_tick)
                    res.series["saved_flash_s"].append(res.saved_flash_s)
                    res.series["flash_hits"].append(res.flash_hits)
                    for r, reg in enumerate(regions):
                        res.series[f"r{r}_bytes"].append(reg.bytes_written)
                        res.series[f"r{r}_sched"].append(reg.scheduled_bytes(next_tick))
                    for key, val in pol.state().items():
                        res.series.setdefault(key, []).append(val)
                next_tick += tick
            if est is not None:
                est.mature(t)
            b = blk[k]
            if kind[k] == REF:
                q = req[k]
                if q != cur_req:
                    cur_req, contiguous = q, True
                c = refs.get(b, 0) + 1
                refs[b] = c
                res.refs += 1
                if c > 1:
                    res.repeat_refs += 1
                p = pos[k]
                if hit[k]:
                    if contiguous:
                        res.hbm_hits += 1
                        res.saved_hbm_s += trec(p)
                    else:
                        res.misses += 1
                else:
                    if est is not None:
                        est.on_hbm_miss_ref(b, t)
                    if contiguous and in_flash(b):
                        res.flash_hits += 1
                        if self.record:
                            res.block_reads[b] = res.block_reads.get(b, 0) + 1
                        res.saved_flash_s += trec(p) - t_read
                    else:
                        contiguous = False
                        res.misses += 1
            else:  # EVICT
                p = pos[k]
                c = refs.get(b, 0)
                # The estimator observes every HBM eviction, including blocks already in flash;
                # observing only the blocks the policy still has to decide on would bias its
                # statistics toward whatever the policy declined earlier.
                if est is not None:
                    est.on_evict(b, t, c, p, hch[k])
                if in_flash(b):
                    continue
                par = parent.get(b)
                parent_ok = bool(pin[k]) or (par is not None and in_flash(par))
                r = pol.decide(b, p, c, parent_ok, t, hch[k])
                if r is not None:
                    regions[r].write(b)
                    pol.notify_write(r, t)
        # close the books at the end of the horizon
        t_final = max(self.t_end, tt[-1])
        if snap is None:   # warm-up longer than the trace: nothing measured
            snap = ({c: getattr(res, c) for c in COUNTERS}, [reg.bytes_written for reg in regions],
                    [reg.blocks_written for reg in regions])
        for c in COUNTERS:
            setattr(res, c, getattr(res, c) - snap[0][c])
        for r, reg in enumerate(regions):
            res.bytes_written.append(reg.bytes_written - snap[1][r])
            res.blocks_written.append(reg.blocks_written - snap[2][r])
            res.endurance_bytes.append(reg.endurance_bytes)
            res.scheduled_bytes.append(reg.rate * max(0.0, t_final - max(t_meas, self.t_start)))
            a = reg.bytes_written - reg.scheduled_bytes(t_final)
            res.max_ahead_bytes.append(max(ahead[r], a))
        return res


def ideal_savings(stream: EventStream, model: ModelSpec, gpu: GPUSpec = B200) -> dict:
    """GPU-seconds an unlimited cache would save: every repeat reference is a hit (prefix
    hashing makes every repeat reference contiguous). Split by whether HBM already holds it."""
    refs: Dict[int, int] = {}
    tot = hbm = 0.0
    for k in range(len(stream.kind)):
        if stream.kind[k] != REF:
            continue
        b = stream.blk[k]
        c = refs.get(b, 0) + 1
        refs[b] = c
        if c > 1:
            s = model.recompute_seconds(stream.pos[k] * BLOCK_TOKENS, gpu.flops)
            tot += s
            if stream.hbm_hit[k]:
                hbm += s
    return {"ideal_total_s": tot, "ideal_hbm_only_s": hbm, "ideal_beyond_hbm_s": tot - hbm}
