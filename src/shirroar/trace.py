"""Trace readers and trace builders.

A request is a prompt split into 512-token blocks. Block ids follow prefix hashing:
the id of block i names the whole prefix up to and including block i, so two requests
share block i only if they share every block before it. The Mooncake trace satisfies
this (each block id appears at one position and has one parent; tests check it).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Optional, Iterable, List, Sequence, Tuple

BLOCK_TOKENS = 512


@dataclass(frozen=True)
class Request:
    t: float                 # arrival time in seconds
    blocks: Tuple[int, ...]  # prefix-chained block ids, one per 512-token block
    input_len: int
    output_len: int
    turn: int = 0            # turn number within its conversation (Qwen-Bailian traces); 0 = unknown


def load_mooncake(path: str) -> List[Request]:
    """Read the public Mooncake trace (JSON lines: timestamp in ms, input_length,
    output_length, hash_ids). Returns requests sorted by arrival time (stable)."""
    reqs: List[Request] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            reqs.append(Request(float(r["timestamp"]) / 1000.0, tuple(int(h) for h in r["hash_ids"]),
                                int(r["input_length"]), int(r["output_length"]), int(r.get("turn", 0))))
    reqs.sort(key=lambda q: q.t)
    return reqs


def check_prefix_hashing(reqs: Sequence[Request]) -> Tuple[int, int]:
    """Return (blocks with more than one parent, blocks seen at more than one position).
    Both are zero for a trace that uses prefix hashing."""
    parent, pos = {}, {}
    bad_parent, bad_pos = set(), set()
    for r in reqs:
        prev = None
        for i, b in enumerate(r.blocks):
            if parent.setdefault(b, prev) != prev:
                bad_parent.add(b)
            if pos.setdefault(b, i) != i:
                bad_pos.add(b)
            prev = b
    return len(bad_parent), len(bad_pos)


def diurnal_multiplier(hour: int, low: float = 0.3, high: float = 1.0, peak_hour: int = 14) -> float:
    """Smooth day curve: `high` at peak_hour, `low` twelve hours away."""
    return low + (high - low) * 0.5 * (1.0 + math.cos(2.0 * math.pi * (hour - peak_hour) / 24.0))


def build_day(base: Sequence[Request], hours: int = 24, low: float = 0.3, high: float = 1.0,
              peak_hour: int = 14, seed: int = 0, namespace: int = 10_000_000,
              base_span: float = 3600.0, hour_offset: int = 0) -> List[Request]:
    """Build a multi-hour trace from a one-hour base trace.

    Hour h keeps each base request independently with probability diurnal_multiplier(h),
    shifts it by h hours, and renames its blocks into a namespace of its own, so no
    block is shared across hours. This removes cross-hour reuse, which makes stored
    copies less valuable than in a real day (a conservative choice for any write policy)."""
    rng = random.Random(seed)
    out: List[Request] = []
    for h in range(hours):
        m = diurnal_multiplier(h, low, high, peak_hour)
        g = h + hour_offset                      # global hour index: own time slot and own namespace
        off = namespace * (g + 1)
        for r in base:
            if rng.random() < m:
                out.append(Request(r.t + base_span * g, tuple(b + off for b in r.blocks), r.input_len, r.output_len, r.turn))
    out.sort(key=lambda q: q.t)
    return out


def build_days(base: Sequence[Request], seeds: Sequence[int], hours: int = 24, **kw) -> List[Request]:
    """Consecutive synthetic days, one per seed, each built like build_day with its own namespaces."""
    out: List[Request] = []
    for d, seed in enumerate(seeds):
        out.extend(build_day(base, hours=hours, seed=seed, hour_offset=d * hours, **kw))
    out.sort(key=lambda q: q.t)
    return out


def random_day_params(seed: int, low=(0.2, 0.5), peak=(11.0, 17.0)) -> dict:
    """Load shape of one 'unknown-load' day: trough level, peak hour (both uniform) and a one-hour burst
    at a random hour that carries 1.5x that hour's normal load."""
    rng = random.Random(10_007 * seed + 3)
    return {"low": rng.uniform(*low), "peak": rng.uniform(*peak), "burst_hour": rng.randrange(24), "burst": 1.5}


def build_days_random(base: Sequence[Request], seeds: Sequence[int], namespace: int = 10_000_000,
                      base_span: float = 3600.0) -> Tuple[List[Request], List[dict], List[float]]:
    """Consecutive days whose load shape is drawn per day (random_day_params): hour h keeps each base
    request with probability m(h) = low + (1 - low)(1 + cos(2 pi (h - peak)/24))/2; in the burst hour every
    request is kept and half of them are added again under new block names. Returns the requests, each
    day's parameters and the realized hourly multipliers."""
    out: List[Request] = []
    params, hourly = [], []
    for d, seed in enumerate(seeds):
        rng = random.Random(seed * 7919 + 17)     # one stream per day, so day 1 of (a, b) equals day (a,)
        p = random_day_params(seed)
        params.append(p)
        for h in range(24):
            m = p["low"] + (1.0 - p["low"]) * 0.5 * (1.0 + math.cos(2.0 * math.pi * (h - p["peak"]) / 24.0))
            g = d * 24 + h
            off = namespace * (g + 1)
            extra = 0.0
            if h == p["burst_hour"]:
                extra = (p["burst"] - 1.0) * m       # added load, as renamed copies
                m = 1.0 if m > 1.0 else m
            hourly.append(m + extra)
            for r in base:
                if rng.random() < m:
                    out.append(Request(r.t + base_span * g, tuple(b + off for b in r.blocks), r.input_len, r.output_len, r.turn))
                if extra > 0 and rng.random() < min(1.0, extra):
                    off2 = off + namespace // 2
                    out.append(Request(r.t + base_span * g, tuple(b + off2 for b in r.blocks), r.input_len, r.output_len, r.turn))
    out.sort(key=lambda q: q.t)
    return out, params, hourly


def build_days_reinject(base: Sequence[Request], seeds: Sequence[int], frac: float, gap_lo_s: float = 3600.0,
                        gap_hi_s: float = 86400.0, hours: int = 24, seed: int = 99,
                        gaps: Optional[Sequence[float]] = None, **kw) -> List[Request]:
    """build_days plus synthetic long-gap reuse: each request is re-issued, with probability `frac`,
    after a gap drawn log-uniformly from [gap_lo_s, gap_hi_s], with the same block ids (the same hour's
    namespace), as a paused session or a repeated job would. Copies that land after the last day are
    dropped. This is the only cross-hour reuse in any synthetic trace here, so it probes how much a
    long FIFO residency (dense cells) is worth against a short one (pSLC). With `gaps` (seconds), each
    gap is drawn uniformly from that empirical list instead (e.g. measured chat turn gaps)."""
    days = build_days(base, seeds, hours=hours, **kw)
    end = len(seeds) * hours * 3600.0 + (base[0].t if base else 0.0)
    rng = random.Random(seed)
    lo, hi = math.log(gap_lo_s), math.log(gap_hi_s)
    extra = []
    for r in days:
        if rng.random() < frac:
            t = r.t + (rng.choice(gaps) if gaps else math.exp(rng.uniform(lo, hi)))
            if t < end:
                extra.append(Request(t, r.blocks, r.input_len, r.output_len, r.turn))
    out = days + extra
    out.sort(key=lambda q: q.t)
    return out


def trace_stats(reqs: Iterable[Request]) -> dict:
    """Reference counts and reuse statistics of a trace."""
    from collections import Counter
    reqs = list(reqs)
    refs = Counter(b for r in reqs for b in r.blocks)
    uniq = len(refs)
    total = sum(refs.values())
    return {
        "requests": len(reqs),
        "span_s": (reqs[-1].t - reqs[0].t) if reqs else 0.0,
        "input_tokens": sum(r.input_len for r in reqs),
        "output_tokens": sum(r.output_len for r in reqs),
        "block_references": total,
        "unique_blocks": uniq,
        "repeat_reference_share": (total - uniq) / total if total else 0.0,
        "never_reused_share": sum(1 for c in refs.values() if c == 1) / uniq if uniq else 0.0,
        "reused_ge": {k: sum(1 for c in refs.values() if c >= k) / uniq for k in (2, 3, 5, 10)},
    }
