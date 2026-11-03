"""Offline reference: choose which blocks to write knowing every future reference.

For each block, the benefit of writing it at its first HBM eviction is the number of later
references that miss in HBM (the ones a flash copy serves) times the GPU time each one saves.
Blocks are taken greedily by benefit; taking a block also takes its ancestors that ever leave
HBM, because a child is useless to a prefix lookup without them. With equal block sizes and no
ancestor constraint, greedy-by-benefit is the exact optimum for unlimited flash capacity; with
the ancestor constraint it is a heuristic, so this is a strong reference, not a proven bound.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set

from .hbm import EVICT, REF, EventStream
from .model import B200, GPUSpec, ModelSpec
from .trace import BLOCK_TOKENS


def future_hbm_miss_refs(stream: EventStream) -> Dict[int, int]:
    first_ev: Dict[int, int] = {}
    for k, kd in enumerate(stream.kind):
        if kd == EVICT:
            first_ev.setdefault(stream.blk[k], k)
    cnt: Dict[int, int] = defaultdict(int)
    for k, kd in enumerate(stream.kind):
        if kd == REF and not stream.hbm_hit[k]:
            b = stream.blk[k]
            f = first_ev.get(b)
            if f is not None and k > f:
                cnt[b] += 1
    return dict(cnt), first_ev


def oracle_select(stream: EventStream, model: ModelSpec, budget_blocks: int, by: str = "value",
                  gpu: GPUSpec = B200, read_bw: float = 6.4e12, closure: bool = True) -> Set[int]:
    if by not in ("value", "count"):
        raise ValueError("by must be 'value' or 'count'")
    cnt, first_ev = future_hbm_miss_refs(stream)
    t_read = model.block_bytes() / read_bw

    def benefit(b: int) -> float:
        if by == "count":
            return float(cnt[b])
        d = stream.position[b] * BLOCK_TOKENS
        return cnt[b] * (model.recompute_seconds(d, gpu.flops) - t_read)

    order = sorted(cnt, key=benefit, reverse=True)
    selected: Set[int] = set()
    used = 0
    parent = stream.parent
    for b in order:
        if benefit(b) <= 0:
            break
        chain = []
        x = b
        while x is not None and x not in selected:
            if x == b or not closure or x in first_ev:
                chain.append(x)
            if not closure:
                break
            x = parent.get(x)
        if used + len(chain) > budget_blocks:
            continue
        selected.update(chain)
        used += len(chain)
        if used >= budget_blocks:
            break
    return selected
