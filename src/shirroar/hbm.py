"""HBM pass: run an LRU block cache over the trace once and emit an event stream.

Every referenced block enters HBM whether it was a hit in HBM, loaded from flash, or
recomputed, so the HBM contents (and therefore the sequence of HBM evictions) do not depend
on the flash policy. The flash simulator replays this stream for each policy.

Event kinds: REF (a request uses a block) and EVICT (HBM evicts a block; the flash
policy decides whether to write it).
"""
from __future__ import annotations

from array import array
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .trace import Request

REF, EVICT = 0, 1


@dataclass
class EventStream:
    kind: List[int] = field(default_factory=list)
    t: List[float] = field(default_factory=list)
    blk: List[int] = field(default_factory=list)
    pos: List[int] = field(default_factory=list)       # block position in its prompt (depth = pos * 512)
    req: List[int] = field(default_factory=list)       # request index for REF, -1 for EVICT
    hbm_hit: List[int] = field(default_factory=list)   # REF: 1 if the block was in HBM
    parent_in_hbm: List[int] = field(default_factory=list)  # EVICT: 1 if parent is in HBM (or no parent)
    has_child: List[int] = field(default_factory=list)      # EVICT: bit 0 = some request has continued past this
    # block; bits 1+ = turn bucket of the last request that referenced it (turn_bucket; 0 when the trace has no turns)
    parent: Dict[int, Optional[int]] = field(default_factory=dict)
    position: Dict[int, int] = field(default_factory=dict)
    hbm_capacity_blocks: int = 0
    n_requests: int = 0

    def __len__(self) -> int:
        return len(self.kind)

    def compact(self) -> "EventStream":
        """Store the per-event fields as typed arrays (about 28 bytes per event instead of about
        150 with Python lists), so multi-day streams fit in memory. Indexing works the same."""
        self.kind = bytearray(self.kind)
        self.hbm_hit = bytearray(self.hbm_hit)
        self.parent_in_hbm = bytearray(self.parent_in_hbm)
        self.has_child = bytearray(self.has_child)
        self.t = array("d", self.t)
        self.blk = array("q", self.blk)
        self.pos = array("i", self.pos)
        self.req = array("i", self.req)
        return self

    @property
    def n_refs(self) -> int:
        return sum(1 for k in self.kind if k == REF)

    @property
    def n_evicts(self) -> int:
        return sum(1 for k in self.kind if k == EVICT)


TURN_EDGES = (2, 3, 5)   # turn buckets: unknown 0 | 1 | 2 | 3-4 | >=5


def turn_bucket(turn: int) -> int:
    if turn <= 0:
        return 0
    return 1 + sum(1 for e in TURN_EDGES if turn >= e)


def hbm_pass(reqs: Sequence[Request], capacity_blocks: int, compact: bool = True) -> EventStream:
    """LRU over blocks. capacity_blocks = 0 means every block leaves HBM right after use,
    which turns write-on-evict into write-at-reference (the model of the Appendix A script)."""
    if capacity_blocks < 0:
        raise ValueError("capacity_blocks must be >= 0")
    lru: "OrderedDict[int, None]" = OrderedDict()
    s = EventStream(hbm_capacity_blocks=capacity_blocks, n_requests=len(reqs))
    if compact:   # build typed arrays directly so peak memory stays small for multi-day traces
        s.kind, s.hbm_hit, s.parent_in_hbm, s.has_child = bytearray(), bytearray(), bytearray(), bytearray()
        s.t, s.blk, s.pos, s.req = array("d"), array("q"), array("i"), array("i")
    kind, tt, blk, pos, req, hit, pin, hch = s.kind, s.t, s.blk, s.pos, s.req, s.hbm_hit, s.parent_in_hbm, s.has_child
    parent, position = s.parent, s.position
    with_child = set()
    use_turn = any(r.turn for r in reqs[:2000])
    last_turn: Dict[int, int] = {}
    for qi, r in enumerate(reqs):
        tb = turn_bucket(r.turn) if use_turn else 0
        prev = None
        for i, b in enumerate(r.blocks):
            if b not in parent:
                parent[b] = prev
                position[b] = i
            if prev is not None:
                with_child.add(prev)
            if b in lru:
                lru.move_to_end(b)
                h = 1
            else:
                lru[b] = None
                h = 0
            kind.append(REF); tt.append(r.t); blk.append(b); pos.append(i); req.append(qi); hit.append(h); pin.append(1)
            hch.append(0)
            if use_turn:
                last_turn[b] = tb
            while len(lru) > capacity_blocks:
                e, _ = lru.popitem(last=False)
                p = parent[e]
                kind.append(EVICT); tt.append(r.t); blk.append(e); pos.append(position[e]); req.append(-1); hit.append(0)
                pin.append(1 if (p is None or p in lru) else 0)
                hch.append((1 if e in with_child else 0) | (last_turn.get(e, 0) << 1))
            prev = b
    return s
