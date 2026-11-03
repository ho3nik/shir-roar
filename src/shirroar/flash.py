"""Flash region model: an append-only log with FIFO eviction and a wear ledger.

A region is written only by appending whole KV blocks; the oldest block is discarded when
the log is full. With whole-erase-block appends and FIFO reclamation there is no garbage
collection copying, so device write amplification is a parameter (default 1.0) that scales
wear, not a simulated quantity.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Set


class Region:
    """One flash region (for example an SLC-mode region or a dense region).

    cache_blocks: how many KV blocks the log holds (None = unlimited).
    endurance_bytes: host bytes the region may absorb over its life (C * N / WAF).
    life_s: the service life over which endurance_bytes is spread, in seconds.
    """

    def __init__(self, name: str, block_bytes: float, endurance_bytes: float, life_s: float,
                 cache_blocks: Optional[int] = None, t0: float = 0.0):
        if endurance_bytes <= 0 or life_s <= 0 or block_bytes <= 0:
            raise ValueError("endurance_bytes, life_s and block_bytes must be positive")
        if cache_blocks is not None and cache_blocks < 0:
            raise ValueError("cache_blocks must be >= 0 or None")
        self.name = name
        self.block_bytes = float(block_bytes)
        self.endurance_bytes = float(endurance_bytes)
        self.life_s = float(life_s)
        self.cache_blocks = cache_blocks
        self.t0 = t0
        self.fifo: Deque[int] = deque()
        self.resident: Set[int] = set()
        self.bytes_written = 0.0
        self.blocks_written = 0
        self.blocks_discarded = 0

    @property
    def rate(self) -> float:
        """Scheduled wear budget in host bytes per second."""
        return self.endurance_bytes / self.life_s

    def scheduled_bytes(self, t: float) -> float:
        return self.rate * max(0.0, t - self.t0)

    def wear_fraction(self) -> float:
        return self.bytes_written / self.endurance_bytes

    def __contains__(self, b: int) -> bool:
        return b in self.resident

    def write(self, b: int) -> None:
        if b in self.resident:
            raise ValueError(f"block {b} is already resident in region {self.name}")
        if self.cache_blocks == 0:
            # A zero-capacity region still wears but keeps nothing.
            self.bytes_written += self.block_bytes
            self.blocks_written += 1
            self.blocks_discarded += 1
            return
        self.fifo.append(b)
        self.resident.add(b)
        self.bytes_written += self.block_bytes
        self.blocks_written += 1
        if self.cache_blocks is not None:
            while len(self.fifo) > self.cache_blocks:
                old = self.fifo.popleft()
                self.resident.discard(old)
                self.blocks_discarded += 1

    def residence_time_at_budget(self) -> float:
        """Seconds a block survives if the region is written exactly at its wear budget."""
        if self.cache_blocks is None:
            return math.inf
        return self.cache_blocks * self.block_bytes / self.rate
