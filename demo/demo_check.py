"""Endurance budgets and residence times per cell type (Eq. 1), recompute times at a few depths, and a
few steps of the wear-price update. Runs in under a second.
Run from the repository root:  python demo/demo_check.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from shirroar.controller import PriceController  # noqa: E402
from shirroar.model import B200, CELLS, DEEPSEEK_V3, LLAMA31_70B  # noqa: E402

for c in CELLS.values():
    print(f"{c.name:20s} budget {c.daily_budget()/1e12:7.1f} TB/day/stack  residence {c.fifo_residence_seconds()/3600:6.2f} h")
print("block MiB", LLAMA31_70B.block_bytes()/2**20, "DSV3 ratio", LLAMA31_70B.kv_bytes_per_token/DEEPSEEK_V3.kv_bytes_per_token)
for d in (0, 4096, 32768):
    print("t_rec depth", d, f"{LLAMA31_70B.recompute_seconds(d, B200.flops)*1000:.1f} ms")
p = PriceController(1.0, 0.05)
for w in [3, 3, 3, 0, 0, 1]:      # wear per interval, in interval budgets
    print(f"w={w} -> price {p.update(w):.4f}")
