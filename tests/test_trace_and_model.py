import math

import pytest

from shirroar.model import (B200, DEEPSEEK_V3, LLAMA31_70B, PSLC_512G, QLC_2048G, SLC_512G, TLC_1536G,
                            tokens_per_second_absorbable)
from shirroar.trace import Request, build_day, check_prefix_hashing, diurnal_multiplier, trace_stats


def test_prefix_hashing_check(toy_requests):
    assert check_prefix_hashing(toy_requests) == (0, 0)
    bad = toy_requests + [Request(4.0, (9, 1), 1024, 1)]   # block 1 under a different parent
    assert check_prefix_hashing(bad)[0] == 1


def test_build_day_namespaces_do_not_collide(toy_requests):
    day = build_day(toy_requests, hours=3, low=1.0, high=1.0, base_span=10.0)
    assert len(day) == 3 * len(toy_requests)
    by_hour = {}
    for r in day:
        h = int(r.t // 10.0)
        by_hour.setdefault(h, set()).update(r.blocks)
    assert not (by_hour[0] & by_hour[1]) and not (by_hour[1] & by_hour[2])
    assert check_prefix_hashing(day) == (0, 0)


def test_build_days_are_disjoint_and_ordered(toy_requests):
    from shirroar.trace import build_days
    two = build_days(toy_requests, seeds=(0, 1), hours=2, low=1.0, high=1.0, base_span=10.0)
    assert len(two) == 4 * len(toy_requests)
    assert all(a.t <= b.t for a, b in zip(two, two[1:]))
    first = {b for r in two if r.t < 20.0 for b in r.blocks}
    second = {b for r in two if r.t >= 20.0 for b in r.blocks}
    assert not (first & second)


def test_diurnal_range():
    vals = [diurnal_multiplier(h) for h in range(24)]
    assert max(vals) == pytest.approx(1.0) and min(vals) >= 0.3 - 1e-12
    assert vals.index(max(vals)) == 14


def test_trace_stats(toy_requests):
    st = trace_stats(toy_requests)
    assert st["unique_blocks"] == 6 and st["block_references"] == 12
    assert st["never_reused_share"] == pytest.approx(3 / 6)


def test_kv_sizes():
    assert LLAMA31_70B.kv_bytes_per_token == 327_680
    assert DEEPSEEK_V3.kv_bytes_per_token == 70_272
    assert LLAMA31_70B.block_bytes() == 160 * 2**20


def test_recompute_times_match_hand_calculation():
    # 64 ms at depth 0 and 221 ms at depth 128K for Llama-3.1-70B on B200 at 50% utilization.
    assert LLAMA31_70B.recompute_seconds(0, B200.flops) == pytest.approx(0.06426, rel=1e-3)
    assert LLAMA31_70B.recompute_seconds(131072, B200.flops) == pytest.approx(0.2206, rel=1e-3)


def test_budgets_match_hand_calculation():
    assert SLC_512G.daily_budget() / 1e12 == pytest.approx(28.04, abs=0.01)
    assert PSLC_512G.daily_budget() / 1e12 == pytest.approx(5.61, abs=0.01)
    assert TLC_1536G.daily_budget() / 1e12 == pytest.approx(2.52, abs=0.01)
    assert QLC_2048G.daily_budget() / 1e12 == pytest.approx(1.12, abs=0.01)
    assert tokens_per_second_absorbable(SLC_512G, LLAMA31_70B) == pytest.approx(990, abs=1)
    assert tokens_per_second_absorbable(TLC_1536G, DEEPSEEK_V3) == pytest.approx(416, abs=1)


def test_fifo_residence_identity():
    # Written at exactly its budget, a FIFO log keeps a block for D*WAF/N, whatever its size.
    for cell in (SLC_512G, PSLC_512G, TLC_1536G, QLC_2048G):
        rate = cell.daily_budget() / 86400.0
        assert cell.capacity_bytes / rate == pytest.approx(cell.fifo_residence_seconds(), rel=1e-12)
    assert SLC_512G.fifo_residence_seconds() / 60 == pytest.approx(26.3, abs=0.1)
    assert TLC_1536G.fifo_residence_seconds() / 3600 == pytest.approx(14.61, abs=0.01)
