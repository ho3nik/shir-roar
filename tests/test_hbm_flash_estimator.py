import pytest

from shirroar.estimator import ReuseEstimator, pos_bucket
from shirroar.flash import Region
from shirroar.hbm import EVICT, REF, hbm_pass


def events(s):
    return [(("R" if k == REF else "E"), b) for k, b in zip(s.kind, s.blk)]


def test_hbm_capacity_zero_evicts_after_each_reference(toy_requests):
    s = hbm_pass(toy_requests, 0)
    ev = events(s)
    assert len(ev) == 2 * 12
    for i in range(0, len(ev), 2):
        assert ev[i][0] == "R" and ev[i + 1] == ("E", ev[i][1])
    assert sum(s.hbm_hit) == 0


def test_hbm_lru_order(toy_requests):
    s = hbm_pass(toy_requests, 3)
    # request 0 fills (0,1,2); request 1 touches 0,1 then inserts 3 -> evicts 2 (LRU)
    ev = events(s)
    assert ev[:3] == [("R", 0), ("R", 1), ("R", 2)]
    assert ev[3:7] == [("R", 0), ("R", 1), ("R", 3), ("E", 2)]
    hits = [h for k, h in zip(s.kind, s.hbm_hit) if k == REF]
    assert hits[:6] == [0, 0, 0, 1, 1, 0]
    # parents are evicted before children when they were last used by the same request
    assert s.parent[3] == 1 and s.position[3] == 2


def test_region_fifo_and_wear():
    r = Region("x", block_bytes=10, endurance_bytes=1000, life_s=100, cache_blocks=2)
    for b in (1, 2, 3):
        r.write(b)
    assert 1 not in r and 2 in r and 3 in r
    assert r.bytes_written == 30 and r.blocks_written == 3 and r.blocks_discarded == 1
    assert r.wear_fraction() == pytest.approx(0.03)
    assert r.scheduled_bytes(50) == pytest.approx(500)
    with pytest.raises(ValueError):
        r.write(3)


def test_region_zero_capacity_keeps_nothing():
    r = Region("x", block_bytes=10, endurance_bytes=1000, life_s=100, cache_blocks=0)
    r.write(1)
    assert 1 not in r and r.bytes_written == 10


def test_estimator_uses_only_closed_windows():
    e = ReuseEstimator(horizon_s=10.0, shrink=0.0, prior=0.0)
    e.on_evict(7, t=0.0, refs=1, pos=0)
    e.on_hbm_miss_ref(7, 3.0)
    e.on_hbm_miss_ref(7, 4.0)
    e.mature(9.99)
    assert e.cnt_all == 0                    # window still open: nothing learned yet
    e.mature(10.0)
    assert e.cnt_all == 1 and e.sum_all == 2
    assert e.expect(1, 0) == pytest.approx(2.0)
    e.on_hbm_miss_ref(7, 11.0)               # after the window: not counted
    assert e.sum_all == 2


def test_estimator_overlapping_windows_same_block():
    e = ReuseEstimator(horizon_s=10.0, shrink=0.0)
    e.on_evict(1, 0.0, 1, 0)
    e.on_evict(1, 5.0, 2, 0)
    e.on_hbm_miss_ref(1, 6.0)                # inside both windows
    e.mature(10.0)
    assert e.sum[(1, 1, 0)] == 1
    e.on_hbm_miss_ref(1, 12.0)               # only inside the second window
    e.mature(15.0)
    assert e.sum[(2, 1, 0)] == 2


def test_estimator_separates_blocks_nobody_continued_past():
    e = ReuseEstimator(horizon_s=1.0, shrink=0.0)
    e.on_evict(1, 0.0, 1, 3, has_child=0)
    e.on_evict(2, 0.0, 1, 3, has_child=1)
    e.on_hbm_miss_ref(2, 0.5)
    e.mature(1.0)
    assert e.expect(1, 3, has_child=0) == 0.0 and e.expect(1, 3, has_child=1) == 1.0


def test_warm_copy_has_stats_but_no_open_windows():
    e = ReuseEstimator(horizon_s=1.0, shrink=0.0)
    e.on_evict(1, 0.0, 1, 0)
    e.on_hbm_miss_ref(1, 0.5)
    e.mature(1.0)
    e.on_evict(5, 2.0, 1, 0)
    w = e.stats_copy()
    assert w.expect(1, 0) == e.expect(1, 0) and not w.open and not w.queue


def test_pos_bucket():
    assert [pos_bucket(p) for p in (0, 1, 2, 3, 4, 7, 8, 15, 16, 500)] == [0, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_turn_bits_do_not_change_base_estimator_and_session_estimator_runs():
    from shirroar import ReuseEstimator, SessionEstimator
    from shirroar.hbm import turn_bucket
    assert [turn_bucket(t) for t in (0, 1, 2, 3, 4, 5, 9)] == [0, 1, 2, 3, 3, 4, 4]
    e = ReuseEstimator()
    assert e.cls(2, 3, 1 | (3 << 1)) == e.cls(2, 3, 1)
    s = SessionEstimator(horizon_s=10.0)
    s.on_evict(7, 0.0, 2, 3, 1 | (2 << 1))
    s.on_hbm_miss_ref(7, 5.0)
    s.mature(20.0)
    assert s.cnt4[(2, 1, s.cls(2, 3, 1)[2], 2)] == 1
    assert s.expect(2, 3, 1 | (2 << 1)) > s.expect(2, 3, 1 | (4 << 1))


def test_waf_loop_learns_the_true_waf():
    from shirroar import LLAMA31_70B, ROAR, Simulator, WAFLoop, WAFRegion, build_day, hbm_pass
    from shirroar.trace import Request
    reqs = [Request(float(i), (1, 2 + i % 5), 1024, 10) for i in range(4000)]
    s = hbm_pass(reqs, 2)
    bb = LLAMA31_70B.block_bytes()
    reg = WAFRegion("f", bb, 50 * bb, 4000.0, lambda t: 2.0)
    pol = WAFLoop(ROAR(eta=0.05), "loop", poll_s=500.0, window_s=1000.0)
    Simulator(s, LLAMA31_70B, [reg], pol, tick_s=10.0, t_start=0.0, t_end=4000.0).run()
    assert abs(reg.waf_hat - 2.0) < 1e-9
    assert reg.device_bytes == 2.0 * reg.bytes_written
