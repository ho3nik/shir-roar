import math

import pytest

from shirroar.flash import Region
from shirroar.hbm import hbm_pass
from shirroar.model import LLAMA31_70B
from shirroar.oracle import oracle_select
from shirroar.policies import (ROAR, AdmitOnK, OracleSet, PeriodQuota, Policy, RandomFraction, StepwiseQuota,
                               WriteAll)
from shirroar.sim import Simulator, ideal_savings

BIG = 1e30


def unlimited(span=3600.0, endurance=BIG):
    return [Region("flash", LLAMA31_70B.block_bytes(), endurance, span, None)]


class ClosureProbe(Policy):
    """Wraps a policy and records writes whose parent is not in flash at write time."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name + "+probe"
        self.orphans = 0
        self.writes = 0

    def reset(self, sim):
        super().reset(sim)
        self.inner.reset(sim)

    def decide(self, b, pos, refs, parent_ok, t, has_child=1):
        r = self.inner.decide(b, pos, refs, parent_ok, t, has_child)
        if r is not None:
            self.writes += 1
            par = self.sim.stream.parent[b]
            if par is not None and not any(par in reg for reg in self.sim.regions):
                self.orphans += 1
        return r


# ---------------- toy-trace tests ----------------

def test_write_all_and_admit_on_k_toy(toy_requests):
    s = hbm_pass(toy_requests, 0)
    res = Simulator(s, LLAMA31_70B, unlimited(4.0), WriteAll(), tick_s=1.0).run()
    assert res.blocks_written[0] == 6
    # repeat references: block 0 x3, block 1 x2, block 2 x1 -> 6 flash hits
    assert res.flash_hits == 6 and res.repeat_refs == 6
    res2 = Simulator(s, LLAMA31_70B, unlimited(4.0), AdmitOnK(2), tick_s=1.0).run()
    # Written at their 2nd reference: block 0 and block 1 (request 1) and block 2 (request 3).
    # Served from flash: block 0 in requests 2 and 3, block 1 in request 3.
    assert res2.blocks_written[0] == 3 and res2.flash_hits == 3


def test_contiguity_stops_at_first_miss(toy_requests):
    s = hbm_pass(toy_requests, 0)
    keep_child_only = OracleSet({1, 2})          # parent 0 never stored
    res = Simulator(s, LLAMA31_70B, unlimited(4.0), keep_child_only, tick_s=1.0).run()
    assert res.flash_hits == 0


# ---------------- Mooncake validation (skipped if the trace is absent) ----------------

def test_reproduces_appendix_numbers(mooncake):
    s = hbm_pass(mooncake, 0)
    uniq = len(s.position)
    out = {}
    for pol in (WriteAll(), AdmitOnK(2), AdmitOnK(3)):
        r = Simulator(s, LLAMA31_70B, unlimited(), pol).run()
        out[pol.name] = (r.blocks_written[0] / uniq, r.flash_hits / r.repeat_refs)
    assert out["write-all"] == (1.0, 1.0)
    assert out["admit-on-2"][0] == pytest.approx(0.2153, abs=5e-4) and out["admit-on-2"][1] == pytest.approx(0.826, abs=1e-3)
    assert out["admit-on-3"][0] == pytest.approx(0.0960, abs=5e-4) and out["admit-on-3"][1] == pytest.approx(0.748, abs=1e-3)
    assert ideal_savings(s, LLAMA31_70B)["ideal_total_s"] == pytest.approx(16504, abs=1)


def test_proposition2_admit_on_k_is_ancestor_closed(mooncake):
    s = hbm_pass(mooncake, 0)
    for k in (2, 3, 5):
        probe = ClosureProbe(AdmitOnK(k))
        Simulator(s, LLAMA31_70B, unlimited(), probe).run()
        assert probe.writes > 0 and probe.orphans == 0


def test_random_admission_is_not_closed(mooncake):
    s = hbm_pass(mooncake, 0)
    probe = ClosureProbe(RandomFraction(0.2, seed=1))
    Simulator(s, LLAMA31_70B, unlimited(), probe).run()
    assert probe.orphans > 0


@pytest.fixture(scope="module")
def stream_hbm(mooncake):
    return hbm_pass(mooncake, 6000)


@pytest.fixture(scope="module")
def wa_bytes(stream_hbm):
    return Simulator(stream_hbm, LLAMA31_70B, unlimited(), WriteAll()).run().bytes_written[0]


def budgeted(wa, frac, span=3600.0):
    return [Region("flash", LLAMA31_70B.block_bytes(), frac * wa, span, None)]


def test_period_quota_never_exceeds_budget(stream_hbm, wa_bytes):
    for k in (1, 2):
        regs = budgeted(wa_bytes, 0.1)
        r = Simulator(stream_hbm, LLAMA31_70B, regs, PeriodQuota(60.0, k), t_start=0.0, t_end=3600.0).run()
        assert r.bytes_written[0] <= 0.1 * wa_bytes + LLAMA31_70B.block_bytes()


def test_stepwise_quota_stays_near_budget(stream_hbm, wa_bytes):
    regs = budgeted(wa_bytes, 0.1)
    r = Simulator(stream_hbm, LLAMA31_70B, regs, StepwiseQuota(300.0), t_start=0.0, t_end=3600.0).run()
    q = 0.1 * wa_bytes * 300.0 / 3600.0
    assert r.bytes_written[0] <= 0.1 * wa_bytes + 3 * q + LLAMA31_70B.block_bytes()


def test_roar_tracks_budget_within_bound(stream_hbm, wa_bytes):
    regs = budgeted(wa_bytes, 0.1)
    pol = ROAR(eta=0.2)
    sim = Simulator(stream_hbm, LLAMA31_70B, regs, pol, tick_s=10.0, t_start=0.0, t_end=3600.0)
    r = sim.run()
    # written bytes within the schedule plus a bounded overshoot (here: under 15% of the hour's budget)
    assert r.bytes_written[0] <= 1.15 * 0.1 * wa_bytes
    assert r.bytes_written[0] >= 0.5 * 0.1 * wa_bytes


def test_oracle_selection_respects_budget_and_closure(stream_hbm):
    sel = oracle_select(stream_hbm, LLAMA31_70B, budget_blocks=2000, by="value")
    assert len(sel) <= 2000
    par = stream_hbm.parent
    from shirroar.oracle import future_hbm_miss_refs
    _, first_ev = future_hbm_miss_refs(stream_hbm)
    for b in sel:
        p = par[b]
        while p is not None:
            if p in first_ev:
                assert p in sel
            p = par[p]


def test_warmup_counts_only_the_measured_window(toy_requests):
    s = hbm_pass(toy_requests, 0)
    full = Simulator(s, LLAMA31_70B, unlimited(4.0), WriteAll(), tick_s=1.0).run()
    late = Simulator(s, LLAMA31_70B, unlimited(4.0), WriteAll(), tick_s=1.0, warmup_s=3.0).run()
    # only request 3 (t = 3.0) is measured: blocks 0, 1, 2 are flash hits, block 5 is new
    assert late.flash_hits == 3 and late.refs == 4 and late.blocks_written[0] == 1
    assert full.flash_hits == 6


def test_adaptive_k_tracks_budget(stream_hbm, wa_bytes):
    from shirroar import AdaptiveK
    for kw in (dict(dither=True, step=0.05), dict(band_s=60.0, dwell_s=10.0)):
        regs = budgeted(wa_bytes, 0.1)
        pol = AdaptiveK(**kw)
        r = Simulator(stream_hbm, LLAMA31_70B, regs, pol, tick_s=10.0, t_start=0.0, t_end=3600.0).run()
        assert 0.5 * 0.1 * wa_bytes <= r.bytes_written[0] <= 1.3 * 0.1 * wa_bytes
        assert all(pol.k_min <= k <= pol.k_max for k in pol.k_hist)


def test_roar_reference_count_value_is_nearly_ancestor_closed(mooncake):
    # with the reference count as its value ROAR never rates a child above its parent at the same price; a
    # child can still pass after its parent was refused at an earlier, higher price, but rarely
    s = hbm_pass(mooncake, 6000)

    class Probe(ROAR):
        orphans = 0

        def decide(self, b, pos, refs, parent_ok, t, has_child=1):
            r = super().decide(b, pos, refs, parent_ok, t, has_child)
            if r is not None and not parent_ok:
                Probe.orphans += 1
            return r
    wa = Simulator(s, LLAMA31_70B, unlimited(), WriteAll()).run().bytes_written[0]
    r = Simulator(s, LLAMA31_70B, budgeted(wa, 0.1), Probe(eta=0.05, value_mode="refs", parent_check=False),
                  tick_s=10.0, t_start=0.0, t_end=3600.0).run()
    assert Probe.orphans <= 0.001 * r.blocks_written[0]


def test_reinjected_copies_reuse_block_ids(toy_requests):
    from shirroar import build_days_reinject
    from shirroar.trace import build_days
    base = build_days(toy_requests, seeds=(0, 1))
    out = build_days_reinject(toy_requests, seeds=(0, 1), frac=1.0, gap_lo_s=3600.0, gap_hi_s=7200.0)
    assert len(out) >= len(base)
    ids = {r.blocks for r in base}
    assert all(r.blocks in ids for r in out)
    assert all(a.t <= b.t for a, b in zip(out, out[1:]))
