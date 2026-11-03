"""Property tests for the wear-price controller (Proposition 1).

Each interval presents a batch of candidate blocks with arbitrary value-to-cost ratios bounded by
rho_max. The controller admits a candidate when value - lam * cost > 0. Wear is measured in
interval budgets. The test checks the stated bound at every horizon T, for adversarial batches.
"""
import math

from hypothesis import given, settings
from hypothesis import strategies as st

from shirroar.controller import PriceController

batch = st.lists(st.tuples(st.floats(0.0, 1.0), st.floats(0.01, 1.0)), max_size=12)  # (ratio share, cost)


def run(mode, eta, lam0, rho_max, batches):
    ctrl = PriceController(lam0, eta, mode)
    w_max = max((sum(c for _, c in b) for b in batches), default=0.0)
    bound = ctrl.overshoot_bound(rho_max, w_max)
    total = 0.0
    for T, b in enumerate(batches, start=1):
        w = 0.0
        for share, cost in b:
            value = share * rho_max * cost          # value / cost <= rho_max
            if ctrl.admits(value, cost):
                w += cost
        total += w
        assert total <= T + bound + 1e-9, (T, total, bound)
        ctrl.update(w)
    return total


@settings(max_examples=300, deadline=None)
@given(st.lists(batch, min_size=1, max_size=200), st.floats(0.01, 2.0), st.floats(0.001, 1.0))
def test_multiplicative_bound(batches, eta, lam_frac):
    rho_max = 50.0
    run("mult", eta, lam_frac * rho_max, rho_max, batches)


@settings(max_examples=300, deadline=None)
@given(st.lists(batch, min_size=1, max_size=200), st.floats(0.5, 50.0), st.floats(0.0, 1.0))
def test_additive_bound(batches, eta, lam_frac):
    rho_max = 50.0
    run("add", eta, lam_frac * rho_max, rho_max, batches)


def test_price_rises_under_overspend_and_falls_under_underspend():
    c = PriceController(1.0, 0.1, "mult")
    assert c.update(3.0) > 1.0
    hi = c.lam
    assert c.update(0.0) < hi
    a = PriceController(1.0, 0.1, "add")
    assert a.update(0.0) == 0.9 and a.update(0.0) == 0.8


def test_constant_overload_is_throttled_to_schedule():
    # Every interval offers 10 budgets of candidates at ratio 0.9 rho_max. Long-run wear must
    # approach one budget per interval.
    rho = 10.0
    ctrl = PriceController(0.1, 0.2, "mult")
    total = 0.0
    T = 2000
    for _ in range(T):
        w = 0.0
        for _ in range(100):
            cost = 0.1
            if ctrl.admits(0.9 * rho * cost, cost):
                w += cost
        total += w
        ctrl.update(w)
    assert total <= T + ctrl.overshoot_bound(rho, 10.0)
    assert abs(total / T - 1.0) < 0.1
