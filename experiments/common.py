"""Shared helpers for the experiment scripts. Every script writes CSV/JSON into results/ and
figures into results/figures/, and prints what it wrote."""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from shirroar import (B200, LLAMA31_70B, ROAR, AdmitOnK, FixedPrice, OracleSet, PeriodQuota,  # noqa: E402
                      RandomFraction, Region, ReuseEstimator, Simulator, StepwiseQuota, WriteAll, build_day,
                      hbm_pass, load_mooncake, oracle_select)

RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

# Traces: the FAST'25 release (tool-and-agent and conversation hours) is what the Mooncake authors
# recommend for new work; the earlier arXiv hour is kept as the tuning trace (controller gain).
TRACES = {
    "toolagent": os.path.join(ROOT, "data", "toolagent_trace.jsonl"),
    "conversation": os.path.join(ROOT, "data", "conversation_trace.jsonl"),
    "arxiv": os.path.join(ROOT, "data", "mooncake_trace.jsonl"),
    # FAST'25 synthetic trace (public datasets, Poisson arrivals, 17 minutes). It shares no requests with
    # either real hour, so it is the tuning trace: base_trace() stretches its timestamps to one hour.
    "synthetic": os.path.join(ROOT, "data", "synthetic_trace.jsonl"),
}
TRACE_NAME = os.environ.get("SHIR_TRACE", "toolagent")
# Qwen-Bailian hours converted by scripts/convert_qwen.py, e.g. SHIR_TRACE=qwenA_h1
TRACE = os.environ.get("MOONCAKE_TRACE", TRACES.get(TRACE_NAME, os.path.join(ROOT, "data", f"{TRACE_NAME}.jsonl")))


def tagged(name):
    """results file name with the trace name inserted before the extension"""
    stem, ext = os.path.splitext(name)
    return f"{stem}_{TRACE_NAME}{ext}"
HBM_BLOCKS = 6000          # about 1.0 TB of Llama-3.1-70B KV: an 8 x B200 node after weights and activations
DAY_S = 86_400.0
TICK_S = 10.0

# Fast gain for the first six hours to find the price level from a cold start, then a slow gain whose
# wear-lead bound is about 1.3 days of budget; no randomized admission band (it lowered value). Chosen on
# the arXiv hour (e5a, e5b, e5d); the held-out FAST'25 synthetic trace selects the same slow gain
# (e9_tune_synthetic.py).
ROAR_DEFAULT = dict(eta=0.0003, mode="mult", soft=0.0, eta0=0.05, warm_s=6 * 3600.0)
EST_DEFAULT = dict(horizon_s=900.0, nmax=6, shrink=5.0, prior=0.5)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def base_trace():
    if not os.path.exists(TRACE):
        raise SystemExit(f"trace {TRACE} not found; run scripts/fetch_mooncake.sh")
    reqs = load_mooncake(TRACE)
    if TRACE_NAME == "synthetic":   # 1,022 s of arrivals spread over one hour, order unchanged
        from shirroar.trace import Request
        t0, span = reqs[0].t, reqs[-1].t - reqs[0].t
        f = 3599.0 / span
        reqs = [Request((r.t - t0) * f, r.blocks, r.input_len, r.output_len, r.turn) for r in reqs]
    return reqs


def day_trace(base, seed=0):
    return build_day(base, hours=24, low=0.3, high=1.0, peak_hour=14, seed=seed)


def region(budget_bytes, span_s, model=LLAMA31_70B, cache_blocks=None, name="flash"):
    return Region(name, model.block_bytes(), budget_bytes, span_s, cache_blocks)


def run(stream, policy, budget_bytes, t_start, t_end, model=LLAMA31_70B, estimator=None, record=False,
        tick_s=TICK_S, regions=None):
    regs = regions if regions is not None else [region(budget_bytes, t_end - t_start, model)]
    sim = Simulator(stream, model, regs, policy, tick_s=tick_s, estimator=estimator, t_start=t_start, t_end=t_end,
                    record=record)
    return sim.run()


def write_all_bytes(stream, t_start, t_end, model=LLAMA31_70B):
    r = run(stream, WriteAll(), 1e30, t_start, t_end, model)
    return r.bytes_written[0], r.saved_flash_s, r.saved_hbm_s


def roar(**kw):
    cfg = dict(ROAR_DEFAULT)
    cfg.update(kw)
    return ROAR(**cfg)


def new_estimator(**kw):
    cfg = dict(EST_DEFAULT)
    cfg.update(kw)
    return ReuseEstimator(**cfg)


def hindsight_static_price(stream, budget_bytes, t_start, t_end, model=LLAMA31_70B, iters=28, **kw):
    """Constant price that spends the budget exactly on this very trace (found by bisection with
    the same online estimator). It needs the future to calibrate, so it is a reference point."""
    def spent(lam):
        p = FixedPrice(lam, soft=kw.get("soft", ROAR_DEFAULT["soft"]))
        r = run(stream, p, budget_bytes, t_start, t_end, model, estimator=new_estimator())
        return r.bytes_written[0] / budget_bytes
    lo, hi = 1e-3, 1e5
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if spent(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return hi


def write_csv(rows, name):
    path = os.path.join(RESULTS, name)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log("wrote", os.path.relpath(path, ROOT))
    return path


def write_json(obj, name):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    log("wrote", os.path.relpath(path, ROOT))
    return path


def stepwise_tuned():
    """The stepwise quota with the setting E15 chose on the held-out synthetic trace (default if not run yet)."""
    import json
    p = os.path.join(RESULTS, "e15_stepwise_choice_synthetic.json")
    if not os.path.exists(p):
        return StepwiseQuota(300.0)
    c = json.load(open(p))
    return StepwiseQuota(c["quantum_s"], tuple(c["steps"]), c["base_k"])
