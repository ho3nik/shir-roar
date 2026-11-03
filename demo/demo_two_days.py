"""Reproduces one cell of Table II (tool and agent, 10% budget, seed pair 0-1).
Two synthetic days (daily load curve 0.3 to 1.0, peak 14:00), measured on day 2.
Run from the repository root:  python demo/demo_two_days.py      (several minutes, ~3 GB RAM)
"""
import os, sys, time
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "src"))
from shirroar import (LLAMA31_70B, ROAR, AdmitOnK, WriteAll, PeriodQuota, Region, Simulator, ReuseEstimator,
                      hbm_pass, load_mooncake, build_days)

DAY, T, BB = 86400.0, 2 * 86400.0, LLAMA31_70B.block_bytes()
t0 = time.time()
base = load_mooncake(os.path.join(ROOT, "data", "toolagent_trace.jsonl"))
s = hbm_pass(build_days(base, seeds=[0, 1]), 6000)
print(f"built two days: {s.n_requests:,} requests ({time.time()-t0:.0f} s)")

def sim(policy, budget_day, est=None):
    reg = Region("flash", BB, 2 * budget_day, T, None)
    return Simulator(s, LLAMA31_70B, [reg], policy, tick_s=10.0, t_start=0.0, t_end=T,
                     warmup_s=DAY, estimator=est).run()

wa = sim(WriteAll(), 1e30)
WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
print(f"day 2 write-all: {WA2/1e12:.0f} TB written, {V2/3600:.1f} GPU-hours saved")
B = 0.10 * WA2
rows = [("period quota (2nd ref)", sim(PeriodQuota(60.0, 2), B)),
        ("admit on 3rd ref (no budget)", sim(AdmitOnK(3), 1e30)),
        ("ROAR", sim(ROAR(eta=0.0003, mode="mult", eta0=0.05, warm_s=6 * 3600.0), B,
                     ReuseEstimator(horizon_s=900.0, nmax=6, shrink=5.0, prior=0.5)))]
print(f"\n{'policy':30s} {'bytes / write-all':>18s} {'kept':>7s}")
for name, r in rows:
    print(f"{name:30s} {r.bytes_written[0]/WA2:17.1%} {r.saved_flash_s/V2:7.1%}")
print(f"\n({time.time()-t0:.0f} s)")
