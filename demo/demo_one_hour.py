"""One-hour demo (about five minutes).
Replays the real one-hour Mooncake tool-and-agent trace once through an HBM LRU cache, then asks
three flash policies what to write: write-all, admit-on-2nd-reference, and ROAR at a 10% budget.
Run from the repository root:  python demo/demo_one_hour.py
"""
import os, sys, time
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "src"))
from shirroar import (LLAMA31_70B, ROAR, AdmitOnK, WriteAll, PeriodQuota, Region, Simulator, ReuseEstimator,
                      hbm_pass, load_mooncake)

t0 = time.time()
reqs = load_mooncake(os.path.join(ROOT, "data", "toolagent_trace.jsonl"))
stream = hbm_pass(reqs, capacity_blocks=6000)          # about 1 TB of HBM for Llama-3.1-70B KV
start, end = stream.t[0], stream.t[-1]
bb = LLAMA31_70B.block_bytes()
print(f"{len(reqs):,} requests, {stream.n_evicts:,} HBM evictions, block = {bb/2**20:.0f} MiB")

def run(policy, budget):
    reg = Region("flash", bb, budget, end - start)       # unlimited capacity, wear budget only
    est = ReuseEstimator(horizon_s=900.0) if policy.uses_estimator else None
    sim = Simulator(stream, LLAMA31_70B, [reg], policy, tick_s=10.0, estimator=est, t_start=start, t_end=end)
    return sim.run()

big = 1e30                                             # no budget
wa = run(WriteAll(), big)
wa_bytes, wa_save = wa.bytes_written[0], wa.saved_flash_s
print(f"\n{'policy':28s} {'bytes written':>14s} {'GPU-s saved':>12s} {'kept':>7s}")
def show(name, r):
    print(f"{name:28s} {r.bytes_written[0]/wa_bytes:13.1%} {r.saved_flash_s:12.0f} {r.saved_flash_s/wa_save:7.1%}")
show("write-all", wa)
show("admit on 2nd reference", run(AdmitOnK(2), big))
show("admit on 3rd reference", run(AdmitOnK(3), big))
show("period quota, 10% budget", run(PeriodQuota(base_k=2, life_s=None), 0.10 * wa_bytes))
show("ROAR, budget 10% of bytes", run(ROAR(eta=0.05), 0.10 * wa_bytes))
print(f"\n({time.time()-t0:.0f} s)")
