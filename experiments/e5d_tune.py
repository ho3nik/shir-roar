"""E5d. Tuning on the arXiv hour only (not an evaluation trace): final gain eta and soft band,
with a fixed warm-up gain (eta0 = 0.05 for the first 6 hours of operation, then eta).
Two synthetic days; measured on day 2."""
import os
import sys
os.environ.setdefault("SHIR_TRACE", "arxiv")
from common import (DAY_S, HBM_BLOCKS, AdmitOnK, WriteAll, base_trace, hbm_pass, log, new_estimator, roar, write_csv)
from shirroar import LLAMA31_70B, Region, Simulator, build_days

T = 2 * DAY_S
BB = LLAMA31_70B.block_bytes()
WARM = dict(eta0=0.05, warm_s=6 * 3600.0)


def sim(s, pol, budget_day, est=None):
    return Simulator(s, LLAMA31_70B, [Region("f", BB, 2 * budget_day, T, None)], pol, tick_s=10.0, t_start=0.0,
                     t_end=T, warmup_s=DAY_S, estimator=est).run()


def main():
    s = hbm_pass(build_days(base_trace(), seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    a3 = sim(s, AdmitOnK(3), 1e30)
    w3 = a3.bytes_written[0] / WA2
    rows = [{"trace": "arxiv", "policy": "admit-on-3", "budget_share": None, "written_share": w3,
             "value_share": a3.saved_flash_s / V2}]
    configs = [(eta, 0.0) for eta in (0.0003, 0.001, 0.003, 0.01)] + [(0.001, 0.3)]
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    budgets = {"low": (0.05, w3), "high": (0.10,), "all": (0.05, w3, 0.10)}[which]
    for b in budgets:
        for eta, soft in configs:
            pol = roar(eta=eta, soft=soft, **WARM)
            r = sim(s, pol, b * WA2, est=new_estimator())
            row = {"trace": "arxiv", "policy": "ROAR", "budget_share": b, "eta": eta, "soft": soft,
                   "value_share": r.saved_flash_s / V2, "written_over_budget": r.bytes_written[0] / (b * WA2),
                   "max_ahead_share_of_daily_budget": r.max_ahead_bytes[0] / (b * WA2),
                   "bound_intervals": pol.bound_intervals()}
            rows.append(row)
            log(f"  budget {b:.4f} eta {eta} soft {soft}: value {row['value_share']:.4f} written/budget "
                f"{row['written_over_budget']:.3f} ahead {row['max_ahead_share_of_daily_budget']:.3f} day "
                f"bound {row['bound_intervals']:.0f} intervals")
    write_csv(rows, f"e5d_tune_arxiv_{which}.csv")


if __name__ == "__main__":
    main()
