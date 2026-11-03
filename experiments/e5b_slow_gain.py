"""E5b. Slower gains and equal-wear comparison with admission on the 3rd reference (tuning trace)."""
import os
os.environ.setdefault("SHIR_TRACE", "arxiv")   # tuning trace only
import sys
from common import (DAY_S, HBM_BLOCKS, AdmitOnK, WriteAll, base_trace, hbm_pass, log, new_estimator, roar, write_csv)
from shirroar import LLAMA31_70B, Region, Simulator, build_days

T = 2 * DAY_S
BB = LLAMA31_70B.block_bytes()


def sim(s, pol, budget_day, est=None):
    return Simulator(s, LLAMA31_70B, [Region("f", BB, 2 * budget_day, T, None)], pol, tick_s=10.0, t_start=0.0,
                     t_end=T, warmup_s=DAY_S, estimator=est).run()


def main():
    s = hbm_pass(build_days(base_trace(), seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    a3 = sim(s, AdmitOnK(3), 1e30)
    w3 = a3.bytes_written[0] / WA2
    log(f"admit-on-3 writes {w3:.4f} of write-all, value {a3.saved_flash_s / V2:.4f}")
    rows = []
    for b in (w3, 0.10):
        for eta in (0.0001, 0.0003, 0.001):
            for soft in (0.3, 0.0):
                pol = roar(eta=eta, soft=soft, eta0=None, warm_s=0.0)
                r = sim(s, pol, b * WA2, est=new_estimator())
                row = {"budget_share": b, "eta": eta, "soft": soft, "value_share": r.saved_flash_s / V2,
                       "written_over_budget": r.bytes_written[0] / (b * WA2),
                       "max_ahead_share_of_daily_budget": r.max_ahead_bytes[0] / (b * WA2),
                       "bound_intervals": pol.bound_intervals()}
                rows.append(row)
                log(f"  budget {b:.4f} eta {eta} soft {soft}: value {row['value_share']:.4f} "
                    f"written/budget {row['written_over_budget']:.3f} ahead {row['max_ahead_share_of_daily_budget']:.3f} day")
            if b != w3:
                break
    write_csv(rows, "e5b_slow_gain_arxiv.csv")


if __name__ == "__main__":
    main()
