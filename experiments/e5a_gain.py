"""E5a. Choosing the controller gain. Two synthetic days back to back (day seeds 0 and 1); every
policy runs both days and is measured on day 2 only, so ROAR's estimator and price are warm, as
they would be in a running system. The gain eta trades how closely wear follows the linear
schedule within a day against how well writes are spread toward the busiest, most valuable hours.
This first sweep used a soft band of 0.3 and no warm-up phase; e5d drops the band and adds the warm-up."""
import os
os.environ.setdefault("SHIR_TRACE", "arxiv")   # tuning trace only
from common import (DAY_S, HBM_BLOCKS, AdmitOnK, PeriodQuota, StepwiseQuota, WriteAll, base_trace, hbm_pass, log,
                    new_estimator, roar, run, write_csv)
from shirroar import build_days

ETAS = (0.001, 0.003, 0.01, 0.03, 0.1)


def main():
    base = base_trace()
    s = hbm_pass(build_days(base, seeds=(0, 1)), HBM_BLOCKS)
    T = 2 * DAY_S
    wa = run(s, WriteAll(), 1e30, 0.0, T)            # whole two days
    wa2 = None
    from shirroar import Simulator, Region, LLAMA31_70B
    r2 = Simulator(s, LLAMA31_70B, [Region("f", LLAMA31_70B.block_bytes(), 1e30, T, None)], WriteAll(),
                   tick_s=10.0, t_start=0.0, t_end=T, warmup_s=DAY_S).run()
    WA2, V2 = r2.bytes_written[0], r2.saved_flash_s
    log(f"two days: write-all {wa.bytes_written[0]/1e12:.1f} TB; day 2 {WA2/1e12:.1f} TB, flash value {V2:.0f} s")
    rows = []
    for b in (0.05, 0.10):
        budget_two_days = 2 * b * WA2

        def go(name, pol, est=None):
            sim = Simulator(s, LLAMA31_70B, [Region("f", LLAMA31_70B.block_bytes(), budget_two_days, T, None)], pol,
                            tick_s=10.0, t_start=0.0, t_end=T, warmup_s=DAY_S, estimator=est)
            res = sim.run()
            row = {"budget_share": b, "policy": name, "value_share_day2": res.saved_flash_s / V2,
                   "written_over_budget_day2": res.bytes_written[0] / (b * WA2),
                   "max_ahead_share_of_daily_budget": res.max_ahead_bytes[0] / (b * WA2)}
            if hasattr(pol, "bound_intervals"):
                row["bound_intervals"] = pol.bound_intervals()
                row["max_ahead_intervals"] = res.max_ahead_bytes[0] / (b * WA2 * 10.0 / DAY_S)
            rows.append(row)
            log(f"  {b:.2f} {name:18s} value {row['value_share_day2']:.4f} written/budget {row['written_over_budget_day2']:.3f} "
                f"ahead {row['max_ahead_share_of_daily_budget']:.3f} of a day's budget")

        for eta in ETAS:
            go(f"ROAR eta={eta}", roar(eta=eta, soft=0.3, eta0=None, warm_s=0.0), new_estimator())
        go("period-quota k=2", PeriodQuota(60.0, 2))
        go("stepwise-quota", StepwiseQuota(300.0))
        go("admit-on-3", AdmitOnK(3))
        go("admit-on-4", AdmitOnK(4))
    write_csv(rows, "e5a_gain_arxiv.csv")


if __name__ == "__main__":
    main()
