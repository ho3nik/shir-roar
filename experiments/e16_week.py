"""E16. Steady state over several days. DAYS known-load days (seeds 0 to DAYS-1, the daily cosine of E3; default
7, the paper uses 5 because 7 need more than 8 GB of memory), a 10% budget of the mean day's write-all bytes,
measured from day 3 so day-1 effects are excluded. Reports savings,
writes per budget and the largest lead over the schedule for ROAR, the period quota (five-year horizon) and
adaptive k given the true load curve. Usage: SHIR_TRACE=conversation python e16_week.py [DAYS]"""
import sys

from common import DAY_S, HBM_BLOCKS, PeriodQuota, Region, WriteAll, base_trace, hbm_pass, log, new_estimator, roar, \
    tagged, write_csv
from e6_adaptive_k import GRID
from e10_load_aware import AK, cosine_profile
from shirroar import LLAMA31_70B, AdaptiveK, LoadProfile, Simulator, build_days
from shirroar.trace import diurnal_multiplier

BB = LLAMA31_70B.block_bytes()


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    T, warm = days * DAY_S, 2 * DAY_S
    s = hbm_pass(build_days(base_trace(), seeds=tuple(range(days))), HBM_BLOCKS)

    def run(pol, budget_day, est=None):
        reg = Region("flash", BB, days * budget_day, T, None)
        return Simulator(s, LLAMA31_70B, [reg], pol, tick_s=10.0, t_start=0.0, t_end=T, warmup_s=warm,
                         estimator=est).run()
    wa = run(WriteAll(), 1e30)
    measured = days - 2
    WA, V = wa.bytes_written[0] / measured, wa.saved_flash_s       # write-all bytes per measured day
    B = 0.10 * WA
    prof = LoadProfile([diurnal_multiplier(h % 24, low=0.3) for h in range(24 * days)])
    rows = []
    for name, mk, est in (("ROAR", roar, True), ("period-quota k=2", lambda: PeriodQuota(60.0, 2), False),
                          ("load-prop adaptive-k", lambda: AdaptiveK(profile=prof, **GRID[AK]), False)):
        r = run(mk(), B, new_estimator() if est else None)
        rows.append({"days": days, "measured_days": measured, "policy": name, "budget_share": 0.10,
                     "value_share": r.saved_flash_s / V, "written_over_budget": r.bytes_written[0] / (measured * B),
                     "max_ahead_days_of_budget": r.max_ahead_bytes[0] / B})
        write_csv(rows, tagged(f"e16_week_{days}d.csv"))
        log(f"  {name:24s} value {rows[-1]['value_share']:.4f} w/b {rows[-1]['written_over_budget']:.3f} "
            f"lead {rows[-1]['max_ahead_days_of_budget']:.2f} days")


if __name__ == "__main__":
    main()
