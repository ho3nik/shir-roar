"""E10. Load-aware baselines and days whose load the policies do not know.

  known   SEED1 SEED2  the E3 days (daily cosine, 0.3 to 1.0, peak 14:00). Adds a period quota and an
                       adaptive k whose schedule follows the true load curve instead of a straight line.
  unknown SEED1 SEED2  days whose trough (0.2-0.5), peak hour (11:00-17:00) and a one-hour 1.5x burst are
                       drawn per day (trace.build_days_random). Load-aware baselines get the family's mean
                       curve (trough 0.35, peak 14:00); ROAR and the flat rules get nothing. A constant
                       price is calibrated on day 1 and held on day 2.

Budgets are shares of day 2's write-all bytes spread over both days, as in E3; everything is measured on
day 2. Rows are appended after every run. Usage: python e10_load_aware.py MODE SEED1 SEED2 [--budgets ...]"""
import csv
import os
import sys

from common import (RESULTS, DAY_S, HBM_BLOCKS, stepwise_tuned, FixedPrice, OracleSet, PeriodQuota, StepwiseQuota, WriteAll, base_trace,
                    hbm_pass, log, new_estimator, oracle_select, roar, tagged, write_csv)
from e3_days import sim
from e6_adaptive_k import GRID, day1_price
from shirroar import LLAMA31_70B, AdaptiveK, LoadProfile, ProfileQuota, Region, Simulator, build_days, build_days_random
from shirroar.trace import diurnal_multiplier

AK = "adaptive-k dither step0.01"      # the adaptive-k setting chosen on the arXiv tuning hour (e6_adaptive_k.py)
BB = LLAMA31_70B.block_bytes()


def cosine_profile(low):
    return LoadProfile([diurnal_multiplier(h % 24, low=low) for h in range(48)])


def yesterday_profile(s, mean):
    """Load model 'yesterday': day 2's schedule follows day 1's measured hourly load (HBM evictions, i.e.
    write-all bytes, per hour); day 1 itself, with no history, follows the mean curve. Each day's weights are
    normalized to the same total, so each day still gets half the run's budget."""
    from shirroar.hbm import EVICT
    hours = [0.0] * 24
    for k in range(len(s.kind)):
        if s.kind[k] == EVICT and s.t[k] < DAY_S:
            hours[int(s.t[k] // 3600) % 24] += 1.0
    d1 = mean.w[:24]
    n1, n2 = sum(d1), sum(hours) or 1.0
    return LoadProfile([x / n1 for x in d1] + [x / n2 for x in hours])


def main():
    mode, seeds = sys.argv[1], (int(sys.argv[2]), int(sys.argv[3]))
    args = sys.argv[4:]
    budgets = (0.05, 0.10, 0.20)
    if "--budgets" in args:
        budgets = tuple(float(x) for x in args[args.index("--budgets") + 1].split(","))
    base = base_trace()
    if mode == "known":
        reqs = build_days(base, seeds=seeds)
        prof = cosine_profile(0.3)
        info = {}
    else:   # "unknown" (budget a share of the day's own write-all) or "unknownfix" (fixed bytes per day)
        reqs, params, hourly = build_days_random(base, seeds)
        prof = cosine_profile(0.35)
        info = {"day1": params[0], "day2": params[1]}
        log("day params", info)
    s = hbm_pass(reqs, HBM_BLOCKS)
    del reqs
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    # Budget base. A real device has a fixed byte budget per day. In "unknownfix" mode the budget is a share
    # of the write-all bytes of a day with the family's mean load curve (trough 0.35, peak 14:00, no burst),
    # so a day with more traffic than the mean gets no more budget; "known" and "unknown" use the day's own.
    BASE = WA2
    if mode == "unknownfix":
        from shirroar import build_day
        sm = hbm_pass(build_day(base, hours=24, low=0.35, high=1.0, peak_hour=14, seed=seeds[1]), HBM_BLOCKS)
        BASE = Simulator(sm, LLAMA31_70B, [Region("f", BB, 1e30, DAY_S, None)], WriteAll(), tick_s=10.0,
                         t_start=0.0, t_end=DAY_S).run().bytes_written[0]
        del sm
        log(f"fixed budget base (mean-curve day): {BASE/1e12:.1f} TB vs this day's {WA2/1e12:.1f} TB")
    tag = f"{mode}_{seeds[0]}-{seeds[1]}"
    out = tagged(f"e10_load_aware_{tag}.csv")
    rows = []
    path = os.path.join(RESULTS, out)
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
    done = {(r["policy"], round(float(r["budget_share"]), 6)) for r in rows}
    log(f"{tag}: write-all {WA2/1e12:.1f} TB, flash value {V2/3600:.2f} GPU-h")

    def add(name, res, b, extra=None):
        row = {"mode": mode, "seeds": f"{seeds[0]}-{seeds[1]}", "policy": name, "budget_share": b,
               "written_share": res.bytes_written[0] / WA2, "written_over_budget": res.bytes_written[0] / (b * BASE),
               "value_share": res.saved_flash_s / V2, "flash_saved_s": res.saved_flash_s,
               "write_all_flash_saved_s": V2, "hbm_saved_s": res.saved_hbm_s, "budget_base_TB": BASE / 1e12,
               "write_all_TB": WA2 / 1e12, "flash_hits": res.flash_hits, "blocks_written": res.blocks_written[0],
               "max_ahead_share_of_daily_budget": res.max_ahead_bytes[0] / (b * BASE)}
        row.update({f"{k}_{kk}": vv for k, v in info.items() for kk, vv in v.items()})
        if extra:
            row.update(extra)
        rows.append(row)
        write_csv(rows, out)
        log(f"  {name:34s} b={b} w/b {row['written_over_budget']:.3f} value {row['value_share']:.4f}")

    SWEEP = (0.045, 0.085, 0.09, 0.095, 0.17, 0.18)   # ROAR just below each budget, for equal-bytes comparisons
    if "--sweep" in args:
        for b in SWEEP:
            if ("ROAR (sweep)", round(b, 6)) not in done:
                add("ROAR (sweep)", sim(s, roar(), b * BASE, est=new_estimator()), b)
        log("done", out)
        return
    yprof = yesterday_profile(s, prof)
    pols = {"load-prop quota": lambda: ProfileQuota(prof, 60.0, 2),
            "load-prop adaptive-k": lambda: AdaptiveK(profile=prof, **GRID[AK]),
            "yesterday adaptive-k": lambda: AdaptiveK(profile=yprof, **GRID[AK]),
            "yesterday quota": lambda: ProfileQuota(yprof, 60.0, 2)}
    lean = "--lean" in args      # extra seed pairs: only what the ROAR vs adaptive-k comparison needs
    if lean:
        pols = {"ROAR": lambda: roar(), "load-prop adaptive-k": pols["load-prop adaptive-k"],
                "yesterday adaptive-k": pols["yesterday adaptive-k"]}
    elif mode in ("unknown", "unknownfix"):
        pols.update({"ROAR": lambda: roar(), "ROAR-refs": lambda: roar(value_mode="refs"),
                     "adaptive-k": lambda: AdaptiveK(**GRID[AK]),
                     "period-quota k=2": lambda: PeriodQuota(60.0, 2), "stepwise-quota": lambda: stepwise_tuned()})
    for b in budgets:
        B = b * BASE
        for name, mk in pols.items():
            if (name, round(b, 6)) in done:
                continue
            pol = mk()
            add(name, sim(s, pol, B, est=new_estimator() if getattr(pol, "uses_estimator", False) else None), b)
        if mode in ("unknown", "unknownfix") and not lean and ("day-1 price", round(b, 6)) not in done:
            # day 1 of this pair, alone (build_days_random draws each day from its own seed)
            lam = day1_price(base, seeds, B, day1_requests=build_days_random(base, (seeds[0],))[0])
            add("day-1 price", sim(s, FixedPrice(lam), B, est=new_estimator()), b, {"lam": lam})
        if mode in ("unknown", "unknownfix") and not lean and b == 0.10 and ("oracle", round(b, 6)) not in done:
            sel = oracle_select(s, LLAMA31_70B, int(2 * B / BB), by="value")
            add("oracle", sim(s, OracleSet(sel), B), b)
        if "--with-sweep" in args:      # the sweep points that sit below this budget
            for bs in [x for x in SWEEP if b - 0.035 < x < b]:
                if ("ROAR (sweep)", round(bs, 6)) not in done:
                    add("ROAR (sweep)", sim(s, roar(), bs * BASE, est=new_estimator()), bs)
    log("done", out)


if __name__ == "__main__":
    main()
