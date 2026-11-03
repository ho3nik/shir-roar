"""E6. Stronger baselines.

  adaptive-k   admit-on-k whose k follows the wear schedule (policies.AdaptiveK), a grid of settings
  ROAR-refs    ROAR's price controller with the value replaced by the plain reference count
  ROAR-refs-d  same, reference count times depth-aware recomputation time
  day-1 price  a constant price that spends exactly the budget on day 1 (found by bisection on day 1
               alone), then held fixed on day 2: the realistic version of the hindsight price

Every run uses the e3_days set-up (two synthetic days, measured on day 2, budget a share of day 2's
write-all bytes). Usage:
  python e6_adaptive_k.py SEED1 SEED2 [--budgets 0.05,0.10,0.20] [--grid] [--only NAME,...]
--grid runs the whole adaptive-k grid (used on the arXiv tuning hour and, as a generous upper bound
for the baseline, on the evaluation traces); without it only the setting chosen on the arXiv hour runs.
Rows are appended to results/e6_adaptive_k_SEEDS_TRACE.csv after every run."""
import csv
import math
import os
import sys

from common import (RESULTS, ROAR_DEFAULT, tagged, DAY_S, HBM_BLOCKS, AdmitOnK, FixedPrice, PeriodQuota,
                    StepwiseQuota, WriteAll, base_trace, hbm_pass, log, new_estimator, roar, write_csv)
from e3_days import sim
from shirroar import AdaptiveK, build_days

GRID = {
    # deterministic integer k, band in seconds of budget, minimum dwell between changes
    "adaptive-k band0 dwell10": dict(band_s=0.0, dwell_s=10.0),
    "adaptive-k band0 dwell300": dict(band_s=0.0, dwell_s=300.0),
    "adaptive-k band600 dwell10": dict(band_s=600.0, dwell_s=10.0),
    "adaptive-k band3600 dwell10": dict(band_s=3600.0, dwell_s=10.0),
    "adaptive-k band3600 dwell300": dict(band_s=3600.0, dwell_s=300.0),
    "adaptive-k band21600 dwell300": dict(band_s=21600.0, dwell_s=300.0),
    "adaptive-k band21600 dwell3600": dict(band_s=21600.0, dwell_s=3600.0),
    # real-valued k with randomized rounding
    "adaptive-k dither step0.001": dict(dither=True, step=0.001),
    "adaptive-k dither step0.01": dict(dither=True, step=0.01),
    "adaptive-k dither step0.05": dict(dither=True, step=0.05),
    "adaptive-k dither step0.01 band3600": dict(dither=True, step=0.01, band_s=3600.0),
}
# chosen on the arXiv hour (e6 --grid on SHIR_TRACE=arxiv); see results/e6_adaptive_k_0-1_arxiv.csv
CHOSEN = os.environ.get("AK_CHOSEN", "adaptive-k dither step0.01")


def day1_price(base, seeds, B_day, day1_requests=None):
    """Bisection on day 1 alone for the constant price that spends B_day over day 1."""
    reqs = day1_requests if day1_requests is not None else build_days(base, seeds=(seeds[0],))
    s1 = hbm_pass(reqs, HBM_BLOCKS)
    from shirroar import LLAMA31_70B, Region, Simulator
    BB = LLAMA31_70B.block_bytes()

    def spend(lam):
        reg = Region("flash", BB, B_day, DAY_S, None)
        r = Simulator(s1, LLAMA31_70B, [reg], FixedPrice(lam), tick_s=10.0, t_start=0.0, t_end=DAY_S,
                      estimator=new_estimator()).run()
        return r.bytes_written[0] / B_day
    lo, hi = 1e-2, 1e3
    for _ in range(int(os.environ.get('DAY1_ITERS', '10'))):      # 10 halvings of ln(1e5) leave the price within about 1%
        mid = math.sqrt(lo * hi)
        if spend(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    del s1
    return hi


def main():
    seeds = (int(sys.argv[1]), int(sys.argv[2]))
    args = sys.argv[3:]
    budgets = (0.05, 0.10, 0.20)
    only = None
    if "--budgets" in args:
        budgets = tuple(float(x) for x in args[args.index("--budgets") + 1].split(","))
    if "--only" in args:
        only = set(args[args.index("--only") + 1].split(","))
    grid = "--grid" in args
    tag = f"{seeds[0]}-{seeds[1]}"
    out = tagged(f"e6_adaptive_k_{tag}.csv")
    rows = []
    path = os.path.join(RESULTS, out)
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
    done = {(r["policy"], round(float(r["budget_share"]), 6)) for r in rows}
    base = base_trace()
    s = hbm_pass(build_days(base, seeds=seeds), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    log(f"seeds {seeds}: write-all {WA2/1e12:.1f} TB, flash value {V2:.0f} s")

    def add(name, res, b, pol=None, extra=None):
        row = {"seeds": tag, "policy": name, "budget_share": b,
               "written_share": res.bytes_written[0] / WA2,
               "written_over_budget": res.bytes_written[0] / (b * WA2),
               "max_ahead_share_of_daily_budget": res.max_ahead_bytes[0] / (b * WA2),
               "value_share": res.saved_flash_s / V2, "flash_saved_s": res.saved_flash_s,
               "write_all_flash_saved_s": V2, "write_all_TB": WA2 / 1e12}
        if extra:
            row.update(extra)
        rows.append(row)
        write_csv(rows, out)
        log(f"  {name:40s} b={b} written/budget {row['written_over_budget']:.3f} value {row['value_share']:.4f}")

    policies = {}
    names = list(GRID) if grid else [CHOSEN]
    for n in names:
        policies[n] = lambda n=n: AdaptiveK(**GRID[n])
    policies["ROAR"] = lambda: roar()
    policies["ROAR-refs"] = lambda: roar(value_mode="refs")
    policies["ROAR-refs-depth"] = lambda: roar(value_mode="refs_depth")
    policies["ROAR-refs no parent check"] = lambda: roar(value_mode="refs", parent_check=False)
    policies["period-quota k=2"] = lambda: PeriodQuota(60.0, 2)
    policies["stepwise-quota"] = lambda: StepwiseQuota(300.0)
    for b in budgets:
        B = b * WA2
        for name, mk in policies.items():
            if only and name not in only:
                continue
            if (name, round(b, 6)) in done:
                continue
            pol = mk()
            est = new_estimator() if getattr(pol, "uses_estimator", False) else None
            r = sim(s, pol, B, est=est)
            extra = {}
            if isinstance(pol, AdaptiveK):
                kh = pol.k_hist[len(pol.k_hist) // 2:]
                extra = {"k_mean_day2": sum(kh) / len(kh), "k_min_day2": min(kh), "k_max_day2": max(kh)}
            add(name, r, b, pol, extra)
        name = "day-1 price"
        if (not only or name in only) and (name, round(b, 6)) not in done:
            # B is spread over two days in sim(); the day-1 budget is therefore B as well
            lam = day1_price(base, seeds, B)
            r = sim(s, FixedPrice(lam), B, est=new_estimator())
            add(name, r, b, extra={"lam": lam})
    log("done", out)


if __name__ == "__main__":
    main()
