"""E3s. Refines the best constant price in hindsight (day pair 0-1, 10% budget).

e3_days.py runs six bisection steps on a bracket of x/4 to x4 around ROAR's day-2 price. That leaves
the price within a factor 16**(1/64) = 1.044 of the boundary, and because day-2 writes change about
twice as fast as the price, the constant price wrote 8-14% less than its budget. This script continues
the bisection inside that final bracket for five more steps (to within 0.14%), so the constant price
spends its budget like the other policies, and records its day-2 series.
Usage: SHIR_TRACE=... python e3_static.py"""
import csv
import math
import os

from common import (RESULTS, ROAR_DEFAULT, DAY_S, HBM_BLOCKS, FixedPrice, WriteAll, base_trace, hbm_pass, log,
                    new_estimator, tagged, write_csv, write_json)
from e3_days import day2_series, sim
from shirroar import build_days

STEPS = 5


def main():
    seeds = (0, 1)
    tag = f"{seeds[0]}-{seeds[1]}"
    prev = [r for r in csv.DictReader(open(os.path.join(RESULTS, tagged(f"e3_days_{tag}.csv"))))
            if r["policy"] == "static price (hindsight)"]
    if not prev:
        raise SystemExit("run e3_days.py 0 1 --full first")
    s = hbm_pass(build_days(base_trace(), seeds=seeds), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    B = 0.10 * WA2
    hi = float(prev[0]["lam"])
    lo = hi / 16.0 ** (1.0 / 64.0)
    best = None
    for i in range(STEPS):
        mid = math.sqrt(lo * hi)
        r = sim(s, FixedPrice(mid, soft=ROAR_DEFAULT["soft"]), B, est=new_estimator(), record=True)
        over = r.bytes_written[0] > B
        log(f"  step {i + 1}: price {mid:.6g} day-2 written/budget {r.bytes_written[0] / B:.4f} value {r.saved_flash_s / V2:.4f}")
        if over:
            lo = mid
        else:
            hi = mid
            best = (mid, r)
    if best is None:
        r = sim(s, FixedPrice(hi, soft=ROAR_DEFAULT["soft"]), B, est=new_estimator(), record=True)
        best = (hi, r)
    lam, r = best
    row = {"seeds": tag, "policy": "static price (hindsight)", "budget_share": 0.10, "eta": None,
           "written_share": r.bytes_written[0] / WA2, "written_over_budget": r.bytes_written[0] / B,
           "max_ahead_share_of_daily_budget": r.max_ahead_bytes[0] / B, "value_share": r.saved_flash_s / V2,
           "flash_saved_s": r.saved_flash_s, "hbm_saved_s": r.saved_hbm_s, "write_all_flash_saved_s": V2,
           "write_all_TB": WA2 / 1e12, "lam": lam, "refined_steps": 6 + STEPS,
           "previous_lam": float(prev[0]["lam"]), "previous_value_share": float(prev[0]["value_share"]),
           "previous_written_share": float(prev[0]["written_share"])}
    write_csv([row], tagged(f"e3_static_{tag}.csv"))
    write_json(day2_series({"static price (hindsight)": r.series}), tagged(f"e3_static_series_{tag}.json"))
    log(f"static price refined: written/budget {row['written_over_budget']:.4f} value {row['value_share']:.4f} "
        f"(was {row['previous_written_share'] / 0.10:.4f} and {row['previous_value_share']:.4f})")
    log("done", tagged(f"e3_static_{tag}.csv"))


if __name__ == "__main__":
    main()
