"""E3. Two synthetic days back to back; every policy runs both days and is measured on
day 2, so learned state (ROAR's estimator and price) is warm. Budgets are shares of day 2's
write-all bytes, spread evenly over both days. Usage: python e3_days.py SEED1 SEED2 [--full]
--full adds the best constant price in hindsight (10% budget) and the extra points of the
equal-wear sweep used for Fig. 1. Results are written after every run, so an interrupted run
keeps what it finished; --resume reruns only the rows that are missing from the CSV file (not
supported together with --full)."""
import math
import os
import sys

from common import (RESULTS, ROAR_DEFAULT, stepwise_tuned, tagged, DAY_S, HBM_BLOCKS, AdmitOnK, FixedPrice, OracleSet, PeriodQuota,
                    StepwiseQuota, WriteAll, base_trace, hbm_pass, log, new_estimator, oracle_select, roar, write_csv,
                    write_json)
from shirroar import LLAMA31_70B, Region, Simulator, build_days

ETA = float(os.environ.get("ROAR_ETA", ROAR_DEFAULT["eta"]))
BUDGETS = (0.05, 0.10, 0.20)
EXTRA_SWEEP = (0.025, 0.30)
T = 2 * DAY_S
BB = LLAMA31_70B.block_bytes()
SWEPT = {"ROAR": "ROAR (sweep)", "period-quota k=2": "period-quota k=2 (sweep)",
         "stepwise-quota": "stepwise-quota (sweep)", "oracle": "oracle (sweep)"}


def sim(stream, policy, budget_day, est=None, record=False):
    reg = Region("flash", BB, 2 * budget_day, T, None)
    return Simulator(stream, LLAMA31_70B, [reg], policy, tick_s=10.0, t_start=0.0, t_end=T, warmup_s=DAY_S,
                     estimator=est, record=record).run()


def day2_series(series):
    """keep day 2 only, one sample per minute, to keep the file small"""
    out = {}
    for k, ser in series.items():
        idx = [i for i in range(len(ser["t"])) if ser["t"][i] >= DAY_S and i % 6 == 5]
        out[k] = {kk: [ser[kk][i] for i in idx] for kk in ser}
    return out


def main(seeds, full, resume=False):
    if full and resume:
        raise SystemExit("--resume is not supported with --full")
    s = hbm_pass(build_days(base_trace(), seeds=seeds), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    log(f"seeds {seeds}: day-2 write-all {WA2/1e12:.1f} TB, flash value {V2:.0f} s, HBM value {wa.saved_hbm_s:.0f} s")
    rows, series = [], {}
    tag = f"{seeds[0]}-{seeds[1]}"
    out_csv = tagged(f"e3_days_{tag}.csv")
    done = set()
    if resume and os.path.exists(os.path.join(RESULTS, out_csv)):
        import csv
        with open(os.path.join(RESULTS, out_csv)) as f:
            rows = list(csv.DictReader(f))
        done = {(r["policy"], round(float(r["budget_share"]), 6) if r["budget_share"] not in ("", "None") else None)
                for r in rows}
        log(f"resuming: {len(rows)} rows already in {out_csv}")

    def todo(name, budget):
        return (name, round(budget, 6) if budget is not None else None) not in done

    def add(name, res, budget, extra=None):
        row = {"seeds": tag, "policy": name, "budget_share": budget, "eta": ETA,
               "written_share": res.bytes_written[0] / WA2,
               "written_over_budget": (res.bytes_written[0] / (budget * WA2)) if budget else None,
               "max_ahead_share_of_daily_budget": (res.max_ahead_bytes[0] / (budget * WA2)) if budget else None,
               "value_share": res.saved_flash_s / V2, "flash_saved_s": res.saved_flash_s,
               "hbm_saved_s": res.saved_hbm_s, "write_all_flash_saved_s": V2, "write_all_TB": WA2 / 1e12}
        if extra:
            row.update(extra)
        rows.append(row)
        if full and name in SWEPT:
            rows.append(dict(row, policy=SWEPT[name]))
        write_csv(rows, out_csv)
        log(f"  {name:26s} budget {budget}  written/WA {row['written_share']:.4f}  value {row['value_share']:.4f}")
        return row

    writes_k = {}
    for k in (2, 3, 4):
        if not todo(f"admit-on-{k}", None):
            continue
        r = sim(s, AdmitOnK(k), 0.10 * WA2, record=(full and k == 3))
        writes_k[k] = r.bytes_written[0] / WA2
        add(f"admit-on-{k}", r, None)
        if full and k == 3:
            series["admit-on-3"] = r.series
    for b in BUDGETS:
        B = b * WA2
        rec = full and b == 0.10
        for name, pol in (("period-quota k=1", PeriodQuota(60.0, 1)), ("period-quota k=2", PeriodQuota(60.0, 2)),
                          ("stepwise-quota", stepwise_tuned())):
            if not todo(name, b):
                continue
            r = sim(s, pol, B, record=rec)
            add(name, r, b)
            if rec:
                series[name] = r.series
        if todo("ROAR", b):
            pol = roar(eta=ETA)
            r = sim(s, pol, B, est=new_estimator(), record=True)
            lam = r.series["lam0"]
            half = len(lam) // 2
            lam_day2 = math.exp(sum(math.log(x) for x in lam[half:]) / max(1, len(lam) - half))
            add("ROAR", r, b, {"bound_intervals": pol.bound_intervals(),
                               "max_ahead_intervals": r.max_ahead_bytes[0] / (B * 10.0 / DAY_S),
                               "lam_day2_geomean": lam_day2})
            if rec:
                series["ROAR"] = r.series
        if full and b == 0.10:   # best constant price in hindsight: bisection in log space around ROAR's day-2 price
            lo, hi = lam_day2 / 4.0, lam_day2 * 4.0
            for _ in range(6):
                mid = math.sqrt(lo * hi)
                rr = sim(s, FixedPrice(mid, soft=ROAR_DEFAULT["soft"]), B, est=new_estimator())
                if rr.bytes_written[0] > B:
                    lo = mid
                else:
                    hi = mid
            r = sim(s, FixedPrice(hi, soft=ROAR_DEFAULT["soft"]), B, est=new_estimator(), record=True)
            add("static price (hindsight)", r, b, {"lam": hi})
            series["static price (hindsight)"] = r.series
        if todo("oracle", b):
            sel = oracle_select(s, LLAMA31_70B, int(2 * B / BB), by="value")
            add("oracle", sim(s, OracleSet(sel), B), b)
        if rec:
            write_json(day2_series(series), tagged(f"e3_days_series_{tag}.json"))
    if full:
        for b in sorted(set(EXTRA_SWEEP + tuple(round(writes_k[k], 4) for k in (2, 3, 4)))):
            B = b * WA2
            add("ROAR", sim(s, roar(eta=ETA), B, est=new_estimator()), b)
            add("period-quota k=2", sim(s, PeriodQuota(60.0, 2), B), b)
            add("stepwise-quota", sim(s, stepwise_tuned(), B), b)
            sel = oracle_select(s, LLAMA31_70B, int(2 * B / BB), by="value")
            add("oracle", sim(s, OracleSet(sel), B), b)
        # the extra points were also recorded under their main names; relabel them so that the main
        # names only hold the three main budgets
        main_b = set(BUDGETS)
        for r in rows:
            if r["policy"] in SWEPT and r["budget_share"] not in main_b:
                r["policy"] = r["policy"] + " (extra)"
        write_csv(rows, out_csv)
    log("done", out_csv)


if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    main((a, b), "--full" in sys.argv, "--resume" in sys.argv)
