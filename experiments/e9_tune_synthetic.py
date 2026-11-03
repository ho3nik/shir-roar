"""E9. Held-out tuning. Chooses ROAR's gain eta, and runs the adaptive-k grid for comparison, on the FAST'25 synthetic
trace (stretched to one hour), which shares no requests with the tool-and-agent or conversation hours,
so every evaluation result is out of sample. Same protocol as e5d (two synthetic days, measured on day
2, warm-up gain 0.05 for six hours). Selection rule, fixed before looking at the results: the highest
mean savings over the 5% and 10% budgets among settings whose day-2 writes stay within 1.05x budget at
both. Usage: SHIR_TRACE=synthetic python e9_tune_synthetic.py"""
import os
os.environ.setdefault("SHIR_TRACE", "synthetic")
from common import (DAY_S, HBM_BLOCKS, WriteAll, base_trace, hbm_pass, log, new_estimator, roar, tagged,  # noqa: E402
                    write_csv)
from e5d_tune import WARM, sim  # noqa: E402
from e6_adaptive_k import GRID  # noqa: E402
from shirroar import AdaptiveK, build_days  # noqa: E402

ETAS = (0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01)
BUDGETS = (0.05, 0.10)


def pick(rows, key):
    by = {}
    for r in rows:
        by.setdefault(r[key], []).append(r)
    ok = {k: v for k, v in by.items() if len(v) == len(BUDGETS) and all(x["written_over_budget"] <= 1.05 for x in v)}
    return max(ok, key=lambda k: sum(x["value_share"] for x in ok[k])) if ok else None


def main():
    s = hbm_pass(build_days(base_trace(), seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    log(f"synthetic: write-all {WA2/1e12:.1f} TB, flash value {V2/3600:.2f} GPU-h")
    rows, ak = [], []
    for b in BUDGETS:
        for eta in ETAS:
            pol = roar(eta=eta, **WARM)
            r = sim(s, pol, b * WA2, est=new_estimator())
            rows.append({"policy": "ROAR", "eta": eta, "budget_share": b, "value_share": r.saved_flash_s / V2,
                         "written_over_budget": r.bytes_written[0] / (b * WA2),
                         "max_ahead_share_of_daily_budget": r.max_ahead_bytes[0] / (b * WA2)})
            write_csv(rows + ak, tagged("e9_tune.csv"))
            log(f"  ROAR eta {eta} b {b}: value {rows[-1]['value_share']:.4f} w/b {rows[-1]['written_over_budget']:.3f}")
        for name, kw in GRID.items():
            r = sim(s, AdaptiveK(**kw), b * WA2)
            ak.append({"policy": name, "eta": None, "budget_share": b, "value_share": r.saved_flash_s / V2,
                       "written_over_budget": r.bytes_written[0] / (b * WA2)})
            write_csv(rows + ak, tagged("e9_tune.csv"))
            log(f"  {name} b {b}: value {ak[-1]['value_share']:.4f} w/b {ak[-1]['written_over_budget']:.3f}")
    log("chosen eta:", pick(rows, "eta"), " chosen adaptive-k:", pick(ak, "policy"))


if __name__ == "__main__":
    main()
