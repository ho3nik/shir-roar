"""E5f. The 60-second-interval variant of E5 with both gains scaled by six, so the price moves at the
same rate per second as the default (E5's plain "tick 60 s" keeps the gain per interval and so moves
six times more slowly). Appends one row to results/e5_sensitivity_toolagent.csv if it is missing;
e5_sensitivity.py now includes this variant, so a full rerun produces the same row."""
import csv
import os

os.environ.setdefault("SHIR_TRACE", "toolagent")
from common import (DAY_S, HBM_BLOCKS, RESULTS, ROAR_DEFAULT, WriteAll, base_trace, hbm_pass, log, new_estimator,  # noqa: E402
                    roar, tagged, write_csv)
from e5_sensitivity import B, sim  # noqa: E402
from shirroar import LLAMA31_70B, build_days  # noqa: E402

NAME = "tick 60 s, gains x6"


def main():
    path = os.path.join(RESULTS, tagged("e5_sensitivity.csv"))
    rows = list(csv.DictReader(open(path)))
    if any(r["variant"] == NAME for r in rows):
        log("already present"); return
    s = hbm_pass(build_days(base_trace(), seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    pol = roar(eta=6 * ROAR_DEFAULT["eta"], eta0=6 * ROAR_DEFAULT["eta0"])
    tick = 60.0
    r = sim(s, pol, B * WA2, LLAMA31_70B, new_estimator(), tick)
    row = {"study": "ROAR variants", "variant": NAME, "value_share": r.saved_flash_s / V2,
           "written_over_budget": r.bytes_written[0] / (B * WA2),
           "max_ahead_share_of_daily_budget": r.max_ahead_bytes[0] / (B * WA2),
           "max_ahead_intervals": r.max_ahead_bytes[0] / (B * WA2 * tick / DAY_S),
           "bound_intervals": pol.bound_intervals()}
    i = max(k for k, x in enumerate(rows) if x["study"] == "ROAR variants") + 1
    rows.insert(i, row)
    write_csv(rows, tagged("e5_sensitivity.csv"))
    log(f"{NAME}: value {row['value_share']:.4f} written/budget {row['written_over_budget']:.3f}")


if __name__ == "__main__":
    main()
