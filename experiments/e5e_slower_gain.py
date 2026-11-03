"""E5e. Tuning trace only (arXiv hour): does value keep rising below the chosen gain? e5d stopped at
eta = 0.0003, the smallest gain it tried and the best. This adds eta = 0.0001 and 0.00003 at the 10%
budget with the same warm-up (eta0 = 0.05 for six hours). Smaller gains loosen the bound on the lead
(it scales with 1/eta) and slow the response to lasting changes in load, which two synthetic days
with the same daily curve cannot show; the defaults are not changed by this run."""
import os
os.environ.setdefault("SHIR_TRACE", "arxiv")
from common import DAY_S, HBM_BLOCKS, WriteAll, base_trace, hbm_pass, log, new_estimator, roar, write_csv  # noqa: E402
from e5d_tune import WARM, sim  # noqa: E402
from shirroar import build_days  # noqa: E402


def main():
    s = hbm_pass(build_days(base_trace(), seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    rows = []
    b = 0.10
    for eta in (0.0001, 0.00003):
        pol = roar(eta=eta, **WARM)
        r = sim(s, pol, b * WA2, est=new_estimator())
        row = {"trace": "arxiv", "policy": "ROAR", "budget_share": b, "eta": eta, "soft": 0.0,
               "value_share": r.saved_flash_s / V2, "written_over_budget": r.bytes_written[0] / (b * WA2),
               "max_ahead_share_of_daily_budget": r.max_ahead_bytes[0] / (b * WA2),
               "bound_intervals": pol.bound_intervals()}
        rows.append(row)
        write_csv(rows, "e5e_slower_gain_arxiv.csv")
        log(f"  eta {eta}: value {row['value_share']:.4f} written/budget {row['written_over_budget']:.3f} "
            f"ahead {row['max_ahead_share_of_daily_budget']:.3f} day bound {row['bound_intervals']:.0f} intervals")
    log("done e5e_slower_gain_arxiv.csv")


if __name__ == "__main__":
    main()
