"""E9b. Sensitivity of ROAR to its gain on an evaluation trace (conversation, day pair 0-1, 10% budget).
Reported only as a range; the gain itself was chosen on the synthetic trace (e9_tune_synthetic.py)."""
import os
os.environ.setdefault("SHIR_TRACE", "conversation")
from common import HBM_BLOCKS, WriteAll, base_trace, hbm_pass, log, new_estimator, roar, tagged, write_csv  # noqa: E402
from e3_days import sim  # noqa: E402
from shirroar import build_days  # noqa: E402


def main():
    s = hbm_pass(build_days(base_trace(), seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    rows = []
    for eta in (0.0001, 0.0003, 0.001):
        r = sim(s, roar(eta=eta), 0.10 * WA2, est=new_estimator())
        rows.append({"eta": eta, "budget_share": 0.10, "value_share": r.saved_flash_s / V2,
                     "written_over_budget": r.bytes_written[0] / (0.10 * WA2)})
        write_csv(rows, tagged("e9b_eta_sensitivity.csv"))
        log(f"eta {eta}: value {rows[-1]['value_share']:.4f} w/b {rows[-1]['written_over_budget']:.3f}")


if __name__ == "__main__":
    main()
