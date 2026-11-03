"""E7. Does the reuse estimator transfer? The evaluation days are re-samplings of one hour, so an
estimator that learns on day 1 is tested on day 2 of the same population. Here the estimator is
trained on one trace and used, frozen, on the other.
  python e7_transfer.py train SEED1 SEED2      (SHIR_TRACE=X) saves results/est_X_SEEDS.pkl
  python e7_transfer.py test SEED1 SEED2 PKL... (SHIR_TRACE=Y) runs ROAR on Y with each saved
      estimator frozen (no learning at all), and warm-started (keeps learning), at 5/10/20%.
Training runs ROAR at a 10% budget for two days (the estimator observes every HBM eviction, so the
budget hardly matters for what it learns)."""
import os
import pickle
import sys

from common import (RESULTS, tagged, HBM_BLOCKS, WriteAll, base_trace, hbm_pass, log, new_estimator, roar, write_csv)
from e3_days import sim
from shirroar import ReuseEstimator, build_days

mode, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
seeds = (a, b)
trace = os.environ.get("SHIR_TRACE", "toolagent")
s = hbm_pass(build_days(base_trace(), seeds=seeds), HBM_BLOCKS)
wa = sim(s, WriteAll(), 1e30)
WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
if mode == "train":
    est = new_estimator()
    sim(s, roar(), 0.10 * WA2, est=est)
    path = os.path.join(RESULTS, f"est_{trace}_{a}-{b}.pkl")
    with open(path, "wb") as f:
        pickle.dump(est.state_dict(), f)
    log("saved", path, "windows closed:", est.cnt_all)
else:
    rows = []
    for bud in (0.05, 0.10, 0.20):
        B = bud * WA2
        variants = [("online (default)", new_estimator())]
        for pk in sys.argv[4:]:
            with open(pk, "rb") as f:
                st = pickle.load(f)
            src = os.path.basename(pk)[4:-4]
            variants.append((f"frozen from {src}", ReuseEstimator.from_state(st, frozen=True)))
            variants.append((f"warm from {src}", ReuseEstimator.from_state(st, frozen=False)))
        for name, est in variants:
            r = sim(s, roar(), B, est=est)
            rows.append({"trace": trace, "seeds": f"{a}-{b}", "estimator": name, "budget_share": bud,
                         "value_share": r.saved_flash_s / V2, "written_over_budget": r.bytes_written[0] / B})
            write_csv(rows, tagged(f"e7_transfer_{a}-{b}.csv"))
            log(f"  {bud} {name:40s} value {rows[-1]['value_share']:.4f} w/b {rows[-1]['written_over_budget']:.3f}")
