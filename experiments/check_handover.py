"""Check where ROAR's lead over the wear schedule comes from (conversation trace, day pair 0-1,
10% budget): price and lead over both days, sampled every two hours, plus the price at the end
of the fast first phase (06:00 on day 1). Writes results/check_handover_<trace>.json."""
import json
import os

from common import DAY_S, HBM_BLOCKS, RESULTS, ROAR_DEFAULT, WriteAll, base_trace, hbm_pass, log, new_estimator, roar, tagged
from e3_days import sim
from shirroar import build_days


def main():
    s = hbm_pass(build_days(base_trace(), seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    B = 0.10 * wa.bytes_written[0]
    r = sim(s, roar(), B, est=new_estimator(), record=True)
    ser = r.series
    t, lam, w, sc = ser["t"], ser["lam0"], ser["r0_bytes"], ser["r0_sched"]
    out = {"budget_day_bytes": B, "warm_s": ROAR_DEFAULT["warm_s"], "samples": []}
    for i in range(len(t)):
        if abs((t[i] % 7200.0)) < 1e-6 or i == len(t) - 1:
            out["samples"].append({"hour": t[i] / 3600.0, "price": lam[i], "lead_days_of_budget": (w[i] - sc[i]) / B})
    hand = min(range(len(t)), key=lambda i: abs(t[i] - ROAR_DEFAULT["warm_s"]))
    out["price_at_handover"] = lam[hand]
    out["lead_at_handover"] = (w[hand] - sc[hand]) / B
    day2 = [x for x, tt in zip(lam, t) if tt >= DAY_S]
    out["day2_price_min"], out["day2_price_max"] = min(day2), max(day2)
    with open(os.path.join(RESULTS, tagged("check_handover.json")), "w") as f:
        json.dump(out, f, indent=1)
    log(f"price at hand-over {out['price_at_handover']:.4g}, lead then {out['lead_at_handover']:.3f} day; day-2 price "
        f"{out['day2_price_min']:.3g}-{out['day2_price_max']:.3g}")
    for x in out["samples"]:
        log(f"  hour {x['hour']:5.1f}  price {x['price']:.4g}  lead {x['lead_days_of_budget']:+.3f} day")
    log("done check_handover")


if __name__ == "__main__":
    main()
