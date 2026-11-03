"""E17. Second production source: the Qwen-Bailian anonymous traces (scripts/convert_qwen.py), one hour at a
time. Same protocol as the known-load days of E3/E10 (daily cosine 0.3-1.0, two days measured on day 2,
budget a share of day 2's write-all bytes), Llama-3.1-70B KV, 1 TB HBM. At the 10% budget it runs ROAR, ROAR
just below 10% (for equal-bytes comparisons), adaptive k with and without the load curve, the period quota and
the load-proportional quota (five-year horizon) and the tuned stepwise quota.
Usage: SHIR_TRACE=qwenA_h1 python e17_qwen.py [SEED1 SEED2]"""
import sys

from common import HBM_BLOCKS, PeriodQuota, WriteAll, base_trace, hbm_pass, log, new_estimator, roar, stepwise_tuned, \
    tagged, write_csv
from e3_days import sim
from e6_adaptive_k import GRID
from e10_load_aware import AK, cosine_profile
from shirroar import AdaptiveK, ProfileQuota, build_days


EXTRA = (0.015, 0.03, 0.05, 0.065, 0.105, 0.11)   # more ROAR points, so every rule's bytes fall inside the sweep


def main():
    if "--extra" in sys.argv:
        return extra()
    seeds = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    s = hbm_pass(build_days(base_trace(), seeds=seeds), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    prof = cosine_profile(0.3)
    rows = []
    runs = [("ROAR", 0.10, lambda: roar(), True)] + \
           [("ROAR (sweep)", b, lambda: roar(), True) for b in (0.08, 0.085, 0.09, 0.095)] + [
        ("adaptive-k", 0.10, lambda: AdaptiveK(**GRID[AK]), False),
        ("load-prop adaptive-k", 0.10, lambda: AdaptiveK(profile=prof, **GRID[AK]), False),
        ("period-quota k=2", 0.10, lambda: PeriodQuota(60.0, 2), False),
        ("load-prop quota", 0.10, lambda: ProfileQuota(prof, 60.0, 2), False),
        ("stepwise-quota", 0.10, stepwise_tuned, False)]
    log(f"write-all {WA2/1e12:.1f} TB/day 2, flash value {V2/3600:.2f} GPU-h, HBM {wa.saved_hbm_s/3600:.2f}")
    for name, b, mk, est in runs:
        r = sim(s, mk(), b * WA2, est=new_estimator() if est else None)
        rows.append({"seeds": f"{seeds[0]}-{seeds[1]}", "policy": name, "budget_share": b,
                     "written_share": r.bytes_written[0] / WA2, "written_over_budget": r.bytes_written[0] / (b * WA2),
                     "value_share": r.saved_flash_s / V2, "write_all_flash_gpu_h": V2 / 3600.0,
                     "hbm_gpu_h": wa.saved_hbm_s / 3600.0, "write_all_TB": WA2 / 1e12})
        write_csv(rows, tagged(f"e17_qwen_{seeds[0]}-{seeds[1]}.csv"))
        log(f"  {name:22s} b={b} w/b {rows[-1]['written_over_budget']:.3f} value {rows[-1]['value_share']:.4f}")


def extra():
    import csv
    import os
    from common import RESULTS
    path = os.path.join(RESULTS, tagged("e17_qwen_0-1.csv"))
    rows = list(csv.DictReader(open(path)))
    have = {round(float(r["budget_share"]), 6) for r in rows if r["policy"].startswith("ROAR")}
    s = hbm_pass(build_days(base_trace(), seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    for b in EXTRA:
        if round(b, 6) in have:
            continue
        r = sim(s, roar(), b * WA2, est=new_estimator())
        rows.append({**{k: "" for k in rows[0]}, "seeds": "0-1", "policy": "ROAR (sweep)", "budget_share": b,
                     "written_share": r.bytes_written[0] / WA2, "written_over_budget": r.bytes_written[0] / (b * WA2),
                     "value_share": r.saved_flash_s / V2, "write_all_flash_gpu_h": V2 / 3600.0,
                     "hbm_gpu_h": wa.saved_hbm_s / 3600.0, "write_all_TB": WA2 / 1e12})
        write_csv(rows, tagged("e17_qwen_0-1.csv"))
        log(f"  sweep b={b} value {rows[-1]['value_share']:.4f}")


if __name__ == "__main__":
    main()
