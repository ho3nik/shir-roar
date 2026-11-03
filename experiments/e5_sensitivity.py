"""E5. Sensitivity at a 10% budget, two synthetic days measured on day 2: soft band, tick length,
estimator horizon, depth weighting, parent check, HBM size and model (KV size and recompute cost).
Also compares the measured lead of cumulative wear over the schedule with the Proposition 1 bound.
Results are written after every run."""
from common import (ROAR_DEFAULT, tagged, DAY_S, HBM_BLOCKS, AdmitOnK, OracleSet, PeriodQuota, StepwiseQuota, WriteAll,
                    base_trace, hbm_pass, log, new_estimator, oracle_select, roar, write_csv)
from shirroar import DEEPSEEK_V3, LLAMA31_70B, Region, Simulator, build_days

B = 0.10
T = 2 * DAY_S


def sim(s, pol, budget_day, model=LLAMA31_70B, est=None, tick=10.0):
    reg = Region("f", model.block_bytes(), 2 * budget_day, T, None)
    return Simulator(s, model, [reg], pol, tick_s=tick, t_start=0.0, t_end=T, warmup_s=DAY_S, estimator=est).run()


def main():
    days = build_days(base_trace(), seeds=(0, 1))
    rows = []

    def block(tag, s, model, variants):
        wa = sim(s, WriteAll(), 1e30, model)
        WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
        for name, pol, est, tick in variants:
            r = sim(s, pol, B * WA2, model, est, tick)
            interval_budget = B * WA2 * tick / DAY_S
            row = {"study": tag, "variant": name, "value_share": r.saved_flash_s / V2,
                   "written_over_budget": r.bytes_written[0] / (B * WA2),
                   "max_ahead_share_of_daily_budget": r.max_ahead_bytes[0] / (B * WA2),
                   "max_ahead_intervals": r.max_ahead_bytes[0] / interval_budget,
                   "bound_intervals": pol.bound_intervals() if hasattr(pol, "bound_intervals") else None}
            rows.append(row)
            write_csv(rows, tagged("e5_sensitivity.csv"))
            log(tag, name, f"value {row['value_share']:.4f} written/budget {row['written_over_budget']:.3f} "
                           f"ahead {row['max_ahead_intervals']:.0f} intervals (bound {row['bound_intervals']})")

    s = hbm_pass(days, HBM_BLOCKS)
    block("ROAR variants", s, LLAMA31_70B, [
        ("default", roar(), new_estimator(), 10.0),
        ("soft band 0.3", roar(soft=0.3), new_estimator(), 10.0),
        ("tick 60 s", roar(), new_estimator(), 60.0),   # same gain per interval: six times slower per second
        ("tick 60 s, gains x6", roar(eta=6 * ROAR_DEFAULT["eta"], eta0=6 * ROAR_DEFAULT["eta0"]), new_estimator(), 60.0),
        ("horizon 300 s", roar(), new_estimator(horizon_s=300.0), 10.0),
        ("horizon 1800 s", roar(), new_estimator(horizon_s=1800.0), 10.0),
        ("no depth weighting", roar(depth_weighted=False), new_estimator(), 10.0),
        ("no parent check", roar(parent_check=False), new_estimator(), 10.0),
        ("no warm-up gain", roar(eta0=None, warm_s=0.0), new_estimator(), 10.0),
    ])
    del s   # free the event stream before building the next one (memory)
    for cap in (1500, 24000):
        sc = hbm_pass(days, cap)
        wa = sim(sc, WriteAll(), 1e30)
        sel = oracle_select(sc, LLAMA31_70B, int(2 * B * wa.bytes_written[0] / LLAMA31_70B.block_bytes()), by="value")
        block(f"HBM {cap} blocks", sc, LLAMA31_70B, [
            ("ROAR", roar(), new_estimator(), 10.0), ("period quota k=2", PeriodQuota(60.0, 2), None, 10.0),
            ("stepwise quota", StepwiseQuota(300.0), None, 10.0), ("admit-on-3", AdmitOnK(3), None, 10.0),
            ("oracle", OracleSet(sel), None, 10.0)])
        del sc, sel
    cap_ds = int(HBM_BLOCKS * LLAMA31_70B.kv_bytes_per_token / DEEPSEEK_V3.kv_bytes_per_token)
    sd = hbm_pass(days, cap_ds)
    block("DeepSeek-V3 (same HBM bytes)", sd, DEEPSEEK_V3, [
        ("ROAR", roar(), new_estimator(), 10.0), ("period quota k=2", PeriodQuota(60.0, 2), None, 10.0),
        ("stepwise quota", StepwiseQuota(300.0), None, 10.0), ("admit-on-3", AdmitOnK(3), None, 10.0)])
    write_csv(rows, tagged("e5_sensitivity.csv"))


def quotas_only():
    """Re-run only the two quota baselines of the HBM-size and model studies (five-year quota horizon, tuned
    stepwise quota from E15) and replace their rows in the existing results file."""
    import csv
    import json
    import os
    from common import RESULTS
    path = os.path.join(RESULTS, tagged("e5_sensitivity.csv"))
    rows = list(csv.DictReader(open(path)))
    c = json.load(open(os.path.join(RESULTS, "e15_stepwise_choice_synthetic.json")))
    days = build_days(base_trace(), seeds=(0, 1))
    cap_ds = int(HBM_BLOCKS * LLAMA31_70B.kv_bytes_per_token / DEEPSEEK_V3.kv_bytes_per_token)
    for tag, cap, model in ((f"HBM 1500 blocks", 1500, LLAMA31_70B), (f"HBM 24000 blocks", 24000, LLAMA31_70B),
                            ("DeepSeek-V3 (same HBM bytes)", cap_ds, DEEPSEEK_V3)):
        sc = hbm_pass(days, cap)
        wa = sim(sc, WriteAll(), 1e30, model)
        WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
        for name, pol in (("period quota k=2", PeriodQuota(60.0, 2)),
                          ("stepwise quota", StepwiseQuota(c["quantum_s"], tuple(c["steps"]), c["base_k"]))):
            r = sim(sc, pol, B * WA2, model)
            for row in rows:
                if row["study"] == tag and row["variant"] == name:
                    row.update({"value_share": r.saved_flash_s / V2, "written_over_budget": r.bytes_written[0] / (B * WA2),
                                "max_ahead_share_of_daily_budget": r.max_ahead_bytes[0] / (B * WA2)})
            write_csv(rows, tagged("e5_sensitivity.csv"))
            log(tag, name, f"value {r.saved_flash_s / V2:.4f}")
        del sc


if __name__ == "__main__":
    import sys
    quotas_only() if "--quotas" in sys.argv else main()
