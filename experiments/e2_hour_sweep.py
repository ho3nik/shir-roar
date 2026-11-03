"""E2. One-hour Mooncake trace, write budget sweep, train/test split by time.

The first 30 minutes train ROAR's reuse estimator (as a running system would have learned from
earlier traffic); all policies are then compared on the last 30 minutes, starting from empty
HBM and flash. Baselines need no training. ROAR is also reported cold (no training) to show the
cost of a cold start. Budgets are shares of the bytes the write-all policy writes in the test
window. Values are shares of the GPU time write-all's flash hits save in that window."""
from common import (tagged, HBM_BLOCKS, AdmitOnK, OracleSet, PeriodQuota, RandomFraction, StepwiseQuota, WriteAll,
                    base_trace, hbm_pass, log, new_estimator, oracle_select, roar, run, write_all_bytes, write_csv)
from shirroar import LLAMA31_70B

BUDGETS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
T0, T1 = 1800.0, 3600.0


def main():
    base = base_trace()
    train = [r for r in base if r.t < T0]
    test = [r for r in base if r.t >= T0]
    rows = []
    for cap in (0, HBM_BLOCKS):
        s_tr, s_te = hbm_pass(train, cap), hbm_pass(test, cap)
        est0 = new_estimator()
        run(s_tr, WriteAll(), 1e30, 0.0, T0, estimator=est0)       # learn from the first half
        wa, wa_val, hbm_val = write_all_bytes(s_te, T0, T1)
        log(f"HBM {cap} blocks: write-all {wa/1e12:.2f} TB, flash value {wa_val:.0f} s, HBM value {hbm_val:.0f} s")
        bb = LLAMA31_70B.block_bytes()

        def add(name, res, budget):
            rows.append({"hbm_blocks": cap, "policy": name, "budget_share": budget,
                         "written_share": res.bytes_written[0] / wa, "value_share": res.saved_flash_s / wa_val,
                         "flash_saved_s": res.saved_flash_s, "hbm_saved_s": res.saved_hbm_s,
                         "write_all_flash_saved_s": wa_val, "flash_hits": res.flash_hits,
                         "max_ahead_share_of_budget": (res.max_ahead_bytes[0] / (budget * wa)) if budget else None})

        for k in (2, 3, 4, 5, 6):
            add(f"admit-on-{k}", run(s_te, AdmitOnK(k), 1e30, T0, T1), None)
        for b in BUDGETS:
            B = b * wa
            add("random", run(s_te, RandomFraction(b, seed=3), B, T0, T1), b)
            add("period-quota k=1", run(s_te, PeriodQuota(60.0, 1), B, T0, T1), b)
            add("period-quota k=2", run(s_te, PeriodQuota(60.0, 2), B, T0, T1), b)
            add("stepwise-quota", run(s_te, StepwiseQuota(300.0), B, T0, T1), b)
            add("ROAR (cold)", run(s_te, roar(), B, T0, T1, estimator=new_estimator()), b)
            add("ROAR (trained)", run(s_te, roar(), B, T0, T1, estimator=est0.stats_copy()), b)
            sel = oracle_select(s_te, LLAMA31_70B, int(B / bb), by="value")
            add("oracle", run(s_te, OracleSet(sel), B, T0, T1), b)
            log(f"  budget {b:.3f} done")
    write_csv(rows, tagged("e2_hour_sweep.csv"))


if __name__ == "__main__":
    main()
