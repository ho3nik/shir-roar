"""E15. Held-out tuning of the stepwise quota (the EMC patent baseline), on the FAST'25 synthetic trace as in
E9, so the setting used on the evaluation traces is out of sample. Grid: quantum 60, 300 or 900 s; slack steps
(2,3,4), (3,4,5) or (2,4,6); within the allowance write every evicted block (base 1) or only blocks with two
or more references (base 2). Selection rule, the same as E9's and fixed before looking at the results: the
highest mean savings over the 5% and 10% budgets among settings whose day-2 writes stay within 1.05x budget
at both. Usage: SHIR_TRACE=synthetic python e15_tune_stepwise.py"""
import itertools
import os
os.environ.setdefault("SHIR_TRACE", "synthetic")
from common import HBM_BLOCKS, WriteAll, base_trace, hbm_pass, log, tagged, write_csv, write_json  # noqa: E402
from e3_days import sim  # noqa: E402
from shirroar import StepwiseQuota, build_days  # noqa: E402

QUANTA = (60.0, 300.0, 900.0)
STEPS = ((2, 3, 4), (3, 4, 5), (2, 4, 6))
BASES = (1, 2)
BUDGETS = (0.05, 0.10)


def main():
    s = hbm_pass(build_days(base_trace(), seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    rows = []
    for q, st, bk in itertools.product(QUANTA, STEPS, BASES):
        key = f"q{q:g} steps{'-'.join(map(str, st))} base{bk}"
        for b in BUDGETS:
            r = sim(s, StepwiseQuota(q, st, bk), b * WA2)
            rows.append({"setting": key, "quantum_s": q, "steps": "-".join(map(str, st)), "base_k": bk,
                         "budget_share": b, "value_share": r.saved_flash_s / V2,
                         "written_over_budget": r.bytes_written[0] / (b * WA2)})
            write_csv(rows, tagged("e15_tune_stepwise.csv"))
            log(f"  {key} b {b}: value {rows[-1]['value_share']:.4f} w/b {rows[-1]['written_over_budget']:.3f}")
    by = {}
    for r in rows:
        by.setdefault(r["setting"], []).append(r)
    ok = {k: v for k, v in by.items() if len(v) == len(BUDGETS) and all(x["written_over_budget"] <= 1.05 for x in v)}
    best = max(ok, key=lambda k: sum(x["value_share"] for x in ok[k]))
    b0 = ok[best][0]
    write_json({"chosen": best, "quantum_s": b0["quantum_s"], "steps": [int(x) for x in b0["steps"].split("-")],
                "base_k": b0["base_k"], "default": "q300 steps2-3-4 base1"}, tagged("e15_stepwise_choice.json"))
    log("chosen stepwise setting:", best)


if __name__ == "__main__":
    main()
