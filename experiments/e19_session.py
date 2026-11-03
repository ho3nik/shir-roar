"""E19. A session-aware reuse estimate on the Qwen-Bailian hours, whose requests carry their turn number
within a conversation. SessionEstimator adds the turn bucket of the last request that referenced the block
(1, 2, 3-4, 5+) to the estimator's classes; a request's turn number is known when it arrives, so the
estimate stays online. Known-load days 0-1, 10% budget, day 2 measured. Reports ROAR with the default
estimate, with the session-aware estimate, with the bare reference count, and the offline oracle, so the
share of the online-to-oracle gap the session feature closes can be read off.
Usage: SHIR_TRACE=qwenA_h1 python e19_session.py"""
from common import EST_DEFAULT, HBM_BLOCKS, OracleSet, WriteAll, base_trace, hbm_pass, log, new_estimator, \
    oracle_select, roar, tagged, write_csv
from e3_days import BB, sim
from shirroar import LLAMA31_70B, SessionEstimator, build_days

B = 0.10


def main():
    base = base_trace()
    s = hbm_pass(build_days(base, seeds=(0, 1)), HBM_BLOCKS)
    turns = sum(1 for r in base if r.turn > 1) / len(base)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    rows = []

    def add(name, r):
        rows.append({"policy": name, "budget_share": B, "value_share": r.saved_flash_s / V2,
                     "written_over_budget": r.bytes_written[0] / (B * WA2), "followup_share": turns})
        write_csv(rows, tagged("e19_session.csv"))
        log(f"  {name:28s} value {rows[-1]['value_share']:.4f} w/b {rows[-1]['written_over_budget']:.3f}")

    add("ROAR", sim(s, roar(), B * WA2, est=new_estimator()))
    add("ROAR session-aware", sim(s, roar(), B * WA2, est=SessionEstimator(**EST_DEFAULT)))
    add("ROAR reference count", sim(s, roar(value_mode="refs"), B * WA2, est=new_estimator()))
    sel = oracle_select(s, LLAMA31_70B, int(2 * B * WA2 / BB), by="value")
    add("oracle", sim(s, OracleSet(sel), B * WA2))


if __name__ == "__main__":
    main()
