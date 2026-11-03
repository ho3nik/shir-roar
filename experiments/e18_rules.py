"""E18. Two admission rules any write policy can apply, and the write volume of an MLA model.

rules: known-load days 0-1 (E3 protocol), day 2 measured. (a) Partial last block: the last block of a prompt
whose length is not a multiple of 512 tokens can only be reused by a prompt identical up to that length
(vLLM and SGLang do not cache it at all). We measure the share of write-all's bytes it takes and the share
of write-all's savings it brings, then apply the rule to ROAR, adaptive k and the period quota at 10%.
(b) Parent check: write only blocks whose parent is in HBM or flash, the only ones a prefix lookup can reach.
deepseek: write-all bytes per day for DeepSeek-V3 (MLA, smaller KV) against Llama-3.1-70B when both get the
same 1 TB of HBM for KV, measured rather than scaled by the per-token KV ratio.
Usage: SHIR_TRACE=toolagent python e18_rules.py rules|deepseek"""
import sys

from common import HBM_BLOCKS, PeriodQuota, WriteAll, base_trace, hbm_pass, log, new_estimator, roar, tagged, \
    write_csv, write_json
from e3_days import sim
from e5_sensitivity import sim as sim_model
from e6_adaptive_k import GRID
from e10_load_aware import AK
from shirroar import DEEPSEEK_V3, LLAMA31_70B, AdaptiveK, ParentCheck, SkipBlocks, build_days


class PartialBlocks:
    """The partial last blocks of the base hour's prompts, under every hour's renaming (build_day adds
    namespace * (hour + 1) to each block id, so id % namespace recovers the base id). Holding the base set
    instead of the multi-day request list keeps memory low."""

    def __init__(self, base, namespace=10_000_000):
        self.ns = namespace
        self.base = {r.blocks[-1] for r in base if r.input_len % 512 and r.blocks}

    def __contains__(self, b):
        return b % self.ns in self.base


def partial_set(base):
    return PartialBlocks(base)


def rules():
    base = base_trace()
    part = partial_set(base)
    s = hbm_pass(build_days(base, seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    rows = []

    def add(name, b, mk, est=False):
        r = sim(s, mk(), 1e30 if b is None else b * WA2, est=new_estimator() if est else None)
        rows.append({"policy": name, "budget_share": b, "written_share": r.bytes_written[0] / WA2,
                     "written_over_budget": None if b is None else r.bytes_written[0] / (b * WA2),
                     "value_share": r.saved_flash_s / V2, "write_all_TB_day": WA2 / 1e12})
        write_csv(rows, tagged("e18_rules.csv"))
        log(f"  {name:34s} b={b} written {rows[-1]['written_share']:.4f} value {rows[-1]['value_share']:.4f}")

    add("write-all", None, WriteAll)
    add("write-all [skip partial]", None, lambda: SkipBlocks(WriteAll(), part, "skip partial"))
    add("write-all [parent check]", None, lambda: ParentCheck(WriteAll()))
    add("write-all [both]", None, lambda: ParentCheck(SkipBlocks(WriteAll(), part, "skip partial")))
    add("ROAR", 0.10, roar, True)
    add("ROAR [skip partial]", 0.10, lambda: SkipBlocks(roar(), part, "skip partial"), True)
    add("ROAR [no parent check]", 0.10, lambda: roar(parent_check=False), True)
    add("adaptive-k", 0.10, lambda: AdaptiveK(**GRID[AK]))
    add("adaptive-k [skip partial]", 0.10, lambda: SkipBlocks(AdaptiveK(**GRID[AK]), part, "skip partial"))
    add("adaptive-k [no parent check]", 0.10, lambda: AdaptiveK(parent_check=False, **GRID[AK]))
    add("period-quota k=2", 0.10, lambda: PeriodQuota(60.0, 2))
    add("period-quota k=2 [skip partial]", 0.10, lambda: SkipBlocks(PeriodQuota(60.0, 2), part, "skip partial"))
    add("period-quota k=2 [parent check]", 0.10, lambda: ParentCheck(PeriodQuota(60.0, 2)))


def deepseek():
    """DeepSeek-V3 only; Llama-3.1-70B's write-all bytes come from the rules run (same days, same HBM)."""
    import csv
    import os
    from common import RESULTS
    base = base_trace()
    part = partial_set(base)
    cap_ds = int(HBM_BLOCKS * LLAMA31_70B.kv_bytes_per_token / DEEPSEEK_V3.kv_bytes_per_token)
    s = hbm_pass(build_days(base, seeds=(0, 1)), cap_ds)
    wa = sim_model(s, WriteAll(), 1e30, DEEPSEEK_V3)
    sk = sim_model(s, SkipBlocks(WriteAll(), part, "skip partial"), 1e30, DEEPSEEK_V3)
    rules_rows = list(csv.DictReader(open(os.path.join(RESULTS, tagged("e18_rules.csv")))))
    ll = {r["policy"]: r for r in rules_rows}
    llama_wa = float(ll["write-all"]["write_all_TB_day"]) * 1e12
    out = {"llama": {"hbm_blocks": HBM_BLOCKS, "block_bytes": LLAMA31_70B.block_bytes(), "write_all_bytes_day2": llama_wa,
                     "write_all_skip_partial_bytes_day2": llama_wa * float(ll["write-all [skip partial]"]["written_share"])},
           "deepseek": {"hbm_blocks": cap_ds, "block_bytes": DEEPSEEK_V3.block_bytes(),
                        "write_all_bytes_day2": wa.bytes_written[0], "write_all_skip_partial_bytes_day2": sk.bytes_written[0],
                        "flash_value_s": wa.saved_flash_s, "hbm_value_s": wa.saved_hbm_s}}
    out["ratio_llama_over_deepseek"] = llama_wa / wa.bytes_written[0]
    out["kv_ratio"] = LLAMA31_70B.kv_bytes_per_token / DEEPSEEK_V3.kv_bytes_per_token
    write_json(out, tagged("e18_deepseek.json"))
    log(f"deepseek write-all {wa.bytes_written[0]/1e12:.1f} TB/day; ratio {out['ratio_llama_over_deepseek']:.2f} "
        f"(KV ratio {out['kv_ratio']:.2f})")


if __name__ == "__main__":
    {"rules": rules, "deepseek": deepseek}[sys.argv[1]]()
