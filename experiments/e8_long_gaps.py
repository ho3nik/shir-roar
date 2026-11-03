"""E8. Long-gap reuse and cell mode. The synthetic days of E3/E4 have no reuse across hours, so they
cannot test whether a dense region's long FIFO residence (TLC: 14.6 h at budget) is worth more than a
pSLC region's larger budget but short residence (2.2 h). Here each request is re-issued, with
probability FRAC, after a log-uniform gap of 1 to 24 hours with the same block ids
(trace.build_days_reinject). Same dies (8 stacks per node), configurations as in E4 plus pSLC at the
break-even endurance (9K cycles, where pSLC's writable bytes equal TLC's). ROAR runs with its default
15-minute estimator window and with a 6-hour window, each with real capacity and with unlimited
capacity. Usage: python e8_long_gaps.py FRAC [SEED1 SEED2]
FRAC = "wildchat": re-issue rate and gaps measured from WildChat-1M (results/wildchat_gaps.json, from
e13_wildchat_gaps): each request returns with the measured probability that a turn is followed by
another turn more than an hour later, after a gap drawn from those measured >1 h gaps.
Rows already in the output file for configurations whose cells did not change are kept (resume)."""
import csv
import json
import os
import sys

from common import (tagged, RESULTS, DAY_S, HBM_BLOCKS, AdmitOnK, PeriodQuota, Region, WriteAll, base_trace, hbm_pass, log,
                    new_estimator, roar, write_csv)
from shirroar import LLAMA31_70B, Simulator, build_days_reinject
from shirroar.model import CellSpec, PSLC_512G, TLC_1536G

T = 2 * DAY_S
BB = LLAMA31_70B.block_bytes()
PSLC_9K = CellSpec("pSLC 9K, 512 GB", 512e9, 9_000)


def regions_for(cells, unlimited=False):
    regs = []
    for name, cell, n in cells:
        cap = None if unlimited else int(n * cell.capacity_bytes // BB)
        regs.append(Region(name, BB, 2 * n * cell.daily_budget(), T, cap))
    return regs


def sim(s, pol, regs, est=None):
    return Simulator(s, LLAMA31_70B, regs, pol, tick_s=10.0, t_start=0.0, t_end=T, warmup_s=DAY_S, estimator=est).run()


CONFIGS = {
    "all TLC": [("TLC", TLC_1536G, 8)],
    "all pSLC 20K": [("pSLC", PSLC_512G, 8)],
    "all pSLC 9K": [("pSLC", PSLC_9K, 8)],
    "half pSLC 20K, half TLC": [("pSLC", PSLC_512G, 4), ("TLC", TLC_1536G, 4)],
}


def main():
    seeds = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    gaps = None
    if sys.argv[1] == "wildchat":
        W = json.load(open(os.path.join(RESULTS, "wildchat_gaps.json")))
        frac, gaps, ftag = W["return_after_1h_per_turn"], W["gaps_over_1h_s"], "wildchat"
    else:
        frac = float(sys.argv[1])
        ftag = f"{frac:g}"
    s = hbm_pass(build_days_reinject(base_trace(), seeds=seeds, frac=frac, gaps=gaps), HBM_BLOCKS)
    wa = sim(s, WriteAll(), [Region("f", BB, 1e30, T, None)])
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    log(f"frac {frac}: write-all {WA2/1e12:.1f} TB/day 2, flash value {V2/3600:.2f} GPU-h, HBM {wa.saved_hbm_s/3600:.2f}")
    tag = f"{ftag}_{seeds[0]}-{seeds[1]}"
    out = tagged(f"e8_long_gaps_{tag}.csv")
    rows = []                                   # resume: configurations already in the results file are kept
    if os.path.exists(os.path.join(RESULTS, out)):
        rows = list(csv.DictReader(open(os.path.join(RESULTS, out))))
    have = {r["config"] for r in rows}
    for label, cells in CONFIGS.items():
        if label in have:
            continue
        pols = [("ROAR", lambda: roar(), dict()), ("ROAR, 6 h window", lambda: roar(), dict(horizon_s=21600.0))]
        if len(cells) == 1:
            pols += [("period-quota k=2", lambda: PeriodQuota(60.0, 2), None), ("admit-on-3", lambda: AdmitOnK(3), None)]
        for name, mk, est_kw in pols:
            for unl in (False, True):
                if unl and est_kw is None:
                    continue
                regs = regions_for(cells, unlimited=unl)
                budget_day = sum(r.endurance_bytes for r in regs) / 2
                res = sim(s, mk(), regs, est=new_estimator(**est_kw) if est_kw is not None else None)
                row = {"frac": frac, "seeds": f"{seeds[0]}-{seeds[1]}", "config": label, "policy": name,
                       "capacity": "unlimited" if unl else "real",
                       "budget_share": budget_day / WA2,
                       "capacity_TB": sum((r.cache_blocks or 0) * r.block_bytes for r in regs) / 1e12,
                       "written_over_budget": sum(res.bytes_written) / budget_day,
                       "flash_saved_gpu_h_per_day": res.saved_flash_s / 3600.0, "value_share": res.saved_flash_s / V2,
                       "write_all_flash_gpu_h": V2 / 3600.0, "hbm_saved_gpu_h": wa.saved_hbm_s / 3600.0,
                       "fifo_discards_two_days": sum(r.blocks_discarded for r in regs)}
                rows.append(row)
                write_csv(rows, out)
                log(label, name, row["capacity"], f"budget {row['budget_share']:.3f} w/b {row['written_over_budget']:.3f} "
                    f"GPU-h/day {row['flash_saved_gpu_h_per_day']:.2f}")


if __name__ == "__main__":
    main()
