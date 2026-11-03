"""E4. Same dies, different cell modes: all TLC, all pSLC, or half and half (two regions).

Each configuration gets its five-year budget per day (C * N / D, WAF 1) and its real capacity as
a FIFO log. Two synthetic days, measured on day 2. ROAR prices each region separately;
single-region baselines run on the single-region configurations. Reports GPU time saved by
flash per day and how many blocks the FIFO logs discarded. Because the FIFO discards blocks
whenever a log is full, whether or not they would be used again, ROAR is also run on every
configuration with unlimited capacity; the difference in savings is what capacity cost.
Results are written after every run."""
import csv
import os

from common import (tagged, RESULTS, DAY_S, HBM_BLOCKS, AdmitOnK, PeriodQuota, Region, WriteAll, base_trace, hbm_pass,
                    log, new_estimator, roar, write_csv)
from shirroar import LLAMA31_70B, Simulator, build_days
from shirroar.model import PSLC_512G, TLC_1536G

T = 2 * DAY_S
BB = LLAMA31_70B.block_bytes()


def regions_for(n_tlc, n_pslc, unlimited=False):
    regs = []
    if n_pslc:
        cap = None if unlimited else int(n_pslc * PSLC_512G.capacity_bytes // BB)
        regs.append(Region("pSLC", BB, 2 * n_pslc * PSLC_512G.daily_budget(), T, cap))
    if n_tlc:
        cap = None if unlimited else int(n_tlc * TLC_1536G.capacity_bytes // BB)
        regs.append(Region("TLC", BB, 2 * n_tlc * TLC_1536G.daily_budget(), T, cap))
    return regs


def sim(s, pol, regs, est=None):
    return Simulator(s, LLAMA31_70B, regs, pol, tick_s=10.0, t_start=0.0, t_end=T, warmup_s=DAY_S, estimator=est).run()


def main():
    s = hbm_pass(build_days(base_trace(), seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), [Region("f", BB, 1e30, T, None)])
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    # resume: all-TLC rows do not depend on the pSLC rating, so keep them from an earlier run
    prev = os.path.join(RESULTS, tagged("e4_cell_modes.csv"))
    # (the period quota is re-run: its horizon moved from the run to the device's five-year life)
    rows = [r for r in csv.DictReader(open(prev)) if r["config"] == "all TLC" and r["policy"] != "period-quota k=2"] \
        if os.path.exists(prev) else []
    have = {(int(r["stacks"]), r["config"], r["policy"]) for r in rows}
    for n in (8, 16, 32):
        for label, (nt, ns) in (("all TLC", (n, 0)), ("all pSLC", (0, n)), ("half pSLC, half TLC", (n // 2, n // 2))):
            pols = [("ROAR", lambda: roar())]
            if nt == 0 or ns == 0:
                pols += [("period-quota k=2", lambda: PeriodQuota(60.0, 2)), ("admit-on-3", lambda: AdmitOnK(3))]
            for name, mk in pols:
                if (n, label, name) in have:
                    continue
                regs = regions_for(nt, ns)
                budget_day = sum(r.endurance_bytes for r in regs) / 2
                res = sim(s, mk(), regs, est=new_estimator() if name == "ROAR" else None)
                row = {"stacks": n, "config": label, "policy": name, "budget_share": budget_day / WA2,
                       "capacity_TB": sum((r.cache_blocks or 0) * r.block_bytes for r in regs) / 1e12,
                       "written_over_budget": sum(res.bytes_written) / budget_day,
                       "flash_saved_gpu_h_per_day": res.saved_flash_s / 3600.0,
                       "value_share": res.saved_flash_s / V2,
                       "fifo_discards_two_days": sum(r.blocks_discarded for r in regs)}
                for r, w in zip(regs, res.bytes_written):
                    row[f"{r.name}_written_over_budget"] = w / (r.endurance_bytes / 2)
                if name == "ROAR":
                    ru = sim(s, roar(), regions_for(nt, ns, unlimited=True), est=new_estimator())
                    row["flash_saved_gpu_h_unlimited_capacity"] = ru.saved_flash_s / 3600.0
                    row["capacity_loss_share"] = 1.0 - res.saved_flash_s / ru.saved_flash_s if ru.saved_flash_s else 0.0
                rows.append(row)
                write_csv(rows, tagged("e4_cell_modes.csv"))
                log(n, label, name, f"budget {budget_day / WA2:.3f} written/budget {row['written_over_budget']:.3f} "
                                    f"GPU-h/day {row['flash_saved_gpu_h_per_day']:.2f} discards {row['fifo_discards_two_days']} "
                                    f"capacity loss {row.get('capacity_loss_share', float('nan')):.4f}")


if __name__ == "__main__":
    main()
