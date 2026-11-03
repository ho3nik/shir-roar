"""E1. Endurance arithmetic against measured write-all volume (derived, not simulated).

For each cell type and number of HBF stacks per node: the five-year write budget per day, that
budget as a share of the day trace's write-all volume, the years a write-all policy would last,
and the FIFO residence time D*WAF/N of a region written exactly at its budget."""
from common import (tagged, DAY_S, HBM_BLOCKS, LLAMA31_70B, base_trace, day_trace, hbm_pass, log, write_all_bytes,
                    write_csv, write_json)
from shirroar.model import CELLS, DEEPSEEK_V3

STACKS = (4, 8, 16, 32)


def main():
    base = base_trace()
    s_hour = hbm_pass(base, HBM_BLOCKS)
    wa_hour, _, _ = write_all_bytes(s_hour, 0.0, 3600.0)
    day = day_trace(base)
    s_day = hbm_pass(day, HBM_BLOCKS)
    wa_day, _, _ = write_all_bytes(s_day, 0.0, DAY_S)
    log(f"write-all, Llama-3.1-70B: peak hour {wa_hour/1e12:.2f} TB/h, diurnal day {wa_day/1e12:.1f} TB/day")
    ratio = DEEPSEEK_V3.kv_bytes_per_token / LLAMA31_70B.kv_bytes_per_token
    rows = []
    for key in ("slc", "pslc", "tlc", "qlc"):
        cell = CELLS[key]
        for n in STACKS:
            budget_day = n * cell.daily_budget()
            for model, wa in (("Llama-3.1-70B", wa_day), ("DeepSeek-V3", wa_day * ratio)):
                share = budget_day / wa
                rows.append({
                    "cell": cell.name, "stacks_per_node": n, "model": model,
                    "budget_TB_per_day": budget_day / 1e12,
                    "write_all_TB_per_day": wa / 1e12,
                    "budget_share_of_write_all": share,
                    "write_all_lifetime_years": 5.0 * share,
                    "fifo_residence_h_at_budget": cell.fifo_residence_seconds() / 3600.0,
                    "capacity_TB": n * cell.capacity_bytes / 1e12,
                })
    write_csv(rows, tagged("e1_endurance.csv"))
    write_json({"write_all_peak_hour_TB": wa_hour / 1e12, "write_all_day_TB": wa_day / 1e12,
                "hbm_blocks": HBM_BLOCKS, "kv_ratio_dsv3_to_70b": ratio}, tagged("e1_endurance_meta.json"))


if __name__ == "__main__":
    main()
