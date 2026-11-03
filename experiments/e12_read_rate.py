"""E12. Peak flash read rate under ROAR at a 10% budget (seed pair 0-1, day 2), against the per-stack
read bandwidth the paper assumes (1.6 TB/s). Reports the busiest hour and the busiest 10-second
interval, as total bytes read from flash and per stack with 8 stacks."""
import json
from common import DAY_S, HBM_BLOCKS, WriteAll, base_trace, hbm_pass, new_estimator, roar, tagged, write_json
from e3_days import sim
from shirroar import LLAMA31_70B, build_days

STACKS, STACK_BW = 8, 1.6e12


def main():
    s = hbm_pass(build_days(base_trace(), seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    r = sim(s, roar(), 0.10 * wa.bytes_written[0], est=new_estimator(), record=True)
    t, h = r.series["t"], r.series["flash_hits"]
    bb = LLAMA31_70B.block_bytes()
    t0 = t[0] - (t[1] - t[0])
    day2 = [(ti, hi) for ti, hi in zip(t, h) if ti >= t0 + DAY_S]
    # busiest 10-second interval and busiest hour on day 2
    tick = max(h[i + 1] - h[i] for i in range(len(h) - 1) if t[i] >= t0 + DAY_S)
    hours = {}
    prev = None
    for ti, hi in day2:
        if prev is not None:
            k = int((ti - t0 - DAY_S) // 3600)
            hours[k] = hours.get(k, 0) + hi - prev
        prev = hi
    peak_h = max(hours.values())
    dt = t[1] - t[0]
    out = {"block_bytes": bb, "tick_s": dt, "peak_hour_hits": peak_h, "peak_tick_hits": tick,
           "peak_hour_TB": peak_h * bb / 1e12,
           "peak_hour_stack_GBs": peak_h * bb / 3600 / STACKS / 1e9,
           "peak_tick_stack_GBs": tick * bb / dt / STACKS / 1e9,
           "peak_tick_share_of_stack_bw": tick * bb / dt / STACKS / STACK_BW,
           "flash_hits_day2": r.flash_hits,
           # reads per block id over both days (a block id rewritten after leaving flash counts once,
           # so this overstates reads per written copy)
           "max_reads_per_block": max(r.block_reads.values()),
           "p99_reads_per_block": sorted(r.block_reads.values())[int(0.99 * (len(r.block_reads) - 1))],
           "mean_reads_per_read_block": sum(r.block_reads.values()) / len(r.block_reads)}
    write_json(out, tagged("e12_read_rate.json"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
