"""Day-to-day spread of write-all bytes on known-load days (same cosine curve, different sampling seeds),
to show that a share of each day's write-all bytes is close to a fixed byte budget there."""
import json
import os
from common import DAY_S, HBM_BLOCKS, RESULTS, LLAMA31_70B, WriteAll, Region, base_trace, hbm_pass, tagged
from shirroar import Simulator
from shirroar import build_days

out = {}
for seeds in ((0, 1), (2, 3), (4, 5)):
    s = hbm_pass(build_days(base_trace(), seeds=seeds), HBM_BLOCKS)
    def wa(warm):
        reg = Region("f", LLAMA31_70B.block_bytes(), 1e30, 2 * DAY_S, None)
        return Simulator(s, LLAMA31_70B, [reg], WriteAll(), tick_s=10.0, t_start=0.0, t_end=2 * DAY_S,
                         warmup_s=warm).run().bytes_written[0]
    tot, d2 = wa(0.0), wa(DAY_S)
    d1 = tot - d2
    out[f"{seeds[0]}-{seeds[1]}"] = [d1 / 1e12, d2 / 1e12]
vals = [x for v in out.values() for x in v]
m = sum(vals) / len(vals)
out["max_rel_dev_from_mean"] = max(abs(x - m) / m for x in vals)
json.dump(out, open(os.path.join(RESULTS, tagged("day_variation.json")), "w"), indent=1)
print(out)
