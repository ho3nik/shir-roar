"""E13. Measured turn gaps from WildChat-1M, for the long-gap test (E8 with FRAC = "wildchat").

Input: parquet shards of allenai/WildChat-1M from Hugging Face (data/train-000NN-of-00014.parquet).
The paper uses shards 0-2 (179,571 conversations). WildChat stamps assistant turns only, so the gap
between consecutive assistant-turn timestamps of one conversation is the user's think time plus the
next response. Writes results/wildchat_gaps.json: counts, shares of follow-up gaps over 1, 2, 15 and
24 hours, the per-turn probability of a follow-up more than an hour later, and the list of those
>1 h gaps (E8 draws re-issue gaps from it). Needs pyarrow.
Usage: python e13_wildchat_gaps.py SHARD.parquet [SHARD.parquet ...]"""
import json
import os
import sys

import pyarrow.parquet as pq

from common import RESULTS


def main():
    files = sys.argv[1:]
    gaps, convs, multi, turns = [], 0, 0, 0
    for fn in files:
        f = pq.ParquetFile(fn)
        for rg in range(f.metadata.num_row_groups):
            t = f.read_row_group(rg, columns=["conversation.list.element.role", "conversation.list.element.timestamp"])
            for conv in t.column(0).to_pylist():
                convs += 1
                ts = [m["timestamp"].timestamp() for m in conv if m["role"] == "assistant" and m["timestamp"] is not None]
                turns += len(ts)
                multi += len(ts) >= 2
                gaps += [b - a for a, b in zip(ts, ts[1:])]
    g = sorted(x for x in gaps if x >= 0)
    n = len(g)
    q = lambda p: g[int(p * (n - 1))]
    over = lambda s: sum(1 for x in g if x > s)
    out = {"files": [os.path.basename(x) for x in files], "conversations": convs, "multi_turn_conversations": multi,
           "assistant_turns": turns, "follow_up_gaps": n, "negative_gaps_dropped": len(gaps) - n,
           "median_gap_s": q(0.5), "p90_gap_s": q(0.9), "p99_gap_s": q(0.99),
           **{f"gaps_over_{h}h": over(h * 3600) for h in (1, 2, 15, 24)},
           **{f"share_of_follow_ups_over_{h}h": over(h * 3600) / n for h in (1, 2, 15, 24)},
           "return_after_1h_per_turn": over(3600) / turns,
           "gaps_over_1h_s": [round(x) for x in g if x > 3600]}
    with open(os.path.join(RESULTS, "wildchat_gaps.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print({k: v for k, v in out.items() if k != "gaps_over_1h_s"})


if __name__ == "__main__":
    main()
