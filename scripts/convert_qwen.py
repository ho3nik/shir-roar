"""Convert the Qwen-Bailian anonymous traces (github.com/alibaba-edu/qwen-bailian-usagetraces-anon,
Apache-2.0) to the Mooncake format this simulator reads, one file per hour.

Qwen records give a timestamp in seconds, input and output lengths, and hash_ids of 16-token blocks. The
simulator uses 512-token blocks named by prefix hashes, so every 32 consecutive Qwen ids become one block
whose name hashes the previous block's name with those 32 ids (a partial last group becomes a partial last
block, as in Mooncake). Names are therefore prefix-chained by construction. Each two-hour file is split into
hours 1 and 2, with timestamps restarted at zero. Each request keeps its turn number within its conversation.
Usage: python convert_qwen.py QWEN_FILE.jsonl NAME   ->  data/NAME_h1.jsonl, data/NAME_h2.jsonl"""
import json
import os
import sys

GROUP = 32          # 32 x 16 tokens = 512 tokens


def main():
    src, name = sys.argv[1], sys.argv[2]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    names, outs = {}, {h: [] for h in (1, 2)}
    with open(src) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            t = float(r["timestamp"])
            hour = 1 if t < 3600.0 else 2 if t < 7200.0 else None
            if hour is None:
                continue
            ids = [int(x) for x in r["hash_ids"]]
            blocks, prev = [], 0
            for i in range(0, len(ids), GROUP):
                key = (prev, tuple(ids[i:i + GROUP]))
                prev = names.setdefault(key, len(names) + 1)
                blocks.append(prev)
            outs[hour].append({"timestamp": round((t - (hour - 1) * 3600.0) * 1000.0, 3),
                               "input_length": int(r["input_length"]), "output_length": int(r["output_length"]),
                               "turn": int(r.get("turn", 0)), "hash_ids": blocks})
    for h, rows in outs.items():
        rows.sort(key=lambda x: x["timestamp"])
        p = os.path.join(out_dir, f"{name}_h{h}.jsonl")
        with open(p + ".tmp", "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
        os.replace(p + ".tmp", p)
        print(p, len(rows), "requests")


if __name__ == "__main__":
    main()
