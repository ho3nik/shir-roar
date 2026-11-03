# Data

| File | Source | Used for |
|---|---|---|
| `toolagent_trace.jsonl` | Mooncake FAST'25 release, tool-and-agent hour, 23,608 requests | evaluation |
| `conversation_trace.jsonl` | Mooncake FAST'25 release, conversation hour, 12,031 requests | evaluation |
| `mooncake_trace.jsonl` | Mooncake arXiv release of the tool-and-agent hour | tuning only |
| `synthetic_trace.jsonl` | Mooncake FAST'25 release, built from public datasets, Poisson arrivals | held-out tuning |
| `qwen{A,B,coder,thinking}_h{1,2}.jsonl` | Qwen-Bailian traces (chat, API, coding, reasoning), one hour per file | second source (E17, E19) |

`scripts/fetch_mooncake.sh` downloads the Mooncake files from kvcache-ai/Mooncake (Apache-2.0) and checks
their SHA-256.

The arXiv file and the tool-and-agent file contain the same requests: block ids map one to one, output
lengths are identical, timestamps differ by a constant factor of 0.9825, and 98.9% of requests have the
same number of blocks. Results on tool-and-agent are therefore not independent of the tuning; results
on the conversation hour are.

Format: one JSON object per request with `timestamp` (ms), `input_length`, `output_length` and
`hash_ids`, one id per 512-token block of the prompt. Ids follow prefix hashing: every id appears at
exactly one position and has exactly one parent, which the test suite checks. The Qwen files also
carry `turn`, the request's turn number in its conversation.

## Qwen-Bailian conversion

The Qwen-Bailian traces (github.com/alibaba-edu/qwen-bailian-usagetraces-anon, Apache-2.0) give hash ids
of 16-token blocks over two hours. `scripts/convert_qwen.py` groups every 32 consecutive ids into one
512-token block, names it by a hash of the previous block's name and those 32 ids, so names are
prefix-chained, and splits each file into two one-hour files with timestamps restarted at zero:

```bash
python scripts/convert_qwen.py qwen_traceA_blksz_16.jsonl qwenA
```

The traces contain no request content and no model identity, and cover one or two hours each.
