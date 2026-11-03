# What the simulator models, and what it leaves out

## Pipeline

1. **Trace.** Requests in arrival order. Each prompt is a chain of 512-token block ids under prefix
   hashing: block i's id names the whole prefix through block i. Evaluation uses the two real hours of
   the Mooncake FAST'25 release (tool-and-agent, 23,608 requests; conversation, 12,031 requests) and
   eight hours of the Qwen-Bailian traces. A 24-hour day keeps each request of the hour with
   probability `0.3 + 0.7 * (1 + cos(2*pi*(h - 14)/24)) / 2` for hour h and gives every hour its own
   block-id namespace, so nothing is shared across hours. That removes cross-hour reuse, which is
   conservative for any cache. Main results run two such days back to back (different seeds) and
   measure the second, so learned state is warm; the write budget covers both days at a constant rate.
   A second family of days (`trace.build_days_random`) draws the trough, the peak hour and a one-hour
   burst at random each day.
2. **HBM pass** (`hbm.py`). An LRU cache of `HBM_BLOCKS` blocks (6,000 blocks, 1.0 TB of
   Llama-3.1-70B KV). Every referenced block enters HBM whether it hit in HBM, was read from flash or
   was recomputed, so HBM contents do not depend on the flash policy. The pass emits reference and
   eviction events once, and every policy replays the same stream. Requests are processed
   instantaneously in arrival order: no queueing, no batching, no decode-time occupancy.
3. **Flash pass** (`sim.py`). Each eviction of a block not already in flash asks the policy whether
   to write it, and to which region. A reference that misses HBM is a flash hit if the block is in
   flash and every earlier block of the same request was a hit. This is the prefix rule of vLLM and
   SGLang: matching stops at the first miss, and everything after it is recomputed.
4. **Flash regions** (`flash.py`). Append-only logs with FIFO discard when full, a wear ledger in host
   bytes, and a linear service-life schedule `rate = endurance_bytes / life_s`. Blocks are stored
   padded to 512 tokens, as paged KV caches allocate whole blocks. There is no garbage-collection
   model, no page or erase-block geometry, no read disturb and no retention error.

## Cost model (`model.py`)

- KV bytes per token: Llama-3.1-70B 327,680 B (80 layers, 8 KV heads of 128, K and V, BF16);
  DeepSeek-V3 70,272 B (61 layers x (512 + 64) x 2 B).
- Recomputation time of a block whose first token sits at depth d:
  `t_rec(d) = 512 * (F_lin + F_att * d) / Phi`, with F_lin = 2 x parameters (141.2 GFLOP for
  Llama-3.1-70B; about 3% high, since it counts the embedding and output layers), F_att the attention
  FLOPs per (query, key) pair, and Phi = 2.25 PFLOP/s x 0.5 (B200 dense BF16 at 50% utilization). It
  ranks blocks and converts savings to GPU time; it is not a kernel-level timing model.
- Flash read time: block bytes / 6.4 TB/s (four stacks at 1.6 TB/s), three orders of magnitude below
  `t_rec`.
- Endurance presets per stack: SLC 100,000 cycles at 512 GB; pSLC 20,000 at 512 GB; TLC 3,000 at
  1.536 TB; QLC 1,000 at 2.048 TB. Five-year life unless stated.

## Metrics

- `value_share`: GPU-seconds saved by flash hits, divided by those saved when every evicted block is
  written to unlimited flash. HBM hits are reported separately and are the same for every policy.
- `written_share`: flash bytes written, divided by write-all's bytes in the same configuration.
- `written_over_budget`: day-2 bytes written divided by one day's budget. Policies with feedback can be
  above 1 on day 2 if they carry a lead from day 1, or below 1 if they cannot find enough eligible
  blocks.
- `max_ahead_share_of_daily_budget`: the largest amount by which cumulative writes ran ahead of the
  linear schedule, checked every 10 seconds, as a share of one day's budget.
- Equal-bytes comparison (`make_paper_inputs.py`, `paper_macros.py`): ROAR's savings at exactly the
  bytes another policy wrote, by linear interpolation along ROAR's own budget sweep on the same seed
  pair. The lead is ROAR's interpolated savings minus the other policy's.

## Reuse estimate (`estimator.py`)

A 15-minute window opens at every HBM eviction of every block, including blocks already in flash, and
counts later references that miss HBM. Classes are the reference count so far (capped at 6), whether a
request has continued past the block, and a position bucket; sparse classes are shrunk toward coarser
ones. Only closed windows are used, so the estimate never looks ahead. `SessionEstimator` adds the turn
number of the last request that referenced the block, which the Qwen-Bailian traces record;
`hbm_pass` stores it in the upper bits of `has_child`, and traces without turns are unaffected.

## Write amplification (`waf.py`)

The main runs count host bytes at WAF = 1. `WAFRegion` gives a region a true WAF, a function of time
the controller never sees, and counts NAND bytes the way the "Physical Media Units Written" field of
the OCP Datacenter NVMe SSD specification (requirement SMART-1) does. `WAFLoop` reads both counters
every hour and sets the host-byte budget to the NAND budget divided by the ratio of NAND to host bytes
over the last six hours. WAF is a scaling here, not a garbage-collection model, so the loop is tested
against constant, stepped and hourly-random WAF, not against a WAF that depends on what is written.

## Baselines

See the module docstring of `policies.py`. They are reimplementations from the published descriptions,
not the original code; where a description leaves a constant open, the value used is stated in the code.
The period and load-proportional quotas spread their allowance over a five-year life. The stepwise
quota and the controller gain were chosen on traces that share no requests with the evaluation hours
(`e15_tune_stepwise.py`, `e9_tune_synthetic.py`); the adaptive-k variant was chosen on the arXiv hour,
which contains the same requests as the tool-and-agent hour.

## Limits

- One or two hours of real traffic per trace. Days are synthetic in their load curve and have no
  cross-hour reuse; `e8_long_gaps.py` adds reuse after more than an hour at the rate and gaps measured
  in WildChat-1M, as a probe, not a measurement of real reuse.
- The seed pairs re-sample one hour, so they show seed spread, not independent workloads.
- No device timing: write bandwidth, read-write interference and queueing are not modeled.
- The best constant price in hindsight cannot spend its budget exactly: blocks of one estimator class at
  the same depth share one value, so a whole group crosses the price threshold at once. `e3_static.py`
  takes the lowest price whose writes stay within the budget, and the paper compares it with ROAR at
  the bytes it actually wrote.
- ROAR's fast first phase ends at 06:00 on day 1 with the low night-time price, so day 2 still carries
  some lead over the schedule, within the proven bound.
- The value model ignores that a flash hit still occupies HBM and interconnect bandwidth.
