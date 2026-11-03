# SHIR-ROAR
Paper live at: https://ho3nik.github.io/shir-roar/

Trace-driven simulator and experiments for the paper *Wear Before Capacity: Lifetime-Budgeted Prefix
KV Caching on High Bandwidth Flash*.

A prefix KV cache keeps the attention keys and values of earlier prompts so that a later request with
the same beginning can read them instead of recomputing them. High Bandwidth Flash (HBF) stacks NAND
dies next to the GPU: it holds far more than HBM, but each cell survives only a rated number of
program/erase cycles. The simulator asks, for every 512-token block that HBM evicts, whether it is
worth writing to flash, given that the flash has to last its rated life.

The two results the code supports:

- At a five-year life, a flash region written at its budget keeps each block for D·WAF/N days (D the
  life in days, WAF the write amplification, N the rated cycles), whatever its capacity. On TLC or QLC
  the write budget, not capacity, limits the cache, and running the dies as pseudo-SLC can matter as
  much as the admission policy.
- ROAR (read-often, append-rarely) writes a block only when its expected recomputation savings exceed
  a price on its wear. The price rises when wear runs ahead of the life schedule and falls when it
  lags, so cumulative wear stays within a proven constant of the schedule without a model of the daily
  load.

Nothing here has run on HBF hardware; all results come from simulation.

## Quick start

```bash
python -m pip install -e ".[test]"      # Python 3.10+; installs numpy, matplotlib, pytest, hypothesis
python -m pytest -q                     # 37 tests, under a minute
python demo/demo_check.py               # budgets, residence times, recompute times, price updates
python demo/demo_one_hour.py            # one real hour through four policies (about 10 s)
python demo/demo_two_days.py            # one cell of Table II (several minutes, about 3 GB of memory)
```

To regenerate every result, figure and `results/SUMMARY.md`:

```bash
scripts/run_all.sh                      # about 14 hours on 2 cores and 8 GB of memory
scripts/run_all.sh path/to/paper        # also writes numbers.tex and the tables into the paper folder
scripts/run_all.sh --tuning             # only the runs that chose the controller settings
```

`run_all.sh` runs two jobs at a time; each needs 2.5–3.5 GB. Every script writes its rows as it goes,
and most skip rows that already exist, so a stopped run can be restarted. Single experiments run from
`experiments/` with the trace chosen by `SHIR_TRACE`, for example
`cd experiments && SHIR_TRACE=conversation python e3_days.py 0 1`.

## Layout

| Path | Contents |
|---|---|
| `src/shirroar/trace.py` | Trace reader, prefix-hash check, synthetic days built from a one-hour trace |
| `src/shirroar/hbm.py` | One LRU pass over HBM that emits reference and eviction events, shared by every policy |
| `src/shirroar/flash.py` | Flash region: append-only FIFO log, wear ledger, linear service-life schedule |
| `src/shirroar/model.py` | KV sizes, recomputation time, cell presets, endurance arithmetic |
| `src/shirroar/estimator.py` | Online reuse estimate from closed 15-minute windows; session-aware variant |
| `src/shirroar/controller.py` | Wear-price update and its bound |
| `src/shirroar/policies.py` | ROAR and the baselines (write-all, admit-on-k, adaptive k, period and stepwise quotas, constant price, oracle set) |
| `src/shirroar/waf.py` | Write-amplification loop on the device's NAND-write counter; admission-rule wrappers |
| `src/shirroar/oracle.py` | Offline greedy selection with knowledge of future requests |
| `src/shirroar/sim.py` | The flash pass: prefix semantics, metrics, warm-up window |
| `experiments/` | One script per experiment (below), plus figure, summary and paper-number generators |
| `demo/` | Three short scripts for a live demonstration |
| `scripts/` | Trace download and conversion, and `run_all.sh` |
| `results/` | Every CSV and JSON the paper's numbers come from; file names carry the trace name |
| `docs/MODEL.md` | What the simulator models, what it leaves out, and the metrics |
| `tests/` | Unit, property (Hypothesis) and validation tests |

## Experiments

| Script | What it measures |
|---|---|
| `e1_endurance.py` | Five-year write budget per node by cell type and stack count (Table I) |
| `e2_hour_sweep.py` | Policies on the bare one-hour traces |
| `e3_days.py`, `e3_static.py` | Two known-load days, measured on day 2 (Table II, Fig. 1); best constant price in hindsight |
| `e4_cell_modes.py` | TLC, pSLC and mixed nodes with real FIFO capacity |
| `e5_sensitivity.py`, `e5f_tick_scaled.py` | Controller settings, HBM size, model, parent check |
| `e5a`–`e5e`, `e9_tune_synthetic.py`, `e9b_eta_sensitivity.py` | Choice of the controller gain on traces held out from evaluation, and its sensitivity |
| `e6_adaptive_k.py` | Budget-tracking adaptive k (11 variants), ROAR with reference counts as its value, constant price set on day 1 |
| `e7_transfer.py` | Reuse estimate trained on one trace and frozen on the other |
| `e8_long_gaps.py`, `e13_wildchat_gaps.py` | Reuse after more than an hour, at the rate and gaps measured in WildChat-1M |
| `e10_load_aware.py` | Load-proportional baselines; days whose load shape is drawn at random (Table III, confidence intervals) |
| `e11_params.py`, `check_handover.py` | Constants of the wear bound; where ROAR's lead over the schedule comes from |
| `e12_read_rate.py` | Peak write and read rates per stack; reads per block |
| `e15_tune_stepwise.py` | Setting of the stepwise quota, chosen on the held-out synthetic trace |
| `e16_week.py` | Several consecutive days, measured from day 3 |
| `e17_qwen.py` | Eight hours of the Qwen-Bailian traces |
| `e18_rules.py` | Skipping partial last blocks; the parent check; DeepSeek-V3 write volume at equal HBM bytes |
| `e19_session.py` | Turn number as an extra estimator class on the Qwen-Bailian hours |
| `e20_waf.py` | Unknown write amplification, corrected from the NAND-write counter |
| `check_day_variation.py` | Day-to-day spread of write-all bytes on known-load days |

## Validation

- The prefix-hash check passes on every trace: each block id has one parent and one position.
- The wear bound of `controller.py` is property-tested on adversarial inputs for both update rules.
- Admission on the k-th reference never writes a block whose parent is missing; random admission does.
- Recomputation times and budgets match hand calculations from the model constants.

## Data and licences

The Mooncake traces (kvcache-ai/Mooncake) and the Qwen-Bailian traces
(alibaba-edu/qwen-bailian-usagetraces-anon) are released under Apache-2.0; `data/README.md` lists the
files and how the Qwen traces were converted. WildChat-1M (allenai) is used only through the turn-gap
statistics in `results/wildchat_gaps.json`. The code in this repository has an MIT license. 
