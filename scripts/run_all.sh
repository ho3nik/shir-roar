#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper with one command:
#
#     scripts/run_all.sh [PAPER_DIR]
#
# PAPER_DIR is the folder of the LaTeX sources; numbers.tex and the tables are written there. Without it,
# the results, figures and results/SUMMARY.md are still produced.
#
# Prerequisites: Python 3.10+ and `pip install -e .[test] pyarrow`. The traces are in data/ (see data/README.md);
# results/wildchat_gaps.json comes from e13_wildchat_gaps.py and WildChat-1M parquet shards.
#
# Two lanes run in parallel. Each job needs 2.5-3.5 GB of memory, so on an 8 GB machine do not run more than
# two at once; e16_week.py runs alone at the end. Wall time on 2 cores: about 14 hours. Logs go to
# results/logs/. Every script writes its rows as it goes; e3_days.py, e8_long_gaps.py, e4_cell_modes.py and
# e10_load_aware.py skip rows that already exist, so a killed lane can simply be restarted.
#
#     scripts/run_all.sh --tuning           # the held-out tuning runs only (arXiv and synthetic traces)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ "${1:-}" == "--tuning" || -z "${1:-}" ]]; then PAPER=""; else PAPER=$(realpath -m "$1"); fi
cd "$ROOT/experiments"
L=../results/logs
mkdir -p "$L"
run() { local name=$1; shift; echo "$(date +%T) $name"; "$@" > "$L/$name.log" 2>&1 || echo "  FAILED: $name"; }

tuning() {   # choices fixed before the evaluation traces were run
  run e5a python3 e5a_gain.py
  run e5b python3 e5b_slow_gain.py
  run e5d python3 e5d_tune.py all
  run e5e python3 e5e_slower_gain.py
  SHIR_TRACE=arxiv run e6_arxiv python3 e6_adaptive_k.py 0 1 --grid --budgets 0.10,0.05
  SHIR_TRACE=synthetic run e9 python3 e9_tune_synthetic.py        # the gain (held out from both real traces)
  SHIR_TRACE=synthetic run e15 python3 e15_tune_stepwise.py       # the stepwise quota setting
}

lane() {     # one evaluation trace; $1 = toolagent | conversation, $2 = short tag
  local t=$1 g=$2
  export SHIR_TRACE=$t
  run e1_$g python3 e1_endurance.py
  run e2_$g python3 e2_hour_sweep.py
  run e3_${g}_0-1 python3 e3_days.py 0 1 --full
  run e3s_$g python3 e3_static.py
  for p in "2 3" "4 5"; do run e3_${g}_${p/ /-} python3 e3_days.py $p; done
  run e7train_$g python3 e7_transfer.py train 0 1
  run e6_${g}_0-1 python3 e6_adaptive_k.py 0 1 --grid
  ONLY="adaptive-k dither step0.01,adaptive-k band3600 dwell300,ROAR-refs,ROAR-refs-depth,day-1 price"
  for p in "2 3" "4 5"; do run e6_${g}_${p/ /-} python3 e6_adaptive_k.py $p --grid --only "$ONLY"; done
  for p in "0 1" "2 3" "4 5"; do       # load-aware baselines, then ROAR just below each budget (equal bytes)
    run e10_${g}_known_${p/ /-} python3 e10_load_aware.py known $p
    run e10_${g}_known_sweep_${p/ /-} python3 e10_load_aware.py known $p --sweep
  done
  for p in "100 101" "102 103" "104 105"; do
    run e10_${g}_unk_${p/ /-} python3 e10_load_aware.py unknownfix $p
    run e10_${g}_unk_sweep_${p/ /-} python3 e10_load_aware.py unknownfix $p --sweep
  done
  for a in 106 108 110 112 114 116 118; do     # extra seed pairs for the confidence interval
    run e10_${g}_lean_$a python3 e10_load_aware.py unknownfix $a $((a+1)) --budgets 0.1 --lean --with-sweep
  done
  run e12_$g python3 e12_read_rate.py
  run e18_${g}_rules python3 e18_rules.py rules
  run e18_${g}_ds python3 e18_rules.py deepseek
  run e20_$g python3 e20_waf.py
}

if [[ "${1:-}" == "--tuning" ]]; then tuning; exit 0; fi
tuning
lane toolagent ta &
lane conversation cv &
wait
# runs that need both traces' first results, or only one trace
SHIR_TRACE=toolagent run e7test_ta python3 e7_transfer.py test 0 1 ../results/est_conversation_0-1.pkl &
SHIR_TRACE=conversation run e7test_cv python3 e7_transfer.py test 0 1 ../results/est_toolagent_0-1.pkl &
wait
SHIR_TRACE=toolagent run e4_ta python3 e4_cell_modes.py &
SHIR_TRACE=toolagent run e5_ta python3 e5_sensitivity.py &
wait
SHIR_TRACE=conversation run e8_02 python3 e8_long_gaps.py 0.2 &
SHIR_TRACE=conversation run e8_0 python3 e8_long_gaps.py 0 &
wait
SHIR_TRACE=conversation run e8_wc python3 e8_long_gaps.py wildchat &
SHIR_TRACE=conversation run e9b python3 e9b_eta_sensitivity.py &
wait
SHIR_TRACE=conversation run handover python3 check_handover.py &
SHIR_TRACE=conversation run e11 python3 e11_params.py &
wait
for t in toolagent conversation; do SHIR_TRACE=$t run dayvar_$t python3 check_day_variation.py; done
# Qwen-Bailian hours (skipped if not converted)
for n in qwenA_h1 qwenA_h2 qwencoder_h1 qwencoder_h2 qwenthinking_h1 qwenthinking_h2 qwenB_h1 qwenB_h2; do
  [[ -f ../data/$n.jsonl ]] || continue
  SHIR_TRACE=$n run e17_$n python3 e17_qwen.py
  SHIR_TRACE=$n run e17x_$n python3 e17_qwen.py --extra
  [[ $n == qwenB_* ]] || SHIR_TRACE=$n run e19_$n python3 e19_session.py
done
SHIR_TRACE=conversation run e16_week python3 e16_week.py 5      # alone: five days in memory (seven do not fit in 8 GB)
for t in toolagent conversation; do SHIR_TRACE=$t run fig_$t python3 make_figures.py; done
run summary python3 make_summary.py
if [[ -n "$PAPER" ]]; then
  run paper python3 make_paper_inputs.py "$PAPER"
  echo "done: results in results/, numbers.tex and tables in $PAPER"
else
  echo "done: results in results/ (pass a paper folder to also write numbers.tex and the tables)"
fi
