#!/usr/bin/env bash
# Download the public Mooncake traces (kvcache-ai/Mooncake, Apache-2.0) and check their SHA-256.
#   FAST25-release/traces/: tool-and-agent and conversation hours (evaluation), synthetic (unused)
#   FAST25-release/arxiv-trace/: the earlier release of the tool-and-agent hour (tuning only)
set -euo pipefail
cd "$(dirname "$0")/../data"
BASE="https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release"
fetch() {  # url file sha256
  curl -fsSL -o "$2" "$1"
  echo "$3  $2" | sha256sum -c -
}
fetch "$BASE/traces/toolagent_trace.jsonl"    toolagent_trace.jsonl    48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71
fetch "$BASE/traces/conversation_trace.jsonl" conversation_trace.jsonl b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df
fetch "$BASE/traces/synthetic_trace.jsonl"    synthetic_trace.jsonl    bd070915a98fc0ed264d7cfef2ce746002eb3076a695ec31ba2674c0111ec131
fetch "$BASE/arxiv-trace/mooncake_trace.jsonl" mooncake_trace.jsonl    b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711
