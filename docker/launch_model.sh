#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${MODEL_NAME:-gpt-oss-20b}"

# Extract values
get_yaml_value() {
  local key="$1"
  awk -v model="$MODEL_KEY" '
    $1 == model ":" { in_model=1; next }
    in_model && /^[^[:space:]]/ { in_model=0 }
    in_model && $1 == key ":" { print $2 }
  ' "./model_registry.yaml"
}

MODEL=$(get_yaml_value model)
DTYPE=$(get_yaml_value dtype)
TP=$(get_yaml_value tensor_parallel_size)
GPU_UTIL=$(get_yaml_value gpu_memory_utilization)
MAX_SEQS=$(get_yaml_value max_num_seqs)

echo "🚀 Launching model: ${MODEL_KEY}"
echo "  → model=$MODEL"
echo "  → dtype=$DTYPE, TP=$TP, GPU_UTIL=$GPU_UTIL, MAX_SEQS=$MAX_SEQS"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "${MASTER_PORT:-8000}" \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs "$MAX_SEQS" \
  --dtype "$DTYPE" \
  --distributed-executor-backend ray \
  --enable-prefix-caching \
  --chunked-prefill-enabled \
  --trust-remote-code
