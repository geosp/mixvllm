#!/usr/bin/env bash
set -euo pipefail

# Default model key (can be overridden with MODEL_NAME env var)
MODEL_KEY="${MODEL_NAME:-gpt-oss-20b}"
REGISTRY_PATH="/opt/vllm/model_registry.yml"

if [[ ! -f "$REGISTRY_PATH" ]]; then
  echo "❌ Registry file not found: $REGISTRY_PATH" >&2
  exit 1
fi

# Extract values from YAML (indentation-tolerant)
get_yaml_value() {
  local key="$1"
  awk -v model="$MODEL_KEY" -v key="$key" '
    $1 == model ":" { in_model=1; next }
    in_model && /^[^[:space:]]/ { in_model=0 }
    in_model && $0 ~ "^[[:space:]]*" key":" {
      val=$0
      sub("^[[:space:]]*" key":[[:space:]]*", "", val)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
      print val
      in_model=0
    }
  ' "$REGISTRY_PATH" | tr -d '\r' | tr -d '[:space:]'
}

MODEL=$(get_yaml_value model)
DTYPE=$(get_yaml_value dtype)
TP=$(get_yaml_value tensor_parallel_size)
GPU_UTIL=$(get_yaml_value gpu_memory_utilization)
MAX_SEQS=$(get_yaml_value max_num_seqs)

# Validation
for var in MODEL DTYPE TP GPU_UTIL MAX_SEQS; do
  if [[ -z "${!var:-}" ]]; then
    echo "❌ Missing '$var' for model '$MODEL_KEY' in $REGISTRY_PATH" >&2
    exit 1
  fi
done

echo "🚀 Launching model: ${MODEL_KEY}"
echo "  → model=$MODEL"
echo "  → dtype=$DTYPE, TP=$TP, GPU_UTIL=$GPU_UTIL, MAX_SEQS=$MAX_SEQS"
echo ""

python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "${MASTER_PORT:-8000}" \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --distributed-executor-backend ray \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs "$MAX_SEQS" \
  --dtype "$DTYPE" \
  --trust-remote-code