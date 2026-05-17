#!/bin/bash
# Launch a local HF model with vllm_launch + serve_local_qwen.py (solo or multi-node Ray cluster).
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

usage() {
    cat <<'EOF'
Usage: run-local-vllm.sh [solo|cluster] [options]

Modes:
  solo      Single node (--solo). Default.
  cluster   Multi-node from .env CLUSTER_NODES; starts Ray + vllm serve with
            --distributed-executor-backend ray.

Options:
  -h, --help    Show this help

Environment (optional):
  HF_HOME                    Host model tree (else read from .env)
  VLLM_MODEL_NAME            Subdir under HF_HOME (default: Qwen_Qwen3.5-2B)
  VLLM_MODEL_PATH            Full model path inside container (overrides VLLM_MODEL_NAME)
  VLLM_TENSOR_PARALLEL_SIZE  TP degree for cluster mode (default: node count)
  VLLM_EXTRA_ARGS            Extra flags passed to vllm serve (both modes)

Examples:
  ./run-local-vllm.sh solo
  ./run-local-vllm.sh cluster
  VLLM_MODEL_NAME=Qwen_Qwen3.6-27B ./run-local-vllm.sh solo
  VLLM_MODEL_NAME=Qwen_Qwen3.6-27B VLLM_TENSOR_PARALLEL_SIZE=2 ./run-local-vllm.sh cluster
EOF
}

cluster_node_count() {
    local nodes="${CLUSTER_NODES:-}"
    if [[ -z "$nodes" && -f "$SCRIPT_DIR/.env" ]]; then
        nodes=$(grep -E '^CLUSTER_NODES=' "$SCRIPT_DIR/.env" | head -1 | cut -d= -f2- | tr -d "'\"")
    fi
    if [[ -z "$nodes" ]]; then
        echo "Error: CLUSTER_NODES not set. Configure $SCRIPT_DIR/.env or export CLUSTER_NODES." >&2
        exit 1
    fi
    local -a arr
    IFS=',' read -ra arr <<< "$nodes"
    echo "${#arr[@]}"
}

MODE="solo"
while [[ $# -gt 0 ]]; do
    case "$1" in
        solo|--solo) MODE="solo"; shift ;;
        cluster|--cluster) MODE="cluster"; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# vllm_launch.sh reads HF_HOME when it starts (before .env is loaded), so export it here.
if [[ -z "${HF_HOME:-}" && -f "$SCRIPT_DIR/.env" ]]; then
    hf_home=$(grep -E '^HF_HOME=' "$SCRIPT_DIR/.env" | head -1 | cut -d= -f2- | tr -d "'\"")
    if [[ -n "$hf_home" ]]; then
        export HF_HOME="$hf_home"
    fi
fi

LAUNCH_ARGS=(--no-cache-dirs --launch-script serve_local_qwen.py)
CONTAINER_ENV=()

if [[ -n "${VLLM_MODEL_NAME:-}" ]]; then
    CONTAINER_ENV+=(-e "VLLM_MODEL_NAME=${VLLM_MODEL_NAME}")
fi
if [[ -n "${VLLM_MODEL_PATH:-}" ]]; then
    CONTAINER_ENV+=(-e "VLLM_MODEL_PATH=${VLLM_MODEL_PATH}")
fi
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    CONTAINER_ENV+=(-e "VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}")
fi

if [[ -n "${VLLM_MODEL_NAME:-}" ]]; then
    echo "Model: ${VLLM_MODEL_NAME} (under HF_HOME)"
elif [[ -n "${VLLM_MODEL_PATH:-}" ]]; then
    echo "Model: ${VLLM_MODEL_PATH}"
else
    echo "Model: Qwen_Qwen3.5-2B (default; set VLLM_MODEL_NAME to override)"
fi

if [[ "$MODE" == "solo" ]]; then
    echo "Mode: solo (single node)"
    LAUNCH_ARGS=(--solo "${LAUNCH_ARGS[@]}")
else
    node_count=$(cluster_node_count)
    tp_size="${VLLM_TENSOR_PARALLEL_SIZE:-$node_count}"
    echo "Mode: cluster (${node_count} node(s) from CLUSTER_NODES, tensor-parallel-size=${tp_size})"
    CONTAINER_ENV+=(
        -e "VLLM_DISTRIBUTED_BACKEND=ray"
        -e "VLLM_TENSOR_PARALLEL_SIZE=${tp_size}"
    )
fi

cd "$SCRIPT_DIR"
./vllm_launch.sh stop
if [[ ${#CONTAINER_ENV[@]} -gt 0 ]]; then
    echo "Container env: ${CONTAINER_ENV[*]}"
fi
./vllm_launch.sh "${CONTAINER_ENV[@]}" "${LAUNCH_ARGS[@]}"
