#!/usr/bin/env bash
set -euo pipefail
#
# Copy a Hugging Face model directory from rank 0 to rank 1 using
# model_transfer.py (PyTorch distributed: set MODEL_TRANSFER_TORCH_BACKEND=nccl|gloo).
# Your container NCCL_* variables are respected automatically.
#
# Usage:
#   ./hf_transfer.sh <rank> <src_dir> <dest_dir> <master_addr> [-- extra model_transfer.py args...]
#
# On spark1 (rank 0), start first. Use the IP that spark2 reaches spark1 on (often the RoCE/data IP).
# On spark2 (rank 1), <src_dir> may be missing; only <dest_dir> must be writable (same path as on spark1 if mounts match).
#
# Examples (inside: docker exec -it sglang_node_tf5 bash):
#   cd /data/sglang/spark-sglang-stack-dashboard/model_transfer   # or wherever this repo is mounted
#   export MODEL_TRANSFER_TORCH_BACKEND=nccl
#   ./hf_transfer.sh 0 /data/hf/Meta-Llama-3-8B /data/hf/Meta-Llama-3-8B 192.168.x.x
#   ./hf_transfer.sh 1 /tmp/.unused /data/hf/Meta-Llama-3-8B 192.168.x.x
#
# If rank 0 has the same --src and --dest and files would be skipped, add: -- --all-files
#
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANK="${1:?rank: 0 (sender) or 1 (receiver)}"
SRC="${2:?source directory (must exist on rank 0)}"
DEST="${3:?destination directory}"
MASTER="${4:?MASTER_ADDR (usually spark1)}"
shift 4
EXTRA=("$@")

MASTER_PORT="${MASTER_PORT:-29500}"

exec python3 "$ROOT/model_transfer.py" \
  --mode rdma \
  --rank "$RANK" \
  --world-size 2 \
  --master-addr "$MASTER" \
  --master-port "$MASTER_PORT" \
  --src "$SRC" \
  --dest "$DEST" \
  "${EXTRA[@]}"
