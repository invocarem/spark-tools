#!/usr/bin/env bash
# Ray head on this machine — start the cluster; workers connect to VLLM_HOST_IP:6379.
# Edit MN_IF_NAME, VLLM_HOST_IP, image/paths for this host.

set -euo pipefail

export MONITOR_REPO_ROOT=/home/chenchen/code/spark-tools
export VLLM_IMAGE=vllm-node:latest
# Use the interface that carries the Ray / NCCL TCP traffic on this host
export MN_IF_NAME=enp1s0f1np1  # e.g. spark1 on 100.xx network
export VLLM_HOST_IP=192.168.100.11
export HF_HOME=/home/chenchen/huggingface

echo "Head: interface ${MN_IF_NAME}, IP ${VLLM_HOST_IP}"

bash "$(dirname "$0")/run_cluster.sh" "${VLLM_IMAGE}" "${VLLM_HOST_IP}" \
  --head "${HF_HOME}" \
  -n vllm_node \
  -e "VLLM_HOST_IP=${VLLM_HOST_IP}" \
  -e "UCX_NET_DEVICES=${MN_IF_NAME}" \
  -e "NCCL_SOCKET_IFNAME=${MN_IF_NAME}" \
  -e "GLOO_SOCKET_IFNAME=${MN_IF_NAME}" \
  -e "TP_SOCKET_IFNAME=${MN_IF_NAME}" \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
  -e NCCL_DEBUG=INFO
