#!/usr/bin/env bash
# Ray worker on this machine — join the cluster at RAY_HEAD_IP.
# Copy and edit: RAY_HEAD_IP, VLLM_HOST_IP, MN_IF_NAME, image/paths for each host.

set -euo pipefail

export MONITOR_REPO_ROOT=/home/chenchen/code/spark-tools
export VLLM_IMAGE=vllm-node:latest
# Interface used for Ray / NCCL TCP (this host)
export MN_IF_NAME=enp1s0f1np1
# This worker's address (must be unique per worker host)
export VLLM_HOST_IP=192.168.100.12
# Ray head address (same on all workers)
export RAY_HEAD_IP=192.168.100.11
export HF_HOME=/home/chenchen/huggingface

echo "Worker: interface ${MN_IF_NAME}, node IP ${VLLM_HOST_IP}, head ${RAY_HEAD_IP}"

bash "$(dirname "$0")/run_cluster.sh" "${VLLM_IMAGE}" "${RAY_HEAD_IP}" \
  --worker "${HF_HOME}" \
  -n vllm_ray_worker \
  -e "VLLM_HOST_IP=${VLLM_HOST_IP}" \
  -e "UCX_NET_DEVICES=${MN_IF_NAME}" \
  -e "NCCL_SOCKET_IFNAME=${MN_IF_NAME}" \
  -e "GLOO_SOCKET_IFNAME=${MN_IF_NAME}" \
  -e "TP_SOCKET_IFNAME=${MN_IF_NAME}" \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
  -e NCCL_DEBUG=INFO
