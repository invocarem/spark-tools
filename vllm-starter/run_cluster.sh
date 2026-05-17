#!/bin/bash
#
# Launch a Ray cluster inside Docker for vLLM inference.
#
# This script can start either a head node or a worker node, depending on the
# --head or --worker flag provided as the third positional argument.
#
# Usage:
# 1. Designate one machine as the head node and execute:
#    bash run_cluster.sh \
#         vllm/vllm-openai \
#         <head_node_ip> \
#         --head \
#         /abs/path/to/huggingface/cache \
#         -e VLLM_HOST_IP=<head_node_ip>
#
# Optional: set a fixed Docker container name (default is node-<random>):
#         --container-name vllm_node
#    or: -n vllm_node
#
# 2. On every worker machine, execute:
#    bash run_cluster.sh \
#         vllm/vllm-openai \
#         <head_node_ip> \
#         --worker \
#         /abs/path/to/huggingface/cache \
#         -e VLLM_HOST_IP=<worker_node_ip>
#
# Each worker requires a unique VLLM_HOST_IP value.
# Keep each terminal session open. Closing a session stops the associated Ray
# node and thereby shuts down the entire cluster.
# Every machine must be reachable at the supplied IP address.
#
# The container is named "node-<random_suffix>" unless you pass
# --container-name or -n. To open a shell inside a running container:
#       docker exec -it <container_name> /bin/bash
#
# Then, you can execute vLLM commands on the Ray cluster as if it were a
# single machine, e.g. vllm serve ...
#
# Containers are run with -it --rm (interactive TTY, removed on exit).
# To stop from another terminal: docker stop <container_name>

# Check for minimum number of required arguments.
if [ $# -lt 4 ]; then
    echo "Usage: $0 docker_image head_node_ip --head|--worker path_to_hf_home [--container-name|-n NAME] [additional_args...]"
    exit 1
fi

# Extract the mandatory positional arguments and remove them from $@.
DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"  # Should be --head or --worker.
PATH_TO_HF_HOME="$4"
shift 4

# Preserve any extra arguments so they can be forwarded to Docker.
ADDITIONAL_ARGS=("$@")

# Optional: --container-name NAME or -n NAME (consumed here, not passed to docker run).
CONTAINER_NAME=""
NEW_ARGS=()
i=0
while [ "${i}" -lt "${#ADDITIONAL_ARGS[@]}" ]; do
    arg="${ADDITIONAL_ARGS[$i]}"
    case "${arg}" in
        --container-name)
            i=$((i + 1))
            if [ "${i}" -ge "${#ADDITIONAL_ARGS[@]}" ]; then
                echo "Error: --container-name requires a value"
                exit 1
            fi
            CONTAINER_NAME="${ADDITIONAL_ARGS[$i]}"
            i=$((i + 1))
            ;;
        -n)
            i=$((i + 1))
            if [ "${i}" -ge "${#ADDITIONAL_ARGS[@]}" ]; then
                echo "Error: -n requires a container name"
                exit 1
            fi
            CONTAINER_NAME="${ADDITIONAL_ARGS[$i]}"
            i=$((i + 1))
            ;;
        *)
            NEW_ARGS+=("${arg}")
            i=$((i + 1))
            ;;
    esac
done
ADDITIONAL_ARGS=("${NEW_ARGS[@]}")

# Validate the NODE_TYPE argument.
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# Extract VLLM_HOST_IP from ADDITIONAL_ARGS (e.g. "-e VLLM_HOST_IP=...").
VLLM_HOST_IP=""
for ((i = 0; i < ${#ADDITIONAL_ARGS[@]}; i++)); do
    arg="${ADDITIONAL_ARGS[$i]}"
    case "${arg}" in
        -e)
            next="${ADDITIONAL_ARGS[$((i + 1))]:-}"
            if [[ "${next}" == VLLM_HOST_IP=* ]]; then
                VLLM_HOST_IP="${next#VLLM_HOST_IP=}"
                break
            fi
            ;;
        -eVLLM_HOST_IP=* | VLLM_HOST_IP=*)
            VLLM_HOST_IP="${arg#*=}"
            break
            ;;
    esac
done

# For the head node, HEAD_NODE_ADDRESS and VLLM_HOST_IP should be consistent.
if [[ "${NODE_TYPE}" == "--head" && -n "${VLLM_HOST_IP}" ]]; then
    if [[ "${VLLM_HOST_IP}" != "${HEAD_NODE_ADDRESS}" ]]; then
        echo "Warning: VLLM_HOST_IP (${VLLM_HOST_IP}) differs from head_node_ip (${HEAD_NODE_ADDRESS})."
        echo "Using VLLM_HOST_IP as the head node address."
        HEAD_NODE_ADDRESS="${VLLM_HOST_IP}"
    fi
fi

# Docker container names must be unique on each host. Default: random suffix so
# multiple Ray containers can run on one machine; use --container-name/-n for a fixed name.
if [ -z "${CONTAINER_NAME}" ]; then
    CONTAINER_NAME="node-${RANDOM}"
fi

# Stop the container on script exit (e.g. SIGINT). --rm removes it after stop.
cleanup() {
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
}
trap cleanup EXIT

# Build the Ray start command based on the node role.
# The head node manages the cluster and accepts connections on port 6379,
# while workers connect to the head's address.
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD+=" --head --node-ip-address=${HEAD_NODE_ADDRESS} --port=6379"
else

    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
    if [ -n "${VLLM_HOST_IP}" ]; then
        RAY_START_CMD+=" --node-ip-address=${VLLM_HOST_IP}"
    fi
fi

# Launch the container with the assembled parameters.
# --network host: Allows Ray nodes to communicate directly via host networking
# --shm-size 10.24g: Increases shared memory
# --gpus all: Gives container access to all GPUs on the host
# -v HF_HOME: Mounts HuggingFace cache to avoid re-downloading models
# -it only when stdout is a TTY; plain -i when run from scripts/API (docker rejects -t without a TTY).
if [ -t 1 ]; then
  DOCKER_TTY_FLAGS=(-it)
else
  DOCKER_TTY_FLAGS=(-i)
fi
docker run "${DOCKER_TTY_FLAGS[@]}" --rm \
    --entrypoint /bin/bash \
    --network host \
    --name "${CONTAINER_NAME}" \
    --shm-size 10.24g \
    --gpus all \
    -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
    -v "${MONITOR_REPO_ROOT}:/workspace" \
    "${ADDITIONAL_ARGS[@]}" \
    "${DOCKER_IMAGE}" -c "${RAY_START_CMD}"
