# vllm_launch

Docker-based launcher for multi-node (or solo) vLLM on DGX Spark-style clusters. Wraps `docker run` with NCCL/Ray wiring, optional mods, and launch scripts copied into the container.

## Prerequisites

- Docker image `vllm-node` (or override with `-t`)
- Passwordless SSH between cluster nodes (for multi-node)
- Models on the host under a single Hugging Face tree (see below)

## Hugging Face mount (`HF_HOME`)

`vllm_launch.sh` bind-mounts the host directory given by **`HF_HOME`** into the container at **`/root/.cache/huggingface`**:

| Host | Container |
|------|-----------|
| `$HF_HOME` (default: `~/.cache/huggingface`) | `/root/.cache/huggingface` |

**Export `HF_HOME` in your shell before calling `vllm_launch.sh`.** The script reads `HF_HOME` when it starts, *before* it loads `.env`. A `HF_HOME=...` line in `.env` does **not** affect the mount unless you export it yourself (or use a wrapper script that does).

```bash
export HF_HOME=/home/chenchen/huggingface

./vllm_launch.sh --solo --launch-script serve_local_qwen.py
```

Verify the mount path without starting a server:

```bash
export HF_HOME=/home/chenchen/huggingface
./vllm_launch.sh --solo --check-config
```

Look for `-v /home/chenchen/huggingface:/root/.cache/huggingface` in the printed Docker args.

### Layout on the host

Models are expected as sibling directories under `HF_HOME`, for example:

```
/home/chenchen/huggingface/
  Qwen_Qwen3.5-2B/
    config.json
    *.safetensors
  hub/          # optional HF hub cache
```

Inside the container that becomes `/root/.cache/huggingface/Qwen_Qwen3.5-2B`.

## Workspace (`/workspace`)

Launch scripts (`--launch-script`) are copied into the container as **`/workspace/exec-script.sh`** and executed from there. Mods land under `/workspace/mods/`. The launcher creates `/workspace` if the image does not already provide it.

| Path in container | Purpose |
|-------------------|---------|
| `/workspace/exec-script.sh` | Your `--launch-script` (e.g. `serve_local_qwen.py`) |
| `/workspace/mods/` | Applied `--apply-mod` patches |
| `/root/.cache/huggingface/` | Host `HF_HOME` bind mount (models live here) |

If a container named `vllm_node` is **already running**, `vllm_launch.sh` skips `docker run` but still **refreshes** `/workspace/exec-script.sh` before exec. To pick up a new `HF_HOME` mount or image, stop first:

```bash
./vllm_launch.sh stop
export HF_HOME=/home/chenchen/huggingface
./vllm_launch.sh --solo --launch-script serve_local_qwen.py
```

## Quick start: local Qwen 3.5 2B

`run-local-vllm.sh` exports `HF_HOME` from `.env` (if unset), stops any existing container, then launches `serve_local_qwen.py`:

```bash
cd vllm_launch
./run-local-vllm.sh solo      # single node (default)
./run-local-vllm.sh cluster   # all CLUSTER_NODES in .env, Ray + tensor parallel
VLLM_MODEL_NAME=Qwen_Qwen3.6-27B ./run-local-vllm.sh cluster
```

(`run-qwen3.5-2b.sh` is a thin alias for the same script; default model remains `Qwen_Qwen3.5-2B` when `VLLM_MODEL_NAME` is unset.)

Cluster mode uses `CLUSTER_NODES` from `.env`, starts containers and Ray on each node, and runs `vllm serve` on the head with `--distributed-executor-backend ray`. Tensor parallel size defaults to the node count; override with `VLLM_TENSOR_PARALLEL_SIZE`.

Equivalent manual steps (solo):

```bash
export HF_HOME=/home/chenchen/huggingface   # or: source from .env (see run-local-vllm.sh)

./vllm_launch.sh stop
./vllm_launch.sh --solo --no-cache-dirs --launch-script serve_local_qwen.py
```

Cluster manual equivalent:

```bash
export HF_HOME=/home/chenchen/huggingface
./vllm_launch.sh stop
./vllm_launch.sh --no-cache-dirs --launch-script serve_local_qwen.py \
  -e VLLM_DISTRIBUTED_BACKEND=ray -e VLLM_TENSOR_PARALLEL_SIZE=2
```

## `serve_local_qwen.py`

Offline vLLM server for a local Qwen checkout. It looks for the model under the container mount path above (not `/workspace/huggingface`).

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLLM_MODEL_NAME` | `Qwen_Qwen3.5-2B` | Subdirectory under the HF root |
| `VLLM_MODEL_PATH` | (unset) | Full path inside the container; overrides name lookup |
| `HF_HOME` | `/root/.cache/huggingface` | HF root *inside the container* (normally leave default) |
| `VLLM_HOST` / `VLLM_PORT` | `0.0.0.0` / `8000` | Bind address for `vllm serve` |
| `VLLM_MAX_MODEL_LEN` | `4096` | Passed to `--max-model-len` |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.9` | Passed to `--gpu-memory-utilization` |
| `VLLM_DISTRIBUTED_BACKEND` | (unset) | e.g. `ray` for multi-node cluster |
| `VLLM_TENSOR_PARALLEL_SIZE` | (unset) | e.g. `2` when using Ray across two GPUs/nodes |
| `VLLM_EXTRA_ARGS` | (unset) | Extra `vllm serve` flags (shell-split) |

Example with a different model directory name:

```bash
export HF_HOME=/home/chenchen/huggingface
./vllm_launch.sh --solo --launch-script serve_local_qwen.py \
  -e VLLM_MODEL_NAME=Qwen_Qwen3.6-27B
```

## Configuration (`.env`)

Copy or generate `.env` in this directory (git-ignored). Use `autodiscover.sh` or `./vllm_launch.sh --setup` to create it from the current host.

Common keys:

| Variable | Purpose |
|----------|---------|
| `CLUSTER_NODES` | Comma-separated node IPs |
| `LOCAL_IP` | This node's IP |
| `ETH_IF` / `IB_IF` | Network interfaces for NCCL |
| `HF_HOME` | Document your model path; **must be exported** before `vllm_launch.sh` (see above) |
| `CONTAINER_*` | Passed into the container as env vars (e.g. `CONTAINER_NCCL_DEBUG=INFO`) |

## `vllm_launch.sh` actions

```bash
export HF_HOME=/path/to/huggingface

./vllm_launch.sh start              # start idle containers (Ray cluster if multi-node)
./vllm_launch.sh stop               # stop containers on all nodes
./vllm_launch.sh status             # container / Ray status
./vllm_launch.sh exec <command>     # run command in head container

./vllm_launch.sh --solo start       # single node, no Ray
./vllm_launch.sh --check-config     # print resolved config and docker args
./vllm_launch.sh -h                 # full flag list
```

Useful flags: `--nodes`, `--launch-script`, `--no-cache-dirs`, `--no-ray`, `--apply-mod`, `-e VAR=val`.

## Files

| File | Role |
|------|------|
| `vllm_launch.sh` | Main launcher |
| `autodiscover.sh` | Probe interfaces and nodes; write `.env` |
| `serve_local_qwen.py` | Example launch script (local Qwen, offline) |
| `run-local-vllm.sh` | Wrapper: exports `HF_HOME`, optional `VLLM_MODEL_NAME`, solo or cluster |
| `run-qwen3.5-2b.sh` | Alias for `run-local-vllm.sh` (backward compatible) |
| `.env` | Local cluster and paths (not committed) |
