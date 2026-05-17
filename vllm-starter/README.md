# vLLM starter (Ray in Docker)

Small scripts to run a **Ray cluster inside Docker** with host networking, then drive **vLLM** with `--distributed-executor-backend ray`.

## Files

| File | Purpose |
|------|---------|
| `run_cluster.sh` | Core launcher: starts Ray **head** or **worker** in a container (`ray start --block`). |
| `head.sh` | Example **head** host wrapper: sets paths, NIC, IPs, calls `run_cluster.sh --head`. |
| `worker.sh` | Example **worker** host wrapper: sets this node’s IP, head IP, calls `run_cluster.sh --worker`. |
| `launch-qwen3.5-2b.sh` | Example `docker exec` into the head container to run `vllm serve` with Ray. |

`head.sh` and `worker.sh` are **templates**: copy or edit `MONITOR_REPO_ROOT`, `VLLM_IMAGE`, `MN_IF_NAME`, `VLLM_HOST_IP`, `RAY_HEAD_IP` (workers only), `HF_HOME`, and container names (`-n`) for each machine.

## Prerequisites

- Docker with GPU support (`--gpus all`).
- An image that includes Ray and vLLM (e.g. your `vllm-node:latest`).
- Every node reachable at the IP you pass for Ray / `VLLM_HOST_IP`.
- Same `HF_HOME` mount path semantics: cache is mounted at `/root/.cache/huggingface` in the container.

## Order of operations

1. On the **head** machine, run `./head.sh` (or `bash head.sh`). Leave the session open; exiting stops the container and the head node.
2. On each **worker** machine, run `./worker.sh` with `RAY_HEAD_IP` set to the head’s address and `VLLM_HOST_IP` set to **that worker’s** address. Each worker needs a **unique** `VLLM_HOST_IP`.
3. From another terminal on the head (or any host with Docker access to the head container), run something like `launch-qwen3.5-2b.sh`: `docker exec` into the **head** container named there (`vllm_node` by default) and start `vllm serve ... --distributed-executor-backend ray`.

Workers use container name `vllm_ray_worker` in the example; change `-n` in `worker.sh` if you run multiple workers on one host or need different names.

## `run_cluster.sh` summary

Positional arguments:

```text
run_cluster.sh <docker_image> <head_node_ip> --head|--worker <path_to_hf_home> [-n NAME] [extra docker run args...]
```

- **Head:** `head_node_ip` is the head’s listen address; pass matching `-e VLLM_HOST_IP=...` for consistency (see script warning if they differ).
- **Worker:** `head_node_ip` is the Ray head address (`address=host:6379`); pass `-e VLLM_HOST_IP=<this_worker_ip>` so Ray uses the correct node IP.

Optional `MONITOR_REPO_ROOT`: if set in the environment, the repo is mounted at `/workspace` inside the container.

## Related

For a higher-level CLI (pull, launch, presets), see `vllm_docker/` and `CLAUDE.md` at the repo root.
