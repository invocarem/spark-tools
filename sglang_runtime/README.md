# sglang runtime helper

`sglang_runtime.py` is a small CLI for operating a custom [SGLang](https://github.com/sgl-project/sglang) stack on DGX Spark–style nodes: sync sources over SSH/rsync, start and stop `python -m sglang.launch_server` locally or across nodes, run a quick OpenAI-compatible API benchmark, and snapshot GPU/system load.

## Requirements

- Python 3.10+ (uses `list[str]` typing and `dataclass`)
- Local: bash, optional `tee` for solo logs
- **deploy**: `ssh`, `rsync`, remote shell access
- **stop**: `lsof` on the target host(s) to resolve listeners on the server port
- **measure**: `nvidia-smi` where you collect metrics

## Usage

```bash
python3 sglang_runtime.py [--verbose] <subcommand> [options]
```

Global `--verbose` prints commands and captured stdout/stderr for debugging.

## Configuration

If `--env-file` is set on a subcommand, variables are loaded from that file. Otherwise, when a `.env` exists in the current working directory, it is loaded automatically. CLI flags override preset fields; preset fields override environment values where the code uses `resolve_value` (see the script for each subcommand).

Common environment variables:

- **Nodes**: `MASTER_NODE`, `WORKER_NODE` — default targets for deploy, cluster launch/stop, and measure when `--hosts` is omitted.
- **Deploy**: `DEPLOY_SET` (name), `DEPLOY_SETS_FILE` (path, falls back to `--deploy-sets-file`), `REMOTE_DIR`, `DEPLOY_SOURCES` (comma-separated paths), `DEPLOY_EXCLUDE` (comma-separated rsync excludes).
- **Launch / stop**: `MODEL_PRESETS_FILE`, `MODEL_PRESET`, `MODEL_PATH`, `VENV_PATH`, `SERVER_PORT`, `TP_SIZE`, `SGLANG_EXTRA_ARGS` (extra flags, shell-split), `DIST_ADDR`, `MASTER_PORT`.
- **NCCL / runtime**: any key listed in `_NCCL_ENV_KEYS` inside the script (for example `NCCL_DEBUG`, `NCCL_SOCKET_IFNAME`) is exported before launch when present in the loaded env.

## Subcommands

### `deploy`

Rsync local paths to `~/runtime-sglang` (or `--remote-dir` / preset / `REMOTE_DIR`) on each host. Hosts come from `--hosts` or `MASTER_NODE` / `WORKER_NODE`.

```bash
python3 sglang_runtime.py deploy --hosts spark-01 spark-02 --sources run.sh,sglang
python3 sglang_runtime.py deploy --list-sets --deploy-sets-file deploy_sets.json
python3 sglang_runtime.py deploy --set my-stack --hosts spark-01
```

Optional: `--ssh-key`, `--ssh-port`, `--exclude`.

### `launch`

Runs `source <venv>/bin/activate && python -m sglang.launch_server ...` after optional NCCL exports.

- **solo** (default): one process; `--tp` defaults to `1`. Use `--host` to run the same command over SSH on one node. `--log-file` tees stdout/stderr locally through `tee`.
- **cluster**: one rank per host via `--dist-init-addr`, `--nnodes`, `--node-rank`; uses `nohup` and `--log-dir` on remotes.

```bash

python sglang_runtime.py --verbose launch --mode cluster --hosts spark1 spark2 --preset qwen3.6-27b --presets-file ./model_presets.json --env-file .env.stack  --log-dir /home/[user]/code/spark-stack/logs


python3 sglang_runtime.py launch --mode solo
python3 sglang_runtime.py launch --mode solo --host spark-01 --preset qwen-2b
python3 sglang_runtime.py launch --mode cluster --hosts spark-01 spark-02
python3 sglang_runtime.py launch --list-presets --presets-file model_presets.json
```

`--command` replaces the generated launch line entirely if you need full control.

### `stop`

Finds PIDs listening on the HTTP port (default `30000` or from preset/env), sends `SIGTERM`, waits `--grace-sec`, then `SIGKILL` if needed, and runs `pkill -f 'python -m sglang.launch_server'` as a backstop.

```bash
python3 sglang_runtime.py stop --mode solo
python3 sglang_runtime.py stop --mode cluster --hosts spark-01 spark-02 --port 30000
```

### `benchmark`

POSTs to `{base_url}/v1/chat/completions` and prints JSON with latency percentiles and throughput.

```bash
python3 sglang_runtime.py benchmark --base-url http://127.0.0.1:30000 --model my-model --requests 50
```

### `measure`

Runs `nvidia-smi` plus a short Python snippet for load average; output is JSON keyed by host. With no hosts and no `MASTER_NODE`/`WORKER_NODE`, measures **local** only. Use the hostname `local` in `--hosts` to force local execution alongside remotes.

```bash
python3 sglang_runtime.py measure
python3 sglang_runtime.py measure --hosts spark-01 spark-02
```

## JSON presets

### Model presets (`model_presets.json`)

Used with `--preset` / `MODEL_PRESET`. Each top-level key is a preset name; values are objects with optional fields:

| Field | Type | Role |
|-------|------|------|
| `model_path` | string | `--model-path` |
| `venv_path` | string | venv containing `sglang` |
| `tp` | int | tensor parallel size |
| `port` | int | HTTP port |
| `sglang_args` | list | Extra server flags (strings; dict/list entries are passed as compact JSON for flags like `--model-loader-extra-config`) |

If `--served-model-name` is not present in merged args, the preset name is appended as `--served-model-name <preset>`.

### Deploy sets (`deploy_sets.json`)

Used with `deploy --set <name>`. Each entry can include `remote_dir`, `sources` (string or list of strings), and `exclude` (string or list), following the same merge rules as the deploy CLI flags and env vars.

## Python API

The module exposes helpers such as `merge_preset_launch_fields`, `build_dashboard_source_launch_command`, and `run_deploy_command` for programmatic use (for example building a launch string or capturing deploy output as structured data).
