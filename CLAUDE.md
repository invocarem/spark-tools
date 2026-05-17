# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spark-tools is a collection of small, standalone CLI utilities for working with Hugging Face models on DGX Spark-style nodes. It has no build step, no tests, and no package manager — just Python scripts and shell wrappers.

## Tools

### `sglang_runtime/` — SGLang runtime operations (venv)

Single-file CLI (`sglang_runtime.py`) for deploying, launching, stopping, and monitoring a custom SGLang stack across nodes via SSH/rsync. Runs inside a local Python venv.

- **Run**: `python3 sglang_runtime/sglang_runtime.py [--verbose] <subcommand>`
- **Subcommands**: `deploy`, `launch`, `stop`, `logs`, `scan` (alias: `refresh`), `benchmark`, `measure`
- **Configuration**: `.env` files (auto-loaded from cwd) or `--env-file`. CLI flags > preset fields > env vars. Model presets are defined in `sglang_runtime/model_presets.json` (git-ignored; contains local paths).
- **Key functions** (exported for programmatic use): `merge_preset_launch_fields`, `build_dashboard_source_launch_command`, `run_deploy_command`
- **Value resolution**: `resolve_value(cli, env, preset, default)` — CLI flags take precedence, then env vars, then preset JSON fields, then hardcoded defaults
- **NCCL env**: Any key in `_NCCL_ENV_KEYS` list is exported before launch when present in the loaded env
- **Deploy sets**: Named collections of `remote_dir`, `sources`, `exclude` stored in `deploy_sets.json` (optional, at repo root)

### `sglang_docker/` — SGLang runtime via Docker

Parallel CLI (`sglang_docker.py`) that mirrors `sglang_runtime` but runs sglang inside Docker containers instead of a local Python venv. Shares all utility code via `sglang_common/`.

- **Run**: `python3 sglang_docker/sglang_docker.py [--verbose] <subcommand>`
- **Subcommands**: `pull`, `launch`, `stop`, `logs`, `scan` (alias: `refresh`), `benchmark`, `measure`
- **Configuration**: same preset resolution pattern; presets file defaults to `sglang_docker/model_presets.json`. Presets use `image` field instead of `venv_path`.
- **Container naming**: `sglang-{preset_name}` for solo, `sglang-{preset_name}-node{idx}` for cluster
- **Default log dir**: `~/sglang-docker-logs`

### `vllm_docker/` — vLLM runtime via Docker

Parallel CLI (`vllm_docker.py`) modeled on `sglang_docker` but launches **`vllm serve`** inside the container. Shares helpers via `sglang_common/` (presets, env, scan, benchmark).

- **Run**: `python3 vllm_docker/vllm_docker.py [--verbose] <subcommand>`
- **Subcommands**: `pull`, `launch`, `stop`, `logs`, `scan` (alias: `refresh`), `benchmark`, `measure`
- **Configuration**: presets default to `vllm_docker/model_presets.json` (same JSON keys as sglang docker: `image`, `model_path`, `tp`, `port`, `sglang_args` for extra `vllm serve` flags). Env `VLLM_EXTRA_ARGS` and CLI `--vllm-args` append after preset rows.
- **Container naming**: `vllm-{preset_name}` for solo, `vllm-{preset_name}-node{idx}` for cluster (with preset); `vllm-node{idx}` when no preset
- **Default log dir**: `~/vllm-docker-logs`

### `sglang_common/` — Shared utilities

Package re-exported by `sglang_runtime`, `sglang_docker`, and `vllm_docker`. Exports CLI helpers, env loading, preset resolution, launch prefix builders, HTTP scan probes, and benchmark logic. Runtime modules import from here instead of from each other.

### `stack_ui/` — Stack web console

FastAPI backend plus Vite + React + TypeScript frontend supporting **venv**, **docker**, and **vllm_docker**. The UI lets the user switch runtime via radio buttons; each API call body includes a `runtime` field (`"venv"`, `"docker"`, or `"vllm_docker"`, defaults to `"venv"`). See `stack_ui/README.md`.

- **API**: `cd stack_ui/backend && uvicorn stack_ui_server:app --host 127.0.0.1 --port 8765`
- **Dev UI**: `cd stack_ui/frontend && npm install && npm run dev` (proxies `/api` to port 8765)
- **Runtime dispatch**: `_RUNTIME_MAP` maps `"venv"` → `sglang_runtime/`, `"docker"` → `sglang_docker/`, `"vllm_docker"` → `vllm_docker/`. The `_run_cli()` helper resolves script path and PYTHONPATH per runtime.
- **Docker-style runtimes**: `/api/preview-launch` builds a `docker run` command for `docker` and `vllm_docker`; `/api/exec` allows `"pull"` for those runtimes, `"deploy"` only for venv.
- **Legacy**: `sglang_runtime/web_ui/server.py` re-exports the same `app` for `uvicorn web_ui.server:app` from `sglang_runtime/`

### `hf_download/` — Hugging Face model download

Downloads a full HF repo using `huggingface_hub.snapshot_download`. No `transformers` dependency.

- **Run**: `python3 hf_download/download.py --model-id <org/repo> [--save-dir /data/hf]`
- **Depends on**: `pip install huggingface_hub`
- Skips `*.h5`, `*.ot`, `*.msgpack`. Prints a disk-free-space heartbeat every 30s.

### `hf_transfer/` — Two-rank NCCL model transfer between hosts

Transfers a model directory from rank 0 (sender) to rank 1 (receiver) using PyTorch distributed (NCCL or Gloo backend).

- **Run**: `./hf_transfer/hf_transfer.sh <rank> <src_dir> <dest_dir> <master_addr>` (starts rank 0 first, then rank 1)
- **Env**: `MODEL_TRANSFER_TORCH_BACKEND=nccl|gloo` (default: gloo, auto-falls-back if CUDA unavailable)
- **Flag**: `--all-files` forces rank 0 to send everything even when `--src` == `--dest`
- `check_nccl_version.py`: utility that prints NCCL version from both PyTorch APIs and the shared library directly

## Conventions

- `sglang_runtime` and `sglang_docker` share code via `sglang_common/` but do not import each other
- No package manager, no `requirements.txt`, no virtual env setup script — dependencies are installed per-environment
- Shell scripts use `set -euo pipefail`
- Python scripts use `raise SystemExit(main())` pattern with `main() -> int`
- Environment variables use `MODEL_TRANSFER_` prefix in the transfer module, raw names (e.g., `MASTER_NODE`, `NCCL_DEBUG`) in sglang_runtime, `DOCKER_IMAGE` in sglang_docker
- `.env` and `.env.stack` files are git-ignored and contain node-specific configuration
- Stack UI backend request bodies use `runtime: "venv" | "docker"` to dispatch between the two CLI modules
