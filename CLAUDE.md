# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spark-tools is a collection of small, standalone CLI utilities for working with Hugging Face models on DGX Spark-style nodes. It has no build step, no tests, and no package manager — just Python scripts and shell wrappers.

## Tools

### `sglang_runtime/` — SGLang runtime operations

Single-file CLI (`sglang_runtime.py`) for deploying, launching, stopping, and monitoring a custom SGLang stack across nodes via SSH/rsync.

- **Run**: `python3 sglang_runtime/sglang_runtime.py [--verbose] <subcommand>`
- **Subcommands**: `deploy`, `launch`, `stop`, `logs`, `scan` (alias: `refresh`), `benchmark`, `measure`
- **Configuration**: `.env` files (auto-loaded from cwd) or `--env-file`. CLI flags > preset fields > env vars. Model presets are defined in `sglang_runtime/model_presets.json` (git-ignored; contains local paths).
- **Key functions** (exported for programmatic use): `merge_preset_launch_fields`, `build_dashboard_source_launch_command`, `run_deploy_command`
- **Value resolution**: `resolve_value(cli, env, preset, default)` — CLI flags take precedence, then env vars, then preset JSON fields, then hardcoded defaults
- **NCCL env**: Any key in `_NCCL_ENV_KEYS` list is exported before launch when present in the loaded env
- **Deploy sets**: Named collections of `remote_dir`, `sources`, `exclude` stored in `deploy_sets.json` (optional, at repo root)

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

- All scripts are self-contained (no `import` between modules in this repo)
- No package manager, no `requirements.txt`, no virtual env setup script — dependencies are installed per-environment
- Shell scripts use `set -euo pipefail`
- Python scripts use `raise SystemExit(main())` pattern with `main() -> int`
- Environment variables use `MODEL_TRANSFER_` prefix in the transfer module, raw names (e.g., `MASTER_NODE`, `NCCL_DEBUG`) in sglang_runtime
- `.env` and `.env.stack` files are git-ignored and contain node-specific configuration
