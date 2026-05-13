# spark-tools

Small utilities for working with Hugging Face models and moving weights between machines.

| Tool | Description |
|------|-------------|
| [sglang_runtime](sglang_runtime/README.md) | Launch/stop/monitor SGLang stack via SSH (Python venv). |
| [sglang_docker](sglang_docker/) | Same operations via Docker containers. |
| [stack_ui](stack_ui/README.md) | Web console supporting both runtimes (venv & docker). |
| [hf_download](hf_download/README.md) | Download a full HF repo with `huggingface_hub.snapshot_download`. |
| [hf_transfer](hf_transfer/README.md) | Two-rank NCCL transfer of a model directory between hosts. |
