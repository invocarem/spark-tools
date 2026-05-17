#!/usr/bin/env python3
"""
vLLM Server for Local Qwen Model
This script serves a local Qwen model without downloading from Hugging Face Hub
"""

import os
import shlex
import sys
from pathlib import Path

# Force offline mode - prevent any downloads
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["VLLM_USE_MODEL_SCOPE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# vllm_launch.sh mounts the host HF tree at /root/.cache/huggingface (see HF_CACHE_DIR).
DEFAULT_HF_ROOT = "/root/.cache/huggingface"
DEFAULT_MODEL_NAME = "Qwen_Qwen3.5-2B"


def resolve_model_path() -> Path:
    """Resolve model directory inside the container."""
    explicit = os.environ.get("VLLM_MODEL_PATH", "").strip()
    if explicit:
        return Path(explicit)

    model_name = os.environ.get("VLLM_MODEL_NAME", DEFAULT_MODEL_NAME).strip()
    hf_root = os.environ.get("HF_HOME", DEFAULT_HF_ROOT).strip() or DEFAULT_HF_ROOT

    candidates = [
        Path(hf_root) / model_name,
        Path(DEFAULT_HF_ROOT) / model_name,
        Path("/workspace/huggingface") / model_name,
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_dir() and (path / "config.json").exists():
            return path
    return candidates[0]


def check_model_path(model_path):
    """Verify model files exist before starting"""
    model_dir = Path(model_path)
    
    if not model_dir.exists():
        print(f"Error: Model path {model_path} does not exist!")
        return False
    
    # Check for required files
    required_files = ["config.json"]
    missing_files = []
    
    for file in required_files:
        if not (model_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: Missing required files: {missing_files}")
        print(f"Model directory contains: {list(model_dir.iterdir())[:10]}")
    
    # Check for model weights
    safetensors = list(model_dir.glob("*.safetensors"))
    bin_files = list(model_dir.glob("*.bin"))
    
    if not safetensors and not bin_files:
        print("Warning: No model weight files (.safetensors or .bin) found!")
    else:
        print(f"Found {len(safetensors)} .safetensors files and {len(bin_files)} .bin files")
    
    return True


def build_vllm_argv(model_path: str) -> list[str]:
    """Build ``vllm serve`` argv (matches the container CLI; avoids internal API drift)."""
    host = os.environ.get("VLLM_HOST", "0.0.0.0")
    port = os.environ.get("VLLM_PORT", "8000")
    argv = [
        "vllm",
        "serve",
        model_path,
        "--host",
        host,
        "--port",
        port,
        "--trust-remote-code",
        "--max-model-len",
        os.environ.get("VLLM_MAX_MODEL_LEN", "4096"),
        "--gpu-memory-utilization",
        os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"),
    ]
    backend = os.environ.get("VLLM_DISTRIBUTED_BACKEND", "").strip()
    if backend:
        argv.extend(["--distributed-executor-backend", backend])
    tp = os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "").strip()
    if tp:
        argv.extend(["--tensor-parallel-size", tp])
    extra = os.environ.get("VLLM_EXTRA_ARGS", "").strip()
    if extra:
        argv.extend(shlex.split(extra))
    return argv


def serve_model() -> None:
    """Start the vLLM OpenAI-compatible API server via ``vllm serve``."""

    model_path = resolve_model_path()
    os.environ.setdefault("HF_HOME", str(model_path.parent))
    os.environ.setdefault("HF_HUB_CACHE", str(model_path.parent / "hub"))

    if not check_model_path(str(model_path)):
        print(f"Error: Cannot find model at {model_path}")
        print("vllm_launch.sh mounts the host HF tree at /root/.cache/huggingface.")
        print("Export HF_HOME on the host before launch (see run-local-vllm.sh), or set VLLM_MODEL_PATH.")
        sys.exit(1)

    model_path_str = str(model_path)
    argv = build_vllm_argv(model_path_str)
    host = os.environ.get("VLLM_HOST", "0.0.0.0")
    port = os.environ.get("VLLM_PORT", "8000")

    print(f"Starting vLLM server with local model: {model_path_str}")
    print("Forcing offline mode - no downloads will occur")
    print("=" * 60)
    print(f"  - Command: {' '.join(shlex.quote(a) for a in argv)}")
    print("=" * 60)
    print(f"Starting server on http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    os.execvp("vllm", argv)


def main():
    """Main entry point"""
    try:
        serve_model()
    except FileNotFoundError:
        print("Error: `vllm` not found on PATH. Run inside the vllm-node container.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
