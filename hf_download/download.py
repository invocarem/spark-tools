#!/usr/bin/env python3
"""Download a HF repo with huggingface_hub.snapshot_download only (no transformers)."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading

from huggingface_hub import snapshot_download

_DISK_HEARTBEAT_SEC = 30.0


def _disk_heartbeat(stop: threading.Event, watch_dir: str) -> None:
    """Periodic free-space lines so UIs / logs show the job is still alive (like watch df -h)."""
    while not stop.wait(_DISK_HEARTBEAT_SEC):
        try:
            usage = shutil.disk_usage(watch_dir)
            free_gib = usage.free / (1024**3)
            total_gib = usage.total / (1024**3)
            print(
                f"[heartbeat] disk free {free_gib:.2f} GiB of {total_gib:.2f} GiB ({watch_dir})",
                flush=True,
            )
        except OSError as e:
            print(f"[heartbeat] disk_usage failed: {e}", flush=True, file=sys.stderr)


def download_hf_snapshot(model_id: str, save_dir: str = "./models") -> str | None:
    model_id = model_id.strip()
    if not model_id:
        print("Error: model_id is empty", file=sys.stderr)
        return None

    print(f"Downloading {model_id}...", flush=True)

    try:
        model_path = os.path.join(save_dir, model_id.replace("/", "_"))
        os.makedirs(model_path, exist_ok=True)

        stop_hb = threading.Event()
        hb_thread = threading.Thread(
            target=_disk_heartbeat,
            args=(stop_hb, model_path),
            name="disk-heartbeat",
            daemon=True,
        )
        hb_thread.start()

        # Use snapshot_download to get ALL files (including processor configs).
        # Progress bars (tqdm) go to stderr — line-buffered when stderr is a pipe.
        print("Downloading all model files using snapshot_download...", flush=True)
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=model_path,
                ignore_patterns=["*.h5", "*.ot", "*.msgpack"],  # Ignore unnecessary files
            )
        finally:
            stop_hb.set()
            hb_thread.join(timeout=5.0)

        print(f"Successfully saved to: {model_path}", flush=True)
        
        # Verify downloaded files
        files = os.listdir(model_path)
        print(f"Downloaded {len(files)} files:", flush=True)
        for f in sorted(files):
            print(f"  - {f}", flush=True)
            
        return os.path.abspath(model_path)

    except Exception as e:
        print(f"Error downloading {model_id}: {e}", file=sys.stderr)
        return None

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face repo via huggingface_hub (snapshot_download).",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help='Model id (e.g. "org/name" or "gpt2")',
    )
    parser.add_argument(
        "--save-dir",
        default="/data/hf",
        help="Parent directory inside the container (default: /data/hf)",
    )
    args = parser.parse_args()
    path = download_hf_snapshot(args.model_id.strip(), save_dir=args.save_dir.strip())
    return 0 if path else 1

if __name__ == "__main__":
    raise SystemExit(main())