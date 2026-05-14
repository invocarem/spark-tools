#!/usr/bin/env python3
"""SGLang bench_serving wrapper: optional presets + delegation to ``benchmark_sglang.py``.

Defaults are tuned for Qwen3.5-397B GPTQ in this repo, but you can point at any server/model
via ``BENCHMARK_*`` env vars or by passing through ``benchmark_sglang.py`` CLI flags (unknown args are forwarded).

Presets (``--preset``) only fill **unset** environment variables so explicit env always wins.

* ``qwen3_397b_gptq`` (default): 10 prompts, concurrency 1, random 512→256, served
  ``qwen3.5-397b``, HF ``Qwen/Qwen3.5-397B-A17B-GPTQ-Int4``.
* ``none``: do not apply preset env; same as running ``benchmark_sglang.py`` directly
  (still applies legacy ``QWEN397_*`` → ``BENCHMARK_*`` mapping below).

Environment (see ``tools/benchmark_sglang.py`` for full list): ``BENCHMARK_BASE_URL``,
``SGLANG_BASE_URL``, ``BENCHMARK_SERVED_MODEL``, ``BENCHMARK_HF_MODEL``, ``BENCHMARK_TOKENIZER``,
``BENCHMARK_NUM_PROMPTS``, ``BENCHMARK_MAX_CONCURRENCY``, ``BENCHMARK_RANDOM_*``, etc.

Legacy names (used only when the corresponding ``BENCHMARK_*`` / ``SGLANG_BASE_URL`` is unset):
``QWEN397_BENCH_BASE_URL``, ``QWEN397_BENCH_SERVED_MODEL``, ``QWEN397_BENCH_HF_MODEL``.

Examples::

  python3 tools/sglang/benchmark_qwen3_397b.py
  python3 tools/sglang/benchmark_qwen3_397b.py --preset none --num-prompts 5
  BENCHMARK_SERVED_MODEL=my-model python3 tools/sglang/benchmark_qwen3_397b.py --preset none
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parent.parent
_BENCHMARK_SGLANG = _TOOLS_ROOT / "benchmark_sglang.py"

PRESET_ENV = {
    "qwen3_397b_gptq": {
        "BENCHMARK_NUM_PROMPTS": "10",
        "BENCHMARK_MAX_CONCURRENCY": "1",
        "BENCHMARK_RANDOM_INPUT_LEN": "512",
        "BENCHMARK_RANDOM_OUTPUT_LEN": "256",
        "BENCHMARK_SERVED_MODEL": "qwen3.5-397b",
        "BENCHMARK_TOKENIZER": "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
        "BENCHMARK_HF_MODEL": "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
    },
}


def _map_legacy_env() -> None:
    """Map QWEN397_BENCH_* to BENCHMARK_* when targets are unset."""
    if not (os.environ.get("BENCHMARK_BASE_URL") or "").strip() and not (
        os.environ.get("SGLANG_BASE_URL") or ""
    ).strip():
        legacy = (os.environ.get("QWEN397_BENCH_BASE_URL") or "").strip()
        if legacy:
            os.environ["BENCHMARK_BASE_URL"] = legacy

    if not (os.environ.get("BENCHMARK_SERVED_MODEL") or "").strip() and not (
        os.environ.get("BENCHMARK_MODEL") or ""
    ).strip():
        legacy = (os.environ.get("QWEN397_BENCH_SERVED_MODEL") or "").strip()
        if legacy:
            os.environ["BENCHMARK_SERVED_MODEL"] = legacy

    legacy_hf = (os.environ.get("QWEN397_BENCH_HF_MODEL") or "").strip()
    if legacy_hf:
        if not (os.environ.get("BENCHMARK_HF_MODEL") or "").strip():
            os.environ["BENCHMARK_HF_MODEL"] = legacy_hf
        if not (os.environ.get("BENCHMARK_TOKENIZER") or "").strip():
            os.environ["BENCHMARK_TOKENIZER"] = legacy_hf


def _apply_preset_env(preset: str) -> None:
    if preset == "none":
        return
    keys = PRESET_ENV.get(preset)
    if not keys:
        print(f"{Path(__file__).name}: unknown preset {preset!r}", file=sys.stderr)
        raise SystemExit(2)
    for key, value in keys.items():
        if not (os.environ.get(key) or "").strip():
            os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=("qwen3_397b_gptq", "none"),
        default="qwen3_397b_gptq",
        help="Fill unset BENCHMARK_* defaults (default: qwen3_397b_gptq).",
    )
    args, rest = parser.parse_known_args()

    if not _BENCHMARK_SGLANG.is_file():
        print(
            f"{Path(__file__).name}: missing {_BENCHMARK_SGLANG} (expected tools layout).",
            file=sys.stderr,
        )
        return 2

    _map_legacy_env()
    _apply_preset_env(args.preset)

    cmd = [sys.executable, str(_BENCHMARK_SGLANG), *rest]
    print("+ " + " ".join(cmd), file=sys.stderr)
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
