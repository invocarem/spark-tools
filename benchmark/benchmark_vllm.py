#!/usr/bin/env python3
"""Run ``vllm bench serve`` with defaults for a vLLM OpenAI-compatible server (typically :8000).

vLLM's bench uses ``--model`` for the Hugging Face id (tokenizer / dataset prep) and
``--served-model-name`` for the OpenAI ``model`` field when it differs from the HF repo.
**Served names** like ``qwen3.5-35b`` are not HF repos — this wrapper passes:

* ``--model <HF id>`` — tokenizer / synthetic prompts (defaults to ``BENCHMARK_TOKENIZER`` or served id when it looks like a HF repo).
* ``--served-model-name <id>`` — value sent to the server (from ``--model`` here, env, or ``GET /v1/models``).

Requires a recent vLLM install with ``python -m vllm bench serve`` (v0.6+). Run from the repo with
``PYTHONPATH`` including the parent of ``benchmark/`` so ``benchmark_common`` imports work.

Env (optional): BENCHMARK_BASE_URL (else VLLM_BASE_URL, else http://127.0.0.1:8000), BENCHMARK_BACKEND,
BENCHMARK_DATASET, BENCHMARK_NUM_PROMPTS, BENCHMARK_RANDOM_INPUT_LEN, BENCHMARK_RANDOM_OUTPUT_LEN,
BENCHMARK_SERVED_MODEL (API id), BENCHMARK_HF_MODEL (HF repo for bench ``--model``),
BENCHMARK_TOKENIZER, BENCHMARK_MAX_CONCURRENCY, BENCHMARK_EXTRA_REQUEST_BODY (JSON merged into
``--extra-body``), BENCHMARK_PRESERVE_THINKING (if true: do not inject ``chat_template_kwargs.enable_thinking: false`` for Qwen3).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from benchmark_common import (
    env_int,
    env_optional_int,
    env_truthy,
    fetch_served_model_id,
    load_json_object,
    pop_json_flag_from_argv,
)

PROG = "benchmark_vllm.py"

DEFAULT_BASE = (
    os.environ.get("BENCHMARK_BASE_URL", "").strip()
    or os.environ.get("VLLM_BASE_URL", "").strip()
    or "http://127.0.0.1:8000"
)
DEFAULT_BACKEND = os.environ.get("BENCHMARK_BACKEND", "openai-chat")
DEFAULT_TOKENIZER = os.environ.get("BENCHMARK_TOKENIZER", "").strip()

_DATASETS_WITH_RANDOM_LEN = frozenset({"random", "random-mm"})


def _vllm_bench_prefix() -> list[str]:
    try:
        import vllm  # noqa: F401
    except ImportError:
        print(
            f"{PROG}: vllm is not installed for {sys.executable!r}. "
            "Install vLLM in this environment or run inside the vLLM container.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return [sys.executable, "-m", "vllm", "bench", "serve"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Thin wrapper around `vllm bench serve` for a vLLM OpenAI API server.",
        prog=PROG,
    )
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE,
        help=f"Server base URL (default {DEFAULT_BASE!r} or BENCHMARK_BASE_URL / VLLM_BASE_URL).",
    )
    p.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"bench serve backend (default {DEFAULT_BACKEND!r} or BENCHMARK_BACKEND).",
    )
    p.add_argument(
        "--dataset-name",
        default=os.environ.get("BENCHMARK_DATASET", "random"),
        help="Dataset name (default: random, or BENCHMARK_DATASET).",
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=env_int("BENCHMARK_NUM_PROMPTS", 3),
        help="Prompt count (default 3 or BENCHMARK_NUM_PROMPTS).",
    )
    p.add_argument(
        "--random-input-len",
        type=int,
        default=env_int("BENCHMARK_RANDOM_INPUT_LEN", 128),
        help="For random datasets: input tokens (default 128).",
    )
    p.add_argument(
        "--random-output-len",
        type=int,
        default=env_int("BENCHMARK_RANDOM_OUTPUT_LEN", 128),
        help="For random datasets: output tokens (default 128).",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=env_optional_int("BENCHMARK_MAX_CONCURRENCY"),
        help="Cap concurrent requests (optional; BENCHMARK_MAX_CONCURRENCY).",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("BENCHMARK_SERVED_MODEL", "") or os.environ.get("BENCHMARK_MODEL", "")
        or "",
        metavar="SERVED_ID",
        help="Served model id for the API (BENCHMARK_SERVED_MODEL / BENCHMARK_MODEL); optional if /v1/models works.",
    )
    p.add_argument(
        "--hf-model",
        default=os.environ.get("BENCHMARK_HF_MODEL", "") or "",
        metavar="HF_REPO",
        help="HF repo for bench --model; default BENCHMARK_HF_MODEL or same as --tokenizer.",
    )
    p.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="HF tokenizer for synthetic prompts (BENCHMARK_TOKENIZER).",
    )
    p.add_argument(
        "--extra-request-body",
        default=None,
        metavar="JSON",
        help="Merged into vllm --extra-body after defaults (same as BENCHMARK_EXTRA_REQUEST_BODY).",
    )
    return p


def main() -> None:
    parser = build_parser()
    args, rest = parser.parse_known_args()

    rest_extra, rest = pop_json_flag_from_argv(rest, "--extra-request-body", PROG)
    rest_extra_body, rest = pop_json_flag_from_argv(rest, "--extra-body", PROG)
    rest_extra.update(rest_extra_body)

    served = (args.model or "").strip() or os.environ.get("BENCHMARK_SERVED_MODEL", "").strip()
    if not served:
        served = fetch_served_model_id(args.base_url) or ""

    if not served:
        print(
            f"{PROG}: could not resolve served model id. Set --model, BENCHMARK_SERVED_MODEL, "
            "or ensure GET {}/v1/models returns a model.".format(args.base_url.rstrip("/")),
            file=sys.stderr,
        )
        raise SystemExit(2)

    tokenizer = args.tokenizer.strip() or DEFAULT_TOKENIZER
    hf_for_bench = (args.hf_model or "").strip()
    if not hf_for_bench and "/" in served:
        hf_for_bench = served
    if not tokenizer:
        tokenizer = hf_for_bench
    if not hf_for_bench:
        hf_for_bench = tokenizer

    if not hf_for_bench:
        print(
            f"{PROG}: could not resolve HF model/tokenizer for vllm bench serve. "
            "Set --hf-model or --tokenizer (or BENCHMARK_HF_MODEL / BENCHMARK_TOKENIZER).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    cmd: list[str] = [
        *_vllm_bench_prefix(),
        "--backend",
        args.backend,
        "--base-url",
        args.base_url,
        "--dataset-name",
        args.dataset_name,
        "--model",
        hf_for_bench,
        "--served-model-name",
        served,
        "--tokenizer",
        tokenizer,
        "--num-prompts",
        str(args.num_prompts),
    ]

    if args.dataset_name in _DATASETS_WITH_RANDOM_LEN:
        cmd.extend(
            [
                "--random-input-len",
                str(args.random_input_len),
                "--random-output-len",
                str(args.random_output_len),
            ]
        )

    if args.max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(args.max_concurrency)])

    extra_body: dict = {}
    env_extra = (os.environ.get("BENCHMARK_EXTRA_REQUEST_BODY") or "").strip()
    if env_extra:
        extra_body.update(load_json_object("BENCHMARK_EXTRA_REQUEST_BODY", env_extra, PROG))
    if args.extra_request_body:
        extra_body.update(load_json_object("--extra-request-body", args.extra_request_body, PROG))
    extra_body.update(rest_extra)
    if not env_truthy("BENCHMARK_PRESERVE_THINKING"):
        ctk = extra_body.get("chat_template_kwargs")
        if not isinstance(ctk, dict):
            ctk = {}
        ctk = {"enable_thinking": False, **ctk}
        extra_body["chat_template_kwargs"] = ctk
    if extra_body:
        cmd.extend(["--extra-body", json.dumps(extra_body)])

    cmd.extend(rest)

    print("+ " + " ".join(cmd), file=sys.stderr)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
