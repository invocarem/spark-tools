#!/usr/bin/env python3
"""Run a small **task** benchmark: send each problem to chat/completions and score with simple checkers.

This complements ``benchmark_sglang.py`` / ``benchmark_vllm.py`` (throughput). Here you measure **rough pass rate** on curated prompts.

Input: JSONL (one JSON object per line). Fields:

- ``id`` (str): stable id
- ``category`` (str): free label for reporting (e.g. ``leetcode``, ``bugfix``, ``tool_use``, ``latin``)
- ``prompt`` (str): user message
- ``system`` (optional str): system message
- ``checker`` (object): how to grade assistant text
    - ``type``: ``regex`` | ``contains`` | ``contains_all``
    - ``regex``: for ``regex`` type, a pattern string; optional ``flags`` (e.g. ``IGNORECASE``)
    - ``value``: for ``contains``, substring required
    - ``values``: for ``contains_all``, list of substrings required
    - ``case_insensitive`` (optional bool, default False for contains/contains_all)

Env: TASK_BENCH_BASE_URL (default http://127.0.0.1:8000), TASK_BENCH_MODEL (served id;
if unset, first id from GET /v1/models), TASK_BENCH_TEMPERATURE, TASK_BENCH_MAX_TOKENS,
TASK_BENCH_TIMEOUT_SEC, TASK_BENCH_PRESERVE_SEPARATE_REASONING, TASK_BENCH_PRESERVE_THINKING.

Usage::

  python3 /workspace/tools/task_benchmark.py \\
    --input /workspace/tools/task_benchmark_seed.jsonl

  python3 /workspace/tools/task_benchmark.py --input /path/to/my_tasks.jsonl --model qwen3.5-35b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from typing import Any


def env_truthy(name: str) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _fetch_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_served_model_id(base_url: str, timeout: float) -> str | None:
    try:
        data = _fetch_json(base_url.rstrip("/") + "/v1/models", timeout)
        rows = data.get("data")
        if not isinstance(rows, list) or not rows:
            return None
        first = rows[0]
        if isinstance(first, dict) and isinstance(first.get("id"), str):
            return first["id"]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    return None


def peer_inference_base_urls(base_url: str) -> list[str]:
    """Primary URL first, then the other common local port (8000 vLLM vs 30000 SGLang in this repo)."""
    primary = base_url.strip().rstrip("/")
    if not primary:
        return []
    parsed = urlsplit(primary if "://" in primary else f"//{primary}", scheme="http")
    scheme = parsed.scheme or "http"
    host = parsed.hostname
    if not host:
        return [primary]
    port = parsed.port
    out: list[str] = [primary]
    if port == 8000:
        alt_netloc = f"{host}:30000"
    elif port == 30000:
        alt_netloc = f"{host}:8000"
    else:
        return out
    alt = urlunsplit((scheme, alt_netloc, parsed.path or "", "", "")).rstrip("/")
    if alt != primary:
        out.append(alt)
    return out


def assistant_text_from_completion(data: object) -> str | None:
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    if isinstance(first.get("text"), str):
        return first["text"]
    msg = first.get("message")
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts) if parts else None
    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and rc:
        return rc
    return None


def run_checker(text: str, checker: object) -> tuple[bool, str]:
    if not isinstance(checker, dict):
        return False, "checker must be an object"
    ctype = checker.get("type")
    if ctype == "regex":
        pat = checker.get("pattern")
        if not isinstance(pat, str):
            return False, "regex checker needs string pattern"
        flags_raw = checker.get("flags")
        flags = 0
        if isinstance(flags_raw, str) and "IGNORECASE" in flags_raw.upper():
            flags |= re.IGNORECASE
        if not re.search(pat, text, flags):
            return False, f"regex did not match: {pat!r}"
        return True, "regex ok"
    if ctype == "contains":
        val = checker.get("value")
        if not isinstance(val, str):
            return False, "contains checker needs string value"
        ci = bool(checker.get("case_insensitive"))
        hay = text.lower() if ci else text
        needle = val.lower() if ci else val
        if needle not in hay:
            return False, f"missing substring: {val!r}"
        return True, "contains ok"
    if ctype == "contains_all":
        vals = checker.get("values")
        if not isinstance(vals, list) or not all(isinstance(x, str) for x in vals):
            return False, "contains_all needs values: list of strings"
        ci = bool(checker.get("case_insensitive"))
        hay = text.lower() if ci else text
        for v in vals:
            n = v.lower() if ci else v
            if n not in hay:
                return False, f"missing substring: {v!r}"
        return True, "contains_all ok"
    return False, f"unknown checker type: {ctype!r}"


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> tuple[int, object | None, str | None]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Keep tool behavior close to dashboard task benchmark defaults:
    # - merge reasoning into content for evaluator parsing
    # - disable visible Qwen "thinking" text unless explicitly preserved
    if not env_truthy("TASK_BENCH_PRESERVE_SEPARATE_REASONING"):
        body["separate_reasoning"] = False
    if not env_truthy("TASK_BENCH_PRESERVE_THINKING"):
        body["chat_template_kwargs"] = {"enable_thinking": False}
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return resp.status, json.loads(raw) if raw else None, None
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else ""
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = raw[:2000] if raw else None
        return e.code, parsed, str(e)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, None, str(e)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"task_benchmark: skip line {line_num}: invalid JSON: {e}", file=sys.stderr)
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def main() -> None:
    default_seed = "/workspace/benchmark/task_benchmark_seed.jsonl"
    here = os.path.dirname(os.path.abspath(__file__))
    fallback_seed = os.path.join(here, "task_benchmark_seed.jsonl")

    p = argparse.ArgumentParser(description="Task-oriented chat benchmark (JSONL problems + checkers).")
    p.add_argument(
        "--input",
        "-i",
        default=os.environ.get("TASK_BENCH_INPUT", default_seed if os.path.isfile(default_seed) else fallback_seed),
        help="JSONL path (default: bundled seed or repo-relative).",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("TASK_BENCH_BASE_URL", "http://127.0.0.1:8000"),
        help="OpenAI-compatible server origin (no path).",
    )
    p.add_argument(
        "--model",
        "-m",
        default=os.environ.get("TASK_BENCH_MODEL", "") or "",
        help="Served model id (default: /v1/models first).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TASK_BENCH_TEMPERATURE", "0.2")),
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("TASK_BENCH_MAX_TOKENS", "1024")),
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("TASK_BENCH_TIMEOUT_SEC", "300")),
        help="Per-request timeout in seconds.",
    )
    args = p.parse_args()

    path = args.input
    if not os.path.isfile(path):
        print(f"task_benchmark: file not found: {path}", file=sys.stderr)
        raise SystemExit(2)

    model = args.model.strip()
    base_url = args.base_url.strip().rstrip("/")
    if not model:
        t_disc = min(30.0, args.timeout)
        for candidate in peer_inference_base_urls(base_url):
            mid = fetch_served_model_id(candidate, t_disc) or ""
            if mid:
                base_url = candidate.rstrip("/")
                model = mid
                break
    if not model:
        print(
            "task_benchmark: set --model or TASK_BENCH_MODEL, or ensure GET /v1/models works.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    cases = load_jsonl(path)
    if not cases:
        print("task_benchmark: no cases loaded.", file=sys.stderr)
        raise SystemExit(2)

    print(f"task_benchmark: model={model!r} cases={len(cases)} base={base_url!r}\n", file=sys.stderr)

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for case in cases:
        cid = str(case.get("id", ""))
        category = str(case.get("category", "unknown"))
        user_prompt = case.get("prompt")
        checker = case.get("checker")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            results.append(
                {
                    "id": cid,
                    "category": category,
                    "ok": False,
                    "error": "missing prompt",
                    "latency_ms": 0,
                }
            )
            continue
        messages: list[dict[str, str]] = []
        sys_msg = case.get("system")
        if isinstance(sys_msg, str) and sys_msg.strip():
            messages.append({"role": "system", "content": sys_msg.strip()})
        messages.append({"role": "user", "content": user_prompt.strip()})

        t_req = time.perf_counter()
        status, data, err = chat_completion(
            base_url,
            model,
            messages,
            args.temperature,
            args.max_tokens,
            args.timeout,
        )
        dt_ms = int((time.perf_counter() - t_req) * 1000)

        if status != 200 or not isinstance(data, dict):
            detail = data if data is not None else err
            results.append(
                {
                    "id": cid,
                    "category": category,
                    "ok": False,
                    "error": f"http {status}: {detail}",
                    "latency_ms": dt_ms,
                }
            )
            continue

        text = assistant_text_from_completion(data) or ""
        ok, reason = run_checker(text, checker)
        preview = (text[:400] + "…") if len(text) > 400 else text
        results.append(
            {
                "id": cid,
                "category": category,
                "ok": ok,
                "reason": reason,
                "latency_ms": dt_ms,
                "preview": preview,
            }
        )
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {cid} ({category}) {dt_ms}ms — {reason}", file=sys.stderr)
        if not ok:
            print(f"  preview: {preview!r}", file=sys.stderr)

    wall_ms = int((time.perf_counter() - t0) * 1000)
    passed = sum(1 for r in results if r.get("ok"))
    by_cat: dict[str, dict[str, int]] = {}
    for r in results:
        c = str(r.get("category", "unknown"))
        bucket = by_cat.setdefault(c, {"pass": 0, "fail": 0})
        if r.get("ok"):
            bucket["pass"] += 1
        else:
            bucket["fail"] += 1

    summary = {
        "model": model,
        "input": path,
        "wall_ms": wall_ms,
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "by_category": by_cat,
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
