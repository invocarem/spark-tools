"""Shared helpers for ``benchmark_sglang.py`` and ``benchmark_vllm.py``."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def env_truthy(name: str) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def load_json_object(label: str, raw: str, prog: str) -> dict:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"{prog}: invalid JSON for {label}: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    if not isinstance(obj, dict):
        print(f"{prog}: {label} must be a JSON object", file=sys.stderr)
        raise SystemExit(2)
    return obj


def pop_json_flag_from_argv(argv: list[str], flag: str, prog: str) -> tuple[dict, list[str]]:
    """Strip ``flag <json>`` pairs from argv and merge JSON objects."""
    merged: dict = {}
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            merged.update(load_json_object(flag, argv[i + 1], prog))
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return merged, out


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def fetch_served_model_id(base_url: str, timeout_sec: float = 15.0) -> str | None:
    url = base_url.rstrip("/") + "/v1/models"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode()
        data = json.loads(raw)
        rows = data.get("data")
        if not isinstance(rows, list) or len(rows) == 0:
            return None
        first = rows[0]
        if isinstance(first, dict) and isinstance(first.get("id"), str):
            return first["id"]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    return None
