#!/usr/bin/env python3
"""Smoke tests for sglang_runtime.py — no SSH, no GPU, no running server needed."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

SCRIPT = pathlib.Path(__file__).parent / "sglang_runtime.py"
PY = sys.executable or "python3"


def _run(args: list[str], **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PY, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
        **kw,
    )


def test_help() -> None:
    """--help exits 0."""
    r = _run(["--help"])
    assert r.returncode == 0, r.stderr
    assert "sglang" in r.stdout.lower()


def test_no_subcommand() -> None:
    """No subcommand produces an error."""
    r = _run([])
    assert r.returncode != 0


SUBCOMMANDS = ("deploy", "launch", "stop", "logs", "scan", "benchmark", "measure")


def test_subcommand_help() -> None:
    """--help for every subcommand exits 0."""
    for sc in SUBCOMMANDS:
        r = _run([sc, "--help"])
        assert r.returncode == 0, f"{sc} --help failed: {r.stderr}"


# ── Preset loading ────────────────────────────────────────────────

def _write_presets(content: dict, tmpdir: pathlib.Path) -> str:
    p = tmpdir / "presets.json"
    p.write_text(json.dumps(content))
    return str(p)


def test_list_presets() -> None:
    """launch --list-presets prints preset names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        presets_file = _write_presets(
            {"alpha": {"model_path": "/a"}, "beta": {"model_path": "/b"}},
            pathlib.Path(tmpdir),
        )
        r = _run(["launch", "--list-presets", "--presets-file", presets_file])
    assert r.returncode == 0, r.stderr
    out = r.stdout.strip().splitlines()
    assert "alpha" in out
    assert "beta" in out


def test_list_presets_missing_file() -> None:
    """launch --list-presets with nonexistent file exits non-zero."""
    r = _run(["launch", "--list-presets", "--presets-file", "/no/such/file.json"])
    assert r.returncode != 0


def test_list_presets_invalid_json() -> None:
    """launch --list-presets with invalid JSON exits non-zero."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write("not json")
        f.flush()
        r = _run(["launch", "--list-presets", "--presets-file", f.name])
    assert r.returncode != 0


def test_list_presets_not_object() -> None:
    """Presets file must be a JSON object, not an array."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write(json.dumps(["a", "b"]))
        f.flush()
        r = _run(["launch", "--list-presets", "--presets-file", f.name])
    assert r.returncode != 0


def test_stop_preset_not_found() -> None:
    """stop with unknown preset exits non-zero."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write(json.dumps({"only": {"model_path": "/x"}}))
        f.flush()
        r = _run(["stop", "--preset", "missing", "--presets-file", f.name])
    assert r.returncode != 0
    assert "missing" in r.stderr


def test_scan_alias() -> None:
    """'refresh' alias for 'scan' should parse (fails on network, not on argparse)."""
    r = _run(["refresh", "--help"])
    assert r.returncode == 0, r.stderr


# ── Value resolution (unit-testable helpers) ──────────────────────

def test_resolve_value_priority() -> None:
    """CLI > env > preset > default."""
    from sglang_runtime import resolve_value
    assert resolve_value("cli", "env", "preset", "def") == "cli"
    assert resolve_value(None, "env", "preset", "def") == "env"
    assert resolve_value(None, None, "preset", "def") == "preset"
    assert resolve_value(None, None, None, "def") == "def"


def test_env_get() -> None:
    """env dict > os.environ > default."""
    import os
    from sglang_runtime import env_get
    assert env_get({"K": "v"}, "K", "x") == "v"
    os.environ["_SG_TEST_KEY"] = "from_os"
    try:
        assert env_get({}, "_SG_TEST_KEY", "x") == "from_os"
        assert env_get({}, "_SG_NOSUCH", "x") == "x"
    finally:
        del os.environ["_SG_TEST_KEY"]


def test_parse_csv() -> None:
    from sglang_runtime import parse_csv
    assert parse_csv("a,b, c ") == ["a", "b", "c"]
    assert parse_csv("") == []


def test_load_presets() -> None:
    from sglang_runtime import load_presets
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write(json.dumps({"x": {"model_path": "/m", "tp": 2}}))
        f.flush()
        presets = load_presets(f.name)
    assert presets == {"x": {"model_path": "/m", "tp": 2}}


# ── Helpers ───────────────────────────────────────────────────────

def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            failed += 1
    total = len(tests)
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
