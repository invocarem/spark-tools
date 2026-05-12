#!/usr/bin/env python3
"""Minimal web UI for sglang_runtime: presets, sglang arg rows, preview, launch/stop/logs.

Run from the ``sglang_runtime`` directory (next to ``sglang_runtime.py``)::

    pip install fastapi 'uvicorn[standard]'
    uvicorn web_ui.server:app --host 127.0.0.1 --port 8765

Open http://127.0.0.1:8765/ — bind to localhost only unless you add auth.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sglang_runtime as rt

_HERE = Path(__file__).resolve().parent
_RUNTIME_DIR = _HERE.parent
_SCRIPT = _RUNTIME_DIR / "sglang_runtime.py"
_DEFAULT_PRESETS = _RUNTIME_DIR / "model_presets.json"
_STATIC = _HERE / "static"

_ALLOWED_SUBCOMMANDS = frozenset(
    {"launch", "stop", "logs", "scan", "refresh", "benchmark", "measure", "deploy"}
)


def _resolve_presets_file(raw: str | None) -> str:
    if raw and str(raw).strip():
        p = Path(raw).expanduser()
        if not p.is_file():
            raise HTTPException(status_code=400, detail=f"presets file not found: {p}")
        return str(p.resolve())
    if not _DEFAULT_PRESETS.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"default presets missing: {_DEFAULT_PRESETS}",
        )
    return str(_DEFAULT_PRESETS.resolve())


def _load_env_dict(env_file: str) -> dict[str, str]:
    if env_file.strip():
        p = Path(env_file).expanduser()
        if not p.is_file():
            raise HTTPException(status_code=400, detail=f"env file not found: {p}")
        return rt.load_dotenv(str(p.resolve()))
    dot = _RUNTIME_DIR / ".env"
    if dot.is_file():
        return rt.load_dotenv(str(dot))
    return {}


def tokens_to_rows(tokens: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                rows.append(
                    {
                        "kind": "pair",
                        "flag": t,
                        "value": tokens[i + 1],
                        "enabled": True,
                    }
                )
                i += 2
            else:
                rows.append({"kind": "switch", "flag": t, "enabled": True})
                i += 1
        else:
            rows.append({"kind": "raw", "value": t, "enabled": True})
            i += 1
    return rows


def rows_to_tokens(rows: list[dict[str, object]]) -> list[str]:
    out: list[str] = []
    for r in rows:
        if not r.get("enabled", True):
            continue
        kind = r.get("kind")
        if kind == "switch":
            out.append(str(r["flag"]))
        elif kind == "pair":
            out.append(str(r["flag"]))
            out.append(str(r.get("value", "")))
        elif kind == "raw":
            out.append(str(r.get("value", "")))
        else:
            raise ValueError(f"unknown row kind: {kind!r}")
    return out


class ArgRow(BaseModel):
    kind: str = Field(pattern="^(switch|pair|raw)$")
    flag: str = ""
    value: str = ""
    enabled: bool = True


class PreviewRequest(BaseModel):
    presets_file: str = ""
    preset: str
    env_file: str = ""
    rows: list[ArgRow] = Field(default_factory=list)
    extra_sglang: str = ""
    override_tp: int | None = None
    override_port: int | None = None
    override_model_path: str = ""
    override_venv_path: str = ""


class LaunchRequest(PreviewRequest):
    mode: str = Field(default="solo", pattern="^(solo|cluster)$")
    host: str = ""
    hosts: list[str] = Field(default_factory=list)
    log_dir: str | None = None
    log_file: str | None = None
    dist_addr: str = ""
    verbose: bool = False


class StopRequest(BaseModel):
    mode: str = Field(default="solo", pattern="^(solo|cluster)$")
    host: str = ""
    hosts: list[str] = Field(default_factory=list)
    preset: str = ""
    presets_file: str = ""
    env_file: str = ""
    port: int | None = None
    grace_sec: int = 5


class LogsRequest(BaseModel):
    mode: str = Field(default="solo", pattern="^(solo|cluster)$")
    host: str = ""
    hosts: list[str] = Field(default_factory=list)
    env_file: str = ""
    log_dir: str | None = None
    log_file: str | None = None
    lines: int = 80
    from_start: bool = False
    role: str = Field(default="head", pattern="^(head|worker)$")
    node_rank: int | None = None


class ExecRequest(BaseModel):
    """Run an allowed subcommand with extra argv (no shell)."""

    subcommand: str
    args: list[str] = Field(default_factory=list)


class ScanRequest(BaseModel):
    """Maps to ``scan`` / ``refresh`` CLI (HTTP probes or SSH remote probe)."""

    presets_file: str = ""
    preset: str = ""
    env_file: str = ""
    port: int | None = None
    base_url: str = ""
    bind_host: str = "127.0.0.1"
    host: str = ""
    api_key: str = "EMPTY"
    timeout_sec: int = 30
    readiness: bool = False


def _probe_ok(block: object) -> bool | None:
    if not isinstance(block, dict):
        return None
    return bool(block.get("ok"))


def _models_from_v1_models(block: object) -> list[str]:
    if not isinstance(block, dict) or not block.get("ok"):
        return []
    body = block.get("body")
    if not isinstance(body, dict):
        return []
    data = body.get("data")
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for item in data:
        if isinstance(item, dict) and "id" in item:
            out.append(str(item["id"]))
    return out


def _server_info_head(block: object, max_chars: int = 500) -> str | None:
    if not isinstance(block, dict) or not block.get("ok"):
        return None
    body = block.get("body")
    if body is None:
        return None
    try:
        text = json.dumps(body, indent=2, default=str)
    except TypeError:
        text = str(body)
    if len(text) > max_chars:
        return text[:max_chars] + "\n…"
    return text


def summarize_scan_payload(payload: dict[str, object]) -> dict[str, object]:
    """Short fields for the UI; full JSON stays in ``payload``."""
    v1 = payload.get("v1_models")
    health = payload.get("health")
    hg = payload.get("health_generate")
    si = payload.get("server_info")
    sia = payload.get("server_info_alt")
    models = _models_from_v1_models(v1)
    info_block = si if _probe_ok(si) else sia
    notes: list[str] = []
    if isinstance(si, dict) and not si.get("ok") and sia is not None:
        notes.append("Used /server_info fallback (get_server_info failed).")
    return {
        "base_url": str(payload.get("base_url", "")),
        "health_ok": _probe_ok(health),
        "health_status": health.get("status") if isinstance(health, dict) else None,
        "health_error": health.get("error") if isinstance(health, dict) else None,
        "readiness_ok": _probe_ok(hg) if hg is not None else None,
        "models": models,
        "server_info_ok": _probe_ok(info_block),
        "server_info_preview": _server_info_head(info_block),
        "v1_models_ok": _probe_ok(v1),
        "notes": notes,
    }


def _base_argv() -> list[str]:
    return [sys.executable, str(_SCRIPT)]


def _run_cli(argv: list[str], timeout: float | None = 600) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = str(_RUNTIME_DIR) + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = str(_RUNTIME_DIR)
    return subprocess.run(
        argv,
        cwd=str(_RUNTIME_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


app = FastAPI(title="sglang_runtime UI", version="0.1.0")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/api/defaults")
def api_defaults() -> dict[str, str]:
    return {
        "presets_file": str(_DEFAULT_PRESETS.resolve())
        if _DEFAULT_PRESETS.is_file()
        else "",
        "runtime_dir": str(_RUNTIME_DIR.resolve()),
        "script": str(_SCRIPT.resolve()),
    }


@app.get("/api/presets")
def api_presets(presets_file: str = "") -> dict[str, object]:
    path = _resolve_presets_file(presets_file or None)
    try:
        data = rt.load_presets(path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        json.dumps(data)
    except (TypeError, ValueError):
        raise HTTPException(status_code=500, detail="presets are not JSON-serializable")
    return {"presets_file": path, "names": sorted(data), "raw": data}


@app.get("/api/preset/{name}/sglang-rows")
def api_preset_rows(name: str, presets_file: str = "") -> dict[str, object]:
    path = _resolve_presets_file(presets_file or None)
    try:
        presets = rt.load_presets(path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if name not in presets:
        raise HTTPException(
            status_code=404,
            detail=f"preset {name!r} not in {path}",
        )
    preset = presets[name]
    tokens = rt.get_preset_sglang_args(preset)
    return {"preset": name, "rows": tokens_to_rows(tokens), "tokens": tokens}


@app.post("/api/preview-launch")
def api_preview_launch(body: PreviewRequest) -> dict[str, object]:
    presets_file = _resolve_presets_file(body.presets_file or None)
    env = _load_env_dict(body.env_file)

    try:
        merged = rt.merge_preset_launch_fields(
            presets_file,
            body.preset,
            env,
            override_model_path=body.override_model_path or None,
            override_venv_path=body.override_venv_path or None,
            override_tp=body.override_tp,
            override_port=body.override_port,
            extra_sglang_args=shlex.split(body.extra_sglang or ""),
            preset_sglang_args=rows_to_tokens([r.model_dump() for r in body.rows]),
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    shell = rt.build_dashboard_source_launch_command(
        presets_file=presets_file,
        preset_name=body.preset,
        env=env,
        override_model_path=body.override_model_path or None,
        override_venv_path=body.override_venv_path or None,
        override_tp=body.override_tp,
        override_port=body.override_port,
        extra_sglang_args=shlex.split(body.extra_sglang or ""),
        preset_sglang_args=rows_to_tokens([r.model_dump() for r in body.rows]),
    )
    return {
        "merged": {
            "model_path": merged.model_path,
            "venv_path": merged.venv_path,
            "tp": merged.tp,
            "port": merged.port,
            "sglang_args": merged.sglang_args,
        },
        "launch_shell": shell,
    }


@app.post("/api/launch")
def api_launch(body: LaunchRequest) -> dict[str, object]:
    presets_file = _resolve_presets_file(body.presets_file or None)
    env = _load_env_dict(body.env_file)
    preset_sglang = rows_to_tokens([r.model_dump() for r in body.rows])

    try:
        shell = rt.build_dashboard_source_launch_command(
            presets_file=presets_file,
            preset_name=body.preset,
            env=env,
            override_model_path=body.override_model_path or None,
            override_venv_path=body.override_venv_path or None,
            override_tp=body.override_tp,
            override_port=body.override_port,
            extra_sglang_args=shlex.split(body.extra_sglang or ""),
            preset_sglang_args=preset_sglang,
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    argv = _base_argv() + [
        "launch",
        "--mode",
        body.mode,
        "--preset",
        body.preset,
        "--presets-file",
        presets_file,
        "--command",
        shell,
    ]
    if body.env_file.strip():
        argv += ["--env-file", str(Path(body.env_file).expanduser().resolve())]
    if body.mode == "solo" and body.host.strip():
        argv += ["--host", body.host.strip()]
    if body.mode == "cluster" and body.hosts:
        argv += ["--hosts", *body.hosts]
    if body.log_dir:
        argv += ["--log-dir", body.log_dir]
    if body.log_file:
        argv += ["--log-file", body.log_file]
    if body.dist_addr.strip():
        argv += ["--dist-addr", body.dist_addr.strip()]
    if body.verbose:
        argv.insert(2, "--verbose")

    proc = _run_cli(argv, timeout=None)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "argv": argv,
    }


@app.post("/api/stop")
def api_stop(body: StopRequest) -> dict[str, object]:
    if body.presets_file.strip():
        presets_file = _resolve_presets_file(body.presets_file)
    elif body.preset.strip():
        presets_file = _resolve_presets_file("")
    else:
        presets_file = ""
    argv = _base_argv() + ["stop", "--mode", body.mode]
    if body.host.strip():
        argv += ["--host", body.host.strip()]
    if body.hosts:
        argv += ["--hosts", *body.hosts]
    if body.preset.strip():
        argv += ["--preset", body.preset.strip()]
    if presets_file:
        argv += ["--presets-file", presets_file]
    if body.env_file.strip():
        argv += ["--env-file", str(Path(body.env_file).expanduser().resolve())]
    if body.port is not None:
        argv += ["--port", str(body.port)]
    if body.grace_sec != 5:
        argv += ["--grace-sec", str(body.grace_sec)]
    proc = _run_cli(argv, timeout=120)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "argv": argv,
    }


@app.post("/api/logs")
def api_logs(body: LogsRequest) -> dict[str, object]:
    argv = _base_argv() + ["logs", "--mode", body.mode, "-n", str(body.lines)]
    if body.host.strip():
        argv += ["--host", body.host.strip()]
    if body.hosts:
        argv += ["--hosts", *body.hosts]
    if body.log_dir:
        argv += ["--log-dir", body.log_dir]
    if body.log_file:
        argv += ["--log-file", body.log_file]
    if body.from_start:
        argv.append("--from-start")
    argv += ["--role", body.role]
    if body.node_rank is not None:
        argv += ["--node-rank", str(body.node_rank)]
    if body.env_file.strip():
        argv += ["--env-file", str(Path(body.env_file).expanduser().resolve())]
    proc = _run_cli(argv, timeout=60)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "argv": argv,
    }


@app.post("/api/scan")
@app.post("/api/refresh")
def api_scan(body: ScanRequest) -> dict[str, object]:
    argv = _base_argv() + ["scan"]
    if body.presets_file.strip():
        argv += ["--presets-file", _resolve_presets_file(body.presets_file)]
    elif body.preset.strip():
        argv += ["--presets-file", _resolve_presets_file("")]
    if body.preset.strip():
        argv += ["--preset", body.preset.strip()]
    if body.env_file.strip():
        argv += ["--env-file", str(Path(body.env_file).expanduser().resolve())]
    if body.port is not None:
        argv += ["--port", str(int(body.port))]
    if body.base_url.strip():
        argv += ["--base-url", body.base_url.strip()]
    if body.bind_host.strip() and body.bind_host.strip() != "127.0.0.1":
        argv += ["--bind-host", body.bind_host.strip()]
    if body.host.strip():
        argv += ["--host", body.host.strip()]
    if body.api_key.strip() and body.api_key.strip() != "EMPTY":
        argv += ["--api-key", body.api_key.strip()]
    if body.timeout_sec != 30:
        argv += ["--timeout-sec", str(int(body.timeout_sec))]
    if body.readiness:
        argv.append("--readiness")

    proc = _run_cli(argv, timeout=float(body.timeout_sec) + 15.0)
    scan_obj: dict[str, object] | None = None
    raw = proc.stdout.strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                scan_obj = parsed
        except json.JSONDecodeError:
            pass

    summary: dict[str, object] | None = None
    if scan_obj is not None:
        summary = summarize_scan_payload(scan_obj)

    return {
        "returncode": proc.returncode,
        "argv": argv,
        "scan": scan_obj,
        "summary": summary,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


@app.get("/api/health")
def api_health() -> dict[str, object]:
    """Cheap probe: confirms you hit this FastAPI app (not a static file server)."""
    return {"ok": True, "service": "sglang_runtime_web_ui"}


@app.post("/api/exec")
def api_exec(body: ExecRequest) -> dict[str, object]:
    sub = body.subcommand.strip()
    if sub not in _ALLOWED_SUBCOMMANDS:
        raise HTTPException(
            status_code=400,
            detail=f"subcommand must be one of: {sorted(_ALLOWED_SUBCOMMANDS)}",
        )
    argv = _base_argv() + [sub, *body.args]
    proc = _run_cli(argv, timeout=3600)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "argv": argv,
    }
