#!/usr/bin/env python3
"""Stack UI API: drives sglang runtimes from repo root.

Supports two runtimes:
- ``venv`` (default): ``sglang_runtime`` — local Python venv
- ``docker``: ``sglang_docker`` — Docker containers

Run::

    cd stack_ui/backend
    pip install -r requirements.txt
    uvicorn stack_ui_server:app --host 127.0.0.1 --port 8765

Dev with Vite (from ``stack_ui/frontend``)::

    npm install && npm run dev

If ``../frontend/dist`` exists (``npm run build``), static assets and SPA
fallback are served from this app on the same port.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]

# --- runtime dispatch -------------------------------------------------------
_VENV_DIR = _REPO_ROOT / "sglang_runtime"
_VENV_SCRIPT = _VENV_DIR / "sglang_runtime.py"
_VENV_PRESETS = _VENV_DIR / "model_presets.json"

_DOCKER_DIR = _REPO_ROOT / "sglang_docker"
_DOCKER_SCRIPT = _DOCKER_DIR / "sglang_docker.py"
_DOCKER_PRESETS = _DOCKER_DIR / "model_presets.json"

_RUNDOT_DIR = _REPO_ROOT / "sglang_runtime"
_DOCKERDOT_DIR = _REPO_ROOT / "sglang_docker"

_RUNTIME_MAP = {
    "venv": {
        "dir": _VENV_DIR,
        "script": _VENV_SCRIPT,
        "presets": _VENV_PRESETS,
        "dotenv": _RUNDOT_DIR / ".env",
    },
    "docker": {
        "dir": _DOCKER_DIR,
        "script": _DOCKER_SCRIPT,
        "presets": _DOCKER_PRESETS,
        "dotenv": _DOCKERDOT_DIR / ".env",
    },
}

_VENV_SUBCOMMANDS = frozenset(
    {"launch", "stop", "logs", "scan", "refresh", "benchmark", "measure", "deploy"}
)
_DOCKER_SUBCOMMANDS = frozenset(
    {"launch", "stop", "logs", "scan", "refresh", "benchmark", "measure", "pull"}
)


def _get_runtime(runtime: str) -> dict[str, Path]:
    if runtime not in _RUNTIME_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"runtime must be one of: {sorted(_RUNTIME_MAP)}",
        )
    return _RUNTIME_MAP[runtime]


def _allowed_subcommands(runtime: str) -> frozenset:
    return _VENV_SUBCOMMANDS if runtime == "venv" else _DOCKER_SUBCOMMANDS


def _effective_runtime(body: BaseModel) -> str:
    """Pull the runtime from a request body, defaulting to 'venv'."""
    return getattr(body, "runtime", "venv")


# ---------------------------------------------------------------------------
_CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "STACK_UI_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if o.strip()
]


def _resolve_presets_file(raw: str | None, runtime: str = "venv") -> str:
    info = _get_runtime(runtime)
    if raw and str(raw).strip():
        p = Path(raw).expanduser()
        if not p.is_file():
            raise HTTPException(status_code=400, detail=f"presets file not found: {p}")
        return str(p.resolve())
    default = info["presets"]
    if not default.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"default presets missing: {default}",
        )
    return str(default.resolve())


def _load_env_dict(env_file: str, runtime: str = "venv") -> dict[str, str]:
    if env_file and env_file.strip():
        p = Path(env_file).expanduser()
        if not p.is_file():
            raise HTTPException(status_code=400, detail=f"env file not found: {p}")
        return load_dotenv_impl(str(p.resolve()))
    info = _get_runtime(runtime)
    dot = info["dotenv"]
    if dot.is_file():
        return load_dotenv_impl(str(dot))
    return {}


# --- import shared utilities once -------------------------------------------
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sglang_common import (  # noqa: E402
    load_dotenv as load_dotenv_impl,
    load_presets,
    get_preset_sglang_args,
    get_preset_int,
    get_preset_string,
    resolve_tp,
    resolve_value,
    env_lookup,
)

if str(_VENV_DIR) not in sys.path:
    sys.path.insert(0, str(_VENV_DIR))
import sglang_runtime as rt  # noqa: E402


# --- helpers ----------------------------------------------------------------

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


# --- request models ---------------------------------------------------------

class ArgRow(BaseModel):
    kind: str = Field(pattern="^(switch|pair|raw)$")
    flag: str = ""
    value: str = ""
    enabled: bool = True


class PreviewRequest(BaseModel):
    runtime: str = Field(default="venv", pattern="^(venv|docker)$")
    presets_file: str = ""
    preset: str
    env_file: str = ""
    rows: list[ArgRow] = Field(default_factory=list)
    extra_sglang: str = ""
    override_tp: int | None = None
    override_port: int | None = None
    override_model_path: str = ""
    override_venv_path: str = ""
    override_image: str = ""


class LaunchRequest(PreviewRequest):
    mode: str = Field(default="solo", pattern="^(solo|cluster)$")
    host: str = ""
    hosts: list[str] = Field(default_factory=list)
    log_dir: str | None = None
    log_file: str | None = None
    dist_addr: str = ""
    verbose: bool = False


class StopRequest(BaseModel):
    runtime: str = Field(default="venv", pattern="^(venv|docker)$")
    mode: str = Field(default="solo", pattern="^(solo|cluster)$")
    host: str = ""
    hosts: list[str] = Field(default_factory=list)
    preset: str = ""
    presets_file: str = ""
    env_file: str = ""
    port: int | None = None
    grace_sec: int = 5


class LogsRequest(BaseModel):
    runtime: str = Field(default="venv", pattern="^(venv|docker)$")
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

    runtime: str = Field(default="venv", pattern="^(venv|docker)$")
    subcommand: str
    args: list[str] = Field(default_factory=list)


class ScanRequest(BaseModel):
    """Maps to ``scan`` / ``refresh`` CLI (HTTP probes or SSH remote probe)."""

    runtime: str = Field(default="venv", pattern="^(venv|docker)$")
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


# --- scan helpers -----------------------------------------------------------

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
        return text[:max_chars] + "…"
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


# --- CLI execution ----------------------------------------------------------

def _base_argv(runtime: str) -> list[str]:
    info = _get_runtime(runtime)
    return [sys.executable, str(info["script"])]


def _run_cli(
    argv: list[str],
    runtime: str = "venv",
    timeout: float | None = 600,
) -> subprocess.CompletedProcess[str]:
    info = _get_runtime(runtime)
    env = os.environ.copy()
    rdir = str(info["dir"])
    # Ensure repo root is on PYTHONPATH so sglang_common is importable
    root_str = str(_REPO_ROOT)
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = rdir + os.pathsep + root_str + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = rdir + os.pathsep + root_str
    return subprocess.run(
        argv,
        cwd=rdir,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


# --- FastAPI app ------------------------------------------------------------

app = FastAPI(title="Stack UI", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/defaults")
def api_defaults() -> dict[str, object]:
    venv_presets = str(_VENV_PRESETS.resolve()) if _VENV_PRESETS.is_file() else ""
    docker_presets = str(_DOCKER_PRESETS.resolve()) if _DOCKER_PRESETS.is_file() else ""
    return {
        "presets_file": venv_presets,
        "runtime_dir": str(_VENV_DIR.resolve()),
        "script": str(_VENV_SCRIPT.resolve()),
        "repo_root": str(_REPO_ROOT.resolve()),
        "runtimes": {
            "venv": {
                "script": str(_VENV_SCRIPT.resolve()),
                "presets_file": venv_presets,
                "dir": str(_VENV_DIR.resolve()),
                "subcommands": sorted(_VENV_SUBCOMMANDS),
            },
            "docker": {
                "script": str(_DOCKER_SCRIPT.resolve()),
                "presets_file": docker_presets,
                "dir": str(_DOCKER_DIR.resolve()),
                "subcommands": sorted(_DOCKER_SUBCOMMANDS),
            },
        },
    }


@app.get("/api/presets")
def api_presets(presets_file: str = "", runtime: str = "venv") -> dict[str, object]:
    path = _resolve_presets_file(presets_file or None, runtime)
    try:
        data = load_presets(path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        json.dumps(data)
    except (TypeError, ValueError):
        raise HTTPException(status_code=500, detail="presets are not JSON-serializable")
    return {"presets_file": path, "names": sorted(data), "raw": data, "runtime": runtime}


@app.get("/api/preset/{name}/sglang-rows")
def api_preset_rows(name: str, presets_file: str = "", runtime: str = "venv") -> dict[str, object]:
    path = _resolve_presets_file(presets_file or None, runtime)
    try:
        presets = load_presets(path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if name not in presets:
        raise HTTPException(
            status_code=404,
            detail=f"preset {name!r} not in {path}",
        )
    preset = presets[name]
    tokens = get_preset_sglang_args(preset)
    return {"preset": name, "rows": tokens_to_rows(tokens), "tokens": tokens}


@app.post("/api/preview-launch")
def api_preview_launch(body: PreviewRequest) -> dict[str, object]:
    presets_file = _resolve_presets_file(body.presets_file or None, body.runtime)
    env = _load_env_dict(body.env_file, body.runtime)

    if body.runtime == "docker":
        return _preview_launch_docker(presets_file, env, body)
    # Default: venv
    return _preview_launch_venv(presets_file, env, body)


def _preview_launch_venv(
    presets_file: str,
    env: dict[str, str],
    body: PreviewRequest,
) -> dict[str, object]:
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
        "runtime": "venv",
        "merged": {
            "model_path": merged.model_path,
            "venv_path": merged.venv_path,
            "tp": merged.tp,
            "port": merged.port,
            "sglang_args": merged.sglang_args,
        },
        "launch_shell": shell,
    }


def _preview_launch_docker(
    presets_file: str,
    env: dict[str, str],
    body: PreviewRequest,
) -> dict[str, object]:
    """Build a docker launch preview without the sglang_runtime venv helpers."""
    presets = load_presets(presets_file)
    preset_name = body.preset
    if preset_name not in presets:
        raise HTTPException(
            status_code=404,
            detail=f"preset {preset_name!r} not in {presets_file}",
        )
    preset = presets[preset_name]

    model_path = str(
        resolve_value(
            body.override_model_path or None,
            env_lookup(env, "MODEL_PATH"),
            get_preset_string(preset, "model_path"),
            "~/huggingface/Qwen_Qwen3.5-2B",
        )
    )
    image = str(
        resolve_value(
            body.override_image or None,
            env_lookup(env, "DOCKER_IMAGE"),
            get_preset_string(preset, "image"),
            "scitrera/dgx-spark-sglang:latest",
        )
    )
    if body.override_tp is not None:
        tp = int(body.override_tp)
    else:
        preset_tp = get_preset_int(preset, "tp")
        tp = int(preset_tp) if preset_tp is not None else 1

    port = int(
        resolve_value(
            body.override_port,
            env_lookup(env, "SERVER_PORT"),
            get_preset_int(preset, "port"),
            30000,
        )
    )
    preset_sglang = rows_to_tokens([r.model_dump() for r in body.rows])
    merged_args = [
        *preset_sglang,
        *shlex.split(env_lookup(env, "SGLANG_EXTRA_ARGS") or ""),
        *shlex.split(body.extra_sglang or ""),
    ]
    if "--served-model-name" not in merged_args:
        merged_args.extend(["--served-model-name", preset_name])

    # Build a docker run command for preview
    from sglang_common import (
        _NCCL_ENV_KEYS,
        build_export_prefix,
        shell_quote_path_allow_home,
    )

    container = f"sglang-{preset_name}"
    expanded_model = os.path.expandvars(os.path.expanduser(model_path))
    nccl_prefix = build_export_prefix(env, _NCCL_ENV_KEYS)

    env_flags = []
    for key in _NCCL_ENV_KEYS:
        if key in env:
            env_flags.extend(["-e", key])

    extra_sglang = " ".join(shlex.quote(arg) for arg in merged_args)

    cmd = (
        f"{nccl_prefix}docker run -d --name {container} --gpus all --network host "
        f"-v {shlex.quote(expanded_model)}:{shlex.quote(expanded_model)}:ro "
        f"{' '.join(env_flags)} "
        f"{shlex.quote(image)} "
        f"python -m sglang.launch_server "
        f"--model-path {shell_quote_path_allow_home(model_path)} "
        f"--tp {tp} --host 0.0.0.0 --port {port}"
    )
    if extra_sglang:
        cmd = f"{cmd} {extra_sglang}"

    return {
        "runtime": "docker",
        "merged": {
            "model_path": model_path,
            "image": image,
            "tp": tp,
            "port": port,
            "sglang_args": merged_args,
        },
        "launch_shell": cmd,
    }


@app.post("/api/launch")
def api_launch(body: LaunchRequest) -> dict[str, object]:
    presets_file = _resolve_presets_file(body.presets_file or None, body.runtime)
    env = _load_env_dict(body.env_file, body.runtime)
    preset_sglang = rows_to_tokens([r.model_dump() for r in body.rows])

    if body.runtime == "docker":
        return _launch_docker(body, presets_file, env, preset_sglang)

    # --- venv path ----------------------------------------------------------
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

    argv = _base_argv(body.runtime) + [
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

    proc = _run_cli(argv, runtime=body.runtime, timeout=None)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "argv": argv,
    }


def _launch_docker(
    body: LaunchRequest,
    presets_file: str,
    env: dict[str, str],
    preset_sglang: list[str],
) -> dict[str, object]:
    """Build argv for sglang_docker launch."""
    argv = _base_argv("docker") + [
        "launch",
        "--mode",
        body.mode,
        "--preset",
        body.preset,
        "--presets-file",
        presets_file,
    ]
    if body.env_file.strip():
        argv += ["--env-file", str(Path(body.env_file).expanduser().resolve())]
    if body.mode == "solo" and body.host.strip():
        argv += ["--host", body.host.strip()]
    if body.mode == "cluster" and body.hosts:
        argv += ["--hosts", *body.hosts]
    if body.override_model_path and body.override_model_path.strip():
        argv += ["--model-path", body.override_model_path.strip()]
    if body.override_image and body.override_image.strip():
        argv += ["--image", body.override_image.strip()]
    if body.override_tp is not None:
        argv += ["--tp", str(body.override_tp)]
    if body.override_port is not None:
        argv += ["--port", str(body.override_port)]
    if body.log_dir:
        argv += ["--log-dir", body.log_dir]
    if body.dist_addr.strip():
        argv += ["--dist-addr", body.dist_addr.strip()]
    extra = body.extra_sglang.strip()
    if extra:
        argv += ["--sglang-args", extra]
    if body.verbose:
        argv.insert(2, "--verbose")

    proc = _run_cli(argv, runtime="docker", timeout=None)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "argv": argv,
    }


@app.post("/api/stop")
def api_stop(body: StopRequest) -> dict[str, object]:
    if body.presets_file.strip():
        presets_file = _resolve_presets_file(body.presets_file, body.runtime)
    elif body.preset.strip():
        presets_file = _resolve_presets_file("", body.runtime)
    else:
        presets_file = ""
    argv = _base_argv(body.runtime) + ["stop", "--mode", body.mode]
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
    proc = _run_cli(argv, runtime=body.runtime, timeout=120)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "argv": argv,
    }


@app.post("/api/logs")
def api_logs(body: LogsRequest) -> dict[str, object]:
    argv = _base_argv(body.runtime) + ["logs", "--mode", body.mode, "-n", str(body.lines)]
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
    if body.runtime == "venv":
        argv += ["--role", body.role]
    else:
        argv += ["--role", body.role]
    if body.node_rank is not None:
        argv += ["--node-rank", str(body.node_rank)]
    if body.env_file.strip():
        argv += ["--env-file", str(Path(body.env_file).expanduser().resolve())]
    proc = _run_cli(argv, runtime=body.runtime, timeout=60)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "argv": argv,
    }


@app.post("/api/scan")
@app.post("/api/refresh")
def api_scan(body: ScanRequest) -> dict[str, object]:
    argv = _base_argv(body.runtime) + ["scan"]
    if body.presets_file.strip():
        argv += ["--presets-file", _resolve_presets_file(body.presets_file, body.runtime)]
    elif body.preset.strip():
        argv += ["--presets-file", _resolve_presets_file("", body.runtime)]
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

    proc = _run_cli(argv, runtime=body.runtime, timeout=float(body.timeout_sec) + 15.0)
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
    return {"ok": True, "service": "stack_ui"}


@app.post("/api/exec")
def api_exec(body: ExecRequest) -> dict[str, object]:
    sub = body.subcommand.strip()
    allowed = _allowed_subcommands(body.runtime)
    if sub not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"subcommand must be one of: {sorted(allowed)}",
        )
    argv = _base_argv(body.runtime) + [sub, *body.args]
    proc = _run_cli(argv, runtime=body.runtime, timeout=3600)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "argv": argv,
    }


# --- SPA serving ------------------------------------------------------------

_DIST = _REPO_ROOT / "stack_ui" / "frontend" / "dist"


def _register_spa() -> None:
    if not (_DIST / "index.html").is_file():
        return
    assets = _DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="vite_assets")

    @app.get("/")
    def spa_index() -> FileResponse:
        return FileResponse(_DIST / "index.html")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="not found")
        candidate = _DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")


_register_spa()
