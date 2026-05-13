#!/usr/bin/env python3
"""Manage custom sglang runtime across DGX Spark nodes.

This tool provides practical operations:
- deploy: sync source/runtime files to one or more remote nodes
- launch: start sglang server locally or remotely
- stop: stop launched sglang server locally or remotely
- scan (alias refresh): query the running server for models, health, and server info
- benchmark: run simple latency/throughput benchmark via OpenAI-compatible API
- measure: capture GPU/CPU/memory snapshots (local or remote)
- logs: show recent log lines (solo log file or cluster head/worker node files)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Ensure repo root is on sys.path so sglang_common is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sglang_common import (
    _NCCL_ENV_KEYS,
    build_export_prefix,
    build_remote_scan_script,
    collect_running_server_scan,
    env_get,
    env_lookup,
    get_preset_csv_or_list,
    get_preset_int,
    get_preset_sglang_args,
    get_preset_string,
    load_env_from_args,
    load_presets,
    normalize_local_sources,
    parse_csv,
    resolve_tp,
    resolve_value,
    run_benchmark,
    run_cmd,
    run_remote,
    run_shell,
    shell_quote_path_allow_home,
)


def deploy(args: argparse.Namespace) -> int:
    env = load_env_from_args(args)
    deploy_set_name = args.set or env_lookup(env, "DEPLOY_SET") or ""
    deploy_sets_file = env_lookup(env, "DEPLOY_SETS_FILE") or args.deploy_sets_file
    deploy_set: dict[str, object] = {}
    deploy_sets: dict[str, dict[str, object]] = {}

    if args.list_sets:
        try:
            deploy_sets = load_presets(deploy_sets_file)
        except FileNotFoundError:
            print(f"Deploy sets file not found: {deploy_sets_file}", file=sys.stderr)
            return 2
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"Failed to parse deploy sets file '{deploy_sets_file}': {exc}", file=sys.stderr)
            return 2
        for name in sorted(deploy_sets):
            print(name)
        return 0

    if deploy_set_name:
        try:
            deploy_sets = load_presets(deploy_sets_file)
        except FileNotFoundError:
            print(f"Deploy sets file not found: {deploy_sets_file}", file=sys.stderr)
            return 2
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"Failed to parse deploy sets file '{deploy_sets_file}': {exc}", file=sys.stderr)
            return 2
        if deploy_set_name not in deploy_sets:
            print(
                f"Deploy set '{deploy_set_name}' not found in {deploy_sets_file}. "
                f"Available: {', '.join(sorted(deploy_sets)) or '(none)'}",
                file=sys.stderr,
            )
            return 2
        deploy_set = deploy_sets[deploy_set_name]

    remote_dir = str(resolve_value(
        args.remote_dir,
        env_lookup(env, "REMOTE_DIR"),
        get_preset_string(deploy_set, "remote_dir"),
        "~/runtime-sglang",
    ))
    default_hosts = [h for h in [env.get("MASTER_NODE"), env.get("WORKER_NODE")] if h]
    hosts = args.hosts or default_hosts
    if not hosts:
        print("No hosts provided. Use --hosts or MASTER_NODE/WORKER_NODE in config.", file=sys.stderr)
        return 2

    sources_raw = str(resolve_value(
        args.sources,
        env_lookup(env, "DEPLOY_SOURCES"),
        get_preset_csv_or_list(deploy_set, "sources"),
        "run.sh,build_wheel.sh,README.md,sglang,vision,pytorch",
    ))
    sources = normalize_local_sources(parse_csv(sources_raw))
    if not sources:
        print("No sources specified.", file=sys.stderr)
        return 2

    ssh_extra: list[str] = []
    if args.ssh_key:
        ssh_extra.extend(["-i", args.ssh_key])
    if args.ssh_port:
        ssh_extra.extend(["-p", str(args.ssh_port)])

    exclude_raw = str(resolve_value(
        args.exclude,
        env_lookup(env, "DEPLOY_EXCLUDE"),
        get_preset_csv_or_list(deploy_set, "exclude"),
        ".git,.venv,__pycache__,*.o,*.a,*.so,*.pt,*.bin",
    ))
    excludes = parse_csv(exclude_raw)
    for host in hosts:
        print(f"[deploy] preparing {host}:{remote_dir}")
        mkdir_cmd = f"mkdir -p {shlex.quote(remote_dir)}"
        result = run_cmd(["ssh", *ssh_extra, host, f"bash -lc {shlex.quote(mkdir_cmd)}"], check=False)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return result.returncode

        rsync_cmd = ["rsync", "-az", "--delete"]
        for item in excludes:
            rsync_cmd.extend(["--exclude", item])
        if ssh_extra:
            rsync_cmd.extend(["-e", "ssh " + " ".join(shlex.quote(part) for part in ssh_extra)])
        rsync_cmd.extend(sources)
        rsync_cmd.append(f"{host}:{remote_dir}/")
        print(f"[deploy] syncing to {host}")
        result = run_cmd(rsync_cmd, check=False)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return result.returncode
    print("[deploy] complete")
    return 0


def run_deploy_command(args: argparse.Namespace) -> dict[str, object]:
    """Run :func:`deploy` and capture stdout/stderr for API callers."""
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        rc = deploy(args)
    return {
        "ok": rc == 0,
        "exit_code": rc,
        "stdout": buf_out.getvalue().strip(),
        "stderr": buf_err.getvalue().strip(),
    }


def default_model_presets_path() -> str:
    """Resolve ``model_presets.json`` at the spark-stack-dashboard repo root."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return str(repo_root / "model_presets.json")


def default_deploy_sets_path() -> str:
    """Resolve ``deploy_sets.json`` at the repo root (optional file for named deploy sets)."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return str(repo_root / "deploy_sets.json")


@dataclass(frozen=True)
class MergedPresetLaunch:
    """Fields merged from ``model_presets.json`` for HTTP/API launches."""

    model_path: str
    venv_path: str
    tp: int
    port: int
    sglang_args: list[str]


def merge_preset_launch_fields(
    presets_file: str,
    preset_name: str,
    env: dict[str, str],
    *,
    override_model_path: str | None,
    override_venv_path: str | None,
    override_tp: int | None,
    override_port: int | None,
    extra_sglang_args: list[str],
    preset_sglang_args: list[str] | None = None,
) -> MergedPresetLaunch:
    presets = load_presets(presets_file)
    if preset_name not in presets:
        raise ValueError(
            f"preset '{preset_name}' not found in {presets_file}; "
            f"available: {', '.join(sorted(presets)) or '(none)'}"
        )
    preset = presets[preset_name]
    model_override = (
        override_model_path.strip() if override_model_path and override_model_path.strip() else None
    )
    venv_override = (
        override_venv_path.strip() if override_venv_path and override_venv_path.strip() else None
    )
    model_path = str(
        resolve_value(
            model_override,
            env_lookup(env, "MODEL_PATH"),
            get_preset_string(preset, "model_path"),
            "~/huggingface/Qwen_Qwen3.5-2B",
        )
    )
    venv_path = str(
        resolve_value(
            venv_override,
            env_lookup(env, "VENV_PATH"),
            get_preset_string(preset, "venv_path"),
            "~/.sglang",
        )
    )
    if override_tp is not None:
        tp = int(override_tp)
    else:
        preset_tp = get_preset_int(preset, "tp")
        tp = int(preset_tp) if preset_tp is not None else 1

    server_port = int(
        resolve_value(
            override_port,
            env_lookup(env, "SERVER_PORT"),
            get_preset_int(preset, "port"),
            30000,
        )
    )
    base_preset_sglang = (
        list(preset_sglang_args)
        if preset_sglang_args is not None
        else get_preset_sglang_args(preset)
    )
    merged_args = [
        *base_preset_sglang,
        *shlex.split(env_lookup(env, "SGLANG_EXTRA_ARGS") or ""),
        *extra_sglang_args,
    ]
    if "--served-model-name" not in merged_args:
        merged_args.extend(["--served-model-name", preset_name])

    return MergedPresetLaunch(
        model_path=model_path,
        venv_path=venv_path,
        tp=tp,
        port=server_port,
        sglang_args=merged_args,
    )


def build_dashboard_source_launch_command(
    *,
    presets_file: str,
    preset_name: str,
    env: dict[str, str],
    override_model_path: str | None,
    override_venv_path: str | None,
    override_tp: int | None,
    override_port: int | None,
    extra_sglang_args: list[str],
    preset_sglang_args: list[str] | None = None,
) -> str:
    merged = merge_preset_launch_fields(
        presets_file,
        preset_name,
        env,
        override_model_path=override_model_path,
        override_venv_path=override_venv_path,
        override_tp=override_tp,
        override_port=override_port,
        extra_sglang_args=extra_sglang_args,
        preset_sglang_args=preset_sglang_args,
    )
    nccl_prefix = build_export_prefix(env, _NCCL_ENV_KEYS)
    venv_activate = f"{shell_quote_path_allow_home(merged.venv_path)}/bin/activate"
    model_path_arg = shell_quote_path_allow_home(merged.model_path)
    extra_sglang = " ".join(shlex.quote(arg) for arg in merged.sglang_args)
    launch_cmd = (
        f"{nccl_prefix}if [ ! -f {venv_activate} ]; then "
        f"echo 'Missing venv activate script at {venv_activate}. "
        f"Pass --venv or set VENV_PATH in .env.' >&2; exit 2; fi && "
        f"source {venv_activate} && "
        f"python -m sglang.launch_server --model-path {model_path_arg} "
        f"--tp {merged.tp} --host 0.0.0.0 --port {merged.port}"
    )
    if extra_sglang:
        launch_cmd = f"{launch_cmd} {extra_sglang}"
    return launch_cmd


def launch(args: argparse.Namespace) -> int:
    env = load_env_from_args(args)
    presets_file = env_lookup(env, "MODEL_PRESETS_FILE") or args.presets_file
    preset_name = args.preset or env_lookup(env, "MODEL_PRESET") or ""
    preset: dict[str, object] = {}
    presets: dict[str, dict[str, object]] = {}

    if args.list_presets:
        try:
            presets = load_presets(presets_file)
        except FileNotFoundError:
            print(f"Presets file not found: {presets_file}", file=sys.stderr)
            return 2
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"Failed to parse presets file '{presets_file}': {exc}", file=sys.stderr)
            return 2
        for name in sorted(presets):
            print(name)
        return 0

    if preset_name:
        try:
            presets = load_presets(presets_file)
        except FileNotFoundError:
            print(f"Presets file not found: {presets_file}", file=sys.stderr)
            return 2
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"Failed to parse presets file '{presets_file}': {exc}", file=sys.stderr)
            return 2
        if preset_name not in presets:
            print(
                f"Preset '{preset_name}' not found in {presets_file}. "
                f"Available: {', '.join(sorted(presets)) or '(none)'}",
                file=sys.stderr,
            )
            return 2
        preset = presets[preset_name]

    model_path = str(resolve_value(
        args.model_path,
        env_lookup(env, "MODEL_PATH"),
        get_preset_string(preset, "model_path"),
        "~/huggingface/Qwen_Qwen3.5-2B",
    ))
    venv_path = str(resolve_value(
        args.venv,
        env_lookup(env, "VENV_PATH"),
        get_preset_string(preset, "venv_path"),
        "~/.sglang",
    ))
    if args.mode == "solo":
        tp = int(args.tp) if args.tp is not None else 1
    else:
        tp = resolve_tp(args.tp, env, preset, preset_name)
    server_port = int(resolve_value(
        args.port,
        env_lookup(env, "SERVER_PORT"),
        get_preset_int(preset, "port"),
        30000,
    ))
    master_node = env_get(env, "MASTER_NODE", "")
    master_port = env_get(env, "MASTER_PORT", "20000")
    default_dist_addr = f"{master_node}:{master_port}" if master_node else args.dist_addr
    dist_addr = env_get(env, "DIST_ADDR", default_dist_addr)
    extra_sglang_args = [
        *get_preset_sglang_args(preset),
        *shlex.split(env_lookup(env, "SGLANG_EXTRA_ARGS") or ""),
        *shlex.split(args.sglang_args or ""),
    ]
    if preset_name and "--served-model-name" not in extra_sglang_args:
        extra_sglang_args.extend(["--served-model-name", preset_name])

    nccl_prefix = build_export_prefix(env, _NCCL_ENV_KEYS)

    log_dir = str(
        resolve_value(
            args.log_dir,
            env_lookup(env, "LOG_DIR"),
            None,
            "~/runtime-sglang/logs",
        )
    )
    log_file = str(
        resolve_value(
            args.log_file,
            env_lookup(env, "LOG_FILE"),
            None,
            "sglang_solo.log",
        )
    )

    launch_cmd = args.command
    if not launch_cmd:
        venv_activate = f"{shell_quote_path_allow_home(venv_path)}/bin/activate"
        model_path_arg = shell_quote_path_allow_home(model_path)
        extra_sglang = " ".join(shlex.quote(arg) for arg in extra_sglang_args)
        launch_cmd = (
            f"{nccl_prefix}if [ ! -f {venv_activate} ]; then "
            f"echo 'Missing venv activate script at {venv_activate}. "
            f"Pass --venv or set VENV_PATH in .env.' >&2; exit 2; fi && "
            f"source {venv_activate} && "
            f"python -m sglang.launch_server --model-path {model_path_arg} "
            f"--tp {tp} --host 0.0.0.0 --port {server_port}"
        )
        if extra_sglang:
            launch_cmd = f"{launch_cmd} {extra_sglang}"

    if args.mode == "solo":
        from sglang_common import _cli

        if _cli.VERBOSE:
            print(f"[launch] command: {launch_cmd}")
        if args.host:
            print(f"[launch] remote solo launch on {args.host}")
            result = run_remote(args.host, launch_cmd, check=False)
            if result.stdout:
                print(result.stdout.strip())
            if result.stderr:
                print(result.stderr.strip(), file=sys.stderr)
            return result.returncode
        print("[launch] local solo launch")
        solo_cmd = launch_cmd
        if log_file:
            log_path = shell_quote_path_allow_home(log_file)
            print(f"[launch] writing local logs to {log_file}")
            solo_cmd = f"set -o pipefail && {{ {launch_cmd}; }} 2>&1 | tee -a {log_path}"
        proc = run_shell(solo_cmd, check=False)
        if proc.stdout:
            print(proc.stdout.strip())
        if proc.stderr:
            print(proc.stderr.strip(), file=sys.stderr)
        return proc.returncode

    if not args.hosts:
        hosts = [h for h in [master_node, env_get(env, "WORKER_NODE", "")] if h]
        if not hosts:
            print("Cluster mode requires --hosts or MASTER_NODE/WORKER_NODE in config.", file=sys.stderr)
            return 2
    else:
        hosts = args.hosts

    rc = 0
    for idx, host in enumerate(hosts):
        node_cmd = (
            f"{launch_cmd} "
            f"--dist-init-addr {dist_addr} "
            f"--nnodes {len(hosts)} --node-rank {idx}"
        )
        print(f"[launch] cluster node {idx} on {host}")
        node_cmd_quoted = shlex.quote(node_cmd)
        result = run_remote(
            host,
            f"nohup bash -lc {node_cmd_quoted} > {log_dir}/sglang_node{idx}.log 2>&1 &",
            check=False,
        )
        if result.returncode != 0:
            rc = result.returncode
            if result.stderr:
                print(result.stderr.strip(), file=sys.stderr)
    return rc


def stop(args: argparse.Namespace) -> int:
    env = load_env_from_args(args)
    presets_file = env_lookup(env, "MODEL_PRESETS_FILE") or args.presets_file
    preset_name = args.preset or env_lookup(env, "MODEL_PRESET") or ""
    preset: dict[str, object] = {}
    presets: dict[str, dict[str, object]] = {}

    if preset_name:
        try:
            presets = load_presets(presets_file)
        except FileNotFoundError:
            print(f"Presets file not found: {presets_file}", file=sys.stderr)
            return 2
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"Failed to parse presets file '{presets_file}': {exc}", file=sys.stderr)
            return 2
        if preset_name not in presets:
            print(
                f"Preset '{preset_name}' not found in {presets_file}. "
                f"Available: {', '.join(sorted(presets)) or '(none)'}",
                file=sys.stderr,
            )
            return 2
        preset = presets[preset_name]

    server_port = int(resolve_value(
        args.port,
        env_lookup(env, "SERVER_PORT"),
        get_preset_int(preset, "port"),
        30000,
    ))

    if args.mode == "solo":
        targets = [args.host] if args.host else ["local"]
    else:
        if args.hosts:
            targets = args.hosts
        else:
            targets = [h for h in [env_get(env, "MASTER_NODE", ""), env_get(env, "WORKER_NODE", "")] if h]
            if not targets:
                print(
                    "Cluster mode stop requires --hosts or MASTER_NODE/WORKER_NODE in config.",
                    file=sys.stderr,
                )
                return 2

    stop_cmd = (
        f"pids=$(lsof -tiTCP:{server_port} -sTCP:LISTEN 2>/dev/null || true); "
        "if [ -n \"$pids\" ]; then "
        "echo \"[stop] port pid(s): $pids\"; "
        "kill $pids 2>/dev/null || true; "
        f"sleep {args.grace_sec}; "
        "remaining=\"\"; "
        "for pid in $pids; do if kill -0 \"$pid\" 2>/dev/null; then remaining=\"$remaining $pid\"; fi; done; "
        "if [ -n \"$remaining\" ]; then "
        "echo \"[stop] force-killing pid(s):$remaining\"; "
        "kill -9 $remaining 2>/dev/null || true; "
        "fi; "
        "else "
        "echo \"[stop] no listener found on target port\"; "
        "fi; "
        "pkill -f 'python -m sglang.launch_server' >/dev/null 2>&1 || true"
    )

    rc = 0
    for target in targets:
        label = target
        if target == "local":
            print(f"[stop] local stop on port {server_port}")
            result = run_shell(stop_cmd, check=False)
        else:
            print(f"[stop] remote stop on {target} port {server_port}")
            result = run_remote(target, stop_cmd, check=False)
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(f"[{label}] {result.stderr.strip()}", file=sys.stderr)
        if result.returncode != 0:
            rc = result.returncode
    return rc


def _expand_log_path(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def logs(args: argparse.Namespace) -> int:
    env = load_env_from_args(args)
    log_dir = str(
        resolve_value(
            args.log_dir,
            env_lookup(env, "LOG_DIR"),
            None,
            "~/runtime-sglang/logs",
        )
    )
    log_file = str(
        resolve_value(
            args.log_file,
            env_lookup(env, "LOG_FILE"),
            None,
            "sglang_solo.log",
        )
    )
    lines = max(1, int(args.lines))
    viewer = "head" if args.from_start else "tail"

    if args.mode == "solo":
        path = _expand_log_path(log_file)
        cmd_parts = [viewer, "-n", str(lines), path]
        if args.host:
            path_q = shell_quote_path_allow_home(log_file)
            inner = f"{viewer} -n {lines} {path_q} 2>&1"
            print(f"[logs] solo on {args.host}: {log_file}", file=sys.stderr)
            result = run_remote(args.host, inner, check=False)
        else:
            print(f"[logs] solo local: {path}", file=sys.stderr)
            if not os.path.isfile(path):
                print(f"[logs] file not found: {path}", file=sys.stderr)
                return 1
            result = subprocess.run(cmd_parts, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout.rstrip("\n"))
        if result.stderr:
            print(result.stderr.rstrip("\n"), file=sys.stderr)
        return 0 if result.returncode == 0 else result.returncode

    if not args.hosts:
        hosts = [h for h in [env_get(env, "MASTER_NODE", ""), env_get(env, "WORKER_NODE", "")] if h]
        if not hosts:
            print(
                "Cluster logs require --hosts or MASTER_NODE/WORKER_NODE in config.",
                file=sys.stderr,
            )
            return 2
    else:
        hosts = args.hosts

    if args.node_rank is not None:
        rank = args.node_rank
    elif args.role == "head":
        rank = 0
    else:
        rank = 1

    if rank < 0 or rank >= len(hosts):
        print(
            f"[logs] invalid node rank {rank} for {len(hosts)} host(s); "
            f"use --role head|worker or --node-rank 0..{len(hosts) - 1}.",
            file=sys.stderr,
        )
        return 2

    remote_log = f"{log_dir.rstrip('/')}/sglang_node{rank}.log"
    path_q = shell_quote_path_allow_home(remote_log)
    inner = f"{viewer} -n {lines} {path_q} 2>&1"
    host = hosts[rank]
    role_label = "head" if rank == 0 else f"worker (rank {rank})"
    print(f"[logs] cluster {role_label} on {host}: {remote_log}", file=sys.stderr)
    result = run_remote(host, inner, check=False)
    if result.stdout:
        print(result.stdout.rstrip("\n"))
    if result.stderr:
        print(result.stderr.rstrip("\n"), file=sys.stderr)
    return 0 if result.returncode == 0 else result.returncode


def scan(args: argparse.Namespace) -> int:
    env = load_env_from_args(args)
    presets_file = env_lookup(env, "MODEL_PRESETS_FILE") or args.presets_file
    preset_name = args.preset or env_lookup(env, "MODEL_PRESET") or ""
    preset: dict[str, object] = {}
    presets: dict[str, dict[str, object]] = {}

    if preset_name:
        try:
            presets = load_presets(presets_file)
        except FileNotFoundError:
            print(f"Presets file not found: {presets_file}", file=sys.stderr)
            return 2
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"Failed to parse presets file '{presets_file}': {exc}", file=sys.stderr)
            return 2
        if preset_name not in presets:
            print(
                f"Preset '{preset_name}' not found in {presets_file}. "
                f"Available: {', '.join(sorted(presets)) or '(none)'}",
                file=sys.stderr,
            )
            return 2
        preset = presets[preset_name]

    server_port = int(
        resolve_value(
            args.port,
            env_lookup(env, "SERVER_PORT"),
            get_preset_int(preset, "port"),
            30000,
        )
    )
    bind_host = (args.bind_host or "127.0.0.1").strip()
    base_url = (args.base_url or "").strip()
    if not base_url:
        base_url = f"http://{bind_host}:{server_port}"

    if args.host:
        remote_url = f"http://127.0.0.1:{server_port}"
        remote_script = build_remote_scan_script(
            remote_url, args.api_key, args.timeout_sec, args.readiness
        )
        result = run_remote(args.host, remote_script, check=False)
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        if result.returncode != 0:
            return result.returncode
        return 0

    payload = collect_running_server_scan(
        base_url,
        api_key=args.api_key,
        timeout_sec=args.timeout_sec,
        readiness=args.readiness,
    )
    print(json.dumps(payload, indent=2))
    return 0


def benchmark(args: argparse.Namespace) -> int:
    result = run_benchmark(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        requests=args.requests,
        timeout_sec=args.timeout_sec,
    )
    if result is None:
        print("No successful benchmark requests.", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def measure(args: argparse.Namespace) -> int:
    env = load_env_from_args(args)
    gpu_cmd = (
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw "
        "--format=csv,noheader,nounits"
    )
    sys_cmd = (
        "python - <<'PY'\n"
        "import json, os\n"
        "load = os.getloadavg()\n"
        "print(json.dumps({'load_1m': load[0], 'load_5m': load[1], 'load_15m': load[2]}))\n"
        "PY"
    )
    command = f"{gpu_cmd} && {sys_cmd}"

    if args.hosts:
        targets = args.hosts
    else:
        config_targets = [h for h in [env.get("MASTER_NODE"), env.get("WORKER_NODE")] if h]
        targets = config_targets if config_targets else ["local"]
    output: dict[str, dict[str, str]] = {}
    for host in targets:
        if host == "local":
            result = run_shell(command, check=False)
        else:
            result = run_remote(host, command, check=False)
        output[host] = {
            "exit_code": str(result.returncode),
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    print(json.dumps(output, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DGX Spark runtime operations for custom sglang stack")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug output")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_deploy = sub.add_parser("deploy", help="Deploy runtime to remote host(s)")
    p_deploy.add_argument("--hosts", nargs="+", help="Remote hosts, e.g. spark-02")
    p_deploy.add_argument("--set", default="", help="Named deploy set from deploy sets file")
    p_deploy.add_argument(
        "--deploy-sets-file",
        default="deploy_sets.json",
        help="Path to JSON deploy sets file",
    )
    p_deploy.add_argument("--list-sets", action="store_true", help="List deploy sets and exit")
    p_deploy.add_argument(
        "--sources",
        default=None,
        help="Comma-separated local paths to sync",
    )
    p_deploy.add_argument("--remote-dir", default=None, help="Remote destination directory")
    p_deploy.add_argument(
        "--exclude",
        default=None,
        help="Comma-separated rsync exclude patterns",
    )
    p_deploy.add_argument("--ssh-key", default="", help="Optional SSH private key path")
    p_deploy.add_argument("--ssh-port", type=int, default=22, help="SSH port")
    p_deploy.add_argument("--env-file", default="", help="Optional path to .env")
    p_deploy.set_defaults(func=deploy)

    p_launch = sub.add_parser("launch", help="Launch sglang runtime")
    p_launch.add_argument("--mode", choices=["solo", "cluster"], default="solo")
    p_launch.add_argument("--host", default="", help="Remote host for solo mode")
    p_launch.add_argument("--hosts", nargs="*", help="Remote hosts for cluster mode")
    p_launch.add_argument("--venv", default=None, help="Python virtual env path")
    p_launch.add_argument("--model-path", default=None)
    p_launch.add_argument("--tp", type=int, default=None)
    p_launch.add_argument("--port", type=int, default=None)
    p_launch.add_argument("--dist-addr", default="spark-01:20000", help="Master addr:port for cluster")
    p_launch.add_argument(
        "--log-dir",
        default=None,
        help="Remote log directory for cluster mode (default: ~/runtime-sglang/logs; env: LOG_DIR)",
    )
    p_launch.add_argument(
        "--log-file",
        default=None,
        help="Local log file for solo mode (default: sglang_solo.log; env: LOG_FILE)",
    )
    p_launch.add_argument("--preset", default="", help="Preset name from presets file")
    p_launch.add_argument("--presets-file", default="model_presets.json", help="Path to JSON presets file")
    p_launch.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    p_launch.add_argument(
        "--sglang-args",
        default="",
        help="Extra arguments appended to sglang.launch_server, e.g. '--enable-metrics --context-length 32768'",
    )
    p_launch.add_argument("--command", default="", help="Optional full launch command override")
    p_launch.add_argument("--env-file", default="", help="Optional path to .env")
    p_launch.set_defaults(func=launch)

    p_stop = sub.add_parser("stop", help="Stop launched sglang runtime")
    p_stop.add_argument("--mode", choices=["solo", "cluster"], default="solo")
    p_stop.add_argument("--host", default="", help="Remote host for solo mode")
    p_stop.add_argument("--hosts", nargs="*", help="Remote hosts for cluster mode")
    p_stop.add_argument("--port", type=int, default=None, help="Server port to stop (default from config/preset)")
    p_stop.add_argument(
        "--grace-sec",
        type=int,
        default=5,
        help="Grace period before force-kill for processes found on target port",
    )
    p_stop.add_argument("--preset", default="", help="Preset name from presets file")
    p_stop.add_argument("--presets-file", default="model_presets.json", help="Path to JSON presets file")
    p_stop.add_argument("--env-file", default="", help="Optional path to .env")
    p_stop.set_defaults(func=stop)

    p_logs = sub.add_parser(
        "logs",
        help="Show log output (solo: LOG_FILE; cluster: LOG_DIR/sglang_node<N>.log on head/worker)",
    )
    p_logs.add_argument("--mode", choices=["solo", "cluster"], default="solo")
    p_logs.add_argument(
        "--host",
        default="",
        help="Solo mode: read log file on this host via SSH (paths from --log-file / LOG_FILE)",
    )
    p_logs.add_argument(
        "--hosts",
        nargs="*",
        help="Cluster mode: host list (order must match launch); default from MASTER_NODE, WORKER_NODE",
    )
    p_logs.add_argument(
        "--role",
        choices=["head", "worker"],
        default="head",
        help="Cluster mode: head is node 0, worker is node 1 (ignored if --node-rank is set)",
    )
    p_logs.add_argument(
        "--node-rank",
        type=int,
        default=None,
        help="Cluster mode: explicit node index matching launch --hosts order and log filename",
    )
    p_logs.add_argument(
        "--log-dir",
        default=None,
        help="Cluster log directory (default: ~/runtime-sglang/logs; env: LOG_DIR)",
    )
    p_logs.add_argument(
        "--log-file",
        default=None,
        help="Solo log file (default: sglang_solo.log; env: LOG_FILE)",
    )
    p_logs.add_argument(
        "-n",
        "--lines",
        type=int,
        default=80,
        metavar="N",
        help="Line count for tail (default) or head (with --from-start)",
    )
    p_logs.add_argument(
        "--from-start",
        action="store_true",
        help="Show the first N lines (head) instead of the last N (tail)",
    )
    p_logs.add_argument("--env-file", default="", help="Optional path to .env")
    p_logs.set_defaults(func=logs)

    p_scan = sub.add_parser(
        "scan",
        aliases=["refresh"],
        help="Probe running server: /v1/models, /health, /get_server_info (alias: refresh)",
    )
    p_scan.add_argument(
        "--base-url",
        default="",
        help="HTTP root of the server (default: http://<bind-host>:<port> from preset/env)",
    )
    p_scan.add_argument(
        "--bind-host",
        default="127.0.0.1",
        help="Host used with --port when --base-url is omitted (ignored when --base-url is set)",
    )
    p_scan.add_argument(
        "--host",
        default="",
        help="If set, run probes over SSH on this host against http://127.0.0.1:<port> (server on that node)",
    )
    p_scan.add_argument("--port", type=int, default=None, help="HTTP port (default from preset/env, else 30000)")
    p_scan.add_argument("--preset", default="", help="Preset name for default port")
    p_scan.add_argument("--presets-file", default="model_presets.json", help="Path to JSON presets file")
    p_scan.add_argument("--api-key", default="EMPTY", help="Bearer token for endpoints that require auth")
    p_scan.add_argument("--timeout-sec", type=int, default=30, help="Per-request timeout")
    p_scan.add_argument(
        "--readiness",
        action="store_true",
        help="Also call /health_generate (may run a short generation; slower than /health)",
    )
    p_scan.add_argument("--env-file", default="", help="Optional path to .env")
    p_scan.set_defaults(func=scan)

    p_bench = sub.add_parser("benchmark", help="Run API benchmark")
    p_bench.add_argument("--base-url", default="http://127.0.0.1:30000")
    p_bench.add_argument("--api-key", default="EMPTY")
    p_bench.add_argument("--model", default="default")
    p_bench.add_argument("--prompt", default="Write a short haiku about distributed inference.")
    p_bench.add_argument("--max-tokens", type=int, default=64)
    p_bench.add_argument("--requests", type=int, default=20)
    p_bench.add_argument("--timeout-sec", type=int, default=120)
    p_bench.set_defaults(func=benchmark)

    p_measure = sub.add_parser("measure", help="Capture utilization snapshots")
    p_measure.add_argument("--hosts", nargs="*", help="If omitted, measure local node only")
    p_measure.add_argument("--env-file", default="", help="Optional path to .env")
    p_measure.set_defaults(func=measure)

    return parser


def main() -> int:
    from sglang_common import _cli

    parser = build_parser()
    args = parser.parse_args()
    _cli.VERBOSE = args.verbose
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
