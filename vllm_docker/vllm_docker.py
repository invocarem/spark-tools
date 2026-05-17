#!/usr/bin/env python3
"""Manage vLLM inference via Docker across DGX Spark–style nodes.

Mirrors ``sglang_docker`` (launch/stop/logs/scan/benchmark/measure plus ``pull``)
but runs ``vllm serve`` inside the container. Presets reuse the same JSON shape
as sglang (``model_path``, ``image``, ``tp``, ``port``, ``sglang_args`` as extra
``vllm serve`` flags).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
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
    get_preset_int,
    get_preset_sglang_args,
    get_preset_string,
    load_env_from_args,
    load_presets,
    resolve_tp,
    resolve_value,
    run_benchmark,
    run_cmd,
    run_remote,
    run_shell,
    shell_quote_path_allow_home,
)


def _container_name(preset_name: str) -> str:
    return f"vllm-{preset_name}"


def default_model_presets_path() -> str:
    here = Path(__file__).resolve().parent
    return str(here / "model_presets.json")


def pull(args: argparse.Namespace) -> int:
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

    image = str(
        resolve_value(
            args.image,
            env_lookup(env, "DOCKER_IMAGE"),
            get_preset_string(preset, "image"),
            "vllm/vllm-openai:latest",
        )
    )

    if args.hosts:
        targets = args.hosts
    else:
        config_targets = [h for h in [env_get(env, "MASTER_NODE", ""), env_get(env, "WORKER_NODE", "")] if h]
        targets = config_targets if config_targets else ["local"]

    rc = 0
    for target in targets:
        if target == "local":
            print(f"[pull] local: {image}")
            result = run_cmd(["docker", "pull", image], check=False)
        else:
            print(f"[pull] {target}: {image}")
            result = run_remote(target, f"docker pull {image}", check=False)
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        if result.returncode != 0:
            rc = result.returncode
    return rc


def _build_docker_launch_command(
    image: str,
    model_path: str,
    tp: int,
    port: int,
    vllm_args: list[str],
    env: dict[str, str],
    preset_name: str,
    log_dir: str,
    dist_args: str,
) -> str:
    """Build the docker run command for ``vllm serve``."""
    container = _container_name(preset_name) if preset_name else "vllm-server"
    expanded_model = os.path.expandvars(os.path.expanduser(model_path))
    expanded_log = os.path.expandvars(os.path.expanduser(log_dir))

    nccl_prefix = build_export_prefix(env, _NCCL_ENV_KEYS)

    # Build -e flags for NCCL env
    env_flags = []
    for key in _NCCL_ENV_KEYS:
        if key in env:
            env_flags.extend(["-e", key])

    extra_vllm = " ".join(shlex.quote(arg) for arg in vllm_args)

    cmd = (
        f"{nccl_prefix}docker run -d --name {container} --gpus all --network host "
        f"-v {shlex.quote(expanded_model)}:{shlex.quote(expanded_model)}:ro "
        f"-v {shlex.quote(expanded_log)}:{shlex.quote(expanded_log)} "
        f"{' '.join(env_flags)} "
        f"{shlex.quote(image)} "
        f"vllm serve {shell_quote_path_allow_home(model_path)} "
        f"--host 0.0.0.0 --port {port} --tensor-parallel-size {tp}"
    )
    if extra_vllm:
        cmd = f"{cmd} {extra_vllm}"
    if dist_args:
        cmd = f"{cmd} {dist_args}"

    return cmd


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

    image = str(resolve_value(
        args.image,
        env_lookup(env, "DOCKER_IMAGE"),
        get_preset_string(preset, "image"),
        "vllm/vllm-openai:latest",
    ))
    model_path = str(resolve_value(
        args.model_path,
        env_lookup(env, "MODEL_PATH"),
        get_preset_string(preset, "model_path"),
        "~/huggingface/Qwen_Qwen3.5-2B",
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
    log_dir = str(
        resolve_value(
            args.log_dir,
            env_lookup(env, "LOG_DIR"),
            None,
            "~/vllm-docker-logs",
        )
    )

    extra_vllm_args = [
        *get_preset_sglang_args(preset),
        *shlex.split(env_lookup(env, "VLLM_EXTRA_ARGS") or ""),
        *shlex.split(args.vllm_args or ""),
    ]
    if preset_name and "--served-model-name" not in extra_vllm_args:
        extra_vllm_args.extend(["--served-model-name", preset_name])

    master_node = env_get(env, "MASTER_NODE", "")

    launch_cmd = args.command
    if not launch_cmd:
        # vLLM multi-node is not wired like sglang; use ``--command`` or preset args for Ray, etc.
        dist_args = ""

        launch_cmd = _build_docker_launch_command(
            image=image,
            model_path=model_path,
            tp=tp,
            port=server_port,
            vllm_args=extra_vllm_args,
            env=env,
            preset_name=preset_name,
            log_dir=log_dir,
            dist_args=dist_args,
        )

    if args.mode == "solo":
        if args.host:
            print(f"[launch] remote solo launch on {args.host}")
            # Remove container if stale, then launch
            cname = _container_name(preset_name) if preset_name else "vllm-server"
            result = run_remote(
                args.host,
                f"docker rm -f {cname} 2>/dev/null || true; bash -lc {shlex.quote(launch_cmd)}",
                check=False,
            )
        else:
            print("[launch] local solo launch")
            cname = _container_name(preset_name) if preset_name else "vllm-server"
            result = run_shell(
                f"docker rm -f {cname} 2>/dev/null || true && bash -lc {shlex.quote(launch_cmd)}",
                check=False,
            )
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        return result.returncode

    # Cluster mode
    if not args.hosts:
        hosts = [h for h in [master_node, env_get(env, "WORKER_NODE", "")] if h]
        if not hosts:
            print("Cluster mode requires --hosts or MASTER_NODE/WORKER_NODE in config.", file=sys.stderr)
            return 2
    else:
        hosts = args.hosts

    rc = 0
    for idx, host in enumerate(hosts):
        suffix = f"{preset_name}-node{idx}" if preset_name else f"node{idx}"
        cname = _container_name(suffix)
        node_cmd = _build_docker_launch_command(
            image=image,
            model_path=model_path,
            tp=tp,
            port=server_port,
            vllm_args=extra_vllm_args,
            env=env,
            preset_name=suffix,
            log_dir=log_dir,
            dist_args="",
        )
        print(f"[launch] cluster node {idx} on {host}")
        result = run_remote(
            host,
            f"docker rm -f {cname} 2>/dev/null || true; bash -lc {shlex.quote(node_cmd)}",
            check=False,
        )
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        if result.returncode != 0:
            rc = result.returncode
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

    if args.mode == "solo":
        cname = _container_name(preset_name) if preset_name else "vllm-server"
        targets = [(args.host, cname)] if args.host else [("local", cname)]
    else:
        if args.hosts:
            hosts = args.hosts
        else:
            hosts = [h for h in [env_get(env, "MASTER_NODE", ""), env_get(env, "WORKER_NODE", "")] if h]
            if not hosts:
                print(
                    "Cluster mode stop requires --hosts or MASTER_NODE/WORKER_NODE in config.",
                    file=sys.stderr,
                )
                return 2
        if preset_name:
            targets = [(h, _container_name(f"{preset_name}-node{idx}")) for idx, h in enumerate(hosts)]
        else:
            targets = [(h, _container_name(f"node{idx}")) for idx, h in enumerate(hosts)]

    stop_cmd = (
        f"if docker ps -a -f name={shlex.quote('{{name}}')} --format '{{{{.Names}}}}' | grep -q .; then "
        f"docker stop {{name}} 2>/dev/null || true; "
        f"sleep {args.grace_sec}; "
        f"docker rm {{name}} 2>/dev/null || true; "
        "echo '[stop] container removed'; "
        "else "
        "echo '[stop] no container found'; "
        "fi"
    )

    rc = 0
    for target, cname in targets:
        cmd = stop_cmd.replace("{name}", cname)
        if target == "local":
            print(f"[stop] local stop container {cname}")
            result = run_shell(cmd, check=False)
        else:
            print(f"[stop] remote stop on {target} container {cname}")
            result = run_remote(target, cmd, check=False)
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(f"[{target}] {result.stderr.strip()}", file=sys.stderr)
        if result.returncode != 0:
            rc = result.returncode
    return rc


def logs(args: argparse.Namespace) -> int:
    env = load_env_from_args(args)
    lines = max(1, int(args.lines))
    preset_name = args.preset or env_lookup(env, "MODEL_PRESET") or ""

    if args.mode == "solo":
        cname = _container_name(preset_name) if preset_name else "vllm-server"
        if args.host:
            print(f"[logs] solo on {args.host}: {cname}", file=sys.stderr)
            inner = f"docker logs --tail {lines} {cname} 2>&1"
            result = run_remote(args.host, inner, check=False)
        else:
            print(f"[logs] solo local: {cname}", file=sys.stderr)
            result = run_cmd(["docker", "logs", "--tail", str(lines), cname], check=False)
        if result.stdout:
            print(result.stdout.rstrip("\n"))
        if result.stderr:
            print(result.stderr.rstrip("\n"), file=sys.stderr)
        return 0 if result.returncode == 0 else result.returncode

    # Cluster mode
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

    if preset_name:
        cname = _container_name(f"{preset_name}-node{rank}")
    else:
        cname = _container_name(f"node{rank}")
    host = hosts[rank]
    role_label = "head" if rank == 0 else f"worker (rank {rank})"
    print(f"[logs] cluster {role_label} on {host}: {cname}", file=sys.stderr)
    inner = f"docker logs --tail {lines} {cname} 2>&1"
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
    parser = argparse.ArgumentParser(description="DGX Spark vLLM runtime via Docker")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug output")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_pull = sub.add_parser("pull", help="Pull Docker image on local or remote hosts")
    p_pull.add_argument("--image", default=None, help="Docker image name:tag")
    p_pull.add_argument("--preset", default="", help="Preset name from presets file")
    p_pull.add_argument("--presets-file", default="model_presets.json", help="Path to JSON presets file")
    p_pull.add_argument("--hosts", nargs="+", help="Remote hosts")
    p_pull.add_argument("--env-file", default="", help="Optional path to .env")
    p_pull.set_defaults(func=pull)

    p_launch = sub.add_parser("launch", help="Launch vLLM via Docker")
    p_launch.add_argument("--mode", choices=["solo", "cluster"], default="solo")
    p_launch.add_argument("--host", default="", help="Remote host for solo mode")
    p_launch.add_argument("--hosts", nargs="*", help="Remote hosts for cluster mode")
    p_launch.add_argument("--image", default=None, help="Docker image name:tag")
    p_launch.add_argument("--model-path", default=None)
    p_launch.add_argument("--tp", type=int, default=None)
    p_launch.add_argument("--port", type=int, default=None)
    p_launch.add_argument("--dist-addr", default="spark-01:20000", help="Master addr:port for cluster")
    p_launch.add_argument(
        "--log-dir",
        default=None,
        help="Log directory (default: ~/vllm-docker-logs; env: LOG_DIR)",
    )
    p_launch.add_argument("--preset", default="", help="Preset name from presets file")
    p_launch.add_argument("--presets-file", default=default_model_presets_path(), help="Path to JSON presets file")
    p_launch.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    p_launch.add_argument(
        "--vllm-args",
        default="",
        help="Extra arguments appended to vllm serve (preset sglang_args + env VLLM_EXTRA_ARGS first)",
    )
    p_launch.add_argument("--command", default="", help="Optional full launch command override")
    p_launch.add_argument("--env-file", default="", help="Optional path to .env")
    p_launch.set_defaults(func=launch)

    p_stop = sub.add_parser("stop", help="Stop vLLM Docker container(s)")
    p_stop.add_argument("--mode", choices=["solo", "cluster"], default="solo")
    p_stop.add_argument("--host", default="", help="Remote host for solo mode")
    p_stop.add_argument("--hosts", nargs="*", help="Remote hosts for cluster mode")
    p_stop.add_argument("--preset", default="", help="Preset name from presets file")
    p_stop.add_argument("--presets-file", default="model_presets.json", help="Path to JSON presets file")
    p_stop.add_argument(
        "--grace-sec",
        type=int,
        default=5,
        help="Grace period before forced removal",
    )
    p_stop.add_argument("--env-file", default="", help="Optional path to .env")
    p_stop.set_defaults(func=stop)

    p_logs = sub.add_parser("logs", help="Show Docker container logs")
    p_logs.add_argument("--mode", choices=["solo", "cluster"], default="solo")
    p_logs.add_argument("--host", default="", help="Remote host for solo mode")
    p_logs.add_argument("--hosts", nargs="*", help="Cluster hosts")
    p_logs.add_argument("--preset", default="", help="Preset name for container name")
    p_logs.add_argument(
        "--role",
        choices=["head", "worker"],
        default="head",
        help="Cluster mode: head is node 0, worker is node 1",
    )
    p_logs.add_argument("--node-rank", type=int, default=None, help="Explicit node index")
    p_logs.add_argument("-n", "--lines", type=int, default=80, metavar="N")
    p_logs.add_argument("--env-file", default="", help="Optional path to .env")
    p_logs.set_defaults(func=logs)

    p_scan = sub.add_parser(
        "scan",
        aliases=["refresh"],
        help="Probe running server",
    )
    p_scan.add_argument("--base-url", default="")
    p_scan.add_argument("--bind-host", default="127.0.0.1")
    p_scan.add_argument("--host", default="", help="SSH host for remote probe")
    p_scan.add_argument("--port", type=int, default=None)
    p_scan.add_argument("--preset", default="", help="Preset name for default port")
    p_scan.add_argument("--presets-file", default="model_presets.json", help="Path to JSON presets file")
    p_scan.add_argument("--api-key", default="EMPTY")
    p_scan.add_argument("--timeout-sec", type=int, default=30)
    p_scan.add_argument("--readiness", action="store_true")
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
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
