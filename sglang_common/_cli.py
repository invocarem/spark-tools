"""Shell and SSH command execution."""

from __future__ import annotations

import shlex
import subprocess
import sys

VERBOSE = False


def debug_log(message: str) -> None:
    if VERBOSE:
        print(f"[verbose] {message}", file=sys.stderr)


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_cmd(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    debug_log(f"running command: {format_command(command)}")
    result = subprocess.run(command, text=True, check=check, capture_output=True)
    debug_log(f"command exit code: {result.returncode}")
    if result.stdout.strip():
        debug_log(f"stdout:\n{result.stdout.strip()}")
    if result.stderr.strip():
        debug_log(f"stderr:\n{result.stderr.strip()}")
    return result


def run_shell(command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_cmd(["bash", "-lc", command], check=check)


def run_remote(host: str, remote_command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    quoted = shlex.quote(remote_command)
    return run_cmd(["ssh", host, f"bash -lc {quoted}"], check=check)
