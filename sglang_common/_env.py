"""Environment file loading and helper functions."""

from __future__ import annotations

import argparse
import os


def load_dotenv(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key:
                values[key] = val
    return values


def load_env_from_args(args: argparse.Namespace) -> dict[str, str]:
    loaded: dict[str, str] = {}
    env_file = getattr(args, "env_file", "")
    if env_file:
        loaded.update(load_dotenv(env_file))
        return loaded

    default_env_file = ".env"
    if os.path.isfile(default_env_file):
        loaded.update(load_dotenv(default_env_file))
    return loaded


def env_get(env: dict[str, str], key: str, default: str) -> str:
    return env.get(key, os.environ.get(key, default))


def env_lookup(env: dict[str, str], key: str) -> str | None:
    value = env.get(key, os.environ.get(key))
    if value is None or value == "":
        return None
    return value
