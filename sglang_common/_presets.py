"""Preset JSON loading and value resolution."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def load_presets(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("presets file must be a JSON object keyed by preset name")
    normalized: dict[str, dict[str, Any]] = {}
    for name, config in data.items():
        if not isinstance(name, str):
            raise ValueError("preset names must be strings")
        if not isinstance(config, dict):
            raise ValueError(f"preset '{name}' must map to an object")
        normalized[name] = config
    return normalized


def get_preset_string(preset: dict[str, Any], key: str) -> str | None:
    value = preset.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"preset key '{key}' must be a string")
    return value


def get_preset_int(preset: dict[str, Any], key: str) -> int | None:
    value = preset.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"preset key '{key}' must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError(f"preset key '{key}' must be an integer")


def get_preset_sglang_args(preset: dict[str, Any]) -> list[str]:
    value = preset.get("sglang_args")
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("preset key 'sglang_args' must be a list")
    args: list[str] = []
    for item in value:
        if isinstance(item, str):
            args.append(item)
            continue
        if isinstance(item, (dict, list)):
            args.append(json.dumps(item, separators=(",", ":")))
            continue
        if isinstance(item, (int, float, bool)):
            args.append(str(item))
            continue
        raise ValueError(
            "preset key 'sglang_args' items must be string/number/bool/object/array"
        )
    return args


def get_preset_csv_or_list(preset: dict[str, Any], key: str) -> str | None:
    value = preset.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"preset key '{key}' list must contain strings only")
            items.append(item)
        return ",".join(items)
    raise ValueError(f"preset key '{key}' must be a string or list of strings")


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def normalize_local_sources(items: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in items:
        expanded = os.path.expandvars(os.path.expanduser(item))
        normalized.append(expanded)
    return normalized


def resolve_value(
    cli_value: str | int | None,
    env_value: str | None,
    preset_value: str | int | None,
    default_value: str | int,
) -> str | int:
    if cli_value is not None:
        return cli_value
    if env_value is not None:
        return env_value
    if preset_value is not None:
        return preset_value
    return default_value


def resolve_tp(
    args_tp: int | None,
    env: dict[str, str],
    preset: dict[str, Any],
    preset_name: str,
) -> int:
    """Pick tensor-parallel width for cluster launch."""
    from sglang_common._env import env_lookup

    if args_tp is not None:
        return int(args_tp)
    preset_tp = get_preset_int(preset, "tp")
    env_tp = env_lookup(env, "TP_SIZE")
    if preset_name.strip():
        if preset_tp is not None:
            return int(preset_tp)
        if env_tp is not None:
            return int(env_tp)
        return 1
    if env_tp is not None:
        return int(env_tp)
    if preset_tp is not None:
        return int(preset_tp)
    return 1
