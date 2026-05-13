"""Launch helpers — NCCL env export, shell path quoting."""

from __future__ import annotations

import shlex


_NCCL_ENV_KEYS = [
    "NCCL_IB_DISABLE",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TIMEOUT",
    "NCCL_IB_RETRY_CNT",
    "NCCL_IB_SL",
    "NCCL_IB_TC",
    "NCCL_IB_QPS_PER_CONNECTION",
    "NCCL_IB_CUDA_SUPPORT",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_NET_GDR_READ",
    "NCCL_P2P_DISABLE",
    "NCCL_IB_HCA",
    "NCCL_PROTO",
    "NCCL_ALGO",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_IFNAME",
    "NCCL_DEBUG",
    "CUDA_GRAPHS",
    "SGLANG_DISABLE_TORCHVISION",
]


def build_export_prefix(env: dict[str, str], keys: list[str]) -> str:
    pairs = []
    for key in keys:
        if key in env:
            pairs.append(f"export {key}={shlex.quote(env[key])}")
    if not pairs:
        return ""
    return " && ".join(pairs) + " && "


def shell_quote_path_allow_home(path: str) -> str:
    """Quote shell path while preserving remote HOME expansion for ~/."""
    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        return "$HOME/" + shlex.quote(path[2:])
    return shlex.quote(path)
