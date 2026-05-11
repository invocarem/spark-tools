#!/usr/bin/env python3
"""Print NCCL version details from multiple sources."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from typing import Optional, Tuple


def parse_nccl_int(version_int: int) -> str:
    """Convert NCCL integer version into major.minor.patch."""
    major = version_int // 1000
    minor = (version_int % 1000) // 100
    patch = version_int % 100
    return f"{major}.{minor}.{patch}"


def nccl_from_torch() -> Tuple[Optional[str], Optional[str]]:
    """Return NCCL version from torch APIs: (version, error)."""
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        return None, f"torch import failed: {exc}"

    # Prefer explicit NCCL API when available.
    try:
        version_obj = torch.cuda.nccl.version()  # pylint: disable=no-member
        if isinstance(version_obj, tuple):
            return ".".join(str(x) for x in version_obj), None
        if isinstance(version_obj, int):
            return parse_nccl_int(version_obj), None
        return str(version_obj), None
    except Exception:
        pass

    # Fallback to torch's compile-time metadata.
    try:
        nccl_attr = getattr(torch.cuda, "nccl", None)
        if nccl_attr is not None and hasattr(nccl_attr, "version"):
            value = nccl_attr.version()
            if isinstance(value, tuple):
                return ".".join(str(x) for x in value), None
            if isinstance(value, int):
                return parse_nccl_int(value), None
            return str(value), None
    except Exception:
        pass

    return None, "torch loaded, but no NCCL version API is accessible"


def nccl_from_library() -> Tuple[Optional[str], Optional[str]]:
    """Return NCCL runtime version via ncclGetVersion: (version, error)."""
    candidates = [
        os.environ.get("NCCL_LIB"),
        "libnccl.so",
        "libnccl.so.2",
        ctypes.util.find_library("nccl"),
    ]
    candidates = [c for c in candidates if c]

    load_errors = []
    for lib_name in candidates:
        try:
            lib = ctypes.CDLL(lib_name)
            break
        except OSError as exc:
            load_errors.append(f"{lib_name}: {exc}")
    else:
        return None, "could not load NCCL library; attempts: " + "; ".join(load_errors)

    try:
        get_version = lib.ncclGetVersion
        get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_version.restype = ctypes.c_int
    except AttributeError:
        return None, "loaded NCCL library but ncclGetVersion symbol not found"

    version_out = ctypes.c_int(0)
    rc = get_version(ctypes.byref(version_out))
    if rc != 0:
        return None, f"ncclGetVersion returned non-zero status: {rc}"

    return parse_nccl_int(version_out.value), None


def main() -> int:
    print("=== NCCL Version Check ===")

    torch_version, torch_err = nccl_from_torch()
    if torch_version:
        print(f"torch NCCL version: {torch_version}")
    else:
        print(f"torch NCCL version: unavailable ({torch_err})")

    lib_version, lib_err = nccl_from_library()
    if lib_version:
        print(f"libnccl runtime version: {lib_version}")
    else:
        print(f"libnccl runtime version: unavailable ({lib_err})")

    # Non-zero exit only if both methods failed.
    return 0 if (torch_version or lib_version) else 1


if __name__ == "__main__":
    sys.exit(main())
