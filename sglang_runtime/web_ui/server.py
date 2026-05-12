"""Compatibility shim: Stack UI lives under ``stack_ui/backend``.

Run from ``sglang_runtime`` (same as before)::

    pip install -r ../stack_ui/backend/requirements.txt
    uvicorn web_ui.server:app --host 127.0.0.1 --port 8765

See ``stack_ui/README.md`` for the recommended layout (backend + Vite frontend).
"""
from __future__ import annotations

import sys
from pathlib import Path

_backend = Path(__file__).resolve().parents[2] / "stack_ui" / "backend"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from stack_ui_server import app  # noqa: E402

__all__ = ["app"]
