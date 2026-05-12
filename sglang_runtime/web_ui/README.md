The control panel moved to **`stack_ui/`** at the repository root (`backend` + `frontend`).

- **Docs**: `stack_ui/README.md`
- **Shim**: `web_ui/server.py` still exposes `app` for `uvicorn web_ui.server:app` when the current working directory is `sglang_runtime/`.
