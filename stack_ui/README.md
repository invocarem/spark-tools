# Stack UI

Web console for **sglang_runtime** (presets, launch/stop/logs, scan). Intended to grow with Docker SGLang and vLLM stacks.

## Layout

- `backend/` — FastAPI (`stack_ui_server.py`), same HTTP API as the former `sglang_runtime/web_ui`.
- `frontend/` — Vite + React + TypeScript; proxies `/api` to the backend in dev.

## Backend

From the repository root (use a **venv** on PEP 668 systems):

```bash
python3 -m venv .venv-stack-ui
. .venv-stack-ui/bin/activate
pip install -r stack_ui/backend/requirements.txt
cd stack_ui/backend
uvicorn stack_ui_server:app --host 127.0.0.1 --port 8765
```

Or reuse an existing env that already has FastAPI (for example `sglang_runtime/.venv-webui` if you use that for the old UI).

Presets and `.env` are read from `sglang_runtime/` next to `sglang_runtime.py` (same as the CLI).

Optional: comma-separated CORS origins (default includes Vite dev ports):

```bash
export STACK_UI_CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:3000
```

## Frontend (development)

```bash
cd stack_ui/frontend
npm install
npm run dev
```

Open the URL Vite prints (usually `http://127.0.0.1:5173`). API calls go to `/api` and are proxied to port 8765.

## Single port (API + built UI)

```bash
cd stack_ui/frontend && npm install && npm run build
cd ../backend && uvicorn stack_ui_server:app --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765/`.

## Legacy entrypoint

From `sglang_runtime/` you can still run:

```bash
cd sglang_runtime
uvicorn web_ui.server:app --host 127.0.0.1 --port 8765
```

That module re-exports the same FastAPI app from `stack_ui/backend`.
