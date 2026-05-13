# Stack UI

Web console for **sglang_runtime** and **sglang_docker** (presets, launch/stop/logs, scan). Switch between runtimes with the radio buttons in the header.

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

Or reuse an existing env that already has FastAPI.

Presets and `.env` are resolved per runtime:
- **venv** → `sglang_runtime/model_presets.json` and `sglang_runtime/.env`
- **docker** → `sglang_docker/model_presets.json` and `sglang_docker/.env`

Optional: comma-separated CORS origins (default includes Vite dev ports):

```bash
export STACK_UI_CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:3000
```

### API

Every endpoint accepts a `runtime` field (`"venv"` or `"docker"`, defaults to `"venv"`). GET endpoints also accept `?runtime=` as a query parameter.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/defaults` | Returns paths for both runtimes, available subcommands |
| GET | `/api/presets?runtime=` | Load presets for the given runtime |
| GET | `/api/preset/{name}/sglang-rows?runtime=` | Parse `sglang_args` rows for a preset |
| POST | `/api/preview-launch` | Build launch command (venv: source+activate; docker: `docker run`) |
| POST | `/api/launch` | Execute launch via the selected runtime CLI |
| POST | `/api/stop` | Stop via the selected runtime CLI |
| POST | `/api/logs` | Fetch logs via the selected runtime CLI |
| POST | `/api/scan`, `/api/refresh` | Probe running server |
| POST | `/api/exec` | Run any allowed subcommand (`deploy` for venv, `pull` for docker) |
| GET | `/api/health` | Health check |

### Runtime-specific behaviour

- **venv**: supports `deploy`, uses `venv_path`, `log_file` for solo mode
- **docker**: supports `pull`, uses `image`, `log_dir` for container log mounts; no `deploy` subcommand

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
