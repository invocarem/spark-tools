"""HTTP helpers for probing running SGLang servers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def http_get_json(
    base_url: str,
    path: str,
    *,
    api_key: str,
    timeout_sec: int,
) -> dict[str, object]:
    url = base_url.rstrip("/") + path
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url=url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
            body: object
            try:
                body = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                body = raw
            return {"ok": True, "status": status, "body": body}
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        parsed: object = err_body
        try:
            if err_body.strip():
                parsed = json.loads(err_body)
        except json.JSONDecodeError:
            pass
        return {
            "ok": False,
            "status": int(exc.code),
            "error": exc.reason,
            "body": parsed,
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def collect_running_server_scan(
    base_url: str,
    *,
    api_key: str,
    timeout_sec: int,
    readiness: bool,
) -> dict[str, object]:
    out: dict[str, object] = {"base_url": base_url.rstrip("/")}
    out["v1_models"] = http_get_json(
        base_url, "/v1/models", api_key=api_key, timeout_sec=timeout_sec
    )
    out["health"] = http_get_json(
        base_url, "/health", api_key=api_key, timeout_sec=timeout_sec
    )
    if readiness:
        out["health_generate"] = http_get_json(
            base_url, "/health_generate", api_key=api_key, timeout_sec=timeout_sec
        )
    out["server_info"] = http_get_json(
        base_url, "/get_server_info", api_key=api_key, timeout_sec=timeout_sec
    )
    si = out["server_info"]
    if isinstance(si, dict) and not si.get("ok") and "status" in si:
        out["server_info_alt"] = http_get_json(
            base_url, "/server_info", api_key=api_key, timeout_sec=timeout_sec
        )
    return out


def build_remote_scan_script(remote_url: str, api_key: str, timeout_sec: int, readiness: bool) -> str:
    template = """python3 - <<'PY'
import json
import urllib.error
import urllib.request

BASE = __BASE__
API_KEY = __API_KEY__
TIMEOUT = __TIMEOUT__
READINESS = __READINESS__


def http_get(path):
    url = BASE.rstrip("/") + path
    headers = {}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY
    req = urllib.request.Request(url=url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
            try:
                body = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                body = raw
            return {"ok": True, "status": status, "body": body}
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            parsed = json.loads(err_body) if err_body.strip() else err_body
        except json.JSONDecodeError:
            parsed = err_body
        return {"ok": False, "status": int(exc.code), "error": exc.reason, "body": parsed}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


out = {"base_url": BASE.rstrip("/"), "v1_models": http_get("/v1/models"), "health": http_get("/health")}
if READINESS:
    out["health_generate"] = http_get("/health_generate")
out["server_info"] = http_get("/get_server_info")
si = out["server_info"]
if not si.get("ok") and "status" in si:
    out["server_info_alt"] = http_get("/server_info")
print(json.dumps(out))
PY"""
    return (
        template.replace("__BASE__", json.dumps(remote_url))
        .replace("__API_KEY__", json.dumps(api_key))
        .replace("__TIMEOUT__", str(int(timeout_sec)))
        .replace("__READINESS__", "True" if readiness else "False")
    )
