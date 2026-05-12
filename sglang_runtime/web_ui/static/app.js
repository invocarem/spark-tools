/** @typedef {{ kind: string, flag?: string, value?: string, enabled: boolean }} ArgRow */

const $ = (id) => document.getElementById(id);

let rows = /** @type {ArgRow[]} */ ([]);

function selectedPreset() {
  return $("presetSelect").value;
}

function setOut(text, ok) {
  const el = $("out");
  el.textContent = text;
  el.classList.remove("status-ok", "status-err");
  if (ok === true) el.classList.add("status-ok");
  if (ok === false) el.classList.add("status-err");
}

function apiUrl(path) {
  const p = path.startsWith("/") ? path.slice(1) : path;
  return new URL(p, document.baseURI).href;
}

async function api(method, path, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const url = apiUrl(path);
  const r = await fetch(url, opt);
  const text = await r.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(text || r.statusText);
  }
  if (!r.ok) {
    let msg = data.detail != null ? JSON.stringify(data.detail) : text;
    if (r.status === 404) {
      msg += ` — requested ${url}. Restart uvicorn after updating \`web_ui/server.py\` if needed.`;
    }
    throw new Error(msg);
  }
  return data;
}

/** Defaults: server-side ``model_presets.json`` and optional ``.env`` in ``sglang_runtime/`` only. */
function collectPayload() {
  return {
    presets_file: "",
    preset: selectedPreset(),
    env_file: "",
    rows,
    extra_sglang: $("extraSglang").value,
    override_tp: null,
    override_port: null,
    override_model_path: "",
    override_venv_path: "",
  };
}

function clusterHostList() {
  const raw = $("clusterHosts").value.trim();
  if (!raw) return [];
  return raw.split(",").map((s) => s.trim()).filter(Boolean);
}

function syncModeBlocks() {
  const cluster = $("runMode").value === "cluster";
  $("soloBlock").hidden = cluster;
  $("clusterBlock").hidden = !cluster;
}

function renderArgRows() {
  const host = $("argRows");
  if (!rows.length) {
    host.innerHTML = '<p class="hint">No sglang_args in preset (or not loaded).</p>';
    return;
  }
  const th = `<tr><th class="narrow">On</th><th>Kind</th><th>Flag / raw</th><th>Value</th><th></th></tr>`;
  const body = rows
    .map(
      (r, i) => `
    <tr data-i="${i}">
      <td class="narrow"><input type="checkbox" data-k="enabled" ${r.enabled ? "checked" : ""} /></td>
      <td><select data-k="kind">
        <option value="switch" ${r.kind === "switch" ? "selected" : ""}>switch</option>
        <option value="pair" ${r.kind === "pair" ? "selected" : ""}>pair</option>
        <option value="raw" ${r.kind === "raw" ? "selected" : ""}>raw</option>
      </select></td>
      <td><input type="text" data-k="flag" value="${esc(r.flag || "")}" /></td>
      <td><input type="text" data-k="value" value="${esc(r.value || "")}" /></td>
      <td><button type="button" data-del="${i}">×</button></td>
    </tr>`
    )
    .join("");
  host.innerHTML = `<table class="arg-table">${th}${body}</table>`;

  host.querySelectorAll("tr[data-i]").forEach((tr) => {
    const i = +tr.getAttribute("data-i");
    tr.querySelector("[data-k=enabled]").addEventListener("change", (e) => {
      rows[i].enabled = /** @type {HTMLInputElement} */ (e.target).checked;
    });
    tr.querySelector("[data-k=kind]").addEventListener("change", (e) => {
      rows[i].kind = /** @type {HTMLSelectElement} */ (e.target).value;
    });
    tr.querySelector("[data-k=flag]").addEventListener("input", (e) => {
      rows[i].flag = /** @type {HTMLInputElement} */ (e.target).value;
    });
    tr.querySelector("[data-k=value]").addEventListener("input", (e) => {
      rows[i].value = /** @type {HTMLInputElement} */ (e.target).value;
    });
    tr.querySelector("[data-del]").addEventListener("click", () => {
      rows.splice(i, 1);
      renderArgRows();
    });
  });
}

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderScanSummary(data) {
  const el = $("scanStatus");
  if (!el) return;
  if (data.returncode !== 0) {
    el.innerHTML =
      `<p class="status-err">scan exited with code ${data.returncode}</p>` +
      (data.stderr ? `<pre class="snippet">${esc(data.stderr)}</pre>` : "");
    return;
  }
  if (!data.summary) {
    const raw = data.stdout || "";
    el.innerHTML =
      '<p class="hint">No JSON in stdout. Raw:</p>' +
      `<pre class="snippet">${esc(raw.slice(0, 4000))}</pre>`;
    return;
  }
  const s = data.summary;
  const parts = [];
  parts.push('<div class="status-head">Current server</div>');
  parts.push(`<div><strong>URL</strong> ${esc(String(s.base_url || ""))}</div>`);
  const hOk = s.health_ok;
  const hTag = hOk === true ? "ok" : hOk === false ? "bad" : "";
  const hLabel = hOk === true ? "OK" : hOk === false ? "FAIL" : "unknown";
  parts.push(
    `<div><strong>Health</strong> <span class="tag ${hTag}">${hLabel}</span>` +
      (s.health_status != null ? ` HTTP ${esc(String(s.health_status))}` : "") +
      (s.health_error ? ` <span class="err-inline">${esc(String(s.health_error))}</span>` : "") +
      `</div>`
  );
  if (s.readiness_ok !== null && s.readiness_ok !== undefined) {
    parts.push(
      `<div><strong>Readiness</strong> <span class="tag ${s.readiness_ok ? "ok" : "bad"}">` +
        `${s.readiness_ok ? "OK" : "FAIL"}</span></div>`
    );
  }
  parts.push('<div><strong>Models</strong> ');
  if (s.models && s.models.length) {
    parts.push(s.models.map((m) => `<code>${esc(m)}</code>`).join(" "));
  } else {
    parts.push('<span class="muted">(none or /v1/models unreachable)</span>');
  }
  parts.push("</div>");
  const siOk = s.server_info_ok;
  parts.push(
    `<div><strong>Server info</strong> <span class="tag ${siOk ? "ok" : "bad"}">` +
      `${siOk ? "OK" : "FAIL"}</span></div>`
  );
  if (s.server_info_preview) {
    parts.push(`<pre class="snippet">${esc(s.server_info_preview)}</pre>`);
  }
  (s.notes || []).forEach((n) => {
    parts.push(`<p class="hint">${esc(String(n))}</p>`);
  });
  el.innerHTML = parts.join("");
}

function renderScanSummaryError(msg) {
  const el = $("scanStatus");
  if (el) el.innerHTML = `<p class="status-err">${esc(msg)}</p>`;
}

async function loadPresetsList() {
  const data = await api("GET", "/api/presets");
  const sel = $("presetSelect");
  const cur = sel.value;
  sel.innerHTML = "";
  for (const name of data.names) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    sel.appendChild(o);
  }
  if (data.names.includes(cur)) sel.value = cur;
  else if (data.names.length) sel.selectedIndex = 0;
}

async function loadRowsForPreset() {
  const name = selectedPreset();
  if (!name) return;
  const data = await api("GET", `/api/preset/${encodeURIComponent(name)}/sglang-rows`);
  rows = data.rows.map((r) => ({
    kind: r.kind,
    flag: r.flag || "",
    value: r.value || "",
    enabled: r.enabled !== false,
  }));
  renderArgRows();
}

async function init() {
  await loadPresetsList();
  await loadRowsForPreset();
  syncModeBlocks();
}

$("runMode").addEventListener("change", () => {
  syncModeBlocks();
});

$("presetSelect").addEventListener("change", async () => {
  try {
    await loadRowsForPreset();
  } catch (e) {
    setOut(String(e), false);
  }
});

$("btnReloadRows").addEventListener("click", async () => {
  try {
    await loadRowsForPreset();
    setOut("Rows reset from preset.", true);
  } catch (e) {
    setOut(String(e), false);
  }
});

$("btnAddPair").addEventListener("click", () => {
  rows.push({ kind: "pair", flag: "--new-flag", value: "", enabled: true });
  renderArgRows();
});

$("btnAddSwitch").addEventListener("click", () => {
  rows.push({ kind: "switch", flag: "--new-switch", value: "", enabled: true });
  renderArgRows();
});

$("btnPreview").addEventListener("click", async () => {
  try {
    const data = await api("POST", "/api/preview-launch", collectPayload());
    setOut(JSON.stringify(data, null, 2), true);
  } catch (e) {
    setOut(String(e), false);
  }
});

$("btnLaunch").addEventListener("click", async () => {
  if (!confirm("Run launch with the built shell command?")) return;
  const mode = $("runMode").value;
  const hosts = clusterHostList();
  if (mode === "cluster" && hosts.length === 0) {
    setOut("Cluster mode needs at least one host in “Cluster hosts”.", false);
    return;
  }
  const body = {
    ...collectPayload(),
    mode,
    host: $("soloHost").value.trim(),
    hosts,
    log_dir: $("logDir").value.trim() || null,
    log_file: $("logFile").value.trim() || null,
    dist_addr: $("distAddr").value.trim(),
    verbose: $("launchVerbose").checked,
  };
  try {
    const data = await api("POST", "/api/launch", body);
    const ok = data.returncode === 0;
    setOut(JSON.stringify(data, null, 2), ok);
  } catch (e) {
    setOut(String(e), false);
  }
});

$("btnStop").addEventListener("click", async () => {
  const mode = $("runMode").value;
  const hosts = clusterHostList();
  const body = {
    mode,
    host: $("soloHost").value.trim(),
    hosts,
    preset: selectedPreset(),
    presets_file: "",
    env_file: "",
  };
  try {
    const data = await api("POST", "/api/stop", body);
    const ok = data.returncode === 0;
    setOut(JSON.stringify(data, null, 2), ok);
  } catch (e) {
    setOut(String(e), false);
  }
});

$("btnLogs").addEventListener("click", async () => {
  const mode = $("runMode").value;
  const hosts = clusterHostList();
  const body = {
    mode,
    host: $("soloHost").value.trim(),
    hosts,
    env_file: "",
    log_dir: $("logDir").value.trim() || null,
    log_file: $("logFile").value.trim() || null,
    lines: parseInt($("logLines").value, 10) || 80,
  };
  try {
    const data = await api("POST", "/api/logs", body);
    const ok = data.returncode === 0;
    setOut(JSON.stringify(data, null, 2), ok);
  } catch (e) {
    setOut(String(e), false);
  }
});

$("btnScan").addEventListener("click", async () => {
  const scanPort = $("scanPort").value.trim();
  let port = null;
  if (scanPort !== "") {
    const n = parseInt(scanPort, 10);
    if (Number.isNaN(n)) {
      renderScanSummaryError("Port must be a number");
      setOut("Invalid port", false);
      return;
    }
    port = n;
  }
  const ssh = $("scanSshHost").value.trim() || $("soloHost").value.trim();
  const body = {
    presets_file: "",
    preset: selectedPreset(),
    env_file: "",
    port,
    base_url: $("scanBaseUrl").value.trim(),
    bind_host: $("scanBindHost").value.trim() || "127.0.0.1",
    host: ssh,
    readiness: $("scanReadiness").checked,
  };
  try {
    const data = await api("POST", "/api/scan", body);
    renderScanSummary(data);
    const healthy = !data.summary || data.summary.health_ok !== false;
    const ok = data.returncode === 0 && healthy;
    setOut(JSON.stringify(data, null, 2), ok);
  } catch (e) {
    renderScanSummaryError(String(e));
    setOut(String(e), false);
  }
});

init().catch((e) => setOut(String(e), false));
