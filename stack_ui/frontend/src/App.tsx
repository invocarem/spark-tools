import { useCallback, useEffect, useState } from "react";

type ArgRow = {
  kind: string;
  flag: string;
  value: string;
  enabled: boolean;
};

type ScanSummary = {
  base_url?: string;
  health_ok?: boolean | null;
  health_status?: unknown;
  health_error?: unknown;
  readiness_ok?: boolean | null;
  models?: string[];
  server_info_ok?: boolean | null;
  server_info_preview?: string | null;
  v1_models_ok?: boolean | null;
  notes?: string[];
};

type ScanResponse = {
  returncode: number;
  stdout?: string;
  stderr?: string;
  summary?: ScanSummary | null;
};

type Runtime = "venv" | "docker";

async function api<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const opt: RequestInit = { method, headers: {} };
  if (body !== undefined) {
    (opt.headers as Record<string, string>)["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const url = path.startsWith("/") ? path : `/${path}`;
  const r = await fetch(url, opt);
  const text = await r.text();
  let data: unknown;
  try {
    data = JSON.parse(text) as unknown;
  } catch {
    throw new Error(text || r.statusText);
  }
  if (!r.ok) {
    const d = data as { detail?: unknown };
    let msg =
      d.detail != null ? JSON.stringify(d.detail) : text || r.statusText;
    if (r.status === 404) {
      msg += ` — requested ${new URL(url, window.location.origin).href}. Restart the API server if routes 404.`;
    }
    throw new Error(msg);
  }
  return data as T;
}

function ScanSummaryView({ data }: { data: ScanResponse }) {
  if (data.returncode !== 0) {
    return (
      <>
        <p className="status-err">scan exited with code {data.returncode}</p>
        {data.stderr ? <pre className="snippet">{data.stderr}</pre> : null}
      </>
    );
  }
  if (!data.summary) {
    const raw = (data.stdout || "").slice(0, 4000);
    return (
      <>
        <p className="hint">No JSON in stdout. Raw:</p>
        <pre className="snippet">{raw}</pre>
      </>
    );
  }
  const s = data.summary;
  const hOk = s.health_ok;
  const hLabel = hOk === true ? "OK" : hOk === false ? "FAIL" : "unknown";
  const hTag = hOk === true ? "ok" : hOk === false ? "bad" : "";
  const siOk = s.server_info_ok;

  return (
    <>
      <div className="status-head">Current server</div>
      <div>
        <strong>URL</strong> {String(s.base_url || "")}
      </div>
      <div>
        <strong>Health</strong>{" "}
        <span className={`tag ${hTag}`}>{hLabel}</span>
        {s.health_status != null ? ` HTTP ${String(s.health_status)}` : ""}
        {s.health_error ? (
          <span className="err-inline"> {String(s.health_error)}</span>
        ) : null}
      </div>
      {s.readiness_ok !== null && s.readiness_ok !== undefined ? (
        <div>
          <strong>Readiness</strong>{" "}
          <span className={`tag ${s.readiness_ok ? "ok" : "bad"}`}>
            {s.readiness_ok ? "OK" : "FAIL"}
          </span>
        </div>
      ) : null}
      <div>
        <strong>Models</strong>{" "}
        {s.models && s.models.length ? (
          s.models.map((m) => (
            <code key={m}>{m} </code>
          ))
        ) : (
          <span className="muted">(none or /v1/models unreachable)</span>
        )}
      </div>
      <div>
        <strong>Server info</strong>{" "}
        <span className={`tag ${siOk ? "ok" : "bad"}`}>{siOk ? "OK" : "FAIL"}</span>
      </div>
      {s.server_info_preview ? <pre className="snippet">{s.server_info_preview}</pre> : null}
      {(s.notes || []).map((n, i) => (
        <p key={i} className="hint">
          {n}
        </p>
      ))}
    </>
  );
}

export default function App() {
  const [runtime, setRuntime] = useState<Runtime>("venv");
  const [presetNames, setPresetNames] = useState<string[]>([]);
  const [preset, setPreset] = useState("");
  const [rows, setRows] = useState<ArgRow[]>([]);
  const [extraSglang, setExtraSglang] = useState("");
  const [overrideImage, setOverrideImage] = useState("");
  const [overrideVenvPath, setOverrideVenvPath] = useState("");
  const [runMode, setRunMode] = useState<"solo" | "cluster">("solo");
  const [soloHost, setSoloHost] = useState("");
  const [logFile, setLogFile] = useState("");
  const [clusterHosts, setClusterHosts] = useState("");
  const [distAddr, setDistAddr] = useState("");
  const [logDir, setLogDir] = useState("");
  const [launchVerbose, setLaunchVerbose] = useState(false);
  const [logLines, setLogLines] = useState(80);
  const [scanBaseUrl, setScanBaseUrl] = useState("");
  const [scanSshHost, setScanSshHost] = useState("");
  const [scanBindHost, setScanBindHost] = useState("127.0.0.1");
  const [scanPort, setScanPort] = useState("");
  const [scanReadiness, setScanReadiness] = useState(false);
  const [out, setOut] = useState("");
  const [outOk, setOutOk] = useState<boolean | undefined>(undefined);
  const [scanStatus, setScanStatus] = useState<ScanResponse | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);

  const setOutStyled = useCallback((text: string, ok?: boolean) => {
    setOut(text);
    setOutOk(ok);
  }, []);

  const clusterHostList = useCallback(() => {
    const raw = clusterHosts.trim();
    if (!raw) return [];
    return raw.split(",").map((s) => s.trim()).filter(Boolean);
  }, [clusterHosts]);

  const collectPayload = useCallback(() => {
    return {
      runtime,
      presets_file: "",
      preset,
      env_file: "",
      rows,
      extra_sglang: extraSglang,
      override_tp: null as number | null,
      override_port: null as number | null,
      override_model_path: "",
      override_venv_path: runtime === "venv" ? (overrideVenvPath || "") : "",
      override_image: runtime === "docker" ? (overrideImage || "") : "",
    };
  }, [runtime, preset, rows, extraSglang, overrideImage, overrideVenvPath]);

  const reloadPresets = useCallback(async () => {
    const data = await api<{ names: string[] }>("GET", `/api/presets?runtime=${runtime}`);
    setPresetNames(data.names);
    setPreset((cur) => (data.names.includes(cur) ? cur : data.names[0] || ""));
  }, [runtime]);

  useEffect(() => {
    (async () => {
      try {
        await reloadPresets();
      } catch (e) {
        setOutStyled(String(e), false);
      }
    })();
  }, [reloadPresets, setOutStyled]);

  const loadRowsForPreset = useCallback(
    async (name: string) => {
      if (!name) return;
      const data = await api<{ rows: ArgRow[] }>(
        "GET",
        `/api/preset/${encodeURIComponent(name)}/sglang-rows?runtime=${runtime}`,
      );
      setRows(
        data.rows.map((r) => ({
          kind: r.kind,
          flag: r.flag || "",
          value: r.value || "",
          enabled: r.enabled !== false,
        })),
      );
    },
    [runtime],
  );

  useEffect(() => {
    if (!preset) return;
    (async () => {
      try {
        await loadRowsForPreset(preset);
      } catch (e) {
        setOutStyled(String(e), false);
      }
    })();
  }, [preset, loadRowsForPreset, setOutStyled]);

  const updateRow = (i: number, patch: Partial<ArgRow>) => {
    setRows((prev) => {
      const next = [...prev];
      next[i] = { ...next[i], ...patch };
      return next;
    });
  };

  const deleteRow = (i: number) => {
    setRows((prev) => prev.filter((_, j) => j !== i));
  };

  const runtimeLabel = runtime === "docker" ? "sglang_docker" : "sglang_runtime";
  const presetsPathLabel =
    runtime === "docker"
      ? "sglang_docker/model_presets.json"
      : "sglang_runtime/model_presets.json";

  return (
    <>
      <header className="hdr">
        <h1>Stack UI — {runtimeLabel}</h1>
        <div className="runtime-switch">
          <span className="hint">Runtime:</span>{" "}
          <label className="radio-label">
            <input
              type="radio"
              name="runtime"
              value="venv"
              checked={runtime === "venv"}
              onChange={() => setRuntime("venv")}
            />{" "}
            venv (sglang_runtime)
          </label>
          <label className="radio-label">
            <input
              type="radio"
              name="runtime"
              value="docker"
              checked={runtime === "docker"}
              onChange={() => setRuntime("docker")}
            />{" "}
            docker (sglang_docker)
          </label>
        </div>
        <p className="sub">
          Uses <code>{presetsPathLabel}</code> and optional{" "}
          <code>{runtimeLabel}/.env</code> on the API host — tweak flags below, then launch solo or
          cluster (<code>nnodes</code> = number of hosts; <code>node-rank</code> = order in the
          list, 0 … n−1).
        </p>
      </header>

      <main className="grid">
        <section className="card span3">
          <h2>
            Preset &amp; <code>sglang_args</code>
          </h2>
          <label>
            Preset{" "}
            <select
              value={preset}
              onChange={(e) => {
                setPreset(e.target.value);
              }}
            >
              {presetNames.length === 0 ? (
                <option value="">(no presets — add model_presets.json)</option>
              ) : null}
              {presetNames.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <p className="hint">
            Toggle rows to drop flags; edit values. Extra argv is appended after the preset merge.
          </p>
          {!rows.length ? (
            <p className="hint">No sglang_args in preset (or not loaded).</p>
          ) : (
            <table className="arg-table">
              <thead>
                <tr>
                  <th className="narrow">On</th>
                  <th>Kind</th>
                  <th>Flag / raw</th>
                  <th>Value</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>
                    <td className="narrow">
                      <input
                        type="checkbox"
                        checked={r.enabled}
                        onChange={(e) => updateRow(i, { enabled: e.target.checked })}
                      />
                    </td>
                    <td>
                      <select
                        value={r.kind}
                        onChange={(e) => updateRow(i, { kind: e.target.value })}
                      >
                        <option value="switch">switch</option>
                        <option value="pair">pair</option>
                        <option value="raw">raw</option>
                      </select>
                    </td>
                    <td>
                      <input
                        type="text"
                        value={r.flag}
                        onChange={(e) => updateRow(i, { flag: e.target.value })}
                        spellCheck={false}
                      />
                    </td>
                    <td>
                      <input
                        type="text"
                        value={r.value}
                        onChange={(e) => updateRow(i, { value: e.target.value })}
                        spellCheck={false}
                      />
                    </td>
                    <td>
                      <button type="button" onClick={() => deleteRow(i)}>
                        ×
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="btnrow">
            <button
              type="button"
              onClick={() =>
                setRows((p) => [...p, { kind: "pair", flag: "--new-flag", value: "", enabled: true }])
              }
            >
              Add flag + value
            </button>
            <button
              type="button"
              onClick={() =>
                setRows((p) => [
                  ...p,
                  { kind: "switch", flag: "--new-switch", value: "", enabled: true },
                ])
              }
            >
              Add switch
            </button>
            <button
              type="button"
              onClick={async () => {
                try {
                  await loadRowsForPreset(preset);
                  setOutStyled("Rows reset from preset.", true);
                } catch (e) {
                  setOutStyled(String(e), false);
                }
              }}
            >
              Reset from preset
            </button>
          </div>
          <label>
            Extra argv (<code>shlex</code> split){" "}
            <input
              type="text"
              value={extraSglang}
              onChange={(e) => setExtraSglang(e.target.value)}
              spellCheck={false}
            />
          </label>
          {runtime === "venv" ? (
            <label>
              Override venv path{" "}
              <input
                type="text"
                value={overrideVenvPath}
                onChange={(e) => setOverrideVenvPath(e.target.value)}
                placeholder="~/.sglang"
                spellCheck={false}
              />
            </label>
          ) : (
            <label>
              Override Docker image{" "}
              <input
                type="text"
                value={overrideImage}
                onChange={(e) => setOverrideImage(e.target.value)}
                placeholder="scitrera/dgx-spark-sglang:latest"
                spellCheck={false}
              />
            </label>
          )}
        </section>

        <section className="card span2">
          <h2>Launch</h2>
          <label>
            Mode{" "}
            <select
              value={runMode}
              onChange={(e) => setRunMode(e.target.value as "solo" | "cluster")}
            >
              <option value="solo">solo</option>
              <option value="cluster">cluster</option>
            </select>
          </label>

          {runMode === "solo" ? (
            <div id="soloBlock">
              <label>
                Solo: SSH host (empty = run on this machine){" "}
                <input
                  type="text"
                  value={soloHost}
                  onChange={(e) => setSoloHost(e.target.value)}
                  placeholder="spark1"
                  spellCheck={false}
                />
              </label>
              {runtime === "venv" ? (
                <label>
                  Solo log file{" "}
                  <input
                    type="text"
                    value={logFile}
                    onChange={(e) => setLogFile(e.target.value)}
                    placeholder="sglang_solo.log"
                    spellCheck={false}
                  />
                </label>
              ) : (
                <label>
                  Log directory{" "}
                  <input
                    type="text"
                    value={logDir}
                    onChange={(e) => setLogDir(e.target.value)}
                    placeholder="~/sglang-docker-logs"
                    spellCheck={false}
                  />
                </label>
              )}
            </div>
          ) : (
            <div id="clusterBlock">
              <label>
                Cluster hosts (comma-separated, launch order = rank){" "}
                <input
                  type="text"
                  value={clusterHosts}
                  onChange={(e) => setClusterHosts(e.target.value)}
                  placeholder="spark1,spark2"
                  spellCheck={false}
                />
              </label>
              <p className="hint">
                Each host gets one launch with <code>--nnodes</code> = list length and{" "}
                <code>--node-rank</code> = index (0 for the first host).
              </p>
              <label>
                <code>--dist-init-addr</code>{" "}
                <input
                  type="text"
                  value={distAddr}
                  onChange={(e) => setDistAddr(e.target.value)}
                  placeholder="spark1:29500"
                  spellCheck={false}
                />
              </label>
              <label>
                Cluster log directory{" "}
                <input
                  type="text"
                  value={logDir}
                  onChange={(e) => setLogDir(e.target.value)}
                  placeholder={
                    runtime === "docker"
                      ? "~/sglang-docker-logs"
                      : "~/runtime-sglang/logs"
                  }
                  spellCheck={false}
                />
              </label>
            </div>
          )}

          <label className="chk">
            <input
              type="checkbox"
              checked={launchVerbose}
              onChange={(e) => setLaunchVerbose(e.target.checked)}
            />{" "}
            Verbose CLI
          </label>
          <div className="btnrow">
            <button
              type="button"
              onClick={async () => {
                try {
                  const data = await api("POST", "/api/preview-launch", collectPayload());
                  setOutStyled(JSON.stringify(data, null, 2), true);
                } catch (e) {
                  setOutStyled(String(e), false);
                }
              }}
            >
              Preview command
            </button>
            <button
              type="button"
              className="danger"
              onClick={async () => {
                if (!window.confirm(`Run ${runtimeLabel} launch with the built command?`)) return;
                const hosts = clusterHostList();
                if (runMode === "cluster" && hosts.length === 0) {
                  setOutStyled('Cluster mode needs at least one host in "Cluster hosts".', false);
                  return;
                }
                const body = {
                  ...collectPayload(),
                  mode: runMode,
                  host: soloHost.trim(),
                  hosts,
                  log_dir: logDir.trim() || null,
                  log_file: runtime === "venv" ? (logFile.trim() || null) : null,
                  dist_addr: distAddr.trim(),
                  verbose: launchVerbose,
                };
                try {
                  const data = await api<{ returncode: number }>("POST", "/api/launch", body);
                  const ok = data.returncode === 0;
                  setOutStyled(JSON.stringify(data, null, 2), ok);
                } catch (e) {
                  setOutStyled(String(e), false);
                }
              }}
            >
              Launch
            </button>
          </div>
        </section>

        <section className="card">
          <h2>Stop / logs</h2>
          <p className="hint">Same mode and host list as Launch.</p>
          <div className="btnrow">
            <button
              type="button"
              className="danger"
              onClick={async () => {
                const hosts = clusterHostList();
                const body = {
                  runtime,
                  mode: runMode,
                  host: soloHost.trim(),
                  hosts,
                  preset,
                  presets_file: "",
                  env_file: "",
                };
                try {
                  const data = await api<{ returncode: number }>("POST", "/api/stop", body);
                  setOutStyled(JSON.stringify(data, null, 2), data.returncode === 0);
                } catch (e) {
                  setOutStyled(String(e), false);
                }
              }}
            >
              Stop
            </button>
            <button
              type="button"
              onClick={async () => {
                const hosts = clusterHostList();
                const body = {
                  runtime,
                  mode: runMode,
                  host: soloHost.trim(),
                  hosts,
                  env_file: "",
                  log_dir: logDir.trim() || null,
                  log_file: runtime === "venv" ? (logFile.trim() || null) : null,
                  lines: logLines || 80,
                };
                try {
                  const data = await api<{ returncode: number }>("POST", "/api/logs", body);
                  setOutStyled(JSON.stringify(data, null, 2), data.returncode === 0);
                } catch (e) {
                  setOutStyled(String(e), false);
                }
              }}
            >
              Logs
            </button>
            {runtime === "docker" ? (
              <button
                type="button"
                onClick={async () => {
                  const body = {
                    runtime: "docker",
                    preset,
                    presets_file: "",
                    hosts: clusterHostList(),
                    env_file: "",
                  };
                  try {
                    const data = await api<{ returncode: number }>(
                      "POST",
                      "/api/exec",
                      { runtime: "docker", subcommand: "pull", args: body },
                    );
                    setOutStyled(JSON.stringify(data, null, 2), data.returncode === 0);
                  } catch (e) {
                    setOutStyled(String(e), false);
                  }
                }}
              >
                Pull image
              </button>
            ) : null}
          </div>
          <label>
            Log lines{" "}
            <input
              type="number"
              value={logLines}
              onChange={(e) => setLogLines(parseInt(e.target.value, 10) || 80)}
              min={1}
              max={5000}
            />
          </label>
        </section>

        <section className="card span2">
          <h2>Scan</h2>
          <p className="hint">
            HTTP probe of the running server. Open{" "}
            <a href="/api/health" target="_blank" rel="noopener noreferrer">
              <code>/api/health</code>
            </a>{" "}
            if routes 404.
          </p>
          <label>
            Server URL{" "}
            <input
              type="text"
              value={scanBaseUrl}
              onChange={(e) => setScanBaseUrl(e.target.value)}
              placeholder="http://spark1:30000"
              spellCheck={false}
            />
          </label>
          <label>
            Or SSH host (probe <code>localhost</code> on that box){" "}
            <input
              type="text"
              value={scanSshHost}
              onChange={(e) => setScanSshHost(e.target.value)}
              spellCheck={false}
            />
          </label>
          <details className="adv">
            <summary>Advanced</summary>
            <label>
              Bind host (no URL only){" "}
              <input
                type="text"
                value={scanBindHost}
                onChange={(e) => setScanBindHost(e.target.value)}
                spellCheck={false}
              />
            </label>
            <label>
              Port override{" "}
              <input
                type="number"
                value={scanPort}
                onChange={(e) => setScanPort(e.target.value)}
                placeholder="from preset"
              />
            </label>
            <label className="chk">
              <input
                type="checkbox"
                checked={scanReadiness}
                onChange={(e) => setScanReadiness(e.target.checked)}
              />{" "}
              <code>/health_generate</code>
            </label>
          </details>
          <div className="btnrow">
            <button
              type="button"
              onClick={async () => {
                const scanPortTrim = scanPort.trim();
                let port: number | null = null;
                if (scanPortTrim !== "") {
                  const n = parseInt(scanPortTrim, 10);
                  if (Number.isNaN(n)) {
                    setScanError("Port must be a number");
                    setOutStyled("Invalid port", false);
                    return;
                  }
                  port = n;
                }
                const ssh = scanSshHost.trim() || soloHost.trim();
                const body = {
                  runtime,
                  presets_file: "",
                  preset,
                  env_file: "",
                  port,
                  base_url: scanBaseUrl.trim(),
                  bind_host: scanBindHost.trim() || "127.0.0.1",
                  host: ssh,
                  readiness: scanReadiness,
                };
                try {
                  const data = await api<ScanResponse>("POST", "/api/scan", body);
                  setScanStatus(data);
                  setScanError(null);
                  const healthy = !data.summary || data.summary.health_ok !== false;
                  const ok = data.returncode === 0 && healthy;
                  setOutStyled(JSON.stringify(data, null, 2), ok);
                } catch (e) {
                  setScanStatus(null);
                  setScanError(String(e));
                  setOutStyled(String(e), false);
                }
              }}
            >
              Refresh status
            </button>
          </div>
          <div className="scan-status" aria-live="polite">
            {scanError ? <p className="status-err">{scanError}</p> : null}
            {scanStatus && !scanError ? <ScanSummaryView data={scanStatus} /> : null}
          </div>
        </section>

        <section className="card span3">
          <h2>Output</h2>
          <pre
            id="out"
            className={`out${outOk === true ? " status-ok" : ""}${outOk === false ? " status-err" : ""}`}
          >
            {out}
          </pre>
        </section>
      </main>
    </>
  );
}
