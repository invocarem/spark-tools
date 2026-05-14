import { useCallback, useEffect, useRef, useState } from "react";

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

type TabId = "configure" | "launch" | "stop" | "logs" | "scan" | "tools";

type ToolId = "benchmark" | "measure" | "pull" | "deploy";

type LogsPayload = {
  returncode: number;
  stdout: string;
  stderr: string;
};

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

const TABS: { id: TabId; label: string }[] = [
  { id: "configure", label: "Configure" },
  { id: "launch", label: "Launch" },
  { id: "stop", label: "Stop" },
  { id: "logs", label: "Logs" },
  { id: "scan", label: "Scan" },
  { id: "tools", label: "Tools" },
];

export default function App() {
  const [tab, setTab] = useState<TabId>("configure");
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
  const [logFromStart, setLogFromStart] = useState(false);
  const [scanBaseUrl, setScanBaseUrl] = useState("");
  const [scanSshHost, setScanSshHost] = useState("");
  const [scanBindHost, setScanBindHost] = useState("127.0.0.1");
  const [scanPort, setScanPort] = useState("");
  const [scanReadiness, setScanReadiness] = useState(false);
  const [out, setOut] = useState("");
  const [outOk, setOutOk] = useState<boolean | undefined>(undefined);
  const [scanStatus, setScanStatus] = useState<ScanResponse | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [lastLogs, setLastLogs] = useState<LogsPayload | null>(null);
  const [toolId, setToolId] = useState<ToolId>("benchmark");
  const [benchApiKey, setBenchApiKey] = useState("EMPTY");
  const [benchPrompt, setBenchPrompt] = useState(
    "Write a short haiku about distributed inference.",
  );
  const [benchMaxTokens, setBenchMaxTokens] = useState(64);
  const [benchRequests, setBenchRequests] = useState(20);
  const [benchTimeoutSec, setBenchTimeoutSec] = useState(120);
  const [measureHostsText, setMeasureHostsText] = useState("");
  const [deploySetName, setDeploySetName] = useState("");

  /** After Preview, do not overwrite preset from scan until the user changes the preset dropdown. */
  const holdPresetAfterPreviewRef = useRef(false);
  /** Last /v1/models id that matched a preset name (used when reloadPresets would otherwise pick names[0]). */
  const lastScanMatchedPresetRef = useRef<string | null>(null);

  const setOutStyled = useCallback((text: string, ok?: boolean) => {
    setOut(text);
    setOutOk(ok);
  }, []);

  const clusterHostList = useCallback(() => {
    const raw = clusterHosts.trim();
    if (!raw) return [];
    return raw.split(",").map((s) => s.trim()).filter(Boolean);
  }, [clusterHosts]);

  const measureHostArgv = useCallback((): string[] => {
    const raw = measureHostsText.trim();
    if (raw) {
      return raw.split(",").map((s) => s.trim()).filter(Boolean);
    }
    if (runMode === "cluster") return clusterHostList();
    const one = soloHost.trim();
    return one ? [one] : [];
  }, [measureHostsText, runMode, clusterHostList, soloHost]);

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

  const runPreviewLaunch = useCallback(async () => {
    try {
      const data = await api("POST", "/api/preview-launch", collectPayload());
      holdPresetAfterPreviewRef.current = true;
      setOutStyled(JSON.stringify(data, null, 2), true);
    } catch (e) {
      setOutStyled(String(e), false);
    }
  }, [collectPayload, setOutStyled]);

  const firstMatchingPresetName = useCallback((models: string[] | undefined, names: string[]) => {
    if (!models?.length || !names.length) return null;
    for (const m of models) {
      if (names.includes(m)) return m;
    }
    return null;
  }, []);

  const reloadPresets = useCallback(async () => {
    const data = await api<{ names: string[] }>("GET", `/api/presets?runtime=${runtime}`);
    setPresetNames(data.names);
    setPreset((cur) => {
      if (data.names.includes(cur)) return cur;
      const scanHit = lastScanMatchedPresetRef.current;
      if (scanHit && data.names.includes(scanHit)) return scanHit;
      return data.names[0] || "";
    });
  }, [runtime]);

  useEffect(() => {
    lastScanMatchedPresetRef.current = null;
    holdPresetAfterPreviewRef.current = false;
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

  useEffect(() => {
    if (runtime === "docker" && toolId === "deploy") setToolId("benchmark");
    if (runtime === "venv" && toolId === "pull") setToolId("benchmark");
  }, [runtime, toolId]);

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

  /** Shared with Scan → Server URL; benchmark uses this (default if empty). */
  const benchmarkBaseUrl = scanBaseUrl.trim() || "http://127.0.0.1:30000";
  /** OpenAI-style model id for benchmark: matches Configure preset name. */
  const benchmarkModelId = preset.trim() || "default";

  const runExec = async (subcommand: string, args: string[]) => {
    const data = await api<{ returncode: number; stdout: string; stderr: string; argv: string[] }>(
      "POST",
      "/api/exec",
      { runtime, subcommand, args },
    );
    const ok = data.returncode === 0;
    setOutStyled(JSON.stringify(data, null, 2), ok);
  };

  const runTool = async () => {
    if (toolId === "pull" && runtime !== "docker") {
      setOutStyled("Pull image is only available for the docker runtime.", false);
      return;
    }
    if (toolId === "deploy" && runtime !== "venv") {
      setOutStyled("Deploy is only available for the venv (sglang_runtime) runtime.", false);
      return;
    }
    try {
      if (toolId === "benchmark") {
        await runExec("benchmark", [
          "--base-url",
          benchmarkBaseUrl,
          "--api-key",
          benchApiKey.trim() || "EMPTY",
          "--model",
          benchmarkModelId,
          "--prompt",
          benchPrompt,
          "--max-tokens",
          String(benchMaxTokens),
          "--requests",
          String(benchRequests),
          "--timeout-sec",
          String(benchTimeoutSec),
        ]);
        return;
      }
      if (toolId === "measure") {
        const hosts = measureHostArgv();
        const argv = hosts.length ? ["--hosts", ...hosts] : [];
        await runExec("measure", argv);
        return;
      }
      if (toolId === "pull") {
        const hosts = clusterHostList();
        let pullArgv: string[] = [];
        if (preset.trim() && hosts.length) {
          pullArgv = ["--preset", preset, "--hosts", ...hosts];
        } else if (preset.trim()) {
          pullArgv = ["--preset", preset];
        } else if (hosts.length) {
          pullArgv = ["--hosts", ...hosts];
        }
        await runExec("pull", pullArgv);
        return;
      }
      if (toolId === "deploy") {
        let hosts = clusterHostList();
        if (hosts.length === 0 && soloHost.trim()) {
          hosts = [soloHost.trim()];
        }
        if (hosts.length === 0) {
          setOutStyled(
            "Deploy needs at least one host: set cluster hosts on Launch or a solo SSH host.",
            false,
          );
          return;
        }
        const argv = ["--hosts", ...hosts];
        if (deploySetName.trim()) argv.push("--set", deploySetName.trim());
        await runExec("deploy", argv);
      }
    } catch (e) {
      setOutStyled(String(e), false);
    }
  };

  return (
    <>
      <header className="hdr">
        <h1>Stack UI</h1>
        <p className="sub">
          <strong>Configure</strong> sets preset, <code>sglang_args</code>, and runtime; the runtime
          choice applies to Launch, Stop, Logs, Scan, and Tools. Each runtime uses its own{" "}
          <code>model_presets.json</code> and optional <code>.env</code> on the API host.
        </p>
      </header>

      <nav className="tabs-bar" role="tablist" aria-label="Stack UI sections">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={tab === t.id}
            className={`tab-btn${tab === t.id ? " active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="tab-main">
        {tab === "configure" ? (
          <section className="card tab-card">
            <h2>
              Preset &amp; <code>sglang_args</code>
            </h2>
            <label>
              Preset{" "}
              <select
                value={preset}
                onChange={(e) => {
                  holdPresetAfterPreviewRef.current = false;
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
              Toggle rows to drop flags; edit values. Extra argv is appended after the preset merge
              on Launch.
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
              <button type="button" onClick={() => void runPreviewLaunch()}>
                Preview
              </button>
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
        ) : null}

        {tab === "launch" ? (
          <section className="card tab-card">
            <h2>Launch</h2>
            <p className="hint">
              Preset: <strong>{preset || "(none)"}</strong> — edit flags on the Configure tab. Cluster
              mode sets <code>--nnodes</code> from host count and <code>--node-rank</code> from list
              order (0 … n−1).
            </p>
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
              <button type="button" onClick={() => void runPreviewLaunch()}>
                Preview
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
        ) : null}

        {tab === "stop" ? (
          <section className="card tab-card">
            <h2>Stop</h2>
            <p className="hint">
              Stops the stack for the current preset using the same mode and host list as on the
              Launch tab ({runMode}
              {runMode === "solo" ? `, host: ${soloHost.trim() || "(local)"}` : `, hosts: ${clusterHosts || "(none)"}`}
              ).
            </p>
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
                Stop current model
              </button>
            </div>
          </section>
        ) : null}

        {tab === "logs" ? (
          <section className="card tab-card">
            <h2>Logs</h2>
            <p className="hint">
              Fetches log tail from the same hosts and paths as Launch (solo host or cluster list,
              log file / log dir).
            </p>
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
            <label className="chk">
              <input
                type="checkbox"
                checked={logFromStart}
                onChange={(e) => setLogFromStart(e.target.checked)}
              />{" "}
              From start of log file (<code>--from-start</code>)
            </label>
            <div className="btnrow">
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
                    from_start: logFromStart,
                  };
                  try {
                    const data = await api<LogsPayload>("POST", "/api/logs", body);
                    setLastLogs({
                      returncode: data.returncode,
                      stdout: String(data.stdout ?? ""),
                      stderr: String(data.stderr ?? ""),
                    });
                    setOutStyled(JSON.stringify(data, null, 2), data.returncode === 0);
                  } catch (e) {
                    setLastLogs(null);
                    setOutStyled(String(e), false);
                  }
                }}
              >
                Fetch logs
              </button>
            </div>
            <div className="log-pane" aria-live="polite">
              {!lastLogs ? (
                <p className="hint">No logs loaded yet.</p>
              ) : (
                <>
                  <div className="log-meta">
                    Exit code{" "}
                    <code className={lastLogs.returncode === 0 ? "ok-code" : "bad-code"}>
                      {lastLogs.returncode}
                    </code>
                  </div>
                  {lastLogs.stderr ? (
                    <div className="log-stderr-block">
                      <div className="log-stderr-label">stderr</div>
                      <pre className="log-stderr">{lastLogs.stderr}</pre>
                    </div>
                  ) : null}
                  <div className="log-stdout-label">stdout</div>
                  <pre className="log-stdout">{lastLogs.stdout || "(empty)"}</pre>
                </>
              )}
            </div>
          </section>
        ) : null}

        {tab === "scan" ? (
          <section className="card tab-card">
            <h2>Scan</h2>
            <p className="hint">
              HTTP probe of the running server (same as CLI <code>scan</code>). Open{" "}
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
            <p className="hint">
              Same value is used for Tools → benchmark base URL (no second copy to maintain).
            </p>
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
                    if (data.returncode === 0) {
                      const matched = firstMatchingPresetName(
                        data.summary?.models,
                        presetNames,
                      );
                      lastScanMatchedPresetRef.current = matched;
                      if (matched && !holdPresetAfterPreviewRef.current) {
                        setPreset(matched);
                      }
                    }
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
        ) : null}

        {tab === "tools" ? (
          <section className="card tab-card">
            <h2>Tools</h2>
            <p className="hint">
              Runs allowed <code>{runtimeLabel}</code> subcommands via the API (<code>benchmark</code>,{" "}
              <code>measure</code>
              {runtime === "docker" ? ", pull" : ", deploy"}). Full CLI output is in Output below.
            </p>
            <label>
              Tool{" "}
              <select
                value={toolId}
                onChange={(e) => setToolId(e.target.value as ToolId)}
              >
                <option value="benchmark">benchmark — API latency / throughput</option>
                <option value="measure">measure — GPU / load snapshots</option>
                {runtime === "docker" ? <option value="pull">pull — Docker image on hosts</option> : null}
                {runtime === "venv" ? <option value="deploy">deploy — rsync runtime to hosts</option> : null}
              </select>
            </label>

            {toolId === "benchmark" ? (
              <div className="tool-fields">
                <label>
                  Base URL (same as Scan → Server URL){" "}
                  <input
                    type="text"
                    value={scanBaseUrl}
                    onChange={(e) => setScanBaseUrl(e.target.value)}
                    placeholder="http://spark1:30000"
                    spellCheck={false}
                  />
                </label>
                <p className="hint">
                  Benchmark calls use <code>{benchmarkBaseUrl}</code>
                  {scanBaseUrl.trim() ? "" : " (default because Server URL is empty)"}.
                </p>
                <label>
                  API key{" "}
                  <input
                    type="text"
                    value={benchApiKey}
                    onChange={(e) => setBenchApiKey(e.target.value)}
                    spellCheck={false}
                  />
                </label>
                <p className="hint">
                  Model id for <code>/v1/chat/completions</code>: <code>{benchmarkModelId}</code> — the
                  current Configure preset name. Change preset on the Configure tab to change the model.
                </p>
                <label>
                  Prompt{" "}
                  <textarea
                    className="textarea"
                    rows={3}
                    value={benchPrompt}
                    onChange={(e) => setBenchPrompt(e.target.value)}
                    spellCheck={false}
                  />
                </label>
                <div className="tool-row2">
                  <label>
                    Max tokens{" "}
                    <input
                      type="number"
                      value={benchMaxTokens}
                      onChange={(e) => setBenchMaxTokens(parseInt(e.target.value, 10) || 64)}
                      min={1}
                    />
                  </label>
                  <label>
                    Requests{" "}
                    <input
                      type="number"
                      value={benchRequests}
                      onChange={(e) => setBenchRequests(parseInt(e.target.value, 10) || 20)}
                      min={1}
                    />
                  </label>
                  <label>
                    Timeout (s){" "}
                    <input
                      type="number"
                      value={benchTimeoutSec}
                      onChange={(e) => setBenchTimeoutSec(parseInt(e.target.value, 10) || 120)}
                      min={1}
                    />
                  </label>
                </div>
              </div>
            ) : null}

            {toolId === "measure" ? (
              <div className="tool-fields">
                <label>
                  Hosts (comma-separated, optional){" "}
                  <input
                    type="text"
                    value={measureHostsText}
                    onChange={(e) => setMeasureHostsText(e.target.value)}
                    placeholder="Leave empty to use Launch solo host or cluster list"
                    spellCheck={false}
                  />
                </label>
                <p className="hint">
                  If empty, hosts default to the Launch tab: solo SSH host, or cluster list in cluster
                  mode. With none set, the CLI measures locally or uses <code>.env</code> nodes.
                </p>
              </div>
            ) : null}

            {toolId === "pull" && runtime === "docker" ? (
              <div className="tool-fields">
                <p className="hint">
                  Uses the <strong>{preset || "(no preset)"}</strong> preset for the image when{" "}
                  <code>--preset</code> is set. Optional cluster hosts from the Launch tab are passed
                  as <code>--hosts</code>; otherwise the CLI uses <code>.env</code> or local pull.
                </p>
              </div>
            ) : null}

            {toolId === "deploy" && runtime === "venv" ? (
              <div className="tool-fields">
                <label>
                  Deploy set name (<code>--set</code>, optional){" "}
                  <input
                    type="text"
                    value={deploySetName}
                    onChange={(e) => setDeploySetName(e.target.value)}
                    placeholder="from deploy_sets.json"
                    spellCheck={false}
                  />
                </label>
                <p className="hint">
                  Hosts default to the cluster list; if that is empty, the solo SSH host from Launch is
                  used.
                </p>
              </div>
            ) : null}

            <div className="btnrow">
              <button type="button" onClick={() => void runTool()}>
                Run tool
              </button>
            </div>
          </section>
        ) : null}

        {tab === "configure" ? (
          <section className="card tab-card" aria-labelledby="runtime-panel-title">
            <h2 id="runtime-panel-title">Runtime</h2>
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
            <p className="hint">
              Active presets file: <code>{presetsPathLabel}</code>. Optional env:{" "}
              <code>{runtimeLabel}/.env</code>. Used by Launch, Stop, Logs, Scan, and Tools.
            </p>
          </section>
        ) : null}

        <section className="card tab-card output-card">
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
