import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import { Link, useParams } from "react-router-dom";
import { FitAddon } from "@xterm/addon-fit";
import { Terminal as XTerm } from "@xterm/xterm";
import "@xterm/xterm/css/xterm.css";
import * as api from "../api/client";
import type { ExecListItem, SessionState, StreamPacket } from "../api/types";
import { Pill, StatusDot } from "../components/ui/Status";
import { useToast } from "../components/ui/Toast";

function execTone(e: ExecListItem): "ok" | "warn" | "bad" | "muted" {
  if (!e.status.started) return "muted";
  if (!e.status.ended) return "warn";
  if (e.status.exit_code === 0) return "ok";
  return "bad";
}

function execLabel(e: ExecListItem): string {
  if (!e.status.started) return "queued";
  if (!e.status.ended) return "running";
  return `exit ${e.status.exit_code ?? "?"}`;
}

/**
 * Split a command line into argv the way a POSIX shell would, honoring
 * single quotes, double quotes, and backslash escapes. Throws on an
 * unterminated quote so the caller can surface a clear error instead of
 * silently passing a malformed argv to the host (which would then fail
 * with confusing shell parse errors like `Syntax error: Unterminated
 * quoted string`).
 */
export function splitCommand(input: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inSingle = false;
  let inDouble = false;
  let hasToken = false;
  for (let i = 0; i < input.length; i++) {
    const c = input[i];
    if (inSingle) {
      if (c === "'") {
        inSingle = false;
      } else {
        cur += c;
      }
      continue;
    }
    if (inDouble) {
      if (c === "\\" && i + 1 < input.length) {
        const next = input[i + 1];
        if (next === '"' || next === "\\" || next === "$" || next === "`") {
          cur += next;
          i++;
        } else {
          cur += c;
        }
      } else if (c === '"') {
        inDouble = false;
      } else {
        cur += c;
      }
      continue;
    }
    if (c === "'") {
      inSingle = true;
      hasToken = true;
    } else if (c === '"') {
      inDouble = true;
      hasToken = true;
    } else if (c === "\\" && i + 1 < input.length) {
      cur += input[i + 1];
      hasToken = true;
      i++;
    } else if (c === " " || c === "\t" || c === "\n") {
      if (hasToken) {
        out.push(cur);
        cur = "";
        hasToken = false;
      }
    } else {
      cur += c;
      hasToken = true;
    }
  }
  if (inSingle || inDouble) {
    throw new Error(`unterminated ${inSingle ? "single" : "double"} quote`);
  }
  if (hasToken) out.push(cur);
  return out;
}

function shortAge(iso?: string): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "—";
  const s = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h`;
}

export function SessionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const toast = useToast();
  const [session, setSession] = useState<SessionState | null>(null);
  const [execs, setExecs] = useState<ExecListItem[]>([]);
  const [error, setError] = useState("");
  const [command, setCommand] = useState("/bin/sh -c 'echo hello'");
  const [keep, setKeep] = useState(true);
  const [envInput, setEnvInput] = useState("");
  const [envs, setEnvs] = useState<string[]>([]);
  const [activeExecId, setActiveExecId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!id) return;
    try {
      const [s, e] = await Promise.all([api.getSession(id), api.listExecs(id)]);
      setSession(s);
      setExecs(e);
    } catch (err) {
      setError(String(err));
    }
  }, [id]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, [refresh]);

  function addEnv() {
    const v = envInput.trim();
    if (!v.includes("=")) {
      toast.error("env must be KEY=VALUE");
      return;
    }
    setEnvs((cur) => [...cur, v]);
    setEnvInput("");
  }

  async function onRun(e: FormEvent) {
    e.preventDefault();
    if (!id) return;
    setError("");
    let parts: string[];
    try {
      parts = splitCommand(command);
    } catch (err) {
      const msg = `invalid command: ${err instanceof Error ? err.message : String(err)}`;
      setError(msg);
      toast.error(msg);
      return;
    }
    if (!parts.length || !parts[0]) {
      const msg = "command is required";
      setError(msg);
      toast.error(msg);
      return;
    }
    try {
      const r = await api.createExec(id, {
        command: parts[0],
        args: parts.slice(1),
        keep,
      });
      setActiveExecId(r.id);
      toast.success(`Exec ${r.id} started`);
      await refresh();
    } catch (err) {
      const msg = String(err);
      setError(msg);
      toast.error(msg);
    }
  }

  async function onRemove(execId: string) {
    if (!id) return;
    try {
      await api.deleteExec(id, execId);
      if (activeExecId === execId) setActiveExecId(null);
      toast.info(`Exec ${execId} removed`);
      await refresh();
    } catch (e) {
      const msg = String(e);
      setError(msg);
      toast.error(msg);
    }
  }

  // SIGTERM, the conventional "please stop" signal. The host forwards it
  // to every node process belonging to the exec.
  async function onKill(execId: string) {
    if (!id) return;
    try {
      await api.signalExec(id, execId, 15);
      toast.info(`Sent SIGTERM to ${execId}`);
      await refresh();
    } catch (e) {
      const msg = String(e);
      setError(msg);
      toast.error(msg);
    }
  }

  if (!id) return <div className="page">missing id</div>;

  const healthy = session?.health.healthy ?? false;
  const stateLabel = session?.health.state ?? (healthy ? "ready" : "pending");

  return (
    <div className="page page-wide">
      <Link to="/sessions" className="back-link">
        &larr; back to sessions
      </Link>

      <div className="session-hero">
        <div>
          <h2 className="session-hero-id">
            Session <code>{id}</code>
          </h2>
          {session && (
            <div className="session-hero-meta">
              <span>Workdir <code>{session.def.workdir}</code></span>
              <span>·</span>
              <span>Created {shortAge(session.def.created_at)} ago</span>
            </div>
          )}
        </div>
        <div className="session-hero-status">
          <StatusDot
            tone={healthy ? "ok" : "warn"}
            ariaLabel={healthy ? "healthy" : "starting"}
          />
          <Pill tone={healthy ? "ok" : "warn"}>{stateLabel}</Pill>
        </div>
      </div>

      {error && <div className="error" role="alert">{error}</div>}

      <div className="session-grid-2col">
        <section className="panel">
          <header className="panel-header">
            <h3>Execs</h3>
            <span className="muted">{execs.length} total</span>
          </header>
          {execs.length === 0 ? (
            <div className="empty-state-sm" data-testid="no-execs">
              No execs yet. Run a command below.
            </div>
          ) : (
            <ul className="exec-list" data-testid="execs-table">
              {execs.map((e) => {
                const tone = execTone(e);
                return (
                  <li
                    key={e.id}
                    className={`exec-row${activeExecId === e.id ? " is-active" : ""}`}
                    data-testid={`exec-row-${e.id}`}
                  >
                    <button
                      type="button"
                      className="exec-row-main"
                      onClick={() => setActiveExecId(e.id)}
                      data-testid={`attach-${e.id}`}
                    >
                      <StatusDot tone={tone} ariaLabel={execLabel(e)} />
                      <div>
                        <div className="exec-row-id"><code>{e.id}</code></div>
                        <div className="exec-row-meta">
                          {execLabel(e)} · {shortAge(e.status.started_at)} ago
                        </div>
                      </div>
                    </button>
                    {e.status.started && !e.status.ended && (
                      <button
                        type="button"
                        className="btn-danger-sm"
                        onClick={() => onKill(e.id)}
                        data-testid={`kill-${e.id}`}
                      >
                        Kill
                      </button>
                    )}
                    <button
                      type="button"
                      className="btn-danger-sm"
                      onClick={() => onRemove(e.id)}
                      data-testid={`remove-${e.id}`}
                    >
                      Remove
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="panel terminal-panel">
          <header className="panel-header">
            <h3>{activeExecId ? `Attached: ${activeExecId}` : "Terminal"}</h3>
            {activeExecId && (
              <button
                type="button"
                className="link-button"
                onClick={() => setActiveExecId(null)}
                data-testid="detach-exec"
              >
                detach
              </button>
            )}
          </header>
          {activeExecId ? (
            <AttachView sessionId={id} execId={activeExecId} />
          ) : (
            <div className="empty-state-sm">
              Pick an exec on the left or run a new one to attach.
            </div>
          )}
        </section>
      </div>

      <section className="panel command-palette">
        <header className="panel-header">
          <h3>Run a command</h3>
          <span className="muted">Streams stdout/stderr into the terminal</span>
        </header>
        <form onSubmit={onRun} className="command-form" data-testid="run-exec">
          <input
            className="command-input"
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            data-testid="exec-command"
            placeholder="/bin/sh -c 'echo hi'"
          />
          <label className="keep-toggle">
            <input
              type="checkbox"
              checked={keep}
              onChange={(e) => setKeep(e.target.checked)}
              data-testid="exec-keep"
            />
            keep
          </label>
          <button
            type="submit"
            className="btn-primary"
            data-testid="submit-exec"
          >
            Run
          </button>
        </form>
        <div className="env-chip-row">
          {envs.map((kv, i) => (
            <span
              key={`${kv}-${i}`}
              className="env-chip"
              data-testid={`env-chip-${i}`}
            >
              {kv}
              <button
                type="button"
                aria-label={`remove ${kv}`}
                onClick={() => setEnvs((cur) => cur.filter((_, j) => j !== i))}
              >
                ×
              </button>
            </span>
          ))}
          <input
            className="env-input"
            placeholder="KEY=VALUE"
            value={envInput}
            data-testid="env-input"
            onChange={(e) => setEnvInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                addEnv();
              }
            }}
          />
        </div>
      </section>
    </div>
  );
}

function AttachView(props: { sessionId: string; execId: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<XTerm | null>(null);
  const fitRef = useRef<FitAddon | null>(null);
  const [output, setOutput] = useState("");
  const [exitCode, setExitCode] = useState<number | null>(null);

  // Mount xterm once per (session, exec).
  useEffect(() => {
    if (!containerRef.current) return;
    const term = new XTerm({
      convertEol: true,
      cursorBlink: true,
      fontFamily:
        '"JetBrains Mono", "Fira Code", "Cascadia Code", Consolas, monospace',
      fontSize: 13,
      theme: { background: "#0d0d0d", foreground: "#e8e8e8" },
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(containerRef.current);
    try {
      fit.fit();
    } catch {
      /* container not measurable yet */
    }
    termRef.current = term;
    fitRef.current = fit;
    // Forward every keystroke typed in the terminal to the program's
    // stdin. xterm emits already-encoded byte sequences (e.g. "\r" for
    // Enter, "\x7f" for Backspace), so we pass them through verbatim.
    // Each keystroke is a separate HTTP POST; to preserve input order we
    // chain the sends so the next one starts only after the previous has
    // been accepted by the daemon (otherwise fast typing can race and
    // arrive scrambled).
    let stdinChain: Promise<unknown> = Promise.resolve();
    const dataSub = term.onData((data) => {
      stdinChain = stdinChain
        .then(() => api.stdinExec(props.sessionId, props.execId, data))
        .catch(() => {
          /* exec may have exited; ignore stdin write failures */
        });
    });
    const ro = new ResizeObserver(() => {
      try {
        fit.fit();
      } catch {
        /* ignore */
      }
    });
    ro.observe(containerRef.current);
    return () => {
      dataSub.dispose();
      ro.disconnect();
      term.dispose();
      termRef.current = null;
      fitRef.current = null;
    };
  }, [props.sessionId, props.execId]);

  useEffect(() => {
    setOutput("");
    setExitCode(null);
    if (termRef.current) {
      termRef.current.clear();
    }
    const ws = new WebSocket(api.attachUrl(props.sessionId, props.execId));
    ws.onmessage = (ev) => {
      try {
        const pkt = JSON.parse(ev.data as string) as StreamPacket;
        if ("Output" in pkt) {
          const text = new TextDecoder().decode(
            new Uint8Array(pkt.Output.data),
          );
          if (termRef.current) termRef.current.write(text);
          setOutput((cur) => cur + text);
        } else if ("ExecExit" in pkt) {
          setExitCode(pkt.ExecExit.exit_code);
        }
      } catch {
        // malformed frames are dropped
      }
    };
    return () => {
      ws.close();
    };
  }, [props.sessionId, props.execId]);

  const exitTone = useMemo<"ok" | "bad" | "muted">(() => {
    if (exitCode === null) return "muted";
    return exitCode === 0 ? "ok" : "bad";
  }, [exitCode]);

  return (
    <>
      <div className="xterm-frame" ref={containerRef} />
      {/* Mirror output as plain text for accessibility + tests. */}
      <pre
        className="attach-output-mirror"
        data-testid="attach-output"
        aria-live="polite"
      >
        {output || "(no output yet)"}
      </pre>
      {exitCode !== null && (
        <div className="attach-exit" data-testid="attach-exit">
          <Pill tone={exitTone}>exit {exitCode}</Pill>
        </div>
      )}
    </>
  );
}
