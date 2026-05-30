import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { Link, useParams } from "react-router-dom";
import * as api from "../api/client";
import type { ExecListItem, SessionState, StreamPacket } from "../api/types";

export function SessionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [session, setSession] = useState<SessionState | null>(null);
  const [execs, setExecs] = useState<ExecListItem[]>([]);
  const [error, setError] = useState("");
  const [command, setCommand] = useState("/bin/sh -c 'echo hello'");
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

  async function onRun(e: FormEvent) {
    e.preventDefault();
    if (!id) return;
    setError("");
    const parts = command.trim().split(/\s+/);
    if (!parts.length || !parts[0]) {
      setError("command is required");
      return;
    }
    try {
      const r = await api.createExec(id, {
        command: parts[0],
        args: parts.slice(1),
        keep: true,
      });
      setActiveExecId(r.id);
      await refresh();
    } catch (err) {
      setError(String(err));
    }
  }

  async function onRemove(execId: string) {
    if (!id) return;
    try {
      await api.deleteExec(id, execId);
      if (activeExecId === execId) setActiveExecId(null);
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  if (!id) return <div className="page">missing id</div>;

  return (
    <div className="page">
      <Link to="/sessions" className="back-link">
        &larr; back to sessions
      </Link>
      <h2>
        Session <code>{id}</code>
      </h2>
      {error && <div className="error" role="alert">{error}</div>}
      {session && (
        <p>
          <strong>Health:</strong> {session.health.healthy ? "healthy" : "not ready"}
          {" — "}
          <strong>State:</strong> {session.health.state ?? ""}
        </p>
      )}

      <h3>Execs</h3>
      {execs.length === 0 ? (
        <p data-testid="no-execs">(no execs)</p>
      ) : (
        <table className="data-table" data-testid="execs-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Started</th>
              <th>Ended</th>
              <th>Exit</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {execs.map((e) => (
              <tr key={e.id} data-testid={`exec-row-${e.id}`}>
                <td>
                  <code>{e.id}</code>
                </td>
                <td>{e.status.started ? "yes" : "no"}</td>
                <td>{e.status.ended ? "yes" : "no"}</td>
                <td>{e.status.exit_code ?? "-"}</td>
                <td>
                  <button
                    type="button"
                    onClick={() => setActiveExecId(e.id)}
                    data-testid={`attach-${e.id}`}
                  >
                    Attach
                  </button>
                  <button
                    type="button"
                    onClick={() => onRemove(e.id)}
                    data-testid={`remove-${e.id}`}
                  >
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h3>Run a command</h3>
      <form onSubmit={onRun} className="form" data-testid="run-exec">
        <label>
          Command
          <input
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            data-testid="exec-command"
          />
        </label>
        <button type="submit" data-testid="submit-exec">
          Run
        </button>
      </form>

      {activeExecId && id && (
        <AttachView sessionId={id} execId={activeExecId} />
      )}
    </div>
  );
}

function AttachView(props: { sessionId: string; execId: string }) {
  const [output, setOutput] = useState("");
  const [exitCode, setExitCode] = useState<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    setOutput("");
    setExitCode(null);
    const ws = new WebSocket(api.attachUrl(props.sessionId, props.execId));
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      try {
        const pkt = JSON.parse(ev.data as string) as StreamPacket;
        if ("Output" in pkt) {
          const text = new TextDecoder().decode(
            new Uint8Array(pkt.Output.data),
          );
          setOutput((cur) => cur + text);
        } else if ("ExecExit" in pkt) {
          setExitCode(pkt.ExecExit.exit_code);
        }
      } catch {
        // ignore malformed frames
      }
    };
    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [props.sessionId, props.execId]);

  return (
    <>
      <h3>Attached: {props.execId}</h3>
      <pre className="terminal-output" data-testid="attach-output">
        {output || "(no output yet)"}
      </pre>
      {exitCode !== null && (
        <p data-testid="attach-exit">Exited with code {exitCode}</p>
      )}
    </>
  );
}
