import { useEffect, useState, useCallback } from "react";
import {
  listTerminals,
  createTerminal,
  closeTerminal,
  listSessions,
} from "../api/client";
import type { TerminalInfo, SessionSummary } from "../api/types";
import { TerminalView } from "../components/TerminalView";

export function RunListPage() {
  const [terminals, setTerminals] = useState<TerminalInfo[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [error, setError] = useState("");
  const [activeTerminal, setActiveTerminal] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  const refresh = useCallback(() => {
    listTerminals().then(setTerminals).catch((e) => setError(String(e)));
  }, []);

  useEffect(refresh, [refresh]);

  useEffect(() => {
    listSessions().then(setSessions).catch(() => {});
  }, []);

  const handleCreate = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError("");
    const fd = new FormData(e.currentTarget);
    const session = fd.get("session") as string;
    const nodeRaw = fd.get("node_index") as string | null;
    const nodeIndex = Math.max(0, Number.parseInt(nodeRaw ?? "0", 10) || 0);
    const program = (fd.get("program") as string | null)?.trim() || "/bin/sh";

    try {
      const res = await createTerminal(session, nodeIndex, program);
      if (!res.ok) {
        setError(res.error ?? "Failed to create terminal");
      } else {
        setShowForm(false);
        setActiveTerminal(res.id);
        refresh();
      }
    } catch (err) {
      setError(String(err));
    }
  };

  const handleClose = async (id: string) => {
    await closeTerminal(id);
    if (activeTerminal === id) setActiveTerminal(null);
    refresh();
  };

  const runningSessions = sessions.filter(
    (s) => s.health_status === "Healthy"
  );

  // If a terminal is active, show it full-screen
  if (activeTerminal) {
    return (
      <div className="page terminal-page">
        <div className="page-header">
          <h2>Terminal</h2>
          <div style={{ display: "flex", gap: "8px" }}>
            <button
              className="btn-secondary"
              onClick={() => setActiveTerminal(null)}
            >
              ← Back
            </button>
            <button
              className="btn-danger-sm"
              onClick={() => handleClose(activeTerminal)}
            >
              Kill Terminal
            </button>
          </div>
        </div>
        <TerminalView
          terminalId={activeTerminal}
          onClose={() => handleClose(activeTerminal)}
          onDead={() => refresh()}
        />
      </div>
    );
  }

  return (
    <div className="page">
      <div className="page-header">
        <h2>Runs</h2>
        <button className="btn-primary" onClick={() => setShowForm(!showForm)}>
          {showForm ? "Cancel" : "+ New Terminal"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {showForm && (
        <form className="create-form labeled-form" onSubmit={handleCreate}>
          <div className="form-field">
            <label htmlFor="tf-session">Session</label>
            <select id="tf-session" name="session" required>
              <option value="" disabled selected>
                Select running session
              </option>
              {runningSessions.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name} — {s.simulator} / {s.profile}
                </option>
              ))}
            </select>
          </div>
          <div className="form-field">
            <label htmlFor="tf-node">Node index</label>
            <input
              id="tf-node"
              name="node_index"
              type="number"
              min={0}
              defaultValue={0}
            />
          </div>
          <div className="form-field">
            <label htmlFor="tf-program">Program</label>
            <input
              id="tf-program"
              name="program"
              type="text"
              placeholder="/bin/sh"
              defaultValue="/bin/sh"
            />
          </div>
          <div className="form-field form-actions">
            <button type="submit" className="btn-primary">
              Open Terminal
            </button>
          </div>
        </form>
      )}

      {terminals.length === 0 ? (
        <p className="empty">
          No terminals. Create one to get an interactive shell inside a session
          container.
        </p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Session</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {terminals.map((t) => (
              <tr key={t.id}>
                <td>
                  <button
                    className="link-button"
                    onClick={() => setActiveTerminal(t.id)}
                  >
                    <strong>{t.id}</strong>
                  </button>
                </td>
                <td>{t.session}</td>
                <td>
                  <span
                    className={`badge ${
                      t.alive ? "badge-healthy" : "badge-unhealthy"
                    }`}
                  >
                    {t.alive ? "running" : "exited"}
                  </span>
                </td>
                <td>
                  <div style={{ display: "flex", gap: "6px" }}>
                    <button
                      className="btn-primary-sm"
                      onClick={() => setActiveTerminal(t.id)}
                    >
                      Attach
                    </button>
                    <button
                      className="btn-danger-sm"
                      onClick={() => handleClose(t.id)}
                    >
                      Close
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
