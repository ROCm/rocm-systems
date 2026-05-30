import { useEffect, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import * as api from "../api/client";
import type { SessionState } from "../api/types";

export function SessionsPage() {
  const [sessions, setSessions] = useState<SessionState[]>([]);
  const [profiles, setProfiles] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [newProfile, setNewProfile] = useState("");

  async function refresh() {
    try {
      const [s, p] = await Promise.all([api.listSessions(), api.listProfiles()]);
      setSessions(s);
      setProfiles(p);
      if (!newProfile && p.length > 0) setNewProfile(p[0]);
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, []);

  async function onStart(e: FormEvent) {
    e.preventDefault();
    if (!newProfile) {
      setError("create a profile first");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await api.createSession({ profile: newProfile, ready_timeout: 10 });
      await refresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function onStop(id: string) {
    if (!confirm(`Stop session ${id}?`)) return;
    setBusy(true);
    try {
      await api.deleteSession(id);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <h2>Sessions</h2>
      {error && <div className="error" role="alert">{error}</div>}

      <h3>Active</h3>
      {sessions.length === 0 ? (
        <p data-testid="no-sessions">(no sessions)</p>
      ) : (
        <table className="data-table" data-testid="sessions-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Profile</th>
              <th>Health</th>
              <th>State</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => (
              <tr key={s.def.id} data-testid={`session-row-${s.def.id}`}>
                <td>
                  <Link to={`/sessions/${encodeURIComponent(s.def.id)}`}>
                    <code>{s.def.id}</code>
                  </Link>
                </td>
                <td>{profileName(s.def.profile)}</td>
                <td>{s.health.healthy ? "healthy" : "starting"}</td>
                <td>{s.health.state ?? ""}</td>
                <td>
                  <button
                    type="button"
                    onClick={() => onStop(s.def.id)}
                    disabled={busy}
                    data-testid={`stop-session-${s.def.id}`}
                  >
                    Stop
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h3>Start a session</h3>
      <form onSubmit={onStart} className="form" data-testid="start-session">
        <label>
          Profile
          <select
            value={newProfile}
            onChange={(e) => setNewProfile(e.target.value)}
            data-testid="new-session-profile"
          >
            {profiles.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
        <button
          type="submit"
          disabled={busy || profiles.length === 0}
          data-testid="submit-session"
        >
          Start
        </button>
      </form>
    </div>
  );
}

function profileName(p: unknown): string {
  if (typeof p === "string") return p;
  if (
    p &&
    typeof p === "object" &&
    "name" in (p as Record<string, unknown>) &&
    typeof (p as { name: unknown }).name === "string"
  ) {
    return (p as { name: string }).name;
  }
  return String(p);
}
