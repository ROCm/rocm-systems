import { useEffect, useState, useCallback } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  listSessions,
  listProfiles,
  createSession,
  deleteSession,
} from "../api/client";
import type { SessionSummary, ProfileDef } from "../api/types";
import { StatusBadge } from "../components/StatusBadge";
import { PhaseBadge } from "../components/PhaseBadge";

export function SessionListPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [profiles, setProfiles] = useState<ProfileDef[]>([]);
  const [error, setError] = useState("");
  const [showForm, setShowForm] = useState(false);
  const navigate = useNavigate();

  const refresh = useCallback(() => {
    listSessions().then(setSessions).catch((e) => setError(String(e)));
  }, []);

  useEffect(refresh, [refresh]);

  // Poll for updates while any session is still booting.
  useEffect(() => {
    const booting = sessions.some(
      (s) => s.phase === "Pulling" || s.phase === "Starting",
    );
    if (!booting) return;
    const h = setInterval(refresh, 1000);
    return () => clearInterval(h);
  }, [sessions, refresh]);

  useEffect(() => {
    listProfiles().then(setProfiles).catch(() => {});
  }, []);

  const handleCreate = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const fd = new FormData(e.currentTarget);
    const session = {
      name: fd.get("name") as string,
      profile: fd.get("profile") as string,
      image: fd.get("image") as string,
    };
    const res = await createSession(session);
    if (!res.ok) {
      setError(res.error);
    } else {
      // Navigate to session detail to see docker pull/run progress
      navigate(`/sessions/${encodeURIComponent(session.name)}`);
    }
  };

  const handleDelete = async (name: string) => {
    const res = await deleteSession(name);
    if (!res.ok) {
      setError(res.error);
    } else {
      refresh();
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <h2>Sessions</h2>
        <button className="btn-primary" onClick={() => setShowForm(!showForm)}>
          {showForm ? "Cancel" : "+ New Session"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {showForm && (
        <form className="create-form labeled-form" onSubmit={handleCreate}>
          <div className="form-field">
            <label htmlFor="sf-name">Session Name</label>
            <input id="sf-name" name="name" required />
          </div>
          <div className="form-field">
            <label htmlFor="sf-profile">Profile</label>
            <select id="sf-profile" name="profile" required>
              <option value="" disabled selected>
                Select profile
              </option>
              {profiles.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name} — {p.simulator} / {p.gpu}
                </option>
              ))}
            </select>
          </div>
          <div className="form-field">
            <label htmlFor="sf-image">Container Image</label>
            <input
              id="sf-image"
              name="image"
              placeholder="e.g. pytorch:latest"
            />
          </div>
          <div className="form-field form-actions">
            <button type="submit" className="btn-primary">
              Create
            </button>
          </div>
        </form>
      )}

      {sessions.length === 0 ? (
        <p className="empty">No sessions.</p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Profile</th>
              <th>Simulator</th>
              <th>Image</th>
              <th>Phase</th>
              <th>Health</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => (
              <tr key={s.name}>
                <td>
                  <Link to={`/sessions/${encodeURIComponent(s.name)}`}>
                    <strong>{s.name}</strong>
                  </Link>
                </td>
                <td>{s.profile}</td>
                <td>{s.simulator}</td>
                <td>
                  <code>{s.image}</code>
                </td>
                <td>
                  <PhaseBadge
                    phase={s.phase}
                    message={s.progress_message}
                  />
                </td>
                <td>
                  <StatusBadge status={s.health_status} />
                </td>
                <td>
                  <button
                    className="btn-danger-sm"
                    onClick={() => handleDelete(s.name)}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
