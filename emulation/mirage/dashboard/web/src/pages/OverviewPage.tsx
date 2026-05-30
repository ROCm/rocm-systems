import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import * as api from "../api/client";
import type { PathsInfo } from "../api/types";

export function OverviewPage() {
  const [paths, setPaths] = useState<PathsInfo | null>(null);
  const [profileCount, setProfileCount] = useState(0);
  const [sessionCount, setSessionCount] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api.getPaths(), api.listProfiles(), api.listSessions()])
      .then(([p, profiles, sessions]) => {
        setPaths(p);
        setProfileCount(profiles.length);
        setSessionCount(sessions.length);
      })
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="page">
      <h2>Overview</h2>
      {error && <div className="error">{error}</div>}
      <div className="card-grid">
        <Link className="stat-card" to="/profiles" data-testid="profile-stat">
          <span className="stat-value">{profileCount}</span>
          <span className="stat-label">Profiles</span>
        </Link>
        <Link className="stat-card" to="/sessions" data-testid="session-stat">
          <span className="stat-value">{sessionCount}</span>
          <span className="stat-label">Sessions</span>
        </Link>
      </div>
      <h3>Storage</h3>
      {paths && (
        <table className="data-table">
          <tbody>
            <tr>
              <th>Config</th>
              <td>
                <code>{paths.config}</code>
              </td>
            </tr>
            <tr>
              <th>Runtime</th>
              <td>
                <code>{paths.runtime}</code>
              </td>
            </tr>
            <tr>
              <th>State</th>
              <td>
                <code>{paths.state}</code>
              </td>
            </tr>
            <tr>
              <th>Profiles dir</th>
              <td>
                <code>{paths.profiles}</code>
              </td>
            </tr>
            <tr>
              <th>Sessions dir</th>
              <td>
                <code>{paths.sessions}</code>
              </td>
            </tr>
          </tbody>
        </table>
      )}
    </div>
  );
}
