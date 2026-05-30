import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getSimulator } from "../api/client";
import type { SimulatorSummary } from "../api/types";

export function SimulatorDetailPage() {
  const { name } = useParams<{ name: string }>();
  const [sim, setSim] = useState<SimulatorSummary | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (name) {
      getSimulator(name)
        .then(setSim)
        .catch((e) => setError(String(e)));
    }
  }, [name]);

  if (error) return <div className="error">{error}</div>;
  if (!sim) return <div className="loading">Loading...</div>;

  return (
    <div className="page">
      <Link to="/simulators" className="back-link">
        &larr; Simulators
      </Link>
      <h2>{sim.name}</h2>
      <span className="version">v{sim.version}</span>
      {sim.description && <p>{sim.description}</p>}

      <div className="detail-grid">
        <div className="detail-item">
          <label>Custom GPUs</label>
          <span>{sim.supports_custom_gpus ? "Supported" : "Not supported"}</span>
        </div>
        <div className="detail-item">
          <label>Modes</label>
          <span>{sim.supported_modes.join(", ")}</span>
        </div>
        <div className="detail-item">
          <label>Active Sessions</label>
          <span>{sim.active_session_count}</span>
        </div>
      </div>

      <h3>Supported GPUs</h3>
      {sim.supported_gpus.length === 0 ? (
        <p className="empty">No GPUs.</p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Architecture</th>
              <th>Family</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            {sim.supported_gpus.map((g) => (
              <tr key={g.name}>
                <td><strong>{g.name}</strong></td>
                <td><code>{g.arch}</code></td>
                <td>{g.family}</td>
                <td>{g.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
