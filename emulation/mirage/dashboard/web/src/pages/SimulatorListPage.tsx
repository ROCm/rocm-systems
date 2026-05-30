import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listSimulators } from "../api/client";
import type { SimulatorSummary } from "../api/types";

export function SimulatorListPage() {
  const [sims, setSims] = useState<SimulatorSummary[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    listSimulators().then(setSims).catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="error">{error}</div>;

  return (
    <div className="page">
      <h2>Simulators</h2>
      {sims.length === 0 ? (
        <p className="empty">No simulators registered.</p>
      ) : (
        <div className="card-grid">
          {sims.map((s) => (
            <Link
              to={`/simulators/${encodeURIComponent(s.name)}`}
              className="card"
              key={s.name}
            >
              <h3>{s.name}</h3>
              <span className="version">v{s.version}</span>
              <p>{s.description}</p>
              <div className="card-meta">
                <span>{s.supported_gpus.length} GPUs</span>
                <span>{s.active_session_count} sessions</span>
                <span>
                  {s.supported_modes.join(", ")}
                </span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
