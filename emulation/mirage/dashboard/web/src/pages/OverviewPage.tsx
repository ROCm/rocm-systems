import { useEffect, useState } from "react";
import { getOverview } from "../api/client";
import type { OverviewData } from "../api/types";

export function OverviewPage() {
  const [data, setData] = useState<OverviewData | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getOverview().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!data) return <div className="loading">Loading...</div>;

  return (
    <div className="page">
      <h2>Overview</h2>
      <div className="card-grid">
        <div className="stat-card">
          <span className="stat-value">{data.simulator_count}</span>
          <span className="stat-label">Simulators</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{data.profile_count}</span>
          <span className="stat-label">Profiles</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{data.session_count}</span>
          <span className="stat-label">Sessions</span>
        </div>
      </div>
    </div>
  );
}
