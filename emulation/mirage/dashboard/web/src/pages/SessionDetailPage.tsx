import { useEffect, useState, useRef } from "react";
import { useParams, Link } from "react-router-dom";
import { getSessionDetail, getSessionLog } from "../api/client";
import type { SessionDetail } from "../api/types";
import { StatusBadge } from "../components/StatusBadge";
import { PhaseBadge } from "../components/PhaseBadge";

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function LogStatusBadge({ status }: { status: string }) {
  const map: Record<string, { label: string; cls: string }> = {
    pulling: { label: "Pulling Image", cls: "badge-pulling" },
    starting: { label: "Starting", cls: "badge-starting" },
    ready: { label: "Ready", cls: "badge-ready" },
    error: { label: "Error", cls: "badge-error" },
  };
  const info = map[status] ?? { label: status || "–", cls: "" };
  return <span className={`session-log-badge ${info.cls}`}>{info.label}</span>;
}

export function SessionDetailPage() {
  const { name } = useParams<{ name: string }>();
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [error, setError] = useState("");
  const [sessionLog, setSessionLog] = useState("");
  const [logStatus, setLogStatus] = useState("");
  const logRef = useRef<HTMLPreElement>(null);

  // Poll session detail (fast during pulling/starting)
  useEffect(() => {
    if (!name) return;
    getSessionDetail(name)
      .then(setDetail)
      .catch((e) => setError(String(e)));

    const booting =
      detail?.phase === "Pulling" || detail?.phase === "Starting";
    const interval = booting ? 1000 : 5000;
    const id = setInterval(() => {
      getSessionDetail(name).then(setDetail).catch(() => {});
    }, interval);
    return () => clearInterval(id);
  }, [name, detail?.phase]);

  // Poll session log (fast while pulling, slow once ready)
  useEffect(() => {
    if (!name) return;
    let active = true;
    const poll = () => {
      if (!active) return;
      getSessionLog(name).then((snap) => {
        if (!active) return;
        setSessionLog(snap.log);
        setLogStatus(snap.status);
        // Poll fast while pulling/starting, slow once ready/error
        const interval =
          snap.status === "pulling" ? 500 :
          snap.status === "starting" ? 1000 : 5000;
        setTimeout(poll, interval);
      }).catch(() => {
        if (active) setTimeout(poll, 5000);
      });
    };
    poll();
    return () => {
      active = false;
    };
  }, [name]);

  // Auto-scroll log to bottom
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [sessionLog]);

  if (error) return <div className="error">{error}</div>;
  if (!detail) return <div className="loading">Loading...</div>;

  return (
    <div className="page">
      <Link to="/sessions" className="back-link">
        &larr; Sessions
      </Link>
      <div className="page-header">
        <h2>{detail.name}</h2>
        <PhaseBadge phase={detail.phase} message={detail.progress_message} />
        <StatusBadge status={detail.health} />
      </div>

      {detail.error_message && (
        <div className="error">{detail.error_message}</div>
      )}

      <div className="detail-grid">
        <div className="detail-item">
          <label>Simulator</label>
          <span>{detail.simulator}</span>
        </div>
        <div className="detail-item">
          <label>Profile</label>
          <span>{detail.profile.name}</span>
        </div>
        <div className="detail-item">
          <label>GPU</label>
          <span>
            <code>{detail.profile.gpu}</code>
          </span>
        </div>
        <div className="detail-item">
          <label>Mode</label>
          <span>{detail.profile.mode}</span>
        </div>
        <div className="detail-item">
          <label>Cluster</label>
          <span>
            {detail.profile.num_gpus} GPU(s) &times;{" "}
            {detail.profile.num_nodes} node(s)
          </span>
        </div>
        <div className="detail-item">
          <label>Image</label>
          <span>
            <code>{detail.image}</code>
          </span>
        </div>
      </div>

      <h3>Performance</h3>
      <div className="card-grid">
        <div className="stat-card">
          <span className="stat-value">
            {formatUptime(detail.uptime.seconds)}
          </span>
          <span className="stat-label">Uptime</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{detail.ticks.toLocaleString()}</span>
          <span className="stat-label">Ticks</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{detail.ipc.toFixed(2)}</span>
          <span className="stat-label">IPC</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">
            {detail.simulation_speed.toFixed(2)}x
          </span>
          <span className="stat-label">Speed</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{detail.active_contexts}</span>
          <span className="stat-label">Active Contexts</span>
        </div>
      </div>

      <div className="session-log-header">
        <h3>Docker Output</h3>
        <LogStatusBadge status={logStatus} />
      </div>
      <pre className="session-log" ref={logRef}>
        {sessionLog || "Waiting for output…"}
      </pre>
    </div>
  );
}
