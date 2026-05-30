import type { SessionPhase } from "../api/types";

const colors: Record<SessionPhase, string> = {
  Pulling: "#4a6cf7",
  Starting: "#f0a020",
  Running: "var(--status-healthy)",
  Failed: "var(--status-unhealthy)",
  ShuttingDown: "#b06030",
  Stale: "#888888",
};

export function PhaseBadge({
  phase,
  message,
}: {
  phase: SessionPhase;
  message?: string;
}) {
  const label = message ? `${phase} · ${message}` : phase;
  return (
    <span
      className="status-badge"
      style={{ background: colors[phase] ?? colors.Running }}
      title={message || phase}
    >
      {label}
    </span>
  );
}
