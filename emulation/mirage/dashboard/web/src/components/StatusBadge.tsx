import type { HealthStatus } from "../api/types";

const colors: Record<HealthStatus, string> = {
  Healthy: "var(--status-healthy)",
  Unhealthy: "var(--status-unhealthy)",
  Unknown: "var(--status-unknown)",
};

export function StatusBadge({ status }: { status: HealthStatus }) {
  return (
    <span
      className="status-badge"
      style={{ background: colors[status] ?? colors.Unknown }}
    >
      {status}
    </span>
  );
}
