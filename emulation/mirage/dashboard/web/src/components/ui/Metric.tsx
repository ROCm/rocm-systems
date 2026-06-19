import type { ReactNode } from "react";
import { Link } from "react-router-dom";

export interface MetricProps {
  label: string;
  value: number | string;
  hint?: ReactNode;
  to?: string;
  testId?: string;
  tone?: "default" | "ok" | "warn" | "bad";
}

export function Metric(props: MetricProps) {
  const { label, value, hint, to, testId, tone = "default" } = props;
  const inner = (
    <>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
      {hint && <span className="metric-hint">{hint}</span>}
    </>
  );
  if (to) {
    return (
      <Link to={to} className={`metric-card metric-${tone}`} data-testid={testId}>
        {inner}
      </Link>
    );
  }
  return (
    <div className={`metric-card metric-${tone}`} data-testid={testId}>
      {inner}
    </div>
  );
}
