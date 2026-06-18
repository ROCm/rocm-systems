import type { ReactNode } from "react";

export type StatusTone = "ok" | "warn" | "bad" | "muted";

export function StatusDot(props: { tone: StatusTone; ariaLabel?: string }) {
  return (
    <span
      className={`status-dot status-dot-${props.tone}`}
      role="img"
      aria-label={props.ariaLabel ?? props.tone}
    />
  );
}

export function Pill(props: { tone: StatusTone; children: ReactNode; testId?: string }) {
  return (
    <span className={`pill pill-${props.tone}`} data-testid={props.testId}>
      {props.children}
    </span>
  );
}
