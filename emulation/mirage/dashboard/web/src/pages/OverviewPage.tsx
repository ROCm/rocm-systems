import { useEffect, useState } from "react";
import * as api from "../api/client";
import type { EmulatorEntry, Metrics, PathsInfo } from "../api/types";
import { Metric } from "../components/ui/Metric";
import { Pill, StatusDot } from "../components/ui/Status";
import { useToast } from "../components/ui/Toast";

export function OverviewPage() {
  const toast = useToast();
  const [paths, setPaths] = useState<PathsInfo | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [emulators, setEmulators] = useState<EmulatorEntry[]>([]);
  const [error, setError] = useState("");
  // Kept for back-compat with the original e2e suite which targeted
  // stat cards by these test-ids and labels.
  const profileCount = metrics?.profiles ?? 0;
  const sessionCount = metrics?.sessions ?? 0;

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [p, m, em] = await Promise.all([
          api.getPaths(),
          api.getMetrics(),
          api.listEmulators(),
        ]);
        if (cancelled) return;
        setPaths(p);
        setMetrics(m);
        setEmulators(em);
        setError("");
      } catch (e) {
        if (cancelled) return;
        const msg = String(e);
        setError(msg);
        toast.error(msg);
      }
    }
    load();
    const t = setInterval(load, 3000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
    // toast is stable from context; safe to omit
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="page page-wide">
      <div className="page-hero">
        <div>
          <h2>Overview</h2>
          <p className="page-subtitle">
            Live health of the mirage daemon, its profiles, and its sessions.
          </p>
        </div>
      </div>
      {error && <div className="error" role="alert">{error}</div>}

      <section className="metrics-grid" data-testid="metrics-grid">
        <Metric
          label="Profiles"
          value={profileCount}
          to="/profiles"
          testId="profile-stat"
          hint="Stored configurations"
        />
        <Metric
          label="Sessions"
          value={sessionCount}
          to="/sessions"
          testId="session-stat"
          hint={
            metrics
              ? `${metrics.sessions_healthy} healthy · ${metrics.sessions_starting} starting`
              : "—"
          }
        />
        <Metric
          label="Healthy"
          value={metrics?.sessions_healthy ?? 0}
          tone={metrics && metrics.sessions_healthy > 0 ? "ok" : "default"}
          testId="healthy-stat"
          hint="Sessions reporting ready"
        />
        <Metric
          label="Execs running"
          value={metrics?.execs_running ?? 0}
          tone={metrics && metrics.execs_running > 0 ? "ok" : "default"}
          testId="execs-stat"
          hint={metrics ? `${metrics.execs_total} total` : "—"}
        />
      </section>

      <section className="panel" data-testid="system-panel">
        <header className="panel-header">
          <h3>System</h3>
          <span className="muted">Detected emulator backends</span>
        </header>
        <ul className="emulator-list">
          {emulators.length === 0 && <li className="muted">(none registered)</li>}
          {emulators.map((e) => (
            <li
              key={e.name}
              className="emulator-row"
              data-testid={`emulator-row-${e.name}`}
            >
              <StatusDot
                tone={e.installed ? "ok" : "muted"}
                ariaLabel={e.installed ? "installed" : "not installed"}
              />
              <div className="emulator-name-block">
                <span className="emulator-name">
                  {e.name}
                  {e.is_default && (
                    <Pill tone="ok" testId={`emulator-default-${e.name}`}>
                      default
                    </Pill>
                  )}
                </span>
                <span className="emulator-desc">{e.description}</span>
              </div>
              <Pill
                tone={e.installed ? "ok" : "muted"}
                testId={`emulator-installed-${e.name}`}
              >
                {e.installed ? "installed" : "not installed"}
              </Pill>
            </li>
          ))}
        </ul>
      </section>

      <section className="panel" data-testid="storage-panel">
        <header className="panel-header">
          <h3>Storage</h3>
          <span className="muted">XDG-derived paths</span>
        </header>
        {paths && (
          <table className="data-table">
            <tbody>
              <tr>
                <th>Config</th>
                <td><code>{paths.config}</code></td>
              </tr>
              <tr>
                <th>Runtime</th>
                <td><code>{paths.runtime}</code></td>
              </tr>
              <tr>
                <th>State</th>
                <td><code>{paths.state}</code></td>
              </tr>
              <tr>
                <th>Profiles dir</th>
                <td><code>{paths.profiles}</code></td>
              </tr>
              <tr>
                <th>Sessions dir</th>
                <td><code>{paths.sessions}</code></td>
              </tr>
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
