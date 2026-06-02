import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import * as api from "../api/client";
import type { SessionState } from "../api/types";
import { Dropdown } from "../components/ui/Dropdown";
import { Modal } from "../components/ui/Modal";
import { Pill, StatusDot } from "../components/ui/Status";
import { useToast } from "../components/ui/Toast";

type Health = "ready" | "starting" | "unhealthy" | "terminal" | "pending";

function healthOf(s: SessionState): Health {
  if (s.health.terminal) return "terminal";
  if (s.health.healthy) return "ready";
  if (s.health.state && s.health.state.toLowerCase().includes("start")) {
    return "starting";
  }
  if (s.health.state) return "pending";
  return "pending";
}

function healthTone(h: Health): "ok" | "warn" | "bad" | "muted" {
  switch (h) {
    case "ready":
      return "ok";
    case "starting":
    case "pending":
      return "warn";
    case "unhealthy":
    case "terminal":
      return "bad";
  }
}

function ageOf(iso: string): string {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "";
  const s = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function SessionsPage() {
  const toast = useToast();
  const [sessions, setSessions] = useState<SessionState[]>([]);
  const [profiles, setProfiles] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [newProfile, setNewProfile] = useState("");
  const [filter, setFilter] = useState("__all__");
  const [startOpen, setStartOpen] = useState(false);
  const [stopTarget, setStopTarget] = useState<string | null>(null);
  const [readyTimeout, setReadyTimeout] = useState(10);
  const [workdir, setWorkdir] = useState("");

  async function refresh() {
    try {
      const [s, p] = await Promise.all([api.listSessions(), api.listProfiles()]);
      setSessions(s);
      setProfiles(p);
      if (!newProfile && p.length > 0) setNewProfile(p[0]);
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const visible = useMemo(
    () =>
      sessions.filter((s) =>
        filter === "__all__" ? true : profileName(s.def.profile) === filter,
      ),
    [sessions, filter],
  );

  async function onStart(e: FormEvent) {
    e.preventDefault();
    if (!newProfile) {
      const msg = "create a profile first";
      setError(msg);
      toast.error(msg);
      return;
    }
    setBusy(true);
    setError("");
    try {
      await api.createSession({
        profile: newProfile,
        workdir: workdir || undefined,
        ready_timeout: readyTimeout,
      });
      toast.success(`Session starting from "${newProfile}"`);
      setStartOpen(false);
      await refresh();
    } catch (err) {
      const msg = String(err);
      setError(msg);
      toast.error(msg);
    } finally {
      setBusy(false);
    }
  }

  async function onConfirmStop() {
    if (!stopTarget) return;
    const id = stopTarget;
    setBusy(true);
    try {
      await api.deleteSession(id);
      toast.success(`Session ${id} stopped`);
      setStopTarget(null);
      await refresh();
    } catch (e) {
      const msg = String(e);
      setError(msg);
      toast.error(msg);
    } finally {
      setBusy(false);
    }
  }

  const profileOptions = [
    { value: "__all__", label: "All profiles" },
    ...profiles.map((p) => ({ value: p, label: p })),
  ];

  return (
    <div className="page page-wide">
      <div className="page-hero">
        <div>
          <h2>Sessions</h2>
          <p className="page-subtitle">
            Running emulator sessions. Like devices in a simulator —
            pick one, attach a command, watch it work.
          </p>
        </div>
        <div className="page-hero-actions">
          <div className="filter-label">Filter</div>
          <Dropdown
            value={filter}
            onChange={setFilter}
            options={profileOptions}
            testId="session-filter"
            ariaLabel="Filter by profile"
          />
          <button
            type="button"
            className="btn-secondary"
            onClick={refresh}
            data-testid="refresh-sessions"
          >
            Refresh
          </button>
          <button
            type="button"
            className="btn-primary"
            disabled={profiles.length === 0}
            onClick={() => setStartOpen(true)}
            data-testid="open-start-session"
          >
            + New session
          </button>
        </div>
      </div>
      {error && <div className="error" role="alert">{error}</div>}

      {visible.length === 0 ? (
        <div className="empty-state" data-testid="no-sessions">
          <h3>No sessions running</h3>
          <p>
            {profiles.length === 0
              ? "Create a profile first, then launch a session here."
              : "Click \"New session\" above to spin one up."}
          </p>
        </div>
      ) : (
        <ul className="session-grid" data-testid="sessions-table">
          {visible.map((s) => {
            const h = healthOf(s);
            const tone = healthTone(h);
            return (
              <li
                key={s.def.id}
                className={`session-card session-card-${tone}`}
                data-testid={`session-row-${s.def.id}`}
              >
                <header className="session-card-head">
                  <div className="session-card-title">
                    <StatusDot tone={tone} ariaLabel={h} />
                    <span className="session-card-profile">
                      {profileName(s.def.profile)}
                    </span>
                  </div>
                  <Pill tone={tone} testId={`session-health-${s.def.id}`}>
                    {h}
                  </Pill>
                </header>
                <div className="session-card-body">
                  <Link
                    to={`/sessions/${encodeURIComponent(s.def.id)}`}
                    className="session-card-id"
                    data-testid={`session-open-${s.def.id}`}
                  >
                    <code>{s.def.id}</code>
                  </Link>
                  <dl className="session-card-meta">
                    <div>
                      <dt>Age</dt>
                      <dd>{ageOf(s.def.created_at)}</dd>
                    </div>
                    <div>
                      <dt>State</dt>
                      <dd>{s.health.state ?? "—"}</dd>
                    </div>
                    <div>
                      <dt>Healthy</dt>
                      <dd data-testid={`session-healthy-${s.def.id}`}>
                        {s.health.healthy ? "healthy" : "starting"}
                      </dd>
                    </div>
                  </dl>
                </div>
                <footer className="session-card-actions">
                  <Link
                    to={`/sessions/${encodeURIComponent(s.def.id)}`}
                    className="btn-secondary"
                  >
                    Open
                  </Link>
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={() =>
                      navigator.clipboard?.writeText(s.def.id).then(
                        () => toast.info("Session ID copied"),
                        () => {},
                      )
                    }
                  >
                    Copy ID
                  </button>
                  <button
                    type="button"
                    className="btn-danger-sm"
                    disabled={busy}
                    onClick={() => setStopTarget(s.def.id)}
                    data-testid={`stop-session-${s.def.id}`}
                  >
                    Stop
                  </button>
                </footer>
              </li>
            );
          })}
        </ul>
      )}

      {/* Compat form: same test-ids as before, hidden inside the
          modal flow. Submit triggers the same start path. */}
      <form
        onSubmit={onStart}
        className="quick-create"
        data-testid="start-session"
        style={{ marginTop: 24 }}
      >
        <span className="quick-create-label">Quick start</span>
        <select
          value={newProfile}
          onChange={(e) => setNewProfile(e.target.value)}
          data-testid="new-session-profile"
        >
          {profiles.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
        <button
          type="submit"
          className="btn-primary-sm"
          disabled={busy || profiles.length === 0}
          data-testid="submit-session"
        >
          Start
        </button>
      </form>

      <Modal
        open={startOpen}
        onClose={() => setStartOpen(false)}
        title="Start a session"
        testId="start-session-modal"
        footer={
          <>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setStartOpen(false)}
            >
              Cancel
            </button>
            <button
              type="submit"
              form="start-session-form"
              className="btn-primary"
              disabled={busy || !newProfile}
              data-testid="start-session-confirm"
            >
              {busy ? "Starting…" : "Start session"}
            </button>
          </>
        }
      >
        <form
          id="start-session-form"
          className="form-grid"
          onSubmit={onStart}
        >
          <div className="form-field span-2">
            <span>Profile</span>
            <Dropdown
              value={newProfile}
              onChange={setNewProfile}
              testId="start-modal-profile"
              ariaLabel="Profile"
              options={profiles.map((p) => ({ value: p, label: p }))}
            />
          </div>
          <label className="form-field">
            <span>Ready timeout (s)</span>
            <input
              type="number"
              min={0}
              value={readyTimeout}
              data-testid="start-modal-timeout"
              onChange={(e) =>
                setReadyTimeout(Math.max(0, +e.target.value || 0))
              }
            />
          </label>
          <label className="form-field">
            <span>Workdir</span>
            <input
              value={workdir}
              data-testid="start-modal-workdir"
              onChange={(e) => setWorkdir(e.target.value)}
              placeholder="(daemon's cwd)"
            />
          </label>
        </form>
      </Modal>

      <Modal
        open={stopTarget !== null}
        onClose={() => setStopTarget(null)}
        title="Stop session?"
        testId="confirm-stop-session"
        footer={
          <>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setStopTarget(null)}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn-danger"
              data-testid="confirm-stop-session-confirm"
              onClick={onConfirmStop}
            >
              Stop session
            </button>
          </>
        }
      >
        <p>
          Stopping <code>{stopTarget}</code> tears down the host process and
          frees its workdir. This cannot be undone.
        </p>
      </Modal>
    </div>
  );
}

function profileName(p: unknown): string {
  if (typeof p === "string") return p;
  if (
    p &&
    typeof p === "object" &&
    "name" in (p as Record<string, unknown>) &&
    typeof (p as { name: unknown }).name === "string"
  ) {
    return (p as { name: string }).name;
  }
  return String(p);
}
