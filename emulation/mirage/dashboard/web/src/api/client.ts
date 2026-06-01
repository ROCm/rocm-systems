/// REST client for the `mirage_daemon` HTTP API.
///
/// Every method is a thin wrapper around `fetch` that delegates to the
/// daemon at `/api/...`. The daemon in turn delegates to
/// `mirage_core::ctl::MirageCtl`.

import type {
  AgentDef,
  EmulatorEntry,
  ExecListItem,
  ExecStatus,
  Metrics,
  PathsInfo,
  ProfileDef,
  SessionDef,
  SessionState,
  SystemInfo,
  TopologyDef,
} from "./types";

const API = "/api";

async function req<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const init: RequestInit = { method };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  const res = await fetch(`${API}${path}`, init);
  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json()).error ?? "";
    } catch {
      detail = await res.text();
    }
    throw new Error(`${method} ${path} failed (${res.status}): ${detail}`);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const get = <T>(p: string) => req<T>("GET", p);
export const post = <T>(p: string, b?: unknown) => req<T>("POST", p, b ?? {});
export const put = <T>(p: string, b?: unknown) => req<T>("PUT", p, b ?? {});
export const del = <T>(p: string) => req<T>("DELETE", p);

// ── Paths ──────────────────────────────────────────────────────────────────

export const getPaths = () => get<PathsInfo>("/paths");
export const getSystem = () => get<SystemInfo>("/system");
export const listEmulators = () => get<EmulatorEntry[]>("/emulators");
export const getMetrics = () => get<Metrics>("/metrics");

// ── Profiles ───────────────────────────────────────────────────────────────

export const listProfiles = () => get<string[]>("/profiles");
export const getProfile = (name: string) =>
  get<ProfileDef>(`/profiles/${encodeURIComponent(name)}`);
export const putProfile = (profile: ProfileDef) =>
  put<{ ok: boolean }>(
    `/profiles/${encodeURIComponent(profile.name)}`,
    profile,
  );
export const deleteProfile = (name: string) =>
  del<{ ok: boolean }>(`/profiles/${encodeURIComponent(name)}`);

// ── Topologies ──────────────────────────────────────────

export const listTopologies = () => get<string[]>("/topologies");
export const getTopology = (name: string) =>
  get<TopologyDef>(`/topologies/${encodeURIComponent(name)}`);
export const putTopology = (name: string, topology: TopologyDef) =>
  put<{ ok: boolean }>(
    `/topologies/${encodeURIComponent(name)}`,
    topology,
  );
export const deleteTopology = (name: string) =>
  del<{ ok: boolean }>(`/topologies/${encodeURIComponent(name)}`);

// ── Agents ──────────────────────────────────────────────
export const listAgents = () => get<string[]>("/agents");
export const getAgent = (name: string) =>
  get<AgentDef>(`/agents/${encodeURIComponent(name)}`);
export const putAgent = (name: string, agent: AgentDef) =>
  put<{ ok: boolean }>(`/agents/${encodeURIComponent(name)}`, agent);
export const deleteAgent = (name: string) =>
  del<{ ok: boolean }>(`/agents/${encodeURIComponent(name)}`);

// ── Sessions ───────────────────────────────────────────────────────────────

export const listSessions = () => get<SessionState[]>("/sessions");
export const getSession = (id: string) =>
  get<SessionState>(`/sessions/${encodeURIComponent(id)}`);
export const createSession = (params: {
  profile: string;
  id?: string;
  workdir?: string;
  ready_timeout?: number;
}) => post<SessionDef>("/sessions", params);
export const deleteSession = (id: string) =>
  del<{ ok: boolean }>(`/sessions/${encodeURIComponent(id)}`);

// ── Execs ──────────────────────────────────────────────────────────────────

export const listExecs = (sessionId: string) =>
  get<ExecListItem[]>(`/sessions/${encodeURIComponent(sessionId)}/execs`);
export const getExec = (sessionId: string, execId: string) =>
  get<ExecStatus>(
    `/sessions/${encodeURIComponent(sessionId)}/execs/${encodeURIComponent(execId)}`,
  );
export const createExec = (
  sessionId: string,
  body: { command: string; args?: string[]; keep?: boolean },
) => post<{ id: string }>(
  `/sessions/${encodeURIComponent(sessionId)}/execs`,
  body,
);
export const deleteExec = (sessionId: string, execId: string) =>
  del<{ ok: boolean }>(
    `/sessions/${encodeURIComponent(sessionId)}/execs/${encodeURIComponent(execId)}`,
  );
export const signalExec = (
  sessionId: string,
  execId: string,
  signal: number,
) =>
  post<{ ok: boolean }>(
    `/sessions/${encodeURIComponent(sessionId)}/execs/${encodeURIComponent(execId)}/signal`,
    { signal },
  );

export const stdinExec = (sessionId: string, execId: string, data: string) =>
  post<{ ok: boolean }>(
    `/sessions/${encodeURIComponent(sessionId)}/execs/${encodeURIComponent(execId)}/stdin`,
    { data },
  );

export function attachUrl(sessionId: string, execId: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${API}/sessions/${encodeURIComponent(sessionId)}/execs/${encodeURIComponent(execId)}/attach`;
}
