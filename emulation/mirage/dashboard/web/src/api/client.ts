/// REST client for the `mirage_daemon` HTTP API.
///
/// Every method is a thin wrapper around `fetch` that delegates to the
/// daemon at `/api/...`. The daemon in turn delegates to
/// `mirage_core::ctl::MirageCtl`.

import type {
  ExecListItem,
  ExecStatus,
  PathsInfo,
  ProfileDef,
  SessionDef,
  SessionState,
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

export function attachUrl(sessionId: string, execId: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${API}/sessions/${encodeURIComponent(sessionId)}/execs/${encodeURIComponent(execId)}/attach`;
}
