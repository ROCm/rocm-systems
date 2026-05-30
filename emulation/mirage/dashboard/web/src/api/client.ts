/// JSON REST client for the Mirage Daemon dashboard API.
///
/// Talks to the axum REST router generated from `ctl_dsl!` in
/// `mirage_schema::ctl::daemon`. Each endpoint accepts a JSON body and
/// returns a JSON body. The daemon exposes these at `/api/<endpoint>`.
///
/// Runs and terminals are modelled on top of the daemon's `exec` + `attach`
/// primitives; records are kept client-side so pages can list past runs in
/// the current session.

import type {
  OverviewData,
  SimulatorSummary,
  ProfileDef,
  SessionSummary,
  SessionDetail,
  ServiceResult,
  SessionDef,
  RunRecord,
  TerminalInfo,
  GpuFamily,
  SimulatorMode,
  HealthStatus,
  SessionPhase,
} from "./types";

// ── Transport ──────────────────────────────────────────────────────────────

const API = "/api";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST ${path} failed (${res.status}): ${text}`);
  }
  return res.json();
}

function websocketUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${API}${path}`;
}

// ── Enum case conversion (TS PascalCase ↔ wire snake_case) ─────────────────

const GPU_FAMILY_FROM_WIRE: Record<string, GpuFamily> = {
  unknown: "Unknown",
  amd_cdna: "AmdCdna",
  amd_rdna: "AmdRdna",
  risc_v: "RiscV",
};

const MODE_FROM_WIRE: Record<string, SimulatorMode> = {
  functional: "Functional",
  clocked: "Clocked",
  cycle_accurate: "CycleAccurate",
};
const MODE_TO_WIRE: Record<SimulatorMode, string> = {
  Functional: "functional",
  Clocked: "clocked",
  CycleAccurate: "cycle_accurate",
};

const HEALTH_FROM_WIRE: Record<string, HealthStatus> = {
  unknown: "Unknown",
  healthy: "Healthy",
  unhealthy: "Unhealthy",
};

const PHASE_FROM_WIRE: Record<string, SessionPhase> = {
  pulling: "Pulling",
  starting: "Starting",
  running: "Running",
  failed: "Failed",
  shutting_down: "ShuttingDown",
  stale: "Stale",
};

interface WireGpuDef {
  name: string;
  arch: string;
  family: string;
  description?: string;
}

function fromWireGpuDef(g: WireGpuDef) {
  return {
    name: g.name,
    arch: g.arch,
    family: GPU_FAMILY_FROM_WIRE[g.family] ?? "Unknown",
    description: g.description ?? "",
  };
}

interface WireSimulatorSummary {
  name?: string;
  version?: string;
  description?: string;
  supported_gpus?: WireGpuDef[];
  supports_custom_gpus?: boolean;
  supported_modes?: string[];
  active_session_count?: number;
}

function fromWireSimulatorSummary(s: WireSimulatorSummary): SimulatorSummary {
  return {
    name: s.name ?? "",
    version: s.version ?? "",
    description: s.description ?? "",
    supported_gpus: (s.supported_gpus ?? []).map(fromWireGpuDef),
    supports_custom_gpus: !!s.supports_custom_gpus,
    supported_modes: (s.supported_modes ?? [])
      .map((m) => MODE_FROM_WIRE[m])
      .filter((m): m is SimulatorMode => !!m),
    active_session_count: s.active_session_count ?? 0,
  };
}

interface WireProfile {
  name: string;
  simulator: string;
  mode: string;
  gpu: string;
  num_gpus: number;
  num_nodes: number;
}

function fromWireProfile(p: WireProfile): ProfileDef {
  return {
    name: p.name,
    simulator: p.simulator,
    mode: MODE_FROM_WIRE[p.mode] ?? "Functional",
    gpu: p.gpu,
    num_gpus: p.num_gpus,
    num_nodes: p.num_nodes,
  };
}

function toWireCreateProfile(p: ProfileDef) {
  return {
    name: p.name,
    simulator: p.simulator,
    mode: MODE_TO_WIRE[p.mode],
    gpu: p.gpu,
    gpus_per_node: p.num_gpus,
    nodes: p.num_nodes,
  };
}

interface WireSessionSummary {
  name?: string;
  profile?: string;
  simulator?: string;
  image?: string;
  health_status?: string;
  phase?: string;
  progress_message?: string;
}

function fromWireSessionSummary(s: WireSessionSummary): SessionSummary {
  return {
    name: s.name ?? "",
    profile: s.profile ?? "",
    simulator: s.simulator ?? "",
    image: s.image ?? "",
    health_status: HEALTH_FROM_WIRE[s.health_status ?? "unknown"] ?? "Unknown",
    phase: PHASE_FROM_WIRE[s.phase ?? "running"] ?? "Running",
    progress_message: s.progress_message ?? "",
  };
}

// ── Overview ───────────────────────────────────────────────────────────────

export async function getOverview(): Promise<OverviewData> {
  return post<OverviewData>("/get_overview", {});
}

// ── Simulators ─────────────────────────────────────────────────────────────

export async function listSimulators(): Promise<SimulatorSummary[]> {
  const r = await post<{ simulators: WireSimulatorSummary[] }>(
    "/list_simulators",
    {},
  );
  return r.simulators.map(fromWireSimulatorSummary);
}

export async function getSimulator(
  name: string,
): Promise<SimulatorSummary | null> {
  try {
    const r = await post<{ simulator?: WireSimulatorSummary }>(
      "/show_simulator",
      { name },
    );
    return r.simulator ? fromWireSimulatorSummary(r.simulator) : null;
  } catch {
    return null;
  }
}

// ── Profiles ───────────────────────────────────────────────────────────────

export async function listProfiles(
  simulatorFilter?: string,
): Promise<ProfileDef[]> {
  const body = simulatorFilter ? { simulator: simulatorFilter } : {};
  const r = await post<{ profiles: WireProfile[] }>("/list_profiles", body);
  return r.profiles.map(fromWireProfile);
}

export async function createProfile(
  profile: ProfileDef,
): Promise<ServiceResult> {
  const r = await post<{ ok: boolean; error?: string }>(
    "/create_profile",
    toWireCreateProfile(profile),
  );
  return { ok: r.ok, error: r.error ?? "" };
}

export async function deleteProfile(name: string): Promise<ServiceResult> {
  const r = await post<{ ok: boolean; error?: string }>("/delete_profile", {
    name,
  });
  return { ok: r.ok, error: r.error ?? "" };
}

// ── Sessions ───────────────────────────────────────────────────────────────

export async function listSessions(
  profileFilter?: string,
): Promise<SessionSummary[]> {
  const body = profileFilter ? { profile: profileFilter } : {};
  const r = await post<{ sessions: WireSessionSummary[] }>(
    "/list_sessions",
    body,
  );
  return r.sessions.map(fromWireSessionSummary);
}

export async function createSession(
  session: SessionDef,
): Promise<ServiceResult> {
  const r = await post<{ ok: boolean; error?: string }>("/boot", {
    name: session.name,
    profile: session.profile,
    image: session.image,
    volumes: [],
  });
  return { ok: r.ok, error: r.error ?? "" };
}

export async function deleteSession(name: string): Promise<ServiceResult> {
  const r = await post<{ ok: boolean; error?: string }>("/shutdown", { name });
  // Remove any client-side runs/terminals for this session.
  for (const [id, run] of clientRuns) {
    if (run.session === name) clientRuns.delete(id);
  }
  for (const [id, t] of clientTerminals) {
    if (t.session === name) clientTerminals.delete(id);
  }
  return { ok: r.ok, error: r.error ?? "" };
}

interface WireStatus {
  name?: string;
  profile?: WireProfile;
  simulator?: string;
  image?: string;
  health: string;
  uptime?: { seconds: number; picoseconds: number };
  error_message?: string;
  ticks: number;
  ipc: number;
  simulation_speed: number;
  active_contexts: number;
  phase?: string;
  progress_message?: string;
}

export async function getSessionDetail(
  name: string,
): Promise<SessionDetail | null> {
  try {
    const r = await post<WireStatus>("/status", { name });
    return {
      name: r.name ?? name,
      profile: r.profile
        ? fromWireProfile(r.profile)
        : {
            name: "",
            simulator: "",
            mode: "Functional",
            gpu: "",
            num_gpus: 1,
            num_nodes: 1,
          },
      simulator: r.simulator ?? "",
      image: r.image ?? "",
      health: HEALTH_FROM_WIRE[r.health] ?? "Unknown",
      uptime: r.uptime ?? { seconds: 0, picoseconds: 0 },
      error_message: r.error_message ?? "",
      ticks: r.ticks,
      ipc: r.ipc,
      simulation_speed: r.simulation_speed,
      active_contexts: r.active_contexts,
      phase: PHASE_FROM_WIRE[r.phase ?? "running"] ?? "Running",
      progress_message: r.progress_message ?? "",
    };
  } catch {
    return null;
  }
}

// ── Session log ────────────────────────────────────────────────────────────

// The daemon does not track an aggregate per-session log. The page displays
// a minimal status line synthesized from the live session detail.
export async function getSessionLog(
  name: string,
): Promise<{ log: string; status: string }> {
  const detail = await getSessionDetail(name);
  if (!detail) return { log: "", status: "" };
  return {
    log: `Session '${detail.name}' using profile '${detail.profile.name}'.\nSimulator: ${detail.simulator}\nStatus: ${detail.health}\n`,
    status: detail.health.toLowerCase(),
  };
}

// ── Runs (client-side, backed by exec + attach) ────────────────────────────

const clientRuns = new Map<string, RunRecord>();
let runCounter = 0;

export async function listRuns(
  sessionFilter?: string,
): Promise<RunRecord[]> {
  const records = Array.from(clientRuns.values());
  return sessionFilter
    ? records.filter((r) => r.session === sessionFilter)
    : records;
}

export async function createRun(
  session: string,
  command: string,
  nodeIndex: number = 0,
): Promise<{ ok: boolean; error?: string; run?: RunRecord }> {
  const parts = command.trim().split(/\s+/);
  if (!parts.length || !parts[0]) return { ok: false, error: "empty command" };
  let execReply: { exec_id: string };
  try {
    execReply = await post<{ exec_id: string }>("/exec", {
      session,
      node_index: nodeIndex,
      command: parts,
    });
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }

  const id = `run-${++runCounter}`;
  const pending: RunRecord = {
    id,
    session,
    command,
    status: "running",
    exit_code: -1,
    output: `$ ${command}\n`,
  };
  clientRuns.set(id, pending);

  // Attach and collect output until the reply frame arrives.
  await new Promise<void>((resolve) => {
    const ws = new WebSocket(
      websocketUrl(`/attach?exec_id=${encodeURIComponent(execReply.exec_id)}`),
    );
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data as string);
        if (msg.type === "output") {
          const bytes = msg.data?.output as number[] | undefined;
          if (bytes && bytes.length) {
            pending.output += String.fromCharCode(...bytes);
            clientRuns.set(id, { ...pending });
          }
        } else if (msg.type === "reply") {
          pending.exit_code = msg.data?.exit_code ?? 0;
          pending.status = pending.exit_code === 0 ? "completed" : "failed";
          clientRuns.set(id, { ...pending });
          ws.close();
        } else if (msg.type === "error") {
          pending.status = "failed";
          pending.output += `\n[attach error: ${msg.message}]`;
          clientRuns.set(id, { ...pending });
          ws.close();
        }
      } catch {
        // ignore malformed frame
      }
    };
    ws.onerror = () => {
      pending.status = "failed";
      pending.output += `\n[websocket error]`;
      clientRuns.set(id, { ...pending });
      resolve();
    };
    ws.onclose = () => resolve();
  });

  return { ok: pending.status === "completed", run: pending };
}

// ── Terminals (client-side, backed by interactive exec) ────────────────────

const clientTerminals = new Map<
  string,
  TerminalInfo & { exec_id: string }
>();
let terminalCounter = 0;

export async function listTerminals(): Promise<TerminalInfo[]> {
  return Array.from(clientTerminals.values()).map(({ id, session, alive }) => ({
    id,
    session,
    alive,
  }));
}

export async function createTerminal(
  session: string,
  nodeIndex: number = 0,
  program: string = "/bin/sh",
): Promise<{ ok: boolean; error: string; id: string }> {
  try {
    const parts = program.trim().split(/\s+/).filter(Boolean);
    const r = await post<{ exec_id: string }>("/exec", {
      session,
      node_index: nodeIndex,
      command: parts.length ? parts : ["/bin/sh"],
    });
    const id = `term-${++terminalCounter}`;
    clientTerminals.set(id, { id, session, alive: true, exec_id: r.exec_id });
    return { ok: true, error: "", id };
  } catch (e) {
    return { ok: false, error: (e as Error).message, id: "" };
  }
}

export async function closeTerminal(id: string): Promise<void> {
  clientTerminals.delete(id);
}

/// Return the underlying exec id so the Terminal page can open a
/// WebSocket attach to stream stdin/stdout.
export function getTerminalExecId(id: string): string | undefined {
  return clientTerminals.get(id)?.exec_id;
}

export function terminalAttachUrl(execId: string): string {
  return websocketUrl(`/attach?exec_id=${encodeURIComponent(execId)}`);
}
