/// TypeScript types matching the simulator.fbs FlatBuffer schema.
/// These mirror mirage.simulator.* exactly.

// ── Enums ──────────────────────────────────────────────────────────────────

export type GpuFamily = "Unknown" | "AmdCdna" | "AmdRdna" | "RiscV";

export type SimulatorMode = "Functional" | "Clocked" | "CycleAccurate";

export type HealthStatus = "Unknown" | "Healthy" | "Unhealthy";

export type SessionPhase =
  | "Pulling"
  | "Starting"
  | "Running"
  | "Failed"
  | "ShuttingDown"
  | "Stale";

// ── Core types (from common.fbs) ───────────────────────────────────────────

export interface GpuDef {
  name: string;
  arch: string;
  family: GpuFamily;
  description: string;
}

export interface ProfileDef {
  name: string;
  simulator: string;
  mode: SimulatorMode;
  gpu: string;
  num_gpus: number;
  num_nodes: number;
}

export interface SessionDef {
  name: string;
  profile: string;
  image: string;
}

export interface Time {
  seconds: number;
  picoseconds: number;
}

// ── Dashboard types (from simulator.fbs) ───────────────────────────────────

export interface OverviewData {
  simulator_count: number;
  profile_count: number;
  session_count: number;
}

export interface SimulatorSummary {
  name: string;
  version: string;
  description: string;
  supported_gpus: GpuDef[];
  supports_custom_gpus: boolean;
  supported_modes: SimulatorMode[];
  active_session_count: number;
}

export interface SessionSummary {
  name: string;
  profile: string;
  simulator: string;
  image: string;
  health_status: HealthStatus;
  phase: SessionPhase;
  progress_message: string;
}

export interface SessionDetail {
  name: string;
  profile: ProfileDef;
  simulator: string;
  image: string;
  health: HealthStatus;
  uptime: Time;
  error_message: string;
  ticks: number;
  ipc: number;
  simulation_speed: number;
  active_contexts: number;
  phase: SessionPhase;
  progress_message: string;
}

export interface ServiceResult {
  ok: boolean;
  error: string;
}

// ── Run types (from simulator.fbs — Dashboard.ListRuns / CreateRun) ────────

export interface RunRecord {
  id: string;
  session: string;
  command: string;
  status: string;
  exit_code: number;
  output: string;
}

// ── Terminal types (from simulator.fbs — Dashboard.Terminal* RPCs) ─────────

export interface TerminalInfo {
  id: string;
  session: string;
  alive: boolean;
}
