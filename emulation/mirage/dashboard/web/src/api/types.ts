/// TypeScript types matching the new mirage_core / mirage_ctl API.
///
/// These are intentionally a thin echo of the Rust types. Fields that
/// the dashboard doesn't render are still typed loosely as
/// `Record<string, unknown>` so future schema changes don't break the
/// build.

export interface ProfileDef {
  name: string;
  description?: string;
  emulator: EmulatorDef;
}

export interface EmulatorDef {
  emulator: string;
  plugins: Record<string, Record<string, unknown>>;
  exec_mode: "Functional" | "Clocked";
  options: Record<string, unknown>;
  /// Either a topology preset name (string) or an inline
  /// [`TopologyDef`].
  topology: string | TopologyDef;
}

export interface TopologyDef {
  racks: number;
  nodes_per_rack: number;
  gpus_per_node: number;
  /// Either an agent preset name (string) or an inline
  /// `AgentDef` (typed loosely here).
  agent: string | AgentDef;
}

/// Loose mirror of `mirage_core::agent::AgentDef`. Only used by the
/// dashboard for display + round-trip; we don't validate inner
/// fields.
export interface AgentDef {
  vm: Record<string, unknown>;
  topology: Record<string, unknown>;
}

export interface SessionHealth {
  timestamp: string;
  healthy: boolean;
  state?: string;
  terminal: boolean;
  message?: string;
}

export interface SessionDef {
  id: string;
  profile: unknown;
  container?: unknown;
  workdir: string;
  created_at: string;
}

export interface SessionState {
  def: SessionDef;
  health: SessionHealth;
}

export interface NodeStatus {
  pid?: number;
  exit_code?: number;
}

export interface ExecStatus {
  started: boolean;
  ended: boolean;
  exit_code?: number;
  started_at?: string;
  ended_at?: string;
  nodes: Record<string, NodeStatus>;
}

export interface ExecListItem {
  id: string;
  status: ExecStatus;
}

export interface PathsInfo {
  config: string;
  runtime: string;
  state: string;
  profiles: string;
  sessions: string;
}

export interface SystemInfo {
  daemon_version: string;
  default_emulator: string;
}

export interface EmulatorEntry {
  name: string;
  description: string;
  installed: boolean;
  is_default: boolean;
  available_plugins: string[];
}

export interface Metrics {
  profiles: number;
  sessions: number;
  sessions_healthy: number;
  sessions_starting: number;
  execs_running: number;
  execs_total: number;
}

export type StreamPacket =
  | {
      Output: { node: number; stream: "Stdout" | "Stderr" | "Stdin"; data: number[] };
    }
  | { NodeExit: { node: number; exit_code: number } }
  | { ExecExit: { exit_code: number } };
