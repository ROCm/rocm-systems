//! `mirage_supervisor`: the in-process engine that owns every session.
//!
//! # What replaced what
//!
//! Mirage used to run a detached `mirage host` process per session. The
//! CLI and that host communicated by writing and polling files: an exec
//! was requested by dropping a `def.json` into a directory the host
//! scanned every 50ms, its output arrived by tailing a file, its stdin
//! was a FIFO, and its completion was a `status.json` the CLI polled for.
//!
//! That design leaked processes, and the reasons are worth stating
//! because they shaped this one:
//!
//! * **No single owner.** The host was detached from the CLI that spawned
//!   it, so when the CLI exited the host was reparented to init. Nothing
//!   was responsible for it, and nothing reaped it.
//! * **Files outlive writers.** A `health.json` claiming `healthy: true`
//!   says nothing about whether its author is alive, so the design needed
//!   a heartbeat and a staleness ladder to *guess* — and a guess is not a
//!   basis for deciding whether to kill something.
//! * **Kill paths were open-loop.** Teardown signalled pids read from
//!   files and then removed the directory. If a signal did not land, or a
//!   grandchild had escaped the process group, nothing noticed: the state
//!   was gone, so the leak was invisible.
//!
//! The supervisor inverts all three. Sessions are values in a map inside
//! one process; every child process is owned by exactly one task that
//! always waits on it; teardown is closed-loop and does not return until
//! the process table agrees.
//!
//! # Structure
//!
//! * [`SessionManager`] — the map of sessions, and the
//!   [`MirageCtl`](mirage_core::ctl::MirageCtl) implementation over it.
//! * [`session::Session`] — one session: profile, emulator runtime,
//!   containers, execs, health.
//! * [`exec::Exec`] — one command invocation's process grid.
//! * [`output::OutputHub`] — replay-plus-live output fan-out.
//! * [`process`] — spawning, supervising and reliably killing processes.

pub mod exec;
pub mod manager;
pub mod output;
pub mod process;
pub mod session;

pub use exec::Exec;
pub use manager::{ManagerConfig, SessionManager};
pub use output::{OutputHub, Subscription};
pub use process::{Exit, ProcessInput, SpawnSpec, Spawned, StdioMode};
pub use session::Session;
