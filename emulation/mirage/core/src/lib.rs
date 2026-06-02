//! `mirage_core` is the shared library for mirage.
//!
//! It contains:
//!
//! * Strongly-typed definitions of mirage's on-disk state
//!   ([`session::SessionDef`], [`exec::ExecDef`], [`profile::ProfileDef`], …)
//! * Path-resolution helpers ([`paths`]) implementing the XDG layout
//! * Atomic file readers/writers ([`state`])
//! * The control-plane trait ([`ctl::MirageCtl`]) and its file-backed
//!   implementation ([`ctl::FileCtl`])
//! * A streaming "attach" helper ([`attach`]) that tails per-node
//!   stdout/stderr and emits structured packets to clients
//!
//! See the crate-level docs of [`paths`] for the on-disk layout.

pub mod agent;
pub mod attach;
pub mod common;
pub mod config;
pub mod container;
pub mod ctl;
pub mod emulator;
pub mod error;
pub mod exec;
pub mod metric;
pub mod paths;
pub mod plugin;
pub mod profile;
pub mod registry;
pub mod session;
pub mod state;
pub mod topology;
pub mod workload;
