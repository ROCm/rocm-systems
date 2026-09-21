//! `mirage_core` is the shared library for mirage.
//!
//! It contains:
//!
//! * Strongly-typed definitions of mirage's documents and control-plane
//!   messages ([`session::SessionDef`], [`exec::ExecDef`],
//!   [`profile::ProfileDef`], …)
//! * Path resolution for mirage's *configuration* ([`paths`]),
//!   implementing the XDG layout
//! * Atomic file readers/writers for those documents ([`state`]) and the
//!   configuration store built on them ([`store`])
//! * The wire protocol a `mirage exec` uses to ask the `mirage run` that
//!   owns a session to describe it ([`proto`])
//! * The emulator backend trait and its link-time registry
//!   ([`emulator`], [`registry`])
//!
//! # What lives where
//!
//! Configuration is on disk; session state is not. Profiles, topologies
//! and agents are user-authored documents that outlive every process, so
//! they are files. A session, its processes and its containers are owned
//! in memory by the single `mirage run` that started them, and cease to
//! exist when it does. See the crate-level docs of [`paths`] for why.

pub mod agent;
pub mod common;
pub mod config;
pub mod container;
pub mod discovery;
pub mod emulator;
pub mod error;
pub mod exec;
pub mod hardware;
pub mod metric;
pub mod paths;
pub mod plugin;
pub mod profile;
pub mod proto;
pub mod reclaim;
pub mod registry;
pub mod session;
pub mod state;
pub mod store;
pub mod topology;
pub mod workload;
