//! `mirage_daemon`: HTTP/WebSocket dashboard server.
//!
//! Mirage's normal command-line operation is fully daemon-free: each
//! session has a per-session [`mirage_host`] process and the CLI talks
//! to it through the filesystem. The daemon is purely a **read/write
//! shim over [`mirage_core::ctl::MirageCtl`]** that exposes the same
//! operations over HTTP so a browser-based dashboard (served by this
//! daemon) and external tools can drive mirage.
//!
//! # Endpoints
//!
//! All JSON endpoints live under `/api`. The dashboard SPA is served
//! at `/` (with single-page-app fallback to `index.html`).
//!
//! | Method | Path                                                 | Body / Response                              |
//! |--------|------------------------------------------------------|----------------------------------------------|
//! | GET    | `/api/paths`                                         | XDG paths                                    |
//! | GET    | `/api/profiles`                                      | `[name]`                                     |
//! | GET    | `/api/profiles/:name`                                | `ProfileDef`                                 |
//! | PUT    | `/api/profiles/:name`                                | `ProfileDef` → `{ok}`                        |
//! | DELETE | `/api/profiles/:name`                                | `{ok}`                                       |
//! | GET    | `/api/sessions`                                      | `[SessionState]`                             |
//! | POST   | `/api/sessions`                                      | `CreateSessionBody` → `SessionDef`           |
//! | GET    | `/api/sessions/:id`                                  | `SessionState`                               |
//! | DELETE | `/api/sessions/:id`                                  | `{ok}`                                       |
//! | GET    | `/api/sessions/:id/execs`                            | `[{id, status}]`                             |
//! | POST   | `/api/sessions/:id/execs`                            | `ExecBody` → `{id}`                          |
//! | GET    | `/api/sessions/:id/execs/:exec`                      | `ExecStatus`                                 |
//! | POST   | `/api/sessions/:id/execs/:exec/signal`               | `{signal:i32}` → `{ok}`                      |
//! | POST   | `/api/sessions/:id/execs/:exec/stdin`                | `{data:string}` → `{ok}`                     |
//! | DELETE | `/api/sessions/:id/execs/:exec`                      | `{ok}`                                       |
//! | WS     | `/api/sessions/:id/execs/:exec/attach`               | server pushes `StreamPacket` JSON frames     |
//!
//! Run with `mirage webui --addr 127.0.0.1:5174`.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use clap::{Args, Subcommand};

mod api;
mod server;
mod service;
mod spa;
mod state;

pub use server::{build_router, serve};
pub use service::InstallArgs;
pub use state::AppState;

/// Command-line flags for `mirage webui`.
#[derive(Args, Debug, Clone)]
pub struct WebuiArgs {
    /// Address to bind the HTTP server on (and to bake into the
    /// systemd unit when installing).
    #[arg(long, default_value = "127.0.0.1:5174", env = "MIRAGE_WEBUI_ADDR")]
    pub addr: SocketAddr,

    #[command(subcommand)]
    pub command: Option<WebuiCmd>,
}

impl Default for WebuiArgs {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:5174".parse().unwrap(),
            command: None,
        }
    }
}

/// Subcommands of `mirage webui`.
#[derive(Subcommand, Debug, Clone)]
pub enum WebuiCmd {
    /// Run the web UI server in the foreground (this is the default
    /// when no subcommand is given).
    Serve,
    /// Install the web UI as a systemd service.
    Install(InstallArgs),
}

/// Entry point for `mirage webui`.
pub fn run(args: WebuiArgs) -> anyhow::Result<()> {
    match &args.command {
        Some(WebuiCmd::Install(install)) => service::install(args.addr, install),
        _ => serve_blocking(args.addr),
    }
}

/// Run the HTTP server until shutdown.
fn serve_blocking(addr: SocketAddr) -> anyhow::Result<()> {
    // Best-effort: write any missing builtin agents/topologies and
    // extract the rocjitsu runtime assets on startup so the
    // dashboard/CLI always see them.
    mirage_ctl::ensure_builtins_present();
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let state = Arc::new(AppState::new());
        serve(addr, build_router(state)).await
    })
}

/// Build a router without binding to a socket. Useful for in-process
/// integration tests.
pub fn router() -> Router {
    build_router(Arc::new(AppState::new()))
}
