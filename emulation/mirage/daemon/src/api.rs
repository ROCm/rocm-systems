//! REST + WebSocket API mounted at `/api`.
//!
//! Every handler delegates to a [`mirage_core::ctl::MirageCtl`]
//! (currently [`mirage_core::ctl::FileCtl`]). The daemon never touches
//! the filesystem layout directly — `ctl` is the single source of
//! truth, exactly the same one the CLI uses.

use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::Router;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use chrono::Utc;
use futures::{SinkExt, StreamExt};
use mirage_core::agent::AgentDef;
use mirage_core::common::MaybeRef;
use mirage_core::ctl::{CreateSessionRequest, MirageCtl, StreamPacket};
use mirage_core::exec::{ExecArgs, ExecDef, ExecId, ExecRef, ExecStatus};
use mirage_core::profile::ProfileDef;
use mirage_core::session::{SessionDef, SessionId, SessionState};
use mirage_core::topology::TopologyDef;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/paths", get(get_paths))
        .route("/system", get(get_system))
        .route("/emulators", get(list_emulators))
        .route("/metrics", get(get_metrics))
        .route("/profiles", get(list_profiles))
        .route("/profiles/{name}", get(get_profile))
        .route("/profiles/{name}", put(put_profile))
        .route("/profiles/{name}", delete(delete_profile))
        .route("/topologies", get(list_topologies))
        .route("/topologies/{name}", get(get_topology))
        .route("/topologies/{name}", put(put_topology))
        .route("/topologies/{name}", delete(delete_topology))
        .route("/agents", get(list_agents))
        .route("/agents/{name}", get(get_agent))
        .route("/agents/{name}", put(put_agent))
        .route("/agents/{name}", delete(delete_agent))
        .route("/sessions", get(list_sessions).post(create_session))
        .route("/sessions/{id}", get(get_session).delete(delete_session))
        .route("/sessions/{id}/execs", get(list_execs).post(create_exec))
        .route(
            "/sessions/{id}/execs/{exec}",
            get(get_exec).delete(delete_exec),
        )
        .route("/sessions/{id}/execs/{exec}/signal", post(signal_exec))
        .route("/sessions/{id}/execs/{exec}/stdin", post(stdin_exec))
        .route("/sessions/{id}/execs/{exec}/attach", get(attach_exec))
        .with_state(state)
}

// ---- helpers ---------------------------------------------------------------

/// Map a `MirageError` into an HTTP response.
struct ApiError {
    status: StatusCode,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(serde_json::json!({"error": self.message}));
        (self.status, body).into_response()
    }
}

impl From<mirage_core::error::MirageError> for ApiError {
    fn from(e: mirage_core::error::MirageError) -> Self {
        use mirage_core::error::MirageError as E;
        let status = match &e {
            E::ProfileNotFound(_) | E::SessionNotFound(_) | E::ExecNotFound(_) => {
                StatusCode::NOT_FOUND
            }
            E::SessionExists(_) => StatusCode::CONFLICT,
            E::Id(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        ApiError {
            status,
            message: e.to_string(),
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: e.to_string(),
        }
    }
}

fn parse_session_id(s: &str) -> Result<SessionId, ApiError> {
    SessionId::from_str(s).map_err(|e| ApiError {
        status: StatusCode::BAD_REQUEST,
        message: format!("invalid session id: {e}"),
    })
}

fn parse_exec_id(s: &str) -> Result<ExecId, ApiError> {
    ExecId::new(s.to_string()).map_err(|e| ApiError {
        status: StatusCode::BAD_REQUEST,
        message: format!("invalid exec id: {e}"),
    })
}

#[derive(Serialize)]
struct Ok {
    ok: bool,
}
fn ok() -> Json<Ok> {
    Json(Ok { ok: true })
}

// ---- paths -----------------------------------------------------------------

#[derive(Serialize)]
struct PathsResponse {
    config: String,
    runtime: String,
    state: String,
    cache: String,
    profiles: String,
    sessions: String,
}

async fn get_paths() -> Json<PathsResponse> {
    Json(PathsResponse {
        config: mirage_core::paths::mirage_config_dir()
            .display()
            .to_string(),
        runtime: mirage_core::paths::mirage_runtime_dir()
            .display()
            .to_string(),
        state: mirage_core::paths::mirage_state_dir().display().to_string(),
        cache: mirage_core::paths::mirage_cache_dir().display().to_string(),
        profiles: mirage_core::paths::profile_root().display().to_string(),
        sessions: mirage_core::paths::session_root().display().to_string(),
    })
}

// ---- system / emulators / metrics -----------------------------------------

#[derive(Serialize)]
struct SystemResponse {
    daemon_version: &'static str,
    default_emulator: String,
}

async fn get_system() -> Json<SystemResponse> {
    Json(SystemResponse {
        daemon_version: env!("CARGO_PKG_VERSION"),
        default_emulator: mirage_ctl::default_emulator().name,
    })
}

#[derive(Serialize)]
struct EmulatorEntry {
    name: String,
    description: String,
    installed: bool,
    is_default: bool,
    /// Whether this host's hardware/environment can run the emulator.
    supported: bool,
    /// Human-readable explanation of the support decision.
    support_reason: String,
    path: Option<std::path::PathBuf>,
    available_plugins: Vec<&'static str>,
}

async fn list_emulators() -> Json<Vec<EmulatorEntry>> {
    let default_name = mirage_ctl::default_emulator().name;
    let entries = mirage_ctl::registry()
        .into_iter()
        .map(|spec| {
            let is_default = spec.name == default_name;
            EmulatorEntry {
                name: spec.name,
                description: spec.description,
                installed: spec.installed,
                is_default,
                supported: spec.support.supported,
                support_reason: spec.support.reason,
                path: spec.path,
                // Plugin discovery requires constructing a live backend
                // instance; the registry doesn't expose a static plugin
                // list yet, so we return an empty set here. Future work:
                // surface declared plugin slots on the emulator
                // description.
                available_plugins: Vec::new(),
            }
        })
        .collect();
    Json(entries)
}

#[derive(Serialize)]
struct MetricsResponse {
    profiles: usize,
    sessions: usize,
    sessions_healthy: usize,
    sessions_starting: usize,
    execs_running: usize,
    execs_total: usize,
}

async fn get_metrics(State(s): State<Arc<AppState>>) -> Result<Json<MetricsResponse>, ApiError> {
    let profiles = s.ctl.profile_list()?.len();
    let session_ids = s.ctl.session_list().unwrap_or_default();
    let mut healthy = 0usize;
    let mut starting = 0usize;
    let mut execs_running = 0usize;
    let mut execs_total = 0usize;
    for id in &session_ids {
        if let Ok(state) = s.ctl.session_state(id) {
            if state.health.healthy {
                healthy += 1;
            } else if !state.health.terminal {
                starting += 1;
            }
        }
        if let Ok(eids) = s.ctl.exec_list(id) {
            execs_total += eids.len();
            for eid in eids {
                let r = ExecRef {
                    session: id.clone(),
                    exec: eid,
                };
                if let Ok(status) = s.ctl.exec_status(&r)
                    && status.started
                    && !status.ended
                {
                    execs_running += 1;
                }
            }
        }
    }
    Ok(Json(MetricsResponse {
        profiles,
        sessions: session_ids.len(),
        sessions_healthy: healthy,
        sessions_starting: starting,
        execs_running,
        execs_total,
    }))
}

// ---- profiles --------------------------------------------------------------

async fn list_profiles(State(s): State<Arc<AppState>>) -> Result<Json<Vec<String>>, ApiError> {
    Ok(Json(s.ctl.profile_list()?))
}

async fn get_profile(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<ProfileDef>, ApiError> {
    Ok(Json(s.ctl.profile_get(&name)?))
}

async fn put_profile(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(mut profile): Json<ProfileDef>,
) -> Result<Json<Ok>, ApiError> {
    // Path is authoritative.
    profile.name = name;
    // Validate against the target emulator so the dashboard surfaces
    // *why* a profile is rejected instead of failing later at session
    // start.
    mirage_ctl::validate_profile(&profile).map_err(|message| ApiError {
        status: StatusCode::BAD_REQUEST,
        message,
    })?;
    s.ctl.profile_put(&profile)?;
    Ok(ok())
}

async fn delete_profile(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<Ok>, ApiError> {
    s.ctl.profile_delete(&name)?;
    Ok(ok())
}

// ---- topologies ------------------------------------------------------------

async fn list_topologies(State(s): State<Arc<AppState>>) -> Result<Json<Vec<String>>, ApiError> {
    Ok(Json(s.ctl.topology_list()?))
}

async fn get_topology(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<TopologyDef>, ApiError> {
    Ok(Json(s.ctl.topology_get(&name)?))
}

async fn put_topology(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(topology): Json<TopologyDef>,
) -> Result<Json<Ok>, ApiError> {
    s.ctl.topology_put(&name, &topology)?;
    Ok(ok())
}

async fn delete_topology(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<Ok>, ApiError> {
    s.ctl.topology_delete(&name)?;
    Ok(ok())
}

// ---- agents ----------------------------------------------------------------

async fn list_agents(State(s): State<Arc<AppState>>) -> Result<Json<Vec<String>>, ApiError> {
    Ok(Json(s.ctl.agent_list()?))
}

async fn get_agent(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<AgentDef>, ApiError> {
    Ok(Json(s.ctl.agent_get(&name)?))
}

async fn put_agent(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(agent): Json<AgentDef>,
) -> Result<Json<Ok>, ApiError> {
    s.ctl.agent_put(&name, &agent)?;
    Ok(ok())
}

async fn delete_agent(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<Ok>, ApiError> {
    s.ctl.agent_delete(&name)?;
    Ok(ok())
}

// ---- sessions --------------------------------------------------------------

#[derive(Deserialize)]
struct CreateSessionBody {
    profile: String,
    #[serde(default)]
    id: Option<SessionId>,
    #[serde(default)]
    workdir: Option<String>,
    /// Override/enable containerisation: run every node inside a
    /// container built from this image.
    #[serde(default)]
    image: Option<String>,
    /// Extra bind mounts (`HOST[:CONTAINER[:ro|rw]]`).
    #[serde(default)]
    mounts: Vec<String>,
    /// Container provider (`podman`, `docker`, or a path). Autodetected
    /// when omitted.
    #[serde(default)]
    provider: Option<String>,
    /// If true (default), the daemon spawns the per-session host
    /// process. Tests can set this to false to drive the host
    /// themselves.
    #[serde(default = "default_true")]
    spawn_host: bool,
    /// Seconds to wait for the host to become healthy. 0 = don't wait.
    #[serde(default = "default_ready_timeout")]
    ready_timeout: u64,
}

fn default_true() -> bool {
    true
}

fn default_ready_timeout() -> u64 {
    10
}

async fn list_sessions(
    State(s): State<Arc<AppState>>,
) -> Result<Json<Vec<SessionState>>, ApiError> {
    let ids = s.ctl.session_list()?;
    let mut states = Vec::with_capacity(ids.len());
    for id in ids {
        match s.ctl.session_state(&id) {
            Ok(state) => states.push(state),
            Err(_) => continue,
        }
    }
    Ok(Json(states))
}

async fn create_session(
    State(s): State<Arc<AppState>>,
    Json(body): Json<CreateSessionBody>,
) -> Result<Json<SessionDef>, ApiError> {
    // validate profile, resolving it so container overrides can apply.
    let mut profile = s.ctl.profile_get(&body.profile)?;
    let profile_ref = if body.image.is_some() || !body.mounts.is_empty() || body.provider.is_some()
    {
        let mut mounts = Vec::with_capacity(body.mounts.len());
        for m in &body.mounts {
            mounts.push(
                mirage_core::profile::FileMount::parse(m)
                    .map_err(mirage_core::error::MirageError::other)?,
            );
        }
        match &mut profile.containerize {
            Some(c) => {
                if let Some(img) = body.image {
                    c.image = img;
                }
                if let Some(p) = body.provider {
                    c.provider = Some(p);
                }
                c.mounts.extend(mounts);
            }
            None => {
                let image = body.image.ok_or_else(|| {
                    mirage_core::error::MirageError::other(
                        "mounts/provider require a containerised profile or image",
                    )
                })?;
                profile.containerize = Some(mirage_core::profile::ContainerizedDef {
                    provider: body.provider,
                    image,
                    mounts,
                    devices: Vec::new(),
                    groups: Vec::new(),
                });
            }
        }
        MaybeRef::Owned(profile)
    } else {
        MaybeRef::Ref(body.profile)
    };
    let def = s.ctl.session_create(CreateSessionRequest {
        id: body.id,
        profile: profile_ref,
        workdir: body.workdir.unwrap_or_else(|| {
            std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or("/".to_string())
        }),
    })?;
    if body.spawn_host {
        mirage_ctl::spawn_host_for(&def.id)?;
        if body.ready_timeout > 0 {
            s.ctl
                .session_wait_ready(&def.id, Duration::from_secs(body.ready_timeout))?;
        }
    }
    Ok(Json(def))
}

async fn get_session(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<SessionState>, ApiError> {
    let id = parse_session_id(&id)?;
    Ok(Json(s.ctl.session_state(&id)?))
}

async fn delete_session(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Ok>, ApiError> {
    let id = parse_session_id(&id)?;
    s.ctl.session_destroy(&id)?;
    Ok(ok())
}

// ---- execs -----------------------------------------------------------------

#[derive(Serialize)]
struct ExecListItem {
    id: ExecId,
    status: ExecStatus,
}

async fn list_execs(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Vec<ExecListItem>>, ApiError> {
    let id = parse_session_id(&id)?;
    let ids = s.ctl.exec_list(&id)?;
    let mut out = Vec::with_capacity(ids.len());
    for eid in ids {
        let r = ExecRef {
            session: id.clone(),
            exec: eid.clone(),
        };
        let status = s.ctl.exec_status(&r).unwrap_or_default();
        out.push(ExecListItem { id: eid, status });
    }
    Ok(Json(out))
}

#[derive(Deserialize)]
struct CreateExecBody {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    workdir: Option<String>,
    #[serde(default)]
    keep: bool,
}

#[derive(Serialize)]
struct CreateExecResp {
    id: ExecId,
}

async fn create_exec(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(body): Json<CreateExecBody>,
) -> Result<Json<CreateExecResp>, ApiError> {
    let session = parse_session_id(&id)?;
    let def = ExecDef {
        timestamp: Utc::now(),
        session: session.clone(),
        exec: ExecArgs {
            command: body.command,
            args: body.args,
            env: body.env,
            workdir: body.workdir,
        },
        worker_exec: None,
        keep: body.keep,
    };
    let r = s.ctl.session_exec(&def)?;
    Ok(Json(CreateExecResp { id: r.exec }))
}

async fn get_exec(
    State(s): State<Arc<AppState>>,
    Path((id, exec)): Path<(String, String)>,
) -> Result<Json<ExecStatus>, ApiError> {
    let r = ExecRef {
        session: parse_session_id(&id)?,
        exec: parse_exec_id(&exec)?,
    };
    Ok(Json(s.ctl.exec_status(&r)?))
}

async fn delete_exec(
    State(s): State<Arc<AppState>>,
    Path((id, exec)): Path<(String, String)>,
) -> Result<Json<Ok>, ApiError> {
    let r = ExecRef {
        session: parse_session_id(&id)?,
        exec: parse_exec_id(&exec)?,
    };
    s.ctl.exec_remove(&r)?;
    Ok(ok())
}

#[derive(Deserialize)]
struct SignalBody {
    signal: i32,
}

async fn signal_exec(
    State(s): State<Arc<AppState>>,
    Path((id, exec)): Path<(String, String)>,
    Json(body): Json<SignalBody>,
) -> Result<Json<Ok>, ApiError> {
    let r = ExecRef {
        session: parse_session_id(&id)?,
        exec: parse_exec_id(&exec)?,
    };
    s.ctl.exec_signal(&r, body.signal)?;
    Ok(ok())
}

#[derive(Deserialize)]
struct StdinBody {
    /// Raw text to write (utf-8). Use `data_b64` for binary.
    #[serde(default)]
    data: Option<String>,
    #[serde(default)]
    data_b64: Option<String>,
}

async fn stdin_exec(
    State(s): State<Arc<AppState>>,
    Path((id, exec)): Path<(String, String)>,
    Json(body): Json<StdinBody>,
) -> Result<Json<Ok>, ApiError> {
    let r = ExecRef {
        session: parse_session_id(&id)?,
        exec: parse_exec_id(&exec)?,
    };
    let bytes: Vec<u8> = if let Some(s) = body.data {
        s.into_bytes()
    } else if let Some(b64) = body.data_b64 {
        use base64_decode::decode;
        decode(&b64).map_err(|e| ApiError {
            status: StatusCode::BAD_REQUEST,
            message: format!("invalid base64: {e}"),
        })?
    } else {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "must supply `data` or `data_b64`".to_string(),
        });
    };
    s.ctl.session_stdin(&r, &bytes)?;
    Ok(ok())
}

// Tiny dependency-free base64 decoder (only need decode for stdin).
mod base64_decode {
    pub fn decode(input: &str) -> Result<Vec<u8>, &'static str> {
        let input: String = input.chars().filter(|c| !c.is_whitespace()).collect();
        let bytes = input.as_bytes();
        let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
        let mut buf: u32 = 0;
        let mut bits: u32 = 0;
        for &b in bytes {
            let v: u32 = match b {
                b'A'..=b'Z' => (b - b'A') as u32,
                b'a'..=b'z' => (b - b'a' + 26) as u32,
                b'0'..=b'9' => (b - b'0' + 52) as u32,
                b'+' => 62,
                b'/' => 63,
                b'=' => break,
                _ => return Err("invalid char"),
            };
            buf = (buf << 6) | v;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push((buf >> bits) as u8);
            }
        }
        Ok(out)
    }
}

// ---- attach (websocket) ----------------------------------------------------

async fn attach_exec(
    State(s): State<Arc<AppState>>,
    Path((id, exec)): Path<(String, String)>,
    ws: WebSocketUpgrade,
) -> Result<Response, ApiError> {
    let r = ExecRef {
        session: parse_session_id(&id)?,
        exec: parse_exec_id(&exec)?,
    };
    // Validate up-front so callers get a clean HTTP error instead of a
    // mysterious closed websocket.
    let _ = s.ctl.exec_status(&r)?;
    Ok(ws.on_upgrade(move |socket| attach_loop(s, r, socket)))
}

async fn attach_loop(state: Arc<AppState>, r: ExecRef, mut socket: WebSocket) {
    let mut stream = match state.ctl.session_attach(&r) {
        Ok(s) => s,
        Err(e) => {
            let _ = socket
                .send(Message::Text(
                    serde_json::to_string(&serde_json::json!({"error": e.to_string()}))
                        .unwrap()
                        .into(),
                ))
                .await;
            let _ = socket.close().await;
            return;
        }
    };
    while let Some(pkt) = stream.next().await {
        let frame = encode_packet(&pkt);
        if socket.send(Message::Text(frame.into())).await.is_err() {
            return;
        }
        if matches!(pkt, StreamPacket::ExecExit { .. }) {
            let _ = socket.close().await;
            return;
        }
    }
}

fn encode_packet(pkt: &StreamPacket) -> String {
    // Re-encode `Output` so `data` is utf-8 text where possible; clients
    // that need raw bytes can pull them out of the Vec<u8> in the
    // canonical encoding.
    serde_json::to_string(pkt).unwrap_or_else(|_| "{}".to_string())
}
