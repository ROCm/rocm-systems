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
use mirage_core::common::MaybeRef;
use mirage_core::ctl::{CreateSessionRequest, MirageCtl, StreamPacket};
use mirage_core::exec::{ExecArgs, ExecDef, ExecId, ExecRef, ExecStatus};
use mirage_core::profile::ProfileDef;
use mirage_core::session::{SessionDef, SessionId, SessionState};
use serde::{Deserialize, Serialize};

use crate::state::AppState;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/paths", get(get_paths))
        .route("/profiles", get(list_profiles))
        .route("/profiles/{name}", get(get_profile))
        .route("/profiles/{name}", put(put_profile))
        .route("/profiles/{name}", delete(delete_profile))
        .route("/sessions", get(list_sessions).post(create_session))
        .route(
            "/sessions/{id}",
            get(get_session).delete(delete_session),
        )
        .route(
            "/sessions/{id}/execs",
            get(list_execs).post(create_exec),
        )
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
    profiles: String,
    sessions: String,
}

async fn get_paths() -> Json<PathsResponse> {
    Json(PathsResponse {
        config: mirage_core::paths::xdg_config_home().display().to_string(),
        runtime: mirage_core::paths::xdg_runtime_dir().display().to_string(),
        state: mirage_core::paths::xdg_state_home().display().to_string(),
        profiles: mirage_core::paths::profile_root().display().to_string(),
        sessions: mirage_core::paths::session_root().display().to_string(),
    })
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

// ---- sessions --------------------------------------------------------------

#[derive(Deserialize)]
struct CreateSessionBody {
    profile: String,
    #[serde(default)]
    id: Option<SessionId>,
    #[serde(default)]
    workdir: Option<String>,
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
    // validate profile.
    s.ctl.profile_get(&body.profile)?;
    let def = s.ctl.session_create(CreateSessionRequest {
        id: body.id,
        profile: MaybeRef::Ref(body.profile),
        workdir: body
            .workdir
            .unwrap_or_else(|| std::env::current_dir().map(|p| p.display().to_string()).unwrap_or("/".to_string())),
        container: None,
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
                out.push((buf >> bits) as u8 & 0xff);
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
