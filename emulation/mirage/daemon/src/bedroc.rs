//! `/api/bedroc` routes: HTTP access to the Project Bedroc planner.
//!
//! These handlers mirror the `mirage bedroc` CLI: list the tool
//! catalogue, plan a proof, plan-and-execute with caching, and fuzz the
//! catalogue. Like the rest of the daemon they are a thin shim — all the
//! logic lives in [`mirage_bedroc`].

use std::path::PathBuf;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use mirage_bedroc::executor::fingerprint_source;
use mirage_bedroc::{
    BedrocRequest, Engine, Environment, Executor, GoalKind, SourceKind, ToolCatalog,
};
use mirage_solver::Cache;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

/// Mount the bedroc routes. They are stateless (they probe the host and
/// registry directly) but share the daemon's state type for nesting.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/bedroc/tools", get(list_tools))
        .route("/bedroc/plan", post(plan))
        .route("/bedroc/run", post(run))
        .route("/bedroc/fuzz", post(fuzz))
}

/// A simple error carrying an HTTP status and message.
struct BedrocError(StatusCode, String);

impl IntoResponse for BedrocError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({"error": self.1}))).into_response()
    }
}

fn bad_request(msg: impl Into<String>) -> BedrocError {
    BedrocError(StatusCode::BAD_REQUEST, msg.into())
}

fn default_tools_dir() -> PathBuf {
    mirage_core::paths::mirage_config_dir()
        .join("bedroc")
        .join("tools")
}

fn cache_path() -> PathBuf {
    mirage_core::paths::mirage_cache_dir()
        .join("bedroc")
        .join("cache.json")
}

fn load_catalog() -> Result<ToolCatalog, BedrocError> {
    let mut catalog = ToolCatalog::builtin();
    catalog.load_dir(&default_tools_dir()).map_err(|e| {
        BedrocError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;
    Ok(catalog)
}

/// Probe GPUs and inject installed emulators as bedroc "tools".
fn probe_environment() -> Environment {
    let mut env = Environment::probe();
    for spec in mirage_ctl::registry() {
        if spec.installed {
            env = env.with_tool(spec.name);
        }
    }
    env
}

/// The request body shared by `plan` and `run`.
#[derive(Debug, Deserialize)]
struct RequestBody {
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    source_kind: Option<String>,
    targets: Vec<String>,
    goals: Vec<String>,
}

fn build_request(body: &RequestBody) -> Result<BedrocRequest, BedrocError> {
    let kind_str = body.source_kind.as_deref().unwrap_or("hip");
    let source_kind =
        SourceKind::parse(kind_str).ok_or_else(|| bad_request(format!("unknown source kind `{kind_str}`")))?;
    if body.targets.is_empty() {
        return Err(bad_request("at least one target is required"));
    }
    if body.goals.is_empty() {
        return Err(bad_request("at least one property to prove is required"));
    }
    let mut goals = Vec::new();
    for g in &body.goals {
        goals.push(GoalKind::parse(g).ok_or_else(|| bad_request(format!("unknown property `{g}`")))?);
    }
    Ok(BedrocRequest::build(
        body.source.clone().unwrap_or_default(),
        source_kind,
        body.targets.clone(),
        goals,
    ))
}

#[derive(Serialize)]
struct ToolEntry {
    #[serde(flatten)]
    manifest: mirage_bedroc::ToolManifest,
    available: bool,
}

async fn list_tools() -> Result<Json<Vec<ToolEntry>>, BedrocError> {
    let catalog = load_catalog()?;
    let env = probe_environment();
    let tools = catalog
        .iter()
        .map(|m| {
            let needs_met = m.needs_tools.iter().all(|t| env.has_tool(t));
            let gpu_met = !m.needs_gpu || env.has_gpu();
            ToolEntry {
                manifest: m.clone(),
                available: needs_met && gpu_met,
            }
        })
        .collect();
    Ok(Json(tools))
}

async fn plan(Json(body): Json<RequestBody>) -> Result<Json<mirage_bedroc::Proof>, BedrocError> {
    let catalog = load_catalog()?;
    let env = probe_environment();
    let request = build_request(&body)?;
    let engine = Engine::new(catalog, env);
    let proof = engine
        .prove(&request)
        .map_err(|e| BedrocError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(proof))
}

#[derive(Serialize)]
struct RunResponse {
    proof: mirage_bedroc::Proof,
    execution: mirage_bedroc::ExecutionReport,
}

async fn run(Json(body): Json<RequestBody>) -> Result<Json<RunResponse>, BedrocError> {
    let catalog = load_catalog()?;
    let env = probe_environment();
    let request = build_request(&body)?;
    let engine = Engine::new(catalog, env);

    let cache_file = cache_path();
    let mut cache = Cache::load(&cache_file).unwrap_or_default();
    let source_fp = fingerprint_source(&request.source);
    let execution = {
        let executor = Executor::new(&engine);
        executor
            .run(&request, &mut cache, &source_fp)
            .map_err(|e| BedrocError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    };
    let proof = engine
        .prove_with_cache(&request, &cache)
        .map_err(|e| BedrocError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    cache.save(&cache_file).ok();

    Ok(Json(RunResponse { proof, execution }))
}

#[derive(Debug, Deserialize)]
struct FuzzBody {
    #[serde(default)]
    iterations: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
}

async fn fuzz(Json(body): Json<FuzzBody>) -> Result<Json<mirage_bedroc::fuzz::FuzzSummary>, BedrocError> {
    let catalog = load_catalog()?;
    // Assume a fully capable environment so every tool route is reachable.
    let mut env = Environment::probe();
    for spec in mirage_ctl::registry() {
        env = env.with_tool(spec.name);
    }
    let engine = Engine::new(catalog, env);
    let summary =
        mirage_bedroc::fuzz::fuzz_catalog(&engine, body.iterations.unwrap_or(200), body.seed.unwrap_or(0));
    Ok(Json(summary))
}
