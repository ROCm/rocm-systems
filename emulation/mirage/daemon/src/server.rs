//! Server entry points: builds the axum [`Router`] and runs it.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::api;
use crate::spa;
use crate::state::AppState;

/// Construct the full router (API + SPA fallback).
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .nest("/api", api::router(state.clone()))
        .fallback(spa::handle)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
}

/// Bind to `addr` and serve `router` until the process is killed.
pub async fn serve(addr: SocketAddr, router: Router) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    tracing::info!("mirage daemon listening on http://{bound}");
    axum::serve(listener, router.into_make_service()).await?;
    Ok(())
}
