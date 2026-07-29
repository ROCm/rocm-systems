//! SPA fallback: serve embedded assets compiled into
//! [`mirage_dashboard`], with `index.html` fallback for client-side
//! routing.

use axum::body::Body;
use axum::http::{HeaderValue, StatusCode, Uri, header};
use axum::response::{IntoResponse as _, Response};

/// Serve an embedded dashboard asset, falling back to the SPA index.
pub(crate) async fn handle(uri: Uri) -> Response {
    let Some(asset) = mirage_dashboard::spa::get_spa(uri.path()) else {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            HeaderValue::from_static(asset.content_type),
        )
        .body(Body::from(asset.bytes))
        // `Response::builder` only fails on an invalid status or header,
        // and both are constants here. Fall back to a plain 500 rather
        // than panicking inside a request handler, which would take the
        // connection down with no explanation.
        .unwrap_or_else(|e| {
            tracing::error!("could not build the SPA response: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        })
}
