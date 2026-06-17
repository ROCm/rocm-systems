//! SPA fallback: serve embedded assets compiled into
//! [`mirage_dashboard`], with `index.html` fallback for client-side
//! routing.

use axum::body::Body;
use axum::http::{HeaderValue, StatusCode, Uri, header};
use axum::response::Response;

pub async fn handle(uri: Uri) -> Response {
    let path = uri.path();
    let asset = mirage_dashboard::spa::get_spa(path);
    match asset {
        Some(a) => Response::builder()
            .status(StatusCode::OK)
            .header(
                header::CONTENT_TYPE,
                HeaderValue::from_static(a.content_type),
            )
            .body(Body::from(a.bytes))
            .unwrap(),
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("not found"))
            .unwrap(),
    }
}
