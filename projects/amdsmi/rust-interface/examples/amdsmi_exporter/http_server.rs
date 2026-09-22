// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use crate::amdsmi_collectors::AmdsmiCollectors;
use axum::response::{IntoResponse, Response};
use axum::{routing::get, Router};
use hyper::Server;
use prometheus_client::encoding::text::encode;
use prometheus_client::registry::Registry;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;

async fn serve_req(registry: Arc<Mutex<Registry>>) -> impl IntoResponse {
    let mut buffer = String::new();
    let registry = registry.lock().await;
    encode(&mut buffer, &*registry).unwrap();
    Response::builder()
        .header("Content-Type", "text/plain; version=0.0.4")
        .body(buffer)
        .unwrap()
}

pub async fn run_http_server(collectors: &Arc<Mutex<AmdsmiCollectors>>, addr: SocketAddr) {
    let app = Router::new().route(
        "/metrics",
        get({
            let collectors = Arc::clone(&collectors);
            move || {
                let collectors = Arc::clone(&collectors);
                async move {
                    let mut collectors = collectors.lock().await;
                    let registry = Arc::new(Mutex::new(collectors.run_collect()));
                    serve_req(registry).await
                }
            }
        }),
    );

    println!("Listening on http://{}", addr);

    Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
