// Copyright (C) 2024 Advanced Micro Devices. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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
