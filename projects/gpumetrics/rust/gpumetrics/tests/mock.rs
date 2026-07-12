// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Integration test against the hardware-free mock plugins (mockA / mockB), the
// same fixtures the C++ test_capi uses. No GPU required.
//
// Point it at the mock .so directory via GPUM_TEST_PLUGIN_DIR (the C++ tests use
// the same variable). If it is unset the test skips, so a plain `cargo test`
// without the mocks built does not spuriously fail.

use gpumetrics::{Collector, Value};

fn mock_dir() -> Option<String> {
    std::env::var("GPUM_TEST_PLUGIN_DIR").ok()
}

#[test]
fn mock_topology_and_read() {
    let Some(dir) = mock_dir() else {
        eprintln!("GPUM_TEST_PLUGIN_DIR not set; skipping mock plugin test");
        return;
    };

    let collector = Collector::builder()
        .plugin_path(dir)
        .plugins(["mockA", "mockB"])
        .provider_priority(["mockA", "mockB"])
        .build()
        .expect("collector should be created with mock plugins");

    // Two mock GPUs (correlated across the two plugins).
    assert_eq!(collector.gpu_count(), 2, "expected 2 mock GPUs");

    // Reading mock.temp on gpu 0 yields 40.0 (40 + local index 0).
    let entity = collector
        .resolve("gpu:0")
        .expect("gpu:0 should resolve to an entity");
    let sample = collector.read(&entity, "mock.temp");
    assert!(sample.is_ok(), "mock.temp read status: {}", sample.status);
    assert_eq!(sample.value(), Some(&Value::F64(40.0)));

    // A missing metric surfaces as a non-OK status, not a panic / error.
    let missing = collector.read(&entity, "does.not.exist");
    assert!(!missing.is_ok());
    assert!(missing.value().is_none());
}

#[test]
fn mock_metric_registry() {
    let Some(dir) = mock_dir() else {
        eprintln!("GPUM_TEST_PLUGIN_DIR not set; skipping mock plugin test");
        return;
    };

    let collector = Collector::builder()
        .plugin_path(dir)
        .plugins(["mockA", "mockB"])
        .provider_priority(["mockA", "mockB"])
        .build()
        .expect("collector should be created with mock plugins");

    let metrics = collector.metrics();
    assert!(!metrics.is_empty(), "expected some metrics");

    // "mock.shared" is offered by both plugins; mockA wins by priority.
    let shared = metrics
        .iter()
        .find(|m| m.key == "mock.shared")
        .expect("mock.shared metric should exist");
    assert_eq!(shared.provider, "mockA");
}

#[test]
fn mock_batch_read() {
    let Some(dir) = mock_dir() else {
        eprintln!("GPUM_TEST_PLUGIN_DIR not set; skipping mock plugin test");
        return;
    };

    let collector = Collector::builder()
        .plugin_path(dir)
        .plugins(["mockA", "mockB"])
        .provider_priority(["mockA", "mockB"])
        .build()
        .expect("collector should be created with mock plugins");

    let entity = collector.resolve("gpu:0").expect("gpu:0 resolves");
    let samples = collector.read_batch(&entity, &["mock.temp", "mock.shared", "missing"]);
    assert_eq!(samples.len(), 3);
    assert!(samples[0].is_ok());
    assert_eq!(samples[0].value(), Some(&Value::F64(40.0)));
    assert!(samples[1].is_ok());
    assert!(!samples[2].is_ok(), "missing key should be non-OK");
}
