// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// A `discover`-style example, loosely mirroring the gpumetrics CLI: create a
// collector, print the sockets / GPUs it found and a handful of metrics, then
// read a few values off GPU 0.
//
// Plugin loading is controlled by env vars so this runs against either the real
// amdsmi plugin or the hardware-free mocks:
//
//   GPUMETRICS_PLUGIN_PATH   extra plugin search dir (e.g. build/plugins/amdsmi)
//   GPUMETRICS_PLUGINS       comma-separated plugin names to restrict to
//
// It degrades gracefully: no plugins / no GPU simply prints an empty topology.

use std::env;

use gpumetrics::{Collector, Value};

fn main() {
    let mut builder = Collector::builder();

    if let Ok(path) = env::var("GPUMETRICS_PLUGIN_PATH") {
        for p in path.split(':').filter(|s| !s.is_empty()) {
            builder = builder.plugin_path(p);
        }
    }
    if let Ok(plugins) = env::var("GPUMETRICS_PLUGINS") {
        for p in plugins.split(',').filter(|s| !s.is_empty()) {
            builder = builder.plugin(p);
        }
    }

    let collector = match builder.build() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to create collector: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "Discovered {} GPU(s) across {} socket(s)\n",
        collector.gpu_count(),
        collector.socket_count()
    );

    println!(
        "{:<5} {:<7} {:<26} {:<14} {:<6} PARTITIONS",
        "GPU", "SOCKET", "NAME", "BDF", "KFD"
    );
    for gpu in collector.gpus() {
        let bdf = gpu.bdf();
        let parts = if gpu.partitions().is_empty() {
            "-".to_string()
        } else {
            gpu.partitions()
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        println!(
            "{:<5} {:<7} {:<26} {:<14} {:<6} {}",
            gpu.ordinal(),
            gpu.socket_id(),
            gpu.name(),
            if bdf.is_empty() { "-" } else { &bdf },
            gpu.kfd_node_id(),
            parts,
        );
    }

    let metrics = collector.metrics();
    println!("\n{} metric(s) available:", metrics.len());
    for m in metrics.iter().take(10) {
        println!(
            "  {:<26} {:<6} [{}] ({}) scope={}",
            m.key, m.unit, m.provider, m.value_type, m.scope
        );
    }
    if metrics.len() > 10 {
        println!("  ... and {} more", metrics.len() - 10);
    }

    // Read a few metrics off GPU 0, if present.
    if collector.gpu_count() == 0 {
        println!("\nNo GPUs to read.");
        return;
    }
    let Some(entity) = collector.gpu_entity(0) else {
        return;
    };

    // Pick up to 5 GPU-scoped metric keys to sample.
    let keys: Vec<String> = metrics
        .iter()
        .filter(|m| m.scope.gpu())
        .take(5)
        .map(|m| m.key.clone())
        .collect();
    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();

    println!("\nReadings for gpu 0:");
    let samples = collector.read_batch(&entity, &key_refs);
    for (key, sample) in key_refs.iter().zip(samples.iter()) {
        match sample.value() {
            Some(Value::Str(s)) => println!("  {key:<26} {s}"),
            Some(v) => println!("  {key:<26} {v}"),
            None => println!("  {key:<26} <{}>", sample.status),
        }
    }
}
