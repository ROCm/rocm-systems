// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native Windows host discovery without an installed GPU provider.
#![cfg(windows)]

#[test]
fn native_host_discovery_reports_memory_and_group_qualified_processors() {
    let snapshot = rocddi::topology::host::graph_snapshot();
    assert!(snapshot.memory_bytes.is_some_and(|bytes| bytes != 0));
    assert!(snapshot.cpu.is_some_and(|cpu| cpu.compute_units != 0));
    assert!(
        snapshot
            .online_cpus
            .is_some_and(|cpus| cpus.first().is_some())
    );
    assert!(snapshot.caches.complete);
    assert!(snapshot.domains_complete);
    assert!(snapshot.cpu_apic_ids.is_empty());
}
