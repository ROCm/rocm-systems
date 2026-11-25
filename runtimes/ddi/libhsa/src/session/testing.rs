// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host-only HSA session fixture using real core session ownership.

#![allow(clippy::unwrap_used)]

use super::*;
use rocddi::topology::host::{HostCaches, HostTopologyGraphSnapshot};

/// Builds an HSA session without discovery or native device acquisition.
pub(crate) fn session_without_devices(native_session: NativeSession) -> HsaSession {
    let inflight = Arc::new(InFlightTracker::new_process());
    inflight.open(1);
    let control = SessionControl::new(inflight);
    let health = control.health.clone();
    HsaSession {
        agents: Agents {
            host_page_size: 4096,
            system_alloc_max_bytes: 0,
            cpus: Vec::new(),
            full_profile: true,
            caches: Vec::new(),
            topology: TopologyGraphSnapshot {
                host: HostTopologyGraphSnapshot {
                    memory_bytes: None,
                    cpu: None,
                    cpu_apic_ids: Vec::new(),
                    online_cpus: None,
                    caches: HostCaches {
                        records: Vec::new(),
                        complete: true,
                    },
                    domains: Vec::new(),
                    domains_complete: true,
                },
                endpoints: Vec::new(),
                sources: Vec::new(),
                nodes: Vec::new(),
                attachments: Vec::new(),
                routes: Vec::new(),
                caches: Vec::new(),
                memory_locations: Vec::new(),
                memory_banks: Vec::new(),
            },
            gpus: Vec::new(),
        },
        config: HsaConfig {
            core_dump: crate::core_dump::Config::default(),
            queue_fault_message_enabled: true,
            yielding_poll_enabled: false,
            sdma_enabled: false,
            peer_sdma_enabled: false,
            disable_fragment_allocator: false,
        },
        control,
        debugger: None,
        logging: Arc::new(LogOutput::new()),
        next_handle: 0x4853_4101_0000_0000,
        queue_state: QueueState::new(4),
        loader: LoaderState::new(),
        image_state: ImageState::new(),
        memory_state: MemoryState::new(health.clone()),
        signal_state: SignalState::new(&native_session).unwrap(),
        pc_sampling_state: PcSamplingState::new(health.clone()),
        events: SystemEvents::new(),
        native_session: Some(native_session),
    }
}
