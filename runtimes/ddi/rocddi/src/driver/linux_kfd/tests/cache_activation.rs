// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Production queue acquisition with ordered KFD and DRM native replies.

use super::super::super::drm;
use super::*;
use std::os::fd::AsRawFd;

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one isolated matrix checks native ordering and retained failure evidence"
)]
fn cache_queue_activation_capture_is_gated_and_brackets_native_create() {
    const CHILD: &str = "ROCDDI_CACHE_ACTIVATION_CHILD";
    let Ok(mode) = std::env::var(CHILD) else {
        for mode in ["unset", "0", "true", "1"] {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            child.args([
                "--exact",
                "driver::linux_kfd::queue::tests::cache_activation::cache_queue_activation_capture_is_gated_and_brackets_native_create",
                "--nocapture",
            ]);
            child.env(CHILD, mode);
            if mode == "unset" {
                child.env_remove("ROCDDI_CODE_CACHE_DIAGNOSTICS");
            } else {
                child.env("ROCDDI_CODE_CACHE_DIAGNOSTICS", mode);
            }
            let output = child.output().unwrap();
            assert!(
                output.status.success(),
                "mode {mode}: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        }
        return;
    };
    let mut previous_ordinal = None;
    for (before, after) in [
        (
            Ok(drm::GpuVmFault::default()),
            Ok(drm::GpuVmFault::default()),
        ),
        (
            Err(5),
            Ok(drm::GpuVmFault {
                address: 0x1234_5000,
                status: 0x0030_0a31,
                vmhub: 0,
            }),
        ),
        (
            Ok(drm::GpuVmFault {
                address: 0x1234_5000,
                status: 0x0030_0a31,
                vmhub: 0,
            }),
            Err(25),
        ),
    ] {
        let mut ordinary_waits = None;
        for code_cache in [false, true] {
            let fixture = Fixture::new(true);
            let fd = fixture.vm.render().unwrap().as_raw_fd();
            let enabled = mode == "1" && code_cache;
            let script = enabled.then_some([
                drm::TestCall::QueryGpuVmFault(fd, before),
                drm::TestCall::QueryGpuVmFault(fd, after),
            ]);
            let mut queue = drm::with_script(script.into_iter().flatten(), || {
                fixture
                    .create_for_purpose(
                        descriptor(aql(QueueProducerMode::Single)),
                        DriverContextLifetime::Process,
                        code_cache,
                    )
                    .unwrap()
            });
            let state = fixture.state.lock().unwrap();
            // The last backing MAP and SVM registration precede the first
            // query; exactly one query has run at native CREATE entry.
            assert_eq!(
                state.fault_queries_remaining_at_last_map,
                Some(if enabled { 2 } else { 0 })
            );
            assert_eq!(
                state.fault_queries_remaining_at_svm,
                Some(if enabled { 2 } else { 0 })
            );
            assert_eq!(
                state.fault_queries_remaining_at_create,
                Some(usize::from(enabled))
            );
            assert_eq!(state.creates, 1);
            assert_eq!(state.native_resets, 0);
            if let Some(waits) = ordinary_waits {
                assert_eq!(state.native_waits, waits);
            } else {
                ordinary_waits = Some(state.native_waits);
            }
            let ring = *state
                .buffers
                .values()
                .find(|allocation| allocation.va == queue.info.ring_device_address)
                .unwrap();
            let eop = *state
                .buffers
                .values()
                .find(|allocation| allocation.flags & uapi::VRAM != 0)
                .unwrap();
            let waits = state.native_waits;
            drop(state);
            // Both snapshots are complete before the constructor returns any
            // producer pointers. Every AQL slot is still INVALID, with no body
            // or producer write-index stores from cache publication.
            let mut ring_bytes = vec![0; usize::try_from(ring.size).unwrap()];
            File::open("/proc/self/mem")
                .unwrap()
                .read_exact_at(&mut ring_bytes, queue.info.ring_host_address as u64)
                .unwrap();
            for packet in ring_bytes.chunks_exact(64) {
                assert_eq!(packet[0], 1);
                assert!(packet[1..].iter().all(|byte| *byte == 0));
            }
            assert_eq!(fixture.read_aql_u64(&queue, AQL_WRITE_OFFSET), 0);
            let mut output = Vec::new();
            drm::with_script([], || {
                queue
                    .write_cache_activation(&mut output, "test-failure")
                    .unwrap();
            });
            assert_eq!(fixture.state.lock().unwrap().native_waits, waits);
            let output = String::from_utf8(output).unwrap();
            assert_eq!(!output.is_empty(), enabled);
            if enabled {
                assert!(
                    output.contains(
                        "queue_id=Some(0); expected_gpu_id=42; create_request=CreateQueue"
                    )
                );
                assert!(
                    output.contains(&format!("eop_address: {}, eop_size: {}", eop.va, eop.size))
                );
                assert!(output.contains("boundary=fresh-render-before-ACQUIRE_VM; drm_gpuvm_fault_cache=unavailable(not_captured_at_render_open)"));
                assert!(output.contains("boundary=code-cache-before-CREATE_QUEUE"));
                assert!(
                    output.contains("boundary=code-cache-after-CREATE_QUEUE-before-producer-store")
                );
                assert!(output.contains("untimestamped_may_be_stale_or_incomplete"));
                for reply in [before, after] {
                    match reply {
                        Ok(fault) => assert!(output.contains(&format!(
                            "raw(address={:#x}, status={:#x}, vmhub={:#x})",
                            fault.address, fault.status, fault.vmhub
                        ))),
                        Err(code) => {
                            assert!(output.contains(&format!("query_failed(errno=Some({code})")));
                        }
                    }
                }
                let ordinal = output
                    .split("cache_queue_capture_ordinal=")
                    .nth(1)
                    .unwrap()
                    .split(';')
                    .next()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap();
                if let Some(previous) = previous_ordinal {
                    assert!(ordinal > previous);
                }
                previous_ordinal = Some(ordinal);
            }
            queue.destroy().unwrap();
            fixture.assert_released();
        }
    }
    // An independently unavailable cache read cannot replace native EACCES.
    let fixture = Fixture::new(true);
    fixture.state.lock().unwrap().create_errno = Some(13);
    let fd = fixture.vm.render().unwrap().as_raw_fd();
    let script = (mode == "1").then_some(drm::TestCall::QueryGpuVmFault(fd, Err(25)));
    let error = drm::with_script(script, || {
        fixture
            .create_for_purpose(
                descriptor(aql(QueueProducerMode::Single)),
                DriverContextLifetime::Process,
                true,
            )
            .err()
            .unwrap()
    });
    assert_eq!(error.native_status().unwrap().errno_code(), Some(13));
    assert!(error.to_string().contains("AMDKFD_IOC_CREATE_QUEUE"));
    fixture.assert_released();
}
