// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Backing diagnostics on queues built through the production constructor.

use super::super::super::drm;
use super::*;
use std::os::fd::AsRawFd;

#[test]
fn queue_failure_boundaries_gate_passive_queries_and_preserve_update_error() {
    const CHILD: &str = "ROCDDI_QUEUE_BOUNDARY_TEST_CHILD";
    let Ok(mode) = std::env::var(CHILD) else {
        // Per-process environment isolates the production opt-in check from
        // parallel host tests without mutating the test runner's environment.
        for mode in ["unset", "0", "true", "1"] {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            child.args([
                "--exact",
                "driver::linux_kfd::queue::tests::backing_ranges::queue_failure_boundaries_gate_passive_queries_and_preserve_update_error",
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
                String::from_utf8_lossy(&output.stderr),
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        }
        return;
    };
    let enabled = mode == "1";
    let fixture = Fixture::new(true);
    let mut queue = fixture
        .create(descriptor(aql(QueueProducerMode::Single)))
        .unwrap();
    {
        let mut state = fixture.state.lock().unwrap();
        state.wait_update_count = 0;
        state.update_errno.push_back(Some(13));
    }
    let render = fixture.vm.render().unwrap().as_raw_fd();
    let script = enabled.then_some([
        drm::TestCall::QueryGpuVmFault(render, Ok(drm::GpuVmFault::default())),
        drm::TestCall::QueryGpuVmFault(render, Err(25)),
    ]);
    drm::with_script(script.into_iter().flatten(), || {
        queue.diagnose_incomplete_cache_wait();
        let error = queue.inactivate().unwrap_err();
        assert!(error.to_string().contains("AMDKFD_IOC_UPDATE_QUEUE"));
        assert_eq!(error.native_status().unwrap().errno_code(), Some(13));
        assert!(queue.active);
    });
    {
        let state = fixture.state.lock().unwrap();
        assert_eq!(state.native_resets, 0);
        let expected: &[(u32, usize)] = if enabled {
            &[(20, 0), (19, 0), (20, 1), (19, 1)]
        } else {
            &[]
        };
        assert_eq!(
            &state.wait_update_boundaries[..state.wait_update_count],
            expected,
        );
    }
    // A successful normal inactivation adds no exception reads or DRM queries,
    // even with diagnostics enabled after the failed update.
    let waits = fixture.state.lock().unwrap().native_waits;
    drm::with_script([], || queue.inactivate().unwrap());
    assert_eq!(fixture.state.lock().unwrap().native_waits, waits);
    queue.destroy().unwrap();
    fixture.assert_released();
}

#[test]
fn backing_range_diagnostic_uses_retained_native_and_svm_extents_without_queries() {
    for native_context in [false, true] {
        let fixture = Fixture::new(true);
        if native_context {
            fixture.state.lock().unwrap().svm_errno = Some(errno::EOPNOTSUPP);
        }
        let mut queue = fixture
            .create(descriptor(aql(QueueProducerMode::Single)))
            .unwrap();
        let (eop, native, svm, waits) = {
            let state = fixture.state.lock().unwrap();
            (
                *state
                    .buffers
                    .values()
                    .find(|buffer| buffer.flags & uapi::VRAM != 0)
                    .unwrap(),
                state
                    .buffers
                    .values()
                    .find(|buffer| buffer.size > 4096)
                    .copied(),
                state.svm_context,
                state.native_waits,
            )
        };
        let mut output = Vec::new();
        queue
            .write_backing_range_diagnostic(&mut output, 0)
            .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert_eq!(fixture.state.lock().unwrap().native_waits, waits);
        assert!(output.contains(&format!(
            "role=eop; kind=vram; cpu=unavailable(no_host_mapping); native_gpu_base={:#x}; native_size_bytes={}; logical_size_bytes={}",
            eop.va, eop.size, eop.size,
        )));
        assert!(output.contains(&format!(
            "native_flags={:#x}; access_bits=0x7; freeing=false; uncertain=false; gpu_map_state=mapped",
            eop.flags,
        )));
        if native_context {
            let native = native.unwrap();
            assert!(svm.is_none());
            assert!(output.contains(&format!(
                "role=context; kind=gtt; cpu={:#x}; native_gpu_base={:#x}; native_size_bytes={}",
                native.va, native.va, native.size,
            )));
            assert!(output.contains(&format!(
                "native_flags={:#x}; access_bits=0x7; freeing=false; uncertain=false; gpu_map_state=mapped",
                native.flags,
            )));
        } else {
            let (address, size) = svm.unwrap();
            assert!(native.is_none());
            assert!(output.contains(&format!(
                "role=context; kind=svm; cpu={address:#x}; gpu={address:#x}; native_size_bytes={size}; cpu_writable=true; svm_registration=completed_at_creation; gpu_permissions=unavailable(not_retained)",
            )));
        }
        assert!(output.contains(
            "context_layout=retained_aql_properties; xcc_count=1; context_stride_bytes=65536; context_span_size_bytes=65536; control_stack_prefix_size_bytes_per_xcc=4096; debug_offset_bytes=65536; debug_size_bytes=unavailable(not_retained); post_context_allocation_tail_size_bytes=4096; vm_trap_owner=unavailable(no_lifecycle_lock)"
        ));
        let mut full = &mut [0_u8; 0][..];
        assert!(queue.write_backing_range_diagnostic(&mut full, 0).is_err());
        assert_eq!(fixture.state.lock().unwrap().native_waits, waits);
        queue.inactivate().unwrap();
        queue.destroy().unwrap();
        fixture.assert_released();
    }
}

#[test]
fn backing_range_diagnostic_does_not_invent_missing_layout_or_allocations() {
    let fixture = Fixture::new(true);
    let mut queue = fixture.create(descriptor(QueueParameters::Sdma)).unwrap();
    let mut output = Vec::new();
    queue
        .write_backing_range_diagnostic(&mut output, 0)
        .unwrap();
    assert_eq!(
        output,
        concat!(
            "rocddi: queue_backing; queue_id=0; role=eop; backing=absent\n",
            "rocddi: queue_backing; queue_id=0; role=context; backing=absent\n",
            "rocddi: queue_backing; queue_id=0; context_layout=unavailable(not_retained); vm_trap_owner=unavailable(no_lifecycle_lock)\n",
        ).as_bytes(),
    );
    queue.inactivate().unwrap();
    queue.destroy().unwrap();
    fixture.assert_released();
}
