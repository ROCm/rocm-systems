// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host-only calls through the production snapshot and native dispatcher.

#![allow(clippy::unwrap_used, clippy::panic)]

use super::super::{ObservedEvents, connection};
use super::*;
use crate::memory::storage::Shared;
use crate::os::ProcessIdentity;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct NativeState {
    /// Pending manual-reset memory event words supplied by the native hook.
    memory: Option<[u64; 4]>,
    /// Pending manual-reset hardware event words supplied by the native hook.
    hardware: Option<[u64; 4]>,
    /// Injected memory-event query error.
    memory_errno: Option<i32>,
    /// Injected hardware-event query error.
    hardware_errno: Option<i32>,
    /// Ordered query/reset calls observed at the native dispatch boundary.
    calls: Vec<(bool, u32)>,
}

fn owner(kfd: Shared<connection::Kfd>) -> LossEvent {
    LossEvent {
        kfd,
        hardware_event_id: AtomicU32::new(19),
        memory_event_id: AtomicU32::new(20),
        hardware_destroy_uncertain: AtomicBool::new(false),
        memory_destroy_uncertain: AtomicBool::new(false),
        observed: Mutex::new(ObservedEvents::new().unwrap()),
        host_aliases: Mutex::new(Vec::new()),
        lost: AtomicBool::new(false),
    }
}

fn fixture() -> (LossEvent, Arc<Mutex<NativeState>>) {
    let state = Arc::new(Mutex::new(NativeState::default()));
    let native = state.clone();
    let kfd = Shared::new(connection::Kfd::with_hook(
        File::open("/dev/null").unwrap(),
        Arc::new(move |call| {
            let mut state = native.lock().unwrap();
            match call {
                connection::Call::Wait(args, event) => {
                    assert_eq!(args.timeout, 0);
                    assert_eq!(args.count, 1);
                    state.calls.push((false, event.event_id));
                    let (payload, errno) = match event.event_id {
                        19 => (state.hardware, state.hardware_errno),
                        20 => (state.memory, state.memory_errno),
                        _ => panic!("read an unowned event ID"),
                    };
                    if let Some(errno) = errno {
                        return Err(io::Error::from_raw_os_error(errno));
                    }
                    args.result = if let Some(payload) = payload {
                        event.payload = payload;
                        super::super::uapi::WAIT_COMPLETE
                    } else {
                        super::super::uapi::WAIT_TIMEOUT
                    };
                }
                connection::Call::ResetEvent(args) => {
                    state.calls.push((true, args.event_id));
                    match args.event_id {
                        19 => state.hardware = None,
                        20 => state.memory = None,
                        _ => panic!("reset an unowned event ID"),
                    }
                }
                connection::Call::DestroyEvent(_) => {}
                _ => panic!("unexpected native operation"),
            }
            Ok(())
        }),
    ))
    .unwrap();
    (owner(kfd), state)
}

/// All mutable observer state, including every retained payload and cursor.
fn observer_state(owner: &LossEvent) -> String {
    let records = owner.observed.lock().unwrap();
    format!(
        "{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
        records
            .events
            .iter()
            .map(|record| (record.sequence, record.gpu_id, record.event))
            .collect::<Vec<_>>(),
        records
            .cursors
            .iter()
            .map(|cursor| cursor.load(Ordering::Acquire))
            .collect::<Vec<_>>(),
        records.next_sequence,
        records.hardware_needs_reset,
        records.memory_needs_reset,
        records.identities,
        owner.lost.load(Ordering::Acquire),
    )
}

#[test]
fn pending_raw_payloads_are_independent_and_do_not_change_owner_state() {
    let (owner, native) = fixture();
    let memory = [1, 2, 0x1234_5000, 0xdead_beef_0000_0043];
    let hardware = [0xfeed_dead_abcd_1234, 6, 7, 8];
    {
        let mut native = native.lock().unwrap();
        native.memory = Some(memory);
        native.hardware = Some(hardware);
    }
    let cursor = Shared::new(std::sync::atomic::AtomicU64::new(7)).unwrap();
    {
        let mut records = owner.observed.lock().unwrap();
        records.cursors.push(cursor);
        records.hardware_needs_reset = true;
        records.memory_needs_reset = true;
    }
    let before = observer_state(&owner);
    let snapshot = owner.passive_snapshot().unwrap();
    assert!(!snapshot.lost_latched);
    assert_eq!(snapshot.retained_memory_sequence, None);
    assert_eq!(snapshot.retained_hardware_sequence, None);
    assert_eq!(snapshot.native_pending_memory_words.unwrap(), Some(memory));
    assert_eq!(
        snapshot.native_pending_hardware_words.unwrap(),
        Some(hardware)
    );
    assert_eq!(observer_state(&owner), before);
    let native = native.lock().unwrap();
    assert_eq!(native.calls, [(false, 20), (false, 19)]);
    assert_eq!(native.memory, Some(memory));
    assert_eq!(native.hardware, Some(hardware));
}

#[test]
fn retained_history_and_terminal_latch_survive_passive_capture() {
    let (owner, native) = fixture();
    {
        let mut native = native.lock().unwrap();
        // Feed the normal observer, including a terminal hardware observation.
        native.hardware = Some([1_u64 << 32, (42_u64 << 32) | 1, 0, 0]);
        native.memory = Some([1, 0, 0x1234_5000, 43]);
    }
    assert_eq!(
        owner.check().unwrap_err().kind(),
        crate::ErrorKind::DeviceLost
    );
    native.lock().unwrap().calls.clear();
    let before = observer_state(&owner);
    let snapshot = owner.passive_snapshot().unwrap();
    assert!(snapshot.lost_latched);
    assert_eq!(snapshot.retained_hardware_sequence, Some(1));
    assert_eq!(snapshot.retained_memory_sequence, Some(2));
    assert!(snapshot.native_pending_memory_words.unwrap().is_none());
    assert!(snapshot.native_pending_hardware_words.unwrap().is_none());
    assert_eq!(observer_state(&owner), before);
    assert_eq!(native.lock().unwrap().calls, [(false, 20), (false, 19)]);
}

#[test]
fn empty_capture_and_each_native_read_error_remain_distinct() {
    for (memory_errno, hardware_errno) in [
        (None, None),
        (Some(5), None),
        (None, Some(13)),
        (Some(5), Some(13)),
    ] {
        let (owner, native) = fixture();
        {
            let mut native = native.lock().unwrap();
            native.memory_errno = memory_errno;
            native.hardware_errno = hardware_errno;
            // A failed channel cannot suppress an available peer payload.
            if hardware_errno.is_some() && memory_errno.is_none() {
                native.memory = Some([1, 2, 3, 4]);
            }
            if memory_errno.is_some() && hardware_errno.is_none() {
                native.hardware = Some([5, 6, 7, 8]);
            }
        }
        let before = observer_state(&owner);
        let snapshot = owner.passive_snapshot().unwrap();
        for (result, expected, peer_error, words) in [
            (
                snapshot.native_pending_memory_words,
                memory_errno,
                hardware_errno,
                [1, 2, 3, 4],
            ),
            (
                snapshot.native_pending_hardware_words,
                hardware_errno,
                memory_errno,
                [5, 6, 7, 8],
            ),
        ] {
            match (result, expected) {
                (Ok(payload), None) => assert_eq!(payload, peer_error.map(|_| words)),
                (Err(PayloadUnavailable::Read(source)), Some(expected)) => {
                    assert_eq!(source.raw_os_error(), Some(expected));
                }
                other => panic!("unexpected independent channel result: {other:?}"),
            }
        }
        assert_eq!(observer_state(&owner), before);
        assert_eq!(native.lock().unwrap().calls, [(false, 20), (false, 19)]);
    }
}

#[test]
fn unavailable_locks_and_event_ids_never_trigger_native_reads() {
    let (owner, native) = fixture();
    let records = owner.observed.lock().unwrap();
    assert!(matches!(
        owner.passive_snapshot(),
        Err(SnapshotUnavailable::Busy)
    ));
    drop(records);
    owner
        .hardware_destroy_uncertain
        .store(true, Ordering::Release);
    owner.memory_event_id.store(0, Ordering::Release);
    let snapshot = owner.passive_snapshot().unwrap();
    assert!(matches!(
        snapshot.native_pending_hardware_words,
        Err(PayloadUnavailable::DestructionUncertain)
    ));
    assert!(matches!(
        snapshot.native_pending_memory_words,
        Err(PayloadUnavailable::Closed)
    ));
    assert!(native.lock().unwrap().calls.is_empty());
    let _ = std::panic::catch_unwind(|| {
        let _records = owner.observed.lock().unwrap();
        panic!("poison observation lock");
    });
    assert!(matches!(
        owner.passive_snapshot(),
        Err(SnapshotUnavailable::Poisoned)
    ));
    assert!(native.lock().unwrap().calls.is_empty());
}

#[test]
fn process_failure_precedes_even_an_unavailable_observation_lock() {
    let kfd = Shared::new(connection::Kfd::new(
        File::open("/dev/null").unwrap(),
        ProcessIdentity::with_pid_for_test(std::process::id().wrapping_add(1)),
    ))
    .unwrap();
    let owner = owner(kfd);
    let _records = owner.observed.lock().unwrap();
    assert!(matches!(
        owner.passive_snapshot(),
        Err(SnapshotUnavailable::Process(_))
    ));
}

#[test]
fn formatting_runs_after_unlock_and_keeps_attribution_separate() {
    struct Output<'a> {
        /// Owner whose observation lock must be released during formatting.
        owner: &'a LossEvent,
        /// Exact bytes produced by the production formatter.
        bytes: Vec<u8>,
    }
    impl Write for Output<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            assert!(self.owner.observed.try_lock().is_ok());
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let (owner, native) = fixture();
    native.lock().unwrap().hardware = Some([1000, 2000, 3000, 4000]);
    let snapshot = owner.passive_snapshot();
    let mut output = Output {
        owner: &owner,
        bytes: Vec::new(),
    };
    write_queue_diagnostic(
        &mut output,
        17,
        42,
        "UPDATE_QUEUE-error",
        Some(&io::Error::from_raw_os_error(13)),
        &snapshot,
        &Ok(drm::GpuVmFault::default()),
    )
    .unwrap();
    let text = String::from_utf8(output.bytes).unwrap();
    assert!(text.contains(
        "boundary=UPDATE_QUEUE-error queue_id=17 expected_gpu_id=42 update_errno=Some(13)"
    ));
    assert!(text.contains("native_pending_hardware_words="));
    assert!(text.contains("drm_gpuvm_fault_cache=raw(address=0x0, status=0x0, vmhub=0x0)"));
    assert!(text.contains("untimestamped_may_be_stale_or_incomplete"));
    assert!(!text.contains("reset_type"));
    assert!(!text.contains("Deadline"));
}

#[test]
fn drm_fault_query_preserves_raw_cache_fields_and_exact_errno() {
    let file = File::open("/dev/null").unwrap();
    for fault in [
        drm::GpuVmFault::default(),
        drm::GpuVmFault {
            address: 0x74a7_2258_3000,
            status: 0xdead_beef,
            vmhub: 0x102,
        },
    ] {
        drm::with_script(
            [drm::TestCall::QueryGpuVmFault(file.as_raw_fd(), Ok(fault))],
            || assert_eq!(drm::gpuvm_fault(&file).unwrap(), fault),
        );
    }
    for errno in [5, 22, 25, 95] {
        drm::with_script(
            [drm::TestCall::QueryGpuVmFault(file.as_raw_fd(), Err(errno))],
            || {
                assert_eq!(
                    drm::gpuvm_fault(&file).unwrap_err().raw_os_error(),
                    Some(errno)
                );
            },
        );
    }
}

#[test]
fn pre_cleanup_formatter_keeps_independent_unavailability_and_raw_cache() {
    let (owner, native) = fixture();
    native.lock().unwrap().memory_errno = Some(5);
    let snapshot = owner.passive_snapshot();
    for fault_cache in [
        Ok(drm::GpuVmFault {
            address: 0x1234_5000,
            status: 0xabcd,
            vmhub: 0x101,
        }),
        Err(FaultCacheUnavailable::Read(io::Error::from_raw_os_error(
            25,
        ))),
        Err(FaultCacheUnavailable::RenderClosed),
    ] {
        let mut output = Vec::new();
        write_queue_diagnostic(
            &mut output,
            0,
            42,
            "code-cache-wait-incomplete-before-UPDATE_QUEUE",
            None,
            &snapshot,
            &fault_cache,
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(
            "boundary=code-cache-wait-incomplete-before-UPDATE_QUEUE queue_id=0 expected_gpu_id=42"
        ));
        assert!(text.contains("native_pending_memory_words=unavailable(read_failed("));
        assert!(text.contains("native_pending_hardware_words=empty"));
        assert!(!text.contains("update_errno="));
        match fault_cache {
            Ok(_) => assert!(text.contains(
                "drm_gpuvm_fault_cache=raw(address=0x12345000, status=0xabcd, vmhub=0x101)"
            )),
            Err(FaultCacheUnavailable::Read(_)) => assert!(
                text.contains("drm_gpuvm_fault_cache=unavailable(query_failed(errno=Some(25)")
            ),
            Err(FaultCacheUnavailable::RenderClosed) => {
                assert!(text.contains("drm_gpuvm_fault_cache=unavailable(render_closed)"));
            }
        }
    }
}

#[test]
fn foreign_process_rejects_new_render_and_queue_captures_before_native_queries() {
    const CHILD: &str = "ROCDDI_FOREIGN_CAPTURE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "driver::linux_kfd::memory::diagnostics::tests::foreign_process_rejects_new_render_and_queue_captures_before_native_queries",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("ROCDDI_CODE_CACHE_DIAGNOSTICS", "1")
            .output().unwrap();
        assert!(
            output.status.success(),
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        return;
    }
    let kfd = Shared::new(connection::Kfd::new(
        File::open("/dev/null").unwrap(),
        ProcessIdentity::with_pid_for_test(std::process::id().wrapping_add(1)),
    ))
    .unwrap();
    let render = File::open("/dev/null").unwrap();
    drm::with_script([], || assert!(fresh_render_fault(&kfd, &render).is_none()));
    // The same production gate is used by private queue activation capture.
    assert!(!enabled(&kfd));
}
