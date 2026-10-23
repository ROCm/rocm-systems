// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Queue ownership, atomic index, and ABI regression tests.

use super::*;
use std::sync::atomic::AtomicUsize;

/// Counts owner drops during queue scratch rollback tests.
struct DropCount(
    /// Number of observed owner drops.
    Arc<AtomicUsize>,
);

impl Drop for DropCount {
    /// Records destruction of the test owner.
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Release);
    }
}

#[test]
/// Verifies that failed scratch update retains both old and candidate owners.
fn failed_scratch_update_retains_both_old_and_candidate_owners() {
    let old_drops = Arc::new(AtomicUsize::new(0));
    let new_drops = Arc::new(AtomicUsize::new(0));
    let mut current = Some(DropCount(old_drops.clone()));
    let mut uncertain = Vec::new();
    assert!(matches!(
        update_scratch_owner(
            &mut current,
            &mut uncertain,
            DropCount(new_drops.clone()),
            || Err(ERROR),
        ),
        Err(ERROR)
    ));
    assert_eq!(old_drops.load(Ordering::Acquire), 0);
    assert_eq!(new_drops.load(Ordering::Acquire), 0);
    assert_eq!(uncertain.len(), 1);
    drop(current);
    drop(uncertain);
    assert_eq!(old_drops.load(Ordering::Acquire), 1);
    assert_eq!(new_drops.load(Ordering::Acquire), 1);
}

#[test]
#[allow(clippy::panic)]
/// Verifies that scratch update unwind keeps candidate owned.
fn scratch_update_unwind_keeps_candidate_owned() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut current = None;
    let mut uncertain = Vec::new();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = update_scratch_owner(
            &mut current,
            &mut uncertain,
            DropCount(drops.clone()),
            || -> Result<(), Status> { panic!("injected scratch update unwind") },
        );
    }));
    assert!(outcome.is_err());
    assert_eq!(drops.load(Ordering::Acquire), 0);
    assert_eq!(uncertain.len(), 1);
    drop(uncertain);
    assert_eq!(drops.load(Ordering::Acquire), 1);
}

#[test]
/// Verifies that stopping a queue joins its event worker.
fn stopping_a_queue_joins_its_event_worker() {
    let alive = Arc::new(AtomicBool::new(true));
    let entered = Arc::new(AtomicBool::new(false));
    let finished = Arc::new(AtomicBool::new(false));
    let worker_alive = alive.clone();
    let worker_entered = entered.clone();
    let worker_finished = finished.clone();
    let mut worker = Some(thread::spawn(move || {
        worker_entered.store(true, Ordering::Release);
        while worker_alive.load(Ordering::Acquire) {
            thread::yield_now();
        }
        thread::sleep(Duration::from_millis(2));
        worker_finished.store(true, Ordering::Release);
    }));
    let deadline = std::time::Instant::now() + Duration::from_millis(100);
    while !entered.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
        thread::yield_now();
    }

    assert!(stop_queue_event_worker(&alive, &mut worker).is_ok());

    assert!(worker.is_none());
    assert!(finished.load(Ordering::Acquire));
}

#[test]
#[allow(clippy::unwrap_used)]
/// Verifies that queue event worker keeps its join handle for later cleanup.
fn queue_event_worker_keeps_its_join_handle_for_later_cleanup() {
    let alive = Arc::new(AtomicBool::new(true));
    let worker_slot = Arc::new(std::sync::Mutex::new(None));
    let (sender, receiver) = std::sync::mpsc::channel();
    let worker_alive = alive.clone();
    let worker_slot_clone = worker_slot.clone();
    let worker = thread::spawn(move || {
        let result = loop {
            let mut slot = worker_slot_clone.lock().unwrap();
            if slot.is_some() {
                break stop_queue_event_worker(&worker_alive, &mut slot);
            }
            drop(slot);
            thread::yield_now();
        };
        sender.send(result.map_err(|error| error.kind())).unwrap();
    });
    *worker_slot.lock().unwrap() = Some(worker);
    assert_eq!(
        receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
        Err(rocddi::ErrorKind::Busy)
    );
    assert!(!alive.load(Ordering::Acquire));
    let worker = worker_slot.lock().unwrap().take().unwrap();
    worker.join().unwrap();
}

#[test]
/// Verifies that async copy profiling entry point matches the public ABI.
fn async_copy_profiling_entry_point_matches_the_public_abi() {
    let _: unsafe extern "C" fn(HsaSignal, *mut ProfilingTime) -> Status =
        hsa_amd_profiling_get_async_copy_time;
}

#[test]
/// Verifies that external semaphore queue entry points match the public ABI.
fn external_semaphore_queue_entry_points_match_the_public_abi() {
    let _: unsafe extern "C" fn(*mut HsaQueue, HsaAmdExternalSemaphore, u64) -> Status =
        hsa_amd_queue_signal_external_semaphore;
    let _: unsafe extern "C" fn(*mut HsaQueue, HsaAmdExternalSemaphore, u64) -> Status =
        hsa_amd_queue_wait_external_semaphore;
}

#[test]
/// Verifies that counted queue entry points match the public ABI.
fn counted_queue_entry_points_match_the_public_abi() {
    let _: unsafe extern "C" fn(
        HsaAgent,
        u32,
        u32,
        QueueErrorCallback,
        *mut c_void,
        u64,
        *mut *mut HsaQueue,
    ) -> Status = hsa_amd_counted_queue_acquire;
    let _: unsafe extern "C" fn(*mut HsaQueue) -> Status = hsa_amd_counted_queue_release;
}

#[test]
/// Verifies that counted queue pool selects the first least used hardware queue.
fn counted_queue_pool_selects_the_first_least_used_hardware_queue() {
    let pool = [
        CountedHardwareQueue {
            queue: 0x1000,
            use_count: 3,
        },
        CountedHardwareQueue {
            queue: 0x2000,
            use_count: 1,
        },
        CountedHardwareQueue {
            queue: 0x3000,
            use_count: 1,
        },
    ];
    assert_eq!(least_used_counted_queue(&pool), Some(0x2000));
    assert_eq!(least_used_counted_queue(&[]), None);
}

#[test]
/// Verifies that counted queue handles are unique copies of the hardware header.
fn counted_queue_handles_are_unique_copies_of_the_hardware_header() {
    let hardware = Box::new(HsaQueue {
        queue_type: QUEUE_TYPE_MULTI,
        features: QUEUE_FEATURE_KERNEL_DISPATCH,
        base_address: 0x1234usize as *mut c_void,
        doorbell_signal: HsaSignal { handle: 0x5678 },
        size: 16_384,
        reserved: 0,
        id: 42,
    });
    let hardware_key = (&raw const *hardware) as usize;
    // SAFETY: hardware remains live while both public headers are copied.
    let mut first = unsafe { CountedQueue::new(hardware_key, (1, 2)) };
    // SAFETY: hardware remains live while both public headers are copied.
    let mut second = unsafe { CountedQueue::new(hardware_key, (1, 2)) };
    let first = first.public_pointer();
    let second = second.public_pointer();
    assert_ne!(first, second);
    assert_eq!(first as usize % 128, COUNTED_QUEUE_HANDLE_BIT);
    assert_eq!(second as usize % 128, COUNTED_QUEUE_HANDLE_BIT);
    // SAFETY: Both pointers refer to live CountedQueue-owned headers.
    unsafe {
        assert_eq!((*first).base_address, hardware.base_address);
        assert_eq!(
            (*first).doorbell_signal.handle,
            hardware.doorbell_signal.handle
        );
        assert_eq!((*first).size, hardware.size);
        assert_eq!((*first).id, hardware.id);
        assert_eq!((*second).base_address, hardware.base_address);
        assert_eq!(
            (*second).doorbell_signal.handle,
            hardware.doorbell_signal.handle
        );
        assert_eq!((*second).size, hardware.size);
        assert_eq!((*second).id, hardware.id);
    }
}

#[repr(C, align(256))]
/// Provides aligned bytes for a public queue fixture.
struct PublicQueueStorage(
    /// Storage large enough for the public queue header.
    [u8; 256],
);

#[test]
/// Verifies that soft queue control uses the direct index path.
fn soft_queue_control_uses_the_direct_index_path() {
    let mut control = Box::new(SoftQueueControl([0; 32]));
    let public = control.0.as_mut_ptr().cast::<HsaQueue>();
    assert_eq!(public as usize % 128, 0);
    // SAFETY: The aligned control allocation retains the read index word.
    unsafe {
        public
            .cast::<u8>()
            .add(READ_INDEX_OFFSET)
            .cast::<AtomicU64>()
            .write(AtomicU64::new(31));
        assert_eq!(hsa_queue_load_read_index_relaxed(public), 31);
    }
}

#[test]
/// Verifies that counted queue index operations reach the hardware control.
fn counted_queue_index_operations_reach_the_hardware_control() {
    let mut hardware = PublicQueueStorage([0; 256]);
    let hardware_queue = hardware.0.as_mut_ptr().cast::<HsaQueue>();
    // SAFETY: The storage is aligned and retains the public header and
    // both atomic index words throughout this test.
    unsafe {
        hardware_queue.write(HsaQueue {
            queue_type: QUEUE_TYPE_MULTI,
            features: QUEUE_FEATURE_KERNEL_DISPATCH,
            base_address: std::ptr::null_mut(),
            doorbell_signal: HsaSignal { handle: 0 },
            size: 64,
            reserved: 0,
            id: 7,
        });
        let read = hardware
            .0
            .as_mut_ptr()
            .add(READ_INDEX_OFFSET)
            .cast::<AtomicU64>();
        let write = hardware
            .0
            .as_mut_ptr()
            .add(WRITE_INDEX_OFFSET)
            .cast::<AtomicU64>();
        read.write(AtomicU64::new(19));
        write.write(AtomicU64::new(23));

        let mut counted = CountedQueue::new(hardware_queue as usize, (1, 2));
        let public = counted.public_pointer();
        assert_eq!(hardware_queue as usize % 128, 0);
        assert_eq!(public as usize % 128, COUNTED_QUEUE_HANDLE_BIT);
        assert_eq!(public as usize % 64, 0);
        assert_eq!(hsa_queue_load_read_index_relaxed(public), 19);
        assert_eq!(hsa_queue_load_read_index_scacquire(public), 19);
        assert_eq!(hsa_queue_add_write_index_relaxed(public, 2), 23);
        assert_eq!((*write).load(Ordering::Relaxed), 25);
        assert_eq!(hsa_queue_load_write_index_relaxed(public), 25);
    }
}

#[test]
/// Verifies that AMD queue create descriptor matches the public x86 64 ABI.
fn amd_queue_create_descriptor_matches_the_public_x86_64_abi() {
    assert_eq!(std::mem::size_of::<HsaAmdComputeQueueParams>(), 32);
    assert_eq!(std::mem::size_of::<HsaAmdSdmaQueueParams>(), 32);
    assert_eq!(std::mem::size_of::<HsaAmdQueueEngineParams>(), 32);
    assert_eq!(std::mem::size_of::<HsaAmdQueueCreateDesc>(), 96);
    assert_eq!(std::mem::align_of::<HsaAmdQueueCreateDesc>(), 8);
    assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, version), 0);
    assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, flags), 2);
    assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, engine_type), 4);
    assert_eq!(
        std::mem::offset_of!(HsaAmdQueueCreateDesc, queue_size_bytes),
        8
    );
    assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, priority), 12);
    assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, callback), 16);
    assert_eq!(
        std::mem::offset_of!(HsaAmdQueueCreateDesc, callback_data),
        24
    );
    assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, queue), 32);
    assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, engine), 40);
    assert_eq!(
        std::mem::offset_of!(HsaAmdQueueCreateDesc, traffic_class),
        72
    );
    assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, reserved), 76);
}

#[test]
/// Verifies that queue create input does not read the uninitialized output.
fn queue_create_input_does_not_read_the_uninitialized_output() {
    let mut descriptor = std::mem::MaybeUninit::<HsaAmdQueueCreateDesc>::uninit();
    let pointer = descriptor.as_mut_ptr();
    // SAFETY: Every input field is initialized below. The reader only
    // copies those fields, leaving the queue output untouched.
    unsafe {
        (&raw mut (*pointer).version).write(AMD_QUEUE_CREATE_DESC_VERSION);
        (&raw mut (*pointer).flags).write(0);
        (&raw mut (*pointer).engine_type).write(AMD_QUEUE_ENGINE_COMPUTE);
        (&raw mut (*pointer).reserved_header).write([0; 3]);
        (&raw mut (*pointer).queue_size_bytes).write(4096);
        (&raw mut (*pointer).priority).write(AMD_QUEUE_PRIORITY_NORMAL);
        (&raw mut (*pointer).callback).write(None);
        (&raw mut (*pointer).callback_data).write(std::ptr::null_mut());
        (&raw mut (*pointer).engine).write(HsaAmdQueueEngineParams {
            compute: HsaAmdComputeQueueParams {
                cu_mask: std::ptr::null(),
                queue_type: QUEUE_TYPE_MULTI,
                private_segment_size: 0,
                cu_mask_count: 0,
                reserved: [0; 3],
            },
        });
        (&raw mut (*pointer).traffic_class).write(0);
        (&raw mut (*pointer).reserved).write([0; 20]);
        let input = read_queue_create_input(pointer);
        assert_eq!(input.version, AMD_QUEUE_CREATE_DESC_VERSION);
        assert_eq!(input.queue_size_bytes, 4096);
    }
}

#[test]
/// Verifies that queue create copies a mask inside the descriptor.
fn queue_create_copies_a_mask_inside_the_descriptor() -> Result<(), Status> {
    let mut descriptor = HsaAmdQueueCreateDesc {
        version: AMD_QUEUE_CREATE_DESC_VERSION,
        flags: 0,
        engine_type: AMD_QUEUE_ENGINE_COMPUTE,
        reserved_header: [0; 3],
        queue_size_bytes: 4096,
        priority: AMD_QUEUE_PRIORITY_NORMAL,
        callback: None,
        callback_data: std::ptr::null_mut(),
        queue: std::ptr::null_mut(),
        engine: HsaAmdQueueEngineParams { reserved: [0; 32] },
        traffic_class: 0,
        reserved: [0; 20],
    };
    descriptor.engine = HsaAmdQueueEngineParams {
        compute: HsaAmdComputeQueueParams {
            cu_mask: &raw const descriptor.queue_size_bytes,
            queue_type: QUEUE_TYPE_MULTI,
            private_segment_size: 0,
            cu_mask_count: 32,
            reserved: [0; 3],
        },
    };
    // SAFETY: The mask points to the initialized u32 field in this live
    // descriptor. No mutable reference to the descriptor is active.
    let copied = unsafe { copy_queue_create_mask(descriptor.engine.compute) }?;
    // SAFETY: The output field is writable and separate from the owned mask.
    unsafe { (&raw mut descriptor.queue).write(std::ptr::null_mut()) };
    assert_eq!(copied, Some(vec![4096]));
    Ok(())
}

#[test]
/// Verifies that firmware queue errors map to public status codes.
fn firmware_queue_errors_map_to_public_status_codes() {
    assert_eq!(queue_error_status(2), INCOMPATIBLE_ARGUMENTS);
    assert_eq!(queue_error_status(4), INVALID_ALLOCATION);
    assert_eq!(queue_error_status(8), INVALID_CODE_OBJECT);
    assert_eq!(queue_error_status(16), MEMORY_FAULT);
    assert_eq!(queue_error_status(32), INVALID_PACKET_FORMAT);
    assert_eq!(queue_error_status(64), INVALID_ARGUMENT);
    assert_eq!(queue_error_status(128), OUT_OF_REGISTERS);
    assert_eq!(queue_error_status(0x2000_0000), MEMORY_APERTURE_VIOLATION);
    assert_eq!(queue_error_status(0x4000_0000), ILLEGAL_INSTRUCTION);
    assert_eq!(queue_error_status(0x8000_0000), EXCEPTION);
}

#[test]
/// Verifies that base atomic entry points preserve queue indices.
fn base_atomic_entry_points_preserve_queue_indices() {
    let mut storage = PublicQueueStorage([0; 256]);
    let queue = storage.0.as_mut_ptr().cast::<HsaQueue>();

    // SAFETY: The aligned backing storage contains initialized atomic
    // index fields at the public amd_queue_v2_t offsets used below.
    unsafe {
        storage
            .0
            .as_mut_ptr()
            .add(WRITE_INDEX_OFFSET)
            .cast::<AtomicU64>()
            .write(AtomicU64::new(0));
        storage
            .0
            .as_mut_ptr()
            .add(READ_INDEX_OFFSET)
            .cast::<AtomicU64>()
            .write(AtomicU64::new(0));

        hsa_queue_store_write_index_relaxed(queue, 4);
        assert_eq!(hsa_queue_load_write_index_acquire(queue), 4);
        assert_eq!(hsa_queue_add_write_index_relaxed(queue, 3), 4);
        assert_eq!(hsa_queue_cas_write_index_scacquire(queue, 6, 9), 7);
        assert_eq!(hsa_queue_load_write_index_relaxed(queue), 7);
        assert_eq!(hsa_queue_cas_write_index_scacq_screl(queue, 7, 9), 7);
        assert_eq!(hsa_queue_load_write_index_scacquire(queue), 9);

        hsa_queue_store_write_index_release(queue, 11);
        assert_eq!(hsa_queue_load_write_index_relaxed(queue), 11);
        hsa_queue_store_read_index_screlease(queue, 5);
        assert_eq!(hsa_queue_load_read_index_acquire(queue), 5);
        hsa_queue_store_read_index_release(queue, 8);
        assert_eq!(hsa_queue_load_read_index_relaxed(queue), 8);
    }
}

#[test]
#[allow(clippy::unwrap_used)]
/// Verifies that repeated queue index access does not wait for the registry
/// mutex.
fn repeated_queue_index_access_does_not_wait_for_the_registry_mutex() {
    use std::sync::mpsc;

    let mut storage = PublicQueueStorage([0; 256]);
    // SAFETY: This aligned storage contains a live atomic at the public
    // write-index offset until the scoped worker exits.
    unsafe {
        storage
            .0
            .as_mut_ptr()
            .add(WRITE_INDEX_OFFSET)
            .cast::<AtomicU64>()
            .write(AtomicU64::new(42));
    }
    let pointer = storage.0.as_ptr() as usize;
    let (ready_tx, ready_rx) = mpsc::channel();
    let (go_tx, go_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    thread::scope(|scope| {
        scope.spawn(move || {
            // SAFETY: The scoped owner keeps the synthetic queue storage
            // live for the read.
            let queue = pointer as *const HsaQueue;
            ready_tx.send(()).unwrap();
            go_rx.recv().unwrap();
            done_tx
                .send(unsafe { hsa_queue_load_write_index_relaxed(queue) })
                .unwrap();
        });
        ready_rx.recv().unwrap();
        let guard = crate::runtime::LIFECYCLE.lock().unwrap();
        go_tx.send(()).unwrap();
        let value = done_rx.recv_timeout(Duration::from_millis(200));
        drop(guard);
        assert_eq!(value.unwrap(), 42);
    });
}
