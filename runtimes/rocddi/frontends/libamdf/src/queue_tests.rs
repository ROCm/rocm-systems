//! Queue lifecycle and publication contracts exercised without a native GPU.

#![allow(clippy::unwrap_used)]

use super::*;
use rocddi::{Error, ErrorKind};
use std::ptr;
use std::sync::Arc;

pub(super) struct Fixture {
    consumed: AtomicU64,
    producer: AtomicU64,
    error: AtomicU64,
    progress_calls: AtomicU64,
    destroy_calls: AtomicU64,
    fail_destroy_once: AtomicBool,
    destroy_errno: AtomicU64,
    backing_live: AtomicBool,
    retire_on_call: AtomicU64,
}

impl Fixture {
    pub(super) fn progress(&self) -> Result<(u64, u64), Error> {
        assert!(
            self.backing_live.load(Ordering::Acquire),
            "native progress touched released backing"
        );
        let call = self.progress_calls.fetch_add(1, Ordering::Relaxed) + 1;
        let kind = match self.error.load(Ordering::Relaxed) {
            LOST => Some(ErrorKind::DeviceLost),
            INTERNAL => Some(ErrorKind::DriverContract),
            BUSY => Some(ErrorKind::Busy),
            _ => None,
        };
        if let Some(kind) = kind {
            return Err(Error::Operation {
                kind,
                detail: "injected queue observation",
            });
        }
        let producer = self.producer.load(Ordering::Acquire);
        if call >= self.retire_on_call.load(Ordering::Relaxed) {
            self.consumed.store(producer, Ordering::Release);
        }
        Ok((self.consumed.load(Ordering::Acquire), producer))
    }

    pub(super) fn destroy(&self) -> Result<(), Error> {
        self.destroy_calls.fetch_add(1, Ordering::Relaxed);
        if self.consumed.load(Ordering::Acquire) < self.producer.load(Ordering::Acquire) {
            return Err(Error::Operation {
                kind: ErrorKind::Busy,
                detail: "unconsumed queue publication",
            });
        }
        self.backing_live.store(false, Ordering::Release);
        if self.fail_destroy_once.swap(false, Ordering::AcqRel) {
            let errno = self.destroy_errno.load(Ordering::Relaxed);
            if errno != 0 {
                return Err(Error::NativeOperation {
                    kind: ErrorKind::Busy,
                    operation: "injected queue teardown",
                    source: std::io::Error::from_raw_os_error(i32::try_from(errno).unwrap()),
                });
            }
            return Err(Error::Operation {
                kind: ErrorKind::Driver,
                detail: "injected failure after backing release",
            });
        }
        Ok(())
    }
}

struct Environment {
    queues: AtomicU64,
    epoch: AtomicU64,
}

impl Environment {
    fn new() -> Self {
        Self {
            queues: AtomicU64::new(1),
            epoch: AtomicU64::new(7),
        }
    }
}

fn fixture_queue(environment: &Environment, allocator: Allocator) -> (Owned<Queue>, Arc<Fixture>) {
    let fixture = Arc::new(Fixture {
        consumed: AtomicU64::new(0),
        producer: AtomicU64::new(0),
        error: AtomicU64::new(0),
        progress_calls: AtomicU64::new(0),
        destroy_calls: AtomicU64::new(0),
        fail_destroy_once: AtomicBool::new(false),
        destroy_errno: AtomicU64::new(0),
        backing_live: AtomicBool::new(true),
        retire_on_call: AtomicU64::new(u64::MAX),
    });
    let id = amdf_queue_id_t { words: [11, 13] };
    let owner = Owned::new(
        Queue {
            native: NativeQueue::Fixture(fixture.clone()),
            allocator,
            device: ptr::NonNull::<Device>::dangling().as_ptr(),
            device_queues: &raw const environment.queues,
            device_reset_epoch: &raw const environment.epoch,
            _scratch_borrow: None,
            info: amdf_user_queue_info_t {
                queue_id: id,
                reset_epoch: 7,
                command_type: AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
                format_version: 1,
                capabilities: AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER
                    | AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
                ring_byte_length: 4096,
                ..Default::default()
            },
            host_mapping: amdf_user_queue_mapping_info_t {
                queue_id: id,
                queue_reset_epoch: 7,
                ring_address: 0x1000,
                ring_byte_length: 4096,
                read_index_address: 0x2000,
                write_index_address: 0x2008,
                doorbell_address: 0x3000,
                index_bits: 64,
                doorbell_bits: 64,
                ..Default::default()
            },
            device_mapping: Some(amdf_user_queue_mapping_info_t {
                producer_device_id: amdf_device_id_t { words: [7, 9] },
                queue_id: id,
                queue_reset_epoch: 7,
                producer_reset_epoch: 7,
                ring_address: 0x4000,
                ring_byte_length: 4096,
                read_index_address: 0x5000,
                write_index_address: 0x5008,
                doorbell_address: 0x6000,
                index_bits: 64,
                doorbell_bits: 64,
                ..Default::default()
            }),
            mappings: AtomicU64::new(0),
            destroying: AtomicBool::new(false),
            terminal: AtomicU64::new(0),
            consumed: AtomicU64::new(0),
            producer: AtomicU64::new(0),
        },
        Allocator::system(),
    )
    .unwrap();
    (owner, fixture)
}

fn structure_size<T>() -> u32 {
    u32::try_from(size_of::<T>()).unwrap()
}

fn status_output() -> amdf_user_queue_status_t {
    amdf_user_queue_status_t {
        r#type: AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS,
        structure_size: structure_size::<amdf_user_queue_status_t>(),
        state: 99,
        consumed_index: 0xface,
        producer_index: 0xbeef,
        ..Default::default()
    }
}

fn family(command: u32) -> amdf_queue_family_info_t {
    let pm4 = command == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
    amdf_queue_family_info_t {
        command_type: command,
        publication_modes: AMDF_QUEUE_PUBLICATION_MODE_USER,
        format_version: 1,
        format_features: if pm4 {
            AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR
        } else if command == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA {
            AMDF_GPU_SDMA_FORMAT_FEATURE_GCR | AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM
        } else {
            0
        },
        user_queue_capabilities: AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER
            | if pm4 {
                0
            } else {
                AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER
            },
        priority_capabilities: AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL,
        producer_modes: AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE,
        minimum_ring_byte_length: if pm4 { 4096 } else { 1024 },
        maximum_ring_byte_length: if pm4 { 4096 } else { 1 << 31 },
        ring_byte_length_alignment: if pm4 { 4096 } else { 1024 },
        ..Default::default()
    }
}

#[test]
fn descriptors_translate_priorities_and_reject_unsupported_options() {
    let sdma_family = family(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
    let mut input = amdf_gpu_user_queue_create_info_t {
        priority: AMDF_QUEUE_PRIORITY_NORMAL,
        producer_mode: AMDF_QUEUE_PRODUCER_MODE_SINGLE,
        ..Default::default()
    };
    let desc = descriptor(&input, &sdma_family, None).unwrap();
    assert_eq!(desc.ring_size_bytes, DEFAULT_RING_BYTES);
    assert_eq!(desc.parameters, QueueParameters::Sdma);
    let pm4 = family(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
    let pm4_desc = descriptor(&input, &pm4, None).unwrap();
    assert_eq!(pm4_desc.parameters, QueueParameters::Pm4);
    assert_eq!(pm4_desc.ring_size_bytes, 4096);
    input.required_capabilities = AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER;
    assert_eq!(descriptor(&input, &pm4, None), Err(UNSUPPORTED));
    assert!(
        descriptor(&input, &sdma_family, None)
            .unwrap()
            .device_producer
    );
    input.required_capabilities = 1 << 63;
    assert_eq!(descriptor(&input, &sdma_family, None), Err(UNSUPPORTED));
    input.required_capabilities = 0;
    input.producer_mode = AMDF_QUEUE_PRODUCER_MODE_MULTI;
    assert_eq!(descriptor(&input, &sdma_family, None), Err(UNSUPPORTED));
    input.producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE;
    input.priority = AMDF_QUEUE_PRIORITY_LOW;
    assert_eq!(descriptor(&input, &sdma_family, None), Err(UNSUPPORTED));
    input.priority = AMDF_QUEUE_PRIORITY_NORMAL;
    input.ring_byte_length = 1536;
    assert_eq!(descriptor(&input, &sdma_family, None), Err(INVALID));
    input.ring_byte_length = 512;
    assert_eq!(descriptor(&input, &sdma_family, None), Err(RANGE));

    let aql = amdf_queue_family_info_t {
        command_type: AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
        publication_modes: AMDF_QUEUE_PUBLICATION_MODE_USER,
        priority_capabilities: AMDF_QUEUE_PRIORITY_CAPABILITY_LOW
            | AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL
            | AMDF_QUEUE_PRIORITY_CAPABILITY_HIGH,
        producer_modes: AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE,
        minimum_ring_byte_length: 1024,
        maximum_ring_byte_length: 1 << 31,
        ring_byte_length_alignment: 1024,
        ..Default::default()
    };
    input.ring_byte_length = 0;
    input.priority = AMDF_QUEUE_PRIORITY_LOW;
    assert_eq!(
        descriptor(&input, &aql, None).unwrap().priority,
        QueuePriority::Low
    );
    input.priority = AMDF_QUEUE_PRIORITY_HIGH;
    assert_eq!(
        descriptor(&input, &aql, None).unwrap().priority,
        QueuePriority::High
    );
}

#[test]
fn scratch_descriptor_requires_an_all_zero_disable_or_complete_aql_request() {
    let device = ptr::NonNull::<Device>::dangling().as_ptr();
    let mut input = amdf_gpu_user_queue_create_info_t::default();
    let aql = amdf_queue_family_info_t {
        command_type: AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
        roles: AMDF_QUEUE_ROLE_COMPUTE,
        ..Default::default()
    };
    // SAFETY: Disabled and structurally incomplete descriptors return before
    // dereferencing the synthetic device pointer.
    unsafe {
        assert_eq!(
            scratch_descriptor(&input, &aql, device),
            Ok((None, ptr::null_mut()))
        );
        input.scratch.byte_length = 4096;
        assert!(matches!(
            scratch_descriptor(&input, &aql, device),
            Err(INVALID)
        ));
        input.scratch.memory = ptr::NonNull::<amdf_memory_t>::dangling().as_ptr();
        input.scratch.maximum_private_segment_byte_length = 16;
        input.scratch.maximum_wave_count = 4;
        let sdma = amdf_queue_family_info_t {
            command_type: AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            roles: AMDF_QUEUE_ROLE_TRANSFER,
            ..Default::default()
        };
        assert!(matches!(
            scratch_descriptor(&input, &sdma, device),
            Err(UNSUPPORTED)
        ));
    }
}

fn native_transport() -> queue::QueueTransport {
    queue::QueueTransport {
        ring_host_address: 0x1000,
        ring_device_address: 0x1000,
        ring_size_bytes: 4096,
        read_index_host_address: 0x2000,
        read_index_device_address: 0x4000,
        write_index_host_address: 0x2008,
        write_index_device_address: 0x4008,
        read_index_width: QueueAccessWidth::Bits64,
        write_index_width: QueueAccessWidth::Bits64,
        index_unit_bytes: 64,
        read_index_wraps: false,
        doorbell_host_address: 0x3000,
        doorbell_device_address: Some(0x5000),
        doorbell_width: QueueAccessWidth::Bits64,
    }
}

#[test]
fn transport_requires_the_exact_advertised_atomic_widths_and_units() {
    let mut native = native_transport();
    let aql = family(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
    let host = transport(
        native,
        aql.command_type,
        aql.format_version,
        aql.format_features,
        false,
    )
    .unwrap();
    assert_eq!(host.format_version, aql.format_version);
    assert_eq!(host.format_features, aql.format_features);
    assert!(
        transport(
            native,
            aql.command_type,
            aql.format_version,
            aql.format_features,
            true
        )
        .is_ok()
    );
    native.read_index_width = QueueAccessWidth::Bits32;
    assert!(matches!(
        transport(
            native,
            aql.command_type,
            aql.format_version,
            aql.format_features,
            false
        ),
        Err(INTERNAL)
    ));
    native.read_index_width = QueueAccessWidth::Bits64;
    let sdma = family(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
    assert!(matches!(
        transport(
            native,
            sdma.command_type,
            sdma.format_version,
            sdma.format_features,
            false
        ),
        Err(INTERNAL)
    ));
    native.index_unit_bytes = 1;
    let host = transport(
        native,
        sdma.command_type,
        sdma.format_version,
        sdma.format_features,
        false,
    )
    .unwrap();
    assert_eq!(host.format_version, sdma.format_version);
    assert_eq!(host.format_features, sdma.format_features);
    native.read_index_wraps = true;
    assert!(matches!(
        transport(
            native,
            sdma.command_type,
            sdma.format_version,
            sdma.format_features,
            false
        ),
        Err(INTERNAL)
    ));
}

#[test]
fn pm4_transport_requires_dword_units_and_a_wrapping_read_index() {
    let pm4 = family(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
    let mut native = native_transport();
    native.index_unit_bytes = 4;
    native.read_index_wraps = true;
    assert!(
        transport(
            native,
            pm4.command_type,
            pm4.format_version,
            pm4.format_features,
            false
        )
        .is_ok()
    );
    native.read_index_wraps = false;
    assert!(matches!(
        transport(
            native,
            pm4.command_type,
            pm4.format_version,
            pm4.format_features,
            false
        ),
        Err(INTERNAL)
    ));
}

#[test]
fn cached_queries_do_not_observe_native_loss_and_mapping_blocks_teardown() {
    #[repr(C)]
    struct Extended {
        info: amdf_user_queue_info_t,
        tail: [u8; 16],
    }
    let environment = Environment::new();
    let (queue, fixture) = fixture_queue(&environment, Allocator::system());
    fixture.error.store(LOST, Ordering::Relaxed);
    let pointer = queue.into_raw().cast();
    let mut extended = Extended {
        info: amdf_user_queue_info_t {
            r#type: AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO,
            structure_size: structure_size::<Extended>(),
            ..Default::default()
        },
        tail: [0xa5; 16],
    };
    let output = &mut extended.info;
    let mut mapping = ptr::null_mut();
    // SAFETY: These live handles and initialized output records have valid extents.
    unsafe {
        assert_eq!(info(pointer, ptr::from_mut(output)), 0);
        assert_eq!(output.structure_size, structure_size::<Extended>());
        assert_eq!(output.queue_id.words, [11, 13]);
        assert_eq!(output.format_features, 0);
        assert_eq!(
            output.capabilities,
            AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER | AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER
        );
        assert_eq!(extended.tail, [0xa5; 16]);
        assert_eq!(map(pointer, ptr::null_mut(), &raw mut mapping), 0);
        let mut mapped = amdf_user_queue_mapping_info_t {
            r#type: AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO,
            structure_size: structure_size::<amdf_user_queue_mapping_info_t>(),
            ..Default::default()
        };
        assert_eq!(mapping_info(mapping, &raw mut mapped), 0);
        assert_eq!(mapped.queue_id.words, [11, 13]);
        assert_eq!(mapped.format_features, 0);
        assert_eq!(mapped.index_bits, 64);
        assert_eq!(mapped.doorbell_bits, 64);
        let mut device_mapping = ptr::null_mut();
        let producer = ptr::NonNull::<Device>::dangling().as_ptr().cast();
        assert_eq!(map(pointer, producer, &raw mut device_mapping), 0);
        assert_eq!(mapping_info(device_mapping, &raw mut mapped), 0);
        assert_eq!(mapped.producer_device_id.words, [7, 9]);
        assert_eq!(mapped.producer_reset_epoch, 7);
        assert_eq!(mapped.ring_address, 0x4000);
        assert_eq!(mapped.read_index_address, 0x5000);
        assert_eq!(mapped.write_index_address, 0x5008);
        assert_eq!(mapped.doorbell_address, 0x6000);
        assert_eq!(unmap(device_mapping), 0);
        assert_eq!(destroy(pointer), BUSY);
        assert_eq!(fixture.destroy_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fixture.progress_calls.load(Ordering::Relaxed), 0);
        // A mapping-only BUSY response must not begin teardown or revoke the
        // existing mapping. New mapping acquisition is still permitted.
        assert_eq!(mapping_info(mapping, &raw mut mapped), 0);
        let mut another_mapping = ptr::null_mut();
        assert_eq!(map(pointer, ptr::null_mut(), &raw mut another_mapping), 0);
        assert_eq!(unmap(another_mapping), 0);
        assert_eq!(unmap(mapping), 0);
        assert_eq!(destroy(pointer), 0);
    }
    assert_eq!(environment.queues.load(Ordering::Relaxed), 0);
}

unsafe extern "C" fn reject_allocation(
    _: *mut std::ffi::c_void,
    _: u64,
    _: u64,
) -> *mut std::ffi::c_void {
    ptr::null_mut()
}
unsafe extern "C" fn empty_free(_: *mut std::ffi::c_void, _: *mut std::ffi::c_void) {}

#[test]
fn failed_mapping_preserves_output_and_does_not_add_a_borrow() {
    let environment = Environment::new();
    // SAFETY: These stateless callbacks remain live and allocation always fails.
    let allocator = unsafe {
        Allocator::from_callbacks(
            ptr::null_mut(),
            Some(reject_allocation),
            None,
            Some(empty_free),
        )
    }
    .unwrap();
    let (queue, _) = fixture_queue(&environment, allocator);
    let pointer = queue.into_raw().cast();
    let sentinel = ptr::NonNull::<amdf_user_queue_mapping_t>::dangling().as_ptr();
    let mut mapping = sentinel;
    // SAFETY: The queue is live and mapping is valid writable pointer storage.
    unsafe {
        let producer = ptr::NonNull::<amdf_device_t>::dangling()
            .as_ptr()
            .cast::<u8>()
            .wrapping_add(1)
            .cast();
        assert_eq!(map(pointer, producer, &raw mut mapping), INVALID);
        assert_eq!(mapping, sentinel);
        assert_eq!(map(pointer, ptr::null_mut(), &raw mut mapping), EXHAUSTED);
        assert_eq!(mapping, sentinel);
        assert_eq!(destroy(pointer), 0);
    }
    assert_eq!(environment.queues.load(Ordering::Relaxed), 0);
}

#[test]
fn short_status_headers_and_native_errors_leave_outputs_unchanged() {
    let environment = Environment::new();
    let (queue, fixture) = fixture_queue(&environment, Allocator::system());
    let pointer = queue.into_raw().cast();
    let mut output = status_output();
    output.structure_size = 8;
    // SAFETY: The declared short record is backed by a complete object; the
    // operation must reject its header before observing native state.
    unsafe { assert_eq!(status(pointer, &raw mut output), INVALID) };
    assert_eq!(output.consumed_index, 0xface);
    assert_eq!(fixture.progress_calls.load(Ordering::Relaxed), 0);
    output.structure_size = structure_size::<amdf_user_queue_status_t>();
    fixture.error.store(LOST, Ordering::Relaxed);
    // SAFETY: The live queue and fully admitted record remain valid.
    unsafe { assert_eq!(status(pointer, &raw mut output), LOST) };
    assert_eq!(output.state, 99);
    assert_eq!(output.consumed_index, 0xface);
    assert_eq!(output.producer_index, 0xbeef);
    assert_eq!(environment.epoch.load(Ordering::Relaxed), 8);
    fixture.error.store(INTERNAL, Ordering::Relaxed);
    // SAFETY: Terminal status is cached and must not re-observe the backend.
    unsafe { assert_eq!(status(pointer, &raw mut output), 0) };
    assert_eq!(output.state, AMDF_QUEUE_STATE_DEVICE_LOST);
    assert_eq!(output.terminal_status, LOST);
    assert_eq!(fixture.progress_calls.load(Ordering::Relaxed), 1);
    fixture.error.store(LOST, Ordering::Relaxed);
    // SAFETY: This fixture has no outstanding publication or mappings.
    unsafe { assert_eq!(destroy(pointer), 0) };
}

#[test]
fn first_terminal_failure_is_sticky_and_transient_errors_do_not_poison_the_queue() {
    let environment = Environment::new();
    let (queue, fixture) = fixture_queue(&environment, Allocator::system());
    fixture.error.store(BUSY, Ordering::Relaxed);
    assert!(matches!(queue.sample(), Err(BUSY)));
    fixture.error.store(0, Ordering::Relaxed);
    assert_eq!(queue.sample().unwrap().state, AMDF_QUEUE_STATE_ACTIVE);
    fixture.error.store(INTERNAL, Ordering::Relaxed);
    assert!(matches!(queue.sample(), Err(INTERNAL)));
    fixture.error.store(LOST, Ordering::Relaxed);
    let status = queue.sample().unwrap();
    assert_eq!(status.state, AMDF_QUEUE_STATE_FAILED);
    assert_eq!(status.terminal_status, INTERNAL);
    assert_eq!(environment.epoch.load(Ordering::Relaxed), 7);
}

#[test]
fn zero_timeout_observes_once_and_wait_uses_a_single_deadline() {
    let environment = Environment::new();
    let (queue, fixture) = fixture_queue(&environment, Allocator::system());
    fixture.producer.store(4, Ordering::Release);
    assert_eq!(
        wait_consumed(&queue, 4, 0, u64::MAX, Instant::now()),
        Err(DEADLINE)
    );
    assert_eq!(fixture.progress_calls.load(Ordering::Relaxed), 1);
    assert_eq!(wait_consumed(&queue, 5, 0, 0, Instant::now()), Err(INVALID));
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert_eq!(
        wait_consumed(&queue, 4, 10, u64::MAX, expired),
        Err(DEADLINE)
    );
    fixture.retire_on_call.store(5, Ordering::Relaxed);
    assert_eq!(
        wait_consumed(&queue, 4, AMDF_TIMEOUT_INFINITE, 0, Instant::now()),
        Ok(())
    );
    assert_eq!(fixture.progress_calls.load(Ordering::Relaxed), 5);
}

#[test]
fn unconsumed_queue_teardown_is_retryable_and_keeps_the_parent_borrow() {
    let environment = Environment::new();
    let (queue, fixture) = fixture_queue(&environment, Allocator::system());
    fixture.producer.store(1, Ordering::Release);
    let pointer = queue.into_raw().cast();
    // SAFETY: Destruction failure leaves this handle live for the retry.
    unsafe { assert_eq!(destroy(pointer), BUSY) };
    assert_eq!(environment.queues.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.destroy_calls.load(Ordering::Relaxed), 0);
    let mut mapping = ptr::null_mut();
    // SAFETY: Work preflight rejected teardown without revoking the transport.
    unsafe {
        assert_eq!(map(pointer, ptr::null_mut(), &raw mut mapping), 0);
        assert_eq!(wait(pointer, 1, 0, 0), DEADLINE);
        assert_eq!(unmap(mapping), 0);
    }
    fixture.consumed.store(1, Ordering::Release);
    // SAFETY: All publication is now consumed and no mappings remain.
    unsafe { assert_eq!(destroy(pointer), 0) };
    assert_eq!(environment.queues.load(Ordering::Relaxed), 0);
}

#[test]
fn partial_native_teardown_revokes_cached_addresses_and_preserves_cleanup_retry() {
    let environment = Environment::new();
    // SAFETY: These static callbacks reject all allocation and own no state.
    let allocator = unsafe {
        Allocator::from_callbacks(
            ptr::null_mut(),
            Some(reject_allocation),
            None,
            Some(empty_free),
        )
    }
    .unwrap();
    let (queue, fixture) = fixture_queue(&environment, allocator);
    fixture.fail_destroy_once.store(true, Ordering::Release);
    let pointer = queue.into_raw().cast();
    // SAFETY: No mappings or publications remain. The injected error occurs
    // after native backing was released but before cleanup completed.
    unsafe { assert_eq!(destroy(pointer), INTERNAL) };
    assert!(!fixture.backing_live.load(Ordering::Acquire));
    assert_eq!(fixture.destroy_calls.load(Ordering::Relaxed), 1);
    assert_eq!(environment.queues.load(Ordering::Relaxed), 1);

    let sentinel = ptr::NonNull::<amdf_user_queue_mapping_t>::dangling().as_ptr();
    let mut mapping = sentinel;
    let mut queue_info = amdf_user_queue_info_t {
        r#type: AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO,
        structure_size: structure_size::<amdf_user_queue_info_t>(),
        ring_byte_length: 0xcafe,
        ..Default::default()
    };
    let mut queue_status = status_output();
    // SAFETY: Failed destruction retains this live cleanup owner. Public use
    // must fail before metadata allocation, cached publication, or native reads.
    unsafe {
        assert_eq!(
            map(pointer, ptr::null_mut(), &raw mut mapping),
            PRECONDITION
        );
        assert_eq!(mapping, sentinel);
        assert_eq!(info(pointer, &raw mut queue_info), PRECONDITION);
        assert_eq!(queue_info.ring_byte_length, 0xcafe);
        assert_eq!(status(pointer, &raw mut queue_status), PRECONDITION);
        assert_eq!(queue_status.consumed_index, 0xface);
        assert_eq!(queue_status.producer_index, 0xbeef);
        assert_eq!(wait(pointer, 0, 0, 0), PRECONDITION);
        assert_eq!(fixture.progress_calls.load(Ordering::Relaxed), 1);
        assert_eq!(destroy(pointer), 0);
    }
    assert_eq!(fixture.destroy_calls.load(Ordering::Relaxed), 2);
    assert_eq!(environment.queues.load(Ordering::Relaxed), 0);
}

#[test]
fn native_busy_after_teardown_begins_does_not_restore_mapping_access() {
    let environment = Environment::new();
    let (queue, fixture) = fixture_queue(&environment, Allocator::system());
    fixture.fail_destroy_once.store(true, Ordering::Release);
    fixture.destroy_errno.store(16, Ordering::Release);
    let pointer = queue.into_raw().cast();
    let native_busy = (u64::from(AMDF_STATUS_DOMAIN_ERRNO) << 32) | 16;
    let sentinel = ptr::NonNull::<amdf_user_queue_mapping_t>::dangling().as_ptr();
    let mut mapping = sentinel;
    // SAFETY: The native failure preserves this owner for cleanup only; it is
    // distinct from the earlier, mutation-free unconsumed-publication check.
    unsafe {
        assert_eq!(destroy(pointer), native_busy);
        assert_eq!(
            map(pointer, ptr::null_mut(), &raw mut mapping),
            PRECONDITION
        );
        assert_eq!(mapping, sentinel);
        assert_eq!(wait(pointer, 0, 0, 0), PRECONDITION);
        assert_eq!(destroy(pointer), 0);
    }
    assert_eq!(fixture.progress_calls.load(Ordering::Relaxed), 1);
    assert_eq!(environment.queues.load(Ordering::Relaxed), 0);
}
