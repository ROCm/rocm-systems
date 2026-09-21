//! Scripted KFD ownership tests use sparse files for real shared CPU mappings.
//! Native replies and errno are independent, so an error can model destruction
//! that already released an ID. No packets are submitted by these tests.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::super::memory;
use super::*;
use crate::host_storage::Allocator;
use crate::session::SessionLifetime;
use std::sync::Arc;
fn shared<T>(value: T) -> Shared<T> {
    Shared::new(value, Allocator::default()).unwrap()
}
use crate::queue::QueueProducerMode;
use std::collections::{BTreeMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};

static FILE_ID: AtomicUsize = AtomicUsize::new(0);

fn backing_file() -> File {
    let path = std::env::temp_dir().join(format!(
        "rocddi-queue-test-{}-{}",
        std::process::id(),
        FILE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    std::fs::remove_file(path).unwrap();
    file.set_len(8 * 1024 * 1024).unwrap();
    file
}

fn native_node() -> sysfs::NativeNode {
    sysfs::NativeNode {
        node: 1,
        gpu_id: 42,
        render_minor: Some(128),
        unique_id: Some(123),
        identity: [0; 16],
        queues: sysfs::NativeQueueProperties {
            gfx_target: 120_001,
            compute_units: 4,
            maximum_wave_count_per_compute_unit: 32,
            maximum_scratch_wave_count_per_compute_unit: 32,
            wavefront_size: 32,
            xcc_count: 1,
            shader_engine_count_per_xcc: 2,
            context_size: 65536,
            control_stack_size: 4096,
            sdma_engines: 2,
            compute_queues: 4,
            sdma_qualified: true,
        },
        local_memory_bytes: 1 << 30,
        public_memory_bytes: 0,
    }
}

fn descriptor(parameters: QueueParameters) -> QueueRequest {
    QueueRequest {
        ring_size_bytes: 4096,
        parameters,
        priority: QueuePriority::Normal,
        device_producer: false,
    }
}

fn aql(producer_mode: QueueProducerMode) -> QueueParameters {
    QueueParameters::Aql {
        producer_mode,
        inactive_signal: None,
        error_event: None,
        scratch: None,
    }
}

#[derive(Clone, Copy, Default, Eq, PartialEq)]
enum AqlRingBacking {
    #[default]
    Userptr,
    Gtt,
}

#[derive(Default)]
struct State {
    next_handle: u64,
    next_offset: u64,
    buffers: BTreeMap<u64, uapi::AllocMemory>,
    creates: usize,
    destroys: usize,
    frees: usize,
    live: bool,
    fail_allocation: Option<u64>,
    reject_userptr: bool,
    expected_aql_backing: AqlRingBacking,
    create_errno: Option<i32>,
    destroy_errno: VecDeque<i32>,
    free_errno: VecDeque<i32>,
    doorbell_map_errno: Option<i32>,
    doorbell_maps: usize,
    doorbell_unmaps: usize,
    maps: Vec<(u64, u32)>,
    unmaps: Vec<(u64, u32)>,
    peer_map_errno: VecDeque<i32>,
    peer_unmap_errno: VecDeque<i32>,
    lost: bool,
    expected_scratch: Option<QueueScratch>,
    expected_inactive_signal: Option<u64>,
    expected_error_event: Option<QueueErrorEvent>,
    expected_aql_queue_type: Option<u32>,
    expected_priority: u32,
    scratch_bases: usize,
    svm_attempts: usize,
    svm_errno: Option<i32>,
    svm_context: Option<(u64, u64)>,
    updates: Vec<uapi::UpdateQueue>,
    cu_masks: Vec<(uapi::SetCuMask, Vec<u32>)>,
}

struct Fixture {
    state: Arc<Mutex<State>>,
    contents: File,
    kfd: Shared<sys::Kfd>,
    vm: Shared<DeviceVm>,
}

impl Fixture {
    #[allow(
        clippy::too_many_lines,
        reason = "the hook checks acquisition and destruction against one observed state"
    )]
    fn new(doorbells_mappable: bool) -> Self {
        let state = Arc::new(Mutex::new(State::default()));
        let observed = state.clone();
        let render = backing_file();
        let contents = render.try_clone().unwrap();
        let observed_contents = contents.try_clone().unwrap();
        let process_memory = File::open("/proc/self/mem").unwrap();
        let endpoint = if doorbells_mappable {
            backing_file()
        } else {
            File::open("/dev/null").unwrap()
        };
        let kfd = shared(sys::Kfd::with_hook(
            endpoint,
            Arc::new(move |call| {
                let mut state = observed.lock().unwrap();
                match call {
                    sys::Call::Wait(args, event) if event.event_id == 19 => {
                        args.result = uapi::WAIT_TIMEOUT;
                    }
                    sys::Call::Wait(args, event) if event.event_id == 20 => {
                        if state.lost {
                            event.payload[0] = u64::from_ne_bytes([0, 0, 0, 0, 1, 0, 0, 0]);
                            event.payload[2] = 0x1234_5000;
                            event.payload[3] = u64::from_ne_bytes([42, 0, 0, 0, 0, 0, 0, 0]);
                            args.result = uapi::WAIT_COMPLETE;
                        } else {
                            args.result = uapi::WAIT_TIMEOUT;
                        }
                    }
                    sys::Call::DestroyEvent(_) => {}
                    sys::Call::SetScratchBackingVa(args) => {
                        assert_eq!(args.gpu_id, 42);
                        assert_ne!(args.va_address, 0);
                        assert_eq!(args.pad, 0);
                        state.scratch_bases += 1;
                    }
                    sys::Call::Allocate(args) => {
                        if state.reject_userptr && args.flags & uapi::USERPTR != 0 {
                            return Err(io::Error::from_raw_os_error(95));
                        }
                        state.next_handle += 1;
                        if state.fail_allocation == Some(state.next_handle) {
                            return Err(io::Error::from_raw_os_error(12));
                        }
                        args.handle = state.next_handle;
                        args.mmap_offset = state.next_offset;
                        state.next_offset += args.size;
                        assert_eq!(args.va % 4096, 0);
                        assert_eq!(args.size % 4096, 0);
                        state.buffers.insert(args.handle, **args);
                    }
                    sys::Call::Map(args, devices) => {
                        let doorbell = state
                            .buffers
                            .get(&args.handle)
                            .is_some_and(|buffer| buffer.flags & uapi::DOORBELL != 0);
                        for device in *devices {
                            state.maps.push((args.handle, *device));
                        }
                        if doorbell {
                            state.doorbell_maps += 1;
                            if let Some(errno) = state.doorbell_map_errno {
                                return Err(io::Error::from_raw_os_error(errno));
                            }
                        } else if devices.contains(&43) {
                            if let Some(errno) = state.peer_map_errno.pop_front() {
                                return Err(io::Error::from_raw_os_error(errno));
                            }
                        }
                        args.success = args.count;
                    }
                    sys::Call::Unmap(args, devices) => {
                        let doorbell = state
                            .buffers
                            .get(&args.handle)
                            .is_some_and(|buffer| buffer.flags & uapi::DOORBELL != 0);
                        for device in *devices {
                            state.unmaps.push((args.handle, *device));
                        }
                        if doorbell {
                            state.doorbell_unmaps += 1;
                        } else if devices.contains(&43) {
                            if let Some(errno) = state.peer_unmap_errno.pop_front() {
                                return Err(io::Error::from_raw_os_error(errno));
                            }
                        }
                        args.success = args.count;
                    }
                    sys::Call::Free(args) => {
                        let doorbell = state
                            .buffers
                            .get(&args.handle)
                            .is_some_and(|buffer| buffer.flags & uapi::DOORBELL != 0);
                        if !doorbell {
                            assert!(
                                !state.live,
                                "released backing before native queue destruction"
                            );
                        }
                        state.frees += 1;
                        if let Some(errno) = state.free_errno.pop_front() {
                            return Err(io::Error::from_raw_os_error(errno));
                        }
                        assert!(state.buffers.remove(&args.handle).is_some());
                    }
                    sys::Call::Svm(args, attributes) => {
                        let flags = uapi::SVM_FLAG_HOST_ACCESS
                            | uapi::SVM_FLAG_GPU_EXECUTE
                            | uapi::SVM_FLAG_GPU_ALWAYS_MAPPED;
                        assert_eq!(args.start_address % CWSR_ALIGNMENT as u64, 0);
                        assert_eq!(args.size, 69632);
                        assert_eq!(args.operation, uapi::SVM_OP_SET_ATTR);
                        assert_eq!(args.attribute_count, 6);
                        assert_eq!(
                            *attributes,
                            [
                                uapi::SvmAttribute {
                                    attribute_type: uapi::SVM_ATTR_PREFETCH_LOCATION,
                                    value: 42,
                                },
                                uapi::SvmAttribute {
                                    attribute_type: uapi::SVM_ATTR_PREFERRED_LOCATION,
                                    value: uapi::SVM_LOCATION_SYSTEM,
                                },
                                uapi::SvmAttribute {
                                    attribute_type: uapi::SVM_ATTR_CLEAR_FLAGS,
                                    value: !flags,
                                },
                                uapi::SvmAttribute {
                                    attribute_type: uapi::SVM_ATTR_SET_FLAGS,
                                    value: flags,
                                },
                                uapi::SvmAttribute {
                                    attribute_type: uapi::SVM_ATTR_ACCESS,
                                    value: 42,
                                },
                                uapi::SvmAttribute {
                                    attribute_type: uapi::SVM_ATTR_GRANULARITY,
                                    value: 0xff,
                                },
                            ]
                        );
                        state.svm_attempts += 1;
                        if let Some(errno) = state.svm_errno {
                            return Err(io::Error::from_raw_os_error(errno));
                        }
                        state.svm_context = Some((args.start_address, args.size));
                    }
                    sys::Call::CreateQueue(args) => {
                        state.creates += 1;
                        let expected_scratch = state.expected_scratch;
                        assert_eq!(args.gpu_id, 42);
                        assert!(matches!(args.queue_type, 0..=2));
                        assert_eq!(args.sdma_engine_id, 0);
                        assert_eq!(args.metadata_ring_size, 0);
                        assert_eq!(args.percentage, 100);
                        assert_eq!(args.priority, state.expected_priority);
                        if args.queue_type == 2 {
                            assert_eq!(state.scratch_bases, 1);
                        }
                        let at = |address| {
                            state
                                .buffers
                                .values()
                                .find(|buffer| {
                                    address >= buffer.va && address < buffer.va + buffer.size
                                })
                                .unwrap()
                        };
                        let ring = at(args.ring_address);
                        assert_eq!(ring.size, u64::from(args.ring_size).div_ceil(4096) * 4096);
                        let expected_ring_flags = if args.queue_type == 2
                            && state.expected_aql_backing == AqlRingBacking::Userptr
                        {
                            uapi::USERPTR
                                | uapi::WRITABLE
                                | uapi::EXECUTABLE
                                | uapi::COHERENT
                                | uapi::UNCACHED
                                | uapi::NO_SUBSTITUTE
                        } else {
                            uapi::GTT
                                | uapi::WRITABLE
                                | uapi::EXECUTABLE
                                | uapi::COHERENT
                                | uapi::NO_SUBSTITUTE
                                | uapi::UNCACHED
                        };
                        assert_eq!(ring.flags, expected_ring_flags);
                        let pointers = at(args.read_pointer);
                        assert_eq!(pointers.size, 4096);
                        let read_context = |address: u64, bytes: &mut [u8]| {
                            if state.svm_context.is_some_and(|(base, size)| {
                                address >= base
                                    && address
                                        .checked_add(bytes.len() as u64)
                                        .is_some_and(|end| end <= base + size)
                            }) {
                                process_memory.read_exact_at(bytes, address).unwrap();
                            } else {
                                let context = at(address);
                                assert_eq!(context.size, 69632);
                                assert_eq!(
                                    context.flags,
                                    uapi::GTT
                                        | uapi::WRITABLE
                                        | uapi::EXECUTABLE
                                        | uapi::COHERENT
                                        | uapi::NO_SUBSTITUTE
                                        | uapi::UNCACHED
                                );
                                observed_contents
                                    .read_exact_at(bytes, context.mmap_offset)
                                    .unwrap();
                            }
                        };
                        if args.queue_type == 2 {
                            assert_eq!(args.read_pointer, pointers.va + AQL_READ_OFFSET as u64);
                            assert_eq!(args.write_pointer, pointers.va + AQL_WRITE_OFFSET as u64);
                            let mut read = [1; 8];
                            let mut write = [1; 8];
                            observed_contents
                                .read_exact_at(
                                    &mut read,
                                    pointers.mmap_offset + AQL_READ_OFFSET as u64,
                                )
                                .unwrap();
                            observed_contents
                                .read_exact_at(
                                    &mut write,
                                    pointers.mmap_offset + AQL_WRITE_OFFSET as u64,
                                )
                                .unwrap();
                            assert_eq!(read, [0; 8]);
                            assert_eq!(write, [0; 8]);
                            let mut offset = [0; 4];
                            observed_contents
                                .read_exact_at(
                                    &mut offset,
                                    pointers.mmap_offset
                                        + offset_of!(
                                            AqlControlLayout,
                                            read_dispatch_id_field_base_byte_offset
                                        ) as u64,
                                )
                                .unwrap();
                            assert_eq!(
                                u32::from_ne_bytes(offset),
                                u32::try_from(AQL_READ_OFFSET).unwrap()
                            );
                            let read_u32 = |offset| {
                                let mut value = [0; 4];
                                observed_contents
                                    .read_exact_at(&mut value, pointers.mmap_offset + offset as u64)
                                    .unwrap();
                                u32::from_ne_bytes(value)
                            };
                            let read_u64 = |offset| {
                                let mut value = [0; 8];
                                observed_contents
                                    .read_exact_at(&mut value, pointers.mmap_offset + offset as u64)
                                    .unwrap();
                                u64::from_ne_bytes(value)
                            };
                            assert_eq!(
                                read_u32(
                                    offset_of!(AqlControlLayout, queue_header)
                                        + offset_of!(AqlQueueHeader, queue_type)
                                ),
                                state.expected_aql_queue_type.unwrap()
                            );
                            assert_eq!(read_u32(offset_of!(AqlControlLayout, max_cu_id)), 3);
                            assert_eq!(read_u32(offset_of!(AqlControlLayout, max_wave_id)), 31);
                            assert_eq!(
                                read_u64(offset_of!(AqlControlLayout, queue_inactive_signal)),
                                state.expected_inactive_signal.unwrap_or(0)
                            );
                            assert_eq!(
                                read_u32(offset_of!(AqlControlLayout, queue_properties)),
                                1 << 1
                            );
                            assert_eq!(
                                read_u64(offset_of!(AqlControlLayout, scratch_max_use_index)),
                                u64::MAX
                            );
                            assert_eq!(
                                read_u64(offset_of!(AqlControlLayout, alt_scratch_max_use_index)),
                                u64::MAX
                            );
                            if let Some(scratch) = expected_scratch {
                                assert_eq!(
                                    read_u32(offset_of!(AqlControlLayout, compute_tmpring_size)),
                                    64 | (2 << 12)
                                );
                                let resource =
                                    offset_of!(AqlControlLayout, scratch_resource_descriptor);
                                assert_eq!(
                                    read_u32(resource),
                                    u32::try_from(scratch.device_address).unwrap()
                                );
                                assert_eq!(
                                    read_u32(resource + 4),
                                    (u32::try_from(scratch.device_address >> 32).unwrap() & 0xffff)
                                        | (1 << 30)
                                );
                                assert_eq!(
                                    read_u32(resource + 8),
                                    u32::try_from(scratch.byte_length).unwrap()
                                );
                                assert_eq!(read_u32(resource + 12), 0x2081_4fac);
                                assert_eq!(
                                    read_u64(offset_of!(
                                        AqlControlLayout,
                                        scratch_backing_memory_location
                                    )),
                                    scratch.device_address
                                );
                                assert_eq!(
                                    read_u64(offset_of!(
                                        AqlControlLayout,
                                        scratch_backing_memory_byte_size
                                    )),
                                    0
                                );
                                assert_eq!(
                                    read_u32(offset_of!(
                                        AqlControlLayout,
                                        scratch_wave64_lane_byte_size
                                    )),
                                    scratch.maximum_private_segment_byte_length / 2
                                );
                            } else {
                                assert_eq!(
                                    read_u32(offset_of!(AqlControlLayout, compute_tmpring_size)),
                                    0
                                );
                                let resource =
                                    offset_of!(AqlControlLayout, scratch_resource_descriptor);
                                assert_eq!(read_u32(resource), 0);
                                assert_eq!(read_u32(resource + 4), 1 << 30);
                                assert_eq!(read_u32(resource + 8), 0);
                                assert_eq!(read_u32(resource + 12), GFX1201_SCRATCH_RESOURCE_WORD3);
                            }
                            for slot in 0..args.ring_size / 64 {
                                let mut header = [0; 2];
                                process_memory
                                    .read_exact_at(&mut header, ring.va + u64::from(slot) * 64)
                                    .unwrap();
                                assert_eq!(header, 1_u16.to_le_bytes());
                            }
                            assert_eq!(args.context_size, 65536);
                            assert_eq!(args.control_stack_size, 4096);
                            assert_eq!(args.eop_size, 4096);
                            assert_eq!(
                                at(args.eop_address).flags,
                                uapi::VRAM
                                    | uapi::WRITABLE
                                    | uapi::EXECUTABLE
                                    | uapi::NO_SUBSTITUTE
                            );
                            let mut header = [0; 40];
                            read_context(args.context_address, &mut header);
                            let mut expected = [0; 40];
                            expected[16..20].copy_from_slice(&65536_u32.to_ne_bytes());
                            expected[20..24].copy_from_slice(&4096_u32.to_ne_bytes());
                            if let Some(error_event) = state.expected_error_event {
                                expected[24..32]
                                    .copy_from_slice(&error_event.payload_address.to_ne_bytes());
                                expected[32..36].copy_from_slice(
                                    &u32::try_from(error_event.native_event_token)
                                        .unwrap()
                                        .to_ne_bytes(),
                                );
                            }
                            assert_eq!(header, expected);
                        } else if args.queue_type == 0 {
                            let cache_line = u64::from(util::host_cache_line_size().unwrap());
                            assert_eq!(args.read_pointer, pointers.va + PM4_READ_OFFSET as u64);
                            assert_eq!(args.write_pointer, pointers.va + cache_line);
                            let mut read = [1; 8];
                            let mut write = [1; 8];
                            let mut error = [1; 8];
                            observed_contents
                                .read_exact_at(
                                    &mut read,
                                    pointers.mmap_offset + PM4_READ_OFFSET as u64,
                                )
                                .unwrap();
                            observed_contents
                                .read_exact_at(&mut write, pointers.mmap_offset + cache_line)
                                .unwrap();
                            observed_contents
                                .read_exact_at(&mut error, pointers.mmap_offset + 2 * cache_line)
                                .unwrap();
                            assert_eq!(read, [0; 8]);
                            assert_eq!(write, [0; 8]);
                            assert_eq!(error, [0; 8]);
                            assert_eq!(args.context_size, 65536);
                            assert_eq!(args.control_stack_size, 4096);
                            assert_eq!(args.eop_size, 4096);
                            let mut header = [0; 32];
                            read_context(args.context_address, &mut header);
                            let mut expected = [0; 32];
                            expected[16..20].copy_from_slice(&65536_u32.to_ne_bytes());
                            expected[20..24].copy_from_slice(&4096_u32.to_ne_bytes());
                            expected[24..32]
                                .copy_from_slice(&(pointers.va + 2 * cache_line).to_ne_bytes());
                            assert_eq!(header, expected);
                        } else {
                            assert_eq!(args.read_pointer, pointers.va + SDMA_READ_OFFSET as u64);
                            assert_eq!(args.write_pointer, pointers.va + SDMA_WRITE_OFFSET as u64);
                            let mut indices = [1; 16];
                            observed_contents
                                .read_exact_at(&mut indices, pointers.mmap_offset)
                                .unwrap();
                            assert_eq!(indices, [0; 16]);
                            assert_eq!(args.eop_address, 0);
                            assert_eq!(args.context_address, 0);
                        }
                        if let Some(errno) = state.create_errno {
                            state.live = errno == 14;
                            return Err(io::Error::from_raw_os_error(errno));
                        }
                        state.live = true;
                        args.queue_id = 0;
                        args.doorbell_offset = 24;
                    }
                    sys::Call::UpdateQueue(args) => {
                        assert!(state.live, "updated a queue after native destruction");
                        assert_eq!(args.queue_id, 0, "zero is a valid queue ID");
                        state.updates.push(**args);
                    }
                    sys::Call::SetCuMask(args, mask) => {
                        assert!(state.live, "masked a queue after native destruction");
                        assert_eq!(args.queue_id, 0, "zero is a valid queue ID");
                        assert_eq!(args.count as usize, mask.len() * 32);
                        assert_eq!(args.mask, mask.as_ptr() as u64);
                        state.cu_masks.push((**args, mask.to_vec()));
                    }
                    sys::Call::DestroyQueue(args) => {
                        assert_eq!(args.queue_id, 0, "zero is a valid queue ID");
                        assert!(
                            state.live,
                            "destroy replayed an ID that could have been reused"
                        );
                        assert_eq!(args.pad, 0);
                        state.destroys += 1;
                        if let Some(errno) = state.destroy_errno.pop_front() {
                            if matches!(errno, 5 | 62 | 14) {
                                state.live = false;
                            }
                            return Err(io::Error::from_raw_os_error(errno));
                        }
                        state.live = false;
                    }
                    _ => panic!("unexpected queue ioctl"),
                }
                Ok(())
            }),
        ));
        let vm = memory::queue_fixture(kfd.clone(), render, native_node());
        Self {
            state,
            contents,
            kfd,
            vm,
        }
    }

    fn create(&self, desc: QueueRequest) -> Result<Owned<KfdQueue>, Error> {
        self.create_with_lifetime(desc, SessionLifetime::Process)
    }

    fn create_with_lifetime(
        &self,
        desc: QueueRequest,
        lifetime: SessionLifetime,
    ) -> Result<Owned<KfdQueue>, Error> {
        let mut state = self.state.lock().unwrap();
        state.expected_aql_backing = if lifetime == SessionLifetime::Session {
            AqlRingBacking::Gtt
        } else {
            AqlRingBacking::Userptr
        };
        state.expected_aql_queue_type = match desc.parameters {
            QueueParameters::Aql {
                producer_mode: QueueProducerMode::Single,
                ..
            } => Some(1),
            QueueParameters::Aql {
                producer_mode: QueueProducerMode::Multiple,
                ..
            } => Some(0),
            QueueParameters::Pm4 | QueueParameters::Sdma => None,
        };
        state.expected_priority = match desc.priority {
            QueuePriority::Low => 0,
            QueuePriority::Normal => 7,
            QueuePriority::High => 15,
        };
        if let QueueParameters::Aql {
            inactive_signal,
            error_event,
            ..
        } = desc.parameters
        {
            state.expected_inactive_signal = inactive_signal;
            state.expected_error_event = error_event;
        }
        drop(state);
        let request = Request::validate(&native_node(), desc)?;
        KfdQueue::create(self.vm.clone(), &request, lifetime)
    }

    fn peer(&self) -> Shared<DeviceVm> {
        self.peer_with_range((0x10000, isize::MAX as u64))
    }

    fn peer_with_range(&self, bounds: (u64, u64)) -> Shared<DeviceVm> {
        let mut node = native_node();
        node.node = 2;
        node.gpu_id = 43;
        node.render_minor = Some(129);
        node.unique_id = Some(456);
        node.identity[0] = 1;
        memory::queue_fixture_with_range(self.kfd.clone(), backing_file(), node, bounds)
    }

    fn assert_released(&self) {
        let state = self.state.lock().unwrap();
        assert!(!state.live);
        assert!(state.buffers.is_empty());
    }

    fn read_aql_u32(&self, queue: &KfdQueue, offset: usize) -> u32 {
        let base = queue.info.read_index_device_address - AQL_READ_OFFSET as u64;
        let file_offset = {
            let state = self.state.lock().unwrap();
            state
                .buffers
                .values()
                .find(|buffer| buffer.va == base)
                .unwrap()
                .mmap_offset
                + offset as u64
        };
        let mut bytes = [0; 4];
        self.contents
            .read_exact_at(&mut bytes, file_offset)
            .unwrap();
        u32::from_ne_bytes(bytes)
    }

    fn read_aql_u64(&self, queue: &KfdQueue, offset: usize) -> u64 {
        let base = queue.info.read_index_device_address - AQL_READ_OFFSET as u64;
        let file_offset = {
            let state = self.state.lock().unwrap();
            state
                .buffers
                .values()
                .find(|buffer| buffer.va == base)
                .unwrap()
                .mmap_offset
                + offset as u64
        };
        let mut bytes = [0; 8];
        self.contents
            .read_exact_at(&mut bytes, file_offset)
            .unwrap();
        u64::from_ne_bytes(bytes)
    }
}

#[test]
fn concurrent_queue_creation_enables_runtime_once_before_acquisition() {
    let enables = Arc::new(AtomicUsize::new(0));
    let allocations = Arc::new(AtomicUsize::new(0));
    let observed_enables = enables.clone();
    let observed_allocations = allocations.clone();
    let mut kfd = shared(sys::Kfd::with_hook(
        File::open("/dev/null").unwrap(),
        Arc::new(move |call| match call {
            sys::Call::RuntimeEnable(args) => {
                assert_eq!(args.r_debug, 0);
                assert_eq!(args.capabilities_mask, 0);
                if args.mode_mask == 1 {
                    observed_enables.fetch_add(1, Ordering::Relaxed);
                }
                Ok(())
            }
            sys::Call::Wait(args, _) => {
                args.result = uapi::WAIT_TIMEOUT;
                Ok(())
            }
            sys::Call::Allocate(_) => {
                observed_allocations.fetch_add(1, Ordering::Relaxed);
                Err(io::Error::from_raw_os_error(12))
            }
            sys::Call::DestroyEvent(_) => Ok(()),
            _ => panic!("unexpected queue admission call"),
        }),
    ));
    let vm = memory::queue_fixture(kfd.clone(), File::open("/dev/null").unwrap(), native_node());
    let native = native_node();
    std::thread::scope(|scope| {
        let threads = (0..8)
            .map(|_| {
                let vm = vm.clone();
                scope.spawn(move || {
                    assert_eq!(
                        create(
                            vm,
                            &native,
                            descriptor(QueueParameters::Sdma),
                            SessionLifetime::Process
                        )
                        .err()
                        .unwrap()
                        .kind(),
                        ErrorKind::ResourceExhausted
                    );
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }
    });
    assert_eq!(enables.load(Ordering::Relaxed), 1);
    assert_eq!(allocations.load(Ordering::Relaxed), 8);
    drop(vm);
    Shared::get_mut(&mut kfd).unwrap().close().unwrap();
}

#[test]
fn runtime_enable_failure_prevents_queue_backing_acquisition() {
    let allocations = Arc::new(AtomicUsize::new(0));
    let observed = allocations.clone();
    let mut kfd = shared(sys::Kfd::with_hook(
        File::open("/dev/null").unwrap(),
        Arc::new(move |call| match call {
            sys::Call::RuntimeEnable(args) if args.mode_mask == 1 => {
                Err(io::Error::from_raw_os_error(5))
            }
            sys::Call::RuntimeEnable(args) if args.mode_mask == 0 => Ok(()),
            sys::Call::Allocate(_) => {
                observed.fetch_add(1, Ordering::Relaxed);
                panic!("queue backing was acquired after runtime-enable failure")
            }
            sys::Call::DestroyEvent(_) => Ok(()),
            _ => panic!("unexpected queue admission call"),
        }),
    ));
    let vm = memory::queue_fixture(kfd.clone(), File::open("/dev/null").unwrap(), native_node());
    assert_eq!(
        create(
            vm.clone(),
            &native_node(),
            descriptor(QueueParameters::Sdma),
            SessionLifetime::Process,
        )
        .err()
        .unwrap()
        .native_error_code(),
        Some(5)
    );
    assert_eq!(allocations.load(Ordering::Relaxed), 0);
    drop(vm);
    Shared::get_mut(&mut kfd).unwrap().close().unwrap();
}

#[test]
fn all_formats_and_priorities_reach_native_queue_creation() {
    for (parameters, priority, unit, read_width, wraps) in [
        (
            QueueParameters::Pm4,
            QueuePriority::Normal,
            4,
            QueueAccessWidth::Bits64,
            true,
        ),
        (
            aql(QueueProducerMode::Single),
            QueuePriority::Low,
            64,
            QueueAccessWidth::Bits64,
            false,
        ),
        (
            aql(QueueProducerMode::Single),
            QueuePriority::Normal,
            64,
            QueueAccessWidth::Bits64,
            false,
        ),
        (
            aql(QueueProducerMode::Single),
            QueuePriority::High,
            64,
            QueueAccessWidth::Bits64,
            false,
        ),
        (
            aql(QueueProducerMode::Multiple),
            QueuePriority::Normal,
            64,
            QueueAccessWidth::Bits64,
            false,
        ),
        (
            QueueParameters::Sdma,
            QueuePriority::Normal,
            1,
            QueueAccessWidth::Bits64,
            false,
        ),
    ] {
        let fixture = Fixture::new(true);
        let mut desc = descriptor(parameters);
        desc.priority = priority;
        let mut queue = fixture.create(desc).unwrap();
        let info = queue.info().unwrap();
        assert_eq!(info.index_unit_bytes, unit);
        assert_eq!(info.read_index_width, read_width);
        assert_eq!(info.read_index_wraps, wraps);
        assert_eq!(info.write_index_width, QueueAccessWidth::Bits64);
        assert_eq!(info.doorbell_width, QueueAccessWidth::Bits64);
        assert_eq!(info.doorbell_host_address % 8, 0);
        assert_eq!(
            crate::test_support::allocation_counter::allocations(|| {
                for _ in 0..100 {
                    assert_eq!(queue.info().unwrap(), info);
                }
            }),
            0
        );
        queue.destroy().unwrap();
        queue.destroy().unwrap();
        assert!(queue.info().is_err());
        drop(queue);
        fixture.assert_released();
        assert_eq!(fixture.state.lock().unwrap().destroys, 1);
    }
}

#[test]
fn instance_aql_ring_uses_coherent_gtt_when_kfd_rejects_userptr() {
    let fixture = Fixture::new(true);
    fixture.state.lock().unwrap().reject_userptr = true;
    let mut queue = fixture
        .create_with_lifetime(
            descriptor(aql(QueueProducerMode::Single)),
            SessionLifetime::Session,
        )
        .unwrap();
    assert_eq!(queue.info().unwrap().index_unit_bytes, 64);
    queue.destroy().unwrap();
    drop(queue);
    fixture.assert_released();
}

#[test]
fn queue_control_updates_forward_the_native_records() {
    let fixture = Fixture::new(true);
    let mut queue = fixture
        .create(descriptor(aql(QueueProducerMode::Single)))
        .unwrap();
    let info = queue.info().unwrap();

    queue.set_priority(QueuePriority::High).unwrap();
    let mask = [0x0000_0005, 0x8000_0000];
    queue.set_cu_mask(&mask).unwrap();
    queue.inactivate().unwrap();
    queue.inactivate().unwrap();

    {
        let state = fixture.state.lock().unwrap();
        assert_eq!(state.updates.len(), 2);
        let priority = state.updates[0];
        assert_eq!(priority.ring_address, info.ring_device_address);
        assert_eq!(priority.queue_id, 0);
        assert_eq!(priority.ring_size, 4096);
        assert_eq!(priority.percentage, 100);
        assert_eq!(priority.priority, 15);
        let inactive = state.updates[1];
        assert_eq!(inactive.ring_address, 0);
        assert_eq!(inactive.queue_id, 0);
        assert_eq!(inactive.ring_size, 0);
        assert_eq!(inactive.percentage, 0);
        assert_eq!(inactive.priority, 15);

        assert_eq!(state.cu_masks.len(), 1);
        assert_eq!(state.cu_masks[0].0.queue_id, 0);
        assert_eq!(state.cu_masks[0].0.count, 64);
        assert_eq!(state.cu_masks[0].1, mask);
    }

    queue.destroy().unwrap();
    fixture.assert_released();
}

#[test]
fn inactive_queue_destruction_ignores_unmatched_indices() {
    let fixture = Fixture::new(true);
    let mut queue = fixture.create(descriptor(QueueParameters::Sdma)).unwrap();
    queue.backing[1]
        .as_mut()
        .unwrap()
        .write_bytes(SDMA_WRITE_OFFSET, &4096_u64.to_ne_bytes())
        .unwrap();
    assert_eq!(queue.progress().unwrap(), (0, 4096));

    queue.inactivate().unwrap();
    queue.destroy().unwrap();

    assert_eq!(fixture.state.lock().unwrap().destroys, 1);
    fixture.assert_released();
}

#[test]
fn gfx1201_scratch_populates_the_firmware_queue_control_fields() {
    let fixture = Fixture::new(true);
    let scratch = QueueScratch {
        device_address: 0x1234_0000,
        byte_length: 64 * 1024,
        maximum_private_segment_byte_length: 16,
        maximum_wave_count: 128,
    };
    fixture.state.lock().unwrap().expected_scratch = Some(scratch);
    let desc = descriptor(QueueParameters::Aql {
        producer_mode: QueueProducerMode::Single,
        inactive_signal: None,
        error_event: None,
        scratch: Some(scratch),
    });
    let mut queue = fixture.create(desc).unwrap();
    queue.destroy().unwrap();
    fixture.assert_released();
}

#[test]
fn gfx1201_queue_publishes_its_event_signals() {
    let fixture = Fixture::new(true);
    let desc = descriptor(QueueParameters::Aql {
        producer_mode: QueueProducerMode::Single,
        inactive_signal: Some(0x1234_5000),
        error_event: Some(QueueErrorEvent {
            payload_address: 0x1234_6008,
            native_event_token: 7,
        }),
        scratch: None,
    });
    let mut queue = fixture.create(desc).unwrap();
    queue.destroy().unwrap();
    fixture.assert_released();
}

#[test]
fn stopped_gfx1201_queue_accepts_replacement_scratch_without_recreation() {
    let fixture = Fixture::new(true);
    let mut queue = fixture
        .create(descriptor(aql(QueueProducerMode::Single)))
        .unwrap();
    let scratch = QueueScratch {
        device_address: 0x1234_0000,
        byte_length: 64 * 1024,
        maximum_private_segment_byte_length: 16,
        maximum_wave_count: 128,
    };
    queue.set_scratch(scratch).unwrap();
    assert_eq!(fixture.state.lock().unwrap().creates, 1);
    assert_eq!(
        fixture.read_aql_u32(&queue, offset_of!(AqlControlLayout, compute_tmpring_size)),
        64 | (2 << 12)
    );
    assert_eq!(
        fixture.read_aql_u64(
            &queue,
            offset_of!(AqlControlLayout, scratch_backing_memory_location)
        ),
        scratch.device_address
    );
    assert_eq!(
        fixture.read_aql_u64(&queue, offset_of!(AqlControlLayout, scratch_max_use_index)),
        u64::MAX
    );
    assert_eq!(
        fixture.read_aql_u64(
            &queue,
            offset_of!(AqlControlLayout, alt_scratch_max_use_index)
        ),
        u64::MAX
    );
    queue.destroy().unwrap();
    fixture.assert_released();
}

#[test]
fn invalid_scratch_geometry_fails_before_native_acquisition() {
    let fixture = Fixture::new(true);
    let base = descriptor(aql(QueueProducerMode::Single));
    let scratch = QueueScratch {
        device_address: 0x1234_0000,
        byte_length: 64 * 1024,
        maximum_private_segment_byte_length: 16,
        maximum_wave_count: 128,
    };
    for invalid in [
        QueueScratch {
            device_address: scratch.device_address + 1,
            ..scratch
        },
        QueueScratch {
            byte_length: scratch.byte_length - 1,
            ..scratch
        },
        QueueScratch {
            maximum_private_segment_byte_length: 7,
            ..scratch
        },
        QueueScratch {
            maximum_wave_count: 129,
            ..scratch
        },
    ] {
        assert!(
            Request::validate(
                &native_node(),
                QueueRequest {
                    parameters: QueueParameters::Aql {
                        producer_mode: QueueProducerMode::Single,
                        inactive_signal: None,
                        error_event: None,
                        scratch: Some(invalid),
                    },
                    ..base
                }
            )
            .is_err()
        );
    }
    assert_eq!(fixture.state.lock().unwrap().next_handle, 0);
}

#[test]
fn unsupported_options_and_missing_context_sizes_fail_before_acquisition() {
    let fixture = Fixture::new(true);
    let base = descriptor(aql(QueueProducerMode::Single));
    for desc in [
        QueueRequest {
            ring_size_bytes: 512,
            ..base
        },
        QueueRequest {
            ring_size_bytes: 1 << 32,
            ..base
        },
        QueueRequest {
            ring_size_bytes: 1536,
            ..base
        },
        QueueRequest {
            parameters: QueueParameters::Sdma,
            priority: QueuePriority::High,
            ..base
        },
        QueueRequest {
            parameters: QueueParameters::Aql {
                producer_mode: QueueProducerMode::Single,
                inactive_signal: None,
                error_event: Some(QueueErrorEvent {
                    payload_address: 0x1000,
                    native_event_token: u64::from(u32::MAX) + 1,
                }),
                scratch: None,
            },
            ..base
        },
    ] {
        assert!(fixture.create(desc).is_err());
    }
    let mut native = native_node();
    native.queues.context_size = 0;
    assert!(Request::validate(&native, base).is_err());
    let sdma = descriptor(QueueParameters::Sdma);
    let mut native = native_node();
    native.queues.sdma_engines = 0;
    assert!(Request::validate(&native, sdma).is_err());
    let mut native = native_node();
    native.queues.sdma_qualified = false;
    assert!(Request::validate(&native, sdma).is_err());
    let pm4 = descriptor(QueueParameters::Pm4);
    for invalid in [
        QueueRequest {
            ring_size_bytes: 8192,
            ..pm4
        },
        QueueRequest {
            priority: QueuePriority::High,
            ..pm4
        },
        QueueRequest {
            device_producer: true,
            ..pm4
        },
    ] {
        assert!(Request::validate(&native_node(), invalid).is_err());
    }
    let mut native = native_node();
    native.queues.gfx_target = 120_000;
    assert!(Request::validate(&native, pm4).is_err());
    assert_eq!(fixture.state.lock().unwrap().next_handle, 0);
}

#[test]
fn backing_and_create_failures_release_only_resources_that_were_acquired() {
    let desc = descriptor(aql(QueueProducerMode::Single));
    for failed in 1..=3 {
        let fixture = Fixture::new(true);
        fixture.state.lock().unwrap().fail_allocation = Some(failed);
        assert!(fixture.create(desc).is_err());
        fixture.assert_released();
        assert_eq!(fixture.state.lock().unwrap().creates, 0);
    }
    let fixture = Fixture::new(true);
    {
        let mut state = fixture.state.lock().unwrap();
        state.svm_errno = Some(95);
        state.fail_allocation = Some(4);
    }
    assert!(fixture.create(desc).is_err());
    fixture.assert_released();
    assert_eq!(fixture.state.lock().unwrap().creates, 0);

    let fixture = Fixture::new(true);
    fixture.state.lock().unwrap().create_errno = Some(12);
    assert!(fixture.create(desc).is_err());
    fixture.assert_released();
    assert_eq!(fixture.state.lock().unwrap().destroys, 0);

    let fixture = Fixture::new(true);
    fixture.state.lock().unwrap().create_errno = Some(14);
    assert!(fixture.create(desc).is_err());
    let state = fixture.state.lock().unwrap();
    assert_eq!(state.creates, 1);
    assert_eq!(state.destroys, 0);
    assert_eq!(state.frees, 0);
    assert_eq!(state.buffers.len(), 3);
}

#[test]
fn compute_queue_falls_back_to_gtt_when_svm_registration_fails() {
    let fixture = Fixture::new(true);
    fixture.state.lock().unwrap().svm_errno = Some(95);
    let mut queue = fixture
        .create(descriptor(aql(QueueProducerMode::Single)))
        .unwrap();
    {
        let state = fixture.state.lock().unwrap();
        assert_eq!(state.svm_attempts, 1);
        assert_eq!(state.buffers.len(), 4);
    }
    queue.destroy().unwrap();
    fixture.assert_released();
}

#[test]
fn doorbell_mapping_failure_destroys_known_queue_before_releasing_backing() {
    for destroy_errno in [None, Some(16)] {
        let fixture = Fixture::new(false);
        fixture
            .state
            .lock()
            .unwrap()
            .destroy_errno
            .extend(destroy_errno);
        assert!(
            fixture
                .create(descriptor(aql(QueueProducerMode::Single)))
                .is_err()
        );
        let state = fixture.state.lock().unwrap();
        assert_eq!(state.destroys, 1);
        assert_eq!(
            state.buffers.len(),
            if destroy_errno.is_some() { 3 } else { 0 }
        );
        assert_eq!(state.live, destroy_errno.is_some());
    }
}

#[test]
fn cleanup_resumes_without_replaying_a_released_native_id() {
    let fixture = Fixture::new(true);
    let mut queue = fixture
        .create(descriptor(aql(QueueProducerMode::Single)))
        .unwrap();
    fixture.state.lock().unwrap().destroy_errno.push_back(16);
    assert_eq!(queue.destroy().unwrap_err().kind(), ErrorKind::Busy);
    assert!(queue.info().is_err());
    assert_eq!(fixture.state.lock().unwrap().frees, 0);
    fixture.state.lock().unwrap().free_errno.push_back(16);
    assert_eq!(queue.destroy().unwrap_err().kind(), ErrorKind::Busy);
    assert_eq!(fixture.state.lock().unwrap().destroys, 2);
    queue.destroy().unwrap();
    assert_eq!(fixture.state.lock().unwrap().destroys, 2);
    drop(queue);
    fixture.assert_released();
}

#[test]
fn ambiguous_destroy_errors_never_replay_ids_or_release_dependencies() {
    for errno in [5, 62, 14, 22] {
        let fixture = Fixture::new(true);
        let mut queue = fixture
            .create(descriptor(aql(QueueProducerMode::Single)))
            .unwrap();
        fixture.state.lock().unwrap().destroy_errno.push_back(errno);
        assert!(queue.destroy().is_err());
        assert_eq!(
            queue.destroy().unwrap_err().kind(),
            ErrorKind::DriverContract
        );
        assert!(queue.info().is_err());
        drop(queue);
        let state = fixture.state.lock().unwrap();
        assert_eq!(state.destroys, 1);
        assert_eq!(state.frees, 0);
        assert_eq!(state.buffers.len(), 3);
    }
}

#[test]
fn reported_loss_rejects_transport_queries_without_preventing_cleanup() {
    let fixture = Fixture::new(true);
    let mut queue = fixture.create(descriptor(QueueParameters::Sdma)).unwrap();
    fixture.state.lock().unwrap().lost = true;
    assert_eq!(queue.info().unwrap_err().kind(), ErrorKind::DeviceLost);
    queue.destroy().unwrap();
    fixture.assert_released();
}

#[test]
fn doorbell_mapping_is_shared_and_rejects_replaced_or_misaligned_slices() {
    let fixture = Fixture::new(true);
    let base = fixture
        .vm
        .doorbells
        .addresses(&fixture.vm, 0, false)
        .unwrap()
        .host;
    assert_eq!(
        fixture
            .vm
            .doorbells
            .addresses(&fixture.vm, 8, false)
            .unwrap()
            .host,
        base + 8
    );
    assert_eq!(
        fixture
            .vm
            .doorbells
            .addresses(&fixture.vm, 16, false)
            .unwrap()
            .host,
        base + 16
    );
    assert!(
        fixture
            .vm
            .doorbells
            .addresses(&fixture.vm, 1, false)
            .is_err()
    );
    assert!(
        fixture
            .vm
            .doorbells
            .addresses(&fixture.vm, 8192, false)
            .is_err()
    );
}

#[test]
fn device_producer_maps_ring_indices_and_doorbell_into_the_queue_vm() {
    let mut fixture = Fixture::new(true);
    let mut host_queue = fixture.create(descriptor(QueueParameters::Sdma)).unwrap();
    assert_eq!(host_queue.info().unwrap().doorbell_device_address, None);
    host_queue.destroy().unwrap();
    drop(host_queue);
    let mut desc = descriptor(aql(QueueProducerMode::Single));
    desc.device_producer = true;
    let mut queue = fixture.create(desc).unwrap();
    let info = queue.info().unwrap();
    assert_eq!(info.ring_device_address, info.ring_host_address as u64);
    assert_eq!(
        info.read_index_device_address,
        info.read_index_host_address as u64
    );
    assert_eq!(
        info.write_index_device_address,
        info.write_index_host_address as u64
    );
    let doorbell = info.doorbell_device_address.unwrap();
    assert_eq!(doorbell, info.doorbell_host_address as u64);
    {
        let state = fixture.state.lock().unwrap();
        let backing = state
            .buffers
            .values()
            .find(|buffer| buffer.flags & uapi::DOORBELL != 0)
            .unwrap();
        assert_eq!(backing.va + 24, doorbell);
        assert_eq!(backing.size, DOORBELL_SLICE as u64);
        assert_eq!(
            backing.flags,
            uapi::DOORBELL | uapi::WRITABLE | uapi::COHERENT | uapi::NO_SUBSTITUTE
        );
        assert_eq!(state.doorbell_maps, 1);
    }
    queue.destroy().unwrap();
    drop(queue);
    Shared::get_mut(&mut fixture.vm).unwrap().close().unwrap();
    assert_eq!(fixture.state.lock().unwrap().doorbell_unmaps, 1);
    fixture.assert_released();
}

#[test]
fn peer_device_producer_maps_each_queue_resource_once_and_retains_the_vm() {
    let mut fixture = Fixture::new(true);
    let mut desc = descriptor(QueueParameters::Sdma);
    desc.device_producer = true;
    let mut queue = fixture.create(desc).unwrap();
    let mut peer = fixture.peer();
    let info = queue.map_device(peer.clone()).unwrap();
    assert_eq!(info.ring_device_address, queue.info.ring_device_address);
    assert_eq!(
        info.read_index_device_address,
        queue.info.read_index_device_address
    );
    assert_eq!(
        info.write_index_device_address,
        queue.info.write_index_device_address
    );
    assert_eq!(
        info.doorbell_device_address,
        queue.info.doorbell_device_address
    );
    assert_eq!(queue.map_device(peer.clone()).unwrap(), info);
    {
        let state = fixture.state.lock().unwrap();
        assert_eq!(
            state
                .maps
                .iter()
                .filter(|(_, gpu_id)| *gpu_id == 43)
                .count(),
            3
        );
    }
    queue.destroy().unwrap();
    drop(queue);
    {
        let state = fixture.state.lock().unwrap();
        assert_eq!(
            state
                .unmaps
                .iter()
                .filter(|(_, gpu_id)| *gpu_id == 43)
                .count(),
            2
        );
    }
    Shared::get_mut(&mut fixture.vm).unwrap().close().unwrap();
    {
        let state = fixture.state.lock().unwrap();
        assert_eq!(
            state
                .unmaps
                .iter()
                .filter(|(_, gpu_id)| *gpu_id == 43)
                .count(),
            3
        );
    }
    Shared::get_mut(&mut peer).unwrap().close().unwrap();
    fixture.assert_released();
}

#[test]
fn peer_device_mapping_requires_the_creation_capability_and_one_kfd_instance() {
    let mut fixture = Fixture::new(true);
    let mut queue = fixture.create(descriptor(QueueParameters::Sdma)).unwrap();
    let mut peer = fixture.peer();
    assert_eq!(
        queue.map_device(peer.clone()).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    queue.destroy().unwrap();
    drop(queue);
    Shared::get_mut(&mut fixture.vm).unwrap().close().unwrap();
    Shared::get_mut(&mut peer).unwrap().close().unwrap();
    fixture.assert_released();

    let mut fixture = Fixture::new(true);
    let mut other = Fixture::new(true);
    let mut desc = descriptor(QueueParameters::Sdma);
    desc.device_producer = true;
    let mut queue = fixture.create(desc).unwrap();
    assert_eq!(
        queue.map_device(other.vm.clone()).unwrap_err().kind(),
        ErrorKind::InvalidArgument
    );
    queue.destroy().unwrap();
    drop(queue);
    Shared::get_mut(&mut fixture.vm).unwrap().close().unwrap();
    Shared::get_mut(&mut other.vm).unwrap().close().unwrap();
    fixture.assert_released();
    other.assert_released();
}

#[test]
fn peer_device_mapping_rejects_an_incompatible_gpu_address_aperture() {
    let mut fixture = Fixture::new(true);
    let mut desc = descriptor(QueueParameters::Sdma);
    desc.device_producer = true;
    let mut queue = fixture.create(desc).unwrap();
    let mut peer = fixture.peer_with_range((1, 1));
    assert_eq!(
        queue.map_device(peer.clone()).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    assert_eq!(
        fixture
            .state
            .lock()
            .unwrap()
            .maps
            .iter()
            .filter(|(_, gpu_id)| *gpu_id == 43)
            .count(),
        0
    );
    queue.destroy().unwrap();
    drop(queue);
    Shared::get_mut(&mut fixture.vm).unwrap().close().unwrap();
    Shared::get_mut(&mut peer).unwrap().close().unwrap();
    fixture.assert_released();
}

#[test]
fn unsupported_peer_doorbell_rolls_back_queue_backing_before_retry() {
    let mut fixture = Fixture::new(true);
    let mut desc = descriptor(QueueParameters::Sdma);
    desc.device_producer = true;
    let mut queue = fixture.create(desc).unwrap();
    let mut peer = fixture.peer();
    fixture.state.lock().unwrap().doorbell_map_errno = Some(95);
    assert_eq!(
        queue.map_device(peer.clone()).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    {
        let state = fixture.state.lock().unwrap();
        assert_eq!(
            state
                .maps
                .iter()
                .filter(|(_, gpu_id)| *gpu_id == 43)
                .count(),
            3
        );
        assert_eq!(
            state
                .unmaps
                .iter()
                .filter(|(_, gpu_id)| *gpu_id == 43)
                .count(),
            2
        );
    }
    fixture.state.lock().unwrap().doorbell_map_errno = None;
    assert!(queue.map_device(peer.clone()).is_ok());
    queue.destroy().unwrap();
    drop(queue);
    Shared::get_mut(&mut fixture.vm).unwrap().close().unwrap();
    Shared::get_mut(&mut peer).unwrap().close().unwrap();
    fixture.assert_released();
}

#[test]
fn peer_queue_cleanup_retries_only_the_unfinished_unmap() {
    let mut fixture = Fixture::new(true);
    let mut desc = descriptor(QueueParameters::Sdma);
    desc.device_producer = true;
    let mut queue = fixture.create(desc).unwrap();
    let mut peer = fixture.peer();
    queue.map_device(peer.clone()).unwrap();
    fixture.state.lock().unwrap().peer_unmap_errno.push_back(16);
    assert_eq!(queue.destroy().unwrap_err().kind(), ErrorKind::Busy);
    let first_unmaps = fixture
        .state
        .lock()
        .unwrap()
        .unmaps
        .iter()
        .filter(|(_, gpu_id)| *gpu_id == 43)
        .count();
    assert_eq!(first_unmaps, 1);
    queue.destroy().unwrap();
    drop(queue);
    let peer_unmaps = fixture
        .state
        .lock()
        .unwrap()
        .unmaps
        .iter()
        .filter(|(_, gpu_id)| *gpu_id == 43)
        .count();
    assert_eq!(peer_unmaps, 3);
    Shared::get_mut(&mut fixture.vm).unwrap().close().unwrap();
    Shared::get_mut(&mut peer).unwrap().close().unwrap();
    fixture.assert_released();
}

#[test]
fn ambiguous_peer_mapping_retains_queue_backing_and_the_peer_vm() {
    let fixture = Fixture::new(true);
    let mut desc = descriptor(QueueParameters::Sdma);
    desc.device_producer = true;
    let mut queue = fixture.create(desc).unwrap();
    let peer = fixture.peer();
    fixture.state.lock().unwrap().peer_map_errno.push_back(14);
    assert_eq!(
        queue.map_device(peer.clone()).unwrap_err().kind(),
        ErrorKind::Driver
    );
    assert_eq!(
        queue.destroy().unwrap_err().kind(),
        ErrorKind::DriverContract
    );
    let state = fixture.state.lock().unwrap();
    assert!(!state.live);
    assert!(!state.buffers.is_empty());
    drop(state);
    drop(queue);
    drop(peer);
}

#[test]
fn failed_device_doorbell_mapping_rolls_back_before_queue_cleanup() {
    let fixture = Fixture::new(true);
    fixture.state.lock().unwrap().doorbell_map_errno = Some(95);
    let mut desc = descriptor(QueueParameters::Sdma);
    desc.device_producer = true;
    assert_eq!(
        fixture.create(desc).err().unwrap().kind(),
        ErrorKind::Unsupported
    );
    let state = fixture.state.lock().unwrap();
    assert_eq!(state.doorbell_maps, 1);
    assert_eq!(state.doorbell_unmaps, 0);
    assert!(!state.live);
    assert!(state.buffers.is_empty());
}

#[test]
fn doorbell_cleanup_retries_only_the_unfinished_native_step() {
    let mut fixture = Fixture::new(true);
    let mut desc = descriptor(QueueParameters::Sdma);
    desc.device_producer = true;
    let mut queue = fixture.create(desc).unwrap();
    queue.destroy().unwrap();
    drop(queue);
    let frees = fixture.state.lock().unwrap().frees;
    fixture.state.lock().unwrap().free_errno.push_back(16);
    let vm = Shared::get_mut(&mut fixture.vm).unwrap();
    assert_eq!(vm.close().unwrap_err().kind(), ErrorKind::Busy);
    {
        let state = fixture.state.lock().unwrap();
        assert_eq!(state.doorbell_unmaps, 1);
        assert_eq!(state.frees, frees + 1);
        assert_eq!(state.buffers.len(), 1);
    }
    vm.close().unwrap();
    {
        let state = fixture.state.lock().unwrap();
        assert_eq!(state.doorbell_unmaps, 1);
        assert_eq!(state.frees, frees + 2);
    }
    fixture.assert_released();
}

#[test]
fn busy_queue_keeps_transport_and_native_owner_until_progress_catches_up() {
    let fixture = Fixture::new(true);
    let mut queue = fixture.create(descriptor(QueueParameters::Sdma)).unwrap();
    queue.backing[1]
        .as_mut()
        .unwrap()
        .write_bytes(8, &4096u64.to_ne_bytes())
        .unwrap();
    assert_eq!(queue.progress().unwrap(), (0, 4096));
    assert_eq!(queue.destroy().unwrap_err().kind(), ErrorKind::Busy);
    assert!(!queue.destroying);
    assert!(queue.info().is_ok());
    assert_eq!(fixture.state.lock().unwrap().destroys, 0);
    queue.backing[1]
        .as_mut()
        .unwrap()
        .write_bytes(0, &4096u64.to_ne_bytes())
        .unwrap();
    queue.destroy().unwrap();
    fixture.assert_released();
}

#[test]
fn pm4_progress_expands_a_ring_relative_read_index_after_multiple_wraps() {
    let fixture = Fixture::new(true);
    let mut queue = fixture.create(descriptor(QueueParameters::Pm4)).unwrap();
    let capacity = u64::from(PM4_RING_SIZE / 4);
    let producer = 3 * capacity + 37;
    let read_offset = queue.read_offset;
    let write_offset = queue.write_offset;
    queue.backing[1]
        .as_mut()
        .unwrap()
        .write_bytes(read_offset, &37_u64.to_ne_bytes())
        .unwrap();
    queue.backing[1]
        .as_mut()
        .unwrap()
        .write_bytes(write_offset, &producer.to_ne_bytes())
        .unwrap();
    assert_eq!(queue.progress().unwrap(), (producer, producer));
    queue.destroy().unwrap();
    fixture.assert_released();
}
