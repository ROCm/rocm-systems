//! The KFD UAPI records used by memory, queues, and native loss reporting.
//!
//! These layouts follow the vendored `ROCm` Systems KFD UAPI in
//! `runtimes/api-headers/include/uapi/linux/kfd_ioctl.h`. Rust unit tests pin
//! selected record sizes and offsets. The queue path requires UAPI 1.17's
//! extended create record. Keep pointer-bearing records here as integers: only
//! `sys` constructs their pointers and keeps referenced storage alive for the
//! complete synchronous ioctl.

use std::mem::{offset_of, size_of};

/// Linux's generic ioctl encoding is shared by the supported 64-bit KFD hosts.
/// The payload size is part of the request, so changing a record changes its ABI.
const fn request(direction: u32, number: u32, size: u32) -> u64 {
    ((direction << 30) | (size << 16) | ((b'K' as u32) << 8) | number) as u64
}

pub(super) const GET_VERSION: u64 = request(2, 0x01, 8);
// Since UAPI 1.17, CREATE includes an SDMA engine ordinal. The final word
// became metadata_ring_size in 1.22; we keep it zero on every supported kernel.
pub(super) const CREATE_QUEUE: u64 = request(3, 0x02, 96);
pub(super) const DESTROY_QUEUE: u64 = request(3, 0x03, 8);
pub(super) const GET_CLOCK_COUNTERS: u64 = request(3, 0x05, 40);
pub(super) const GET_AVAILABLE_MEMORY: u64 = request(3, 0x23, 16);
pub(super) const UPDATE_QUEUE: u64 = request(1, 0x07, 24);
pub(super) const CREATE_EVENT: u64 = request(3, 0x08, 32);
pub(super) const DESTROY_EVENT: u64 = request(1, 0x09, 8);
pub(super) const WAIT_EVENTS: u64 = request(3, 0x0c, 24);
pub(super) const SET_SCRATCH_BACKING_VA: u64 = request(3, 0x11, 16);
pub(super) const SET_TRAP_HANDLER: u64 = request(1, 0x13, 24);
pub(super) const GET_APERTURES: u64 = request(3, 0x14, 16);
pub(super) const ACQUIRE_VM: u64 = request(1, 0x15, 8);
pub(super) const RUNTIME_ENABLE: u64 = request(3, 0x25, 16);
pub(super) const CREATE_PROCESS: u64 = request(3, 0x27, 8);
pub(super) const ALLOC_MEMORY: u64 = request(3, 0x16, 40);
pub(super) const FREE_MEMORY: u64 = request(1, 0x17, 8);
pub(super) const MAP_MEMORY: u64 = request(3, 0x18, 24);
pub(super) const UNMAP_MEMORY: u64 = request(3, 0x19, 24);
pub(super) const SET_CU_MASK: u64 = request(1, 0x1a, 16);
pub(super) const GET_DMABUF_INFO: u64 = request(3, 0x1c, 32);
pub(super) const IMPORT_DMABUF: u64 = request(3, 0x1d, 24);
pub(super) const EXPORT_DMABUF: u64 = request(3, 0x24, 16);
pub(super) const SVM: u64 = request(3, 0x20, 24);
pub(super) const IPC_IMPORT_HANDLE: u64 = request(3, 0x80, 48);
pub(super) const IPC_EXPORT_HANDLE: u64 = request(3, 0x81, 32);
pub(super) const SPM: u64 = request(3, 0x84, 32);
pub(super) const PC_SAMPLE: u64 = request(3, 0x85, 32);

pub(super) const SVM_OP_SET_ATTR: u32 = 0;
pub(super) const SVM_OP_GET_ATTR: u32 = 1;
pub(super) const SVM_LOCATION_SYSTEM: u32 = 0;
pub(super) const SVM_LOCATION_UNDEFINED: u32 = u32::MAX;
pub(super) const SVM_ATTR_PREFERRED_LOCATION: u32 = 0;
pub(super) const SVM_ATTR_PREFETCH_LOCATION: u32 = 1;
pub(super) const SVM_ATTR_ACCESS: u32 = 2;
pub(super) const SVM_ATTR_ACCESS_IN_PLACE: u32 = 3;
pub(super) const SVM_ATTR_NO_ACCESS: u32 = 4;
pub(super) const SVM_ATTR_SET_FLAGS: u32 = 5;
pub(super) const SVM_ATTR_CLEAR_FLAGS: u32 = 6;
pub(super) const SVM_ATTR_GRANULARITY: u32 = 7;
pub(super) const SPM_OP_ACQUIRE: u32 = 0;
pub(super) const SPM_OP_RELEASE: u32 = 1;
pub(super) const SPM_OP_SET_DESTINATION: u32 = 2;
pub(super) const PC_SAMPLE_OP_QUERY_CAPABILITIES: u32 = 0;
pub(super) const PC_SAMPLE_OP_CREATE: u32 = 1;
pub(super) const PC_SAMPLE_OP_DESTROY: u32 = 2;
pub(super) const PC_SAMPLE_OP_START: u32 = 3;
pub(super) const PC_SAMPLE_OP_STOP: u32 = 4;
pub(super) const PC_SAMPLE_METHOD_HOSTTRAP: u32 = 1;
pub(super) const PC_SAMPLE_METHOD_STOCHASTIC: u32 = 2;
pub(super) const PC_SAMPLE_TYPE_TIME_US: u32 = 0;
pub(super) const PC_SAMPLE_TYPE_CLOCK_CYCLES: u32 = 1;
pub(super) const PC_SAMPLE_TYPE_INSTRUCTIONS: u32 = 2;

const IOCTL_SIZE_MAX: usize = (1 << 14) - 1;

pub(super) fn svm_request(attribute_count: usize) -> Option<u64> {
    let extra = attribute_count.checked_mul(size_of::<SvmAttribute>())?;
    let bytes = size_of::<SvmArgs>().checked_add(extra)?;
    (bytes <= IOCTL_SIZE_MAX).then(|| SVM + ((extra as u64) << 16))
}

pub(super) const VRAM: u32 = 1;
pub(super) const GTT: u32 = 1 << 1;
pub(super) const USERPTR: u32 = 1 << 2;
pub(super) const DOORBELL: u32 = 1 << 3;
pub(super) const MMIO_REMAP: u32 = 1 << 4;
pub(super) const CONTIGUOUS: u32 = 1 << 23;
pub(super) const COHERENT: u32 = 1 << 26;
pub(super) const UNCACHED: u32 = 1 << 25;
pub(super) const WRITABLE: u32 = 1 << 31;
pub(super) const EXECUTABLE: u32 = 1 << 30;
pub(super) const PUBLIC: u32 = 1 << 29;
pub(super) const NO_SUBSTITUTE: u32 = 1 << 28;
pub(super) const SIGNAL_EVENT: u32 = 0;
pub(super) const HW_EXCEPTION: u32 = 3;
pub(super) const MEMORY_EXCEPTION: u32 = 8;
pub(super) const SIGNAL_EVENT_LIMIT: u32 = 4096;
pub(super) const WAIT_COMPLETE: u32 = 0;
pub(super) const WAIT_TIMEOUT: u32 = 1;

pub(super) const SVM_FLAG_HOST_ACCESS: u32 = 0x01;
pub(super) const SVM_FLAG_COHERENT: u32 = 0x02;
pub(super) const SVM_FLAG_HIVE_LOCAL: u32 = 0x04;
pub(super) const SVM_FLAG_GPU_READ_ONLY: u32 = 0x08;
pub(super) const SVM_FLAG_GPU_EXECUTE: u32 = 0x10;
pub(super) const SVM_FLAG_GPU_READ_MOSTLY: u32 = 0x20;
pub(super) const SVM_FLAG_GPU_ALWAYS_MAPPED: u32 = 0x40;
pub(super) const SVM_FLAG_EXT_COHERENT: u32 = 0x80;
pub(super) const SVM_FLAGS: u32 = SVM_FLAG_HOST_ACCESS
    | SVM_FLAG_COHERENT
    | SVM_FLAG_HIVE_LOCAL
    | SVM_FLAG_GPU_READ_ONLY
    | SVM_FLAG_GPU_EXECUTE
    | SVM_FLAG_GPU_READ_MOSTLY
    | SVM_FLAG_GPU_ALWAYS_MAPPED
    | SVM_FLAG_EXT_COHERENT;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Version {
    pub major: u32,
    pub minor: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ClockCounters {
    pub gpu_clock_counter: u64,
    pub cpu_clock_counter: u64,
    pub system_clock_counter: u64,
    pub system_clock_frequency: u64,
    pub gpu_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct AvailableMemory {
    pub available: u64,
    pub gpu_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CreateQueue {
    pub ring_address: u64,
    pub write_pointer: u64,
    pub read_pointer: u64,
    pub doorbell_offset: u64,
    pub ring_size: u32,
    pub gpu_id: u32,
    pub queue_type: u32,
    pub percentage: u32,
    pub priority: u32,
    pub queue_id: u32,
    pub eop_address: u64,
    pub eop_size: u64,
    pub context_address: u64,
    pub context_size: u32,
    pub control_stack_size: u32,
    pub sdma_engine_id: u32,
    pub metadata_ring_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DestroyQueue {
    pub queue_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct UpdateQueue {
    pub ring_address: u64,
    pub queue_id: u32,
    pub ring_size: u32,
    pub percentage: u32,
    pub priority: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SetCuMask {
    pub queue_id: u32,
    pub count: u32,
    pub mask: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Aperture {
    pub lds_base: u64,
    pub lds_limit: u64,
    pub scratch_base: u64,
    pub scratch_limit: u64,
    pub gpuvm_base: u64,
    pub gpuvm_limit: u64,
    pub gpu_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Apertures {
    pub pointer: u64,
    pub count: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct AcquireVm {
    pub drm_fd: u32,
    pub gpu_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RuntimeEnable {
    pub r_debug: u64,
    pub mode_mask: u32,
    pub capabilities_mask: u32,
}

/// Selects a secondary KFD context on the owned endpoint descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CreateProcess {
    pub flags: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct AllocMemory {
    pub va: u64,
    pub size: u64,
    pub handle: u64,
    pub mmap_offset: u64,
    pub gpu_id: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct FreeMemory {
    pub handle: u64,
}

/// MAP and UNMAP intentionally share a layout. `success` is both an input
/// retry prefix and an output, including when the ioctl reports an error.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct MapMemory {
    pub handle: u64,
    pub devices: u64,
    pub count: u32,
    pub success: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DmaBufInfo {
    pub size: u64,
    pub metadata: u64,
    pub metadata_size: u32,
    pub gpu_id: u32,
    pub flags: u32,
    pub descriptor: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ImportDmaBuf {
    pub va: u64,
    pub handle: u64,
    pub gpu_id: u32,
    pub descriptor: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct ExportDmaBuf {
    pub handle: u64,
    pub flags: u32,
    pub descriptor: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct IpcExportHandle {
    pub handle: u64,
    pub share_handle: [u32; 4],
    pub gpu_id: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct IpcImportHandle {
    pub handle: u64,
    pub va_addr: u64,
    pub mmap_offset: u64,
    pub share_handle: [u32; 4],
    pub gpu_id: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SvmArgs {
    pub start_address: u64,
    pub size: u64,
    pub operation: u32,
    pub attribute_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct SvmAttribute {
    pub attribute_type: u32,
    pub value: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Spm {
    pub destination: u64,
    pub size: u32,
    pub operation: u32,
    pub timeout: u32,
    pub gpu_id: u32,
    pub bytes_copied: u32,
    pub has_data_loss: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct PcSampleInfo {
    pub interval: u64,
    pub interval_min: u64,
    pub interval_max: u64,
    pub flags: u64,
    pub method: u32,
    pub sample_type: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PcSample {
    pub sample_info: u64,
    pub sample_info_count: u32,
    pub operation: u32,
    pub gpu_id: u32,
    pub trace_id: u32,
    pub flags: u32,
    pub version: u32,
}

impl Default for ExportDmaBuf {
    fn default() -> Self {
        Self {
            handle: 0,
            flags: 0,
            descriptor: u32::MAX,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CreateEvent {
    pub page_offset: u64,
    pub trigger_data: u32,
    pub event_type: u32,
    pub auto_reset: u32,
    pub node_id: u32,
    pub event_id: u32,
    pub slot_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DestroyEvent {
    pub event_id: u32,
    pub pad: u32,
}

/// The kernel's event union occupies 32 bytes and has eight-byte alignment.
/// `sys` decodes the hardware- and memory-exception layouts only after a
/// successful wait on an event created with the matching type.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct EventData {
    pub payload: [u64; 4],
    pub extension: u64,
    pub event_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct WaitEvents {
    pub events: u64,
    pub count: u32,
    pub wait_all: u32,
    pub timeout: u32,
    pub result: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SetScratchBackingVa {
    pub va_address: u64,
    pub gpu_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SetTrapHandler {
    pub tba_address: u64,
    pub tma_address: u64,
    pub gpu_id: u32,
    pub pad: u32,
}

// Match the independent C layout checks, including offsets after pointer and
// union fields. Failing at compile time prevents a wrong request from reaching
// a real driver, even when its enclosing Rust code would otherwise compile.
const _: () = {
    assert!(size_of::<CreateQueue>() == 96);
    assert!(offset_of!(CreateQueue, queue_id) == 52);
    assert!(offset_of!(CreateQueue, context_address) == 72);
    assert!(offset_of!(CreateQueue, sdma_engine_id) == 88);
    assert!(size_of::<DestroyQueue>() == 8);
    assert!(size_of::<UpdateQueue>() == 24);
    assert!(offset_of!(UpdateQueue, queue_id) == 8);
    assert!(size_of::<SetCuMask>() == 16);
    assert!(offset_of!(SetCuMask, mask) == 8);
    assert!(size_of::<Version>() == 8);
    assert!(size_of::<ClockCounters>() == 40);
    assert!(offset_of!(ClockCounters, gpu_id) == 32);
    assert!(size_of::<AvailableMemory>() == 16);
    assert!(offset_of!(AvailableMemory, gpu_id) == 8);
    assert!(size_of::<Aperture>() == 56);
    assert!(offset_of!(Aperture, gpu_id) == 48);
    assert!(size_of::<Apertures>() == 16);
    assert!(offset_of!(Apertures, count) == 8);
    assert!(size_of::<AcquireVm>() == 8);
    assert!(size_of::<RuntimeEnable>() == 16);
    assert!(offset_of!(RuntimeEnable, mode_mask) == 8);
    assert!(offset_of!(RuntimeEnable, capabilities_mask) == 12);
    assert!(size_of::<CreateProcess>() == 8);
    assert!(size_of::<AllocMemory>() == 40);
    assert!(offset_of!(AllocMemory, gpu_id) == 32);
    assert!(size_of::<FreeMemory>() == 8);
    assert!(size_of::<MapMemory>() == 24);
    assert!(offset_of!(MapMemory, success) == 20);
    assert!(size_of::<DmaBufInfo>() == 32);
    assert!(offset_of!(DmaBufInfo, metadata_size) == 16);
    assert!(offset_of!(DmaBufInfo, descriptor) == 28);
    assert!(size_of::<ImportDmaBuf>() == 24);
    assert!(offset_of!(ImportDmaBuf, descriptor) == 20);
    assert!(size_of::<ExportDmaBuf>() == 16);
    assert!(offset_of!(ExportDmaBuf, descriptor) == 12);
    assert!(size_of::<IpcExportHandle>() == 32);
    assert!(offset_of!(IpcExportHandle, share_handle) == 8);
    assert!(offset_of!(IpcExportHandle, gpu_id) == 24);
    assert!(size_of::<IpcImportHandle>() == 48);
    assert!(offset_of!(IpcImportHandle, share_handle) == 24);
    assert!(offset_of!(IpcImportHandle, gpu_id) == 40);
    assert!(size_of::<SvmArgs>() == 24);
    assert!(offset_of!(SvmArgs, operation) == 16);
    assert!(offset_of!(SvmArgs, attribute_count) == 20);
    assert!(size_of::<SvmAttribute>() == 8);
    assert!(size_of::<Spm>() == 32);
    assert!(offset_of!(Spm, operation) == 12);
    assert!(offset_of!(Spm, timeout) == 16);
    assert!(offset_of!(Spm, gpu_id) == 20);
    assert!(offset_of!(Spm, bytes_copied) == 24);
    assert!(offset_of!(Spm, has_data_loss) == 28);
    assert!(size_of::<PcSampleInfo>() == 40);
    assert!(offset_of!(PcSampleInfo, interval_min) == 8);
    assert!(offset_of!(PcSampleInfo, flags) == 24);
    assert!(offset_of!(PcSampleInfo, method) == 32);
    assert!(offset_of!(PcSampleInfo, sample_type) == 36);
    assert!(size_of::<PcSample>() == 32);
    assert!(offset_of!(PcSample, sample_info_count) == 8);
    assert!(offset_of!(PcSample, operation) == 12);
    assert!(offset_of!(PcSample, gpu_id) == 16);
    assert!(offset_of!(PcSample, trace_id) == 20);
    assert!(offset_of!(PcSample, flags) == 24);
    assert!(offset_of!(PcSample, version) == 28);
    assert!(size_of::<CreateEvent>() == 32);
    assert!(offset_of!(CreateEvent, event_id) == 24);
    assert!(size_of::<DestroyEvent>() == 8);
    assert!(size_of::<EventData>() == 48);
    assert!(offset_of!(EventData, event_id) == 40);
    assert!(size_of::<WaitEvents>() == 24);
    assert!(offset_of!(WaitEvents, result) == 20);
    assert!(size_of::<SetScratchBackingVa>() == 16);
    assert!(offset_of!(SetScratchBackingVa, gpu_id) == 8);
    assert!(size_of::<SetTrapHandler>() == 24);
    assert!(offset_of!(SetTrapHandler, tma_address) == 8);
    assert!(offset_of!(SetTrapHandler, gpu_id) == 16);
};
