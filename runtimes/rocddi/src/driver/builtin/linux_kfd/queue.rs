//! Concrete KFD queue ownership. Every ring and pointer page is a separate BO:
//! `CREATE_QUEUE` checks their GPU mapping extents, so suballocating from a larger
//! mapping would violate the kernel contract. Compute queues also retain EOP
//! and context-save storage; the process/device VM retains its doorbell slice.

use crate::host_storage::{Buffer, Owned, Shared};
use std::mem::{offset_of, size_of};
use std::sync::Mutex;

use super::memory::{BufferKind, DeviceVm, KfdAllocation, error, native_error};
use super::{sys, sysfs, uapi, util};
use crate::memory::AllocationDesc;
use crate::memory::DeviceAccess;
use crate::queue::{
    QueueAccessWidth, QueueErrorEvent, QueueParameters, QueuePriority, QueueProducerMode,
    QueueRequest, QueueScratch, QueueTransport,
};
use crate::session::SessionLifetime;

use crate::{Error, ErrorKind};

const DOORBELL_SLICE: usize = 8192;
const PM4_RING_SIZE: u32 = 4096;
const PM4_READ_OFFSET: usize = 0;
const SDMA_READ_OFFSET: usize = 0;
const SDMA_WRITE_OFFSET: usize = 8;
const CWSR_ALIGNMENT: usize = 2 * 1024 * 1024;
const SCRATCH_ALIGNMENT: u64 = 256;
const MAX_PRIVATE_SEGMENT_BYTES: u32 = 262_128;
const GFX1201_SCRATCH_RESOURCE_WORD3: u32 =
    4 | (5 << 3) | (6 << 6) | (7 << 9) | (0x14 << 12) | (1 << 23) | (2 << 28);

#[allow(dead_code)]
#[repr(C)]
struct AqlQueueHeader {
    queue_type: u32,
    features: u32,
    base_address: u64,
    doorbell_signal: u64,
    size: u32,
    reserved: u32,
    id: u64,
}

#[allow(dead_code)]
#[repr(C, align(64))]
struct AqlControlLayout {
    queue_header: AqlQueueHeader,
    caps: u32,
    reserved1: [u32; 3],
    write_dispatch_id: u64,
    group_segment_aperture_base_hi: u32,
    private_segment_aperture_base_hi: u32,
    max_cu_id: u32,
    max_wave_id: u32,
    max_legacy_doorbell_dispatch_id_plus_1: u64,
    legacy_doorbell_lock: u32,
    reserved2: [u32; 9],
    read_dispatch_id: u64,
    read_dispatch_id_field_base_byte_offset: u32,
    compute_tmpring_size: u32,
    scratch_resource_descriptor: [u32; 4],
    scratch_backing_memory_location: u64,
    scratch_backing_memory_byte_size: u64,
    scratch_wave64_lane_byte_size: u32,
    queue_properties: u32,
    scratch_max_use_index: u64,
    queue_inactive_signal: u64,
    alt_scratch_max_use_index: u64,
    alt_scratch_resource_descriptor: [u32; 4],
    alt_scratch_backing_memory_location: u64,
    alt_scratch_dispatch_limit: [u32; 3],
    alt_scratch_wave64_lane_byte_size: u32,
    alt_compute_tmpring_size: u32,
    reserved5: u32,
}

const AQL_CONTROL_SIZE: usize = size_of::<AqlControlLayout>();
const AQL_WRITE_OFFSET: usize = offset_of!(AqlControlLayout, write_dispatch_id);
const AQL_READ_OFFSET: usize = offset_of!(AqlControlLayout, read_dispatch_id);

const _: () = assert!(size_of::<AqlQueueHeader>() == 40);
const _: () = assert!(AQL_CONTROL_SIZE == 256);
const _: () = assert!(AQL_WRITE_OFFSET == 56);
const _: () = assert!(AQL_READ_OFFSET == 128);

/// KFD has one doorbell VMA per device/process. Repeated discovery and concurrent
/// creation therefore share this mapping through the exact acquired `DeviceVm`.
/// A device-producer request upgrades that retained CPU mapping with a KFD
/// doorbell BO at the same VA and maps it into the owning GPU VM.
#[derive(Default)]
pub(super) struct Doorbells {
    mapping: Mutex<Option<DoorbellMapping>>,
}

struct DoorbellMapping {
    storage: Option<sys::Reservation>,
    offset: u64,
    gpu_id: u32,
    handle: Option<u64>,
    gpu_mapping: DoorbellGpuMapping,
    peers: Buffer<DoorbellPeer>,
    uncertain: bool,
}

/// Peer VM retaining its device-visible mapping of shared doorbell storage.
struct DoorbellPeer {
    vm: Shared<DeviceVm>,
    gpu_mapping: DoorbellGpuMapping,
    uncertain: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DoorbellGpuMapping {
    Unmapped,
    Mapping(u32),
    Mapped,
    Unmapping(u32),
}

/// CPU and optional GPU addresses for one allocated doorbell slot.
#[derive(Clone, Copy)]
struct DoorbellAddresses {
    host: usize,
    device: Option<u64>,
}

fn doorbell_error(operation: &'static str, source: std::io::Error) -> Error {
    if matches!(source.raw_os_error(), Some(22 | 25 | 95)) {
        Error::NativeOperation {
            kind: ErrorKind::Unsupported,
            operation,
            source,
        }
    } else {
        native_error(operation, source)
    }
}

impl DoorbellPeer {
    fn transfer(&mut self, handle: u64, map: bool) -> Result<(), Error> {
        let (DoorbellGpuMapping::Mapping(completed) | DoorbellGpuMapping::Unmapping(completed)) =
            &mut self.gpu_mapping
        else {
            return Err(error(
                ErrorKind::Internal,
                "peer doorbell mapping has an invalid transfer state",
            ));
        };
        let result = self
            .vm
            .kfd()
            .transfer(handle, &[self.vm.gpu_id()], completed, map);
        if result.as_ref().err().is_some_and(|source| {
            source.kind() == std::io::ErrorKind::InvalidData || source.raw_os_error() == Some(14)
        }) {
            self.uncertain = true;
        }
        result.map_err(|source| {
            doorbell_error(
                if map {
                    "AMDKFD_IOC_MAP_MEMORY_TO_GPU for peer doorbells"
                } else {
                    "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU for peer doorbells"
                },
                source,
            )
        })
    }

    fn enable(&mut self, handle: u64) -> Result<(), Error> {
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "peer doorbell mapping outcome is uncertain",
            ));
        }
        if matches!(self.gpu_mapping, DoorbellGpuMapping::Mapped) {
            return Ok(());
        }
        if !matches!(self.gpu_mapping, DoorbellGpuMapping::Mapping(_)) {
            return Err(error(
                ErrorKind::DriverContract,
                "peer doorbell mapping cleanup is incomplete",
            ));
        }
        self.transfer(handle, true)?;
        self.gpu_mapping = DoorbellGpuMapping::Mapped;
        self.vm.check()
    }

    fn close(&mut self, handle: u64) -> Result<(), Error> {
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "peer doorbell mapping outcome is uncertain",
            ));
        }
        match self.gpu_mapping {
            DoorbellGpuMapping::Mapping(0) => {
                self.gpu_mapping = DoorbellGpuMapping::Unmapped;
            }
            DoorbellGpuMapping::Mapping(1) | DoorbellGpuMapping::Mapped => {
                self.gpu_mapping = DoorbellGpuMapping::Unmapping(0);
            }
            DoorbellGpuMapping::Mapping(_) => {
                self.uncertain = true;
                return Err(error(
                    ErrorKind::DriverContract,
                    "peer doorbell mapping returned an invalid completion count",
                ));
            }
            DoorbellGpuMapping::Unmapped | DoorbellGpuMapping::Unmapping(_) => {}
        }
        if matches!(self.gpu_mapping, DoorbellGpuMapping::Unmapping(_)) {
            self.transfer(handle, false)?;
            self.gpu_mapping = DoorbellGpuMapping::Unmapped;
        }
        Ok(())
    }
}

impl DoorbellMapping {
    fn address(&self) -> Result<usize, Error> {
        self.storage
            .as_ref()
            .map(sys::Reservation::address)
            .ok_or_else(|| error(ErrorKind::Internal, "doorbell mapping has no storage"))
    }

    fn transfer(&mut self, kfd: &sys::Kfd, map: bool) -> Result<(), Error> {
        let handle = self
            .handle
            .ok_or_else(|| error(ErrorKind::Internal, "doorbell mapping has no handle"))?;
        let (DoorbellGpuMapping::Mapping(completed) | DoorbellGpuMapping::Unmapping(completed)) =
            &mut self.gpu_mapping
        else {
            return Err(error(
                ErrorKind::Internal,
                "doorbell mapping has an invalid transfer state",
            ));
        };
        let result = kfd.transfer(handle, &[self.gpu_id], completed, map);
        if result.as_ref().err().is_some_and(|source| {
            source.kind() == std::io::ErrorKind::InvalidData || source.raw_os_error() == Some(14)
        }) {
            self.uncertain = true;
        }
        result.map_err(|source| {
            doorbell_error(
                if map {
                    "AMDKFD_IOC_MAP_MEMORY_TO_GPU for doorbells"
                } else {
                    "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU for doorbells"
                },
                source,
            )
        })
    }

    fn rollback_device(&mut self, kfd: &sys::Kfd) -> Result<(), Error> {
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "doorbell GPU mapping outcome is uncertain",
            ));
        }
        match self.gpu_mapping {
            DoorbellGpuMapping::Mapping(0) => {
                self.gpu_mapping = DoorbellGpuMapping::Unmapped;
            }
            DoorbellGpuMapping::Mapping(1) | DoorbellGpuMapping::Mapped => {
                self.gpu_mapping = DoorbellGpuMapping::Unmapping(0);
            }
            DoorbellGpuMapping::Mapping(_) => {
                self.uncertain = true;
                return Err(error(
                    ErrorKind::DriverContract,
                    "doorbell mapping returned an invalid completion count",
                ));
            }
            DoorbellGpuMapping::Unmapped | DoorbellGpuMapping::Unmapping(_) => {}
        }
        if matches!(self.gpu_mapping, DoorbellGpuMapping::Unmapping(_)) {
            self.transfer(kfd, false)?;
            self.gpu_mapping = DoorbellGpuMapping::Unmapped;
        }
        if let Some(handle) = self.handle {
            kfd.free(handle)
                .map_err(|source| native_error("KFD doorbell BO free", source))?;
            self.handle = None;
        }
        Ok(())
    }

    fn enable_device(&mut self, vm: &DeviceVm) -> Result<u64, Error> {
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "doorbell GPU mapping outcome is uncertain",
            ));
        }
        if matches!(self.gpu_mapping, DoorbellGpuMapping::Mapped) {
            return Ok(self.address()? as u64);
        }
        if matches!(self.gpu_mapping, DoorbellGpuMapping::Mapping(_)) {
            self.transfer(vm.kfd(), true)?;
            self.gpu_mapping = DoorbellGpuMapping::Mapped;
            return Ok(self.address()? as u64);
        }
        if self.handle.is_some() || !matches!(self.gpu_mapping, DoorbellGpuMapping::Unmapped) {
            return Err(error(
                ErrorKind::DriverContract,
                "doorbell GPU mapping cleanup is incomplete",
            ));
        }
        let mut args = uapi::AllocMemory {
            va: self.address()? as u64,
            size: DOORBELL_SLICE as u64,
            gpu_id: vm.gpu_id(),
            flags: uapi::DOORBELL | uapi::WRITABLE | uapi::COHERENT | uapi::NO_SUBSTITUTE,
            ..uapi::AllocMemory::default()
        };
        let result = vm.kfd().allocate_doorbells(&mut args);
        self.handle = (args.handle != 0).then_some(args.handle);
        self.uncertain = result
            .as_ref()
            .err()
            .is_some_and(|source| source.raw_os_error() == Some(14));
        if let Err(source) = result {
            let failure = doorbell_error("KFD doorbell BO allocation", source);
            if !self.uncertain {
                self.rollback_device(vm.kfd())?;
            }
            return Err(failure);
        }
        if self.handle.is_none() {
            self.uncertain = true;
            return Err(error(
                ErrorKind::DriverContract,
                "KFD doorbell allocation succeeded without a handle",
            ));
        }
        self.gpu_mapping = DoorbellGpuMapping::Mapping(0);
        if let Err(failure) = self.transfer(vm.kfd(), true) {
            if !self.uncertain {
                self.rollback_device(vm.kfd())?;
            }
            return Err(failure);
        }
        self.gpu_mapping = DoorbellGpuMapping::Mapped;
        vm.check()?;
        Ok(self.address()? as u64)
    }

    fn enable_peer(
        &mut self,
        owner: &Shared<DeviceVm>,
        peer: Shared<DeviceVm>,
    ) -> Result<u64, Error> {
        if Shared::ptr_eq(owner, &peer) {
            return self.enable_device(owner);
        }
        if !owner.shares_kfd(&peer) {
            return Err(error(
                ErrorKind::InvalidArgument,
                "queue producer must share one KFD session",
            ));
        }
        let address = self.address()? as u64;
        let size = self
            .storage
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "doorbell mapping has no storage"))?
            .usable_size() as u64;
        let end = address
            .checked_add(size.saturating_sub(1))
            .ok_or_else(|| error(ErrorKind::Internal, "doorbell address overflow"))?;
        let (base, limit) = peer.address_range();
        if address < base || end > limit {
            return Err(error(
                ErrorKind::Unsupported,
                "doorbell mapping is outside the producer GPU address range",
            ));
        }
        let handle = self
            .handle
            .ok_or_else(|| error(ErrorKind::Unsupported, "device doorbell mapping is absent"))?;
        if let Some(existing) = self
            .peers
            .iter_mut()
            .find(|existing| Shared::ptr_eq(&existing.vm, &peer))
        {
            existing.enable(handle)?;
            return Ok(address);
        }
        self.peers.try_reserve(1)?;
        self.peers.try_push(DoorbellPeer {
            vm: peer,
            gpu_mapping: DoorbellGpuMapping::Mapping(0),
            uncertain: false,
        })?;
        let index = self.peers.len() - 1;
        if let Err(failure) = self.peers[index].enable(handle) {
            if !self.peers[index].uncertain {
                self.peers[index].close(handle)?;
                let _ = self.peers.pop();
            }
            return Err(failure);
        }
        Ok(address)
    }

    fn close(&mut self, kfd: &sys::Kfd) -> Result<(), Error> {
        if let Some(handle) = self.handle {
            for peer in self.peers.iter_mut().rev() {
                peer.close(handle)?;
            }
            self.peers.clear();
        }
        self.rollback_device(kfd)?;
        if let Some(storage) = &mut self.storage {
            storage
                .release()
                .map_err(|source| native_error("doorbell munmap", source))?;
        }
        self.storage = None;
        Ok(())
    }
}

impl Drop for DoorbellMapping {
    fn drop(&mut self) {
        if self.handle.is_some()
            || !matches!(self.gpu_mapping, DoorbellGpuMapping::Unmapped)
            || self.uncertain
        {
            if let Some(storage) = self.storage.take() {
                std::mem::forget(storage);
            }
            for peer in &self.peers {
                if peer.uncertain || !matches!(peer.gpu_mapping, DoorbellGpuMapping::Unmapped) {
                    std::mem::forget(peer.vm.clone());
                }
            }
        }
    }
}

impl Doorbells {
    pub(super) fn close(&mut self, kfd: &sys::Kfd) -> Result<(), Error> {
        let mapping = self
            .mapping
            .get_mut()
            .map_err(|_| error(ErrorKind::Internal, "doorbell mapping lock was poisoned"))?;
        if let Some(mapping) = mapping {
            mapping.close(kfd)?;
        }
        *mapping = None;
        Ok(())
    }

    fn addresses(
        &self,
        vm: &DeviceVm,
        offset: u64,
        device: bool,
    ) -> Result<DoorbellAddresses, Error> {
        // A fork child may inherit this lock held by a thread that no longer
        // exists. Check the endpoint before even reading initialization state.
        vm.kfd()
            .check_process()
            .map_err(|source| native_error("KFD doorbell mapping", source))?;
        let within = usize::try_from(offset & (DOORBELL_SLICE as u64 - 1)).map_err(|_| {
            error(
                ErrorKind::DriverContract,
                "doorbell offset exceeds host width",
            )
        })?;
        let base = offset - within as u64;
        if within % 8 != 0 || within + 8 > DOORBELL_SLICE {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD returned an invalid doorbell offset",
            ));
        }
        let mut mapping = self.mapping.lock().map_err(|_| {
            error(
                ErrorKind::Internal,
                "KFD doorbell initialization lock was poisoned",
            )
        })?;
        if mapping.is_none() {
            let storage = vm
                .kfd()
                .map_doorbells(base, DOORBELL_SLICE, vm.address_range())
                .map_err(|source| native_error("KFD doorbell mmap", source))?;
            *mapping = Some(DoorbellMapping {
                storage: Some(storage),
                offset: base,
                gpu_id: vm.gpu_id(),
                handle: None,
                gpu_mapping: DoorbellGpuMapping::Unmapped,
                peers: Buffer::new(vm.allocator()),
                uncertain: false,
            });
        }
        let mapping = mapping.as_mut().ok_or_else(|| {
            error(
                ErrorKind::Internal,
                "KFD doorbell mapping was not published",
            )
        })?;
        if mapping.offset != base {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD changed a retained doorbell slice",
            ));
        }
        let host = mapping.address()? + within;
        let device = if device {
            Some(
                mapping
                    .enable_device(vm)?
                    .checked_add(within as u64)
                    .ok_or_else(|| error(ErrorKind::Internal, "doorbell address overflow"))?,
            )
        } else {
            None
        };
        Ok(DoorbellAddresses { host, device })
    }

    fn peer_address(
        &self,
        owner: &Shared<DeviceVm>,
        peer: Shared<DeviceVm>,
        offset: u64,
    ) -> Result<u64, Error> {
        owner
            .kfd()
            .check_process()
            .map_err(|source| native_error("KFD peer doorbell mapping", source))?;
        let within = usize::try_from(offset & (DOORBELL_SLICE as u64 - 1)).map_err(|_| {
            error(
                ErrorKind::DriverContract,
                "doorbell offset exceeds host width",
            )
        })?;
        let base = offset - within as u64;
        if within % 8 != 0 || within + 8 > DOORBELL_SLICE {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD returned an invalid doorbell offset",
            ));
        }
        let mut mapping = self.mapping.lock().map_err(|_| {
            error(
                ErrorKind::Internal,
                "KFD doorbell initialization lock was poisoned",
            )
        })?;
        let mapping = mapping.as_mut().ok_or_else(|| {
            error(
                ErrorKind::DriverContract,
                "device doorbell mapping was not created",
            )
        })?;
        if mapping.offset != base {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD changed a retained doorbell slice",
            ));
        }
        mapping
            .enable_peer(owner, peer)?
            .checked_add(within as u64)
            .ok_or_else(|| error(ErrorKind::Internal, "doorbell address overflow"))
    }
}

/// Computed context-save allocation requirements for a compute queue.
struct ComputeStorage {
    context_size: u32,
    control_stack_size: u32,
    debug_size: u32,
    xcc_count: u32,
    total_size: u64,
}

/// Fully validated native queue request used by the acquisition path.
struct Request {
    ring_size: u32,
    queue_type: u32,
    priority: u32,
    device_producer: bool,
    compute: Option<ComputeStorage>,
    aql: Option<AqlControl>,
    pm4: Option<Pm4Control>,
}

/// A peer VM's mappings of the queue ring and control/index allocation.
struct QueuePeerMapping {
    vm: Shared<DeviceVm>,
    handles: [u64; 2],
    mappings: [DoorbellGpuMapping; 2],
    uncertain: bool,
}

impl QueuePeerMapping {
    fn new(vm: Shared<DeviceVm>, handles: [u64; 2]) -> Self {
        Self {
            vm,
            handles,
            mappings: [DoorbellGpuMapping::Unmapped; 2],
            uncertain: false,
        }
    }

    fn transfer(&mut self, index: usize, map: bool) -> Result<(), Error> {
        let (DoorbellGpuMapping::Mapping(completed) | DoorbellGpuMapping::Unmapping(completed)) =
            &mut self.mappings[index]
        else {
            return Err(error(
                ErrorKind::Internal,
                "queue peer mapping has an invalid transfer state",
            ));
        };
        let result =
            self.vm
                .kfd()
                .transfer(self.handles[index], &[self.vm.gpu_id()], completed, map);
        if result.as_ref().err().is_some_and(|source| {
            source.kind() == std::io::ErrorKind::InvalidData || source.raw_os_error() == Some(14)
        }) {
            self.uncertain = true;
        }
        result.map_err(|source| {
            let operation = if map {
                "AMDKFD_IOC_MAP_MEMORY_TO_GPU for a queue producer"
            } else {
                "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU for a queue producer"
            };
            if matches!(source.raw_os_error(), Some(22 | 25 | 95)) {
                Error::NativeOperation {
                    kind: ErrorKind::Unsupported,
                    operation,
                    source,
                }
            } else {
                native_error(operation, source)
            }
        })
    }

    fn enable(&mut self) -> Result<(), Error> {
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "queue peer mapping outcome is uncertain",
            ));
        }
        for index in 0..self.mappings.len() {
            match self.mappings[index] {
                DoorbellGpuMapping::Unmapped => {
                    self.mappings[index] = DoorbellGpuMapping::Mapping(0);
                }
                DoorbellGpuMapping::Mapping(_) => {}
                DoorbellGpuMapping::Mapped => continue,
                DoorbellGpuMapping::Unmapping(_) => {
                    return Err(error(
                        ErrorKind::DriverContract,
                        "queue peer mapping cleanup is incomplete",
                    ));
                }
            }
            self.transfer(index, true)?;
            self.mappings[index] = DoorbellGpuMapping::Mapped;
        }
        self.vm.check()
    }

    fn close(&mut self) -> Result<(), Error> {
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "queue peer mapping outcome is uncertain",
            ));
        }
        for index in (0..self.mappings.len()).rev() {
            match self.mappings[index] {
                DoorbellGpuMapping::Mapping(0) => {
                    self.mappings[index] = DoorbellGpuMapping::Unmapped;
                }
                DoorbellGpuMapping::Mapping(1) | DoorbellGpuMapping::Mapped => {
                    self.mappings[index] = DoorbellGpuMapping::Unmapping(0);
                }
                DoorbellGpuMapping::Mapping(_) => {
                    self.uncertain = true;
                    return Err(error(
                        ErrorKind::DriverContract,
                        "queue peer mapping returned an invalid completion count",
                    ));
                }
                DoorbellGpuMapping::Unmapping(_) => {}
                DoorbellGpuMapping::Unmapped => continue,
            }
            if matches!(self.mappings[index], DoorbellGpuMapping::Unmapping(_)) {
                self.transfer(index, false)?;
                self.mappings[index] = DoorbellGpuMapping::Unmapped;
            }
        }
        Ok(())
    }

    fn is_mapped(&self) -> bool {
        !self.uncertain
            && self
                .mappings
                .iter()
                .all(|mapping| matches!(mapping, DoorbellGpuMapping::Mapped))
    }
}

/// Hardware scratch registers derived from one validated scratch allocation.
#[derive(Clone, Copy)]
struct ScratchControl {
    tmpring_size: u32,
    resource: [u32; 4],
    backing_address: u64,
    wave64_lane_bytes: u32,
}

/// AQL-specific queue metadata needed for creation and later scratch updates.
#[derive(Clone, Copy)]
struct AqlControl {
    producer_mode: QueueProducerMode,
    properties: sysfs::NativeQueueProperties,
    inactive_signal: Option<u64>,
    error_event: Option<QueueErrorEvent>,
    scratch: Option<ScratchControl>,
}

/// PM4-specific control-page offsets and wrapped read-index mask.
#[derive(Clone, Copy)]
struct Pm4Control {
    write_offset: usize,
    error_offset: usize,
    read_mask: u64,
}

fn validate_pm4(
    native: &sysfs::NativeNode,
    desc: QueueRequest,
    ring_size: u32,
) -> Result<Pm4Control, Error> {
    if desc.device_producer {
        return Err(error(
            ErrorKind::Unsupported,
            "native PM4 is qualified only for a host producer",
        ));
    }
    if desc.priority != QueuePriority::Normal {
        return Err(error(
            ErrorKind::Unsupported,
            "native PM4 is qualified only at normal priority",
        ));
    }
    if ring_size != PM4_RING_SIZE {
        return Err(error(
            ErrorKind::Unsupported,
            "native PM4 requires a 4096-byte ring",
        ));
    }
    pm4_control(native).ok_or_else(|| {
        error(
            ErrorKind::Unsupported,
            "native PM4 context is not qualified",
        )
    })
}

fn validate_aql(
    properties: sysfs::NativeQueueProperties,
    desc: QueueRequest,
) -> Result<AqlControl, Error> {
    let QueueParameters::Aql {
        producer_mode,
        inactive_signal,
        error_event,
        scratch,
    } = desc.parameters
    else {
        return Err(error(
            ErrorKind::InvalidArgument,
            "AQL validation requires AQL queue parameters",
        ));
    };
    if inactive_signal == Some(0) {
        return Err(error(
            ErrorKind::InvalidArgument,
            "AQL inactive signal handle is null",
        ));
    }
    if error_event.is_some_and(|event| {
        event.payload_address == 0
            || event.native_event_token == 0
            || event.native_event_token > u64::from(u32::MAX)
    }) {
        return Err(error(
            ErrorKind::InvalidArgument,
            "AQL error event is incomplete",
        ));
    }
    Ok(AqlControl {
        producer_mode,
        properties,
        inactive_signal,
        error_event,
        scratch: validate_scratch(properties, scratch)?,
    })
}

fn validate_sdma(
    properties: sysfs::NativeQueueProperties,
    desc: QueueRequest,
) -> Result<(), Error> {
    if !properties.sdma_qualified {
        return Err(error(
            ErrorKind::Unsupported,
            "native SDMA packet ABI is not qualified",
        ));
    }
    if desc.priority != QueuePriority::Normal {
        return Err(error(
            ErrorKind::Unsupported,
            "KFD SDMA does not implement this scheduling priority",
        ));
    }
    if properties.sdma_engines == 0 {
        return Err(error(
            ErrorKind::Unsupported,
            "device reports no general-purpose SDMA engines",
        ));
    }
    Ok(())
}

impl Request {
    fn validate(native: &sysfs::NativeNode, desc: QueueRequest) -> Result<Self, Error> {
        let properties = native.queues;
        // These generations share eight-byte doorbells, the pointer protocol,
        // and the 32-waves-per-CU debugger tail. Other generations need their
        // own checked sizing rules before they can use this allocation path.
        if native.gpu_id == 0 || !(100_100..=120_001).contains(&properties.gfx_target) {
            return Err(error(
                ErrorKind::Unsupported,
                "KFD queue backing is supported on GFX10.1 through GFX12.0",
            ));
        }
        if util::page_size().map_err(|source| native_error("queue page size", source))? != 4096 {
            return Err(error(
                ErrorKind::Unsupported,
                "KFD queue allocation requires a checked 4 KiB host-page layout",
            ));
        }
        let ring_size = u32::try_from(desc.ring_size_bytes).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "queue ring exceeds the KFD size field",
            )
        })?;
        if ring_size < 1024 || !ring_size.is_power_of_two() {
            return Err(error(
                ErrorKind::InvalidArgument,
                "KFD ring size must be a power of two of at least 1024 bytes",
            ));
        }
        let (queue_type, aql, pm4) = match desc.parameters {
            QueueParameters::Pm4 => (0, None, Some(validate_pm4(native, desc, ring_size)?)),
            QueueParameters::Aql { .. } if supports_aql(native) => {
                (2, Some(validate_aql(properties, desc)?), None)
            }
            QueueParameters::Aql { .. } => {
                return Err(error(
                    ErrorKind::Unsupported,
                    "native AQL context is not qualified",
                ));
            }
            QueueParameters::Sdma => {
                validate_sdma(properties, desc)?;
                (1, None, None)
            }
        };
        let compute = if queue_type == 0 || queue_type == 2 {
            Some(compute_storage(properties)?)
        } else {
            None
        };
        Ok(Self {
            ring_size,
            queue_type,
            device_producer: desc.device_producer,
            compute,
            aql,
            pm4,
            priority: match desc.priority {
                QueuePriority::Low => 0,
                QueuePriority::Normal => 7,
                QueuePriority::High => 15,
            },
        })
    }
}

fn validate_scratch(
    properties: sysfs::NativeQueueProperties,
    scratch: Option<QueueScratch>,
) -> Result<Option<ScratchControl>, Error> {
    let Some(scratch) = scratch else {
        return Ok(None);
    };
    if properties.gfx_target != 120_001
        || properties.wavefront_size != 32
        || properties.xcc_count != 1
    {
        return Err(error(
            ErrorKind::Unsupported,
            "fixed AQL scratch is qualified only for GFX1201 wave32",
        ));
    }
    if scratch.device_address % SCRATCH_ALIGNMENT != 0
        || scratch.byte_length % SCRATCH_ALIGNMENT != 0
        || scratch.maximum_private_segment_byte_length == 0
        || scratch.maximum_private_segment_byte_length > MAX_PRIVATE_SEGMENT_BYTES
        || scratch.maximum_private_segment_byte_length % 8 != 0
        || scratch.maximum_wave_count == 0
    {
        return Err(error(
            ErrorKind::InvalidArgument,
            "scratch range or limits do not meet GFX1201 alignment",
        ));
    }
    let shader_engines = properties
        .shader_engine_count_per_xcc
        .checked_mul(properties.xcc_count)
        .filter(|count| *count != 0)
        .ok_or_else(|| error(ErrorKind::InvalidData, "invalid shader-engine geometry"))?;
    let maximum_waves = properties
        .compute_units
        .checked_mul(properties.maximum_scratch_wave_count_per_compute_unit)
        .ok_or_else(|| error(ErrorKind::InvalidData, "scratch-wave limit overflows"))?;
    if scratch.maximum_wave_count > maximum_waves
        || scratch.maximum_wave_count % shader_engines != 0
    {
        return Err(error(
            ErrorKind::InvalidArgument,
            "scratch wave count exceeds or does not divide target geometry",
        ));
    }
    let wave_bytes = u64::from(properties.wavefront_size)
        .checked_mul(u64::from(scratch.maximum_private_segment_byte_length))
        .ok_or_else(|| error(ErrorKind::InvalidArgument, "scratch wave size overflows"))?;
    let wave_units = wave_bytes.div_ceil(SCRATCH_ALIGNMENT);
    let required = wave_units
        .checked_mul(SCRATCH_ALIGNMENT)
        .and_then(|size| size.checked_mul(u64::from(scratch.maximum_wave_count)))
        .ok_or_else(|| error(ErrorKind::InvalidArgument, "scratch extent overflows"))?;
    let waves_per_engine = scratch.maximum_wave_count / shader_engines;
    if required > scratch.byte_length
        || wave_units > 0x3_ffff
        || waves_per_engine > 0x0fff
        || scratch.byte_length > u64::from(u32::MAX)
        || scratch.device_address >> 48 != 0
        || scratch
            .device_address
            .checked_add(scratch.byte_length)
            .is_none()
    {
        return Err(error(
            ErrorKind::InvalidArgument,
            "scratch backing cannot represent the requested execution limits",
        ));
    }
    let wave_units = u32::try_from(wave_units).map_err(|_| {
        error(
            ErrorKind::Internal,
            "validated scratch wave size overflowed",
        )
    })?;
    let address_low =
        u32::try_from(scratch.device_address & u64::from(u32::MAX)).map_err(|_| {
            error(
                ErrorKind::Internal,
                "validated scratch address low word overflowed",
            )
        })?;
    let address_high = u32::try_from(scratch.device_address >> 32).map_err(|_| {
        error(
            ErrorKind::Internal,
            "validated scratch address high word overflowed",
        )
    })?;
    let backing_size = u32::try_from(scratch.byte_length)
        .map_err(|_| error(ErrorKind::Internal, "validated scratch size overflowed"))?;
    Ok(Some(ScratchControl {
        tmpring_size: waves_per_engine | (wave_units << 12),
        resource: [
            address_low,
            (address_high & 0xffff) | (1 << 30),
            backing_size,
            GFX1201_SCRATCH_RESOURCE_WORD3,
        ],
        backing_address: scratch.device_address,
        wave64_lane_bytes: scratch.maximum_private_segment_byte_length / 2,
    }))
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_ne_bytes());
}

fn put_aql_scratch(bytes: &mut [u8], gfx_target: u32, scratch: Option<ScratchControl>) {
    if gfx_target == 120_001 {
        let resource = offset_of!(AqlControlLayout, scratch_resource_descriptor);
        put_u32(bytes, resource + 4, 1 << 30);
        put_u32(bytes, resource + 12, GFX1201_SCRATCH_RESOURCE_WORD3);
    }
    if let Some(scratch) = scratch {
        put_u32(
            bytes,
            offset_of!(AqlControlLayout, compute_tmpring_size),
            scratch.tmpring_size,
        );
        let resource = offset_of!(AqlControlLayout, scratch_resource_descriptor);
        for (index, value) in scratch.resource.into_iter().enumerate() {
            put_u32(bytes, resource + index * 4, value);
        }
        put_u64(
            bytes,
            offset_of!(AqlControlLayout, scratch_backing_memory_location),
            scratch.backing_address,
        );
        put_u32(
            bytes,
            offset_of!(AqlControlLayout, scratch_wave64_lane_byte_size),
            scratch.wave64_lane_bytes,
        );
    }
    // AMD_QUEUE_PROPERTIES_IS_PTR64. Scratch recovery keeps this invariant
    // while replacing the adjacent target-specific control fields.
    put_u32(
        bytes,
        offset_of!(AqlControlLayout, queue_properties),
        1 << 1,
    );
    put_u64(
        bytes,
        offset_of!(AqlControlLayout, scratch_max_use_index),
        u64::MAX,
    );
    put_u64(
        bytes,
        offset_of!(AqlControlLayout, alt_scratch_max_use_index),
        u64::MAX,
    );
}

fn initialize_aql_control(
    storage: &mut KfdAllocation,
    request: &Request,
    ring_address: u64,
    apertures: (u64, u64),
) -> Result<(), Error> {
    let aql = request
        .aql
        .as_ref()
        .ok_or_else(|| error(ErrorKind::Internal, "missing AQL control description"))?;
    let mut bytes = [0_u8; AQL_CONTROL_SIZE];
    let read_offset = u32::try_from(AQL_READ_OFFSET)
        .map_err(|_| error(ErrorKind::Internal, "AQL read offset is not representable"))?;
    put_u32(
        &mut bytes,
        offset_of!(AqlControlLayout, queue_header) + offset_of!(AqlQueueHeader, queue_type),
        match aql.producer_mode {
            QueueProducerMode::Single => 1,
            QueueProducerMode::Multiple => 0,
        },
    );
    put_u32(
        &mut bytes,
        offset_of!(AqlControlLayout, queue_header) + offset_of!(AqlQueueHeader, features),
        1,
    );
    put_u64(
        &mut bytes,
        offset_of!(AqlControlLayout, queue_header) + offset_of!(AqlQueueHeader, base_address),
        ring_address,
    );
    put_u32(
        &mut bytes,
        offset_of!(AqlControlLayout, queue_header) + offset_of!(AqlQueueHeader, size),
        request.ring_size / 64,
    );
    put_u64(
        &mut bytes,
        offset_of!(AqlControlLayout, queue_header) + offset_of!(AqlQueueHeader, id),
        u64::MAX,
    );
    put_u32(
        &mut bytes,
        offset_of!(AqlControlLayout, group_segment_aperture_base_hi),
        (apertures.0 >> 32) as u32,
    );
    put_u32(
        &mut bytes,
        offset_of!(AqlControlLayout, private_segment_aperture_base_hi),
        (apertures.1 >> 32) as u32,
    );
    put_u32(
        &mut bytes,
        offset_of!(AqlControlLayout, max_cu_id),
        aql.properties.compute_units - 1,
    );
    put_u32(
        &mut bytes,
        offset_of!(AqlControlLayout, max_wave_id),
        aql.properties.maximum_wave_count_per_compute_unit - 1,
    );
    put_u32(
        &mut bytes,
        offset_of!(AqlControlLayout, read_dispatch_id_field_base_byte_offset),
        read_offset,
    );
    if let Some(signal) = aql.inactive_signal {
        put_u64(
            &mut bytes,
            offset_of!(AqlControlLayout, queue_inactive_signal),
            signal,
        );
    }
    put_aql_scratch(&mut bytes, aql.properties.gfx_target, aql.scratch);
    storage.write_bytes(0, &bytes)
}

fn compute_storage(properties: sysfs::NativeQueueProperties) -> Result<ComputeStorage, Error> {
    let xcc_count = properties.xcc_count;
    let context_size = properties.context_size;
    let control_stack_size = properties.control_stack_size;
    if xcc_count == 0
        || properties.compute_units == 0
        || properties.compute_units % xcc_count != 0
        || context_size == 0
        || context_size % 4096 != 0
        || control_stack_size < 40
        || control_stack_size % 4096 != 0
        || control_stack_size > context_size
    {
        return Err(error(
            ErrorKind::Unsupported,
            "KFD did not report usable queue context-save sizes",
        ));
    }
    let debug_size = (properties.compute_units / xcc_count)
        .checked_mul(32 * 32)
        .ok_or_else(|| error(ErrorKind::DriverContract, "KFD debugger storage overflows"))?;
    let total_size = (u64::from(context_size) + u64::from(debug_size)) * u64::from(xcc_count);
    // DebugOffset and DebugSize in the native header are u32. Check their
    // complete multi-XCC extent before any buffer is acquired or initialized.
    if total_size > u64::from(u32::MAX) {
        return Err(error(
            ErrorKind::Unsupported,
            "KFD context-save extent exceeds its header fields",
        ));
    }
    Ok(ComputeStorage {
        context_size,
        control_stack_size,
        debug_size,
        xcc_count,
        total_size: total_size.div_ceil(4096) * 4096,
    })
}

fn pm4_control(native: &sysfs::NativeNode) -> Option<Pm4Control> {
    let properties = native.queues;
    if !cfg!(target_arch = "x86_64")
        || properties.gfx_target != 120_001
        || properties.compute_queues == 0
        || properties.wavefront_size != 32
        || properties.xcc_count != 1
        || properties.maximum_wave_count_per_compute_unit == 0
        || compute_storage(properties).is_err()
        || util::page_size().ok()? != 4096
    {
        return None;
    }
    let cache_line = usize::try_from(util::host_cache_line_size().ok()?).ok()?;
    if cache_line < size_of::<u64>() || !cache_line.is_power_of_two() || cache_line > 4096 / 3 {
        return None;
    }
    Some(Pm4Control {
        write_offset: cache_line,
        error_offset: 2 * cache_line,
        read_mask: u64::from(PM4_RING_SIZE / 4 - 1),
    })
}

pub(super) fn supports_aql(native: &sysfs::NativeNode) -> bool {
    (100_100..=120_001).contains(&native.queues.gfx_target)
        && native.queues.compute_queues != 0
        && compute_storage(native.queues).is_ok()
}

pub(super) fn supports_pm4(native: &sysfs::NativeNode) -> bool {
    native.gpu_id != 0 && pm4_control(native).is_some()
}

pub(super) fn create(
    vm: Shared<DeviceVm>,
    native: &sysfs::NativeNode,
    desc: QueueRequest,
    lifetime: SessionLifetime,
) -> Result<Owned<KfdQueue>, Error> {
    let request = Request::validate(native, desc)?;
    vm.kfd()
        .enable_runtime()
        .map_err(|source| native_error("AMDKFD_IOC_RUNTIME_ENABLE", source))?;
    KfdQueue::create(vm, &request, lifetime)
}

/// Queue context backing, either registered SVM or an owned native allocation.
enum QueueContext {
    Svm(sys::Reservation),
    Native(Owned<KfdAllocation>),
}

impl QueueContext {
    fn address(&self, vm: &Shared<DeviceVm>) -> Result<u64, Error> {
        match self {
            Self::Svm(storage) => Ok(storage.address() as u64),
            Self::Native(storage) => storage.device_address(vm),
        }
    }

    fn prepare_zeroed(&mut self) -> Result<(), Error> {
        match self {
            // This SVM context is a fresh MAP_PRIVATE | MAP_ANONYMOUS mapping.
            // Its untouched pages are already zero-filled; faulting all of them
            // before KFD registration adds milliseconds to queue creation.
            Self::Svm(_) => Ok(()),
            Self::Native(storage) => storage.zero(),
        }
    }

    fn write_bytes(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        match self {
            Self::Svm(storage) => storage
                .write_bytes(offset, bytes)
                .map_err(|source| native_error("queue context initialization", source)),
            Self::Native(storage) => storage.write_bytes(offset, bytes),
        }
    }

    fn register_svm(&self, vm: &DeviceVm) -> std::io::Result<()> {
        let Self::Svm(storage) = self else {
            return Ok(());
        };
        let flags = uapi::SVM_FLAG_HOST_ACCESS
            | uapi::SVM_FLAG_GPU_EXECUTE
            | uapi::SVM_FLAG_GPU_ALWAYS_MAPPED;
        let mut attributes = [
            uapi::SvmAttribute {
                attribute_type: uapi::SVM_ATTR_PREFETCH_LOCATION,
                value: vm.gpu_id(),
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
                value: vm.gpu_id(),
            },
            uapi::SvmAttribute {
                attribute_type: uapi::SVM_ATTR_GRANULARITY,
                value: 0xff,
            },
        ];
        vm.kfd().svm_attributes(
            storage.address() as u64,
            storage.usable_size() as u64,
            uapi::SVM_OP_SET_ATTR,
            &mut attributes,
        )
    }

    fn release(&mut self) -> Result<(), Error> {
        match self {
            Self::Svm(storage) => storage
                .release()
                .map_err(|source| native_error("queue context munmap", source)),
            Self::Native(storage) => storage.free(),
        }
    }
}

fn initialize_queue_context(
    storage: &mut QueueContext,
    request: &Request,
    compute: &ComputeStorage,
    pointer_device_address: u64,
) -> Result<(), Error> {
    storage.prepare_zeroed()?;
    for xcc in 0..compute.xcc_count {
        let offset = xcc as usize * compute.context_size as usize;
        storage.write_bytes(
            offset + 16,
            &((compute.xcc_count - xcc) * compute.context_size).to_ne_bytes(),
        )?;
        storage.write_bytes(
            offset + 20,
            &(compute.debug_size * compute.xcc_count).to_ne_bytes(),
        )?;
        if let Some(error_event) = request.aql.and_then(|aql| aql.error_event) {
            let event_id = u32::try_from(error_event.native_event_token).map_err(|_| {
                error(
                    ErrorKind::InvalidArgument,
                    "KFD queue event identity exceeds the native field",
                )
            })?;
            storage.write_bytes(offset + 24, &error_event.payload_address.to_ne_bytes())?;
            storage.write_bytes(offset + 32, &event_id.to_ne_bytes())?;
        } else if let Some(pm4) = request.pm4 {
            storage.write_bytes(
                offset + 24,
                &(pointer_device_address + pm4.error_offset as u64).to_ne_bytes(),
            )?;
        }
    }
    Ok(())
}

/// Complete KFD queue owner with resumable teardown state.
///
/// Ring, pointer, optional context, peer mappings, and doorbell state outlive
/// the native queue ID. Destruction records each completed phase so retry does
/// not replay an ioctl or release backing still reachable by hardware.
pub(crate) struct KfdQueue {
    vm: Shared<DeviceVm>,
    backing: [Option<Owned<KfdAllocation>>; 3],
    context: Option<QueueContext>,
    peer_mappings: Mutex<Buffer<QueuePeerMapping>>,
    read_offset: usize,
    write_offset: usize,
    aql: Option<AqlControl>,
    read_index_mask: Option<u64>,
    doorbell_offset: u64,
    id: Option<u32>,
    active: bool,
    priority: u32,
    destroying: bool,
    uncertain: bool,
    info: QueueTransport,
}

impl KfdQueue {
    #[allow(
        clippy::too_many_lines,
        reason = "keep native queue acquisition and publication in one auditable path"
    )]
    fn create(
        vm: Shared<DeviceVm>,
        request: &Request,
        lifetime: SessionLifetime,
    ) -> Result<Owned<Self>, Error> {
        vm.check()?;
        if vm.version.major != 1 || vm.version.minor < 17 {
            return Err(error(
                ErrorKind::Unsupported,
                "queue allocation requires KFD UAPI 1.17 or newer",
            ));
        }
        if request.aql.is_some() {
            vm.initialize_scratch()?;
        }
        let allocate = |size, kind, permissions| {
            KfdAllocation::create(
                vm.clone(),
                AllocationDesc {
                    size,
                    alignment: 4096,
                },
                kind,
                permissions,
            )
        };
        let ring_kind = if request.aql.is_some() && lifetime == SessionLifetime::Process {
            BufferKind::OwnedUserptr { uncached: true }
        } else {
            // KFD USERPTR cannot be allocated in a secondary INSTANCE VM.
            // Coherent GTT provides a host view of the ring in that context.
            BufferKind::Gtt
        };
        let mut ring = allocate(
            u64::from(request.ring_size).div_ceil(4096) * 4096,
            ring_kind,
            DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
        )?;
        ring.zero()?;
        if request.queue_type == 2 {
            // An empty AQL slot has type INVALID (1); zero is vendor-specific.
            // Initialize every slot before KFD can observe the ring.
            let mut packet = [0; 64];
            packet[0] = 1;
            ring.fill_records(&packet, request.ring_size as usize / packet.len())?;
        }
        let mut pointers = allocate(
            4096,
            BufferKind::Gtt,
            DeviceAccess::READ | DeviceAccess::WRITE,
        )?;
        pointers.zero()?;
        let ring_host_address = ring.host_address()?;
        let ring_device_address = ring.device_address(&vm)?;
        let pointer_host_address = pointers.host_address()?;
        let pointer_device_address = pointers.device_address(&vm)?;
        let mut eop = None;
        let mut context = None;
        if let Some(compute) = &request.compute {
            eop = Some(allocate(
                4096,
                BufferKind::Vram {
                    public: false,
                    coherent: false,
                    uncached: false,
                    contiguous: false,
                },
                DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
            )?);
            let context_size = usize::try_from(compute.total_size).map_err(|_| {
                error(
                    ErrorKind::ResourceExhausted,
                    "queue context size exceeds the host address width",
                )
            })?;
            let mut storage = match sys::Reservation::new_aligned_host(
                context_size,
                CWSR_ALIGNMENT,
                (0, isize::MAX as u64),
            ) {
                Ok(reservation) => QueueContext::Svm(reservation),
                Err(_) => QueueContext::Native(allocate(
                    compute.total_size,
                    BufferKind::Gtt,
                    DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
                )?),
            };
            if let QueueContext::Svm(reservation) = &storage {
                let _ = reservation.dont_fork();
            }
            initialize_queue_context(&mut storage, request, compute, pointer_device_address)?;
            if storage.register_svm(&vm).is_err() {
                storage = QueueContext::Native(allocate(
                    compute.total_size,
                    BufferKind::Gtt,
                    DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
                )?);
                initialize_queue_context(&mut storage, request, compute, pointer_device_address)?;
            }
            context = Some(storage);
        }
        let (read_offset, write_offset) = if request.aql.is_some() {
            initialize_aql_control(
                &mut pointers,
                request,
                ring_device_address,
                vm.queue_apertures(),
            )?;
            (AQL_READ_OFFSET, AQL_WRITE_OFFSET)
        } else if let Some(pm4) = request.pm4 {
            (PM4_READ_OFFSET, pm4.write_offset)
        } else {
            (SDMA_READ_OFFSET, SDMA_WRITE_OFFSET)
        };
        let info = QueueTransport {
            ring_host_address,
            ring_device_address,
            ring_size_bytes: u64::from(request.ring_size),
            read_index_host_address: pointer_host_address + read_offset,
            read_index_device_address: pointer_device_address + read_offset as u64,
            write_index_host_address: pointer_host_address + write_offset,
            write_index_device_address: pointer_device_address + write_offset as u64,
            read_index_width: QueueAccessWidth::Bits64,
            write_index_width: QueueAccessWidth::Bits64,
            index_unit_bytes: match request.queue_type {
                0 => 4,
                2 => 64,
                _ => 1,
            },
            read_index_wraps: request.pm4.is_some(),
            doorbell_host_address: 0,
            doorbell_device_address: None,
            doorbell_width: QueueAccessWidth::Bits64,
        };
        let mut args = uapi::CreateQueue {
            ring_address: ring_device_address,
            write_pointer: pointer_device_address + write_offset as u64,
            read_pointer: pointer_device_address + read_offset as u64,
            ring_size: request.ring_size,
            gpu_id: vm.gpu_id(),
            queue_type: request.queue_type,
            percentage: 100,
            priority: request.priority,
            ..uapi::CreateQueue::default()
        };
        if let (Some(eop), Some(context), Some(compute)) = (&eop, &context, &request.compute) {
            args.eop_address = eop.device_address(&vm)?;
            args.eop_size = 4096;
            args.context_address = context.address(&vm)?;
            args.context_size = compute.context_size;
            args.control_stack_size = compute.control_stack_size;
        }
        let allocator = vm.allocator();
        let mut queue = Owned::new(
            Self {
                vm,
                backing: [Some(ring), Some(pointers), eop],
                context,
                peer_mappings: Mutex::new(Buffer::new(allocator)),
                read_offset,
                write_offset,
                aql: request.aql,
                read_index_mask: request.pm4.map(|pm4| pm4.read_mask),
                doorbell_offset: 0,
                id: None,
                active: true,
                priority: request.priority,
                destroying: false,
                uncertain: false,
                info,
            },
            allocator,
        )?;
        let result = queue.vm.kfd().create_queue(&mut args);
        if result.is_ok() {
            // Zero is a valid native queue ID. Only successful CREATE transfers
            // the returned ID; a copy fault may conceal an entirely different ID.
            queue.id = Some(args.queue_id);
        } else {
            queue.uncertain = result
                .as_ref()
                .err()
                .is_some_and(|source| source.raw_os_error() == Some(14));
        }
        result.map_err(|source| native_error("AMDKFD_IOC_CREATE_QUEUE", source))?;
        let doorbell = queue.vm.doorbells.addresses(
            &queue.vm,
            args.doorbell_offset,
            request.device_producer,
        )?;
        queue.info.doorbell_host_address = doorbell.host;
        queue.info.doorbell_device_address = doorbell.device;
        queue.doorbell_offset = args.doorbell_offset;
        queue.vm.check()?;
        Ok(queue)
    }
}

impl KfdQueue {
    pub(crate) fn cached_info(&self) -> QueueTransport {
        self.info
    }
    pub(crate) fn check(&self) -> Result<(), Error> {
        self.info().map(|_| ())
    }
    pub(crate) fn progress(&self) -> Result<(u64, u64), Error> {
        self.check()?;
        let pointers = self.backing[1]
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "queue pointer backing is missing"))?;
        if let Some(mask) = self.read_index_mask {
            pointers.read_wrapping_indices(self.read_offset, self.write_offset, mask)
        } else {
            pointers.read_indices(self.read_offset, self.write_offset)
        }
    }

    pub(crate) fn set_scratch(&mut self, scratch: QueueScratch) -> Result<(), Error> {
        self.check()?;
        let aql = self.aql.as_mut().ok_or_else(|| {
            error(
                ErrorKind::Unsupported,
                "scratch backing requires an AQL compute queue",
            )
        })?;
        let control = validate_scratch(aql.properties, Some(scratch))?;
        let pointers = self.backing[1]
            .as_mut()
            .ok_or_else(|| error(ErrorKind::Internal, "queue pointer backing is missing"))?;
        let start = offset_of!(AqlControlLayout, compute_tmpring_size);
        let end = offset_of!(AqlControlLayout, queue_properties);
        let mut bytes = [0_u8; 40];
        debug_assert_eq!(bytes.len(), end - start);
        if let Some(control) = control {
            put_u32(
                &mut bytes,
                offset_of!(AqlControlLayout, compute_tmpring_size) - start,
                control.tmpring_size,
            );
            let resource = offset_of!(AqlControlLayout, scratch_resource_descriptor) - start;
            for (index, value) in control.resource.into_iter().enumerate() {
                put_u32(&mut bytes, resource + index * 4, value);
            }
            put_u64(
                &mut bytes,
                offset_of!(AqlControlLayout, scratch_backing_memory_location) - start,
                control.backing_address,
            );
            put_u32(
                &mut bytes,
                offset_of!(AqlControlLayout, scratch_wave64_lane_byte_size) - start,
                control.wave64_lane_bytes,
            );
        }
        pointers.write_bytes(start, &bytes)?;
        pointers.write_bytes(
            offset_of!(AqlControlLayout, scratch_max_use_index),
            &u64::MAX.to_ne_bytes(),
        )?;
        pointers.write_bytes(
            offset_of!(AqlControlLayout, alt_scratch_max_use_index),
            &u64::MAX.to_ne_bytes(),
        )?;
        aql.scratch = control;
        Ok(())
    }

    pub(crate) fn map_device(&self, device: Shared<DeviceVm>) -> Result<QueueTransport, Error> {
        self.info()?;
        if Shared::ptr_eq(&self.vm, &device) {
            return Ok(self.info);
        }
        if self.info.doorbell_device_address.is_none() {
            return Err(error(
                ErrorKind::Unsupported,
                "queue was not created for device producers",
            ));
        }
        let (ring_handle, ring_address) = self.backing[0]
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "queue ring backing is missing"))?
            .peer_mapping_source(&device)?;
        let (pointer_handle, pointer_address) = self.backing[1]
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "queue pointer backing is missing"))?
            .peer_mapping_source(&device)?;
        let mut peers = self
            .peer_mappings
            .lock()
            .map_err(|_| error(ErrorKind::Internal, "queue peer mapping lock was poisoned"))?;
        let index = if let Some(index) = peers
            .iter()
            .position(|mapping| Shared::ptr_eq(&mapping.vm, &device))
        {
            index
        } else {
            peers.try_reserve(1)?;
            peers.try_push(QueuePeerMapping::new(
                device.clone(),
                [ring_handle, pointer_handle],
            ))?;
            peers.len() - 1
        };
        if !peers[index].is_mapped() {
            if let Err(failure) = peers[index].enable() {
                if !peers[index].uncertain {
                    peers[index].close()?;
                    if index + 1 == peers.len() {
                        let _ = peers.pop();
                    }
                }
                return Err(failure);
            }
        }
        let doorbell = match self
            .vm
            .doorbells
            .peer_address(&self.vm, device, self.doorbell_offset)
        {
            Ok(address) => address,
            Err(failure) => {
                peers[index].close()?;
                if index + 1 == peers.len() {
                    let _ = peers.pop();
                }
                return Err(failure);
            }
        };
        Ok(QueueTransport {
            ring_device_address: ring_address,
            read_index_device_address: pointer_address + self.read_offset as u64,
            write_index_device_address: pointer_address + self.write_offset as u64,
            doorbell_device_address: Some(doorbell),
            ..self.info
        })
    }
}

impl KfdQueue {
    pub(crate) fn info(&self) -> Result<QueueTransport, Error> {
        self.vm.check()?;
        if self.destroying || self.uncertain || self.id.is_none() {
            return Err(error(
                ErrorKind::Unsupported,
                "queue transport is no longer available",
            ));
        }
        Ok(self.info)
    }

    pub(crate) fn inactivate(&mut self) -> Result<(), Error> {
        self.vm
            .kfd()
            .check_process()
            .map_err(|source| native_error("KFD queue inactivation", source))?;
        if self.destroying || self.uncertain {
            return Err(error(ErrorKind::Unsupported, "queue is not available"));
        }
        if !self.active {
            return Ok(());
        }
        let id = self
            .id
            .ok_or_else(|| error(ErrorKind::Unsupported, "queue has no native identifier"))?;
        self.vm
            .kfd()
            .update_queue(&mut uapi::UpdateQueue {
                ring_address: 0,
                queue_id: id,
                ring_size: 0,
                percentage: 0,
                priority: self.priority,
            })
            .map_err(|source| native_error("AMDKFD_IOC_UPDATE_QUEUE", source))?;
        self.active = false;
        Ok(())
    }

    pub(crate) fn set_priority(&mut self, priority: QueuePriority) -> Result<(), Error> {
        self.check()?;
        if !self.active {
            return Err(error(ErrorKind::Unsupported, "queue is inactive"));
        }
        let id = self
            .id
            .ok_or_else(|| error(ErrorKind::Unsupported, "queue has no native identifier"))?;
        let priority = match priority {
            QueuePriority::Low => 0,
            QueuePriority::Normal => 7,
            QueuePriority::High => 15,
        };
        self.vm
            .kfd()
            .update_queue(&mut uapi::UpdateQueue {
                ring_address: self.info.ring_device_address,
                queue_id: id,
                ring_size: u32::try_from(self.info.ring_size_bytes).map_err(|_| {
                    error(ErrorKind::Internal, "queue ring size exceeds native field")
                })?,
                percentage: 100,
                priority,
            })
            .map_err(|source| native_error("AMDKFD_IOC_UPDATE_QUEUE", source))?;
        self.priority = priority;
        Ok(())
    }

    pub(crate) fn set_cu_mask(&mut self, mask: &[u32]) -> Result<(), Error> {
        self.check()?;
        if self.aql.is_none() {
            return Err(error(
                ErrorKind::Unsupported,
                "CU masking requires an AQL compute queue",
            ));
        }
        let id = self
            .id
            .ok_or_else(|| error(ErrorKind::Unsupported, "queue has no native identifier"))?;
        self.vm
            .kfd()
            .set_cu_mask(id, mask)
            .map_err(|source| native_error("AMDKFD_IOC_SET_CU_MASK", source))
    }

    pub(crate) fn destroy(&mut self) -> Result<(), Error> {
        self.vm
            .kfd()
            .check_process()
            .map_err(|source| native_error("KFD queue destroy", source))?;
        if self.active && !self.destroying && !self.uncertain && self.id.is_some() {
            match self.progress() {
                Ok((read, write)) if read != write => {
                    return Err(error(ErrorKind::Busy, "queue still has published work"));
                }
                Ok(_) => (),
                Err(e) if e.kind() == ErrorKind::DeviceLost => (),
                Err(e) => return Err(e),
            }
        }
        self.destroying = true;
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD queue outcome is uncertain; dependencies are retained until process teardown",
            ));
        }
        if let Some(id) = self.id {
            if let Err(source) = self.vm.kfd().destroy_queue(id) {
                // KFD can return ETIME/EIO after removing the queue and making
                // its ID reusable. Replay only errors whose paths retain it.
                // Unknown errors are conservative: keep backing, never replay.
                self.uncertain = !matches!(source.raw_os_error(), Some(4 | 16 | 512));
                return Err(native_error("AMDKFD_IOC_DESTROY_QUEUE", source));
            }
            // Backing cleanup may fail. Clear the released ID first so that a
            // later cleanup attempt cannot destroy another queue reusing it.
            self.id = None;
        }
        let peers = self
            .peer_mappings
            .get_mut()
            .map_err(|_| error(ErrorKind::Internal, "queue peer mapping lock was poisoned"))?;
        for peer in peers.iter_mut().rev() {
            peer.close()?;
        }
        peers.clear();
        for slot in &mut self.backing {
            if let Some(allocation) = slot {
                allocation.free()?;
                *slot = None;
            }
        }
        if let Some(context) = &mut self.context {
            context.release()?;
            self.context = None;
        }
        Ok(())
    }
}

impl Drop for KfdQueue {
    fn drop(&mut self) {
        if self.destroy().is_err() {
            // Native queue references can outlive an unsuccessful destructor.
            // Retain their exact BOs and mappings without allocating cleanup
            // work; the kernel's process teardown remains the final owner.
            for slot in &mut self.backing {
                if let Some(allocation) = slot.take() {
                    std::mem::forget(allocation);
                }
            }
            if let Some(context) = self.context.take() {
                std::mem::forget(context);
            }
            let peers = self
                .peer_mappings
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for peer in peers.iter() {
                if peer.uncertain
                    || peer
                        .mappings
                        .iter()
                        .any(|mapping| !matches!(mapping, DoorbellGpuMapping::Unmapped))
                {
                    std::mem::forget(peer.vm.clone());
                }
            }
            std::mem::forget(self.vm.clone());
        }
    }
}

#[cfg(test)]
#[path = "tests/queue.rs"]
mod tests;
