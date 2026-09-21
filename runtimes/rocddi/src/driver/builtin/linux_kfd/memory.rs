//! Session-owned VM bindings and independently owned KFD allocations.
//!
//! A primary-process VM may survive session destruction in the kernel. This
//! session retains exact render files for device recreation while it remains
//! live; there is no process-wide registry or deferred resource collection.
use super::{drm, sys, sysfs, uapi, util};
use crate::event::GpuMemoryFault;
use crate::host_storage::{Allocator, Buffer, Owned, Shared};
use crate::memory::interop::linux::{DmaBuf, DmaBufInfo, KfdIpcMemoryHandle};
use crate::memory::{AllocationDesc, AllocationInfo, DeviceAccess};
use crate::{Error, ErrorKind};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Process-local set of activated device VMs and their shared loss events.
///
/// `process` rejects use after `fork`, while the mutex serializes publication,
/// reuse, and shutdown of the exact VM owners retained by this session.
pub(super) struct VmBindings {
    process: AtomicU32,
    bindings: Mutex<Bindings>,
}

/// Mutable process binding set serialized across activation and shutdown.
struct Bindings {
    loss: Option<Shared<LossEvent>>,
    devices: Buffer<Shared<DeviceVm>>,
}

const SCRATCH_BASE_ALIGNMENT: usize = 64 * 1024;
const GRAPHICS_IMPORT_ALIGNMENT: usize = 256 * 1024;
const IPC_PAGE_SIZE: u64 = 4096;
const IPC_APERTURE_DGPU: u32 = 1;
const IPC_APERTURE_DGPU_ALT: u32 = 2;
const IPC_APERTURE_GPUVM: u32 = 3;
const IPC_FRAGMENT: u32 = 1 << 31;
const SCRATCH_BYTES_PER_XCC: u64 = 4 << 30;
const GFX12_SCRATCH_BYTES_PER_XCC: u64 = 8 << 30;

/// One free or allocated interval in the process scratch aperture.
#[derive(Clone, Copy)]
struct ScratchRange {
    offset: u64,
    size: u64,
    allocated: bool,
}

/// Lazily reserved scratch aperture shared by queues on one device VM.
///
/// The reservation is retained if programming its KFD base has an ambiguous
/// outcome; reusing that virtual range could otherwise alias live GPU state.
struct ScratchPool {
    capacity: u64,
    reservation: Option<sys::Reservation>,
    ranges: Buffer<ScratchRange>,
    uncertain: bool,
}

impl ScratchPool {
    fn new(properties: sysfs::NativeQueueProperties, allocator: Allocator) -> Self {
        let per_xcc = if properties.gfx_target / 10_000 >= 12 {
            GFX12_SCRATCH_BYTES_PER_XCC
        } else {
            SCRATCH_BYTES_PER_XCC
        };
        let capacity = per_xcc
            .checked_mul(u64::from(properties.xcc_count))
            .filter(|capacity| *capacity != 0 && usize::try_from(*capacity).is_ok())
            .unwrap_or(0);
        Self {
            capacity,
            reservation: None,
            ranges: Buffer::new(allocator),
            uncertain: false,
        }
    }

    fn initialize(&mut self, vm: &DeviceVm) -> Result<(), Error> {
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD scratch-base programming outcome is uncertain",
            ));
        }
        if self.reservation.is_some() {
            return Ok(());
        }
        if self.capacity == 0 {
            return Err(error(
                ErrorKind::Unsupported,
                "native target has no scratch aperture capacity",
            ));
        }
        self.ranges.try_reserve_exact(1)?;
        let reservation = sys::Reservation::new(
            usize::try_from(self.capacity).map_err(|_| {
                error(
                    ErrorKind::ResourceExhausted,
                    "scratch capacity exceeds the host address width",
                )
            })?,
            SCRATCH_BASE_ALIGNMENT,
            vm.address_range(),
            false,
        )
        .map_err(|source| native_error("scratch address reservation", source))?;
        let address = reservation.address() as u64;
        let result = vm.loss.kfd.set_scratch_backing_va(vm.gpu_id, address);
        self.reservation = Some(reservation);
        if let Err(source) = result {
            // The ioctl has no output that can prove whether a failing call
            // changed the process base. Retain the VA until process teardown.
            self.uncertain = true;
            return Err(native_error("AMDKFD_IOC_SET_SCRATCH_BACKING_VA", source));
        }
        self.ranges.try_push(ScratchRange {
            offset: 0,
            size: self.capacity,
            allocated: false,
        })?;
        Ok(())
    }

    fn allocate(&mut self, vm: &DeviceVm, size: u64) -> Result<u64, Error> {
        self.initialize(vm)?;
        let index = self
            .ranges
            .iter()
            .position(|range| !range.allocated && range.size >= size)
            .ok_or_else(|| {
                error(
                    ErrorKind::ResourceExhausted,
                    "scratch aperture is exhausted",
                )
            })?;
        let range = self.ranges[index];
        if range.size != size {
            self.ranges.try_push(ScratchRange {
                offset: range.offset + size,
                size: range.size - size,
                allocated: false,
            })?;
        }
        self.ranges[index].size = size;
        self.ranges[index].allocated = true;
        self.reservation
            .as_ref()
            .and_then(|reservation| (reservation.address() as u64).checked_add(range.offset))
            .ok_or_else(|| error(ErrorKind::Internal, "scratch address overflow"))
    }

    fn release(&mut self, address: u64, size: u64) -> Result<(), Error> {
        let base = self
            .reservation
            .as_ref()
            .map(|reservation| reservation.address() as u64)
            .ok_or_else(|| error(ErrorKind::Internal, "scratch aperture is not initialized"))?;
        let offset = address
            .checked_sub(base)
            .ok_or_else(|| error(ErrorKind::Internal, "scratch address precedes its aperture"))?;
        let range = self
            .ranges
            .iter_mut()
            .find(|range| range.allocated && range.offset == offset && range.size == size)
            .ok_or_else(|| error(ErrorKind::Internal, "scratch range is not allocated"))?;
        range.allocated = false;
        loop {
            let mut pair = None;
            for first in 0..self.ranges.len() {
                if self.ranges[first].allocated {
                    continue;
                }
                for second in first + 1..self.ranges.len() {
                    if self.ranges[second].allocated {
                        continue;
                    }
                    let first_end = self.ranges[first]
                        .offset
                        .checked_add(self.ranges[first].size);
                    let second_end = self.ranges[second]
                        .offset
                        .checked_add(self.ranges[second].size);
                    if first_end == Some(self.ranges[second].offset)
                        || second_end == Some(self.ranges[first].offset)
                    {
                        pair = Some((first, second));
                        break;
                    }
                }
                if pair.is_some() {
                    break;
                }
            }
            let Some((first, second)) = pair else {
                break;
            };
            let offset = self.ranges[first].offset.min(self.ranges[second].offset);
            let size = self.ranges[first]
                .size
                .checked_add(self.ranges[second].size)
                .ok_or_else(|| error(ErrorKind::Internal, "scratch range size overflow"))?;
            self.ranges[first] = ScratchRange {
                offset,
                size,
                allocated: false,
            };
            let last = self
                .ranges
                .pop()
                .ok_or_else(|| error(ErrorKind::Internal, "scratch range list is empty"))?;
            if second < self.ranges.len() {
                self.ranges[second] = last;
            }
        }
        Ok(())
    }
}

impl VmBindings {
    pub(super) fn shutdown(&mut self) -> Result<(), Error> {
        let bindings = self
            .bindings
            .get_mut()
            .map_err(|_| error(ErrorKind::Internal, "VM bindings lock poisoned"))?;
        while let Some(vm) = bindings.devices.last_mut() {
            // An abandoned native resource can retain callback-allocated VM
            // metadata. PROCESS permits kernel state to survive, but callers
            // may release their allocator when session destruction succeeds.
            // Keep this binding and fail teardown while that dependency exists.
            let vm = Shared::get_mut(vm).ok_or_else(|| {
                error(
                    ErrorKind::DriverContract,
                    "retained native VM dependencies prevent session destruction",
                )
            })?;
            vm.close()?;
            bindings.devices.pop();
        }
        if let Some(loss) = bindings.loss.as_mut() {
            let loss = Shared::get_mut(loss).ok_or_else(|| {
                error(
                    ErrorKind::DriverContract,
                    "retained native loss-event dependencies prevent session destruction",
                )
            })?;
            loss.close()?;
        }
        bindings.loss = None;
        Ok(())
    }

    #[cfg(test)]
    #[allow(clippy::unwrap_used)]
    pub(super) fn is_empty_for_test(&self) -> bool {
        self.bindings.lock().unwrap().devices.is_empty()
    }
    pub(super) fn new(allocator: Allocator) -> Self {
        Self {
            process: AtomicU32::new(0),
            bindings: Mutex::new(Bindings {
                loss: None,
                devices: Buffer::new(allocator),
            }),
        }
    }

    pub(super) fn svm_gpu_id(&self, identity: [u8; 16]) -> Result<u32, Error> {
        let bindings = self.bindings.lock().map_err(|_| {
            error(
                ErrorKind::Internal,
                "KFD VM bindings lock was poisoned during SVM translation",
            )
        })?;
        bindings
            .devices
            .iter()
            .find(|device| device.identity == identity)
            .map(|device| device.gpu_id)
            .ok_or_else(|| {
                error(
                    ErrorKind::InvalidArgument,
                    "SVM attribute names an inactive or foreign endpoint",
                )
            })
    }

    pub(super) fn svm_identity(&self, gpu_id: u32) -> Result<[u8; 16], Error> {
        let bindings = self.bindings.lock().map_err(|_| {
            error(
                ErrorKind::Internal,
                "KFD VM bindings lock was poisoned during SVM translation",
            )
        })?;
        bindings
            .devices
            .iter()
            .find(|device| device.gpu_id == gpu_id)
            .map(|device| device.identity)
            .ok_or_else(|| {
                error(
                    ErrorKind::DriverContract,
                    "KFD returned an unknown GPU identifier for an SVM attribute",
                )
            })
    }

    pub(super) fn device(
        &self,
        kfd: &Shared<sys::Kfd>,
        node: &sysfs::NativeNode,
    ) -> Result<Shared<DeviceVm>, Error> {
        self.device_with_render(kfd, node, sysfs::open_render)
    }

    // Supplying the opener lets tests exercise concurrent initialization and
    // native conflicts without needing a render device or a process-global fake.
    #[allow(
        clippy::too_many_lines,
        reason = "keep VM validation, acquisition, and publication under one lifecycle lock"
    )]
    fn device_with_render(
        &self,
        kfd: &Shared<sys::Kfd>,
        node: &sysfs::NativeNode,
        open_render: impl FnOnce(u32) -> io::Result<File>,
    ) -> Result<Shared<DeviceVm>, Error> {
        kfd.check_process()
            .map_err(|source| native_error("KFD VM acquisition", source))?;
        let mut runtime = kfd
            .runtime_guard()
            .map_err(|source| native_error("KFD runtime serialization", source))?;
        let process = std::process::id();
        let owner = self
            .process
            .compare_exchange(0, process, Ordering::AcqRel, Ordering::Acquire)
            .unwrap_or_else(|owner| owner);
        if owner != 0 && owner != process {
            return Err(error(
                ErrorKind::Unsupported,
                "inherited KFD VM state requires exec",
            ));
        }
        let mut bindings = self.bindings.lock().map_err(|_| {
            error(
                ErrorKind::Internal,
                "KFD VM initialization lock was poisoned",
            )
        })?;
        if let Some(vm) = bindings.devices.iter().find(|vm| vm.gpu_id == node.gpu_id) {
            if vm.identity != node.identity
                || vm.unique_id != node.unique_id
                || Some(vm.render_minor) != node.render_minor
            {
                return Err(error(
                    ErrorKind::DeviceLost,
                    "KFD reused a GPU identifier for a different native device",
                ));
            }
            vm.check()?;
            kfd.enable_runtime_locked(&mut runtime)
                .map_err(|source| native_error("AMDKFD_IOC_RUNTIME_ENABLE", source))?;
            return Ok(vm.clone());
        }
        let kfd_allocator = kfd.allocator();
        let version = kfd
            .version()
            .map_err(|source| native_error("AMDKFD_IOC_GET_VERSION", source))?;
        if version.major != 1 || version.minor < 17 {
            return Err(error(
                ErrorKind::Unsupported,
                "KFD UAPI 1.17 or newer is required",
            ));
        }
        let aperture = kfd
            .apertures()
            .map_err(|source| native_error("AMDKFD_IOC_GET_PROCESS_APERTURES_NEW", source))?
            .into_iter()
            .find(|entry| entry.gpu_id == node.gpu_id)
            .ok_or_else(|| error(ErrorKind::DeviceLost, "KFD device has no process aperture"))?;
        if aperture.gpuvm_base == 0
            || aperture.gpuvm_base > aperture.gpuvm_limit
            || aperture.gpuvm_limit > isize::MAX as u64
        {
            return Err(error(
                ErrorKind::Unsupported,
                "KFD device does not provide a canonical host-reservable GPU aperture",
            ));
        }
        let render_minor = node
            .render_minor
            .ok_or_else(|| error(ErrorKind::Unsupported, "KFD device has no render endpoint"))?;
        let render = open_render(render_minor)
            .map_err(|source| native_error("KFD render node open", source))?;
        let system_dma_buf_import = drm::supports_system_dma_buf_import(&render);
        if bindings.loss.is_none() {
            bindings.loss = Some(LossEvent::create(kfd.clone())?);
        }
        let loss = bindings
            .loss
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "KFD loss event was not published"))?
            .clone();
        loss.check()?;
        bindings.devices.try_reserve(1).map_err(|_| {
            error(
                ErrorKind::ResourceExhausted,
                "KFD VM owner storage is exhausted",
            )
        })?;
        let vm = Shared::new(
            DeviceVm {
                loss,
                render: Some(render),
                system_dma_buf_import,
                gpu_id: node.gpu_id,
                render_minor,
                identity: node.identity,
                unique_id: node.unique_id,
                base: aperture.gpuvm_base,
                limit: aperture.gpuvm_limit,
                lds_base: aperture.lds_base,
                scratch_base: aperture.scratch_base,
                scratch: Mutex::new(ScratchPool::new(node.queues, kfd_allocator)),
                vmem: Mutex::new(super::vmem::VmState::new(kfd_allocator)),
                version,
                doorbells: super::queue::Doorbells::default(),
            },
            kfd_allocator,
        )?;
        vm.loss
            .kfd
            .acquire_vm(vm.render()?, vm.gpu_id)
            .map_err(|source| native_error("AMDKFD_IOC_ACQUIRE_VM", source))?;
        // No fallible allocation occurs after acquisition. KFD now retains this
        // exact render file until process teardown, so publish the same owner
        // before runtime activation can fail or block.
        bindings.devices.try_push(vm.clone())?;
        kfd.enable_runtime_locked(&mut runtime)
            .map_err(|source| native_error("AMDKFD_IOC_RUNTIME_ENABLE", source))?;
        Ok(vm)
    }
}

/// One activated device VM and every process-scoped dependency needed to use it.
///
/// The exact render file is retained because KFD associates the acquired VM
/// with that file description. Scratch, virtual-memory mappings, and doorbells
/// are closed before the render endpoint itself.
pub(crate) struct DeviceVm {
    loss: Shared<LossEvent>,
    render: Option<File>,
    system_dma_buf_import: bool,
    gpu_id: u32,
    render_minor: u32,
    identity: [u8; 16],
    unique_id: Option<u64>,
    base: u64,
    limit: u64,
    lds_base: u64,
    scratch_base: u64,
    scratch: Mutex<ScratchPool>,
    pub(super) vmem: Mutex<super::vmem::VmState>,
    pub(super) version: uapi::Version,
    pub(super) doorbells: super::queue::Doorbells,
}

impl DeviceVm {
    pub(super) fn render_minor(&self) -> u32 {
        self.render_minor
    }

    pub(super) fn supports_system_dma_buf_import(&self) -> bool {
        self.system_dma_buf_import
    }
    pub(super) fn render(&self) -> Result<&File, Error> {
        self.render
            .as_ref()
            .ok_or_else(|| error(ErrorKind::InvalidArgument, "render endpoint was closed"))
    }
    pub(super) fn close(&mut self) -> Result<(), Error> {
        self.loss
            .kfd
            .check_process()
            .map_err(|e| native_error("render close process check", e))?;
        let render = self
            .render
            .as_ref()
            .ok_or_else(|| error(ErrorKind::InvalidArgument, "render endpoint was closed"))?;
        self.vmem
            .get_mut()
            .map_err(|_| error(ErrorKind::Internal, "virtual-memory mapping lock poisoned"))?
            .close(render)?;
        self.doorbells.close(&self.loss.kfd)?;
        util::close_file(&mut self.render).map_err(|e| native_error("render endpoint close", e))
    }

    pub(super) fn check(&self) -> Result<(), Error> {
        self.loss.check()
    }

    pub(super) fn poll_memory_fault(&self) -> Result<Option<GpuMemoryFault>, Error> {
        self.loss.poll_memory_fault()
    }

    pub(super) fn has_observed_loss(&self) -> bool {
        self.loss.has_observed_loss()
    }

    pub(super) fn kfd(&self) -> &sys::Kfd {
        &self.loss.kfd
    }

    pub(super) fn kfd_owner(&self) -> Shared<sys::Kfd> {
        self.loss.kfd.clone()
    }

    pub(super) fn address_range(&self) -> (u64, u64) {
        (self.base, self.limit)
    }

    pub(super) fn gpu_id(&self) -> u32 {
        self.gpu_id
    }

    pub(super) fn shares_kfd(&self, other: &Self) -> bool {
        Shared::ptr_eq(&self.loss.kfd, &other.loss.kfd)
    }

    pub(super) fn allocator(&self) -> Allocator {
        self.loss.allocator()
    }

    pub(super) fn queue_apertures(&self) -> (u64, u64) {
        (self.lds_base, self.scratch_base)
    }

    pub(super) fn initialize_scratch(&self) -> Result<(), Error> {
        self.check()?;
        self.scratch
            .lock()
            .map_err(|_| error(ErrorKind::Internal, "scratch allocator lock poisoned"))?
            .initialize(self)
    }

    fn allocate_scratch(&self, size: u64) -> Result<u64, Error> {
        self.initialize_scratch()?;
        if size == 0 || size % 4096 != 0 {
            return Err(error(
                ErrorKind::InvalidArgument,
                "scratch extent is not page aligned",
            ));
        }
        self.scratch
            .lock()
            .map_err(|_| error(ErrorKind::Internal, "scratch allocator lock poisoned"))?
            .allocate(self, size)
    }

    fn release_scratch(&self, address: u64, size: u64) -> Result<(), Error> {
        self.scratch
            .lock()
            .map_err(|_| error(ErrorKind::Internal, "scratch allocator lock poisoned"))?
            .release(address, size)
    }
}

/// Shared hardware- and memory-exception events for all owners of one KFD VM.
///
/// Once loss is observed it remains sticky. The memory event can be claimed by
/// detailed fault reporting so the general health check does not consume it a
/// second time.
struct LossEvent {
    kfd: Shared<sys::Kfd>,
    hardware_event_id: AtomicU32,
    memory_event_id: AtomicU32,
    hardware_destroy_uncertain: AtomicBool,
    memory_destroy_uncertain: AtomicBool,
    memory_event_claimed: Mutex<bool>,
    lost: AtomicBool,
}

impl LossEvent {
    fn close(&mut self) -> Result<(), Error> {
        self.kfd
            .check_process()
            .map_err(|e| native_error("loss event close process check", e))?;
        let memory = self.destroy_event(
            &self.memory_event_id,
            &self.memory_destroy_uncertain,
            "KFD memory-exception event destruction",
        );
        let hardware = self.destroy_event(
            &self.hardware_event_id,
            &self.hardware_destroy_uncertain,
            "KFD hardware-exception event destruction",
        );
        memory.and(hardware)
    }

    fn destroy_event(
        &self,
        event_id: &AtomicU32,
        uncertain: &AtomicBool,
        operation: &'static str,
    ) -> Result<(), Error> {
        if uncertain.load(Ordering::Acquire) {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD event destruction outcome is uncertain",
            ));
        }
        let id = event_id.load(Ordering::Acquire);
        if id != 0 {
            if let Err(source) = self.kfd.destroy_event(id) {
                // The ioctl can fail after destroying the event. Keep its ID
                // for diagnostics, but never replay it: KFD may reuse the ID.
                uncertain.store(true, Ordering::Release);
                return Err(native_error(operation, source));
            }
            event_id.store(0, Ordering::Release);
        }
        Ok(())
    }

    fn create(kfd: Shared<sys::Kfd>) -> Result<Shared<Self>, Error> {
        let allocator = kfd.allocator();
        let event = Shared::new(
            Self {
                kfd,
                hardware_event_id: AtomicU32::new(0),
                memory_event_id: AtomicU32::new(0),
                hardware_destroy_uncertain: AtomicBool::new(false),
                memory_destroy_uncertain: AtomicBool::new(false),
                memory_event_claimed: Mutex::new(false),
                lost: AtomicBool::new(false),
            },
            allocator,
        )?;
        let mut args = uapi::CreateEvent::default();
        let result = event
            .kfd
            .create_exception_event(uapi::HW_EXCEPTION, &mut args);
        event
            .hardware_event_id
            .store(args.event_id, Ordering::Release);
        result.map_err(|source| native_error("KFD hardware-exception event creation", source))?;
        if args.event_id == 0 {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD returned an invalid hardware-exception event",
            ));
        }
        let mut args = uapi::CreateEvent::default();
        let result = event
            .kfd
            .create_exception_event(uapi::MEMORY_EXCEPTION, &mut args);
        event
            .memory_event_id
            .store(args.event_id, Ordering::Release);
        result.map_err(|source| native_error("KFD memory-exception event creation", source))?;
        if args.event_id == 0 {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD returned an invalid memory-exception event",
            ));
        }
        Ok(event)
    }

    fn check(&self) -> Result<(), Error> {
        self.kfd
            .check_process()
            .map_err(|source| native_error("KFD memory query", source))?;
        if !self.lost.load(Ordering::Acquire) {
            let hardware_lost = self
                .kfd
                .hardware_memory_lost(self.hardware_event_id.load(Ordering::Acquire))
                .map_err(|source| native_error("KFD hardware-exception poll", source))?;
            let memory_fault = if hardware_lost {
                false
            } else {
                let claimed = self.memory_event_claimed.lock().map_err(|_| {
                    error(
                        ErrorKind::Internal,
                        "KFD memory-event ownership lock poisoned",
                    )
                })?;
                if *claimed {
                    false
                } else {
                    self.kfd
                        .memory_exception(self.memory_event_id.load(Ordering::Acquire))
                        .map_err(|source| native_error("KFD memory-exception poll", source))?
                        .is_some()
                }
            };
            if hardware_lost || memory_fault {
                self.lost.store(true, Ordering::Release);
            }
        }
        if self.lost.load(Ordering::Acquire) {
            Err(error(
                ErrorKind::DeviceLost,
                "KFD reported loss of device memory; process restart is required",
            ))
        } else {
            Ok(())
        }
    }

    fn has_observed_loss(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    fn poll_memory_fault(&self) -> Result<Option<GpuMemoryFault>, Error> {
        let mut claimed = self.memory_event_claimed.lock().map_err(|_| {
            error(
                ErrorKind::Internal,
                "KFD memory-event ownership lock poisoned",
            )
        })?;
        let exception = self
            .kfd
            .memory_exception(self.memory_event_id.load(Ordering::Acquire))
            .map_err(|source| native_error("KFD memory-exception poll", source))?;
        *claimed = true;
        Ok(exception.map(|fault| GpuMemoryFault {
            kfd_gpu_id: fault.gpu_id,
            virtual_address: fault.address,
            page_not_present: fault.not_present != 0,
            read_only: fault.read_only != 0,
            no_execute: fault.no_execute != 0,
            imprecise: fault.imprecise != 0,
            error_type: fault.error_type,
        }))
    }
}

impl Drop for LossEvent {
    fn drop(&mut self) {
        let _ = self.destroy_event(
            &self.memory_event_id,
            &self.memory_destroy_uncertain,
            "KFD memory-exception event destruction",
        );
        let _ = self.destroy_event(
            &self.hardware_event_id,
            &self.hardware_destroy_uncertain,
            "KFD hardware-exception event destruction",
        );
    }
}

/// The prefix counts PTE work, not completion of the ioctl. Even a full prefix
/// stays in Mapping/Unmapping until synchronization also succeeds. Keeping the
/// two retry phases distinct prevents an UNMAP from being replayed from zero.
enum GpuMapping {
    NotStarted,
    Mapping(u32),
    Mapped,
    Unmapping(u32),
    Unmapped,
}

/// Concrete KFD buffer placement. GTT supplies SYSTEM allocations and queue
/// backing; CPU-only allocations use separate OS mappings and acquire no GPU
/// access as a side effect.
#[derive(Clone, Copy)]
pub(super) enum BufferKind {
    Vram {
        public: bool,
        coherent: bool,
        uncached: bool,
        contiguous: bool,
    },
    Gtt,
    Mmio,
    OwnedUserptr {
        uncached: bool,
    },
    Userptr {
        address: usize,
        uncached: bool,
    },
}

/// One peer VM retained for the complete lifetime of its native mapping.
struct PeerVm {
    vm: Shared<DeviceVm>,
}

/// Complete ownership record for a KFD allocation and all of its mappings.
///
/// Optional fields encode teardown progress. Successful cleanup clears each
/// field exactly once; failure leaves the remaining handles and reservations in
/// this object so a later `free` call can resume safely.
pub(crate) struct KfdAllocation {
    vm: Shared<DeviceVm>,
    peers: Buffer<PeerVm>,
    gpu_ids: Buffer<u32>,
    reservation: Option<sys::Reservation>,
    handle: Option<u64>,
    host_address: Option<usize>,
    device_byte_offset: usize,
    logical_size: u64,
    origin_gpu_id: u32,
    native_flags: u32,
    physical_backing_id: [u64; 2],
    metadata: Buffer<u8>,
    scratch_range: Option<(u64, u64)>,
    mapping: GpuMapping,
    freeing: bool,
    uncertain: bool,
}

fn allocation_source(kind: BufferKind, size: usize) -> Result<(Option<usize>, usize, u64), Error> {
    let BufferKind::Userptr { address, .. } = kind else {
        return Ok((None, 0, 0));
    };
    let page =
        util::page_size().map_err(|source| native_error("registered host page size", source))?;
    let offset = address & (page - 1);
    let page_base = address - offset;
    if address == 0 || page_base.checked_add(size).is_none() {
        return Err(error(
            ErrorKind::InvalidArgument,
            "registered host page cover is invalid",
        ));
    }
    Ok((Some(address), offset, page_base as u64))
}

impl KfdAllocation {
    pub(super) fn create(
        vm: Shared<DeviceVm>,
        desc: AllocationDesc,
        kind: BufferKind,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self>, Error> {
        Self::create_with_peers(
            vm,
            std::iter::empty::<Shared<DeviceVm>>(),
            desc,
            kind,
            permissions,
        )
    }

    #[allow(
        clippy::too_many_lines,
        reason = "validation, common-VA selection, and native ownership form one transaction"
    )]
    pub(super) fn create_with_peers(
        vm: Shared<DeviceVm>,
        peers: impl ExactSizeIterator<Item = Shared<DeviceVm>>,
        desc: AllocationDesc,
        kind: BufferKind,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self>, Error> {
        if !permissions.contains(DeviceAccess::READ) {
            return Err(error(
                ErrorKind::Unsupported,
                "GPU allocation requires READ permission",
            ));
        }
        let peer_count = peers.len();
        let allocator = vm.allocator();
        let mut peer_vms: Buffer<PeerVm> = Buffer::try_with_capacity(peer_count, allocator)?;
        let mut gpu_ids = Buffer::try_with_capacity(
            peer_count
                .checked_add(1)
                .ok_or_else(|| error(ErrorKind::ResourceExhausted, "device count overflow"))?,
            allocator,
        )?;
        gpu_ids.try_push(vm.gpu_id)?;
        let mut bounds = vm.address_range();
        for peer in peers {
            if !Shared::ptr_eq(&vm.loss.kfd, &peer.loss.kfd) {
                return Err(error(
                    ErrorKind::InvalidArgument,
                    "allocation devices must share one KFD session",
                ));
            }
            if Shared::ptr_eq(&vm, &peer)
                || peer_vms
                    .iter()
                    .any(|retained| Shared::ptr_eq(&retained.vm, &peer))
            {
                continue;
            }
            if peer.gpu_id == vm.gpu_id || gpu_ids.iter().any(|gpu_id| *gpu_id == peer.gpu_id) {
                return Err(error(
                    ErrorKind::InvalidArgument,
                    "distinct VM bindings must have distinct GPU IDs",
                ));
            }
            peer.check()?;
            let peer_bounds = peer.address_range();
            bounds.0 = bounds.0.max(peer_bounds.0);
            bounds.1 = bounds.1.min(peer_bounds.1);
            if bounds.0 > bounds.1 {
                return Err(error(
                    ErrorKind::Unsupported,
                    "allocation devices have no common GPU address range",
                ));
            }
            gpu_ids.try_push(peer.gpu_id)?;
            peer_vms.try_push(PeerVm { vm: peer })?;
        }
        let permission_flags = if permissions.contains(DeviceAccess::WRITE) {
            uapi::WRITABLE
        } else {
            0
        } | if permissions.contains(DeviceAccess::EXECUTE) {
            uapi::EXECUTABLE
        } else {
            0
        };
        let size = usize::try_from(desc.size).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "allocation size exceeds host width",
            )
        })?;
        let alignment = usize::try_from(desc.alignment)
            .map_err(|_| error(ErrorKind::InvalidArgument, "alignment exceeds host width"))?;
        let (mut host_address, device_byte_offset, mut mmap_offset) =
            allocation_source(kind, size)?;
        vm.check()?;
        let owned_userptr = matches!(kind, BufferKind::OwnedUserptr { .. });
        let reservation = if owned_userptr {
            sys::Reservation::new_shared_host(size, alignment, bounds)
        } else {
            sys::Reservation::new(size, alignment, bounds, false)
        }
        .map_err(|source| native_error("GPU address reservation", source))?;
        if owned_userptr {
            host_address = Some(reservation.address());
            mmap_offset = reservation.address() as u64;
        }
        // Establish the concrete owner before KFD can create a buffer. Every
        // later failure, including incomplete creation, goes through its Drop.
        let allocator = vm.allocator();
        let origin_gpu_id = vm.gpu_id;
        let mut allocation = Owned::new(
            Self {
                vm,
                peers: peer_vms,
                gpu_ids,
                reservation: Some(reservation),
                handle: None,
                host_address,
                device_byte_offset,
                logical_size: desc.size,
                origin_gpu_id,
                native_flags: 0,
                physical_backing_id: [0; 2],
                metadata: Buffer::new(allocator),
                scratch_range: None,
                mapping: GpuMapping::NotStarted,
                freeing: false,
                uncertain: false,
            },
            allocator,
        )?;
        let mut args = uapi::AllocMemory {
            va: allocation.address()? as u64,
            size: desc.size,
            mmap_offset,
            gpu_id: allocation.vm.gpu_id,
            flags: permission_flags
                | match kind {
                    BufferKind::Vram {
                        public,
                        coherent,
                        uncached,
                        contiguous,
                    } => {
                        uapi::VRAM
                            | uapi::NO_SUBSTITUTE
                            | if public { uapi::PUBLIC } else { 0 }
                            | if coherent { uapi::COHERENT } else { 0 }
                            | if uncached { uapi::UNCACHED } else { 0 }
                            | if contiguous { uapi::CONTIGUOUS } else { 0 }
                    }
                    BufferKind::Gtt => {
                        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE
                    }
                    BufferKind::Mmio => uapi::MMIO_REMAP | uapi::COHERENT,
                    BufferKind::OwnedUserptr { uncached } => {
                        uapi::USERPTR
                            | uapi::COHERENT
                            | uapi::NO_SUBSTITUTE
                            | if uncached { uapi::UNCACHED } else { 0 }
                    }
                    BufferKind::Userptr { uncached, .. } => {
                        uapi::USERPTR
                            | uapi::NO_SUBSTITUTE
                            | if uncached {
                                uapi::UNCACHED
                            } else {
                                uapi::COHERENT
                            }
                    }
                },
            ..uapi::AllocMemory::default()
        };
        let result = if matches!(kind, BufferKind::Mmio) {
            allocation.vm.loss.kfd.allocate_mmio(&mut args)
        } else {
            allocation.vm.loss.kfd.allocate(&mut args)
        };
        allocation.native_flags = args.flags;
        allocation.handle = (args.handle != 0).then_some(args.handle);
        // A copy fault is unexpected with our live writable ioctl body. If one
        // occurs, the kernel may have allocated a buffer without returning its
        // handle. Do not recycle a VA that could still belong to that buffer.
        allocation.uncertain = result
            .as_ref()
            .err()
            .is_some_and(|source| source.raw_os_error() == Some(14));
        result.map_err(|source| native_error("AMDKFD_IOC_ALLOC_MEMORY_OF_GPU", source))?;
        if allocation.handle.is_none() {
            allocation.uncertain = true;
            return Err(error(
                ErrorKind::DriverContract,
                "KFD allocation succeeded without a handle",
            ));
        }
        if matches!(kind, BufferKind::Mmio) {
            let allocation_ref = &mut *allocation;
            allocation_ref
                .vm
                .loss
                .kfd
                .map_mmio(
                    allocation_ref.reservation.as_mut().ok_or_else(|| {
                        error(
                            ErrorKind::Internal,
                            "MMIO allocation lost its VA reservation",
                        )
                    })?,
                    args.mmap_offset,
                )
                .map_err(|source| native_error("KFD MMIO CPU mmap", source))?;
            allocation.host_address = Some(allocation.address()?);
        } else if matches!(
            kind,
            BufferKind::Gtt | BufferKind::Vram { public: true, .. }
        ) {
            let allocation_ref = &mut *allocation;
            allocation_ref
                .reservation
                .as_mut()
                .ok_or_else(|| error(ErrorKind::Internal, "allocation lost its VA reservation"))?
                .map_render(allocation_ref.vm.render()?, args.mmap_offset)
                .map_err(|source| native_error("KFD buffer CPU mmap", source))?;
            allocation.host_address = Some(allocation.address()?);
        }
        allocation.mapping = GpuMapping::Mapping(0);
        allocation.finish_map()?;
        allocation.vm.check()?;
        Ok(allocation)
    }

    pub(super) fn create_mmio(vm: &Shared<DeviceVm>) -> Result<Owned<Self>, Error> {
        let page = util::page_size().map_err(|source| native_error("native page size", source))?;
        Self::create(
            vm.clone(),
            AllocationDesc {
                size: page as u64,
                alignment: page as u64,
            },
            BufferKind::Mmio,
            DeviceAccess::READ | DeviceAccess::WRITE,
        )
    }

    pub(super) fn create_scratch(
        vm: &Shared<DeviceVm>,
        desc: AllocationDesc,
    ) -> Result<Owned<Self>, Error> {
        let size = usize::try_from(desc.size).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "scratch size exceeds the host address width",
            )
        })?;
        if desc.alignment > 4096 || size == 0 || size % 4096 != 0 {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid native scratch extent or alignment",
            ));
        }
        let allocator = vm.allocator();
        let origin_gpu_id = vm.gpu_id;
        let peers = Buffer::new(allocator);
        let mut gpu_ids = Buffer::try_with_capacity(1, allocator)?;
        gpu_ids.try_push(vm.gpu_id)?;
        let address = vm.allocate_scratch(desc.size)?;
        let reservation = sys::Reservation::view(
            usize::try_from(address).map_err(|_| {
                error(
                    ErrorKind::ResourceExhausted,
                    "scratch address exceeds the host address width",
                )
            })?,
            size,
        );
        let mut allocation = match Owned::new(
            Self {
                vm: vm.clone(),
                peers,
                gpu_ids,
                reservation: Some(reservation),
                handle: None,
                host_address: None,
                device_byte_offset: 0,
                logical_size: desc.size,
                origin_gpu_id,
                native_flags: 0,
                physical_backing_id: [0; 2],
                metadata: Buffer::new(allocator),
                scratch_range: Some((address, desc.size)),
                mapping: GpuMapping::NotStarted,
                freeing: false,
                uncertain: false,
            },
            allocator,
        ) {
            Ok(allocation) => allocation,
            Err(source) => {
                vm.release_scratch(address, desc.size)?;
                return Err(source.into());
            }
        };
        let mut args = uapi::AllocMemory {
            va: address,
            size: desc.size,
            gpu_id: allocation.vm.gpu_id,
            flags: uapi::VRAM | uapi::WRITABLE | uapi::NO_SUBSTITUTE,
            ..uapi::AllocMemory::default()
        };
        let result = allocation.vm.loss.kfd.allocate_scratch(&mut args);
        allocation.native_flags = args.flags;
        allocation.handle = (args.handle != 0).then_some(args.handle);
        allocation.uncertain = result
            .as_ref()
            .err()
            .is_some_and(|source| source.raw_os_error() == Some(14));
        result.map_err(|source| native_error("AMDKFD_IOC_ALLOC_MEMORY_OF_GPU", source))?;
        if allocation.handle.is_none() {
            allocation.uncertain = true;
            return Err(error(
                ErrorKind::DriverContract,
                "KFD scratch allocation succeeded without a handle",
            ));
        }
        let allocation_ref = &mut *allocation;
        allocation_ref
            .reservation
            .as_mut()
            .ok_or_else(|| error(ErrorKind::Internal, "scratch allocation lost its VA view"))?
            .map_render_inaccessible(allocation_ref.vm.render()?, args.mmap_offset)
            .map_err(|source| native_error("KFD scratch buffer CPU mmap", source))?;
        allocation.mapping = GpuMapping::Mapping(0);
        allocation.finish_map()?;
        allocation.vm.check()?;
        Ok(allocation)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "validation, native acquisition, and cleanup ownership form one transaction"
    )]
    pub(super) fn import_dma_buf(
        vm: Shared<DeviceVm>,
        descriptor: i32,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self>, Error> {
        let required_permissions = DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE;
        if permissions != required_permissions {
            return Err(error(
                ErrorKind::Unsupported,
                "DMA-BUF import requires read/write/execute GPU access",
            ));
        }
        let page = util::page_size().map_err(|source| native_error("native page size", source))?;
        let alignment = usize::try_from(alignment)
            .map_err(|_| error(ErrorKind::InvalidArgument, "alignment exceeds host width"))?;
        if byte_length == 0
            || !alignment.is_power_of_two()
            || alignment < page
            || source_offset % alignment as u64 != 0
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid DMA-BUF logical extent or alignment",
            ));
        }
        let dma_buf = util::duplicate_file(descriptor)
            .map_err(|source| native_error("DMA-BUF descriptor duplication", source))?;
        let file_info = util::dma_buf_file_info(&dma_buf)
            .map_err(|source| native_error("DMA-BUF file information", source))?;
        vm.check()?;
        let details = vm
            .loss
            .kfd
            .dma_buf_info(dma_buf.as_raw_fd())
            .map_err(|source| native_error("AMDKFD_IOC_GET_DMABUF_INFO", source))?;
        let native_info = details.info;
        if native_info.gpu_id != vm.gpu_id {
            return Err(error(
                ErrorKind::Unsupported,
                "DMA-BUF originates from a different GPU",
            ));
        }
        if native_info.flags & !uapi::PUBLIC != uapi::GTT {
            return Err(error(
                ErrorKind::Unsupported,
                "DMA-BUF is not supported SYSTEM/GTT storage",
            ));
        }
        if native_info.size != file_info.size {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD and DMA-BUF report different backing extents",
            ));
        }
        let native_size = usize::try_from(native_info.size).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "DMA-BUF backing exceeds host address width",
            )
        })?;
        if native_size == 0
            || native_size % page != 0
            || source_offset
                .checked_add(byte_length)
                .is_none_or(|end| end > native_info.size)
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "DMA-BUF logical range exceeds its backing",
            ));
        }
        let device_byte_offset = usize::try_from(source_offset).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "DMA-BUF source offset exceeds host address width",
            )
        })?;
        let reservation = sys::Reservation::new(native_size, alignment, (vm.base, vm.limit), false)
            .map_err(|source| native_error("DMA-BUF GPU address reservation", source))?;
        let allocator = vm.allocator();
        let peers = Buffer::new(allocator);
        let mut gpu_ids = Buffer::try_with_capacity(1, allocator)?;
        gpu_ids.try_push(vm.gpu_id)?;
        let mut allocation = Owned::new(
            Self {
                vm,
                peers,
                gpu_ids,
                reservation: Some(reservation),
                handle: None,
                host_address: None,
                device_byte_offset,
                logical_size: byte_length,
                origin_gpu_id: native_info.gpu_id,
                native_flags: native_info.flags,
                physical_backing_id: file_info.physical_id,
                metadata: details.metadata,
                scratch_range: None,
                mapping: GpuMapping::NotStarted,
                freeing: false,
                uncertain: false,
            },
            allocator,
        )?;
        let mut args = uapi::ImportDmaBuf {
            va: allocation.address()? as u64,
            gpu_id: allocation.vm.gpu_id,
            descriptor: u32::try_from(dma_buf.as_raw_fd()).map_err(|_| {
                error(
                    ErrorKind::Internal,
                    "duplicated DMA-BUF descriptor is invalid",
                )
            })?,
            ..uapi::ImportDmaBuf::default()
        };
        let result = allocation.vm.loss.kfd.import_dma_buf(&mut args);
        allocation.handle = (args.handle != 0).then_some(args.handle);
        allocation.uncertain = result
            .as_ref()
            .err()
            .is_some_and(|source| source.raw_os_error() == Some(14));
        result.map_err(|source| native_error("AMDKFD_IOC_IMPORT_DMABUF", source))?;
        if allocation.handle.is_none() {
            allocation.uncertain = true;
            return Err(error(
                ErrorKind::DriverContract,
                "KFD import succeeded without a handle",
            ));
        }
        let logical_host = allocation
            .address()?
            .checked_add(device_byte_offset)
            .ok_or_else(|| error(ErrorKind::Internal, "DMA-BUF host address overflow"))?;
        allocation
            .reservation
            .as_mut()
            .ok_or_else(|| error(ErrorKind::Internal, "DMA-BUF lost its VA reservation"))?
            .map_dma_buf(&dma_buf)
            .map_err(|source| native_error("DMA-BUF CPU mmap", source))?;
        allocation.host_address = Some(logical_host);
        allocation.mapping = GpuMapping::Mapping(0);
        allocation.finish_map()?;
        allocation.vm.check()?;
        Ok(allocation)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "graphics import validates a foreign backing and owns every mapping step"
    )]
    pub(super) fn import_graphics_dma_buf(
        vm: Shared<DeviceVm>,
        peers: impl ExactSizeIterator<Item = Shared<DeviceVm>>,
        descriptor: i32,
        _size_hint: u64,
    ) -> Result<Owned<Self>, Error> {
        let page = util::page_size().map_err(|source| native_error("native page size", source))?;
        let peer_count = peers.len();
        let allocator = vm.allocator();
        let mut peer_vms: Buffer<PeerVm> = Buffer::try_with_capacity(peer_count, allocator)?;
        let mut gpu_ids = Buffer::try_with_capacity(
            peer_count
                .checked_add(1)
                .ok_or_else(|| error(ErrorKind::ResourceExhausted, "device count overflow"))?,
            allocator,
        )?;
        gpu_ids.try_push(vm.gpu_id)?;
        let mut bounds = vm.address_range();
        vm.check()?;
        for peer in peers {
            if !Shared::ptr_eq(&vm.loss.kfd, &peer.loss.kfd) {
                return Err(error(
                    ErrorKind::InvalidArgument,
                    "graphics import devices must share one KFD session",
                ));
            }
            if Shared::ptr_eq(&vm, &peer)
                || peer_vms
                    .iter()
                    .any(|retained| Shared::ptr_eq(&retained.vm, &peer))
            {
                continue;
            }
            if peer.gpu_id == vm.gpu_id || gpu_ids.iter().any(|gpu_id| *gpu_id == peer.gpu_id) {
                return Err(error(
                    ErrorKind::InvalidArgument,
                    "distinct VM bindings must have distinct GPU IDs",
                ));
            }
            peer.check()?;
            let peer_bounds = peer.address_range();
            bounds.0 = bounds.0.max(peer_bounds.0);
            bounds.1 = bounds.1.min(peer_bounds.1);
            if bounds.0 > bounds.1 {
                return Err(error(
                    ErrorKind::Unsupported,
                    "graphics import devices have no common GPU address range",
                ));
            }
            gpu_ids.try_push(peer.gpu_id)?;
            peer_vms.try_push(PeerVm { vm: peer })?;
        }

        let dma_buf = util::duplicate_file(descriptor)
            .map_err(|source| native_error("DMA-BUF descriptor duplication", source))?;
        let file_info = util::dma_buf_file_info(&dma_buf)
            .map_err(|source| native_error("DMA-BUF file information", source))?;
        let details = vm
            .loss
            .kfd
            .dma_buf_info(dma_buf.as_raw_fd())
            .map_err(|source| native_error("AMDKFD_IOC_GET_DMABUF_INFO", source))?;
        let native_info = details.info;
        let placement = native_info.flags & (uapi::VRAM | uapi::GTT);
        if native_info.gpu_id == 0 || !matches!(placement, uapi::VRAM | uapi::GTT) {
            return Err(error(
                ErrorKind::Unsupported,
                "DMA-BUF has unsupported graphics storage",
            ));
        }
        if native_info.size != file_info.size {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD and DMA-BUF report different backing extents",
            ));
        }
        let native_size = usize::try_from(native_info.size).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "DMA-BUF backing exceeds host address width",
            )
        })?;
        if native_size == 0 || native_size % page != 0 {
            return Err(error(
                ErrorKind::InvalidArgument,
                "DMA-BUF backing is not page aligned",
            ));
        }
        let reservation =
            sys::Reservation::new(native_size, GRAPHICS_IMPORT_ALIGNMENT, bounds, false).map_err(
                |source| native_error("graphics DMA-BUF GPU address reservation", source),
            )?;
        let mut allocation = Owned::new(
            Self {
                vm,
                peers: peer_vms,
                gpu_ids,
                reservation: Some(reservation),
                handle: None,
                host_address: None,
                device_byte_offset: 0,
                logical_size: native_info.size,
                origin_gpu_id: native_info.gpu_id,
                native_flags: native_info.flags,
                physical_backing_id: file_info.physical_id,
                metadata: details.metadata,
                scratch_range: None,
                mapping: GpuMapping::NotStarted,
                freeing: false,
                uncertain: false,
            },
            allocator,
        )?;
        let mut args = uapi::ImportDmaBuf {
            va: allocation.address()? as u64,
            gpu_id: native_info.gpu_id,
            descriptor: u32::try_from(dma_buf.as_raw_fd()).map_err(|_| {
                error(
                    ErrorKind::Internal,
                    "duplicated DMA-BUF descriptor is invalid",
                )
            })?,
            ..uapi::ImportDmaBuf::default()
        };
        let result = allocation.vm.loss.kfd.import_dma_buf(&mut args);
        allocation.handle = (args.handle != 0).then_some(args.handle);
        allocation.uncertain = result
            .as_ref()
            .err()
            .is_some_and(|source| source.raw_os_error() == Some(14));
        result.map_err(|source| native_error("AMDKFD_IOC_IMPORT_DMABUF", source))?;
        if allocation.handle.is_none() {
            allocation.uncertain = true;
            return Err(error(
                ErrorKind::DriverContract,
                "KFD graphics import succeeded without a handle",
            ));
        }
        if placement == uapi::GTT || native_info.flags & uapi::PUBLIC != 0 {
            let address = allocation.address()?;
            allocation
                .reservation
                .as_mut()
                .ok_or_else(|| {
                    error(
                        ErrorKind::Internal,
                        "graphics import lost its VA reservation",
                    )
                })?
                .map_dma_buf(&dma_buf)
                .map_err(|source| native_error("graphics DMA-BUF CPU mmap", source))?;
            allocation.host_address = Some(address);
        }
        allocation.mapping = GpuMapping::Mapping(0);
        allocation.finish_map()?;
        allocation.vm.check()?;
        Ok(allocation)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "IPC validation, common-VA selection, import, and mapping form one transaction"
    )]
    pub(super) fn import_ipc(
        vm: Shared<DeviceVm>,
        mapping_vms: impl ExactSizeIterator<Item = Shared<DeviceVm>>,
        words: [u32; 8],
        logical_size: u64,
    ) -> Result<Owned<Self>, Error> {
        let aperture = words[4];
        if words[..4] == [0; 4]
            || !matches!(
                aperture,
                IPC_APERTURE_DGPU | IPC_APERTURE_DGPU_ALT | IPC_APERTURE_GPUVM
            )
            || (aperture != IPC_APERTURE_GPUVM && words[5] != 0)
            || words[6] & IPC_FRAGMENT != 0
            || words[7] == 0
            || words[7] != vm.gpu_id
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid or unsupported KFD IPC memory handle",
            ));
        }
        let native_size = u64::from(words[6])
            .checked_mul(IPC_PAGE_SIZE)
            .filter(|size| *size != 0)
            .ok_or_else(|| error(ErrorKind::InvalidArgument, "invalid IPC memory extent"))?;
        if logical_size == 0 || logical_size > native_size {
            return Err(error(
                ErrorKind::InvalidArgument,
                "IPC logical extent exceeds its native backing",
            ));
        }
        let native_size_usize = usize::try_from(native_size).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "IPC memory extent exceeds host address width",
            )
        })?;

        let mapping_count = mapping_vms.len();
        let allocator = vm.allocator();
        let mut peers: Buffer<PeerVm> = Buffer::try_with_capacity(mapping_count, allocator)?;
        let mut gpu_ids = Buffer::try_with_capacity(mapping_count, allocator)?;
        let mut bounds = vm.address_range();
        vm.check()?;
        for mapping_vm in mapping_vms {
            if !Shared::ptr_eq(&vm.loss.kfd, &mapping_vm.loss.kfd) {
                return Err(error(
                    ErrorKind::InvalidArgument,
                    "IPC mapping devices must share one KFD session",
                ));
            }
            if gpu_ids.iter().any(|gpu_id| *gpu_id == mapping_vm.gpu_id) {
                continue;
            }
            mapping_vm.check()?;
            let mapping_bounds = mapping_vm.address_range();
            bounds.0 = bounds.0.max(mapping_bounds.0);
            bounds.1 = bounds.1.min(mapping_bounds.1);
            if bounds.0 > bounds.1 {
                return Err(error(
                    ErrorKind::Unsupported,
                    "IPC mapping devices have no common GPU address range",
                ));
            }
            gpu_ids.try_push(mapping_vm.gpu_id)?;
            if !Shared::ptr_eq(&vm, &mapping_vm) {
                peers.try_push(PeerVm { vm: mapping_vm })?;
            }
        }

        let reservation = sys::Reservation::new(native_size_usize, 4096, bounds, false)
            .map_err(|source| native_error("IPC memory address reservation", source))?;
        let mut allocation = Owned::new(
            Self {
                vm,
                peers,
                gpu_ids,
                reservation: Some(reservation),
                handle: None,
                host_address: None,
                device_byte_offset: 0,
                logical_size,
                origin_gpu_id: words[7],
                native_flags: 0,
                physical_backing_id: [0; 2],
                metadata: Buffer::new(allocator),
                scratch_range: None,
                mapping: GpuMapping::NotStarted,
                freeing: false,
                uncertain: false,
            },
            allocator,
        )?;
        let mut args = uapi::IpcImportHandle {
            va_addr: allocation.address()? as u64,
            share_handle: [words[0], words[1], words[2], words[3]],
            gpu_id: words[7],
            ..uapi::IpcImportHandle::default()
        };
        let result = allocation.vm.loss.kfd.import_ipc_handle(&mut args);
        allocation.handle = (args.handle != 0).then_some(args.handle);
        allocation.native_flags = args.flags;
        allocation.uncertain = result
            .as_ref()
            .err()
            .is_some_and(|source| source.raw_os_error() == Some(14));
        result.map_err(|source| native_error("AMDKFD_IOC_IPC_IMPORT_HANDLE", source))?;
        if allocation.handle.is_none() {
            allocation.uncertain = true;
            return Err(error(
                ErrorKind::DriverContract,
                "KFD IPC import succeeded without a handle",
            ));
        }
        if args.mmap_offset != 0 {
            let address = allocation.address()?;
            let allocation_ref = &mut *allocation;
            allocation_ref
                .reservation
                .as_mut()
                .ok_or_else(|| error(ErrorKind::Internal, "IPC import lost its VA reservation"))?
                .map_render(allocation_ref.vm.render()?, args.mmap_offset)
                .map_err(|source| native_error("KFD IPC CPU mmap", source))?;
            allocation.host_address = Some(address);
        }
        if allocation.gpu_ids.is_empty() {
            allocation.mapping = GpuMapping::Mapped;
        } else {
            allocation.mapping = GpuMapping::Mapping(0);
            allocation.finish_map()?;
        }
        allocation.vm.check()?;
        Ok(allocation)
    }

    pub(crate) fn cached_info(&self) -> AllocationInfo {
        AllocationInfo {
            device_address: self.reservation.as_ref().map_or(0, |reservation| {
                reservation.address() as u64 + self.device_byte_offset as u64
            }),
            host_address: self.host_address,
            size: self.logical_size,
            native_size: self
                .reservation
                .as_ref()
                .map_or(0, |reservation| reservation.usable_size() as u64),
            physical_backing_id: self.physical_backing_id,
        }
    }

    pub(super) fn is_owned_by(&self, device: &Shared<DeviceVm>) -> bool {
        self.vm.shares_kfd(device) && self.origin_gpu_id == device.gpu_id
    }

    pub(crate) fn metadata(&self) -> &[u8] {
        self.metadata.as_slice()
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        self.check_address()
    }

    pub(super) fn signal_event_page_handle(&self, device: &Shared<DeviceVm>) -> Result<u64, Error> {
        self.check_address()?;
        if !self.vm.shares_kfd(device)
            || self.device_byte_offset != 0
            || self.host_address.is_none()
            || self.logical_size < u64::from(uapi::SIGNAL_EVENT_LIMIT) * 8
            || self.native_flags & (uapi::VRAM | uapi::GTT) != uapi::GTT
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid KFD signal event page allocation",
            ));
        }
        self.handle.ok_or_else(|| {
            error(
                ErrorKind::DriverContract,
                "signal event page has no KFD allocation handle",
            )
        })
    }

    pub(super) fn retain_signal_event_page(&mut self) -> Result<(), Error> {
        if self.freeing
            || !matches!(self.mapping, GpuMapping::Mapped)
            || self.device_byte_offset != 0
            || self.host_address.is_none()
            || self.logical_size < u64::from(uapi::SIGNAL_EVENT_LIMIT) * 8
            || self.native_flags & (uapi::VRAM | uapi::GTT) != uapi::GTT
            || self.handle.is_none()
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid KFD signal event page retention",
            ));
        }
        let reservation = self.reservation.take().ok_or_else(|| {
            error(
                ErrorKind::DriverContract,
                "signal event page lost its address reservation",
            )
        })?;
        std::mem::forget(reservation);
        self.handle = None;
        self.mapping = GpuMapping::Unmapped;
        self.gpu_ids.clear();
        self.peers.clear();
        self.freeing = true;
        self.uncertain = false;
        Ok(())
    }

    pub(super) fn read_indices(
        &self,
        read_offset: usize,
        write_offset: usize,
    ) -> Result<(u64, u64), Error> {
        self.reservation
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "queue pointer mapping is missing"))?
            .read_indices(read_offset, write_offset)
            .map_err(|e| native_error("queue index read", e))
    }

    pub(super) fn read_wrapping_indices(
        &self,
        read_offset: usize,
        write_offset: usize,
        read_mask: u64,
    ) -> Result<(u64, u64), Error> {
        self.reservation
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "queue pointer mapping is missing"))?
            .read_wrapping_indices(read_offset, write_offset, read_mask)
            .map_err(|e| native_error("queue index read", e))
    }

    /// Queue construction initializes only owned CPU-visible backing, before
    /// `CREATE_QUEUE` can expose it to the device. These methods never map memory.
    pub(super) fn zero(&mut self) -> Result<(), Error> {
        self.host_address()?;
        self.reservation
            .as_mut()
            .ok_or_else(|| error(ErrorKind::Internal, "missing queue storage"))?
            .zero()
            .map_err(|source| native_error("queue backing initialization", source))
    }

    pub(super) fn write_bytes(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        self.host_address()?;
        self.reservation
            .as_mut()
            .ok_or_else(|| error(ErrorKind::Internal, "missing queue storage"))?
            .write_bytes(offset, bytes)
            .map_err(|source| native_error("queue backing initialization", source))
    }

    pub(super) fn fill_records(&mut self, record: &[u8], count: usize) -> Result<(), Error> {
        self.host_address()?;
        self.reservation
            .as_mut()
            .ok_or_else(|| error(ErrorKind::Internal, "missing queue storage"))?
            .fill_records(record, count)
            .map_err(|source| native_error("queue ring initialization", source))
    }

    fn address(&self) -> Result<usize, Error> {
        self.reservation
            .as_ref()
            .map(sys::Reservation::address)
            .ok_or_else(|| {
                error(
                    ErrorKind::InvalidArgument,
                    "GPU allocation has no live address reservation",
                )
            })
    }

    fn transfer(&mut self) -> Result<(), Error> {
        let handle = self.handle.ok_or_else(|| {
            error(
                ErrorKind::DriverContract,
                "KFD mapping has no allocation handle",
            )
        })?;
        let (map, progress) = match &mut self.mapping {
            GpuMapping::Mapping(progress) => (true, progress),
            GpuMapping::Unmapping(progress) => (false, progress),
            _ => {
                return Err(error(
                    ErrorKind::Internal,
                    "KFD transfer has no pending mapping phase",
                ));
            }
        };
        let result = self
            .vm
            .loss
            .kfd
            .transfer(handle, self.gpu_ids.as_slice(), progress, map);
        if result.as_ref().err().is_some_and(|source| {
            source.kind() == io::ErrorKind::InvalidData || source.raw_os_error() == Some(14)
        }) {
            self.uncertain = true;
        }
        result.map_err(|source| {
            native_error(
                if map {
                    "AMDKFD_IOC_MAP_MEMORY_TO_GPU"
                } else {
                    "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU"
                },
                source,
            )
        })
    }

    fn finish_map(&mut self) -> Result<(), Error> {
        self.transfer()?;
        self.mapping = GpuMapping::Mapped;
        Ok(())
    }

    fn check_address(&self) -> Result<(), Error> {
        self.vm.check()?;
        if self.freeing || !matches!(self.mapping, GpuMapping::Mapped) {
            Err(error(
                ErrorKind::Unsupported,
                "GPU allocation is not available for access",
            ))
        } else {
            Ok(())
        }
    }
}

impl KfdAllocation {
    pub(super) fn peer_mapping_source(
        &self,
        device: &Shared<DeviceVm>,
    ) -> Result<(u64, u64), Error> {
        self.check_address()?;
        if !self.vm.shares_kfd(device) {
            return Err(error(
                ErrorKind::InvalidArgument,
                "queue producer must share one KFD session",
            ));
        }
        device.check()?;
        let address = self.address()? as u64;
        let size = self
            .reservation
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "GPU allocation lost its reservation"))?
            .usable_size() as u64;
        let end = address
            .checked_add(size.saturating_sub(1))
            .ok_or_else(|| error(ErrorKind::Internal, "GPU allocation address overflow"))?;
        let (base, limit) = device.address_range();
        if address < base || end > limit {
            return Err(error(
                ErrorKind::Unsupported,
                "queue backing is outside the producer GPU address range",
            ));
        }
        let handle = self.handle.ok_or_else(|| {
            error(
                ErrorKind::DriverContract,
                "queue backing has no KFD allocation handle",
            )
        })?;
        Ok((handle, address + self.device_byte_offset as u64))
    }

    pub(super) fn device_address(&self, device: &Shared<DeviceVm>) -> Result<u64, Error> {
        self.check_address()?;
        if !self.vm.shares_kfd(device)
            || !self.gpu_ids.iter().any(|gpu_id| *gpu_id == device.gpu_id)
        {
            return Err(error(
                ErrorKind::Unsupported,
                "allocation is not mapped to this device",
            ));
        }
        self.address()
            .map(|address| address as u64 + self.device_byte_offset as u64)
    }

    pub(super) fn host_address(&self) -> Result<usize, Error> {
        self.check_address()?;
        self.host_address
            .ok_or_else(|| error(ErrorKind::Unsupported, "private VRAM has no CPU mapping"))
    }

    pub(super) fn export_dma_buf(&self) -> Result<DmaBuf, Error> {
        self.check_address()?;
        if self.scratch_range.is_some() || self.native_flags & uapi::MMIO_REMAP != 0 {
            return Err(error(
                ErrorKind::Unsupported,
                "private runtime backing cannot be exported",
            ));
        }
        let handle = self.handle.ok_or_else(|| {
            error(
                ErrorKind::DriverContract,
                "DMA-BUF export has no allocation handle",
            )
        })?;
        let file = self
            .vm
            .loss
            .kfd
            .export_dma_buf(handle)
            .map_err(|source| native_error("AMDKFD_IOC_EXPORT_DMABUF", source))?;
        let file_info = util::dma_buf_file_info(&file)
            .map_err(|source| native_error("exported DMA-BUF file information", source))?;
        let expected_size = self
            .reservation
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "DMA-BUF export lost its reservation"))?
            .usable_size() as u64;
        if file_info.size != expected_size {
            return Err(error(
                ErrorKind::DriverContract,
                "exported DMA-BUF has an unexpected backing extent",
            ));
        }
        if self.physical_backing_id != [0; 2] && file_info.physical_id != self.physical_backing_id {
            return Err(error(
                ErrorKind::DriverContract,
                "re-exported DMA-BUF changed physical identity",
            ));
        }
        Ok(DmaBuf::new(
            file.into(),
            DmaBufInfo {
                byte_length: file_info.size,
                source_offset: self.device_byte_offset as u64,
                physical_backing_id: file_info.physical_id,
            },
        ))
    }

    pub(super) fn export_ipc_memory(&self) -> Result<KfdIpcMemoryHandle, Error> {
        self.check_address()?;
        if self.scratch_range.is_some()
            || self.device_byte_offset != 0
            || !matches!(
                self.native_flags & (uapi::VRAM | uapi::GTT),
                uapi::VRAM | uapi::GTT
            )
        {
            return Err(error(
                ErrorKind::Unsupported,
                "allocation backing cannot be exported through KFD IPC",
            ));
        }
        let handle = self.handle.ok_or_else(|| {
            error(
                ErrorKind::DriverContract,
                "IPC export has no allocation handle",
            )
        })?;
        let native_size = self
            .reservation
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "IPC export lost its reservation"))?
            .usable_size() as u64;
        let pages = native_size
            .checked_div(IPC_PAGE_SIZE)
            .and_then(|pages| u32::try_from(pages).ok())
            .filter(|pages| *pages != 0 && *pages & IPC_FRAGMENT == 0)
            .ok_or_else(|| error(ErrorKind::InvalidArgument, "IPC allocation is too large"))?;
        if native_size % IPC_PAGE_SIZE != 0 {
            return Err(error(
                ErrorKind::DriverContract,
                "IPC allocation backing is not page aligned",
            ));
        }
        let share_handle = self
            .vm
            .loss
            .kfd
            .export_ipc_handle(handle, self.origin_gpu_id, self.native_flags)
            .map_err(|source| native_error("AMDKFD_IOC_IPC_EXPORT_HANDLE", source))?;
        Ok(KfdIpcMemoryHandle::from_words([
            share_handle[0],
            share_handle[1],
            share_handle[2],
            share_handle[3],
            IPC_APERTURE_DGPU,
            0,
            pages,
            self.origin_gpu_id,
        ]))
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        self.vm
            .loss
            .kfd
            .check_process()
            .map_err(|source| native_error("KFD allocation free", source))?;
        self.freeing = true;
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                concat!(
                    "KFD allocation outcome is uncertain; ",
                    "native state is retained until process teardown"
                ),
            ));
        }
        if matches!(self.mapping, GpuMapping::Mapping(_)) {
            self.finish_map()?;
        }
        if matches!(self.mapping, GpuMapping::Mapped) {
            self.mapping = if self.gpu_ids.is_empty() {
                GpuMapping::Unmapped
            } else {
                GpuMapping::Unmapping(0)
            };
        }
        if matches!(self.mapping, GpuMapping::Unmapping(_)) {
            self.transfer()?;
            self.mapping = GpuMapping::Unmapped;
        }
        if let Some(handle) = self.handle {
            self.vm
                .loss
                .kfd
                .free(handle)
                .map_err(|source| native_error("AMDKFD_IOC_FREE_MEMORY_OF_GPU", source))?;
            self.handle = None;
        }
        if let Some(reservation) = &mut self.reservation {
            reservation
                .release()
                .map_err(|source| native_error("GPU address munmap", source))?;
            self.reservation = None;
        }
        if let Some((address, size)) = self.scratch_range {
            self.vm.release_scratch(address, size)?;
            self.scratch_range = None;
        }
        Ok(())
    }
}

impl Drop for KfdAllocation {
    fn drop(&mut self) {
        if self.free().is_err() {
            // A still-live BO must never outlive its VA reservation or render
            // dependencies. Kernel process teardown is the final cleanup owner;
            // retaining these concrete resources requires no cleanup allocation.
            if let Some(reservation) = self.reservation.take() {
                std::mem::forget(reservation);
            }
            std::mem::forget(self.vm.clone());
            for peer in &self.peers {
                std::mem::forget(peer.vm.clone());
            }
        }
    }
}

pub(super) fn error(kind: ErrorKind, detail: &'static str) -> Error {
    Error::Operation { kind, detail }
}

pub(super) fn native_error(operation: &'static str, source: io::Error) -> Error {
    let kind = match source.kind() {
        io::ErrorKind::PermissionDenied => ErrorKind::PermissionDenied,
        io::ErrorKind::OutOfMemory => ErrorKind::ResourceExhausted,
        io::ErrorKind::Unsupported => ErrorKind::Unsupported,
        io::ErrorKind::InvalidData => ErrorKind::DriverContract,
        _ => match source.raw_os_error() {
            Some(11 | 16) => ErrorKind::Busy,
            Some(12 | 28) => ErrorKind::ResourceExhausted,
            Some(25 | 95) => ErrorKind::Unsupported,
            _ => ErrorKind::Driver,
        },
    };
    Error::NativeOperation {
        kind,
        operation,
        source,
    }
}

#[cfg(test)]
#[path = "tests/memory.rs"]
mod tests;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
pub(super) fn queue_fixture(
    kfd: Shared<sys::Kfd>,
    render: File,
    node: sysfs::NativeNode,
) -> Shared<DeviceVm> {
    queue_fixture_with_range(kfd, render, node, (0x10000, isize::MAX as u64))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
pub(super) fn queue_fixture_with_range(
    kfd: Shared<sys::Kfd>,
    render: File,
    node: sysfs::NativeNode,
    bounds: (u64, u64),
) -> Shared<DeviceVm> {
    let allocator = kfd.allocator();
    let loss = Shared::new(
        LossEvent {
            kfd,
            hardware_event_id: AtomicU32::new(19),
            memory_event_id: AtomicU32::new(20),
            hardware_destroy_uncertain: AtomicBool::new(false),
            memory_destroy_uncertain: AtomicBool::new(false),
            memory_event_claimed: Mutex::new(false),
            lost: AtomicBool::new(false),
        },
        allocator,
    )
    .unwrap();
    Shared::new(
        DeviceVm {
            loss,
            render: Some(render),
            system_dma_buf_import: false,
            gpu_id: node.gpu_id,
            render_minor: node.render_minor.unwrap(),
            identity: node.identity,
            unique_id: node.unique_id,
            base: bounds.0,
            limit: bounds.1,
            lds_base: 0x1000_0000_0000,
            scratch_base: 0x2000_0000_0000,
            scratch: Mutex::new(ScratchPool::new(
                sysfs::NativeQueueProperties {
                    gfx_target: 120_001,
                    xcc_count: 1,
                    ..sysfs::NativeQueueProperties::default()
                },
                allocator,
            )),
            vmem: Mutex::new(super::vmem::VmState::new(allocator)),
            version: uapi::Version {
                major: 1,
                minor: 23,
            },
            doorbells: super::queue::Doorbells::default(),
        },
        allocator,
    )
    .unwrap()
}

#[cfg(test)]
#[path = "tests/shutdown.rs"]
mod shutdown_tests;
