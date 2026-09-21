//! Linux DRM/KFD virtual-memory ownership and mapping transactions.
//!
//! Virtual address reservations, physical memory handles, and host/device
//! mappings are independently owned so partial teardown can resume without
//! replaying successful native operations. DMA-BUF imports share one DRM GEM
//! handle per physical backing identity and reference-count that handle across
//! mappings. Timeline sync objects order DRM VM updates before KFD-visible use.
//!
//! Public owners expose cached addresses and extents, never Rust references into
//! mappings. A failed unmap or handle release leaves its native dependency live
//! and attached to the owner for an explicit retry.

use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::host_storage::{Allocator, Buffer, Owned, Shared};
use crate::memory::interop::linux::{DmaBuf, DmaBufInfo};
use crate::memory::{
    AllocationDesc, AllocationLimits, DeviceAccess, MemoryKind, VirtualMemoryInfo,
};
use crate::{Error, ErrorKind};

use super::memory::{DeviceVm, error, native_error};
use super::{drm, sys, uapi, util};

/// Imported GEM handle shared by mappings of the same physical backing.
struct ImportedGem {
    backing_id: [u64; 2],
    handle: u32,
    references: usize,
}

/// A failed submission can still have reached the driver.
struct SubmitError {
    error: Error,
    update: Option<(u32, u64)>,
}

/// Per-device DRM virtual-memory state serialized by its owning `DeviceVm`.
pub(super) struct VmState {
    syncobj: u32,
    next_point: u64,
    gems: Buffer<ImportedGem>,
    // A failed MAP has no public mapping owner. Keep its timeline point and
    // GEM handle in this VM; the reservation and backing quarantine themselves.
    pending_maps: Buffer<(u32, u64)>,
}

impl VmState {
    pub(super) fn new(allocator: Allocator) -> Self {
        Self {
            syncobj: 0,
            next_point: 0,
            gems: Buffer::new(allocator),
            pending_maps: Buffer::new(allocator),
        }
    }

    pub(super) fn close(&mut self, render: &File) -> Result<(), Error> {
        if !self.gems.is_empty() || !self.pending_maps.is_empty() {
            return Err(error(
                ErrorKind::DriverContract,
                "live virtual-memory mappings prevent render shutdown",
            ));
        }
        if self.syncobj != 0 {
            drm::destroy_syncobj(render, self.syncobj)
                .map_err(|source| native_error("DRM sync object destruction", source))?;
            self.syncobj = 0;
        }
        Ok(())
    }

    fn ensure_syncobj(&mut self, render: &File) -> io::Result<u32> {
        if self.syncobj == 0 {
            self.syncobj = drm::create_syncobj(render)?;
        }
        Ok(self.syncobj)
    }

    fn next_point(&mut self) -> io::Result<u64> {
        self.next_point = self
            .next_point
            .checked_add(1)
            .ok_or_else(|| io::Error::from(io::ErrorKind::OutOfMemory))?;
        Ok(self.next_point)
    }

    fn acquire_gem(
        &mut self,
        render: &File,
        dma_buf: &File,
        backing_id: [u64; 2],
    ) -> Result<u32, Error> {
        if let Some(gem) = self
            .gems
            .iter_mut()
            .find(|gem| gem.backing_id == backing_id)
        {
            gem.references = gem.references.checked_add(1).ok_or_else(|| {
                error(
                    ErrorKind::ResourceExhausted,
                    "DRM memory handle reference count overflow",
                )
            })?;
            return Ok(gem.handle);
        }
        self.gems.try_reserve(1)?;
        let handle = drm::import_dma_buf(render, dma_buf.as_raw_fd())
            .map_err(|source| native_error("DRM DMA-BUF import", source))?;
        if let Err(source) = self.gems.try_push(ImportedGem {
            backing_id,
            handle,
            references: 1,
        }) {
            let _ = drm::close_gem(render, handle);
            return Err(source.into());
        }
        Ok(handle)
    }

    fn release_gem(&mut self, render: &File, backing_id: [u64; 2]) -> Result<(), Error> {
        let index = self
            .gems
            .iter()
            .position(|gem| gem.backing_id == backing_id)
            .ok_or_else(|| {
                error(
                    ErrorKind::DriverContract,
                    "virtual-memory mapping lost its DRM memory handle",
                )
            })?;
        if self.gems[index].references > 1 {
            self.gems[index].references -= 1;
            return Ok(());
        }
        drm::close_gem(render, self.gems[index].handle)
            .map_err(|source| native_error("DRM memory handle close", source))?;
        let last = self
            .gems
            .pop()
            .ok_or_else(|| error(ErrorKind::Internal, "DRM memory handle cache is empty"))?;
        if index < self.gems.len() {
            self.gems[index] = last;
        }
        Ok(())
    }

    fn submit_map(
        &mut self,
        render: &File,
        handle: u32,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<(u32, u64), SubmitError> {
        let syncobj = self.ensure_syncobj(render).map_err(|source| SubmitError {
            error: native_error("DRM sync object creation", source),
            update: None,
        })?;
        let point = self.next_point().map_err(|source| SubmitError {
            error: native_error("DRM VM timeline advance", source),
            update: None,
        })?;
        drm::map(
            render,
            handle,
            address,
            offset,
            size,
            permissions.bits(),
            syncobj,
            point,
        )
        .map_err(|source| SubmitError {
            error: native_error("DRM virtual-memory map", source),
            update: Some((syncobj, point)),
        })?;
        Ok((syncobj, point))
    }

    fn submit_unmap(
        &mut self,
        render: &File,
        handle: u32,
        address: u64,
        offset: u64,
        size: u64,
    ) -> Result<(u32, u64), SubmitError> {
        let syncobj = self.ensure_syncobj(render).map_err(|source| SubmitError {
            error: native_error("DRM sync object creation", source),
            update: None,
        })?;
        let point = self.next_point().map_err(|source| SubmitError {
            error: native_error("DRM VM timeline advance", source),
            update: None,
        })?;
        drm::unmap(render, handle, address, offset, size, syncobj, point).map_err(|source| {
            SubmitError {
                error: native_error("DRM virtual-memory unmap", source),
                update: Some((syncobj, point)),
            }
        })?;
        Ok((syncobj, point))
    }

    fn wait_map(render: &File, update: (u32, u64)) -> Result<(), Error> {
        let (syncobj, point) = update;
        drm::wait(render, syncobj, point)
            .map_err(|source| native_error("DRM virtual-memory map wait", source))
    }

    fn wait_unmap(render: &File, update: (u32, u64)) -> Result<(), Error> {
        let (syncobj, point) = update;
        drm::wait(render, syncobj, point)
            .map_err(|source| native_error("DRM virtual-memory unmap wait", source))
    }
}

/// Owned virtual-address reservation with explicit release progress.
pub(crate) struct KfdVirtualAddress {
    reservation: Option<sys::Reservation>,
    mapping_granularity: u64,
    uncertain: AtomicBool,
}

impl KfdVirtualAddress {
    pub(super) fn reserve(
        bounds: (u64, u64),
        size: u64,
        alignment: u64,
        address: u64,
        allocator: Allocator,
    ) -> Result<Owned<Self>, Error> {
        let desc = AllocationDesc { size, alignment };
        let page = util::page_size()
            .map_err(|source| native_error("virtual-address page size", source))?
            as u64;
        let limits = AllocationLimits {
            alignment: page,
            granularity: page,
            maximum_size: isize::MAX as u64,
        };
        if !limits.supports(desc) {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid virtual-address extent or alignment",
            ));
        }
        let size = usize::try_from(size).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "virtual-address extent exceeds host width",
            )
        })?;
        let alignment = usize::try_from(alignment).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "virtual-address alignment exceeds host width",
            )
        })?;
        let address = usize::try_from(address).unwrap_or(0);
        let owner = Owned::try_new_uninit(allocator)?;
        let reservation = sys::Reservation::new_at(size, alignment, bounds, address)
            .map_err(|source| native_error("virtual-address reservation", source))?;
        Ok(owner.write(Self {
            reservation: Some(reservation),
            mapping_granularity: limits.granularity,
            uncertain: AtomicBool::new(false),
        }))
    }

    pub(crate) fn address(&self) -> u64 {
        self.reservation
            .as_ref()
            .map_or(0, |reservation| reservation.address() as u64)
    }

    pub(crate) fn size(&self) -> u64 {
        self.reservation
            .as_ref()
            .map_or(0, |reservation| reservation.usable_size() as u64)
    }

    pub(crate) const fn mapping_granularity(&self) -> u64 {
        self.mapping_granularity
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        self.check()?;
        if let Some(reservation) = &mut self.reservation {
            reservation
                .release()
                .map_err(|source| native_error("virtual-address release", source))?;
            self.reservation = None;
        }
        Ok(())
    }

    fn check(&self) -> Result<(), Error> {
        if self.reservation.is_none() {
            return Err(error(
                ErrorKind::InvalidArgument,
                "virtual-address reservation was already released",
            ));
        }
        if self.uncertain.load(Ordering::Acquire) {
            return Err(error(
                ErrorKind::DriverContract,
                "virtual-address range may still have a native GPU mapping",
            ));
        }
        Ok(())
    }

    fn mark_uncertain(&self) {
        self.uncertain.store(true, Ordering::Release);
    }
}

impl Drop for KfdVirtualAddress {
    fn drop(&mut self) {
        if self.free().is_err() {
            if let Some(reservation) = self.reservation.take() {
                std::mem::forget(reservation);
            }
        }
    }
}

/// Physical KFD allocation or imported DMA-BUF backing for virtual mappings.
pub(crate) struct KfdVirtualMemory {
    vm: Option<Shared<DeviceVm>>,
    reservation: Option<sys::Reservation>,
    handle: Option<u64>,
    dma_buf: Option<File>,
    info: VirtualMemoryInfo,
    freeing: bool,
    uncertain: bool,
    mapping_uncertain: AtomicBool,
}

impl KfdVirtualMemory {
    #[allow(
        clippy::too_many_lines,
        reason = "keep detached KFD allocation acquisition and rollback in one owner constructor"
    )]
    pub(super) fn create(
        vm: Shared<DeviceVm>,
        kind: MemoryKind,
        size: u64,
        pinned: bool,
        uncached: bool,
    ) -> Result<Owned<Self>, Error> {
        let desc = AllocationDesc {
            size,
            alignment: 4096,
        };
        let limits = AllocationLimits {
            alignment: 4096,
            granularity: 4096,
            maximum_size: isize::MAX as u64,
        };
        if !limits.supports(desc) {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid virtual-memory physical extent",
            ));
        }
        let size_usize = usize::try_from(size).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "virtual-memory physical extent exceeds host width",
            )
        })?;
        let native_kind = match kind {
            MemoryKind::System => uapi::GTT | uapi::COHERENT,
            MemoryKind::DeviceLocal { coherent, .. } => {
                uapi::VRAM | if coherent { uapi::COHERENT } else { 0 }
            }
            MemoryKind::OwnedHost | MemoryKind::RegisteredHost { .. } => {
                return Err(error(
                    ErrorKind::Unsupported,
                    "virtual-memory handles require platform-managed physical backing",
                ));
            }
        };
        vm.check()?;
        let allocator = vm.allocator();
        let owner = Owned::try_new_uninit(allocator)?;
        let reservation = sys::Reservation::new(size_usize, 4096, vm.address_range(), false)
            .map_err(|source| native_error("virtual-memory physical reservation", source))?;
        let mut memory = owner.write(Self {
            vm: Some(vm),
            reservation: Some(reservation),
            handle: None,
            dma_buf: None,
            info: VirtualMemoryInfo {
                size,
                mapping_granularity: 4096,
                physical_backing_id: [0; 2],
            },
            freeing: false,
            uncertain: false,
            mapping_uncertain: AtomicBool::new(false),
        });
        let gpu_id = memory
            .vm
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "virtual memory lost its owner VM"))?
            .gpu_id();
        let mut args = uapi::AllocMemory {
            va: memory
                .reservation
                .as_ref()
                .ok_or_else(|| {
                    error(
                        ErrorKind::Internal,
                        "virtual memory lost its physical reservation",
                    )
                })?
                .address() as u64,
            size,
            gpu_id,
            flags: native_kind
                | uapi::WRITABLE
                | if pinned { uapi::NO_SUBSTITUTE } else { 0 }
                | if uncached { uapi::UNCACHED } else { 0 },
            ..uapi::AllocMemory::default()
        };
        let result = memory
            .vm
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "virtual memory lost its owner VM"))?
            .kfd()
            .allocate(&mut args);
        memory.handle = (args.handle != 0).then_some(args.handle);
        memory.uncertain = result
            .as_ref()
            .err()
            .is_some_and(|source| source.raw_os_error() == Some(14));
        result.map_err(|source| native_error("AMDKFD_IOC_ALLOC_MEMORY_OF_GPU", source))?;
        let handle = memory.handle.ok_or_else(|| {
            memory.uncertain = true;
            error(
                ErrorKind::DriverContract,
                "KFD virtual-memory allocation succeeded without a handle",
            )
        })?;
        let dma_buf = memory
            .vm
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "virtual memory lost its owner VM"))?
            .kfd()
            .export_dma_buf(handle)
            .map_err(|source| native_error("AMDKFD_IOC_EXPORT_DMABUF", source))?;
        let file_info = util::dma_buf_file_info(&dma_buf)
            .map_err(|source| native_error("virtual-memory DMA-BUF information", source))?;
        if file_info.size != size {
            return Err(error(
                ErrorKind::DriverContract,
                "virtual-memory DMA-BUF has an unexpected extent",
            ));
        }
        memory.info.physical_backing_id = file_info.physical_id;
        memory.dma_buf = Some(dma_buf);
        Ok(memory)
    }

    pub(super) fn import(descriptor: i32, allocator: Allocator) -> Result<Owned<Self>, Error> {
        let dma_buf = util::duplicate_file(descriptor)
            .map_err(|source| native_error("virtual-memory descriptor duplication", source))?;
        let file_info = util::dma_buf_file_info(&dma_buf)
            .map_err(|source| native_error("virtual-memory DMA-BUF information", source))?;
        let page = util::page_size()
            .map_err(|source| native_error("virtual-memory page size", source))?
            as u64;
        Owned::new(
            Self {
                vm: None,
                reservation: None,
                handle: None,
                dma_buf: Some(dma_buf),
                info: VirtualMemoryInfo {
                    size: file_info.size,
                    mapping_granularity: page,
                    physical_backing_id: file_info.physical_id,
                },
                freeing: false,
                uncertain: false,
                mapping_uncertain: AtomicBool::new(false),
            },
            allocator,
        )
        .map_err(Into::into)
    }

    pub(crate) fn info(&self) -> VirtualMemoryInfo {
        self.info
    }

    pub(crate) fn export_dma_buf(&self) -> Result<DmaBuf, Error> {
        if self.freeing {
            return Err(error(
                ErrorKind::InvalidArgument,
                "virtual-memory handle release already began",
            ));
        }
        let file = self.dma_buf.as_ref().ok_or_else(|| {
            error(
                ErrorKind::DriverContract,
                "virtual-memory handle lost its DMA-BUF",
            )
        })?;
        let duplicate = util::duplicate_file(file.as_raw_fd())
            .map_err(|source| native_error("virtual-memory descriptor duplication", source))?;
        Ok(DmaBuf::new(
            duplicate.into(),
            DmaBufInfo {
                byte_length: self.info.size,
                source_offset: 0,
                physical_backing_id: self.info.physical_backing_id,
            },
        ))
    }

    pub(super) fn dma_buf(&self) -> Result<&File, Error> {
        if self.freeing {
            return Err(error(
                ErrorKind::InvalidArgument,
                "virtual-memory handle release already began",
            ));
        }
        self.dma_buf.as_ref().ok_or_else(|| {
            error(
                ErrorKind::DriverContract,
                "virtual-memory handle lost its DMA-BUF",
            )
        })
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        self.freeing = true;
        if self.mapping_uncertain.load(Ordering::Acquire) {
            return Err(error(
                ErrorKind::DriverContract,
                "virtual-memory backing may still have a native GPU mapping",
            ));
        }
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "virtual-memory allocation outcome is uncertain",
            ));
        }
        if let Some(handle) = self.handle {
            let vm = self.vm.as_ref().ok_or_else(|| {
                error(
                    ErrorKind::Internal,
                    "virtual-memory allocation lost its owner VM",
                )
            })?;
            vm.kfd()
                .free(handle)
                .map_err(|source| native_error("AMDKFD_IOC_FREE_MEMORY_OF_GPU", source))?;
            self.handle = None;
        }
        if let Some(reservation) = &mut self.reservation {
            reservation.release().map_err(|source| {
                native_error("virtual-memory physical reservation release", source)
            })?;
            self.reservation = None;
        }
        self.dma_buf = None;
        self.vm = None;
        Ok(())
    }
}

impl Drop for KfdVirtualMemory {
    fn drop(&mut self) {
        if self.free().is_err() {
            if let Some(reservation) = self.reservation.take() {
                std::mem::forget(reservation);
            }
            if let Some(vm) = self.vm.take() {
                std::mem::forget(vm);
            }
        }
    }
}

/// Device mapping joining one address reservation to one physical allocation.
pub(crate) struct KfdVirtualDeviceMapping {
    vm: Shared<DeviceVm>,
    backing_id: [u64; 2],
    handle: u32,
    address: u64,
    offset: u64,
    size: u64,
    mapped: bool,
    pending_unmap: Option<(u32, u64)>,
    owns_gem: bool,
    uncertain: bool,
}

impl KfdVirtualDeviceMapping {
    pub(super) fn create(
        memory: &KfdVirtualMemory,
        reservation: &KfdVirtualAddress,
        vm: Shared<DeviceVm>,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self>, Error> {
        let supported_permissions =
            DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE;
        if !supported_permissions.contains(permissions) {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid virtual-memory access permission",
            ));
        }
        vm.check()?;
        reservation.check()?;
        let dma_buf = memory.dma_buf()?;
        let allocator = vm.allocator();
        let owner = Owned::try_new_uninit(allocator)?;
        let render = vm.render()?;
        let mut state = vm.vmem.lock().map_err(|_| {
            error(
                ErrorKind::Internal,
                "virtual-memory mapping lock was poisoned",
            )
        })?;
        state.pending_maps.try_reserve(1)?;
        let handle = state.acquire_gem(render, dma_buf, memory.info.physical_backing_id)?;
        let update = match state.submit_map(render, handle, address, offset, size, permissions) {
            Ok(update) => update,
            Err(source) => {
                if let Some(update) = source.update {
                    reservation.mark_uncertain();
                    memory.mapping_uncertain.store(true, Ordering::Release);
                    let _ = state.pending_maps.try_push(update);
                    drop(state);
                    std::mem::forget(vm);
                } else {
                    state.release_gem(render, memory.info.physical_backing_id)?;
                }
                return Err(source.error);
            }
        };
        if let Err(source) = VmState::wait_map(render, update) {
            reservation.mark_uncertain();
            memory.mapping_uncertain.store(true, Ordering::Release);
            let _ = state.pending_maps.try_push(update);
            drop(state);
            std::mem::forget(vm);
            return Err(source);
        }
        drop(state);
        Ok(owner.write(Self {
            vm,
            backing_id: memory.info.physical_backing_id,
            handle,
            address,
            offset,
            size,
            mapped: true,
            pending_unmap: None,
            owns_gem: true,
            uncertain: false,
        }))
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        if !self.mapped && !self.owns_gem {
            return Ok(());
        }
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "virtual-memory unmap outcome is uncertain",
            ));
        }
        self.vm.check()?;
        let render = self.vm.render()?;
        let mut state = self.vm.vmem.lock().map_err(|_| {
            error(
                ErrorKind::Internal,
                "virtual-memory mapping lock was poisoned",
            )
        })?;
        if self.mapped {
            let update = if let Some(update) = self.pending_unmap {
                update
            } else {
                match state.submit_unmap(render, self.handle, self.address, self.offset, self.size)
                {
                    Ok(update) => {
                        self.pending_unmap = Some(update);
                        update
                    }
                    Err(source) => {
                        if source.update.is_some() {
                            self.uncertain = true;
                        }
                        return Err(source.error);
                    }
                }
            };
            VmState::wait_unmap(render, update)?;
            self.pending_unmap = None;
            self.mapped = false;
        }
        if self.owns_gem {
            state.release_gem(render, self.backing_id)?;
            self.owns_gem = false;
        }
        Ok(())
    }
}

impl Drop for KfdVirtualDeviceMapping {
    fn drop(&mut self) {
        if self.free().is_err() {
            std::mem::forget(self.vm.clone());
        }
    }
}

/// Host mapping joining one address reservation to CPU-accessible backing.
pub(crate) struct KfdVirtualHostMapping {
    reservation: Option<sys::Reservation>,
}

impl KfdVirtualHostMapping {
    pub(super) fn create(
        memory: &KfdVirtualMemory,
        address_owner: &KfdVirtualAddress,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
        allocator: Allocator,
    ) -> Result<Owned<Self>, Error> {
        let supported_permissions = DeviceAccess::READ | DeviceAccess::WRITE;
        if !supported_permissions.contains(permissions) {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid virtual-memory host access permission",
            ));
        }
        address_owner.check()?;
        let address = usize::try_from(address).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "virtual-memory host address exceeds host width",
            )
        })?;
        let size = usize::try_from(size).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "virtual-memory host extent exceeds host width",
            )
        })?;
        let owner = Owned::try_new_uninit(allocator)?;
        let mut reservation = sys::Reservation::view(address, size);
        reservation
            .map_dma_buf_with_permissions(memory.dma_buf()?, offset, permissions.bits())
            .map_err(|source| native_error("virtual-memory host map", source))?;
        Ok(owner.write(Self {
            reservation: Some(reservation),
        }))
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        if let Some(reservation) = &mut self.reservation {
            reservation
                .release()
                .map_err(|source| native_error("virtual-memory host unmap", source))?;
            self.reservation = None;
        }
        Ok(())
    }
}

impl Drop for KfdVirtualHostMapping {
    fn drop(&mut self) {
        let _ = self.free();
    }
}
