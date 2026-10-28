// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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

use crate::memory::storage::TryPush;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::memory::device::VirtualMemoryPlacement;
use crate::memory::interop::linux::{DmaBuf, DmaBufInfo};
use crate::memory::storage::{Owned, Shared};
use crate::memory::{
    AllocationDesc, AllocationLimits, DeviceAccess, MemoryKind, VirtualMemoryInfo,
};
use crate::{Error, ErrorKind};

use super::memory::DeviceVm;
use super::{drm, errno, error, native_error, os_file, os_memory, uapi};

/// Imported GEM handle shared by mappings of the same physical backing.
struct ImportedGem {
    /// Native physical backing identity used to share this GEM import.
    backing_id: [u64; 2],
    /// Process-local GEM handle tied to the creating render file.
    handle: u32,
    /// Mappings currently using this imported handle.
    references: usize,
}

/// A failed submission can still have reached the driver.
struct SubmitError {
    /// Native failure returned to the caller.
    error: Error,
    /// Mapping update that may need completion or quarantine.
    update: Option<(u32, u64)>,
}

/// Per-device DRM virtual-memory state serialized by its owning `DeviceVm`.
pub(super) struct VmState {
    /// Timeline sync object shared by mapping transactions on this VM.
    syncobj: u32,
    /// Its native destruction may have completed despite an error.
    syncobj_destroy_uncertain: bool,
    /// Next timeline point assigned to a VM update.
    next_point: u64,
    /// Imported GEM handles shared by backing identity.
    gems: Vec<ImportedGem>,
    /// Failed MAP updates with no public owner. The reservation and backing
    /// quarantine themselves while these timeline points remain reachable.
    pending_maps: Vec<(u32, u64)>,
}

impl VmState {
    pub(super) fn new() -> Self {
        Self {
            syncobj: 0,
            syncobj_destroy_uncertain: false,
            next_point: 0,
            gems: Vec::new(),
            pending_maps: Vec::new(),
        }
    }

    pub(super) fn close(&mut self, render: &File) -> Result<(), Error> {
        if !self.gems.is_empty() || !self.pending_maps.is_empty() {
            return Err(error(
                ErrorKind::DriverContract,
                "live virtual-memory mappings prevent render shutdown",
            ));
        }
        if self.syncobj_destroy_uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "DRM sync object destruction outcome is uncertain",
            ));
        }
        if self.syncobj != 0 {
            if let Err(source) = drm::destroy_syncobj(render, self.syncobj) {
                self.syncobj_destroy_uncertain = source.raw_os_error() == Some(errno::EFAULT);
                return Err(native_error("DRM sync object destruction", source));
            }
            self.syncobj = 0;
        }
        Ok(())
    }

    fn ensure_syncobj(&mut self, render: &File) -> io::Result<u32> {
        if self.syncobj_destroy_uncertain {
            return Err(io::Error::from(io::ErrorKind::InvalidData));
        }
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
    /// Process VA range retained until every mapping ends.
    reservation: Option<os_memory::Reservation>,
    /// Required alignment and mapping size multiple.
    mapping_granularity: u64,
    /// A native mapping result may still reference this range.
    uncertain: AtomicBool,
}

impl KfdVirtualAddress {
    pub(super) fn reserve(
        bounds: (u64, u64),
        size: u64,
        alignment: u64,
        address: u64,
    ) -> Result<Owned<Self>, Error> {
        let desc = AllocationDesc { size, alignment };
        let page = os_memory::page_size()
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
        let owner = Owned::try_new_uninit()?;
        let reservation = os_memory::Reservation::new_at(size, alignment, bounds, address)
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
        let Some(reservation) = self.reservation.as_ref() else {
            return Err(error(
                ErrorKind::InvalidArgument,
                "virtual-address reservation was already released",
            ));
        };
        if !reservation.is_live() {
            return Err(error(
                ErrorKind::DriverContract,
                "virtual-address reservation ownership is uncertain",
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
    /// Creating VM retained while native backing exists.
    vm: Option<Shared<DeviceVm>>,
    /// Process address cover used by host-visible backing.
    reservation: Option<os_memory::Reservation>,
    /// Native KFD memory handle until release succeeds.
    handle: Option<u64>,
    /// Imported DMA-BUF descriptor retained with physical backing.
    dma_buf: Option<File>,
    /// Cached extent, granularity, and backing identity.
    info: VirtualMemoryInfo,
    /// Prevents new maps once native release begins.
    freeing: bool,
    /// Native release outcome cannot be safely replayed.
    uncertain: bool,
    /// A failed map may have left device page tables referring to backing.
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
        let host_page = os_memory::page_size()
            .map_err(|source| native_error("virtual-memory page size", source))?;
        let page = host_page as u64;
        let desc = AllocationDesc {
            size,
            alignment: page,
        };
        let limits = AllocationLimits {
            alignment: page,
            granularity: page,
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
        if let MemoryKind::DeviceLocal {
            coherent,
            extended_coherent: true,
            uncached: kind_uncached,
            ..
        } = kind
            && (!coherent || kind_uncached || uncached)
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "extended local coherence requires a cached coherent mapping",
            ));
        }
        let native_kind = match kind {
            MemoryKind::System => uapi::GTT | uapi::COHERENT,
            MemoryKind::DeviceLocal {
                coherent,
                extended_coherent,
                uncached,
                ..
            } => {
                uapi::VRAM
                    | if coherent { uapi::COHERENT } else { 0 }
                    | if extended_coherent {
                        uapi::EXT_COHERENT
                    } else {
                        0
                    }
                    | if uncached { uapi::UNCACHED } else { 0 }
            }
            MemoryKind::OwnedHost { .. } => {
                return Err(error(
                    ErrorKind::Unsupported,
                    "virtual-memory handles require platform-managed physical backing",
                ));
            }
        };
        vm.check()?;

        let gpu_id = vm.gpu_id;

        let owner = Owned::try_new_uninit()?;
        let reservation =
            os_memory::Reservation::new(size_usize, host_page, vm.address_range(), false)
                .map_err(|source| native_error("virtual-memory physical reservation", source))?;
        let mut memory = owner.write(Self {
            vm: Some(vm),
            reservation: Some(reservation),
            handle: None,
            dma_buf: None,
            info: VirtualMemoryInfo {
                size,
                mapping_granularity: page,
                physical_backing_id: [0; 2],
                placement: Some(if native_kind & uapi::VRAM != 0 {
                    VirtualMemoryPlacement::Device(gpu_id)
                } else {
                    VirtualMemoryPlacement::System
                }),
            },
            freeing: false,
            uncertain: false,
            mapping_uncertain: AtomicBool::new(false),
        });
        let gpu_id = memory
            .vm
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "virtual memory lost its owner VM"))?
            .gpu_id;
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
            .is_some_and(|source| source.raw_os_error() == Some(errno::EFAULT));
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
        let file_info = os_file::dma_buf_file_info(&dma_buf)
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

    pub(super) fn import(descriptor: i32) -> Result<Owned<Self>, Error> {
        Self::import_impl(descriptor, None)
    }

    pub(super) fn import_with_kfd(
        descriptor: i32,
        kfd: &super::sys::Kfd,
    ) -> Result<Owned<Self>, Error> {
        Self::import_impl(descriptor, Some(kfd))
    }

    fn import_impl(descriptor: i32, kfd: Option<&super::sys::Kfd>) -> Result<Owned<Self>, Error> {
        let dma_buf = os_file::duplicate_file(descriptor)
            .map_err(|source| native_error("virtual-memory descriptor duplication", source))?;
        let file_info = os_file::dma_buf_file_info(&dma_buf)
            .map_err(|source| native_error("virtual-memory DMA-BUF information", source))?;
        let page = os_memory::page_size()
            .map_err(|source| native_error("virtual-memory page size", source))?
            as u64;
        // Older KFD versions can import the DMA-BUF but cannot describe its
        // origin. Preserve that distinction for the HSA handle-info query.
        let placement = kfd
            .and_then(|kfd| kfd.dma_buf_info(dma_buf.as_raw_fd()).ok())
            .and_then(|details| {
                let info = details.info;
                if info.size != file_info.size {
                    return None;
                }
                match info.flags & (uapi::VRAM | uapi::GTT) {
                    uapi::GTT => Some(VirtualMemoryPlacement::System),
                    uapi::VRAM if info.gpu_id != 0 => {
                        Some(VirtualMemoryPlacement::Device(info.gpu_id))
                    }
                    _ => None,
                }
            });
        Owned::new(Self {
            vm: None,
            reservation: None,
            handle: None,
            dma_buf: Some(dma_buf),
            info: VirtualMemoryInfo {
                size: file_info.size,
                mapping_granularity: page,
                physical_backing_id: file_info.physical_id,
                placement,
            },
            freeing: false,
            uncertain: false,
            mapping_uncertain: AtomicBool::new(false),
        })
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
        let duplicate = os_file::duplicate_file(file.as_raw_fd())
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

    pub(crate) fn fabric_ready(&self) -> Result<bool, Error> {
        let vm = self.vm.as_ref().ok_or_else(|| {
            error(
                ErrorKind::Unsupported,
                "imported virtual memory has no fabric export owner",
            )
        })?;
        vm.check()?;
        let path = format!(
            "/sys/class/drm/renderD{}/device/ualink/accel_state",
            vm.render_minor
        );
        Ok(std::fs::read_to_string(path)
            .is_ok_and(|state| matches!(state.trim(), "ready" | "active")))
    }

    pub(crate) fn export_fabric_handle(&self) -> Result<[u8; 16], Error> {
        if self.freeing || self.uncertain || self.mapping_uncertain.load(Ordering::Acquire) {
            return Err(error(
                ErrorKind::DriverContract,
                "virtual-memory backing is unavailable for fabric export",
            ));
        }
        let vm = self.vm.as_ref().ok_or_else(|| {
            error(
                ErrorKind::Unsupported,
                "imported virtual memory has no fabric export owner",
            )
        })?;
        vm.check()?;
        if !self.fabric_ready()? {
            return Err(error(
                ErrorKind::Busy,
                "UALink accelerator is not ready for fabric export",
            ));
        }
        // GEM handles are scoped to a DRM file. A dedicated file makes the
        // temporary PRIME import close even if export fails.
        let render = super::sysfs::open_render(vm.render_minor)
            .map_err(|source| native_error("fabric render open", source))?;
        let dma_buf = self.dma_buf()?;
        let gem = drm::import_dma_buf(&render, dma_buf.as_raw_fd())
            .map_err(|source| native_error("fabric DMA-BUF import", source))?;
        drm::export_fabric_handle(&render, gem)
            .map_err(|source| native_error("DRM UALink fabric export", source))
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
    /// Render VM whose page tables contain the mapping.
    vm: Shared<DeviceVm>,
    /// Physical identity used to share imported GEM handles.
    backing_id: [u64; 2],
    /// GEM handle mapped in this render VM.
    handle: u32,
    /// First mapped GPU virtual byte.
    address: u64,
    /// Byte offset into physical backing.
    offset: u64,
    /// Mapped byte extent.
    size: u64,
    /// Native map and synchronization completed.
    mapped: bool,
    /// Timeline update awaiting proof of unmap completion.
    pending_unmap: Option<(u32, u64)>,
    /// Whether this mapping must release a GEM import reference.
    owns_gem: bool,
    /// Native update may have completed despite reporting an error.
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

        let owner = Owned::try_new_uninit()?;
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
    /// Host VA mapping restored to reservation state on free.
    reservation: Option<os_memory::Reservation>,
}

impl KfdVirtualHostMapping {
    pub(crate) fn set_permissions(&mut self, permissions: DeviceAccess) -> Result<(), Error> {
        let Some(reservation) = &mut self.reservation else {
            return Err(error(
                ErrorKind::InvalidArgument,
                "virtual-memory host mapping is no longer live",
            ));
        };
        reservation
            .set_permissions(permissions.bits())
            .map_err(|source| native_error("virtual-memory host protect", source))
    }

    pub(super) fn create(
        memory: &KfdVirtualMemory,
        address_owner: &KfdVirtualAddress,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
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
        let owner = Owned::try_new_uninit()?;
        let parent = address_owner.reservation.as_ref().ok_or_else(|| {
            error(
                ErrorKind::InvalidArgument,
                "virtual-address reservation was already released",
            )
        })?;
        let mut reservation = os_memory::Reservation::view_in(parent, address, size);
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

#[cfg(test)]
#[path = "tests/vmem.rs"]
mod tests;
