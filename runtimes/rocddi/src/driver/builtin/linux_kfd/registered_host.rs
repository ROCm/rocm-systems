//! Borrowed host pages registered through DRM in a secondary KFD context.
//!
//! A KFD USERPTR allocation is unavailable in that context, but its acquired
//! render VM can map a DRM GEM USERPTR object. This owner retains the caller's
//! GPU VA, each render VM, GEM handle, and timeline until native cleanup ends.
//! The caller continues to own and keep the complete source page cover live.

use super::drm;
use super::memory::{DeviceVm, error, native_error};
use super::sys;
use super::util;
use crate::host_storage::{Buffer, Owned, Shared};
use crate::memory::{AllocationDesc, AllocationInfo, DeviceAccess};
use crate::{Error, ErrorKind};

#[derive(Clone, Copy, Eq, PartialEq)]
enum MappingPhase {
    NotMapped,
    Mapping,
    Mapped,
    Unmapping,
    Unmapped,
}

struct VmMapping {
    vm: Shared<DeviceVm>,
    handle: u32,
    timeline: u32,
    phase: MappingPhase,
    uncertain: bool,
}

impl VmMapping {
    fn new(vm: Shared<DeviceVm>) -> Self {
        Self {
            vm,
            handle: 0,
            timeline: 0,
            phase: MappingPhase::NotMapped,
            uncertain: false,
        }
    }

    fn map(
        &mut self,
        host_base: u64,
        address: u64,
        size: u64,
        permissions: DeviceAccess,
        uncached: bool,
    ) -> Result<(), Error> {
        let render = self.vm.render()?;
        let result = drm::register_userptr(render, host_base, size, &mut self.handle);
        if let Err(source) = result {
            if self.handle != 0
                || source.raw_os_error() == Some(14)
                || source.kind() == std::io::ErrorKind::InvalidData
            {
                self.uncertain = true;
            }
            return Err(native_error("DRM GEM USERPTR registration", source));
        }
        self.timeline = drm::create_syncobj(render)
            .map_err(|source| native_error("DRM host registration timeline", source))?;
        self.phase = MappingPhase::Mapping;
        if let Err(source) = drm::map_with_cache(
            render,
            self.handle,
            address,
            0,
            size,
            permissions.bits(),
            uncached,
            self.timeline,
            1,
        ) {
            // An ioctl error may follow submission. Reusing the VA or host
            // pages would then be unsafe, even if no timeline point arrives.
            self.uncertain = true;
            return Err(native_error("DRM host registration map", source));
        }
        drm::wait(render, self.timeline, 1)
            .map_err(|source| native_error("DRM host registration map wait", source))?;
        self.phase = MappingPhase::Mapped;
        Ok(())
    }

    fn free(&mut self, address: u64, size: u64) -> Result<(), Error> {
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "DRM host registration outcome is uncertain",
            ));
        }
        let render = self.vm.render()?;
        if self.phase == MappingPhase::Mapping {
            drm::wait(render, self.timeline, 1)
                .map_err(|source| native_error("DRM host registration map wait", source))?;
            self.phase = MappingPhase::Mapped;
        }
        if self.phase == MappingPhase::Mapped {
            if let Err(source) = drm::unmap(render, self.handle, address, 0, size, self.timeline, 2)
            {
                self.uncertain = true;
                return Err(native_error("DRM host registration unmap", source));
            }
            self.phase = MappingPhase::Unmapping;
        }
        if self.phase == MappingPhase::Unmapping {
            drm::wait(render, self.timeline, 2)
                .map_err(|source| native_error("DRM host registration unmap wait", source))?;
            self.phase = MappingPhase::Unmapped;
        }
        if self.timeline != 0 {
            drm::destroy_syncobj(render, self.timeline)
                .map_err(|source| native_error("DRM host registration timeline close", source))?;
            self.timeline = 0;
        }
        if self.handle != 0 {
            drm::close_gem(render, self.handle)
                .map_err(|source| native_error("DRM GEM USERPTR close", source))?;
            self.handle = 0;
        }
        Ok(())
    }
}

/// One common GPU VA for a caller-owned page cover across requested render VMs.
pub(crate) struct DrmRegisteredHost {
    owner: Shared<DeviceVm>,
    mappings: Buffer<VmMapping>,
    reservation: Option<sys::Reservation>,
    host_address: usize,
    byte_offset: usize,
    size: u64,
    freeing: bool,
}

impl DrmRegisteredHost {
    pub(super) fn create(
        owner: Shared<DeviceVm>,
        peers: impl ExactSizeIterator<Item = Shared<DeviceVm>>,
        desc: AllocationDesc,
        host_address: usize,
        permissions: DeviceAccess,
        uncached: bool,
    ) -> Result<Owned<Self>, Error> {
        if !permissions.contains(DeviceAccess::READ) {
            return Err(error(
                ErrorKind::Unsupported,
                "registered GPU pages require READ permission",
            ));
        }
        let page = util::page_size()
            .map_err(|source| native_error("registered host page size", source))?;
        let size = usize::try_from(desc.size).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "registered host extent exceeds host address width",
            )
        })?;
        let byte_offset = host_address & (page - 1);
        let host_base = host_address - byte_offset;
        if host_address == 0
            || size == 0
            || size % page != 0
            || host_base.checked_add(size).is_none()
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "registered host page cover is invalid",
            ));
        }
        owner.check()?;
        let allocator = owner.allocator();
        let mut mappings = Buffer::try_with_capacity(
            peers.len().checked_add(1).ok_or_else(|| {
                error(
                    ErrorKind::ResourceExhausted,
                    "registered device count overflow",
                )
            })?,
            allocator,
        )?;
        mappings.try_push(VmMapping::new(owner.clone()))?;
        let mut bounds = owner.address_range();
        for peer in peers {
            if !owner.shares_kfd(&peer) {
                return Err(error(
                    ErrorKind::InvalidArgument,
                    "registered devices must share one KFD session",
                ));
            }
            if Shared::ptr_eq(&owner, &peer)
                || mappings
                    .iter()
                    .any(|mapping| Shared::ptr_eq(&mapping.vm, &peer))
            {
                continue;
            }
            if mappings
                .iter()
                .any(|mapping| mapping.vm.gpu_id() == peer.gpu_id())
            {
                return Err(error(
                    ErrorKind::InvalidArgument,
                    "distinct registered VMs must have distinct GPU IDs",
                ));
            }
            peer.check()?;
            let range = peer.address_range();
            bounds.0 = bounds.0.max(range.0);
            bounds.1 = bounds.1.min(range.1);
            if bounds.0 > bounds.1 {
                return Err(error(
                    ErrorKind::Unsupported,
                    "registered devices have no common GPU address range",
                ));
            }
            mappings.try_push(VmMapping::new(peer))?;
        }
        let owner_slot = Owned::try_new_uninit(allocator)?;
        let alignment = usize::try_from(desc.alignment)
            .map_err(|_| error(ErrorKind::InvalidArgument, "alignment exceeds host width"))?;
        let reservation = sys::Reservation::new(size, alignment, bounds, false)
            .map_err(|source| native_error("registered GPU address reservation", source))?;
        let mut allocation = owner_slot.write(Self {
            owner,
            mappings,
            reservation: Some(reservation),
            host_address,
            byte_offset,
            size: desc.size,
            freeing: false,
        });
        let address = allocation.address()?;
        for mapping in &mut allocation.mappings {
            mapping.map(host_base as u64, address, desc.size, permissions, uncached)?;
        }
        allocation.owner.check()?;
        Ok(allocation)
    }

    fn address(&self) -> Result<u64, Error> {
        self.reservation
            .as_ref()
            .map(|reservation| reservation.address() as u64)
            .ok_or_else(|| error(ErrorKind::InvalidArgument, "registered GPU VA was released"))
    }

    pub(crate) fn cached_info(&self) -> AllocationInfo {
        AllocationInfo {
            device_address: self.reservation.as_ref().map_or(0, |reservation| {
                reservation.address() as u64 + self.byte_offset as u64
            }),
            host_address: Some(self.host_address),
            size: self.size,
            native_size: self.size,
            physical_backing_id: [0; 2],
        }
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        self.owner.check()?;
        if self.freeing
            || self
                .mappings
                .iter()
                .any(|mapping| mapping.phase != MappingPhase::Mapped)
        {
            return Err(error(
                ErrorKind::Unsupported,
                "registered host mapping is unavailable",
            ));
        }
        Ok(())
    }

    pub(crate) fn device_address(&self, device: &Shared<DeviceVm>) -> Result<u64, Error> {
        self.check()?;
        if !self
            .mappings
            .iter()
            .any(|mapping| Shared::ptr_eq(&mapping.vm, device))
        {
            return Err(error(
                ErrorKind::Unsupported,
                "registered host mapping is unavailable on this device",
            ));
        }
        device.check()?;
        self.address()
            .map(|address| address + self.byte_offset as u64)
    }

    pub(crate) fn is_owned_by(&self, device: &Shared<DeviceVm>) -> bool {
        self.owner.shares_kfd(device) && self.owner.gpu_id() == device.gpu_id()
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        self.owner
            .kfd()
            .check_process()
            .map_err(|source| native_error("DRM host registration process check", source))?;
        self.freeing = true;
        if self.reservation.is_none() {
            return Ok(());
        }
        let address = self.address()?;
        for mapping in self.mappings.iter_mut().rev() {
            mapping.free(address, self.size)?;
        }
        if let Some(reservation) = self.reservation.as_mut() {
            reservation
                .release()
                .map_err(|source| native_error("registered GPU address munmap", source))?;
            self.reservation = None;
        }
        Ok(())
    }
}

impl Drop for DrmRegisteredHost {
    fn drop(&mut self) {
        if self.free().is_err() {
            if let Some(reservation) = self.reservation.take() {
                std::mem::forget(reservation);
            }
            std::mem::forget(self.owner.clone());
            for mapping in &self.mappings {
                std::mem::forget(mapping.vm.clone());
            }
        }
    }
}
