//! Qualified same-device GTT DMA-BUF attachment with an independent DRM owner.
//!
//! The BO is validated through its KFD and DRM views before it is published.
//! Detached virtual-memory owners provide one exact GPU PTE mapping and one
//! write-back host view; neither the caller's descriptor nor the AMDF transport
//! callback is retained. Failed native detach retains the VA and backing so a
//! GPU address cannot be reused while it may still be mapped.

use std::os::fd::AsRawFd;

use super::drm;
use super::memory::{DeviceVm, error, native_error};
use super::sysfs;
use super::uapi;
use super::util;
use super::vmem::{
    KfdVirtualAddress, KfdVirtualDeviceMapping, KfdVirtualHostMapping, KfdVirtualMemory,
};
use crate::host_storage::{Buffer, Owned, Shared};
use crate::memory::interop::linux::DmaBuf;
use crate::memory::{AllocationInfo, DeviceAccess};
use crate::{Error, ErrorKind};

fn local_gem(
    render: &std::fs::File,
    handle: u32,
    allocator: crate::host_storage::Allocator,
) -> Result<bool, Error> {
    let mut entries = Buffer::new(allocator);
    for _ in 0..4 {
        let count = drm::list_gem_handles(render, entries.as_mut_slice())
            .map_err(|source| native_error("DRM GEM handle query", source))?;
        if count > entries.len() {
            entries.try_reserve(count - entries.len())?;
            while entries.len() < count {
                entries.try_push(drm::GemHandleInfo::default())?;
            }
            continue;
        }
        return entries
            .iter()
            .take(count)
            .find(|entry| entry.handle == handle)
            .map(|entry| !entry.is_imported())
            .ok_or_else(|| {
                error(
                    ErrorKind::DriverContract,
                    "DRM lost the imported GEM handle",
                )
            });
    }
    Err(error(
        ErrorKind::ResourceExhausted,
        "DRM GEM handle table changed during import qualification",
    ))
}

fn validate_backing(vm: &DeviceVm, memory: &KfdVirtualMemory) -> Result<(), Error> {
    let dma_buf = memory.dma_buf()?;
    let file_info = util::dma_buf_file_info(dma_buf)
        .map_err(|source| native_error("DMA-BUF backing information", source))?;
    let details = vm
        .kfd()
        .dma_buf_info(dma_buf.as_raw_fd())
        .map_err(|source| native_error("AMDKFD_IOC_GET_DMABUF_INFO", source))?;
    let kfd = details.info;
    if kfd.gpu_id != vm.gpu_id() || kfd.flags & !uapi::PUBLIC != uapi::GTT {
        return Err(error(
            ErrorKind::Unsupported,
            "DMA-BUF is not same-device SYSTEM/GTT storage",
        ));
    }
    if kfd.size != file_info.size || kfd.size != memory.info().size {
        return Err(error(
            ErrorKind::DriverContract,
            "KFD and DMA-BUF report different backing extents",
        ));
    }

    // A separate render file gives qualification its own GEM namespace. File
    // closure releases this temporary handle without touching any live VM
    // mapping, including an earlier import of the same physical backing.
    let render = sysfs::open_render(vm.render_minor())
        .map_err(|source| native_error("DMA-BUF qualification render open", source))?;
    let handle = drm::import_dma_buf(&render, dma_buf.as_raw_fd())
        .map_err(|source| native_error("DRM DMA-BUF qualification import", source))?;
    let info = drm::gem_create_info(&render, handle)
        .map_err(|source| native_error("DRM GEM creation information", source))?;
    let required = drm::GEM_CREATE_COHERENT | drm::GEM_CREATE_UNCACHED;
    let excluded = drm::GEM_CREATE_CPU_GTT_USWC
        | drm::GEM_CREATE_NO_CPU_ACCESS
        | drm::GEM_CREATE_ENCRYPTED
        | drm::GEM_CREATE_SPARSE
        | drm::GEM_CREATE_DISCARDABLE
        | drm::GEM_CREATE_GFX12_DCC;
    if info.size != file_info.size
        || info.domains != drm::GEM_DOMAIN_GTT
        || info.flags & required != required
        || info.flags & excluded != 0
        || !local_gem(&render, handle, vm.allocator())?
    {
        return Err(error(
            ErrorKind::Unsupported,
            "DMA-BUF lacks a qualified same-device coherent GTT backing",
        ));
    }
    Ok(())
}

/// One imported physical backing and its host/device views at a common VA.
pub(crate) struct DrmImportedSystem {
    owner: Shared<DeviceVm>,
    memory: Option<Owned<KfdVirtualMemory>>,
    address: Option<Owned<KfdVirtualAddress>>,
    mapping: Option<Owned<KfdVirtualDeviceMapping>>,
    host: Option<Owned<KfdVirtualHostMapping>>,
    source_offset: u64,
    logical_size: u64,
    native_size: u64,
    physical_backing_id: [u64; 2],
    freeing: bool,
}

impl DrmImportedSystem {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create(
        vm: Shared<DeviceVm>,
        descriptor: i32,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self>, Error> {
        if !vm.supports_system_dma_buf_import() {
            return Err(error(
                ErrorKind::Unsupported,
                "DRM system import is unavailable",
            ));
        }
        let page = util::page_size()
            .map_err(|source| native_error("system import page size", source))?
            as u64;
        if byte_length == 0
            || !alignment.is_power_of_two()
            || alignment < page
            || source_offset % alignment != 0
            || !permissions.contains(DeviceAccess::READ)
            || permissions.bits() & !7 != 0
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid system DMA-BUF range, alignment, or permissions",
            ));
        }
        vm.check()?;
        let allocator = vm.allocator();
        let memory = KfdVirtualMemory::import(descriptor, allocator)?;
        let native_size = memory.info().size;
        if native_size == 0
            || native_size % page != 0
            || source_offset
                .checked_add(byte_length)
                .is_none_or(|end| end > native_size)
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "system DMA-BUF logical range exceeds its backing",
            ));
        }
        validate_backing(&vm, &memory)?;
        let address =
            KfdVirtualAddress::reserve(vm.address_range(), native_size, alignment, 0, allocator)?;
        let slot = Owned::<Self>::try_new_uninit(allocator)?;
        let mut imported = slot.write(Self {
            owner: vm.clone(),
            physical_backing_id: memory.info().physical_backing_id,
            memory: Some(memory),
            address: Some(address),
            mapping: None,
            host: None,
            source_offset,
            logical_size: byte_length,
            native_size,
            freeing: false,
        });
        let base = imported.base()?;
        let mapping = KfdVirtualDeviceMapping::create(
            imported.memory.as_ref().ok_or_else(|| {
                error(
                    ErrorKind::Internal,
                    "system import lost its physical backing",
                )
            })?,
            imported.address.as_ref().ok_or_else(|| {
                error(
                    ErrorKind::Internal,
                    "system import lost its address reservation",
                )
            })?,
            vm,
            base,
            0,
            native_size,
            permissions,
        )?;
        imported.mapping = Some(mapping);
        let host = KfdVirtualHostMapping::create(
            imported.memory.as_ref().ok_or_else(|| {
                error(
                    ErrorKind::Internal,
                    "system import lost its physical backing",
                )
            })?,
            imported.address.as_ref().ok_or_else(|| {
                error(
                    ErrorKind::Internal,
                    "system import lost its address reservation",
                )
            })?,
            base,
            0,
            native_size,
            DeviceAccess::READ | DeviceAccess::WRITE,
            allocator,
        )?;
        imported.host = Some(host);
        imported.owner.check()?;
        Ok(imported)
    }

    fn base(&self) -> Result<u64, Error> {
        self.address
            .as_ref()
            .map(|address| address.address())
            .ok_or_else(|| error(ErrorKind::InvalidArgument, "system import was released"))
    }

    pub(crate) fn cached_info(&self) -> AllocationInfo {
        let base = self.address.as_ref().map_or(0, |address| address.address());
        AllocationInfo {
            device_address: base + self.source_offset,
            host_address: usize::try_from(base + self.source_offset).ok(),
            size: self.logical_size,
            native_size: self.native_size,
            physical_backing_id: self.physical_backing_id,
        }
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        self.owner.check()?;
        if self.freeing || self.mapping.is_none() || self.host.is_none() {
            return Err(error(
                ErrorKind::InvalidArgument,
                "system import was released",
            ));
        }
        Ok(())
    }

    pub(crate) fn device_address(&self, device: &Shared<DeviceVm>) -> Result<u64, Error> {
        self.check()?;
        if !Shared::ptr_eq(&self.owner, device) {
            return Err(error(
                ErrorKind::Unsupported,
                "system import is not mapped into this device VM",
            ));
        }
        Ok(self.base()? + self.source_offset)
    }

    pub(crate) fn is_owned_by(&self, device: &Shared<DeviceVm>) -> bool {
        self.owner.shares_kfd(device) && self.owner.gpu_id() == device.gpu_id()
    }

    pub(crate) fn export_dma_buf(&self) -> Result<DmaBuf, Error> {
        self.check()?;
        self.memory
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "system import lost its backing"))?
            .export_dma_buf()
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        self.freeing = true;
        if let Some(mapping) = &mut self.mapping {
            mapping.free()?;
            self.mapping = None;
        }
        if let Some(host) = &mut self.host {
            host.free()?;
            self.host = None;
        }
        if let Some(address) = &mut self.address {
            address.free()?;
            self.address = None;
        }
        if let Some(memory) = &mut self.memory {
            memory.free()?;
            self.memory = None;
        }
        Ok(())
    }
}

impl Drop for DrmImportedSystem {
    fn drop(&mut self) {
        if self.free().is_err() {
            if let Some(mapping) = self.mapping.take() {
                std::mem::forget(mapping);
            }
            if let Some(host) = self.host.take() {
                std::mem::forget(host);
            }
            if let Some(address) = self.address.take() {
                std::mem::forget(address);
            }
            if let Some(memory) = self.memory.take() {
                std::mem::forget(memory);
            }
            std::mem::forget(self.owner.clone());
        }
    }
}
