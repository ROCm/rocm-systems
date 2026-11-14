// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Qualified GTT DMA-BUF attachment with independent mappings in each GPU VM.
//!
//! The BO is validated through its KFD and DRM views before it is published.
//! Detached virtual-memory owners provide exact GPU PTE mappings and one
//! write-back host view; neither the caller's descriptor nor the AMDF transport
//! callback is retained. Failed native detach retains the VA and backing so a
//! GPU address cannot be reused while it may still be mapped.

use crate::memory::storage::TryPush;
use std::os::fd::AsRawFd;

use super::drm;
use super::memory::DeviceVm;
use super::sysfs;
use super::uapi;
use super::vmem::{
    KfdVirtualAddress, KfdVirtualDeviceMapping, KfdVirtualHostMapping, KfdVirtualMemory,
};
use super::{error, native_error};
use super::{os_file, os_memory};
use crate::memory::interop::linux::DmaBuf;
use crate::memory::storage::{Owned, Shared};
use crate::memory::{AllocationInfo, DeviceAccess};
use crate::{Error, ErrorKind};

fn local_gem(render: &std::fs::File, handle: u32) -> Result<bool, Error> {
    let mut entries = Vec::new();
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
    let file_info = os_file::dma_buf_file_info(dma_buf)
        .map_err(|source| native_error("DMA-BUF backing information", source))?;
    let details = vm
        .kfd()
        .dma_buf_info(dma_buf.as_raw_fd())
        .map_err(|source| native_error("AMDKFD_IOC_GET_DMABUF_INFO", source))?;
    let kfd = details.info;
    if kfd.gpu_id != vm.gpu_id || kfd.flags & !uapi::PUBLIC != uapi::GTT {
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
    let render = sysfs::open_render(vm.render_minor)
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
        || !local_gem(&render, handle)?
    {
        return Err(error(
            ErrorKind::Unsupported,
            "DMA-BUF lacks a qualified same-device coherent GTT backing",
        ));
    }
    Ok(())
}

/// Every participating VM must belong to the same native connection and offer
/// the exact DRM system-import mapping contract used by the owning VM.
fn validate_vm(owner: &DeviceVm, device: &DeviceVm) -> Result<(), Error> {
    device.check()?;
    if !device.system_dma_buf_import || !owner.shares_kfd(device) {
        return Err(error(
            ErrorKind::Unsupported,
            "system import requires compatible native GPU VMs",
        ));
    }
    Ok(())
}

/// One imported physical backing and its host/device views at a common VA.
struct SystemMapping {
    /// Exact VM identity, including its creating render context.
    vm: Shared<DeviceVm>,
    /// Independently retryable native map/GEM owner.
    mapping: Owned<KfdVirtualDeviceMapping>,
    /// An incomplete revocation is not reported as current accessibility.
    retiring: bool,
}

/// One imported physical backing and its host/device views at a common VA.
pub(crate) struct DrmImportedSystem {
    /// Creating render VM retained through import cleanup.
    owner: Shared<DeviceVm>,
    /// Imported physical backing owner.
    memory: Option<Owned<KfdVirtualMemory>>,
    /// Common virtual address reservation.
    address: Option<Owned<KfdVirtualAddress>>,
    /// Device mapping of the imported backing.
    mappings: Vec<SystemMapping>,
    /// Host mapping of the imported backing.
    host: Option<Owned<KfdVirtualHostMapping>>,
    /// Byte offset into the source DMA-BUF.
    source_offset: u64,
    /// Public byte extent requested by the caller.
    logical_size: u64,
    /// Native physical extent covering that request.
    native_size: u64,
    /// Stable identity of the imported physical backing.
    physical_backing_id: [u64; 2],
    /// Immutable GPU permissions applied to every access-set member.
    permissions: DeviceAccess,
    /// Prevents new mappings once release begins.
    freeing: bool,
}

impl DrmImportedSystem {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create(
        vm: Shared<DeviceVm>,
        peers: impl ExactSizeIterator<Item = Shared<DeviceVm>>,
        descriptor: i32,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self>, Error> {
        if !vm.system_dma_buf_import {
            return Err(error(
                ErrorKind::Unsupported,
                "DRM system import is unavailable",
            ));
        }
        let page = os_memory::page_size()
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
        let mut devices = Vec::new();
        devices.try_reserve(peers.len() + 1)?;
        devices.push(vm.clone());
        for peer in peers {
            if !devices.iter().any(|device| Shared::ptr_eq(device, &peer)) {
                devices.push(peer);
            }
        }
        let mut bounds = vm.address_range();
        for device in &devices {
            validate_vm(&vm, device)?;
            let range = device.address_range();
            bounds = (bounds.0.max(range.0), bounds.1.min(range.1));
        }

        let memory = KfdVirtualMemory::import(descriptor)?;
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
        let address = KfdVirtualAddress::reserve(bounds, native_size, alignment, 0)?;
        Self::map_backing(
            vm,
            &devices,
            memory,
            address,
            source_offset,
            byte_length,
            permissions,
        )
    }

    /// Acquires per-VM views only after the backing and common VA are validated.
    #[allow(clippy::too_many_arguments)]
    fn map_backing(
        vm: Shared<DeviceVm>,
        devices: &[Shared<DeviceVm>],
        memory: Owned<KfdVirtualMemory>,
        address: Owned<KfdVirtualAddress>,
        source_offset: u64,
        byte_length: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self>, Error> {
        let native_size = memory.info().size;
        let slot = Owned::<Self>::try_new_uninit()?;
        let mut imported = slot.write(Self {
            owner: vm,
            physical_backing_id: memory.info().physical_backing_id,
            memory: Some(memory),
            address: Some(address),
            mappings: Vec::new(),
            host: None,
            source_offset,
            logical_size: byte_length,
            native_size,
            permissions,
            freeing: false,
        });
        let base = imported.base()?;
        imported.mappings.try_reserve(devices.len())?;
        for device in devices {
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
                device.clone(),
                base,
                0,
                native_size,
                permissions,
            )?;
            imported.mappings.push(SystemMapping {
                vm: device.clone(),
                mapping,
                retiring: false,
            });
        }
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
        if self.freeing || self.address.is_none() || self.memory.is_none() || self.host.is_none() {
            return Err(error(
                ErrorKind::InvalidArgument,
                "system import was released",
            ));
        }
        Ok(())
    }

    pub(crate) fn device_address(&self, device: &Shared<DeviceVm>) -> Result<u64, Error> {
        self.check()?;
        device.check()?;
        if !self
            .mappings
            .iter()
            .any(|mapping| Shared::ptr_eq(&mapping.vm, device) && !mapping.retiring)
        {
            return Err(error(
                ErrorKind::Unsupported,
                "system import is not mapped into this device VM",
            ));
        }
        Ok(self.base()? + self.source_offset)
    }

    /// Maps newly requested VMs before revoking old mappings, preserving each
    /// partial native transition for retry and accurate accessibility queries.
    pub(super) fn set_access(&mut self, devices: &[&Shared<DeviceVm>]) -> Result<(), Error> {
        self.check()?;
        // An uncertain peer MAP has no returned mapping owner here. Even a
        // shrinking access set must not report success while it is unresolved.
        self.address
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "system import lost reservation"))?
            .check()?;
        let base = self.base()?;
        let end = base
            .checked_add(self.native_size - 1)
            .ok_or_else(|| error(ErrorKind::InvalidArgument, "system import address overflow"))?;
        for device in devices {
            validate_vm(&self.owner, device)?;
            let range = device.address_range();
            if base < range.0 || end > range.1 {
                return Err(error(
                    ErrorKind::Unsupported,
                    "system import lies outside a requested GPU aperture",
                ));
            }
        }
        self.mappings.try_reserve(devices.len())?;
        // Finish a previously interrupted revocation before attempting to map
        // that same VA again, even when a retry requests the VM back.
        let mut index = 0;
        while index < self.mappings.len() {
            if self.mappings[index].retiring {
                self.mappings[index].mapping.free()?;
                self.mappings.remove(index);
            } else {
                index += 1;
            }
        }
        for &device in devices {
            if self
                .mappings
                .iter()
                .any(|mapping| Shared::ptr_eq(&mapping.vm, device))
            {
                continue;
            }
            let mapping = KfdVirtualDeviceMapping::create(
                self.memory
                    .as_ref()
                    .ok_or_else(|| error(ErrorKind::Internal, "system import lost backing"))?,
                self.address
                    .as_ref()
                    .ok_or_else(|| error(ErrorKind::Internal, "system import lost reservation"))?,
                device.clone(),
                base,
                0,
                self.native_size,
                self.permissions,
            )?;
            self.mappings.push(SystemMapping {
                vm: device.clone(),
                mapping,
                retiring: false,
            });
        }
        let mut index = 0;
        while index < self.mappings.len() {
            if devices
                .iter()
                .any(|device| Shared::ptr_eq(device, &self.mappings[index].vm))
            {
                index += 1;
                continue;
            }
            self.mappings[index].retiring = true;
            self.mappings[index].mapping.free()?;
            self.mappings.remove(index);
        }
        Ok(())
    }

    pub(crate) fn is_owned_by(&self, device: &Shared<DeviceVm>) -> bool {
        self.owner.shares_kfd(device) && self.owner.gpu_id == device.gpu_id
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
        while let Some(mapping) = self.mappings.last_mut() {
            mapping.retiring = true;
            mapping.mapping.free()?;
            self.mappings.pop();
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
        if let Err(error) = self.free() {
            crate::memory::device::record_rollback(error, 0);
            for mapping in self.mappings.drain(..) {
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::{memory, sys};
    use super::*;
    use std::fs::{File, OpenOptions};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FILE_ID: AtomicUsize = AtomicUsize::new(0);

    fn page() -> u64 {
        os_memory::page_size().unwrap() as u64
    }

    fn backing() -> File {
        let path = std::env::temp_dir().join(format!(
            "rocddi-system-import-{}-{}",
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
        file.set_len(page()).unwrap();
        file
    }

    fn fixtures() -> [Shared<DeviceVm>; 2] {
        let kfd = Shared::new(sys::Kfd::with_hook(
            File::open("/dev/null").unwrap(),
            Arc::new(|call| {
                if let sys::Call::Wait(args, _) = call {
                    args.result = uapi::WAIT_TIMEOUT;
                }
                Ok(())
            }),
        ))
        .unwrap();
        std::array::from_fn(|index| {
            let node = sysfs::KfdNode {
                host_atomics: false,
                node: u32::try_from(index).unwrap() + 1,
                gpu_id: u32::try_from(index).unwrap() + 42,
                render_minor: Some(u32::try_from(index).unwrap() + 128),
                unique_id: Some(index as u64 + 123),
                identity: [u8::try_from(index).unwrap(); 16],
                queues: sysfs::KfdQueueProperties {
                    gfx_target: 120_001,
                    target_profile: crate::topology::GpuTargetProfile::Gfx1201,
                    xcc_count: 1,
                    ..sysfs::KfdQueueProperties::default()
                },
                local_memory_bytes: 1 << 30,
                public_memory_bytes: 0,
            };
            let mut vm = memory::queue_fixture(kfd.clone(), backing(), node);
            Shared::get_mut(&mut vm).unwrap().system_dma_buf_import = true;
            vm
        })
    }

    fn mapped(devices: &[Shared<DeviceVm>]) -> Result<Owned<DrmImportedSystem>, Error> {
        let source = backing();
        let memory = KfdVirtualMemory::import(source.as_raw_fd())?;
        let address = KfdVirtualAddress::reserve((0x10000, isize::MAX as u64), page(), page(), 0)?;
        DrmImportedSystem::map_backing(
            devices[0].clone(),
            devices,
            memory,
            address,
            0,
            page(),
            DeviceAccess::READ | DeviceAccess::WRITE,
        )
    }

    #[test]
    fn system_import_publishes_one_address_and_keeps_two_independent_vm_owners() {
        let devices = fixtures();
        drm::with_script(
            [
                drm::TestCall::ImportGem(11),
                drm::TestCall::CreateSyncobj(1),
                drm::TestCall::Map(Ok(())),
                drm::TestCall::Wait(Ok(())),
                drm::TestCall::ImportGem(22),
                drm::TestCall::CreateSyncobj(2),
                drm::TestCall::Map(Ok(())),
                drm::TestCall::Wait(Ok(())),
                drm::TestCall::Unmap(Ok(())),
                drm::TestCall::Wait(Ok(())),
                drm::TestCall::CloseGem(Ok(())),
                drm::TestCall::Unmap(Ok(())),
                drm::TestCall::Wait(Ok(())),
                drm::TestCall::CloseGem(Ok(())),
                drm::TestCall::DestroySyncobj,
                drm::TestCall::DestroySyncobj,
            ],
            || {
                let mut imported = mapped(&devices).unwrap();
                assert_eq!(imported.mappings.len(), 2);
                let address = imported.device_address(&devices[0]).unwrap();
                assert_eq!(imported.device_address(&devices[1]).unwrap(), address);
                assert_eq!(
                    imported.cached_info().host_address,
                    Some(usize::try_from(address).unwrap())
                );
                imported
                    .set_access(&[&devices[0], &devices[1], &devices[0]])
                    .unwrap();
                assert_eq!(imported.mappings.len(), 2);
                imported.free().unwrap();
                for device in &devices {
                    device
                        .vmem
                        .lock()
                        .unwrap()
                        .close(device.render().unwrap())
                        .unwrap();
                }
            },
        );
    }

    #[test]
    fn system_import_maps_new_vm_before_revoking_old_and_retries_only_unfinished_unmap() {
        let devices = fixtures();
        drm::with_script(
            [
                drm::TestCall::ImportGem(11),
                drm::TestCall::CreateSyncobj(1),
                drm::TestCall::Map(Ok(())),
                drm::TestCall::Wait(Ok(())),
                drm::TestCall::ImportGem(22),
                drm::TestCall::CreateSyncobj(2),
                drm::TestCall::Map(Ok(())),
                drm::TestCall::Wait(Ok(())),
                drm::TestCall::Unmap(Ok(())),
                drm::TestCall::Wait(Err(super::super::errno::EIO)),
                drm::TestCall::Wait(Ok(())),
                drm::TestCall::CloseGem(Ok(())),
                drm::TestCall::Unmap(Ok(())),
                drm::TestCall::Wait(Ok(())),
                drm::TestCall::CloseGem(Ok(())),
                drm::TestCall::DestroySyncobj,
                drm::TestCall::DestroySyncobj,
            ],
            || {
                let mut imported = mapped(&devices[..1]).unwrap();
                let original = imported.device_address(&devices[0]).unwrap();
                assert!(imported.device_address(&devices[1]).is_err());
                assert!(imported.set_access(&[&devices[1], &devices[1]]).is_err());
                assert!(imported.device_address(&devices[0]).is_err());
                assert_eq!(imported.device_address(&devices[1]).unwrap(), original);
                // Query hides the old VM during its unresolved unmap. The retry
                // reuses the completed peer import and does not submit UNMAP twice.
                imported.set_access(&[&devices[1]]).unwrap();
                assert_eq!(imported.mappings.len(), 1);
                imported.free().unwrap();
                for device in &devices {
                    device
                        .vmem
                        .lock()
                        .unwrap()
                        .close(device.render().unwrap())
                        .unwrap();
                }
            },
        );
    }

    #[test]
    fn uncertain_peer_map_rejects_access_retries_but_cleans_known_mappings() {
        for failed_wait in [false, true] {
            let devices = fixtures();
            let mut calls = vec![
                drm::TestCall::ImportGem(11),
                drm::TestCall::CreateSyncobj(1),
                drm::TestCall::Map(Ok(())),
                drm::TestCall::Wait(Ok(())),
                drm::TestCall::ImportGem(22),
                drm::TestCall::CreateSyncobj(2),
            ];
            if failed_wait {
                calls.extend([
                    drm::TestCall::Map(Ok(())),
                    drm::TestCall::Wait(Err(super::super::errno::ETIME)),
                ]);
            } else {
                calls.push(drm::TestCall::Map(Err(super::super::errno::EFAULT)));
            }
            calls.extend([
                drm::TestCall::Unmap(Ok(())),
                drm::TestCall::Wait(Ok(())),
                drm::TestCall::CloseGem(Ok(())),
            ]);
            drm::with_script(calls, || {
                let mut imported = mapped(&devices[..1]).unwrap();
                let original = imported.device_address(&devices[0]).unwrap();
                assert!(imported.set_access(&[&devices[0], &devices[1]]).is_err());
                for requested in [&[&devices[0]][..], &[][..], &[&devices[1]][..]] {
                    assert_eq!(
                        imported.set_access(requested).unwrap_err().kind(),
                        ErrorKind::DriverContract
                    );
                }
                assert_eq!(imported.device_address(&devices[0]).unwrap(), original);
                assert!(imported.device_address(&devices[1]).is_err());
                assert_eq!(
                    imported.free().unwrap_err().kind(),
                    ErrorKind::DriverContract
                );
                assert!(imported.mappings.is_empty());
                assert!(imported.host.is_none());
                assert!(imported.address.is_some());
                assert!(imported.memory.is_some());
                // B's accepted or ambiguous MAP remains owned by its VM. No
                // retry may invent an UNMAP or release its backing.
                assert!(
                    devices[1]
                        .vmem
                        .lock()
                        .unwrap()
                        .close(devices[1].render().unwrap())
                        .is_err()
                );
            });
        }
    }

    #[test]
    fn ambiguous_second_vm_import_retains_backing_after_cleaning_known_first_mapping() {
        let devices = fixtures();
        drm::with_script(
            [
                drm::TestCall::ImportGem(11),
                drm::TestCall::CreateSyncobj(1),
                drm::TestCall::Map(Ok(())),
                drm::TestCall::Wait(Ok(())),
                drm::TestCall::ImportGem(22),
                drm::TestCall::CreateSyncobj(2),
                drm::TestCall::Map(Err(super::super::errno::EFAULT)),
                drm::TestCall::Unmap(Ok(())),
                drm::TestCall::Wait(Ok(())),
                drm::TestCall::CloseGem(Ok(())),
            ],
            || {
                let result = mapped(&devices);
                assert!(result.is_err());
                // The uncertain second map never receives an invented UNMAP or
                // GEM close. Its lower VM owner retains the marked backing/VA.
                assert!(
                    devices[1]
                        .vmem
                        .lock()
                        .unwrap()
                        .close(devices[1].render().unwrap())
                        .is_err()
                );
            },
        );
    }
}
