// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux allocation owner selected by KFD or DRM backing.
//!
//! The core sees one allocation contract. KFD and DRM retain separate
//! lifecycle records so each cleanup path resumes only its own unfinished
//! work.

use super::error;
use super::imported_system::DrmImportedSystem;
use super::memory::{self, DeviceVm, KfdAllocation, KfdHostAlias};
use super::registered_host::DrmRegisteredHost;
use crate::memory::host::HostBackingLease;
use crate::memory::interop::linux::{AisFileOperation, AisFileResult, DmaBuf, KfdIpcMemoryHandle};
use crate::memory::storage::{Owned, Shared};
use crate::memory::{AllocationDesc, AllocationInfo, DeviceAccess};
use crate::{Error, ErrorKind};

/// Allocation ownership selected by the native backing and import path.
pub(crate) enum LinuxAllocation {
    /// KFD allocation with its retained VM mappings.
    Kfd(Owned<KfdAllocation>),
    /// Borrow of a mapped KFD BO's CPU range without another USERPTR BO.
    KfdHostAlias(Owned<KfdHostAlias>),
    /// Borrowed or runtime-owned host pages registered through DRM USERPTR.
    DrmRegisteredHost(Owned<DrmRegisteredHost>),
    /// Imported system-memory DMA-BUF mapped through the render VM.
    DrmImportedSystem(Owned<DrmImportedSystem>),
}

impl LinuxAllocation {
    fn from_kfd(allocation: Owned<KfdAllocation>) -> Result<Owned<Self>, Error> {
        Ok(Owned::new(Self::Kfd(allocation))?)
    }

    pub(super) fn from_host_alias(alias: Owned<KfdHostAlias>) -> Result<Owned<Self>, Error> {
        Ok(Owned::new(Self::KfdHostAlias(alias))?)
    }

    pub(super) fn create_scratch(
        vm: &Shared<DeviceVm>,
        desc: AllocationDesc,
    ) -> Result<Owned<Self>, Error> {
        Self::from_kfd(KfdAllocation::create_scratch(vm, desc)?)
    }

    pub(super) fn create_mmio(vm: &Shared<DeviceVm>) -> Result<Owned<Self>, Error> {
        Self::from_kfd(KfdAllocation::create_mmio(vm)?)
    }

    pub(super) fn create_with_peers(
        vm: Shared<DeviceVm>,
        peers: impl ExactSizeIterator<Item = Shared<DeviceVm>>,
        desc: AllocationDesc,
        kind: memory::BufferKind,
        permissions: DeviceAccess,
        backing: Option<HostBackingLease>,
    ) -> Result<Owned<Self>, Error> {
        Self::from_kfd(KfdAllocation::create_with_peers(
            vm,
            peers,
            desc,
            kind,
            permissions,
            backing,
        )?)
    }

    /// # Safety
    /// Borrowed callers retain and synchronize the entire host page cover
    /// through cleanup; an owned request transfers that obligation in `backing`.
    #[allow(unsafe_code)]
    pub(super) unsafe fn create_registered_host(
        vm: Shared<DeviceVm>,
        peers: impl ExactSizeIterator<Item = Shared<DeviceVm>>,
        desc: AllocationDesc,
        address: usize,
        permissions: DeviceAccess,
        uncached: bool,
        backing: Option<HostBackingLease>,
    ) -> Result<Owned<Self>, Error> {
        let owner = Owned::try_new_uninit()?;
        // SAFETY: The caller of this function owns the retained-page contract.
        let registration = unsafe {
            DrmRegisteredHost::create(vm, peers, desc, address, permissions, uncached, backing)?
        };
        Ok(owner.write(Self::DrmRegisteredHost(registration)))
    }

    pub(super) fn import_dma_buf(
        vm: Shared<DeviceVm>,
        descriptor: i32,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self>, Error> {
        Self::from_kfd(KfdAllocation::import_dma_buf(
            vm,
            descriptor,
            source_offset,
            byte_length,
            alignment,
            permissions,
        )?)
    }

    pub(super) fn import_system_dma_buf(
        vm: Shared<DeviceVm>,
        descriptor: i32,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self>, Error> {
        let owner = Owned::try_new_uninit()?;
        let imported = DrmImportedSystem::create(
            vm,
            descriptor,
            source_offset,
            byte_length,
            alignment,
            permissions,
        )?;
        Ok(owner.write(Self::DrmImportedSystem(imported)))
    }

    pub(super) fn import_graphics_dma_buf(
        vm: Shared<DeviceVm>,
        peers: impl ExactSizeIterator<Item = Shared<DeviceVm>>,
        descriptor: i32,
        size_hint: u64,
    ) -> Result<Owned<Self>, Error> {
        Self::from_kfd(KfdAllocation::import_graphics_dma_buf(
            vm, peers, descriptor, size_hint,
        )?)
    }

    pub(super) fn import_ipc(
        vm: Shared<DeviceVm>,
        mappings: impl ExactSizeIterator<Item = Shared<DeviceVm>>,
        words: [u32; 8],
        size: u64,
    ) -> Result<Owned<Self>, Error> {
        Self::from_kfd(KfdAllocation::import_ipc(vm, mappings, words, size)?)
    }

    pub(crate) fn cached_info(&self) -> AllocationInfo {
        match self {
            Self::Kfd(allocation) => allocation.cached_info(),
            Self::KfdHostAlias(alias) => alias.cached_info(),
            Self::DrmRegisteredHost(allocation) => allocation.cached_info(),
            Self::DrmImportedSystem(allocation) => allocation.cached_info(),
        }
    }

    pub(crate) fn host_alias_source_offset(&self) -> Option<u64> {
        match self {
            Self::KfdHostAlias(alias) => Some(alias.source_offset()),
            Self::Kfd(_) | Self::DrmRegisteredHost(_) | Self::DrmImportedSystem(_) => None,
        }
    }

    pub(crate) fn set_host_alias_identity(&self, identity: [u64; 2]) -> Result<(), Error> {
        match self {
            Self::Kfd(allocation) => allocation.set_host_alias_identity(identity),
            Self::KfdHostAlias(_) | Self::DrmRegisteredHost(_) | Self::DrmImportedSystem(_) => {
                Err(error(
                    ErrorKind::Unsupported,
                    "allocation cannot publish a host alias identity",
                ))
            }
        }
    }

    pub(crate) fn metadata(&self) -> &[u8] {
        match self {
            Self::Kfd(allocation) => allocation.metadata(),
            Self::KfdHostAlias(_) | Self::DrmRegisteredHost(_) | Self::DrmImportedSystem(_) => &[],
        }
    }

    pub(super) fn check(&self) -> Result<(), Error> {
        match self {
            Self::Kfd(allocation) => allocation.check(),
            Self::KfdHostAlias(alias) => alias.check(),
            Self::DrmRegisteredHost(allocation) => allocation.check(),
            Self::DrmImportedSystem(allocation) => allocation.check(),
        }
    }

    pub(super) fn is_device_local(&self) -> bool {
        match self {
            Self::Kfd(allocation) => allocation.is_device_local(),
            Self::KfdHostAlias(_) | Self::DrmRegisteredHost(_) | Self::DrmImportedSystem(_) => {
                false
            }
        }
    }

    pub(super) fn set_access(&mut self, devices: &[&Shared<DeviceVm>]) -> Result<(), Error> {
        match self {
            Self::Kfd(allocation) => allocation.set_access(devices),
            Self::KfdHostAlias(_) | Self::DrmRegisteredHost(_) | Self::DrmImportedSystem(_) => {
                Err(error(
                    ErrorKind::Unsupported,
                    "this allocation cannot change its native device mappings",
                ))
            }
        }
    }

    pub(super) fn device_address(&self, device: &Shared<DeviceVm>) -> Result<u64, Error> {
        match self {
            Self::Kfd(allocation) => allocation.device_address(device),
            Self::KfdHostAlias(alias) => alias.device_address(device),
            Self::DrmRegisteredHost(allocation) => allocation.device_address(device),
            Self::DrmImportedSystem(allocation) => allocation.device_address(device),
        }
    }

    pub(super) fn is_owned_by(&self, device: &Shared<DeviceVm>) -> bool {
        match self {
            Self::Kfd(allocation) => allocation.is_owned_by(device),
            Self::KfdHostAlias(alias) => alias.is_owned_by(device),
            Self::DrmRegisteredHost(allocation) => allocation.is_owned_by(device),
            Self::DrmImportedSystem(allocation) => allocation.is_owned_by(device),
        }
    }

    pub(super) fn free(&mut self) -> Result<(), Error> {
        match self {
            Self::Kfd(allocation) => allocation.free(),
            Self::KfdHostAlias(alias) => {
                alias.free();
                Ok(())
            }
            Self::DrmRegisteredHost(allocation) => allocation.free(),
            Self::DrmImportedSystem(allocation) => allocation.free(),
        }
    }

    pub(crate) fn remaining_backing_byte_length(&self) -> u64 {
        match self {
            Self::Kfd(allocation) => allocation.remaining_backing_byte_length(),
            Self::KfdHostAlias(_) | Self::DrmRegisteredHost(_) | Self::DrmImportedSystem(_) => 0,
        }
    }

    pub(super) fn export_dma_buf(&self) -> Result<DmaBuf, Error> {
        match self {
            Self::Kfd(allocation) => allocation.export_dma_buf(),
            Self::KfdHostAlias(_) | Self::DrmRegisteredHost(_) => Err(error(
                ErrorKind::Unsupported,
                "borrowed host pages have no DMA-BUF export",
            )),
            Self::DrmImportedSystem(allocation) => allocation.export_dma_buf(),
        }
    }

    pub(super) fn export_ipc_memory(&self) -> Result<KfdIpcMemoryHandle, Error> {
        match self {
            Self::Kfd(allocation) => allocation.export_ipc_memory(),
            Self::KfdHostAlias(_) | Self::DrmRegisteredHost(_) => Err(error(
                ErrorKind::Unsupported,
                "borrowed host pages have no KFD IPC export",
            )),
            Self::DrmImportedSystem(_) => Err(error(
                ErrorKind::Unsupported,
                "DRM-imported storage has no KFD IPC export",
            )),
        }
    }

    pub(super) fn ais_transfer(
        &self,
        descriptor: i32,
        offset: u64,
        size: u64,
        file_offset: i64,
        operation: AisFileOperation,
    ) -> Result<AisFileResult, Error> {
        let Self::Kfd(allocation) = self else {
            return Err(error(
                ErrorKind::Unsupported,
                "AIS requires a KFD VRAM allocation",
            ));
        };
        let operation = match operation {
            AisFileOperation::Read => super::uapi::AIS_READ,
            AisFileOperation::Write => super::uapi::AIS_WRITE,
        };
        let output = allocation.ais_transfer(descriptor, offset, size, file_offset, operation)?;
        Ok(AisFileResult {
            size_copied: output.size_copied,
            status: output.status,
        })
    }

    pub(super) fn signal_event_page_handle(&self, device: &Shared<DeviceVm>) -> Result<u64, Error> {
        match self {
            Self::Kfd(allocation) => allocation.signal_event_page_handle(device),
            Self::KfdHostAlias(_) => Err(error(
                ErrorKind::InvalidArgument,
                "host alias is not a KFD event page",
            )),
            Self::DrmRegisteredHost(_) => Err(error(
                ErrorKind::InvalidArgument,
                "DRM registration is not a KFD event page",
            )),
            Self::DrmImportedSystem(_) => Err(error(
                ErrorKind::InvalidArgument,
                "DRM-imported storage is not a KFD event page",
            )),
        }
    }

    pub(super) fn retain_signal_event_page(&mut self) -> Result<(), Error> {
        match self {
            Self::Kfd(allocation) => allocation.retain_signal_event_page(),
            Self::KfdHostAlias(_) => Err(error(
                ErrorKind::InvalidArgument,
                "host alias is not a KFD event page",
            )),
            Self::DrmRegisteredHost(_) => Err(error(
                ErrorKind::InvalidArgument,
                "DRM registration is not a KFD event page",
            )),
            Self::DrmImportedSystem(_) => Err(error(
                ErrorKind::InvalidArgument,
                "DRM-imported storage is not a KFD event page",
            )),
        }
    }
}
