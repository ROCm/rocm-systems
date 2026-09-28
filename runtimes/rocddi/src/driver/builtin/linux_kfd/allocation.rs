//! Linux allocation owner selected by the native backing mechanism.
//!
//! The core sees one allocation contract. KFD and DRM retain separate
//! lifecycle records so neither native cleanup mechanism contaminates the
//! other's state machine.

use super::imported_system::DrmImportedSystem;
use super::memory::{self, DeviceVm, KfdAllocation, error};
use super::registered_host::DrmRegisteredHost;
use crate::host_storage::{Owned, Shared};
use crate::memory::interop::linux::{DmaBuf, KfdIpcMemoryHandle};
use crate::memory::{AllocationDesc, AllocationInfo, DeviceAccess};
use crate::{Error, ErrorKind};

pub(crate) enum NativeAllocation {
    Kfd(Owned<KfdAllocation>),
    DrmRegisteredHost(Owned<DrmRegisteredHost>),
    DrmImportedSystem(Owned<DrmImportedSystem>),
}

impl NativeAllocation {
    fn from_kfd(allocation: Owned<KfdAllocation>) -> Result<Owned<Self>, Error> {
        let allocator = allocation.allocator();
        Ok(Owned::new(Self::Kfd(allocation), allocator)?)
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
    ) -> Result<Owned<Self>, Error> {
        Self::from_kfd(KfdAllocation::create_with_peers(
            vm,
            peers,
            desc,
            kind,
            permissions,
        )?)
    }

    pub(super) fn create_registered_host(
        vm: Shared<DeviceVm>,
        peers: impl ExactSizeIterator<Item = Shared<DeviceVm>>,
        desc: AllocationDesc,
        address: usize,
        permissions: DeviceAccess,
        uncached: bool,
    ) -> Result<Owned<Self>, Error> {
        let allocator = vm.allocator();
        let owner = Owned::try_new_uninit(allocator)?;
        let registration =
            DrmRegisteredHost::create(vm, peers, desc, address, permissions, uncached)?;
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
        let allocator = vm.allocator();
        let owner = Owned::try_new_uninit(allocator)?;
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
            Self::DrmRegisteredHost(allocation) => allocation.cached_info(),
            Self::DrmImportedSystem(allocation) => allocation.cached_info(),
        }
    }

    pub(crate) fn metadata(&self) -> &[u8] {
        match self {
            Self::Kfd(allocation) => allocation.metadata(),
            Self::DrmRegisteredHost(_) | Self::DrmImportedSystem(_) => &[],
        }
    }

    pub(super) fn check(&self) -> Result<(), Error> {
        match self {
            Self::Kfd(allocation) => allocation.check(),
            Self::DrmRegisteredHost(allocation) => allocation.check(),
            Self::DrmImportedSystem(allocation) => allocation.check(),
        }
    }

    pub(super) fn device_address(&self, device: &Shared<DeviceVm>) -> Result<u64, Error> {
        match self {
            Self::Kfd(allocation) => allocation.device_address(device),
            Self::DrmRegisteredHost(allocation) => allocation.device_address(device),
            Self::DrmImportedSystem(allocation) => allocation.device_address(device),
        }
    }

    pub(super) fn is_owned_by(&self, device: &Shared<DeviceVm>) -> bool {
        match self {
            Self::Kfd(allocation) => allocation.is_owned_by(device),
            Self::DrmRegisteredHost(allocation) => allocation.is_owned_by(device),
            Self::DrmImportedSystem(allocation) => allocation.is_owned_by(device),
        }
    }

    pub(super) fn free(&mut self) -> Result<(), Error> {
        match self {
            Self::Kfd(allocation) => allocation.free(),
            Self::DrmRegisteredHost(allocation) => allocation.free(),
            Self::DrmImportedSystem(allocation) => allocation.free(),
        }
    }

    pub(super) fn export_dma_buf(&self) -> Result<DmaBuf, Error> {
        match self {
            Self::Kfd(allocation) => allocation.export_dma_buf(),
            Self::DrmRegisteredHost(_) => Err(error(
                ErrorKind::Unsupported,
                "borrowed host pages have no DMA-BUF export",
            )),
            Self::DrmImportedSystem(allocation) => allocation.export_dma_buf(),
        }
    }

    pub(super) fn export_ipc_memory(&self) -> Result<KfdIpcMemoryHandle, Error> {
        match self {
            Self::Kfd(allocation) => allocation.export_ipc_memory(),
            Self::DrmRegisteredHost(_) => Err(error(
                ErrorKind::Unsupported,
                "borrowed host pages have no KFD IPC export",
            )),
            Self::DrmImportedSystem(_) => Err(error(
                ErrorKind::Unsupported,
                "DRM-imported storage has no KFD IPC export",
            )),
        }
    }

    pub(super) fn signal_event_page_handle(&self, device: &Shared<DeviceVm>) -> Result<u64, Error> {
        match self {
            Self::Kfd(allocation) => allocation.signal_event_page_handle(device),
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
