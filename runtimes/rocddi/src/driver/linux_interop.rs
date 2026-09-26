//! Private Linux memory-interop contract implemented by the selected backend.
//!
//! Keeping this contract separate from the portable driver facets prevents DMA-BUF file
//! descriptors from becoming requirements for future non-Linux drivers.

use std::os::fd::{BorrowedFd, RawFd};

use crate::Error;
use crate::event::GpuMemoryFault;
use crate::host_storage::Owned;
use crate::memory::DeviceAccess;
use crate::memory::interop::linux::{DmaBuf, KfdIpcMemoryHandle, KfdSvmAttribute};

use super::{
    AllocationDriver, DeviceDriver, DeviceState, NativeAllocation, NativeSignalEvent,
    NativeVirtualMemory, VirtualMemoryDriver,
};

/// Linux-only DMA-BUF operations supplied by the active native driver.
pub(crate) trait LinuxMemoryInteropDriver: AllocationDriver + VirtualMemoryDriver {
    fn supports_system_dma_buf_import(device: &DeviceState) -> bool;

    fn import_virtual_memory(
        &self,
        descriptor: BorrowedFd<'_>,
    ) -> Result<Owned<NativeVirtualMemory>, Error>;

    fn export_virtual_memory(memory: &NativeVirtualMemory) -> Result<DmaBuf, Error>;

    fn import_dma_buf(
        &self,
        device: &DeviceState,
        descriptor: BorrowedFd<'_>,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<NativeAllocation>, Error>;

    fn import_system_dma_buf(
        &self,
        devices: &[&DeviceState],
        descriptor: RawFd,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<NativeAllocation>, Error>;

    fn import_graphics_dma_buf(
        &self,
        devices: &[&DeviceState],
        descriptor: BorrowedFd<'_>,
        size_hint: u64,
    ) -> Result<Owned<NativeAllocation>, Error>;

    fn export_dma_buf(allocation: &NativeAllocation) -> Result<DmaBuf, Error>;

    fn import_kfd_ipc_memory(
        &self,
        devices: &[&DeviceState],
        mapping_devices: &[&DeviceState],
        handle: KfdIpcMemoryHandle,
        size: u64,
    ) -> Result<Owned<NativeAllocation>, Error>;

    fn export_kfd_ipc_memory(allocation: &NativeAllocation) -> Result<KfdIpcMemoryHandle, Error>;

    fn set_kfd_svm_attributes(
        &self,
        address: u64,
        size: u64,
        attributes: &[KfdSvmAttribute],
    ) -> Result<(), Error>;

    fn get_kfd_svm_attributes(
        &self,
        address: u64,
        size: u64,
        attributes: &mut [KfdSvmAttribute],
    ) -> Result<(), Error>;

    fn retain_kfd_signal_event_page(allocation: &mut NativeAllocation) -> Result<(), Error>;
}

/// Linux KFD event operations supplied by the active native driver.
pub(crate) trait LinuxGpuEventDriver: DeviceDriver {
    fn create_kfd_signal_event(
        &self,
        device: &DeviceState,
        event_page: Option<&NativeAllocation>,
    ) -> Result<Owned<NativeSignalEvent>, Error>;

    fn destroy_kfd_signal_event(event: &mut NativeSignalEvent) -> Result<(), Error>;

    fn poll_kfd_memory_fault(&self, device: &DeviceState) -> Result<Option<GpuMemoryFault>, Error>;
}
