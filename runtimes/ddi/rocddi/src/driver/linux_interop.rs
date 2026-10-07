// SPDX-License-Identifier: MIT

//! Private Linux memory-interop contract implemented by the selected backend.
//!
//! Keeping this contract separate from the portable driver facets prevents DMA-BUF file
//! descriptors from becoming requirements for future non-Linux drivers.

use std::io;
use std::os::fd::{BorrowedFd, OwnedFd, RawFd};

use crate::Error;
use crate::event::GpuMemoryFault;
use crate::host_storage::Owned;
use crate::memory::DeviceAccess;
use crate::memory::interop::linux::{
    AisFileOperation, AisFileResult, DmaBuf, KfdIpcMemoryHandle, KfdSvmAttribute,
};

use super::{
    AllocationDriver, DeviceDriver, DeviceState, NativeAllocation, NativeSignalEvent,
    VirtualMemoryDriver,
};

/// Linux-only DMA-BUF operations supplied by the active native driver.
pub(crate) trait LinuxMemoryInteropDriver: AllocationDriver + VirtualMemoryDriver {
    #[allow(unsafe_code)]
    unsafe fn close_owned_descriptor(descriptor: RawFd) -> io::Result<()>;
    fn duplicate_descriptor(descriptor: RawFd) -> Result<OwnedFd, Error>;
    fn descriptor_length(descriptor: RawFd) -> io::Result<u64>;
    fn read_descriptor_exact_at(
        descriptor: RawFd,
        buffer: &mut [u8],
        offset: u64,
    ) -> io::Result<()>;
    fn read_descriptor_at(descriptor: RawFd, buffer: &mut [u8], offset: i64) -> io::Result<usize>;
    fn write_descriptor_at(descriptor: RawFd, buffer: &[u8], offset: i64) -> io::Result<usize>;
    fn ais_transfer(
        allocation: &Self::Allocation,
        descriptor: RawFd,
        allocation_offset: u64,
        size: u64,
        file_offset: i64,
        operation: AisFileOperation,
    ) -> Result<AisFileResult, Error>;
    fn supports_system_dma_buf_import(device: &Self::DeviceState) -> bool;

    fn import_virtual_memory(
        &self,
        descriptor: BorrowedFd<'_>,
    ) -> Result<Owned<Self::VirtualMemory>, Error>;

    fn export_virtual_memory(memory: &Self::VirtualMemory) -> Result<DmaBuf, Error>;

    fn import_dma_buf(
        &self,
        device: &Self::DeviceState,
        descriptor: BorrowedFd<'_>,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self::Allocation>, Error>;

    fn import_system_dma_buf(
        &self,
        devices: &[&Self::DeviceState],
        descriptor: RawFd,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self::Allocation>, Error>;

    fn import_graphics_dma_buf(
        &self,
        devices: &[&Self::DeviceState],
        descriptor: BorrowedFd<'_>,
        size_hint: u64,
    ) -> Result<Owned<Self::Allocation>, Error>;

    fn export_dma_buf(allocation: &Self::Allocation) -> Result<DmaBuf, Error>;

    fn import_kfd_ipc_memory(
        &self,
        devices: &[&Self::DeviceState],
        mapping_devices: &[&Self::DeviceState],
        handle: KfdIpcMemoryHandle,
        size: u64,
    ) -> Result<Owned<Self::Allocation>, Error>;

    fn export_kfd_ipc_memory(allocation: &Self::Allocation) -> Result<KfdIpcMemoryHandle, Error>;

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
}

/// Linux KFD event operations supplied by the active native driver.
pub(crate) trait LinuxGpuEventDriver: DeviceDriver {
    fn retain_kfd_signal_event_page(allocation: &mut NativeAllocation) -> Result<(), Error>;

    fn create_kfd_signal_event(
        &self,
        device: &DeviceState,
        event_page: Option<&NativeAllocation>,
        page_offered: &mut bool,
    ) -> Result<Owned<NativeSignalEvent>, Error>;

    fn destroy_kfd_signal_event(event: &mut NativeSignalEvent) -> Result<(), Error>;

    fn poll_kfd_memory_fault(&self, device: &DeviceState) -> Result<Option<GpuMemoryFault>, Error>;
}
