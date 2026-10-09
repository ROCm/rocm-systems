// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux file-descriptor, shared-memory, and KFD event operations.
//!
//! These are concrete Linux services used by rocddi's Linux interop
//! modules. They do not belong to the portable driver capabilities because
//! their inputs and outputs carry Linux descriptors or KFD event state.
//! Imports retain or duplicate the descriptor state needed after return;
//! resource owners keep it through successful cleanup or a retryable failure.

// This module extends the concrete driver's Linux implementation scope.
#[allow(clippy::wildcard_imports)]
use super::*;

/// Closes a raw descriptor whose ownership the caller transfers to this
/// function.
///
/// # Safety
/// If `descriptor` names an open file, the caller must own it. No other
/// owner may close it or use the integer after this call, even when close
/// fails. An invalid descriptor is permitted and reported by the kernel.
#[allow(unsafe_code)]
pub(crate) unsafe fn close_owned_descriptor(descriptor: RawFd) -> io::Result<()> {
    util::close_descriptor(descriptor)
}

pub(crate) fn duplicate_descriptor(descriptor: RawFd) -> Result<OwnedFd, Error> {
    util::duplicate_file(descriptor)
        .map(Into::into)
        .map_err(|source| Error::NativeOperation {
            kind: match source.raw_os_error() {
                // Linux EBADF and the explicit negative-descriptor rejection.
                Some(9) => ErrorKind::InvalidArgument,
                // Linux ENFILE, EMFILE, and ENOMEM.
                Some(23 | 24 | 12) => ErrorKind::ResourceExhausted,
                _ if source.kind() == io::ErrorKind::InvalidInput => ErrorKind::InvalidArgument,
                _ => ErrorKind::Driver,
            },
            operation: "descriptor duplication",
            source,
        })
}

pub(crate) fn descriptor_length(descriptor: RawFd) -> io::Result<u64> {
    util::descriptor_length(descriptor)
}

pub(crate) fn read_descriptor_exact_at(
    descriptor: RawFd,
    buffer: &mut [u8],
    offset: u64,
) -> io::Result<()> {
    util::read_descriptor_exact_at(descriptor, buffer, offset)
}

pub(crate) fn read_descriptor_at(
    descriptor: RawFd,
    buffer: &mut [u8],
    offset: i64,
) -> io::Result<usize> {
    util::read_descriptor_at(descriptor, buffer, offset)
}

pub(crate) fn write_descriptor_at(
    descriptor: RawFd,
    buffer: &[u8],
    offset: i64,
) -> io::Result<usize> {
    util::write_descriptor_at(descriptor, buffer, offset)
}

impl LinuxKfdDriver {
    pub(crate) fn ais_transfer(
        allocation: &LinuxAllocation,
        descriptor: RawFd,
        allocation_offset: u64,
        size: u64,
        file_offset: i64,
        operation: AisFileOperation,
    ) -> Result<AisFileResult, Error> {
        allocation.ais_transfer(descriptor, allocation_offset, size, file_offset, operation)
    }

    pub(crate) fn supports_system_dma_buf_import(device: &DeviceState) -> bool {
        device.supports_system_dma_buf_import()
    }

    /// Imports a borrowed DMA-BUF as a virtual-memory owner.
    ///
    /// The returned owner retains the native object independently of the
    /// caller's descriptor and must be released through the normal VM path.
    pub(crate) fn import_virtual_memory(
        &self,
        descriptor: BorrowedFd<'_>,
    ) -> Result<Owned<KfdVirtualMemory>, Error> {
        self.ensure_open()?;
        KfdVirtualMemory::import(descriptor.as_raw_fd(), self.allocator)
    }

    pub(crate) fn export_virtual_memory(memory: &KfdVirtualMemory) -> Result<DmaBuf, Error> {
        memory.export_dma_buf()
    }

    pub(crate) fn import_dma_buf(
        &self,
        device: &DeviceState,
        descriptor: BorrowedFd<'_>,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<LinuxAllocation>, Error> {
        self.ensure_open()?;
        LinuxAllocation::import_dma_buf(
            device.vm.clone(),
            descriptor.as_raw_fd(),
            source_offset,
            byte_length,
            alignment,
            permissions,
        )
    }

    pub(crate) fn import_system_dma_buf(
        &self,
        devices: &[&DeviceState],
        descriptor: RawFd,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<LinuxAllocation>, Error> {
        self.ensure_open()?;
        let (owner, peers) = devices.split_first().ok_or_else(|| {
            error(
                ErrorKind::InvalidArgument,
                "system import requires a device",
            )
        })?;
        if !peers.iter().all(|peer| owner.shares_vm(peer)) {
            return Err(error(
                ErrorKind::Unsupported,
                "system import requires one native GPU address domain",
            ));
        }
        LinuxAllocation::import_system_dma_buf(
            owner.vm.clone(),
            descriptor,
            source_offset,
            byte_length,
            alignment,
            permissions,
        )
    }

    pub(crate) fn import_graphics_dma_buf(
        &self,
        devices: &[&DeviceState],
        descriptor: BorrowedFd<'_>,
        size_hint: u64,
    ) -> Result<Owned<LinuxAllocation>, Error> {
        self.ensure_open()?;
        let (device, peers) = devices.split_first().ok_or_else(|| {
            error(
                ErrorKind::InvalidArgument,
                "graphics import requires at least one device",
            )
        })?;
        LinuxAllocation::import_graphics_dma_buf(
            device.vm.clone(),
            peers.iter().map(|peer| peer.vm.clone()),
            descriptor.as_raw_fd(),
            size_hint,
        )
    }

    pub(crate) fn export_dma_buf(allocation: &LinuxAllocation) -> Result<DmaBuf, Error> {
        allocation.export_dma_buf()
    }

    pub(crate) fn import_kfd_ipc_memory(
        &self,
        devices: &[&DeviceState],
        mapping_devices: &[&DeviceState],
        handle: KfdIpcMemoryHandle,
        size: u64,
    ) -> Result<Owned<LinuxAllocation>, Error> {
        self.ensure_open()?;
        let words = handle.words();
        let owner = devices
            .iter()
            .find(|device| device.vm.gpu_id() == words[7])
            .ok_or_else(|| {
                error(
                    ErrorKind::InvalidArgument,
                    "IPC exporting GPU is unavailable in this session",
                )
            })?;
        LinuxAllocation::import_ipc(
            owner.vm.clone(),
            mapping_devices.iter().map(|device| device.vm.clone()),
            words,
            size,
        )
    }

    pub(crate) fn export_kfd_ipc_memory(
        allocation: &LinuxAllocation,
    ) -> Result<KfdIpcMemoryHandle, Error> {
        allocation.export_ipc_memory()
    }

    pub(crate) fn set_kfd_svm_attributes(
        &self,
        address: u64,
        size: u64,
        attributes: &[KfdSvmAttribute],
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if attributes.is_empty() {
            return Ok(());
        }
        let mut native = self.encode_svm_attributes(attributes)?;
        self.kfd()?
            .svm_attributes(address, size, uapi::SVM_OP_SET_ATTR, native.as_mut_slice())
            .map_err(|source| native_error("AMDKFD_IOC_SVM set attributes", source))
    }

    pub(crate) fn get_kfd_svm_attributes(
        &self,
        address: u64,
        size: u64,
        attributes: &mut [KfdSvmAttribute],
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if attributes.is_empty() {
            return Ok(());
        }
        let mut native = self.encode_svm_attributes(attributes)?;
        self.kfd()?
            .svm_attributes(address, size, uapi::SVM_OP_GET_ATTR, native.as_mut_slice())
            .map_err(|source| native_error("AMDKFD_IOC_SVM get attributes", source))?;
        self.decode_svm_attributes(attributes, native.as_slice())
    }

    pub(crate) fn retain_kfd_signal_event_page(
        allocation: &mut LinuxAllocation,
    ) -> Result<(), Error> {
        allocation.retain_signal_event_page()
    }

    /// Creates a KFD signal event while preserving an offered event page on
    /// ambiguous native failure.
    pub(crate) fn create_kfd_signal_event(
        &self,
        device: &DeviceState,
        event_page: Option<&LinuxAllocation>,
        page_offered: &mut bool,
    ) -> Result<Owned<KfdSignalEvent>, Error> {
        *page_offered = false;
        self.ensure_open()?;
        let event_page_handle = event_page
            .map(|page| page.signal_event_page_handle(&device.vm))
            .transpose()?;
        KfdSignalEvent::create(
            device.vm.kfd_owner(),
            event_page_handle,
            device.vm.allocator(),
            page_offered,
        )
    }

    pub(crate) fn destroy_kfd_signal_event(event: &mut KfdSignalEvent) -> Result<(), Error> {
        event.destroy()
    }
}
