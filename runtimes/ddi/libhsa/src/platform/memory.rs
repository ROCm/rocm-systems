// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Qualified native memory codecs around shared HSA transactions.
//!
//! HSA's descriptor extensions carry Linux DMA-BUFs. IPC words and fabric
//! tokens retain their native peer protocol; matching widths do not establish
//! compatibility with another provider. No Windows transport is qualified yet.

use rocddi::device::Device;
use rocddi::memory::device::{Allocation, VirtualMemory};
#[cfg(target_os = "linux")]
use rocddi::memory::interop::linux;
use rocddi::session::Session;
use rocddi::topology::Endpoint;
use rocddi::{Error, ErrorKind};
#[cfg(target_os = "linux")]
use std::os::fd::{AsFd, IntoRawFd};

/// An independently retained descriptor used only by the native adapter.
pub(crate) struct Descriptor {
    #[cfg(target_os = "linux")]
    /// Retained Linux transport owner.
    native: std::os::fd::OwnedFd,
    #[cfg(not(target_os = "linux"))]
    #[allow(dead_code, reason = "no descriptor transport on this target")]
    /// No transport can construct this owner on the selected target.
    unavailable: std::convert::Infallible,
}

/// Typed export kept owned until common validation and public publication.
pub(crate) struct DmaBuf {
    #[cfg(target_os = "linux")]
    /// Retained Linux transport owner.
    native: linux::DmaBuf,
    #[cfg(not(target_os = "linux"))]
    #[allow(dead_code, reason = "no DMA-BUF transport on this target")]
    /// No transport can construct this owner on the selected target.
    unavailable: std::convert::Infallible,
}

/// Export geometry consumed by the public HSA transaction.
pub(crate) struct DmaBufInfo {
    /// Start of the allocation's logical range in the exported backing.
    pub source_offset: u64,
}

impl DmaBuf {
    /// Reports the logical offset while the descriptor stays owned.
    pub(crate) fn info(&self) -> DmaBufInfo {
        #[cfg(target_os = "linux")]
        {
            DmaBufInfo {
                source_offset: self.native.info().source_offset,
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            match self.unavailable {}
        }
    }

    /// Transfers ownership at the successful ABI publication point.
    pub(crate) fn into_descriptor(self) -> i32 {
        #[cfg(target_os = "linux")]
        {
            self.native.into_fd().into_raw_fd()
        }
        #[cfg(not(target_os = "linux"))]
        {
            match self.unavailable {}
        }
    }
}

/// Checks native identity before advertising the descriptor transport.
pub(crate) fn supports_dma_buf(endpoint: &Endpoint) -> bool {
    #[cfg(target_os = "linux")]
    {
        rocddi::topology::platform::linux::kfd_drm_info(endpoint).is_some()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = endpoint;
        false
    }
}

/// Duplicates a caller descriptor without consuming its ownership.
pub(crate) fn duplicate_descriptor(descriptor: i32) -> Result<Descriptor, Error> {
    #[cfg(target_os = "linux")]
    {
        linux::duplicate_descriptor(descriptor).map(|native| Descriptor { native })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = descriptor;
        Err(unsupported())
    }
}

/// Consumes the descriptor passed back to the HSA close extension.
///
/// # Safety
/// The caller relinquishes an exclusively owned descriptor even if close fails.
pub(crate) unsafe fn close_owned_descriptor(descriptor: i32) -> Result<(), Error> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: The public release operation transfers descriptor ownership.
        unsafe { linux::close_owned_descriptor(descriptor) }.map_err(|source| {
            Error::NativeOperation {
                kind: ErrorKind::Driver,
                operation: "HSA descriptor release",
                source,
            }
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = descriptor;
        Err(unsupported())
    }
}

/// Exports allocation backing while retaining descriptor ownership.
pub(crate) fn export_dma_buf(allocation: &Allocation) -> Result<DmaBuf, Error> {
    #[cfg(target_os = "linux")]
    {
        linux::export_dma_buf(allocation).map(|native| DmaBuf { native })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = allocation;
        Err(unsupported())
    }
}

/// Exports detached backing while retaining descriptor ownership.
pub(crate) fn export_virtual_memory(memory: &VirtualMemory) -> Result<DmaBuf, Error> {
    #[cfg(target_os = "linux")]
    {
        linux::export_virtual_memory(memory).map(|native| DmaBuf { native })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = memory;
        Err(unsupported())
    }
}

/// Imports backing through the retained descriptor.
pub(crate) fn import_virtual_memory(
    session: &Session,
    descriptor: &Descriptor,
) -> Result<VirtualMemory, Error> {
    #[cfg(target_os = "linux")]
    {
        linux::import_virtual_memory(session, descriptor.native.as_fd())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (session, descriptor);
        Err(unsupported())
    }
}

/// Imports graphics backing with the exact requested mapping devices.
pub(crate) fn import_graphics_dma_buf(
    session: &Session,
    devices: &[&Device],
    descriptor: &Descriptor,
    size: u64,
) -> Result<Allocation, Error> {
    #[cfg(target_os = "linux")]
    {
        linux::import_graphics_dma_buf(session, devices, descriptor.native.as_fd(), size)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (session, devices, descriptor, size);
        Err(unsupported())
    }
}

/// Encodes only the qualified KFD IPC peer protocol into HSA's opaque words.
pub(crate) fn export_ipc_memory(allocation: &Allocation) -> Result<[u32; 8], Error> {
    #[cfg(target_os = "linux")]
    {
        linux::export_kfd_ipc_memory(allocation).map(|handle| handle.words)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = allocation;
        Err(unsupported())
    }
}

/// Decodes the wire representation without exposing KFD types to HSA owners.
pub(crate) fn import_ipc_memory(
    session: &Session,
    devices: &[&Device],
    mappings: &[&Device],
    words: [u32; 8],
    size: u64,
) -> Result<Allocation, Error> {
    #[cfg(target_os = "linux")]
    {
        linux::import_kfd_ipc_memory(
            session,
            devices,
            mappings,
            linux::KfdIpcMemoryHandle { words },
            size,
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (session, devices, mappings, words, size);
        Err(unsupported())
    }
}

/// Queries the native fabric exporter without changing ownership.
pub(crate) fn fabric_virtual_memory_ready(memory: &VirtualMemory) -> Result<bool, Error> {
    #[cfg(target_os = "linux")]
    {
        linux::fabric_virtual_memory_ready(memory)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = memory;
        Err(unsupported())
    }
}

/// Encodes the qualified fabric peer protocol.
pub(crate) fn export_fabric_virtual_memory(memory: &VirtualMemory) -> Result<[u8; 16], Error> {
    #[cfg(target_os = "linux")]
    {
        linux::export_fabric_virtual_memory(memory)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = memory;
        Err(unsupported())
    }
}

/// Imports a fabric token through the selected device.
pub(crate) fn import_fabric_virtual_memory(
    device: &Device,
    token: [u8; 16],
) -> Result<VirtualMemory, Error> {
    #[cfg(target_os = "linux")]
    {
        linux::import_fabric_virtual_memory(device, token)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (device, token);
        Err(unsupported())
    }
}

#[cfg(not(target_os = "linux"))]
/// Reports an unavailable native transport.
fn unsupported() -> Error {
    Error::Operation {
        kind: ErrorKind::Unsupported,
        detail: "no provider implements this HSA native transport",
    }
}

/// The HSA VMM path is qualified for the installed KFD provider.
/// New providers must separately establish reservation, host/device projection,
/// and permissions support before this API family is advertised.
pub(crate) fn supports_virtual_memory(endpoint: &Endpoint) -> bool {
    #[cfg(target_os = "linux")]
    {
        rocddi::topology::platform::linux::kfd_drm_info(endpoint).is_some()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = endpoint;
        false
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    /// No descriptor ownership is manufactured on an unsupported target.
    fn descriptor_transports_reject_without_acquiring_native_owners() {
        for descriptor in [-1, 0, i32::MAX] {
            assert!(matches!(duplicate_descriptor(descriptor), Err(error)
                if error.kind() == ErrorKind::Unsupported));
            // SAFETY: No descriptor transport exists and the adapter does not
            // dereference or close these deliberately invalid native inputs.
            assert!(
                matches!(unsafe { close_owned_descriptor(descriptor) }, Err(error)
                if error.kind() == ErrorKind::Unsupported)
            );
        }
    }
}
