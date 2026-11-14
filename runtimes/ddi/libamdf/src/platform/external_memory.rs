// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! AMDF native transport codecs around shared memory construction transactions.
//!
//! Payload ownership remains typed until the complete public export is ready.
//! Unsupported transports never reinterpret descriptor bits as another OS handle.

use crate::generated::amdf::*;
use rocddi::device::Device;
use rocddi::memory::device::{Allocation, DeviceAccess};
use rocddi::session::Session;
use rocddi::topology::Endpoint;
use rocddi::{Error, ErrorKind};

#[cfg(target_os = "linux")]
use rocddi::memory::interop::linux;

/// Native export geometry used by the common AMDF transaction.
pub(crate) struct ExportInfo {
    pub byte_length: u64,
    pub physical_backing_id: [u64; 2],
}

/// Typed owner for a qualified external-memory transport.
pub(crate) enum Export {
    #[cfg(target_os = "linux")]
    DmaBuf(linux::DmaBuf),
    #[cfg(not(target_os = "linux"))]
    #[allow(
        dead_code,
        reason = "no native transport can construct an export on this target"
    )]
    Unavailable(std::convert::Infallible),
}

impl Export {
    pub(crate) fn info(&self) -> ExportInfo {
        match self {
            #[cfg(target_os = "linux")]
            Self::DmaBuf(owner) => ExportInfo {
                byte_length: owner.info().byte_length,
                physical_backing_id: owner.info().physical_backing_id,
            },
            #[cfg(not(target_os = "linux"))]
            Self::Unavailable(never) => match *never {},
        }
    }

    /// Transfers native ownership only after common export validation succeeds.
    pub(crate) fn into_abi(self) -> amdf_external_memory_t {
        match self {
            #[cfg(target_os = "linux")]
            Self::DmaBuf(owner) => {
                use std::os::fd::IntoRawFd;
                amdf_external_memory_t {
                    r#type: AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
                    payload: amdf_external_memory_payload_t {
                        file_descriptor: i64::from(owner.into_fd().into_raw_fd()),
                    },
                    release: Some(close_dma_buf),
                    ..Default::default()
                }
            }
            #[cfg(not(target_os = "linux"))]
            Self::Unavailable(never) => match never {},
        }
    }
}

#[cfg(target_os = "linux")]
unsafe extern "C" fn close_dma_buf(
    _user_data: *mut std::ffi::c_void,
    external_type: u32,
    payload: amdf_external_memory_payload_t,
) {
    if external_type != AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD {
        return;
    }
    // SAFETY: This callback is attached only to a transferred owned descriptor.
    if let Ok(descriptor) = i32::try_from(unsafe { payload.file_descriptor }) {
        let _ = unsafe { linux::close_owned_descriptor(descriptor) };
    }
}

pub(crate) fn supports_system_import(device: &Device) -> bool {
    #[cfg(target_os = "linux")]
    {
        linux::supports_system_dma_buf_import(device)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = device;
        false
    }
}

pub(crate) fn supports_device_export(endpoint: &Endpoint) -> bool {
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

/// Selects a concrete transport after the common placement rules are met.
pub(crate) fn support(page: u64, maximum: u64, import: bool) -> amdf_external_memory_support_t {
    // Currently called only for the qualified Linux KFD system-memory shape.
    amdf_external_memory_support_t {
        r#type: AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
        flags: AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT
            | if import {
                AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT
            } else {
                0
            }
            | AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET
            | AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS,
        source_offset_alignment: page,
        byte_length_alignment: 1,
        maximum_byte_length: maximum,
        ..Default::default()
    }
}

pub(crate) fn export(allocation: &Allocation) -> Result<Export, Error> {
    #[cfg(target_os = "linux")]
    {
        linux::export_dma_buf(allocation).map(Export::DmaBuf)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = allocation;
        Err(unsupported())
    }
}

/// Borrows the ABI payload; the common transaction consumes it only on success.
///
/// # Safety
/// The external payload and all requested devices remain valid during import.
pub(crate) unsafe fn import_system(
    session: &Session,
    devices: &[&Device],
    external: &amdf_external_memory_t,
    alignment: u64,
    permissions: DeviceAccess,
) -> Result<Allocation, Error> {
    #[cfg(target_os = "linux")]
    {
        if external.r#type != AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD {
            return Err(unsupported());
        }
        // SAFETY: The validated transport tag selects this payload member.
        let descriptor =
            i32::try_from(unsafe { external.payload.file_descriptor }).map_err(|_| {
                Error::Operation {
                    kind: ErrorKind::InvalidArgument,
                    detail: "invalid AMDF descriptor payload",
                }
            })?;
        linux::import_system_dma_buf(
            session,
            devices,
            descriptor,
            external.source_byte_offset,
            external.byte_length,
            alignment,
            permissions,
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (session, devices, external, alignment, permissions);
        Err(unsupported())
    }
}

fn unsupported() -> Error {
    Error::Operation {
        kind: ErrorKind::Unsupported,
        detail: "no provider implements this AMDF external transport",
    }
}
