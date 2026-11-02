// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux KFD and descriptor translation for the HSA frontend.

pub(crate) use rocddi::device::gpu::event::linux as event;
pub(crate) use rocddi::memory::interop::linux as memory;
use rocddi::topology::platform::linux::kfd_drm_info;
use rocddi::topology::{Endpoint, NativeNodeId, TopologyNode};

/// Returns the KFD node ID of an endpoint, or the host default.
pub(crate) fn node_id(endpoint: Option<&Endpoint>) -> u32 {
    endpoint
        .and_then(kfd_drm_info)
        .map_or(0, |info| info.node_id)
}

/// Returns the KFD ordinal used by the HSA CPU node query.
pub(crate) fn native_cpu_node_ordinal(node: &TopologyNode) -> Option<u32> {
    match node.native_id {
        NativeNodeId::LinuxKfd(id) => Some(id),
        _ => None,
    }
}

/// Returns the native driver UID of an endpoint, or the host default.
pub(crate) fn driver_uid(endpoint: Option<&Endpoint>) -> u32 {
    endpoint
        .and_then(kfd_drm_info)
        .map_or(0, |info| info.gpu_id)
}

/// File descriptor support for this frontend.
pub(crate) mod fd {
    pub(crate) use std::os::fd::{AsFd, IntoRawFd};
}

/// Loader URI for a snapshotted code object supplied through a Linux descriptor.
pub(crate) fn code_object_file_uri(descriptor: i32, offset: usize, size: usize) -> Option<Vec<u8>> {
    use std::os::unix::ffi::OsStrExt;

    /// Uppercase digits used to percent-encode descriptor path bytes.
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let path = memory::descriptor_path(descriptor).ok()?;
    let mut uri = b"file://".to_vec();
    for &byte in path.as_os_str().as_bytes() {
        if matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'_' | b'.' | b'~' | b'-')
        {
            uri.push(byte);
        } else {
            uri.extend_from_slice(&[
                b'%',
                HEX[usize::from(byte >> 4)],
                HEX[usize::from(byte & 15)],
            ]);
        }
    }
    uri.extend_from_slice(format!("#offset={offset}&size={size}").as_bytes());
    Some(uri)
}
