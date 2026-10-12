// SPDX-License-Identifier: MIT

//! Linux KFD and descriptor translation for the HSA frontend.

pub(crate) use rocddi::gpu::event::linux as event;
pub(crate) use rocddi::memory::interop::linux as memory;
use rocddi::topology::Endpoint;
pub(crate) use rocddi::topology::platform::linux::host;

pub(crate) fn node_id(endpoint: Option<&Endpoint>) -> u32 {
    endpoint
        .and_then(Endpoint::linux_kfd_drm_info)
        .map_or(0, |info| info.node_id)
}

pub(crate) fn driver_uid(endpoint: Option<&Endpoint>) -> u32 {
    endpoint
        .and_then(Endpoint::linux_kfd_drm_info)
        .map_or(0, |info| info.gpu_id)
}

pub(crate) fn fault_matches_endpoint(endpoint: &Endpoint, fault: &event::GpuMemoryFault) -> bool {
    endpoint
        .linux_kfd_drm_info()
        .is_some_and(|info| info.gpu_id == fault.kfd_gpu_id)
}

pub(crate) mod fd {
    pub(crate) use std::os::fd::{AsFd, IntoRawFd};
}

/// Loader URI for a snapshotted code object supplied through a Linux descriptor.
pub(crate) fn code_object_file_uri(descriptor: i32, offset: usize, size: usize) -> Vec<u8> {
    format!("file:///proc/self/fd/{descriptor}#offset={offset}&size={size}").into_bytes()
}
