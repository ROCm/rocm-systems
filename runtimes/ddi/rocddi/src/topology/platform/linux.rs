// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux KFD and DRM identities associated with a discovered endpoint.

use crate::topology::Endpoint;
use crate::topology::host::attached_numa_node_for_render;
use std::path::Path;

/// Linux-native identities retained for KFD and DRM interoperability.
///
/// These values are not stable rocddi endpoint identities. KFD identifiers are
/// scoped to the running system, and a DRM render minor names a Linux device
/// node. Portable code should use [`Endpoint::id`] instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KfdDrmEndpointInfo {
    /// KFD topology node ordinal.
    pub node_id: u32,
    /// KFD's per-boot GPU identifier.
    pub gpu_id: u32,
    /// DRM render-node minor, when the endpoint has a render node.
    pub render_minor: Option<u32>,
}

/// Returns the Linux KFD/DRM identity associated with an endpoint.
///
/// Linux API frontends use this when translating rocddi objects to an
/// established KFD or DRM contract. Portable code uses [`Endpoint::id`].
#[must_use]
pub fn kfd_drm_info(endpoint: &Endpoint) -> Option<KfdDrmEndpointInfo> {
    match &endpoint.selector {
        crate::driver::EndpointSelector::LinuxKfd { native, .. } => Some(KfdDrmEndpointInfo {
            node_id: native.node,
            gpu_id: native.gpu_id,
            render_minor: native.render_minor,
        }),
        crate::driver::EndpointSelector::Opaque(_) => None,
    }
}

/// Returns the OS NUMA node at a DRM endpoint's attachment when it has an
/// online host CPU. This is a best-effort host locality hint, not a KFD node
/// identity, execution permission, or part of endpoint activation identity.
#[must_use]
pub fn host_numa_node(endpoint: &Endpoint) -> Option<u32> {
    let minor = kfd_drm_info(endpoint)?.render_minor?;
    attached_numa_node_for_render(
        Path::new("/sys/class/drm"),
        Path::new("/sys/devices/system/node"),
        Path::new("/sys/devices/system/cpu/online"),
        minor,
    )
}
