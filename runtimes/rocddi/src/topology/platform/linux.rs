//! Linux KFD and DRM identities associated with a discovered endpoint.

use crate::topology::Endpoint;

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

impl Endpoint {
    /// Returns the Linux KFD/DRM identity associated with this endpoint.
    ///
    /// Portable callers should not use this extension. It exists for Linux API
    /// frontends that must translate rocddi objects to an established KFD or
    /// DRM contract.
    #[must_use]
    pub fn linux_kfd_drm_info(&self) -> KfdDrmEndpointInfo {
        KfdDrmEndpointInfo {
            node_id: self.native.node,
            gpu_id: self.native.gpu_id,
            render_minor: self.native.render_minor,
        }
    }
}
