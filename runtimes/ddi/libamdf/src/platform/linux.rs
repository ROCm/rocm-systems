// SPDX-License-Identifier: MIT

//! Linux descriptor and memory interop for the AMDF frontend.

use crate::generated::amdf::{
    AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE, amdf_endpoint_native_identity_t,
    amdf_endpoint_native_identity_t__value, amdf_endpoint_native_identity_t__value__linux_device,
};
use rocddi::topology::Endpoint;

pub(crate) use rocddi::memory::interop::linux as memory;
pub(crate) use std::os::fd::IntoRawFd;

pub(crate) fn render_supported(endpoint: &Endpoint) -> bool {
    endpoint
        .linux_kfd_drm_info()
        .and_then(|info| info.render_minor)
        .is_some()
}

pub(crate) fn native_identity(endpoint: &Endpoint) -> amdf_endpoint_native_identity_t {
    endpoint
        .linux_kfd_drm_info()
        .and_then(|info| info.render_minor)
        .map_or_else(amdf_endpoint_native_identity_t::default, |minor| {
            amdf_endpoint_native_identity_t {
                r#type: AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE,
                value: amdf_endpoint_native_identity_t__value {
                    linux_device: amdf_endpoint_native_identity_t__value__linux_device {
                        // Linux reserves character-device major 226 for DRM.
                        major: 226,
                        minor,
                    },
                },
            }
        })
}
