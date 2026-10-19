// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux descriptor and memory interop for the AMDF frontend.

use crate::generated::amdf::{
    AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE, amdf_endpoint_native_identity_t,
    amdf_endpoint_native_identity_t__value, amdf_endpoint_native_identity_t__value__linux_device,
};
use rocddi::topology::Endpoint;
use rocddi::topology::platform::linux::kfd_drm_info;

pub(crate) use rocddi::memory::interop::linux as memory;
pub(crate) use std::os::fd::IntoRawFd;

pub(crate) fn render_supported(endpoint: &Endpoint) -> bool {
    kfd_drm_info(endpoint)
        .and_then(|info| info.render_minor)
        .is_some()
}

pub(crate) fn host_numa_node(endpoint: &Endpoint) -> u32 {
    let Some(minor) = kfd_drm_info(endpoint).and_then(|info| info.render_minor) else {
        return u32::MAX;
    };
    let Some(node) =
        std::fs::read_to_string(format!("/sys/class/drm/renderD{minor}/device/numa_node"))
            .ok()
            .and_then(|value| value.trim().parse::<i64>().ok())
            .and_then(|value| u32::try_from(value).ok())
    else {
        return u32::MAX;
    };
    let cpus = std::fs::read_to_string(format!("/sys/devices/system/node/node{node}/cpulist"));
    if cpus.is_ok_and(|value| {
        value
            .trim_start()
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_digit)
    }) {
        node
    } else {
        u32::MAX
    }
}

pub(crate) fn native_identity(endpoint: &Endpoint) -> amdf_endpoint_native_identity_t {
    kfd_drm_info(endpoint)
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
