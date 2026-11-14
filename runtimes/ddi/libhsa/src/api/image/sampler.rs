// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Sampler descriptor entry points.

use crate::ffi::*;
use crate::image_abi::{create_sampler, destroy_sampler};
use crate::runtime::boundary;

#[cfg_attr(
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_sampler_create")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Creates a sampler from an HSA descriptor.
pub unsafe extern "C" fn hsa_ext_sampler_create(
    agent: HsaAgent,
    sampler_descriptor: *const HsaExtSamplerDescriptor,
    sampler: *mut HsaExtSampler,
) -> Status {
    boundary(|| {
        if agent.handle == 0 {
            return INVALID_AGENT;
        }
        if sampler_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The HSA caller supplies one initialized descriptor.
        let descriptor = unsafe { *sampler_descriptor };
        // SAFETY: create_sampler checks output before writing it.
        unsafe {
            create_sampler(
                agent,
                HsaExtSamplerDescriptorV3 {
                    coordinate_mode: descriptor.coordinate_mode,
                    filter_mode: descriptor.filter_mode,
                    mipmap_filter_mode: 2,
                    address_modes: [descriptor.address_mode; 3],
                    max_anisotropy: 0,
                },
                sampler,
            )
        }
    })
}

#[cfg_attr(
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_sampler_create_v2")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Creates a sampler from a version-two descriptor.
pub unsafe extern "C" fn hsa_ext_sampler_create_v2(
    agent: HsaAgent,
    sampler_descriptor: *const HsaExtSamplerDescriptorV2,
    sampler: *mut HsaExtSampler,
) -> Status {
    boundary(|| {
        if agent.handle == 0 {
            return INVALID_AGENT;
        }
        if sampler_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The HSA caller supplies one initialized descriptor.
        let descriptor = unsafe { *sampler_descriptor };
        // SAFETY: create_sampler checks output before writing it.
        unsafe {
            create_sampler(
                agent,
                HsaExtSamplerDescriptorV3 {
                    coordinate_mode: descriptor.coordinate_mode,
                    filter_mode: descriptor.filter_mode,
                    mipmap_filter_mode: descriptor.mipmap_filter_mode,
                    address_modes: descriptor.address_modes,
                    max_anisotropy: 0,
                },
                sampler,
            )
        }
    })
}

#[cfg_attr(
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_sampler_create_v3")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Creates a sampler with independently selected addressing and anisotropy.
pub unsafe extern "C" fn hsa_ext_sampler_create_v3(
    agent: HsaAgent,
    sampler_descriptor: *const HsaExtSamplerDescriptorV3,
    sampler: *mut HsaExtSampler,
) -> Status {
    boundary(|| {
        if agent.handle == 0 {
            return INVALID_AGENT;
        }
        if sampler_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplies one readable descriptor and writable
        // output storage; create_sampler validates the output pointer.
        unsafe { create_sampler(agent, *sampler_descriptor, sampler) }
    })
}

#[cfg_attr(
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_sampler_destroy")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
/// Releases a sampler descriptor handle.
pub extern "C" fn hsa_ext_sampler_destroy(agent: HsaAgent, sampler: HsaExtSampler) -> Status {
    destroy_sampler(agent, sampler)
}
