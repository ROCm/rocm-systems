// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Sampler descriptor entry points.

use crate::ffi::*;
use crate::image_abi::{create_sampler, destroy_sampler};
use crate::runtime::boundary;

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_sampler_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
                HsaExtSamplerDescriptorV2 {
                    coordinate_mode: descriptor.coordinate_mode,
                    filter_mode: descriptor.filter_mode,
                    mipmap_filter_mode: 2,
                    address_modes: [descriptor.address_mode; 3],
                },
                sampler,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_sampler_create_v2")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        unsafe { create_sampler(agent, descriptor, sampler) }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_sampler_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Releases a sampler descriptor handle.
pub extern "C" fn hsa_ext_sampler_destroy(agent: HsaAgent, sampler: HsaExtSampler) -> Status {
    destroy_sampler(agent, sampler)
}
