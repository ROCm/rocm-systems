// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Public image rejection and output-preservation tests.

use super::*;
use crate::ffi::*;

#[test]
fn sampler_v3_rejects_invalid_modes_without_output_changes() {
    let mut descriptor = HsaExtSamplerDescriptorV3 {
        coordinate_mode: 1,
        filter_mode: 0,
        mipmap_filter_mode: 2,
        address_modes: [1; 3],
        max_anisotropy: 16,
    };
    let mut output = HsaExtSampler { handle: 0xfeed };
    descriptor.address_modes[1] = 5;
    assert_eq!(
        unsafe {
            hsa_ext_sampler_create_v3(
                HsaAgent { handle: 1 },
                &raw const descriptor,
                &raw mut output,
            )
        },
        SAMPLER_DESCRIPTOR_UNSUPPORTED
    );
    assert_eq!(output.handle, 0xfeed);
}

#[test]
/// Checks rejected image requests leave caller output storage untouched.
fn unsupported_image_entry_points_leave_outputs_untouched() {
    let agent = HsaAgent { handle: 0 };
    let mut capability = 0xfeed_u32;
    let mut image = HsaExtImage { handle: 0xfeed };
    let mut sampler = HsaExtSampler { handle: 0xfeed };
    assert_eq!(
        unsafe { hsa_ext_image_get_capability(agent, 0, std::ptr::null(), &raw mut capability) },
        INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            hsa_ext_image_create(agent, std::ptr::null(), std::ptr::null(), 0, &raw mut image)
        },
        INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { hsa_ext_sampler_create(agent, std::ptr::null(), &raw mut sampler) },
        INVALID_AGENT
    );
    assert_eq!(hsa_ext_image_destroy(agent, image), NOT_INITIALIZED);
    assert_eq!(hsa_ext_sampler_destroy(agent, sampler), INVALID_AGENT);
    assert_eq!(capability, 0xfeed);
    assert_eq!(image.handle, 0xfeed);
    assert_eq!(sampler.handle, 0xfeed);
}
