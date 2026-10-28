// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Image transfer and clear entry points.

use std::ffi::c_void;

use crate::ffi::*;
use crate::image_abi::{hsa_ext_image_clear_impl, hsa_ext_image_copy_impl, transfer_image};
use crate::runtime::boundary;

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_copy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Copies a region between two images.
pub unsafe extern "C" fn hsa_ext_image_copy(
    agent: HsaAgent,
    src_image: HsaExtImage,
    src_offset: *const HsaDim3,
    dst_image: HsaExtImage,
    dst_offset: *const HsaDim3,
    range: *const HsaDim3,
) -> Status {
    // SAFETY: The image owner validates caller pointers and retains image
    // state through this synchronous operation.
    unsafe { hsa_ext_image_copy_impl(agent, src_image, src_offset, dst_image, dst_offset, range) }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_import")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Copies host image data into an image region.
pub unsafe extern "C" fn hsa_ext_image_import(
    agent: HsaAgent,
    src_memory: *const c_void,
    src_row_pitch: usize,
    src_slice_pitch: usize,
    dst_image: HsaExtImage,
    image_region: *const HsaExtImageRegion,
) -> Status {
    boundary(|| unsafe {
        transfer_image(
            agent,
            dst_image,
            src_memory.cast_mut(),
            src_row_pitch,
            src_slice_pitch,
            image_region,
            true,
        )
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_export")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Copies an image region into host storage.
pub unsafe extern "C" fn hsa_ext_image_export(
    agent: HsaAgent,
    src_image: HsaExtImage,
    dst_memory: *mut c_void,
    dst_row_pitch: usize,
    dst_slice_pitch: usize,
    image_region: *const HsaExtImageRegion,
) -> Status {
    boundary(|| unsafe {
        transfer_image(
            agent,
            src_image,
            dst_memory,
            dst_row_pitch,
            dst_slice_pitch,
            image_region,
            false,
        )
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_clear")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Fills an image region with an encoded clear color.
pub unsafe extern "C" fn hsa_ext_image_clear(
    agent: HsaAgent,
    image: HsaExtImage,
    data: *const c_void,
    image_region: *const HsaExtImageRegion,
) -> Status {
    // SAFETY: The image owner validates caller pointers and retains image
    // state through this synchronous operation.
    unsafe { hsa_ext_image_clear_impl(agent, image, data, image_region) }
}
