// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Image capability and allocation queries.

use std::ffi::c_void;

use crate::ffi::*;
use crate::image_abi::{
    IMAGE_LAYOUT_LINEAR, IMAGE_LAYOUT_OPAQUE, get_info, image_capability, native_mip_data_info,
};
use crate::runtime::{boundary, lock};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_get_capability")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Reports image capabilities for a format and geometry.
pub unsafe extern "C" fn hsa_ext_image_get_capability(
    agent: HsaAgent,
    geometry: u32,
    image_format: *const HsaExtImageFormat,
    capability_mask: *mut u32,
) -> Status {
    boundary(|| {
        if image_format.is_null() || capability_mask.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        // SAFETY: The caller supplies one readable format and one writable mask.
        let format = unsafe { *image_format };
        let mask = match image_capability(runtime, agent, geometry, format) {
            Ok(mask) => mask,
            Err(status) => return status,
        };
        unsafe { capability_mask.write(mask) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_get_capability_with_layout")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Reports image capabilities for a specified layout.
pub unsafe extern "C" fn hsa_ext_image_get_capability_with_layout(
    agent: HsaAgent,
    geometry: u32,
    image_format: *const HsaExtImageFormat,
    image_data_layout: u32,
    capability_mask: *mut u32,
) -> Status {
    if image_data_layout != IMAGE_LAYOUT_LINEAR {
        return INVALID_ARGUMENT;
    }
    unsafe { hsa_ext_image_get_capability(agent, geometry, image_format, capability_mask) }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_data_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Reports allocation size and alignment for an image.
pub unsafe extern "C" fn hsa_ext_image_data_get_info(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptor,
    access_permission: u32,
    image_data_info: *mut HsaExtImageDataInfo,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        unsafe {
            get_info(
                agent,
                descriptor,
                access_permission,
                IMAGE_LAYOUT_OPAQUE,
                0,
                0,
                image_data_info,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_data_get_info_with_layout")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Reports allocation requirements for a chosen image layout.
pub unsafe extern "C" fn hsa_ext_image_data_get_info_with_layout(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptor,
    access_permission: u32,
    image_data_layout: u32,
    image_data_row_pitch: usize,
    image_data_slice_pitch: usize,
    image_data_info: *mut HsaExtImageDataInfo,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() || image_data_layout != IMAGE_LAYOUT_LINEAR {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        unsafe {
            get_info(
                agent,
                descriptor,
                access_permission,
                image_data_layout,
                image_data_row_pitch,
                image_data_slice_pitch,
                image_data_info,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_data_get_info_v2")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Reports allocation requirements for a mipmapped image.
pub unsafe extern "C" fn hsa_ext_image_data_get_info_v2(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptorV2,
    access_permission: u32,
    image_data_info: *mut HsaExtImageDataInfo,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        if descriptor.mipmap_levels > 1 {
            if image_data_info.is_null() {
                return INVALID_ARGUMENT;
            }
            let info = match native_mip_data_info(agent, descriptor, access_permission) {
                Ok(info) => info,
                Err(status) => return status,
            };
            // SAFETY: The caller supplied writable output storage for one
            // image data information record.
            unsafe { image_data_info.write(info) };
            return SUCCESS;
        }
        unsafe {
            get_info(
                agent,
                descriptor.into(),
                access_permission,
                IMAGE_LAYOUT_OPAQUE,
                0,
                0,
                image_data_info,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_image_get_info_max_dim")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Reports maximum dimensions supported for an image geometry.
pub unsafe extern "C" fn hsa_amd_image_get_info_max_dim(
    agent: HsaAgent,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    if !(EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS..=EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS)
        .contains(&attribute)
    {
        return INVALID_ARGUMENT;
    }
    // SAFETY: The HSA caller provides storage of the type associated with
    // this image limit; hsa_agent_get_info validates the agent and output.
    unsafe { crate::hsa_agent_get_info(agent, attribute, value) }
}
