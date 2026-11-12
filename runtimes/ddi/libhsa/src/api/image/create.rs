// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Image creation, view, and destruction entry points.

use std::ffi::c_void;

use crate::ffi::*;
use crate::image_abi::{
    IMAGE_LAYOUT_LINEAR, IMAGE_LAYOUT_OPAQUE, ImageRequest, create_image, create_imported_image,
    create_native_mip_image, destroy_image, image_mipmap_array_get_level,
};
use crate::runtime::boundary;

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Creates an image using runtime-selected native layout.
pub unsafe extern "C" fn hsa_ext_image_create(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptor,
    image_data: *const c_void,
    access_permission: u32,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        unsafe {
            create_image(
                agent,
                ImageRequest::new(
                    descriptor,
                    image_data,
                    access_permission,
                    IMAGE_LAYOUT_OPAQUE,
                    0,
                    0,
                    image,
                ),
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_create_with_layout")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Creates an image using the requested native layout.
pub unsafe extern "C" fn hsa_ext_image_create_with_layout(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptor,
    image_data: *const c_void,
    access_permission: u32,
    image_data_layout: u32,
    image_data_row_pitch: usize,
    image_data_slice_pitch: usize,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() || image_data_layout != IMAGE_LAYOUT_LINEAR {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        unsafe {
            create_image(
                agent,
                ImageRequest::new(
                    descriptor,
                    image_data,
                    access_permission,
                    image_data_layout,
                    image_data_row_pitch,
                    image_data_slice_pitch,
                    image,
                ),
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_create_v2")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Creates a mipmapped image from a version-two descriptor.
pub unsafe extern "C" fn hsa_ext_image_create_v2(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptorV2,
    image_data: *const c_void,
    access_permission: u32,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        if descriptor.mipmap_levels > 1 {
            // SAFETY: The caller supplies image backing and one writable
            // output handle for this synchronous creation call.
            return unsafe {
                create_native_mip_image(agent, descriptor, image_data, access_permission, image)
            };
        }
        unsafe {
            create_image(
                agent,
                ImageRequest::new(
                    descriptor.into(),
                    image_data,
                    access_permission,
                    IMAGE_LAYOUT_OPAQUE,
                    0,
                    0,
                    image,
                ),
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_image_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Creates an AMD image using native descriptor data.
pub unsafe extern "C" fn hsa_amd_image_create(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptor,
    image_layout: *const HsaAmdImageDescriptor,
    image_data: *const c_void,
    access_permission: u32,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        if !image_layout.is_null() {
            // SAFETY: The C caller supplies a readable versioned layout,
            // image backing, and output storage for this synchronous call.
            return unsafe {
                create_imported_image(
                    agent,
                    descriptor,
                    1,
                    image_layout,
                    image_data,
                    access_permission,
                    image,
                )
            };
        }
        unsafe {
            create_image(
                agent,
                ImageRequest::new(
                    descriptor,
                    image_data,
                    access_permission,
                    IMAGE_LAYOUT_OPAQUE,
                    0,
                    0,
                    image,
                ),
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_image_create_v2")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
/// Creates a version-two AMD image using native descriptor data.
pub unsafe extern "C" fn hsa_amd_image_create_v2(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptorV2,
    image_layout: *const HsaAmdImageDescriptor,
    image_data: *const c_void,
    access_permission: u32,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        if descriptor.mipmap_levels > 1 && image_layout.is_null() {
            // SAFETY: The caller supplies image backing and one writable
            // output handle for this synchronous creation call.
            return unsafe {
                create_native_mip_image(agent, descriptor, image_data, access_permission, image)
            };
        }
        if !image_layout.is_null() {
            // SAFETY: The C caller supplies a readable versioned layout,
            // image backing, and output storage for this synchronous call.
            return unsafe {
                create_imported_image(
                    agent,
                    descriptor.into(),
                    descriptor.mipmap_levels.max(1),
                    image_layout,
                    image_data,
                    access_permission,
                    image,
                )
            };
        }
        unsafe {
            create_image(
                agent,
                ImageRequest::new(
                    descriptor.into(),
                    image_data,
                    access_permission,
                    IMAGE_LAYOUT_OPAQUE,
                    0,
                    0,
                    image,
                ),
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Releases an image handle and its native backing.
pub extern "C" fn hsa_ext_image_destroy(agent: HsaAgent, image: HsaExtImage) -> Status {
    boundary(|| destroy_image(agent, image))
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_destroy_v2")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Destroys a version-two image handle and its native backing.
pub extern "C" fn hsa_ext_image_destroy_v2(agent: HsaAgent, image: HsaExtImage) -> Status {
    boundary(|| destroy_image(agent, image))
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_mipmap_array_get_level")
)]
/// # Safety
/// The non-null pointers must address readable parent and optional descriptor
/// records and writable storage for one output image handle.
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_mipmap_array_get_level(
    agent: HsaAgent,
    mipmapped_array: *const HsaExtImage,
    mip_level: u32,
    image_descriptor: *const HsaExtImageDescriptorV2,
    level_image_out: *mut HsaExtImage,
) -> Status {
    // SAFETY: The owner validates the C pointers and retains the image state
    // through this synchronous view creation call.
    unsafe {
        image_mipmap_array_get_level(
            agent,
            mipmapped_array,
            mip_level,
            image_descriptor,
            level_image_out,
        )
    }
}
