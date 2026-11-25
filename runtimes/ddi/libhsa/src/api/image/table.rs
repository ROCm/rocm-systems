// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! The public HSA image extension table.

use std::ffi::c_void;

use super::*;
use crate::ffi::*;

/// Function table in the exact order of `hsa_ext_images_1_pfn_t`.
#[cfg(any(test, not(hsa_tool_dispatch)))]
#[repr(C)]
pub(crate) struct ImageExtensionTable {
    /// Queries format and geometry capabilities.
    get_capability:
        unsafe extern "C" fn(HsaAgent, u32, *const HsaExtImageFormat, *mut u32) -> Status,
    /// Reports required image allocation size and alignment.
    data_get_info: unsafe extern "C" fn(
        HsaAgent,
        *const HsaExtImageDescriptor,
        u32,
        *mut HsaExtImageDataInfo,
    ) -> Status,
    /// Creates an image using the runtime-selected layout.
    create: unsafe extern "C" fn(
        HsaAgent,
        *const HsaExtImageDescriptor,
        *const c_void,
        u32,
        *mut HsaExtImage,
    ) -> Status,
    /// Releases a published image handle.
    destroy: extern "C" fn(HsaAgent, HsaExtImage) -> Status,
    /// Copies a region between two images.
    copy: unsafe extern "C" fn(
        HsaAgent,
        HsaExtImage,
        *const HsaDim3,
        HsaExtImage,
        *const HsaDim3,
        *const HsaDim3,
    ) -> Status,
    /// Copies host data into an image region.
    import: unsafe extern "C" fn(
        HsaAgent,
        *const c_void,
        usize,
        usize,
        HsaExtImage,
        *const HsaExtImageRegion,
    ) -> Status,
    /// Copies an image region into host storage.
    export: unsafe extern "C" fn(
        HsaAgent,
        HsaExtImage,
        *mut c_void,
        usize,
        usize,
        *const HsaExtImageRegion,
    ) -> Status,
    /// Fills an image region with a clear color.
    clear: unsafe extern "C" fn(
        HsaAgent,
        HsaExtImage,
        *const c_void,
        *const HsaExtImageRegion,
    ) -> Status,
    /// Creates a sampler from the original descriptor layout.
    sampler_create: unsafe extern "C" fn(
        HsaAgent,
        *const HsaExtSamplerDescriptor,
        *mut HsaExtSampler,
    ) -> Status,
    /// Releases a sampler handle.
    sampler_destroy: extern "C" fn(HsaAgent, HsaExtSampler) -> Status,
    /// Queries format capabilities for an explicit layout.
    get_capability_with_layout:
        unsafe extern "C" fn(HsaAgent, u32, *const HsaExtImageFormat, u32, *mut u32) -> Status,
    /// Reports allocation requirements for an explicit layout.
    data_get_info_with_layout: unsafe extern "C" fn(
        HsaAgent,
        *const HsaExtImageDescriptor,
        u32,
        u32,
        usize,
        usize,
        *mut HsaExtImageDataInfo,
    ) -> Status,
    /// Creates an image using an explicit layout and pitches.
    create_with_layout: unsafe extern "C" fn(
        HsaAgent,
        *const HsaExtImageDescriptor,
        *const c_void,
        u32,
        u32,
        usize,
        usize,
        *mut HsaExtImage,
    ) -> Status,
    /// Creates a sampler from the extended descriptor layout.
    sampler_create_v2: unsafe extern "C" fn(
        HsaAgent,
        *const HsaExtSamplerDescriptorV2,
        *mut HsaExtSampler,
    ) -> Status,
    /// Creates a sampler with normalized anisotropy.
    sampler_create_v3: unsafe extern "C" fn(
        HsaAgent,
        *const HsaExtSamplerDescriptorV3,
        *mut HsaExtSampler,
    ) -> Status,
}

#[cfg(any(test, not(hsa_tool_dispatch)))]
const _: () =
    assert!(std::mem::size_of::<ImageExtensionTable>() == 15 * std::mem::size_of::<usize>());

#[cfg(any(test, not(hsa_tool_dispatch)))]
/// Builds the image extension table from this frontend’s entry points.
pub(crate) fn image_extension_table() -> ImageExtensionTable {
    ImageExtensionTable {
        get_capability: hsa_ext_image_get_capability,
        data_get_info: hsa_ext_image_data_get_info,
        create: hsa_ext_image_create,
        destroy: hsa_ext_image_destroy,
        copy: hsa_ext_image_copy,
        import: hsa_ext_image_import,
        export: hsa_ext_image_export,
        clear: hsa_ext_image_clear,
        sampler_create: hsa_ext_sampler_create,
        sampler_destroy: hsa_ext_sampler_destroy,
        get_capability_with_layout: hsa_ext_image_get_capability_with_layout,
        data_get_info_with_layout: hsa_ext_image_data_get_info_with_layout,
        create_with_layout: hsa_ext_image_create_with_layout,
        sampler_create_v2: hsa_ext_sampler_create_v2,
        sampler_create_v3: hsa_ext_sampler_create_v3,
    }
}
