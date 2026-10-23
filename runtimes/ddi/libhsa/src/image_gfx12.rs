// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Translation between HSA image descriptors and GFX12 hardware rules.

use crate::ffi::{
    HsaExtImageDataInfo, HsaExtImageDescriptor, HsaExtImageFormat, IMAGE_FORMAT_UNSUPPORTED,
    IMAGE_PITCH_UNSUPPORTED, IMAGE_SIZE_UNSUPPORTED, Status,
};
use rocaddr3::SurfaceLayout;
use rocddi::device::gpu::image::{
    self as hardware, ChannelOrder, ChannelType, Description, Geometry, ImageError, MipChainLayout,
};

pub(crate) use hardware::{BUFFER_MAX_ELEMENTS, Format, Layout, ROW_ALIGNMENT, supported_gpu};

/// Maps image error to the corresponding HSA representation.
fn map_image_error(error: ImageError) -> Status {
    match error {
        ImageError::Format => IMAGE_FORMAT_UNSUPPORTED,
        ImageError::Size => IMAGE_SIZE_UNSUPPORTED,
        ImageError::Pitch => IMAGE_PITCH_UNSUPPORTED,
    }
}

/// Converts an HSA image descriptor to a validated GFX12 surface description.
fn description(desc: HsaExtImageDescriptor) -> Result<Description, Status> {
    let (geometry, height, slices) = match desc.geometry {
        0 if desc.height == 0 && desc.depth == 0 && desc.array_size == 0 => {
            (Geometry::Texture1D, 1, 1)
        }
        1 if desc.height != 0 && desc.depth == 0 && desc.array_size == 0 => {
            (Geometry::Texture2D, desc.height, 1)
        }
        2 if desc.height != 0 && desc.depth != 0 && desc.array_size == 0 => {
            (Geometry::Texture3D, desc.height, desc.depth)
        }
        3 if desc.height == 0 && desc.depth == 0 && desc.array_size != 0 => {
            (Geometry::Texture1DArray, 1, desc.array_size)
        }
        4 if desc.height != 0 && desc.depth == 0 && desc.array_size != 0 => {
            (Geometry::Texture2DArray, desc.height, desc.array_size)
        }
        5 if desc.height == 0 && desc.depth == 0 && desc.array_size == 0 => {
            (Geometry::Buffer, 1, 1)
        }
        _ => return Err(IMAGE_SIZE_UNSUPPORTED),
    };
    Ok(Description {
        geometry,
        width: desc.width,
        height,
        slices,
    })
}

/// Checks whether the image geometry is supported by the GFX12 backend.
pub(crate) fn supported_geometry(geometry: u32) -> bool {
    matches!(geometry, 0..=5)
}

/// Maps an HSA channel format to a native image format.
pub(crate) fn format(format: HsaExtImageFormat) -> Option<Format> {
    let order = match format.channel_order {
        0 => ChannelOrder::A,
        1 => ChannelOrder::R,
        3 => ChannelOrder::Rg,
        5 => ChannelOrder::Ra,
        8 => ChannelOrder::Rgba,
        9 => ChannelOrder::Bgra,
        10 => ChannelOrder::Argb,
        _ => return None,
    };
    let channel_type = match format.channel_type {
        0 => ChannelType::Snorm8,
        1 => ChannelType::Snorm16,
        2 => ChannelType::Unorm8,
        3 => ChannelType::Unorm16,
        8 => ChannelType::Sint8,
        9 => ChannelType::Sint16,
        10 => ChannelType::Sint32,
        11 => ChannelType::Uint8,
        12 => ChannelType::Uint16,
        13 => ChannelType::Uint32,
        14 => ChannelType::Float16,
        15 => ChannelType::Float32,
        _ => return None,
    };
    hardware::format(order, channel_type)
}

/// Calculates the GFX12 surface layout for an HSA image descriptor.
pub(crate) fn layout(
    desc: HsaExtImageDescriptor,
    format: Format,
    row_pitch: usize,
    slice_pitch: usize,
) -> Result<Layout, Status> {
    hardware::layout(description(desc)?, format, row_pitch, slice_pitch).map_err(map_image_error)
}

/// Returns the required image allocation size and alignment.
pub(crate) fn info(layout: Layout) -> HsaExtImageDataInfo {
    HsaExtImageDataInfo {
        size: layout.size,
        alignment: hardware::BASE_ALIGNMENT,
    }
}

/// Adjusts GPU image descriptor words to the HSA metadata layout.
fn hsa_metadata(mut words: [u32; 12], desc: HsaExtImageDescriptor) -> [u32; 12] {
    words[8] = desc.format.channel_type;
    words[9] = desc.format.channel_order;
    words
}

/// Encodes a native image layout into GPU descriptor words.
pub(crate) fn descriptor(
    desc: HsaExtImageDescriptor,
    format: Format,
    layout: Layout,
    data: u64,
) -> Result<[u32; 12], Status> {
    hardware::descriptor(description(desc)?, format, layout, data)
        .map(|words| hsa_metadata(words, desc))
        .map_err(map_image_error)
}

/// Encodes the descriptor for one level of a mipmapped image.
pub(crate) fn native_mip_descriptor(
    desc: HsaExtImageDescriptor,
    format: Format,
    layout: &SurfaceLayout,
    data: u64,
) -> Result<[u32; 12], Status> {
    let levels = u32::try_from(layout.mips.len()).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
    hardware::native_mip_descriptor(
        description(desc)?,
        format,
        MipChainLayout {
            alignment: layout.alignment,
            swizzle_mode: layout.swizzle_mode,
            levels,
        },
        data,
    )
    .map(|words| hsa_metadata(words, desc))
    .map_err(map_image_error)
}

/// Builds a GPU descriptor for imported image storage.
pub(crate) fn imported_descriptor(
    desc: HsaExtImageDescriptor,
    format: Format,
    layout: Layout,
    data: u64,
    source: [u32; 8],
) -> Result<[u32; 12], Status> {
    hardware::imported_descriptor(description(desc)?, format, layout, data, source)
        .map(|words| hsa_metadata(words, desc))
        .map_err(map_image_error)
}

/// Builds descriptors for every level of an imported mip chain.
pub(crate) fn imported_mip_chain(
    words: [u32; 12],
    desc: HsaExtImageDescriptor,
    source: [u32; 8],
    levels: usize,
) -> Result<[u32; 12], Status> {
    hardware::imported_mip_chain(words, description(desc)?, source, levels).map_err(map_image_error)
}

/// Returns a view of one mip level from an imported image.
pub(crate) fn mip_level_view(
    words: [u32; 12],
    level: u32,
    desc: HsaExtImageDescriptor,
    format: Format,
) -> [u32; 12] {
    hsa_metadata(hardware::mip_level_view(words, level, format), desc)
}
