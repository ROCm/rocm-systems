// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Translation between HSA image contracts and GPU resource descriptors.

use crate::ffi::{
    HsaExtImageDataInfo, HsaExtImageDescriptor, HsaExtImageFormat, IMAGE_FORMAT_UNSUPPORTED,
    IMAGE_PITCH_UNSUPPORTED, IMAGE_SIZE_UNSUPPORTED, Status,
};
use rocaddr3::SurfaceLayout;
use rocddi::device::gpu::image::{
    self as hardware, ChannelOrder, ChannelType, Description, Geometry, ImageError, MipChainLayout,
};

pub(crate) use hardware::{
    BUFFER_MAX_ELEMENTS, Format, Generation, Layout, ROW_ALIGNMENT, supported_gpu,
};

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
        1 | 6 if desc.height != 0 && desc.depth == 0 && desc.array_size == 0 => {
            (Geometry::Texture2D, desc.height, 1)
        }
        2 if desc.height != 0 && desc.depth != 0 && desc.array_size == 0 => {
            (Geometry::Texture3D, desc.height, desc.depth)
        }
        3 if desc.height == 0 && desc.depth == 0 && desc.array_size != 0 => {
            (Geometry::Texture1DArray, 1, desc.array_size)
        }
        4 | 7 if desc.height != 0 && desc.depth == 0 && desc.array_size != 0 => {
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
    matches!(geometry, 0..=7)
}

/// Maps an HSA channel format to a native image format.
pub(crate) fn format(format: HsaExtImageFormat) -> Option<Format> {
    let order = match format.channel_order {
        0 => ChannelOrder::A,
        1 => ChannelOrder::R,
        3 => ChannelOrder::Rg,
        5 => ChannelOrder::Ra,
        6 => ChannelOrder::Rgb,
        8 => ChannelOrder::Rgba,
        9 => ChannelOrder::Bgra,
        10 => ChannelOrder::Argb,
        14 => ChannelOrder::Srgba,
        16 => ChannelOrder::Intensity,
        17 => ChannelOrder::Luminance,
        18 => ChannelOrder::Depth,
        _ => return None,
    };
    let channel_type = match format.channel_type {
        0 => ChannelType::Snorm8,
        1 => ChannelType::Snorm16,
        2 => ChannelType::Unorm8,
        3 => ChannelType::Unorm16,
        5 => ChannelType::Unorm555,
        6 => ChannelType::Unorm565,
        7 => ChannelType::Unorm101010,
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

/// Permitted kernel access modes for one format/geometry combination.
pub(crate) fn access_capabilities(geometry: u32, image_format: HsaExtImageFormat) -> u32 {
    if !supported_geometry(geometry)
        || format(image_format).is_none()
        || (matches!(geometry, 6 | 7) && image_format.channel_order != 18)
        || (geometry == 5
            && (image_format.channel_order == 14 || matches!(image_format.channel_type, 5 | 6)))
    {
        return 0;
    }
    match image_format.channel_order {
        14 => 1, // sRGBA supports read-only kernel accesses.
        18 => 3, // Depth supports separate read-only and write-only handles.
        _ => 7,
    }
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
    generation: Generation,
    desc: HsaExtImageDescriptor,
    format: Format,
    layout: Layout,
    data: u64,
) -> Result<[u32; 12], Status> {
    hardware::descriptor_for(generation, description(desc)?, format, layout, data)
        .map(|words| hsa_metadata(words, desc))
        .map_err(map_image_error)
}

/// Encodes the descriptor for one level of a mipmapped image.
pub(crate) fn native_mip_descriptor(
    generation: Generation,
    desc: HsaExtImageDescriptor,
    format: Format,
    layout: &SurfaceLayout,
    data: u64,
) -> Result<[u32; 12], Status> {
    let levels = u32::try_from(layout.mips.len()).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
    hardware::native_mip_descriptor_for(
        generation,
        description(desc)?,
        format,
        MipChainLayout {
            alignment: layout.alignment,
            swizzle_mode: layout.swizzle_mode,
            levels,
        },
        layout.pitch,
        data,
    )
    .map(|words| hsa_metadata(words, desc))
    .map_err(map_image_error)
}

/// Builds a GPU descriptor for imported image storage.
pub(crate) fn imported_descriptor(
    generation: Generation,
    desc: HsaExtImageDescriptor,
    format: Format,
    layout: Layout,
    data: u64,
    source: [u32; 8],
) -> Result<[u32; 12], Status> {
    hardware::imported_descriptor_for(generation, description(desc)?, format, layout, data, source)
        .map(|words| hsa_metadata(words, desc))
        .map_err(map_image_error)
}

/// Builds descriptors for every level of an imported mip chain.
pub(crate) fn imported_mip_chain(
    generation: Generation,
    words: [u32; 12],
    desc: HsaExtImageDescriptor,
    source: [u32; 8],
    levels: usize,
) -> Result<[u32; 12], Status> {
    hardware::imported_mip_chain_for(generation, words, description(desc)?, source, levels)
        .map_err(map_image_error)
}

/// Returns a view of one mip level from an imported image.
pub(crate) fn mip_level_view(
    generation: Generation,
    words: [u32; 12],
    level: u32,
    desc: HsaExtImageDescriptor,
    format: Format,
) -> Result<[u32; 12], Status> {
    hardware::mip_level_view_for(generation, words, level, format)
        .map(|words| hsa_metadata(words, desc))
        .map_err(map_image_error)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn packed_srgb_and_depth_capabilities_match_hardware_access() {
        for (order, channel_type, bytes, code, access) in [
            (6, 5, 2, 69, 7),
            (6, 6, 2, 68, 7),
            (6, 7, 4, 36, 7),
            (8, 7, 4, 36, 7),
            (14, 2, 4, 66, 1),
            (18, 3, 2, 7, 3),
        ] {
            let image_format = HsaExtImageFormat {
                channel_order: order,
                channel_type,
            };
            let native = format(image_format).unwrap();
            assert_eq!(native.bytes, bytes);
            assert_eq!(access_capabilities(1, image_format), access);
            let descriptor = HsaExtImageDescriptor {
                geometry: 1,
                width: 16,
                height: 8,
                depth: 0,
                array_size: 0,
                format: image_format,
            };
            let surface = layout(descriptor, native, 0, 0).unwrap();
            let words =
                self::descriptor(Generation::Gfx12, descriptor, native, surface, 0x10000).unwrap();
            assert_eq!((words[1] >> 17) & 0xff, code);
            if order == 14 || channel_type == 5 || channel_type == 6 {
                assert_eq!(access_capabilities(5, image_format), 0);
            }
        }
        let color = HsaExtImageFormat {
            channel_order: 8,
            channel_type: 2,
        };
        assert_eq!(access_capabilities(6, color), 0);
        for geometry in [6, 7] {
            let depth = HsaExtImageFormat {
                channel_order: 18,
                channel_type: 15,
            };
            assert_eq!(access_capabilities(geometry, depth), 3);
        }
        assert!(
            format(HsaExtImageFormat {
                channel_order: 18,
                channel_type: 4
            })
            .is_none()
        );
        assert!(
            format(HsaExtImageFormat {
                channel_order: 19,
                channel_type: 3
            })
            .is_none()
        );
    }
}
