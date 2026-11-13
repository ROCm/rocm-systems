// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GFX12 image formats, linear layouts, and texture resource descriptors.
//!
//! The encodings below follow `ROCr`'s gfx12 image resource definition;
//! the linear row alignment follows `AddrLib3`'s GFX12 surface calculation.

use crate::topology::GpuInfo;

/// Required base alignment for a GFX12 texture resource.
pub const BASE_ALIGNMENT: usize = 256;
/// Required byte alignment for a linear image row.
pub const ROW_ALIGNMENT: usize = 256;
// The buffer SRD has a 32-bit byte count and formats can occupy 16 bytes.
/// Largest element count representable by every supported buffer format.
pub const BUFFER_MAX_ELEMENTS: u32 = u32::MAX / 16;

/// Whether this GPU has a qualified GFX12 image descriptor path.
#[must_use]
pub fn supported_gpu(info: &GpuInfo) -> bool {
    info.target_profile.supports_image_sampler()
}

/// Encoded GFX12 image format and channel-selection fields.
#[derive(Clone, Copy)]
pub struct Format {
    /// Bytes per uncompressed format element.
    pub bytes: usize,
    /// Hardware resource descriptor format code.
    code: u32,
    /// Component selectors written into the resource descriptor.
    swizzle: [u32; 4],
    /// Buffer descriptor component selector.
    bc_swizzle: u32,
}

/// Checked linear image dimensions and byte strides.
#[derive(Clone, Copy)]
pub struct Layout {
    /// Width in format elements.
    pub width: usize,
    /// Height in format elements.
    pub height: usize,
    /// Depth or array-slice count.
    pub slices: usize,
    /// Bytes between adjacent rows.
    pub row_pitch: usize,
    /// Bytes between adjacent slices.
    pub slice_pitch: usize,
    /// Total byte extent of the image.
    pub size: usize,
}

/// Texture shape represented by a GFX12 resource descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Geometry {
    /// One-dimensional texture.
    Texture1D,
    /// Two-dimensional texture.
    Texture2D,
    /// Three-dimensional texture.
    Texture3D,
    /// Array of one-dimensional textures.
    Texture1DArray,
    /// Array of two-dimensional textures.
    Texture2DArray,
    /// Byte-addressed buffer image.
    Buffer,
}

/// Image extent after frontend-specific shape validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Description {
    /// Hardware texture shape.
    pub geometry: Geometry,
    /// Width in format elements.
    pub width: usize,
    /// Height in format elements, or one for 1D textures.
    pub height: usize,
    /// Depth or array extent, or one for non-array textures.
    pub slices: usize,
}

/// Descriptor fields taken from a completed native mip-chain layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MipChainLayout {
    /// Required base address alignment.
    pub alignment: u32,
    /// Hardware swizzle mode selected by the surface layout library.
    pub swizzle_mode: u32,
    /// Number of levels in the complete chain.
    pub levels: u32,
}

/// Component order visible to a shader.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelOrder {
    /// Alpha only.
    A,
    /// Red only.
    R,
    /// Red and green.
    Rg,
    /// Red and alpha.
    Ra,
    /// Red, green, blue, and alpha.
    Rgba,
    /// Blue, green, red, and alpha.
    Bgra,
    /// Alpha, red, green, and blue.
    Argb,
}

/// Numeric interpretation of each format component.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelType {
    /// Signed normalized 8-bit integer.
    Snorm8,
    /// Signed normalized 16-bit integer.
    Snorm16,
    /// Unsigned normalized 8-bit integer.
    Unorm8,
    /// Unsigned normalized 16-bit integer.
    Unorm16,
    /// Signed 8-bit integer.
    Sint8,
    /// Signed 16-bit integer.
    Sint16,
    /// Signed 32-bit integer.
    Sint32,
    /// Unsigned 8-bit integer.
    Uint8,
    /// Unsigned 16-bit integer.
    Uint16,
    /// Unsigned 32-bit integer.
    Uint32,
    /// IEEE binary16 floating point.
    Float16,
    /// IEEE binary32 floating point.
    Float32,
}

/// Failure class for an image layout or descriptor request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageError {
    /// The shape, format, or descriptor type cannot be encoded.
    Format,
    /// The extent exceeds a hardware field or supported dimension.
    Size,
    /// A pitch or base address violates hardware alignment.
    Pitch,
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    value
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
}

/// Returns the GFX12 combined format and component selectors for HIP formats.
/// Unsupported packed, depth, and sRGB formats stay absent from capability
/// queries until their access and conversion rules are implemented.
#[must_use]
pub fn format(order: ChannelOrder, channel_type: ChannelType) -> Option<Format> {
    let (channels, swizzle, bc_swizzle) = match order {
        ChannelOrder::A => (1, [0, 0, 0, 4], 3),    // A
        ChannelOrder::R => (1, [4, 0, 0, 1], 1),    // R
        ChannelOrder::Rg => (2, [4, 5, 0, 1], 0),   // RG
        ChannelOrder::Ra => (2, [4, 0, 0, 5], 1),   // RA
        ChannelOrder::Rgba => (4, [4, 5, 6, 7], 0), // RGBA
        ChannelOrder::Bgra => (4, [6, 5, 4, 7], 4), // BGRA
        ChannelOrder::Argb => (4, [5, 6, 7, 4], 3), // ARGB
    };
    if matches!(order, ChannelOrder::Bgra | ChannelOrder::Argb)
        && !matches!(
            channel_type,
            ChannelType::Snorm8 | ChannelType::Unorm8 | ChannelType::Sint8 | ChannelType::Uint8
        )
    {
        return None;
    }
    // These are FORMAT register values from resource_gfx12.h. The indexes
    // into ROCr's FormatLUT are different and cannot be written into an SRD.
    let (component_bytes, code) = match (channels, channel_type) {
        (1, ChannelType::Snorm8) => (1, 2),
        (1, ChannelType::Snorm16) => (2, 8),
        (1, ChannelType::Unorm8) => (1, 1),
        (1, ChannelType::Unorm16) => (2, 7),
        (1, ChannelType::Sint8) => (1, 6),
        (1, ChannelType::Sint16) => (2, 12),
        (1, ChannelType::Sint32) => (4, 21),
        (1, ChannelType::Uint8) => (1, 5),
        (1, ChannelType::Uint16) => (2, 11),
        (1, ChannelType::Uint32) => (4, 20),
        (1, ChannelType::Float16) => (2, 13),
        (1, ChannelType::Float32) => (4, 22),
        (2, ChannelType::Snorm8) => (1, 15),
        (2, ChannelType::Snorm16) => (2, 24),
        (2, ChannelType::Unorm8) => (1, 14),
        (2, ChannelType::Unorm16) => (2, 23),
        (2, ChannelType::Sint8) => (1, 19),
        (2, ChannelType::Sint16) => (2, 28),
        (2, ChannelType::Sint32) => (4, 49),
        (2, ChannelType::Uint8) => (1, 18),
        (2, ChannelType::Uint16) => (2, 27),
        (2, ChannelType::Uint32) => (4, 48),
        (2, ChannelType::Float16) => (2, 29),
        (2, ChannelType::Float32) => (4, 50),
        (4, ChannelType::Snorm8) => (1, 43),
        (4, ChannelType::Snorm16) => (2, 52),
        (4, ChannelType::Unorm8) => (1, 42),
        (4, ChannelType::Unorm16) => (2, 51),
        (4, ChannelType::Sint8) => (1, 47),
        (4, ChannelType::Sint16) => (2, 56),
        (4, ChannelType::Sint32) => (4, 62),
        (4, ChannelType::Uint8) => (1, 46),
        (4, ChannelType::Uint16) => (2, 55),
        (4, ChannelType::Uint32) => (4, 61),
        (4, ChannelType::Float16) => (2, 57),
        (4, ChannelType::Float32) => (4, 63),
        _ => return None,
    };
    Some(Format {
        bytes: channels * component_bytes,
        code,
        swizzle,
        bc_swizzle,
    })
}

fn dimensions(desc: Description) -> Result<(usize, usize, usize), ImageError> {
    let maxima = match desc.geometry {
        Geometry::Texture1D => (16_384, 1, 1),
        Geometry::Texture2D => (16_384, 16_384, 1),
        Geometry::Texture3D | Geometry::Texture2DArray => (16_384, 16_384, 8_192),
        Geometry::Texture1DArray => (16_384, 1, 8_192),
        Geometry::Buffer => (BUFFER_MAX_ELEMENTS as usize, 1, 1),
    };
    if desc.width == 0
        || desc.height == 0
        || desc.slices == 0
        || desc.width > maxima.0
        || desc.height > maxima.1
        || desc.slices > maxima.2
    {
        return Err(ImageError::Size);
    }
    Ok((desc.width, desc.height, desc.slices))
}

/// Computes checked linear pitches and allocation extent for a texture.
///
/// # Errors
/// Returns `Size` for unsupported extents and `Pitch` for invalid strides.
pub fn layout(
    desc: Description,
    format: Format,
    requested_row_pitch: usize,
    requested_slice_pitch: usize,
) -> Result<Layout, ImageError> {
    let (width, height, slices) = dimensions(desc)?;
    let minimum_row = width.checked_mul(format.bytes).ok_or(ImageError::Size)?;
    if desc.geometry == Geometry::Buffer {
        // A 1DB uses a buffer SRD. Its byte count is exact and the caller's
        // allocation need not include texture-row padding. Pitch arguments do
        // not describe a buffer image and are ignored, as in ROCr.
        u32::try_from(minimum_row).map_err(|_| ImageError::Size)?;
        return Ok(Layout {
            width,
            height: 1,
            slices: 1,
            row_pitch: minimum_row,
            slice_pitch: minimum_row,
            size: minimum_row,
        });
    }
    let aligned_row = align_up(minimum_row, ROW_ALIGNMENT).ok_or(ImageError::Size)?;
    let row_pitch = if requested_row_pitch == 0 {
        aligned_row
    } else {
        requested_row_pitch
    };
    if row_pitch < aligned_row
        || row_pitch % ROW_ALIGNMENT != 0
        || (matches!(
            desc.geometry,
            Geometry::Texture3D | Geometry::Texture1DArray | Geometry::Texture2DArray
        ) && row_pitch != aligned_row)
    {
        return Err(ImageError::Pitch);
    }
    let minimum_slice = row_pitch.checked_mul(height).ok_or(ImageError::Size)?;
    let slice_pitch = if requested_slice_pitch == 0 {
        minimum_slice
    } else {
        requested_slice_pitch
    };
    if (slices == 1 && requested_slice_pitch != 0)
        || slice_pitch < minimum_slice
        || slice_pitch % row_pitch != 0
        || (slices > 1 && slice_pitch != minimum_slice)
    {
        return Err(ImageError::Pitch);
    }
    let size = slice_pitch.checked_mul(slices).ok_or(ImageError::Size)?;
    Ok(Layout {
        width,
        height,
        slices,
        row_pitch,
        slice_pitch,
        size,
    })
}

/// Encodes a linear image or buffer resource descriptor.
///
/// # Errors
/// Returns `Size` for unrepresentable extents and `Pitch` for invalid bases.
#[allow(
    clippy::cast_possible_truncation,
    reason = "descriptor words intentionally hold low address bits"
)]
pub fn descriptor(
    desc: Description,
    format: Format,
    layout: Layout,
    data: u64,
) -> Result<[u32; 12], ImageError> {
    if data == 0 || data % BASE_ALIGNMENT as u64 != 0 || data >> 48 != 0 {
        return Err(ImageError::Pitch);
    }
    if desc.geometry == Geometry::Buffer {
        let size = u32::try_from(layout.size).map_err(|_| ImageError::Size)?;
        let mut words = [0_u32; 12];
        words[0] = data as u32;
        words[1] = ((data >> 32) as u32 & 0xffff) | (format.bytes as u32) << 16;
        words[2] = size;
        words[3] = format.swizzle[0]
            | format.swizzle[1] << 3
            | format.swizzle[2] << 6
            | format.swizzle[3] << 9
            | format.code << 12
            | ((format.bytes as u32) & 3) << 21;
        words[10] = u32::try_from(layout.width).map_err(|_| ImageError::Size)?;
        words[11] = 1;
        return Ok(words);
    }
    let width = u32::try_from(layout.width - 1).map_err(|_| ImageError::Size)?;
    let height = u32::try_from(layout.height - 1).map_err(|_| ImageError::Size)?;
    let pitch = u32::try_from(layout.row_pitch / format.bytes - 1).map_err(|_| ImageError::Size)?;
    if pitch > u32::from(u16::MAX) {
        return Err(ImageError::Size);
    }
    let type_code = match desc.geometry {
        Geometry::Texture1D => 8,
        Geometry::Texture2D => 9,
        Geometry::Texture3D => 10,
        Geometry::Texture1DArray => 12,
        Geometry::Texture2DArray => 13,
        Geometry::Buffer => return Err(ImageError::Format),
    };
    let mut words = [0_u32; 12];
    words[0] = (data >> 8) as u32;
    words[1] = ((data >> 40) as u32 & 0xff) | format.code << 17 | (width & 3) << 30;
    words[2] = ((width >> 2) & 0x3fff) | height << 14;
    words[3] = format.swizzle[0]
        | format.swizzle[1] << 3
        | format.swizzle[2] << 6
        | format.swizzle[3] << 9
        | format.bc_swizzle << 25
        | type_code << 28;
    words[4] = if matches!(
        desc.geometry,
        Geometry::Texture1DArray | Geometry::Texture2DArray
    ) {
        u32::try_from(layout.slices - 1).map_err(|_| ImageError::Size)?
    } else if desc.geometry == Geometry::Texture3D {
        u32::try_from(layout.slices - 1).map_err(|_| ImageError::Size)?
    } else {
        pitch
    };
    words[10] = u32::try_from(layout.width).map_err(|_| ImageError::Size)?;
    words[11] = 1;
    Ok(words)
}

/// Encodes an AddrLib3-laid-out native mip chain for the gfx12 texture unit.
/// The complete chain must be allocated at the base alignment reported by
/// `AddrLib3`; each level view changes only `BASE_LEVEL` and `LAST_LEVEL`.
///
/// # Errors
/// Returns `Size` for unsupported mip layout or extent, and `Format` for an
/// unsupported texture shape.
#[allow(
    clippy::cast_possible_truncation,
    reason = "descriptor words intentionally hold low address bits"
)]
pub fn native_mip_descriptor(
    desc: Description,
    format: Format,
    layout: MipChainLayout,
    data: u64,
) -> Result<[u32; 12], ImageError> {
    let levels = layout.levels;
    if !(2..=32).contains(&levels)
        || data == 0
        || data % u64::from(layout.alignment) != 0
        || data >> 48 != 0
        || layout.swizzle_mode == 0
        || layout.swizzle_mode > 7
    {
        return Err(ImageError::Size);
    }
    let (width, height, slices) = dimensions(desc)?;
    let width = u32::try_from(width - 1).map_err(|_| ImageError::Size)?;
    let height = u32::try_from(height - 1).map_err(|_| ImageError::Size)?;
    let type_code = match desc.geometry {
        Geometry::Texture1D => 8,
        Geometry::Texture2D => 9,
        Geometry::Texture3D => 10,
        Geometry::Texture1DArray => 12,
        Geometry::Texture2DArray => 13,
        Geometry::Buffer => return Err(ImageError::Format),
    };
    let mut words = [0u32; 12];
    words[0] = (data >> 8) as u32;
    words[1] =
        ((data >> 40) as u32 & 0xff) | (levels - 1) << 12 | format.code << 17 | (width & 3) << 30;
    words[2] = ((width >> 2) & 0x3fff) | height << 14;
    words[3] = format.swizzle[0]
        | format.swizzle[1] << 3
        | format.swizzle[2] << 6
        | format.swizzle[3] << 9
        | (levels - 1) << 15
        | layout.swizzle_mode << 20
        | format.bc_swizzle << 25
        | type_code << 28;
    words[4] = if matches!(
        desc.geometry,
        Geometry::Texture3D | Geometry::Texture1DArray | Geometry::Texture2DArray
    ) {
        u32::try_from(slices - 1).map_err(|_| ImageError::Size)?
    } else {
        0
    };
    words[10] = u32::try_from(desc.width).map_err(|_| ImageError::Size)?;
    words[11] = levels;
    Ok(words)
}

/// Rebase a driver-supplied gfx12 image SRD onto an imported allocation.
/// The surface and compression fields belong to the exporting driver. The
/// consumer controls only the GPU address, format selectors, and view extent.
///
/// # Errors
/// Returns a format, size, or pitch error for invalid source descriptors,
/// extents, or base addresses.
#[allow(
    clippy::cast_possible_truncation,
    reason = "descriptor words intentionally hold low address bits"
)]
pub fn imported_descriptor(
    desc: Description,
    format: Format,
    layout: Layout,
    data: u64,
    source: [u32; 8],
) -> Result<[u32; 12], ImageError> {
    if desc.geometry == Geometry::Buffer {
        return descriptor(desc, format, layout, data);
    }
    if data == 0 || data % BASE_ALIGNMENT as u64 != 0 || data >> 48 != 0 {
        return Err(ImageError::Pitch);
    }
    if !matches!(source[3] >> 28, 8..=15) {
        return Err(ImageError::Format);
    }
    let mut words = [0_u32; 12];
    words[..8].copy_from_slice(&source);
    words[0] = (data >> 8) as u32;
    words[1] = (words[1] & !0xff) | ((data >> 40) as u32 & 0xff);
    words[1] = (words[1] & !(0xff << 17)) | (format.code << 17);
    words[3] = (words[3] & !0xfff)
        | format.swizzle[0]
        | format.swizzle[1] << 3
        | format.swizzle[2] << 6
        | format.swizzle[3] << 9;
    if matches!(
        desc.geometry,
        Geometry::Texture1D | Geometry::Texture1DArray
    ) {
        let type_code = if desc.geometry == Geometry::Texture1D {
            8
        } else {
            12
        };
        words[3] = (words[3] & !(0xf << 28)) | type_code << 28;
    }
    if matches!(
        desc.geometry,
        Geometry::Texture2D | Geometry::Texture2DArray
    ) {
        let requested_width = u32::try_from(layout.width - 1).map_err(|_| ImageError::Size)?;
        let requested_height = u32::try_from(layout.height - 1).map_err(|_| ImageError::Size)?;
        let srd_width = ((words[2] & 0x3fff) << 2) | (words[1] >> 30);
        let srd_height = (words[2] >> 14) & 0xffff;
        if requested_width < srd_width || requested_height < srd_height {
            words[1] = (words[1] & !(3 << 30)) | ((requested_width & 3) << 30);
            words[2] = (words[2] & 0xc000_0000)
                | ((requested_height & 0xffff) << 14)
                | ((requested_width >> 2) & 0x3fff);
        }
    }
    words[10] = u32::try_from(layout.width).map_err(|_| ImageError::Size)?;
    words[11] = 1;
    Ok(words)
}

/// Selects a complete chain already laid out by the exporting driver. The
/// SRD's maximum mip count describes that layout; reducing `LAST_LEVEL` only
/// limits the view and does not change any tile or compression fields.
///
/// # Errors
/// Returns `Size` if the requested view does not fit the imported mip chain.
pub fn imported_mip_chain(
    mut words: [u32; 12],
    desc: Description,
    source: [u32; 8],
    levels: usize,
) -> Result<[u32; 12], ImageError> {
    let source_width = (((source[2] & 0x3fff) << 2) | (source[1] >> 30)) as usize + 1;
    let source_height = ((source[2] >> 14) & 0xffff) as usize + 1;
    let source_depth = (source[4] & 0x3fff) as usize + 1;
    if !(2..=32).contains(&levels)
        || desc.width != source_width
        || desc.height != source_height
        || (matches!(
            desc.geometry,
            Geometry::Texture3D | Geometry::Texture1DArray | Geometry::Texture2DArray
        ) && desc.slices != source_depth)
        || (words[1] >> 25) & 0x1f != 0
        || levels > (((words[1] >> 12) & 0x1f) + 1) as usize
        || levels > (((words[3] >> 15) & 0x1f) + 1) as usize
    {
        return Err(ImageError::Size);
    }
    let levels = u32::try_from(levels).map_err(|_| ImageError::Size)?;
    words[3] = (words[3] & !(0x1f << 15)) | ((levels - 1) << 15);
    words[11] = levels;
    Ok(words)
}

/// Makes one independent mip-level descriptor from a driver-laid-out chain.
/// The base dimensions and all tiling fields remain those of the parent SRD.
#[must_use]
pub fn mip_level_view(mut words: [u32; 12], level: u32, format: Format) -> [u32; 12] {
    words[1] = (words[1] & !(0x1f << 25)) | (level << 25);
    words[1] = (words[1] & !(0xff << 17)) | (format.code << 17);
    words[3] = (words[3] & !(0x1f << 15)) | (level << 15);
    words[3] = (words[3] & !0xfff)
        | format.swizzle[0]
        | format.swizzle[1] << 3
        | format.swizzle[2] << 6
        | format.swizzle[3] << 9;
    words
}
