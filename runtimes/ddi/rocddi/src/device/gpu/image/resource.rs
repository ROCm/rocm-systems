// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GFX9, GFX10 and GFX11 register layouts from `ROCr`'s resource headers.

use super::{
    BASE_ALIGNMENT, Description, Format, Generation, Geometry, ImageError, Layout, MipChainLayout,
    ROW_ALIGNMENT, align_up, descriptor, imported_descriptor, imported_mip_chain, mip_level_view,
    native_mip_descriptor,
};

/// Converts a GFX11/12 combined code to GFX10 and GFX9 fields.
fn legacy_format(code: u32) -> Result<(u32, u32, u32), ImageError> {
    Ok(match code {
        1 => (1, 1, 0),
        2 => (2, 1, 1),
        5 => (5, 1, 4),
        6 => (6, 1, 5),
        7 => (7, 2, 0),
        8 => (8, 2, 1),
        11 => (11, 2, 4),
        12 => (12, 2, 5),
        13 => (13, 2, 7),
        14 => (14, 3, 0),
        15 => (15, 3, 1),
        18 => (18, 3, 4),
        19 => (19, 3, 5),
        20 => (20, 4, 4),
        21 => (21, 4, 5),
        22 => (22, 4, 7),
        23 => (23, 5, 0),
        24 => (24, 5, 1),
        27 => (27, 5, 4),
        28 => (28, 5, 5),
        29 => (29, 5, 7),
        36 => (50, 9, 0),
        42 => (56, 10, 0),
        43 => (57, 10, 1),
        46 => (60, 10, 4),
        47 => (61, 10, 5),
        48 => (62, 11, 4),
        49 => (63, 11, 5),
        50 => (64, 11, 7),
        51 => (65, 12, 0),
        52 => (66, 12, 1),
        55 => (69, 12, 4),
        56 => (70, 12, 5),
        57 => (71, 12, 7),
        61 => (75, 14, 4),
        62 => (76, 14, 5),
        63 => (77, 14, 7),
        66 => (130, 10, 9),
        68 => (133, 16, 0),
        69 => (134, 17, 0),
        _ => return Err(ImageError::Format),
    })
}

/// Places the texture's numeric format at this generation's register offset.
fn format_word(generation: Generation, format: Format) -> Result<(u32, u32), ImageError> {
    let (gfx10, data, numeric) = legacy_format(format.code)?;
    Ok(match generation {
        Generation::Gfx9 => ((data << 20) | (numeric << 26), 0x3ff << 20),
        Generation::Gfx10 | Generation::Gfx103 => (gfx10 << 20, 0x1ff << 20),
        Generation::Gfx11 => (format.code << 20, 0xff << 20),
        Generation::Gfx12 => (format.code << 17, 0xff << 17),
    })
}

/// Builds linear texture or buffer words for the specified register generation.
///
/// # Errors
/// Rejects incompatible addresses, dimensions, formats, or explicit pitches.
#[allow(
    clippy::cast_possible_truncation,
    reason = "validated resource fields use low address words"
)]
pub fn descriptor_for(
    generation: Generation,
    desc: Description,
    format: Format,
    layout: Layout,
    data: u64,
) -> Result<[u32; 12], ImageError> {
    if generation == Generation::Gfx12 {
        return descriptor(desc, format, layout, data);
    }
    // The shared validator covers the limits common to all four generations.
    let mut words = descriptor(desc, format, layout, data)?;
    let (gfx10, data_format, numeric) = legacy_format(format.code)?;
    if desc.geometry == Geometry::Buffer {
        words[3] &= !0x001f_f000;
        words[3] |= match generation {
            Generation::Gfx9 => (data_format << 15) | (numeric << 12),
            Generation::Gfx10 | Generation::Gfx103 => (gfx10 << 12) | (1 << 24),
            _ => format.code << 12,
        };
        return Ok(words);
    }
    let width = (layout.width - 1) as u32;
    let height = (layout.height - 1) as u32;
    let pitch = (layout.row_pitch / format.bytes - 1) as u32;
    words[1] = ((data >> 40) as u32 & 0xff) | format_word(generation, format)?.0;
    if generation == Generation::Gfx9 {
        words[2] = width | (height << 14);
        words[3] &= !(7 << 25);
        let depth = if matches!(
            desc.geometry,
            Geometry::Texture1DArray | Geometry::Texture2DArray | Geometry::Texture3D
        ) {
            (layout.slices - 1) as u32
        } else {
            0
        };
        words[4] = depth | (pitch << 13) | (format.bc_swizzle << 29);
    } else {
        words[1] |= (width & 3) << 30;
        words[2] = width >> 2 | height << 14;
        if matches!(generation, Generation::Gfx10 | Generation::Gfx103) {
            words[2] |= 1 << 31;
        }
        if generation == Generation::Gfx10
            && matches!(desc.geometry, Geometry::Texture1D | Geometry::Texture2D)
        {
            if layout.row_pitch
                != align_up(layout.width * format.bytes, ROW_ALIGNMENT).ok_or(ImageError::Size)?
            {
                return Err(ImageError::Pitch);
            }
            words[4] = 0;
        } else if pitch >= 16384
            && matches!(desc.geometry, Geometry::Texture1D | Geometry::Texture2D)
        {
            return Err(ImageError::Pitch);
        }
    }
    Ok(words)
}

/// Builds a native mip-chain SRD using the matching `AddrLib` generation.
///
/// # Errors
/// Rejects unrepresentable levels, swizzle modes, pitches or base alignment.
pub fn native_mip_descriptor_for(
    generation: Generation,
    desc: Description,
    format: Format,
    layout: MipChainLayout,
    pitch: u32,
    data: u64,
) -> Result<[u32; 12], ImageError> {
    if generation == Generation::Gfx12 {
        return native_mip_descriptor(desc, format, layout, data);
    }
    if !(2..=16).contains(&layout.levels)
        || layout.swizzle_mode >= 32
        || !layout.alignment.is_power_of_two()
        || data % u64::from(layout.alignment) != 0
        || pitch == 0
        || pitch > 65536
        || desc.geometry == Geometry::Buffer
    {
        return Err(ImageError::Size);
    }
    let surface = super::layout(desc, format, 0, 0)?;
    let mut words = descriptor_for(generation, desc, format, surface, data)?;
    words[3] |= (layout.levels - 1) << 16 | layout.swizzle_mode << 20;
    match generation {
        Generation::Gfx9 => {
            words[4] = (words[4] & !(0xffff << 13)) | ((pitch - 1) << 13);
            words[5] |= (layout.levels - 1) << 28;
        }
        Generation::Gfx10 | Generation::Gfx103 => {
            words[5] |= (layout.levels - 1) << 4;
            if matches!(desc.geometry, Geometry::Texture1D | Geometry::Texture2D) {
                words[4] = 0;
            }
        }
        Generation::Gfx11 => {
            words[1] |= (layout.levels - 1) << 16;
            if matches!(desc.geometry, Geometry::Texture1D | Geometry::Texture2D) {
                words[4] = 0;
            }
        }
        Generation::Gfx12 => {}
    }
    words[11] = layout.levels;
    Ok(words)
}

/// Returns the base mip field without changing producer storage geometry.
#[must_use]
pub fn base_mip_level(generation: Generation, words: &[u32; 12]) -> u32 {
    if generation == Generation::Gfx12 {
        (words[1] >> 25) & 31
    } else {
        (words[3] >> 12) & 15
    }
}

/// Makes one format-compatible mip view while preserving compression and tiling.
///
/// # Errors
/// Rejects formats or levels that this register generation cannot represent.
pub fn mip_level_view_for(
    generation: Generation,
    mut words: [u32; 12],
    level: u32,
    format: Format,
) -> Result<[u32; 12], ImageError> {
    if generation == Generation::Gfx12 {
        return (level < 32)
            .then(|| mip_level_view(words, level, format))
            .ok_or(ImageError::Size);
    }
    if level >= 16 {
        return Err(ImageError::Size);
    }
    let (encoded, mask) = format_word(generation, format)?;
    words[1] = (words[1] & !mask) | encoded;
    words[3] = (words[3] & !0xfffff)
        | level << 12
        | level << 16
        | format.swizzle[0]
        | format.swizzle[1] << 3
        | format.swizzle[2] << 6
        | format.swizzle[3] << 9;
    Ok(words)
}

/// Rebases producer SRDs without recalculating their surface or metadata layout.
///
/// # Errors
/// Rejects incompatible resource types, alignment or format fields.
#[allow(
    clippy::cast_possible_truncation,
    reason = "validated addresses use hardware low and high words"
)]
pub fn imported_descriptor_for(
    generation: Generation,
    desc: Description,
    format: Format,
    layout: Layout,
    data: u64,
    source: [u32; 8],
) -> Result<[u32; 12], ImageError> {
    if generation == Generation::Gfx12 {
        return imported_descriptor(desc, format, layout, data, source);
    }
    if desc.geometry == Geometry::Buffer {
        return descriptor_for(generation, desc, format, layout, data);
    }
    if data == 0 || data % BASE_ALIGNMENT as u64 != 0 || data >> 48 != 0 {
        return Err(ImageError::Pitch);
    }
    if !matches!(source[3] >> 28, 8..=15) {
        return Err(ImageError::Format);
    }
    if generation == Generation::Gfx9 {
        // The GFX9 importer requires a same-size view of the producer's format.
        let bytes = match (source[1] >> 20) & 63 {
            1 => 1,
            2 | 3 | 16 | 17 => 2,
            4..=10 => 4,
            11 | 12 => 8,
            14 => 16,
            _ => return Err(ImageError::Format),
        };
        if bytes != format.bytes {
            return Err(ImageError::Format);
        }
    }
    let mut words = [0; 12];
    words[..8].copy_from_slice(&source);
    words[0] = (data >> 8) as u32;
    words[1] = (words[1] & !0xff) | ((data >> 40) as u32 & 0xff);
    let (encoded, mask) = format_word(generation, format)?;
    words[1] = (words[1] & !mask) | encoded;
    words[3] = (words[3] & !0xfff)
        | format.swizzle[0]
        | format.swizzle[1] << 3
        | format.swizzle[2] << 6
        | format.swizzle[3] << 9;
    if matches!(
        desc.geometry,
        Geometry::Texture1D | Geometry::Texture1DArray
    ) {
        let resource_type = if desc.geometry == Geometry::Texture1D {
            8
        } else {
            12
        };
        words[3] = (words[3] & !(15 << 28)) | resource_type << 28;
    }
    if generation != Generation::Gfx9
        && matches!(
            desc.geometry,
            Geometry::Texture2D | Geometry::Texture2DArray
        )
    {
        // The producer owns storage geometry, but sampling uses the requested
        // view extent. GFX10/11 split WIDTH across two words and use 14-bit HEIGHT.
        let width = u32::try_from(layout.width - 1).map_err(|_| ImageError::Size)?;
        let height = u32::try_from(layout.height - 1).map_err(|_| ImageError::Size)?;
        let source_width = ((words[2] & 0xfff) << 2) | (words[1] >> 30);
        let source_height = (words[2] >> 14) & 0x3fff;
        if width < source_width || height < source_height {
            words[1] = (words[1] & !(3 << 30)) | ((width & 3) << 30);
            words[2] = (words[2] & !(0xfff | (0x3fff << 14)))
                | ((width >> 2) & 0xfff)
                | ((height & 0x3fff) << 14);
        }
    }
    // Legacy exported metadata encodes a byte offset from the image allocation,
    // unlike GFX12's inline compression information. Rebase both address halves.
    let metadata_offset = if generation == Generation::Gfx9 {
        (u64::from((source[5] >> 17) & 0xff) << 40) | (u64::from(source[7]) << 8)
    } else {
        (u64::from(source[7]) << 16) | (u64::from(source[6] >> 24) << 8)
    };
    let metadata = data
        .checked_add(metadata_offset)
        .filter(|address| address >> 48 == 0)
        .ok_or(ImageError::Size)?;
    if generation == Generation::Gfx9 {
        words[5] = (words[5] & !(0xff << 17)) | (((metadata >> 40) as u32) << 17);
        words[7] = (metadata >> 8) as u32;
    } else {
        words[6] = (words[6] & 0x00ff_ffff) | (((metadata >> 8) as u32 & 0xff) << 24);
        words[7] = (metadata >> 16) as u32;
    }
    words[10] = u32::try_from(layout.width).map_err(|_| ImageError::Size)?;
    words[11] = 1;
    Ok(words)
}

/// Restricts a producer mip chain's view without changing its maximum storage mip.
///
/// # Errors
/// Rejects chains outside the producer SRD's existing mip extent.
pub fn imported_mip_chain_for(
    generation: Generation,
    mut words: [u32; 12],
    desc: Description,
    source: [u32; 8],
    levels: usize,
) -> Result<[u32; 12], ImageError> {
    if generation == Generation::Gfx12 {
        return imported_mip_chain(words, desc, source, levels);
    }
    let maximum = match generation {
        Generation::Gfx9 => (source[5] >> 28) & 15,
        Generation::Gfx10 | Generation::Gfx103 => (source[5] >> 4) & 15,
        _ => (source[1] >> 16) & 15,
    };
    let base = base_mip_level(generation, &words);
    let levels = u32::try_from(levels).map_err(|_| ImageError::Size)?;
    let last = base
        .checked_add(levels)
        .and_then(|end| end.checked_sub(1))
        .ok_or(ImageError::Size)?;
    if levels < 2 || last > maximum || last > 15 {
        return Err(ImageError::Size);
    }
    words[3] = (words[3] & !(15 << 16)) | last << 16;
    words[11] = levels;
    Ok(words)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::{ChannelOrder, ChannelType, format, layout};
    use super::*;

    #[test]
    fn linear_srd_fields_follow_each_generation() {
        let desc = Description {
            geometry: Geometry::Texture2D,
            width: 37,
            height: 19,
            slices: 1,
        };
        let format = format(ChannelOrder::Rgba, ChannelType::Unorm8).unwrap();
        let layout = layout(desc, format, 0, 0).unwrap();
        let base = 0x1234_5600_u64;
        for (generation, code, pitch, resource_level) in [
            (Generation::Gfx10, 56, 0, 1 << 31),
            (Generation::Gfx103, 56, 63, 1 << 31),
            (Generation::Gfx11, 42, 63, 0),
        ] {
            let words = descriptor_for(generation, desc, format, layout, base).unwrap();
            assert_eq!(words[0], 0x0012_3456);
            assert_eq!(words[1], code << 20);
            assert_eq!(words[2], 9 | 18 << 14 | resource_level);
            assert_eq!(words[3] >> 28, 9);
            assert_eq!(words[4], pitch);
        }
        let words = descriptor_for(Generation::Gfx9, desc, format, layout, base).unwrap();
        assert_eq!(words[1], 10 << 20);
        assert_eq!(words[2], 0x24 | 18 << 14);
        assert_eq!(words[4], 63 << 13);
    }

    #[test]
    fn native_chain_and_raw_views_preserve_producer_metadata() {
        let desc = Description {
            geometry: Geometry::Texture2DArray,
            width: 64,
            height: 32,
            slices: 4,
        };
        let format = format(ChannelOrder::Rgba, ChannelType::Unorm8).unwrap();
        let raw = super::super::format(ChannelOrder::Rgba, ChannelType::Uint8).unwrap();
        for generation in [
            Generation::Gfx9,
            Generation::Gfx10,
            Generation::Gfx103,
            Generation::Gfx11,
        ] {
            let mut words = native_mip_descriptor_for(
                generation,
                desc,
                format,
                MipChainLayout {
                    alignment: 65536,
                    swizzle_mode: 20,
                    levels: 4,
                },
                64,
                0x10000,
            )
            .unwrap();
            assert_eq!((words[3] >> 16) & 15, 3);
            assert_eq!((words[3] >> 20) & 31, 20);
            words[6] = 0x1234_5678;
            words[7] = 0x9876_5432;
            let view = mip_level_view_for(generation, words, 2, raw).unwrap();
            assert_eq!(base_mip_level(generation, &view), 2);
            assert_eq!((view[3] >> 16) & 15, 2);
            assert_eq!((view[3] >> 20) & 31, 20);
            assert_eq!(&view[4..8], &words[4..8]);
            let mut source = [0; 8];
            source.copy_from_slice(&words[..8]);
            let imported = imported_mip_chain_for(generation, words, desc, source, 3).unwrap();
            assert_eq!((imported[3] >> 16) & 15, 2);
            assert!(imported_mip_chain_for(generation, words, desc, source, 5).is_err());
        }
    }

    #[test]
    fn imported_2d_views_apply_requested_dimensions_without_changing_storage() {
        let format = format(ChannelOrder::Rgba, ChannelType::Unorm8).unwrap();
        for generation in [Generation::Gfx10, Generation::Gfx103, Generation::Gfx11] {
            for geometry in [Geometry::Texture2D, Geometry::Texture2DArray] {
                let producer = Description {
                    geometry,
                    width: 67,
                    height: 35,
                    slices: if geometry == Geometry::Texture2DArray {
                        4
                    } else {
                        1
                    },
                };
                let words = native_mip_descriptor_for(
                    generation,
                    producer,
                    format,
                    MipChainLayout {
                        alignment: 65536,
                        swizzle_mode: 20,
                        levels: 4,
                    },
                    128,
                    0x10000,
                )
                .unwrap();
                let mut source: [u32; 8] = words[..8].try_into().unwrap();
                source[6] = 1 << 21;
                source[7] = 3;
                for (width, height) in [(32, 16), (32, 35), (67, 16), (67, 35)] {
                    let requested = Description {
                        width,
                        height,
                        ..producer
                    };
                    let imported = imported_descriptor_for(
                        generation,
                        requested,
                        format,
                        layout(requested, format, 0, 0).unwrap(),
                        0x10000,
                        source,
                    )
                    .unwrap();
                    let width = u32::try_from(width).unwrap();
                    let height = u32::try_from(height).unwrap();
                    assert_eq!(
                        (((imported[2] & 0xfff) << 2) | (imported[1] >> 30)) + 1,
                        width,
                    );
                    assert_eq!(((imported[2] >> 14) & 0x3fff) + 1, height);
                    assert_eq!(imported[1] & 0x3fff_ffff, source[1] & 0x3fff_ffff);
                    assert_eq!(imported[2] & 0xf000_3000, source[2] & 0xf000_3000);
                    assert_eq!(&imported[3..6], &source[3..6]);
                    assert_eq!(imported[6], source[6]);
                    assert_eq!(imported[7], 4);
                    assert_eq!(imported[10], width);
                }
            }
        }
    }

    #[test]
    fn gfx9_buffers_encode_distinct_numeric_and_data_fields() {
        let desc = Description {
            geometry: Geometry::Buffer,
            width: 16,
            height: 1,
            slices: 1,
        };
        let format = format(ChannelOrder::Rgba, ChannelType::Sint16).unwrap();
        let layout = layout(desc, format, 0, 0).unwrap();
        let words = descriptor_for(Generation::Gfx9, desc, format, layout, 0x10000).unwrap();
        assert_eq!((words[3] >> 12) & 7, 5);
        assert_eq!((words[3] >> 15) & 15, 12);
        assert_eq!(words[2], 16 * 8);
    }

    #[test]
    fn imported_compression_metadata_offsets_are_rebased() {
        let desc = Description {
            geometry: Geometry::Texture1D,
            width: 64,
            height: 1,
            slices: 1,
        };
        let format = format(ChannelOrder::Rgba, ChannelType::Unorm8).unwrap();
        let layout = layout(desc, format, 0, 0).unwrap();
        for generation in [Generation::Gfx9, Generation::Gfx10, Generation::Gfx11] {
            let mut source = [0; 8];
            source[1] = 10 << 20;
            source[3] = 9 << 28;
            source[6] = 1 << 21;
            if generation == Generation::Gfx9 {
                source[7] = 0x345;
            } else {
                source[6] |= 0x45 << 24;
                source[7] = 3;
            }
            let words =
                imported_descriptor_for(generation, desc, format, layout, 0x1200_0000, source)
                    .unwrap();
            assert_eq!(words[3] >> 28, 8);
            assert_eq!(words[6] & (1 << 21), 1 << 21);
            let address = if generation == Generation::Gfx9 {
                u64::from(words[7]) << 8 | u64::from((words[5] >> 17) & 255) << 40
            } else {
                u64::from(words[7]) << 16 | u64::from(words[6] >> 24) << 8
            };
            assert_eq!(address, 0x1203_4500);
        }
    }
}
