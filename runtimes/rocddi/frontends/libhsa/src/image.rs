//! HSA image and sampler compatibility over rocddi-owned memory.
//!
//! This module validates public image geometry and format combinations,
//! computes overflow-checked linear and mipmapped layouts, constructs the
//! qualified GFX12 shader-resource descriptors, and implements host-side
//! import, export, copy, and clear operations. Runtime maps own every image and
//! sampler handle; native allocations stay live until explicit destruction.
//!
//! Capability queries describe only combinations implemented here. They must
//! not infer support from a recognizable format alone, because geometry,
//! access mode, layout, and target-generation constraints are independent.

use std::ffi::c_void;

use rocddi::memory::{Allocation, DeviceAccess, MemoryKind};

use crate::ffi::*;
use crate::memory::resolve_memory_range;
use crate::runtime::{Runtime, boundary, initialized_mut, lock, map_error};

const IMAGE_ALLOCATION_BYTES: u64 = 4096;
const IMAGE_DESCRIPTOR_DWORDS: usize = 12;
const SAMPLER_ALLOCATION_BYTES: u64 = 4096;
const SAMPLER_DESCRIPTOR_DWORDS: usize = 8;

const SAMPLER_COORDINATE_MODE_UNNORMALIZED: u32 = 0;
const SAMPLER_COORDINATE_MODE_NORMALIZED: u32 = 1;
const SAMPLER_FILTER_MODE_NEAREST: u32 = 0;
const SAMPLER_FILTER_MODE_LINEAR: u32 = 1;
const SAMPLER_FILTER_MODE_NONE: u32 = 2;
const SAMPLER_ADDRESSING_MODE_UNDEFINED: u32 = 0;
const SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE: u32 = 1;
const SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER: u32 = 2;
const SAMPLER_ADDRESSING_MODE_REPEAT: u32 = 3;
const SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT: u32 = 4;
const IMAGE_CAPABILITY_RW: u32 =
    IMAGE_CAPABILITY_READ_ONLY | IMAGE_CAPABILITY_WRITE_ONLY | IMAGE_CAPABILITY_READ_WRITE;
const IMAGE_CAPABILITY_ROWO: u32 = IMAGE_CAPABILITY_READ_ONLY | IMAGE_CAPABILITY_WRITE_ONLY;
const IMAGE_ALIGNMENT: usize = 256;
const LINEAR_ROW_PITCH_ALIGNMENT: usize = 256;

/// Capability bits and byte size for one validated public image format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ImageProperty {
    capability: u32,
    element_size: usize,
}

/// Fully expanded storage geometry, including every mip-level byte offset.
#[derive(Clone, Debug, Eq, PartialEq)]
struct ImageLayout {
    size: usize,
    alignment: usize,
    row_pitch: usize,
    slice_pitch: usize,
    mip_offsets: Vec<usize>,
}

/// Runtime-owned image backing and the immutable facts needed by later calls.
///
/// The public image handle is the key of this object in `Runtime::images`.
/// Keeping the allocation here makes handle removal and native destruction one
/// ownership transaction.
pub(crate) struct Image {
    allocation: Allocation,
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptor,
    access_permission: u32,
    layout: ImageLayout,
    data_device_address: u64,
    data_host_address: Option<usize>,
    mipmap_levels: usize,
}

/// Validated image state staged before the public handle is published.
struct ImageCreateInfo {
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptor,
    access_permission: u32,
    layout: ImageLayout,
    data_device_address: u64,
    data_host_address: Option<usize>,
    mipmap_levels: usize,
}

/// Normalized inputs shared by the standard and AMD image-create entry points.
#[derive(Clone, Copy)]
struct AmdImageCreateRequest {
    descriptor: HsaExtImageDescriptor,
    layout: *const HsaAmdImageDescriptor,
    data: *const c_void,
    access_permission: u32,
    mipmap_levels: usize,
    allow_missing_layout: bool,
}

impl Image {
    fn free(&mut self) -> Result<(), rocddi::Error> {
        self.allocation.free()
    }
}

fn scalar_property(channel_type: u32) -> Option<usize> {
    match channel_type {
        0 | 2 | 8 | 11 => Some(1),
        1 | 3 | 9 | 12 | 14 => Some(2),
        10 | 13 | 15 => Some(4),
        _ => None,
    }
}

fn image_property(format: HsaExtImageFormat, geometry: u32) -> Option<ImageProperty> {
    if geometry > IMAGE_GEOMETRY_2DADEPTH || format.channel_order > 19 {
        return None;
    }
    let (capability, element_size) = match format.channel_order {
        0 | 1 => (IMAGE_CAPABILITY_RW, scalar_property(format.channel_type)?),
        3 | 5 => (
            IMAGE_CAPABILITY_RW,
            scalar_property(format.channel_type)?.checked_mul(2)?,
        ),
        6 => match format.channel_type {
            5 | 6 => (IMAGE_CAPABILITY_RW, 2),
            7 => (IMAGE_CAPABILITY_RW, 4),
            _ => return None,
        },
        8 => match format.channel_type {
            0 | 2 | 8 | 11 => (IMAGE_CAPABILITY_RW, 4),
            1 | 3 | 9 | 12 | 14 => (IMAGE_CAPABILITY_RW, 8),
            7 => (IMAGE_CAPABILITY_RW, 4),
            10 | 13 | 15 => (IMAGE_CAPABILITY_RW, 16),
            _ => return None,
        },
        9 | 10 => match format.channel_type {
            0 | 2 | 8 | 11 => (IMAGE_CAPABILITY_RW, 4),
            _ => return None,
        },
        14 => match format.channel_type {
            2 => (IMAGE_CAPABILITY_READ_ONLY, 4),
            _ => return None,
        },
        16 | 17 => match format.channel_type {
            0 | 2 => (IMAGE_CAPABILITY_RW, 1),
            1 | 3 | 14 => (IMAGE_CAPABILITY_RW, 2),
            15 => (IMAGE_CAPABILITY_RW, 4),
            _ => return None,
        },
        18 => match format.channel_type {
            3 => (IMAGE_CAPABILITY_ROWO, 2),
            15 => (IMAGE_CAPABILITY_ROWO, 4),
            _ => return None,
        },
        _ => return None,
    };
    if geometry == IMAGE_GEOMETRY_1DB
        && (matches!(format.channel_order, 12..=15) || matches!(format.channel_type, 5 | 6))
    {
        return None;
    }
    if matches!(geometry, IMAGE_GEOMETRY_2DDEPTH | IMAGE_GEOMETRY_2DADEPTH)
        && !matches!(format.channel_order, 18 | 19)
    {
        return None;
    }
    Some(ImageProperty {
        capability,
        element_size,
    })
}

fn max_dimensions(geometry: u32) -> Option<(usize, usize, usize, usize)> {
    match geometry {
        IMAGE_GEOMETRY_1D => Some((16_384, 1, 1, 1)),
        IMAGE_GEOMETRY_2D => Some((16_384, 16_384, 1, 1)),
        IMAGE_GEOMETRY_3D => Some((16_384, 16_384, 8192, 1)),
        IMAGE_GEOMETRY_1DA => Some((16_384, 1, 1, 8192)),
        IMAGE_GEOMETRY_2DA => Some((16_384, 16_384, 1, 8192)),
        IMAGE_GEOMETRY_1DB => Some((u32::MAX as usize, 1, 1, 1)),
        IMAGE_GEOMETRY_2DDEPTH => Some((16_384, 16_384, 1, 1)),
        IMAGE_GEOMETRY_2DADEPTH => Some((16_384, 16_384, 1, 8192)),
        _ => None,
    }
}

fn valid_dimensions(descriptor: HsaExtImageDescriptor) -> bool {
    let Some((max_width, max_height, max_depth, max_array_size)) =
        max_dimensions(descriptor.geometry)
    else {
        return false;
    };
    if descriptor.width == 0
        || descriptor.width > max_width
        || descriptor.height > max_height
        || descriptor.depth > max_depth
        || descriptor.array_size > max_array_size
    {
        return false;
    }
    match descriptor.geometry {
        IMAGE_GEOMETRY_1D | IMAGE_GEOMETRY_1DB => {
            descriptor.height == 0 && descriptor.depth == 0 && descriptor.array_size == 0
        }
        IMAGE_GEOMETRY_2D | IMAGE_GEOMETRY_2DDEPTH => {
            descriptor.height != 0 && descriptor.depth == 0 && descriptor.array_size == 0
        }
        IMAGE_GEOMETRY_3D => {
            descriptor.height != 0 && descriptor.depth != 0 && descriptor.array_size == 0
        }
        IMAGE_GEOMETRY_1DA => {
            descriptor.height == 0 && descriptor.depth == 0 && descriptor.array_size != 0
        }
        IMAGE_GEOMETRY_2DA | IMAGE_GEOMETRY_2DADEPTH => {
            descriptor.height != 0 && descriptor.depth == 0 && descriptor.array_size != 0
        }
        _ => false,
    }
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    value
        .checked_add(alignment.checked_sub(1)?)
        .map(|value| value & !(alignment - 1))
}

fn level_extent(descriptor: HsaExtImageDescriptor, level: usize) -> HsaExtImageDescriptor {
    let scaled = |value: usize| value.checked_shr(level as u32).unwrap_or(0).max(1);
    HsaExtImageDescriptor {
        width: scaled(descriptor.width),
        height: if descriptor.height == 0 {
            0
        } else {
            scaled(descriptor.height)
        },
        depth: if descriptor.depth == 0 {
            0
        } else {
            scaled(descriptor.depth)
        },
        ..descriptor
    }
}

fn maximum_mip_levels(descriptor: HsaExtImageDescriptor) -> usize {
    let largest = match descriptor.geometry {
        IMAGE_GEOMETRY_1D | IMAGE_GEOMETRY_1DA | IMAGE_GEOMETRY_1DB => descriptor.width,
        IMAGE_GEOMETRY_2D
        | IMAGE_GEOMETRY_2DA
        | IMAGE_GEOMETRY_2DDEPTH
        | IMAGE_GEOMETRY_2DADEPTH => descriptor.width.max(descriptor.height),
        IMAGE_GEOMETRY_3D => descriptor
            .width
            .max(descriptor.height)
            .max(descriptor.depth),
        _ => 0,
    };
    (usize::BITS - largest.leading_zeros()) as usize
}

fn level_layout(
    descriptor: HsaExtImageDescriptor,
    element_size: usize,
    explicit_pitch: Option<(usize, usize)>,
) -> Result<(usize, usize, usize), Status> {
    let minimum_row_pitch = descriptor
        .width
        .checked_mul(element_size)
        .ok_or(IMAGE_SIZE_UNSUPPORTED)?;
    let row_pitch = match explicit_pitch {
        Some((0, _)) => minimum_row_pitch,
        Some((row_pitch, _)) => {
            if row_pitch < minimum_row_pitch
                || row_pitch % element_size != 0
                || row_pitch % LINEAR_ROW_PITCH_ALIGNMENT != 0
            {
                return Err(IMAGE_PITCH_UNSUPPORTED);
            }
            row_pitch
        }
        None => {
            align_up(minimum_row_pitch, LINEAR_ROW_PITCH_ALIGNMENT).ok_or(IMAGE_SIZE_UNSUPPORTED)?
        }
    };
    let rows = descriptor.height.max(1);
    let minimum_slice_pitch = match descriptor.geometry {
        IMAGE_GEOMETRY_1DA => row_pitch,
        IMAGE_GEOMETRY_3D | IMAGE_GEOMETRY_2DA | IMAGE_GEOMETRY_2DADEPTH => {
            row_pitch.checked_mul(rows).ok_or(IMAGE_SIZE_UNSUPPORTED)?
        }
        _ => 0,
    };
    let slice_pitch = match explicit_pitch {
        Some((_, 0)) => minimum_slice_pitch,
        Some((_, slice_pitch)) => {
            if minimum_slice_pitch == 0
                || slice_pitch < minimum_slice_pitch
                || slice_pitch % row_pitch != 0
            {
                return Err(IMAGE_PITCH_UNSUPPORTED);
            }
            slice_pitch
        }
        None => minimum_slice_pitch,
    };
    let size = match descriptor.geometry {
        IMAGE_GEOMETRY_1D | IMAGE_GEOMETRY_1DB => row_pitch,
        IMAGE_GEOMETRY_2D | IMAGE_GEOMETRY_2DDEPTH => {
            row_pitch.checked_mul(rows).ok_or(IMAGE_SIZE_UNSUPPORTED)?
        }
        IMAGE_GEOMETRY_3D => slice_pitch
            .checked_mul(descriptor.depth)
            .ok_or(IMAGE_SIZE_UNSUPPORTED)?,
        IMAGE_GEOMETRY_1DA | IMAGE_GEOMETRY_2DA | IMAGE_GEOMETRY_2DADEPTH => slice_pitch
            .checked_mul(descriptor.array_size)
            .ok_or(IMAGE_SIZE_UNSUPPORTED)?,
        _ => return Err(INVALID_ARGUMENT),
    };
    Ok((row_pitch, slice_pitch, size))
}

fn image_layout(
    descriptor: HsaExtImageDescriptor,
    access_permission: u32,
    mipmap_levels: usize,
    explicit_pitch: Option<(usize, usize)>,
) -> Result<ImageLayout, Status> {
    if !matches!(
        access_permission,
        ACCESS_PERMISSION_RO | ACCESS_PERMISSION_WO | ACCESS_PERMISSION_RW
    ) {
        return Err(INVALID_ARGUMENT);
    }
    let property =
        image_property(descriptor.format, descriptor.geometry).ok_or(IMAGE_FORMAT_UNSUPPORTED)?;
    let required_capability = match access_permission {
        ACCESS_PERMISSION_RO => IMAGE_CAPABILITY_READ_ONLY,
        ACCESS_PERMISSION_WO => IMAGE_CAPABILITY_WRITE_ONLY,
        ACCESS_PERMISSION_RW => IMAGE_CAPABILITY_READ_WRITE,
        _ => return Err(INVALID_ARGUMENT),
    };
    if property.capability & required_capability == 0 {
        return Err(IMAGE_FORMAT_UNSUPPORTED);
    }
    if !valid_dimensions(descriptor) {
        return Err(IMAGE_SIZE_UNSUPPORTED);
    }
    let levels = mipmap_levels.max(1);
    if levels > maximum_mip_levels(descriptor) || (levels > 1 && explicit_pitch.is_some()) {
        return Err(INVALID_ARGUMENT);
    }
    let mut size = 0_usize;
    let mut row_pitch = 0;
    let mut slice_pitch = 0;
    let mut mip_offsets = Vec::new();
    mip_offsets
        .try_reserve_exact(levels)
        .map_err(|_| OUT_OF_RESOURCES)?;
    for level in 0..levels {
        size = align_up(size, IMAGE_ALIGNMENT).ok_or(IMAGE_SIZE_UNSUPPORTED)?;
        mip_offsets.push(size);
        let (level_row_pitch, level_slice_pitch, level_size) = level_layout(
            level_extent(descriptor, level),
            property.element_size,
            explicit_pitch,
        )?;
        if level == 0 {
            row_pitch = level_row_pitch;
            slice_pitch = level_slice_pitch;
        }
        size = size.checked_add(level_size).ok_or(IMAGE_SIZE_UNSUPPORTED)?;
    }
    size = align_up(size, IMAGE_ALIGNMENT).ok_or(IMAGE_SIZE_UNSUPPORTED)?;
    Ok(ImageLayout {
        size,
        alignment: IMAGE_ALIGNMENT,
        row_pitch,
        slice_pitch,
        mip_offsets,
    })
}

fn scalar_combined_format(channel_type: u32, components: usize) -> Option<u32> {
    match (components, channel_type) {
        (1, 0) => Some(2),
        (1, 1) => Some(8),
        (1, 2) => Some(1),
        (1, 3) => Some(7),
        (1, 8) => Some(6),
        (1, 9) => Some(12),
        (1, 10) => Some(21),
        (1, 11) => Some(5),
        (1, 12) => Some(11),
        (1, 13) => Some(20),
        (1, 14) => Some(13),
        (1, 15) => Some(22),
        (2, 0) => Some(15),
        (2, 1) => Some(24),
        (2, 2) => Some(14),
        (2, 3) => Some(23),
        (2, 8) => Some(19),
        (2, 9) => Some(28),
        (2, 10) => Some(49),
        (2, 11) => Some(18),
        (2, 12) => Some(27),
        (2, 13) => Some(48),
        (2, 14) => Some(29),
        (2, 15) => Some(50),
        (4, 0) => Some(43),
        (4, 1) => Some(52),
        (4, 2) => Some(42),
        (4, 3) => Some(51),
        (4, 8) => Some(47),
        (4, 9) => Some(56),
        (4, 10) => Some(62),
        (4, 11) => Some(46),
        (4, 12) => Some(55),
        (4, 13) => Some(61),
        (4, 14) => Some(57),
        (4, 15) => Some(63),
        _ => None,
    }
}

fn combined_format(format: HsaExtImageFormat) -> Option<u32> {
    match format.channel_order {
        0 | 1 | 16 | 17 => scalar_combined_format(format.channel_type, 1),
        3 | 5 => scalar_combined_format(format.channel_type, 2),
        6 => match format.channel_type {
            5 => Some(69),
            6 => Some(68),
            7 => Some(36),
            _ => None,
        },
        8 => match format.channel_type {
            7 => Some(36),
            _ => scalar_combined_format(format.channel_type, 4),
        },
        9 | 10 => scalar_combined_format(format.channel_type, 4),
        14 => (format.channel_type == 2).then_some(66),
        18 => match format.channel_type {
            3 => Some(7),
            15 => Some(22),
            _ => None,
        },
        _ => None,
    }
}

fn image_swizzle(channel_order: u32) -> Option<[u32; 4]> {
    match channel_order {
        0 => Some([0, 0, 0, 4]),
        1 => Some([4, 0, 0, 1]),
        3 => Some([4, 5, 0, 1]),
        5 => Some([4, 0, 0, 5]),
        6 => Some([6, 5, 4, 1]),
        8 => Some([4, 5, 6, 7]),
        9 => Some([6, 5, 4, 7]),
        10 => Some([5, 6, 7, 4]),
        14 => Some([4, 5, 6, 7]),
        16 => Some([4, 4, 4, 4]),
        17 => Some([4, 4, 4, 1]),
        18 => Some([4, 0, 0, 0]),
        _ => None,
    }
}

fn geometry_type(geometry: u32) -> Option<u32> {
    match geometry {
        IMAGE_GEOMETRY_1D => Some(8),
        IMAGE_GEOMETRY_2D | IMAGE_GEOMETRY_2DDEPTH => Some(9),
        IMAGE_GEOMETRY_3D => Some(10),
        IMAGE_GEOMETRY_1DA => Some(12),
        IMAGE_GEOMETRY_2DA | IMAGE_GEOMETRY_2DADEPTH => Some(13),
        IMAGE_GEOMETRY_1DB => Some(0),
        _ => None,
    }
}

fn bc_swizzle(swizzle: [u32; 4]) -> u32 {
    let [r, g, b, a] = swizzle;
    if a == 4 {
        if b == 5 {
            2
        } else if r == 4 && g == 4 && b == 4 {
            0
        } else {
            3
        }
    } else if r == 4 {
        u32::from(!(g == 5 || (g == 4 && b == 4 && a == 7)))
    } else if g == 4 {
        5
    } else if b == 4 {
        4
    } else {
        0
    }
}

fn gfx12_image_srd(
    descriptor: HsaExtImageDescriptor,
    layout: &ImageLayout,
    device_address: u64,
    mipmap_levels: usize,
) -> Result<[u32; IMAGE_DESCRIPTOR_DWORDS], Status> {
    let property =
        image_property(descriptor.format, descriptor.geometry).ok_or(IMAGE_FORMAT_UNSUPPORTED)?;
    let format = combined_format(descriptor.format).ok_or(IMAGE_FORMAT_UNSUPPORTED)?;
    let swizzle = image_swizzle(descriptor.format.channel_order).ok_or(IMAGE_FORMAT_UNSUPPORTED)?;
    let geometry_type = geometry_type(descriptor.geometry).ok_or(INVALID_ARGUMENT)?;
    let mut srd = [0_u32; IMAGE_DESCRIPTOR_DWORDS];

    if descriptor.geometry == IMAGE_GEOMETRY_1DB {
        srd[0] = device_address as u32;
        srd[1] = ((device_address >> 32) as u32 & 0xffff)
            | ((property.element_size as u32 & 0x3fff) << 16);
        srd[2] = u32::try_from(
            descriptor
                .width
                .checked_mul(property.element_size)
                .ok_or(IMAGE_SIZE_UNSUPPORTED)?,
        )
        .map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
        srd[3] = swizzle[0]
            | (swizzle[1] << 3)
            | (swizzle[2] << 6)
            | (swizzle[3] << 9)
            | (format << 12)
            | ((property.element_size as u32 & 0x3) << 21)
            | (geometry_type << 30);
        srd[8] = descriptor.format.channel_type;
        srd[9] = descriptor.format.channel_order;
        srd[10] = descriptor.width as u32;
        srd[11] = if mipmap_levels > 1 {
            mipmap_levels as u32
        } else {
            0
        };
        return Ok(srd);
    }

    if device_address % IMAGE_ALIGNMENT as u64 != 0 {
        return Err(INVALID_ARGUMENT);
    }
    let width = u32::try_from(
        descriptor
            .width
            .checked_sub(1)
            .ok_or(IMAGE_SIZE_UNSUPPORTED)?,
    )
    .map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
    let height = u32::try_from(descriptor.height.max(1) - 1).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
    let levels = u32::try_from(mipmap_levels.max(1) - 1).map_err(|_| INVALID_ARGUMENT)?;
    if levels > 0x1f {
        return Err(INVALID_ARGUMENT);
    }
    srd[0] = (device_address >> 8) as u32;
    srd[1] = ((device_address >> 40) as u32 & 0xff)
        | (levels << 12)
        | (format << 17)
        | ((width & 0x3) << 30);
    srd[2] = ((width >> 2) & 0x3fff) | ((height & 0xffff) << 14);
    srd[3] = swizzle[0]
        | (swizzle[1] << 3)
        | (swizzle[2] << 6)
        | (swizzle[3] << 9)
        | (levels << 15)
        | (bc_swizzle(swizzle) << 25)
        | (geometry_type << 28);
    if matches!(
        descriptor.geometry,
        IMAGE_GEOMETRY_1DA | IMAGE_GEOMETRY_2DA | IMAGE_GEOMETRY_2DADEPTH
    ) {
        srd[4] = u32::try_from(descriptor.array_size - 1).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
    } else if descriptor.geometry == IMAGE_GEOMETRY_3D {
        srd[4] = u32::try_from(descriptor.depth - 1).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
    } else {
        let pitch = layout
            .row_pitch
            .checked_div(property.element_size)
            .and_then(|pitch| pitch.checked_sub(1))
            .and_then(|pitch| u32::try_from(pitch).ok())
            .filter(|pitch| *pitch <= 0xffff)
            .ok_or(IMAGE_PITCH_UNSUPPORTED)?;
        srd[4] = (pitch & 0x3fff) | ((pitch >> 14) << 14);
    }
    srd[8] = descriptor.format.channel_type;
    srd[9] = descriptor.format.channel_order;
    srd[10] = descriptor.width as u32;
    srd[11] = if mipmap_levels > 1 {
        mipmap_levels as u32
    } else {
        0
    };
    Ok(srd)
}

fn gfx12_amd_image_srd(
    descriptor: HsaExtImageDescriptor,
    device_address: u64,
    mipmap_levels: usize,
    words: [u32; 8],
) -> Result<[u32; IMAGE_DESCRIPTOR_DWORDS], Status> {
    let property =
        image_property(descriptor.format, descriptor.geometry).ok_or(IMAGE_FORMAT_UNSUPPORTED)?;
    let format = combined_format(descriptor.format).ok_or(IMAGE_FORMAT_UNSUPPORTED)?;
    let swizzle = image_swizzle(descriptor.format.channel_order).ok_or(IMAGE_FORMAT_UNSUPPORTED)?;
    let mut srd = [0_u32; IMAGE_DESCRIPTOR_DWORDS];
    srd[..8].copy_from_slice(&words);
    if descriptor.geometry == IMAGE_GEOMETRY_1DB {
        srd[0] = device_address as u32;
        srd[1] = (srd[1] & 0xc000_0000)
            | ((device_address >> 32) as u32 & 0xffff)
            | ((property.element_size as u32 & 0x3fff) << 16);
        srd[3] &= !0x0063_ffff;
        srd[3] |= swizzle[0]
            | (swizzle[1] << 3)
            | (swizzle[2] << 6)
            | (swizzle[3] << 9)
            | (format << 12)
            | ((property.element_size as u32 & 0x3) << 21);
    } else {
        if device_address % IMAGE_ALIGNMENT as u64 != 0 {
            return Err(INVALID_ALLOCATION);
        }
        srd[0] = (device_address >> 8) as u32;
        srd[1] = (srd[1] & !0x01fe_00ff) | ((device_address >> 40) as u32 & 0xff) | (format << 17);
        srd[3] = (srd[3] & !0x0000_0fff)
            | swizzle[0]
            | (swizzle[1] << 3)
            | (swizzle[2] << 6)
            | (swizzle[3] << 9);
        if matches!(descriptor.geometry, IMAGE_GEOMETRY_1D | IMAGE_GEOMETRY_1DA) {
            let resource_type = geometry_type(descriptor.geometry).ok_or(INVALID_ARGUMENT)?;
            srd[3] = (srd[3] & 0x0fff_ffff) | (resource_type << 28);
        }
        if matches!(descriptor.geometry, IMAGE_GEOMETRY_2D | IMAGE_GEOMETRY_2DA) {
            let supplied_width = ((srd[2] & 0x3fff) << 2) | (srd[1] >> 30);
            let supplied_height = (srd[2] >> 14) & 0xffff;
            let width = u32::try_from(descriptor.width - 1).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
            let height =
                u32::try_from(descriptor.height.max(1) - 1).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
            if width < supplied_width || height < supplied_height {
                srd[1] = (srd[1] & 0x3fff_ffff) | ((width & 0x3) << 30);
                srd[2] =
                    (srd[2] & 0xc000_0000) | ((width >> 2) & 0x3fff) | ((height & 0xffff) << 14);
            }
        }
    }
    srd[8] = descriptor.format.channel_type;
    srd[9] = descriptor.format.channel_order;
    srd[10] = descriptor.width as u32;
    srd[11] = if mipmap_levels > 1 {
        mipmap_levels as u32
    } else {
        0
    };
    Ok(srd)
}

fn store_image(
    runtime: &mut Runtime,
    gpu_index: usize,
    create: ImageCreateInfo,
    output: *mut HsaExtImage,
) -> Status {
    let srd = match gfx12_image_srd(
        create.descriptor,
        &create.layout,
        create.data_device_address,
        create.mipmap_levels,
    ) {
        Ok(srd) => srd,
        Err(status) => return status,
    };
    store_image_srd(runtime, gpu_index, create, srd, output)
}

fn store_image_srd(
    runtime: &mut Runtime,
    gpu_index: usize,
    create: ImageCreateInfo,
    srd: [u32; IMAGE_DESCRIPTOR_DWORDS],
    output: *mut HsaExtImage,
) -> Status {
    if runtime.images.try_reserve(1).is_err() {
        return OUT_OF_RESOURCES;
    }
    let allocation = match runtime.gpus[gpu_index].device.allocate(
        MemoryKind::System,
        IMAGE_ALLOCATION_BYTES,
        IMAGE_ALLOCATION_BYTES,
        DeviceAccess::READ | DeviceAccess::WRITE,
    ) {
        Ok(allocation) => allocation,
        Err(error) => return map_error(error),
    };
    let info = allocation.info();
    let Some(host_address) = info.host_address else {
        return OUT_OF_RESOURCES;
    };
    if info.device_address == 0 || runtime.images.contains_key(&info.device_address) {
        return OUT_OF_RESOURCES;
    }
    // SAFETY: The dedicated GPU-visible allocation has room for the complete
    // image object and remains owned by the image map for the handle lifetime.
    unsafe {
        std::ptr::copy_nonoverlapping(
            srd.as_ptr().cast::<u8>(),
            (host_address as *mut c_void).cast::<u8>(),
            size_of_val(&srd),
        );
        output.write(HsaExtImage {
            handle: info.device_address,
        });
    }
    runtime.images.insert(
        info.device_address,
        Image {
            allocation,
            agent: create.agent,
            descriptor: create.descriptor,
            access_permission: create.access_permission,
            layout: create.layout,
            data_device_address: create.data_device_address,
            data_host_address: create.data_host_address,
            mipmap_levels: create.mipmap_levels,
        },
    );
    SUCCESS
}

fn create_image(
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptor,
    image_data: *const c_void,
    access_permission: u32,
    mipmap_levels: usize,
    explicit_pitch: Option<(usize, usize)>,
    output: *mut HsaExtImage,
) -> Status {
    // SAFETY: All public callers reject a null output pointer.
    unsafe { output.write(HsaExtImage { handle: 0 }) };
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let runtime = match initialized_mut(&mut guard) {
        Ok(runtime) => runtime,
        Err(status) => return status,
    };
    let Some(gpu_index) = runtime.gpu_index(agent) else {
        return INVALID_AGENT;
    };
    let layout = match image_layout(descriptor, access_permission, mipmap_levels, explicit_pitch) {
        Ok(layout) => layout,
        Err(status) => return status,
    };
    if (image_data as usize) % layout.alignment != 0 {
        return INVALID_ARGUMENT;
    }
    let backing = match resolve_memory_range(runtime, agent, image_data as usize, layout.size) {
        Ok(backing) => backing,
        Err(INVALID_ALLOCATION) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    store_image(
        runtime,
        gpu_index,
        ImageCreateInfo {
            agent,
            descriptor,
            access_permission,
            layout,
            data_device_address: backing.device_address,
            data_host_address: backing.host_address,
            mipmap_levels: mipmap_levels.max(1),
        },
        output,
    )
}

unsafe fn create_amd_image(
    agent: HsaAgent,
    request: AmdImageCreateRequest,
    output: *mut HsaExtImage,
) -> Status {
    // SAFETY: All public callers reject a null output pointer.
    unsafe { output.write(HsaExtImage { handle: 0 }) };
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let runtime = match initialized_mut(&mut guard) {
        Ok(runtime) => runtime,
        Err(status) => return status,
    };
    let Some(gpu_index) = runtime.gpu_index(agent) else {
        return INVALID_AGENT;
    };
    let layout = match image_layout(
        request.descriptor,
        request.access_permission,
        request.mipmap_levels,
        None,
    ) {
        Ok(layout) => layout,
        Err(status) => return status,
    };
    if (request.data as usize) % IMAGE_ALIGNMENT != 0 {
        return INVALID_ALLOCATION;
    }
    let backing = match resolve_memory_range(runtime, agent, request.data as usize, layout.size) {
        Ok(backing) => backing,
        Err(status) => return status,
    };
    let srd = if request.layout.is_null() {
        if !request.allow_missing_layout {
            return INVALID_ARGUMENT;
        }
        match gfx12_image_srd(
            request.descriptor,
            &layout,
            backing.device_address,
            request.mipmap_levels.max(1),
        ) {
            Ok(srd) => srd,
            Err(status) => return status,
        }
    } else {
        // SAFETY: The AMD interop ABI requires a descriptor header followed
        // by at least eight driver SRD words for version one.
        let metadata = unsafe { &*request.layout };
        if metadata.version != 1
            || metadata.device_id
                != 0x1002_0000
                    | (runtime.gpus[gpu_index]
                        .endpoint
                        .pci
                        .map_or(0, |pci| pci.device_id)
                        & 0xffff)
        {
            return IMAGE_FORMAT_UNSUPPORTED;
        }
        let mut words = [0_u32; 8];
        // SAFETY: Version one defines eight consecutive words beginning at data.
        unsafe {
            std::ptr::copy_nonoverlapping(metadata.data.as_ptr(), words.as_mut_ptr(), words.len())
        };
        match gfx12_amd_image_srd(
            request.descriptor,
            backing.device_address,
            request.mipmap_levels.max(1),
            words,
        ) {
            Ok(srd) => srd,
            Err(status) => return status,
        }
    };
    let swizzle_mode = ((srd[3] >> 20) as u8) & 0x1f;
    let linear_backing = request.layout.is_null()
        || request.descriptor.geometry == IMAGE_GEOMETRY_1DB
        || swizzle_mode == 0;
    store_image_srd(
        runtime,
        gpu_index,
        ImageCreateInfo {
            agent,
            descriptor: request.descriptor,
            access_permission: request.access_permission,
            layout,
            data_device_address: backing.device_address,
            data_host_address: linear_backing.then_some(backing.host_address).flatten(),
            mipmap_levels: request.mipmap_levels.max(1),
        },
        srd,
        output,
    )
}

/// Borrowed, host-visible image geometry used by software copy operations.
///
/// Construction fails for tiled or device-only backing, so row walkers never
/// manufacture a CPU pointer for storage that lacks a valid host mapping.
#[derive(Clone, Copy)]
struct ImageView {
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptor,
    access_permission: u32,
    host_address: usize,
    row_pitch: usize,
    slice_pitch: usize,
    element_size: usize,
}

fn image_view(runtime: &Runtime, agent: HsaAgent, image: HsaExtImage) -> Result<ImageView, Status> {
    let image = runtime.images.get(&image.handle).ok_or(INVALID_ARGUMENT)?;
    if image.agent != agent {
        return Err(INVALID_AGENT);
    }
    let property = image_property(image.descriptor.format, image.descriptor.geometry)
        .ok_or(IMAGE_FORMAT_UNSUPPORTED)?;
    Ok(ImageView {
        agent: image.agent,
        descriptor: image.descriptor,
        access_permission: image.access_permission,
        host_address: image.data_host_address.ok_or(OUT_OF_RESOURCES)?,
        row_pitch: image.layout.row_pitch,
        slice_pitch: image.layout.slice_pitch,
        element_size: property.element_size,
    })
}

fn image_extent(descriptor: HsaExtImageDescriptor) -> HsaDim3 {
    match descriptor.geometry {
        IMAGE_GEOMETRY_1D | IMAGE_GEOMETRY_1DB => HsaDim3 {
            x: descriptor.width as u32,
            y: 1,
            z: 1,
        },
        IMAGE_GEOMETRY_2D | IMAGE_GEOMETRY_2DDEPTH => HsaDim3 {
            x: descriptor.width as u32,
            y: descriptor.height as u32,
            z: 1,
        },
        IMAGE_GEOMETRY_3D => HsaDim3 {
            x: descriptor.width as u32,
            y: descriptor.height as u32,
            z: descriptor.depth as u32,
        },
        IMAGE_GEOMETRY_1DA => HsaDim3 {
            x: descriptor.width as u32,
            y: descriptor.array_size as u32,
            z: 1,
        },
        IMAGE_GEOMETRY_2DA | IMAGE_GEOMETRY_2DADEPTH => HsaDim3 {
            x: descriptor.width as u32,
            y: descriptor.height as u32,
            z: descriptor.array_size as u32,
        },
        _ => HsaDim3 { x: 0, y: 0, z: 0 },
    }
}

fn region_fits(descriptor: HsaExtImageDescriptor, offset: HsaDim3, range: HsaDim3) -> bool {
    let extent = image_extent(descriptor);
    offset
        .x
        .checked_add(range.x)
        .is_some_and(|end| end <= extent.x)
        && offset
            .y
            .checked_add(range.y)
            .is_some_and(|end| end <= extent.y)
        && offset
            .z
            .checked_add(range.z)
            .is_some_and(|end| end <= extent.z)
}

fn compatible_copy_formats(source: HsaExtImageFormat, destination: HsaExtImageFormat) -> bool {
    source == destination
        || (source.channel_type == 2
            && destination.channel_type == 2
            && matches!(
                (source.channel_order, destination.channel_order),
                (8, 14) | (14, 8)
            ))
}

fn row_address(
    base: usize,
    row_pitch: usize,
    slice_pitch: usize,
    element_size: usize,
    origin: HsaDim3,
    row: u32,
    slice: u32,
) -> Option<usize> {
    base.checked_add((origin.x as usize).checked_mul(element_size)?)?
        .checked_add(
            (origin.y as usize)
                .checked_add(row as usize)?
                .checked_mul(row_pitch)?,
        )?
        .checked_add(
            (origin.z as usize)
                .checked_add(slice as usize)?
                .checked_mul(slice_pitch)?,
        )
}

fn copy_image_rows(
    source: ImageView,
    source_origin: HsaDim3,
    destination: ImageView,
    destination_origin: HsaDim3,
    range: HsaDim3,
) -> Status {
    if source.agent != destination.agent
        || source.access_permission == ACCESS_PERMISSION_WO
        || destination.access_permission == ACCESS_PERMISSION_RO
        || !compatible_copy_formats(source.descriptor.format, destination.descriptor.format)
        || source.element_size != destination.element_size
        || !region_fits(source.descriptor, source_origin, range)
        || !region_fits(destination.descriptor, destination_origin, range)
    {
        return INVALID_ARGUMENT;
    }
    let Some(row_bytes) = (range.x as usize).checked_mul(source.element_size) else {
        return INVALID_ARGUMENT;
    };
    let convert_srgb =
        source.descriptor.format.channel_order != destination.descriptor.format.channel_order;
    for slice in 0..range.z {
        for row in 0..range.y {
            let Some(source_row) = row_address(
                source.host_address,
                source.row_pitch,
                source.slice_pitch,
                source.element_size,
                source_origin,
                row,
                slice,
            ) else {
                return INVALID_ARGUMENT;
            };
            let Some(destination_row) = row_address(
                destination.host_address,
                destination.row_pitch,
                destination.slice_pitch,
                destination.element_size,
                destination_origin,
                row,
                slice,
            ) else {
                return INVALID_ARGUMENT;
            };
            if convert_srgb {
                for column in 0..range.x as usize {
                    let byte_offset = column * 4;
                    // SAFETY: Both image regions were checked against their
                    // live backing layouts and use four-byte RGBA elements.
                    unsafe {
                        let source_pixel = (source_row + byte_offset) as *const u8;
                        let destination_pixel = (destination_row + byte_offset) as *mut u8;
                        let convert = if source.descriptor.format.channel_order == 14 {
                            standard_to_linear_byte
                        } else {
                            linear_to_standard_byte
                        };
                        destination_pixel.write(convert(source_pixel.read()));
                        destination_pixel
                            .add(1)
                            .write(convert(source_pixel.add(1).read()));
                        destination_pixel
                            .add(2)
                            .write(convert(source_pixel.add(2).read()));
                        destination_pixel.add(3).write(source_pixel.add(3).read());
                    }
                }
            } else {
                // SAFETY: The caller owns non-overlapping image regions and
                // both checked layouts contain row_bytes at these addresses.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        source_row as *const u8,
                        destination_row as *mut u8,
                        row_bytes,
                    )
                };
            }
        }
    }
    SUCCESS
}

fn linear_pitches(
    geometry: u32,
    element_size: usize,
    range: HsaDim3,
    row_pitch: usize,
    slice_pitch: usize,
) -> Result<(usize, usize, usize), Status> {
    let row_bytes = (range.x as usize)
        .checked_mul(element_size)
        .ok_or(INVALID_ARGUMENT)?;
    let row_pitch = row_pitch.max(row_bytes);
    let minimum_slice = if geometry == IMAGE_GEOMETRY_1DA {
        row_pitch
    } else if matches!(
        geometry,
        IMAGE_GEOMETRY_3D | IMAGE_GEOMETRY_2DA | IMAGE_GEOMETRY_2DADEPTH
    ) {
        row_pitch
            .checked_mul(range.y as usize)
            .ok_or(INVALID_ARGUMENT)?
    } else {
        0
    };
    let slice_pitch = slice_pitch.max(minimum_slice);
    let extent = if range.x == 0 || range.y == 0 || range.z == 0 {
        0
    } else {
        (range.z as usize - 1)
            .checked_mul(slice_pitch)
            .and_then(|value| value.checked_add((range.y as usize - 1).checked_mul(row_pitch)?))
            .and_then(|value| value.checked_add(row_bytes))
            .ok_or(INVALID_ARGUMENT)?
    };
    Ok((row_pitch, slice_pitch, extent))
}

fn host_copy_address(
    runtime: &Runtime,
    agent: HsaAgent,
    address: usize,
    size: usize,
) -> Result<usize, Status> {
    if size == 0 {
        return Ok(address);
    }
    match resolve_memory_range(runtime, agent, address, size) {
        Ok(resolved) => resolved.host_address.ok_or(INVALID_ALLOCATION),
        Err(INVALID_ALLOCATION) => Ok(address),
        Err(status) => Err(status),
    }
}

fn normalize_byte(value: u8) -> f32 {
    f32::from(value) / f32::from(u8::MAX)
}

fn denormalize_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * f32::from(u8::MAX))
        .round_ties_even()
        .clamp(0.0, f32::from(u8::MAX)) as u8
}

fn standard_to_linear_byte(value: u8) -> u8 {
    let value = normalize_byte(value);
    denormalize_byte(if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    })
}

fn linear_to_standard_byte(value: u8) -> u8 {
    let value = normalize_byte(value);
    denormalize_byte(if value < 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(5.0 / 12.0) - 0.055
    })
}

fn float_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = (bits >> 23) & 0xff;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0 && mantissa == 0 {
        return sign;
    }
    if exponent == 0xff {
        return sign
            | if mantissa == 0 {
                0x7c00
            } else if mantissa & 0x40_0000 != 0 {
                0x7e00
            } else {
                0x7c01
            };
    }
    let maximum_normal_exponent = 0x477f_e000_u32 >> 23;
    let minimum_normal_exponent = 0x3880_0000_u32 >> 23;
    let minimum_subnormal_exponent = 0x3380_0000_u32 >> 23;
    if exponent > maximum_normal_exponent {
        sign | 0x7bff
    } else if exponent < minimum_subnormal_exponent {
        sign
    } else if exponent < minimum_normal_exponent {
        sign | (((0x0400 | (mantissa >> 13)) >> (127 - exponent - 14)) as u16)
    } else {
        sign | ((((exponent - 127 + 15) << 10) | (mantissa >> 13)) as u16)
    }
}

fn component_indices(channel_order: u32) -> Option<&'static [usize]> {
    match channel_order {
        0 => Some(&[3]),
        1 => Some(&[0]),
        3 => Some(&[0, 1]),
        5 => Some(&[0, 3]),
        6 => Some(&[0, 1, 2]),
        8 | 14 => Some(&[0, 1, 2, 3]),
        9 => Some(&[2, 1, 0, 3]),
        10 => Some(&[3, 0, 1, 2]),
        16..=18 => Some(&[0]),
        _ => None,
    }
}

unsafe fn pattern_f32(pattern: *const c_void, index: usize) -> f32 {
    // SAFETY: The image clear ABI supplies four readable access components.
    unsafe { pattern.cast::<f32>().add(index).read_unaligned() }
}

unsafe fn pattern_i32(pattern: *const c_void, index: usize) -> i32 {
    // SAFETY: The image clear ABI supplies four readable access components.
    unsafe { pattern.cast::<i32>().add(index).read_unaligned() }
}

unsafe fn pattern_u32(pattern: *const c_void, index: usize) -> u32 {
    // SAFETY: The image clear ABI supplies four readable access components.
    unsafe { pattern.cast::<u32>().add(index).read_unaligned() }
}

fn format_clear_pattern(
    format: HsaExtImageFormat,
    pattern: *const c_void,
    element_size: usize,
) -> Result<[u8; 16], Status> {
    let indices = component_indices(format.channel_order).ok_or(IMAGE_FORMAT_UNSUPPORTED)?;
    let mut output = [0_u8; 16];
    match format.channel_type {
        5..=7 => {
            // SAFETY: Packed normalized formats use floating-point access components.
            let component = |index| unsafe { pattern_f32(pattern, index) }.clamp(0.0, 1.0);
            let r = component(indices[0]);
            let g = component(indices[1]);
            let b = component(indices[2]);
            let packed = match format.channel_type {
                5 => {
                    (((r * 31.0).round_ties_even() as u32) << 10)
                        | (((g * 31.0).round_ties_even() as u32) << 5)
                        | (b * 31.0).round_ties_even() as u32
                }
                6 => {
                    (((r * 31.0).round_ties_even() as u32) << 11)
                        | (((g * 63.0).round_ties_even() as u32) << 5)
                        | (b * 31.0).round_ties_even() as u32
                }
                _ => {
                    (((r * 1023.0).round_ties_even() as u32) << 20)
                        | (((g * 1023.0).round_ties_even() as u32) << 10)
                        | (b * 1023.0).round_ties_even() as u32
                }
            };
            output[..element_size].copy_from_slice(&packed.to_le_bytes()[..element_size]);
        }
        channel_type => {
            for (component, &index) in indices.iter().enumerate() {
                let byte = match channel_type {
                    0 => {
                        let value = unsafe { pattern_f32(pattern, index) };
                        let value = (value * f32::from(i8::MAX))
                            .round_ties_even()
                            .clamp(f32::from(i8::MIN), f32::from(i8::MAX))
                            as i8;
                        output[component] = value as u8;
                        1
                    }
                    1 => {
                        let value = unsafe { pattern_f32(pattern, index) };
                        let value = (value * f32::from(i16::MAX))
                            .round_ties_even()
                            .clamp(f32::from(i16::MIN), f32::from(i16::MAX))
                            as i16;
                        let offset = component * 2;
                        output[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
                        2
                    }
                    2 => {
                        let mut value = unsafe { pattern_f32(pattern, index) };
                        if format.channel_order == 14 && component < 3 {
                            value = if value.is_nan() {
                                0.0
                            } else if value > 1.0 {
                                1.0
                            } else if value < 0.0 {
                                0.0
                            } else if value < 0.003_130_8 {
                                12.92 * value
                            } else {
                                1.055 * value.powf(5.0 / 12.0) - 0.055
                            };
                        }
                        output[component] = denormalize_byte(value);
                        1
                    }
                    3 => {
                        let value = unsafe { pattern_f32(pattern, index) };
                        let value = (value * f32::from(u16::MAX))
                            .round_ties_even()
                            .clamp(0.0, f32::from(u16::MAX))
                            as u16;
                        let offset = component * 2;
                        output[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
                        2
                    }
                    8 => {
                        output[component] = unsafe { pattern_i32(pattern, index) } as i8 as u8;
                        1
                    }
                    9 => {
                        let offset = component * 2;
                        output[offset..offset + 2].copy_from_slice(
                            &(unsafe { pattern_i32(pattern, index) } as i16).to_le_bytes(),
                        );
                        2
                    }
                    10 => {
                        let offset = component * 4;
                        output[offset..offset + 4]
                            .copy_from_slice(&unsafe { pattern_i32(pattern, index) }.to_le_bytes());
                        4
                    }
                    11 => {
                        output[component] = unsafe { pattern_u32(pattern, index) } as u8;
                        1
                    }
                    12 => {
                        let offset = component * 2;
                        output[offset..offset + 2].copy_from_slice(
                            &(unsafe { pattern_u32(pattern, index) } as u16).to_le_bytes(),
                        );
                        2
                    }
                    13 => {
                        let offset = component * 4;
                        output[offset..offset + 4]
                            .copy_from_slice(&unsafe { pattern_u32(pattern, index) }.to_le_bytes());
                        4
                    }
                    14 => {
                        let offset = component * 2;
                        output[offset..offset + 2].copy_from_slice(
                            &float_to_half(unsafe { pattern_f32(pattern, index) }).to_le_bytes(),
                        );
                        2
                    }
                    15 => {
                        let offset = component * 4;
                        output[offset..offset + 4]
                            .copy_from_slice(&unsafe { pattern_f32(pattern, index) }.to_le_bytes());
                        4
                    }
                    _ => return Err(IMAGE_FORMAT_UNSUPPORTED),
                };
                if (component + 1) * byte > element_size {
                    return Err(IMAGE_FORMAT_UNSUPPORTED);
                }
            }
        }
    }
    Ok(output)
}

fn query_capability(
    agent: HsaAgent,
    geometry: u32,
    format: HsaExtImageFormat,
) -> Result<u32, Status> {
    if geometry > IMAGE_GEOMETRY_2DADEPTH {
        return Err(INVALID_ARGUMENT);
    }
    let guard = lock()?;
    let Some(runtime) = guard.as_ref() else {
        return Err(NOT_INITIALIZED);
    };
    if !runtime.is_agent(agent) {
        return Err(INVALID_AGENT);
    }
    Ok(if runtime.gpu_index(agent).is_some() {
        image_property(format, geometry).map_or(0, |property| property.capability)
    } else {
        0
    })
}

#[unsafe(no_mangle)]
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
        // SAFETY: The caller supplied a readable format descriptor.
        let format = unsafe { image_format.read() };
        match query_capability(agent, geometry, format) {
            Ok(capability) => {
                // SAFETY: The caller supplied writable output storage.
                unsafe { capability_mask.write(capability) };
                SUCCESS
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_get_capability_with_layout(
    agent: HsaAgent,
    geometry: u32,
    image_format: *const HsaExtImageFormat,
    image_data_layout: u32,
    capability_mask: *mut u32,
) -> Status {
    boundary(|| {
        if image_data_layout != IMAGE_DATA_LAYOUT_LINEAR {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The pointer contract is identical to the base capability API.
        unsafe { hsa_ext_image_get_capability(agent, geometry, image_format, capability_mask) }
    })
}

fn query_image_data_info(
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptor,
    access_permission: u32,
    mipmap_levels: usize,
    explicit_pitch: Option<(usize, usize)>,
    image_data_info: *mut HsaExtImageDataInfo,
) -> Status {
    if image_data_info.is_null() {
        return INVALID_ARGUMENT;
    }
    let guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(runtime) = guard.as_ref() else {
        return NOT_INITIALIZED;
    };
    if !runtime.is_agent(agent) {
        return INVALID_AGENT;
    }
    // SAFETY: The caller supplied writable output storage.
    unsafe { image_data_info.write(HsaExtImageDataInfo::default()) };
    if runtime.gpu_index(agent).is_none() {
        return IMAGE_FORMAT_UNSUPPORTED;
    }
    match image_layout(descriptor, access_permission, mipmap_levels, explicit_pitch) {
        Ok(layout) => {
            // SAFETY: The caller supplied writable output storage.
            unsafe {
                image_data_info.write(HsaExtImageDataInfo {
                    size: layout.size,
                    alignment: layout.alignment,
                });
            }
            SUCCESS
        }
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_data_get_info(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptor,
    access_permission: u32,
    image_data_info: *mut HsaExtImageDataInfo,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() || image_data_info.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable image descriptor.
        let descriptor = unsafe { image_descriptor.read() };
        query_image_data_info(
            agent,
            descriptor,
            access_permission,
            1,
            None,
            image_data_info,
        )
    })
}

#[unsafe(no_mangle)]
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
        if image_descriptor.is_null()
            || image_data_info.is_null()
            || image_data_layout != IMAGE_DATA_LAYOUT_LINEAR
        {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable image descriptor.
        let descriptor = unsafe { image_descriptor.read() };
        query_image_data_info(
            agent,
            descriptor,
            access_permission,
            1,
            Some((image_data_row_pitch, image_data_slice_pitch)),
            image_data_info,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_data_get_info_v2(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptorV2,
    access_permission: u32,
    image_data_info: *mut HsaExtImageDataInfo,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() || image_data_info.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable version-two image descriptor.
        let descriptor = unsafe { image_descriptor.read() };
        query_image_data_info(
            agent,
            HsaExtImageDescriptor {
                geometry: descriptor.geometry,
                width: descriptor.width,
                height: descriptor.height,
                depth: descriptor.depth,
                array_size: descriptor.array_size,
                format: descriptor.format,
            },
            access_permission,
            descriptor.mipmap_levels,
            None,
            image_data_info,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_create(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptor,
    image_data: *const c_void,
    access_permission: u32,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() || image_data.is_null() || image.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable descriptor.
        let descriptor = unsafe { image_descriptor.read() };
        create_image(
            agent,
            descriptor,
            image_data,
            access_permission,
            1,
            None,
            image,
        )
    })
}

#[unsafe(no_mangle)]
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
        if image_descriptor.is_null()
            || image_data.is_null()
            || image.is_null()
            || image_data_layout != IMAGE_DATA_LAYOUT_LINEAR
        {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable descriptor.
        let descriptor = unsafe { image_descriptor.read() };
        create_image(
            agent,
            descriptor,
            image_data,
            access_permission,
            1,
            Some((image_data_row_pitch, image_data_slice_pitch)),
            image,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_create_v2(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptorV2,
    image_data: *const c_void,
    access_permission: u32,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() || image_data.is_null() || image.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable version-two descriptor.
        let descriptor = unsafe { image_descriptor.read() };
        create_image(
            agent,
            HsaExtImageDescriptor {
                geometry: descriptor.geometry,
                width: descriptor.width,
                height: descriptor.height,
                depth: descriptor.depth,
                array_size: descriptor.array_size,
                format: descriptor.format,
            },
            image_data,
            access_permission,
            descriptor.mipmap_levels.max(1),
            None,
            image,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_image_create(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptor,
    image_layout: *const HsaAmdImageDescriptor,
    image_data: *const c_void,
    access_permission: u32,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null()
            || image_layout.is_null()
            || image_data.is_null()
            || image.is_null()
        {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable image descriptor and AMD
        // metadata whose lifetime covers this synchronous creation call.
        unsafe {
            create_amd_image(
                agent,
                AmdImageCreateRequest {
                    descriptor: image_descriptor.read(),
                    layout: image_layout,
                    data: image_data,
                    access_permission,
                    mipmap_levels: 1,
                    allow_missing_layout: false,
                },
                image,
            )
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_image_create_v2(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptorV2,
    image_layout: *const HsaAmdImageDescriptor,
    image_data: *const c_void,
    access_permission: u32,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() || image_data.is_null() || image.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable version-two descriptor.
        let descriptor = unsafe { image_descriptor.read() };
        // SAFETY: The optional AMD metadata, when present, remains readable
        // for this synchronous creation call.
        unsafe {
            create_amd_image(
                agent,
                AmdImageCreateRequest {
                    descriptor: HsaExtImageDescriptor {
                        geometry: descriptor.geometry,
                        width: descriptor.width,
                        height: descriptor.height,
                        depth: descriptor.depth,
                        array_size: descriptor.array_size,
                        format: descriptor.format,
                    },
                    layout: image_layout,
                    data: image_data,
                    access_permission,
                    mipmap_levels: descriptor.mipmap_levels.max(1),
                    allow_missing_layout: true,
                },
                image,
            )
        }
    })
}

fn destroy_image(agent: HsaAgent, image: HsaExtImage) -> Status {
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let runtime = match initialized_mut(&mut guard) {
        Ok(runtime) => runtime,
        Err(status) => return status,
    };
    if runtime.gpu_index(agent).is_none() {
        return INVALID_AGENT;
    }
    let Some(mut owned) = runtime.images.remove(&image.handle) else {
        return INVALID_ARGUMENT;
    };
    if owned.agent != agent {
        runtime.images.insert(image.handle, owned);
        return INVALID_AGENT;
    }
    match owned.free() {
        Ok(()) => SUCCESS,
        Err(error) => {
            runtime.images.insert(image.handle, owned);
            map_error(error)
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ext_image_destroy(agent: HsaAgent, image: HsaExtImage) -> Status {
    boundary(|| destroy_image(agent, image))
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ext_image_destroy_v2(agent: HsaAgent, image: HsaExtImage) -> Status {
    boundary(|| destroy_image(agent, image))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_mipmap_array_get_level(
    agent: HsaAgent,
    mipmapped_array: *const HsaExtImage,
    mip_level: u32,
    image_descriptor: *const HsaExtImageDescriptorV2,
    level_image_out: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if mipmapped_array.is_null() || level_image_out.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied writable output storage.
        unsafe { level_image_out.write(HsaExtImage { handle: 0 }) };
        // SAFETY: The caller supplied a readable parent handle.
        let parent_handle = unsafe { mipmapped_array.read() };
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(gpu_index) = runtime.gpu_index(agent) else {
            return INVALID_AGENT;
        };
        let Some(parent) = runtime.images.get(&parent_handle.handle) else {
            return INVALID_ARGUMENT;
        };
        if parent.agent != agent {
            return INVALID_AGENT;
        }
        let level = mip_level as usize;
        if parent.mipmap_levels <= 1 || level >= parent.mipmap_levels {
            return INVALID_ARGUMENT;
        }
        let mut descriptor = level_extent(parent.descriptor, level);
        if !image_descriptor.is_null() {
            // ROCr uses the optional descriptor to override the view format.
            let requested = unsafe { image_descriptor.read() };
            let Some(parent_property) =
                image_property(parent.descriptor.format, parent.descriptor.geometry)
            else {
                return IMAGE_FORMAT_UNSUPPORTED;
            };
            let Some(requested_property) = image_property(requested.format, descriptor.geometry)
            else {
                return IMAGE_FORMAT_UNSUPPORTED;
            };
            if requested_property.element_size != parent_property.element_size {
                return INVALID_ARGUMENT;
            }
            descriptor.format = requested.format;
        }
        let layout = match image_layout(descriptor, parent.access_permission, 1, None) {
            Ok(layout) => layout,
            Err(status) => return status,
        };
        let Some(&offset) = parent.layout.mip_offsets.get(level) else {
            return INVALID_ARGUMENT;
        };
        let Some(data_device_address) = parent.data_device_address.checked_add(offset as u64)
        else {
            return IMAGE_SIZE_UNSUPPORTED;
        };
        let data_host_address = parent
            .data_host_address
            .and_then(|address| address.checked_add(offset));
        let access_permission = parent.access_permission;
        store_image(
            runtime,
            gpu_index,
            ImageCreateInfo {
                agent,
                descriptor,
                access_permission,
                layout,
                data_device_address,
                data_host_address,
                mipmap_levels: 1,
            },
            level_image_out,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_copy(
    agent: HsaAgent,
    src_image: HsaExtImage,
    src_offset: *const HsaDim3,
    dst_image: HsaExtImage,
    dst_offset: *const HsaDim3,
    range: *const HsaDim3,
) -> Status {
    boundary(|| {
        if src_image.handle == 0
            || dst_image.handle == 0
            || src_offset.is_null()
            || dst_offset.is_null()
            || range.is_null()
        {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied readable coordinate descriptors.
        let (src_offset, dst_offset, range) =
            unsafe { (src_offset.read(), dst_offset.read(), range.read()) };
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        let source = match image_view(runtime, agent, src_image) {
            Ok(image) => image,
            Err(status) => return status,
        };
        let destination = match image_view(runtime, agent, dst_image) {
            Ok(image) => image,
            Err(status) => return status,
        };
        copy_image_rows(source, src_offset, destination, dst_offset, range)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_import(
    agent: HsaAgent,
    src_memory: *const c_void,
    src_row_pitch: usize,
    src_slice_pitch: usize,
    dst_image: HsaExtImage,
    image_region: *const HsaExtImageRegion,
) -> Status {
    boundary(|| {
        if src_memory.is_null() || dst_image.handle == 0 || image_region.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable region descriptor.
        let region = unsafe { image_region.read() };
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        let destination = match image_view(runtime, agent, dst_image) {
            Ok(image) => image,
            Err(status) => return status,
        };
        if destination.access_permission == ACCESS_PERMISSION_RO
            || !region_fits(destination.descriptor, region.offset, region.range)
        {
            return INVALID_ARGUMENT;
        }
        let (row_pitch, slice_pitch, extent) = match linear_pitches(
            destination.descriptor.geometry,
            destination.element_size,
            region.range,
            src_row_pitch,
            src_slice_pitch,
        ) {
            Ok(pitches) => pitches,
            Err(status) => return status,
        };
        let host_address = match host_copy_address(runtime, agent, src_memory as usize, extent) {
            Ok(address) => address,
            Err(status) => return status,
        };
        let source = ImageView {
            agent,
            descriptor: destination.descriptor,
            access_permission: ACCESS_PERMISSION_RO,
            host_address,
            row_pitch,
            slice_pitch,
            element_size: destination.element_size,
        };
        copy_image_rows(
            source,
            HsaDim3 { x: 0, y: 0, z: 0 },
            destination,
            region.offset,
            region.range,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_export(
    agent: HsaAgent,
    src_image: HsaExtImage,
    dst_memory: *mut c_void,
    dst_row_pitch: usize,
    dst_slice_pitch: usize,
    image_region: *const HsaExtImageRegion,
) -> Status {
    boundary(|| {
        if src_image.handle == 0 || dst_memory.is_null() || image_region.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable region descriptor.
        let region = unsafe { image_region.read() };
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        let source = match image_view(runtime, agent, src_image) {
            Ok(image) => image,
            Err(status) => return status,
        };
        if source.access_permission == ACCESS_PERMISSION_WO
            || !region_fits(source.descriptor, region.offset, region.range)
        {
            return INVALID_ARGUMENT;
        }
        let (row_pitch, slice_pitch, extent) = match linear_pitches(
            source.descriptor.geometry,
            source.element_size,
            region.range,
            dst_row_pitch,
            dst_slice_pitch,
        ) {
            Ok(pitches) => pitches,
            Err(status) => return status,
        };
        let host_address = match host_copy_address(runtime, agent, dst_memory as usize, extent) {
            Ok(address) => address,
            Err(status) => return status,
        };
        let destination = ImageView {
            agent,
            descriptor: source.descriptor,
            access_permission: ACCESS_PERMISSION_WO,
            host_address,
            row_pitch,
            slice_pitch,
            element_size: source.element_size,
        };
        copy_image_rows(
            source,
            region.offset,
            destination,
            HsaDim3 { x: 0, y: 0, z: 0 },
            region.range,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_clear(
    agent: HsaAgent,
    image: HsaExtImage,
    data: *const c_void,
    image_region: *const HsaExtImageRegion,
) -> Status {
    boundary(|| {
        if image.handle == 0 || data.is_null() || image_region.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable region descriptor.
        let region = unsafe { image_region.read() };
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        let image = match image_view(runtime, agent, image) {
            Ok(image) => image,
            Err(status) => return status,
        };
        if image.access_permission == ACCESS_PERMISSION_RO
            || !region_fits(image.descriptor, region.offset, region.range)
        {
            return INVALID_ARGUMENT;
        }
        let pattern = match format_clear_pattern(image.descriptor.format, data, image.element_size)
        {
            Ok(pattern) => pattern,
            Err(status) => return status,
        };
        for slice in 0..region.range.z {
            for row in 0..region.range.y {
                let Some(row_address) = row_address(
                    image.host_address,
                    image.row_pitch,
                    image.slice_pitch,
                    image.element_size,
                    region.offset,
                    row,
                    slice,
                ) else {
                    return INVALID_ARGUMENT;
                };
                for column in 0..region.range.x as usize {
                    let Some(pixel) = row_address.checked_add(column * image.element_size) else {
                        return INVALID_ARGUMENT;
                    };
                    // SAFETY: The region and image layout checks establish a
                    // complete writable element at this backing address.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            pattern.as_ptr(),
                            pixel as *mut u8,
                            image.element_size,
                        )
                    };
                }
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_image_get_info_max_dim(
    agent: HsaAgent,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if !runtime.is_agent(agent) {
            return INVALID_AGENT;
        }
        let gpu = runtime.gpu_index(agent).is_some();
        // SAFETY: Each attribute selects its documented writable output extent.
        unsafe {
            match attribute {
                EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS => {
                    value.cast::<u32>().write(if gpu { 16_384 } else { 0 });
                }
                EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS => {
                    value.cast::<u32>().write(if gpu { 16_384 } else { 0 });
                }
                EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS => {
                    value.cast::<u32>().write(if gpu { u32::MAX } else { 0 });
                }
                EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS
                | EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS
                | EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS
                | EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS => {
                    value
                        .cast::<[u32; 2]>()
                        .write(if gpu { [16_384, 16_384] } else { [0; 2] });
                }
                EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS => {
                    value.cast::<[u32; 3]>().write(if gpu {
                        [16_384, 16_384, 8192]
                    } else {
                        [0; 3]
                    });
                }
                EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS => {
                    value.cast::<u32>().write(if gpu { 8192 } else { 0 });
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

/// Runtime-owned sampler descriptor allocation and its creating agent.
pub(crate) struct Sampler {
    allocation: Allocation,
    agent: HsaAgent,
}

impl Sampler {
    fn free(&mut self) -> Result<(), rocddi::Error> {
        self.allocation.free()
    }
}

fn clamp(address_mode: u32) -> Option<u32> {
    match address_mode {
        SAMPLER_ADDRESSING_MODE_UNDEFINED | SAMPLER_ADDRESSING_MODE_REPEAT => Some(0),
        SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE => Some(2),
        SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER => Some(6),
        SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT => Some(1),
        _ => None,
    }
}

fn sampler_srd(descriptor: HsaExtSamplerDescriptorV2) -> Result<[u32; 8], Status> {
    if !matches!(
        descriptor.coordinate_mode,
        SAMPLER_COORDINATE_MODE_UNNORMALIZED | SAMPLER_COORDINATE_MODE_NORMALIZED
    ) || !matches!(
        descriptor.filter_mode,
        SAMPLER_FILTER_MODE_NEAREST | SAMPLER_FILTER_MODE_LINEAR
    ) || !matches!(
        descriptor.mipmap_filter_mode,
        SAMPLER_FILTER_MODE_NEAREST | SAMPLER_FILTER_MODE_LINEAR | SAMPLER_FILTER_MODE_NONE
    ) {
        return Err(SAMPLER_DESCRIPTOR_UNSUPPORTED);
    }

    let clamp_x = clamp(descriptor.address_modes[0]).ok_or(SAMPLER_DESCRIPTOR_UNSUPPORTED)?;
    let clamp_y = clamp(descriptor.address_modes[1]).ok_or(SAMPLER_DESCRIPTOR_UNSUPPORTED)?;
    let clamp_z = clamp(descriptor.address_modes[2]).ok_or(SAMPLER_DESCRIPTOR_UNSUPPORTED)?;

    let mut srd = [0_u32; SAMPLER_DESCRIPTOR_DWORDS];
    srd[0] = clamp_x | (clamp_y << 3) | (clamp_z << 6);
    if descriptor.coordinate_mode == SAMPLER_COORDINATE_MODE_UNNORMALIZED {
        srd[0] |= 1 << 15;
    }
    srd[1] = 4095 << 13;
    if descriptor.filter_mode == SAMPLER_FILTER_MODE_LINEAR {
        srd[2] |= 1 << 20;
        srd[2] |= 1 << 22;
    }
    srd[2] |= match descriptor.mipmap_filter_mode {
        SAMPLER_FILTER_MODE_NEAREST => 1 << 26,
        SAMPLER_FILTER_MODE_LINEAR => 2 << 26,
        _ => 0,
    };
    Ok(srd)
}

fn create_sampler(
    agent: HsaAgent,
    descriptor: HsaExtSamplerDescriptorV2,
    sampler: *mut HsaExtSampler,
) -> Status {
    if sampler.is_null() {
        return INVALID_ARGUMENT;
    }
    let srd = match sampler_srd(descriptor) {
        Ok(srd) => srd,
        Err(status) => return status,
    };
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let runtime = match initialized_mut(&mut guard) {
        Ok(runtime) => runtime,
        Err(status) => return status,
    };
    let Some(index) = runtime.gpu_index(agent) else {
        return INVALID_AGENT;
    };
    if runtime.samplers.try_reserve(1).is_err() {
        return OUT_OF_RESOURCES;
    }
    let allocation = match runtime.gpus[index].device.allocate(
        MemoryKind::System,
        SAMPLER_ALLOCATION_BYTES,
        SAMPLER_ALLOCATION_BYTES,
        DeviceAccess::READ | DeviceAccess::WRITE,
    ) {
        Ok(allocation) => allocation,
        Err(error) => return map_error(error),
    };
    let info = allocation.info();
    let Some(host_address) = info.host_address else {
        return OUT_OF_RESOURCES;
    };
    if info.device_address == 0 || runtime.samplers.contains_key(&info.device_address) {
        return OUT_OF_RESOURCES;
    }
    // SAFETY: The dedicated GPU-visible allocation has room for the complete
    // descriptor and remains owned by the sampler map for the handle lifetime.
    unsafe {
        std::ptr::copy_nonoverlapping(
            srd.as_ptr().cast::<u8>(),
            (host_address as *mut c_void).cast::<u8>(),
            size_of_val(&srd),
        );
        sampler.write(HsaExtSampler {
            handle: info.device_address,
        });
    }
    runtime
        .samplers
        .insert(info.device_address, Sampler { allocation, agent });
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_sampler_create(
    agent: HsaAgent,
    sampler_descriptor: *const HsaExtSamplerDescriptor,
    sampler: *mut HsaExtSampler,
) -> Status {
    boundary(|| {
        if sampler_descriptor.is_null() || sampler.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable descriptor.
        let descriptor = unsafe { sampler_descriptor.read() };
        create_sampler(
            agent,
            HsaExtSamplerDescriptorV2 {
                coordinate_mode: descriptor.coordinate_mode,
                filter_mode: descriptor.filter_mode,
                mipmap_filter_mode: SAMPLER_FILTER_MODE_NONE,
                address_modes: [descriptor.address_mode; 3],
            },
            sampler,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_sampler_create_v2(
    agent: HsaAgent,
    sampler_descriptor: *const HsaExtSamplerDescriptorV2,
    sampler: *mut HsaExtSampler,
) -> Status {
    boundary(|| {
        if sampler_descriptor.is_null() || sampler.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable descriptor.
        let descriptor = unsafe { sampler_descriptor.read() };
        create_sampler(agent, descriptor, sampler)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ext_sampler_destroy(agent: HsaAgent, sampler: HsaExtSampler) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        let Some(mut owned) = runtime.samplers.remove(&sampler.handle) else {
            return INVALID_ARGUMENT;
        };
        if owned.agent != agent {
            runtime.samplers.insert(sampler.handle, owned);
            return INVALID_AGENT;
        }
        match owned.free() {
            Ok(()) => SUCCESS,
            Err(error) => {
                runtime.samplers.insert(sampler.handle, owned);
                map_error(error)
            }
        }
    })
}

pub(crate) fn image_extension_table() -> [usize; 14] {
    [
        hsa_ext_image_get_capability as *const () as usize,
        hsa_ext_image_data_get_info as *const () as usize,
        hsa_ext_image_create as *const () as usize,
        hsa_ext_image_destroy as *const () as usize,
        hsa_ext_image_copy as *const () as usize,
        hsa_ext_image_import as *const () as usize,
        hsa_ext_image_export as *const () as usize,
        hsa_ext_image_clear as *const () as usize,
        hsa_ext_sampler_create as *const () as usize,
        hsa_ext_sampler_destroy as *const () as usize,
        hsa_ext_image_get_capability_with_layout as *const () as usize,
        hsa_ext_image_data_get_info_with_layout as *const () as usize,
        hsa_ext_image_create_with_layout as *const () as usize,
        hsa_ext_sampler_create_v2 as *const () as usize,
    ]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::mem::{align_of, size_of};

    #[test]
    fn sampler_abi_matches_public_headers() {
        assert_eq!(size_of::<HsaExtSampler>(), 8);
        assert_eq!(align_of::<HsaExtSampler>(), 8);
        assert_eq!(size_of::<HsaExtSamplerDescriptor>(), 12);
        assert_eq!(align_of::<HsaExtSamplerDescriptor>(), 4);
        assert_eq!(size_of::<HsaExtSamplerDescriptorV2>(), 24);
        assert_eq!(align_of::<HsaExtSamplerDescriptorV2>(), 4);
        let _: unsafe extern "C" fn(
            HsaAgent,
            *const HsaExtSamplerDescriptor,
            *mut HsaExtSampler,
        ) -> Status = hsa_ext_sampler_create;
        let _: unsafe extern "C" fn(
            HsaAgent,
            *const HsaExtSamplerDescriptorV2,
            *mut HsaExtSampler,
        ) -> Status = hsa_ext_sampler_create_v2;
        let _: extern "C" fn(HsaAgent, HsaExtSampler) -> Status = hsa_ext_sampler_destroy;
    }

    #[test]
    fn image_query_abi_matches_public_headers() {
        assert_eq!(size_of::<HsaExtImage>(), 8);
        assert_eq!(align_of::<HsaExtImage>(), 8);
        assert_eq!(size_of::<HsaExtImageFormat>(), 8);
        assert_eq!(align_of::<HsaExtImageFormat>(), 4);
        assert_eq!(size_of::<HsaExtImageDescriptor>(), 48);
        assert_eq!(align_of::<HsaExtImageDescriptor>(), 8);
        assert_eq!(size_of::<HsaExtImageDescriptorV2>(), 56);
        assert_eq!(align_of::<HsaExtImageDescriptorV2>(), 8);
        assert_eq!(size_of::<HsaExtImageDataInfo>(), 16);
        assert_eq!(align_of::<HsaExtImageDataInfo>(), 8);
        let _: unsafe extern "C" fn(HsaAgent, u32, *const HsaExtImageFormat, *mut u32) -> Status =
            hsa_ext_image_get_capability;
        let _: unsafe extern "C" fn(
            HsaAgent,
            u32,
            *const HsaExtImageFormat,
            u32,
            *mut u32,
        ) -> Status = hsa_ext_image_get_capability_with_layout;
        let _: unsafe extern "C" fn(
            HsaAgent,
            *const HsaExtImageDescriptor,
            u32,
            *mut HsaExtImageDataInfo,
        ) -> Status = hsa_ext_image_data_get_info;
        let _: unsafe extern "C" fn(
            HsaAgent,
            *const HsaExtImageDescriptor,
            u32,
            u32,
            usize,
            usize,
            *mut HsaExtImageDataInfo,
        ) -> Status = hsa_ext_image_data_get_info_with_layout;
        let _: unsafe extern "C" fn(
            HsaAgent,
            *const HsaExtImageDescriptorV2,
            u32,
            *mut HsaExtImageDataInfo,
        ) -> Status = hsa_ext_image_data_get_info_v2;
        let _: unsafe extern "C" fn(HsaAgent, u32, *mut c_void) -> Status =
            hsa_amd_image_get_info_max_dim;
    }

    #[test]
    fn image_handle_entry_points_match_the_public_abi() {
        assert_eq!(size_of::<HsaDim3>(), 12);
        assert_eq!(align_of::<HsaDim3>(), 4);
        assert_eq!(size_of::<HsaExtImageRegion>(), 24);
        assert_eq!(align_of::<HsaExtImageRegion>(), 4);
        let _: unsafe extern "C" fn(
            HsaAgent,
            *const HsaExtImageDescriptor,
            *const c_void,
            u32,
            *mut HsaExtImage,
        ) -> Status = hsa_ext_image_create;
        let _: unsafe extern "C" fn(
            HsaAgent,
            *const HsaExtImageDescriptor,
            *const c_void,
            u32,
            u32,
            usize,
            usize,
            *mut HsaExtImage,
        ) -> Status = hsa_ext_image_create_with_layout;
        let _: unsafe extern "C" fn(
            HsaAgent,
            *const HsaExtImageDescriptorV2,
            *const c_void,
            u32,
            *mut HsaExtImage,
        ) -> Status = hsa_ext_image_create_v2;
        let _: extern "C" fn(HsaAgent, HsaExtImage) -> Status = hsa_ext_image_destroy;
        let _: extern "C" fn(HsaAgent, HsaExtImage) -> Status = hsa_ext_image_destroy_v2;
        let _: unsafe extern "C" fn(
            HsaAgent,
            *const HsaExtImage,
            u32,
            *const HsaExtImageDescriptorV2,
            *mut HsaExtImage,
        ) -> Status = hsa_ext_image_mipmap_array_get_level;
        let _: unsafe extern "C" fn(
            HsaAgent,
            HsaExtImage,
            *const HsaDim3,
            HsaExtImage,
            *const HsaDim3,
            *const HsaDim3,
        ) -> Status = hsa_ext_image_copy;
        let _: unsafe extern "C" fn(
            HsaAgent,
            *const c_void,
            usize,
            usize,
            HsaExtImage,
            *const HsaExtImageRegion,
        ) -> Status = hsa_ext_image_import;
        let _: unsafe extern "C" fn(
            HsaAgent,
            HsaExtImage,
            *mut c_void,
            usize,
            usize,
            *const HsaExtImageRegion,
        ) -> Status = hsa_ext_image_export;
        let _: unsafe extern "C" fn(
            HsaAgent,
            HsaExtImage,
            *const c_void,
            *const HsaExtImageRegion,
        ) -> Status = hsa_ext_image_clear;
        let _: unsafe extern "C" fn(
            HsaAgent,
            *const HsaExtImageDescriptor,
            *const HsaAmdImageDescriptor,
            *const c_void,
            u32,
            *mut HsaExtImage,
        ) -> Status = hsa_amd_image_create;
        let _: unsafe extern "C" fn(
            HsaAgent,
            *const HsaExtImageDescriptorV2,
            *const HsaAmdImageDescriptor,
            *const c_void,
            u32,
            *mut HsaExtImage,
        ) -> Status = hsa_amd_image_create_v2;
        assert!(image_extension_table().into_iter().all(|entry| entry != 0));
    }

    #[test]
    fn gfx12_image_capability_table_matches_rocr() {
        let property = image_property(
            HsaExtImageFormat {
                channel_type: 2,
                channel_order: 8,
            },
            IMAGE_GEOMETRY_2D,
        );
        assert_eq!(
            property,
            Some(ImageProperty {
                capability: IMAGE_CAPABILITY_RW,
                element_size: 4,
            })
        );
        assert_eq!(
            image_property(
                HsaExtImageFormat {
                    channel_type: 2,
                    channel_order: 14,
                },
                IMAGE_GEOMETRY_2D,
            ),
            Some(ImageProperty {
                capability: IMAGE_CAPABILITY_READ_ONLY,
                element_size: 4,
            })
        );
        assert_eq!(
            image_property(
                HsaExtImageFormat {
                    channel_type: 2,
                    channel_order: 14,
                },
                IMAGE_GEOMETRY_1DB,
            ),
            None
        );
        assert_eq!(
            image_property(
                HsaExtImageFormat {
                    channel_type: 3,
                    channel_order: 18,
                },
                IMAGE_GEOMETRY_2DDEPTH,
            ),
            Some(ImageProperty {
                capability: IMAGE_CAPABILITY_ROWO,
                element_size: 2,
            })
        );
    }

    fn rgba8_2d(width: usize, height: usize) -> HsaExtImageDescriptor {
        HsaExtImageDescriptor {
            geometry: IMAGE_GEOMETRY_2D,
            width,
            height,
            depth: 0,
            array_size: 0,
            format: HsaExtImageFormat {
                channel_type: 2,
                channel_order: 8,
            },
        }
    }

    #[test]
    fn linear_image_layout_validates_pitches_and_extents() {
        let descriptor = rgba8_2d(17, 3);
        assert_eq!(
            image_layout(descriptor, ACCESS_PERMISSION_RW, 1, None),
            Ok(ImageLayout {
                size: 768,
                alignment: 256,
                row_pitch: 256,
                slice_pitch: 0,
                mip_offsets: vec![0],
            })
        );
        assert_eq!(
            image_layout(descriptor, ACCESS_PERMISSION_RW, 1, Some((256, 0))),
            Ok(ImageLayout {
                size: 768,
                alignment: 256,
                row_pitch: 256,
                slice_pitch: 0,
                mip_offsets: vec![0],
            })
        );
        assert_eq!(
            image_layout(descriptor, ACCESS_PERMISSION_RW, 1, Some((128, 0))),
            Err(IMAGE_PITCH_UNSUPPORTED)
        );
        assert_eq!(
            image_layout(
                HsaExtImageDescriptor {
                    width: 0,
                    ..descriptor
                },
                ACCESS_PERMISSION_RW,
                1,
                None,
            ),
            Err(IMAGE_SIZE_UNSUPPORTED)
        );
    }

    #[test]
    fn mipmapped_image_layout_tracks_every_level() {
        assert_eq!(
            image_layout(rgba8_2d(64, 32), ACCESS_PERMISSION_RO, 3, None),
            Ok(ImageLayout {
                size: 14_336,
                alignment: 256,
                row_pitch: 256,
                slice_pitch: 0,
                mip_offsets: vec![0, 8192, 12_288],
            })
        );
        assert_eq!(
            image_layout(rgba8_2d(2, 2), ACCESS_PERMISSION_RO, 3, None),
            Err(INVALID_ARGUMENT)
        );
    }

    #[test]
    fn gfx12_linear_image_srd_matches_rocr_fields() {
        let descriptor = rgba8_2d(17, 3);
        let layout = image_layout(descriptor, ACCESS_PERMISSION_RW, 1, None).unwrap();
        assert_eq!(
            gfx12_image_srd(descriptor, &layout, 0x0000_1234_5678_9000, 1),
            Ok([
                0x3456_7890,
                0x0054_0012,
                0x0000_8004,
                0x9000_0fac,
                0x0000_003f,
                0,
                0,
                0,
                2,
                8,
                17,
                0,
            ])
        );
    }

    #[test]
    fn gfx12_buffer_image_srd_uses_byte_extent() {
        let descriptor = HsaExtImageDescriptor {
            geometry: IMAGE_GEOMETRY_1DB,
            width: 257,
            height: 0,
            depth: 0,
            array_size: 0,
            format: HsaExtImageFormat {
                channel_type: 13,
                channel_order: 1,
            },
        };
        let layout = image_layout(descriptor, ACCESS_PERMISSION_RW, 1, None).unwrap();
        assert_eq!(
            gfx12_image_srd(descriptor, &layout, 0x0000_1234_5678_9abc, 1),
            Ok([
                0x5678_9abc,
                0x0004_1234,
                1028,
                0x0001_4204,
                0,
                0,
                0,
                0,
                13,
                1,
                257,
                0,
            ])
        );
    }

    #[test]
    fn gfx12_mipmap_srd_exposes_the_full_level_range() {
        let descriptor = rgba8_2d(64, 32);
        let layout = image_layout(descriptor, ACCESS_PERMISSION_RO, 3, None).unwrap();
        let srd = gfx12_image_srd(descriptor, &layout, 0x0000_1234_5678_4000, 3).unwrap();
        assert_eq!((srd[1] >> 12) & 0x1f, 2);
        assert_eq!((srd[3] >> 15) & 0x1f, 2);
        assert_eq!(srd[11], 3);
    }

    #[test]
    fn gfx12_amd_image_srd_preserves_layout_and_patches_identity() {
        let descriptor = rgba8_2d(17, 3);
        let words = [
            0xdead_beef,
            3 << 30,
            15 | (31 << 14),
            (3 << 20) | (9 << 28),
            0x1234,
            0x5678,
            0x9abc,
            0xdef0,
        ];
        assert_eq!(
            gfx12_amd_image_srd(descriptor, 0x0000_1234_5678_9000, 1, words),
            Ok([
                0x3456_7890,
                0x0054_0012,
                0x0000_8004,
                0x9030_0fac,
                0x1234,
                0x5678,
                0x9abc,
                0xdef0,
                2,
                8,
                17,
                0,
            ])
        );
    }

    #[test]
    fn linear_image_copy_honors_origins_and_pitches() {
        let descriptor = rgba8_2d(3, 2);
        let source = [
            1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 0, 0, 0, 0,
        ];
        let mut destination = [0_u8; 32];
        let view = |address, access_permission| ImageView {
            agent: HsaAgent { handle: 1 },
            descriptor,
            access_permission,
            host_address: address,
            row_pitch: 16,
            slice_pitch: 0,
            element_size: 4,
        };
        assert_eq!(
            copy_image_rows(
                view(source.as_ptr() as usize, ACCESS_PERMISSION_RO),
                HsaDim3 { x: 1, y: 0, z: 0 },
                view(destination.as_mut_ptr() as usize, ACCESS_PERMISSION_WO),
                HsaDim3 { x: 0, y: 0, z: 0 },
                HsaDim3 { x: 2, y: 2, z: 1 },
            ),
            SUCCESS
        );
        assert_eq!(&destination[..8], &source[4..12]);
        assert_eq!(&destination[16..24], &source[20..28]);
    }

    #[test]
    fn clear_patterns_match_rocr_component_conversion() {
        let rgba = [1.0_f32, 0.5, 0.0, 1.0];
        assert_eq!(
            &format_clear_pattern(
                HsaExtImageFormat {
                    channel_type: 2,
                    channel_order: 8,
                },
                rgba.as_ptr().cast(),
                4,
            )
            .unwrap()[..4],
            &[255, 128, 0, 255]
        );
        assert_eq!(
            &format_clear_pattern(
                HsaExtImageFormat {
                    channel_type: 6,
                    channel_order: 6,
                },
                rgba.as_ptr().cast(),
                2,
            )
            .unwrap()[..2],
            &[0x00, 0xfc]
        );
    }

    #[test]
    fn srgb_copy_converts_color_and_preserves_alpha() {
        assert_eq!(standard_to_linear_byte(0), 0);
        assert_eq!(standard_to_linear_byte(255), 255);
        assert_eq!(linear_to_standard_byte(0), 0);
        assert_eq!(linear_to_standard_byte(255), 255);
        assert_eq!(standard_to_linear_byte(128), 55);
        assert_eq!(linear_to_standard_byte(55), 128);
    }

    #[test]
    fn gfx12_sampler_srd_matches_rocr_layout() {
        assert_eq!(
            sampler_srd(HsaExtSamplerDescriptorV2 {
                coordinate_mode: SAMPLER_COORDINATE_MODE_NORMALIZED,
                filter_mode: SAMPLER_FILTER_MODE_NEAREST,
                mipmap_filter_mode: SAMPLER_FILTER_MODE_NONE,
                address_modes: [SAMPLER_ADDRESSING_MODE_REPEAT; 3],
            }),
            Ok([0, 0x01ff_e000, 0, 0, 0, 0, 0, 0])
        );
        assert_eq!(
            sampler_srd(HsaExtSamplerDescriptorV2 {
                coordinate_mode: SAMPLER_COORDINATE_MODE_UNNORMALIZED,
                filter_mode: SAMPLER_FILTER_MODE_LINEAR,
                mipmap_filter_mode: SAMPLER_FILTER_MODE_LINEAR,
                address_modes: [
                    SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE,
                    SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER,
                    SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT,
                ],
            }),
            Ok([0x8072, 0x01ff_e000, 0x0850_0000, 0, 0, 0, 0, 0])
        );
    }

    #[test]
    fn invalid_sampler_fields_are_rejected() {
        let valid = HsaExtSamplerDescriptorV2 {
            coordinate_mode: SAMPLER_COORDINATE_MODE_NORMALIZED,
            filter_mode: SAMPLER_FILTER_MODE_NEAREST,
            mipmap_filter_mode: SAMPLER_FILTER_MODE_NONE,
            address_modes: [SAMPLER_ADDRESSING_MODE_REPEAT; 3],
        };
        assert_eq!(
            sampler_srd(HsaExtSamplerDescriptorV2 {
                coordinate_mode: 2,
                ..valid
            }),
            Err(SAMPLER_DESCRIPTOR_UNSUPPORTED)
        );
        assert_eq!(
            sampler_srd(HsaExtSamplerDescriptorV2 {
                address_modes: [SAMPLER_ADDRESSING_MODE_REPEAT, 5, 0],
                ..valid
            }),
            Err(SAMPLER_DESCRIPTOR_UNSUPPORTED)
        );
    }

    #[test]
    fn sampler_error_has_the_rocr_status_string() {
        let mut output = std::ptr::null();
        // SAFETY: output is writable pointer storage for the returned static string.
        assert_eq!(
            unsafe { crate::hsa_status_string(SAMPLER_DESCRIPTOR_UNSUPPORTED, &raw mut output) },
            SUCCESS
        );
        // SAFETY: hsa_status_string returned a static NUL-terminated string.
        assert_eq!(
            unsafe { CStr::from_ptr(output) }.to_bytes(),
            b"HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED: Sampler descriptor is not supported or invalid."
        );
    }

    #[test]
    fn image_errors_have_the_rocr_status_strings() {
        let cases = [
            (
                IMAGE_FORMAT_UNSUPPORTED,
                b"HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED: Image format is not supported."
                    .as_slice(),
            ),
            (
                IMAGE_SIZE_UNSUPPORTED,
                b"HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED: Image size is not supported."
                    .as_slice(),
            ),
            (
                IMAGE_PITCH_UNSUPPORTED,
                b"HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED: Image pitch is not supported or invalid."
                    .as_slice(),
            ),
        ];
        for (status, expected) in cases {
            let mut output = std::ptr::null();
            // SAFETY: output is writable pointer storage for the returned static string.
            assert_eq!(
                unsafe { crate::hsa_status_string(status, &raw mut output) },
                SUCCESS
            );
            // SAFETY: hsa_status_string returned a static NUL-terminated string.
            assert_eq!(unsafe { CStr::from_ptr(output) }.to_bytes(), expected);
        }
    }
}
