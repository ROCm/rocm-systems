// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Checked GFX12 surface layouts computed through AMD `AddrLib3`.
//!
//! The library owns no GPU resource. Callers supply hardware address geometry
//! from the same device that will access a surface, then retain the returned
//! layout while constructing descriptors and transferring image data.

use std::ffi::c_void;
use std::ptr::NonNull;

/// Maximum mip levels accepted by GFX12 resource descriptors.
pub const MAX_MIP_LEVELS: usize = 32;
const MAX_MIP_LEVELS_U32: u32 = 32;

/// Texture dimensions passed to `AddrLib3`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ResourceType {
    /// One-dimensional texture or texture array.
    Texture1D = 0,
    /// Two-dimensional texture or texture array.
    Texture2D = 1,
    /// Three-dimensional texture.
    Texture3D = 2,
}

/// Hardware and image dimensions for a surface-layout query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceRequest {
    /// ASIC family ID reported by the GPU driver.
    pub family: u32,
    /// Hardware `GB_ADDR_CONFIG` reported by the GPU driver.
    pub gb_addr_config: u32,
    /// Uncompressed format size in bits per pixel.
    pub bits_per_pixel: u32,
    /// Base mip width.
    pub width: u32,
    /// Base mip height, or one for 1D images.
    pub height: u32,
    /// Base mip depth or array extent.
    pub slices: u32,
    /// Number of mip levels, including the base level.
    pub mip_levels: u32,
    /// Texture dimension class.
    pub resource_type: ResourceType,
}

/// `AddrLib3`'s layout for one mip level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MipLayout {
    /// Byte offset used by the hardware view of this mip level.
    pub offset: u64,
    /// Padded pitch in format elements.
    pub pitch: u32,
    /// Padded height in format elements.
    pub height: u32,
    /// Padded depth or array extent.
    pub depth: u32,
}

/// Complete surface layout for the requested mip chain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SurfaceLayout {
    /// Total byte extent of the surface.
    pub size: u64,
    /// Extent of one array slice, including the mip chain.
    pub slice_size: u64,
    /// Required GPU base-address alignment.
    pub alignment: u32,
    /// Padded base mip pitch in format elements.
    pub pitch: u32,
    /// Padded base mip height in format elements.
    pub height: u32,
    /// GFX12 resource-descriptor swizzle mode.
    pub swizzle_mode: u32,
    /// First mip level packed into the mip tail.
    pub first_mip_in_tail: u32,
    /// One layout per requested mip level.
    pub mips: Vec<MipLayout>,
}

/// Surface layout query failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// The request has invalid dimensions, format size, or mip count.
    InvalidRequest,
    /// The device address configuration offers no supported surface layout.
    NoLayout,
}

/// C ABI request passed to the local `AddrLib3` bridge.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawInput {
    family: u32,
    gb_addr_config: u32,
    bits_per_pixel: u32,
    width: u32,
    height: u32,
    slices: u32,
    mip_levels: u32,
    resource_type: u32,
}

/// C ABI result for one mip level from `AddrLib3`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawMip {
    offset: u64,
    pitch: u32,
    height: u32,
    depth: u32,
}

/// Fixed-capacity C ABI result returned by the local `AddrLib3` bridge.
#[repr(C)]
struct RawOutput {
    size: u64,
    slice_size: u64,
    alignment: u32,
    pitch: u32,
    height: u32,
    swizzle_mode: u32,
    first_mip_in_tail: u32,
    mips: [RawMip; MAX_MIP_LEVELS],
}

impl Default for RawOutput {
    fn default() -> Self {
        Self {
            size: 0,
            slice_size: 0,
            alignment: 0,
            pitch: 0,
            height: 0,
            swizzle_mode: 0,
            first_mip_in_tail: 0,
            mips: [RawMip::default(); MAX_MIP_LEVELS],
        }
    }
}

unsafe extern "C" {
    fn rocaddr3_compute(request: *const RawInput, result: *mut RawOutput) -> i32;
    fn rocaddr3_mapper_create(request: *const RawInput) -> *mut c_void;
    fn rocaddr3_mapper_destroy(mapper: *mut c_void);
    fn rocaddr3_mapper_map_row(
        mapper: *mut c_void,
        request: *const RawInput,
        swizzle_mode: u32,
        mip: u32,
        x: u32,
        y: u32,
        slice: u32,
        count: u32,
        offsets: *mut u64,
    ) -> i32;
}

fn valid_request(request: SurfaceRequest) -> bool {
    request.width != 0
        && request.height != 0
        && request.slices != 0
        && (1..=MAX_MIP_LEVELS_U32).contains(&request.mip_levels)
        && matches!(request.bits_per_pixel, 8 | 16 | 32 | 64 | 128)
}

impl From<SurfaceRequest> for RawInput {
    fn from(request: SurfaceRequest) -> Self {
        Self {
            family: request.family,
            gb_addr_config: request.gb_addr_config,
            bits_per_pixel: request.bits_per_pixel,
            width: request.width,
            height: request.height,
            slices: request.slices,
            mip_levels: request.mip_levels,
            resource_type: request.resource_type as u32,
        }
    }
}

/// Computes a native GFX12 texture layout using the device's address geometry.
///
/// # Errors
/// Returns `InvalidRequest` for invalid dimensions or format size and
/// `NoLayout` when `AddrLib3` cannot select a supported swizzle and extent.
pub fn compute(request: SurfaceRequest) -> Result<SurfaceLayout, Error> {
    if !valid_request(request) {
        return Err(Error::InvalidRequest);
    }
    let raw = RawInput::from(request);
    let mut result = RawOutput::default();
    // SAFETY: Both pointers refer to initialized, uniquely owned fixed-size
    // records. The bridge writes only the output record before returning.
    if unsafe { rocaddr3_compute(&raw const raw, &raw mut result) } != 1 {
        return Err(Error::NoLayout);
    }
    if result.size == 0 || !result.alignment.is_power_of_two() {
        return Err(Error::NoLayout);
    }
    let mips = result.mips[..request.mip_levels as usize]
        .iter()
        .map(|mip| MipLayout {
            offset: mip.offset,
            pitch: mip.pitch,
            height: mip.height,
            depth: mip.depth,
        })
        .collect();
    Ok(SurfaceLayout {
        size: result.size,
        slice_size: result.slice_size,
        alignment: result.alignment,
        pitch: result.pitch,
        height: result.height,
        swizzle_mode: result.swizzle_mode,
        first_mip_in_tail: result.first_mip_in_tail,
        mips,
    })
}

/// One native `AddrLib` mapper retained across rows of a surface transfer.
/// The opaque handle is confined to its owner and released on drop.
pub struct SurfaceMapper {
    /// Native `AddrLib` handle owned by this mapper.
    handle: NonNull<c_void>,
    /// Dimensions and hardware geometry used to create the handle.
    request: RawInput,
    /// Selected swizzle mode from the corresponding surface layout.
    swizzle_mode: u32,
    /// Complete byte extent used to validate returned offsets.
    layout_size: u64,
}

impl SurfaceMapper {
    /// Creates a mapper for a layout computed from the same request.
    ///
    /// # Errors
    /// Returns `InvalidRequest` for mismatched layout metadata and `NoLayout`
    /// when `AddrLib` cannot initialize for this device geometry.
    pub fn new(request: SurfaceRequest, layout: &SurfaceLayout) -> Result<Self, Error> {
        if !valid_request(request)
            || layout.mips.len() != request.mip_levels as usize
            || layout.swizzle_mode >= 8
        {
            return Err(Error::InvalidRequest);
        }
        let raw = RawInput::from(request);
        // SAFETY: The bridge reads one initialized request and returns an
        // independently owned handle or null.
        let handle = NonNull::new(unsafe { rocaddr3_mapper_create(&raw const raw) })
            .ok_or(Error::NoLayout)?;
        Ok(Self {
            handle,
            request: raw,
            swizzle_mode: layout.swizzle_mode,
            layout_size: layout.size,
        })
    }

    /// Maps one row to byte offsets in the complete surface backing.
    /// `slice` selects a 3D depth plane or an array layer.
    ///
    /// # Errors
    /// Returns `InvalidRequest` for an empty row or invalid mip and `NoLayout`
    /// if `AddrLib` cannot map a coordinate inside the surface extent.
    pub fn map_row(
        &mut self,
        mip: u32,
        x: u32,
        y: u32,
        slice: u32,
        offsets: &mut [u64],
    ) -> Result<(), Error> {
        if mip >= self.request.mip_levels || offsets.is_empty() {
            return Err(Error::InvalidRequest);
        }
        let count = u32::try_from(offsets.len()).map_err(|_| Error::InvalidRequest)?;
        // SAFETY: This mapper owns the live handle and the bridge writes at
        // most `count` offsets into the uniquely borrowed slice.
        if unsafe {
            rocaddr3_mapper_map_row(
                self.handle.as_ptr(),
                &raw const self.request,
                self.swizzle_mode,
                mip,
                x,
                y,
                slice,
                count,
                offsets.as_mut_ptr(),
            )
        } != 1
        {
            return Err(Error::NoLayout);
        }
        let element_bytes = u64::from(self.request.bits_per_pixel / 8);
        if offsets.iter().any(|&offset| {
            offset
                .checked_add(element_bytes)
                .is_none_or(|end| end > self.layout_size)
        }) {
            return Err(Error::NoLayout);
        }
        Ok(())
    }
}

impl Drop for SurfaceMapper {
    fn drop(&mut self) {
        // SAFETY: This owner releases its unique handle exactly once.
        unsafe { rocaddr3_mapper_destroy(self.handle.as_ptr()) };
    }
}
