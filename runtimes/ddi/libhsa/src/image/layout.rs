// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA image layout requests and generation-bound native layout work.

use super::*;

/// Mip layout tied to the generation that supplied the device address configuration.
pub(super) struct NativeMipLayout {
    /// Register generation for this native layout.
    pub(super) architecture: Generation,
    /// Format encoded in the resource descriptor.
    pub(super) format: Format,
    /// Checked pitches and total backing extent.
    pub(super) surface: Layout,
    /// Device geometry passed to `AddrLib`.
    pub(super) request: SurfaceRequest,
    /// Computed native offsets and swizzle mode.
    pub(super) native: SurfaceLayout,
    /// Runtime generation that owns the queried device.
    pub(super) generation: u64,
    /// Keeps that generation's device state alive through publication.
    pub(super) _inflight: InFlightToken,
}

/// Computes the image capability flags for a format and geometry.
pub(crate) fn image_capability(
    runtime: &HsaState,
    agent: HsaAgent,
    geometry: u32,
    format: HsaExtImageFormat,
) -> Result<u32, Status> {
    image_gpu_index(runtime, agent)?;
    if geometry > 7 {
        return Err(INVALID_ARGUMENT);
    }
    let access = descriptor::access_capabilities(geometry, format);
    if access == 0 {
        return Ok(0);
    }
    Ok(access | IMAGE_CAPABILITY_INVARIANT_LAYOUT)
}

/// Computes the native image layout requested by an HSA descriptor.
pub(super) fn image_layout(
    runtime: &HsaState,
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptor,
    permission: u32,
    layout: u32,
    row_pitch: usize,
    slice_pitch: usize,
) -> Result<(usize, Format, Layout), Status> {
    let index = image_gpu_index(runtime, agent)?;
    if !matches!(
        permission,
        ACCESS_PERMISSION_RO | ACCESS_PERMISSION_WO | ACCESS_PERMISSION_RW
    ) || !matches!(layout, IMAGE_LAYOUT_OPAQUE | IMAGE_LAYOUT_LINEAR)
    {
        return Err(INVALID_ARGUMENT);
    }
    let access = descriptor::access_capabilities(descriptor.geometry, descriptor.format);
    let required = match permission {
        ACCESS_PERMISSION_RO => IMAGE_CAPABILITY_READ_ONLY,
        ACCESS_PERMISSION_WO => IMAGE_CAPABILITY_WRITE_ONLY,
        _ => IMAGE_CAPABILITY_READ_WRITE,
    };
    if access & required == 0 {
        return Err(IMAGE_FORMAT_UNSUPPORTED);
    }
    let format = descriptor::format(descriptor.format).ok_or(IMAGE_FORMAT_UNSUPPORTED)?;
    let info = runtime.gpus[index].info();
    let architecture = Generation::from_gpu(&info).ok_or(NOT_SUPPORTED)?;
    let mut surface = if architecture == Generation::Gfx12 {
        descriptor::layout(descriptor, format, row_pitch, slice_pitch)?
    } else {
        descriptor::layout(descriptor, format, 0, 0)?
    };
    if architecture != Generation::Gfx12 && descriptor.geometry != 5 {
        let config = runtime.gpus[index]
            .device
            .gpu()
            .and_then(|gpu| gpu.surface_config())
            .map_err(map_error)?;
        let request = SurfaceRequest {
            family: config.asic_family_id,
            chip_revision: config.chip_revision,
            gb_addr_config: config.gb_addr_config,
            bits_per_pixel: (format.bytes * 8) as u32,
            width: surface.width as u32,
            height: surface.height as u32,
            slices: surface.slices as u32,
            mip_levels: 1,
            resource_type: match descriptor.geometry {
                0 | 3 => ResourceType::Texture1D,
                2 => ResourceType::Texture3D,
                _ => ResourceType::Texture2D,
            },
        };
        let explicit_pitch = architecture == Generation::Gfx9
            || (architecture != Generation::Gfx10 && matches!(descriptor.geometry, 0 | 1 | 6));
        if row_pitch % format.bytes != 0 {
            return Err(IMAGE_PITCH_UNSUPPORTED);
        }
        let pitch = if explicit_pitch {
            u32::try_from(row_pitch / format.bytes).map_err(|_| IMAGE_PITCH_UNSUPPORTED)?
        } else {
            0
        };
        let computed =
            rocaddr3::compute_linear(request, pitch).map_err(|_| IMAGE_PITCH_UNSUPPORTED)?;
        surface.row_pitch = computed.pitch as usize * format.bytes;
        surface.slice_pitch =
            usize::try_from(computed.slice_size).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
        surface.size = usize::try_from(computed.size).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
        if (row_pitch != 0 && row_pitch != surface.row_pitch)
            || (slice_pitch != 0 && slice_pitch != surface.slice_pitch)
        {
            return Err(IMAGE_PITCH_UNSUPPORTED);
        }
    }
    // Size queries and creation must agree on representability of pitch fields.
    descriptor::descriptor(architecture, descriptor, format, surface, 0x10000)?;
    Ok((index, format, surface))
}

/// Limits the mip count to the surface dimensions and descriptor.
pub(super) fn maximum_mip_levels(descriptor: HsaExtImageDescriptor, surface: Layout) -> usize {
    let maximum_dimension = match descriptor.geometry {
        0 | 3 => surface.width,
        1 | 4 | 6 | 7 => surface.width.max(surface.height),
        2 => surface.width.max(surface.height).max(surface.slices),
        _ => 0,
    };
    (usize::BITS - maximum_dimension.leading_zeros()) as usize
}

/// Builds a native layout for the requested mip level chain.
pub(super) fn native_mip_layout(
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptorV2,
    permission: u32,
) -> Result<NativeMipLayout, Status> {
    let simple = descriptor.into();
    let (format, mut surface, device, generation, inflight) = {
        let guard = lock()?;
        let generation = guard.active_generation().ok_or(NOT_INITIALIZED)?;
        let runtime = guard.as_ref().ok_or(NOT_INITIALIZED)?;
        let (index, format, surface) = image_layout(
            runtime,
            agent,
            simple,
            permission,
            IMAGE_LAYOUT_OPAQUE,
            0,
            0,
        )?;
        let token = runtime.inflight.enter().ok_or(OUT_OF_RESOURCES)?;
        (
            format,
            surface,
            runtime.gpus[index].device.clone(),
            generation,
            token,
        )
    };
    if descriptor.mipmap_levels < 2
        || descriptor.mipmap_levels > maximum_mip_levels(simple, surface)
        || descriptor.geometry == 5
    {
        return Err(IMAGE_SIZE_UNSUPPORTED);
    }
    let config = device
        .gpu()
        .and_then(|gpu| gpu.surface_config())
        .map_err(map_error)?;
    let resource_type = match descriptor.geometry {
        0 | 3 => ResourceType::Texture1D,
        1 | 4 | 6 | 7 => ResourceType::Texture2D,
        2 => ResourceType::Texture3D,
        _ => return Err(IMAGE_SIZE_UNSUPPORTED),
    };
    let request = SurfaceRequest {
        family: config.asic_family_id,
        chip_revision: config.chip_revision,
        gb_addr_config: config.gb_addr_config,
        bits_per_pixel: u32::try_from(format.bytes.checked_mul(8).ok_or(IMAGE_SIZE_UNSUPPORTED)?)
            .map_err(|_| IMAGE_SIZE_UNSUPPORTED)?,
        width: u32::try_from(surface.width).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?,
        height: u32::try_from(surface.height).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?,
        slices: u32::try_from(surface.slices).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?,
        mip_levels: u32::try_from(descriptor.mipmap_levels).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?,
        resource_type,
    };
    let native = rocaddr3::compute(request).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
    surface.row_pitch = usize::try_from(native.pitch)
        .ok()
        .and_then(|pitch| pitch.checked_mul(format.bytes))
        .ok_or(IMAGE_SIZE_UNSUPPORTED)?;
    surface.slice_pitch = usize::try_from(native.slice_size).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
    surface.size = usize::try_from(native.size)
        .ok()
        .filter(|size| isize::try_from(*size).is_ok())
        .ok_or(IMAGE_SIZE_UNSUPPORTED)?;
    Ok(NativeMipLayout {
        architecture: Generation::from_gpu(device.gpu().map_err(map_error)?.info())
            .ok_or(NOT_SUPPORTED)?,
        format,
        surface,
        request,
        native,
        generation,
        _inflight: inflight,
    })
}

/// Reports the allocation requirements of a native mip chain.
pub(crate) fn native_mip_data_info(
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptorV2,
    permission: u32,
) -> Result<HsaExtImageDataInfo, Status> {
    let layout = native_mip_layout(agent, descriptor, permission)?;
    Ok(HsaExtImageDataInfo {
        size: layout.surface.size,
        alignment: layout.native.alignment as usize,
    })
}

#[allow(unsafe_code)]
/// Calculates image storage requirements and writes the result to the caller.
pub(crate) unsafe fn get_info(
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptor,
    permission: u32,
    layout: u32,
    row_pitch: usize,
    slice_pitch: usize,
    output: *mut HsaExtImageDataInfo,
) -> Status {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(runtime) = guard.as_ref() else {
        return NOT_INITIALIZED;
    };
    let (_, _, surface) = match image_layout(
        runtime,
        agent,
        descriptor,
        permission,
        layout,
        row_pitch,
        slice_pitch,
    ) {
        Ok(result) => result,
        Err(status) => return status,
    };
    // SAFETY: The caller supplied writable output storage for one info record.
    unsafe { output.write(descriptor::info(surface)) };
    SUCCESS
}
