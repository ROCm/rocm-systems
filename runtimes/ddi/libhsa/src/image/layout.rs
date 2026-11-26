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
    runtime: &HsaSession,
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
    runtime: &HsaSession,
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
    let info = runtime.agents.gpus[index].info();
    let architecture = Generation::from_gpu(&info).ok_or(NOT_SUPPORTED)?;
    let mut surface = descriptor::layout(descriptor, format, 0, 0)?;
    if descriptor.geometry != 5 {
        let config = runtime.agents.gpus[index]
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
        surface = linear_surface_layout(
            architecture,
            descriptor.geometry,
            format,
            request,
            row_pitch,
            slice_pitch,
        )?;
    }
    // Size queries and creation must agree on representability of pitch fields.
    descriptor::descriptor(architecture, descriptor, format, surface, 0x10000)?;
    Ok((index, format, surface))
}

/// Computes representable linear strides while preserving native allocation padding.
fn linear_surface_layout(
    architecture: Generation,
    geometry: u32,
    format: Format,
    request: SurfaceRequest,
    row_pitch: usize,
    slice_pitch: usize,
) -> Result<Layout, Status> {
    // GFX10 cannot encode a custom pitch; GFX11/12 can encode it only for
    // ordinary 1D/2D images. Array and 3D descriptors use that field for depth.
    let explicit_pitch = architecture == Generation::Gfx9
        || (architecture != Generation::Gfx10 && matches!(geometry, 0 | 1 | 6));
    if row_pitch % format.bytes != 0 {
        return Err(IMAGE_PITCH_UNSUPPORTED);
    }
    let pitch = if explicit_pitch {
        u32::try_from(row_pitch / format.bytes).map_err(|_| IMAGE_PITCH_UNSUPPORTED)?
    } else {
        0
    };
    let computed = rocaddr3::compute_linear(request, pitch).map_err(|_| IMAGE_PITCH_UNSUPPORTED)?;
    let computed_row = computed.pitch as usize * format.bytes;
    // AddrLib trims trailing padding for a single non-3D slice. HSA's default
    // slice pitch is the row extent there, even when the allocation is padded.
    let computed_slice =
        if architecture == Generation::Gfx12 && geometry != 2 && request.slices == 1 {
            computed_row
                .checked_mul(computed.height as usize)
                .ok_or(IMAGE_SIZE_UNSUPPORTED)?
        } else {
            usize::try_from(computed.slice_size).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?
        };
    if (row_pitch != 0 && row_pitch != computed_row)
        || (slice_pitch != 0 && slice_pitch != computed_slice)
    {
        return Err(IMAGE_PITCH_UNSUPPORTED);
    }
    Ok(Layout {
        width: request.width as usize,
        height: request.height as usize,
        slices: request.slices as usize,
        row_pitch: computed_row,
        slice_pitch: computed_slice,
        size: usize::try_from(computed.size).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?,
    })
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
        let token = runtime.control.inflight.enter().ok_or(OUT_OF_RESOURCES)?;
        (
            format,
            surface,
            runtime.agents.gpus[index].device.clone(),
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Builds native geometry for the fixed GFX12 host test device.
    fn request(descriptor: HsaExtImageDescriptor, format: Format) -> SurfaceRequest {
        SurfaceRequest {
            family: 0x98,
            chip_revision: 1,
            gb_addr_config: 0,
            bits_per_pixel: (format.bytes * 8) as u32,
            width: descriptor.width as u32,
            height: descriptor.height.max(1) as u32,
            slices: descriptor.array_size.max(descriptor.depth).max(1) as u32,
            mip_levels: 1,
            resource_type: match descriptor.geometry {
                0 | 3 => ResourceType::Texture1D,
                2 => ResourceType::Texture3D,
                _ => ResourceType::Texture2D,
            },
        }
    }

    #[test]
    fn gfx12_linear_layout_separates_hsa_slice_pitch_from_allocation_padding() {
        for (channel_type, width, row_pitch, slice_pitch, size) in
            [(9, 179, 384, 1152, 1280), (15, 30, 128, 384, 512)]
        {
            let desc = HsaExtImageDescriptor {
                geometry: 1,
                width,
                height: 3,
                depth: 0,
                array_size: 0,
                format: HsaExtImageFormat {
                    channel_order: 1,
                    channel_type,
                },
            };
            let format = descriptor::format(desc.format).unwrap();
            let request = request(desc, format);
            for requested_row in [0, row_pitch] {
                for requested_slice in [0, slice_pitch] {
                    let surface = linear_surface_layout(
                        Generation::Gfx12,
                        desc.geometry,
                        format,
                        request,
                        requested_row,
                        requested_slice,
                    )
                    .unwrap();
                    assert_eq!(surface.row_pitch, row_pitch);
                    assert_eq!(surface.slice_pitch, slice_pitch);
                    assert_eq!(surface.size, size);
                    let words =
                        descriptor::descriptor(Generation::Gfx12, desc, format, surface, 0x10000)
                            .unwrap();
                    assert_eq!(words[4] as usize, row_pitch / format.bytes - 1);
                }
            }
            assert_eq!(
                linear_surface_layout(
                    Generation::Gfx12,
                    desc.geometry,
                    format,
                    request,
                    row_pitch,
                    size
                )
                .err(),
                Some(IMAGE_PITCH_UNSUPPORTED)
            );
        }
    }

    #[test]
    fn gfx12_array_and_3d_layouts_preserve_native_slice_padding() {
        for (geometry, height, depth, array_size, slice_pitch) in
            [(3, 0, 0, 2, 256), (4, 3, 0, 2, 768), (2, 3, 2, 0, 768)]
        {
            let desc = HsaExtImageDescriptor {
                geometry,
                width: 30,
                height,
                depth,
                array_size,
                format: HsaExtImageFormat {
                    channel_order: 1,
                    channel_type: 15,
                },
            };
            let format = descriptor::format(desc.format).unwrap();
            let request = request(desc, format);
            for requested_row in [0, 128] {
                let surface = linear_surface_layout(
                    Generation::Gfx12,
                    geometry,
                    format,
                    request,
                    requested_row,
                    slice_pitch,
                )
                .unwrap();
                assert_eq!(surface.row_pitch, 128);
                assert_eq!(surface.slice_pitch, slice_pitch);
                assert_eq!(surface.size, slice_pitch * 2);
                let words =
                    descriptor::descriptor(Generation::Gfx12, desc, format, surface, 0x10000)
                        .unwrap();
                assert_eq!(words[4], 1);
            }
            // This field holds array/depth extent, so a custom row pitch cannot
            // be represented even when it meets the hardware's alignment.
            assert_eq!(
                linear_surface_layout(
                    Generation::Gfx12,
                    geometry,
                    format,
                    request,
                    256,
                    slice_pitch
                )
                .err(),
                Some(IMAGE_PITCH_UNSUPPORTED)
            );
            assert_eq!(
                linear_surface_layout(
                    Generation::Gfx12,
                    geometry,
                    format,
                    request,
                    128,
                    128 * request.height as usize,
                )
                .err(),
                Some(IMAGE_PITCH_UNSUPPORTED)
            );
        }
    }
}
