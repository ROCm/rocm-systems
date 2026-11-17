// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Image instructions preserve producer tiling and compression during imports.

use super::*;
use crate::dispatch::Dispatch;

/// Compiled source is reproducible with the adjacent `generate_transfer.py`.
const GFX1200: &[u8] = include_bytes!("image_transfer_gfx1200.hsaco");
/// GFX1201 requires its own code-object machine identifier.
const GFX1201: &[u8] = include_bytes!("image_transfer_gfx1201.hsaco");

/// Registry lease for the caller's complete image allocation or reservation.
struct BackingLease {
    /// Runtime generation in which the borrow was acquired.
    generation: u64,
    /// Canonical allocation or VM reservation base.
    base: usize,
}

impl Drop for BackingLease {
    fn drop(&mut self) {
        if let Ok(mut guard) = lock() {
            if guard.active_generation() == Some(self.generation) {
                if let Some(runtime) = guard.as_mut() {
                    runtime.memory_state.release_copy_borrow(self.base);
                }
            }
        }
    }
}

/// All storage reachable by an image-transfer packet.
struct TransferOwners {
    /// Original public descriptor must survive concurrent image destruction.
    _snapshot: ImageSnapshot,
    /// Registry owner prevents the image allocation from being freed or unmapped.
    _backing: BackingLease,
    /// Unsigned view used for lossless byte transfers through image instructions.
    _descriptor: Allocation,
    /// Bounded host-visible linear staging buffer.
    staging: Allocation,
    /// Device address of the unsigned image descriptor.
    descriptor_address: u64,
}

/// Unsigned format retaining the hardware component widths and stored ordering.
fn raw_format(format: HsaExtImageFormat) -> Result<(HsaExtImageFormat, u32, u32), Status> {
    let (order, channel_type, channels) = match format.channel_type {
        5 | 6 => (1, 12, 1),
        7 => (1, 13, 1),
        channel_type => {
            let (order, channels) = match format.channel_order {
                0 | 1 | 16..=18 => (1, 1),
                3 | 5 => (3, 2),
                8..=10 | 14 => (8, 4),
                _ => return Err(IMAGE_FORMAT_UNSUPPORTED),
            };
            let raw_type = match channel_type {
                0 | 2 | 8 | 11 => 11,
                1 | 3 | 9 | 12 | 14 => 12,
                10 | 13 | 15 => 13,
                _ => return Err(IMAGE_FORMAT_UNSUPPORTED),
            };
            (order, raw_type, channels)
        }
    };
    Ok((
        HsaExtImageFormat {
            channel_type,
            channel_order: order,
        },
        channels,
        1 << (channel_type - 11),
    ))
}

/// Creates a private unsigned resource view without changing its tiling fields.
fn raw_words(snapshot: &ImageSnapshot) -> Result<([u32; 12], u32, u32), Status> {
    let (raw, channels, component_bytes) = raw_format(snapshot.descriptor.format)?;
    let format = image_gfx12::format(raw).ok_or(IMAGE_FORMAT_UNSUPPORTED)?;
    if channels as usize * component_bytes as usize != snapshot.format.bytes {
        return Err(IMAGE_FORMAT_UNSUPPORTED);
    }
    let mut descriptor = snapshot.descriptor;
    descriptor.format = raw;
    let level = rocddi::device::gpu::image::base_mip_level(
        snapshot.record.architecture,
        &snapshot.record.words,
    );
    Ok((
        image_gfx12::mip_level_view(
            snapshot.record.architecture,
            snapshot.record.words,
            level,
            descriptor,
            format,
        )?,
        channels,
        component_bytes,
    ))
}

/// Creates a queue whose retained operands cover the complete imported image.
#[allow(unsafe_code)]
fn prepare(snapshot: &ImageSnapshot, import: bool) -> Result<Dispatch<TransferOwners>, Status> {
    let (words, _, _) = raw_words(snapshot)?;
    let (device, page, object, backing) = {
        let mut guard = lock()?;
        let generation = guard.active_generation().ok_or(NOT_INITIALIZED)?;
        let runtime = initialized_mut(&mut guard)?;
        let index = image_gpu_index(runtime, snapshot.agent)?;
        let info = runtime.gpus[index].info();
        let object: &[u8] = match (info.gfx_major, info.gfx_minor, info.gfx_stepping) {
            (9, 0, 8) => include_bytes!("image_transfer_gfx908.hsaco"),
            (9, 0, 10) => include_bytes!("image_transfer_gfx90a.hsaco"),
            (9, 4 | 5, _) => include_bytes!("image_transfer_gfx9-4-generic.hsaco"),
            (10, 1, _) => include_bytes!("image_transfer_gfx10-1-generic.hsaco"),
            (10, 3, _) => include_bytes!("image_transfer_gfx10-3-generic.hsaco"),
            (11, _, _) => include_bytes!("image_transfer_gfx11-generic.hsaco"),
            (12, 0, 0) => GFX1200,
            (12, 0, 1) => GFX1201,
            _ => return Err(NOT_SUPPORTED),
        };
        // The importer already validated the SRD's layout. Borrow its containing
        // owner; compressed byte extent is driver-owned and not inferred here.
        let (_, base) = crate::memory::image_copy_borrow(
            runtime,
            usize::try_from(snapshot.data).map_err(|_| INVALID_ALLOCATION)?,
            1,
            snapshot.agent,
            if import {
                ACCESS_PERMISSION_WO
            } else {
                ACCESS_PERMISSION_RO
            },
        )?;
        runtime.memory_state.retain_copy_borrow(base)?;
        (
            runtime.gpus[index].device.clone(),
            runtime.host_page_size as u64,
            object,
            BackingLease { generation, base },
        )
    };
    let descriptor = device
        .allocate(MemoryKind::System, page, page, DeviceAccess::READ)
        .map_err(map_error)?;
    let staging = device
        .allocate(
            MemoryKind::System,
            TILED_TRANSFER_CHUNK_BYTES as u64,
            page,
            DeviceAccess::READ | DeviceAccess::WRITE,
        )
        .map_err(map_error)?;
    let host = descriptor.info().host_address.ok_or(NOT_SUPPORTED)?;
    if staging.info().host_address.is_none() {
        return Err(NOT_SUPPORTED);
    }
    let descriptor_address = descriptor.device_address(&device).map_err(map_error)?;
    // SAFETY: The owned descriptor allocation contains a writable page.
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), host as *mut u32, words.len()) };
    Dispatch::new(
        snapshot.agent,
        Arc::from(object),
        TransferOwners {
            _snapshot: snapshot.clone(),
            _backing: backing,
            _descriptor: descriptor,
            staging,
            descriptor_address,
        },
    )
}

/// Encodes the explicit `OpenCL` arguments at their metadata-defined offsets.
fn arguments(
    buffer: u64,
    image: u64,
    origin: [u32; 3],
    extent: [u32; 3],
    geometry: u32,
    channels: u32,
    component_bytes: u32,
) -> [u8; 92] {
    let mut bytes = [0; 92];
    bytes[..8].copy_from_slice(&buffer.to_le_bytes());
    for offset in (8..48).step_by(8) {
        bytes[offset..offset + 8].copy_from_slice(&image.to_le_bytes());
    }
    for (offset, coordinates) in [(48, origin), (64, extent)] {
        for (axis, value) in coordinates.into_iter().enumerate() {
            bytes[offset + axis * 4..offset + axis * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    for (offset, value) in [(80, geometry), (84, channels), (88, component_bytes)] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Destination for exported bytes, or source of an import or clear operation.
#[derive(Clone, Copy)]
enum Transfer<'a> {
    /// Caller-owned rows to read or write.
    Rows {
        /// Base of caller-owned host or GPU storage.
        address: usize,
        /// Bytes between consecutive external rows.
        row_pitch: usize,
        /// Bytes between consecutive external slices.
        slice_pitch: usize,
        /// Whether bytes flow from external storage to the image.
        import: bool,
    },
    /// One encoded pixel repeated across the image region.
    Pixel(&'a [u8]),
}

/// Performs bounded transfers while one private queue owns the native resources.
#[allow(unsafe_code)]
unsafe fn run(snapshot: &ImageSnapshot, region: Region, transfer: Transfer<'_>) -> Status {
    let import = !matches!(transfer, Transfer::Rows { import: false, .. });
    let (_, channels, component_bytes) = match raw_words(snapshot) {
        Ok(format) => format,
        Err(status) => return status,
    };
    let row_bytes = region.range[0] * snapshot.format.bytes;
    if row_bytes > TILED_TRANSFER_CHUNK_BYTES {
        return IMAGE_SIZE_UNSUPPORTED;
    }
    let rows_per_chunk = (TILED_TRANSFER_CHUNK_BYTES / row_bytes).max(1);
    let mut dispatch = match prepare(snapshot, import) {
        Ok(dispatch) => dispatch,
        Err(status) => return status,
    };
    let status = (|| {
        for z in 0..region.range[2] {
            for y in (0..region.range[1]).step_by(rows_per_chunk) {
                let rows = (region.range[1] - y).min(rows_per_chunk);
                let owners = dispatch.dependencies().ok_or(ERROR)?;
                let host = owners.staging.info().host_address.ok_or(ERROR)?;
                let args = arguments(
                    owners.staging.info().device_address,
                    owners.descriptor_address,
                    [
                        region.offset[0] as u32,
                        (region.offset[1] + y) as u32,
                        (region.offset[2] + z) as u32,
                    ],
                    [region.range[0] as u32, rows as u32, 1],
                    snapshot.descriptor.geometry,
                    channels,
                    component_bytes,
                );
                if import {
                    match transfer {
                        Transfer::Pixel(pixel) => {
                            // SAFETY: The chunk fits in the owned writable staging mapping.
                            let bytes = unsafe {
                                std::slice::from_raw_parts_mut(host as *mut u8, rows * row_bytes)
                            };
                            for element in bytes.chunks_exact_mut(pixel.len()) {
                                element.copy_from_slice(pixel);
                            }
                        }
                        Transfer::Rows {
                            address,
                            row_pitch,
                            slice_pitch,
                            ..
                        } => {
                            for row in 0..rows {
                                let offset = row_offset(z, y + row, row_pitch, slice_pitch)
                                    .ok_or(INVALID_ARGUMENT)?;
                                let source = address.checked_add(offset).ok_or(INVALID_ARGUMENT)?;
                                // SAFETY: Both rows are live, and HSA validates registered ranges.
                                let status = unsafe {
                                    crate::memory::hsa_memory_copy(
                                        (host + row * row_bytes) as *mut c_void,
                                        source as *const c_void,
                                        row_bytes,
                                    )
                                };
                                if status != SUCCESS {
                                    return Err(status);
                                }
                            }
                        }
                    }
                }
                // SAFETY: Dispatch retains the SRD, image backing and staging.
                let status = unsafe {
                    dispatch.launch(
                        if import {
                            "image_write_bytes"
                        } else {
                            "image_read_bytes"
                        },
                        &args,
                        (rows * region.range[0]) as u32,
                    )
                };
                if status != SUCCESS {
                    return Err(status);
                }
                if let Transfer::Rows {
                    address,
                    row_pitch,
                    slice_pitch,
                    import: false,
                } = transfer
                {
                    for row in 0..rows {
                        let offset = row_offset(z, y + row, row_pitch, slice_pitch)
                            .ok_or(INVALID_ARGUMENT)?;
                        let destination = address.checked_add(offset).ok_or(INVALID_ARGUMENT)?;
                        // SAFETY: Kernel completion made the staging bytes visible to the CPU.
                        let status = unsafe {
                            crate::memory::hsa_memory_copy(
                                destination as *mut c_void,
                                (host + row * row_bytes) as *const c_void,
                                row_bytes,
                            )
                        };
                        if status != SUCCESS {
                            return Err(status);
                        }
                    }
                }
            }
        }
        Ok(())
    })()
    .err()
    .unwrap_or(SUCCESS);
    dispatch.finish(status).status
}

/// Copies imported image storage through typed image load/store instructions.
#[allow(unsafe_code)]
pub(super) unsafe fn transfer(
    snapshot: &ImageSnapshot,
    region: Region,
    address: usize,
    row_pitch: usize,
    slice_pitch: usize,
    import: bool,
) -> Status {
    // SAFETY: The caller owns both external rows and the image during the call.
    unsafe {
        run(
            snapshot,
            region,
            Transfer::Rows {
                address,
                row_pitch,
                slice_pitch,
                import,
            },
        )
    }
}

/// Clears an imported image using a previously encoded storage pixel.
#[allow(unsafe_code)]
pub(super) unsafe fn clear(snapshot: &ImageSnapshot, region: Region, pixel: &[u8]) -> Status {
    if pixel.len() != snapshot.format.bytes {
        return INVALID_ARGUMENT;
    }
    // SAFETY: The caller retains the validated pattern and image through run.
    unsafe { run(snapshot, region, Transfer::Pixel(pixel)) }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn transfer_objects_have_expected_kernel_contracts() {
        let objects: &[(&[u8], &str, u32)] = &[
            (include_bytes!("image_transfer_gfx908.hsaco"), "gfx908", 64),
            (include_bytes!("image_transfer_gfx90a.hsaco"), "gfx90a", 64),
            (
                include_bytes!("image_transfer_gfx9-4-generic.hsaco"),
                "gfx9-4-generic",
                64,
            ),
            (
                include_bytes!("image_transfer_gfx10-1-generic.hsaco"),
                "gfx10-1-generic",
                32,
            ),
            (
                include_bytes!("image_transfer_gfx10-3-generic.hsaco"),
                "gfx10-3-generic",
                32,
            ),
            (
                include_bytes!("image_transfer_gfx11-generic.hsaco"),
                "gfx11-generic",
                32,
            ),
            (GFX1200, "gfx1200", 32),
            (GFX1201, "gfx1201", 32),
        ];
        for &(bytes, target, wave) in objects {
            let object = rocloader::CodeObject::parse(bytes).unwrap();
            assert_eq!(object.target().unwrap().0, target);
            let symbols = object
                .symbols(rocloader::SymbolTableOrder::DynamicFirst)
                .unwrap();
            assert!(symbols.iter().all(|symbol| symbol.is_definition));
            for name in ["image_read_bytes", "image_write_bytes"] {
                let symbol = symbols
                    .iter()
                    .find(|symbol| symbol.name.strip_suffix(".kd").unwrap_or(&symbol.name) == name)
                    .unwrap();
                assert_eq!(symbol.kind, rocloader::SymbolKind::Kernel);
                assert_eq!(symbol.kernarg_size, 92);
                assert_eq!(symbol.kernarg_alignment, 16);
                assert_eq!(symbol.group_size, 0);
                assert_eq!(symbol.private_size, 0);
                assert_eq!(symbol.wavefront_size, wave);
            }
            let plan = rocelf::LoadPlan::new(bytes, object.layout(), 4096).unwrap();
            let mut image = vec![0; plan.allocation_size()];
            plan.write_amdgpu_image(0x10000, &mut image, |_: &str| None)
                .unwrap();
        }
    }

    #[test]
    fn transfer_arguments_preserve_image_vector_alignment() {
        let args = arguments(0x1000, 0x2000, [3, 5, 7], [11, 13, 1], 4, 2, 4);
        assert_eq!(rocelf::read_u64(&args, 0), Some(0x1000));
        for offset in (8..48).step_by(8) {
            assert_eq!(rocelf::read_u64(&args, offset), Some(0x2000));
        }
        for (offset, expected) in [
            (48, 3),
            (52, 5),
            (56, 7),
            (60, 0),
            (64, 11),
            (68, 13),
            (72, 1),
            (76, 0),
            (80, 4),
            (84, 2),
            (88, 4),
        ] {
            assert_eq!(rocelf::read_u32(&args, offset), Some(expected));
        }
    }

    #[test]
    fn unsigned_views_keep_compression_component_widths() {
        for (order, channel_type, raw_order, raw_type, channels, bytes) in [
            (8, 2, 8, 11, 4, 1),
            (14, 2, 8, 11, 4, 1),
            (9, 8, 8, 11, 4, 1),
            (5, 14, 3, 12, 2, 2),
            (18, 15, 1, 13, 1, 4),
            (16, 3, 1, 12, 1, 2),
            (6, 5, 1, 12, 1, 2),
            (6, 6, 1, 12, 1, 2),
            (8, 7, 1, 13, 1, 4),
        ] {
            let (format, count, component) = raw_format(HsaExtImageFormat {
                channel_order: order,
                channel_type,
            })
            .unwrap();
            assert_eq!(format.channel_order, raw_order);
            assert_eq!(format.channel_type, raw_type);
            assert_eq!((count, component), (channels, bytes));
        }
    }
}
