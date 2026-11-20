// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Image transfer planning and retained synchronous operations.

use super::color::{clear_pixel, convert_srgb_pixels};
use super::*;

mod gpu;

/// Converts external pitches to the strides of logical y and z coordinates.
fn external_pitches(
    geometry: u32,
    row_bytes: usize,
    rows: usize,
    row_pitch: usize,
    slice_pitch: usize,
) -> Result<(usize, usize), Status> {
    let row_pitch = row_pitch.max(row_bytes);
    if geometry == 3 {
        // A 1DA layer advances y, and its external stride is the slice pitch.
        let layer_pitch = slice_pitch.max(row_pitch);
        return Ok((layer_pitch, layer_pitch));
    }
    let minimum_slice = row_pitch.checked_mul(rows).ok_or(INVALID_ARGUMENT)?;
    let slice_pitch = slice_pitch.max(minimum_slice);
    Ok((row_pitch, slice_pitch))
}

/// Computes a linear image row offset, checking overflow.
fn row_offset(z: usize, y: usize, row_pitch: usize, slice_pitch: usize) -> Option<usize> {
    z.checked_mul(slice_pitch)?
        .checked_add(y.checked_mul(row_pitch)?)
}

/// Tiled transfer chunk size in bytes.
const TILED_TRANSFER_CHUNK_BYTES: usize = 1024 * 1024;

/// Allocates and initializes a staging buffer of the requested size.
fn zeroed_bytes(size: usize) -> Result<Vec<u8>, Status> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| OUT_OF_RESOURCES)?;
    bytes.resize(size, 0);
    Ok(bytes)
}

/// Constructs an address mapper for a tiled image surface.
fn tiled_mapper(snapshot: &ImageSnapshot) -> Result<rocaddr3::SurfaceMapper, Status> {
    let native = snapshot.native.as_ref().ok_or(NOT_SUPPORTED)?;
    rocaddr3::SurfaceMapper::new(native.request, &native.layout).map_err(|_| IMAGE_SIZE_UNSUPPORTED)
}

/// Entries are sorted by physical byte offset so adjacent texels can use one
/// HSA copy. Each entry's second field selects a texel in the row-major chunk.
fn tiled_chunk_addresses(
    mapper: &mut rocaddr3::SurfaceMapper,
    snapshot: &ImageSnapshot,
    region: Region,
    z: usize,
    first_row: usize,
    rows: usize,
) -> Result<Vec<(u64, usize)>, Status> {
    let native = snapshot.native.as_ref().ok_or(NOT_SUPPORTED)?;
    let width = region.range[0];
    let count = width.checked_mul(rows).ok_or(INVALID_ARGUMENT)?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(count)
        .map_err(|_| OUT_OF_RESOURCES)?;
    let mut offsets = Vec::<u64>::new();
    offsets
        .try_reserve_exact(width)
        .map_err(|_| OUT_OF_RESOURCES)?;
    offsets.resize(width, 0);
    for row in 0..rows {
        let logical_y = region.offset[1] + first_row + row;
        let (y, slice) = if snapshot.descriptor.geometry == 3 {
            (0, logical_y)
        } else {
            (logical_y, region.offset[2] + z)
        };
        mapper
            .map_row(
                native.level,
                u32::try_from(region.offset[0]).map_err(|_| INVALID_ARGUMENT)?,
                u32::try_from(y).map_err(|_| INVALID_ARGUMENT)?,
                u32::try_from(slice).map_err(|_| INVALID_ARGUMENT)?,
                &mut offsets,
            )
            .map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
        for (x, &offset) in offsets.iter().enumerate() {
            entries.push((offset, row * width + x));
        }
    }
    entries.sort_unstable_by_key(|entry| entry.0);
    if entries.windows(2).any(|pair| {
        pair[0]
            .0
            .checked_add(snapshot.format.bytes as u64)
            .is_none_or(|end| end > pair[1].0)
    }) {
        return Err(ERROR);
    }
    Ok(entries)
}

/// Finds the next contiguous run of tiled element addresses.
fn next_tiled_run(entries: &[(u64, usize)], first: usize, element_bytes: usize) -> usize {
    let mut end = first + 1;
    while end < entries.len()
        && entries[end - 1].0.checked_add(element_bytes as u64) == Some(entries[end].0)
    {
        end += 1;
    }
    end
}

/// Source for writing one tiled image region.
#[derive(Clone, Copy)]
enum TiledSource<'a> {
    /// Strided caller-owned host rows and slices.
    Rows {
        /// Base address of caller-owned source pixels.
        address: usize,
        /// Byte stride between adjacent rows of the tiled source.
        row_pitch: usize,
        /// Byte stride between adjacent slices of the tiled source.
        slice_pitch: usize,
    },
    /// One pixel repeated across the region.
    Pixel(&'a [u8]),
}

#[allow(unsafe_code)]
/// Writes tiled rows to the validated destination.
unsafe fn write_tiled_rows(
    snapshot: &ImageSnapshot,
    region: Region,
    source: TiledSource<'_>,
) -> Status {
    let element_bytes = snapshot.format.bytes;
    let Some(row_bytes) = region.range[0].checked_mul(element_bytes) else {
        return INVALID_ARGUMENT;
    };
    if matches!(source, TiledSource::Pixel(pixel) if pixel.len() != element_bytes) {
        return INVALID_ARGUMENT;
    }
    let chunk_rows = (TILED_TRANSFER_CHUNK_BYTES / row_bytes).max(1);
    let Ok(base) = usize::try_from(snapshot.data) else {
        return INVALID_ALLOCATION;
    };
    let mut mapper = match tiled_mapper(snapshot) {
        Ok(mapper) => mapper,
        Err(status) => return status,
    };
    for z in 0..region.range[2] {
        for first_row in (0..region.range[1]).step_by(chunk_rows) {
            let rows = (region.range[1] - first_row).min(chunk_rows);
            let Some(chunk_bytes) = row_bytes.checked_mul(rows) else {
                return INVALID_ARGUMENT;
            };
            let mut chunk = match zeroed_bytes(chunk_bytes) {
                Ok(bytes) => bytes,
                Err(status) => return status,
            };
            match source {
                TiledSource::Rows {
                    address,
                    row_pitch,
                    slice_pitch,
                } => {
                    for row in 0..rows {
                        let Some(offset) = row_offset(z, first_row + row, row_pitch, slice_pitch)
                        else {
                            return INVALID_ARGUMENT;
                        };
                        let Some(source) = address.checked_add(offset) else {
                            return INVALID_ARGUMENT;
                        };
                        // SAFETY: HSA validates registered GPU ranges; the
                        // caller owns each external row for this operation.
                        let status = unsafe {
                            crate::memory::copy(
                                chunk[row * row_bytes..].as_mut_ptr().cast(),
                                source as *const c_void,
                                row_bytes,
                            )
                        };
                        if let Err(status) = status {
                            return status;
                        }
                    }
                }
                TiledSource::Pixel(pixel) => {
                    for element in chunk.chunks_exact_mut(element_bytes) {
                        element.copy_from_slice(pixel);
                    }
                }
            }
            let entries =
                match tiled_chunk_addresses(&mut mapper, snapshot, region, z, first_row, rows) {
                    Ok(entries) => entries,
                    Err(status) => return status,
                };
            let mut run = match zeroed_bytes(chunk_bytes) {
                Ok(bytes) => bytes,
                Err(status) => return status,
            };
            let mut first = 0;
            while first < entries.len() {
                let end = next_tiled_run(&entries, first, element_bytes);
                let bytes = (end - first) * element_bytes;
                for (index, &(_, logical)) in entries[first..end].iter().enumerate() {
                    let source = logical * element_bytes;
                    let target = index * element_bytes;
                    run[target..target + element_bytes]
                        .copy_from_slice(&chunk[source..source + element_bytes]);
                }
                let Ok(offset) = usize::try_from(entries[first].0) else {
                    return INVALID_ALLOCATION;
                };
                let Some(destination) = base.checked_add(offset) else {
                    return INVALID_ALLOCATION;
                };
                // SAFETY: AddrLib mapped every texel inside the caller-owned
                // image backing and the copy source is initialized host data.
                let status = unsafe {
                    crate::memory::copy(destination as *mut c_void, run.as_ptr().cast(), bytes)
                };
                if let Err(status) = status {
                    return status;
                }
                first = end;
            }
        }
    }
    SUCCESS
}

#[allow(unsafe_code)]
/// Reads tiled rows from the supplied source.
unsafe fn read_tiled_rows(
    snapshot: &ImageSnapshot,
    region: Region,
    destination: usize,
    row_pitch: usize,
    slice_pitch: usize,
) -> Status {
    let element_bytes = snapshot.format.bytes;
    let Some(row_bytes) = region.range[0].checked_mul(element_bytes) else {
        return INVALID_ARGUMENT;
    };
    let chunk_rows = (TILED_TRANSFER_CHUNK_BYTES / row_bytes).max(1);
    let Ok(base) = usize::try_from(snapshot.data) else {
        return INVALID_ALLOCATION;
    };
    let mut mapper = match tiled_mapper(snapshot) {
        Ok(mapper) => mapper,
        Err(status) => return status,
    };
    for z in 0..region.range[2] {
        for first_row in (0..region.range[1]).step_by(chunk_rows) {
            let rows = (region.range[1] - first_row).min(chunk_rows);
            let Some(chunk_bytes) = row_bytes.checked_mul(rows) else {
                return INVALID_ARGUMENT;
            };
            let mut chunk = match zeroed_bytes(chunk_bytes) {
                Ok(bytes) => bytes,
                Err(status) => return status,
            };
            let entries =
                match tiled_chunk_addresses(&mut mapper, snapshot, region, z, first_row, rows) {
                    Ok(entries) => entries,
                    Err(status) => return status,
                };
            let mut run = match zeroed_bytes(chunk_bytes) {
                Ok(bytes) => bytes,
                Err(status) => return status,
            };
            let mut first = 0;
            while first < entries.len() {
                let end = next_tiled_run(&entries, first, element_bytes);
                let bytes = (end - first) * element_bytes;
                let Ok(offset) = usize::try_from(entries[first].0) else {
                    return INVALID_ALLOCATION;
                };
                let Some(source) = base.checked_add(offset) else {
                    return INVALID_ALLOCATION;
                };
                // SAFETY: AddrLib mapped this contiguous texel span inside
                // the caller-owned image backing.
                let status = unsafe {
                    crate::memory::copy(run.as_mut_ptr().cast(), source as *const c_void, bytes)
                };
                if let Err(status) = status {
                    return status;
                }
                for (index, &(_, logical)) in entries[first..end].iter().enumerate() {
                    let source = index * element_bytes;
                    let target = logical * element_bytes;
                    chunk[target..target + element_bytes]
                        .copy_from_slice(&run[source..source + element_bytes]);
                }
                first = end;
            }
            for row in 0..rows {
                let Some(offset) = row_offset(z, first_row + row, row_pitch, slice_pitch) else {
                    return INVALID_ARGUMENT;
                };
                let Some(address) = destination.checked_add(offset) else {
                    return INVALID_ARGUMENT;
                };
                // SAFETY: The caller owns the external row and HSA validates
                // any registered GPU destination range.
                let status = unsafe {
                    crate::memory::copy(
                        address as *mut c_void,
                        chunk[row * row_bytes..].as_ptr().cast(),
                        row_bytes,
                    )
                };
                if let Err(status) = status {
                    return status;
                }
            }
        }
    }
    SUCCESS
}

#[allow(unsafe_code)]
/// Copies image rows after validating its source and destination.
unsafe fn copy_image_rows(
    snapshot: &ImageSnapshot,
    external: usize,
    row_pitch: usize,
    slice_pitch: usize,
    region: Region,
    import: bool,
) -> Status {
    if !snapshot.record.linear
        && (snapshot.native.is_none() || snapshot.record.architecture != Generation::Gfx12)
    {
        // SAFETY: The external rows and image are live throughout this call.
        return unsafe {
            gpu::transfer(snapshot, region, external, row_pitch, slice_pitch, import)
        };
    }
    if snapshot.native.is_some() {
        // SAFETY: The external range and image backing are caller-owned for
        // the synchronous operation.
        return unsafe {
            if import {
                write_tiled_rows(
                    snapshot,
                    region,
                    TiledSource::Rows {
                        address: external,
                        row_pitch,
                        slice_pitch,
                    },
                )
            } else {
                read_tiled_rows(snapshot, region, external, row_pitch, slice_pitch)
            }
        };
    }
    let Some(row_bytes) = region.range[0].checked_mul(snapshot.format.bytes) else {
        return INVALID_ARGUMENT;
    };
    for z in 0..region.range[2] {
        for y in 0..region.range[1] {
            let Some(image_address) = snapshot.region_row_address(
                region.offset[0],
                region.offset[1] + y,
                region.offset[2] + z,
            ) else {
                return INVALID_ARGUMENT;
            };
            let Some(external_offset) = row_offset(z, y, row_pitch, slice_pitch) else {
                return INVALID_ARGUMENT;
            };
            let Some(external_address) = external.checked_add(external_offset) else {
                return INVALID_ARGUMENT;
            };
            // SAFETY: The HSA caller owns both memory ranges and the image
            // backing. the memory service validates registered GPU ranges.
            let status = if import {
                unsafe {
                    crate::memory::copy(
                        image_address as *mut c_void,
                        external_address as *const c_void,
                        row_bytes,
                    )
                }
            } else {
                unsafe {
                    crate::memory::copy(
                        external_address as *mut c_void,
                        image_address as *const c_void,
                        row_bytes,
                    )
                }
            };
            if let Err(status) = status {
                return status;
            }
        }
    }
    SUCCESS
}

#[allow(unsafe_code)]
/// Transfers a validated image region between image and host storage.
pub(crate) unsafe fn transfer_image(
    agent: HsaAgent,
    image: HsaExtImage,
    external: *mut c_void,
    row_pitch: usize,
    slice_pitch: usize,
    region: *const HsaExtImageRegion,
    import: bool,
) -> Status {
    if external.is_null() || region.is_null() {
        return INVALID_ARGUMENT;
    }
    let (snapshot, _inflight) = match image_snapshot(agent, image) {
        Ok(snapshot) => snapshot,
        Err(status) => return status,
    };
    // SAFETY: The caller supplies one initialized region record.
    let region = match snapshot.region(unsafe { *region }) {
        Ok(region) => region,
        Err(status) => return status,
    };
    let row_bytes = match region.range[0].checked_mul(snapshot.format.bytes) {
        Some(bytes) => bytes,
        None => return INVALID_ARGUMENT,
    };
    let (row_pitch, slice_pitch) = match external_pitches(
        snapshot.descriptor.geometry,
        row_bytes,
        region.range[1],
        row_pitch,
        slice_pitch,
    ) {
        Ok(pitches) => pitches,
        Err(status) => return status,
    };
    // SAFETY: The caller provides the external range for this synchronous
    // copy; the image snapshot retains the runtime while it is in flight.
    unsafe {
        copy_image_rows(
            &snapshot,
            external as usize,
            row_pitch,
            slice_pitch,
            region,
            import,
        )
    }
}

/// Performs the validated copy operation under image ownership.
#[allow(unsafe_code)]
pub(crate) unsafe fn copy_images(
    agent: HsaAgent,
    src_image: HsaExtImage,
    src_offset: *const HsaDim3,
    dst_image: HsaExtImage,
    dst_offset: *const HsaDim3,
    range: *const HsaDim3,
) -> Status {
    if src_offset.is_null() || dst_offset.is_null() || range.is_null() {
        return INVALID_ARGUMENT;
    }
    let (source, _source_inflight) = match image_snapshot(agent, src_image) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let (destination, _destination_inflight) = match image_snapshot(agent, dst_image) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let source_format = source.descriptor.format;
    let destination_format = destination.descriptor.format;
    let srgb_conversion = source_format.channel_type == 2
        && destination_format.channel_type == 2
        && matches!(
            (
                source_format.channel_order,
                destination_format.channel_order
            ),
            (8, 14) | (14, 8)
        );
    if source_format != destination_format && !srgb_conversion {
        return IMAGE_FORMAT_UNSUPPORTED;
    }
    let src_region = HsaExtImageRegion {
        offset: unsafe { *src_offset },
        range: unsafe { *range },
    };
    let dst_region = HsaExtImageRegion {
        offset: unsafe { *dst_offset },
        range: unsafe { *range },
    };
    let src_region = match source.region(src_region) {
        Ok(region) => region,
        Err(status) => return status,
    };
    let dst_region = match destination.region(dst_region) {
        Ok(region) => region,
        Err(status) => return status,
    };
    let row_bytes = match src_region.range[0].checked_mul(source.format.bytes) {
        Some(bytes) => bytes,
        None => return INVALID_ARGUMENT,
    };
    let row_count = match src_region.range[1].checked_mul(src_region.range[2]) {
        Some(rows) => rows,
        None => return INVALID_ARGUMENT,
    };
    let size = match row_bytes.checked_mul(row_count) {
        Some(size) => size,
        None => return INVALID_ARGUMENT,
    };
    let mut staging = Vec::<u8>::new();
    if staging.try_reserve_exact(size).is_err() {
        return OUT_OF_RESOURCES;
    }
    staging.resize(size, 0);
    let Some(slice_pitch) = row_bytes.checked_mul(src_region.range[1]) else {
        return INVALID_ARGUMENT;
    };
    // SAFETY: Staging owns the entire tightly packed source region until
    // both synchronous image copies finish.
    let status = unsafe {
        copy_image_rows(
            &source,
            staging.as_mut_ptr() as usize,
            row_bytes,
            slice_pitch,
            src_region,
            false,
        )
    };
    if status != SUCCESS {
        return status;
    }
    if srgb_conversion {
        convert_srgb_pixels(&mut staging, destination_format.channel_order == 14);
    }
    // SAFETY: The same staging buffer remains live for the destination
    // copy, including when the two images share backing.
    unsafe {
        copy_image_rows(
            &destination,
            staging.as_mut_ptr() as usize,
            row_bytes,
            slice_pitch,
            dst_region,
            true,
        )
    }
}

/// Performs the validated clear operation under image ownership.
#[allow(unsafe_code)]
pub(crate) unsafe fn clear_image(
    agent: HsaAgent,
    image: HsaExtImage,
    data: *const c_void,
    image_region: *const HsaExtImageRegion,
) -> Status {
    if data.is_null() || image_region.is_null() {
        return INVALID_ARGUMENT;
    }
    let (snapshot, _inflight) = match image_snapshot(agent, image) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let region = match snapshot.region(unsafe { *image_region }) {
        Ok(region) => region,
        Err(status) => return status,
    };
    let pixel =
        match unsafe { clear_pixel(snapshot.descriptor.format, snapshot.format.bytes, data) } {
            Ok(pixel) => pixel,
            Err(status) => return status,
        };
    let row_bytes = match region.range[0].checked_mul(snapshot.format.bytes) {
        Some(bytes) => bytes,
        None => return INVALID_ARGUMENT,
    };
    if !snapshot.record.linear
        && (snapshot.native.is_none() || snapshot.record.architecture != Generation::Gfx12)
    {
        // SAFETY: The copied clear pattern and retained image cover the call.
        return unsafe { gpu::clear(&snapshot, region, &pixel[..snapshot.format.bytes]) };
    }
    if snapshot.native.is_some() {
        // SAFETY: The caller owns the image backing during this
        // synchronous clear; the clear pattern was copied above.
        return unsafe {
            write_tiled_rows(
                &snapshot,
                region,
                TiledSource::Pixel(&pixel[..snapshot.format.bytes]),
            )
        };
    }
    let mut row = Vec::new();
    if row.try_reserve_exact(row_bytes).is_err() {
        return OUT_OF_RESOURCES;
    }
    row.resize(row_bytes, 0);
    for element in row.chunks_exact_mut(snapshot.format.bytes) {
        element.copy_from_slice(&pixel[..snapshot.format.bytes]);
    }
    for z in 0..region.range[2] {
        for y in 0..region.range[1] {
            let Some(address) = snapshot.region_row_address(
                region.offset[0],
                region.offset[1] + y,
                region.offset[2] + z,
            ) else {
                return INVALID_ARGUMENT;
            };
            let status = unsafe {
                crate::memory::copy(address as *mut c_void, row.as_ptr().cast(), row_bytes)
            };
            if let Err(status) = status {
                return status;
            }
        }
    }
    SUCCESS
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn external_pitches_clamp_to_complete_rows_and_slices() {
        assert_eq!(external_pitches(2, 32, 3, 0, 0), Ok((32, 96)));
        assert_eq!(external_pitches(2, 32, 3, 8, 64), Ok((32, 96)));
        assert_eq!(external_pitches(2, 32, 3, 64, 256), Ok((64, 256)));
        assert_eq!(
            external_pitches(2, usize::MAX, 2, 0, 0),
            Err(INVALID_ARGUMENT)
        );
    }

    #[test]
    fn external_1da_layers_follow_slice_pitch() {
        for (row_pitch, slice_pitch, expected) in [(4, 64, 64), (0, 0, 4), (8, 2, 8), (4, 6, 6)] {
            let (y_stride, z_stride) = external_pitches(3, 4, 2, row_pitch, slice_pitch).unwrap();
            assert_eq!(row_offset(0, 0, y_stride, z_stride), Some(0));
            assert_eq!(row_offset(0, 1, y_stride, z_stride), Some(expected));
        }
    }
}
