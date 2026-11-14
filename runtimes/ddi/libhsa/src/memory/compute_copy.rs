// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Private compute kernels and bounded host staging for HSA copy fallback.

use super::*;
use crate::dispatch::{Dispatch, DispatchResult};

/// Maximum reusable staging allocation for a host/device transfer.
const STAGING_BYTES: usize = 4 * 1024 * 1024;

/// Selects a compiler-produced object matching this GPU's actual ISA family.
fn object(gfx: (u32, u32, u32)) -> Option<&'static [u8]> {
    Some(match gfx {
        (9, 0, 8) => include_bytes!("compute_copy/gfx908.hsaco"),
        (9, 0, 10) => include_bytes!("compute_copy/gfx90a.hsaco"),
        (9, 4, 2) | (9, 5, 0) => include_bytes!("compute_copy/gfx9-4-generic.hsaco"),
        (10, 1, _) => include_bytes!("compute_copy/gfx10-1-generic.hsaco"),
        (10, 3, _) => include_bytes!("compute_copy/gfx10-3-generic.hsaco"),
        (11, 0 | 5, _) => include_bytes!("compute_copy/gfx11-generic.hsaco"),
        (12, 0, _) => include_bytes!("compute_copy/gfx12-generic.hsaco"),
        (12, 5, _) => include_bytes!("compute_copy/gfx12-5-generic.hsaco"),
        _ => return None,
    })
}

/// Whether a private compute-copy executable exists for the selected device.
pub(crate) fn supported(device: &Device) -> bool {
    device.endpoint().gpu().is_some_and(supports_target)
}

/// Whether this passive target has the mandatory HSA compute-copy executable.
pub(crate) fn supports_target(info: &rocddi::topology::GpuInfo) -> bool {
    object((info.gfx_major, info.gfx_minor, info.gfx_stepping)).is_some()
}

/// Finds the live HSA agent without confusing devices from another VM domain.
fn prepare<D>(device: &Device, dependencies: D) -> Result<Dispatch<D>, CopyRouteFailure> {
    let info = device
        .endpoint()
        .gpu()
        .ok_or_else(|| CopyRouteFailure::retired(INVALID_AGENT))?;
    let bytes = object((info.gfx_major, info.gfx_minor, info.gfx_stepping))
        .ok_or_else(|| CopyRouteFailure::retired(NOT_SUPPORTED))?;
    let agent = {
        let guard = lock().map_err(CopyRouteFailure::retired)?;
        let runtime = guard
            .as_ref()
            .ok_or_else(|| CopyRouteFailure::retired(NOT_INITIALIZED))?;
        let index = runtime
            .gpus
            .iter()
            .position(|gpu| {
                gpu.endpoint.id == device.endpoint().id && gpu.device.shares_address_domain(device)
            })
            .ok_or_else(|| CopyRouteFailure::retired(INVALID_AGENT))?;
        HsaAgent {
            handle: GPU_AGENT_BASE + index as u64,
        }
    };
    Dispatch::new(agent, Arc::from(bytes), dependencies).map_err(CopyRouteFailure::retired)
}

/// Translates private queue retirement while preserving operand uncertainty.
fn finish(result: &DispatchResult) -> Result<(), CopyRouteFailure> {
    if result.status == SUCCESS {
        Ok(())
    } else {
        Err(CopyRouteFailure {
            status: result.status,
            operands_may_be_live: result.operands_may_be_live,
        })
    }
}

/// Bounds the grid while allowing kernels to stride over a `size_t` extent.
fn work_items(size: u64) -> u32 {
    (size.min(65536) as u32).max(1).div_ceil(64) * 64
}

/// Encodes explicit copy arguments without exposing Rust padding to the GPU.
fn copy_arguments(destination: u64, source: u64, size: u64) -> ([u8; 28], u32) {
    let mut args = [0; 28];
    let work_items = work_items(size);
    args[..8].copy_from_slice(&destination.to_ne_bytes());
    args[8..16].copy_from_slice(&source.to_ne_bytes());
    args[16..24].copy_from_slice(&size.to_ne_bytes());
    args[24..28].copy_from_slice(&work_items.to_ne_bytes());
    (args, work_items)
}

/// Copies two already validated and retained GPU address ranges.
pub(crate) fn copy(
    device: &Device,
    destination: u64,
    source: u64,
    size: u64,
) -> Result<(), CopyRouteFailure> {
    if size == 0 {
        return Ok(());
    }
    let mut dispatch = prepare(device, device.clone())?;
    let (args, work_items) = copy_arguments(destination, source, size);
    // SAFETY: The HSA route retains both checked GPU operands through finish.
    let status = unsafe { dispatch.launch("copy_bytes", &args, work_items) };
    finish(&dispatch.finish(status))
}

/// Fills a retained GPU address range with repeated 32-bit values.
pub(crate) fn fill(
    device: &Device,
    destination: u64,
    value: u32,
    count: u64,
) -> Result<(), CopyRouteFailure> {
    if count == 0 {
        return Ok(());
    }
    let mut dispatch = prepare(device, device.clone())?;
    let work_items = work_items(count);
    let mut args = [0; 28];
    args[..8].copy_from_slice(&destination.to_ne_bytes());
    args[8..12].copy_from_slice(&value.to_ne_bytes());
    args[16..24].copy_from_slice(&count.to_ne_bytes());
    args[24..28].copy_from_slice(&work_items.to_ne_bytes());
    // SAFETY: The caller retains the complete validated writable GPU range.
    let status = unsafe { dispatch.launch("fill_words", &args, work_items) };
    finish(&dispatch.finish(status))
}

/// Owns reusable staging until the private kernel queue is retired.
fn staging(device: &Device, size: usize) -> Result<Dispatch<Allocation>, CopyRouteFailure> {
    let page = rocddi::memory::host::host_page_size()
        .map_err(|error| CopyRouteFailure::retired(map_error(error)))?;
    let size = (size.min(STAGING_BYTES) as u64).div_ceil(page) * page;
    let allocation = device
        .allocate(
            MemoryKind::System,
            size,
            page,
            DeviceAccess::READ | DeviceAccess::WRITE,
        )
        .map_err(|error| CopyRouteFailure::retired(map_error(error)))?;
    if allocation.info().host_address.is_none() {
        return Err(CopyRouteFailure::retired(NOT_SUPPORTED));
    }
    prepare(device, allocation)
}

/// Stages ordinary host bytes into GPU-visible storage after dependencies.
pub(crate) fn copy_from_host(
    device: &Device,
    destination: u64,
    bytes: &[u8],
) -> Result<(), CopyRouteFailure> {
    if bytes.is_empty() {
        return Ok(());
    }
    let mut dispatch = staging(device, bytes.len())?;
    let info = dispatch
        .dependencies()
        .ok_or_else(|| CopyRouteFailure::retired(ERROR))?
        .info();
    let host = info
        .host_address
        .ok_or_else(|| CopyRouteFailure::retired(ERROR))?;
    let mut status = SUCCESS;
    for (index, chunk) in bytes.chunks(STAGING_BYTES).enumerate() {
        // SAFETY: The previous launch retired before this retained staging is reused.
        unsafe { std::ptr::copy_nonoverlapping(chunk.as_ptr(), host as *mut u8, chunk.len()) };
        let (args, work_items) = copy_arguments(
            destination + (index * STAGING_BYTES) as u64,
            info.device_address,
            chunk.len() as u64,
        );
        // SAFETY: Staging is in D and the caller retains the checked GPU destination.
        status = unsafe { dispatch.launch("copy_bytes", &args, work_items) };
        if status != SUCCESS {
            break;
        }
    }
    finish(&dispatch.finish(status))
}

/// Publishes bytes to the host only after the staging kernel has completed.
pub(crate) fn copy_to_host(
    device: &Device,
    bytes: &mut [u8],
    source: u64,
) -> Result<(), CopyRouteFailure> {
    if bytes.is_empty() {
        return Ok(());
    }
    let mut dispatch = staging(device, bytes.len())?;
    let info = dispatch
        .dependencies()
        .ok_or_else(|| CopyRouteFailure::retired(ERROR))?
        .info();
    let host = info
        .host_address
        .ok_or_else(|| CopyRouteFailure::retired(ERROR))?;
    let mut status = SUCCESS;
    for (index, chunk) in bytes.chunks_mut(STAGING_BYTES).enumerate() {
        let (args, work_items) = copy_arguments(
            info.device_address,
            source + (index * STAGING_BYTES) as u64,
            chunk.len() as u64,
        );
        // SAFETY: Staging is in D and the caller retains the checked GPU source.
        status = unsafe { dispatch.launch("copy_bytes", &args, work_items) };
        if status != SUCCESS {
            break;
        }
        // SAFETY: Completion acquired the whole staging write before this host read.
        unsafe {
            std::ptr::copy_nonoverlapping(host as *const u8, chunk.as_mut_ptr(), chunk.len())
        };
    }
    finish(&dispatch.finish(status))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn private_copy_objects_match_supported_targets_and_explicit_argument_abi() {
        let targets = [
            (9, 0, 8),
            (9, 0, 10),
            (9, 4, 2),
            (9, 5, 0),
            (10, 1, 0),
            (10, 1, 3),
            (10, 3, 0),
            (10, 3, 6),
            (11, 0, 0),
            (11, 0, 3),
            (11, 5, 0),
            (11, 5, 3),
            (12, 0, 0),
            (12, 0, 1),
            (12, 5, 0),
            (12, 5, 1),
        ];
        for (major, minor, stepping) in targets {
            let object =
                rocloader::CodeObject::parse(object((major, minor, stepping)).unwrap()).unwrap();
            for xnack in [false, true] {
                let target = rocloader::Target {
                    gfx_major: major,
                    gfx_minor: minor,
                    gfx_stepping: stepping,
                    asic_revision: 1,
                    xnack_enabled: Some(xnack),
                    sram_ecc_enabled: Some(true),
                };
                object.validate_target(target).unwrap();
            }
            let symbols = object
                .symbols(rocloader::SymbolTableOrder::DynamicFirst)
                .unwrap();
            for name in ["copy_bytes", "fill_words"] {
                let symbol = symbols
                    .iter()
                    .find(|symbol| symbol.name == format!("{name}.kd"))
                    .unwrap();
                assert_eq!(symbol.kind, rocloader::SymbolKind::Kernel);
                assert_eq!(symbol.kernarg_size, 28);
                assert_eq!(symbol.kernarg_alignment, 16);
                assert_eq!(symbol.private_size, 0);
                assert_eq!(symbol.group_size, 0);
            }
        }
        assert!(object((11, 7, 0)).is_none());
    }

    #[test]
    fn large_copy_arguments_preserve_full_addresses_and_strided_extents() {
        let size = u64::from(u32::MAX) + 257;
        let (args, grid) = copy_arguments(0xabcd_1234_5678, 0xdcba_4321_8765, size);
        assert_eq!(
            u64::from_ne_bytes(args[..8].try_into().unwrap()),
            0xabcd_1234_5678
        );
        assert_eq!(
            u64::from_ne_bytes(args[8..16].try_into().unwrap()),
            0xdcba_4321_8765
        );
        assert_eq!(u64::from_ne_bytes(args[16..24].try_into().unwrap()), size);
        assert_eq!(u32::from_ne_bytes(args[24..28].try_into().unwrap()), grid);
        assert_eq!(grid, 65536);
        for size in [1, 63, 64, 65, 65535, 65536, 65537] {
            let grid = u64::from(work_items(size));
            assert_eq!(grid % 64, 0);
            assert!(grid <= 65536);
            let mut visited = vec![false; size as usize];
            for thread in 0..grid {
                for index in (thread..size).step_by(grid as usize) {
                    assert!(!visited[index as usize]);
                    visited[index as usize] = true;
                }
            }
            assert!(visited.into_iter().all(|visited| visited));
        }
    }
}
