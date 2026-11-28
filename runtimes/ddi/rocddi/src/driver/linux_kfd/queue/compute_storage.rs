// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Cold queue-context planning before native resource acquisition.

use super::{Error, ErrorKind, error, native_error, os_memory, sysfs};

/// Computed context-save allocation requirements for a compute queue.
#[derive(Clone, Copy)]
pub(super) struct ComputeStorage {
    /// Native context-save extent per XCC.
    pub(super) context_size: u32,
    /// Native control-stack extent.
    pub(super) control_stack_size: u32,
    /// Additional debug state extent.
    pub(super) debug_size: u32,
    /// Number of XCCs requiring context storage.
    pub(super) xcc_count: u32,
    /// Total allocation extent after native alignment.
    pub(super) total_size: u64,
}

// Debugger backing follows the trap-save wave bound even when KFD reports
// authoritative context and control-stack sizes. Resident occupancy cannot
// size this separately allocated area.
fn gfx942_saved_waves(
    properties: sysfs::KfdQueueProperties,
    compute_units_per_xcc: u32,
) -> Result<u32, Error> {
    if properties.simd_arrays_per_engine == 0 || properties.shader_engine_count_per_xcc == 0 {
        return Err(error(
            ErrorKind::Unsupported,
            "missing gfx942 save-wave geometry",
        ));
    }
    let overflow = || {
        error(
            ErrorKind::DriverContract,
            "gfx942 context-save sizing overflows",
        )
    };
    // The trap-save bound is 40 waves/CU, independently of KFD's resident
    // occupancy (32 waves/CU on gfx942).
    // NumShaderBanks = array_count / simd_arrays_per_engine, while
    // NumArrays = simd_arrays_per_engine. Retain ROCr's second division.
    let wave_limit = properties
        .shader_engine_count_per_xcc
        .checked_mul(properties.xcc_count)
        .map(|engines| engines / properties.simd_arrays_per_engine)
        .and_then(|arrays| arrays.checked_mul(512))
        .filter(|limit| *limit != 0)
        .ok_or_else(overflow)?;
    let saved_waves = compute_units_per_xcc
        .checked_mul(40)
        .ok_or_else(overflow)?
        .min(wave_limit);
    Ok(saved_waves)
}

// The gfx942 trap-save protocol without positive size overrides matches
// libhsakmt's update_ctx_save_restore_size: each XCC owns a page-aligned
// control stack followed by CU register/LDS state. Only an absent or zero pair
// reaches this path; positive sizes remain authoritative. Other targets need their
// own qualified register-file and control-stack contracts.
fn legacy_gfx942_sizes(
    properties: sysfs::KfdQueueProperties,
    compute_units_per_xcc: u32,
    saved_waves: u32,
) -> Result<(u32, u32), Error> {
    if properties.gfx_target != 90402
        || properties.wavefront_size != 64
        || properties.maximum_wave_count_per_compute_unit == 0
        || properties.local_data_share_byte_length != 0x10000
        || os_memory::page_size().map_err(|source| native_error("queue page size", source))? != 4096
    {
        return Err(error(
            ErrorKind::Unsupported,
            "missing KFD context sizes require qualified gfx942 geometry",
        ));
    }
    let overflow = || {
        error(
            ErrorKind::DriverContract,
            "gfx942 context-save sizing overflows",
        )
    };
    // 40-byte native header, eight bytes per saved wave and an eight-byte
    // stack trailer. gfx942 saves 512 KiB VGPR, 16 KiB SGPR, 64 KiB LDS
    // and 4 KiB HW registers per CU.
    let control_stack_size = saved_waves
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(40 + 8 + 4095))
        .map(|bytes| bytes / 4096 * 4096)
        .ok_or_else(overflow)?;
    let context_size = compute_units_per_xcc
        .checked_mul(0x80000 + 0x4000 + 0x10000 + 0x1000)
        .and_then(|bytes| bytes.checked_add(control_stack_size))
        .ok_or_else(overflow)?;
    Ok((context_size, control_stack_size))
}

pub(super) fn compute_storage(
    properties: sysfs::KfdQueueProperties,
) -> Result<ComputeStorage, Error> {
    let xcc_count = properties.xcc_count;
    let mut context_size = properties.context_size;
    let mut control_stack_size = properties.control_stack_size;
    if xcc_count == 0
        || properties.compute_units == 0
        || properties.compute_units % xcc_count != 0
        || properties.maximum_wave_count_per_compute_unit == 0
    {
        return Err(error(
            ErrorKind::Unsupported,
            "KFD did not report usable queue context-save sizes",
        ));
    }
    let compute_units_per_xcc = properties.compute_units / xcc_count;
    let saved_waves = if properties.gfx_target == 90402 {
        gfx942_saved_waves(properties, compute_units_per_xcc)?
    } else {
        compute_units_per_xcc
            .checked_mul(properties.maximum_wave_count_per_compute_unit)
            .ok_or_else(|| error(ErrorKind::DriverContract, "KFD wave count overflows"))?
    };
    if context_size == 0 && control_stack_size == 0 {
        (context_size, control_stack_size) =
            legacy_gfx942_sizes(properties, compute_units_per_xcc, saved_waves)?;
    }
    if context_size == 0
        || context_size % 4096 != 0
        || control_stack_size < 40
        || control_stack_size % 4096 != 0
        || control_stack_size > context_size
    {
        return Err(error(
            ErrorKind::Unsupported,
            "KFD did not report usable queue context-save sizes",
        ));
    }
    let debug_size = saved_waves
        .checked_mul(32)
        .and_then(|bytes| bytes.checked_add(63))
        .map(|bytes| bytes / 64 * 64)
        .ok_or_else(|| error(ErrorKind::DriverContract, "KFD debugger storage overflows"))?;
    let total_size = (u64::from(context_size) + u64::from(debug_size))
        .checked_mul(u64::from(xcc_count))
        .ok_or_else(|| error(ErrorKind::Unsupported, "KFD context-save extent overflows"))?;
    let page =
        os_memory::page_size().map_err(|source| native_error("queue page size", source))? as u64;
    let total_size = total_size
        .div_ceil(page)
        .checked_mul(page)
        .ok_or_else(|| error(ErrorKind::Unsupported, "KFD context-save extent overflows"))?;
    // DebugOffset and DebugSize in the native header are u32. Check their
    // complete multi-XCC extent before any buffer is acquired or initialized.
    if total_size > u64::from(u32::MAX) {
        return Err(error(
            ErrorKind::Unsupported,
            "KFD context-save extent exceeds its header fields",
        ));
    }
    Ok(ComputeStorage {
        context_size,
        control_stack_size,
        debug_size,
        xcc_count,
        total_size,
    })
}
