// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Qualified gfx942 layout when KFD omits queue context sizes.

use super::{Error, ErrorKind, error, native_error, os_memory, sysfs};

// The gfx942 trap-save protocol without positive size overrides matches
// libhsakmt's update_ctx_save_restore_size: each XCC owns a page-aligned
// control stack followed by CU register/LDS state. Only an absent or zero pair
// reaches this path; positive sizes remain authoritative. Other targets need their
// own qualified register-file and control-stack contracts.
pub(super) fn legacy_gfx942_sizes(
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
