// SPDX-License-Identifier: MIT

//! KFD compute-queue context backing, including per-XCC and debugger extents.
//!
//! GFX10.1 through GFX12.0 use the sizing rules in libhsakmt's queues.c:
//! `update_ctx_save_restore_size`, `get_num_waves`, and the register-size helpers.
//! KFD may provide authoritative context sizes; older interfaces require the
//! userspace calculation. Both paths pass the same native layout validation.

use super::error;
use super::sysfs::KfdQueueProperties;
use crate::{Error, ErrorKind};

/// Kernel-provided queue context extents within one XCC.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct QueueContextSizes {
    /// Context header, control stack and wave-state storage, in bytes.
    pub(super) context_size: u32,
    /// Header and control-stack prefix of the context, in bytes.
    pub(super) control_stack_size: u32,
}

/// Validated context-save allocation requirements for one compute queue.
pub(super) struct ComputeStorage {
    /// Context storage per XCC, excluding debugger storage, in bytes.
    pub(super) context_size: u32,
    /// Header and control-stack storage per XCC, in bytes.
    pub(super) control_stack_size: u32,
    /// Debugger storage per XCC, in bytes.
    pub(super) debug_size: u32,
    /// Number of consecutive XCC context images in the allocation.
    pub(super) xcc_count: u32,
    /// Host-page-aligned allocation extent for all context and debugger images.
    pub(super) total_size: u64,
}

impl ComputeStorage {
    pub(super) fn new(properties: KfdQueueProperties, page_size: u64) -> Result<Self, Error> {
        if page_size < 4096
            || !page_size.is_power_of_two()
            || properties.xcc_count == 0
            || properties.compute_units == 0
            || properties.compute_units % properties.xcc_count != 0
            || properties.maximum_wave_count_per_compute_unit == 0
        {
            return Err(error(
                ErrorKind::Unsupported,
                "invalid queue context geometry or host page size",
            ));
        }
        let compute_units = u64::from(properties.compute_units / properties.xcc_count);
        let waves = compute_units * u64::from(properties.maximum_wave_count_per_compute_unit);
        let sizes = match properties.context_sizes {
            Some(sizes) => sizes,
            None => derived_sizes(properties, compute_units, waves, page_size)?,
        };
        if sizes.context_size == 0
            || sizes.context_size % 4096 != 0
            || sizes.control_stack_size < 40
            || sizes.control_stack_size % 4096 != 0
            || sizes.control_stack_size > sizes.context_size
        {
            return Err(error(
                ErrorKind::Unsupported,
                "invalid queue context-save sizes",
            ));
        }
        let debug_size = align(waves.checked_mul(32).ok_or_else(size_overflow)?, 64)?;
        let total_size = align(
            u64::from(sizes.context_size)
                .checked_add(debug_size)
                .and_then(|size| size.checked_mul(u64::from(properties.xcc_count)))
                .ok_or_else(size_overflow)?,
            page_size,
        )?;
        // Every XCC header's DebugOffset and DebugSize use u32 byte extents.
        if total_size > u64::from(u32::MAX) {
            return Err(size_overflow());
        }
        Ok(Self {
            context_size: sizes.context_size,
            control_stack_size: sizes.control_stack_size,
            debug_size: u32::try_from(debug_size).map_err(|_| size_overflow())?,
            xcc_count: properties.xcc_count,
            total_size,
        })
    }
}

fn derived_sizes(
    properties: KfdQueueProperties,
    compute_units: u64,
    waves: u64,
    page_size: u64,
) -> Result<QueueContextSizes, Error> {
    let vector_register_bytes: u64 = match properties.gfx_target {
        100_100..=100_103 | 100_300..=100_306 => 0x40000,
        110_000..=110_003 | 110_500..=110_504 | 110_700..=110_701 | 120_000..=120_001 => 0x60000,
        _ => {
            return Err(error(
                ErrorKind::Unsupported,
                "GPU target has no queue context sizing model",
            ));
        }
    };
    if properties.local_data_share_bytes_per_compute_unit == 0 {
        return Err(error(
            ErrorKind::Unsupported,
            "queue context sizing requires the GPU LDS capacity",
        ));
    }
    // The native header is 40 bytes; two stack entries require another 8 bytes.
    let mut control_stack_size = align(
        waves
            .checked_mul(12)
            .and_then(|size| size.checked_add(40 + 8))
            .ok_or_else(size_overflow)?,
        page_size,
    )?;
    if properties.gfx_target / 10000 == 10 {
        // GFX10 hardware limits the control stack to 0x7000 bytes.
        control_stack_size = control_stack_size.min(0x7000);
    }
    // SGPRs occupy 16 KiB and hardware registers 4 KiB per CU on these targets.
    let wave_state_size = align(
        compute_units
            .checked_mul(
                vector_register_bytes
                    + 0x4000
                    + 0x1000
                    + u64::from(properties.local_data_share_bytes_per_compute_unit),
            )
            .ok_or_else(size_overflow)?,
        page_size,
    )?;
    Ok(QueueContextSizes {
        context_size: u32::try_from(
            control_stack_size
                .checked_add(wave_state_size)
                .ok_or_else(size_overflow)?,
        )
        .map_err(|_| size_overflow())?,
        control_stack_size: u32::try_from(control_stack_size).map_err(|_| size_overflow())?,
    })
}

fn align(size: u64, alignment: u64) -> Result<u64, Error> {
    size.div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or_else(size_overflow)
}

fn size_overflow() -> Error {
    error(
        ErrorKind::Unsupported,
        "queue context-save extent exceeds its native fields",
    )
}

#[cfg(test)]
#[path = "tests/queue_context.rs"]
mod tests;
