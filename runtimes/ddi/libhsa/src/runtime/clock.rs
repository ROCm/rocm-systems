// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GPU and system clock correlation for HSA profiling.

use super::*;

static PRECISE_TIMESTAMPS: AtomicBool = AtomicBool::new(false);

/// Choose the KFD correlation least exposed to preemption between its reads.
pub(crate) fn clock_counters(
    device: &Device,
) -> Result<rocddi::device::gpu::profiling::ClockCounters, rocddi::Error> {
    let gpu = device.gpu()?;
    let samples = if PRECISE_TIMESTAMPS.load(Ordering::Acquire) {
        4
    } else {
        1
    };
    best_clock_sample(samples, || {
        let before = Instant::now();
        let counters = gpu.clock_counters()?;
        Ok((counters, before.elapsed()))
    })
}

pub(super) fn best_clock_sample<T, E>(
    count: usize,
    mut sample: impl FnMut() -> Result<(T, Duration), E>,
) -> Result<T, E> {
    let (first, elapsed) = sample()?;
    let mut best = (first, elapsed);
    for _ in 1..count {
        match sample() {
            Ok((value, elapsed)) if elapsed < best.1 => best = (value, elapsed),
            Ok(_) => {}
            Err(_) => return Ok(best.0),
        }
    }
    Ok(best.0)
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_enable_high_precision_timestamps")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_amd_enable_high_precision_timestamps() -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        PRECISE_TIMESTAMPS.store(true, Ordering::Release);
        SUCCESS
    })
}

pub(crate) fn translate_gpu_interval(
    device: &Device,
    start: u64,
    end: u64,
) -> Result<(u64, u64), Status> {
    if start == 0 || end == 0 {
        return Ok((0, 0));
    }
    let counters = clock_counters(device).map_err(map_error)?;
    Ok((
        translate_gpu_tick(counters, start)?,
        translate_gpu_tick(counters, end)?,
    ))
}

pub(crate) fn translate_gpu_tick(
    counters: rocddi::device::gpu::profiling::ClockCounters,
    tick: u64,
) -> Result<u64, Status> {
    if counters.system_frequency == 0 || counters.gpu_frequency == 0 {
        return Err(ERROR);
    }
    let scaled = |delta: u64| {
        u64::try_from(
            u128::from(delta) * u128::from(counters.system_frequency)
                / u128::from(counters.gpu_frequency),
        )
        .unwrap_or(u64::MAX)
    };
    Ok(if tick >= counters.gpu {
        counters.system.wrapping_add(scaled(tick - counters.gpu))
    } else {
        counters.system.wrapping_sub(scaled(counters.gpu - tick))
    })
}
