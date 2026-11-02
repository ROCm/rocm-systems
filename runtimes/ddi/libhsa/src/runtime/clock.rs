// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GPU and system clock correlation for HSA profiling.

use super::*;

/// Whether profiling requests high precision GPU clock samples.
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

/// Chooses the native clock sample with the smallest host read interval.
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

/// Enables high precision sampling for subsequent native clock queries.
pub(crate) fn enable_high_precision_timestamps() {
    PRECISE_TIMESTAMPS.store(true, Ordering::Release);
}

/// Converts a GPU-clock interval to the system-clock domain.
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

/// Converts one GPU clock tick to the system-clock domain.
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
