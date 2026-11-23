// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GPU and system clock correlation for HSA profiling.

use super::*;
use rocddi::device::gpu::profiling::ClockCounters;

/// Native sampling and host-clock dependencies of one GPU's conversion.
pub(crate) trait ClockSource {
    /// Samples the originating device, including its native admission checks.
    fn sample(&self) -> Result<ClockCounters, Status>;

    /// Reads the exact epoch and nanosecond units used by `ClockCounters::host`.
    fn host_clock(&self) -> Result<u64, Status> {
        rocddi::host::clock::read().map_err(map_error)
    }

    /// Sleeps between host-clock checks without polling the device.
    fn sleep(&self, duration: Duration) {
        thread::sleep(duration);
    }
}

impl ClockSource for Device {
    fn sample(&self) -> Result<ClockCounters, Status> {
        clock_counters(self).map_err(map_error)
    }
}

/// Shared measured map for one activated GPU and uninterrupted counter epoch.
pub(crate) struct GpuClock<Source = Device> {
    /// Retains the activated device and performs a live query on every conversion.
    source: Source,
    /// Startup observation from the originating activated device's epoch.
    startup: ClockCounters,
    /// Serializes live observations, first-map publication, and conversion.
    state: Mutex<ClockState>,
}

/// Native health observations and the separately published conversion map.
struct ClockState {
    /// Latest validated native observation, including failed output conversions.
    latest: ClockCounters,
    /// First adequate measured ratio, immutable after a successful conversion.
    ratio: Option<ClockRatio>,
}

/// Immutable positive measured ratio, in host ticks per GPU tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ClockRatio {
    /// Host ticks in the first adequate startup-to-sample span.
    host: u64,
    /// GPU ticks in the first adequate startup-to-sample span.
    gpu: u64,
}

impl ClockRatio {
    /// Requires a full host calibration horizon before any map can be published.
    fn measured(startup: ClockCounters, sample: ClockCounters) -> Result<Self, Status> {
        let gpu = sample.gpu.checked_sub(startup.gpu).ok_or(ERROR)?;
        let host = sample.host.checked_sub(startup.host).ok_or(ERROR)?;
        if gpu == 0 || host < rocddi::host::clock::info().frequency_hz / 16 {
            return Err(ERROR);
        }
        Ok(Self { host, gpu })
    }

    /// Scales a magnitude without floating-point rounding or unchecked narrowing.
    fn scale(self, ticks: u64) -> Result<u64, Status> {
        u64::try_from(u128::from(ticks) * u128::from(self.host) / u128::from(self.gpu))
            .map_err(|_| ERROR)
    }

    /// Converts around an anchor with checked signed displacement.
    fn translate(self, anchor: ClockCounters, tick: u64) -> Result<u64, Status> {
        let delta = anchor
            .gpu_info
            .tick_delta(anchor.gpu, tick)
            .map_err(map_error)?;
        let magnitude = u64::try_from(delta.unsigned_abs()).map_err(|_| ERROR)?;
        let scaled = self.scale(magnitude)?;
        if delta >= 0 {
            anchor.host.checked_add(scaled)
        } else {
            anchor.host.checked_sub(scaled)
        }
        .ok_or(ERROR)
    }
}

impl<Source: ClockSource> GpuClock<Source> {
    /// Samples startup during normal GPU construction; failure uses init rollback.
    pub(crate) fn new(source: Source) -> Result<Self, Status> {
        let startup = source.sample()?;
        translate_gpu_tick(startup, startup.gpu)?;
        Ok(Self {
            source,
            startup,
            state: Mutex::new(ClockState {
                latest: startup,
                ratio: None,
            }),
        })
    }

    /// Rejects invalid metadata and regression behind the retained observation.
    fn validate(&self, previous: ClockCounters, sample: ClockCounters) -> Result<(), Status> {
        translate_gpu_tick(sample, sample.gpu)?;
        if sample.gpu_info != self.startup.gpu_info
            || sample.gpu < previous.gpu
            || sample.host < previous.host
        {
            return Err(ERROR);
        }
        Ok(())
    }

    /// Obtains one adequate measured baseline before publishing any conversion.
    fn initialize(&self, state: &mut ClockState) -> Result<ClockRatio, Status> {
        let sample = state.latest;
        let elapsed = sample.host.checked_sub(self.startup.host).ok_or(ERROR)?;
        // Use the reference refresh horizon as a minimum calibration span.
        // This limits short-baseline sensitivity, not absolute conversion error.
        let minimum_span = rocddi::host::clock::info().frequency_hz / 16;
        if elapsed < minimum_span {
            let deadline = self.startup.host.checked_add(minimum_span).ok_or(ERROR)?;
            let reached = self.wait_until(deadline, sample.host)?;
            let next = self.source.sample()?;
            self.validate(sample, next)?;
            if next.host < reached {
                return Err(ERROR);
            }
            state.latest = next;
        }
        // Native sampling is not retried: the post-deadline sample must honor
        // the same host domain and provide positive GPU progress.
        ClockRatio::measured(self.startup, state.latest)
    }

    /// Reaches a fixed calibration-span target, not an operation timeout.
    fn wait_until(&self, deadline: u64, mut previous: u64) -> Result<u64, Status> {
        loop {
            let now = self.source.host_clock()?;
            if now < previous {
                return Err(ERROR);
            }
            if now >= deadline {
                return Ok(now);
            }
            // The sleep clock can run at a different rate or wake early. Only
            // another read of the calibration's host domain establishes its span.
            self.source.sleep(Duration::from_nanos(deadline - now));
            previous = now;
        }
    }

    /// Converts both endpoints using the same immutable map.
    pub(crate) fn interval(&self, start: u64, end: u64) -> Result<(u64, u64), Status> {
        if start == 0 || end == 0 {
            return Ok((0, 0));
        }
        self.convert(start, end)
    }

    /// Converts a retained originating-agent tick with the shared measured map.
    pub(crate) fn tick(&self, tick: u64) -> Result<u64, Status> {
        self.convert(tick, tick).map(|(_, end)| end)
    }

    /// Keeps native failure visible even when no new calibration is needed.
    fn convert(&self, start: u64, end: u64) -> Result<(u64, u64), Status> {
        let mut state = self.state.lock().map_err(|_| ERROR)?;
        let sample = self.source.sample()?;
        self.validate(state.latest, sample)?;
        state.latest = sample;
        let ratio = if let Some(ratio) = state.ratio {
            ratio
        } else {
            self.initialize(&mut state)?
        };
        let end = ratio.translate(self.startup, end)?;
        let start = ratio.translate(self.startup, start)?;
        // A later slope would reinterpret values already cached by callers.
        // Keep one map for the epoch; its initial error and drift remain unbounded.
        state.ratio = Some(ratio);
        Ok((start, end))
    }
}

/// Whether profiling requests high precision GPU clock samples.
static PRECISE_TIMESTAMPS: AtomicBool = AtomicBool::new(false);

/// Choose the provider correlation least exposed to preemption between its reads.
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

/// Converts one tick using a native sample and validates its nominal timebase.
pub(super) fn translate_gpu_tick(counters: ClockCounters, tick: u64) -> Result<u64, Status> {
    // Narrow counters need provider epoch extension before this path can
    // distinguish retained ticks from a nearby wrap.
    if matches!(counters.gpu_info.counter_bit_count, 1..=63) {
        return Err(NOT_SUPPORTED);
    }
    counters.to_host(tick).map_err(map_error)
}

#[cfg(test)]
mod tests;
