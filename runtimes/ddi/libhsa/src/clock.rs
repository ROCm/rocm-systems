// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GPU and system clock correlation for HSA profiling.

use crate::ffi::*;
use crate::status::map_error;
use rocddi::device::Device;
use rocddi::device::gpu::profiling::ClockCounters;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Native sampling and host-clock dependencies of one GPU's conversion.
pub(crate) trait ClockSource {
    /// Performs one native query, including admission, and measures its duration.
    fn sample(&self) -> Result<(ClockCounters, Duration), Status>;

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
    fn sample(&self) -> Result<(ClockCounters, Duration), Status> {
        native_clock_sample(self).map_err(map_error)
    }
}

/// Shared measured map for one activated GPU and uninterrupted counter epoch.
pub(crate) struct GpuClock<Source = Device> {
    /// Retains the activated device for each conversion's fresh native observations.
    source: Source,
    /// Observations begin at first use, independently of general GPU availability.
    state: Mutex<Option<ClockState>>,
}

/// Native health observations and the separately published conversion map.
struct ClockState {
    /// Selected initial anchor from the originating device's first sampling batch.
    startup: ClockCounters,
    /// Latest validated native observation, including failed output conversions.
    latest: ClockCounters,
    /// Greatest native or RAW host observation, retained across failed calls.
    host_high_water: u64,
    /// First adequate measured ratio, immutable after a successful conversion.
    ratio: Option<ClockRatio>,
    /// A contradictory observation permanently ends this owner's clock epoch.
    invalid: bool,
}

impl ClockState {
    /// Admits an initial observation without publishing any converted output.
    fn new(startup: ClockCounters) -> Result<Self, Status> {
        translate_gpu_tick(startup, startup.gpu)?;
        Ok(Self {
            startup,
            latest: startup,
            host_high_water: startup.host,
            ratio: None,
            invalid: false,
        })
    }

    /// Records host progress before any following operation can fail.
    fn observe_host(&mut self, host: u64) -> Result<(), Status> {
        if host < self.host_high_water {
            self.invalid = true;
            return Err(ERROR);
        }
        self.host_high_water = host;
        Ok(())
    }

    /// Ends this epoch on invalid metadata or contradictory native counters.
    fn validate(&mut self, sample: ClockCounters) -> Result<(), Status> {
        self.observe_host(sample.host)?;
        let result = translate_gpu_tick(sample, sample.gpu).and_then(|_| {
            if sample.gpu_info != self.startup.gpu_info || sample.gpu < self.latest.gpu {
                Err(ERROR)
            } else {
                Ok(())
            }
        });
        if result.is_err() {
            self.invalid = true;
        } else {
            self.latest = sample;
        }
        result
    }
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
    /// Retains one shared owner without making profiling a GPU startup prerequisite.
    pub(crate) fn new(source: Source) -> Self {
        Self {
            source,
            state: Mutex::new(None),
        }
    }

    /// Admits every native observation before choosing the least exposed sample.
    fn sample(&self, state: &mut Option<ClockState>) -> Result<ClockCounters, Status> {
        if state.as_ref().is_some_and(|state| state.invalid) {
            return Err(ERROR);
        }
        let starting = state.is_none();
        let count = clock_sample_count();
        let (mut best, mut elapsed) = self.source.sample()?;
        let state = match state {
            Some(state) => {
                state.validate(best)?;
                state
            }
            slot @ None => slot.insert(ClockState::new(best)?),
        };
        for _ in 1..count {
            let (sample, duration) = self.source.sample()?;
            state.validate(sample)?;
            if duration < elapsed {
                best = sample;
                elapsed = duration;
                if starting {
                    state.startup = sample;
                }
            }
        }
        Ok(best)
    }

    /// Obtains one adequate measured baseline before publishing any conversion.
    fn initialize(
        &self,
        state: &mut Option<ClockState>,
        mut sample: ClockCounters,
    ) -> Result<ClockRatio, Status> {
        let observed = state.as_mut().ok_or(ERROR)?;
        let startup = observed.startup;
        let elapsed = sample.host.checked_sub(startup.host).ok_or(ERROR)?;
        // Use the reference refresh horizon as a minimum calibration span.
        // This limits short-baseline sensitivity, not absolute conversion error.
        let minimum_span = rocddi::host::clock::info().frequency_hz / 16;
        if elapsed < minimum_span {
            let deadline = startup.host.checked_add(minimum_span).ok_or(ERROR)?;
            self.wait_until(observed, deadline)?;
            sample = self.sample(state)?;
        }
        // Native sampling is not retried: the post-deadline batch must honor
        // the same host domain and provide positive GPU progress.
        ClockRatio::measured(startup, sample)
    }

    /// Reaches a fixed calibration-span target, not an operation timeout.
    fn wait_until(&self, state: &mut ClockState, deadline: u64) -> Result<(), Status> {
        loop {
            let now = self.source.host_clock()?;
            state.observe_host(now)?;
            if now >= deadline {
                return Ok(());
            }
            // The sleep clock can run at a different rate or wake early. Only
            // another read of the calibration's host domain establishes its span.
            self.source.sleep(Duration::from_nanos(deadline - now));
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
        if state.is_none() {
            self.sample(&mut state)?;
        }
        let sample = self.sample(&mut state)?;
        let ratio = if let Some(ratio) = state.as_ref().ok_or(ERROR)?.ratio {
            ratio
        } else {
            self.initialize(&mut state, sample)?
        };
        let state = state.as_mut().ok_or(ERROR)?;
        let end = ratio.translate(state.startup, end)?;
        let start = ratio.translate(state.startup, start)?;
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
    best_clock_sample(clock_sample_count(), || native_clock_sample(device))
}

/// One timed provider query, before health admission or sample selection.
fn native_clock_sample(device: &Device) -> Result<(ClockCounters, Duration), rocddi::Error> {
    let gpu = device.gpu()?;
    let before = Instant::now();
    let counters = gpu.clock_counters()?;
    Ok((counters, before.elapsed()))
}

/// Uses the same process setting for owned conversion and raw counter queries.
fn clock_sample_count() -> usize {
    if PRECISE_TIMESTAMPS.load(Ordering::Acquire) {
        4
    } else {
        1
    }
}

/// Chooses the raw-counter API's sample with the smallest host read interval.
pub(crate) fn best_clock_sample<T, E>(
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
pub(crate) fn translate_gpu_tick(counters: ClockCounters, tick: u64) -> Result<u64, Status> {
    // Narrow counters need provider epoch extension before this path can
    // distinguish retained ticks from a nearby wrap.
    if matches!(counters.gpu_info.counter_bit_count, 1..=63) {
        return Err(NOT_SUPPORTED);
    }
    counters.to_host(tick).map_err(map_error)
}

#[cfg(test)]
mod tests;
