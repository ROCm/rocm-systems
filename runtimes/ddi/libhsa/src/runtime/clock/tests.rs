// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host tests of the production clock owner with clock dependencies injected.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use std::collections::VecDeque;

mod anchors;

/// Scripted native dependency shared with the test's observation handle.
#[derive(Clone)]
struct Samples {
    /// Native results consumed by the real constructor and conversion methods.
    values: Arc<Mutex<VecDeque<Result<ClockCounters, Status>>>>,
    /// Number of actual dependency calls, including startup and failures.
    calls: Arc<AtomicUsize>,
    /// Requested cold waits; replay advances through supplied observations.
    waits: Arc<Mutex<Vec<Duration>>>,
    /// Virtual host time in the same domain as the scripted native observations.
    host_time: Arc<Mutex<u64>>,
}

impl Samples {
    /// Supplies explicit native observations without manufacturing a device.
    fn new(values: impl IntoIterator<Item = Result<ClockCounters, Status>>) -> Self {
        Self {
            values: Arc::new(Mutex::new(values.into_iter().collect())),
            calls: Arc::new(AtomicUsize::new(0)),
            waits: Arc::new(Mutex::new(Vec::new())),
            host_time: Arc::new(Mutex::new(0)),
        }
    }
}

impl ClockSource for Samples {
    fn sample(&self) -> Result<ClockCounters, Status> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let result = self
            .values
            .lock()
            .unwrap()
            .pop_front()
            .expect("native sample");
        if let Ok(counters) = result {
            *self.host_time.lock().unwrap() = counters.host;
        }
        result
    }

    fn host_clock(&self) -> Result<u64, Status> {
        Ok(*self.host_time.lock().unwrap())
    }

    fn sleep(&self, duration: Duration) {
        self.waits.lock().unwrap().push(duration);
        let elapsed = u64::try_from(duration.as_nanos()).unwrap();
        let mut time = self.host_time.lock().unwrap();
        *time = time.checked_add(elapsed).unwrap();
    }
}

/// Full-width synthetic raw-monotonic correlation at a nominal 100 MHz.
fn sample(gpu: u64, host: u64) -> ClockCounters {
    ClockCounters {
        gpu,
        host,
        gpu_info: rocddi::host::clock::ClockInfo {
            frequency_hz: 100_000_000,
            counter_bit_count: 0,
        },
        system: None,
    }
}

#[test]
fn constructor_samples_and_rejects_failed_or_unsupported_sources() {
    assert!(matches!(
        GpuClock::new(Samples::new([Err(ERROR)])),
        Err(ERROR)
    ));
    for (frequency_hz, counter_bit_count, status) in [
        (0, 0, ERROR),
        (100_000_000, 32, NOT_SUPPORTED),
        (100_000_000, 65, ERROR),
    ] {
        let counters = ClockCounters {
            gpu_info: rocddi::host::clock::ClockInfo {
                frequency_hz,
                counter_bit_count,
            },
            ..sample(100, 1_000)
        };
        assert!(
            matches!(GpuClock::new(Samples::new([Ok(counters)])), Err(error) if error == status)
        );
    }
    let source = Samples::new([Ok(sample(100, 1_000))]);
    let _clock = GpuClock::new(source.clone()).unwrap();
    assert_eq!(source.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn first_conversion_waits_once_for_a_measured_baseline_without_nominal_output() {
    let source = Samples::new([
        Ok(sample(100, 1_000_000_000)),
        // Three GPU ticks and 5 us of sequential-read noise are not a slope.
        Ok(sample(103, 1_000_005_000)),
        Ok(sample(8_000_100, 1_080_120_000)),
    ]);
    let clock = GpuClock::new(source.clone()).unwrap();
    assert!(clock.calibration.lock().unwrap().is_none());
    assert_eq!(clock.tick(102), Ok(1_000_000_020));
    let map = clock.calibration.lock().unwrap().unwrap();
    assert_eq!(map.latest, sample(8_000_100, 1_080_120_000));
    assert_eq!(map.ratio.host, 80_120_000);
    assert_eq!(map.ratio.gpu, 8_000_000);
    assert_eq!(
        *source.waits.lock().unwrap(),
        [Duration::from_micros(62_495)]
    );
    assert_eq!(source.calls.load(Ordering::Relaxed), 3);
}

#[test]
fn retained_and_reverse_queries_share_the_map_that_avoids_the_captured_inversion() {
    // These equivalent anchors reproduce captured outputs; they are not the
    // native samples, which the original capture did not retain.
    let early = sample(1_260_825_916_689_703, 12_627_073_945_037_079);
    let late = sample(1_260_825_916_693_115, 12_627_073_945_059_316);
    let earlier_tick = 1_260_825_916_688_703;
    let later_tick = 1_260_825_916_689_115;
    let independent_earlier = translate_gpu_tick(early, earlier_tick).unwrap();
    let independent_later = translate_gpu_tick(late, later_tick).unwrap();
    assert_eq!(
        i128::from(independent_later) - i128::from(independent_earlier),
        -7_763
    );

    let startup = sample(early.gpu - 100_000_000, early.host - 1_000_000_000);
    let source = Samples::new([Ok(startup), Ok(early), Ok(late), Ok(late), Ok(late)]);
    let clock = GpuClock::new(source.clone()).unwrap();
    let later = clock.tick(later_tick).unwrap();
    let earlier = clock.tick(earlier_tick).unwrap();
    assert_eq!(later - earlier, 4_120);
    assert_eq!(
        clock.interval(earlier_tick, later_tick),
        Ok((earlier, later))
    );
    assert_eq!(clock.tick(later_tick), Ok(later));
    assert_eq!(source.calls.load(Ordering::Relaxed), 5);
}

#[test]
fn interval_and_tick_use_the_same_measured_host_ratio_and_ignore_boottime() {
    let startup = sample(100, 1_000_000_000);
    let mut latest = sample(10_000_100, 1_100_150_000);
    latest.system = Some(rocddi::device::gpu::profiling::SystemClockSample {
        domain: rocddi::device::gpu::profiling::SystemClockDomain::BootTime,
        ticks: 9_000_000_000_000,
        info: rocddi::host::clock::info(),
    });
    let clock = GpuClock::new(Samples::new([
        Ok(startup),
        Ok(latest),
        Ok(latest),
        Ok(latest),
    ]))
    .unwrap();
    let interval = clock.interval(9_000_100, 10_000_100).unwrap();
    assert_eq!(interval, (1_090_135_000, 1_100_150_000));
    assert_eq!(clock.tick(10_000_100), Ok(interval.1));
    assert_eq!(clock.tick(9_000_100), Ok(interval.0));
}

#[test]
fn quarter_span_refresh_starts_strictly_after_the_boundary() {
    let startup = sample(100, 1_000_000_000);
    let latest = sample(10_000_100, 1_100_000_000);
    let clock = GpuClock::new(Samples::new([
        Ok(startup),
        Ok(latest),
        Ok(sample(12_600_100, 1_126_000_000)),
        Ok(sample(13_000_100, 1_130_000_000)),
    ]))
    .unwrap();
    assert_eq!(clock.tick(latest.gpu), Ok(latest.host));
    assert_eq!(clock.tick(12_500_100), Ok(1_125_000_000));
    assert_eq!(clock.calibration.lock().unwrap().unwrap().latest, latest);
    assert_eq!(clock.tick(12_500_101), Ok(1_125_000_010));
    assert_eq!(
        clock.calibration.lock().unwrap().unwrap().latest.gpu,
        13_000_100
    );
}

#[test]
fn host_horizon_refresh_starts_strictly_after_one_sixteenth_second() {
    let startup = sample(100, 1_000_000_000);
    let latest = sample(100_000_100, 2_000_000_000);
    let clock = GpuClock::new(Samples::new([
        Ok(startup),
        Ok(latest),
        Ok(sample(106_300_100, 2_063_000_000)),
        Ok(sample(106_400_100, 2_064_000_000)),
    ]))
    .unwrap();
    assert_eq!(clock.tick(latest.gpu), Ok(latest.host));
    assert_eq!(clock.tick(106_250_100), Ok(2_062_500_000));
    assert_eq!(clock.calibration.lock().unwrap().unwrap().latest, latest);
    assert_eq!(clock.tick(106_250_101), Ok(2_062_500_010));
    assert_eq!(
        clock.calibration.lock().unwrap().unwrap().latest.gpu,
        106_400_100
    );
}

#[test]
fn first_measured_map_removes_the_synthetic_nominal_bootstrap_inversion() {
    let clock = GpuClock::new(Samples::new([
        Ok(sample(100, 1_000_000_000)),
        Ok(sample(6_249_100, 1_062_490_000)),
        Ok(sample(6_260_100, 1_062_590_000)),
        Ok(sample(6_260_100, 1_062_590_000)),
    ]))
    .unwrap();
    let earlier = clock.tick(6_249_000).unwrap();
    let later = clock.tick(6_249_412).unwrap();
    eprintln!(
        "measured bootstrap: ticks=(6249000,6249412) translated=({earlier},{later}) gap_ns={}",
        i128::from(later) - i128::from(earlier)
    );
    assert!(
        later > earlier,
        "the first successful value already uses the measured map"
    );
}

#[test]
fn captured_anchor_replay_removes_the_reverse_retained_bootstrap_inversion() {
    // Three recorded gfx942 correlations, in acquisition order. Endpoint ticks
    // are synthetic and precede both query samples; no GPU work runs here.
    let startup = sample(142_666_430_882_407, 1_428_622_627_227_254);
    let short = sample(142_666_432_121_002, 1_428_622_639_631_825);
    let measured = sample(142_666_442_127_170, 1_428_622_739_844_311);
    let earlier_tick = 142_666_432_119_590;
    let later_tick = earlier_tick + 412;
    assert!(startup.gpu < earlier_tick && later_tick < short.gpu);
    assert!(short.gpu < measured.gpu && short.host < measured.host);
    let source = Samples::new([Ok(startup), Ok(short), Ok(measured), Ok(measured)]);
    let clock = GpuClock::new(source.clone()).unwrap();
    let later = clock.tick(later_tick).unwrap();
    let earlier = clock.tick(earlier_tick).unwrap();
    eprintln!(
        "captured-anchor replay: synthetic ticks=({earlier_tick},{later_tick}) reverse-query translated=({earlier},{later}) ordered_gap_ns={}",
        i128::from(later) - i128::from(earlier)
    );
    assert_eq!(earlier, 1_428_622_639_617_724);
    assert_eq!(later - earlier, 4_127);
    assert_eq!(source.waits.lock().unwrap().len(), 1);
}

#[test]
fn refreshed_slope_can_remap_and_reverse_close_retained_endpoints() {
    // Positive increasing anchors alone cannot establish cross-refresh order.
    let latest = sample(100_000_100, 2_000_000_000);
    let refreshed = sample(125_000_100, 2_249_875_000);
    let clock = GpuClock::new(Samples::new([
        Ok(sample(100, 1_000_000_000)),
        Ok(latest),
        Ok(refreshed),
        Ok(refreshed),
        Ok(refreshed),
    ]))
    .unwrap();
    let earlier = clock.tick(latest.gpu).unwrap();
    clock.tick(110_000_100).unwrap();
    let later = clock.tick(latest.gpu + 412).unwrap();
    let repeated_earlier = clock.tick(latest.gpu).unwrap();
    eprintln!(
        "refresh counterexample: ticks=(100000100,100000512) translated=({earlier},{later}) gap_ns={} repeated_earlier={repeated_earlier}",
        i128::from(later) - i128::from(earlier)
    );
    assert!(
        later < earlier,
        "mutable calibration does not establish historical ordering"
    );
    assert_ne!(repeated_earlier, earlier);
}

#[test]
fn pre_startup_ticks_keep_their_first_ratio_across_refreshes() {
    let latest = sample(11_000_100, 1_100_150_000);
    let refreshed = sample(21_000_100, 1_200_200_000);
    let clock = GpuClock::new(Samples::new([
        Ok(sample(1_000_100, 1_000_000_000)),
        Ok(latest),
        Ok(refreshed),
        Ok(refreshed),
    ]))
    .unwrap();
    assert_eq!(clock.tick(100), Ok(989_985_000));
    assert_eq!(clock.tick(refreshed.gpu), Ok(refreshed.host));
    assert_eq!(clock.tick(100), Ok(989_985_000));
}

#[test]
fn live_failure_never_returns_a_cached_success_and_is_retryable() {
    let startup = sample(100, 1_000_000_000);
    let latest = sample(10_000_100, 1_100_000_000);
    let source = Samples::new([
        Ok(startup),
        Ok(latest),
        Err(INVALID_RUNTIME_STATE),
        Ok(latest),
    ]);
    let clock = GpuClock::new(source.clone()).unwrap();
    let value = clock.tick(9_000_100).unwrap();
    assert_eq!(clock.tick(9_000_100), Err(INVALID_RUNTIME_STATE));
    assert_eq!(clock.tick(9_000_100), Ok(value));
    assert_eq!(source.calls.load(Ordering::Relaxed), 4);
}

#[test]
fn invalid_or_regressing_samples_leave_calibration_unchanged_for_retry() {
    let startup = sample(100, 1_000_000_000);
    for invalid in [
        sample(99, startup.host + 1),
        sample(101, startup.host - 1),
        sample(100, startup.host + 100_000_000),
        ClockCounters {
            gpu_info: rocddi::host::clock::ClockInfo {
                frequency_hz: 200_000_000,
                counter_bit_count: 0,
            },
            ..sample(101, startup.host + 1)
        },
    ] {
        let latest = sample(10_000_100, 1_100_000_000);
        let clock = GpuClock::new(Samples::new([Ok(startup), Ok(invalid), Ok(latest)])).unwrap();
        assert_eq!(clock.tick(1_000_100), Err(ERROR));
        assert!(clock.calibration.lock().unwrap().is_none());
        assert_eq!(clock.tick(10_000_100), Ok(latest.host));
    }
}

#[test]
fn checked_conversion_overflows_remain_errors_and_do_not_publish_calibration() {
    let source = Samples::new([
        Ok(sample(100, u64::MAX - 100_000_000)),
        Ok(sample(10_000_100, u64::MAX)),
    ]);
    let clock = GpuClock::new(source).unwrap();
    assert_eq!(clock.tick(10_000_101), Err(ERROR));
    assert!(clock.calibration.lock().unwrap().is_none());
    let clock = GpuClock::new(Samples::new([
        Ok(sample(100_000_000, 1_000_000)),
        Ok(sample(110_000_000, 101_000_000)),
    ]))
    .unwrap();
    assert_eq!(clock.tick(0), Err(ERROR));
    let counters = ClockCounters {
        gpu_info: rocddi::host::clock::ClockInfo {
            frequency_hz: 1,
            counter_bit_count: 0,
        },
        ..sample(0, 1)
    };
    let clock = GpuClock::new(Samples::new([
        Ok(counters),
        Ok(ClockCounters {
            gpu: 1,
            host: 62_500_001,
            ..counters
        }),
    ]))
    .unwrap();
    assert_eq!(clock.tick(u64::MAX), Err(ERROR));
}

#[test]
fn full_width_counter_keeps_the_native_half_period_ambiguity_check() {
    let counters = ClockCounters {
        gpu_info: rocddi::host::clock::ClockInfo {
            frequency_hz: 1_000_000_000,
            counter_bit_count: 64,
        },
        ..sample(0, 1)
    };
    let clock = GpuClock::new(Samples::new([
        Ok(counters),
        Ok(ClockCounters {
            gpu: 100_000_000,
            host: 100_000_001,
            ..counters
        }),
    ]))
    .unwrap();
    assert_eq!(clock.tick(1 << 63), Err(ERROR));
}

#[test]
fn absent_interval_endpoints_keep_the_zero_sentinel_without_sampling() {
    let source = Samples::new([Ok(sample(100, 1_000))]);
    let clock = GpuClock::new(source.clone()).unwrap();
    assert_eq!(clock.interval(0, 101), Ok((0, 0)));
    assert_eq!(clock.interval(99, 0), Ok((0, 0)));
    assert_eq!(source.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn first_query_uses_an_adequate_startup_age_without_waiting() {
    let source = Samples::new([
        Ok(sample(100, 1_000_000_000)),
        Ok(sample(10_000_100, 1_100_150_000)),
    ]);
    let clock = GpuClock::new(source.clone()).unwrap();
    assert_eq!(clock.tick(5_000_100), Ok(1_050_075_000));
    assert!(source.waits.lock().unwrap().is_empty());
    assert_eq!(source.calls.load(Ordering::Relaxed), 2);
}

#[test]
fn post_deadline_native_shortfall_leaves_initialization_retryable() {
    let startup = sample(100, 1_000_000_000);
    let latest = sample(10_000_100, 1_100_150_000);
    let source = Samples::new([Ok(startup), Ok(startup), Ok(startup), Ok(latest)]);
    let clock = GpuClock::new(source.clone()).unwrap();
    assert_eq!(clock.tick(101), Err(ERROR));
    assert!(clock.calibration.lock().unwrap().is_none());
    assert_eq!(
        *source.waits.lock().unwrap(),
        [Duration::from_micros(62_500)]
    );
    assert_eq!(clock.tick(latest.gpu), Ok(latest.host));
    assert_eq!(source.calls.load(Ordering::Relaxed), 4);
    assert_eq!(source.waits.lock().unwrap().len(), 1);
}

#[test]
fn post_wait_native_failure_or_regression_never_publishes_a_first_map() {
    let startup = sample(100, 1_000_000_000);
    let first = sample(200, 1_000_001_000);
    let latest = sample(10_000_100, 1_100_150_000);
    for bad in [Err(INVALID_RUNTIME_STATE), Ok(sample(199, 1_070_000_000))] {
        let expected = if bad.is_err() {
            INVALID_RUNTIME_STATE
        } else {
            ERROR
        };
        let source = Samples::new([Ok(startup), Ok(first), bad, Ok(latest)]);
        let clock = GpuClock::new(source.clone()).unwrap();
        assert_eq!(clock.tick(150), Err(expected));
        assert!(clock.calibration.lock().unwrap().is_none());
        assert_eq!(source.waits.lock().unwrap().len(), 1);
        assert_eq!(clock.tick(latest.gpu), Ok(latest.host));
    }
}

/// CPU-derived native dependency using the production host reader and sleep.
struct RealWaitSamples {
    /// Raw host origin for the synthetic GPU domain, in nanoseconds.
    origin: u64,
    /// Every synthetic native query, including startup and post-deadline sample.
    samples: Mutex<Vec<ClockCounters>>,
}

impl ClockSource for Arc<RealWaitSamples> {
    fn sample(&self) -> Result<ClockCounters, Status> {
        let host = rocddi::host::clock::read().map_err(map_error)?;
        let counters = sample(100 + (host - self.origin) / 10, host);
        self.samples.lock().unwrap().push(counters);
        Ok(counters)
    }
}

#[test]
fn cold_calibration_measures_the_production_wait_on_the_host() {
    let source = Arc::new(RealWaitSamples {
        origin: rocddi::host::clock::read().unwrap(),
        samples: Mutex::new(Vec::new()),
    });
    let clock = GpuClock::new(source.clone()).unwrap();
    let raw_before = rocddi::host::clock::read().unwrap();
    let before = Instant::now();
    assert!(clock.tick(100).unwrap() > 0);
    let elapsed = before.elapsed();
    let raw_elapsed = rocddi::host::clock::read().unwrap() - raw_before;
    let calibration = clock.calibration.lock().unwrap().unwrap();
    let span = calibration.latest.host - clock.startup.host;
    let samples = source.samples.lock().unwrap();
    let first_query_age = samples[1].host - samples[0].host;
    let expected_samples = if first_query_age < 62_500_000 { 3 } else { 2 };
    eprintln!(
        "CPU RAW deadline: monotonic_query_ns={} raw_query_ns={raw_elapsed} first_query_age_ns={first_query_age} measured_host_span_ns={span} native_samples={}",
        elapsed.as_nanos(),
        samples.len()
    );
    assert!(span >= 62_500_000);
    assert_eq!(samples.len(), expected_samples);
}

/// Independent scripted native and raw-host dependencies for deadline failures.
struct ScriptedHostSamples {
    /// Native observations and recorded positive sleep requests.
    native: Samples,
    /// Exact raw-host reads, including native host-read failures.
    host_times: Mutex<VecDeque<Result<u64, Status>>>,
    /// Number of host observations, independent of GPU sampling.
    reads: AtomicUsize,
}

impl ScriptedHostSamples {
    /// Supplies both clock domains without changing owner control flow.
    fn new(
        native: impl IntoIterator<Item = Result<ClockCounters, Status>>,
        host_times: impl IntoIterator<Item = Result<u64, Status>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            native: Samples::new(native),
            host_times: Mutex::new(host_times.into_iter().collect()),
            reads: AtomicUsize::new(0),
        })
    }
}

impl ClockSource for Arc<ScriptedHostSamples> {
    fn sample(&self) -> Result<ClockCounters, Status> {
        self.native.sample()
    }

    fn host_clock(&self) -> Result<u64, Status> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.host_times
            .lock()
            .unwrap()
            .pop_front()
            .expect("host read")
    }

    fn sleep(&self, duration: Duration) {
        self.native.sleep(duration);
    }
}

#[test]
fn short_wakes_recheck_the_raw_deadline_without_resampling_the_gpu() {
    let startup = sample(100, 1_000_000_000);
    let first = sample(200, 1_000_001_000);
    let deadline = 1_062_500_000;
    let source = ScriptedHostSamples::new(
        [Ok(startup), Ok(first), Ok(sample(6_240_100, deadline))],
        [
            Ok(first.host),
            Ok(deadline - 40_000),
            Ok(deadline - 1),
            Ok(deadline),
        ],
    );
    let clock = GpuClock::new(source.clone()).unwrap();
    assert!(clock.tick(150).is_ok());
    assert_eq!(
        *source.native.waits.lock().unwrap(),
        [
            Duration::from_micros(62_499),
            Duration::from_micros(40),
            Duration::from_nanos(1)
        ]
    );
    assert_eq!(source.reads.load(Ordering::Relaxed), 4);
    assert_eq!(source.native.calls.load(Ordering::Relaxed), 3);
    eprintln!("RAW short wakes: sleeps_ns=[62499000,40000,1] host_reads=4 native_samples=3");
}

#[test]
fn a_raw_deadline_already_reached_adds_no_sleep() {
    let startup = sample(100, 1_000_000_000);
    let first = sample(200, 1_000_001_000);
    let deadline = 1_062_500_000;
    let source = ScriptedHostSamples::new(
        [Ok(startup), Ok(first), Ok(sample(6_240_100, deadline))],
        [Ok(deadline)],
    );
    let clock = GpuClock::new(source.clone()).unwrap();
    assert!(clock.tick(150).is_ok());
    assert!(source.native.waits.lock().unwrap().is_empty());
    assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    assert_eq!(source.native.calls.load(Ordering::Relaxed), 3);
}

#[test]
fn raw_host_errors_or_regressions_propagate_without_another_gpu_sample() {
    let startup = sample(100, 1_000_000_000);
    let first = sample(200, 1_000_001_000);
    let latest = sample(10_000_100, 1_100_150_000);
    for reads in [
        vec![Err(INVALID_RUNTIME_STATE)],
        vec![Ok(first.host), Err(INVALID_RUNTIME_STATE)],
        vec![Ok(first.host - 1)],
        vec![Ok(first.host), Ok(first.host - 1)],
    ] {
        let expected = if reads.iter().any(Result::is_err) {
            INVALID_RUNTIME_STATE
        } else {
            ERROR
        };
        let expected_waits = reads.len() - 1;
        let source = ScriptedHostSamples::new([Ok(startup), Ok(first), Ok(latest)], reads);
        let clock = GpuClock::new(source.clone()).unwrap();
        assert_eq!(clock.tick(150), Err(expected));
        assert!(clock.calibration.lock().unwrap().is_none());
        assert_eq!(source.native.calls.load(Ordering::Relaxed), 2);
        assert_eq!(source.native.waits.lock().unwrap().len(), expected_waits);
        assert_eq!(clock.tick(latest.gpu), Ok(latest.host));
        assert_eq!(source.native.calls.load(Ordering::Relaxed), 3);
    }
}

#[test]
fn checked_raw_deadline_overflow_returns_before_host_waiting() {
    let startup = sample(100, u64::MAX - 10);
    let source = Samples::new([Ok(startup), Ok(startup)]);
    let clock = GpuClock::new(source.clone()).unwrap();
    assert_eq!(clock.tick(100), Err(ERROR));
    assert!(clock.calibration.lock().unwrap().is_none());
    assert!(source.waits.lock().unwrap().is_empty());
    assert_eq!(source.calls.load(Ordering::Relaxed), 2);
}

#[test]
fn post_deadline_native_host_must_follow_the_observed_raw_clock() {
    let startup = sample(100, 1_000_000_000);
    let source = ScriptedHostSamples::new(
        [
            Ok(startup),
            Ok(sample(200, 1_000_001_000)),
            Ok(sample(6_240_100, 1_062_505_000)),
            Ok(sample(10_000_100, 1_100_150_000)),
        ],
        [Ok(1_062_510_000)],
    );
    let clock = GpuClock::new(source.clone()).unwrap();
    assert_eq!(clock.tick(150), Err(ERROR));
    assert!(clock.calibration.lock().unwrap().is_none());
    assert_eq!(source.native.calls.load(Ordering::Relaxed), 3);
    assert_eq!(clock.tick(10_000_100), Ok(1_100_150_000));
    assert_eq!(source.native.calls.load(Ordering::Relaxed), 4);
}

/// Rational raw-clock progress during a duration measured by the sleep clock.
struct RateHostTime {
    /// Raw host nanoseconds already elapsed in this synthetic clock epoch.
    now: u64,
    /// Fractional nanoseconds retained across successive sleep requests.
    remainder: u128,
    /// Requested sleeps, each performed before checking the raw clock again.
    sleeps: Vec<Duration>,
}

/// Native-clock model with an independently scaled sleep timebase.
struct RateHostSamples {
    /// Raw-clock rate in millionths of the sleep clock's rate.
    numerator: u64,
    /// Shared host time advanced only by explicit sleep requests in this test.
    time: Mutex<RateHostTime>,
    /// Native-query count, independent of sleep and host-read counts.
    native_calls: AtomicUsize,
    /// Host observations made while awaiting the fixed raw deadline.
    host_reads: AtomicUsize,
}

impl ClockSource for Arc<RateHostSamples> {
    fn sample(&self) -> Result<ClockCounters, Status> {
        self.native_calls.fetch_add(1, Ordering::Relaxed);
        let host = self.time.lock().unwrap().now;
        Ok(sample(100 + (host - 1_000_000_000) / 10, host))
    }

    fn host_clock(&self) -> Result<u64, Status> {
        self.host_reads.fetch_add(1, Ordering::Relaxed);
        Ok(self.time.lock().unwrap().now)
    }

    fn sleep(&self, duration: Duration) {
        let mut time = self.time.lock().unwrap();
        let scaled = duration.as_nanos() * u128::from(self.numerator) + time.remainder;
        time.now += u64::try_from(scaled / 1_000_000).unwrap();
        time.remainder = scaled % 1_000_000;
        time.sleeps.push(duration);
    }
}

#[test]
fn differing_sleep_and_raw_rates_reach_one_raw_deadline_without_gpu_retries() {
    for numerator in [999_500, 1_000_500] {
        let source = Arc::new(RateHostSamples {
            numerator,
            time: Mutex::new(RateHostTime {
                now: 1_000_000_000,
                remainder: 0,
                sleeps: Vec::new(),
            }),
            native_calls: AtomicUsize::new(0),
            host_reads: AtomicUsize::new(0),
        });
        let clock = GpuClock::new(source.clone()).unwrap();
        source.time.lock().unwrap().now += 1_000;
        assert!(clock.tick(150).is_ok());
        let map = clock.calibration.lock().unwrap().unwrap();
        let time = source.time.lock().unwrap();
        let sleep_count = time.sleeps.len();
        assert!(map.latest.host - clock.startup.host >= 62_500_000);
        assert!(time.sleeps.iter().all(|duration| !duration.is_zero()));
        assert_eq!(source.native_calls.load(Ordering::Relaxed), 3);
        assert_eq!(source.host_reads.load(Ordering::Relaxed), sleep_count + 1);
        if numerator < 1_000_000 {
            assert!(sleep_count > 1);
        } else {
            assert_eq!(sleep_count, 1);
        }
        eprintln!(
            "RAW rate model: raw_per_sleep_millionths={numerator} sleeps_ns={:?} measured_host_span_ns={} native_samples=3",
            time.sleeps
                .iter()
                .map(Duration::as_nanos)
                .collect::<Vec<_>>(),
            map.latest.host - clock.startup.host
        );
    }
}

#[test]
fn captured_short_baselines_use_one_measured_first_map_in_both_query_orders() {
    let startup = sample(anchors::SHORT[0].0, anchors::SHORT[0].1);
    let measured = sample(anchors::SPAN[1].0, anchors::SPAN[1].1);
    let mut minimum_gap = u64::MAX;
    let mut cases = 0;
    for &(gpu, host) in anchors::SHORT.iter().skip(1) {
        let short = sample(gpu, host);
        let earlier_tick = gpu - 1_412;
        let later_tick = earlier_tick + 412;
        assert!(startup.gpu < earlier_tick && later_tick < short.gpu);
        for reverse in [false, true] {
            let source = Samples::new([
                Ok(startup),
                Ok(short),
                Ok(measured),
                Ok(measured),
                Ok(measured),
                Ok(measured),
            ]);
            let clock = GpuClock::new(source.clone()).unwrap();
            let (first, second) = if reverse {
                (later_tick, earlier_tick)
            } else {
                (earlier_tick, later_tick)
            };
            let first_value = clock.tick(first).unwrap();
            let frozen = clock.calibration.lock().unwrap().unwrap();
            let second_value = clock.tick(second).unwrap();
            let (earlier, later) = if reverse {
                (second_value, first_value)
            } else {
                (first_value, second_value)
            };
            assert_eq!(frozen.latest, measured);
            assert_eq!(frozen.translate(startup, earlier_tick), Ok(earlier));
            assert_eq!(frozen.translate(startup, later_tick), Ok(later));
            assert_eq!(clock.tick(earlier_tick), Ok(earlier));
            assert_eq!(clock.tick(later_tick), Ok(later));
            assert!(later > earlier);
            minimum_gap = minimum_gap.min(later - earlier);
            assert_eq!(
                *source.waits.lock().unwrap(),
                [Duration::from_nanos(62_500_000 - (host - startup.host))]
            );
            cases += 1;
        }
    }
    eprintln!(
        "captured short-baseline replay: cases={cases} minimum_ordered_gap_ns={minimum_gap} measured_span_host_ns={} measured_span_gpu_ticks={}",
        measured.host - startup.host,
        measured.gpu - startup.gpu
    );
    assert_eq!(cases, 254);
}

#[test]
fn captured_refresh_boundaries_and_retained_requeries_have_a_frozen_measured_control() {
    let startup = sample(anchors::SHORT[0].0, anchors::SHORT[0].1);
    let mut minimum_gap = i128::MAX;
    let mut maximum_retained_change = 0_u128;
    let mut cases = 0;
    // span[0] is too young; span[1] is the first adequate measured baseline.
    for index in 1..anchors::SPAN.len() - 1 {
        let previous = sample(anchors::SPAN[index].0, anchors::SPAN[index].1);
        let next = sample(anchors::SPAN[index + 1].0, anchors::SPAN[index + 1].1);
        for reverse in [false, true] {
            let source = Samples::new([
                Ok(startup),
                Ok(previous),
                Ok(next),
                Ok(next),
                Ok(next),
                Ok(next),
            ]);
            let clock = GpuClock::new(source.clone()).unwrap();
            assert_eq!(clock.tick(previous.gpu), Ok(previous.host));
            let frozen = clock.calibration.lock().unwrap().unwrap();
            let host_limit = rocddi::host::clock::info().frequency_hz / 16;
            let time_limit = u64::try_from(
                (u128::from(host_limit + 1) * u128::from(frozen.ratio.gpu) - 1)
                    / u128::from(frozen.ratio.host),
            )
            .unwrap();
            let earlier_tick = previous.gpu + ((previous.gpu - startup.gpu) / 4).min(time_limit);
            let later_tick = earlier_tick + 412;
            assert!(later_tick < next.gpu);
            assert!(!frozen.needs_refresh(startup, earlier_tick).unwrap());
            assert!(frozen.needs_refresh(startup, later_tick).unwrap());
            let (first, second) = if reverse {
                (later_tick, earlier_tick)
            } else {
                (earlier_tick, later_tick)
            };
            let first_value = clock.tick(first).unwrap();
            let second_value = clock.tick(second).unwrap();
            let (earlier, later) = if reverse {
                (second_value, first_value)
            } else {
                (first_value, second_value)
            };
            let retained_earlier = clock.tick(earlier_tick).unwrap();
            let retained_later = clock.tick(later_tick).unwrap();
            let frozen_gap = frozen.translate(startup, later_tick).unwrap()
                - frozen.translate(startup, earlier_tick).unwrap();
            let gap = i128::from(later) - i128::from(earlier);
            let retained_change = i128::from(retained_earlier) - i128::from(earlier);
            let selected = clock.calibration.lock().unwrap().unwrap();
            assert_eq!(selected.latest, next);
            assert_eq!(retained_later, later);
            assert!(gap > 0, "captured +412-tick boundary pair inverted");
            assert!(source.waits.lock().unwrap().is_empty());
            minimum_gap = minimum_gap.min(gap);
            maximum_retained_change = maximum_retained_change.max(retained_change.unsigned_abs());
            eprintln!(
                "captured refresh: span_index={index} reverse={reverse} old_ratio={}/{} new_ratio={}/{} ticks=({earlier_tick},{later_tick}) converted=({earlier},{later}) gap_ns={gap} frozen_gap_ns={frozen_gap} retained_earlier_change_ns={retained_change}",
                frozen.ratio.host, frozen.ratio.gpu, selected.ratio.host, selected.ratio.gpu
            );
            cases += 1;
        }
    }
    eprintln!(
        "captured refresh summary: cases={cases} minimum_ordered_gap_ns={minimum_gap} maximum_abs_retained_change_ns={maximum_retained_change}"
    );
    assert_eq!(cases, 38);
}

#[test]
fn captured_trajectory_keeps_the_first_measured_map_as_an_immutable_control() {
    let startup = sample(anchors::SHORT[0].0, anchors::SHORT[0].1);
    let short = sample(anchors::SHORT[127].0, anchors::SHORT[127].1);
    let measured = sample(anchors::SPAN[1].0, anchors::SPAN[1].1);
    let mut samples = vec![Ok(startup), Ok(short), Ok(measured)];
    for &(gpu, host) in anchors::SPAN.iter().skip(2) {
        samples.extend([Ok(sample(gpu, host)); 3]);
    }
    let source = Samples::new(samples);
    let clock = GpuClock::new(source.clone()).unwrap();
    let earlier_tick = short.gpu - 1_412;
    let later_tick = earlier_tick + 412;
    let first_later = clock.tick(later_tick).unwrap();
    let frozen = clock.calibration.lock().unwrap().unwrap();
    let mut maximum_control_difference = 0_u128;
    let mut minimum_cached_retained_gap = i128::MAX;
    for (index, &(gpu, host)) in anchors::SPAN.iter().enumerate().skip(2) {
        let interval = clock.interval(gpu - 1_412, gpu - 1_000).unwrap();
        let earlier = clock.tick(earlier_tick).unwrap();
        let later = clock.tick(later_tick).unwrap();
        assert!(interval.1 > interval.0 && later > earlier);
        let selected = clock.calibration.lock().unwrap().unwrap();
        assert_eq!(selected.latest, sample(gpu, host));
        let control = frozen.translate(startup, gpu - 1_000).unwrap();
        let control_difference = i128::from(interval.1) - i128::from(control);
        let cached_retained_gap = i128::from(first_later) - i128::from(earlier);
        assert!(
            cached_retained_gap > 0,
            "the retained earlier endpoint overtook the cached later endpoint"
        );
        maximum_control_difference =
            maximum_control_difference.max(control_difference.unsigned_abs());
        minimum_cached_retained_gap = minimum_cached_retained_gap.min(cached_retained_gap);
        eprintln!(
            "captured trajectory: span_index={index} end={} frozen_first_map_end={control} control_difference_ns={control_difference} retained_gap_ns={} cached_retained_gap_ns={cached_retained_gap}",
            interval.1,
            later - earlier
        );
    }
    eprintln!(
        "captured trajectory summary: refreshes=19 maximum_abs_frozen_control_difference_ns={maximum_control_difference} minimum_cached_retained_gap_ns={minimum_cached_retained_gap}"
    );
    assert_eq!(source.waits.lock().unwrap().len(), 1);
}

/// Increasing native observations for concurrent owner calls.
struct IncreasingSamples {
    /// Sequence also exposes the exact number of live native observations.
    calls: AtomicUsize,
}

impl ClockSource for Arc<IncreasingSamples> {
    fn sample(&self) -> Result<ClockCounters, Status> {
        let index = u64::try_from(self.calls.fetch_add(1, Ordering::Relaxed)).unwrap();
        Ok(sample(
            100 + index * 10_000_000,
            1_000_000_000 + index * 100_000_000,
        ))
    }
}

#[test]
fn concurrent_clones_share_one_serialized_owner_and_live_sampling() {
    let source = Arc::new(IncreasingSamples {
        calls: AtomicUsize::new(0),
    });
    let clock = Arc::new(GpuClock::new(source.clone()).unwrap());
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let clock = clock.clone();
            thread::spawn(move || clock.tick(5_000_100))
        })
        .collect();
    for worker in workers {
        assert_eq!(worker.join().unwrap(), Ok(1_050_000_000));
    }
    assert_eq!(source.calls.load(Ordering::Relaxed), 9);
}
