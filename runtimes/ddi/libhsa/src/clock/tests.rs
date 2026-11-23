// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host tests of the production clock owner with clock dependencies injected.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

mod anchors;
mod precision;

/// Scripted native dependency shared with the test's observation handle.
#[derive(Clone)]
struct Samples {
    /// Native results consumed lazily by the production conversion methods.
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
    fn sample(&self) -> Result<(ClockCounters, Duration), Status> {
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
        result.map(|sample| (sample, Duration::ZERO))
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
fn construction_defers_native_failures_and_unsupported_clocks_to_profiling() {
    for (initial, status) in [
        (Err(INVALID_RUNTIME_STATE), INVALID_RUNTIME_STATE),
        (Err(ERROR), ERROR),
        (
            Ok(ClockCounters {
                gpu_info: rocddi::host::clock::ClockInfo {
                    frequency_hz: 0,
                    counter_bit_count: 0,
                },
                ..sample(100, 1_000)
            }),
            ERROR,
        ),
        (
            Ok(ClockCounters {
                gpu_info: rocddi::host::clock::ClockInfo {
                    frequency_hz: 100_000_000,
                    counter_bit_count: 32,
                },
                ..sample(100, 1_000)
            }),
            NOT_SUPPORTED,
        ),
        (
            Ok(ClockCounters {
                gpu_info: rocddi::host::clock::ClockInfo {
                    frequency_hz: 100_000_000,
                    counter_bit_count: 65,
                },
                ..sample(100, 1_000)
            }),
            ERROR,
        ),
    ] {
        let source = Samples::new([
            initial,
            Ok(sample(100, 1_000_000_000)),
            Ok(sample(10_000_100, 1_100_150_000)),
        ]);
        let clock = GpuClock::new(source.clone());
        assert_eq!(source.calls.load(Ordering::Relaxed), 0);
        assert!(clock.state.lock().unwrap().is_none());
        assert_eq!(clock.interval(0, 101), Ok((0, 0)));
        assert_eq!(clock.interval(99, 0), Ok((0, 0)));
        assert_eq!(source.calls.load(Ordering::Relaxed), 0);
        assert_eq!(clock.tick(100), Err(status));
        assert_eq!(source.calls.load(Ordering::Relaxed), 1);
        assert!(clock.state.lock().unwrap().is_none());
        assert_eq!(clock.tick(5_000_100), Ok(1_050_075_000));
        assert_eq!(source.calls.load(Ordering::Relaxed), 3);
    }
}

#[test]
fn first_conversion_waits_once_for_a_measured_baseline_without_nominal_output() {
    let source = Samples::new([
        Ok(sample(100, 1_000_000_000)),
        // Three GPU ticks and 5 us of sequential-read noise are not a slope.
        Ok(sample(103, 1_000_005_000)),
        Ok(sample(8_000_100, 1_080_120_000)),
    ]);
    let clock = GpuClock::new(source.clone());
    assert!(clock.state.lock().unwrap().is_none());
    assert_eq!(clock.tick(102), Ok(1_000_000_020));
    let map = clock.state.lock().unwrap().as_ref().unwrap().ratio.unwrap();
    assert_eq!(
        clock.state.lock().unwrap().as_ref().unwrap().latest,
        sample(8_000_100, 1_080_120_000)
    );
    assert_eq!(map.host, 80_120_000);
    assert_eq!(map.gpu, 8_000_000);
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
    let clock = GpuClock::new(source.clone());
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
    ]));
    let interval = clock.interval(9_000_100, 10_000_100).unwrap();
    assert_eq!(interval, (1_090_135_000, 1_100_150_000));
    assert_eq!(clock.tick(10_000_100), Ok(interval.1));
    assert_eq!(clock.tick(9_000_100), Ok(interval.0));
}

#[test]
fn crossing_the_quarter_span_boundary_keeps_the_map_and_records_live_samples() {
    let startup = sample(100, 1_000_000_000);
    let latest = sample(10_000_100, 1_100_000_000);
    let clock = GpuClock::new(Samples::new([
        Ok(startup),
        Ok(latest),
        Ok(sample(12_600_100, 1_125_900_000)),
        Ok(sample(13_000_100, 1_129_800_000)),
    ]));
    assert_eq!(clock.tick(latest.gpu), Ok(latest.host));
    assert_eq!(clock.tick(12_500_100), Ok(1_125_000_000));
    let ratio = clock.state.lock().unwrap().as_ref().unwrap().ratio.unwrap();
    assert_eq!(
        clock.state.lock().unwrap().as_ref().unwrap().latest,
        sample(12_600_100, 1_125_900_000)
    );
    assert_eq!(clock.tick(12_500_101), Ok(1_125_000_010));
    assert_eq!(
        clock.state.lock().unwrap().as_ref().unwrap().latest.gpu,
        13_000_100
    );
    assert_eq!(
        clock.state.lock().unwrap().as_ref().unwrap().ratio,
        Some(ratio)
    );
}

#[test]
fn crossing_the_host_horizon_keeps_the_map_and_records_live_samples() {
    let startup = sample(100, 1_000_000_000);
    let latest = sample(100_000_100, 2_000_000_000);
    let clock = GpuClock::new(Samples::new([
        Ok(startup),
        Ok(latest),
        Ok(sample(106_300_100, 2_062_900_000)),
        Ok(sample(106_400_100, 2_063_800_000)),
    ]));
    assert_eq!(clock.tick(latest.gpu), Ok(latest.host));
    assert_eq!(clock.tick(106_250_100), Ok(2_062_500_000));
    let ratio = clock.state.lock().unwrap().as_ref().unwrap().ratio.unwrap();
    assert_eq!(
        clock.state.lock().unwrap().as_ref().unwrap().latest,
        sample(106_300_100, 2_062_900_000)
    );
    assert_eq!(clock.tick(106_250_101), Ok(2_062_500_010));
    assert_eq!(
        clock.state.lock().unwrap().as_ref().unwrap().latest.gpu,
        106_400_100
    );
    assert_eq!(
        clock.state.lock().unwrap().as_ref().unwrap().ratio,
        Some(ratio)
    );
}

#[test]
fn first_measured_map_removes_the_synthetic_nominal_bootstrap_inversion() {
    let clock = GpuClock::new(Samples::new([
        Ok(sample(100, 1_000_000_000)),
        Ok(sample(6_249_100, 1_062_490_000)),
        Ok(sample(6_260_100, 1_062_590_000)),
        Ok(sample(6_260_100, 1_062_590_000)),
    ]));
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
    let clock = GpuClock::new(source.clone());
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
fn later_slope_changes_preserve_cached_close_endpoints_in_both_query_orders() {
    // These increasing anchors used to move a cached +412-tick pair by -95,881 ns.
    let latest = sample(100_000_100, 2_000_000_000);
    let refreshed = sample(125_000_100, 2_249_875_000);
    for reverse in [false, true] {
        let source = Samples::new([
            Ok(sample(100, 1_000_000_000)),
            Ok(latest),
            Ok(refreshed),
            Ok(refreshed),
            Ok(refreshed),
            Ok(refreshed),
        ]);
        let clock = GpuClock::new(source.clone());
        let (first, second) = if reverse {
            (latest.gpu + 412, latest.gpu)
        } else {
            (latest.gpu, latest.gpu + 412)
        };
        let first_value = clock.tick(first).unwrap();
        assert_eq!(clock.tick(110_000_100), Ok(2_100_000_000));
        let second_value = clock.tick(second).unwrap();
        let (earlier, later) = if reverse {
            (second_value, first_value)
        } else {
            (first_value, second_value)
        };
        assert_eq!(earlier, 2_000_000_000);
        assert_eq!(later, 2_000_004_120);
        assert!(later > earlier);
        assert_eq!(clock.tick(first), Ok(first_value));
        assert_eq!(
            clock.interval(latest.gpu, latest.gpu + 412),
            Ok((earlier, later))
        );
        assert_eq!(
            clock.state.lock().unwrap().as_ref().unwrap().latest,
            refreshed
        );
        assert_eq!(source.calls.load(Ordering::Relaxed), 6);
        eprintln!(
            "retained slope witness: reverse={reverse} gap_ns={} repeated_first_change_ns=0",
            later - earlier
        );
    }
}

#[test]
fn pre_startup_and_later_ticks_keep_their_first_measured_ratio() {
    let latest = sample(11_000_100, 1_100_150_000);
    let refreshed = sample(21_000_100, 1_200_200_000);
    let clock = GpuClock::new(Samples::new([
        Ok(sample(1_000_100, 1_000_000_000)),
        Ok(latest),
        Ok(refreshed),
        Ok(refreshed),
    ]));
    assert_eq!(clock.tick(100), Ok(989_985_000));
    assert_eq!(clock.tick(refreshed.gpu), Ok(1_200_300_000));
    assert_eq!(clock.tick(100), Ok(989_985_000));
}

#[test]
fn live_failure_never_returns_a_cached_success_and_is_retryable() {
    for status in [ERROR, INVALID_RUNTIME_STATE] {
        let startup = sample(100, 1_000_000_000);
        let latest = sample(10_000_100, 1_100_000_000);
        let source = Samples::new([Ok(startup), Ok(latest), Err(status), Ok(latest)]);
        let clock = GpuClock::new(source.clone());
        let value = clock.tick(9_000_100).unwrap();
        let ratio = clock.state.lock().unwrap().as_ref().unwrap().ratio;
        assert_eq!(clock.tick(9_000_100), Err(status));
        assert_eq!(clock.state.lock().unwrap().as_ref().unwrap().latest, latest);
        assert_eq!(clock.state.lock().unwrap().as_ref().unwrap().ratio, ratio);
        assert_eq!(clock.tick(9_000_100), Ok(value));
        assert_eq!(source.calls.load(Ordering::Relaxed), 4);
    }
}

#[test]
fn native_contradictions_before_first_map_permanently_end_the_epoch() {
    let startup = sample(100, 1_000_000_000);
    let latest = sample(10_000_100, 1_100_000_000);
    for invalid in [
        sample(99, startup.host + 1),
        sample(101, startup.host - 1),
        ClockCounters {
            gpu_info: rocddi::host::clock::ClockInfo {
                frequency_hz: 200_000_000,
                counter_bit_count: 0,
            },
            ..sample(101, startup.host + 1)
        },
    ] {
        let source = Samples::new([Ok(startup), Ok(invalid), Ok(latest)]);
        let clock = GpuClock::new(source.clone());
        assert_eq!(clock.tick(1_000_100), Err(ERROR));
        {
            let state = clock.state.lock().unwrap();
            let state = state.as_ref().unwrap();
            assert!(state.invalid && state.ratio.is_none());
            assert_eq!(state.latest, startup);
        }
        // Even a later native value above the old high-water mark cannot admit
        // this owner again; a contradictory observation already ended its epoch.
        assert_eq!(clock.tick(latest.gpu), Err(ERROR));
        assert_eq!(clock.interval(latest.gpu, latest.gpu + 412), Err(ERROR));
        assert_eq!(source.calls.load(Ordering::Relaxed), 2);
    }
}

#[test]
fn absent_initial_gpu_progress_does_not_prove_epoch_loss() {
    let startup = sample(100, 1_000_000_000);
    let latest = sample(10_000_100, 1_100_000_000);
    let source = Samples::new([
        Ok(startup),
        Ok(sample(startup.gpu, latest.host)),
        Ok(latest),
    ]);
    let clock = GpuClock::new(source.clone());
    assert_eq!(clock.tick(1_000_100), Err(ERROR));
    {
        let state = clock.state.lock().unwrap();
        let state = state.as_ref().unwrap();
        assert!(!state.invalid && state.ratio.is_none());
    }
    assert_eq!(clock.tick(latest.gpu), Ok(latest.host));
    assert_eq!(source.calls.load(Ordering::Relaxed), 3);
}

#[test]
fn transient_native_failure_before_first_map_preserves_the_valid_anchor() {
    for status in [ERROR, INVALID_RUNTIME_STATE] {
        let startup = sample(100, 1_000_000_000);
        let latest = sample(10_000_100, 1_100_000_000);
        let source = Samples::new([Ok(startup), Err(status), Ok(latest)]);
        let clock = GpuClock::new(source.clone());
        assert_eq!(clock.tick(1_000_100), Err(status));
        {
            let state = clock.state.lock().unwrap();
            let state = state.as_ref().unwrap();
            assert!(!state.invalid && state.ratio.is_none());
            assert_eq!(state.startup, startup);
        }
        assert_eq!(clock.tick(latest.gpu), Ok(latest.host));
        assert_eq!(source.calls.load(Ordering::Relaxed), 3);
    }
}

#[test]
fn published_map_never_reenters_after_native_contradiction_and_catchup() {
    let startup = sample(100, 1_000_000_000);
    let measured = sample(10_000_100, 1_100_000_000);
    let observed = sample(30_000_100, 1_299_900_000);
    let caught_up = sample(observed.gpu + 200, observed.host + 2_000);
    let next = sample(caught_up.gpu + 10_000_000, caught_up.host + 100_000_000);
    let invalid_metadata = |frequency_hz, counter_bit_count| ClockCounters {
        gpu_info: rocddi::host::clock::ClockInfo {
            frequency_hz,
            counter_bit_count,
        },
        ..caught_up
    };
    for (invalid, status) in [
        (sample(observed.gpu - 1, observed.host + 1), ERROR),
        (sample(observed.gpu + 1, observed.host - 1), ERROR),
        (invalid_metadata(0, 0), ERROR),
        (invalid_metadata(200_000_000, 0), ERROR),
        (invalid_metadata(100_000_000, 32), NOT_SUPPORTED),
        (invalid_metadata(100_000_000, 64), ERROR),
        (invalid_metadata(100_000_000, 65), ERROR),
    ] {
        let source = Samples::new([
            Ok(startup),
            Ok(measured),
            Ok(observed),
            Ok(invalid),
            Ok(caught_up),
            Ok(next),
        ]);
        let clock = GpuClock::new(source.clone());
        let value = clock.tick(9_000_100).unwrap();
        let ratio = clock.state.lock().unwrap().as_ref().unwrap().ratio;
        assert_eq!(clock.tick(9_000_100), Ok(value));
        assert_eq!(clock.tick(9_000_100), Err(status));
        {
            let state = clock.state.lock().unwrap();
            let state = state.as_ref().unwrap();
            assert!(state.invalid);
            assert_eq!(state.latest, observed);
            assert_eq!(state.ratio, ratio);
        }
        // Caught-up counters remain queued, so refusing more reads also proves
        // that neither query order nor the new epoch's progress can reopen it.
        assert_eq!(clock.tick(caught_up.gpu), Err(ERROR));
        assert_eq!(clock.tick(9_000_100), Err(ERROR));
        assert_eq!(clock.interval(caught_up.gpu + 412, 9_000_100), Err(ERROR));
        assert_eq!(source.calls.load(Ordering::Relaxed), 4);
        let fresh = GpuClock::new(source.clone());
        assert_eq!(
            fresh.interval(caught_up.gpu, next.gpu),
            Ok((caught_up.host, next.host))
        );
        assert_eq!(source.calls.load(Ordering::Relaxed), 6);
        // Even a fresh owner's success cannot revive the invalidated owner.
        assert_eq!(clock.tick(caught_up.gpu), Err(ERROR));
        assert_eq!(source.calls.load(Ordering::Relaxed), 6);
    }
}

#[test]
fn failed_output_arithmetic_keeps_native_high_water_for_epoch_validation() {
    let startup = sample(100, 1_000_000_000);
    let measured = sample(10_000_100, 1_100_000_000);
    let observed = sample(30_000_100, 1_299_900_000);
    for initialized in [false, true] {
        let latest = if initialized { observed } else { measured };
        let regressed = sample(latest.gpu - 1, latest.host + 1);
        let caught_up = sample(latest.gpu + 100, latest.host + 1_000);
        let mut values = vec![Ok(startup), Ok(measured)];
        if initialized {
            values.push(Ok(observed));
        }
        values.extend([Ok(regressed), Ok(caught_up)]);
        let source = Samples::new(values);
        let clock = GpuClock::new(source.clone());
        if initialized {
            assert_eq!(clock.tick(1_000_100), Ok(1_010_000_000));
        }
        let before = clock
            .state
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|state| state.ratio);
        // The end converts first; failure of the start must not publish a map.
        assert_eq!(clock.interval(u64::MAX, 1_000_100), Err(ERROR));
        {
            let state = clock.state.lock().unwrap();
            let state = state.as_ref().unwrap();
            assert_eq!(state.latest, latest);
            assert_eq!(state.ratio, before);
            assert!(!state.invalid);
        }
        assert_eq!(clock.tick(1_000_100), Err(ERROR));
        assert!(clock.state.lock().unwrap().as_ref().unwrap().invalid);
        assert_eq!(clock.tick(1_000_100), Err(ERROR));
        assert_eq!(
            source.calls.load(Ordering::Relaxed),
            if initialized { 4 } else { 3 }
        );
    }
}

#[test]
fn output_arithmetic_failure_does_not_invalidate_a_healthy_epoch() {
    let startup = sample(100, 1_000_000_000);
    let measured = sample(10_000_100, 1_100_000_000);
    let source = Samples::new([Ok(startup), Ok(measured), Ok(measured)]);
    let clock = GpuClock::new(source.clone());
    assert_eq!(clock.interval(u64::MAX, 1_000_100), Err(ERROR));
    assert!(!clock.state.lock().unwrap().as_ref().unwrap().invalid);
    assert_eq!(clock.tick(1_000_100), Ok(1_010_000_000));
    assert_eq!(source.calls.load(Ordering::Relaxed), 3);
}

#[test]
fn arbitrary_future_ticks_and_reversed_intervals_share_the_immutable_map() {
    let measured = sample(10_000_100, 1_100_000_000);
    let later = sample(125_000_100, 2_249_875_000);
    let source = Samples::new([
        Ok(sample(100, 1_000_000_000)),
        Ok(measured),
        Ok(later),
        Ok(later),
        Ok(later),
        Ok(later),
    ]);
    let clock = GpuClock::new(source.clone());
    let future_tick = 1_000_000_000_100;
    let future_value = clock.tick(future_tick).unwrap();
    assert_eq!(future_value, 10_001_000_000_000);
    let earlier = clock.tick(measured.gpu).unwrap();
    let later_value = clock.tick(measured.gpu + 412).unwrap();
    assert_eq!(earlier, measured.host);
    assert_eq!(later_value - earlier, 4_120);
    assert_eq!(clock.tick(future_tick), Ok(future_value));
    assert_eq!(
        clock.interval(measured.gpu + 412, measured.gpu),
        Ok((later_value, earlier))
    );
    assert_eq!(clock.state.lock().unwrap().as_ref().unwrap().latest, later);
    assert_eq!(source.calls.load(Ordering::Relaxed), 6);
}

#[test]
fn checked_conversion_overflows_remain_errors_and_do_not_publish_calibration() {
    let source = Samples::new([
        Ok(sample(100, u64::MAX - 100_000_000)),
        Ok(sample(10_000_100, u64::MAX)),
    ]);
    let clock = GpuClock::new(source);
    assert_eq!(clock.tick(10_000_101), Err(ERROR));
    assert!(
        clock
            .state
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .ratio
            .is_none()
    );
    let clock = GpuClock::new(Samples::new([
        Ok(sample(100_000_000, 1_000_000)),
        Ok(sample(110_000_000, 101_000_000)),
    ]));
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
    ]));
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
    ]));
    assert_eq!(clock.tick(1 << 63), Err(ERROR));
}

#[test]
fn absent_interval_endpoints_keep_the_zero_sentinel_without_sampling() {
    let source = Samples::new([Ok(sample(100, 1_000))]);
    let clock = GpuClock::new(source.clone());
    assert_eq!(clock.interval(0, 101), Ok((0, 0)));
    assert_eq!(clock.interval(99, 0), Ok((0, 0)));
    assert_eq!(source.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn first_query_uses_an_adequate_startup_age_without_waiting() {
    let source = Samples::new([
        Ok(sample(100, 1_000_000_000)),
        Ok(sample(10_000_100, 1_100_150_000)),
    ]);
    let clock = GpuClock::new(source.clone());
    assert_eq!(clock.tick(5_000_100), Ok(1_050_075_000));
    assert!(source.waits.lock().unwrap().is_empty());
    assert_eq!(source.calls.load(Ordering::Relaxed), 2);
}

#[test]
fn post_deadline_native_host_shortfall_permanently_invalidates_the_epoch() {
    let startup = sample(100, 1_000_000_000);
    let caught_up = sample(10_000_100, 1_100_150_000);
    let source = Samples::new([Ok(startup), Ok(startup), Ok(startup), Ok(caught_up)]);
    let clock = GpuClock::new(source.clone());
    assert_eq!(clock.tick(101), Err(ERROR));
    {
        let state = clock.state.lock().unwrap();
        let state = state.as_ref().unwrap();
        assert!(state.invalid && state.ratio.is_none());
    }
    assert_eq!(
        *source.waits.lock().unwrap(),
        [Duration::from_micros(62_500)]
    );
    assert_eq!(clock.tick(caught_up.gpu), Err(ERROR));
    assert_eq!(source.calls.load(Ordering::Relaxed), 3);
    assert_eq!(source.waits.lock().unwrap().len(), 1);
}

#[test]
fn post_wait_transient_failure_recovers_but_native_regression_ends_the_epoch() {
    let startup = sample(100, 1_000_000_000);
    let first = sample(200, 1_000_001_000);
    let latest = sample(10_000_100, 1_100_150_000);
    for bad in [Err(INVALID_RUNTIME_STATE), Ok(sample(199, 1_070_000_000))] {
        let transient = bad.is_err();
        let expected = if transient {
            INVALID_RUNTIME_STATE
        } else {
            ERROR
        };
        let source = Samples::new([Ok(startup), Ok(first), bad, Ok(latest)]);
        let clock = GpuClock::new(source.clone());
        assert_eq!(clock.tick(150), Err(expected));
        {
            let state = clock.state.lock().unwrap();
            let state = state.as_ref().unwrap();
            assert!(state.ratio.is_none());
            assert_eq!(state.invalid, !transient);
        }
        assert_eq!(source.waits.lock().unwrap().len(), 1);
        assert_eq!(
            clock.tick(latest.gpu),
            if transient {
                Ok(latest.host)
            } else {
                Err(ERROR)
            }
        );
        assert_eq!(
            source.calls.load(Ordering::Relaxed),
            if transient { 4 } else { 3 }
        );
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
    fn sample(&self) -> Result<(ClockCounters, Duration), Status> {
        let host = rocddi::host::clock::read().map_err(map_error)?;
        let counters = sample(100 + (host - self.origin) / 10, host);
        self.samples.lock().unwrap().push(counters);
        Ok((counters, Duration::ZERO))
    }
}

#[test]
fn cold_calibration_measures_the_production_wait_on_the_host() {
    let source = Arc::new(RealWaitSamples {
        origin: rocddi::host::clock::read().unwrap(),
        samples: Mutex::new(Vec::new()),
    });
    let clock = GpuClock::new(source.clone());
    let raw_before = rocddi::host::clock::read().unwrap();
    let before = Instant::now();
    assert!(clock.tick(100).unwrap() > 0);
    let elapsed = before.elapsed();
    let raw_elapsed = rocddi::host::clock::read().unwrap() - raw_before;
    let span = clock
        .state
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .ratio
        .unwrap()
        .host;
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
    fn sample(&self) -> Result<(ClockCounters, Duration), Status> {
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
fn interrupted_native_sample_retains_the_raw_witness_and_rejects_a_behind_retry() {
    let startup = sample(100, 1_000_000_000);
    let first = sample(200, 1_000_001_000);
    let raw = 1_062_510_000;
    let behind = sample(6_240_100, 1_062_505_000);
    let caught_up = sample(10_000_100, 1_100_150_000);
    let next = sample(20_000_100, 1_200_150_000);
    let source = ScriptedHostSamples::new(
        [
            Ok(startup),
            Ok(first),
            Err(INVALID_RUNTIME_STATE),
            Ok(behind),
            Ok(caught_up),
            Ok(next),
        ],
        [Ok(raw)],
    );
    let clock = GpuClock::new(source.clone());
    assert_eq!(clock.tick(150), Err(INVALID_RUNTIME_STATE));
    {
        let state = clock.state.lock().unwrap();
        let state = state.as_ref().unwrap();
        assert_eq!(state.latest, first);
        assert_eq!(state.host_high_water, raw);
        assert!(!state.invalid && state.ratio.is_none());
    }
    // Both counters exceed the latest native pair and the host horizon, but
    // the retained RAW read proves this native host value is in the past.
    assert!(behind.gpu > first.gpu && behind.host > first.host);
    assert!(behind.host - startup.host >= 62_500_000);
    assert_eq!(clock.tick(behind.gpu), Err(ERROR));
    {
        let state = clock.state.lock().unwrap();
        let state = state.as_ref().unwrap();
        assert_eq!(state.host_high_water, raw);
        assert!(state.invalid && state.ratio.is_none());
    }
    assert_eq!(clock.tick(caught_up.gpu), Err(ERROR));
    assert_eq!(source.native.calls.load(Ordering::Relaxed), 4);
    let fresh = GpuClock::new(source.clone());
    assert_eq!(
        fresh.interval(caught_up.gpu, next.gpu),
        Ok((caught_up.host, next.host))
    );
    assert_eq!(clock.tick(caught_up.gpu), Err(ERROR));
    assert_eq!(source.native.calls.load(Ordering::Relaxed), 6);
    assert_eq!(source.reads.load(Ordering::Relaxed), 1);
}

#[test]
fn interrupted_native_sample_recovers_only_at_or_above_the_retained_raw_witness() {
    let raw = 1_062_510_000;
    for status in [ERROR, INVALID_RUNTIME_STATE] {
        let recovered = sample(6_240_100, raw + 1);
        let source = ScriptedHostSamples::new(
            [
                Ok(sample(100, 1_000_000_000)),
                Ok(sample(200, 1_000_001_000)),
                Err(status),
                Ok(recovered),
                Ok(recovered),
                Ok(recovered),
            ],
            [Ok(raw)],
        );
        let clock = GpuClock::new(source.clone());
        assert_eq!(clock.tick(150), Err(status));
        assert_eq!(
            clock
                .state
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .host_high_water,
            raw
        );
        assert_eq!(clock.tick(recovered.gpu), Ok(recovered.host));
        let reversed = clock.interval(recovered.gpu, 150).unwrap();
        assert_eq!(reversed.0, recovered.host);
        assert!(reversed.1 < reversed.0);
        assert_eq!(clock.tick(150), Ok(reversed.1));
        {
            let state = clock.state.lock().unwrap();
            let state = state.as_ref().unwrap();
            assert_eq!(state.host_high_water, recovered.host);
            assert!(!state.invalid && state.ratio.is_some());
        }
        assert_eq!(source.native.calls.load(Ordering::Relaxed), 6);
        assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn interrupted_host_read_retains_an_intermediate_raw_witness() {
    let startup = sample(100, 1_000_000_000);
    let first = sample(200, 1_000_001_000);
    let raw = 1_050_000_000;
    let caught_up = sample(10_000_100, 1_100_150_000);
    for behind in [true, false] {
        let retry = sample(4_900_100, if behind { raw - 1_000 } else { raw + 1 });
        let source = ScriptedHostSamples::new(
            [Ok(startup), Ok(first), Ok(retry), Ok(caught_up)],
            [Ok(raw), Err(INVALID_RUNTIME_STATE), Ok(1_062_500_000)],
        );
        let clock = GpuClock::new(source.clone());
        assert_eq!(clock.tick(150), Err(INVALID_RUNTIME_STATE));
        {
            let state = clock.state.lock().unwrap();
            let state = state.as_ref().unwrap();
            assert_eq!(state.host_high_water, raw);
            assert_eq!(state.latest, first);
            assert!(!state.invalid && state.ratio.is_none());
        }
        assert_eq!(
            *source.native.waits.lock().unwrap(),
            [Duration::from_micros(12_500)]
        );
        assert_eq!(
            clock.tick(caught_up.gpu),
            if behind {
                Err(ERROR)
            } else {
                Ok(caught_up.host)
            }
        );
        if behind {
            assert_eq!(clock.tick(first.gpu), Err(ERROR));
            assert_eq!(source.native.calls.load(Ordering::Relaxed), 3);
            assert_eq!(source.reads.load(Ordering::Relaxed), 2);
        } else {
            let state = clock.state.lock().unwrap();
            let state = state.as_ref().unwrap();
            assert_eq!(state.host_high_water, caught_up.host);
            assert!(!state.invalid && state.ratio.is_some());
            assert_eq!(source.native.calls.load(Ordering::Relaxed), 4);
            assert_eq!(source.reads.load(Ordering::Relaxed), 3);
        }
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
    let clock = GpuClock::new(source.clone());
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
    let clock = GpuClock::new(source.clone());
    assert!(clock.tick(150).is_ok());
    assert!(source.native.waits.lock().unwrap().is_empty());
    assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    assert_eq!(source.native.calls.load(Ordering::Relaxed), 3);
}

#[test]
fn raw_host_errors_recover_but_observed_regressions_end_the_epoch() {
    let startup = sample(100, 1_000_000_000);
    let first = sample(200, 1_000_001_000);
    let latest = sample(10_000_100, 1_100_150_000);
    for reads in [
        vec![Err(INVALID_RUNTIME_STATE)],
        vec![Ok(first.host), Err(INVALID_RUNTIME_STATE)],
        vec![Ok(first.host - 1)],
        vec![Ok(first.host), Ok(first.host - 1)],
    ] {
        let transient = reads.iter().any(Result::is_err);
        let expected = if transient {
            INVALID_RUNTIME_STATE
        } else {
            ERROR
        };
        let expected_waits = reads.len() - 1;
        let source = ScriptedHostSamples::new([Ok(startup), Ok(first), Ok(latest)], reads);
        let clock = GpuClock::new(source.clone());
        assert_eq!(clock.tick(150), Err(expected));
        {
            let state = clock.state.lock().unwrap();
            let state = state.as_ref().unwrap();
            assert!(state.ratio.is_none());
            assert_eq!(state.invalid, !transient);
        }
        assert_eq!(source.native.calls.load(Ordering::Relaxed), 2);
        assert_eq!(source.native.waits.lock().unwrap().len(), expected_waits);
        assert_eq!(
            clock.tick(latest.gpu),
            if transient {
                Ok(latest.host)
            } else {
                Err(ERROR)
            }
        );
        assert_eq!(
            source.native.calls.load(Ordering::Relaxed),
            if transient { 3 } else { 2 }
        );
    }
}

#[test]
fn checked_raw_deadline_overflow_returns_before_host_waiting() {
    let startup = sample(100, u64::MAX - 10);
    let source = Samples::new([Ok(startup), Ok(startup)]);
    let clock = GpuClock::new(source.clone());
    assert_eq!(clock.tick(100), Err(ERROR));
    assert!(
        clock
            .state
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .ratio
            .is_none()
    );
    assert!(source.waits.lock().unwrap().is_empty());
    assert_eq!(source.calls.load(Ordering::Relaxed), 2);
}

#[test]
fn native_host_behind_observed_raw_clock_permanently_ends_the_epoch() {
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
    let clock = GpuClock::new(source.clone());
    assert_eq!(clock.tick(150), Err(ERROR));
    {
        let state = clock.state.lock().unwrap();
        let state = state.as_ref().unwrap();
        assert!(state.invalid && state.ratio.is_none());
    }
    assert_eq!(source.native.calls.load(Ordering::Relaxed), 3);
    assert_eq!(clock.tick(10_000_100), Err(ERROR));
    assert_eq!(source.native.calls.load(Ordering::Relaxed), 3);
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
    fn sample(&self) -> Result<(ClockCounters, Duration), Status> {
        self.native_calls.fetch_add(1, Ordering::Relaxed);
        let host = self.time.lock().unwrap().now;
        Ok((
            sample(100 + (host - 1_000_000_000) / 10, host),
            Duration::ZERO,
        ))
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
        let clock = GpuClock::new(source.clone());
        source.time.lock().unwrap().now += 1_000;
        assert!(clock.tick(150).is_ok());
        let map = clock.state.lock().unwrap().as_ref().unwrap().ratio.unwrap();
        let time = source.time.lock().unwrap();
        let sleep_count = time.sleeps.len();
        assert!(map.host >= 62_500_000);
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
            map.host
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
            let clock = GpuClock::new(source.clone());
            let (first, second) = if reverse {
                (later_tick, earlier_tick)
            } else {
                (earlier_tick, later_tick)
            };
            let first_value = clock.tick(first).unwrap();
            let frozen = clock.state.lock().unwrap().as_ref().unwrap().ratio.unwrap();
            let second_value = clock.tick(second).unwrap();
            let (earlier, later) = if reverse {
                (second_value, first_value)
            } else {
                (first_value, second_value)
            };
            assert_eq!(
                clock.state.lock().unwrap().as_ref().unwrap().latest,
                measured
            );
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
fn captured_old_refresh_boundaries_keep_cached_values_exactly_stable() {
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
            let clock = GpuClock::new(source.clone());
            assert_eq!(clock.tick(previous.gpu), Ok(previous.host));
            let frozen = clock.state.lock().unwrap().as_ref().unwrap().ratio.unwrap();
            let host_limit = rocddi::host::clock::info().frequency_hz / 16;
            let time_limit = u64::try_from(
                (u128::from(host_limit + 1) * u128::from(frozen.gpu) - 1) / u128::from(frozen.host),
            )
            .unwrap();
            let earlier_tick = previous.gpu + ((previous.gpu - startup.gpu) / 4).min(time_limit);
            let later_tick = earlier_tick + 412;
            assert!(later_tick < next.gpu);
            let earlier_extrapolation = earlier_tick - previous.gpu;
            let later_extrapolation = later_tick - previous.gpu;
            assert!(earlier_extrapolation <= (previous.gpu - startup.gpu) / 4);
            assert!(frozen.scale(earlier_extrapolation).unwrap() <= host_limit);
            assert!(
                later_extrapolation > (previous.gpu - startup.gpu) / 4
                    || frozen.scale(later_extrapolation).unwrap() > host_limit
            );
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
            let selected = clock.state.lock().unwrap().as_ref().unwrap().ratio.unwrap();
            assert_eq!(clock.state.lock().unwrap().as_ref().unwrap().latest, next);
            assert_eq!(selected, frozen);
            assert_eq!(earlier, frozen.translate(startup, earlier_tick).unwrap());
            assert_eq!(later, frozen.translate(startup, later_tick).unwrap());
            assert_eq!(retained_earlier, earlier);
            assert_eq!(retained_later, later);
            assert!(gap > 0, "captured +412-tick boundary pair inverted");
            assert!(source.waits.lock().unwrap().is_empty());
            minimum_gap = minimum_gap.min(gap);
            maximum_retained_change = maximum_retained_change.max(retained_change.unsigned_abs());
            eprintln!(
                "captured old refresh boundary: span_index={index} reverse={reverse} ratio={}/{} ticks=({earlier_tick},{later_tick}) converted=({earlier},{later}) gap_ns={gap} frozen_gap_ns={frozen_gap} retained_earlier_change_ns={retained_change}",
                frozen.host, frozen.gpu
            );
            cases += 1;
        }
    }
    eprintln!(
        "captured refresh summary: cases={cases} minimum_ordered_gap_ns={minimum_gap} maximum_abs_retained_change_ns={maximum_retained_change}"
    );
    assert_eq!(cases, 38);
    assert_eq!(maximum_retained_change, 0);
}

#[test]
fn captured_trajectory_keeps_the_first_measured_map_as_an_immutable_control() {
    let startup = sample(anchors::SHORT[0].0, anchors::SHORT[0].1);
    let short = sample(anchors::SHORT[127].0, anchors::SHORT[127].1);
    let measured = sample(anchors::SPAN[1].0, anchors::SPAN[1].1);
    let mut samples = vec![Ok(startup), Ok(short), Ok(measured), Ok(measured)];
    for &(gpu, host) in anchors::SPAN.iter().skip(2) {
        samples.extend([Ok(sample(gpu, host)); 3]);
    }
    let source = Samples::new(samples);
    let clock = GpuClock::new(source.clone());
    let earlier_tick = short.gpu - 1_412;
    let later_tick = earlier_tick + 412;
    let first_later = clock.tick(later_tick).unwrap();
    let first_earlier = clock.tick(earlier_tick).unwrap();
    let frozen = clock.state.lock().unwrap().as_ref().unwrap().ratio.unwrap();
    let mut maximum_control_difference = 0_u128;
    let mut minimum_cached_retained_gap = i128::MAX;
    for (index, &(gpu, host)) in anchors::SPAN.iter().enumerate().skip(2) {
        let interval = clock.interval(gpu - 1_412, gpu - 1_000).unwrap();
        let earlier = clock.tick(earlier_tick).unwrap();
        let later = clock.tick(later_tick).unwrap();
        assert_eq!(earlier, first_earlier);
        assert!(interval.1 > interval.0 && later > earlier);
        let selected = clock.state.lock().unwrap().as_ref().unwrap().ratio.unwrap();
        assert_eq!(
            clock.state.lock().unwrap().as_ref().unwrap().latest,
            sample(gpu, host)
        );
        assert_eq!(selected, frozen);
        let control = frozen.translate(startup, gpu - 1_000).unwrap();
        let control_difference = i128::from(interval.1) - i128::from(control);
        assert_eq!(control_difference, 0);
        assert_eq!(later, first_later);
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
        "captured trajectory summary: later_samples=19 maximum_abs_frozen_control_difference_ns={maximum_control_difference} minimum_cached_retained_gap_ns={minimum_cached_retained_gap}"
    );
    assert_eq!(maximum_control_difference, 0);
    assert_eq!(source.waits.lock().unwrap().len(), 1);
}

/// Increasing native observations for concurrent owner calls.
struct IncreasingSamples {
    /// Sequence also exposes the exact number of live native observations.
    calls: AtomicUsize,
}

impl ClockSource for Arc<IncreasingSamples> {
    fn sample(&self) -> Result<(ClockCounters, Duration), Status> {
        let index = u64::try_from(self.calls.fetch_add(1, Ordering::Relaxed)).unwrap();
        Ok((
            sample(
                100 + index * 10_000_000,
                1_000_000_000 + index * 100_000_000 - index.saturating_sub(1) * 12_500,
            ),
            Duration::ZERO,
        ))
    }
}

#[test]
fn concurrent_reverse_and_future_queries_share_one_map_despite_changing_samples() {
    let source = Arc::new(IncreasingSamples {
        calls: AtomicUsize::new(0),
    });
    let clock = Arc::new(GpuClock::new(source.clone()));
    let ready = Arc::new(std::sync::Barrier::new(8));
    let workers: Vec<_> = (0..8)
        .map(|index| {
            let clock = clock.clone();
            let ready = ready.clone();
            thread::spawn(move || {
                let tick = if index == 0 {
                    50
                } else if index % 2 == 0 {
                    5_000_100 + index
                } else {
                    1_000_000_100 + index
                };
                let expected =
                    u64::try_from(1_000_000_000 + (i128::from(tick) - 100) * 10).unwrap();
                ready.wait();
                assert_eq!(clock.tick(tick), Ok(expected));
                assert_eq!(
                    clock.interval(tick + 412, tick),
                    Ok((expected + 4_120, expected))
                );
                assert_eq!(clock.tick(tick), Ok(expected));
                assert_eq!(clock.tick(tick + 412), Ok(expected + 4_120));
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(source.calls.load(Ordering::Relaxed), 33);
    assert_eq!(
        clock.state.lock().unwrap().as_ref().unwrap().latest.gpu,
        320_000_100
    );
    assert_eq!(
        clock.state.lock().unwrap().as_ref().unwrap().ratio,
        Some(ClockRatio {
            host: 100_000_000,
            gpu: 10_000_000
        })
    );
}

#[test]
/// Verifies raw-counter quality selection independently of owned conversion.
fn raw_counter_precision_selection_keeps_the_shortest_complete_sample() {
    let mut samples = [
        Ok::<_, &str>((10, Duration::from_micros(40))),
        Ok((20, Duration::from_micros(3))),
        Ok((30, Duration::from_micros(12))),
        Ok((40, Duration::from_micros(1))),
    ]
    .into_iter();
    assert_eq!(best_clock_sample(4, || samples.next().unwrap()), Ok(40));

    let mut calls = 0;
    assert_eq!(
        best_clock_sample(4, || {
            calls += 1;
            if calls == 2 {
                Err("late ioctl failure")
            } else {
                Ok((17, Duration::from_micros(2)))
            }
        }),
        Ok(17)
    );
    assert_eq!(calls, 2);
}

#[test]
/// Verifies that GPU ticks translate around the correlated host sample.
fn gpu_ticks_translate_around_the_correlated_host_sample() {
    let counters = ClockCounters {
        gpu: 100,
        host: 1_000,
        system: None,
        gpu_info: rocddi::host::clock::ClockInfo {
            frequency_hz: 100_000_000,
            counter_bit_count: 0,
        },
    };
    assert_eq!(
        translate_gpu_tick(
            ClockCounters {
                gpu_info: rocddi::host::clock::ClockInfo {
                    counter_bit_count: 32,
                    ..counters.gpu_info
                },
                ..counters
            },
            110
        ),
        Err(NOT_SUPPORTED)
    );
    assert_eq!(translate_gpu_tick(counters, 110), Ok(1_100));
    assert_eq!(translate_gpu_tick(counters, 90), Ok(900));
    assert_eq!(
        translate_gpu_tick(
            ClockCounters {
                gpu_info: rocddi::host::clock::ClockInfo {
                    frequency_hz: 200_000_000,
                    ..counters.gpu_info
                },
                ..counters
            },
            110,
        ),
        Ok(1_050)
    );
    assert_eq!(
        translate_gpu_tick(
            ClockCounters {
                gpu_info: rocddi::host::clock::ClockInfo {
                    frequency_hz: 0,
                    ..counters.gpu_info
                },
                ..counters
            },
            100,
        ),
        Err(ERROR)
    );
}
