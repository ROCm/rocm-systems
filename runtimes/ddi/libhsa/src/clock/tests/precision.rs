// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Precision conversion tests with only individual native queries replaced.

use super::*;

/// Individual query results, including observed duration before batch selection.
struct NativeQueries {
    /// Successful and failed provider operations in their actual query order.
    values: Mutex<VecDeque<Result<(ClockCounters, Duration), Status>>>,
    /// Number of provider operations, including failures and discarded candidates.
    calls: AtomicUsize,
}

impl NativeQueries {
    /// Injects raw provider operations below production batching and admission.
    fn new(
        values: impl IntoIterator<Item = Result<(ClockCounters, Duration), Status>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            values: Mutex::new(values.into_iter().collect()),
            calls: AtomicUsize::new(0),
        })
    }
}

impl ClockSource for Arc<NativeQueries> {
    fn sample(&self) -> Result<(ClockCounters, Duration), Status> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.values
            .lock()
            .unwrap()
            .pop_front()
            .expect("native query")
    }
}

/// Constructs one timed native result without selecting or validating it.
fn query(gpu: u64, host: u64, microseconds: u64) -> (ClockCounters, Duration) {
    (sample(gpu, host), Duration::from_micros(microseconds))
}

/// Isolates the production process-wide setting without resetting it in tests.
fn in_fresh_process(test: impl FnOnce()) {
    const CHILD: &str = "ROCDDI_CLOCK_PRECISION_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let name = thread::current().name().unwrap().to_owned();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &name, "--nocapture"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    assert_eq!(clock_sample_count(), 1);
    test();
}

/// Enables the real public setting after normal host-session publication.
fn enable_precision() {
    use crate::api::system::hsa_amd_enable_high_precision_timestamps;
    use crate::lifecycle::lock;
    use crate::session::testing::session_without_devices;
    use rocddi::session::{DriverContextLifetime, Session};

    assert!(crate::admission::process_admission().enter().is_none());
    assert_eq!(hsa_amd_enable_high_precision_timestamps(), NOT_INITIALIZED);
    assert_eq!(clock_sample_count(), 1);
    let session = session_without_devices(Session::new(DriverContextLifetime::Session).unwrap());
    {
        let mut lifecycle = lock().unwrap();
        let generation = lifecycle.begin_init().unwrap().created().unwrap();
        lifecycle.publish_init(generation, session).unwrap();
        lifecycle.finish_init(generation).unwrap();
    }
    // The real non-test ABI admission precondition is satisfied as well as the
    // function's registry check; enabling requires no native device acquisition.
    let token = crate::admission::process_admission().enter().unwrap();
    assert_eq!(hsa_amd_enable_high_precision_timestamps(), SUCCESS);
    drop(token);
    assert_eq!(clock_sample_count(), 4);
}

/// Startup and first live batches whose selected anchors produce exactly 10 ns/tick.
fn initial_queries() -> Vec<Result<(ClockCounters, Duration), Status>> {
    vec![
        Ok(query(100, 1_000_000_000, 40)),
        Ok(query(200, 1_000_001_000, 2)),
        Ok(query(300, 1_000_002_000, 20)),
        Ok(query(400, 1_000_003_000, 30)),
        Ok(query(10_000_200, 1_100_001_000, 9)),
        Ok(query(10_000_300, 1_100_002_000, 1)),
        Ok(query(10_000_400, 1_100_003_100, 7)),
        Ok(query(10_000_500, 1_100_004_200, 8)),
    ]
}

#[test]
fn slower_gpu_regression_invalidates_before_another_query() {
    in_fresh_process(|| {
        enable_precision();
        let mut values = initial_queries();
        values.extend([
            Ok(query(20_000_100, 1_200_000_000, 1)),
            Ok(query(20_000_099, 1_200_000_001, 50)),
            Ok(query(30_000_100, 1_300_000_000, 100)),
        ]);
        let source = NativeQueries::new(values);
        let clock = GpuClock::new(source.clone());
        assert_eq!(source.calls.load(Ordering::Relaxed), 0);
        assert_eq!(clock.tick(10_000_300), Ok(1_100_002_000));
        assert_eq!(clock.tick(10_000_300), Err(ERROR));
        assert_eq!(source.calls.load(Ordering::Relaxed), 10);
        assert_eq!(clock.interval(10_000_300, 200), Err(ERROR));
        assert_eq!(source.calls.load(Ordering::Relaxed), 10);
    });
}

#[test]
fn slower_host_or_metadata_contradiction_cannot_hide_behind_the_winner() {
    in_fresh_process(|| {
        enable_precision();
        for (invalid, status) in [
            (sample(20_000_200, 1_199_999_999), ERROR),
            (
                ClockCounters {
                    gpu_info: rocddi::host::clock::ClockInfo {
                        frequency_hz: 200_000_000,
                        counter_bit_count: 0,
                    },
                    ..sample(20_000_200, 1_200_001_000)
                },
                ERROR,
            ),
            (
                ClockCounters {
                    gpu_info: rocddi::host::clock::ClockInfo {
                        frequency_hz: 100_000_000,
                        counter_bit_count: 32,
                    },
                    ..sample(20_000_200, 1_200_001_000)
                },
                NOT_SUPPORTED,
            ),
        ] {
            let mut values = initial_queries();
            values.extend([
                Ok(query(20_000_100, 1_200_000_000, 1)),
                Ok((invalid, Duration::from_micros(50))),
                Ok(query(30_000_100, 1_300_000_000, 100)),
            ]);
            let source = NativeQueries::new(values);
            let clock = GpuClock::new(source.clone());
            assert_eq!(clock.tick(10_000_300), Ok(1_100_002_000));
            assert_eq!(clock.tick(10_000_300), Err(status));
            assert_eq!(clock.tick(10_000_300), Err(ERROR));
            assert_eq!(source.calls.load(Ordering::Relaxed), 10);
        }
    });
}

#[test]
fn late_batch_error_retains_nonwinning_high_water_for_retry_validation() {
    in_fresh_process(|| {
        enable_precision();
        for kind in ["gpu regression", "host regression", "healthy recovery"] {
            let retry = match kind {
                "gpu regression" => sample(25_000_100, 1_350_000_000),
                "host regression" => sample(31_000_100, 1_250_000_000),
                _ => sample(31_000_100, 1_310_000_000),
            };
            let mut values = initial_queries();
            values.extend([
                Ok(query(20_000_100, 1_200_000_000, 1)),
                Ok(query(30_000_100, 1_300_000_000, 50)),
                Err(INVALID_RUNTIME_STATE),
                Ok((retry, Duration::from_micros(1))),
                Ok(query(32_000_100, 1_320_000_000, 10)),
                Ok(query(33_000_100, 1_330_000_000, 20)),
                Ok(query(34_000_100, 1_340_000_000, 30)),
            ]);
            let source = NativeQueries::new(values);
            let clock = GpuClock::new(source.clone());
            let retained = clock.tick(10_000_300).unwrap();
            assert_eq!(clock.tick(10_000_300), Err(INVALID_RUNTIME_STATE));
            {
                let state = clock.state.lock().unwrap();
                let state = state.as_ref().unwrap();
                assert_eq!(state.latest, sample(30_000_100, 1_300_000_000));
                assert_eq!(state.host_high_water, 1_300_000_000);
                assert!(!state.invalid);
            }
            if kind == "healthy recovery" {
                assert_eq!(clock.tick(10_000_300), Ok(retained));
                assert_eq!(source.calls.load(Ordering::Relaxed), 15);
            } else {
                assert_eq!(clock.tick(10_000_300), Err(ERROR));
                assert_eq!(clock.tick(10_000_300), Err(ERROR));
                assert_eq!(source.calls.load(Ordering::Relaxed), 12);
            }
        }
    });
}

#[test]
fn first_precision_batch_retains_health_before_its_late_error() {
    in_fresh_process(|| {
        enable_precision();
        for healthy in [false, true] {
            let retry = if healthy {
                sample(10_000_100, 1_100_000_000)
            } else {
                sample(199, 1_100_000_000)
            };
            let source = NativeQueries::new([
                Ok(query(100, 1_000_000_000, 1)),
                Ok(query(200, 1_000_001_000, 50)),
                Err(ERROR),
                Ok((retry, Duration::from_micros(100))),
                Ok(query(10_000_200, 1_100_001_000, 100)),
                Ok(query(10_000_300, 1_100_002_000, 100)),
                Ok(query(10_000_400, 1_100_003_000, 100)),
            ]);
            let clock = GpuClock::new(source.clone());
            assert_eq!(source.calls.load(Ordering::Relaxed), 0);
            assert_eq!(clock.tick(150), Err(ERROR));
            {
                let state = clock.state.lock().unwrap();
                let state = state.as_ref().unwrap();
                assert_eq!(state.startup, sample(100, 1_000_000_000));
                assert_eq!(state.latest, sample(200, 1_000_001_000));
                assert_eq!(state.host_high_water, 1_000_001_000);
                assert!(!state.invalid && state.ratio.is_none());
            }
            if healthy {
                assert_eq!(clock.tick(150), Ok(1_000_000_500));
                assert_eq!(source.calls.load(Ordering::Relaxed), 7);
            } else {
                assert_eq!(clock.tick(150), Err(ERROR));
                assert_eq!(clock.tick(150), Err(ERROR));
                assert_eq!(source.calls.load(Ordering::Relaxed), 4);
            }
        }
    });
}

#[test]
fn healthy_precision_uses_best_anchors_and_retains_all_health_observations() {
    in_fresh_process(|| {
        enable_precision();
        let mut values = initial_queries();
        values.extend([Ok(query(20_000_500, 1_200_100_000, 10)); 12]);
        let source = NativeQueries::new(values);
        let clock = Arc::new(GpuClock::new(source.clone()));
        let raw = clock.clone();
        let dispatch = clock.clone();
        let copy = clock.clone();
        let later = raw.tick(10_000_300).unwrap();
        assert_eq!(later, 1_100_002_000);
        {
            let state = clock.state.lock().unwrap();
            let state = state.as_ref().unwrap();
            assert_eq!(state.startup, sample(200, 1_000_001_000));
            assert_eq!(state.latest, sample(10_000_500, 1_100_004_200));
            assert_eq!(state.host_high_water, 1_100_004_200);
            assert_eq!(
                state.ratio,
                Some(ClockRatio {
                    host: 100_001_000,
                    gpu: 10_000_100
                })
            );
        }
        let interval = dispatch.interval(10_000_300 - 412, 10_000_300).unwrap();
        assert_eq!(interval, (later - 4_120, later));
        assert_eq!(
            copy.interval(10_000_300, 10_000_300 - 412),
            Ok((later, interval.0))
        );
        assert_eq!(raw.tick(10_000_300), Ok(later));
        assert_eq!(source.calls.load(Ordering::Relaxed), 20);
    });
}

#[test]
fn public_precision_enablement_preserves_an_existing_shared_map() {
    in_fresh_process(|| {
        let mut values = vec![
            Ok(query(100, 1_000_000_000, 10)),
            Ok(query(10_000_100, 1_100_000_000, 20)),
        ];
        values.extend([Ok(query(20_000_100, 1_200_100_000, 30)); 12]);
        let source = NativeQueries::new(values);
        let clock = Arc::new(GpuClock::new(source.clone()));
        let raw = clock.clone();
        let dispatch = clock.clone();
        let copy = clock.clone();
        let later_tick = 9_000_512;
        let later = raw.tick(later_tick).unwrap();
        assert_eq!(source.calls.load(Ordering::Relaxed), 2);
        enable_precision();
        assert_eq!(
            dispatch.interval(later_tick - 412, later_tick),
            Ok((later - 4_120, later))
        );
        assert_eq!(source.calls.load(Ordering::Relaxed), 6);
        assert_eq!(
            copy.interval(later_tick, later_tick - 412),
            Ok((later, later - 4_120))
        );
        assert_eq!(raw.tick(later_tick), Ok(later));
        assert_eq!(source.calls.load(Ordering::Relaxed), 14);
    });
}
