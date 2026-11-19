// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use super::*;
use rocddi::device::gpu::profiling::ClockCounters;

#[test]
/// Verifies that precise clock sampling keeps the shortest complete sample.
fn precise_clock_sampling_keeps_the_shortest_complete_sample() {
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
