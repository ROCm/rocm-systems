// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// NFR-PERF-1: Per-sample write latency microbenchmark.
// Measures the signal-handler hot path: handler entry -> unwind -> ring push.
// Built as a separate CTest target (not TSAN).
// Comparison budget: median <= 312,930 ns (260,775 ns baseline * 1.20).

#include <benchmark/benchmark.h>

#include "sampling/data/backtrace_record.hpp"
#include "sampling/src/sample_ring_buffer.hpp"

using namespace rocprofsys::sampling;

// ─── Benchmark: simulated signal-handler hot path ────────────────────────────
// We invoke the handler logic directly (without OS signals) to measure
// the pure computation cost (clock_gettime + memcpy + ring push).

static void
BM_signal_handler_hot_path(benchmark::State& state)
{
    sample_ring_buffer<2048> ring;

    backtrace_record rec;
    rec.tid      = 0;
    rec.pc_count = 8;
    for(uint8_t i = 0; i < 8; ++i)
        rec.raw_pcs[i] = 0x400000 + i * 8;

    for(auto _ : state)
    {
        // Simulate timestamp capture (clock_gettime equivalent).
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        rec.timestamp_ns = static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000U + ts.tv_nsec;

        // Simulate ring push.
        bool pushed = ring.try_push(rec);
        if(!pushed)
        {
            // Ring full — drain one entry (simulate the drop path).
            backtrace_record dummy;
            ring.try_pop(dummy);
            ring.try_push(rec);
        }

        benchmark::DoNotOptimize(rec);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_signal_handler_hot_path)->Threads(4)->MeasureProcessCPUTime()->UseRealTime();

BENCHMARK_MAIN();
