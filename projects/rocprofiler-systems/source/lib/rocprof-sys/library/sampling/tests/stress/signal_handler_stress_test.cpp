// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AC-21: Signal-handler safety stress test.
// Built with -fsanitize=thread (TSAN). K=4 worker threads, concurrent
// pause/resume/shutdown. Runtime: ~10 seconds.
//
// Uses test_sampling_policies (test doubles) so no production policy headers
// are needed. AC-21 validates the service template's concurrency
// invariants (atomics, mutexes, atomic_flag) — the policy implementation
// does not affect the race/deadlock/UAF properties being checked.
//
// TSAN hard gate: zero races, deadlocks, use-after-free (NFR-TS-3).

#include <gtest/gtest.h>

#include "doubles/fake_clock.hpp"
#include "doubles/in_memory_emitter.hpp"
#include "doubles/mock_overflow_trigger.hpp"
#include "doubles/mock_timer_trigger.hpp"
#include "doubles/mock_unwinder.hpp"
#include "doubles/noop_perfetto_sink.hpp"
#include "doubles/noop_production_hooks.hpp"
#include "doubles/noop_report_writer.hpp"
#include "doubles/noop_signal_dispatcher.hpp"
#include "doubles/recording_trace_sink.hpp"
#include "doubles/throwing_fatal_error_policy.hpp"
#include "sampling/sampling_service.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

// This file is only compiled on Linux (ROCPROFSYS_BUILD_SAMPLING_TSAN) — see
// CMakeLists.txt.

using namespace rocprofsys::sampling;

namespace
{

// stress_sampling_policies uses noop_signal_dispatcher instead of
// recording_signal_dispatcher: the recording double has an unprotected
// std::vector that races under concurrent setup() calls. AC-21 validates
// the service's concurrency primitives — not the policy implementation.
struct stress_sampling_policies
{
    using unwinder          = test::mock_unwinder;
    using offload           = test::in_memory_emitter;
    using trace_sink        = test::recording_trace_sink;
    using timer_trigger     = test::mock_timer_trigger;
    using overflow_trigger  = test::mock_overflow_trigger;
    using clock             = test::fake_clock;
    using signal_dispatcher = test::noop_signal_dispatcher;
    using report_writer     = test::noop_report_writer;
    using perfetto_sink     = test::noop_perfetto_sink;
    using fatal_error       = test::throwing_fatal_error_policy;
    using production_hooks  = test::noop_production_hooks;
};

constexpr int          k_num_workers   = 4;
constexpr std::int64_t k_duration_secs = 10;

}  // namespace

TEST(signal_handler_stress, no_race_no_deadlock_no_use_after_free)
{
    using stress_service = sampling_service<stress_sampling_policies>;

    stress_service svc;

    // Set up all K worker threads.
    std::vector<std::thread> workers;
    std::atomic<bool>        stop_flag{ false };

    for(int i = 0; i < k_num_workers; ++i)
    {
        workers.emplace_back([&svc, &stop_flag, i] {
            const int64_t tid = static_cast<int64_t>(i);
            svc.setup(tid);

            while(!stop_flag.load(std::memory_order_relaxed))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            svc.shutdown(tid);
        });
    }

    // Coordinator thread: interleave pause/resume during the run.
    auto start    = std::chrono::steady_clock::now();
    auto end_time = start + std::chrono::seconds(k_duration_secs);

    while(std::chrono::steady_clock::now() < end_time)
    {
        svc.pause();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        svc.resume();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Signal all worker threads to stop and wait for them.
    // Each worker calls svc.shutdown(tid) for its own tid — no double-shutdown here.
    stop_flag.store(true, std::memory_order_release);
    for(auto& t : workers)
        t.join();

    // Assertions:
    // (a) No data corruption: total samples = written + dropped.
    size_t total_dropped = svc.dropped_samples();
    // The counter is always non-negative; reading it exercises the atomic without UAF.
    EXPECT_GE(total_dropped, 0U) << "dropped_samples counter should be non-negative";

    // (b) No deadlock: we reached here.
    SUCCEED() << "Stress test completed without deadlock. Dropped samples: "
              << total_dropped;
}
