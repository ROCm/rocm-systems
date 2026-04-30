// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Configuration data for the sampling subsystem. Constructed once in the
// production facade (services_accessor.cpp) from rocprofsys::get_*() calls
// and injected into sampling_service at construction time.
//
// No main-library includes — only standard headers. This allows the sampling
// subsystem to be compiled and tested without linking the full rocprof-sys
// library.

#include "sampling/thread_info_data.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <string>

namespace rocprofsys::sampling
{

struct sampling_config
{
    // Signal assignments
    int realtime_signal = 0;
    int cputime_signal  = 0;
    int overflow_signal = 0;

    // Frequencies and delays (Hz / seconds)
    double realtime_freq  = 0.0;
    double cputime_freq   = 0.0;
    double realtime_delay = 0.0;
    double cputime_delay  = 0.0;
    double overflow_freq  = 0.0;
    double duration       = 0.0;

    // Overflow event name (perf_event type string)
    std::string overflow_event = "PERF_COUNT_SW_CPU_CLOCK";

    // Feature flags
    bool use_causal           = false;
    bool use_perfetto         = false;
    bool perfetto_annotations = false;
    bool trace_legacy         = false;
    bool use_process_sampling = false;
    bool use_amd_smi          = false;

    // Per-tid signal resolver. Returns the set of sampling signals configured
    // for the given logical tid. Production: wired to rocprofsys::get_sampling_signals().
    // Test: returns a fixed set injected by the test.
    std::function<std::set<int>(int64_t)> resolve_signals = [](int64_t) {
        return std::set<int>{};
    };

    // Thread eligibility guard. Returns true if the thread may be sampled.
    // Production: checks ThreadState::Disabled + thread_info::is_offset.
    // Test: returns true (all threads eligible).
    std::function<bool(int64_t)> is_thread_eligible = [](int64_t) { return true; };

    // System thread-id provider. Returns the kernel tid of the calling thread.
    // Production: wired to threading::get_sys_tid().
    // Test: returns a fixed value.
    std::function<int64_t()> get_sys_tid = []() -> int64_t { return 0; };

    // Metadata registration callbacks for trace_cache track setup.
    // Production: wired to trace_cache::get_metadata_registry().
    // Test: noops.
    std::function<void()>                         register_sampling_categories = []() {};
    std::function<void(int, int, std::size_t)>    register_thread_info = [](int, int,
                                                                         std::size_t) {};
    std::function<void(std::string, std::size_t)> register_track       = [](std::string,
                                                                      std::size_t) {};

    // Thread-info resolver. Returns thread system/sequent IDs and lifetime.
    // Production: wraps thread_info::get() with SequentTID/InternalTID fallback.
    // Default: returns nullopt (no thread info available).
    std::function<std::optional<thread_info_data>(int64_t)> resolve_thread_info =
        [](int64_t) -> std::optional<thread_info_data> { return std::nullopt; };

    // Overflow event perf_event_attr configurator.
    // Production: wired to rocprofsys::perf::config_overflow_sampling().
    // Test: noop (mock trigger ignores pe_attr).
    std::function<void(void*, std::string const&, double)> configure_overflow_pe_attr =
        [](void*, std::string const&, double) {};

    // PMC postfork callbacks. Called from do_postfork_parent_reinit /
    // do_postfork_child_cleanup when use_process_sampling && use_amd_smi.
    std::function<void()> postfork_parent_reinit = []() {};
    std::function<void()> postfork_child_cleanup = []() {};
};

}  // namespace rocprofsys::sampling
