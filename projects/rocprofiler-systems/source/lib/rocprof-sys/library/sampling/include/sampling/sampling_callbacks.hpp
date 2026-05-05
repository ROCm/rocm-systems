// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Type-erased callbacks injected into sampling_service at construction.
// Separated from sampling_config (POD values) so the config struct stays
// lightweight and copyable without std::function overhead.

#include "core/trace_cache/sample_type.hpp"
#include "sampling/data/limits.hpp"
#include "sampling/thread_info_data.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <string>

namespace rocprofsys::sampling
{

// Signal-safe function pointer type for reading HW counters from the signal handler.
// Writes up to PAPI_EVENT_COUNT values into `out`; returns the number actually written.
using hw_counter_read_fn = std::size_t (*)(int64_t tid, long long* out,
                                           std::size_t max_count);

struct sampling_callbacks
{
    std::function<std::set<int>(int64_t)> resolve_signals = [](int64_t) {
        return std::set<int>{};
    };

    std::function<bool(int64_t)> is_thread_eligible = [](int64_t) { return true; };

    std::function<int64_t()> get_sys_tid = []() -> int64_t { return 0; };

    std::function<std::optional<thread_info_data>(int64_t)> resolve_thread_info =
        [](int64_t) -> std::optional<thread_info_data> { return std::nullopt; };

    std::function<void()>                         register_sampling_categories = []() {};
    std::function<void(int, int, std::size_t)>    register_thread_info = [](int, int,
                                                                         std::size_t) {};
    std::function<void(std::string, std::size_t)> register_track       = [](std::string,
                                                                      std::size_t) {};

    std::function<void(void*, std::string const&, double)> configure_overflow_pe_attr =
        [](void*, std::string const&, double) {};

    std::function<void()> postfork_parent_reinit = []() {};
    std::function<void()> postfork_child_cleanup = []() {};

    // HW counter (PAPI) lifecycle — setup/teardown use std::function (not signal-safe),
    // read uses a raw function pointer (async-signal-safe).
    std::function<void(int64_t)> setup_hw_counters    = [](int64_t) {};
    std::function<void(int64_t)> teardown_hw_counters = [](int64_t) {};
    hw_counter_read_fn           read_hw_counters     = nullptr;

    // Trace cache storage — type-erased so unit tests can intercept without
    // depending on the global trace_cache::buffer_storage singleton.
    // Production wiring in services_accessor.cpp.
    std::function<void(trace_cache::backtrace_region_sample const&)> store_region_sample =
        [](trace_cache::backtrace_region_sample const&) {};
    std::function<void(trace_cache::pmc_event_with_sample const&)> store_pmc_event =
        [](trace_cache::pmc_event_with_sample const&) {};
};

}  // namespace rocprofsys::sampling
