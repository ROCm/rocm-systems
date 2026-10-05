// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/categories.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unistd.h>

namespace rocprofsys::rocprofiler_sdk
{

/// Thin API wrapper policy for marker_writer.
/// Each method wraps a single raw API call. No business logic.
struct default_marker_policy
{
    static void push_timemory(std::string_view name);
    static void pop_timemory(std::string_view name);

    static void add_string(std::string_view string_value);
    static void store_region(const trace_cache::region_sample& sample);
    static void add_thread_info(const rocprofsys::trace_cache::info::thread& thread_info);
};

/// Output layer for writing marker data to timemory and cache.
/// Contains the logic for building region samples.
/// Delegates raw API calls to the policy, which can be mocked for testing.
///
/// @tparam MarkerWriterPolicy Compile-time policy providing thin API wrappers.
template <typename MarkerWriterPolicy = default_marker_policy>
class marker_writer
{
public:
    explicit marker_writer(bool use_timemory)
    : m_use_timemory(use_timemory)
    {
        MarkerWriterPolicy::add_string(
            tim::trait::name<tim::category::rocm_marker_api>::value);
    }

    ~marker_writer() = default;

    marker_writer(const marker_writer&)            = delete;
    marker_writer& operator=(const marker_writer&) = delete;
    marker_writer(marker_writer&&)                 = default;
    marker_writer& operator=(marker_writer&&)      = default;

    void write_begin(std::string_view name) const
    {
        if(m_use_timemory)
        {
            MarkerWriterPolicy::push_timemory(name);
        }
    }

    void write_end(std::string_view name, std::uint64_t begin_ts, std::uint64_t end_ts,
                   std::string_view                      args,
                   rocprofiler_callback_tracing_record_t record) const
    {
        if(m_use_timemory)
        {
            MarkerWriterPolicy::pop_timemory(name);
        }

        constexpr size_t UNKNOWN_TIME = 0;
        MarkerWriterPolicy::add_thread_info(
            { getppid(), getpid(), record.thread_id, UNKNOWN_TIME, UNKNOWN_TIME, "{}" });

        MarkerWriterPolicy::store_region(trace_cache::region_sample{
            record.thread_id, name, record.correlation_id.internal,
            record.correlation_id.external.value, begin_ts, end_ts, "{}", args,
            tim::trait::name<tim::category::rocm_marker_api>::value });
    }

private:
    bool m_use_timemory{ false };
};

}  // namespace rocprofsys::rocprofiler_sdk
