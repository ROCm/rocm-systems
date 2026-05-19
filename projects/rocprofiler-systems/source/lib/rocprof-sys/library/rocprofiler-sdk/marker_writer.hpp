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
#include <variant>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

/// Annotation key-value pair for perfetto events.
/// Writer builds these; policy iterates and calls add_perfetto_annotation for each.
struct annotation_entry
{
    annotation_entry(const char*                                      anno_key,
                     const std::variant<std::string, std::uint64_t, std::int64_t, double,
                                        std::int32_t, std::uint32_t>& anno_value)
    : key(anno_key)
    , value(anno_value)
    {}
    const char* key;
    std::variant<std::string, std::uint64_t, std::int64_t, double, std::int32_t,
                 std::uint32_t>
        value;
};

/// Thin API wrapper policy for marker_writer.
/// Each method wraps a single raw API call. No business logic.
struct default_marker_policy
{
    static void push_timemory(std::string_view name);
    static void pop_timemory(std::string_view name);

    static void push_perfetto_ts(const char* name, std::uint64_t ts,
                                 std::uint64_t                        flow_id,
                                 const std::vector<annotation_entry>& annotations);
    static void pop_perfetto_ts(const char* name, std::uint64_t ts,
                                const std::vector<annotation_entry>& annotations);

    static void add_string(std::string_view string_value);
    static void store_region(const trace_cache::region_sample& sample);
    static void add_thread_info(const rocprofsys::trace_cache::info::thread& thread_info);
};

/// Output layer for writing marker data to Perfetto, timemory, and cache.
/// Contains the logic for building region samples and perfetto annotations.
/// Delegates raw API calls to the policy, which can be mocked for testing.
///
/// @tparam MarkerWriterPolicy Compile-time policy providing thin API wrappers.
template <typename MarkerWriterPolicy = default_marker_policy>
class marker_writer
{
public:
    marker_writer(bool use_perfetto, bool use_timemory, bool perfetto_annotations)
    : m_use_perfetto(use_perfetto)
    , m_use_timemory(use_timemory)
    , m_perfetto_annotations(perfetto_annotations)
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
                   const std::string&                    args,
                   rocprofiler_callback_tracing_record_t record) const
    {
        if(m_use_timemory)
        {
            MarkerWriterPolicy::pop_timemory(name);
        }

        if(m_use_perfetto)
        {
            auto push_annotations = std::vector<annotation_entry>{};
            auto pop_annotations  = std::vector<annotation_entry>{};
            if(m_perfetto_annotations)
            {
                push_annotations.emplace_back("begin_ns", begin_ts);
                push_annotations.emplace_back("stack_id", record.correlation_id.internal);
                pop_annotations.emplace_back("end_ns", end_ts);
            }
            MarkerWriterPolicy::push_perfetto_ts(
                name.data(), begin_ts, record.correlation_id.internal, push_annotations);
            MarkerWriterPolicy::pop_perfetto_ts(name.data(), end_ts, pop_annotations);
        }

        constexpr size_t UNKNOWN_TIME = 0;
        MarkerWriterPolicy::add_thread_info(
            { getppid(), getpid(), record.thread_id, UNKNOWN_TIME, UNKNOWN_TIME, "{}" });

        MarkerWriterPolicy::store_region(trace_cache::region_sample{
            record.thread_id, std::string{ name }.c_str(), record.correlation_id.internal,
            record.correlation_id.external.value, begin_ts, end_ts, "{}", args,
            tim::trait::name<tim::category::rocm_marker_api>::value });
    }

private:
    bool m_use_perfetto{ false };
    bool m_use_timemory{ false };
    bool m_perfetto_annotations{ false };
};

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
