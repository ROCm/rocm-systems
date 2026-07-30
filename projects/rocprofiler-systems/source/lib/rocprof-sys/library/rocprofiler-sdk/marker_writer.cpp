// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/marker_writer.hpp"

#include "core/categories.hpp"
#include "core/perfetto.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/tracing.hpp"
#include "library/tracing/annotation.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <variant>
#include <vector>

#include <unistd.h>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

void
default_marker_policy::push_timemory(std::string_view name)
{
    tracing::push_timemory(category::rocm_marker_api{}, name);
}

void
default_marker_policy::pop_timemory(std::string_view name)
{
    tracing::pop_timemory(category::rocm_marker_api{}, name);
}

void
default_marker_policy::push_perfetto_ts(const char* name, std::uint64_t ts,
                                        std::uint64_t                        flow_id,
                                        const std::vector<annotation_entry>& annotations)
{
    tracing::push_perfetto_ts(
        category::rocm_marker_api{}, name, ts, ::perfetto::Flow::ProcessScoped(flow_id),
        [&](::perfetto::EventContext ctx) {
            for(const auto& ann : annotations)
            {
                std::visit(
                    [&](const auto& v) {
                        tracing::add_perfetto_annotation(ctx, ann.key, v);
                    },
                    ann.value);
            }
        });
}

void
default_marker_policy::pop_perfetto_ts(const char* name, std::uint64_t ts,
                                       const std::vector<annotation_entry>& annotations)
{
    tracing::pop_perfetto_ts(
        category::rocm_marker_api{}, name, ts, [&](::perfetto::EventContext ctx) {
            for(const auto& ann : annotations)
            {
                std::visit(
                    [&](const auto& v) {
                        tracing::add_perfetto_annotation(ctx, ann.key, v);
                    },
                    ann.value);
            }
        });
}

void
default_marker_policy::add_string(std::string_view string_value)
{
    trace_cache::get_metadata_registry().add_string(string_value);
}

void
default_marker_policy::store_region(const trace_cache::region_sample& sample)
{
    trace_cache::get_buffer_storage().store(sample);
}

void
default_marker_policy::add_thread_info(const trace_cache::info::thread& thread_info)
{
    trace_cache::get_metadata_registry().add_thread_info(thread_info);
}

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
