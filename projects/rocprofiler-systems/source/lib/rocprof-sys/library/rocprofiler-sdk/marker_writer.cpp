// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/marker_writer.hpp"

#include "core/perfetto.hpp"
#include "library/tracing.hpp"
#include "library/tracing/annotation.hpp"

#include <rocprofiler-sdk/fwd.h>

#include "core/categories.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <unistd.h>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace
{

void
cache_add_thread_info(rocprofiler_thread_id_t thread_id)
{
    constexpr size_t UNKNOWN_TIME = 0;
    trace_cache::get_metadata_registry().add_thread_info(
        { getppid(), getpid(), thread_id, UNKNOWN_TIME, UNKNOWN_TIME, "{}" });
}

void
cache_category()
{
    trace_cache::get_metadata_registry().add_string(
        trait::name<category::rocm_marker_api>::value);
}

void
write_to_cache(const rocprofiler_callback_tracing_record_t& record, std::string_view name,
               uint64_t begin_ts, uint64_t end_ts, const std::string& args)
{
    cache_add_thread_info(record.thread_id);

    trace_cache::get_buffer_storage().store(trace_cache::region_sample{
        record.thread_id, std::string{ name }.c_str(), record.correlation_id.internal,
        record.correlation_id.external.value, begin_ts, end_ts, "{}", args,
        trait::name<category::rocm_marker_api>::value });
}
}  // namespace

marker_writer::marker_writer(bool use_perfetto, bool use_timemory)
: m_use_perfetto(use_perfetto)
, m_use_timemory(use_timemory)
{
    cache_category();
}

void
marker_writer::write_begin(std::string_view name) const
{
    if(m_use_timemory)
    {
        tracing::push_timemory(category::rocm_marker_api{}, name);
    }
}

void
marker_writer::write_end(std::string_view name, uint64_t begin_ts, uint64_t end_ts,
                         const std::string&                    args,
                         rocprofiler_callback_tracing_record_t record) const
{
    if(m_use_timemory)
    {
        tracing::pop_timemory(category::rocm_marker_api{}, name);
    }

    if(m_use_perfetto)
    {
        tracing::push_perfetto_ts(
            category::rocm_marker_api{}, name.data(), begin_ts,
            ::perfetto::Flow::ProcessScoped(record.correlation_id.internal),
            [&](::perfetto::EventContext ctx) {
                if(config::get_perfetto_annotations())
                {
                    tracing::add_perfetto_annotation(ctx, "begin_ns", begin_ts);
                    tracing::add_perfetto_annotation(ctx, "stack_id",
                                                     record.correlation_id.internal);
                }
            });
        tracing::pop_perfetto_ts(category::rocm_marker_api{}, name.data(), end_ts,
                                 [&](::perfetto::EventContext ctx) {
                                     if(config::get_perfetto_annotations())
                                         tracing::add_perfetto_annotation(ctx, "end_ns",
                                                                          end_ts);
                                 });
    }

    write_to_cache(record, name, begin_ts, end_ts, args);
}

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
