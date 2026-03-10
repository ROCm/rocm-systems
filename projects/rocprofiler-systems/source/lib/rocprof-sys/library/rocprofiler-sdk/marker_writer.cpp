// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/marker_writer.hpp"

#include "core/config.hpp"
#include "core/timemory.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/tracing.hpp"

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace
{
bool
get_use_perfetto()
{
    return config::get_use_perfetto();
}

bool
get_use_timemory()
{
    return config::get_use_timemory();
}

uint64_t
get_parent_stack_id(rocprofiler_correlation_id_t corr_id)
{
    return corr_id.external.value;
}

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
               uint64_t begin_ts, uint64_t end_ts, std::string& args)
{
    cache_category();
    cache_add_thread_info(record.thread_id);

    trace_cache::get_buffer_storage().store(trace_cache::region_sample{
        record.thread_id, std::string{ name }.c_str(), record.correlation_id.internal,
        get_parent_stack_id(record.correlation_id), begin_ts, end_ts, "{}", args,
        trait::name<category::rocm_marker_api>::value });
}
}  // namespace

void
marker_writer::write_push_start(std::string_view name)
{
    if(get_use_timemory())
    {
        tracing::push_timemory(category::rocm_marker_api{}, name);
    }
}

void
marker_writer::write_pop(std::string_view name, uint64_t begin_ts, uint64_t end_ts,
                         std::string& args, rocprofiler_callback_tracing_record_t record)
{
    if(get_use_timemory())
    {
        tracing::pop_timemory(category::rocm_marker_api{}, name);
    }

    if(get_use_perfetto())
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

void
marker_writer::write_range_start(std::string_view name)
{
    if(get_use_timemory())
    {
        tracing::push_timemory(category::rocm_marker_api{}, name);
    }
}

void
marker_writer::write_range_stop(std::string_view name, uint64_t begin_ts, uint64_t end_ts,
                                std::string&                          args,
                                rocprofiler_callback_tracing_record_t record)
{
    if(get_use_timemory())
    {
        tracing::pop_timemory(category::rocm_marker_api{}, name);
    }

    if(get_use_perfetto())
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

void
marker_writer::write_mark(std::string_view name, uint64_t begin_ts, uint64_t end_ts,
                          std::string& args, rocprofiler_callback_tracing_record_t record)
{
    if(get_use_timemory())
    {
        tracing::pop_timemory(category::rocm_marker_api{}, name);
    }

    if(get_use_perfetto())
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

void
marker_writer::write_api_call(std::string_view name, uint64_t begin_ts, uint64_t end_ts,
                              std::string&                          args,
                              rocprofiler_callback_tracing_record_t record)
{
    if(get_use_timemory())
    {
        tracing::pop_timemory(category::rocm_marker_api{}, name);
    }

    if(get_use_perfetto())
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
