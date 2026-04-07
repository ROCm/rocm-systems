// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/roctx_client.hpp"
#include "library/rocprofiler-sdk/marker_writer.hpp"
#include "library/rocprofiler-sdk/trace_control.hpp"

#include "core/common_types.hpp"
#include "core/demangler.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <timemory/hash/types.hpp>

#include "logger/debug.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

template <typename MarkerWriterPolicy>
thread_local typename roctx_client<MarkerWriterPolicy>::marker_range_stack_t
    roctx_client<MarkerWriterPolicy>::m_pushed_ranges{};

template <typename MarkerWriterPolicy>
thread_local typename roctx_client<MarkerWriterPolicy>::marker_range_stack_t
    roctx_client<MarkerWriterPolicy>::m_started_ranges{};

namespace
{

int
iterate_args_callback(rocprofiler_callback_tracing_kind_t, int32_t, uint32_t arg_number,
                      const void* const, int32_t, const char*                arg_type,
                      const char* arg_name, const char* arg_value_str, int32_t,
                      void* data)
{
    auto* args = static_cast<function_args_t*>(data);
    if(arg_type && arg_name && arg_value_str)
    {
        args->emplace_back(argument_info{ arg_number,
                                          rocprofsys::utility::demangle(arg_type),
                                          arg_name, arg_value_str });
    }
    return 0;
}

void
configure_callback_tracing(rocprofiler_context_id_t               context_id,
                           rocprofiler_callback_tracing_kind_t    kind,
                           const rocprofiler_tracing_operation_t* operations,
                           size_t                                 operations_count,
                           rocprofiler_callback_tracing_cb_t      callback,
                           void*                                  callback_args)
{
    auto status = rocprofiler_configure_callback_tracing_service(
        context_id, kind, operations, operations_count, callback, callback_args);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to configure marker core callback : {}",
                    rocprofiler_get_status_string(status));
    }
}

std::string
collect_args(rocprofiler_callback_tracing_record_t record)
{
    auto args = function_args_t{};
    rocprofiler_iterate_callback_tracing_kind_operation_args(
        record, iterate_args_callback, 2, &args);
    return get_args_string(args);
}

}  // namespace

template <typename MarkerWriterPolicy>
roctx_client<MarkerWriterPolicy>::roctx_client(const roctx_client_config& roctx_cfg)
: m_config{ roctx_cfg }
, m_writer{ roctx_cfg.use_perfetto, roctx_cfg.use_timemory,
            roctx_cfg.perfetto_annotations }
, m_controller{ std::make_shared<control::trace_control>(
      roctx_cfg.selected_trace_regions) }
{}

template <typename MarkerWriterPolicy>
void
roctx_client<MarkerWriterPolicy>::configure_services(rocprofiler_context_id_t ctx)
{
    m_ctx = ctx;

    configure_callback_tracing(m_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                               nullptr, 0, marker_core_callback, this);

    if(m_config.pause_resume_enabled)
    {
        auto control_ops = std::array<rocprofiler_tracing_operation_t, 2>{
            ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause,
            ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume
        };
        configure_callback_tracing(m_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                                   control_ops.data(), control_ops.size(),
                                   marker_control_callback, this);
    }
}

template <typename MarkerWriterPolicy>
void
roctx_client<MarkerWriterPolicy>::handle_marker_core_enter(
    rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data,
    rocprofiler_timestamp_t ts)
{
    auto* data =
        static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);
    const bool write_enabled = m_controller->should_write_markers();

    switch(record.operation)
    {
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA:
        {
            const char* name = data->args.roctxRangePushA.message;
            m_pushed_ranges.push_back({ tim::add_hash_id(name), ts, write_enabled });
            if(write_enabled)
            {
                m_writer.write_begin(name);
            }
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA:
        {
            tim::add_hash_id(data->args.roctxRangeStartA.message);
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA:
        {
            const char* name = data->args.roctxMarkA.message;
            tim::add_hash_id(name);
            if(write_enabled)
            {
                m_writer.write_begin(name);
            }
            break;
        }
        default:
        {
            if(write_enabled)
            {
                const auto& name =
                    trace_cache::get_metadata_registry().get_callback_tracing_info().at(
                        record.kind, record.operation);
                m_writer.write_begin(name);
            }
            break;
        }
    }

    user_data->value = ts;
}

template <typename MarkerWriterPolicy>
void
roctx_client<MarkerWriterPolicy>::handle_marker_core_exit(
    rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data,
    rocprofiler_timestamp_t ts)
{
    auto* data =
        static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);
    const uint64_t begin_ts = user_data->value;
    const auto     args_str = collect_args(record);

    auto pop_and_write = [&](marker_range_stack_t& stack) {
        auto        range = stack.back();
        const char* name  = nullptr;
        stack.pop_back();
        tim::get_hash_identifier_fast(range.hash, name);
        if(range.write_enabled && name)
        {
            m_writer.write_end(name, range.begin_ts, ts, args_str, record);
        }
    };

    switch(record.operation)
    {
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop:
        {
            if(m_pushed_ranges.empty())
            {
                LOG_CRITICAL("roctxRangePop does not have corresponding roctxRangePush "
                             "(skipping)");
                return;
            }

            pop_and_write(m_pushed_ranges);
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop:
        {
            if(m_started_ranges.empty())
            {
                LOG_CRITICAL("roctxRangeStop does not have corresponding roctxRangeStart "
                             "(skipping)");
                return;
            }

            pop_and_write(m_started_ranges);
            m_controller->handle_range_stop(data->args.roctxRangeStop.id);
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA:
        {
            if(m_controller->should_write_markers())
            {
                m_writer.write_end(data->args.roctxMarkA.message, begin_ts, ts, args_str,
                                   record);
            }
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA:
        {
            return;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA:
        {
            const char* name     = data->args.roctxRangeStartA.message;
            auto        range_id = data->retval.roctx_range_id_t_retval;

            m_controller->handle_range_start(range_id, name);

            const bool write_enabled = m_controller->should_write_markers();
            m_started_ranges.push_back(
                { tim::get_hash_id(name), begin_ts, write_enabled });
            if(write_enabled)
            {
                m_writer.write_begin(name);
            }
            return;
        }
        default:
        {
            if(m_controller->should_write_markers())
            {
                const auto& name =
                    trace_cache::get_metadata_registry().get_callback_tracing_info().at(
                        record.kind, record.operation);
                m_writer.write_end(name, begin_ts, ts, args_str, record);
            }
            break;
        }
    }
}

template <typename MarkerWriterPolicy>
void
roctx_client<MarkerWriterPolicy>::handle_marker_control(
    rocprofiler_callback_tracing_record_t record)
{
    if(record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        m_controller->handle_pause();
    }
    else if(record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume &&
            record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        m_controller->handle_resume();
    }
}

template <typename MarkerWriterPolicy>
void
roctx_client<MarkerWriterPolicy>::marker_core_callback(
    rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data,
    void* callback_data)
{
    if(!callback_data)
    {
        return;
    }
    auto* client = static_cast<roctx_client*>(callback_data);

    rocprofiler_timestamp_t ts = 0;
    rocprofiler_get_timestamp(&ts);

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        client->handle_marker_core_enter(record, user_data, ts);
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        client->handle_marker_core_exit(record, user_data, ts);
    }
}

template <typename MarkerWriterPolicy>
void
roctx_client<MarkerWriterPolicy>::marker_control_callback(
    rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*,
    void*                                 callback_data)
{
    if(!callback_data)
    {
        return;
    }
    static_cast<roctx_client*>(callback_data)->handle_marker_control(record);
}

template class roctx_client<default_marker_policy>;

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
