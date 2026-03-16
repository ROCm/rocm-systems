// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/roctx_client.hpp"
#include "library/rocprofiler-sdk/trace_control.hpp"
#include "library/tracing.hpp"

#include "core/categories.hpp"
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
#include <tuple>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

thread_local roctx_client::marker_range_stack_t roctx_client::m_pushed_ranges{};
thread_local roctx_client::marker_range_stack_t roctx_client::m_started_ranges{};

namespace
{
int
iterate_args_callback(rocprofiler_callback_tracing_kind_t /*kind*/, int32_t /*operation*/,
                      uint32_t arg_number, const void* const /*arg_value_addr*/,
                      int32_t /*arg_indirection_count*/, const char* arg_type,
                      const char* arg_name, const char*        arg_value_str,
                      int32_t /*arg_dereference_count*/, void* data)
{
    auto* _data = static_cast<function_args_t*>(data);
    if(arg_type && arg_name && arg_value_str)
        _data->emplace_back(argument_info{ arg_number,
                                           rocprofsys::utility::demangle(arg_type),
                                           arg_name, arg_value_str });
    return 0;
}

void
configure_rocprofiler_callback_tracing(rocprofiler_context_id_t               context_id,
                                       rocprofiler_callback_tracing_kind_t    kind,
                                       const rocprofiler_tracing_operation_t* operations,
                                       size_t                            operations_count,
                                       rocprofiler_callback_tracing_cb_t callback,
                                       void*                             callback_args)
{
    auto status = rocprofiler_configure_callback_tracing_service(
        context_id, kind, operations, operations_count, callback, callback_args);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to configure marker core callback : {}",
                    rocprofiler_get_status_string(status));
    }
}

std::string_view
get_operation_name(rocprofiler_callback_tracing_record_t record)
{
    return trace_cache::get_metadata_registry().get_callback_tracing_info().at(
        record.kind, record.operation);
}
}  // namespace

roctx_client::roctx_client(roctx_client_config roctx_cfg)
: m_config{ roctx_cfg }
, m_writer{ roctx_cfg.use_perfetto, roctx_cfg.use_timemory }
, m_controller{ std::make_shared<control::trace_control>(
      roctx_cfg.selected_trace_regions) }
{}

bool
roctx_client::should_write_markers() const
{
    return (m_config.is_write_enabled && m_controller->should_write_markers());
}

void
roctx_client::shutdown()
{
    m_controller->shutdown();
}

void
roctx_client::configure_services(rocprofiler_context_id_t ctx)
{
    m_ctx = ctx;

    // Configure MARKER_CORE_API for all marker operations
    configure_rocprofiler_callback_tracing(m_ctx,
                                           ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                                           nullptr, 0, marker_core_callback, this);

    // Configure MARKER_CONTROL_API for pause/resume
    auto control_ops = std::array<rocprofiler_tracing_operation_t, 2>{
        ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause,
        ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume
    };
    configure_rocprofiler_callback_tracing(
        m_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API, control_ops.data(),
        control_ops.size(), marker_control_callback, this);
}

void
roctx_client::handle_marker_core_enter(rocprofiler_callback_tracing_record_t record,
                                       rocprofiler_user_data_t*              user_data,
                                       rocprofiler_timestamp_t               ts)
{
    auto* data =
        static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);
    bool write_enabled = should_write_markers();

    switch(record.operation)
    {
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA:
        {
            const char* name = data->args.roctxRangePushA.message;
            const auto  hash = tim::add_hash_id(name);
            m_pushed_ranges.emplace_back(hash, ts, write_enabled);

            if(write_enabled)
            {
                m_writer.write_begin(name);
            }
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA:
        {
            // retval (range_id) is not available in ENTER phase
            // Just register the hash, actual handling in EXIT phase
            const char* name = data->args.roctxRangeStartA.message;
            tim::add_hash_id(name);
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA:
        {
            const char* name = data->args.roctxMarkA.message;
            tim::add_hash_id(name);

            if(write_enabled && m_config.use_timemory)
            {
                tracing::push_timemory(category::rocm_marker_api{}, name);
            }
            break;
        }
        default:
        {
            // For other operations (like roctxGetThreadId), push to timemory
            // They will be written in EXIT phase with duration
            if(write_enabled && m_config.use_timemory)
            {
                auto name = get_operation_name(record);
                tracing::push_timemory(category::rocm_marker_api{}, name);
            }
            break;
        }
    }

    // Store timestamp for EXIT phase
    user_data->value = ts;
}

void
roctx_client::handle_marker_core_exit(rocprofiler_callback_tracing_record_t record,
                                      rocprofiler_user_data_t*              user_data,
                                      rocprofiler_timestamp_t               ts)
{
    auto* data =
        static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);
    const uint64_t begin_ts = user_data->value;

    auto args = function_args_t{};
    rocprofiler_iterate_callback_tracing_kind_operation_args(
        record, iterate_args_callback, 2, &args);

    const std::string args_str = get_args_string(args);

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

            auto& entry                          = m_pushed_ranges.back();
            auto [hash, begin_ts, write_enabled] = entry;
            m_pushed_ranges.pop_back();

            const char* name = nullptr;
            tim::get_hash_identifier_fast(hash, name);

            // Only write if writing was enabled at push time
            if(write_enabled && name)
            {
                m_writer.write_end(name, begin_ts, ts, args_str, record);
            }
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

            auto& entry                          = m_started_ranges.back();
            auto [hash, begin_ts, write_enabled] = entry;
            m_started_ranges.pop_back();

            const char* name = nullptr;
            tim::get_hash_identifier_fast(hash, name);
            auto range_id = data->args.roctxRangeStop.id;

            if(write_enabled && name)
            {
                m_writer.write_end(name, begin_ts, ts, args_str, record);
            }

            m_controller->handle_range_stop(range_id);
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA:
        {
            // roctxMarkA writes with duration (ENTER to EXIT timestamps)
            const char* name = data->args.roctxMarkA.message;

            if(should_write_markers())
            {
                m_writer.write_end(name, begin_ts, ts, args_str, record);
            }
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA:
        {
            return;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA:
        {
            // Handle entirely in EXIT phase where retval (range_id) is available
            const char* name     = data->args.roctxRangeStartA.message;
            auto        hash     = tim::get_hash_id(name);
            auto        range_id = data->retval.roctx_range_id_t_retval;

            // Call range_start first so region becomes active
            m_controller->handle_range_start(range_id, name);

            // Now check if we should write (region is now active)
            bool range_write_enabled = should_write_markers();
            m_started_ranges.emplace_back(hash, begin_ts, range_write_enabled);

            if(range_write_enabled)
            {
                m_writer.write_begin(name);
            }
            return;
        }
        default:
        {
            // For other operations (like roctxGetThreadId), write with duration
            if(should_write_markers())
            {
                auto name = get_operation_name(record);
                m_writer.write_end(name, begin_ts, ts, args_str, record);
            }
            break;
        }
    }
}

void
roctx_client::handle_marker_control(rocprofiler_callback_tracing_record_t record)
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

void
roctx_client::marker_core_callback(rocprofiler_callback_tracing_record_t record,
                                   rocprofiler_user_data_t*              user_data,
                                   void*                                 callback_data)
{
    if(!callback_data) return;
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

void
roctx_client::marker_control_callback(rocprofiler_callback_tracing_record_t record,
                                      rocprofiler_user_data_t* /*user_data*/,
                                      void* callback_data)
{
    if(!callback_data) return;
    auto* client = static_cast<roctx_client*>(callback_data);
    client->handle_marker_control(record);
}

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
