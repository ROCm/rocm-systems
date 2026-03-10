// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/marker_client.hpp"

#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/demangler.hpp"
#include "core/timemory.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "library/trace_control.hpp"
#include "library/tracing.hpp"

#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <timemory/hash/types.hpp>
#include <timemory/utility/delimit.hpp>

#include "logger/debug.hpp"

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace
{
// Thread-local storage for tracking marker ranges
// Tuple: (hash, timestamp, was_write_enabled_at_push)
auto&
get_marker_pushed_ranges()
{
    static thread_local auto _v =
        std::vector<std::tuple<tim::hash_value_t, rocprofiler_timestamp_t, bool>>{};
    return _v;
}

auto&
get_marker_started_ranges()
{
    static thread_local auto _v =
        std::vector<std::tuple<tim::hash_value_t, rocprofiler_timestamp_t, bool>>{};
    return _v;
}

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
check_rocprofiler_status(rocprofiler_status_t status, const char* msg)
{
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("{}: {}", msg, rocprofiler_get_status_string(status));
    }
}

// Get operation name from callback tracing info
std::string_view
get_operation_name(rocprofiler_callback_tracing_record_t record)
{
    return trace_cache::get_metadata_registry().get_callback_tracing_info().at(
        record.kind, record.operation);
}

bool
get_use_timemory()
{
    return config::get_use_timemory();
}
}  // namespace

marker_client::marker_client(control::trace_control& controller)
: m_controller(controller)
{}

bool
marker_client::is_write_enabled() const
{
    // Check if marker_api or roctx is in ROCPROFSYS_ROCM_DOMAINS
    static const bool enabled = []() {
        auto domains =
            tim::delimit(config::get_setting_value<std::string>("ROCPROFSYS_ROCM_DOMAINS")
                             .value_or(std::string{}),
                         " ,;:\t\n");
        return std::find(domains.begin(), domains.end(), "marker_api") != domains.end() ||
               std::find(domains.begin(), domains.end(), "roctx") != domains.end();
    }();
    return enabled;
}

bool
marker_client::region_filter_active() const
{
    return m_controller.region_filter_active();
}

bool
marker_client::should_write_markers() const
{
    return is_write_enabled() && m_controller.should_write_markers();
}

void
marker_client::shutdown()
{
    m_controller.shutdown();
}

void
marker_client::configure_services(rocprofiler_context_id_t ctx)
{
    m_ctx = ctx;

    // Configure MARKER_CORE_API for all marker operations
    check_rocprofiler_status(rocprofiler_configure_callback_tracing_service(
                                 m_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                                 nullptr, 0, marker_core_callback, this),
                             "Failed to configure marker core callback");

    // Configure MARKER_CONTROL_API for pause/resume
    auto control_ops = std::array<rocprofiler_tracing_operation_t, 2>{
        ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause,
        ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume
    };
    check_rocprofiler_status(rocprofiler_configure_callback_tracing_service(
                                 m_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                                 control_ops.data(), control_ops.size(),
                                 marker_control_callback, this),
                             "Failed to configure marker control callback");
}

void
marker_client::handle_marker_core_enter(rocprofiler_callback_tracing_record_t record,
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
            auto        hash = tim::add_hash_id(name);
            get_marker_pushed_ranges().emplace_back(hash, ts, write_enabled);

            if(write_enabled)
            {
                m_writer.write_push_start(name);
            }
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA:
        {
            const char* name     = data->args.roctxRangeStartA.message;
            auto        hash     = tim::add_hash_id(name);
            auto        range_id = data->retval.roctx_range_id_t_retval;

            // Notify trace_control for region filtering BEFORE checking write state
            m_controller.handle_range_start(range_id, name);
            get_marker_started_ranges().emplace_back(hash, ts, write_enabled);

            if(write_enabled)
            {
                m_writer.write_range_start(name);
            }
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA:
        {
            // roctxMarkA is handled in EXIT phase with duration (ENTER to EXIT)
            // Just register the hash here
            const char* name = data->args.roctxMarkA.message;
            tim::add_hash_id(name);

            if(write_enabled && get_use_timemory())
            {
                tracing::push_timemory(category::rocm_marker_api{}, name);
            }
            break;
        }
        default:
        {
            // For other operations (like roctxGetThreadId), push to timemory
            // They will be written in EXIT phase with duration
            if(write_enabled && get_use_timemory())
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
marker_client::handle_marker_core_exit(rocprofiler_callback_tracing_record_t record,
                                       rocprofiler_user_data_t*              user_data,
                                       rocprofiler_timestamp_t               ts)
{
    auto* data =
        static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);
    uint64_t begin_ts = user_data->value;

    auto args = function_args_t{};
    rocprofiler_iterate_callback_tracing_kind_operation_args(
        record, iterate_args_callback, 2, &args);

    std::string args_str = get_args_string(args);

    switch(record.operation)
    {
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop:
        {
            if(get_marker_pushed_ranges().empty())
            {
                LOG_CRITICAL("roctxRangePop does not have corresponding roctxRangePush "
                             "(skipping)");
                return;
            }

            auto& entry        = get_marker_pushed_ranges().back();
            auto  hash         = std::get<0>(entry);
            begin_ts           = std::get<1>(entry);
            bool write_enabled = std::get<2>(entry);
            get_marker_pushed_ranges().pop_back();

            const char* name = nullptr;
            tim::get_hash_identifier_fast(hash, name);

            // Only write if writing was enabled at push time
            if(write_enabled && name)
            {
                m_writer.write_pop(name, begin_ts, ts, args_str, record);
            }
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop:
        {
            if(get_marker_started_ranges().empty())
            {
                LOG_CRITICAL("roctxRangeStop does not have corresponding roctxRangeStart "
                             "(skipping)");
                return;
            }

            auto& entry        = get_marker_started_ranges().back();
            auto  hash         = std::get<0>(entry);
            begin_ts           = std::get<1>(entry);
            bool write_enabled = std::get<2>(entry);
            get_marker_started_ranges().pop_back();

            const char* name = nullptr;
            tim::get_hash_identifier_fast(hash, name);
            auto range_id = data->args.roctxRangeStop.id;

            // Notify trace_control for region filtering
            m_controller.handle_range_stop(range_id);

            // Only write if writing was enabled at start time
            if(write_enabled && name)
            {
                m_writer.write_range_stop(name, begin_ts, ts, args_str, record);
            }
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA:
        {
            // roctxMarkA writes with duration (ENTER to EXIT timestamps)
            const char* name = data->args.roctxMarkA.message;

            if(should_write_markers())
            {
                m_writer.write_mark(name, begin_ts, ts, args_str, record);
            }
            break;
        }
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA:
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA:
        {
            // Push/Start operations are handled in their respective Pop/Stop EXIT cases
            // Nothing to do here in EXIT phase
            return;
        }
        default:
        {
            // For other operations (like roctxGetThreadId), write with duration
            if(should_write_markers())
            {
                auto name = get_operation_name(record);
                m_writer.write_api_call(name, begin_ts, ts, args_str, record);
            }
            break;
        }
    }
}

void
marker_client::handle_marker_control(rocprofiler_callback_tracing_record_t record)
{
    if(record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        m_controller.handle_pause();
    }
    else if(record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume &&
            record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        m_controller.handle_resume();
    }
}

void
marker_client::marker_core_callback(rocprofiler_callback_tracing_record_t record,
                                    rocprofiler_user_data_t*              user_data,
                                    void*                                 callback_data)
{
    if(!callback_data) return;
    auto* client = static_cast<marker_client*>(callback_data);

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
marker_client::marker_control_callback(rocprofiler_callback_tracing_record_t record,
                                       rocprofiler_user_data_t* /*user_data*/,
                                       void* callback_data)
{
    if(!callback_data) return;
    auto* client = static_cast<marker_client*>(callback_data);
    client->handle_marker_control(record);
}

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
