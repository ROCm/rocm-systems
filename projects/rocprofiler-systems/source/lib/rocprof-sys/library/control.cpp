// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/control.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <timemory/utility/delimit.hpp>

#include "library/components/category_region.hpp"
#include "logger/debug.hpp"

namespace rocprofsys
{
namespace control
{
namespace
{
inline void
check_rocprofiler_status(rocprofiler_status_t status, const char* msg)
{
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("{}: {}", msg, rocprofiler_get_status_string(status));
    }
}

void
marker_watch_start_callback(rocprofiler_callback_tracing_record_t record,
                            rocprofiler_user_data_t* /*user_data*/, void* callback_data)
{
    if(!callback_data) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;
    auto* controller = static_cast<trace_controller*>(callback_data);
    auto* data =
        static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);

    if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA)
    {
        LOG_CRITICAL("MARKER WATCH START {}", record.operation);
        controller->handle_range_start(data->retval.roctx_range_id_t_retval,
                                       data->args.roctxRangeStartA.message);
        component::category_region<tim::category::rocm_marker_api>::start(
            data->args.roctxRangeStartA.message);
    }
    else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop)
    {
        LOG_CRITICAL("MARKER WATCH STOP {}", record.operation);
        controller->handle_range_stop(data->args.roctxRangeStop.id);
        component::category_region<tim::category::rocm_marker_api>::stop(
            data->args.roctxRangeStartA.message);
    }
}

void
marker_watch_pause_callback(rocprofiler_callback_tracing_record_t record,
                            rocprofiler_user_data_t* /*user_data*/, void* callback_data)
{
    if(!callback_data) return;

    auto* controller = static_cast<trace_controller*>(callback_data);

    // Handle pause
    if(record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        controller->handle_pause();
        return;
    }

    // Handle resume
    if(record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        controller->handle_resume();
    }
}
}  // namespace

trace_controller::trace_controller(std::string_view         trace_regions,
                                   rocprofiler_context_id_t ctx)
: m_marker_watch_ctx{ ctx }
{
    // Parse trace region names from parameter
    if(!trace_regions.empty())
    {
        for(auto& name : tim::delimit(std::string{ trace_regions }, ","))
        {
            auto start = name.find_first_not_of(" \t");
            auto end   = name.find_last_not_of(" \t");
            if(start != std::string::npos)
                m_trace_regions.insert(name.substr(start, end - start + 1));
        }
        std::string names;
        for(const auto& n : m_trace_regions)
        {
            if(!names.empty()) names += ", ";
            names += n;
        }
        LOG_INFO("Trace controller: region filter active for regions: [{}]", names);
    }
}

void
trace_controller::handle_range_start(uint64_t range_id, const char* message)
{
    if(message == nullptr || m_trace_regions.count(message) == 0) return;

    bool was_empty = false;
    {
        std::lock_guard<std::mutex> lk(m_region_mutex);
        was_empty = m_active_range_ids.empty();
        m_active_range_ids.insert(range_id);
    }

    // First target region became active - trigger start callbacks
    if(was_empty && !m_user_paused.load(std::memory_order_relaxed))
    {
        std::lock_guard<std::mutex> lk(m_callback_mutex);
        trigger_callbacks(m_start_callbacks);
    }
}

void
trace_controller::handle_range_stop(uint64_t range_id)
{
    bool now_empty = false;
    {
        std::lock_guard<std::mutex> lk(m_region_mutex);
        auto                        it = m_active_range_ids.find(range_id);
        if(it != m_active_range_ids.end())
        {
            m_active_range_ids.erase(it);
            now_empty = m_active_range_ids.empty();
        }
    }

    // Last target region exited - trigger stop callbacks
    if(now_empty)
    {
        std::lock_guard<std::mutex> lk(m_callback_mutex);
        trigger_callbacks(m_stop_callbacks);
    }
}

void
trace_controller::handle_pause()
{
    if(region_filter_active()) m_user_paused.store(true, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(m_callback_mutex);
    trigger_callbacks(m_stop_callbacks);
}

void
trace_controller::handle_resume()
{
    bool should_resume = true;
    if(region_filter_active())
    {
        m_user_paused.store(false, std::memory_order_relaxed);
        // Only resume if we're currently inside a target region
        std::lock_guard<std::mutex> lk(m_region_mutex);
        should_resume = !m_active_range_ids.empty();
    }

    if(should_resume)
    {
        std::lock_guard<std::mutex> lk(m_callback_mutex);
        trigger_callbacks(m_start_callbacks);
    }
}

void
trace_controller::configure_services(rocprofiler_context_id_t ctx)
{
    if(ctx.handle != 0) m_marker_watch_ctx = ctx;

    if(region_filter_active())
    {
        auto start_op = std::array<rocprofiler_tracing_operation_t, 2>{
            ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA,
            ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop
        };
        check_rocprofiler_status(
            rocprofiler_configure_callback_tracing_service(
                m_marker_watch_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                start_op.data(), start_op.size(), marker_watch_start_callback, this),
            "Failed to configure marker start callback");
    }

    auto control_ops = std::array<rocprofiler_tracing_operation_t, 2>{
        ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause,
        ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume
    };
    check_rocprofiler_status(
        rocprofiler_configure_callback_tracing_service(
            m_marker_watch_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
            control_ops.data(), control_ops.size(), marker_watch_pause_callback, this),
        "Failed to configure marker control callback");

    // Context will be auto-started by rocprofiler-sdk when tool_init returns 0
}

void
trace_controller::shutdown()
{
    // NOTE: We don't explicitly stop marker_watch_ctx here because:
    // 1. It interferes with rocprofiler-sdk's async callback processing
    // 2. rocprofiler-sdk will handle stopping all contexts during finalization
    // 3. Explicitly stopping it causes hangs waiting for async copy callbacks

    // Clear callback vectors
    {
        std::lock_guard<std::mutex> lk(m_callback_mutex);
        m_start_callbacks.clear();
        m_stop_callbacks.clear();
    }

    // Clear region filter state
    {
        std::lock_guard<std::mutex> lk(m_region_mutex);
        m_active_range_ids.clear();
        m_trace_regions.clear();
    }
}

void
trace_controller::register_region_start_callback(callback_t callback)
{
    std::lock_guard<std::mutex> lk(m_callback_mutex);
    m_start_callbacks.push_back(std::move(callback));
}

void
trace_controller::register_region_stop_callback(callback_t callback)
{
    std::lock_guard<std::mutex> lk(m_callback_mutex);
    m_stop_callbacks.push_back(std::move(callback));
}

bool
trace_controller::region_filter_active() const
{
    return !m_trace_regions.empty();
}

void
trace_controller::trigger_callbacks(const std::vector<callback_t>& callbacks)
{
    for(const auto& cb : callbacks)
    {
        if(cb) cb();
    }
}

}  // namespace control
}  // namespace rocprofsys
