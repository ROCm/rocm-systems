// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "library/control.hpp"
#include "core/config.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <timemory/utility/delimit.hpp>

#include "logger/debug.hpp"

#if !defined(ROCPROFILER_CALL)
#    define ROCPROFILER_CALL(result)                                                     \
        {                                                                                \
            rocprofiler_status_t _rocp_status = (result);                                \
            if(_rocp_status != ROCPROFILER_STATUS_SUCCESS)                               \
            {                                                                            \
                std::string status_msg = rocprofiler_get_status_string(_rocp_status);    \
                LOG_WARNING("[{}][{}:{}] rocprofiler-sdk call [{}] failed with error "   \
                            "code {} :: {}",                                             \
                            #result, __FILE__, __LINE__, #result,                        \
                            static_cast<int>(_rocp_status), status_msg);                 \
            }                                                                            \
        }
#endif

namespace rocprofsys
{
namespace control
{
void
marker_watch_start_callback(rocprofiler_callback_tracing_record_t record,
                            rocprofiler_user_data_t* /*user_data*/, void* callback_data)
{
    if(!callback_data) return;
    auto* client = static_cast<control_client*>(callback_data);

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        auto* _data =
            static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);

        const char* _msg = _data->args.roctxRangeStartA.message;
        if(_msg != nullptr && client->m_trace_regions.count(_msg) > 0)
        {
            auto _range_id  = _data->retval.roctx_range_id_t_retval;
            bool _was_empty = false;
            {
                std::lock_guard<std::mutex> _lk(client->m_region_mutex);
                _was_empty = client->m_active_range_ids.empty();
                client->m_active_range_ids.insert(_range_id);
            }

            // First target region became active - trigger start callbacks
            if(_was_empty && !client->m_user_paused.load(std::memory_order_relaxed))
            {
                std::lock_guard<std::mutex> _lk(client->m_callback_mutex);
                client->trigger_callbacks(client->m_start_callbacks);
            }
        }
    }
}

void
marker_watch_stop_callback(rocprofiler_callback_tracing_record_t record,
                           rocprofiler_user_data_t* /*user_data*/, void* callback_data)
{
    if(!callback_data) return;
    auto* client = static_cast<control_client*>(callback_data);

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        auto* _data =
            static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);

        auto _range_id  = _data->args.roctxRangeStop.id;
        bool _now_empty = false;
        {
            std::lock_guard<std::mutex> _lk(client->m_region_mutex);
            auto                        _it = client->m_active_range_ids.find(_range_id);
            if(_it != client->m_active_range_ids.end())
            {
                client->m_active_range_ids.erase(_it);
                _now_empty = client->m_active_range_ids.empty();
            }
        }

        // Last target region exited - trigger stop callbacks
        if(_now_empty)
        {
            std::lock_guard<std::mutex> _lk(client->m_callback_mutex);
            client->trigger_callbacks(client->m_stop_callbacks);
        }
    }
}

void
marker_watch_pause_callback(rocprofiler_callback_tracing_record_t record,
                            rocprofiler_user_data_t* /*user_data*/, void* callback_data)
{
    if(!callback_data) return;

    auto* client = static_cast<control_client*>(callback_data);

    // Handle pause
    if(record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        if(client->region_filter_active())
            client->m_user_paused.store(true, std::memory_order_relaxed);

        std::lock_guard<std::mutex> _lk(client->m_callback_mutex);
        client->trigger_callbacks(client->m_stop_callbacks);
    }
    // Handle resume
    else if(record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume &&
            record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        bool _should_resume = true;
        if(client->region_filter_active())
        {
            client->m_user_paused.store(false, std::memory_order_relaxed);
            // Only resume if we're currently inside a target region
            std::lock_guard<std::mutex> _lk(client->m_region_mutex);
            _should_resume = !client->m_active_range_ids.empty();
        }

        if(_should_resume)
        {
            std::lock_guard<std::mutex> _lk(client->m_callback_mutex);
            client->trigger_callbacks(client->m_start_callbacks);
        }
    }
}

control_client::control_client(rocprofiler_context_id_t ctx)
: m_marker_watch_ctx{ ctx }
{
    // Parse trace region names from config
    auto _region_str = config::get_trace_region();
    if(!_region_str.empty())
    {
        for(auto& _name : tim::delimit(_region_str, ","))
        {
            auto _start = _name.find_first_not_of(" \t");
            auto _end   = _name.find_last_not_of(" \t");
            if(_start != std::string::npos)
                m_trace_regions.insert(_name.substr(_start, _end - _start + 1));
        }
        std::string _names;
        for(const auto& _n : m_trace_regions)
        {
            if(!_names.empty()) _names += ", ";
            _names += _n;
        }
        LOG_INFO("Control client: region filter active for regions: [{}]", _names);
    }
}

void
control_client::set_context(rocprofiler_context_id_t ctx)
{
    m_marker_watch_ctx = ctx;
}

void
control_client::configure_services(rocprofiler_context_id_t ctx)
{
    if(ctx.handle != 0) m_marker_watch_ctx = ctx;

    if(region_filter_active())
    {
        auto start_op = std::array<rocprofiler_tracing_operation_t, 1>{
            ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA
        };
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
            m_marker_watch_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
            start_op.data(), start_op.size(), marker_watch_start_callback, this));

        auto stop_op = std::array<rocprofiler_tracing_operation_t, 1>{
            ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop
        };
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
            m_marker_watch_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
            stop_op.data(), stop_op.size(), marker_watch_stop_callback, this));
    }

    auto control_ops = std::array<rocprofiler_tracing_operation_t, 2>{
        ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause,
        ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume
    };
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
        m_marker_watch_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
        control_ops.data(), control_ops.size(), marker_watch_pause_callback, this));

    // Context will be auto-started by rocprofiler-sdk when tool_init returns 0
}

void
control_client::shutdown()
{
    // NOTE: We don't explicitly stop marker_watch_ctx here because:
    // 1. It interferes with rocprofiler-sdk's async callback processing
    // 2. rocprofiler-sdk will handle stopping all contexts during finalization
    // 3. Explicitly stopping it causes hangs waiting for async copy callbacks

    // Clear callback vectors
    {
        std::lock_guard<std::mutex> _lk(m_callback_mutex);
        m_start_callbacks.clear();
        m_stop_callbacks.clear();
    }

    // Clear region filter state
    {
        std::lock_guard<std::mutex> _lk(m_region_mutex);
        m_active_range_ids.clear();
        m_trace_regions.clear();
    }
}

void
control_client::register_region_start_callback(callback_t callback)
{
    std::lock_guard<std::mutex> _lk(m_callback_mutex);
    m_start_callbacks.push_back(std::move(callback));
}

void
control_client::register_region_stop_callback(callback_t callback)
{
    std::lock_guard<std::mutex> _lk(m_callback_mutex);
    m_stop_callbacks.push_back(std::move(callback));
}

bool
control_client::region_filter_active() const
{
    return !m_trace_regions.empty();
}

void
control_client::trigger_callbacks(const std::vector<callback_t>& callbacks)
{
    for(const auto& cb : callbacks)
    {
        if(cb) cb();
    }
}

}  // namespace control
}  // namespace rocprofsys
