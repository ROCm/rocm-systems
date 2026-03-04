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

#include "core/config.hpp"
#include "library/control.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <timemory/utility/delimit.hpp>

#include "logger/debug.hpp"

#include <atomic>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

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
namespace
{
// Control context
rocprofiler_context_id_t g_marker_watch_ctx = { 0 };

// Region filter state
std::set<std::string, std::less<>> g_trace_regions;     // parsed region names
std::unordered_set<uint64_t>       g_active_range_ids;  // currently active target ranges
std::mutex                         g_region_mutex;      // protects g_active_range_ids
std::atomic<bool>                  g_user_paused{ false };

// Callback registrations
std::vector<callback_t> g_start_callbacks;
std::vector<callback_t> g_stop_callbacks;
std::vector<callback_t> g_pause_callbacks;
std::vector<callback_t> g_resume_callbacks;
std::mutex              g_callback_mutex;

inline bool
region_filter_active()
{
    return !g_trace_regions.empty();
}

// Trigger all registered callbacks for a given event
void
trigger_callbacks(const std::vector<callback_t>& callbacks)
{
    for(const auto& cb : callbacks)
    {
        if(cb) cb();
    }
}

// Marker watch callback - watches roctxRangeStart/Stop and roctxProfilerPause/Resume
void
tool_marker_watch_callback(rocprofiler_callback_tracing_record_t record,
                           rocprofiler_user_data_t* /*user_data*/,
                           void* /*callback_data*/)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API &&
       record.kind != ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API)
        return;

    // Handle roctxProfilerPause/Resume (control API)
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API)
    {
        if(record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause &&
           record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            if(region_filter_active())
                g_user_paused.store(true, std::memory_order_relaxed);

            std::lock_guard<std::mutex> _lk(g_callback_mutex);
            trigger_callbacks(g_pause_callbacks);
        }
        else if(record.operation ==
                    ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume &&
                record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
        {
            bool _should_resume = true;
            if(region_filter_active())
            {
                g_user_paused.store(false, std::memory_order_relaxed);
                // Only resume if we're currently inside a target region
                std::lock_guard<std::mutex> _lk(g_region_mutex);
                _should_resume = !g_active_range_ids.empty();
            }

            if(_should_resume)
            {
                std::lock_guard<std::mutex> _lk(g_callback_mutex);
                trigger_callbacks(g_resume_callbacks);
            }
        }
        return;
    }

    // Handle roctxRangeStart/Stop (marker core API) for region filtering
    if(!region_filter_active()) return;

    auto* _data =
        static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        // Nothing to do in ENTER phase for region filtering
        return;
    }

    // EXIT phase
    if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA)
    {
        const char* _msg = _data->args.roctxRangeStartA.message;
        if(_msg != nullptr && g_trace_regions.count(_msg) > 0)
        {
            auto _range_id  = _data->retval.roctx_range_id_t_retval;
            bool _was_empty = false;
            {
                std::lock_guard<std::mutex> _lk(g_region_mutex);
                _was_empty = g_active_range_ids.empty();
                g_active_range_ids.insert(_range_id);
            }

            // First target region became active - trigger start callbacks
            if(_was_empty && !g_user_paused.load(std::memory_order_relaxed))
            {
                std::lock_guard<std::mutex> _lk(g_callback_mutex);
                trigger_callbacks(g_start_callbacks);
            }
        }
    }
    else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop)
    {
        auto _range_id  = _data->args.roctxRangeStop.id;
        bool _now_empty = false;
        {
            std::lock_guard<std::mutex> _lk(g_region_mutex);
            auto                        _it = g_active_range_ids.find(_range_id);
            if(_it != g_active_range_ids.end())
            {
                g_active_range_ids.erase(_it);
                _now_empty = g_active_range_ids.empty();
            }
        }

        // Last target region exited - trigger stop callbacks
        if(_now_empty)
        {
            std::lock_guard<std::mutex> _lk(g_callback_mutex);
            trigger_callbacks(g_stop_callbacks);
        }
    }
}

}  // namespace

void
register_start_callback(callback_t callback)
{
    std::lock_guard<std::mutex> _lk(g_callback_mutex);
    g_start_callbacks.push_back(std::move(callback));
}

void
register_stop_callback(callback_t callback)
{
    std::lock_guard<std::mutex> _lk(g_callback_mutex);
    g_stop_callbacks.push_back(std::move(callback));
}

void
register_pause_callback(callback_t callback)
{
    std::lock_guard<std::mutex> _lk(g_callback_mutex);
    g_pause_callbacks.push_back(std::move(callback));
}

void
register_resume_callback(callback_t callback)
{
    std::lock_guard<std::mutex> _lk(g_callback_mutex);
    g_resume_callbacks.push_back(std::move(callback));
}

void
setup()
{
    // Parse trace region names
    auto _region_str = config::get_trace_region();
    if(!_region_str.empty())
    {
        for(auto& _name : tim::delimit(_region_str, ","))
        {
            // trim whitespace
            auto _start = _name.find_first_not_of(" \t");
            auto _end   = _name.find_last_not_of(" \t");
            if(_start != std::string::npos)
                g_trace_regions.insert(_name.substr(_start, _end - _start + 1));
        }
        std::string _names;
        for(const auto& _n : g_trace_regions)
        {
            if(!_names.empty()) _names += ", ";
            _names += _n;
        }
        LOG_INFO("Control client: region filter active for regions: [{}]", _names);
    }

    // Create marker watch context (always-on for pause/resume support)
    ROCPROFILER_CALL(rocprofiler_create_context(&g_marker_watch_ctx));

    // Configure marker core API (for roctxRangeStart/Stop) if region filter active
    if(region_filter_active())
    {
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
            g_marker_watch_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API, nullptr, 0,
            tool_marker_watch_callback, nullptr));
    }

    // Always configure marker control API (for roctxProfilerPause/Resume)
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
        g_marker_watch_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API, nullptr, 0,
        tool_marker_watch_callback, nullptr));

    // Context will be auto-started by rocprofiler-sdk when tool_init returns 0
}

void
shutdown()
{
    // Cleanup callback vectors
    {
        std::lock_guard<std::mutex> _lk(g_callback_mutex);
        g_start_callbacks.clear();
        g_stop_callbacks.clear();
        g_pause_callbacks.clear();
        g_resume_callbacks.clear();
    }

    // Clear region filter state
    {
        std::lock_guard<std::mutex> _lk(g_region_mutex);
        g_active_range_ids.clear();
    }
    g_trace_regions.clear();
}

}  // namespace control
}  // namespace rocprofsys
