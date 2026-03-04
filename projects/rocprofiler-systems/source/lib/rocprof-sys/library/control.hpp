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

#pragma once

#pragma once

#include "core/defines.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace rocprofsys
{
namespace control
{

using callback_t = std::function<void()>;

class control_client
{
public:
    /// Setup the control client.
    control_client();

    /// Shutdown the control client.
    ~control_client();

    /// Register a callback to be triggered when tracing should start.
    /// Called when:
    /// - roctxRangeStartA matches a target region (ROCPROFSYS_TRACE_REGION)
    /// - First target region becomes active (0→1 active regions)
    void register_region_start_callback(callback_t callback) ROCPROFSYS_INTERNAL_API;

    /// Register a callback to be triggered when tracing should stop.
    /// Called when:
    /// - roctxRangeStop exits the last active target region (1→0 active regions)
    void register_region_stop_callback(callback_t callback) ROCPROFSYS_INTERNAL_API;

private:
    // rocprofiler-sdk context for marker watching
    rocprofiler_context_id_t m_marker_watch_ctx{ 0 };

    // Region filter state
    std::set<std::string, std::less<>> m_trace_regions;
    std::unordered_set<uint64_t>       m_active_range_ids;
    std::atomic<bool>                  m_user_paused{ false };

    /// Check if region filter is active
    bool region_filter_active() const;

    std::vector<callback_t> m_start_callbacks;
    std::vector<callback_t> m_stop_callbacks;

    std::mutex m_region_mutex;
    std::mutex m_callback_mutex;

    // Trigger all registered callbacks for a given event
    void trigger_callbacks(const std::vector<callback_t>& callbacks);

private:
    friend void marker_watch_start_callback(rocprofiler_callback_tracing_record_t,
                                            rocprofiler_user_data_t*, void*);
    friend void marker_watch_stop_callback(rocprofiler_callback_tracing_record_t,
                                           rocprofiler_user_data_t*, void*);
    friend void marker_watch_resume_callback(rocprofiler_callback_tracing_record_t,
                                             rocprofiler_user_data_t*, void*);
    friend void marker_watch_pause_callback(rocprofiler_callback_tracing_record_t,
                                            rocprofiler_user_data_t*, void*);
};
}  // namespace control
}  // namespace rocprofsys
