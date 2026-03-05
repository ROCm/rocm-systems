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

/// @brief Handles roctx-based tracing control: region filtering and pause/resume.
///
/// This class implements the "two-context pattern" for rocprofiler-sdk marker control.
/// It subscribes to MARKER_CONTROL_API (pause/resume) and optionally MARKER_CORE_API
/// (roctxRangeStart/Stop for region filtering) on a dedicated "always-on" context,
/// then triggers callbacks to start/stop the main tracing contexts.
///
/// @par Why this class doesn't own the context
///
/// rocprofiler-sdk requires all contexts to be created during the `tool_init` callback
/// (see rocprofiler-sdk registration.h). The control_client is instantiated earlier
/// (during library setup) so it can register start/stop callbacks before tool_init runs.
/// The actual context is created in tool_init and passed to configure_services().
///
/// This separation allows:
/// - Early callback registration (before rocprofiler-sdk initialization)
/// - Context creation at the required time (during tool_init)
/// - Proper lifecycle management (context owned by client_data, cleaned up by
/// rocprofiler-sdk)
///
/// @par Two-Context Pattern
///
/// The control context must remain active even when the main tracing context is paused.
/// If pause/resume were handled by the same context being paused, the resume callback
/// would never fire (because the context listening for it is stopped).
class control_client
{
public:
    explicit control_client(rocprofiler_context_id_t ctx = { 0 });

    ~control_client() = default;

    void set_context(rocprofiler_context_id_t ctx) ROCPROFSYS_INTERNAL_API;

    void configure_services(rocprofiler_context_id_t ctx = { 0 }) ROCPROFSYS_INTERNAL_API;

    void shutdown() ROCPROFSYS_INTERNAL_API;

    void register_region_start_callback(callback_t callback) ROCPROFSYS_INTERNAL_API;

    void register_region_stop_callback(callback_t callback) ROCPROFSYS_INTERNAL_API;

    bool region_filter_active() const;

private:
    rocprofiler_context_id_t m_marker_watch_ctx{ 0 };

    // Region filter state
    std::set<std::string, std::less<>> m_trace_regions;
    std::unordered_set<uint64_t>       m_active_range_ids;
    std::atomic<bool>                  m_user_paused{ false };

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
    friend void marker_watch_pause_callback(rocprofiler_callback_tracing_record_t,
                                            rocprofiler_user_data_t*, void*);
};
}  // namespace control
}  // namespace rocprofsys
