// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/defines.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
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
/// (see rocprofiler-sdk registration.h). The trace_controller is instantiated earlier
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
class trace_controller
{
public:
    explicit trace_controller(std::string_view         trace_regions = {},
                              rocprofiler_context_id_t ctx           = { 0 });

    ~trace_controller() = default;

    void configure_services(rocprofiler_context_id_t ctx = { 0 }) ROCPROFSYS_INTERNAL_API;

    void shutdown() ROCPROFSYS_INTERNAL_API;

    void register_region_start_callback(callback_t callback) ROCPROFSYS_INTERNAL_API;

    void register_region_stop_callback(callback_t callback) ROCPROFSYS_INTERNAL_API;

    bool region_filter_active() const;

    // Handler methods called from callbacks
    void handle_range_start(uint64_t range_id, const char* message);
    void handle_range_stop(uint64_t range_id);
    void handle_pause();
    void handle_resume();

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
    static void trigger_callbacks(const std::vector<callback_t>& callbacks);
};
}  // namespace control
}  // namespace rocprofsys
