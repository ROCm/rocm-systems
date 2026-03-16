// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace control
{

using callback_t = std::function<void()>;

// Handles roctx-based tracing control: region filtering and pause/resume.
// Provides handler methods that roctx_client calls from its callbacks.
// Triggers registered start/stop callbacks to control main tracing contexts.
class trace_control
{
public:
    explicit trace_control(std::string_view trace_regions = {});
    ~trace_control() = default;

    void shutdown();

    void register_region_start_callback(callback_t callback);
    void register_region_stop_callback(callback_t callback);

    bool region_filter_active() const;

    // Returns true if currently inside an active filtered region (or if no filter active)
    bool should_write_markers() const;

    // Handler methods called by roctx_client
    void handle_range_start(uint64_t range_id, const char* message);
    void handle_range_stop(uint64_t range_id);
    void handle_pause();
    void handle_resume();

private:
    // Region filter state
    std::set<std::string, std::less<>> m_trace_regions;
    std::unordered_set<uint64_t>       m_active_range_ids;
    std::atomic<bool>                  m_user_paused{ false };

    std::vector<callback_t> m_start_callbacks;
    std::vector<callback_t> m_stop_callbacks;

    mutable std::mutex m_region_mutex;
    std::mutex         m_callback_mutex;

    void trigger_callbacks(const std::vector<callback_t>& callbacks);
};
}  // namespace control
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
