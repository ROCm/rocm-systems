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

class trace_control
{
public:
    explicit trace_control(std::string_view trace_regions = {});
    ~trace_control() = default;

    void shutdown();

    void register_region_pauser_resume_callbacks(callback_t resume_callback,
                                                 callback_t pause_callback);

    bool region_filter_active() const
    {
        return m_region_filter_active.load(std::memory_order_relaxed);
    }
    bool should_write_markers() const;

    void force_initial_pause();

    void handle_range_start(uint64_t range_id, const char* message);
    void handle_range_stop(uint64_t range_id);
    void handle_pause();
    void handle_resume();

private:
    std::set<std::string, std::less<>> m_trace_regions;
    std::unordered_set<uint64_t>       m_active_range_ids;
    std::atomic<bool>                  m_region_filter_active{ false };
    std::atomic<uint32_t>              m_active_region_count{ 0 };
    std::atomic<bool>                  m_user_paused{ false };

    std::vector<callback_t> m_resume_callbacks;
    std::vector<callback_t> m_pause_callbacks;

    std::mutex m_region_mutex;
    std::mutex m_callback_mutex;

    void trigger_callbacks(const std::vector<callback_t>& callbacks);
};
}  // namespace control
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
