// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/trace_control.hpp"

#include "common/delimit.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "logger/debug.hpp"
#include "spdlog/fmt/bundled/ranges.h"

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace control
{

trace_control::trace_control(std::string_view trace_regions)
{
    if(trace_regions.empty())
    {
        return;
    }

    const auto delimited = rocprofsys::common::delimit(std::string{ trace_regions }, ",");
    m_trace_regions.insert(delimited.begin(), delimited.end());

    LOG_INFO("Trace controller: region filter active for regions: [{}]",
             fmt::join(m_trace_regions, ", "));
}

void
trace_control::force_initial_pause()
{
    if(!region_filter_active()) return;
    trigger_callbacks(m_stop_callbacks);
}

void
trace_control::handle_range_start(uint64_t range_id, const char* message)
{
    if(message == nullptr || m_trace_regions.count(message) == 0)
    {
        return;
    }

    bool was_empty = false;
    {
        std::lock_guard<std::mutex> const lk{ m_region_mutex };
        was_empty = m_active_range_ids.empty();
        m_active_range_ids.insert(range_id);
    }

    // First target region became active - trigger start callbacks
    if(was_empty && !m_user_paused.load(std::memory_order_relaxed))
    {
        trigger_callbacks(m_start_callbacks);
    }
}

void
trace_control::handle_range_stop(uint64_t range_id)
{
    bool now_empty = false;
    {
        std::lock_guard<std::mutex> const lk{ m_region_mutex };
        auto                              it = m_active_range_ids.find(range_id);
        if(it != m_active_range_ids.end())
        {
            m_active_range_ids.erase(it);
            now_empty = m_active_range_ids.empty();
        }
    }

    // Last target region exited - trigger stop callbacks
    if(now_empty)
    {
        if(m_user_paused.load(std::memory_order_relaxed))
        {
            // Region ended while user had it paused — stop callbacks were already
            // fired by handle_pause(), so skip them to avoid double-stop.
            LOG_WARNING(
                "Target region ended while paused. Subsequent resume will be ignored.");
            m_user_paused.store(false, std::memory_order_relaxed);
        }
        else
        {
            trigger_callbacks(m_stop_callbacks);
        }
    }
}

void
trace_control::handle_pause()
{
    if(region_filter_active())
    {
        std::lock_guard<std::mutex> const lk{ m_region_mutex };
        if(m_active_range_ids.empty())
        {
            LOG_WARNING("Pause requested outside of target region - ignoring");
            return;
        }
    }

    if(m_user_paused.load(std::memory_order_relaxed))
    {
        LOG_WARNING("Pause requested but tracing is already paused - ignoring");
        return;
    }

    m_user_paused.store(true, std::memory_order_relaxed);
    LOG_INFO("Pausing tracing session...");
    trigger_callbacks(m_stop_callbacks);
}

void
trace_control::handle_resume()
{
    if(!m_user_paused.load(std::memory_order_relaxed))
    {
        LOG_WARNING("Resume requested but tracing was not paused by user - ignoring");
        return;
    }

    if(region_filter_active())
    {
        std::lock_guard<std::mutex> const lk{ m_region_mutex };
        if(m_active_range_ids.empty())
        {
            LOG_WARNING("Resume requested outside of target region - ignoring");
            return;
        }
    }

    m_user_paused.store(false, std::memory_order_relaxed);
    LOG_INFO("Resuming tracing session...");
    trigger_callbacks(m_start_callbacks);
}

void
trace_control::shutdown()
{
    // Clear callback vectors
    {
        std::lock_guard<std::mutex> const lk{ m_callback_mutex };
        m_start_callbacks.clear();
        m_stop_callbacks.clear();
    }

    // Clear region filter state
    {
        std::lock_guard<std::mutex> const lk{ m_region_mutex };
        m_active_range_ids.clear();
        m_trace_regions.clear();
    }
}

void
trace_control::register_region_start_stop_callbacks(callback_t start_callback,
                                                    callback_t stop_callback)
{
    std::lock_guard<std::mutex> const lk{ m_callback_mutex };
    m_start_callbacks.push_back(std::move(start_callback));
    m_stop_callbacks.push_back(std::move(stop_callback));
}

bool
trace_control::region_filter_active() const
{
    return !m_trace_regions.empty();
}

bool
trace_control::should_write_markers() const
{
    // If no region filter, always write
    if(!region_filter_active())
    {
        return true;
    }

    // If paused by user, don't write
    if(m_user_paused.load(std::memory_order_relaxed))
    {
        return false;
    }

    // Only write if inside an active filtered region
    std::lock_guard<std::mutex> const lk{ m_region_mutex };
    return !m_active_range_ids.empty();
}

void
trace_control::trigger_callbacks(const std::vector<callback_t>& callbacks)
{
    std::lock_guard<std::mutex> const lk{ m_callback_mutex };
    for(const auto& cb : callbacks)
    {
        if(cb) cb();
    }
}

}  // namespace control
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
