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
#include <spdlog/fmt/ranges.h>

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
    m_region_filter_active.store(!m_trace_regions.empty(), std::memory_order_relaxed);

    LOG_INFO("Trace controller: region filter active for regions: [{}]",
             fmt::join(m_trace_regions, ", "));
}

void
trace_control::force_initial_pause()
{
    if(!region_filter_active())
    {
        return;
    }
    trigger_callbacks(m_pause_callbacks);
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
        std::scoped_lock const lk{ m_region_mutex };
        was_empty = m_active_range_ids.empty();
        m_active_range_ids.insert(range_id);
        m_active_region_count.store(static_cast<uint32_t>(m_active_range_ids.size()),
                                    std::memory_order_relaxed);
    }

    if(was_empty && !m_user_paused.load(std::memory_order_relaxed))
    {
        trigger_callbacks(m_resume_callbacks);
    }
}

void
trace_control::handle_range_stop(uint64_t range_id)
{
    bool now_empty = false;
    {
        std::scoped_lock const lk{ m_region_mutex };
        auto                   it = m_active_range_ids.find(range_id);
        if(it != m_active_range_ids.end())
        {
            m_active_range_ids.erase(it);
            now_empty = m_active_range_ids.empty();
            m_active_region_count.store(static_cast<uint32_t>(m_active_range_ids.size()),
                                        std::memory_order_relaxed);
        }
    }

    if(now_empty)
    {
        if(m_user_paused.load(std::memory_order_relaxed))
        {
            LOG_WARNING(
                "Target region ended while paused. Subsequent resume will be ignored.");
            m_user_paused.store(false, std::memory_order_relaxed);
        }
        else
        {
            trigger_callbacks(m_pause_callbacks);
        }
    }
}

void
trace_control::handle_pause()
{
    if(region_filter_active())
    {
        std::scoped_lock const lk{ m_region_mutex };
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
    trigger_callbacks(m_pause_callbacks);
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
        std::scoped_lock const lk{ m_region_mutex };
        if(m_active_range_ids.empty())
        {
            LOG_WARNING("Resume requested outside of target region - ignoring");
            return;
        }
    }

    m_user_paused.store(false, std::memory_order_relaxed);
    LOG_INFO("Resuming tracing session...");
    trigger_callbacks(m_resume_callbacks);
}

void
trace_control::shutdown()
{
    {
        std::scoped_lock const lk{ m_callback_mutex };
        m_resume_callbacks.clear();
        m_pause_callbacks.clear();
    }

    {
        std::scoped_lock const lk{ m_region_mutex };
        m_active_range_ids.clear();
        m_active_region_count.store(0, std::memory_order_relaxed);
        m_trace_regions.clear();
        m_region_filter_active.store(false, std::memory_order_relaxed);
    }
}

void
trace_control::register_region_pauser_resume_callbacks(callback_t start_callback,
                                                       callback_t stop_callback)
{
    std::scoped_lock const lk{ m_callback_mutex };
    m_resume_callbacks.push_back(std::move(start_callback));
    m_pause_callbacks.push_back(std::move(stop_callback));
}

bool
trace_control::should_write_markers() const
{
    if(!region_filter_active())
    {
        return true;
    }

    if(m_user_paused.load(std::memory_order_relaxed))
    {
        return false;
    }

    return m_active_region_count.load(std::memory_order_relaxed) > 0;
}

void
trace_control::trigger_callbacks(const std::vector<callback_t>& callbacks)
{
    std::scoped_lock const lk{ m_callback_mutex };
    for(const auto& cb : callbacks)
    {
        if(cb)
        {
            cb();
        }
    }
}

}  // namespace control
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
