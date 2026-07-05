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
trace_control::handle_range_start(std::uint64_t range_id, const char* message)
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
        m_active_region_count.store(static_cast<std::uint32_t>(m_active_range_ids.size()),
                                    std::memory_order_relaxed);
    }

    if(was_empty && !m_user_paused.load(std::memory_order_relaxed))
    {
        trigger_callbacks(m_resume_callbacks);
    }
}

void
trace_control::handle_range_stop(std::uint64_t range_id)
{
    bool now_empty  = false;
    bool had_paused = false;
    {
        std::scoped_lock const lk{ m_region_mutex };
        auto                   it = m_active_range_ids.find(range_id);
        if(it != m_active_range_ids.end())
        {
            m_active_range_ids.erase(it);
            now_empty = m_active_range_ids.empty();
            m_active_region_count.store(
                static_cast<std::uint32_t>(m_active_range_ids.size()),
                std::memory_order_relaxed);
        }
    }

    if(now_empty)
    {
        had_paused = m_user_paused.load(std::memory_order_relaxed);
        if(had_paused)
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
trace_control::handle_pause(std::uint64_t tid)
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
    LOG_INFO("Pausing tracing session (thread {})...", tid);
    trigger_callbacks(m_pause_callbacks);
}

void
trace_control::handle_resume(std::uint64_t tid)
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
    LOG_INFO("Resuming tracing session (thread {})...", tid);
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

    m_user_paused.store(false, std::memory_order_relaxed);
}

void
trace_control::register_region_pause_resume_callbacks(callback_t resume_callback,
                                                      callback_t pause_callback)
{
    std::scoped_lock const lk{ m_callback_mutex };
    m_resume_callbacks.push_back(std::move(resume_callback));
    m_pause_callbacks.push_back(std::move(pause_callback));
}

bool
trace_control::should_write_markers() const
{
    if(m_user_paused.load(std::memory_order_relaxed))
    {
        return false;
    }

    if(!region_filter_active())
    {
        return true;
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
