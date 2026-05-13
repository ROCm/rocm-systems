// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/trace_control.hpp"

#include "common/delimit.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unistd.h>
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

    if(was_empty && m_paused_thread_count.load(std::memory_order_relaxed) == 0)
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

        if(now_empty && !m_paused_thread_ids.empty())
        {
            had_paused = true;
            m_paused_thread_ids.clear();
            m_paused_thread_count.store(0, std::memory_order_relaxed);
        }
    }

    if(now_empty)
    {
        if(had_paused)
        {
            LOG_WARNING(
                "Target region ended while paused. Subsequent resume will be ignored.");
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

    bool first_pause = false;
    {
        std::scoped_lock const lk{ m_region_mutex };
        auto [it, inserted] = m_paused_thread_ids.insert(tid);
        if(!inserted)
        {
            LOG_WARNING("Pause requested but thread {} is already paused - ignoring",
                        tid);
            return;
        }
        first_pause = (m_paused_thread_ids.size() == 1);
        m_paused_thread_count.store(
            static_cast<std::uint32_t>(m_paused_thread_ids.size()),
            std::memory_order_relaxed);
    }

    if(first_pause)
    {
        LOG_INFO("Pausing tracing session (thread {})...", tid);
        trigger_callbacks(m_pause_callbacks);
    }
}

void
trace_control::handle_resume(std::uint64_t tid)
{
    if(region_filter_active())
    {
        std::scoped_lock const lk{ m_region_mutex };
        if(m_active_range_ids.empty())
        {
            LOG_WARNING("Resume requested outside of target region - ignoring");
            return;
        }
    }

    bool last_resume = false;
    {
        std::scoped_lock const lk{ m_region_mutex };
        auto                   it = m_paused_thread_ids.find(tid);
        if(it == m_paused_thread_ids.end())
        {
            LOG_WARNING(
                "Resume requested but thread {} was not paused by user - ignoring", tid);
            return;
        }
        m_paused_thread_ids.erase(it);
        last_resume = m_paused_thread_ids.empty();
        m_paused_thread_count.store(
            static_cast<std::uint32_t>(m_paused_thread_ids.size()),
            std::memory_order_relaxed);
    }

    if(last_resume)
    {
        LOG_INFO("Resuming tracing session (thread {})...", tid);
        trigger_callbacks(m_resume_callbacks);
    }
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
        m_paused_thread_ids.clear();
        m_paused_thread_count.store(0, std::memory_order_relaxed);
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
    // Fast path: if any threads are paused, check whether the calling thread is one of
    // them. Skip the lock entirely when no threads are paused (the common case).
    if(m_paused_thread_count.load(std::memory_order_relaxed) > 0)
    {
        const auto             tid = static_cast<std::uint64_t>(::gettid());
        std::scoped_lock const lk{ m_region_mutex };
        if(m_paused_thread_ids.count(tid) > 0)
        {
            return false;
        }
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
