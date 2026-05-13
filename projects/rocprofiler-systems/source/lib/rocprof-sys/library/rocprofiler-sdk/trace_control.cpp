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

namespace
{
// True on exactly the threads that have called handle_pause() without a matching
// handle_resume(). Checked in should_write_markers() with zero overhead (no lock,
// no syscall) on every roctx marker event.
thread_local bool tl_thread_paused = false;
}  // namespace

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
    }

    if(now_empty)
    {
        // Reset the count atomically. Paused threads still carry tl_thread_paused=true
        // and will clear it when they call handle_resume(); the early-return in that
        // path (region filter active + no active ranges) prevents a spurious callback.
        const std::uint32_t prev =
            m_paused_thread_count.exchange(0, std::memory_order_relaxed);
        had_paused = (prev > 0);

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

    if(tl_thread_paused)
    {
        LOG_WARNING("Pause requested but thread {} is already paused - ignoring", tid);
        return;
    }

    tl_thread_paused = true;
    const bool first_pause =
        (m_paused_thread_count.fetch_add(1, std::memory_order_relaxed) == 0);

    if(first_pause)
    {
        LOG_INFO("Pausing tracing session (thread {})...", tid);
        trigger_callbacks(m_pause_callbacks);
    }
}

void
trace_control::handle_resume(std::uint64_t tid)
{
    if(!tl_thread_paused)
    {
        LOG_WARNING("Resume requested but thread {} was not paused - ignoring", tid);
        return;
    }

    // Clear the per-thread flag unconditionally so the thread can write markers
    // even if we skip the callbacks below (e.g. region ended while paused).
    tl_thread_paused = false;

    if(region_filter_active())
    {
        std::scoped_lock const lk{ m_region_mutex };
        if(m_active_range_ids.empty())
        {
            // Region ended while paused; count was already reset by handle_range_stop.
            LOG_WARNING("Resume requested outside of target region - ignoring");
            return;
        }
    }

    const bool last_resume =
        (m_paused_thread_count.fetch_sub(1, std::memory_order_relaxed) == 1);

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
    if(tl_thread_paused)
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
