// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "session.hpp"

#include "common/delimit.hpp"

#include "logger/debug.hpp"
#include <spdlog/fmt/ranges.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::control
{
session::session(std::string_view trace_regions)
{
    if(trace_regions.empty()) return;

    const auto delimited = rocprofsys::delimit(std::string{ trace_regions }, ",");
    m_trace_regions.insert(delimited.begin(), delimited.end());
    m_region_filter_active.store(!m_trace_regions.empty(), std::memory_order_relaxed);

    if(region_filter_active())
    {
        recompute_active();
        LOG_INFO("Trace controller: region filter active for regions: [{}]",
                 fmt::join(m_trace_regions, ", "));
    }
}

void
session::force_initial_pause()
{
    if(!region_filter_active()) return;
    recompute_active();
    notify_pause();
}

void
session::handle_range_start(std::uint64_t range_id, const char* message)
{
    if(message == nullptr || m_trace_regions.count(message) == 0) return;

    bool was_empty = false;
    {
        const std::scoped_lock lk{ m_region_mutex };
        was_empty = m_active_range_ids.empty();
        m_active_range_ids.insert(range_id);
        m_active_region_count.store(static_cast<std::uint32_t>(m_active_range_ids.size()),
                                    std::memory_order_relaxed);
    }

    if(was_empty && !m_user_paused.load(std::memory_order_relaxed))
    {
        recompute_active();
        notify_resume();
    }
}

void
session::handle_range_stop(std::uint64_t range_id)
{
    bool now_empty  = false;
    bool had_paused = false;
    {
        const std::scoped_lock lk{ m_region_mutex };
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
            recompute_active();
        }
        else
        {
            recompute_active();
            notify_pause();
        }
    }
}

void
session::handle_pause(std::uint64_t tid)
{
    if(region_filter_active())
    {
        const std::scoped_lock lk{ m_region_mutex };
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
    recompute_active();
    LOG_INFO("Pausing tracing session (thread {})...", tid);
    notify_pause();
}

void
session::handle_resume(std::uint64_t tid)
{
    if(!m_user_paused.load(std::memory_order_relaxed))
    {
        LOG_WARNING("Resume requested but tracing was not paused by user - ignoring");
        return;
    }

    if(region_filter_active())
    {
        const std::scoped_lock lk{ m_region_mutex };
        if(m_active_range_ids.empty())
        {
            LOG_WARNING("Resume requested outside of target region - ignoring");
            return;
        }
    }

    m_user_paused.store(false, std::memory_order_relaxed);
    recompute_active();
    LOG_INFO("Resuming tracing session (thread {})...", tid);
    notify_resume();
}

void
session::shutdown()
{
    {
        const std::scoped_lock lk{ m_subscribers_mutex };
        m_subscribers.clear();
    }

    {
        const std::scoped_lock lk{ m_region_mutex };
        m_active_range_ids.clear();
        m_active_region_count.store(0, std::memory_order_relaxed);
        m_trace_regions.clear();
        m_region_filter_active.store(false, std::memory_order_relaxed);
    }

    m_user_paused.store(false, std::memory_order_relaxed);
    recompute_active();
}

void
session::subscribe(subscriber sub)
{
    const std::scoped_lock lk{ m_subscribers_mutex };
    m_subscribers.push_back(std::move(sub));
}

// Mirrors the original should_write_markers() formula:
//   no filter            -> always active
//   filter && paused     -> inactive
//   filter && !paused    -> active iff at least one target region is open
void
session::recompute_active()
{
    const bool filter   = m_region_filter_active.load(std::memory_order_relaxed);
    const bool paused   = m_user_paused.load(std::memory_order_relaxed);
    const auto in_range = m_active_region_count.load(std::memory_order_relaxed) > 0;
    m_active.store(!filter || (!paused && in_range), std::memory_order_relaxed);
}

void
session::notify_pause()
{
    const std::scoped_lock lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        if(sub.on_pause) sub.on_pause();
    }
}

void
session::notify_resume()
{
    const std::scoped_lock lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        if(sub.on_resume) sub.on_resume();
    }
}
}  // namespace rocprofsys::control
