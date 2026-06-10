// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "roctx.hpp"

#include "common/delimit.hpp"
#include "core/control/session.hpp"

#include "logger/debug.hpp"
#include <spdlog/fmt/ranges.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>

namespace rocprofsys::control::triggers
{
roctx::roctx(session& sess, std::string_view trace_regions)
: m_session{ sess }
{
    if(!trace_regions.empty())
    {
        const auto delimited =
            rocprofsys::common::delimit(std::string{ trace_regions }, ",");
        m_trace_regions.insert(delimited.begin(), delimited.end());
    }

    if(filter_active())
    {
        LOG_INFO("roctx trigger: filter active for regions: [{}]",
                 fmt::join(m_trace_regions, ", "));
    }

    m_should_write.store(compute_should_write(), std::memory_order_relaxed);
}

vote
roctx::initial_vote() const noexcept
{
    return compute_vote();
}

void
roctx::on_range_start(std::uint64_t range_id, const char* message)
{
    if(message == nullptr || m_trace_regions.count(message) == 0) return;

    bool became_in_region = false;
    {
        std::scoped_lock const lk{ m_mutex };
        const bool             was_empty = m_active_range_ids.empty();
        m_active_range_ids.insert(range_id);
        if(was_empty)
        {
            // store under the mutex so the atomic snapshot is consistent
            // with the set (compute_vote / compute_should_write read both)
            m_in_region.store(true, std::memory_order_relaxed);
            became_in_region = true;
        }
    }

    if(became_in_region) refresh_state();
}

void
roctx::on_range_stop(std::uint64_t range_id)
{
    bool became_out_of_region = false;
    bool was_user_paused      = false;
    {
        std::scoped_lock const lk{ m_mutex };
        if(m_active_range_ids.erase(range_id) == 0) return;
        if(!m_active_range_ids.empty()) return;

        // Region ended while paused: silently clear user_paused so a later
        // region push behaves as a fresh start. Subsequent on_resume becomes
        // a no-op. Both atomic stores happen under the mutex so the set,
        // m_in_region, and m_user_paused snapshot consistently.
        was_user_paused = m_user_paused.exchange(false, std::memory_order_relaxed);
        m_in_region.store(false, std::memory_order_relaxed);
        became_out_of_region = true;
    }

    if(was_user_paused)
    {
        LOG_WARNING(
            "Target region ended while paused. Subsequent resume will be ignored.");
    }

    if(became_out_of_region) refresh_state();
}

void
roctx::on_pause()
{
    if(filter_active())
    {
        std::scoped_lock const lk{ m_mutex };
        if(m_active_range_ids.empty())
        {
            LOG_WARNING("Pause requested outside of target region - ignoring");
            return;
        }
    }

    bool expected = false;
    if(!m_user_paused.compare_exchange_strong(expected, true, std::memory_order_relaxed))
    {
        LOG_WARNING("Pause requested but tracing is already paused - ignoring");
        return;
    }

    LOG_INFO("Pausing tracing session...");
    refresh_state();
}

void
roctx::on_resume()
{
    if(!m_user_paused.load(std::memory_order_relaxed))
    {
        LOG_WARNING("Resume requested but tracing was not paused by user - ignoring");
        return;
    }

    if(filter_active())
    {
        std::scoped_lock const lk{ m_mutex };
        if(m_active_range_ids.empty())
        {
            LOG_WARNING("Resume requested outside of target region - ignoring");
            return;
        }
    }

    m_user_paused.store(false, std::memory_order_relaxed);
    LOG_INFO("Resuming tracing session...");
    refresh_state();
}

vote
roctx::compute_vote() const noexcept
{
    const bool filter    = filter_active();
    const bool in_region = m_in_region.load(std::memory_order_relaxed);
    const bool paused    = m_user_paused.load(std::memory_order_relaxed);
    const bool active    = !paused && (!filter || in_region);
    return active ? vote::active : vote::paused;
}

bool
roctx::compute_should_write() const noexcept
{
    const bool filter    = filter_active();
    const bool in_region = m_in_region.load(std::memory_order_relaxed);
    const bool paused    = m_user_paused.load(std::memory_order_relaxed);
    return !paused && (!filter || in_region);
}

void
roctx::refresh_state()
{
    m_should_write.store(compute_should_write(), std::memory_order_relaxed);
    m_session.publish(*this, compute_vote());
}
}  // namespace rocprofsys::control::triggers
