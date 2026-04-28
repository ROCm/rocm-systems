// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Implementation included from pause_interval_registry.hpp (template body).

#include <algorithm>

// Log includes — uses the existing rocprof-sys logging infrastructure.
#include "logger/debug.hpp"

namespace rocprofsys::sampling
{

template <class ClockPolicy>
bool
pause_interval_registry<ClockPolicy>::pause() noexcept
{
    bool expected = false;
    if(!paused_.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                        std::memory_order_relaxed))
    {
        LOG_WARNING("sampling::pause() called but sampling is already paused");
        return false;
    }
    pending_pause_ts_.store(clock_.now_ns(), std::memory_order_seq_cst);
    LOG_DEBUG("Pausing sampling...");
    return true;
}

template <class ClockPolicy>
bool
pause_interval_registry<ClockPolicy>::resume() noexcept
{
    bool expected = true;
    if(!paused_.compare_exchange_strong(expected, false, std::memory_order_acq_rel,
                                        std::memory_order_relaxed))
    {
        LOG_WARNING("sampling::resume() called but sampling is not paused");
        return false;
    }
    uint64_t pause_ts  = pending_pause_ts_.exchange(0, std::memory_order_seq_cst);
    uint64_t resume_ts = clock_.now_ns();
    {
        std::lock_guard<std::mutex> lk(pause_mutex_);
        intervals_.push_back({ pause_ts, resume_ts });
    }
    LOG_DEBUG("Resuming sampling...");
    return true;
}

template <class ClockPolicy>
bool
pause_interval_registry<ClockPolicy>::spans_pause_interval(uint64_t beg_ns,
                                                           uint64_t end_ns) const noexcept
{
    std::lock_guard<std::mutex> lk(pause_mutex_);
    // Intervals are appended in monotonic order; binary search on resume_ns.
    auto it = std::lower_bound(
        intervals_.begin(), intervals_.end(), beg_ns,
        [](pause_interval const& iv, uint64_t t) { return iv.resume_ns < t; });
    for(; it != intervals_.end(); ++it)
    {
        if(it->pause_ns >= end_ns) break;
        // interval [pause_ns, resume_ns) overlaps [beg_ns, end_ns): half-open convention.
        // A sample ending exactly at pause_ns is NOT considered overlapping (it completed
        // before the pause began), and a sample starting exactly at resume_ns is
        // similarly not filtered.
        if(it->resume_ns > beg_ns && it->pause_ns < end_ns) return true;
    }
    return false;
}

template <class ClockPolicy>
uint64_t
pause_interval_registry<ClockPolicy>::max_resume_ns_overlapping(
    uint64_t beg_ns, uint64_t end_ns) const noexcept
{
    std::lock_guard<std::mutex> lk(pause_mutex_);
    uint64_t                    max_resume = 0;
    auto                        it         = std::lower_bound(
        intervals_.begin(), intervals_.end(), beg_ns,
        [](pause_interval const& iv, uint64_t t) { return iv.resume_ns < t; });
    for(; it != intervals_.end(); ++it)
    {
        if(it->pause_ns >= end_ns) break;
        if(it->resume_ns > beg_ns && it->pause_ns < end_ns)
        {
            if(it->resume_ns > max_resume) max_resume = it->resume_ns;
        }
    }
    return max_resume;
}

}  // namespace rocprofsys::sampling
