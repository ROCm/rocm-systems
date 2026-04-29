// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/pause_interval.hpp"

#include "logger/debug.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace rocprofsys::sampling
{

// pause_interval_registry — manages pause/resume intervals for sample filtering.
// Thread-safety:
//   - paused_ is a standalone atomic (lock-free reads from any thread)
//   - intervals_ is protected by pause_mutex_ (write on resume; reads under lock)
//   - DEC-5: PauseMutex is order-index 4; never held while another std::mutex is held
template <class ClockPolicy>
class pause_interval_registry
{
public:
    explicit pause_interval_registry(ClockPolicy& clock) noexcept
    : clock_(clock)
    {}

    // Record pause timestamp and set paused flag.
    // Returns false (with L37) if already paused.
    bool pause() noexcept;

    // Record resume timestamp, store interval, clear paused flag.
    // Returns false (with L39) if not currently paused.
    bool resume() noexcept;

    // Returns true if paused flag is currently set.
    [[nodiscard]] bool is_paused() const noexcept
    {
        return paused_.load(std::memory_order_relaxed);
    }

    // Returns true if the interval [beg_ns, end_ns] overlaps any stored pause interval.
    // Thread-safe read under pause_mutex_.
    [[nodiscard]] bool spans_pause_interval(uint64_t beg_ns,
                                            uint64_t end_ns) const noexcept;

    // Returns the maximum resume_ns among all pause intervals that overlap [beg_ns,
    // end_ns], or 0 if no interval overlaps. Used by parse_timer to advance last_ts past
    // the pause.
    [[nodiscard]] uint64_t max_resume_ns_overlapping(uint64_t beg_ns,
                                                     uint64_t end_ns) const noexcept;

    // Access recorded intervals (for testing).
    [[nodiscard]] std::vector<pause_interval> const& intervals() const noexcept
    {
        return intervals_;
    }

private:
    ClockPolicy&                clock_;
    std::atomic<bool>           paused_{ false };
    std::atomic<uint64_t>       pending_pause_ts_{ 0 };
    mutable std::mutex          pause_mutex_;
    std::vector<pause_interval> intervals_;
};

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

namespace pause_detail
{
// Iterate every pause interval that overlaps [beg_ns, end_ns) in half-open
// convention (a sample ending exactly at pause_ns is NOT overlapping; a sample
// starting at resume_ns is similarly not filtered). The on_overlap callable
// receives a const pause_interval& and may return false to stop early.
template <class Intervals, class OnOverlap>
inline void
for_each_overlapping(Intervals const& intervals, uint64_t beg_ns, uint64_t end_ns,
                     OnOverlap&& on_overlap)
{
    auto it = std::lower_bound(
        intervals.begin(), intervals.end(), beg_ns,
        [](pause_interval const& iv, uint64_t t) { return iv.resume_ns < t; });
    for(; it != intervals.end(); ++it)
    {
        if(it->pause_ns >= end_ns) break;
        if(it->resume_ns > beg_ns && it->pause_ns < end_ns)
        {
            if(!on_overlap(*it)) return;
        }
    }
}
}  // namespace pause_detail

template <class ClockPolicy>
bool
pause_interval_registry<ClockPolicy>::spans_pause_interval(uint64_t beg_ns,
                                                           uint64_t end_ns) const noexcept
{
    std::lock_guard<std::mutex> const lock(pause_mutex_);
    bool                              hit = false;
    pause_detail::for_each_overlapping(intervals_, beg_ns, end_ns,
                                       [&hit](pause_interval const&) {
                                           hit = true;
                                           return false;  // stop on first overlap
                                       });
    return hit;
}

template <class ClockPolicy>
uint64_t
pause_interval_registry<ClockPolicy>::max_resume_ns_overlapping(
    uint64_t beg_ns, uint64_t end_ns) const noexcept
{
    std::lock_guard<std::mutex> const lock(pause_mutex_);
    uint64_t                          max_resume = 0;
    pause_detail::for_each_overlapping(intervals_, beg_ns, end_ns,
                                       [&max_resume](pause_interval const& iv) {
                                           if(iv.resume_ns > max_resume)
                                               max_resume = iv.resume_ns;
                                           return true;  // continue scanning
                                       });
    return max_resume;
}

}  // namespace rocprofsys::sampling
