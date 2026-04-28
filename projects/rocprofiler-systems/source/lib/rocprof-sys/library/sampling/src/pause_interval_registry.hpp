// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/pause_interval.hpp"

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

}  // namespace rocprofsys::sampling

#include "pause_interval_registry_impl.hpp"
