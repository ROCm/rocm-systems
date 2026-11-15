// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>

namespace rocprofsys::collection
{
/** The timing decisions of a polled source; no thread, no clock. Not thread-safe. */
class tick_grid
{
public:
    using clock      = std::chrono::steady_clock;
    using time_point = clock::time_point;

    enum class step : std::uint8_t
    {
        wait,
        fire,
        idle,
        exit
    };

    struct decision
    {
        step what = step::wait;
        /** Only meaningful for step::wait. */
        time_point until;
    };

    /** Longest interval kept; a longer one is clamped to it so the grid arithmetic
     * cannot overflow. */
    static constexpr std::chrono::nanoseconds k_max_interval =
        std::chrono::nanoseconds::max() / 4;

    /** An interval above k_max_interval (about 73 years) is clamped to it.
     * @throws std::invalid_argument when interval <= 0 */
    tick_grid(time_point start, std::chrono::nanoseconds interval);

    void pause() noexcept;
    /** The next tick is at `now`. No-op when not paused. */
    void resume(time_point now) noexcept;
    /** The callback returned false or threw: exit. */
    void finish() noexcept;

    /** exit, else idle while paused, else fire once `now` reached the next tick, else
     * wait until it. */
    [[nodiscard]] decision next(time_point now) const noexcept;

    /** A tick ran: the next tick is the first grid point after `now`, so missed ticks
     * are skipped without drift. */
    void advance(time_point now) noexcept;

private:
    time_point               m_next;
    std::chrono::nanoseconds m_interval;
    bool                     m_paused   = false;
    bool                     m_finished = false;
};
}  // namespace rocprofsys::collection
