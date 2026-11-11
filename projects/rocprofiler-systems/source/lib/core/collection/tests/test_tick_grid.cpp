// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/collection/tick_grid.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>

namespace rocprofsys::collection
{
namespace
{
using std::chrono::nanoseconds;

constexpr nanoseconds  k_interval{ 100 };
constexpr std::int64_t k_start_ns      = 1000;
constexpr std::int64_t k_jitter_ns     = 30;
constexpr std::int64_t k_tick_count    = 1000;
constexpr std::int64_t k_late_ns       = 1020;
constexpr std::int64_t k_very_late_ns  = 1250;
constexpr std::int64_t k_next_ns       = 1100;
constexpr std::int64_t k_skipped_to_ns = 1300;
constexpr std::int64_t k_resume_ns     = 5555;

[[nodiscard]] tick_grid::time_point
at(std::int64_t offset_ns)
{
    return tick_grid::time_point{ nanoseconds{ offset_ns } };
}

[[nodiscard]] tick_grid
make_grid()
{
    return tick_grid{ at(k_start_ns), k_interval };
}

void
expect_wait_until(const tick_grid& grid, tick_grid::time_point now,
                  tick_grid::time_point until)
{
    const auto decision = grid.next(now);
    EXPECT_EQ(decision.what, tick_grid::step::wait);
    EXPECT_EQ(decision.until, until);
}
}  // namespace

TEST(tick_grid, first_tick_fires_at_start)
{
    const auto grid = make_grid();

    EXPECT_EQ(grid.next(at(k_start_ns)).what, tick_grid::step::fire);
    expect_wait_until(grid, at(k_start_ns - 1), at(k_start_ns));
}

TEST(tick_grid, late_ticks_do_not_drift_over_many_ticks)
{
    auto grid = make_grid();
    for(std::int64_t tick = 0; tick < k_tick_count; ++tick)
    {
        const auto planned = at(k_start_ns + tick * k_interval.count());
        ASSERT_EQ(grid.next(planned + nanoseconds{ k_jitter_ns }).what,
                  tick_grid::step::fire);
        grid.advance(planned + nanoseconds{ k_jitter_ns });
    }

    expect_wait_until(grid, at(k_start_ns),
                      at(k_start_ns + k_tick_count * k_interval.count()));
}

TEST(tick_grid, late_tick_moves_to_the_next_grid_point)
{
    auto grid = make_grid();

    grid.advance(at(k_late_ns));
    expect_wait_until(grid, at(k_late_ns), at(k_next_ns));

    grid.advance(at(k_very_late_ns));
    expect_wait_until(grid, at(k_very_late_ns), at(k_skipped_to_ns));
}

TEST(tick_grid, tick_on_a_grid_point_moves_one_interval)
{
    auto grid = make_grid();

    grid.advance(at(k_next_ns));

    expect_wait_until(grid, at(k_next_ns), at(k_next_ns + k_interval.count()));
}

TEST(tick_grid, paused_grid_is_idle_even_when_a_tick_is_due)
{
    auto grid = make_grid();

    grid.pause();

    EXPECT_EQ(grid.next(at(k_start_ns)).what, tick_grid::step::idle);
    EXPECT_EQ(grid.next(at(k_skipped_to_ns)).what, tick_grid::step::idle);
}

TEST(tick_grid, resume_fires_at_the_resume_time)
{
    auto grid = make_grid();
    grid.pause();

    grid.resume(at(k_resume_ns));

    expect_wait_until(grid, at(k_resume_ns - 1), at(k_resume_ns));
    EXPECT_EQ(grid.next(at(k_resume_ns)).what, tick_grid::step::fire);
}

TEST(tick_grid, resume_without_pause_keeps_the_grid)
{
    auto grid = make_grid();

    grid.resume(at(k_resume_ns));

    expect_wait_until(grid, at(k_start_ns - 1), at(k_start_ns));
}

TEST(tick_grid, finish_exits_also_while_paused)
{
    auto grid = make_grid();
    grid.pause();

    grid.finish();

    EXPECT_EQ(grid.next(at(k_start_ns)).what, tick_grid::step::exit);
    grid.resume(at(k_resume_ns));
    EXPECT_EQ(grid.next(at(k_resume_ns)).what, tick_grid::step::exit);
}

TEST(tick_grid, huge_interval_is_clamped_and_does_not_overflow)
{
    tick_grid grid{ at(k_start_ns), nanoseconds::max() };

    grid.advance(at(k_start_ns));

    expect_wait_until(grid, at(k_start_ns), at(k_start_ns) + tick_grid::k_max_interval);
}

TEST(tick_grid, non_positive_interval_throws)
{
    EXPECT_THROW(tick_grid(at(k_start_ns), nanoseconds::zero()), std::invalid_argument);
    EXPECT_THROW(tick_grid(at(k_start_ns), -k_interval), std::invalid_argument);
}
}  // namespace rocprofsys::collection
