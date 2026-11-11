// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/collection/tick_grid.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace rocprofsys::collection
{
tick_grid::tick_grid(time_point start, std::chrono::nanoseconds interval)
: m_next(start)
, m_interval(std::min(interval, k_max_interval))
{
    if(interval <= std::chrono::nanoseconds::zero())
    {
        throw std::invalid_argument("tick_grid: the interval must be positive");
    }
}

void
tick_grid::pause() noexcept
{
    m_paused = true;
}

void
tick_grid::resume(time_point now) noexcept
{
    if(!m_paused)
    {
        return;
    }
    m_paused = false;
    m_next   = now;
}

void
tick_grid::finish() noexcept
{
    m_finished = true;
}

tick_grid::decision
tick_grid::next(time_point now) const noexcept
{
    if(m_finished)
    {
        return { .what = step::exit, .until = {} };
    }
    if(m_paused)
    {
        return { .what = step::idle, .until = {} };
    }
    if(now >= m_next)
    {
        return { .what = step::fire, .until = {} };
    }
    return { .what = step::wait, .until = m_next };
}

void
tick_grid::advance(time_point now) noexcept
{
    if(now < m_next)
    {
        m_next += m_interval;
        return;
    }
    m_next += m_interval * ((now - m_next) / m_interval + 1);
}
}  // namespace rocprofsys::collection
