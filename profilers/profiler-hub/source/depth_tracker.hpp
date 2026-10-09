// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <queue>
#include <vector>

namespace profiler_hub
{

/**
 * Tracks how many events of one track overlap at a time. Events must be added in
 * non-decreasing start order. An event that starts exactly when another ends does not
 * overlap it.
 */
class depth_tracker
{
public:
    /**
     * Adds the next event of the track.
     * @return how many events, this one included, are active at @p start.
     */
    std::size_t add(std::size_t start, std::size_t end)
    {
        while(!m_active_ends.empty() && m_active_ends.top() <= start)
        {
            m_active_ends.pop();
        }
        m_active_ends.push(end);
        m_depth = std::max(m_depth, m_active_ends.size());
        return m_active_ends.size();
    }

    /** The largest overlap seen since construction or the last reset(); 0 if none. */
    [[nodiscard]] std::size_t depth() const noexcept { return m_depth; }

    void reset()
    {
        m_active_ends = {};
        m_depth       = 0;
    }

private:
    using min_heap_t =
        std::priority_queue<std::size_t, std::vector<std::size_t>, std::greater<>>;

    min_heap_t  m_active_ends;
    std::size_t m_depth{};
};

}  // namespace profiler_hub
