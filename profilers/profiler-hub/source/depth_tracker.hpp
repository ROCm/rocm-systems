// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace profiler_hub
{

/**
 * Assigns each event the lowest level that is free when it starts. Events must
 * be added in non-decreasing start order. An event that starts exactly when
 * another ends does not overlap it. Levels count from 1.
 */
class depth_tracker
{
public:
    /**
     * Adds the next event of the track.
     * @return the lowest level free at @p start. 1 means nothing on this track
     *         overlaps the event.
     */
    std::size_t add(std::size_t start, std::size_t end)
    {
        std::size_t level = 0;
        auto        it    = m_active.begin();
        while(it != m_active.end())
        {
            if(start >= it->end)
            {
                it = m_active.erase(it);
                continue;
            }
            if(end >= it->start)
            {
                // A higher level is occupied while this one is free.
                if(level < it->level)
                {
                    break;
                }
                level = it->level + 1;
            }
            ++it;
        }

        m_active.push_back(active_event{ start, end, level });
        std::sort(m_active.begin(),
                  m_active.end(),
                  [](const active_event& lhs, const active_event& rhs) {
                      return lhs.level < rhs.level;
                  });

        const std::size_t depth = level + 1;
        m_depth                 = std::max(m_depth, depth);
        return depth;
    }

    /** The largest level assigned since construction or the last reset(); 0 if none. */
    [[nodiscard]] std::size_t depth() const noexcept { return m_depth; }

    void reset()
    {
        m_active.clear();
        m_depth = 0;
    }

private:
    struct active_event
    {
        std::size_t start;
        std::size_t end;
        std::size_t level;
    };

    std::vector<active_event> m_active;
    std::size_t               m_depth{};
};

}  // namespace profiler_hub
