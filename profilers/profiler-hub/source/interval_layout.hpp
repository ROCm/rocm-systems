// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/reader_types.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace profiler_hub::detail
{

// Computes the geometric packing `lane` for every event (always valid), and the
// containment `parent_id`/`level` for `stack` tracks only, over an interval list. Returns
// the track's peak concurrency (number of lanes = max_lane). Reader-internal; exposed in
// a header only so it can be unit-tested directly with exact coordinates.
//
// DESIGN DECISION (gap #2, 2026-07-20): `lane` is greedy interval packing so overlapping
// bars never collide; `parent_id` is a true containment edge computed ONLY when
// nesting == stack, and carries the opaque event_id_t (task 028), never a raw row id. The
// containment walk searches DOWN the ancestor stack for the nearest true container
// instead of testing only the immediate top — fixing the deeper-ancestor bug where a
// partial-overlap sibling hid a real container (A=[0,100], B=[10,60], C=[50,90]: C is a
// child of A, not top-level). `level` is retained for backward compatibility (Optiq reads
// it for height): containment depth on stack tracks, == lane on lane tracks; height
// consumers should migrate to track_info_t::max_lane. Open for Anthony review — see
// design/draft_api_2026-06-22.md §4-5.
inline uint32_t
compute_interval_layout(reader_types::interval_entry_list_t& events,
                        reader_types::nesting_model_t        nesting)
{
    std::stable_sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
        if(a.start != b.start) return a.start < b.start;
        return a.end > b.end;
    });

    // Greedy lane packing over ALL tracks: each event takes the lowest-numbered lane
    // whose last event ends at or before this event's start; else a new lane. Because
    // events are processed in start order, the lane count equals the track's peak
    // concurrency.
    std::vector<reader_types::timestamp_t> lane_end;  // last end time per lane
    for(auto& ev : events)
    {
        auto assigned = static_cast<uint32_t>(lane_end.size());
        for(uint32_t l = 0; l < lane_end.size(); ++l)
        {
            if(lane_end[l] <= ev.start)
            {
                assigned = l;
                break;
            }
        }
        if(assigned == lane_end.size())
            lane_end.push_back(ev.end);
        else
            lane_end[assigned] = ev.end;
        ev.lane = assigned;
    }
    const auto max_lane = static_cast<uint32_t>(lane_end.size());

    if(nesting != reader_types::nesting_model_t::stack)
    {
        // Concurrency track: overlap is not containment. parent is always no-parent;
        // level mirrors the packing lane so consumers still reading `level` for height
        // see rows.
        for(auto& ev : events)
        {
            ev.parent_id = std::nullopt;
            ev.level     = static_cast<int>(ev.lane);
        }
        return max_lane;
    }

    // Stack track: assign each event its nearest enclosing ancestor. Pop tops that end at
    // or before this event's start (disjoint), then search top->down for the nearest
    // ancestor that truly contains this event (end >= this.end). Skipped partial-overlap
    // siblings stay on the stack — they may still contain a later event.
    std::vector<size_t> stack;
    for(size_t i = 0; i < events.size(); ++i)
    {
        while(!stack.empty() && events[stack.back()].end <= events[i].start)
        {
            stack.pop_back();
        }
        size_t parent = events.size();  // sentinel: no container
        for(size_t s = stack.size(); s-- > 0;)
        {
            if(events[stack[s]].end >= events[i].end)
            {
                parent = stack[s];
                break;
            }
        }
        if(parent != events.size())
        {
            events[i].level     = events[parent].level + 1;
            events[i].parent_id = events[parent].id;
        }
        else
        {
            events[i].level     = 0;
            events[i].parent_id = std::nullopt;
        }
        stack.push_back(i);
    }
    return max_lane;
}

}  // namespace profiler_hub::detail
