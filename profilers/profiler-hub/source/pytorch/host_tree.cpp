// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "host_tree.hpp"

#include <algorithm>
#include <array>

namespace profiler_hub::pytorch
{
namespace
{
std::int64_t
duration_of(const event_t& e)
{
    return std::max<std::int64_t>(e.dur_ns, 0);
}
}  // namespace

tree_result_t
build_host_tree(std::span<const event_t> events_of_one_thread,
                const loaded_trace_t&    t,
                const tree_options_t&    o)
{
    tree_result_t r{};
    r.depth.assign(events_of_one_thread.size(), k_not_in_tree);

    std::array<bool, 256> in_tree{};
    for(std::size_t i = 0; i < t.cats.size() && i < in_tree.size(); ++i)
    {
        in_tree[i] = t.cat_classes[i].in_host_tree;
    }

    std::vector<std::uint32_t> order;
    for(std::size_t i = 0; i < events_of_one_thread.size(); ++i)
    {
        if(in_tree[events_of_one_thread[i].cat_id])
        {
            order.push_back(static_cast<std::uint32_t>(i));
        }
    }

    std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        const auto& x = events_of_one_thread[a];
        const auto& y = events_of_one_thread[b];
        if(x.ts_ns != y.ts_ns) return x.ts_ns < y.ts_ns;
        return o.equal_start_longer_is_parent && duration_of(x) > duration_of(y);
    });

    std::vector<std::int64_t> open_ends;
    for(const auto idx : order)
    {
        const auto&        e  = events_of_one_thread[idx];
        const std::int64_t t0 = e.ts_ns;
        const std::int64_t t1 = e.ts_ns + duration_of(e);

        while(!open_ends.empty() && t0 >= open_ends.back())
        {
            open_ends.pop_back();
        }

        if(!open_ends.empty())
        {
            const std::int64_t top_end = open_ends.back();
            if(t1 > top_end) ++r.stats.children_ending_after_parent;
            if(t1 - top_end > o.bleed_ns)
            {
                if(top_end - t0 < o.bleed_ns && open_ends.size() >= 2)
                {
                    open_ends.pop_back();
                    ++r.stats.small_overlap_pops;
                }
                else
                {
                    ++r.stats.orphans_kept_as_child;
                }
            }
        }

        const auto level  = static_cast<std::uint16_t>(open_ends.size());
        r.depth[idx]      = level;
        r.stats.max_level = std::max(r.stats.max_level, level);
        ++r.stats.placed;
        open_ends.push_back(t1);
    }
    return r;
}
}  // namespace profiler_hub::pytorch
