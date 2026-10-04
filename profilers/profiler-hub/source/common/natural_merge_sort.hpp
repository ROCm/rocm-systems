// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

namespace profiler_hub::common
{

template <typename RandomIt, typename Compare>
void
natural_merge_sort(RandomIt first, RandomIt last, Compare comp)
{
    std::vector<RandomIt> bounds{ first };
    for(auto it = first; it != last;)
    {
        it = std::is_sorted_until(it, last, comp);
        bounds.push_back(it);
    }

    while(bounds.size() > 2)
    {
        std::vector<RandomIt> merged{ bounds.front() };

        size_t i = 0;
        for(; i + 2 < bounds.size(); i += 2)
        {
            std::inplace_merge(bounds[i], bounds[i + 1], bounds[i + 2], comp);
            merged.push_back(bounds[i + 2]);
        }
        if(merged.back() != bounds.back())
        {
            merged.push_back(bounds.back());
        }

        bounds = std::move(merged);
    }
}

}  // namespace profiler_hub::common
