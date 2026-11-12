// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

namespace profiler_hub::common
{

template <std::random_access_iterator RandomIt, typename Compare>
    requires std::sortable<RandomIt, Compare>
void
natural_merge_sort(RandomIt first, RandomIt last, Compare comp)
{
    // Bounds are distances from `first`. A Debug MSVC iterator registers
    // itself on the vector it points into. Destroying a vector full of those
    // iterators unlinks them one at a time, and each unlink walks every
    // iterator that is still registered. A track with one run per event makes
    // that walk take time proportional to the square of the event count, so
    // the sort looks stuck. An integer has no registration.
    using difference_type = std::iter_difference_t<RandomIt>;

    std::vector<difference_type> bounds{ 0 };
    for(auto it = first; it != last;)
    {
        it = std::is_sorted_until(it, last, comp);
        bounds.push_back(it - first);
    }

    while(bounds.size() > 2)
    {
        std::vector<difference_type> merged{ bounds.front() };

        size_t i = 0;
        for(; i + 2 < bounds.size(); i += 2)
        {
            std::inplace_merge(first + bounds[i],
                               first + bounds[i + 1],
                               first + bounds[i + 2],
                               comp);
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
