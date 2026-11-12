// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "event_partitions.hpp"

#include "common/natural_merge_sort.hpp"

#include <algorithm>

namespace profiler_hub
{

std::vector<id_range>
plan_id_ranges(std::span<const table_id_span> spans, size_t parts)
{
    size_t total_span = 0;
    for(const auto& span : spans)
    {
        if(span.last < span.first) continue;
        total_span += span.last - span.first + 1;
    }

    std::vector<id_range> ranges;
    for(const auto& span : spans)
    {
        if(span.last < span.first) continue;

        const size_t low   = span.first;
        const size_t high  = span.last + 1;
        const size_t share = (high - low) * parts;
        const size_t count = std::max<size_t>(1, (share + total_span / 2) / total_span);
        const size_t step  = (high - low + count - 1) / count;
        for(size_t begin = low; begin < high; begin += step)
        {
            ranges.push_back({ span.type, begin, std::min(begin + step, high) });
        }
    }
    return ranges;
}

std::vector<ph_event_t>
merge_event_partitions(std::span<const id_range>             ranges,
                       std::vector<std::vector<ph_event_t>>& outputs)
{
    size_t total = 0;
    for(const auto& output : outputs)
    {
        total += output.size();
    }

    std::vector<ph_event_t> events;
    events.reserve(total);

    std::vector<size_t> table_ends;
    for(size_t i = 0; i < ranges.size(); ++i)
    {
        events.insert(events.end(), outputs[i].begin(), outputs[i].end());
        std::vector<ph_event_t>().swap(outputs[i]);
        if(i + 1 == ranges.size() || ranges[i + 1].type != ranges[i].type)
        {
            table_ends.push_back(events.size());
        }
    }

    const auto by_start = [](const ph_event_t& lhs, const ph_event_t& rhs) {
        return lhs.start < rhs.start;
    };

    size_t table_begin = 0;
    for(const size_t table_end : table_ends)
    {
        common::natural_merge_sort(
            events.begin() + static_cast<std::ptrdiff_t>(table_begin),
            events.begin() + static_cast<std::ptrdiff_t>(table_end),
            by_start);
        table_begin = table_end;
    }
    for(size_t i = 1; i < table_ends.size(); ++i)
    {
        std::inplace_merge(events.begin(),
                           events.begin() +
                               static_cast<std::ptrdiff_t>(table_ends[i - 1]),
                           events.begin() + static_cast<std::ptrdiff_t>(table_ends[i]),
                           by_start);
    }

    return events;
}

}  // namespace profiler_hub
