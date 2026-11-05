// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "memory_histogram.hpp"

#include "cached_track_reader.hpp"
#include "reader_catalog.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace profiler_hub::pytorch
{

namespace
{

// make_grid and the overlap accumulation are copied from histogram.cpp, which keeps them
// file-local; keep the two in step.

struct histogram_grid_t
{
    reader_types::timestamp_ns_t origin{};
    reader_types::timestamp_ns_t bucket_size{};
    size_t                       bucket_count{};
};

histogram_grid_t
make_grid(const reader_catalog_t& catalog, size_t max_buckets)
{
    if(max_buckets == 0)
    {
        throw std::invalid_argument{ "max_buckets must be non-zero" };
    }

    bool                         any    = false;
    reader_types::timestamp_ns_t origin = 0;
    reader_types::timestamp_ns_t end    = 0;
    for(const auto& track : catalog.tracks)
    {
        if(track->event_count == 0) continue;
        if(!any || track->start_ts < origin) origin = track->start_ts;
        if(!any || track->end_ts > end) end = track->end_ts;
        any = true;
    }
    if(!any || end < origin)
    {
        return {};
    }

    const reader_types::timestamp_ns_t duration = end - origin;
    reader_types::timestamp_ns_t bucket_size    = (duration + max_buckets) / max_buckets;
    if(bucket_size == 0) bucket_size = 1;
    const size_t bucket_count =
        duration == 0 ? 1 : (duration + bucket_size) / bucket_size;
    return { origin, bucket_size, bucket_count };
}

struct bucket_accumulator_t
{
    size_t event_count{};
    double value{};
};

reader_types::histogram_bucket_list_t
accumulate_overlaps(const ph_event_list_t& events, const histogram_grid_t& grid)
{
    std::vector<bucket_accumulator_t> buckets(grid.bucket_count);
    for(uint32_t index = 0; index < events.list_size; ++index)
    {
        const auto& event = events.events[index];
        if(event.end <= event.start || grid.bucket_size == 0) continue;
        const size_t end_mark = event.end - 1;
        if(end_mark < grid.origin) continue;
        size_t start_bucket = event.start <= grid.origin
                                  ? 0
                                  : (event.start - grid.origin) / grid.bucket_size;
        size_t end_bucket   = (end_mark - grid.origin) / grid.bucket_size;
        if(start_bucket >= grid.bucket_count) continue;
        if(end_bucket >= grid.bucket_count) end_bucket = grid.bucket_count - 1;
        if(end_bucket < start_bucket) continue;
        for(size_t bucket = start_bucket; bucket <= end_bucket; ++bucket)
        {
            const size_t bucket_start  = grid.origin + bucket * grid.bucket_size;
            const size_t bucket_end    = bucket_start + grid.bucket_size;
            const size_t overlap_start = std::max<size_t>(event.start, bucket_start);
            const size_t overlap_end   = std::min<size_t>(event.end, bucket_end);
            if(overlap_end <= overlap_start) continue;
            buckets[bucket].event_count += 1;
            buckets[bucket].value += static_cast<double>(overlap_end - overlap_start);
        }
    }

    reader_types::histogram_bucket_list_t result;
    for(size_t index = 0; index < buckets.size(); ++index)
    {
        if(buckets[index].event_count == 0) continue;
        result.push_back(reader_types::histogram_bucket_t{
            index, buckets[index].event_count, buckets[index].value });
    }
    return result;
}

}  // namespace

reader_types::histogram_t
memory_histogram(const reader_catalog_t& catalog,
                 cached_track_reader&    reader,
                 std::size_t             max_buckets)
{
    const histogram_grid_t    grid = make_grid(catalog, max_buckets);
    reader_types::histogram_t histogram;
    histogram.origin_ts    = grid.origin;
    histogram.bucket_size  = grid.bucket_size;
    histogram.bucket_count = grid.bucket_count;
    histogram.tracks.reserve(catalog.tracks.size());
    for(const reader_types::track_info_ptr_t& track : catalog.tracks)
    {
        reader_types::track_histogram_t track_histogram;
        track_histogram.track = track;
        if(grid.bucket_count != 0)
        {
            track_histogram.buckets =
                accumulate_overlaps(reader.events(track, 0, 0), grid);
        }
        histogram.tracks.push_back(std::move(track_histogram));
    }
    return histogram;
}

}  // namespace profiler_hub::pytorch
