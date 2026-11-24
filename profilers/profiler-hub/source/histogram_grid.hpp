// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader_types.hpp"
#include "reader_catalog.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace profiler_hub
{

struct histogram_grid_t
{
    reader_types::timestamp_ns_t origin{};
    reader_types::timestamp_ns_t bucket_size{};
    size_t                       bucket_count{};
};

// max_buckets is an upper bound. The grid never has more buckets than
// requested, and a short trace has fewer. any is false when the span is empty.
inline histogram_grid_t
make_histogram_grid(reader_types::timestamp_ns_t origin,
                    reader_types::timestamp_ns_t end,
                    bool                         any,
                    size_t                       max_buckets)
{
    if(max_buckets == 0)
    {
        throw std::invalid_argument{ "max_buckets must be non-zero" };
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

inline histogram_grid_t
make_histogram_grid(const reader_catalog_t& catalog, size_t max_buckets)
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
    return make_histogram_grid(origin, end, any, max_buckets);
}

/** @brief Counts the rows overlapping each bucket and sums the overlap.
 *         Row has start_ts and end_ts. */
template <typename Row>
reader_types::histogram_bucket_list_t
accumulate_overlaps(const std::vector<Row>& rows, const histogram_grid_t& grid)
{
    struct bucket_accumulator_t
    {
        size_t event_count{};
        double value{};
    };
    const auto origin       = grid.origin;
    const auto bucket_size  = grid.bucket_size;
    const auto bucket_count = grid.bucket_count;

    std::vector<bucket_accumulator_t> buckets(bucket_count);
    for(const auto& row : rows)
    {
        if(row.end_ts <= row.start_ts || bucket_size == 0) continue;
        const size_t end_mark = row.end_ts - 1;
        if(end_mark < origin) continue;

        size_t start_bucket =
            row.start_ts <= origin ? 0 : (row.start_ts - origin) / bucket_size;
        size_t end_bucket = (end_mark - origin) / bucket_size;
        if(start_bucket >= bucket_count) continue;
        if(end_bucket >= bucket_count) end_bucket = bucket_count - 1;
        if(end_bucket < start_bucket) continue;

        for(size_t bucket = start_bucket; bucket <= end_bucket; ++bucket)
        {
            const size_t bucket_start  = origin + bucket * bucket_size;
            const size_t bucket_end    = bucket_start + bucket_size;
            const size_t overlap_start = std::max<size_t>(row.start_ts, bucket_start);
            const size_t overlap_end   = std::min<size_t>(row.end_ts, bucket_end);
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

/** @brief Averages the samples in each bucket and holds the last value across
 *         empty buckets between samples. Row has timestamp and value. */
template <typename Row>
reader_types::histogram_bucket_list_t
accumulate_samples(const std::vector<Row>& rows, const histogram_grid_t& grid)
{
    struct sample_accumulator_t
    {
        size_t count{};
        double sum{};
    };
    const auto origin       = grid.origin;
    const auto bucket_size  = grid.bucket_size;
    const auto bucket_count = grid.bucket_count;

    std::vector<sample_accumulator_t> buckets(bucket_count);
    for(const auto& row : rows)
    {
        if(row.timestamp < origin || bucket_size == 0) continue;
        const size_t index = (row.timestamp - origin) / bucket_size;
        if(index >= bucket_count) continue;
        buckets[index].count += 1;
        buckets[index].sum += row.value;
    }

    reader_types::histogram_bucket_list_t populated;
    for(size_t index = 0; index < buckets.size(); ++index)
    {
        if(buckets[index].count == 0) continue;
        populated.push_back(reader_types::histogram_bucket_t{
            index,
            buckets[index].count,
            buckets[index].sum / static_cast<double>(buckets[index].count) });
    }

    if(populated.size() < 2) return populated;

    reader_types::histogram_bucket_list_t filled;
    for(size_t index = 0; index < populated.size(); ++index)
    {
        filled.push_back(populated[index]);
        if(index + 1 == populated.size()) break;
        for(size_t gap = populated[index].bucket_index + 1;
            gap < populated[index + 1].bucket_index;
            ++gap)
        {
            filled.push_back(
                reader_types::histogram_bucket_t{ gap, 0, populated[index].value });
        }
    }
    return filled;
}

/** @brief One histogram per catalog track on a supplied grid.
 *         buckets_of(track, grid) returns the buckets for one track. */
template <typename BucketsOf>
reader_types::histogram_t
build_histogram(const reader_catalog_t& catalog,
                const histogram_grid_t& grid,
                BucketsOf&&             buckets_of)
{
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
            track_histogram.buckets = buckets_of(track, grid);
        }
        histogram.tracks.push_back(std::move(track_histogram));
    }
    return histogram;
}

/** @brief One histogram per catalog track. The grid covers this catalog only. */
template <typename BucketsOf>
reader_types::histogram_t
build_histogram(const reader_catalog_t& catalog,
                size_t                  max_buckets,
                BucketsOf&&             buckets_of)
{
    return build_histogram(catalog,
                           make_histogram_grid(catalog, max_buckets),
                           std::forward<BucketsOf>(buckets_of));
}

}  // namespace profiler_hub
