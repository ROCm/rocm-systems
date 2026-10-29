// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cached_track_reader.hpp"

#include "common/helper_group.hpp"
#include "common/natural_merge_sort.hpp"
#include "depth_tracker.hpp"
#include "event_partitions.hpp"
#include "track_window.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <tuple>

namespace profiler_hub
{

namespace
{

constexpr std::array event_tables{ reader_types::event_type_t::region,
                                   reader_types::event_type_t::kernel_dispatch,
                                   reader_types::event_type_t::memory_allocate,
                                   reader_types::event_type_t::memory_copy };

std::uint32_t
to_list_size(size_t size)
{
    if(size > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::length_error("result has more elements than the C API can report");
    }
    return static_cast<std::uint32_t>(size);
}

std::optional<std::size_t>
event_type_index(ph_event_type_t type)
{
    switch(type)
    {
        case PH_EVENT_TYPE_REGION: return 0;
        case PH_EVENT_TYPE_KERNEL_DISPATCH: return 1;
        case PH_EVENT_TYPE_MEMORY_COPY: return 2;
        case PH_EVENT_TYPE_MEMORY_ALLOCATE: return 3;
        case PH_EVENT_TYPE_HIP_EVENT: return 4;
    }
    return std::nullopt;
}

ph_event_type_t
to_ph_event_type(reader_types::event_type_t type)
{
    switch(type)
    {
        case reader_types::event_type_t::region: return PH_EVENT_TYPE_REGION;
        case reader_types::event_type_t::kernel_dispatch:
            return PH_EVENT_TYPE_KERNEL_DISPATCH;
        case reader_types::event_type_t::memory_copy: return PH_EVENT_TYPE_MEMORY_COPY;
        case reader_types::event_type_t::memory_allocate:
            return PH_EVENT_TYPE_MEMORY_ALLOCATE;
        case reader_types::event_type_t::hip_event: return PH_EVENT_TYPE_HIP_EVENT;
        case reader_types::event_type_t::sample:
        case reader_types::event_type_t::pmc_event: break;
    }
    throw std::invalid_argument("event type is not a duration event");
}

ph_event_t
to_ph_event(const reader_types::timeline_event_t& event, uint32_t track_id)
{
    return ph_event_t{
        .id            = event.unique_identifier.id,
        .start         = event.start_timestamp,
        .end           = event.end_timestamp,
        .name          = event.display_name.empty() ? "" : event.display_name.data(),
        .type          = to_ph_event_type(event.unique_identifier.type),
        .nesting_depth = 0,
        .track_id      = track_id
    };
}

ph_sample_t
to_ph_sample(const reader_types::counter_timeline_event_t& sample)
{
    return ph_sample_t{ .timestamp = sample.timestamp, .value = sample.value };
}

std::vector<ph_event_t>
to_ph_events(const reader_types::timeline_event_list_t&    events,
             const reader_types::track_info_ptr_t&         track)
{
    const auto track_id = static_cast<uint32_t>(track->id);
    // A host thread query also selects device events on that thread.
    // resolve_track places those on the queue or stream, so they do not
    // belong here. Stream and queue tracks keep every row their own query
    // selected.
    const bool host_thread = track->category == reader_types::track_kind_t::thread;
    std::vector<ph_event_t> converted;
    converted.reserve(events.size());
    for(const auto& event : events)
    {
        if(host_thread && event.track &&
           static_cast<uint32_t>(event.track->id) != track_id)
        {
            continue;
        }
        converted.push_back(to_ph_event(event, track_id));
    }
    return converted;
}

std::vector<ph_sample_t>
to_ph_samples(const reader_types::counter_timeline_event_list_t& samples)
{
    std::vector<ph_sample_t> converted;
    converted.reserve(samples.size());
    for(const auto& sample : samples)
    {
        converted.push_back(to_ph_sample(sample));
    }
    return converted;
}

void
sort_by_start(std::vector<ph_event_t>& events)
{
    common::natural_merge_sort(
        events.begin(), events.end(), [](const ph_event_t& lhs, const ph_event_t& rhs) {
            return lhs.start < rhs.start;
        });
}

void
assign_depths(std::vector<ph_event_t>& events)
{
    depth_tracker tracker;
    for(auto& event : events)
    {
        event.nesting_depth =
            static_cast<std::uint32_t>(tracker.add(event.start, event.end));
    }
}

std::vector<ph_event_t>
overlapping(const std::vector<ph_event_t>& sorted,
            uint64_t                       window_start,
            uint64_t                       window_end)
{
    const auto last = std::upper_bound(
        sorted.begin(),
        sorted.end(),
        window_end,
        [](uint64_t bound, const ph_event_t& event) { return bound < event.start; });

    std::vector<ph_event_t> result;
    std::copy_if(sorted.begin(),
                 last,
                 std::back_inserter(result),
                 [&](const ph_event_t& event) { return event.end >= window_start; });
    return result;
}

void
sort_by_timestamp(std::vector<ph_sample_t>& samples)
{
    common::natural_merge_sort(samples.begin(),
                               samples.end(),
                               [](const ph_sample_t& lhs, const ph_sample_t& rhs) {
                                   return lhs.timestamp < rhs.timestamp;
                               });
}

// The timeline turns each sample into a segment that ends at the next later
// timestamp, and it fetches the track one short slice at a time. A slice that
// contains only the samples inside it cannot draw a step that crosses the
// slice, so keep the sample just before the window and the sample just after
// it. When nothing follows the window, repeat the last value at the track end
// so the step holds for the rest of the track.
std::vector<ph_sample_t>
samples_for_window(const std::vector<ph_sample_t>& sorted,
                   uint64_t                        window_start,
                   uint64_t                        window_end,
                   uint64_t                        track_end)
{
    std::vector<ph_sample_t> selected;
    if(sorted.empty()) return selected;

    const uint64_t upper =
        window_end == 0 ? std::numeric_limits<uint64_t>::max() : window_end;

    const auto first_inside = std::lower_bound(
        sorted.begin(),
        sorted.end(),
        window_start,
        [](const ph_sample_t& sample, uint64_t timestamp) {
            return sample.timestamp < timestamp;
        });
    const auto first_after = std::upper_bound(
        sorted.begin(),
        sorted.end(),
        upper,
        [](uint64_t timestamp, const ph_sample_t& sample) {
            return timestamp < sample.timestamp;
        });

    auto begin = first_inside;
    if(begin != sorted.begin())
    {
        --begin;
    }
    auto end = first_after;
    if(end != sorted.end())
    {
        ++end;
    }

    selected.assign(begin, end);
    if(first_after == sorted.end() && !selected.empty() &&
       track_end > selected.back().timestamp)
    {
        ph_sample_t hold = selected.back();
        hold.timestamp   = track_end;
        selected.push_back(hold);
    }
    return selected;
}

ph_event_list_t
as_list(std::vector<ph_event_t>& events)
{
    return ph_event_list_t{ .list_size = to_list_size(events.size()),
                            .events    = events.data() };
}

ph_sample_list_t
as_list(std::vector<ph_sample_t>& samples)
{
    return ph_sample_list_t{ .list_size = to_list_size(samples.size()),
                             .samples   = samples.data() };
}

}  // namespace

cached_track_reader::cached_track_reader(connection_source&   source,
                                         common::thread_pool& workers,
                                         track_read_options   options)
: m_source{ source }
, m_workers{ workers }
, m_options{ options }
{}

template <typename Item, typename Build>
std::vector<Item>&
cached_track_reader::cached(cache_t<Item>&                        cache,
                            const reader_types::track_info_ptr_t& track,
                            Build&&                               build)
{
    cache_entry<Item>* entry = nullptr;
    {
        const std::scoped_lock lock{ m_cache_mutex };
        entry = &cache.try_emplace(static_cast<size_t>(track->id)).first->second;
    }

    std::call_once(entry->once, [&] { entry->items = std::forward<Build>(build)(); });
    return entry->items;
}

ph_event_list_t
cached_track_reader::events(const reader_types::track_info_ptr_t& track,
                            uint64_t                              start_ts,
                            uint64_t                              end_ts)
{
    if(!track || track->category == reader_types::track_kind_t::pmc_agent)
    {
        return ph_event_list_t{ .list_size = 0, .events = nullptr };
    }

    if(start_ts == 0 && end_ts == 0)
    {
        return as_list(cached(m_events_cache, track, [&] {
            return with_reader(m_source, [&](track_row_reader& reader) {
                return build_sorted_events(reader, track);
            });
        }));
    }

    const auto& whole = cached(m_events_cache, track, [&] {
        return with_reader(m_source, [&](track_row_reader& reader) {
            return build_sorted_events(reader, track);
        });
    });

    const auto window   = make_window_filter(start_ts, end_ts).time_window;
    auto       windowed = overlapping(whole, window.start.value(), window.end.value());

    const std::scoped_lock lock{ m_windowed_mutex };
    return as_list(m_windowed_events.emplace_back(std::move(windowed)));
}

uint32_t
cached_track_reader::nesting_depth(const reader_types::track_info_ptr_t& track)
{
    if(!track || track->category == reader_types::track_kind_t::pmc_agent) return 0;

    const auto& whole = cached(m_events_cache, track, [&] {
        return with_reader(m_source, [&](track_row_reader& reader) {
            return build_sorted_events(reader, track);
        });
    });

    const auto deepest = std::max_element(
        whole.begin(), whole.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.nesting_depth < rhs.nesting_depth;
        });
    return deepest == whole.end() ? 0 : deepest->nesting_depth;
}

std::optional<uint32_t>
cached_track_reader::nesting_depth_of(const reader_types::track_info_ptr_t& track,
                                      uint64_t                              id,
                                      ph_event_type_t                       type)
{
    const auto type_index = event_type_index(type);
    if(!track || !type_index.has_value() ||
       track->category == reader_types::track_kind_t::pmc_agent)
    {
        return std::nullopt;
    }

    const auto             whole = events(track, 0, 0);
    const std::scoped_lock lock{ m_depth_mutex };
    auto&                  index = m_event_depths[static_cast<size_t>(track->id)];
    std::call_once(index.once, [&] {
        if(whole.events == nullptr) return;
        for(uint32_t event_index = 0; event_index < whole.list_size; ++event_index)
        {
            const auto& event      = whole.events[event_index];
            const auto  event_slot = event_type_index(event.type);
            if(!event_slot.has_value()) continue;
            index.by_type[event_slot.value()].emplace(event.id, event.nesting_depth);
        }
    });

    const auto& depths = index.by_type[type_index.value()];
    const auto  found  = depths.find(id);
    if(found == depths.end()) return std::nullopt;
    return found->second;
}

ph_sample_list_t
cached_track_reader::samples(const reader_types::track_info_ptr_t& track,
                             uint64_t                              start_ts,
                             uint64_t                              end_ts)
{
    if(!track ||
       (track->category != reader_types::track_kind_t::pmc_agent &&
        track->category != reader_types::track_kind_t::pmc_memory))
    {
        return ph_sample_list_t{ .list_size = 0, .samples = nullptr };
    }

    if(start_ts == 0 && end_ts == 0)
    {
        return as_list(cached(m_samples_cache, track, [&] {
            return with_reader(m_source, [&](track_row_reader& reader) {
                auto sorted = to_ph_samples(reader.counter_events_for_track(track, {}));
                sort_by_timestamp(sorted);
                return sorted;
            });
        }));
    }

    const auto& whole = cached(m_samples_cache, track, [&] {
        return with_reader(m_source, [&](track_row_reader& reader) {
            auto sorted = to_ph_samples(reader.counter_events_for_track(track, {}));
            sort_by_timestamp(sorted);
            return sorted;
        });
    });

    auto windowed = samples_for_window(
        whole, start_ts, end_ts, static_cast<uint64_t>(track->end_ts));

    const std::scoped_lock lock{ m_windowed_mutex };
    return as_list(m_windowed_samples.emplace_back(std::move(windowed)));
}

std::vector<ph_event_t>
cached_track_reader::build_sorted_events(track_row_reader&                     reader,
                                         const reader_types::track_info_ptr_t& track)
{
    if(track->category == reader_types::track_kind_t::thread &&
       track->event_count >= m_options.parallel_read_min_events)
    {
        auto merged = read_thread_track_in_parts(reader, track);
        assign_depths(merged);
        return merged;
    }

    auto sorted = to_ph_events(reader.events_for_track(track, {}), track);
    sort_by_start(sorted);
    assign_depths(sorted);
    return sorted;
}

std::vector<ph_event_t>
cached_track_reader::read_thread_track_in_parts(
    track_row_reader&                     reader,
    const reader_types::track_info_ptr_t& track)
{
    std::vector<table_id_span> spans;
    for(const auto type : event_tables)
    {
        if(const auto span = reader.event_id_span(type); span.has_value())
        {
            spans.push_back({ type, span->first, span->second });
        }
    }

    const auto ranges = plan_id_ranges(spans, m_options.parallel_read_parts);
    std::vector<std::vector<ph_event_t>> outputs(ranges.size());

    size_t total_ids = 0;
    for(const auto& range : ranges)
    {
        total_ids += range.end - range.begin;
    }
    const auto expected_events = [&](const id_range& range) {
        const size_t share = track->event_count * (range.end - range.begin) / total_ids;
        return share + share / 8 + 16;
    };

    const uint32_t track_id = static_cast<uint32_t>(track->id);
    const auto     visitor  = [](void*                                    context,
                            reader_types::unique_timeline_event_id_t id,
                            reader_types::timestamp_ns_t             start,
                            reader_types::timestamp_ns_t             end,
                            std::string_view                         name) {
        static_cast<std::vector<ph_event_t>*>(context)->push_back(
            ph_event_t{ .id            = id.id,
                             .start         = start,
                             .end           = end,
                             .name          = name.empty() ? "" : name.data(),
                             .type          = to_ph_event_type(id.type),
                             .nesting_depth = 0,
                             .track_id      = 0 });
    };

    std::atomic<size_t> next{ 0 };
    std::atomic<bool>   failed{ false };
    std::mutex          error_mutex;
    std::exception_ptr  first_error;
    const auto          worker = [&](track_row_reader& worker_reader) {
        try
        {
            for(size_t i = next.fetch_add(1); i < ranges.size() && !failed.load();
                i = next.fetch_add(1))
            {
                std::vector<ph_event_t> events;
                events.reserve(expected_events(ranges[i]));
                worker_reader.visit_events_in_id_range(track,
                                                       ranges[i].type,
                                                       ranges[i].begin,
                                                       ranges[i].end,
                                                       visitor,
                                                       &events);
                for(auto& event : events)
                {
                    event.track_id = track_id;
                }
                outputs[i] = std::move(events);
            }
        } catch(...)
        {
            const std::scoped_lock lock{ error_mutex };
            if(!first_error) first_error = std::current_exception();
            failed.store(true);
        }
    };

    const auto helper = [&](const std::stop_token&) {
        if(next.load() >= ranges.size()) return;
        const auto helper_reader = m_source.try_acquire();
        if(helper_reader) worker(*helper_reader);
    };

    {
        common::helper_group helpers{ m_workers };
        const size_t wanted = std::min(m_options.parallel_read_parts, ranges.size());
        for(size_t i = 1; i < wanted; ++i)
        {
            helpers.add(helper);
        }
        worker(reader);
    }

    if(first_error) std::rethrow_exception(first_error);

    return merge_event_partitions(ranges, outputs);
}

}  // namespace profiler_hub
