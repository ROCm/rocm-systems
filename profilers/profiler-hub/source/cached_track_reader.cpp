// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cached_track_reader.hpp"

#include "common/natural_merge_sort.hpp"
#include "event_partitions.hpp"
#include "track_window.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
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

ph_event_t
to_ph_event(const reader_types::timeline_event_t& event)
{
    return ph_event_t{ .start = event.start_timestamp,
                       .end   = event.end_timestamp,
                       .name =
                           event.display_name.empty() ? "" : event.display_name.data() };
}

ph_sample_t
to_ph_sample(const reader_types::counter_timeline_event_t& sample)
{
    return ph_sample_t{ .timestamp = sample.timestamp, .value = sample.value };
}

std::vector<ph_event_t>
to_ph_events(const reader_types::timeline_event_list_t& events)
{
    std::vector<ph_event_t> converted;
    converted.reserve(events.size());
    for(const auto& event : events)
    {
        converted.push_back(to_ph_event(event));
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
sort_by_timestamp(std::vector<ph_sample_t>& samples)
{
    common::natural_merge_sort(samples.begin(),
                               samples.end(),
                               [](const ph_sample_t& lhs, const ph_sample_t& rhs) {
                                   return lhs.timestamp < rhs.timestamp;
                               });
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
        auto&                  slot = cache[static_cast<size_t>(track->id)];
        if(!slot) slot = std::make_unique<cache_entry<Item>>();
        entry = slot.get();
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

    auto windowed = with_reader(m_source, [&](track_row_reader& reader) {
        return to_ph_events(
            reader.events_for_track(track, make_window_filter(start_ts, end_ts)));
    });

    const std::scoped_lock lock{ m_windowed_mutex };
    return as_list(m_windowed_events.emplace_back(std::move(windowed)));
}

ph_sample_list_t
cached_track_reader::samples(const reader_types::track_info_ptr_t& track,
                             uint64_t                              start_ts,
                             uint64_t                              end_ts)
{
    if(!track || track->category != reader_types::track_kind_t::pmc_agent)
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

    auto windowed = with_reader(m_source, [&](track_row_reader& reader) {
        return to_ph_samples(
            reader.counter_events_for_track(track, make_window_filter(start_ts, end_ts)));
    });

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
        return read_thread_track_in_parts(reader, track);
    }

    auto sorted = to_ph_events(reader.events_for_track(track, {}));
    sort_by_start(sorted);
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

    const auto visitor = [](void*                        context,
                            reader_types::timestamp_ns_t start,
                            reader_types::timestamp_ns_t end,
                            std::string_view             name) {
        static_cast<std::vector<ph_event_t>*>(context)->push_back(ph_event_t{
            .start = start, .end = end, .name = name.empty() ? "" : name.data() });
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

    struct helper_group
    {
        std::vector<common::thread_pool::task_handle> handles;

        ~helper_group()
        {
            for(const auto& handle : handles)
            {
                std::ignore = handle.cancel();
                handle.wait();
            }
        }
    };

    {
        helper_group helpers;
        const size_t wanted = std::min(m_options.parallel_read_parts, ranges.size());
        for(size_t i = 1; i < wanted; ++i)
        {
            helpers.handles.push_back(m_workers.submit(helper));
        }
        worker(reader);
    }

    if(first_error) std::rethrow_exception(first_error);

    return merge_event_partitions(ranges, outputs);
}

}  // namespace profiler_hub
