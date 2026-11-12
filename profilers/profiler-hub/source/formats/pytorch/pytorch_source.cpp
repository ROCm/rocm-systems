// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "pytorch_source.hpp"

#include "cached_track_reader.hpp"
#include "debug.hpp"
#include "pytorch_catalog.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

namespace profiler_hub::pytorch
{

namespace
{

// cached_track_reader only asks for whole tracks. A filter this reader would ignore is
// refused, so a later caller cannot get unfiltered rows without knowing it.
void
require_no_filter(const reader_types::event_filter_t& filter)
{
    if(filter.time_window.start || filter.time_window.end || filter.pagination.limit ||
       filter.pagination.offset || filter.sort || !filter.types.empty())
    {
        throw std::invalid_argument{ "pytorch: track reads take no filter" };
    }
}

}  // namespace

class pytorch_row_reader final : public track_row_reader
{
public:
    explicit pytorch_row_reader(const pytorch_source& source)
    : m_source{ source }
    {}

    reader_types::timeline_event_list_t events_for_track(
        const reader_types::track_info_ptr_t& track,
        const reader_types::event_filter_t&   filter) override
    {
        require_no_filter(filter);
        if(track->category == reader_types::track_kind_t::pmc_memory) return {};
        const auto& held = m_source.track_rows(*track);
        std::array<std::size_t, pytorch_source::k_type_slots> next_id{};
        for(std::size_t slot = 0; slot < next_id.size(); ++slot)
        {
            next_id.at(slot) = held.by_type.at(slot).first_id;
        }
        reader_types::timeline_event_list_t events;
        events.reserve(held.rows.size());
        for(const auto& row : held.rows)
        {
            const auto slot = static_cast<std::size_t>(row.type);
            events.push_back(reader_types::timeline_event_t{
                .unique_identifier = { .id = next_id.at(slot)++, .type = row.type },
                .start_timestamp   = row.start,
                .end_timestamp     = row.end,
                .display_name      = m_source.m_names[row.name_id],
                .category          = m_source.m_cats[m_source.m_events[row.event].cat_id],
                .track             = track });
        }
        return events;
    }

    reader_types::counter_timeline_event_list_t counter_events_for_track(
        const reader_types::track_info_ptr_t& track,
        const reader_types::event_filter_t&   filter) override
    {
        require_no_filter(filter);
        if(track->category != reader_types::track_kind_t::pmc_memory) return {};
        const auto* points = m_source.m_catalog->memory_activity_points(track->id);
        if(points == nullptr) return {};
        reader_types::counter_timeline_event_list_t events;
        events.reserve(points->size());
        for(const auto& point : *points)
        {
            events.push_back(
                reader_types::counter_timeline_event_t{ .unique_identifier = {},
                                                        .timestamp         = point.start,
                                                        .value             = point.value,
                                                        .track             = track });
        }
        return events;
    }

    std::optional<std::pair<size_t, size_t>> event_id_span(
        reader_types::event_type_t type) override
    {
        const auto slot = static_cast<std::size_t>(type);
        if(slot >= pytorch_source::k_type_slots) return std::nullopt;
        std::optional<size_t> first;
        size_t                last = 0;
        for(const auto& held : m_source.m_tracks)
        {
            const auto& run = held.by_type.at(slot);
            if(run.rows.empty()) continue;
            if(!first || run.first_id < first.value()) first = run.first_id;
            last = std::max(last, run.first_id + run.rows.size() - 1);
        }
        if(!first) return std::nullopt;
        return std::make_pair(first.value(), last);
    }

    void visit_events_in_id_range(const reader_types::track_info_ptr_t& track,
                                  reader_types::event_type_t            type,
                                  size_t                                id_begin,
                                  size_t                                id_end,
                                  reader_t::event_visitor_t             visitor,
                                  void*                                 context) override
    {
        const auto slot = static_cast<std::size_t>(type);
        if(slot >= pytorch_source::k_type_slots) return;
        const auto&  held = m_source.track_rows(*track);
        const auto&  run  = held.by_type.at(slot);
        const size_t low  = std::max(id_begin, run.first_id);
        const size_t high = std::min(id_end, run.first_id + run.rows.size());
        for(size_t id = low; id < high; ++id)
        {
            const auto& row = held.rows[run.rows[id - run.first_id]];
            visitor(context,
                    { .id = id, .type = type },
                    row.start,
                    row.end,
                    m_source.m_names[row.name_id]);
        }
    }

private:
    const pytorch_source& m_source;
};

std::unique_ptr<track_row_reader>
pytorch_source::row_seam::acquire()
{
    return std::make_unique<pytorch_row_reader>(m_owner);
}

std::unique_ptr<track_row_reader>
pytorch_source::row_seam::try_acquire()
{
    return acquire();
}

pytorch_source::pytorch_source(const source_request_t& request)
: format_source{ "pytorch" }
, m_path{ request.path }
, m_catalog{ std::make_shared<reader_catalog_t>() }
, m_seam{ *this }
{}

const std::shared_ptr<reader_catalog_t>&
pytorch_source::catalog() const
{
    return m_catalog;
}

void
pytorch_source::load_catalog(common::thread_pool& /*workers*/,
                             const catalog_cache_t& cache)
{
    auto                         store  = load_store(m_path, cache.stop_requested);
    [[maybe_unused]] std::size_t events = 0;
    for(const auto& track : store.tracks)
    {
        events += track.rows.size();
    }
    LOG_INFO("[Profiler-Hub] source: events={} tracks={} dropped_without_ts_dur={} "
             "clamped_timestamps={} unbound_flows={} other_phases={} gunzip_ms={} "
             "parse_ms={} derive_ms={}",
             events,
             store.tracks.size(),
             store.diag.bad_events,
             store.diag.clamped_events,
             store.diag.flow_unbound,
             store.diag.other_events,
             store.diag.gunzip_ms,
             store.diag.parse_ms,
             store.derive_ms);
    const std::size_t first_name_id = fill_catalog(*m_catalog, store);

    for(const auto& [id, text] : m_catalog->strings_by_id)
    {
        auto&        table = id < first_name_id ? m_cats : m_names;
        const size_t index = id < first_name_id ? id - 1 : id - first_name_id;
        if(index >= table.size()) table.resize(index + 1);
        table[index] = text;
    }

    // Ids start at 1, as rocpd row ids do.
    std::array<std::size_t, k_type_slots> next_id{};
    next_id.fill(1);
    m_tracks.reserve(store.tracks.size());
    for(auto& track : store.tracks)
    {
        track_rows_t held;
        held.rows = std::move(track.rows);
        for(std::size_t index = 0; index < held.rows.size(); ++index)
        {
            const auto slot = static_cast<std::size_t>(held.rows[index].type);
            auto&      run  = held.by_type.at(slot);
            if(run.rows.empty()) run.first_id = next_id.at(slot);
            run.rows.push_back(static_cast<std::uint32_t>(index));
            ++next_id.at(slot);
        }
        m_tracks.push_back(std::move(held));
    }
    m_events       = std::move(store.events);
    m_args         = std::move(store.args);
    m_memory       = std::move(store.memory);
    m_cpu_agent_id = store.cpu_agent_id;

    for(std::size_t track = 0; track < m_tracks.size(); ++track)
    {
        for(std::size_t slot = 0; slot < k_type_slots; ++slot)
        {
            const auto& run = m_tracks[track].by_type.at(slot);
            if(run.rows.empty()) continue;
            m_runs.push_back(run_t{ static_cast<reader_types::event_type_t>(slot),
                                    run.first_id,
                                    run.rows.size(),
                                    static_cast<std::uint32_t>(track) });
        }
    }
    std::sort(m_runs.begin(), m_runs.end(), [](const run_t& a, const run_t& b) {
        if(a.type != b.type) return a.type < b.type;
        return a.first_id < b.first_id;
    });

    add_memory_tracks();
    m_catalog->publish_memory_activity_pmc_infos();
}

std::uint64_t
pytorch_source::agent_of(const memory_sample_t& sample) const
{
    if(sample.device_type == 0) return m_cpu_agent_id;
    return sample.device_id >= 0 ? static_cast<std::uint64_t>(sample.device_id) + 1 : 0;
}

const memory_sample_t*
pytorch_source::sample_of(std::uint32_t event) const
{
    const auto it =
        std::lower_bound(m_memory.begin(),
                         m_memory.end(),
                         event,
                         [](const memory_sample_t& sample, std::uint32_t wanted) {
                             return sample.event < wanted;
                         });
    return it != m_memory.end() && it->event == event ? &*it : nullptr;
}

// Each agent with memory events gets two series. The totals are read from the file's
// "Total Allocated" and "Total Reserved", never summed from the events: a trace starts
// mid-run, so a sum from zero would be wrong.
void
pytorch_source::add_memory_tracks()
{
    struct series_t
    {
        reader_types::track_info_ptr_t       owner;
        std::vector<memory_activity_point_t> allocated;
        std::vector<memory_activity_point_t> reserved;
    };
    std::map<std::uint64_t, series_t> by_agent;
    const auto                        slot =
        static_cast<std::size_t>(reader_types::event_type_t::memory_allocate);
    for(std::size_t track = 0; track < m_tracks.size(); ++track)
    {
        const auto& held = m_tracks[track];
        const auto& run  = held.by_type.at(slot);
        for(std::size_t index = 0; index < run.rows.size(); ++index)
        {
            const auto& row    = held.rows[run.rows[index]];
            const auto* sample = sample_of(row.event);
            if(sample == nullptr) continue;
            const auto agent = agent_of(*sample);
            if(agent == 0) continue;
            auto& series = by_agent[agent];
            if(!series.owner) series.owner = m_catalog->tracks[track];
            const std::uint64_t id = run.first_id + index;
            if(sample->allocated >= 0)
            {
                series.allocated.push_back(
                    { id, row.start, row.end, static_cast<double>(sample->allocated) });
            }
            if(sample->reserved >= 0)
            {
                series.reserved.push_back(
                    { id, row.start, row.end, static_cast<double>(sample->reserved) });
            }
        }
    }

    for(auto& [agent, series] : by_agent)
    {
        for(auto* points : { &series.allocated, &series.reserved })
        {
            if(points->empty()) continue;
            auto track = std::make_shared<reader_types::track_info_t>();
            track->id  = m_catalog->tracks.size();
            track->name =
                points == &series.allocated ? "ALLOCATED MEMORY" : "RESERVED MEMORY";
            track->category    = reader_types::track_kind_t::pmc_memory;
            track->agent_id    = agent;
            track->event_count = points->size();
            track->start_ts    = std::numeric_limits<std::size_t>::max();
            double low         = points->front().value;
            double high        = low;
            for(const auto& point : *points)
            {
                track->start_ts = std::min<std::size_t>(track->start_ts, point.start);
                track->end_ts   = std::max<std::size_t>(track->end_ts, point.end);
                low             = std::min(low, point.value);
                high            = std::max(high, point.value);
            }
            track->value_range  = reader_types::value_range_t{ low, high };
            track->node_info    = series.owner->node_info;
            track->process_info = series.owner->process_info;
            m_catalog->add_memory_activity_track(std::move(track), std::move(*points));
        }
    }
}

void
pytorch_source::attach_reader(cached_track_reader& reader)
{
    m_reader = &reader;
}

connection_source&
pytorch_source::track_reads()
{
    return m_seam;
}

reader_types::track_info_ptr_t
pytorch_source::track_of(const reader_types::timeline_event_t& event)
{
    const auto type  = event.unique_identifier.type;
    const auto id    = event.unique_identifier.id;
    const auto after = std::upper_bound(
        m_runs.begin(),
        m_runs.end(),
        event.unique_identifier,
        [](const reader_types::unique_timeline_event_id_t& wanted, const run_t& run) {
            if(wanted.type != run.type) return wanted.type < run.type;
            return wanted.id < run.first_id;
        });
    if(after == m_runs.begin()) return nullptr;
    const run_t& run = *std::prev(after);
    if(run.type != type || id >= run.first_id + run.count) return nullptr;
    return m_catalog->tracks[run.track];
}

reader_types::histogram_t
pytorch_source::histogram(const histogram_grid_t& grid)
{
    if(m_reader == nullptr)
    {
        throw std::logic_error{ "pytorch: histogram needs the track reader" };
    }
    struct span_t
    {
        std::uint64_t start_ts;
        std::uint64_t end_ts;
    };
    struct sample_t
    {
        std::uint64_t timestamp;
        double        value;
    };
    return build_histogram(
        *m_catalog,
        grid,
        [this](const reader_types::track_info_ptr_t& track, const histogram_grid_t& on) {
            if(track->category == reader_types::track_kind_t::pmc_memory)
            {
                std::vector<sample_t> samples;
                if(const auto* points = m_catalog->memory_activity_points(track->id))
                {
                    for(const auto& point : *points)
                    {
                        samples.push_back({ point.start, point.value });
                    }
                }
                return accumulate_samples(samples, on);
            }
            const ph_event_list_t events = m_reader->events(track, 0, 0);
            std::vector<span_t>   spans;
            spans.reserve(events.list_size);
            for(std::uint32_t index = 0; index < events.list_size; ++index)
            {
                spans.push_back({ events.events[index].start, events.events[index].end });
            }
            return accumulate_overlaps(spans, on);
        });
}

const pytorch_source::track_rows_t&
pytorch_source::track_rows(const reader_types::track_info_t& track) const
{
    return m_tracks.at(track.id);
}

}  // namespace profiler_hub::pytorch
