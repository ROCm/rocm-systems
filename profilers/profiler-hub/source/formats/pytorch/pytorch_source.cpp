// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "pytorch_source.hpp"

#include "cached_track_reader.hpp"
#include "debug.hpp"
#include "pytorch_catalog.hpp"

#include <algorithm>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <utility>

namespace profiler_hub::pytorch
{

namespace
{

constexpr const char* k_released = "pytorch: rows were released after loading";

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
                .category          = {},
                .track             = track });
        }
        return events;
    }

    reader_types::counter_timeline_event_list_t counter_events_for_track(
        const reader_types::track_info_ptr_t& /*track*/,
        const reader_types::event_filter_t& filter) override
    {
        require_no_filter(filter);
        return {};
    }

    std::optional<std::pair<size_t, size_t>> event_id_span(
        reader_types::event_type_t type) override
    {
        m_source.require_rows();
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
    m_owner.require_rows();
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
             "clamped_timestamps={} other_phases={} gunzip_ms={} parse_ms={} "
             "derive_ms={}",
             events,
             store.tracks.size(),
             store.diag.bad_events,
             store.diag.clamped_events,
             store.diag.other_events,
             store.diag.gunzip_ms,
             store.diag.parse_ms,
             store.derive_ms);
    const std::size_t first_name_id = fill_catalog(*m_catalog, store);

    for(const auto& [id, text] : m_catalog->strings_by_id)
    {
        if(id < first_name_id) continue;
        const size_t name_id = id - first_name_id;
        if(name_id >= m_names.size()) m_names.resize(name_id + 1);
        m_names[name_id] = text.c_str();
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
}

void
pytorch_source::finish_load()
{
    if(m_released.exchange(true)) return;
    std::vector<track_rows_t>{}.swap(m_tracks);
    std::vector<const char*>{}.swap(m_names);
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
    return build_histogram(
        *m_catalog,
        grid,
        [this](const reader_types::track_info_ptr_t& track, const histogram_grid_t& on) {
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

void
pytorch_source::require_rows() const
{
    if(m_released) throw std::logic_error{ k_released };
}

const pytorch_source::track_rows_t&
pytorch_source::track_rows(const reader_types::track_info_t& track) const
{
    require_rows();
    return m_tracks.at(track.id);
}

}  // namespace profiler_hub::pytorch
