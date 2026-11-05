// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "memory_source.hpp"

#include "debug.hpp"
#include "pytorch_catalog.hpp"
#include "reader_catalog.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <utility>

namespace profiler_hub::pytorch
{

namespace
{

std::uint64_t
peak_rss_kb()
{
    std::ifstream f{ "/proc/self/status" };
    std::string   line;
    while(std::getline(f, line))
    {
        if(line.rfind("VmHWM:", 0) == 0) return std::stoull(line.substr(6));
    }
    return 0;
}

std::string
summary_line(const std::string& path, const pytorch_store_t& store)
{
    std::string line = fmt::format("[Profiler-Hub] pytorch: file={}", path);
    for(const auto& track : store.tracks)
    {
        line += fmt::format(" track={} events={}", track.id, track.rows.size());
        if(track.category == reader_types::track_kind_t::thread)
        {
            line += fmt::format(" tree_placed={} tree_max_level={}",
                                track.tree.placed,
                                track.tree.max_level);
        }
    }
    line += fmt::format(" parse_ms={} derive_ms={} peak_rss_kb={}",
                        store.parse_ms,
                        store.derive_ms,
                        peak_rss_kb());
    return line;
}

}  // namespace

class memory_row_reader final : public track_row_reader
{
public:
    explicit memory_row_reader(const memory_source& source)
    : m_source{ source }
    {}

    // cached_track_reader only asks for whole tracks, so the filter is always empty.
    reader_types::timeline_event_list_t events_for_track(
        const reader_types::track_info_ptr_t& track,
        const reader_types::event_filter_t&) override
    {
        const auto& held = m_source.track_rows(*track);
        std::array<std::size_t, memory_source::k_type_slots> next_id{};
        for(std::size_t slot = 0; slot < next_id.size(); ++slot)
        {
            next_id[slot] = held.by_type[slot].first_id;
        }
        reader_types::timeline_event_list_t events;
        events.reserve(held.rows.size());
        for(const auto& row : held.rows)
        {
            const auto slot = static_cast<std::size_t>(row.type);
            events.push_back(reader_types::timeline_event_t{
                .unique_identifier = { .id = next_id[slot]++, .type = row.type },
                .start_timestamp   = row.start,
                .end_timestamp     = row.end,
                .display_name      = m_source.m_names[row.name_id],
                .category          = {},
                .track             = track });
        }
        return events;
    }

    reader_types::counter_timeline_event_list_t counter_events_for_track(
        const reader_types::track_info_ptr_t&,
        const reader_types::event_filter_t&) override
    {
        return {};
    }

    std::optional<std::pair<size_t, size_t>> event_id_span(
        reader_types::event_type_t type) override
    {
        if(m_source.m_released) throw std::logic_error{ "PyTorch rows were released" };
        const auto            slot = static_cast<std::size_t>(type);
        std::optional<size_t> first;
        size_t                last = 0;
        for(const auto& held : m_source.m_tracks)
        {
            const auto& run = held.by_type[slot];
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
        const auto&  held = m_source.track_rows(*track);
        const auto&  run  = held.by_type[static_cast<std::size_t>(type)];
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
    const memory_source& m_source;
};

std::unique_ptr<memory_source>
memory_source::open(const std::string& path, reader_catalog_t& catalog)
{
    auto store = load_store(path);
    LOG_INFO("{}", summary_line(path, store));
    fill_catalog(catalog, store);

    std::unique_ptr<memory_source> source{ new memory_source };
    for(const auto& [id, text] : catalog.strings_by_id)
    {
        if(id < k_catalog_name_id_base) continue;
        const size_t name_id = id - k_catalog_name_id_base;
        if(name_id >= source->m_names.size()) source->m_names.resize(name_id + 1);
        source->m_names[name_id] = text.c_str();
    }

    // Ids start at 1, as rocpd row ids do.
    std::array<std::size_t, k_type_slots> next_id;
    next_id.fill(1);
    source->m_tracks.reserve(store.tracks.size());
    for(auto& track : store.tracks)
    {
        track_rows_t held;
        held.rows = std::move(track.rows);
        for(std::size_t index = 0; index < held.rows.size(); ++index)
        {
            const auto slot = static_cast<std::size_t>(held.rows[index].type);
            auto&      run  = held.by_type[slot];
            if(run.rows.empty()) run.first_id = next_id[slot];
            run.rows.push_back(static_cast<std::uint32_t>(index));
            ++next_id[slot];
        }
        source->m_tracks.push_back(std::move(held));
    }
    return source;
}

std::unique_ptr<track_row_reader>
memory_source::acquire()
{
    if(m_released) throw std::logic_error{ "PyTorch rows were released" };
    return std::make_unique<memory_row_reader>(*this);
}

std::unique_ptr<track_row_reader>
memory_source::try_acquire()
{
    return acquire();
}

void
memory_source::release_rows()
{
    m_released = true;
    std::vector<track_rows_t>{}.swap(m_tracks);
    std::vector<const char*>{}.swap(m_names);
}

const memory_source::track_rows_t&
memory_source::track_rows(const reader_types::track_info_t& track) const
{
    if(m_released) throw std::logic_error{ "PyTorch rows were released" };
    return m_tracks.at(track.id - 1);
}

}  // namespace profiler_hub::pytorch
