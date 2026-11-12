// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "pytorch_source.hpp"

#include "cached_track_reader.hpp"
#include "debug.hpp"
#include "profiler-hub/c/profiler_hub_types.h"
#include "pytorch_catalog.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>

namespace profiler_hub::pytorch
{

namespace
{

// What an event's args object says about a kernel launch. Missing keys leave a field
// unset.
struct launch_geometry_t
{
    std::optional<std::array<std::uint64_t, 3>> grid;
    std::optional<std::array<std::uint64_t, 3>> block;
    std::optional<std::uint64_t>                shared_memory;
};

// What an event's args object says about a copy.
struct copy_fields_t
{
    std::optional<std::uint64_t> bytes;
    std::optional<std::uint64_t> src;
    std::optional<std::uint64_t> dst;
    std::string                  kind;
};

// What a stream wait event says about the event it waits on and the stream that recorded
// it.
struct hip_fields_t
{
    std::optional<std::uint64_t> event;
    std::optional<std::uint64_t> stream;
};

enum class arg_scope_t : std::uint8_t
{
    host_span,      // operator inputs, Module Hierarchy, Python ids
    runtime_call,   // also the call's dst, src, bytes, kind, ptr and size
    profiler_range  // also the smsp__ counters of a CUDA profiler range
};

using json_t = nlohmann::ordered_json;

json_t
parse_object(std::string_view text)
{
    if(text.empty()) return json_t::object();
    auto parsed = json_t::parse(text.begin(), text.end(), nullptr, false);
    return parsed.is_object() ? parsed : json_t::object();
}

// A number, or a string holding one ("0x7fda0e810200" or "4096").
std::optional<std::uint64_t>
as_unsigned(const json_t& value)
{
    if(value.is_number_unsigned()) return value.get<std::uint64_t>();
    if(value.is_number_integer())
    {
        const auto v = value.get<std::int64_t>();
        return v < 0 ? std::nullopt
                     : std::optional<std::uint64_t>{ static_cast<std::uint64_t>(v) };
    }
    if(value.is_number_float())
    {
        const auto       v           = value.get<double>();
        constexpr double k_two_to_64 = 18446744073709551616.0;
        if(v < 0 || v >= k_two_to_64) return std::nullopt;
        return static_cast<std::uint64_t>(v);
    }
    if(!value.is_string()) return std::nullopt;

    const auto& text = value.get_ref<const std::string&>();
    int         base = 10;
    const auto* from = text.data();
    if(text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X'))
    {
        base = 16;
        from += 2;
    }
    std::uint64_t     number = 0;
    const auto* const to     = text.data() + text.size();
    const auto        result = std::from_chars(from, to, number, base);
    if(result.ec != std::errc{} || result.ptr != to) return std::nullopt;
    return number;
}

std::optional<std::array<std::uint64_t, 3>>
as_triple(const json_t& value)
{
    if(!value.is_array() || value.empty() || value.size() > 3) return std::nullopt;
    std::array<std::uint64_t, 3> out{ 1, 1, 1 };
    for(std::size_t i = 0; i < value.size(); ++i)
    {
        const auto number = as_unsigned(value[i]);
        if(!number) return std::nullopt;
        out.at(i) = *number;
    }
    return out;
}

bool
named_in(std::string_view key, arg_scope_t scope)
{
    constexpr std::array<std::string_view, 7> k_span{
        "Input Dims",       "Input type", "Input Strides",   "Concrete Inputs",
        "Module Hierarchy", "Python id",  "Python parent id"
    };
    constexpr std::array<std::string_view, 6> k_call{ "dst",  "src", "bytes",
                                                      "kind", "ptr", "size" };
    if(std::find(k_span.begin(), k_span.end(), key) != k_span.end()) return true;
    if(scope == arg_scope_t::runtime_call)
    {
        return std::find(k_call.begin(), k_call.end(), key) != k_call.end();
    }
    return scope == arg_scope_t::profiler_range && key.starts_with("smsp__");
}

// @p args_json is the args object as the loader kept it; empty or not an object reads as
// no args.
launch_geometry_t
launch_geometry_of(std::string_view args_json)
{
    const auto        args = parse_object(args_json);
    launch_geometry_t out;
    if(const auto it = args.find("grid"); it != args.end()) out.grid = as_triple(*it);
    if(const auto it = args.find("block"); it != args.end()) out.block = as_triple(*it);
    if(const auto it = args.find("shared memory"); it != args.end())
    {
        out.shared_memory = as_unsigned(*it);
    }
    return out;
}

copy_fields_t
copy_fields_of(std::string_view args_json)
{
    const auto    args = parse_object(args_json);
    copy_fields_t out;
    if(const auto it = args.find("bytes"); it != args.end()) out.bytes = as_unsigned(*it);
    if(!out.bytes)
    {
        if(const auto it = args.find("size"); it != args.end())
        {
            out.bytes = as_unsigned(*it);
        }
    }
    if(const auto it = args.find("src"); it != args.end()) out.src = as_unsigned(*it);
    if(const auto it = args.find("dst"); it != args.end()) out.dst = as_unsigned(*it);
    if(const auto it = args.find("kind"); it != args.end() && it->is_string())
    {
        out.kind = it->get<std::string>();
    }
    return out;
}

hip_fields_t
hip_fields_of(std::string_view args_json)
{
    const auto   args = parse_object(args_json);
    hip_fields_t out;
    if(const auto it = args.find("wait_on_cuda_event_id"); it != args.end())
    {
        out.event = as_unsigned(*it);
    }
    if(const auto it = args.find("wait_on_stream"); it != args.end())
    {
        out.stream = as_unsigned(*it);
    }
    return out;
}

// The named values of a host span, as rocpd's argument table has them: type empty,
// position the order in the file.
std::vector<reader_types::arg_data_t>
named_args_of(std::string_view args_json, arg_scope_t scope)
{
    const auto                            args = parse_object(args_json);
    std::vector<reader_types::arg_data_t> out;
    for(const auto& [key, value] : args.items())
    {
        if(!named_in(key, scope)) continue;
        reader_types::arg_data_t arg;
        arg.position = out.size();
        arg.name     = key;
        arg.value    = value.is_string() ? value.get<std::string>() : value.dump();
        arg.extdata  = "{}";
        out.push_back(std::move(arg));
    }
    return out;
}

// The args object with a "flows" member added for the ends in @p flows, or "{}" when
// there are neither.
std::string
extdata_with_flows(std::string_view              args_json,
                   std::span<const bound_flow_t> flows,
                   std::string_view              flow_args_arena)
{
    std::string text{ args_json.empty() ? std::string_view{ "{}" } : args_json };
    if(flows.empty()) return text;

    std::string list = "\"flows\":[";
    for(std::size_t i = 0; i < flows.size(); ++i)
    {
        const auto& flow = flows[i];
        if(i != 0) list.push_back(',');
        list += "{\"category\":";
        append_json_string(list, flow.category);
        list += ",\"id\":" + std::to_string(flow.id) +
                ",\"end\":" + (flow.phase == 's' ? "\"start\"" : "\"finish\"");
        if(flow.args_len != 0)
        {
            list += ",\"args\":";
            list.append(flow_args_arena.substr(flow.args_off, flow.args_len));
        }
        list.push_back('}');
    }
    list.push_back(']');

    text.pop_back();  // the closing brace
    if(text.size() > 1) text.push_back(',');
    return text + list + "}";
}

// An allocation event allocates when its Bytes are positive and frees when they are
// negative.
std::string_view
alloc_type_of(std::int64_t bytes)
{
    if(bytes > 0) return "ALLOC";
    if(bytes < 0) return "FREE";
    return {};
}

std::uint64_t
alloc_size_of(std::int64_t bytes)
{
    return static_cast<std::uint64_t>(bytes < 0 ? -bytes : bytes);
}

bool
overlaps(const time_window& window, std::uint64_t start, std::uint64_t end)
{
    return start <= window.end && end >= window.start;
}

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
    m_first_name_id = fill_catalog(*m_catalog, store);

    for(const auto& [id, text] : m_catalog->strings_by_id)
    {
        auto&        table = id < m_first_name_id ? m_cats : m_names;
        const size_t index = id < m_first_name_id ? id - 1 : id - m_first_name_id;
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
    m_flows        = std::move(store.flows);
    m_cat_classes  = std::move(store.cat_classes);
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

    m_streams.resize(m_catalog->tracks.size());
    std::size_t next_stream = 0;
    for(std::size_t track = 0; track < m_catalog->tracks.size(); ++track)
    {
        if(m_catalog->tracks[track]->category == reader_types::track_kind_t::stream)
        {
            m_streams[track] = m_catalog->streams[next_stream++];
        }
    }

    for(const auto& held : m_tracks)
    {
        for(std::size_t slot = 0; slot < k_type_slots; ++slot)
        {
            const auto& run = held.by_type.at(slot);
            for(std::size_t index = 0; index < run.rows.size(); ++index)
            {
                const auto link = m_events[held.rows[run.rows[index]].event].corr;
                if(link > 0)
                {
                    m_correlations.push_back(
                        { link, run.first_id + index, static_cast<std::uint8_t>(slot) });
                }
            }
        }
    }
    std::sort(m_correlations.begin(),
              m_correlations.end(),
              [](const correlation_t& a, const correlation_t& b) {
                  if(a.link != b.link) return a.link < b.link;
                  if(a.type != b.type) return a.type < b.type;
                  return a.id < b.id;
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
    const auto at = locate(event.unique_identifier);
    return at ? m_catalog->tracks[at->track] : nullptr;
}

std::optional<pytorch_source::located_t>
pytorch_source::locate(const reader_types::unique_timeline_event_id_t& id) const
{
    const auto slot = static_cast<std::size_t>(id.type);
    if(slot >= k_type_slots) return std::nullopt;
    const auto after = std::upper_bound(
        m_runs.begin(),
        m_runs.end(),
        id,
        [](const reader_types::unique_timeline_event_id_t& wanted, const run_t& run) {
            if(wanted.type != run.type) return wanted.type < run.type;
            return wanted.id < run.first_id;
        });
    if(after == m_runs.begin()) return std::nullopt;
    const run_t& run = *std::prev(after);
    if(run.type != id.type || id.id >= run.first_id + run.count) return std::nullopt;
    const auto& held = m_tracks[run.track];
    const auto  row  = held.by_type.at(slot).rows[id.id - run.first_id];
    return located_t{ run.track, &held.rows[row], id.id };
}

std::string_view
pytorch_source::args_of(const event_t& event) const
{
    return event.args_len == 0
               ? std::string_view{}
               : std::string_view{ m_args }.substr(event.args_off, event.args_len);
}

std::string
pytorch_source::extdata_of(const store_row_t& row) const
{
    const auto first = std::lower_bound(
        m_flows.begin(),
        m_flows.end(),
        row.event,
        [](const bound_flow_t& flow, std::uint32_t event) { return flow.event < event; });
    const auto last = std::upper_bound(
        first,
        m_flows.end(),
        row.event,
        [](std::uint32_t event, const bound_flow_t& flow) { return event < flow.event; });
    return extdata_with_flows(args_of(m_events[row.event]),
                              std::span<const bound_flow_t>{ first, last },
                              m_args);
}

reader_types::event_data_ptr_t
pytorch_source::event_data_of(const store_row_t& row) const
{
    const auto& event = m_events[row.event];
    auto        data  = std::make_shared<reader_types::event_data_t>();
    // Kineto's correlation plays the part of the rocpd stack id, which is the link
    // correlated_events joins on; the parent is its own copy, as for an event with no
    // ancestor. External id is the correlation id.
    data->stack_id = static_cast<std::size_t>(std::max<std::int64_t>(event.corr, 0));
    data->parent_stack_id = data->stack_id;
    data->correlation_id = static_cast<std::size_t>(std::max<std::int64_t>(event.ext, 0));
    data->event_category = std::string{ m_cats[event.cat_id] };
    data->extdata        = "{}";
    return data;
}

reader_types::timeline_event_t
pytorch_source::timeline_event_of(const located_t&           at,
                                  reader_types::event_type_t type) const
{
    const auto& row = *at.row;
    return reader_types::timeline_event_t{ .unique_identifier = { .id   = at.id,
                                                                  .type = type },
                                           .start_timestamp   = row.start,
                                           .end_timestamp     = row.end,
                                           .display_name      = m_names[row.name_id],
                                           .category = m_cats[m_events[row.event].cat_id],
                                           .track    = m_catalog->tracks[at.track] };
}

reader_types::agent_info_ptr_t
pytorch_source::agent_info(std::uint64_t agent) const
{
    const auto found = m_catalog->agents_by_id.find(agent);
    return found == m_catalog->agents_by_id.end() ? nullptr : found->second;
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

namespace
{

using reader_types::event_type_t;

// The event types an event of @p source reaches through a shared stack id, as the rocpd
// v3 reader joins them.
bool
reaches(event_type_t source, event_type_t target)
{
    if(target == event_type_t::region) return true;
    switch(source)
    {
        case event_type_t::region:
            return target == event_type_t::kernel_dispatch ||
                   target == event_type_t::memory_copy ||
                   target == event_type_t::memory_allocate;
        case event_type_t::kernel_dispatch:
            return target == event_type_t::kernel_dispatch;
        case event_type_t::memory_copy: return target == event_type_t::memory_copy;
        default: return false;
    }
}

}  // namespace

reader_types::timeline_event_list_t
pytorch_source::correlated_events(const reader_types::timeline_event_t& event)
{
    const auto at = locate(event.unique_identifier);
    if(!at) return {};
    const auto link = m_events[at->row->event].corr;
    if(link <= 0) return {};

    const auto first =
        std::lower_bound(m_correlations.begin(),
                         m_correlations.end(),
                         link,
                         [](const correlation_t& entry, std::int64_t wanted) {
                             return entry.link < wanted;
                         });
    reader_types::timeline_event_list_t found;
    for(auto it = first; it != m_correlations.end() && it->link == link; ++it)
    {
        const auto type = static_cast<event_type_t>(it->type);
        if(type == event.unique_identifier.type && it->id == event.unique_identifier.id)
        {
            continue;
        }
        if(!reaches(event.unique_identifier.type, type)) continue;
        const auto other = locate({ it->id, type });
        if(other) found.push_back(timeline_event_of(*other, type));
    }
    return found;
}

// The rocpd v3 nesting rule: the regions of the event's track are walked in start order,
// each one is nested under the open regions it falls inside, and the stack is the event
// and its ancestors, innermost first.
reader_types::call_stack_t
pytorch_source::call_stack(const reader_types::timeline_event_t& event)
{
    if(event.unique_identifier.type != event_type_t::region) return {};
    const auto at = locate(event.unique_identifier);
    if(!at) return {};

    struct active_t
    {
        const store_row_t* row;
        std::size_t        id;
        std::size_t        level;
    };
    const auto& held = m_tracks[at->track];
    const auto& run  = held.by_type.at(static_cast<std::size_t>(event_type_t::region));
    std::vector<active_t> active;
    for(std::size_t index = 0; index < run.rows.size(); ++index)
    {
        const store_row_t& region = held.rows[run.rows[index]];
        const std::size_t  id     = run.first_id + index;
        active.erase(std::remove_if(active.begin(),
                                    active.end(),
                                    [&](const active_t& open) {
                                        return region.start >= open.row->end;
                                    }),
                     active.end());

        std::size_t level = 0;
        for(const auto& ancestor : active)
        {
            if(region.end < ancestor.row->start) continue;
            if(level < ancestor.level) break;
            level = ancestor.level + 1;
        }

        if(id == event.unique_identifier.id)
        {
            reader_types::call_stack_t stack;
            const auto add_frame = [&](const store_row_t& row, std::size_t frame_id) {
                reader_types::stack_frame_t          frame;
                reader_types::program_counter_info_t counter;
                counter.event_id      = frame_id;
                counter.function      = std::string{ m_names[row.name_id] };
                frame.program_counter = std::move(counter);
                stack.push_back(std::move(frame));
            };
            add_frame(region, id);
            for(const auto& ancestor : active | std::views::reverse)
            {
                if(ancestor.level >= level) continue;
                add_frame(*ancestor.row, ancestor.id);
            }
            return stack;
        }

        active.push_back({ &region, id, level });
        std::stable_sort(
            active.begin(), active.end(), [](const active_t& a, const active_t& b) {
                return a.level < b.level;
            });
    }
    return {};
}

std::optional<reader_types::region_data_t>
pytorch_source::region_details(const reader_types::timeline_event_t& event)
{
    const auto at = locate(event.unique_identifier);
    if(!at || at->row->type != event_type_t::region) return std::nullopt;
    const auto& row    = *at->row;
    const auto& stored = m_events[row.event];

    reader_types::region_data_t data;
    data.event           = event_data_of(row);
    data.start_timestamp = row.start;
    data.end_timestamp   = row.end;
    data.name            = std::string{ m_names[row.name_id] };
    data.extdata         = extdata_of(row);
    auto scope           = arg_scope_t::host_span;
    if(m_cats[stored.cat_id] == "cuda_profiler_range")
    {
        scope = arg_scope_t::profiler_range;
    }
    else if(m_cat_classes[stored.cat_id].launch_site)
    {
        scope = arg_scope_t::runtime_call;
    }
    data.args = named_args_of(args_of(stored), scope);
    return data;
}

namespace
{

// A launch call that already records the grid in work-items, not in blocks.
bool
records_work_items(std::string_view call)
{
    return call == "hipExtModuleLaunchKernel" || call == "hipHccModuleLaunchKernel";
}

}  // namespace

pytorch_source::kernel_shape_t
pytorch_source::kernel_shape_of(const event_t& stored) const
{
    // The grid is in blocks, so the dispatch's work-item grid is its product with the
    // block size, unless the launch call is one that already records work-items. A kernel
    // whose launch call is not in the file has no such call to go by.
    bool totals = false;
    if(stored.corr > 0)
    {
        const auto first =
            std::lower_bound(m_correlations.begin(),
                             m_correlations.end(),
                             stored.corr,
                             [](const correlation_t& entry, std::int64_t wanted) {
                                 return entry.link < wanted;
                             });
        for(auto it = first; it != m_correlations.end() && it->link == stored.corr; ++it)
        {
            if(it->type != static_cast<std::uint8_t>(event_type_t::region)) continue;
            const auto call = locate({ it->id, event_type_t::region });
            if(!call) continue;
            const auto& launch = m_events[call->row->event];
            if(!m_cat_classes[launch.cat_id].launch_site) continue;
            totals = records_work_items(m_names[call->row->name_id]);
            break;
        }
    }

    const auto     geometry = launch_geometry_of(args_of(stored));
    kernel_shape_t shape;
    if(geometry.block) shape.workgroup = *geometry.block;
    if(geometry.grid)
    {
        const bool multiply = geometry.block.has_value() && !totals;
        for(std::size_t axis = 0; axis < 3; ++axis)
        {
            shape.grid.at(axis) =
                geometry.grid->at(axis) * (multiply ? geometry.block->at(axis) : 1);
        }
    }
    shape.group_segment = geometry.shared_memory;
    return shape;
}

std::optional<reader_types::kernel_dispatch_data_t>
pytorch_source::kernel_dispatch_details(const reader_types::timeline_event_t& event)
{
    const auto at = locate(event.unique_identifier);
    if(!at || at->row->type != event_type_t::kernel_dispatch) return std::nullopt;
    const auto& row    = *at->row;
    const auto& stored = m_events[row.event];
    const auto& track  = *m_catalog->tracks[at->track];

    const auto                           shape = kernel_shape_of(stored);
    reader_types::kernel_dispatch_data_t data;
    data.dispatch_id        = at->id;
    data.start_timestamp    = row.start;
    data.end_timestamp      = row.end;
    data.workgroup_size_x   = shape.workgroup[0];
    data.workgroup_size_y   = shape.workgroup[1];
    data.workgroup_size_z   = shape.workgroup[2];
    data.grid_size_x        = shape.grid[0];
    data.grid_size_y        = shape.grid[1];
    data.grid_size_z        = shape.grid[2];
    data.group_segment_size = shape.group_segment;
    data.name               = std::string{ m_names[row.name_id] };
    data.extdata            = extdata_of(row);
    data.event              = event_data_of(row);
    data.node_info          = track.node_info;
    data.process_info       = track.process_info;
    data.thread_info        = track.thread_info;
    data.agent_info         = agent_info(track.agent_id);
    data.stream_info        = m_streams[at->track];
    return data;
}

std::optional<reader_types::memory_copy_data_t>
pytorch_source::memory_copy_details(const reader_types::timeline_event_t& event)
{
    const auto at = locate(event.unique_identifier);
    if(!at || at->row->type != event_type_t::memory_copy) return std::nullopt;
    const auto& row   = *at->row;
    const auto& track = *m_catalog->tracks[at->track];
    const auto  copy  = copy_fields_of(args_of(m_events[row.event]));

    reader_types::memory_copy_data_t data;
    data.start_timestamp = row.start;
    data.end_timestamp   = row.end;
    data.dst_address     = copy.dst;
    data.src_address     = copy.src;
    data.size            = copy.bytes.value_or(0);
    data.name            = std::string{ m_names[row.name_id] };
    data.extdata         = extdata_of(row);
    data.event           = event_data_of(row);
    // The event's device is one end of the copy. The host has no agent here, so the other
    // end stays unset.
    if(copy.kind == "HtoD" || copy.kind == "DtoD")
    {
        data.dst_agent_id = agent_info(track.agent_id);
    }
    else if(copy.kind == "DtoH")
    {
        data.src_agent_id = agent_info(track.agent_id);
    }
    data.node_info    = track.node_info;
    data.process_info = track.process_info;
    data.thread_info  = track.thread_info;
    data.stream_info  = m_streams[at->track];
    return data;
}

std::optional<reader_types::memory_alloc_data_t>
pytorch_source::memory_alloc_details(const reader_types::timeline_event_t& event)
{
    const auto at = locate(event.unique_identifier);
    if(!at || at->row->type != event_type_t::memory_allocate) return std::nullopt;
    const auto& row    = *at->row;
    const auto& track  = *m_catalog->tracks[at->track];
    const auto* sample = sample_of(row.event);

    reader_types::memory_alloc_data_t data;
    data.start_timestamp = row.start;
    data.end_timestamp   = row.end;
    data.extdata         = extdata_of(row);
    data.event           = event_data_of(row);
    data.node_info       = track.node_info;
    data.process_info    = track.process_info;
    data.thread_info     = track.thread_info;
    if(sample != nullptr)
    {
        data.type       = std::string{ alloc_type_of(sample->bytes) };
        data.address    = static_cast<std::size_t>(sample->address);
        data.size       = static_cast<std::size_t>(alloc_size_of(sample->bytes));
        data.agent_info = agent_info(agent_of(*sample));
    }
    return data;
}

namespace
{

// The reader type a table query's event type reads, and false for the sample types, which
// a Kineto file has none of.
std::optional<event_type_t>
table_type_of(uint32_t event_type)
{
    switch(event_type)
    {
        case PH_EVENT_TYPE_REGION:
        case PH_EVENT_TYPE_REGION_SAMPLE: return event_type_t::region;
        case PH_EVENT_TYPE_KERNEL_DISPATCH: return event_type_t::kernel_dispatch;
        case PH_EVENT_TYPE_MEMORY_COPY: return event_type_t::memory_copy;
        case PH_EVENT_TYPE_MEMORY_ALLOCATE: return event_type_t::memory_allocate;
        case PH_EVENT_TYPE_HIP_EVENT: return event_type_t::hip_event;
        default: return std::nullopt;
    }
}

bool
asks_for(const resolved_event_query& query, uint32_t event_type)
{
    return query.event_types.empty() ||
           std::find(query.event_types.begin(), query.event_types.end(), event_type) !=
               query.event_types.end();
}

// The memory tracks' points, in id order across the tracks the query names.
uint32_t
read_memory_points(const reader_catalog_t&     catalog,
                   const resolved_event_query& query,
                   uint64_t                    after_id,
                   uint64_t                    last_id,
                   ph_table_event*             rows,
                   uint32_t                    capacity)
{
    struct cursor_t
    {
        const std::vector<memory_activity_point_t>* points;
        std::size_t                                 index;
    };
    std::vector<cursor_t> cursors;
    for(const auto track_id : query.tracks)
    {
        const auto* const points = catalog.memory_activity_points(track_id);
        if(points == nullptr) continue;
        if(query.agent_id && catalog.tracks[track_id]->agent_id != *query.agent_id)
        {
            continue;
        }
        const auto begin =
            std::lower_bound(points->begin(),
                             points->end(),
                             after_id,
                             [](const memory_activity_point_t& point, std::uint64_t id) {
                                 return point.id <= id;
                             });
        cursors.push_back({ points, static_cast<std::size_t>(begin - points->begin()) });
    }
    uint32_t filled = 0;
    while(filled < capacity)
    {
        const memory_activity_point_t* best        = nullptr;
        cursor_t*                      best_cursor = nullptr;
        for(auto& cursor : cursors)
        {
            while(cursor.index < cursor.points->size())
            {
                const auto& point = (*cursor.points)[cursor.index];
                if(point.id > last_id)
                {
                    cursor.index = cursor.points->size();
                    break;
                }
                if(query.time && !overlaps(*query.time, point.start, point.end))
                {
                    ++cursor.index;
                    continue;
                }
                if(best == nullptr || point.id < best->id)
                {
                    best        = &point;
                    best_cursor = &cursor;
                }
                break;
            }
        }
        if(best == nullptr) break;
        rows[filled]       = ph_table_event{};
        rows[filled].id    = best->id;
        rows[filled].start = best->start;
        rows[filled].end   = best->end;
        rows[filled].value = best->value;
        ++filled;
        ++best_cursor->index;
    }
    return filled;
}

}  // namespace

std::uint64_t
pytorch_source::copy_agent_of(const located_t& at) const
{
    const auto kind = copy_fields_of(args_of(m_events[at.row->event])).kind;
    return kind == "DtoH" ? 0 : m_catalog->tracks[at.track]->agent_id;
}

// The filter has the meaning the rocpd v3 table queries give it. pid and tid are the
// process and thread row ids; a region has no agent, so an agent filter excludes it; the
// name of a memory allocation is not searched.
bool
pytorch_source::table_match(const resolved_event_query& query,
                            event_type_t                type,
                            const located_t&            at) const
{
    const auto& row   = *at.row;
    const auto& track = *m_catalog->tracks[at.track];
    if(query.time && !overlaps(*query.time, row.start, row.end))
    {
        return false;
    }
    if(query.node_id && track.node_info && *query.node_id != track.node_info->node_id)
    {
        return false;
    }
    if(query.process_id &&
       (!track.process_info || *query.process_id != track.process_info->id))
    {
        return false;
    }
    if(query.thread_id &&
       (!track.thread_info || *query.thread_id != track.thread_info->id))
    {
        return false;
    }
    if(!query.tracks.empty() &&
       std::find(query.tracks.begin(), query.tracks.end(), at.track) ==
           query.tracks.end())
    {
        return false;
    }
    if(query.agent_id)
    {
        std::uint64_t agent = 0;
        switch(type)
        {
            case event_type_t::kernel_dispatch:
            case event_type_t::hip_event: agent = track.agent_id; break;
            case event_type_t::memory_copy: agent = copy_agent_of(at); break;
            case event_type_t::memory_allocate:
                if(const auto* sample = sample_of(row.event)) agent = agent_of(*sample);
                break;
            default: return false;
        }
        if(agent != *query.agent_id) return false;
    }
    if(query.has_string_filter)
    {
        const auto holds = [&query](std::size_t id) {
            return std::find(query.string_ids.begin(), query.string_ids.end(), id) !=
                   query.string_ids.end();
        };
        const bool by_name =
            type != event_type_t::memory_allocate && holds(m_first_name_id + row.name_id);
        const auto cat_id      = m_events[row.event].cat_id;
        const bool by_category = query.include_category && !m_cats[cat_id].empty() &&
                                 holds(static_cast<std::size_t>(cat_id) + 1);
        if(!by_name && !by_category) return false;
    }
    return true;
}

void
pytorch_source::publish_row(event_type_t     type,
                            const located_t& at,
                            ph_table_event&  out) const
{
    const auto& row    = *at.row;
    const auto& stored = m_events[row.event];
    const auto& track  = *m_catalog->tracks[at.track];
    out                = ph_table_event{};
    out.id             = at.id;
    out.start          = row.start;
    out.end            = row.end;
    if(type != event_type_t::memory_allocate) out.name_id = m_first_name_id + row.name_id;
    if(!m_cats[stored.cat_id].empty()) out.category_id = stored.cat_id + 1;
    if(track.node_info) out.nid = track.node_info->node_id;
    if(track.process_info)
    {
        out.process_id = track.process_info->id;
        out.pid        = track.process_info->pid;
    }
    if(track.thread_info)
    {
        out.thread_id = track.thread_info->id;
        out.tid       = track.thread_info->thread_id;
    }
    if(track.category == reader_types::track_kind_t::stream)
    {
        out.stream_id = track.stream_id;
    }

    switch(type)
    {
        case event_type_t::kernel_dispatch:
        {
            const auto shape       = kernel_shape_of(stored);
            out.agent_id           = track.agent_id;
            out.grid_size_x        = shape.grid[0];
            out.grid_size_y        = shape.grid[1];
            out.grid_size_z        = shape.grid[2];
            out.workgroup_size_x   = shape.workgroup[0];
            out.workgroup_size_y   = shape.workgroup[1];
            out.workgroup_size_z   = shape.workgroup[2];
            out.group_segment_size = shape.group_segment.value_or(0);
            break;
        }
        case event_type_t::memory_copy:
        {
            const auto copy  = copy_fields_of(args_of(stored));
            out.agent_id     = copy.kind == "DtoH" ? 0 : track.agent_id;
            out.src_agent_id = copy.kind == "DtoH" ? track.agent_id : 0;
            out.size         = copy.bytes.value_or(0);
            out.address      = copy.dst.value_or(0);
            out.src_address  = copy.src.value_or(0);
            break;
        }
        case event_type_t::memory_allocate:
            if(const auto* sample = sample_of(row.event))
            {
                const std::string_view kind = alloc_type_of(sample->bytes);
                std::copy(kind.begin(), kind.end(), out.mem_type);
                out.agent_id = agent_of(*sample);
                out.size     = alloc_size_of(sample->bytes);
                out.address  = sample->address;
            }
            break;
        case event_type_t::hip_event:
        {
            const auto hip = hip_fields_of(args_of(stored));
            out.agent_id   = track.agent_id;
            out.size       = hip.event.value_or(0);
            out.address    = hip.stream.value_or(0);
            break;
        }
        default: break;
    }
}

uint32_t
pytorch_source::read_table_events(const resolved_event_query& query,
                                  uint32_t                    event_type,
                                  uint64_t                    after_id,
                                  uint64_t                    last_id,
                                  ph_table_event*             rows,
                                  uint32_t                    capacity)
{
    if(rows == nullptr || capacity == 0) return 0;
    if(event_type == PH_EVENT_TYPE_MEMORY_ACTIVITY)
    {
        return read_memory_points(*m_catalog, query, after_id, last_id, rows, capacity);
    }

    const auto type = table_type_of(event_type);
    if(!type || event_type == PH_EVENT_TYPE_REGION_SAMPLE || !asks_for(query, event_type))
    {
        return 0;
    }
    if(query.agent_id && *type == event_type_t::region) return 0;
    if(query.has_string_filter && query.string_ids.empty()) return 0;

    uint32_t filled = 0;
    for(const auto& run : m_runs)
    {
        if(run.type != *type) continue;
        const std::uint64_t run_last = run.first_id + run.count - 1;
        if(run_last <= after_id) continue;
        const auto& held = m_tracks[run.track];
        const auto& by   = held.by_type.at(static_cast<std::size_t>(*type));
        for(std::uint64_t id = std::max<std::uint64_t>(run.first_id, after_id + 1);
            id <= std::min<std::uint64_t>(run_last, last_id);
            ++id)
        {
            const located_t at{ run.track, &held.rows[by.rows[id - run.first_id]], id };
            if(!table_match(query, *type, at)) continue;
            publish_row(*type, at, rows[filled]);
            if(++filled == capacity) return filled;
        }
    }
    return filled;
}

bool
pytorch_source::table_event_id_span(uint32_t  event_type,
                                    uint64_t& first_id,
                                    uint64_t& last_id)
{
    first_id = 0;
    last_id  = 0;
    if(event_type == PH_EVENT_TYPE_MEMORY_ACTIVITY)
    {
        return m_catalog->memory_activity_id_span(first_id, last_id);
    }
    const auto type = table_type_of(event_type);
    if(!type) return false;

    bool found = false;
    for(const auto& run : m_runs)
    {
        if(run.type != *type) continue;
        if(!found) first_id = run.first_id;
        last_id = std::max<std::uint64_t>(last_id, run.first_id + run.count - 1);
        found   = true;
    }
    return found;
}

bool
pytorch_source::summarize_name_durations(const resolved_event_query&           query,
                                         uint32_t                              event_type,
                                         std::vector<name_duration_summary_t>& rows)
{
    rows.clear();
    const auto type = table_type_of(event_type);
    if(!type || event_type == PH_EVENT_TYPE_REGION_SAMPLE || !asks_for(query, event_type))
    {
        return true;
    }
    if(query.agent_id && *type == event_type_t::region) return true;
    if(query.has_string_filter && query.string_ids.empty()) return true;

    std::map<std::string, name_duration_summary_t> groups;
    for(const auto& run : m_runs)
    {
        if(run.type != *type) continue;
        const auto& held = m_tracks[run.track];
        const auto& by   = held.by_type.at(static_cast<std::size_t>(*type));
        for(std::size_t index = 0; index < run.count; ++index)
        {
            const located_t at{ run.track,
                                &held.rows[by.rows[index]],
                                run.first_id + index };
            if(!table_match(query, *type, at)) continue;

            std::string name;
            if(*type == event_type_t::memory_allocate)
            {
                const auto* sample = sample_of(at.row->event);
                name = sample != nullptr ? std::string{ alloc_type_of(sample->bytes) }
                                         : std::string{};
            }
            else
            {
                name = std::string{ m_names[at.row->name_id] };
            }
            const std::uint64_t duration = at.row->end - at.row->start;
            auto&               group    = groups[name];
            group.min_duration =
                group.count == 0 ? duration : std::min(group.min_duration, duration);
            group.max_duration = std::max(group.max_duration, duration);
            group.total_duration += duration;
            ++group.count;
        }
    }
    rows.reserve(groups.size());
    for(auto& [name, group] : groups)
    {
        group.name = name;
        rows.push_back(std::move(group));
    }
    return true;
}

}  // namespace profiler_hub::pytorch
