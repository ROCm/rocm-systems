// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "pytorch_store.hpp"

#include "utility.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace profiler_hub::pytorch
{

namespace
{

using track_key_t = std::pair<std::int64_t, std::int64_t>;

// A time before zero is clamped to 0 and reported through @p clamped. A base plus ts that
// does not fit in an int64 rejects the trace.
std::uint64_t
to_ns(std::uint64_t base_ns, std::int64_t ts_ns, bool& clamped)
{
    constexpr auto k_max = std::numeric_limits<std::int64_t>::max();
    if(base_ns > static_cast<std::uint64_t>(k_max) ||
       (ts_ns > 0 && static_cast<std::int64_t>(base_ns) > k_max - ts_ns))
    {
        throw std::runtime_error("timestamp out of range");
    }
    const std::int64_t v = static_cast<std::int64_t>(base_ns) + ts_ns;
    clamped              = v < 0;
    return clamped ? 0 : static_cast<std::uint64_t>(v);
}

std::int64_t
link_of(const loaded_trace_t& t, const event_t& e)
{
    return t.link_by_correlation ? e.corr : e.ext;
}

bool
event_order(const store_row_t& a, const store_row_t& b)
{
    if(a.start != b.start) return a.start < b.start;
    return a.end > b.end;
}

// Unknown categories, memsets, GPU annotations and the CUDA syncs other than a stream
// wait have no event type of their own and are served as regions.
std::vector<store_row_t>
sorted_rows(loaded_trace_t& t, const std::vector<std::uint32_t>& idxs)
{
    std::vector<store_row_t> rows;
    rows.reserve(idxs.size());
    for(const auto i : idxs)
    {
        const auto& e       = t.events[i];
        bool        clamped = false;
        const auto  start   = to_ns(t.base_time_ns, e.ts_ns, clamped);
        const auto  dur     = e.dur_ns < 0 ? 0 : e.dur_ns;
        if(clamped || e.dur_ns < 0) ++t.diag.clamped_events;
        rows.push_back({ start,
                         start + static_cast<std::uint64_t>(dur),
                         e.name_id,
                         e.phase == phase_t::memory
                             ? reader_types::event_type_t::memory_allocate
                             : event_type_of(t.cat_classes[e.cat_id], t.names[e.name_id]),
                         i });
    }
    std::stable_sort(rows.begin(), rows.end(), event_order);
    return rows;
}

// Each flow end binds to the innermost event of its lane that holds its time: the event
// with the latest start among those that started and have not ended. Ends with no such
// event are left out and counted.
void
bind_flows(loaded_trace_t&                           trace,
           pytorch_store_t&                          store,
           const std::map<track_key_t, std::size_t>& host_track_of,
           const std::map<track_key_t, std::size_t>& gpu_track_of)
{
    struct pending_t
    {
        std::uint64_t ts;
        std::size_t   flow;
    };
    std::map<std::size_t, std::vector<pending_t>> by_track;
    for(std::size_t i = 0; i < trace.flows.size(); ++i)
    {
        const auto&                flow = trace.flows[i];
        const track_key_t          key{ flow.pid, flow.tid };
        std::optional<std::size_t> track;
        if(const auto it = host_track_of.find(key); it != host_track_of.end())
        {
            track = it->second;
        }
        else if(const auto lane = gpu_track_of.find(key); lane != gpu_track_of.end())
        {
            track = lane->second;
        }
        if(!track)
        {
            ++trace.diag.flow_unbound;
            continue;
        }
        bool clamped = false;
        by_track[*track].push_back({ to_ns(trace.base_time_ns, flow.ts_ns, clamped), i });
    }

    for(auto& [track, flows] : by_track)
    {
        const auto& rows = store.tracks[track].rows;
        std::stable_sort(
            flows.begin(), flows.end(), [](const pending_t& a, const pending_t& b) {
                return a.ts < b.ts;
            });
        std::vector<std::size_t> open;
        std::size_t              next = 0;
        for(const auto& pending : flows)
        {
            while(next < rows.size() && rows[next].start <= pending.ts)
            {
                open.push_back(next++);
            }
            while(!open.empty() && rows[open.back()].end < pending.ts)
            {
                open.pop_back();
            }
            if(open.empty())
            {
                ++trace.diag.flow_unbound;
                continue;
            }
            const auto& flow = trace.flows[pending.flow];
            store.flows.push_back({ rows[open.back()].event,
                                    flow.phase,
                                    flow.id,
                                    flow.category,
                                    flow.args_off,
                                    flow.args_len });
        }
    }
    std::stable_sort(
        store.flows.begin(),
        store.flows.end(),
        [](const bound_flow_t& a, const bound_flow_t& b) { return a.event < b.event; });
}

}  // namespace

pytorch_store_t
load_store(const std::string& path, std::function<bool()> cancelled)
{
    const auto t0    = common::steady_clock_t::now();
    auto       trace = load_kineto(path, std::move(cancelled));
    if(trace.events.empty())
    {
        throw std::runtime_error("no X events with ts and dur in " + path);
    }
    pytorch_store_t store;
    store.parse_ms      = common::elapsed_ms(t0);
    const auto t_derive = common::steady_clock_t::now();

    std::map<track_key_t, std::vector<std::uint32_t>> host_groups;
    std::map<track_key_t, std::vector<std::uint32_t>> gpu_groups;
    std::map<track_key_t, std::size_t>                host_track_of;
    std::map<track_key_t, std::size_t>                gpu_track_of;
    std::map<std::int64_t, std::uint64_t>             host_events_by_pid;
    std::unordered_map<std::int64_t, std::int64_t>    launcher_pid_by_link;
    for(std::size_t i = 0; i < trace.events.size(); ++i)
    {
        const auto& e   = trace.events[i];
        const auto  idx = static_cast<std::uint32_t>(i);
        const auto& cls = trace.cat_classes[e.cat_id];
        if(cls.gpu)
        {
            const track_key_t k{ e.device >= 0 ? e.device : e.pid,
                                 e.stream >= 0 ? e.stream : e.tid };
            gpu_groups[k].push_back(idx);
        }
        else
        {
            host_groups[{ e.pid, e.tid }].push_back(idx);
            ++host_events_by_pid[e.pid];
            if(cls.launch_site && link_of(trace, e) >= 0)
            {
                launcher_pid_by_link.emplace(link_of(trace, e), e.pid);
            }
        }
    }

    std::int64_t  busiest_pid   = 0;
    std::uint64_t busiest_count = 0;
    for(const auto& [pid, count] : host_events_by_pid)
    {
        if(count > busiest_count)
        {
            busiest_count = count;
            busiest_pid   = pid;
        }
    }

    for(auto& [key, idxs] : host_groups)
    {
        if(const auto lane = trace.lane_names.find(key.second);
           lane != trace.lane_names.end())
        {
            trace.thread_names.emplace(key, lane->second);
        }
        store_track_t track{};
        track.id       = static_cast<std::uint32_t>(store.tracks.size());
        const auto it  = trace.thread_names.find(key);
        track.name     = it != trace.thread_names.end() && !it->second.empty()
                             ? it->second
                             : "Thread " + std::to_string(key.second);
        track.category = reader_types::track_kind_t::thread;
        track.pid      = key.first;
        track.tid      = key.second;
        track.rows     = sorted_rows(trace, idxs);
        std::vector<std::uint32_t>{}.swap(idxs);
        host_track_of.emplace(key, store.tracks.size());
        store.tracks.push_back(std::move(track));
    }

    // A stream belongs to the process that launched its first matched kernel.
    for(auto& [key, idxs] : gpu_groups)
    {
        std::int64_t launcher_pid = -1;
        for(const auto i : idxs)
        {
            const auto link = link_of(trace, trace.events[i]);
            const auto it   = launcher_pid_by_link.find(link);
            if(link >= 0 && it != launcher_pid_by_link.end())
            {
                launcher_pid = it->second;
                break;
            }
        }
        store_track_t track{};
        track.id        = static_cast<std::uint32_t>(store.tracks.size());
        track.name      = key.second < 0
                              ? fmt::format("GPU {} / Device-wide", key.first)
                              : fmt::format("GPU {} / Stream {}", key.first, key.second);
        track.category  = reader_types::track_kind_t::stream;
        track.pid       = launcher_pid < 0 ? busiest_pid : launcher_pid;
        track.agent_id  = static_cast<std::uint64_t>(key.first) + 1;
        track.stream_id = static_cast<std::uint64_t>(key.second);
        track.rows      = sorted_rows(trace, idxs);
        std::vector<std::uint32_t>{}.swap(idxs);
        gpu_track_of.emplace(key, store.tracks.size());
        store.tracks.push_back(std::move(track));
    }
    bind_flows(trace, store, host_track_of, gpu_track_of);

    for(auto& track : store.tracks)
    {
        track.start_ts = track.rows.front().start;
        for(const auto& row : track.rows)
        {
            track.end_ts = std::max(track.end_ts, row.end);
        }
    }

    store.events        = std::move(trace.events);
    store.memory        = std::move(trace.memory);
    store.cat_classes   = std::move(trace.cat_classes);
    store.args          = std::move(trace.args);
    store.names         = std::move(trace.names);
    store.cats          = std::move(trace.cats);
    store.thread_names  = std::move(trace.thread_names);
    store.process_names = std::move(trace.process_names);
    store.process_sort  = std::move(trace.process_sort);
    store.thread_sort   = std::move(trace.thread_sort);
    store.busiest_pid   = busiest_pid;
    store.trace_info    = std::move(trace.trace_info);
    if(!trace.markers.empty())
    {
        auto& info = store.trace_info;
        if(!info.empty()) info.push_back(',');
        info += "\"markers\":[";
        for(std::size_t i = 0; i < trace.markers.size(); ++i)
        {
            const auto& marker  = trace.markers[i];
            bool        clamped = false;
            if(i != 0) info.push_back(',');
            info += "{\"name\":";
            append_json_string(info, marker.name);
            info += ",\"ts_ns\":" +
                    std::to_string(to_ns(trace.base_time_ns, marker.ts_ns, clamped)) +
                    ",\"pid\":" + std::to_string(marker.pid) +
                    ",\"tid\":" + std::to_string(marker.tid);
            if(marker.args_len != 0)
            {
                info += ",\"args\":";
                info.append(trace.args, marker.args_off, marker.args_len);
            }
            info.push_back('}');
        }
        info.push_back(']');
    }
    store.host_name    = std::move(trace.host_name);
    store.device_names = std::move(trace.device_names);
    store.device_json  = std::move(trace.device_json);
    store.diag         = trace.diag;
    store.derive_ms    = common::elapsed_ms(t_derive);
    return store;
}

}  // namespace profiler_hub::pytorch
