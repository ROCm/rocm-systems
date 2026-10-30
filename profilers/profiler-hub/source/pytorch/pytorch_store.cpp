// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "pytorch_store.hpp"
#include "pytorch_catalog.hpp"
#include "reader_catalog.hpp"
#include "utility.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_map>
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

std::uint64_t
to_ns(std::uint64_t base_ns, std::int64_t ts_ns)
{
    const long long v = static_cast<long long>(base_ns) + ts_ns;
    return v < 0 ? 0 : static_cast<std::uint64_t>(v);
}

using track_key_t = std::pair<std::int32_t, std::int32_t>;

struct sort_event_t
{
    std::uint64_t start{};
    std::uint64_t end{};
    std::uint32_t name_id{};
    std::uint16_t depth{};
    std::uint8_t  category_id{};
};

bool
event_order(const sort_event_t& a, const sort_event_t& b)
{
    if(a.start != b.start) return a.start < b.start;
    return a.end > b.end;
}
}  // namespace

std::shared_ptr<const pytorch_store_t>
pytorch_store_t::open(const std::string& path)
{
    const auto t0 = common::steady_clock_t::now();

    auto trace = load_kineto(path);
    if(trace.events.empty())
    {
        throw std::runtime_error("no X events with ts and dur in " + path);
    }

    std::shared_ptr<pytorch_store_t> s{ new pytorch_store_t };
    s->m_path           = path;
    s->m_gzip           = trace.gzip;
    s->m_base_time_ns   = trace.base_time_ns;
    s->m_schema_version = trace.schema_version;
    s->m_device_count   = trace.device_names.size();
    s->m_diag           = trace.diag;

    const auto t_derive = common::steady_clock_t::now();
    s->derive(trace);
    s->m_derive_ms   = common::elapsed_ms(t_derive);
    s->m_total_ms    = common::elapsed_ms(t0);
    s->m_peak_rss_kb = peak_rss_kb();
    return s;
}

void
pytorch_store_t::derive(loaded_trace_t& t)
{
    std::array<bool, 256> gpu_cat{};
    for(std::size_t i = 0; i < t.cats.size(); ++i)
    {
        gpu_cat[i] = t.cat_classes[i].gpu;
    }

    std::map<track_key_t, std::vector<std::uint32_t>> host_groups;
    std::map<track_key_t, std::vector<std::uint32_t>> gpu_groups;
    std::map<std::int32_t, std::uint64_t>             host_events_by_pid;
    std::unordered_map<std::int64_t, std::int32_t>    launcher_pid_by_link;

    for(std::size_t i = 0; i < t.events.size(); ++i)
    {
        const auto& e   = t.events[i];
        const auto  idx = static_cast<std::uint32_t>(i);
        if(gpu_cat[e.cat_id])
        {
            const track_key_t k{ e.device >= 0 ? e.device : e.pid,
                                 e.stream >= 0 ? e.stream : e.tid };
            gpu_groups[k].push_back(idx);
        }
        else
        {
            host_groups[{ e.pid, e.tid }].push_back(idx);
            ++host_events_by_pid[e.pid];
            if(t.cat_classes[e.cat_id].launch_site && e.correlation >= 0)
            {
                launcher_pid_by_link.emplace(e.correlation, e.pid);
            }
        }
    }

    std::int32_t  busiest_pid   = 0;
    std::uint64_t busiest_count = 0;
    for(const auto& [pid, count] : host_events_by_pid)
    {
        if(count > busiest_count)
        {
            busiest_count = count;
            busiest_pid   = pid;
        }
    }

    m_catalog = make_catalog(t);
    std::vector<const char*> name_ptr(t.names.size());
    for(std::size_t i = 0; i < name_ptr.size(); ++i)
    {
        name_ptr[i] = m_catalog->string_utility.at(k_catalog_name_id_base + i).c_str();
    }

    std::vector<std::vector<sort_event_t>> sorted_by_track;

    auto sort_events = [&](const std::vector<std::uint32_t>& idxs,
                           const std::vector<std::uint16_t>* depth) {
        std::vector<sort_event_t> sorted;
        sorted.reserve(idxs.size());
        for(std::size_t k = 0; k < idxs.size(); ++k)
        {
            const auto& e     = t.events[idxs[k]];
            const auto  start = to_ns(m_base_time_ns, e.ts_ns);
            auto        dur   = e.dur_ns;
            if(dur < 0)
            {
                dur = 0;
                ++m_clamped_negative_dur;
            }
            sorted.push_back({ start,
                               start + static_cast<std::uint64_t>(dur),
                               e.name_id,
                               depth != nullptr ? (*depth)[k] : k_not_in_tree,
                               e.cat_id });
        }
        std::stable_sort(sorted.begin(), sorted.end(), event_order);
        sorted_by_track.push_back(std::move(sorted));
    };

    const tree_options_t options{};

    for(const auto& [key, idxs] : host_groups)
    {
        std::vector<event_t> thread_events;
        thread_events.reserve(idxs.size());
        for(const auto i : idxs)
        {
            thread_events.push_back(t.events[i]);
        }
        auto tree     = build_host_tree(thread_events, t, options);
        thread_events = std::vector<event_t>{};

        store_track_t track{};
        track.id       = static_cast<std::uint32_t>(m_tracks.size() + 1);
        const auto it  = t.thread_names.find(key);
        track.name     = it != t.thread_names.end() && !it->second.empty()
                             ? it->second
                             : "Thread " + std::to_string(key.second);
        track.category = PH_TRACK_CATEGORY_THREAD;
        track.pid      = static_cast<std::uint32_t>(key.first);
        track.tid      = static_cast<std::uint32_t>(key.second);
        sort_events(idxs, &tree.depth);
        m_tracks.push_back(std::move(track));
        m_tree_stats.push_back(tree.stats);
    }

    for(const auto& [key, idxs] : gpu_groups)
    {
        std::int32_t launcher_pid = -1;
        for(const auto i : idxs)
        {
            ++m_gpu_events;
            const auto it = launcher_pid_by_link.find(t.events[i].correlation);
            if(t.events[i].correlation < 0 || it == launcher_pid_by_link.end())
            {
                ++m_gpu_unmatched;
            }
            else if(launcher_pid < 0)
            {
                launcher_pid = it->second;
            }
        }
        if(launcher_pid < 0)
        {
            launcher_pid = busiest_pid;
            ++m_pid_fallback_tracks;
        }

        store_track_t track{};
        track.id        = static_cast<std::uint32_t>(m_tracks.size() + 1);
        track.name      = fmt::format("GPU {} / Stream {}", key.first, key.second);
        track.category  = PH_TRACK_CATEGORY_STREAM;
        track.pid       = static_cast<std::uint32_t>(launcher_pid);
        track.agent_id  = static_cast<std::uint32_t>(key.first) + 1;
        track.stream_id = static_cast<std::uint32_t>(key.second);
        sort_events(idxs, nullptr);
        m_tracks.push_back(std::move(track));
        m_tree_stats.push_back(tree_stats_t{});
    }

    // Dropping the loader's events before the per-track arrays are built keeps the peak
    // below the sum of the two.
    std::vector<event_t>{}.swap(t.events);
    host_groups.clear();
    gpu_groups.clear();

    for(std::size_t k = 0; k < m_tracks.size(); ++k)
    {
        auto& track  = m_tracks[k];
        auto& sorted = sorted_by_track[k];

        track.start_ts = sorted.front().start;
        track.end_ts   = 0;
        track.events.reserve(sorted.size());
        track.category_ids.reserve(sorted.size());
        for(const auto& ev : sorted)
        {
            track.end_ts = std::max(track.end_ts, ev.end);
            track.events.push_back({ ev.start,
                                     ev.end,
                                     name_ptr[ev.name_id],
                                     ev.depth == k_not_in_tree ? 0u : ev.depth + 1u });
            track.category_ids.push_back(ev.category_id);
        }
        sorted = std::vector<sort_event_t>{};
    }

    fill_catalog(*m_catalog, m_tracks, t);
}

const std::vector<ph_event_t>&
pytorch_store_t::track_events(std::uint32_t id) const
{
    if(!has_track(id)) throw std::out_of_range("unknown pytorch track id");
    return m_tracks[id - 1].events;
}

std::vector<ph_event_t>
pytorch_store_t::query(std::uint32_t id,
                       std::uint64_t start_ts,
                       std::uint64_t end_ts) const
{
    std::vector<ph_event_t> out;
    for(const auto& e : track_events(id))
    {
        if(end_ts != 0 && e.start > end_ts) break;  // sorted by start
        if(e.end < start_ts) continue;
        out.push_back(e);
    }
    return out;
}

const tree_stats_t&
pytorch_store_t::tree_stats(std::uint32_t host_track_id) const
{
    if(!has_track(host_track_id)) throw std::out_of_range("unknown pytorch track id");
    return m_tree_stats[host_track_id - 1];
}

std::string
pytorch_store_t::summary_line() const
{
    constexpr const char* p = "[Profiler-Hub] pytorch:";
    std::string           s;
    auto                  line = [&](const std::string& text) {
        if(!s.empty()) s += '\n';
        s += fmt::format("{} {}", p, text);
    };

    line(fmt::format("file={} encoding={}", m_path, m_gzip ? "gzip" : "plain"));
    line(fmt::format("base_time_ns={} schemaVersion={} devices={}",
                     m_base_time_ns,
                     m_schema_version,
                     m_device_count));
    line(fmt::format("events total={} X={} flow_s={} flow_f={} M={} other={} "
                     "bad_X_without_ts_or_dur={} "
                     "clamped_negative_dur={}",
                     m_diag.total,
                     m_diag.x_events,
                     m_diag.flow_s,
                     m_diag.flow_f,
                     m_diag.meta_events,
                     m_diag.other_events,
                     m_diag.bad_events,
                     m_clamped_negative_dur));
    line(fmt::format("gpu_join matched={}/{} tracks_with_pid_fallback={}",
                     m_gpu_events - m_gpu_unmatched,
                     m_gpu_events,
                     m_pid_fallback_tracks));
    line(fmt::format(
        "catalog nodes={} processes={} threads={} agents={} tracks={} strings={}",
        m_catalog->nodes.size(),
        m_catalog->processes.size(),
        m_catalog->threads.size(),
        m_catalog->agents.size(),
        m_catalog->tracks.size(),
        m_catalog->string_utility.size()));

    std::uint64_t retained_events = 0;
    for(const auto& tr : m_tracks)
    {
        retained_events += tr.events.size();
    }
    line(fmt::format("retained events={} bytes={}",
                     retained_events,
                     retained_events * (sizeof(ph_event_t) + sizeof(std::uint8_t))));
    for(const auto& tr : m_tracks)
    {
        line(fmt::format(
            "track id={} category={} name=\"{}\" pid={} tid={} agent_id={} stream_id={} "
            "events={} start_ns={} end_ns={}",
            tr.id,
            tr.category == PH_TRACK_CATEGORY_THREAD ? "THREAD" : "STREAM",
            tr.name,
            tr.pid,
            tr.tid,
            tr.agent_id,
            tr.stream_id,
            tr.events.size(),
            tr.start_ts,
            tr.end_ts));
        if(tr.category != PH_TRACK_CATEGORY_THREAD) continue;
        const auto& ts = m_tree_stats[tr.id - 1];
        line(fmt::format(
            "tree track={} placed={} orphans_kept_as_child={} small_overlap_pops={} "
            "children_ending_after_parent={} max_level={}",
            tr.id,
            ts.placed,
            ts.orphans_kept_as_child,
            ts.small_overlap_pops,
            ts.children_ending_after_parent,
            ts.max_level));
    }
    line(fmt::format(
        "time gunzip_ms={} parse_ms={} derive_ms={} total_ms={} peak_rss_kb={}",
        m_diag.gunzip_ms,
        m_diag.parse_ms,
        m_derive_ms,
        m_total_ms,
        m_peak_rss_kb));
    return s;
}
}  // namespace profiler_hub::pytorch
