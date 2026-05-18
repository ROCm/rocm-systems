// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/profiled_event_processor.hpp"

#include "core/config.hpp"
#include "core/output_file_registry.hpp"
#include "library/runtime.hpp"

#include <timemory/backends/dmp.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/components/timing/wall_clock.hpp>
#include <timemory/data/stream.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/manager/manager.hpp>
#include <timemory/math/compute.hpp>
#include <timemory/operations/types/print.hpp>
#include <timemory/operations/types/print_header.hpp>
#include <timemory/settings.hpp>
#include <timemory/storage/node.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofsys::trace_cache
{
namespace
{

constexpr const char* output_metric = "wall_clock_evt";

struct slot_t
{
    std::uint64_t exec_id        = 0;
    std::uint64_t parent_exec_id = 0;
    std::uint64_t thread_id      = 0;
    std::uint32_t depth          = 0;
    std::string   name;
    std::uint64_t enter_wall_ns   = 0;
    std::uint64_t exit_wall_ns    = 0;
    std::uint64_t enter_steady_ns = 0;
    bool          has_exit        = false;
};

struct out_node_t
{
    std::uint32_t              id        = 0;
    std::uint32_t              parent    = 0;
    std::int32_t               depth     = -1;
    std::int64_t               thread_id = -1;
    std::string                name;
    std::uint64_t              exec_id        = 0;
    std::uint64_t              parent_exec_id = 0;
    std::uint64_t              start_ns       = 0;
    std::uint64_t              end_ns         = 0;
    std::vector<std::uint32_t> children;
};

std::string
format_prefix(int d, const std::string& name)
{
    if(d <= 0) return name;
    std::string p;
    if(d > 1) p.append(std::string(static_cast<size_t>(d - 1) * 2, ' '));
    p += "|_";
    p += name;
    return p;
}

/// Maps captured OS thread ids to timemory-style labels (\c 0, \c 1, …) for JSON/text
/// prefixes. When \c collapse_threads is set, every scope uses label \c 0 like timemory.
struct thread_label_ctx_t
{
    explicit thread_label_ctx_t(bool collapse_threads)
    : collapse{ collapse_threads }
    {}

    std::int64_t label(std::int64_t sys_tid)
    {
        if(collapse) return 0;
        auto it = sys_to_label.find(sys_tid);
        if(it != sys_to_label.end()) return it->second;
        const std::int64_t lbl = next_label++;
        sys_to_label.emplace(sys_tid, lbl);
        return lbl;
    }

    bool                                           collapse = false;
    std::int64_t                                   next_label{ 0 };
    std::unordered_map<std::int64_t, std::int64_t> sys_to_label;
};

/// Prefix string in timemory flat JSON (\c ranks[].graph), e.g. \c "|0>>> |_MPI_Init".
std::string
timemory_json_prefix(std::int64_t display_tid, int depth, const std::string& name)
{
    const std::string body = (depth <= 0) ? name : format_prefix(depth, name);
    return std::string("|") + std::to_string(display_tid) + ">>> " + body;
}

tim::hash::hash_value_t
node_name_hash(const std::string& name)
{
    return tim::get_hash_id(name);
}

/// Timemory flat JSON \c rolling_hash is the sum of region hashes along the parent chain.
tim::hash::hash_value_t
rolling_hash_for_node(const std::vector<out_node_t>& nodes, std::uint32_t id)
{
    tim::hash::hash_value_t rolling = node_name_hash(nodes.at(id).name);
    for(std::uint32_t cur = nodes.at(id).parent; cur != 0; cur = nodes.at(cur).parent)
        rolling += node_name_hash(nodes.at(cur).name);
    return rolling;
}

std::string
flat_graph_agg_key(const std::vector<out_node_t>& nodes, std::uint32_t id,
                   thread_label_ctx_t& tlabels)
{
    const auto& n           = nodes.at(id);
    const auto  display_tid = tlabels.label(n.thread_id);
    return std::to_string(n.depth) + '\x1e' + std::to_string(node_name_hash(n.name)) +
           '\x1e' + std::to_string(rolling_hash_for_node(nodes, id)) + '\x1e' +
           timemory_json_prefix(display_tid, n.depth, n.name);
}

double
duration_sec(const out_node_t& n)
{
    return (n.end_ns > n.start_ns) ? (n.end_ns - n.start_ns) / 1.e9 : 0.0;
}

std::uint64_t
duration_ns(const out_node_t& n)
{
    return (n.end_ns > n.start_ns) ? (n.end_ns - n.start_ns) : 0;
}

nlohmann::json
wall_clock_entry_json(std::uint64_t accum_ns, std::int64_t laps)
{
    const double   sec = static_cast<double>(accum_ns) / 1.e9;
    nlohmann::json entry;
    entry["laps"]         = laps;
    entry["value"]        = accum_ns;
    entry["accum"]        = accum_ns;
    entry["repr_data"]    = sec;
    entry["repr_display"] = sec;
    return entry;
}

nlohmann::json
wall_clock_stats_json(const std::vector<double>& lap_secs)
{
    nlohmann::json stats;
    if(lap_secs.empty())
    {
        stats["sum"]    = 0.0;
        stats["count"]  = 0;
        stats["min"]    = 0.0;
        stats["max"]    = 0.0;
        stats["sqr"]    = 0.0;
        stats["mean"]   = 0.0;
        stats["stddev"] = 0.0;
        return stats;
    }

    double sum   = 0.0;
    double sumsq = 0.0;
    double minv  = lap_secs.front();
    double maxv  = lap_secs.front();
    for(double s : lap_secs)
    {
        sum += s;
        sumsq += s * s;
        minv = std::min(minv, s);
        maxv = std::max(maxv, s);
    }
    const auto   count = static_cast<std::int64_t>(lap_secs.size());
    const double mean  = sum / static_cast<double>(count);
    const double var_p =
        (count > 0) ? (sumsq / static_cast<double>(count) - mean * mean) : 0.0;
    const double var = (var_p > 0.0) ? var_p : 0.0;

    stats["sum"]    = sum;
    stats["count"]  = count;
    stats["min"]    = minv;
    stats["max"]    = maxv;
    stats["sqr"]    = sumsq;
    stats["mean"]   = mean;
    stats["stddev"] = std::sqrt(var);
    return stats;
}

struct flat_graph_agg_t
{
    std::string             prefix;
    std::int32_t            depth    = 0;
    tim::hash::hash_value_t hash     = 0;
    tim::hash::hash_value_t rolling  = 0;
    std::int64_t            laps     = 0;
    std::uint64_t           accum_ns = 0;
    std::vector<double>     lap_secs;
};

void
collect_flat_graph_agg(const std::vector<out_node_t>& nodes, std::uint32_t id,
                       std::unordered_map<std::string, flat_graph_agg_t>& agg,
                       std::vector<std::string>& order, thread_label_ctx_t& tlabels)
{
    if(id != 0)
    {
        const auto& n   = nodes.at(id);
        const auto  key = flat_graph_agg_key(nodes, id, tlabels);
        auto&       a   = agg[key];
        if(a.laps == 0)
        {
            order.push_back(key);
            a.prefix  = timemory_json_prefix(tlabels.label(n.thread_id), n.depth, n.name);
            a.depth   = n.depth;
            a.hash    = node_name_hash(n.name);
            a.rolling = rolling_hash_for_node(nodes, id);
        }
        const double sec = duration_sec(n);
        ++a.laps;
        a.accum_ns += duration_ns(n);
        a.lap_secs.push_back(sec);
    }
    for(auto cid : nodes.at(id).children)
        collect_flat_graph_agg(nodes, cid, agg, order, tlabels);
}

std::vector<flat_graph_agg_t>
build_flat_graph_aggregate(const std::vector<out_node_t>& nodes)
{
    const bool collapse_threads =
        tim::settings::instance() && tim::settings::instance()->get_collapse_threads();
    thread_label_ctx_t tlabels{ collapse_threads };

    std::unordered_map<std::string, flat_graph_agg_t> agg;
    std::vector<std::string>                          order;
    collect_flat_graph_agg(nodes, 0, agg, order, tlabels);

    std::vector<flat_graph_agg_t> rows;
    rows.reserve(order.size());
    for(const auto& key : order)
        rows.push_back(agg.at(key));
    return rows;
}

nlohmann::json
build_timemory_flat_graph(const std::vector<out_node_t>& nodes)
{
    const auto rows = build_flat_graph_aggregate(nodes);

    nlohmann::json graph = nlohmann::json::array();
    for(const auto& a : rows)
    {
        nlohmann::json row;
        row["hash"]         = a.hash;
        row["prefix"]       = a.prefix;
        row["depth"]        = a.depth;
        row["entry"]        = wall_clock_entry_json(a.accum_ns, a.laps);
        row["stats"]        = wall_clock_stats_json(a.lap_secs);
        row["rolling_hash"] = a.rolling;
        graph.push_back(std::move(row));
    }
    return graph;
}

void
write_text(const std::vector<out_node_t>& nodes, std::ofstream& os)
{
    using wc_t            = tim::component::wall_clock;
    using result_t        = tim::node::result<wc_t>;
    using stats_t         = typename result_t::stats_type;
    using get_return_type = decltype(std::declval<const wc_t>().get());
    using compute_type    = tim::math::compute<get_return_type>;

    const auto rows = build_flat_graph_aggregate(nodes);
    if(rows.empty()) return;

    // print_header<> returns immediately when runtime_enabled is false (common after
    // timemory_finalize), but print<> still writes rows and requires headers.
    const bool saved_runtime = tim::trait::runtime_enabled<wc_t>::get();
    tim::trait::runtime_enabled<wc_t>::set(true);

    std::vector<result_t> storage;
    storage.reserve(rows.size());
    std::vector<result_t*> flattened;
    flattened.reserve(rows.size());

    std::int64_t max_depth = 0;
    for(const auto& a : rows)
    {
        max_depth = std::max(max_depth, static_cast<std::int64_t>(a.depth));

        stats_t stats{};
        for(double sec : a.lap_secs)
            stats += sec;

        wc_t wc{};
        wc.set_value(static_cast<wc_t::value_type>(a.accum_ns));
        wc.set_accum(static_cast<wc_t::value_type>(a.accum_ns));
        wc.set_laps(a.laps);

        std::vector<tim::hash::hash_value_t> hierarchy{};
        storage.emplace_back(a.hash, wc, std::string(" ") + a.prefix, a.depth, a.rolling,
                             hierarchy, stats, 0, tim::process::get_id());
        flattened.push_back(&storage.back());
    }

    auto stream = std::make_shared<tim::utility::stream>(
        '|', '-', wc_t::get_format_flags(), wc_t::get_width(), wc_t::get_precision());

    std::string banner = wc_t::description();
    for(char& c : banner)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    stream->set_banner(banner);

    // Same flattened preorder walk and % SELF logic as
    // tim::operation::finalize::print<wc_t>::write_stream.
    bool header_done = false;
    for(auto itr = flattened.begin(); itr != flattened.end(); ++itr)
    {
        auto& itr_obj    = (*itr)->data();
        auto& itr_prefix = (*itr)->prefix();
        auto& itr_depth  = (*itr)->depth();
        auto  itr_laps   = itr_obj.get_laps();

        if(itr_depth < 0 || itr_depth > max_depth) continue;

        get_return_type exclusive_values{};
        if(itr_depth < max_depth)
        {
            auto         eitr       = itr;
            std::int64_t nexclusive = 0;
            std::advance(eitr, 1);
            exclusive_values = get_return_type{};
            if(eitr != flattened.end())
            {
                auto eitr_depth = (*eitr)->depth();
                while(eitr_depth != itr_depth)
                {
                    auto& eitr_obj = (*eitr)->data();
                    if(eitr_depth == itr_depth + 1)
                    {
                        if(nexclusive == 0)
                            exclusive_values = eitr_obj.get();
                        else
                            compute_type::plus(exclusive_values, eitr_obj.get());
                        ++nexclusive;
                    }
                    ++eitr;
                    if(eitr == flattened.end()) break;
                    eitr_depth = (*eitr)->depth();
                }
            }
        }

        const auto itr_self = compute_type::percent_diff(exclusive_values, itr_obj.get());
        auto&      itr_stats = (*itr)->stats();

        if(!header_done)
        {
            tim::operation::print_header<wc_t>(itr_obj, *stream, itr_stats);
            header_done = true;
        }

        tim::operation::print<wc_t>(itr_obj, *stream, itr_prefix, itr_laps, itr_depth,
                                    itr_self, itr_stats);
        stream->add_row();
    }

    tim::trait::runtime_enabled<wc_t>::set(saved_runtime);

    os << *stream << std::flush;
}

bool
replay_to_nodes(const std::vector<wall_clock_event_sample>& ev,
                std::vector<out_node_t>&                    out_nodes)
{
    if(ev.empty()) return false;

    auto legacy_replay_less = [](const wall_clock_event_sample& a,
                                 const wall_clock_event_sample& b) {
        if(a.steady_ns != b.steady_ns) return a.steady_ns < b.steady_ns;
        if(a.thread_id != b.thread_id) return a.thread_id < b.thread_id;
        if(a.exec_id != b.exec_id) return a.exec_id < b.exec_id;
        return a.event_kind < b.event_kind;
    };

    std::vector<wall_clock_event_sample> sorted = ev;
    std::sort(sorted.begin(), sorted.end(),
              [&](const wall_clock_event_sample& a, const wall_clock_event_sample& b) {
                  if(a.record_seq != b.record_seq) return a.record_seq < b.record_seq;
                  return legacy_replay_less(a, b);
              });

    std::uint64_t                                                 max_wall = 0;
    std::unordered_map<std::uint64_t, slot_t>                     slots;
    std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> by_parent;

    for(const auto& s : sorted)
    {
        max_wall = std::max(max_wall, s.wall_ns);

        if(s.event_kind == static_cast<std::uint8_t>(wall_clock_scope_event_kind::enter))
        {
            slot_t sl;
            sl.exec_id         = s.exec_id;
            sl.parent_exec_id  = s.parent_exec_id;
            sl.thread_id       = s.thread_id;
            sl.depth           = s.depth;
            sl.name            = s.name;
            sl.enter_wall_ns   = s.wall_ns;
            sl.enter_steady_ns = s.steady_ns;
            sl.has_exit        = false;
            slots[s.exec_id]   = std::move(sl);
            by_parent[s.parent_exec_id].push_back(s.exec_id);
        }
        else if(s.event_kind ==
                static_cast<std::uint8_t>(wall_clock_scope_event_kind::exit))
        {
            auto it = slots.find(s.exec_id);
            if(it != slots.end())
            {
                it->second.exit_wall_ns = s.wall_ns;
                it->second.has_exit     = true;
            }
        }
    }

    for(auto& kv : slots)
    {
        if(!kv.second.has_exit)
        {
            kv.second.exit_wall_ns = max_wall;
            kv.second.has_exit     = true;
        }
    }

    out_nodes.clear();
    out_node_t root{};
    root.id        = 0;
    root.parent    = 0;
    root.depth     = -1;
    root.thread_id = -1;
    root.start_ns  = 0;
    root.end_ns    = 0;
    out_nodes.push_back(root);

    std::function<void(std::uint64_t, std::uint32_t)> visit;
    visit = [&](std::uint64_t exec_id, std::uint32_t parent_out) {
        auto sit = slots.find(exec_id);
        if(sit == slots.end()) return;
        const slot_t& sl = sit->second;

        const auto nid = static_cast<std::uint32_t>(out_nodes.size());
        out_node_t n{};
        n.id             = nid;
        n.parent         = parent_out;
        n.depth          = out_nodes.at(parent_out).depth + 1;
        n.thread_id      = static_cast<std::int64_t>(sl.thread_id);
        n.name           = sl.name;
        n.exec_id        = sl.exec_id;
        n.parent_exec_id = sl.parent_exec_id;
        n.start_ns       = sl.enter_wall_ns;
        n.end_ns         = sl.exit_wall_ns;
        out_nodes.at(parent_out).children.push_back(nid);
        out_nodes.push_back(std::move(n));

        auto pit = by_parent.find(exec_id);
        if(pit != by_parent.end())
        {
            for(auto ce : pit->second)
                visit(ce, nid);
        }
    };

    auto roots_it = by_parent.find(0);
    if(roots_it != by_parent.end())
    {
        for(auto rid : roots_it->second)
            visit(rid, 0);
    }

    return out_nodes.size() > 1;
}

}  // namespace

profiled_event_processor_t::profiled_event_processor_t(int pid, int ppid,
                                                       output_file_registry& registry)
: m_pid(pid)
, m_ppid(ppid)
, m_registry(registry)
{
    (void) m_pid;
    (void) m_ppid;
}

void
profiled_event_processor_t::prepare_for_processing()
{
    m_events.clear();
}

void
profiled_event_processor_t::finalize_processing()
{
    if(!config::get_use_timemory()) return;
    if(m_events.empty()) return;
    if(!config::output_filtering::is_output_enabled_for_current_mpi_rank()) return;

    // Trace-cache replay can run before tim::timemory_finalize(); timemory's
    // file_output() flag is often still false here, while timemory wall_clock files are
    // written after finalize when it becomes true. Do not gate on file_output() — use
    // get_use_timemory() (already checked) and successful opens below.

    std::vector<out_node_t> nodes;
    if(!replay_to_nodes(m_events, nodes)) return;

    auto path_cfg       = settings::compose_filename_config{};
    path_cfg.use_suffix = config::get_use_pid();
    path_cfg.suffix     = settings::default_process_suffix();
    path_cfg.make_dir   = true;

    const std::string json_path =
        settings::compose_output_filename(output_metric, "json", path_cfg);
    const std::string txt_path =
        settings::compose_output_filename(output_metric, "txt", path_cfg);
    if(json_path.empty() || txt_path.empty()) return;

    nlohmann::json flat_graph = build_timemory_flat_graph(nodes);

    // Match timemory \c wall_clock.json layout (key \c "wall_clock") for hatchet/tools.
    // Output path remains \c wall_clock_evt-* via \p output_metric.
    constexpr const char* json_component_key = "wall_clock";

    nlohmann::json root;
    auto&          wc       = root["timemory"][json_component_key];
    wc["type"]              = "wall_clock";
    wc["description"]       = "Real-clock timer (i.e. wall-clock timer)";
    wc["unit_value"]        = 1000000000;
    wc["unit_repr"]         = "sec";
    wc["thread_scope_only"] = false;
    wc["thread_count"]      = tim::manager::get_thread_count();
    wc["mpi_size"]          = dmp::size();
    wc["upcxx_size"]        = 1;
    wc["process_count"]     = 1;
    wc["num_ranks"]         = dmp::size();
    wc["concurrency"]       = tim::manager::get_thread_count();
    wc["ranks"]             = nlohmann::json::array();

    nlohmann::json rank_entry;
    rank_entry["rank"]       = dmp::rank();
    rank_entry["graph_size"] = flat_graph.size();
    rank_entry["graph"]      = std::move(flat_graph);
    wc["ranks"].push_back(std::move(rank_entry));

    bool json_ok = false;
    bool txt_ok  = false;
    {
        std::ofstream os{ json_path };
        json_ok = static_cast<bool>(os);
        if(json_ok) os << root.dump(4);
    }
    {
        std::ofstream os{ txt_path };
        txt_ok = static_cast<bool>(os);
        if(txt_ok) write_text(nodes, os);
    }

    if(json_ok) m_registry.register_file(json_path, output_format::json, output_metric);
    if(txt_ok) m_registry.register_file(txt_path, output_format::text, output_metric);
}

void
profiled_event_processor_t::handle(const wall_clock_event_sample& s)
{
    m_events.push_back(s);
}

void
profiled_event_processor_t::handle(const kernel_dispatch_sample&)
{}

void
profiled_event_processor_t::handle(const scratch_memory_sample&)
{}

void
profiled_event_processor_t::handle(const memory_copy_sample&)
{}

void
profiled_event_processor_t::handle(const memory_allocate_sample&)
{}

void
profiled_event_processor_t::handle(const region_sample&)
{}

void
profiled_event_processor_t::handle(const in_time_sample&)
{}

void
profiled_event_processor_t::handle(const pmc_event_with_sample&)
{}

void
profiled_event_processor_t::handle(const gpu_pmc_sample&)
{}

void
profiled_event_processor_t::handle(const ainic_pmc_sample&)
{}

void
profiled_event_processor_t::handle(const cpu_pmc_sample&)
{}

void
profiled_event_processor_t::handle(const backtrace_region_sample&)
{}

void
profiled_event_processor_t::handle(const kfd_sample&)
{}

}  // namespace rocprofsys::trace_cache
