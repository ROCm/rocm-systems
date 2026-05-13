// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/profiled_event_processor.hpp"

#include "core/config.hpp"
#include "core/output_file_registry.hpp"
#include "library/runtime.hpp"

#include <timemory/backends/dmp.hpp>
#include <timemory/manager/manager.hpp>
#include <timemory/settings.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
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
    std::uint64_t              start_ns = 0;
    std::uint64_t              end_ns   = 0;
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

std::string
timemory_label_cell(std::int64_t tid, int depth, const std::string& name)
{
    const std::string body = (depth <= 0) ? name : format_prefix(depth, name);
    return std::string(" |") + std::to_string(tid) + ">>> " + body;
}

double
duration_sec(const out_node_t& n)
{
    return (n.end_ns > n.start_ns) ? (n.end_ns - n.start_ns) / 1.e9 : 0.0;
}

void
dfs_graph(const std::vector<out_node_t>& nodes, std::uint32_t nid, nlohmann::json& graph)
{
    const auto& n = nodes.at(nid);
    if(n.id != 0)
    {
        const double   sec = duration_sec(n);
        nlohmann::json row;
        row["prefix"]         = std::string{ ">>>  " } + format_prefix(n.depth, n.name);
        row["depth"]          = n.depth;
        row["hash"]           = 0;
        row["rolling_hash"]   = 0;
        row["entry"]          = nlohmann::json::object();
        row["entry"]["laps"]  = 1;
        row["entry"]["value"] = sec;
        row["entry"]["accum_value"] = sec;
        graph.push_back(std::move(row));
    }
    for(auto cid : n.children)
        dfs_graph(nodes, cid, graph);
}

void
fill_inc_exc(const std::vector<out_node_t>& nodes, std::uint32_t id,
             std::vector<double>& inclusive, std::vector<double>& exclusive)
{
    const auto&  n   = nodes.at(id);
    const double inc = duration_sec(n);
    double       ch  = 0.0;
    for(auto cid : n.children)
    {
        fill_inc_exc(nodes, cid, inclusive, exclusive);
        ch += inclusive.at(cid);
    }
    inclusive.at(id) = inc;
    exclusive.at(id) = std::max(0.0, inc - ch);
}

void
compute_inc_exc_vectors(const std::vector<out_node_t>& nodes,
                        std::vector<double>& inclusive, std::vector<double>& exclusive)
{
    inclusive.assign(nodes.size(), 0.0);
    exclusive.assign(nodes.size(), 0.0);
    fill_inc_exc(nodes, 0, inclusive, exclusive);
}

struct flat_row_t
{
    std::string  key;
    std::int32_t depth     = 0;
    double       inclusive = 0;
    double       exclusive = 0;
};

void
collect_preorder(const std::vector<out_node_t>& nodes, std::uint32_t id,
                 const std::vector<double>& inclusive,
                 const std::vector<double>& exclusive, std::vector<flat_row_t>& out)
{
    if(id != 0)
    {
        const auto& n = nodes.at(id);
        out.push_back({ timemory_label_cell(n.thread_id, n.depth, n.name), n.depth,
                        inclusive.at(id), exclusive.at(id) });
    }
    for(auto cid : nodes.at(id).children)
        collect_preorder(nodes, cid, inclusive, exclusive, out);
}

struct agg_t
{
    std::int32_t depth    = 0;
    std::int64_t count    = 0;
    double       sum      = 0;
    double       minv     = 0;
    double       maxv     = 0;
    double       sumsq    = 0;
    double       sum_excl = 0;
};

std::string
center_text(const std::string& s, size_t w)
{
    if(s.size() >= w) return s.substr(0, w);
    const size_t pad_l = (w - s.size()) / 2;
    const size_t pad_r = w - pad_l - s.size();
    return std::string(pad_l, ' ') + s + std::string(pad_r, ' ');
}

template <typename T>
std::string
right_field(const T& v, size_t w)
{
    std::ostringstream o;
    o << std::right << std::setw(static_cast<int>(w)) << v;
    return o.str();
}

std::string
left_field(std::string s, size_t w)
{
    if(s.size() > w)
        s.resize(w);
    else if(s.size() < w)
        s.append(w - s.size(), ' ');
    return s;
}

std::string
right_fixed(double v, size_t w, int prec)
{
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << std::right
      << std::setw(static_cast<int>(w)) << v;
    return o.str();
}

void
write_text(const std::vector<out_node_t>& nodes, std::ofstream& os)
{
    constexpr int inner_w = 149;
    std::string   border  = "|";
    border.append(static_cast<size_t>(inner_w), '-');
    border += "|\n";

    os << border;
    {
        const char* title = "REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)";
        const int   tl    = static_cast<int>(std::strlen(title));
        const int   padl  = std::max(0, (inner_w - tl) / 2);
        const int   padr  = std::max(0, inner_w - padl - tl);
        os << '|' << std::string(static_cast<size_t>(padl), ' ') << title
           << std::string(static_cast<size_t>(padr), ' ') << "|\n";
    }
    os << border;

    constexpr int wl = 36;
    os << '|' << center_text("LABEL", static_cast<size_t>(wl)) << '|'
       << center_text("COUNT", 8) << '|' << center_text("DEPTH", 8) << '|'
       << center_text("METRIC", 12) << '|' << center_text("UNITS", 8) << '|'
       << center_text("SUM", 10) << '|' << center_text("MEAN", 10) << '|'
       << center_text("MIN", 10) << '|' << center_text("MAX", 10) << '|'
       << center_text("VAR", 10) << '|' << center_text("STDDEV", 10) << '|'
       << center_text("% SELF", 8) << "|\n";

    os << "|" << std::string(36, '-') << "|" << std::string(8, '-') << "|"
       << std::string(8, '-') << "|" << std::string(12, '-') << "|" << std::string(8, '-')
       << "|" << std::string(10, '-') << "|" << std::string(10, '-') << "|"
       << std::string(10, '-') << "|" << std::string(10, '-') << "|"
       << std::string(10, '-') << "|" << std::string(10, '-') << "|"
       << std::string(8, '-') << "|\n";

    std::vector<double> inclusive;
    std::vector<double> exclusive;
    compute_inc_exc_vectors(nodes, inclusive, exclusive);

    std::vector<flat_row_t> flat;
    collect_preorder(nodes, 0, inclusive, exclusive, flat);

    std::unordered_map<std::string, agg_t> agg;
    std::vector<std::string>               order;
    order.reserve(flat.size());
    for(const auto& r : flat)
    {
        auto& a = agg[r.key];
        if(a.count == 0)
        {
            order.push_back(r.key);
            a.depth = r.depth;
            a.minv = a.maxv = r.inclusive;
        }
        else
        {
            a.minv = std::min(a.minv, r.inclusive);
            a.maxv = std::max(a.maxv, r.inclusive);
        }
        ++a.count;
        a.sum += r.inclusive;
        a.sumsq += r.inclusive * r.inclusive;
        a.sum_excl += r.exclusive;
    }

    for(const auto& key : order)
    {
        const auto&  a     = agg.at(key);
        const double mean  = a.sum / static_cast<double>(a.count);
        const double mean2 = mean * mean;
        const double var_p =
            (a.count > 0) ? (a.sumsq / static_cast<double>(a.count) - mean2) : 0.0;
        const double var    = (var_p > 0) ? var_p : 0.0;
        const double stddev = std::sqrt(var);
        const double pct    = (a.sum > 1e-30) ? (100.0 * a.sum_excl / a.sum) : 0.0;

        std::string lbl = key;
        if(static_cast<int>(lbl.size()) > wl)
            lbl = lbl.substr(0, static_cast<size_t>(wl - 3)) + "...";

        os << '|' << left_field(lbl, static_cast<size_t>(wl)) << '|'
           << right_field(a.count, 8) << '|' << right_field(a.depth, 8) << '|'
           << center_text("wall_clock", 12) << '|' << center_text("sec", 8) << '|'
           << right_fixed(a.sum, 10, 6) << '|' << right_fixed(mean, 10, 6) << '|'
           << right_fixed(a.minv, 10, 6) << '|' << right_fixed(a.maxv, 10, 6) << '|'
           << right_fixed(var, 10, 6) << '|' << right_fixed(stddev, 10, 6) << '|'
           << right_fixed(pct, 8, 1) << "|\n";
    }

    os << border;
}

bool
replay_to_nodes(const std::vector<wall_clock_event_sample>& ev,
                std::vector<out_node_t>&                    out_nodes)
{
    if(ev.empty()) return false;

    std::vector<wall_clock_event_sample> sorted = ev;
    std::sort(sorted.begin(), sorted.end(),
              [](const wall_clock_event_sample& a, const wall_clock_event_sample& b) {
                  if(a.steady_ns != b.steady_ns) return a.steady_ns < b.steady_ns;
                  if(a.thread_id != b.thread_id) return a.thread_id < b.thread_id;
                  if(a.exec_id != b.exec_id) return a.exec_id < b.exec_id;
                  return a.event_kind < b.event_kind;
              });

    std::uint64_t max_wall = 0;
    for(const auto& s : sorted)
        max_wall = std::max(max_wall, s.wall_ns);

    std::unordered_map<std::uint64_t, slot_t> slots;
    for(const auto& s : sorted)
    {
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

    std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> by_parent;
    for(const auto& kv : slots)
    {
        by_parent[kv.second.parent_exec_id].push_back(kv.first);
    }

    for(auto& kv : by_parent)
    {
        auto& ch = kv.second;
        // Sibling order: first ENTER time (steady clock), not inclusive duration.
        // Sorting by duration reorders unrelated regions under the same parent (e.g.
        // _pthread_barrier_wait before _launch_child) and still does not match timemory's
        // graph sibling order for pthread_create → start_thread.
        std::sort(ch.begin(), ch.end(), [&](std::uint64_t a, std::uint64_t b) {
            const auto& sa = slots.at(a);
            const auto& sb = slots.at(b);
            if(sa.enter_steady_ns != sb.enter_steady_ns)
                return sa.enter_steady_ns < sb.enter_steady_ns;
            if(sa.enter_wall_ns != sb.enter_wall_ns)
                return sa.enter_wall_ns < sb.enter_wall_ns;
            if(sa.thread_id != sb.thread_id) return sa.thread_id < sb.thread_id;
            return sa.exec_id < sb.exec_id;
        });
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
        n.id        = nid;
        n.parent    = parent_out;
        n.depth     = out_nodes.at(parent_out).depth + 1;
        n.thread_id = static_cast<std::int64_t>(sl.thread_id);
        n.name      = sl.name;
        n.start_ns  = sl.enter_wall_ns;
        n.end_ns    = sl.exit_wall_ns;
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

    nlohmann::json root;
    root["timemory"][output_metric]["properties"] = nlohmann::json::object();
    root["timemory"][output_metric]["type"]       = output_metric;
    root["timemory"][output_metric]["description"] =
        "Wall-clock tree reconstructed offline from trace-cache scope events "
        "(wall_clock_event)";
    root["timemory"][output_metric]["unit_value"]        = 1.e9;
    root["timemory"][output_metric]["unit_repr"]         = "sec";
    root["timemory"][output_metric]["thread_scope_only"] = false;
    root["timemory"][output_metric]["thread_count"] = tim::manager::get_thread_count();
    root["timemory"][output_metric]["mpi_size"]     = dmp::size();
    root["timemory"][output_metric]["ranks"]        = nlohmann::json::array();
    nlohmann::json rank_entry;
    rank_entry["graph"] = nlohmann::json::array();
    dfs_graph(nodes, 0, rank_entry["graph"]);
    root["timemory"][output_metric]["ranks"].push_back(std::move(rank_entry));

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
