// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// timemory_processor_t — trace_cache processor that consumes backtrace_region_sample
// records and emits timemory-compatible JSON + TXT report files:
//   sampling_wall_clock.json/txt, sampling_percent.json/txt, trip_count.json/txt
//
// Builds a proper call tree from per-frame records, matching baseline timemory behavior:
// - Frames grouped into samples by {thread_id, beg_ns, end_ns}
// - Tree nodes merged by name hash at each depth level
// - Wall-clock interval applied to every frame in the stack
// - sampling_percent: flat profile, per-sample deduplication

#include "core/trace_cache/sample_type.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocprofsys
{
namespace trace_cache
{

class timemory_processor_t
{
    struct stats
    {
        uint64_t count  = 0;
        double   sum    = 0.0;
        double   sum_sq = 0.0;
        double   min_v  = 0.0;
        double   max_v  = 0.0;

        void accumulate(double v)
        {
            count++;
            sum += v;
            sum_sq += v * v;
            if(count == 1)
            {
                min_v = v;
                max_v = v;
            }
            else
            {
                if(v < min_v) min_v = v;
                if(v > max_v) max_v = v;
            }
        }
    };

    struct tree_node
    {
        std::string name;
        uint64_t    name_hash    = 0;
        uint64_t    rolling_hash = 0;
        int         depth        = 0;

        stats    wall;
        stats    cpu;
        uint64_t trip_count = 0;

        std::vector<std::unique_ptr<tree_node>> children;

        tree_node* find_or_create_child(const std::string& child_name,
                                        uint64_t child_hash, int child_depth)
        {
            for(auto& c : children)
            {
                if(c->name_hash == child_hash) return c.get();
            }
            auto child          = std::make_unique<tree_node>();
            child->name         = child_name;
            child->name_hash    = child_hash;
            child->rolling_hash = rolling_hash + child_hash;
            child->depth        = child_depth;
            auto* ptr           = child.get();
            children.push_back(std::move(child));
            return ptr;
        }
    };

    struct buffered_frame
    {
        uint64_t    thread_id;
        std::string name;
        uint64_t    beg_ns;
        uint64_t    end_ns;
        int         depth;
        int64_t     cpu_ns;
        std::string category;
        std::string track_name;
    };

    struct sample_key
    {
        uint64_t thread_id;
        uint64_t beg_ns;
        uint64_t end_ns;
        bool     operator<(const sample_key& o) const noexcept
        {
            if(thread_id != o.thread_id) return thread_id < o.thread_id;
            if(beg_ns != o.beg_ns) return beg_ns < o.beg_ns;
            return end_ns < o.end_ns;
        }
    };

    struct pct_entry
    {
        uint64_t count = 0;
        double   pct   = 0.0;
    };

    struct flat_node
    {
        uint64_t    tid;
        std::string name;
        uint64_t    name_hash;
        uint64_t    rolling_hash;
        int         depth;
        stats       wall;
        stats       cpu;
        uint64_t    trip_count;
        double      self_pct;
    };

public:
    explicit timemory_processor_t(std::string output_dir)
    : m_output_dir(std::move(output_dir))
    {}

    void prepare_for_processing() {}

    void finalize_processing()
    {
        if(m_output_dir.empty()) return;

        build_thread_seq_map_();
        group_and_process_();

        auto all_nodes = flatten_all_trees_();

        if(!all_nodes.empty())
        {
            emit_metric_files_("sampling_wall_clock", "Wall clock time (via sampling)",
                               1000000000, "sec", all_nodes, false);

            bool has_cpu =
                std::any_of(all_nodes.begin(), all_nodes.end(),
                            [](const flat_node& n) { return n.cpu.count > 0; });
            if(has_cpu)
            {
                emit_metric_files_("sampling_cpu_clock", "CPU clock time (via sampling)",
                                   1000000000, "sec", all_nodes, true);
            }

            emit_trip_count_files_(all_nodes);
        }

        if(!m_pct_agg.empty())
        {
            emit_percent_files_();
        }
    }

    void handle(const backtrace_region_sample& sample)
    {
        int     depth  = 0;
        int64_t cpu_ns = -1;
        try
        {
            auto ext = nlohmann::json::parse(sample.extdata);
            if(!ext.contains("depth")) return;
            depth = ext["depth"].get<int>();
            if(ext.contains("cpu_ns")) cpu_ns = ext["cpu_ns"].get<int64_t>();
        } catch(...)
        {
            return;
        }

        m_thread_ids.insert(sample.thread_id);
        if(m_tid_to_seq.find(sample.thread_id) == m_tid_to_seq.end())
            parse_seq_from_track_(sample.track_name, sample.thread_id);

        if(sample.category != "timer_sampling" && sample.category != "overflow_sampling")
            return;

        m_frames.push_back(buffered_frame{
            sample.thread_id, sample.name, sample.start_timestamp, sample.end_timestamp,
            depth, cpu_ns, sample.category, sample.track_name });
    }

    void handle(const kernel_dispatch_sample&) noexcept {}
    void handle(const scratch_memory_sample&) noexcept {}
    void handle(const memory_copy_sample&) noexcept {}
    void handle(const memory_allocate_sample&) noexcept {}
    void handle(const region_sample&) noexcept {}
    void handle(const in_time_sample&) noexcept {}
    void handle(const pmc_event_with_sample&) noexcept {}
    void handle(const kfd_sample&) noexcept {}

private:
    // ── Grouping + tree building ─────────────────────────────────────────────

    void group_and_process_()
    {
        std::map<sample_key, std::vector<buffered_frame*>> samples;
        for(auto& f : m_frames)
            samples[sample_key{ f.thread_id, f.beg_ns, f.end_ns }].push_back(&f);

        std::map<uint64_t, uint64_t> thread_frame_counts;
        for(auto& [sk, frames] : samples)
            thread_frame_counts[sk.thread_id] += frames.size();

        for(auto& [sk, frames] : samples)
        {
            std::sort(frames.begin(), frames.end(),
                      [](const buffered_frame* a, const buffered_frame* b) {
                          return a->depth < b->depth;
                      });

            double dur_s = static_cast<double>(sk.end_ns - sk.beg_ns) * 1.0e-9;

            auto& root = m_trees[sk.thread_id];
            if(!root)
            {
                root        = std::make_unique<tree_node>();
                root->depth = -1;
            }

            tree_node* cur = root.get();
            for(auto* fp : frames)
            {
                uint64_t   h     = hash_label_(fp->name);
                tree_node* child = cur->find_or_create_child(fp->name, h, fp->depth);

                child->wall.accumulate(dur_s);
                child->trip_count++;

                if(fp->cpu_ns >= 0)
                {
                    double cpu_s = static_cast<double>(fp->cpu_ns) * 1.0e-9;
                    child->cpu.accumulate(cpu_s);
                }

                cur = child;
            }

            uint64_t total_frames = thread_frame_counts[sk.thread_id];
            double   per_frame    = (total_frames > 0)
                                        ? (1.0 / static_cast<double>(total_frames)) * 100.0
                                        : 0.0;

            for(auto* fp : frames)
            {
                auto& entry = m_pct_agg[std::make_pair(sk.thread_id, fp->name)];
                entry.count++;
                entry.pct += per_frame;
            }
        }
    }

    // ── Tree traversal ───────────────────────────────────────────────────────

    static double compute_self_pct_(const tree_node& node)
    {
        if(node.wall.sum <= 0.0) return 100.0;
        double child_sum = 0.0;
        for(auto& c : node.children)
            child_sum += c->wall.sum;
        double ratio = std::min(child_sum / node.wall.sum, 1.0);
        return std::max((1.0 - ratio) * 100.0, 0.0);
    }

    void flatten_tree_(uint64_t tid, const tree_node& node,
                       std::vector<flat_node>& out) const
    {
        for(auto& child : node.children)
        {
            out.push_back(flat_node{
                tid, child->name, child->name_hash, child->rolling_hash, child->depth,
                child->wall, child->cpu, child->trip_count, compute_self_pct_(*child) });
            flatten_tree_(tid, *child, out);
        }
    }

    std::vector<flat_node> flatten_all_trees_() const
    {
        std::vector<flat_node> result;
        for(auto& [tid, root] : m_trees)
            flatten_tree_(tid, *root, result);
        return result;
    }

    // ── Thread seq mapping ───────────────────────────────────────────────────

    void parse_seq_from_track_(const std::string& track_name, uint64_t tid)
    {
        auto pos = track_name.find("Thread ");
        if(pos != std::string::npos)
        {
            pos += 7;
            auto end = track_name.find(' ', pos);
            if(end != std::string::npos)
            {
                try
                {
                    uint64_t seq      = std::stoull(track_name.substr(pos, end - pos));
                    m_tid_to_seq[tid] = seq;
                    return;
                } catch(...)
                {}
            }
        }
        m_tid_to_seq[tid] = m_next_seq++;
    }

    void build_thread_seq_map_()
    {
        for(auto tid : m_thread_ids)
        {
            if(m_tid_to_seq.find(tid) == m_tid_to_seq.end())
                m_tid_to_seq[tid] = m_next_seq++;
        }
    }

    uint64_t seq_for_(uint64_t tid) const
    {
        auto it = m_tid_to_seq.find(tid);
        return it != m_tid_to_seq.end() ? it->second : tid;
    }

    // ── Formatting ───────────────────────────────────────────────────────────

    std::string make_prefix_(uint64_t tid, int depth, const std::string& label) const
    {
        std::string pfx = "|" + std::to_string(seq_for_(tid)) + ">>> ";
        if(depth == 0)
        {
            pfx += label;
        }
        else
        {
            for(int i = 0; i < depth - 1; ++i)
                pfx += "  ";
            pfx += "|_" + label;
        }
        return pfx;
    }

    static uint64_t hash_label_(const std::string& label)
    {
        return std::hash<std::string>{}(label);
    }

    std::string make_filepath_(const std::string& name, const std::string& ext) const
    {
        return m_output_dir + "/" + name + ext;
    }

    static double variance_(uint64_t n, double sum, double sum_sq)
    {
        if(n < 2) return 0.0;
        double v =
            (sum_sq - (sum * sum) / static_cast<double>(n)) / static_cast<double>(n - 1);
        return v < 0.0 ? 0.0 : v;
    }

    static std::string fmt_(double v, int prec = 6)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(prec) << v;
        return ss.str();
    }

    // ── JSON + TXT emission ──────────────────────────────────────────────────

    void emit_metric_files_(const std::string& metric_name,
                            const std::string& description, uint64_t unit_value,
                            const std::string&            unit_repr,
                            const std::vector<flat_node>& all_nodes, bool is_cpu)
    {
        auto graph_arr = nlohmann::json::array();

        for(auto& n : all_nodes)
        {
            const stats& st = is_cpu ? n.cpu : n.wall;
            if(st.count == 0) continue;
            double mean   = st.sum / static_cast<double>(st.count);
            double stddev = std::sqrt(variance_(st.count, st.sum, st.sum_sq));

            std::string prefix = make_prefix_(n.tid, n.depth, n.name);

            nlohmann::json entry;
            entry["cereal_class_version"] = 0;
            entry["laps"]                 = st.count;
            entry["value"]                = st.sum;
            entry["repr_data"]            = st.sum;
            entry["repr_display"]         = st.sum;

            nlohmann::json stats_obj;
            stats_obj["cereal_class_version"] = 0;
            stats_obj["sum"]                  = st.sum;
            stats_obj["count"]                = st.count;
            stats_obj["min"]                  = st.min_v;
            stats_obj["max"]                  = st.max_v;
            stats_obj["sqr"]                  = st.sum_sq;
            stats_obj["mean"]                 = mean;
            stats_obj["stddev"]               = stddev;

            nlohmann::json node_j;
            node_j["hash"]         = n.name_hash;
            node_j["prefix"]       = prefix;
            node_j["depth"]        = n.depth;
            node_j["entry"]        = entry;
            node_j["stats"]        = stats_obj;
            node_j["rolling_hash"] = n.rolling_hash;
            graph_arr.push_back(std::move(node_j));
        }

        nlohmann::json rank_obj;
        rank_obj["rank"]       = 0;
        rank_obj["graph_size"] = graph_arr.size();
        rank_obj["graph"]      = std::move(graph_arr);

        uint64_t       thread_count = m_thread_ids.size();
        nlohmann::json metric;
        metric["properties"]        = { { "cereal_class_version", 0 } };
        metric["type"]              = metric_name;
        metric["description"]       = description;
        metric["unit_value"]        = unit_value;
        metric["unit_repr"]         = unit_repr;
        metric["thread_scope_only"] = false;
        metric["thread_count"]      = thread_count;
        metric["mpi_size"]          = 1;
        metric["upcxx_size"]        = 1;
        metric["process_count"]     = 1;
        metric["num_ranks"]         = 1;
        metric["concurrency"]       = thread_count;
        metric["ranks"]             = nlohmann::json::array({ rank_obj });

        nlohmann::json root;
        root["timemory"]              = nlohmann::json::object();
        root["timemory"][metric_name] = std::move(metric);

        {
            std::ofstream ofs(make_filepath_(metric_name, ".json"));
            if(ofs.is_open()) ofs << root.dump();
        }

        {
            std::ofstream ofs(make_filepath_(metric_name, ".txt"));
            if(ofs.is_open())
                emit_metric_txt_(ofs, metric_name, description, unit_repr, all_nodes,
                                 is_cpu);
        }
    }

    void emit_metric_txt_(std::ostream& out, const std::string& metric_name,
                          const std::string& title, const std::string& unit,
                          const std::vector<flat_node>& all_nodes, bool is_cpu) const
    {
        std::size_t max_label = 5;
        for(auto& n : all_nodes)
        {
            const stats& st = is_cpu ? n.cpu : n.wall;
            if(st.count == 0) continue;
            auto pfx  = make_prefix_(n.tid, n.depth, n.name);
            max_label = std::max(max_label, pfx.size());
        }
        max_label += 2;

        auto sep = [&](std::ostream& o) {
            o << "|" << std::string(max_label + 120, '-') << "|\n";
        };

        sep(out);
        {
            std::string upper_title;
            for(char c : title)
                upper_title += static_cast<char>(std::toupper(c));
            auto total_w = max_label + 120;
            auto pad     = (total_w > upper_title.size() + 2)
                               ? (total_w - upper_title.size()) / 2
                               : 1;
            out << "|" << std::string(pad, ' ') << upper_title
                << std::string(total_w - pad - upper_title.size(), ' ') << "|\n";
        }
        sep(out);

        out << "| " << std::left << std::setw(static_cast<int>(max_label)) << "LABEL"
            << " | " << std::right << std::setw(6) << "COUNT"
            << " | " << std::setw(6) << "DEPTH"
            << " | " << std::left << std::setw(19) << "METRIC"
            << " | " << std::left << std::setw(6) << "UNITS"
            << " | " << std::right << std::setw(8) << "SUM"
            << " | " << std::setw(8) << "MEAN"
            << " | " << std::setw(8) << "MIN"
            << " | " << std::setw(8) << "MAX"
            << " | " << std::setw(8) << "VAR"
            << " | " << std::setw(8) << "STDDEV"
            << " | " << std::setw(6) << "% SELF"
            << " |\n";

        sep(out);

        for(auto& n : all_nodes)
        {
            const stats& st = is_cpu ? n.cpu : n.wall;
            if(st.count == 0) continue;
            auto   prefix = make_prefix_(n.tid, n.depth, n.name);
            double mean   = st.sum / static_cast<double>(st.count);
            double var    = variance_(st.count, st.sum, st.sum_sq);
            double stddev = std::sqrt(var);

            out << "| " << std::left << std::setw(static_cast<int>(max_label)) << prefix
                << " | " << std::right << std::setw(6) << st.count << " | "
                << std::setw(6) << n.depth << " | " << std::left << std::setw(19)
                << metric_name << " | " << std::left << std::setw(6) << unit << " | "
                << std::right << std::setw(8) << fmt_(st.sum) << " | " << std::setw(8)
                << fmt_(mean) << " | " << std::setw(8) << fmt_(st.min_v) << " | "
                << std::setw(8) << fmt_(st.max_v) << " | " << std::setw(8) << fmt_(var)
                << " | " << std::setw(8) << fmt_(stddev) << " | " << std::setw(6)
                << fmt_(n.self_pct, 1) << " |\n";
        }

        sep(out);
    }

    // ── Trip count emission ──────────────────────────────────────────────────

    void emit_trip_count_files_(const std::vector<flat_node>& all_nodes)
    {
        auto graph_arr = nlohmann::json::array();
        for(auto& n : all_nodes)
        {
            std::string prefix = make_prefix_(n.tid, n.depth, n.name);

            nlohmann::json entry;
            entry["cereal_class_version"] = 0;
            entry["laps"]                 = n.trip_count;
            entry["value"]                = static_cast<int64_t>(n.trip_count);
            entry["repr_data"]            = static_cast<int64_t>(n.trip_count);
            entry["repr_display"]         = static_cast<int64_t>(n.trip_count);

            nlohmann::json node_j;
            node_j["hash"]         = n.name_hash;
            node_j["prefix"]       = prefix;
            node_j["depth"]        = n.depth;
            node_j["entry"]        = entry;
            node_j["stats"]        = nlohmann::json::object();
            node_j["rolling_hash"] = n.rolling_hash;
            graph_arr.push_back(std::move(node_j));
        }

        nlohmann::json rank_obj;
        rank_obj["rank"]       = 0;
        rank_obj["graph_size"] = graph_arr.size();
        rank_obj["graph"]      = std::move(graph_arr);

        uint64_t       thread_count = m_thread_ids.size();
        nlohmann::json metric;
        metric["properties"]        = { { "cereal_class_version", 0 } };
        metric["type"]              = "trip_count";
        metric["description"]       = "Counts number of invocations";
        metric["unit_value"]        = 1;
        metric["unit_repr"]         = "";
        metric["thread_scope_only"] = false;
        metric["thread_count"]      = thread_count;
        metric["mpi_size"]          = 1;
        metric["upcxx_size"]        = 1;
        metric["process_count"]     = 1;
        metric["num_ranks"]         = 1;
        metric["concurrency"]       = thread_count;
        metric["ranks"]             = nlohmann::json::array({ rank_obj });

        nlohmann::json root;
        root["timemory"]               = nlohmann::json::object();
        root["timemory"]["trip_count"] = std::move(metric);

        {
            std::ofstream ofs(make_filepath_("trip_count", ".json"));
            if(ofs.is_open()) ofs << root.dump();
        }

        {
            std::ofstream ofs(make_filepath_("trip_count", ".txt"));
            if(ofs.is_open()) emit_trip_count_txt_(ofs, all_nodes);
        }
    }

    void emit_trip_count_txt_(std::ostream&                 out,
                              const std::vector<flat_node>& all_nodes) const
    {
        std::size_t max_label = 5;
        for(auto& n : all_nodes)
        {
            auto pfx  = make_prefix_(n.tid, n.depth, n.name);
            max_label = std::max(max_label, pfx.size());
        }
        max_label += 2;

        auto sep = [&](std::ostream& o) {
            o << "|" << std::string(max_label + 50, '-') << "|\n";
        };

        sep(out);
        {
            std::string title   = "COUNTS NUMBER OF INVOCATIONS";
            auto        total_w = max_label + 50;
            auto pad = (total_w > title.size() + 2) ? (total_w - title.size()) / 2 : 1;
            out << "|" << std::string(pad, ' ') << title
                << std::string(total_w - pad - title.size(), ' ') << "|\n";
        }
        sep(out);

        out << "| " << std::left << std::setw(static_cast<int>(max_label)) << "LABEL"
            << " | " << std::right << std::setw(6) << "COUNT"
            << " | " << std::setw(6) << "DEPTH"
            << " | " << std::left << std::setw(10) << "METRIC"
            << " | " << std::right << std::setw(6) << "SUM"
            << " |\n";
        sep(out);

        for(auto& n : all_nodes)
        {
            auto prefix = make_prefix_(n.tid, n.depth, n.name);
            out << "| " << std::left << std::setw(static_cast<int>(max_label)) << prefix
                << " | " << std::right << std::setw(6) << n.trip_count << " | "
                << std::setw(6) << n.depth << " | " << std::left << std::setw(10)
                << "trip_count"
                << " | " << std::right << std::setw(6) << n.trip_count << " |\n";
        }
        sep(out);
    }

    // ── Percent emission ─────────────────────────────────────────────────────

    void emit_percent_files_()
    {
        std::map<uint64_t, double> thread_totals;
        for(auto& [key, entry] : m_pct_agg)
            thread_totals[key.first] += entry.pct;

        auto graph_arr = nlohmann::json::array();

        for(auto& [key, entry] : m_pct_agg)
        {
            uint64_t           tid   = key.first;
            const std::string& name  = key.second;
            double             total = thread_totals[tid];
            double             pct   = (total > 0.0) ? (entry.pct / total) * 100.0 : 0.0;

            uint64_t    h      = hash_label_(name);
            std::string prefix = make_prefix_(tid, 0, name);

            nlohmann::json entry_j;
            entry_j["cereal_class_version"] = 0;
            entry_j["laps"]                 = entry.count;
            entry_j["value"]                = pct;
            entry_j["repr_data"]            = pct;
            entry_j["repr_display"]         = pct;

            nlohmann::json node_j;
            node_j["hash"]         = h;
            node_j["prefix"]       = prefix;
            node_j["depth"]        = 0;
            node_j["entry"]        = entry_j;
            node_j["stats"]        = nlohmann::json::object();
            node_j["rolling_hash"] = h;
            graph_arr.push_back(std::move(node_j));
        }

        nlohmann::json rank_obj;
        rank_obj["rank"]       = 0;
        rank_obj["graph_size"] = graph_arr.size();
        rank_obj["graph"]      = std::move(graph_arr);

        uint64_t       thread_count = m_thread_ids.size();
        nlohmann::json metric;
        metric["properties"]        = { { "cereal_class_version", 0 } };
        metric["type"]              = "sampling_percent";
        metric["description"]       = "Percentage of samples";
        metric["unit_value"]        = 1;
        metric["unit_repr"]         = "%";
        metric["thread_scope_only"] = false;
        metric["thread_count"]      = thread_count;
        metric["mpi_size"]          = 1;
        metric["upcxx_size"]        = 1;
        metric["process_count"]     = 1;
        metric["num_ranks"]         = 1;
        metric["concurrency"]       = thread_count;
        metric["ranks"]             = nlohmann::json::array({ rank_obj });

        nlohmann::json root;
        root["timemory"]                     = nlohmann::json::object();
        root["timemory"]["sampling_percent"] = std::move(metric);

        {
            std::ofstream ofs(make_filepath_("sampling_percent", ".json"));
            if(ofs.is_open()) ofs << root.dump();
        }

        {
            std::ofstream ofs(make_filepath_("sampling_percent", ".txt"));
            if(ofs.is_open()) emit_percent_txt_(ofs);
        }
    }

    void emit_percent_txt_(std::ostream& out) const
    {
        std::map<uint64_t, double> thread_totals;
        for(auto& [key, entry] : m_pct_agg)
            thread_totals[key.first] += entry.pct;

        std::size_t max_label = 5;
        for(auto& [key, entry] : m_pct_agg)
        {
            auto pfx  = make_prefix_(key.first, 0, key.second);
            max_label = std::max(max_label, pfx.size());
        }
        max_label += 2;

        auto sep = [&](std::ostream& o) {
            o << "|" << std::string(max_label + 60, '-') << "|\n";
        };

        sep(out);
        {
            std::string title   = "PERCENTAGE OF SAMPLES";
            auto        total_w = max_label + 60;
            auto pad = (total_w > title.size() + 2) ? (total_w - title.size()) / 2 : 1;
            out << "|" << std::string(pad, ' ') << title
                << std::string(total_w - pad - title.size(), ' ') << "|\n";
        }
        sep(out);

        out << "| " << std::left << std::setw(static_cast<int>(max_label)) << "LABEL"
            << " | " << std::right << std::setw(5) << "COUNT"
            << " | " << std::setw(5) << "DEPTH"
            << " | " << std::left << std::setw(16) << "METRIC"
            << " | " << std::left << std::setw(5) << "UNITS"
            << " | " << std::right << std::setw(6) << "SUM"
            << " |\n";
        sep(out);

        for(auto& [key, entry] : m_pct_agg)
        {
            double total = thread_totals[key.first];
            double pct   = (total > 0.0) ? (entry.pct / total) * 100.0 : 0.0;

            auto prefix = make_prefix_(key.first, 0, key.second);
            out << "| " << std::left << std::setw(static_cast<int>(max_label)) << prefix
                << " | " << std::right << std::setw(5) << entry.count << " | "
                << std::setw(5) << 0 << " | " << std::left << std::setw(16)
                << "sampling_percent"
                << " | " << std::left << std::setw(5) << "%"
                << " | " << std::right << std::setw(6) << fmt_(pct, 3) << " |\n";
        }
        sep(out);
    }

    // ── Members ──────────────────────────────────────────────────────────────

    std::vector<buffered_frame> m_frames;

    std::map<uint64_t, std::unique_ptr<tree_node>> m_trees;

    std::map<std::pair<uint64_t, std::string>, pct_entry> m_pct_agg;

    std::set<uint64_t>           m_thread_ids;
    std::map<uint64_t, uint64_t> m_tid_to_seq;
    uint64_t                     m_next_seq = 0;

    std::string m_output_dir;
};

}  // namespace trace_cache
}  // namespace rocprofsys
