// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// timemory_processor_t — trace_cache processor that consumes backtrace_region_sample
// records and emits timemory-compatible JSON + TXT report files:
//   sampling_wall_clock.json/txt, sampling_percent.json/txt, trip_count.json/txt
//
// Replaces tsv_processor_t. Same aggregation logic, different output format.
// CRTP-free for testability (see timemory_processor_adapter.hpp for production).

#include "core/trace_cache/sample_type.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
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
    struct row_key
    {
        uint64_t    thread_id;
        int         depth;
        std::string label;
        bool        operator<(const row_key& o) const noexcept
        {
            if(thread_id != o.thread_id) return thread_id < o.thread_id;
            if(depth != o.depth) return depth < o.depth;
            return label < o.label;
        }
    };

    struct stats
    {
        uint64_t count  = 0;
        double   sum    = 0.0;
        double   sum_sq = 0.0;
        double   min_v  = 0.0;
        double   max_v  = 0.0;
    };

    struct pct_key
    {
        uint64_t    thread_id;
        std::string label;
        bool        operator<(const pct_key& o) const noexcept
        {
            if(thread_id != o.thread_id) return thread_id < o.thread_id;
            return label < o.label;
        }
    };

    struct pct_stats
    {
        uint64_t count = 0;
        double   sum   = 0.0;
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

        if(!m_wall_agg.empty())
        {
            emit_metric_files_("sampling_wall_clock", "Wall clock time (via sampling)",
                               1000000000, "sec", m_wall_agg, true);
        }

        if(!m_trip_agg.empty())
        {
            emit_trip_count_files_();
        }

        if(!m_pct_agg.empty())
        {
            emit_percent_files_();
        }
    }

    void handle(const backtrace_region_sample& sample)
    {
        int depth = 0;
        try
        {
            auto ext = nlohmann::json::parse(sample.extdata);
            if(!ext.contains("depth")) return;
            depth = ext["depth"].get<int>();
        } catch(...)
        {
            return;
        }

        double dur_s =
            static_cast<double>(sample.end_timestamp - sample.start_timestamp) * 1.0e-9;
        row_key k{ sample.thread_id, depth, sample.name };

        if(sample.category == "timer_sampling")
        {
            accumulate_(m_wall_agg[k], dur_s);
            m_trip_agg[k]++;

            pct_key pk{ sample.thread_id, sample.name };
            m_pct_agg[pk].count++;
            m_pct_agg[pk].sum += dur_s;
        }

        m_thread_ids.insert(sample.thread_id);

        if(m_tid_to_seq.find(sample.thread_id) == m_tid_to_seq.end())
        {
            parse_seq_from_track_(sample.track_name, sample.thread_id);
        }
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
    static void accumulate_(stats& st, double v)
    {
        st.count++;
        st.sum += v;
        st.sum_sq += v * v;
        if(st.count == 1)
        {
            st.min_v = v;
            st.max_v = v;
        }
        else
        {
            if(v < st.min_v) st.min_v = v;
            if(v > st.max_v) st.max_v = v;
        }
    }

    static double variance_(uint64_t n, double sum, double sum_sq)
    {
        if(n < 2) return 0.0;
        double v =
            (sum_sq - (sum * sum) / static_cast<double>(n)) / static_cast<double>(n - 1);
        return v < 0.0 ? 0.0 : v;
    }

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
        return m_output_dir + "/" + name + "-" + std::to_string(::getpid()) + ext;
    }

    std::map<std::pair<uint64_t, int>, double> compute_children_sum_(
        const std::map<row_key, stats>& agg) const
    {
        std::map<std::pair<uint64_t, int>, double> result;
        for(auto const& [k, v] : agg)
            result[{ k.thread_id, k.depth }] += v.sum;
        return result;
    }

    double compute_self_pct_(
        const row_key& k, double sum,
        const std::map<std::pair<uint64_t, int>, double>& csums) const
    {
        if(sum <= 0.0) return 100.0;
        auto   it    = csums.find({ k.thread_id, k.depth + 1 });
        double csum  = (it != csums.end()) ? it->second : 0.0;
        double ratio = std::min(csum / sum, 1.0);
        return std::max((1.0 - ratio) * 100.0, 0.0);
    }

    // ── JSON emission ────────────────────────────────────────────────────────

    void emit_metric_files_(const std::string& metric_name,
                            const std::string& description, uint64_t unit_value,
                            const std::string&              unit_repr,
                            const std::map<row_key, stats>& agg, bool with_stats)
    {
        auto csums     = compute_children_sum_(agg);
        auto graph_arr = nlohmann::json::array();

        std::map<uint64_t, std::vector<uint64_t>> hash_stacks;

        for(auto const& [k, v] : agg)
        {
            uint64_t h = hash_label_(k.label);

            auto& stack = hash_stacks[k.thread_id];
            while(static_cast<int>(stack.size()) > k.depth)
                stack.pop_back();

            uint64_t parent_rh    = stack.empty() ? 0 : stack.back();
            uint64_t rolling_hash = h ^ parent_rh;
            stack.push_back(rolling_hash);

            double mean   = v.sum / static_cast<double>(v.count);
            double stddev = std::sqrt(variance_(v.count, v.sum, v.sum_sq));

            std::string prefix = make_prefix_(k.thread_id, k.depth, k.label);

            nlohmann::json entry;
            entry["cereal_class_version"] = 0;
            entry["laps"]                 = v.count;
            entry["value"]                = v.sum;
            entry["repr_data"]            = v.sum;
            entry["repr_display"]         = v.sum;

            nlohmann::json stats_obj;
            if(with_stats)
            {
                stats_obj["cereal_class_version"] = 0;
                stats_obj["sum"]                  = v.sum;
                stats_obj["count"]                = v.count;
                stats_obj["min"]                  = v.min_v;
                stats_obj["max"]                  = v.max_v;
                stats_obj["sqr"]                  = v.sum_sq;
                stats_obj["mean"]                 = mean;
                stats_obj["stddev"]               = stddev;
            }

            nlohmann::json node;
            node["hash"]         = h;
            node["prefix"]       = prefix;
            node["depth"]        = k.depth;
            node["entry"]        = entry;
            node["stats"]        = with_stats ? stats_obj : nlohmann::json::object();
            node["rolling_hash"] = rolling_hash;

            graph_arr.push_back(std::move(node));
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

        auto json_path = make_filepath_(metric_name, ".json");
        {
            std::ofstream ofs(json_path);
            if(ofs.is_open()) ofs << root.dump();
        }

        auto txt_path = make_filepath_(metric_name, ".txt");
        {
            std::ofstream ofs(txt_path);
            if(ofs.is_open())
                emit_txt_table_(ofs, metric_name, description, unit_repr, agg, csums,
                                with_stats);
        }
    }

    void emit_trip_count_files_()
    {
        auto                                      graph_arr = nlohmann::json::array();
        std::map<uint64_t, std::vector<uint64_t>> hash_stacks;

        for(auto const& [k, v] : m_trip_agg)
        {
            uint64_t h     = hash_label_(k.label);
            auto&    stack = hash_stacks[k.thread_id];
            while(static_cast<int>(stack.size()) > k.depth)
                stack.pop_back();
            uint64_t parent_rh = stack.empty() ? 0 : stack.back();
            uint64_t rh        = h ^ parent_rh;
            stack.push_back(rh);

            std::string prefix = make_prefix_(k.thread_id, k.depth, k.label);

            nlohmann::json entry;
            entry["cereal_class_version"] = 0;
            entry["laps"]                 = v;
            entry["value"]                = static_cast<int64_t>(v);
            entry["repr_data"]            = static_cast<int64_t>(v);
            entry["repr_display"]         = static_cast<int64_t>(v);

            nlohmann::json node;
            node["hash"]         = h;
            node["prefix"]       = prefix;
            node["depth"]        = k.depth;
            node["entry"]        = entry;
            node["stats"]        = nlohmann::json::object();
            node["rolling_hash"] = rh;
            graph_arr.push_back(std::move(node));
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

        auto json_path = make_filepath_("trip_count", ".json");
        {
            std::ofstream ofs(json_path);
            if(ofs.is_open()) ofs << root.dump();
        }

        auto txt_path = make_filepath_("trip_count", ".txt");
        {
            std::ofstream ofs(txt_path);
            if(ofs.is_open()) emit_trip_count_txt_(ofs);
        }
    }

    void emit_percent_files_()
    {
        std::map<uint64_t, double> thread_totals;
        for(auto const& [k, v] : m_pct_agg)
            thread_totals[k.thread_id] += v.sum;

        auto graph_arr = nlohmann::json::array();

        for(auto const& [k, v] : m_pct_agg)
        {
            double total = thread_totals[k.thread_id];
            double pct   = (total > 0.0) ? (v.sum / total) * 100.0 : 0.0;

            uint64_t    h      = hash_label_(k.label);
            std::string prefix = make_prefix_(k.thread_id, 0, k.label);

            nlohmann::json entry;
            entry["cereal_class_version"] = 0;
            entry["laps"]                 = v.count;
            entry["value"]                = pct;
            entry["repr_data"]            = pct;
            entry["repr_display"]         = pct;

            nlohmann::json node;
            node["hash"]         = h;
            node["prefix"]       = prefix;
            node["depth"]        = 0;
            node["entry"]        = entry;
            node["stats"]        = nlohmann::json::object();
            node["rolling_hash"] = h;
            graph_arr.push_back(std::move(node));
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

        auto json_path = make_filepath_("sampling_percent", ".json");
        {
            std::ofstream ofs(json_path);
            if(ofs.is_open()) ofs << root.dump();
        }

        auto txt_path = make_filepath_("sampling_percent", ".txt");
        {
            std::ofstream ofs(txt_path);
            if(ofs.is_open()) emit_percent_txt_(ofs);
        }
    }

    // ── TXT table emission ───────────────────────────────────────────────────

    static std::string fmt_(double v, int prec = 6)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(prec) << v;
        return ss.str();
    }

    void emit_txt_table_(std::ostream& out, const std::string& metric_name,
                         const std::string& title, const std::string& unit,
                         const std::map<row_key, stats>&                   agg,
                         const std::map<std::pair<uint64_t, int>, double>& csums,
                         bool with_stats) const
    {
        std::size_t max_label = 5;
        for(auto const& [k, v] : agg)
        {
            auto pfx  = make_prefix_(k.thread_id, k.depth, k.label);
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
            << " | " << std::left << std::setw(6) << "UNITS";
        if(with_stats)
        {
            out << " | " << std::right << std::setw(8) << "SUM"
                << " | " << std::setw(8) << "MEAN"
                << " | " << std::setw(8) << "MIN"
                << " | " << std::setw(8) << "MAX"
                << " | " << std::setw(8) << "VAR"
                << " | " << std::setw(8) << "STDDEV"
                << " | " << std::setw(6) << "% SELF";
        }
        else
        {
            out << " | " << std::right << std::setw(6) << "SUM";
        }
        out << " |\n";

        sep(out);

        for(auto const& [k, v] : agg)
        {
            auto prefix = make_prefix_(k.thread_id, k.depth, k.label);
            out << "| " << std::left << std::setw(static_cast<int>(max_label)) << prefix
                << " | " << std::right << std::setw(6) << v.count << " | " << std::setw(6)
                << k.depth << " | " << std::left << std::setw(19) << metric_name << " | "
                << std::left << std::setw(6) << unit;
            if(with_stats)
            {
                double mean   = v.sum / static_cast<double>(v.count);
                double var    = variance_(v.count, v.sum, v.sum_sq);
                double stddev = std::sqrt(var);
                double ps     = compute_self_pct_(k, v.sum, csums);

                out << " | " << std::right << std::setw(8) << fmt_(v.sum) << " | "
                    << std::setw(8) << fmt_(mean) << " | " << std::setw(8)
                    << fmt_(v.min_v) << " | " << std::setw(8) << fmt_(v.max_v) << " | "
                    << std::setw(8) << fmt_(var) << " | " << std::setw(8) << fmt_(stddev)
                    << " | " << std::setw(6) << fmt_(ps, 1);
            }
            else
            {
                out << " | " << std::right << std::setw(6) << fmt_(v.sum);
            }
            out << " |\n";
        }

        sep(out);
    }

    void emit_trip_count_txt_(std::ostream& out) const
    {
        std::size_t max_label = 5;
        for(auto const& [k, v] : m_trip_agg)
        {
            auto pfx  = make_prefix_(k.thread_id, k.depth, k.label);
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

        for(auto const& [k, v] : m_trip_agg)
        {
            auto prefix = make_prefix_(k.thread_id, k.depth, k.label);
            out << "| " << std::left << std::setw(static_cast<int>(max_label)) << prefix
                << " | " << std::right << std::setw(6) << v << " | " << std::setw(6)
                << k.depth << " | " << std::left << std::setw(10) << "trip_count"
                << " | " << std::right << std::setw(6) << v << " |\n";
        }
        sep(out);
    }

    void emit_percent_txt_(std::ostream& out) const
    {
        std::map<uint64_t, double> thread_totals;
        for(auto const& [k, v] : m_pct_agg)
            thread_totals[k.thread_id] += v.sum;

        std::size_t max_label = 5;
        for(auto const& [k, v] : m_pct_agg)
        {
            auto pfx  = make_prefix_(k.thread_id, 0, k.label);
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

        for(auto const& [k, v] : m_pct_agg)
        {
            double total = thread_totals[k.thread_id];
            double pct   = (total > 0.0) ? (v.sum / total) * 100.0 : 0.0;

            auto prefix = make_prefix_(k.thread_id, 0, k.label);
            out << "| " << std::left << std::setw(static_cast<int>(max_label)) << prefix
                << " | " << std::right << std::setw(5) << v.count << " | " << std::setw(5)
                << 0 << " | " << std::left << std::setw(16) << "sampling_percent"
                << " | " << std::left << std::setw(5) << "%"
                << " | " << std::right << std::setw(6) << fmt_(pct, 3) << " |\n";
        }
        sep(out);
    }

    // ── Members ──────────────────────────────────────────────────────────────

    std::map<row_key, stats>     m_wall_agg;
    std::map<pct_key, pct_stats> m_pct_agg;
    std::map<row_key, uint64_t>  m_trip_agg;

    std::set<uint64_t>           m_thread_ids;
    std::map<uint64_t, uint64_t> m_tid_to_seq;
    uint64_t                     m_next_seq = 0;

    std::string m_output_dir;
};

}  // namespace trace_cache
}  // namespace rocprofsys
