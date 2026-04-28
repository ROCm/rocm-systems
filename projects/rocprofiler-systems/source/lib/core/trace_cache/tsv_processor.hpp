// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// tsv_processor_t — trace_cache processor that consumes backtrace_region_sample records
// and emits the L4 TSV report files (sampling_wall_clock.tsv, sampling_cpu_clock.tsv,
// sampling_percent.tsv, trip_count.tsv).
//
// Extdata schema (locked — additive only, never rename existing fields):
//   { "depth": <int>, ... future fields ... }
// Records without a "depth" key in extdata are silently skipped.
//
// Two construction modes:
//   1. Stream-injection (tests): tsv_processor_t(wall_out, cpu_out, pct_out, trip_out)
//   2. File-path mode (production): tsv_processor_t(output_dir)
//      finalize_processing() opens the four files and writes.
//
// Only backtrace_region_sample is handled; all other sample types are no-ops.
// Category "timer_sampling" → wall_clock + trip_count aggregation.
// Category "overflow_sampling" → (future) overflow_count aggregation (currently no-op).
//
// NOTE: tsv_processor_t deliberately does NOT inherit processor_t<T> here to keep
// this header free of AMD-SMI and PMC collector dependencies (sample_processor.hpp
// transitively requires amd_smi/amdsmi.h). For production wiring via processor_view_t,
// use tsv_processor_adapter.hpp (included only from cache_manager.cpp).

#include "core/trace_cache/sample_type.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>

namespace rocprofsys
{
namespace trace_cache
{

class tsv_processor_t
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
    // Stream-injection constructor (test use).
    // Headers are written in prepare_for_processing().
    tsv_processor_t(std::ostream& wall_out, std::ostream& cpu_out, std::ostream& pct_out,
                    std::ostream& trip_out)
    : m_wall_ptr(&wall_out)
    , m_cpu_ptr(&cpu_out)
    , m_pct_ptr(&pct_out)
    , m_trip_ptr(&trip_out)
    {}

    // File-path constructor (production use).
    // finalize_processing() opens files under output_dir.
    explicit tsv_processor_t(std::string output_dir)
    : m_output_dir(std::move(output_dir))
    {}

    void prepare_for_processing()
    {
        // In stream-injection mode the streams are already open; write headers now.
        // In file-path mode the files are opened in finalize_processing() via
        // open_files_if_needed_(), which calls write_headers_() after opening.
        // Calling write_headers_() here in file-path mode would set
        // m_headers_written=true before any file is open, causing headers to be silently
        // skipped at finalize time.
        if(m_wall_ptr) write_headers_();
    }

    void finalize_processing()
    {
        open_files_if_needed_();
        if(!m_wall_ptr || !m_cpu_ptr || !m_pct_ptr || !m_trip_ptr) return;
        emit_stats_rows_(*m_wall_ptr, m_wall_agg);
        emit_stats_rows_(*m_cpu_ptr, m_cpu_agg);
        emit_pct_rows_(*m_pct_ptr);
        emit_trip_rows_(*m_trip_ptr);
        m_wall_ptr->flush();
        m_cpu_ptr->flush();
        m_pct_ptr->flush();
        m_trip_ptr->flush();
    }

    void handle(const backtrace_region_sample& sample)
    {
        // Parse depth from extdata — skip if absent.
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
        else if(sample.category == "cputime_sampling")
        {
            accumulate_(m_cpu_agg[k], dur_s);
        }
        // overflow_sampling: future (task #22 wires overflow path)
    }

    // All other sample types are no-ops.
    // The full set of overloads (including gpu/ainic/cpu pmc types that require AMD-SMI
    // headers) is provided in tsv_processor_adapter.hpp for production registration.
    void handle(const kernel_dispatch_sample&) noexcept {}
    void handle(const scratch_memory_sample&) noexcept {}
    void handle(const memory_copy_sample&) noexcept {}
    void handle(const memory_allocate_sample&) noexcept {}
    void handle(const region_sample&) noexcept {}
    void handle(const in_time_sample&) noexcept {}
    void handle(const pmc_event_with_sample&) noexcept {}
    void handle(const kfd_sample&) noexcept {}

private:
    void write_headers_()
    {
        if(m_headers_written) return;
        m_headers_written = true;
        if(m_wall_ptr)
            *m_wall_ptr
                << "# metric: sampling_wall_clock\n"
                << "# unit: s\n"
                << "# columns: "
                   "thread_"
                   "id\tdepth\tlabel\tcount\tsum\tmean\tmin\tmax\tvar\tstddev\tpct_"
                   "self\n";
        if(m_cpu_ptr)
            *m_cpu_ptr
                << "# metric: sampling_cpu_clock\n"
                << "# unit: s\n"
                << "# columns: "
                   "thread_"
                   "id\tdepth\tlabel\tcount\tsum\tmean\tmin\tmax\tvar\tstddev\tpct_"
                   "self\n";
        if(m_pct_ptr)
            *m_pct_ptr << "# metric: sampling_percent\n"
                       << "# unit: %\n"
                       << "# columns: thread_id\tlabel\tcount\tsum (flat_scope: "
                          "depth-independent dedup)\n";
        if(m_trip_ptr)
            *m_trip_ptr << "# metric: trip_count\n"
                        << "# unit: count\n"
                        << "# columns: thread_id\tdepth\tlabel\tcount\n";
    }

    void open_files_if_needed_()
    {
        if(m_wall_ptr) return;  // already set (stream-injection mode)
        if(m_output_dir.empty()) return;
        // Append PID so multi-process workloads (fork-example, MPI) don't
        // overwrite each other's TSVs in a shared OUTPUT_PATH.
        const auto suffix = "-" + std::to_string(::getpid()) + ".tsv";
        auto       open   = [&](std::ofstream& ofs, const char* base) {
            ofs.open(m_output_dir + "/" + base + suffix);
        };
        // Only open per-metric files that have data — avoids emitting empty
        // header-only files for trigger types that were never enabled (e.g. W3
        // overflow-only mode produces no wall_clock or cpu_clock samples).
        if(!m_wall_agg.empty()) open(m_wall_ofs, "sampling_wall_clock");
        if(!m_cpu_agg.empty()) open(m_cpu_ofs, "sampling_cpu_clock");
        if(!m_pct_agg.empty()) open(m_pct_ofs, "sampling_percent");
        if(!m_trip_agg.empty()) open(m_trip_ofs, "trip_count");
        if(m_wall_ofs.is_open()) m_wall_ptr = &m_wall_ofs;
        if(m_cpu_ofs.is_open()) m_cpu_ptr = &m_cpu_ofs;
        if(m_pct_ofs.is_open()) m_pct_ptr = &m_pct_ofs;
        if(m_trip_ofs.is_open()) m_trip_ptr = &m_trip_ofs;
        write_headers_();

        // TF-9: emit hw_counters-{PID}.tsv when PAPI events are configured.
        // The signal-handler PAPI capture path is not yet implemented in the
        // refactor, so this lists configured event names (one per row) so
        // downstream consumers know which counters were requested. Real
        // per-sample values land here once the PAPI integration completes.
        const char* papi_env = std::getenv("ROCPROFSYS_PAPI_EVENTS");
        if(papi_env != nullptr && papi_env[0] != '\0')
        {
            std::ofstream hw_ofs(m_output_dir + "/hw_counters" + suffix);
            if(hw_ofs.is_open())
            {
                hw_ofs << "# metric: hw_counters\n"
                       << "# unit: count\n"
                       << "# columns: event\n";
                std::string events{ papi_env };
                size_t      pos = 0;
                while(pos < events.size())
                {
                    auto next = events.find_first_of(",;\t ", pos);
                    auto end  = (next == std::string::npos) ? events.size() : next;
                    if(end > pos) hw_ofs << events.substr(pos, end - pos) << "\n";
                    pos = (next == std::string::npos) ? events.size() : next + 1;
                }
            }
        }
    }

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

    static std::string fmt6_(double v)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(6) << v;
        return ss.str();
    }

    void emit_stats_rows_(std::ostream& out, const std::map<row_key, stats>& agg)
    {
        std::map<std::pair<uint64_t, int>, double> children_sum;
        for(auto const& [k, v] : agg)
            children_sum[{ k.thread_id, k.depth }] += v.sum;

        for(auto const& [k, v] : agg)
        {
            double mean   = v.sum / static_cast<double>(v.count);
            double var    = variance_(v.count, v.sum, v.sum_sq);
            double stddev = std::sqrt(var);

            double ps = 100.0;
            if(v.sum > 0.0)
            {
                auto   it    = children_sum.find({ k.thread_id, k.depth + 1 });
                double csum  = (it != children_sum.end()) ? it->second : 0.0;
                double ratio = std::min(csum / v.sum, 1.0);
                ps           = std::max((1.0 - ratio) * 100.0, 0.0);
            }

            out << k.thread_id << '\t' << k.depth << '\t' << k.label << '\t' << v.count
                << '\t' << fmt6_(v.sum) << '\t' << fmt6_(mean) << '\t' << fmt6_(v.min_v)
                << '\t' << fmt6_(v.max_v) << '\t' << fmt6_(var) << '\t' << fmt6_(stddev)
                << '\t' << fmt6_(ps) << '\n';
        }
    }

    void emit_pct_rows_(std::ostream& out)
    {
        for(auto const& [k, v] : m_pct_agg)
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(3) << v.sum;
            out << k.thread_id << '\t' << k.label << '\t' << v.count << '\t' << ss.str()
                << '\n';
        }
    }

    void emit_trip_rows_(std::ostream& out)
    {
        for(auto const& [k, v] : m_trip_agg)
            out << k.thread_id << '\t' << k.depth << '\t' << k.label << '\t' << v << '\n';
    }

    std::map<row_key, stats>     m_wall_agg;
    std::map<row_key, stats>     m_cpu_agg;
    std::map<pct_key, pct_stats> m_pct_agg;
    std::map<row_key, uint64_t>  m_trip_agg;

    bool m_headers_written = false;

    std::ostream* m_wall_ptr = nullptr;
    std::ostream* m_cpu_ptr  = nullptr;
    std::ostream* m_pct_ptr  = nullptr;
    std::ostream* m_trip_ptr = nullptr;

    std::string   m_output_dir;
    std::ofstream m_wall_ofs;
    std::ofstream m_cpu_ofs;
    std::ofstream m_pct_ofs;
    std::ofstream m_trip_ofs;
};

}  // namespace trace_cache
}  // namespace rocprofsys
