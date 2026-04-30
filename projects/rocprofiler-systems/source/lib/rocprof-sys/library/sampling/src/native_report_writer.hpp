// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace rocprofsys::sampling
{

// TSV report writer — L4 format spec.
//
// Two construction modes:
//   1. Stream-injection (test): native_report_writer(wall_out, cpu_out, pct_out,
//   trip_out)
//      flush() writes to the injected ostream references.
//   2. Default (production): native_report_writer()
//      flush() calls open_output_files_() to obtain streams, then writes.
//      Production callers must set the file-path factory before flush() via
//      set_file_path_factory(). If no factory is set, flush() is a no-op
//      (safe for unit tests that default-construct and never call flush()).
//
// Wall-clock and CPU-clock files: 11 columns
//   thread_id  depth  label  count  sum  mean  min  max  var  stddev  pct_self
//
// Percent file (flat_scope — labels dedup'd across depths): 4 columns
//   thread_id  label  count  sum
//
// Trip-count file: 4 columns
//   thread_id  depth  label  count
//
// Each file has 3 leading comment lines: "# metric:", "# unit:", "# columns:".
// % SELF = (1 - sum_children_at_depth+1/self)*100, clamped [0,100].
// VAR = sample variance, Bessel-corrected (÷(n-1)); 0 when n<2; never NaN.
// Labels are written untruncated. Delimiter is a single tab.
class native_report_writer
{
    struct row_key
    {
        int64_t     tid;
        int         depth;
        std::string label;
        bool        operator<(row_key const& o) const noexcept
        {
            if(tid != o.tid) return tid < o.tid;
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
        int64_t     tid;
        std::string label;
        bool        operator<(pct_key const& o) const noexcept
        {
            if(tid != o.tid) return tid < o.tid;
            return label < o.label;
        }
    };
    struct pct_stats
    {
        uint64_t count = 0;
        double   sum   = 0.0;
    };

public:
    // Stream-injection constructor — caller provides output streams directly.
    // Headers are written immediately so they appear even with zero samples.
    native_report_writer(std::ostream& wall_clock_out, std::ostream& cpu_clock_out,
                         std::ostream& percent_out, std::ostream& trip_count_out)
    : m_wall_ptr(&wall_clock_out)
    , m_cpu_ptr(&cpu_clock_out)
    , m_pct_ptr(&percent_out)
    , m_trip_ptr(&trip_count_out)
    {
        write_headers();
    }

    // Default constructor — streams provided later via set_streams().
    // Accumulates data; flush() writes to streams set by set_streams().
    native_report_writer() = default;

    // Called by production code (e.g. sampling_service_production_hooks) to
    // inject opened ofstream handles before flush() runs.
    // Takes non-owning pointers; caller must keep streams alive until flush().
    void set_streams(std::ostream* wall, std::ostream* cpu, std::ostream* pct,
                     std::ostream* trip)
    {
        m_wall_ptr = wall;
        m_cpu_ptr  = cpu;
        m_pct_ptr  = pct;
        m_trip_ptr = trip;
        write_headers();
    }

    void write_timer_samples(int64_t tid, std::vector<timer_sample> const& samples)
    {
        if(samples.empty()) return;

        int64_t total_frames = 0;
        for(auto const& s : samples)
            total_frames += static_cast<int64_t>(s.stack.size());

        for(auto const& s : samples)
        {
            double dur_s = static_cast<double>(s.end_ns - s.beg_ns) * 1.0e-9;
            int    depth = 0;
            for(auto const& frame : s.stack)
            {
                row_key k{ tid, depth, frame.name };

                accumulate(m_wall_agg[k], dur_s);

                if(s.metrics.valid.any())
                {
                    double cpu_s = static_cast<double>(s.metrics.cpu_ns) * 1.0e-9;
                    accumulate(m_cpu_agg[k], cpu_s);
                }

                if(total_frames > 0)
                {
                    double pct_v = 100.0 / static_cast<double>(total_frames);
                    auto&  p     = m_pct_agg[pct_key{ tid, frame.name }];
                    p.count++;
                    p.sum += pct_v;
                }

                m_trip_agg[k]++;
                ++depth;
            }
        }
    }

    void write_overflow_samples(int64_t /*tid*/,
                                std::vector<overflow_sample> const& /*samples*/)
    {}

    void flush()
    {
        if(!m_wall_ptr || !m_cpu_ptr || !m_pct_ptr || !m_trip_ptr) return;
        emit_stats_rows(*m_wall_ptr, m_wall_agg);
        emit_stats_rows(*m_cpu_ptr, m_cpu_agg);
        emit_pct_rows(*m_pct_ptr, m_pct_agg);
        emit_trip_rows(*m_trip_ptr, m_trip_agg);
        m_wall_ptr->flush();
        m_cpu_ptr->flush();
        m_pct_ptr->flush();
        m_trip_ptr->flush();
    }

private:
    void write_headers()
    {
        if(m_headers_written) return;
        m_headers_written = true;
        if(m_wall_ptr)
            *m_wall_ptr << "# metric: sampling_wall_clock\n"
                        << "# unit: s\n"
                        << "# columns: "
                           "thread_"
                           "id\tdepth\tlabel\tcount\tsum\tmean\tmin\tmax\tvar\tstddev\tpc"
                           "t_self\n";
        if(m_cpu_ptr)
            *m_cpu_ptr << "# metric: sampling_cpu_clock\n"
                       << "# unit: s\n"
                       << "# columns: "
                          "thread_"
                          "id\tdepth\tlabel\tcount\tsum\tmean\tmin\tmax\tvar\tstddev\tpct"
                          "_self\n";
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

    static void accumulate(stats& st, double v)
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

    // Sample variance, Bessel-corrected (÷(n-1)); 0 when n<2; never NaN.
    static double variance(uint64_t n, double sum, double sum_sq)
    {
        if(n < 2) return 0.0;
        double var =
            (sum_sq - (sum * sum) / static_cast<double>(n)) / static_cast<double>(n - 1);
        return var < 0.0 ? 0.0 : var;
    }

    void emit_stats_rows(std::ostream& out, std::map<row_key, stats> const& agg)
    {
        // Build children_sum[{tid, depth}] = total sum of all rows at that depth.
        // Used to compute % SELF = (1 - children_sum_at_depth+1 / self_sum)*100.
        // O(n log n) total instead of O(n²).
        std::map<std::pair<int64_t, int>, double> children_sum;
        for(auto const& [k, v] : agg)
            children_sum[{ k.tid, k.depth }] += v.sum;

        for(auto const& [k, v] : agg)
        {
            double mean   = v.sum / static_cast<double>(v.count);
            double var    = variance(v.count, v.sum, v.sum_sq);
            double stddev = std::sqrt(var);

            double ps = 100.0;
            if(v.sum > 0.0)
            {
                auto   it    = children_sum.find({ k.tid, k.depth + 1 });
                double csum  = (it != children_sum.end()) ? it->second : 0.0;
                double ratio = std::min(csum / v.sum, 1.0);
                ps           = std::max((1.0 - ratio) * 100.0, 0.0);
            }

            out << k.tid << '\t' << k.depth << '\t' << k.label << '\t' << v.count << '\t'
                << fmt_fixed<6>(v.sum) << '\t' << fmt_fixed<6>(mean) << '\t'
                << fmt_fixed<6>(v.min_v) << '\t' << fmt_fixed<6>(v.max_v) << '\t'
                << fmt_fixed<6>(var) << '\t' << fmt_fixed<6>(stddev) << '\t'
                << fmt_fixed<6>(ps) << '\n';
        }
    }

    static void emit_pct_rows(std::ostream& out, std::map<pct_key, pct_stats> const& agg)
    {
        for(auto const& [k, v] : agg)
            out << k.tid << '\t' << k.label << '\t' << v.count << '\t'
                << fmt_fixed<3>(v.sum) << '\n';
    }

    static void emit_trip_rows(std::ostream& out, std::map<row_key, uint64_t> const& agg)
    {
        for(auto const& [k, v] : agg)
            out << k.tid << '\t' << k.depth << '\t' << k.label << '\t' << v << '\n';
    }

    template <int N>
    static std::string fmt_fixed(double v)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(N) << v;
        return ss.str();
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
};

}  // namespace rocprofsys::sampling
