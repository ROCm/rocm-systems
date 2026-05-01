// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profile_report_processor.hpp"

#include "core/config.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/sample_type.hpp"

#include "logger/debug.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <rocprofiler-sdk/version.h>

namespace rocprofsys
{
namespace trace_cache
{
namespace
{
constexpr double NS_TO_SEC = 1e-9;

const char*
source_string(wall_clock_event_source s)
{
    switch(s)
    {
        case wall_clock_event_source::buffered_kernel_dispatch:
            return "buffered_kernel_dispatch";
        case wall_clock_event_source::buffered_scratch_memory:
            return "buffered_scratch_memory";
        case wall_clock_event_source::buffered_memory_copy: return "buffered_memory_copy";
        case wall_clock_event_source::tool_callback_api: return "tool_callback_api";
        case wall_clock_event_source::ompt_callback_api: return "ompt_callback_api";
        default: return "unknown";
    }
}

std::string
format_timemory_style_label(uint64_t timemory_tid, const std::string& raw_label)
{
    return "|" + std::to_string(timemory_tid) + ">>> " + raw_label;
}

std::string
fmt_fixed(double v, int prec)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

std::string
horizontal_rule(size_t width)
{
    return std::string(width, '-');
}

}  // namespace

profile_report_processor_t::profile_report_processor_t(
    int pid, int ppid, output_file_registry& output_registry)
: processor_t<profile_report_processor_t>()
, m_pid(pid)
, m_ppid(ppid)
, m_output_registry(output_registry)
{}

void
profile_report_processor_t::prepare_for_processing()
{}

void
profile_report_processor_t::finalize_processing()
{
    auto _cfg       = settings::compose_filename_config{};
    _cfg.use_suffix = config::get_use_pid();
    _cfg.suffix     = settings::default_process_suffix();
    // Default compose_filename_config::make_dir is false; without creating the directory,
    // ofstream fails when the session path under ROCPROFSYS_OUTPUT_PATH does not exist.
    _cfg.make_dir = true;

    auto path = settings::compose_output_filename("wall_clock_cache", "txt", _cfg);
    if(path.empty())
    {
        path = std::string{ trace_cache::tmp_directory } + "wall_clock_cache_" +
               std::to_string(m_ppid) + "_" + std::to_string(m_pid) + ".txt";
    }

    std::ofstream ofs(path);
    if(!ofs.good())
    {
        LOG_WARNING("Could not open trace_cache wall_clock report path: {}", path);
        return;
    }

    using row_entry_t = std::pair<aggregate_key_t, aggregate_row>;
    std::vector<row_entry_t> rows(m_aggregates.begin(), m_aggregates.end());
    std::sort(rows.begin(), rows.end(), [](const row_entry_t& a, const row_entry_t& b) {
        const auto& [la, sa, ra, ma, da] = a.first;
        const auto& [lb, sb, rb, mb, db] = b.first;
        if(la != lb) return la < lb;
        if(da != db) return da < db;
        if(ma != mb) return ma < mb;
        if(ra != rb) return ra < rb;
        if(sa != sb) return static_cast<int>(sa) < static_cast<int>(sb);
        return false;
    });

    size_t max_label_display = 40;
    for(const auto& [key, row] : rows)
    {
        if(row.count == 0) continue;
        const auto& [label, src_unused, roc_unused, tim_tid, depth_unused] = key;
        (void) src_unused;
        (void) roc_unused;
        (void) depth_unused;
        (void) row;
        const auto disp   = format_timemory_style_label(tim_tid, label);
        max_label_display = std::max(max_label_display, disp.size());
    }
    max_label_display = std::min(max_label_display, size_t{ 120 });

    constexpr size_t w_count  = 8;
    constexpr size_t w_depth  = 7;
    constexpr size_t w_metric = 11;
    constexpr size_t w_units  = 7;
    constexpr size_t w_float  = 12;
    constexpr size_t w_pct    = 8;

    const size_t inner_width = max_label_display + 3 + w_count + 3 + w_depth + 3 +
                               w_metric + 3 + w_units + 3 + w_float * 7 + 3 + w_pct;

    const std::string title =
        "REAL-CLOCK TIMER (TRACE CACHE — rocprofiler timestamp durations)";
    const size_t banner_w = std::max(inner_width, title.size() + 4);

    ofs << '|' << horizontal_rule(banner_w - 2) << "|\n";
    ofs << "| " << std::left << std::setw(static_cast<int>(banner_w - 4))
        << std::setfill(' ') << title << " |\n";
    ofs << '|' << horizontal_rule(banner_w - 2) << "|\n";

    ofs << '|' << std::setw(static_cast<int>(max_label_display)) << std::left << "LABEL"
        << " | " << std::setw(static_cast<int>(w_count)) << std::right << "COUNT"
        << " | " << std::setw(static_cast<int>(w_depth)) << "DEPTH"
        << " | " << std::setw(static_cast<int>(w_metric)) << std::left << "METRIC"
        << " | " << std::setw(static_cast<int>(w_units)) << std::left << "UNITS"
        << " | " << std::setw(static_cast<int>(w_float)) << std::right << "SUM"
        << " | " << std::setw(static_cast<int>(w_float)) << "MEAN"
        << " | " << std::setw(static_cast<int>(w_float)) << "MIN"
        << " | " << std::setw(static_cast<int>(w_float)) << "MAX"
        << " | " << std::setw(static_cast<int>(w_float)) << "VAR"
        << " | " << std::setw(static_cast<int>(w_float)) << "STDDEV"
        << " | " << std::setw(static_cast<int>(w_pct)) << "% SELF"
        << " |\n";
    ofs << '|' << horizontal_rule(banner_w - 2) << "|\n";

    for(const auto& [key, row] : rows)
    {
        if(row.count == 0) continue;

        const auto& [label, src_u2, roc_tid_u, tim_tid, depth_val] = key;
        (void) src_u2;
        (void) roc_tid_u;

        const double sum_sec  = static_cast<double>(row.total_nsec) * NS_TO_SEC;
        const double mean_sec = sum_sec / static_cast<double>(row.count);
        const double min_sec  = static_cast<double>(row.min_nsec) * NS_TO_SEC;
        const double max_sec  = static_cast<double>(row.max_nsec) * NS_TO_SEC;

        double var_sec = 0.0;
        if(row.count > 0)
        {
            const double mean_sq = row.sum_sq_sec / static_cast<double>(row.count);
            var_sec              = mean_sq - mean_sec * mean_sec;
            if(var_sec < 0.0 && var_sec > -1e-18) var_sec = 0.0;
        }
        const double stddev_sec =
            (row.count > 0) ? std::sqrt(std::max(0.0, var_sec)) : 0.0;

        double pct_self = 0.0;
        if(row.total_nsec > 0)
        {
            pct_self = 100.0 * static_cast<double>(row.sum_exclusive_nsec) /
                       static_cast<double>(row.total_nsec);
            if(pct_self > 100.0) pct_self = 100.0;
            if(pct_self < 0.0) pct_self = 0.0;
        }

        std::string disp_label = format_timemory_style_label(tim_tid, label);
        if(disp_label.size() > max_label_display)
        {
            disp_label.resize(max_label_display > 3 ? max_label_display - 3
                                                    : max_label_display);
            disp_label += "...";
        }

        ofs << "| " << std::setw(static_cast<int>(max_label_display)) << std::left
            << disp_label << " | " << std::setw(static_cast<int>(w_count)) << std::right
            << row.count << " | " << std::setw(static_cast<int>(w_depth)) << std::right
            << depth_val << " | " << std::setw(static_cast<int>(w_metric)) << std::left
            << "wall_clock"
            << " | " << std::setw(static_cast<int>(w_units)) << std::left << "sec"
            << " | " << std::setw(static_cast<int>(w_float)) << std::right
            << fmt_fixed(sum_sec, 6) << " | " << std::setw(static_cast<int>(w_float))
            << fmt_fixed(mean_sec, 6) << " | " << std::setw(static_cast<int>(w_float))
            << fmt_fixed(min_sec, 6) << " | " << std::setw(static_cast<int>(w_float))
            << fmt_fixed(max_sec, 6) << " | " << std::setw(static_cast<int>(w_float))
            << fmt_fixed(var_sec, 6) << " | " << std::setw(static_cast<int>(w_float))
            << fmt_fixed(stddev_sec, 6) << " | " << std::setw(static_cast<int>(w_pct))
            << std::right << fmt_fixed(pct_self, 1) << " |\n";
    }

    ofs << '|' << horizontal_rule(banner_w - 2) << "|\n";
    ofs << "# DEPTH: instrumentation stack depth (rocprofiler-sdk) and/or nested "
           "callback scope.\n";
    ofs << "# % SELF: 100 * sum(exclusive_nsec) / sum(inclusive_nsec) per aggregate "
           "bucket.\n";
    ofs << "# Source dimension is listed in the debug lines below.\n\n";

    ofs << "# --- aggregate keys (label, source, rocprofiler_thread_id, timemory_tid, "
           "depth) ---\n";
    for(const auto& [key, row] : rows)
    {
        if(row.count == 0) continue;
        const auto& [lab, src, roc_tid, tim_tid, dep] = key;
        ofs << '#' << " label=" << lab << " source=" << source_string(src)
            << " rocprofiler_thread_id=" << roc_tid << " timemory_tid=" << tim_tid
            << " depth=" << dep << " count=" << row.count
            << " inclusive_nsec=" << row.total_nsec
            << " exclusive_nsec=" << row.sum_exclusive_nsec << '\n';
    }

    m_output_registry.register_file(path, output_format::text, "trace_cache_wall_clock");
    LOG_INFO("Wrote trace_cache wall_clock summary ({} aggregate rows): {}",
             m_aggregates.size(), path);
}

void
profile_report_processor_t::handle(const wall_clock_event_sample& sample)
{
    const aggregate_key_t key{ sample.label, sample.source, sample.thread_id,
                               sample.timemory_tid, sample.depth };
    auto&                 agg = m_aggregates[key];

    const uint64_t dur =
        (sample.end_ns > sample.begin_ns) ? (sample.end_ns - sample.begin_ns) : 0;

    ++agg.count;
    agg.total_nsec += dur;
    agg.sum_exclusive_nsec += sample.exclusive_nsec;
    const double d_sec = static_cast<double>(dur) * NS_TO_SEC;
    agg.sum_sq_sec += d_sec * d_sec;

    if(agg.count == 1u)
    {
        agg.min_nsec = agg.max_nsec = dur;
    }
    else
    {
        agg.min_nsec = std::min(agg.min_nsec, dur);
        agg.max_nsec = std::max(agg.max_nsec, dur);
    }
}

void
profile_report_processor_t::handle(const kernel_dispatch_sample&)
{}

void
profile_report_processor_t::handle(const scratch_memory_sample&)
{}

void
profile_report_processor_t::handle(const memory_copy_sample&)
{}

#if(ROCPROFILER_VERSION >= 600)
void
profile_report_processor_t::handle(const memory_allocate_sample&)
{}
#endif

void
profile_report_processor_t::handle(const region_sample&)
{}

void
profile_report_processor_t::handle(const in_time_sample&)
{}

void
profile_report_processor_t::handle(const pmc_event_with_sample&)
{}

void
profile_report_processor_t::handle(const gpu_pmc_sample&)
{}

void
profile_report_processor_t::handle(const ainic_pmc_sample&)
{}

void
profile_report_processor_t::handle(const cpu_pmc_sample&)
{}

void
profile_report_processor_t::handle(const backtrace_region_sample&)
{}

void
profile_report_processor_t::handle(const kfd_sample&)
{}

}  // namespace trace_cache
}  // namespace rocprofsys
