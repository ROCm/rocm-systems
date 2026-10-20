// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The report is rendered into a stream and parsed back into rows.

#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "core/trace_cache/wall_clock_processor.hpp"
#include "core/trace_cache/wall_clock_report.hpp"

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using rocprofsys::trace_cache::kernel_dispatch_sample;
using rocprofsys::trace_cache::metadata_registry;
using rocprofsys::trace_cache::wall_clock_interval;
using rocprofsys::trace_cache::wall_clock_nanoseconds;
using rocprofsys::trace_cache::wall_clock_processor_t;
using rocprofsys::trace_cache::wall_clock_report;

namespace
{
constexpr std::uint64_t k_main_thread  = 1;
constexpr std::uint64_t k_other_thread = 2;
constexpr std::uint64_t k_launch_id    = 7;

// Timeline
constexpr wall_clock_nanoseconds k_start        = {};
constexpr wall_clock_nanoseconds k_ms           = std::chrono::milliseconds{ 1 };
constexpr wall_clock_nanoseconds k_launch_begin = k_start;
constexpr wall_clock_nanoseconds k_launch_end   = k_ms;
constexpr wall_clock_nanoseconds k_kernel_begin =
    5 * k_ms;  // after the launch has returned
constexpr wall_clock_nanoseconds k_kernel_end = 7 * k_ms;

// Column positions of a table row
constexpr std::size_t k_columns_per_row = 12;
constexpr std::size_t k_count_column    = 1;
constexpr std::size_t k_depth_column    = 2;
constexpr std::size_t k_sum_column      = 5;
constexpr std::size_t k_min_column      = 7;
constexpr std::size_t k_max_column      = 8;
constexpr std::size_t k_self_column     = 11;

struct time_span
{
    wall_clock_nanoseconds begin;
    wall_clock_nanoseconds end;
};

[[nodiscard]] constexpr time_span
span(wall_clock_nanoseconds begin, wall_clock_nanoseconds end)
{
    return { .begin = begin, .end = end };
}

[[nodiscard]] wall_clock_interval
interval(std::string name, time_span when, std::uint64_t correlation = 0,
         std::uint64_t thread = k_main_thread)
{
    return { .thread_id      = thread,
             .name           = std::move(name),
             .correlation_id = correlation,
             .begin          = when.begin,
             .end            = when.end };
}

struct report_row
{
    std::string   name;
    std::string   thread;  // "0" for the label prefix "|0>>>"
    std::uint64_t count;
    std::uint64_t depth;
    std::string   sum;
    std::string   min;
    std::string   max;
    std::string   self_percent;
};

std::string
trim(const std::string& text)
{
    auto const first = text.find_first_not_of(' ');
    auto const last  = text.find_last_not_of(' ');
    return first == std::string::npos ? std::string{}
                                      : text.substr(first, last - first + 1);
}

/// Parses one table row: "| <label> | COUNT | DEPTH | METRIC | UNITS | SUM | MEAN | MIN |
/// MAX | VAR | STDDEV | % SELF |". Returns false for separators and the header.
bool
parse_row(const std::string& line, report_row& row)
{
    constexpr std::string_view k_separator = " | ";
    if(!line.starts_with("| |"))
    {
        return false;
    }
    auto const body = line.substr(2, line.size() - 4);  // drop leading "| " and " |"

    std::vector<std::string> cells;
    for(std::size_t pos = 0;;)
    {
        auto const next = body.find(k_separator, pos);
        cells.push_back(trim(body.substr(pos, next - pos)));
        if(next == std::string::npos)
        {
            break;
        }
        pos = next + k_separator.size();
    }
    if(cells.size() != k_columns_per_row)
    {
        return false;
    }

    auto const& label  = cells.front();  // "|<thread>>>> [indent|_]<name>"
    auto const  marker = label.find(">>> ");
    row.thread         = label.substr(1, marker - 1);
    row.name           = trim(label.substr(marker + std::string_view{ ">>> " }.size()));
    if(row.name.starts_with("|_"))
    {
        row.name = row.name.substr(2);
    }
    row.count        = std::stoull(cells[k_count_column]);
    row.depth        = std::stoull(cells[k_depth_column]);
    row.sum          = cells[k_sum_column];
    row.min          = cells[k_min_column];
    row.max          = cells[k_max_column];
    row.self_percent = cells[k_self_column];
    return true;
}

std::vector<std::string>
lines_of(wall_clock_report& report)
{
    std::ostringstream out;
    report.write_text(out);
    std::istringstream       input{ out.str() };
    std::vector<std::string> lines;
    for(std::string line; std::getline(input, line);)
    {
        lines.push_back(line);
    }
    return lines;
}

std::vector<report_row>
rows_of(wall_clock_report& report)
{
    std::vector<report_row> rows;
    for(const auto& line : lines_of(report))
    {
        if(report_row row{}; parse_row(line, row))
        {
            rows.push_back(std::move(row));
        }
    }
    return rows;
}

void
expect_node(const report_row& row, std::string_view name, std::uint64_t depth)
{
    EXPECT_EQ(row.name, name);
    EXPECT_EQ(row.depth, depth);
}

/// A launch at the start of the timeline and the kernel it started, after it returned.
void
add_launch_with_kernel(wall_clock_report& report)
{
    report.add_region(
        interval("hipLaunchKernel", span(k_launch_begin, k_launch_end), k_launch_id));
    report.add_gpu_activity(
        interval("tiny", span(k_kernel_begin, k_kernel_end), k_launch_id));
}
}  // namespace

TEST(wall_clock_report, kernel_nests_under_launch_with_same_correlation_id)
{
    wall_clock_report report{ {} };
    report.add_region(
        interval("hipLaunchKernel", span(k_launch_begin, k_launch_end), k_launch_id));
    // another thread id: only the correlation id ties it to the launch
    report.add_gpu_activity(interval("tiny", span(k_kernel_begin, k_kernel_end),
                                     k_launch_id, k_other_thread));

    auto const rows = rows_of(report);
    ASSERT_EQ(rows.size(), 2U);
    expect_node(rows[0], "hipLaunchKernel", 0);
    expect_node(rows[1], "tiny", 1);
    EXPECT_EQ(rows[1].count, 1U);
    EXPECT_EQ(rows[1].sum, "0.002000");
}

TEST(wall_clock_report, kernel_without_matching_launch_is_top_level)
{
    wall_clock_report report{ {} };
    report.add_region(
        interval("hipLaunchKernel", span(k_launch_begin, k_launch_end), k_launch_id));
    report.add_gpu_activity(
        interval("tiny", span(k_kernel_begin, k_kernel_end), k_launch_id + 1));

    auto const rows = rows_of(report);
    ASSERT_EQ(rows.size(), 2U);
    expect_node(rows[0], "hipLaunchKernel", 0);
    expect_node(rows[1], "tiny", 0);
}

TEST(wall_clock_report, repeated_launches_aggregate_into_one_row)
{
    constexpr std::uint64_t          k_first_launch = 100;
    constexpr wall_clock_nanoseconds k_stride       = 10 * k_ms;
    wall_clock_report                report{ {} };
    for(std::uint64_t i = 0; i < 3; ++i)
    {
        auto const begin = i * k_stride;
        report.add_region(interval("hipLaunchKernel", span(begin, begin + (i + 1) * k_ms),
                                   k_first_launch + i));
        report.add_gpu_activity(
            interval("tiny", span(begin + k_kernel_begin, begin + k_kernel_end),
                     k_first_launch + i));
    }

    auto const rows = rows_of(report);
    ASSERT_EQ(rows.size(), 2U);
    EXPECT_EQ(rows[0].count, 3U);
    EXPECT_EQ(rows[0].sum, "0.006000");
    EXPECT_EQ(rows[0].min, "0.001000");
    EXPECT_EQ(rows[0].max, "0.003000");
    EXPECT_EQ(rows[1].count, 3U);
}

TEST(wall_clock_report, contained_regions_nest_and_later_ones_do_not)
{
    constexpr time_span k_outer = span(k_start, 100 * k_ms);
    constexpr time_span k_inner = span(10 * k_ms, 20 * k_ms);
    constexpr time_span k_after = span(200 * k_ms, 300 * k_ms);
    wall_clock_report   report{ {} };
    // fed out of order on purpose: nesting comes from the timestamps, not arrival order
    report.add_region(interval("inner", k_inner));
    report.add_region(interval("after", k_after));
    report.add_region(interval("outer", k_outer));

    auto const rows = rows_of(report);
    ASSERT_EQ(rows.size(), 3U);
    expect_node(rows[0], "outer", 0);
    expect_node(rows[1], "inner", 1);
    expect_node(rows[2], "after", 0);
}

TEST(wall_clock_report, self_percent_excludes_only_cpu_children)
{
    constexpr time_span k_outer        = span(k_start, 4 * k_ms);
    constexpr time_span k_inner        = span(k_start, k_ms);
    constexpr time_span k_later_launch = span(10 * k_ms, 11 * k_ms);
    // the kernel outlasts its launch; subtracting it would make self time negative
    constexpr time_span k_long_kernel = span(12 * k_ms, 20 * k_ms);
    wall_clock_report   report{ {} };
    report.add_region(interval("outer", k_outer));
    report.add_region(interval("inner", k_inner));
    report.add_region(interval("hipLaunchKernel", k_later_launch, k_launch_id));
    report.add_gpu_activity(interval("tiny", k_long_kernel, k_launch_id));

    auto const rows = rows_of(report);
    ASSERT_EQ(rows.size(), 4U);
    EXPECT_EQ(rows[0].name, "outer");
    EXPECT_EQ(rows[0].self_percent, "75.0");
    EXPECT_EQ(rows[2].name, "hipLaunchKernel");
    EXPECT_EQ(rows[2].self_percent, "100.0");
}

TEST(wall_clock_report, each_thread_gets_its_own_ordinal_tree)
{
    wall_clock_report report{ {} };
    report.add_region(
        interval("work", span(k_launch_begin, k_launch_end), 0, k_other_thread));
    report.add_region(interval("work", span(k_launch_begin, k_launch_end)));

    auto const rows = rows_of(report);
    ASSERT_EQ(rows.size(), 2U);
    EXPECT_EQ(rows[0].thread, "0");
    EXPECT_EQ(rows[1].thread, "1");
}

TEST(wall_clock_report, flat_layout_puts_every_row_directly_under_its_thread)
{
    constexpr time_span k_outer = span(k_start, 4 * k_ms);
    constexpr time_span k_inner = span(k_start, k_ms);
    wall_clock_report   report{ { .flat = true } };
    report.add_region(interval("outer", k_outer));
    report.add_region(interval("inner", k_inner));
    add_launch_with_kernel(report);

    auto const rows = rows_of(report);
    ASSERT_EQ(rows.size(), 4U);
    for(const auto& row : rows)
    {
        EXPECT_EQ(row.depth, 0U) << row.name;
        EXPECT_EQ(row.self_percent, "100.0") << row.name;
    }
}

TEST(wall_clock_report, flat_layout_still_merges_repeated_calls)
{
    wall_clock_report report{ { .flat = true } };
    report.add_region(interval("work", span(k_launch_begin, k_launch_end)));
    report.add_region(interval("work", span(k_kernel_begin, k_kernel_end)));

    auto const rows = rows_of(report);
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].count, 2U);
}

TEST(wall_clock_report, timeline_layout_keeps_one_row_per_call_in_call_order)
{
    constexpr std::uint64_t k_second_launch = k_launch_id + 1;
    constexpr time_span     k_second_begin  = span(10 * k_ms, 11 * k_ms);
    constexpr time_span     k_second_kernel = span(12 * k_ms, 13 * k_ms);
    wall_clock_report       report{ { .timeline = true } };
    add_launch_with_kernel(report);
    report.add_region(interval("hipLaunchKernel", k_second_begin, k_second_launch));
    report.add_gpu_activity(interval("tiny", k_second_kernel, k_second_launch));

    auto const rows = rows_of(report);
    ASSERT_EQ(rows.size(), 4U);
    expect_node(rows[0], "hipLaunchKernel", 0);
    expect_node(rows[1], "tiny", 1);
    expect_node(rows[2], "hipLaunchKernel", 0);
    expect_node(rows[3], "tiny", 1);
    for(const auto& row : rows)
    {
        EXPECT_EQ(row.count, 1U) << row.name;
    }
}

TEST(wall_clock_report, flat_and_timeline_together_give_one_top_level_row_per_call)
{
    wall_clock_report report{ { .flat = true, .timeline = true } };
    report.add_region(interval("work", span(k_launch_begin, k_launch_end)));
    report.add_region(interval("work", span(k_kernel_begin, k_kernel_end)));

    auto const rows = rows_of(report);
    ASSERT_EQ(rows.size(), 2U);
    expect_node(rows[0], "work", 0);
    expect_node(rows[1], "work", 0);
}

TEST(wall_clock_report, has_timemory_table_layout)
{
    wall_clock_report report{ {} };
    add_launch_with_kernel(report);

    auto const lines = lines_of(report);
    // banner, title, banner, header, column separator, two rows, closing banner
    ASSERT_EQ(lines.size(), 8U);
    EXPECT_THAT(lines[1],
                ::testing::HasSubstr("REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)"));
    EXPECT_THAT(lines[3], ::testing::HasSubstr("| % SELF |"));
    for(const auto& line : lines)
    {
        EXPECT_EQ(line.size(), lines.front().size()) << line;
    }
}

TEST(wall_clock_report, json_graph_has_the_rows_the_timemory_validator_reads)
{
    wall_clock_report report{ {} };
    add_launch_with_kernel(report);
    std::ostringstream out;
    report.write_json(out, 0, 1);

    auto const  json = nlohmann::json::parse(out.str());
    auto const& wall = json.at("timemory").at("wall_clock");
    EXPECT_EQ(wall.at("unit_repr"), "sec");
    EXPECT_EQ(wall.at("unit_value"), 1'000'000'000);
    auto const& graph = wall.at("ranks").at(0).at("graph");
    ASSERT_EQ(graph.size(), 2U);
    EXPECT_EQ(wall.at("ranks").at(0).at("graph_size"), 2);
    EXPECT_EQ(graph[0].at("prefix"), "|0>>> hipLaunchKernel");
    EXPECT_EQ(graph[0].at("depth"), 0);
    EXPECT_EQ(graph[0].at("entry").at("laps"), 1);
    EXPECT_EQ(graph[0].at("entry").at("value"), k_ms.count());
    EXPECT_EQ(graph[1].at("prefix"), "|0>>> |_tiny");
    EXPECT_EQ(graph[1].at("depth"), 1);
    EXPECT_EQ(graph[1].at("entry").at("value"), (k_kernel_end - k_kernel_begin).count());
}

/// The stats object of the only row of a report with three calls of 1, 2 and 3 ms.
nlohmann::json
merged_calls_stats()
{
    wall_clock_report report{ {} };
    for(std::uint64_t i = 0; i < 3; ++i)
    {
        auto const begin = i * 10 * k_ms;
        report.add_region(interval("work", span(begin, begin + (i + 1) * k_ms)));
    }
    std::ostringstream out;
    report.write_json(out, 0, 1);
    return nlohmann::json::parse(out.str())
        .at("timemory")
        .at("wall_clock")
        .at("ranks")
        .at(0)
        .at("graph")
        .at(0)
        .at("stats");
}

TEST(wall_clock_report, json_stats_total_and_extremes_of_the_merged_calls)
{
    auto const stats = merged_calls_stats();
    EXPECT_EQ(stats.at("count"), 3);
    EXPECT_NEAR(stats.at("sum").get<double>(), 0.006, 1e-12);
    EXPECT_NEAR(stats.at("min").get<double>(), 0.001, 1e-12);
    EXPECT_NEAR(stats.at("max").get<double>(), 0.003, 1e-12);
}

TEST(wall_clock_report, json_stats_spread_of_the_merged_calls)
{
    auto const stats = merged_calls_stats();
    EXPECT_NEAR(stats.at("mean").get<double>(), 0.002, 1e-12);
    EXPECT_NEAR(stats.at("sqr").get<double>(), 14e-6, 1e-15);
    EXPECT_NEAR(stats.at("stddev").get<double>(), 0.001, 1e-12);  // sample (n-1)
}

TEST(wall_clock_report, intervals_ending_before_they_begin_throw)
{
    wall_clock_report report{ {} };
    EXPECT_THROW(
        report.add_region(interval("broken", span(k_launch_end, k_launch_begin))),
        std::runtime_error);
    EXPECT_THROW(
        report.add_gpu_activity(interval("broken", span(k_launch_end, k_launch_begin))),
        std::runtime_error);
    EXPECT_TRUE(report.empty());
}

TEST(wall_clock_processor, kernel_with_unregistered_symbol_throws)
{
    constexpr std::uint64_t k_unknown_kernel = 999;
    wall_clock_processor_t  processor{
        std::make_shared<metadata_registry>(), nullptr, 1, {}
    };
    EXPECT_THROW(processor.handle(kernel_dispatch_sample{
                     k_launch_begin.count(), k_launch_end.count(), k_main_thread, 0,
                     k_unknown_kernel, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0 }),
                 std::runtime_error);
}
