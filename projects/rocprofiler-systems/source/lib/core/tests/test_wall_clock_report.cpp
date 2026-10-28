// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/wall_clock_report.hpp"

#include <nlohmann/json.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using rocprofsys::find_wall_clock_unit;
using rocprofsys::wall_clock_interval;
using rocprofsys::wall_clock_nanoseconds;
using rocprofsys::wall_clock_report;
using rocprofsys::wall_clock_settings;
using rocprofsys::wall_clock_unit;

namespace
{
constexpr std::uint64_t   k_main_thread  = 1;
constexpr std::uint64_t   k_other_thread = 2;
constexpr std::uint64_t   k_launch_id    = 7;
constexpr wall_clock_unit k_milliseconds{ .name   = "ms",
                                          .length = std::chrono::milliseconds{ 1 } };

constexpr wall_clock_nanoseconds k_start        = {};
constexpr wall_clock_nanoseconds k_ms           = std::chrono::milliseconds{ 1 };
constexpr wall_clock_nanoseconds k_launch_begin = k_start;
constexpr wall_clock_nanoseconds k_launch_end   = k_ms;
constexpr wall_clock_nanoseconds k_kernel_begin = 5 * k_ms;
constexpr wall_clock_nanoseconds k_kernel_end   = 7 * k_ms;

constexpr std::size_t k_columns_per_row = 12;
constexpr std::size_t k_count_column    = 1;
constexpr std::size_t k_depth_column    = 2;
constexpr std::size_t k_units_column    = 4;
constexpr std::size_t k_sum_column      = 5;
constexpr std::size_t k_min_column      = 7;
constexpr std::size_t k_max_column      = 8;
constexpr std::size_t k_var_column      = 9;
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
    std::string   thread;  // "0" for the label prefix "|0>>>", empty when collapsed
    std::uint64_t count;
    std::uint64_t depth;
    std::string   units;
    std::string   sum;
    std::string   min;
    std::string   max;
    std::string   var;
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
    if(!line.starts_with("| |") && !line.starts_with("| >>> "))
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

    auto const& label  = cells.front();  // "[|<thread>]>>> [indent|_]<name>"
    auto const  marker = label.find(">>> ");
    row.thread         = label.starts_with('|') ? label.substr(1, marker - 1) : "";
    row.name           = trim(label.substr(marker + std::string_view{ ">>> " }.size()));
    if(row.name.starts_with("|_"))
    {
        row.name = row.name.substr(2);
    }
    row.count        = std::stoull(cells[k_count_column]);
    row.depth        = std::stoull(cells[k_depth_column]);
    row.units        = cells[k_units_column];
    row.sum          = cells[k_sum_column];
    row.min          = cells[k_min_column];
    row.max          = cells[k_max_column];
    row.var          = cells[k_var_column];
    row.self_percent = cells[k_self_column];
    return true;
}

std::vector<std::string>
text_lines(const wall_clock_report& report)
{
    std::istringstream       input{ report.text() };
    std::vector<std::string> lines;
    for(std::string line; std::getline(input, line);)
    {
        lines.push_back(line);
    }
    return lines;
}

std::vector<report_row>
parse_rows(const wall_clock_report& report)
{
    std::vector<report_row> rows;
    for(const auto& line : text_lines(report))
    {
        report_row row{};
        if(parse_row(line, row))
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

struct intervals
{
    std::vector<wall_clock_interval> regions;
    std::vector<wall_clock_interval> gpu_activity;
};

[[nodiscard]] wall_clock_report
make_report(const intervals& input, wall_clock_settings settings = {})
{
    wall_clock_report report{ settings };
    for(const auto& region : input.regions)
    {
        report.add_region(region);
    }
    for(const auto& activity : input.gpu_activity)
    {
        report.add_gpu_activity(activity);
    }
    return report;
}

/// A launch at the start of the timeline and the kernel it started, after it returned.
void
add_launch_with_kernel(intervals& input)
{
    input.regions.push_back(
        interval("hipLaunchKernel", span(k_launch_begin, k_launch_end), k_launch_id));
    input.gpu_activity.push_back(
        interval("tiny", span(k_kernel_begin, k_kernel_end), k_launch_id));
}
}  // namespace

TEST(wall_clock_report, kernel_nests_under_launch_with_same_correlation_id)
{
    intervals input;
    input.regions.push_back(
        interval("hipLaunchKernel", span(k_launch_begin, k_launch_end), k_launch_id));
    // another thread id: only the correlation id ties it to the launch
    input.gpu_activity.push_back(interval("tiny", span(k_kernel_begin, k_kernel_end),
                                          k_launch_id, k_other_thread));

    auto const rows = parse_rows(make_report(input));
    ASSERT_EQ(rows.size(), 2U);
    expect_node(rows[0], "hipLaunchKernel", 0);
    expect_node(rows[1], "tiny", 1);
    EXPECT_EQ(rows[1].count, 1U);
    EXPECT_EQ(rows[1].sum, "0.002000");
}

TEST(wall_clock_report, kernel_without_matching_launch_is_top_level)
{
    intervals input;
    input.regions.push_back(
        interval("hipLaunchKernel", span(k_launch_begin, k_launch_end), k_launch_id));
    input.gpu_activity.push_back(
        interval("tiny", span(k_kernel_begin, k_kernel_end), k_launch_id + 1));

    auto const rows = parse_rows(make_report(input));
    ASSERT_EQ(rows.size(), 2U);
    expect_node(rows[0], "hipLaunchKernel", 0);
    expect_node(rows[1], "tiny", 0);
}

TEST(wall_clock_report, repeated_launches_aggregate_into_one_row)
{
    constexpr std::uint64_t          k_first_launch = 100;
    constexpr wall_clock_nanoseconds k_stride       = 10 * k_ms;
    intervals                        input;
    for(std::uint64_t i = 0; i < 3; ++i)
    {
        auto const begin = i * k_stride;
        input.regions.push_back(interval(
            "hipLaunchKernel", span(begin, begin + (i + 1) * k_ms), k_first_launch + i));
        input.gpu_activity.push_back(
            interval("tiny", span(begin + k_kernel_begin, begin + k_kernel_end),
                     k_first_launch + i));
    }

    auto const rows = parse_rows(make_report(input));
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
    // fed out of order on purpose: nesting comes from the timestamps, not arrival order
    intervals const input{ .regions      = { interval("inner", k_inner),
                                             interval("after", k_after),
                                             interval("outer", k_outer) },
                           .gpu_activity = {} };

    auto const rows = parse_rows(make_report(input));
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
    intervals const     input{
            .regions      = { interval("outer", k_outer), interval("inner", k_inner),
                              interval("hipLaunchKernel", k_later_launch, k_launch_id) },
            .gpu_activity = { interval("tiny", k_long_kernel, k_launch_id) }
    };

    auto const rows = parse_rows(make_report(input));
    ASSERT_EQ(rows.size(), 4U);
    EXPECT_EQ(rows[0].name, "outer");
    EXPECT_EQ(rows[0].self_percent, "75.0");
    EXPECT_EQ(rows[2].name, "hipLaunchKernel");
    EXPECT_EQ(rows[2].self_percent, "100.0");
}

TEST(wall_clock_report, each_thread_gets_its_own_ordinal_tree)
{
    intervals const input{
        .regions      = { interval("work", span(k_launch_begin, k_launch_end), 0,
                                   k_other_thread),
                          interval("work", span(k_launch_begin, k_launch_end)) },
        .gpu_activity = {}
    };

    auto const rows = parse_rows(make_report(input));
    ASSERT_EQ(rows.size(), 2U);
    EXPECT_EQ(rows[0].thread, "0");
    EXPECT_EQ(rows[1].thread, "1");
}

TEST(wall_clock_report, flat_layout_puts_every_row_directly_under_its_thread)
{
    constexpr time_span k_outer = span(k_start, 4 * k_ms);
    constexpr time_span k_inner = span(k_start, k_ms);
    intervals           input{ .regions      = { interval("outer", k_outer),
                                                 interval("inner", k_inner) },
                               .gpu_activity = {} };
    add_launch_with_kernel(input);

    auto const rows = parse_rows(make_report(input, { .flat = true }));
    ASSERT_EQ(rows.size(), 4U);
    for(const auto& row : rows)
    {
        EXPECT_EQ(row.depth, 0U) << row.name;
        EXPECT_EQ(row.self_percent, "100.0") << row.name;
    }
}

TEST(wall_clock_report, flat_layout_still_merges_repeated_calls)
{
    intervals const input{
        .regions      = { interval("work", span(k_launch_begin, k_launch_end)),
                          interval("work", span(k_kernel_begin, k_kernel_end)) },
        .gpu_activity = {}
    };

    auto const rows = parse_rows(make_report(input, { .flat = true }));
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].count, 2U);
}

TEST(wall_clock_report, timeline_layout_keeps_one_row_per_call_in_call_order)
{
    constexpr std::uint64_t k_second_launch = k_launch_id + 1;
    constexpr time_span     k_second_begin  = span(10 * k_ms, 11 * k_ms);
    constexpr time_span     k_second_kernel = span(12 * k_ms, 13 * k_ms);
    intervals               input;
    add_launch_with_kernel(input);
    input.regions.push_back(interval("hipLaunchKernel", k_second_begin, k_second_launch));
    input.gpu_activity.push_back(interval("tiny", k_second_kernel, k_second_launch));

    auto const rows = parse_rows(make_report(input, { .timeline = true }));
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
    intervals const input{
        .regions      = { interval("work", span(k_launch_begin, k_launch_end)),
                          interval("work", span(k_kernel_begin, k_kernel_end)) },
        .gpu_activity = {}
    };

    auto const rows = parse_rows(make_report(input, { .flat = true, .timeline = true }));
    ASSERT_EQ(rows.size(), 2U);
    expect_node(rows[0], "work", 0);
    expect_node(rows[1], "work", 0);
}

TEST(wall_clock_report, collapsed_threads_share_one_tree_without_thread_number)
{
    intervals const input{
        .regions      = { interval("work", span(k_launch_begin, k_launch_end), 0,
                                   k_other_thread),
                          interval("work", span(k_launch_begin, k_launch_end)) },
        .gpu_activity = {}
    };
    auto const report = make_report(input, { .collapse_threads = true });

    auto const rows = parse_rows(report);
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].thread, "");
    EXPECT_EQ(rows[0].count, 2U);

    auto const  json = nlohmann::json::parse(report.json());
    auto const& wall = json.at("rocprofiler-systems").at("wall_clock");
    EXPECT_EQ(wall.at("thread_count"), 2);
    EXPECT_EQ(wall.at("ranks").at(0).at("graph").at(0).at("prefix"), ">>> work");
}

TEST(wall_clock_report, has_timemory_table_layout)
{
    intervals input;
    add_launch_with_kernel(input);

    auto const lines = text_lines(make_report(input));
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

TEST(wall_clock_report, value_columns_grow_to_fit_long_values)
{
    constexpr wall_clock_nanoseconds k_twelve_seconds = std::chrono::seconds{ 12 };
    intervals const                  input{
                         .regions = { interval("long", span(k_start, k_twelve_seconds)),
                                      interval("short", span(k_twelve_seconds, k_twelve_seconds + k_ms)) },
                         .gpu_activity = {}
    };

    auto const lines = text_lines(make_report(input));
    ASSERT_FALSE(lines.empty());
    for(const auto& line : lines)
    {
        EXPECT_EQ(line.size(), lines.front().size()) << line;
    }
    auto const rows = parse_rows(make_report(input));
    ASSERT_EQ(rows.size(), 2U);
    EXPECT_EQ(rows[0].sum, "12.000000");
}

TEST(wall_clock_report, json_graph_has_the_rows_the_timemory_validator_reads)
{
    intervals input;
    add_launch_with_kernel(input);

    auto const  json = nlohmann::json::parse(make_report(input).json());
    auto const& wall = json.at("rocprofiler-systems").at("wall_clock");
    EXPECT_EQ(wall.at("unit_repr"), "s");
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

TEST(wall_clock_report, json_replaces_invalid_utf8_in_labels)
{
    intervals const input{ .regions      = { interval("caf\xE9", span(k_start, k_ms)) },
                           .gpu_activity = {} };

    auto const json = nlohmann::json::parse(make_report(input).json());
    EXPECT_EQ(json.at("rocprofiler-systems")
                  .at("wall_clock")
                  .at("ranks")
                  .at(0)
                  .at("graph")
                  .at(0)
                  .at("prefix"),
              "|0>>> caf\xEF\xBF\xBD");
}

/// Three calls of 1, 2 and 3 ms that merge into one row.
intervals
merged_calls()
{
    intervals input;
    for(std::uint64_t i = 0; i < 3; ++i)
    {
        auto const begin = i * 10 * k_ms;
        input.regions.push_back(interval("work", span(begin, begin + (i + 1) * k_ms)));
    }
    return input;
}

nlohmann::json
merged_calls_row(wall_clock_settings settings = {})
{
    return nlohmann::json::parse(make_report(merged_calls(), settings).json())
        .at("rocprofiler-systems")
        .at("wall_clock")
        .at("ranks")
        .at(0)
        .at("graph")
        .at(0);
}

nlohmann::json
merged_calls_stats()
{
    return merged_calls_row().at("stats");
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
    auto const        broken = interval("broken", span(k_launch_end, k_launch_begin));
    wall_clock_report report{ {} };
    EXPECT_THROW(report.add_region(broken), std::runtime_error);
    EXPECT_THROW(report.add_gpu_activity(broken), std::runtime_error);
    EXPECT_TRUE(report.empty());
}

TEST(wall_clock_report,
     max_depth_drops_deeper_rows_and_their_parent_shows_all_time_as_self)
{
    intervals const input{ .regions      = { interval("outer", span(k_start, 4 * k_ms)),
                                             interval("inner", span(k_start, k_ms)) },
                           .gpu_activity = {} };

    auto const rows = parse_rows(make_report(input, { .max_depth = 1 }));
    ASSERT_EQ(rows.size(), 1U);
    expect_node(rows[0], "outer", 0);
    EXPECT_EQ(rows[0].self_percent, "100.0");
    EXPECT_TRUE(make_report(input, { .max_depth = 0 }).text().empty());
}

TEST(wall_clock_report, text_label_is_cut_to_max_width_and_json_keeps_it)
{
    constexpr std::string_view k_long_name = "abcdefghijklmnopqrstuvwxyz0123";
    struct width_case
    {
        std::int32_t     max_width;
        std::string_view name;
    };
    // The label "|0>>> abcdefghijklmnopqrstuvwxyz0123" has 36 characters
    constexpr std::array k_cases{ width_case{ .max_width = 25,
                                              .name      = "abcdefghijklmnop..." },
                                  width_case{ .max_width = 10, .name = "abcd" },
                                  width_case{ .max_width = 0, .name = k_long_name } };
    intervals const      input{ .regions      = { interval(std::string{ k_long_name },
                                                           span(k_start, k_ms)) },
                                .gpu_activity = {} };
    for(const auto& [max_width, name] : k_cases)
    {
        SCOPED_TRACE(max_width);
        auto const report = make_report(input, { .max_width = max_width });

        auto const rows = parse_rows(report);
        ASSERT_EQ(rows.size(), 1U);
        EXPECT_EQ(rows[0].name, name);
        auto const json = nlohmann::json::parse(report.json());
        EXPECT_EQ(json.at("rocprofiler-systems")
                      .at("wall_clock")
                      .at("ranks")
                      .at(0)
                      .at("graph")
                      .at(0)
                      .at("prefix"),
                  "|0>>> abcdefghijklmnopqrstuvwxyz0123");
    }
}

TEST(wall_clock_report, timing_units_scale_text_columns_and_square_the_variance)
{
    auto const rows = parse_rows(make_report(merged_calls(), { .unit = k_milliseconds }));
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].units, "ms");
    EXPECT_EQ(rows[0].sum, "6.000000");
    EXPECT_EQ(rows[0].min, "1.000000");
    EXPECT_EQ(rows[0].var, "1.000000");
}

TEST(wall_clock_report, timing_units_scale_json_display_and_stats_but_not_raw_values)
{
    auto const row = merged_calls_row({ .unit = k_milliseconds });
    EXPECT_EQ(row.at("entry").at("value"), (6 * k_ms).count());
    EXPECT_NEAR(row.at("entry").at("repr_data").get<double>(), 6.0, 1e-9);
    EXPECT_NEAR(row.at("stats").at("sum").get<double>(), 6.0, 1e-9);
    EXPECT_NEAR(row.at("stats").at("sqr").get<double>(), 14.0, 1e-9);
    EXPECT_NEAR(row.at("stats").at("stddev").get<double>(), 1.0, 1e-9);

    auto const document = nlohmann::json::parse(
        make_report(merged_calls(), { .unit = k_milliseconds }).json());
    EXPECT_EQ(document.at("rocprofiler-systems").at("wall_clock").at("unit_repr"), "ms");
    EXPECT_EQ(document.at("rocprofiler-systems").at("wall_clock").at("unit_value"),
              1'000'000);
}

TEST(wall_clock_report, timing_precision_sets_the_decimals_of_value_columns)
{
    auto const rows = parse_rows(make_report(merged_calls(), { .precision = 3 }));
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].sum, "0.006");
    EXPECT_EQ(rows[0].self_percent, "100.0");
}

TEST(wall_clock_report, timing_unit_lookup_accepts_the_timemory_names_in_any_case)
{
    using namespace std::chrono_literals;
    struct unit_case
    {
        std::string_view         setting;
        std::string_view         name;
        std::chrono::nanoseconds length;
    };
    constexpr std::array k_cases{
        unit_case{ .setting = "nsec", .name = "ns", .length = 1ns },
        unit_case{ .setting = "microseconds", .name = "us", .length = 1us },
        unit_case{ .setting = "MSEC", .name = "ms", .length = 1ms },
        unit_case{ .setting = "csec", .name = "cs", .length = 10ms },
        unit_case{ .setting = "ds", .name = "ds", .length = 100ms },
        unit_case{ .setting = "", .name = "s", .length = 1s },
        unit_case{ .setting = "sec", .name = "s", .length = 1s },
        unit_case{ .setting = "Seconds", .name = "s", .length = 1s },
        unit_case{ .setting = "minute", .name = "min", .length = 1min },
        unit_case{ .setting = "hr", .name = "h", .length = 1h },
    };
    for(const auto& [setting, name, length] : k_cases)
    {
        SCOPED_TRACE(setting);
        auto const unit = find_wall_clock_unit(setting);
        ASSERT_TRUE(unit.has_value());
        EXPECT_EQ(unit->name, name);
        EXPECT_EQ(unit->length, length);
    }
}

TEST(wall_clock_report, timing_unit_lookup_rejects_names_timemory_rejects)
{
    for(std::string_view const name : { "h", "fortnight" })
    {
        SCOPED_TRACE(name);
        EXPECT_FALSE(find_wall_clock_unit(name).has_value());
    }
}
