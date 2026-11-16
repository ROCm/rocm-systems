// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/wall_clock_report.hpp"
#include "common/string_utility.hpp"

#include <fmt/base.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <optional>
#include <ranges>
#include <ratio>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys
{
namespace
{
using wall_clock_detail::interval_store;
using wall_clock_detail::stored_interval;

constexpr std::size_t      k_indent_per_depth = 2;
constexpr std::string_view k_metric           = "wall_clock";
constexpr std::string_view k_title = "REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)";
constexpr std::uint64_t    k_collapsed_thread    = 0;
constexpr std::string_view k_collapsed_prefix    = ">>> ";
constexpr std::string_view k_branch_marker       = "|_";
constexpr std::size_t      k_narrow_column_width = 6;
constexpr std::size_t      k_value_column_width  = 8;
constexpr std::size_t      k_frame_overhead      = 3;
constexpr double           k_percent             = 100.0;
// Wider labels are cut with an ellipsis, narrower ones without, as timemory does
constexpr std::int32_t     k_min_width_with_ellipsis = 20;
constexpr std::string_view k_ellipsis                = "...";

constexpr std::array<std::string_view, 6> k_value_column_names{ "SUM", "MEAN", "MIN",
                                                                "MAX", "VAR",  "STDDEV" };
constexpr std::size_t                     k_value_columns = k_value_column_names.size();

struct timing_stats
{
    std::uint64_t count    = 0;
    std::uint64_t total_ns = 0;
    double        sum      = 0.0;
    double        min      = std::numeric_limits<double>::max();
    double        max      = 0.0;
    double        mean     = 0.0;
    double        m2       = 0.0;

    void add(wall_clock_nanoseconds elapsed) noexcept
    {
        const auto seconds = std::chrono::duration<double>{ elapsed }.count();
        ++count;
        total_ns += elapsed.count();
        sum += seconds;
        min              = std::min(min, seconds);
        max              = std::max(max, seconds);
        const auto delta = seconds - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (seconds - mean);
    }

    [[nodiscard]] double sum_of_squares() const noexcept
    {
        return m2 + static_cast<double>(count) * mean * mean;
    }

    [[nodiscard]] double variance() const noexcept
    {
        return count > 1 ? m2 / static_cast<double>(count - 1) : 0.0;
    }
};

enum class call_kind
{
    cpu,
    gpu
};

enum class child_policy
{
    merge,
    append
};

struct call_node
{
    std::string          name;
    call_kind            kind = call_kind::cpu;
    timing_stats         data = {};
    std::list<call_node> children;

    /// Linear search keeps first-seen order; it is cheap unless a flat or collapsed
    /// layout puts many distinct names under one node.
    call_node& child(std::string_view child_name, call_kind child_kind,
                     child_policy policy)
    {
        if(policy == child_policy::merge)
        {
            const auto existing =
                std::ranges::find(children, child_name, &call_node::name);
            if(existing != children.end())
            {
                return *existing;
            }
        }
        return children.emplace_back(call_node{
            .name = std::string{ child_name }, .kind = child_kind, .children = {} });
    }

    /// Share of this node's time not spent in CPU children. GPU children run
    /// asynchronously and may outlast the launch, so they are not subtracted.
    [[nodiscard]] double self_percent() const noexcept
    {
        if(data.sum <= 0.0)
        {
            return k_percent;
        }
        auto const cpu_children = std::accumulate(children.begin(), children.end(), 0.0,
                                                  [](double total, const call_node& kid) {
                                                      return kid.kind == call_kind::gpu
                                                                 ? total
                                                                 : total + kid.data.sum;
                                                  });
        return k_percent * (data.sum - cpu_children) / data.sum;
    }
};

struct thread_trees
{
    std::map<std::uint64_t, call_node> by_thread;
    std::size_t                        thread_count = 0;
};

[[nodiscard]] double
scale_to_unit(const wall_clock_settings& settings)
{
    return std::chrono::duration<double>{ std::chrono::seconds{ 1 } } /
           settings.unit.length;
}

void
validate_interval(const wall_clock_interval& interval)
{
    if(interval.end < interval.begin)
    {
        throw std::runtime_error(
            fmt::format("wall_clock: '{}' (thread {}, correlation {}) ends at {} before "
                        "it begins at {}",
                        interval.name, interval.thread_id, interval.correlation_id,
                        interval.end.count(), interval.begin.count()));
    }
}

struct report_row
{
    std::string      label;
    const call_node* entry;
    std::size_t      depth;
};

struct row_builder
{
    std::vector<report_row> rows;
    std::string_view        prefix;
    std::size_t             max_depth;

    void add_children(const call_node& parent, std::size_t depth)
    {
        if(depth >= max_depth)
        {
            return;
        }
        for(const auto& kid : parent.children)
        {
            auto const label = depth == 0 ? fmt::format("{}{}", prefix, kid.name)
                                          : fmt::format("{}{:{}}{}{}", prefix, "",
                                                        k_indent_per_depth * (depth - 1),
                                                        k_branch_marker, kid.name);
            rows.push_back({ .label = label, .entry = &kid, .depth = depth });
            add_children(kid, depth + 1);
        }
    }
};

[[nodiscard]] std::vector<report_row>
build_rows(const thread_trees& trees, const wall_clock_settings& settings)
{
    row_builder rows{ .rows = {}, .prefix = {}, .max_depth = settings.max_depth };
    std::size_t thread_ordinal = 0;
    for(const auto& [thread_id, root] : trees.by_thread)
    {
        auto const prefix = settings.collapse_threads
                                ? std::string{ k_collapsed_prefix }
                                : fmt::format("|{}>>> ", thread_ordinal);
        rows.prefix       = prefix;
        rows.add_children(root, 0);
        ++thread_ordinal;
    }
    return std::move(rows.rows);
}

struct tree_builder
{
    const wall_clock_settings&                    settings;
    const interval_store&                         intervals;
    child_policy                                  policy;
    thread_trees                                  trees                    = {};
    std::set<std::uint64_t>                       threads                  = {};
    std::unordered_map<std::uint64_t, call_node*> launch_by_correlation_id = {};

    call_node& tree_of(std::uint64_t thread_id)
    {
        threads.insert(thread_id);
        return trees
            .by_thread[settings.collapse_threads ? k_collapsed_thread : thread_id];
    }

    void nest_regions()
    {
        std::vector<const stored_interval*> sorted;
        sorted.reserve(intervals.regions.size());
        std::ranges::transform(intervals.regions, std::back_inserter(sorted),
                               [](const stored_interval& region) { return &region; });
        std::ranges::sort(sorted,
                          [](const stored_interval* lhs, const stored_interval* rhs) {
                              return std::tie(lhs->thread_id, lhs->begin, rhs->end) <
                                     std::tie(rhs->thread_id, rhs->begin, lhs->end);
                          });

        std::vector<std::pair<call_node*, wall_clock_nanoseconds>> open_regions;
        auto open_regions_thread = std::optional<std::uint64_t>{};
        for(const auto* region : sorted)
        {
            if(open_regions_thread != region->thread_id)
            {
                open_regions.clear();
                open_regions_thread = region->thread_id;
            }

            std::erase_if(open_regions,
                          [&](const auto& open) { return region->begin >= open.second; });

            auto const reversed  = open_regions | std::views::reverse;
            auto const enclosing = std::ranges::find_if(
                reversed, [&](const auto& open) { return open.second >= region->end; });
            auto& parent = (settings.flat || enclosing == std::ranges::end(reversed))
                               ? tree_of(region->thread_id)
                               : *enclosing->first;
            auto& entry =
                parent.child(intervals.names.at(region->name), call_kind::cpu, policy);
            entry.data.add(region->end - region->begin);
            open_regions.emplace_back(&entry, region->end);
            if(region->correlation_id != k_no_correlation)
            {
                launch_by_correlation_id[region->correlation_id] = &entry;
            }
        }
    }

    void attach_gpu_activity()
    {
        for(const auto& activity : intervals.gpu_activity)
        {
            auto const launch = launch_by_correlation_id.find(activity.correlation_id);
            auto& parent = (launch != launch_by_correlation_id.end() && !settings.flat)
                               ? *launch->second
                               : tree_of(activity.thread_id);
            parent.child(intervals.names.at(activity.name), call_kind::gpu, policy)
                .data.add(activity.end - activity.begin);
        }
    }
};

[[nodiscard]] thread_trees
build_thread_trees(const wall_clock_settings& settings, const interval_store& intervals)
{
    tree_builder builder{ .settings  = settings,
                          .intervals = intervals,
                          .policy    = settings.timeline ? child_policy::append
                                                         : child_policy::merge };
    builder.nest_regions();
    builder.attach_gpu_activity();
    builder.trees.thread_count = builder.threads.size();
    return std::move(builder.trees);
}

void
truncate_labels(std::vector<report_row>& rows, std::int32_t max_width_setting)
{
    auto const max_width = static_cast<std::size_t>(std::max(max_width_setting, 0));
    auto const ellipsis =
        max_width_setting > k_min_width_with_ellipsis ? k_ellipsis : std::string_view{};
    for(auto& row : rows)
    {
        if(max_width > 0 && row.label.size() > max_width)
        {
            row.label = fmt::format(
                "{}{}", row.label.substr(0, max_width - ellipsis.size()), ellipsis);
        }
    }
}

struct value_cells
{
    std::vector<std::array<double, k_value_columns>> values;
    std::array<std::size_t, k_value_columns>         widths = {};
};

[[nodiscard]] value_cells
compute_value_cells(const std::vector<report_row>& rows,
                    const wall_clock_settings&     settings)
{
    auto const  scale = scale_to_unit(settings);
    value_cells cells;
    for(const auto& row : rows)
    {
        const auto&                               stats = row.entry->data;
        std::array<double, k_value_columns> const values{
            stats.sum * scale,
            stats.mean * scale,
            stats.min * scale,
            stats.max * scale,
            stats.variance() * scale * scale,
            std::sqrt(stats.variance()) * scale
        };
        for(std::size_t column = 0; column < k_value_columns; ++column)
        {
            cells.widths[column] = std::max(
                { cells.widths[column], k_value_column_width,
                  fmt::formatted_size("{:.{}f}", values[column], settings.precision) });
        }
        cells.values.push_back(values);
    }
    return cells;
}

[[nodiscard]] std::string
format_header(std::size_t                                     label_width,
              const std::array<std::size_t, k_value_columns>& widths)
{
    auto header = fmt::format("| {:^{}} | {:^{}} | {:^{}} | {:^{}} | {:^{}} | ", "LABEL",
                              label_width, "COUNT", k_narrow_column_width, "DEPTH",
                              k_narrow_column_width, "METRIC", k_metric.size(), "UNITS",
                              k_narrow_column_width);
    for(std::size_t column = 0; column < k_value_columns; ++column)
    {
        // timemory left-aligns the last value heading
        auto const last = column + 1 == k_value_columns;
        fmt::format_to(std::back_inserter(header),
                       fmt::runtime(last ? "{:<{}} | " : "{:^{}} | "),
                       k_value_column_names[column], widths[column]);
    }
    fmt::format_to(std::back_inserter(header), "{:^{}} |\n", "% SELF",
                   k_narrow_column_width);
    return header;
}

[[nodiscard]] nlohmann::json
row_to_json(const report_row& row, double scale)
{
    auto const& stats = row.entry->data;
    return { { "prefix", row.label },
             { "depth", row.depth },
             { "entry",
               { { "laps", stats.count },
                 { "value", stats.total_ns },
                 { "accum", stats.total_ns },
                 { "repr_data", stats.sum * scale },
                 { "repr_display", stats.sum * scale } } },
             { "stats",
               { { "sum", stats.sum * scale },
                 { "count", stats.count },
                 { "min", stats.min * scale },
                 { "max", stats.max * scale },
                 { "sqr", stats.sum_of_squares() * scale * scale },
                 { "mean", stats.mean * scale },
                 { "stddev", std::sqrt(stats.variance()) * scale } } } };
}

}  // namespace

std::optional<wall_clock_unit>
find_wall_clock_unit(std::string_view name)
{
    using namespace std::chrono_literals;
    if(name.empty())
    {
        return k_wall_clock_seconds;
    }
    struct unit_spellings
    {
        wall_clock_unit                 unit;
        std::array<std::string_view, 4> spellings;
    };
    constexpr std::array k_units{
        unit_spellings{ .unit      = { .name = "nsec", .length = 1ns },
                        .spellings = { "nsec", "ns", "nanosecond", "nanoseconds" } },
        unit_spellings{ .unit      = { .name = "usec", .length = 1us },
                        .spellings = { "usec", "us", "microsecond", "microseconds" } },
        unit_spellings{ .unit      = { .name = "msec", .length = 1ms },
                        .spellings = { "msec", "ms", "millisecond", "milliseconds" } },
        unit_spellings{
            .unit      = { .name   = "csec",
                           .length = std::chrono::duration<std::int64_t, std::centi>{ 1 } },
            .spellings = { "csec", "cs", "centisecond", "centiseconds" } },
        unit_spellings{
            .unit      = { .name   = "dsec",
                           .length = std::chrono::duration<std::int64_t, std::deci>{ 1 } },
            .spellings = { "dsec", "ds", "decisecond", "deciseconds" } },
        unit_spellings{ .unit      = k_wall_clock_seconds,
                        .spellings = { "sec", "s", "second", "seconds" } },
        unit_spellings{ .unit      = { .name = "min", .length = 1min },
                        .spellings = { "min", "minute", "minutes" } },
        unit_spellings{ .unit      = { .name = "hr", .length = 1h },
                        .spellings = { "hr", "hour", "hours" } },
    };
    auto const is_spelling_of = [name](const unit_spellings& entry) {
        return std::ranges::any_of(entry.spellings, [name](std::string_view spelling) {
            return utility::string::equals_ignore_case(spelling, name);
        });
    };
    auto const found = std::ranges::find_if(k_units, is_spelling_of);
    return found == k_units.end() ? std::nullopt : std::optional{ found->unit };
}

wall_clock_report::wall_clock_report(wall_clock_settings settings)
: m_settings{ settings }
{}

void
wall_clock_report::add_region(const wall_clock_interval& region)
{
    validate_interval(region);
    m_intervals.regions.push_back(m_intervals.compact(region));
}

void
wall_clock_report::add_gpu_activity(const wall_clock_interval& activity)
{
    validate_interval(activity);
    m_intervals.gpu_activity.push_back(m_intervals.compact(activity));
}

bool
wall_clock_report::empty() const noexcept
{
    return m_intervals.regions.empty() && m_intervals.gpu_activity.empty();
}

std::string
wall_clock_report::text() const
{
    auto const trees = build_thread_trees(m_settings, m_intervals);
    auto       rows  = build_rows(trees, m_settings);
    if(rows.empty())
    {
        return {};
    }

    truncate_labels(rows, m_settings.max_width);
    auto const cells = compute_value_cells(rows, m_settings);
    auto const label_width =
        std::ranges::max(rows | std::views::transform([](const report_row& row) {
                             return row.label.size();
                         }));
    auto const header           = format_header(label_width, cells.widths);
    auto       column_separator = header;
    std::ranges::replace_if(
        column_separator, [](char chr) { return chr != '|' && chr != '\n'; }, '-');
    auto const frame_width = header.size() - k_frame_overhead;
    auto const banner      = fmt::format("|{:-<{}}|\n", "", frame_width);
    auto const title       = fmt::format("|{:^{}}|\n", k_title, frame_width);

    auto table =
        fmt::format("{}{}{}{}{}", banner, title, banner, header, column_separator);
    for(std::size_t index = 0; index < rows.size(); ++index)
    {
        const auto& [label, entry, depth] = rows[index];
        fmt::format_to(std::back_inserter(table),
                       "| {:<{}} | {:>{}} | {:>{}} | {:<{}} | {:<{}} | ", label,
                       label_width, entry->data.count, k_narrow_column_width, depth,
                       k_narrow_column_width, k_metric, k_metric.size(),
                       m_settings.unit.name, k_narrow_column_width);
        for(std::size_t column = 0; column < k_value_columns; ++column)
        {
            fmt::format_to(std::back_inserter(table), "{:>{}.{}f} | ",
                           cells.values[index][column], cells.widths[column],
                           m_settings.precision);
        }
        auto const self =
            depth + 1 < m_settings.max_depth ? entry->self_percent() : k_percent;
        fmt::format_to(std::back_inserter(table), "{:>{}.1f} |\n", self,
                       k_narrow_column_width);
    }
    table += banner;
    return table;
}

std::string
wall_clock_report::json() const
{
    constexpr std::size_t k_single_process = 1;
    auto const            trees            = build_thread_trees(m_settings, m_intervals);
    auto const            rows             = build_rows(trees, m_settings);
    auto const            scale            = scale_to_unit(m_settings);
    auto                  graph            = nlohmann::json::array();
    for(const auto& row : rows)
    {
        graph.push_back(row_to_json(row, scale));
    }

    return nlohmann::json{
        { "rocprofiler-systems",
          { { "wall_clock",
              { { "type", k_metric },
                { "description", "Real-clock timer (i.e. wall-clock timer)" },
                { "unit_value", m_settings.unit.length.count() },
                { "unit_repr", m_settings.unit.name },
                { "thread_scope_only", false },
                { "thread_count", trees.thread_count },
                { "mpi_size", m_settings.mpi_size },
                { "upcxx_size", k_single_process },
                { "process_count", k_single_process },
                { "num_ranks", k_single_process },
                { "concurrency", trees.thread_count },
                { "ranks",
                  { { { "rank", m_settings.rank },
                      { "graph_size", rows.size() },
                      { "graph", graph } } } } } } } }
    }.dump(4, ' ', false, nlohmann::json::error_handler_t::replace);
}

}  // namespace rocprofsys
