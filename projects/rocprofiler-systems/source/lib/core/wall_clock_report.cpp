// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/wall_clock_report.hpp"
#include "common/delimit.hpp"
#include "common/string_utility.hpp"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <list>
#include <map>
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
constexpr std::size_t      k_indent_per_depth = 2;
constexpr std::string_view k_metric           = "wall_clock";
constexpr std::string_view k_title          = "REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)";
constexpr std::uint64_t    k_no_correlation = 0;
constexpr std::uint64_t    k_collapsed_thread    = 0;
constexpr std::string_view k_collapsed_prefix    = ">>> ";
constexpr std::size_t      k_narrow_column_width = 6;
constexpr std::size_t      k_value_column_width  = 8;
constexpr std::size_t      k_value_columns       = 6;
constexpr double           k_percent             = 100.0;
// Wider labels are cut with an ellipsis, narrower ones without, as timemory does
constexpr std::int32_t     k_min_width_with_ellipsis = 20;
constexpr std::string_view k_ellipsis                = "...";

struct timing_stats
{
    std::uint64_t count = 0;
    double        sum   = 0.0;
    double        min   = std::numeric_limits<double>::max();
    double        max   = 0.0;
    double        mean  = 0.0;
    double        m2    = 0.0;

    void add(std::chrono::duration<double> elapsed) noexcept
    {
        const auto seconds = elapsed.count();
        ++count;
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

struct call_node
{
    std::string          name;
    call_kind            kind = call_kind::cpu;
    timing_stats         data = {};
    std::list<call_node> children;

    /// Returns the child called `child_name`. With `merge`, an existing child is
    /// reused (linear search: fan-out per node is small, so a map would only lose
    /// first-seen ordering); otherwise a new one is always appended.
    call_node& child(std::string_view child_name, call_kind child_kind, bool merge)
    {
        if(merge)
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
        auto cpu_children = 0.0;
        for(const auto& kid : children)
        {
            cpu_children += kid.kind == call_kind::gpu ? 0.0 : kid.data.sum;
        }
        return k_percent * (data.sum - cpu_children) / data.sum;
    }
};

struct thread_trees
{
    std::map<std::uint64_t, call_node> by_thread;
    std::size_t                        thread_count = 0;
};

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

void
flatten_tree_into_rows(std::vector<report_row>& rows, std::string_view prefix,
                       const call_node& parent, std::size_t depth)
{
    for(const auto& kid : parent.children)
    {
        auto const label = depth == 0
                               ? fmt::format("{}{}", prefix, kid.name)
                               : fmt::format("{}{:{}}|_{}", prefix, "",
                                             k_indent_per_depth * (depth - 1), kid.name);
        rows.push_back({ .label = label, .entry = &kid, .depth = depth });
        flatten_tree_into_rows(rows, prefix, kid, depth + 1);
    }
}

[[nodiscard]] std::vector<report_row>
build_rows(const thread_trees& trees, const wall_clock_settings& settings)
{
    std::vector<report_row> rows;
    std::size_t             thread_ordinal = 0;
    for(const auto& [thread_id, root] : trees.by_thread)
    {
        auto const prefix = settings.collapse_threads
                                ? std::string{ k_collapsed_prefix }
                                : fmt::format("|{}>>> ", thread_ordinal++);
        flatten_tree_into_rows(rows, prefix, root, 0);
    }
    std::erase_if(rows, [&settings](const report_row& row) {
        return row.depth >= settings.max_depth;
    });
    return rows;
}

[[nodiscard]] thread_trees
build_thread_trees(const wall_clock_settings&              settings,
                   const std::vector<wall_clock_interval>& regions,
                   const std::vector<wall_clock_interval>& gpu_activity)
{
    auto const merge = !settings.timeline;

    thread_trees                                  trees;
    std::set<std::uint64_t>                       threads;
    std::unordered_map<std::uint64_t, call_node*> launch_by_correlation_id;

    auto const tree_of = [&](std::uint64_t thread_id) -> call_node& {
        threads.insert(thread_id);
        return trees
            .by_thread[settings.collapse_threads ? k_collapsed_thread : thread_id];
    };

    std::vector<const wall_clock_interval*> sorted;
    sorted.reserve(regions.size());
    std::ranges::transform(regions, std::back_inserter(sorted),
                           [](const wall_clock_interval& region) { return &region; });
    std::ranges::sort(sorted,
                      [](const wall_clock_interval* lhs, const wall_clock_interval* rhs) {
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

        auto& parent = (settings.flat || open_regions.empty())
                           ? tree_of(region->thread_id)
                           : *open_regions.back().first;
        auto& entry  = parent.child(region->name, call_kind::cpu, merge);
        entry.data.add(region->end - region->begin);
        open_regions.emplace_back(&entry, region->end);
        if(region->correlation_id != k_no_correlation)
        {
            launch_by_correlation_id[region->correlation_id] = &entry;
        }
    }

    for(const auto& activity : gpu_activity)
    {
        auto const launch = launch_by_correlation_id.find(activity.correlation_id);
        auto&      parent = (launch != launch_by_correlation_id.end() && !settings.flat)
                                ? *launch->second
                                : tree_of(activity.thread_id);
        parent.child(activity.name, call_kind::gpu, merge)
            .data.add(activity.end - activity.begin);
    }
    trees.thread_count = threads.size();
    return trees;
}

}  // namespace

std::optional<wall_clock_unit>
find_wall_clock_unit(std::string_view name)
{
    using namespace std::chrono_literals;
    struct unit_spellings
    {
        wall_clock_unit  unit;
        std::string_view spellings;
    };
    constexpr std::array k_units{
        unit_spellings{ .unit      = { .name = "ns", .length = 1ns },
                        .spellings = "nsec ns nanosecond nanoseconds" },
        unit_spellings{ .unit      = { .name = "us", .length = 1us },
                        .spellings = "usec us microsecond microseconds" },
        unit_spellings{ .unit      = { .name = "ms", .length = 1ms },
                        .spellings = "msec ms millisecond milliseconds" },
        unit_spellings{
            .unit      = { .name   = "cs",
                           .length = std::chrono::duration<std::int64_t, std::centi>{ 1 } },
            .spellings = "csec cs centisecond centiseconds" },
        unit_spellings{
            .unit      = { .name   = "ds",
                           .length = std::chrono::duration<std::int64_t, std::deci>{ 1 } },
            .spellings = "dsec ds decisecond deciseconds" },
        unit_spellings{ .unit      = k_wall_clock_seconds,
                        .spellings = "sec s second seconds" },
        unit_spellings{ .unit      = { .name = "min", .length = 1min },
                        .spellings = "min minute minutes" },
        unit_spellings{ .unit      = { .name = "h", .length = 1h },
                        .spellings = "hr hour hours" },
    };
    if(name.empty())
    {
        return k_wall_clock_seconds;
    }
    auto const is_spelling_of = [name](const unit_spellings& entry) {
        return std::ranges::any_of(delimit(std::string{ entry.spellings }, " "),
                                   [name](const std::string& spelling) {
                                       return utility::string::equals_ignore_case(
                                           spelling, name);
                                   });
    };
    auto const found = std::ranges::find_if(k_units, is_spelling_of);
    return found == k_units.end() ? std::nullopt : std::optional{ found->unit };
}

wall_clock_report::wall_clock_report(wall_clock_settings settings) noexcept
: m_settings{ settings }
{}

void
wall_clock_report::add_region(wall_clock_interval region)
{
    validate_interval(region);
    m_regions.push_back(std::move(region));
}

void
wall_clock_report::add_gpu_activity(wall_clock_interval activity)
{
    validate_interval(activity);
    m_gpu_activity.push_back(std::move(activity));
}

bool
wall_clock_report::empty() const noexcept
{
    return m_regions.empty() && m_gpu_activity.empty();
}

std::string
wall_clock_report::text() const
{
    auto const trees = build_thread_trees(m_settings, m_regions, m_gpu_activity);
    auto       rows  = build_rows(trees, m_settings);
    if(rows.empty())
    {
        return {};
    }

    auto const max_width = static_cast<std::size_t>(std::max(m_settings.max_width, 0));
    auto const ellipsis  = m_settings.max_width > k_min_width_with_ellipsis
                               ? k_ellipsis
                               : std::string_view{};
    for(auto& row : rows)
    {
        if(max_width > 0 && row.label.size() > max_width)
        {
            row.label = fmt::format(
                "{}{}", row.label.substr(0, max_width - ellipsis.size()), ellipsis);
        }
    }

    // SUM, MEAN, MIN, MAX, VAR and STDDEV of each row in the unit, VAR in the unit
    // squared. Each value column is as wide as its widest cell.
    auto const scale = std::chrono::duration<double>{ std::chrono::seconds{ 1 } } /
                       m_settings.unit.length;
    auto values = std::vector<std::array<double, k_value_columns>>{};
    auto widths = std::array<std::size_t, k_value_columns>{};
    for(const auto& row : rows)
    {
        const auto& stats = row.entry->data;
        values.push_back({ stats.sum * scale, stats.mean * scale, stats.min * scale,
                           stats.max * scale, stats.variance() * scale * scale,
                           std::sqrt(stats.variance()) * scale });
        for(std::size_t column = 0; column < k_value_columns; ++column)
        {
            widths[column] =
                std::max({ widths[column], k_value_column_width,
                           fmt::formatted_size("{:.{}f}", values.back()[column],
                                               m_settings.precision) });
        }
    }

    auto const label_width =
        std::ranges::max(rows | std::views::transform([](const report_row& row) {
                             return row.label.size();
                         }));
    auto const header = fmt::format(
        "| {:^{}} | {:^{}} | {:^{}} | {:^{}} | {:^{}} | {:^{}} | {:^{}} | {:^{}} | "
        "{:^{}} | {:^{}} | {:<{}} | {:^{}} |\n",
        "LABEL", label_width, "COUNT", k_narrow_column_width, "DEPTH",
        k_narrow_column_width, "METRIC", k_metric.size(), "UNITS", k_narrow_column_width,
        "SUM", widths[0], "MEAN", widths[1], "MIN", widths[2], "MAX", widths[3], "VAR",
        widths[4], "STDDEV", widths[5], "% SELF", k_narrow_column_width);
    auto column_separator = header;
    std::ranges::replace_if(
        column_separator, [](char chr) { return chr != '|' && chr != '\n'; }, '-');
    auto const banner = fmt::format("|{:-<{}}|\n", "", header.size() - 3);
    auto const title  = fmt::format("|{:^{}}|\n", k_title, header.size() - 3);

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
                           values[index][column], widths[column], m_settings.precision);
        }
        auto const self =
            depth + 1 < m_settings.max_depth ? entry->self_percent() : k_percent;
        fmt::format_to(std::back_inserter(table), "{:>{}.1f} |\n", self,
                       k_narrow_column_width);
    }
    return table + banner;
}

std::string
wall_clock_report::json() const
{
    constexpr std::size_t k_single_process = 1;
    auto const trees = build_thread_trees(m_settings, m_regions, m_gpu_activity);
    auto const rows  = build_rows(trees, m_settings);
    auto const scale = std::chrono::duration<double>{ std::chrono::seconds{ 1 } } /
                       m_settings.unit.length;
    auto graph = nlohmann::json::array();
    for(const auto& [label, entry, depth] : rows)
    {
        auto const& stats  = entry->data;
        auto const  sum_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>{ stats.sum });
        graph.push_back({ { "prefix", label },
                          { "depth", depth },
                          { "entry",
                            { { "laps", stats.count },
                              { "value", sum_ns.count() },
                              { "accum", sum_ns.count() },
                              { "repr_data", stats.sum * scale },
                              { "repr_display", stats.sum * scale } } },
                          { "stats",
                            { { "sum", stats.sum * scale },
                              { "count", stats.count },
                              { "min", stats.min * scale },
                              { "max", stats.max * scale },
                              { "sqr", stats.sum_of_squares() * scale * scale },
                              { "mean", stats.mean * scale },
                              { "stddev", std::sqrt(stats.variance()) * scale } } } });
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
