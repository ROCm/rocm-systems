// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/wall_clock_report.hpp"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <ranges>
#include <ratio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::trace_cache
{
namespace
{
constexpr std::size_t      k_indent_per_depth = 2;
constexpr std::string_view k_metric           = "wall_clock";
constexpr std::string_view k_units            = "sec";
constexpr std::string_view k_title          = "REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)";
constexpr std::uint64_t    k_no_correlation = 0;
constexpr std::size_t      k_narrow_column_width = 6;
constexpr std::size_t      k_value_column_width  = 8;
constexpr double           k_percent             = 100.0;

/// Running statistics of one call-tree node, in seconds (Welford's online algorithm).
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

struct call_node
{
    std::string          name;
    bool                 is_gpu = false;
    timing_stats         data;
    std::list<call_node> children;

    /// Returns the child called `child_name`. With `merge`, an existing child is
    /// reused (linear search: fan-out per node is small, so a map would only lose
    /// first-seen ordering); otherwise a new one is always appended.
    call_node& child(std::string_view child_name, bool gpu, bool merge)
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
        return children.emplace_back(
            call_node{ .name = std::string{ child_name }, .is_gpu = gpu });
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
            cpu_children += kid.is_gpu ? 0.0 : kid.data.sum;
        }
        return k_percent * (data.sum - cpu_children) / data.sum;
    }
};

void
validate_and_append(std::vector<wall_clock_interval>& into, wall_clock_interval interval)
{
    if(interval.end < interval.begin)
    {
        throw std::runtime_error(
            fmt::format("wall_clock report: '{}' ends ({}) before it begins ({})",
                        interval.name, interval.end.count(), interval.begin.count()));
    }
    into.push_back(std::move(interval));
}

struct report_row
{
    std::string      label;
    const call_node* entry;
    std::size_t      depth;
};

void
flatten_tree_into_rows(std::vector<report_row>& rows, std::size_t thread_ordinal,
                       const call_node& parent, std::size_t depth)
{
    auto const prefix = fmt::format("|{}>>> ", thread_ordinal);
    for(const auto& kid : parent.children)
    {
        auto const label = depth == 0
                               ? fmt::format("{}{}", prefix, kid.name)
                               : fmt::format("{}{:{}}|_{}", prefix, "",
                                             k_indent_per_depth * (depth - 1), kid.name);
        rows.push_back({ .label = label, .entry = &kid, .depth = depth });
        flatten_tree_into_rows(rows, thread_ordinal, kid, depth + 1);
    }
}

std::vector<report_row>
rows_of(const std::map<std::uint64_t, call_node>& thread_trees)
{
    std::vector<report_row> rows;
    std::size_t             thread_ordinal = 0;
    for(const auto& [thread_id, root] : thread_trees)
    {
        flatten_tree_into_rows(rows, thread_ordinal++, root, 0);
    }
    return rows;
}

std::map<std::uint64_t, call_node>
build_thread_trees(const wall_clock_layout&                layout,
                   std::vector<wall_clock_interval>&       regions,
                   const std::vector<wall_clock_interval>& gpu_activity)
{
    std::ranges::sort(regions,
                      [](const wall_clock_interval& lhs, const wall_clock_interval& rhs) {
                          return std::tie(lhs.thread_id, lhs.begin, rhs.end) <
                                 std::tie(rhs.thread_id, rhs.begin, lhs.end);
                      });

    auto const merge = !layout.timeline;

    std::map<std::uint64_t, call_node>            thread_trees;
    std::unordered_map<std::uint64_t, call_node*> launch_by_correlation_id;
    // Open regions of the current thread, outermost first
    std::vector<std::pair<call_node*, wall_clock_nanoseconds>> open_regions;
    auto open_regions_thread = std::optional<std::uint64_t>{};
    for(const auto& region : regions)
    {
        if(open_regions_thread != region.thread_id)
        {
            open_regions.clear();
            open_regions_thread = region.thread_id;
        }
        // Ends only shrink going down the stack, so the finished regions are a suffix.
        std::erase_if(open_regions,
                      [&](const auto& open) { return region.begin >= open.second; });
        // Flat puts every row directly under its thread
        auto& parent = (layout.flat || open_regions.empty())
                           ? thread_trees[region.thread_id]
                           : *open_regions.back().first;
        auto& entry  = parent.child(region.name, false, merge);
        entry.data.add(region.end - region.begin);
        open_regions.emplace_back(&entry, region.end);
        if(region.correlation_id != k_no_correlation)
        {
            launch_by_correlation_id[region.correlation_id] = &entry;
        }
    }

    for(const auto& activity : gpu_activity)
    {
        auto const launch = launch_by_correlation_id.find(activity.correlation_id);
        auto&      parent = (launch != launch_by_correlation_id.end() && !layout.flat)
                                ? *launch->second
                                : thread_trees[activity.thread_id];
        parent.child(activity.name, true, merge).data.add(activity.end - activity.begin);
    }
    return thread_trees;
}

}  // namespace

struct wall_clock_thread_trees
{
    std::map<std::uint64_t, call_node> by_thread;
};

wall_clock_report::wall_clock_report(wall_clock_layout layout)
: m_layout{ layout }
{}

void
wall_clock_report::add_region(wall_clock_interval region)
{
    validate_and_append(m_regions, std::move(region));
}

void
wall_clock_report::add_gpu_activity(wall_clock_interval activity)
{
    validate_and_append(m_gpu_activity, std::move(activity));
}

bool
wall_clock_report::empty() const noexcept
{
    return m_trees ? m_trees->by_thread.empty()
                   : m_regions.empty() && m_gpu_activity.empty();
}

const wall_clock_thread_trees&
wall_clock_report::trees()
{
    if(!m_trees)
    {
        m_trees = std::make_shared<const wall_clock_thread_trees>(wall_clock_thread_trees{
            build_thread_trees(m_layout, m_regions, m_gpu_activity) });
        std::vector<wall_clock_interval>{}.swap(m_regions);
        std::vector<wall_clock_interval>{}.swap(m_gpu_activity);
    }
    return *m_trees;
}

void
wall_clock_report::write_text(std::ostream& out)
{
    auto const rows = rows_of(trees().by_thread);
    if(rows.empty())
    {
        return;
    }
    auto const label_width =
        std::ranges::max(rows | std::views::transform([](const report_row& line) {
                             return line.label.size();
                         }));
    auto const header = fmt::format(
        "| {:^{}} | {:^{}} | {:^{}} | {:^{}} | {:^{}} | {:^{}} | {:^{}} | {:^{}} | "
        "{:^{}} | {:^{}} | {:<{}} | {:^{}} |\n",
        "LABEL", label_width, "COUNT", k_narrow_column_width, "DEPTH",
        k_narrow_column_width, "METRIC", k_metric.size(), "UNITS", k_narrow_column_width,
        "SUM", k_value_column_width, "MEAN", k_value_column_width, "MIN",
        k_value_column_width, "MAX", k_value_column_width, "VAR", k_value_column_width,
        "STDDEV", k_value_column_width, "% SELF", k_narrow_column_width);

    auto column_separator = header;
    std::ranges::replace_if(
        column_separator, [](char chr) { return chr != '|' && chr != '\n'; }, '-');
    auto const banner = fmt::format("|{:-<{}}|\n", "", header.size() - 3);
    auto const title  = fmt::format("|{:^{}}|\n", k_title, header.size() - 3);

    out << banner << title << banner << header << column_separator;
    for(const auto& [label, entry, depth] : rows)
    {
        auto const& values = entry->data;
        out << fmt::format(
            "| {:<{}} | {:>{}} | {:>{}} | {:<{}} | {:<{}} | {:>{}.6f} | {:>{}.6f} | "
            "{:>{}.6f} | {:>{}.6f} | {:>{}.6f} | {:>{}.6f} | {:>{}.1f} |\n",
            label, label_width, values.count, k_narrow_column_width, depth,
            k_narrow_column_width, k_metric, k_metric.size(), k_units,
            k_narrow_column_width, values.sum, k_value_column_width, values.mean,
            k_value_column_width, values.min, k_value_column_width, values.max,
            k_value_column_width, values.variance(), k_value_column_width,
            std::sqrt(values.variance()), k_value_column_width, entry->self_percent(),
            k_narrow_column_width);
    }
    out << banner;
}

void
wall_clock_report::write_json(std::ostream& out, std::int32_t rank, std::int32_t mpi_size)
{
    const auto&           thread_trees     = trees().by_thread;
    auto const            rows             = rows_of(thread_trees);
    auto const            thread_count     = thread_trees.size();
    constexpr std::size_t k_single_process = 1;
    auto                  graph            = nlohmann::json::array();
    for(const auto& [label, entry, depth] : rows)
    {
        auto const& values = entry->data;
        auto const  sum_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>{ values.sum });
        graph.push_back({ { "prefix", label },
                          { "depth", depth },
                          { "entry",
                            { { "laps", values.count },
                              { "value", sum_ns.count() },
                              { "accum", sum_ns.count() },
                              { "repr_data", values.sum },
                              { "repr_display", values.sum } } },
                          { "stats",
                            { { "sum", values.sum },
                              { "count", values.count },
                              { "min", values.min },
                              { "max", values.max },
                              { "sqr", values.sum_of_squares() },
                              { "mean", values.mean },
                              { "stddev", std::sqrt(values.variance()) } } } });
    }

    out << nlohmann::json{
        { "timemory",
          { { "wall_clock",
              { { "type", k_metric },
                { "description", "Real-clock timer (i.e. wall-clock timer)" },
                { "unit_value", std::nano::den },
                { "unit_repr", k_units },
                { "thread_scope_only", false },
                { "thread_count", thread_count },
                { "mpi_size", mpi_size },
                { "upcxx_size", k_single_process },
                { "process_count", k_single_process },
                { "num_ranks", k_single_process },
                { "concurrency", thread_count },
                { "ranks",
                  { { { "rank", rank },
                      { "graph_size", rows.size() },
                      { "graph", graph } } } } } } } }
    }.dump(4);
}

}  // namespace rocprofsys::trace_cache
