// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/output/output_summary.hpp"

#include "common/units/data_size.hpp"
#include "logger/debug.hpp"

#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <map>
#include <numeric>
#include <ranges>
#include <set>
#include <system_error>
#include <utility>

namespace rocprofsys::output
{

using rocprofsys::common::units::bytes;
using rocprofsys::common::units::data_size_cast;
using rocprofsys::common::units::megabytes;

namespace
{
[[nodiscard]] std::uint64_t
get_file_size(const std::string& path)
{
    std::error_code ec;
    const auto      size = std::filesystem::file_size(path, ec);
    if(ec)
    {
        LOG_WARNING("registry: failed to read size of '{}' ({}); reporting size as 0",
                    path, ec.message());
        return 0;
    }
    return static_cast<std::uint64_t>(size);
}
}  // namespace

registry&
registry::instance()
{
    static registry inst{};
    return inst;
}

void
registry::register_file(std::string path, output_format format)
{
    artifact entry{};
    entry.pid        = getpid();
    entry.size_bytes = get_file_size(path);
    entry.path       = std::move(path);
    entry.format     = format;

    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.push_back(std::move(entry));
}

void
registry::record_process(process_metadata meta)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto [iter, inserted] = m_processes.try_emplace(meta.pid, meta);
    if(!inserted)
    {
        if(meta.ppid != k_no_pid)
        {
            iter->second.ppid = meta.ppid;
        }
        if(!meta.command.empty())
        {
            iter->second.command = std::move(meta.command);
        }
    }
}

std::vector<artifact>
registry::rows() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_files;
}

std::vector<process_metadata>
registry::processes() const
{
    std::lock_guard<std::mutex>   lock(m_mutex);
    std::vector<process_metadata> result;
    result.reserve(m_processes.size());
    for(const auto& [pid, meta] : m_processes)
    {
        result.push_back(meta);
    }
    return result;
}

void
registry::start_new_session()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.clear();
    m_processes.clear();
}

namespace
{
// Metadata is attached by pid; a pid with rows but no metadata keeps the
// default k_no_pid and is reported as missing.
[[nodiscard]] std::map<pid_t, process_node>
build_nodes(std::span<const artifact> rows, std::span<const process_metadata> processes,
            process_tree_diagnostics& diagnostics)
{
    std::map<pid_t, process_node> nodes;
    for(const auto& row : rows)
    {
        nodes[row.pid].rows.push_back(row);
    }

    for(const auto& meta : processes)
    {
        if(auto node = nodes.find(meta.pid); node != nodes.end())
        {
            node->second.meta = meta;
        }
    }

    for(auto& [pid, node] : nodes)
    {
        if(node.meta.pid == k_no_pid)
        {
            node.meta.pid = pid;
            diagnostics.missing_metadata_pids.push_back(pid);
        }
        std::ranges::sort(node.rows, std::greater{}, &artifact::size_bytes);
    }
    return nodes;
}

// Extracting from `nodes` means whatever remains afterwards was never reached
// from a root, i.e. it sits on (or hangs off) a ppid cycle.
[[nodiscard]] process_node
extract_subtree(pid_t pid, std::map<pid_t, process_node>& nodes,
                const std::map<pid_t, std::vector<pid_t>>& children_of)
{
    process_node node = std::move(nodes.extract(pid).mapped());
    if(auto iter = children_of.find(pid); iter != children_of.end())
    {
        for(const pid_t child : iter->second)
        {
            node.children.push_back(extract_subtree(child, nodes, children_of));
        }
    }
    return node;
}
}  // namespace

process_tree::process_tree(std::span<const artifact>         rows,
                           std::span<const process_metadata> processes)
{
    auto nodes = build_nodes(rows, processes, m_diagnostics);

    std::map<pid_t, std::vector<pid_t>> children_of;
    std::vector<pid_t>                  root_pids;
    for(const auto& [pid, node] : nodes)
    {
        (nodes.contains(node.meta.ppid) ? children_of[node.meta.ppid] : root_pids)
            .push_back(pid);
    }

    for(const pid_t pid : root_pids)
    {
        m_roots.push_back(extract_subtree(pid, nodes, children_of));
    }

    std::ranges::copy(std::views::keys(nodes),
                      std::back_inserter(m_diagnostics.cyclic_ppid_pids));
}

namespace
{
inline constexpr std::string_view k_unknown_value_placeholder = "?";

inline constexpr std::size_t k_format_name_width = 9;
inline constexpr std::size_t k_file_size_width   = 10;
}  // namespace

run_metadata
run_metadata::capture(std::chrono::steady_clock::time_point load_baseline)
{
    run_metadata meta{};

    const auto tt =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    meta.run_label = fmt::format("{:%FT%TZ}", fmt::gmtime(tt));

    meta.duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - load_baseline);

    return meta;
}

namespace
{
std::string
strip_terminal_control_chars(std::string_view s)
{
    std::string out{ s };
    std::erase_if(out, [](char c) {
        const auto byte = static_cast<unsigned char>(c);
        return (byte < 0x20 && byte != 0x09 && byte != 0x0A) || byte == 0x7F;
    });
    return out;
}

[[nodiscard]] std::string
summarize_command(std::string_view command)
{
    const auto program = command.substr(0, command.find_first_of(" \t"));
    return strip_terminal_control_chars(program.substr(program.find_last_of('/') + 1));
}

std::string
format_duration(std::chrono::nanoseconds dur)
{
    if(dur.count() <= 0)
    {
        return std::string{ k_unknown_value_placeholder };
    }
    const double seconds = std::chrono::duration<double>(dur).count();
    return fmt::format("{:.2f}s", seconds);
}

std::string
format_data_size(std::uint64_t size_bytes)
{
    const auto value = bytes{ static_cast<double>(size_bytes) };
    return fmt::format("{:.2f}", data_size_cast<megabytes>(value));
}

struct format_badge
{
    std::string_view glyph;
    std::string_view name;
    std::string_view viewer_hint;
};

[[nodiscard]] constexpr format_badge
badge_for(output_format format) noexcept
{
    switch(format)
    {
        case output_format::perfetto:
            return { .glyph       = "◈",
                     .name        = "perfetto",
                     .viewer_hint = "https://ui.perfetto.dev" };
        case output_format::rocpd:
            return { .glyph       = "◆",
                     .name        = "rocpd",
                     .viewer_hint = "sqlite3 / ROCm Optiq" };
        case output_format::json:
            return { .glyph = "▪", .name = "json", .viewer_hint = "jq" };
        case output_format::text:
            return { .glyph = "▪", .name = "text", .viewer_hint = "cat" };
    }
    return { .glyph = "▪", .name = "output", .viewer_hint = "" };
}

inline void
report_diagnostics(const process_tree_diagnostics& diagnostics)
{
    if(!diagnostics.missing_metadata_pids.empty())
    {
        LOG_WARNING("Output Summary: missing process metadata for pid(s) [{}]; "
                    "they render at root depth without role/parent",
                    fmt::join(diagnostics.missing_metadata_pids, ","));
    }
    if(!diagnostics.cyclic_ppid_pids.empty())
    {
        LOG_WARNING("Output Summary: pid(s) [{}] excluded — their parent-process "
                    "chain forms a cycle (corrupted metadata) instead of reaching a "
                    "real root",
                    fmt::join(diagnostics.cyclic_ppid_pids, ","));
    }
}

[[nodiscard]] std::string
format_process_label(const process_node& node, pid_t main_pid)
{
    const std::string program = summarize_command(node.meta.command);
    std::string       label   = program.empty() ? fmt::format("[{}]", node.meta.pid)
                                                : fmt::format("[{}] {}", node.meta.pid, program);
    if(node.meta.pid == main_pid)
    {
        label += "  main";
    }
    return label;
}

[[nodiscard]] std::string
display_path(const std::string& path, const std::filesystem::path& cwd)
{
    std::filesystem::path p{ path };
    if(!p.is_absolute()) p = cwd / p;
    return strip_terminal_control_chars(p.string());
}

[[nodiscard]] std::string
file_row_line(std::string_view branch, const artifact& file,
              const std::filesystem::path& cwd)
{
    const auto badge = badge_for(file.format);
    return fmt::format("{}{} {:<{}} {:>{}}  {}\n", branch, badge.glyph, badge.name,
                       k_format_name_width, format_data_size(file.size_bytes),
                       k_file_size_width, display_path(file.path, cwd));
}

[[nodiscard]] std::string
format_process_subtree(const process_node& node, std::string_view connector,
                       const std::string& prefix, pid_t main_pid,
                       const std::filesystem::path& cwd)
{
    std::string out =
        fmt::format("{}● {}\n", connector, format_process_label(node, main_pid));

    const bool has_children = !node.children.empty();
    for(std::size_t i = 0; i < node.rows.size(); ++i)
    {
        const bool last = (i + 1 == node.rows.size()) && !has_children;
        out += file_row_line(prefix + (last ? "└─ " : "├─ "), node.rows[i], cwd);
    }
    if(!node.rows.empty() && has_children) out += prefix + "│\n";

    for(std::size_t i = 0; i < node.children.size(); ++i)
    {
        const bool last = (i + 1 == node.children.size());
        if(i > 0)
        {
            out += prefix + "│\n";
        }
        out += format_process_subtree(node.children[i], last ? "└─" : "├─",
                                      prefix + (last ? "    " : "│   "), main_pid, cwd);
    }
    return out;
}

[[nodiscard]] std::string
derive_output_dir(std::span<const artifact> rows)
{
    if(rows.empty())
    {
        return std::string{ k_unknown_value_placeholder };
    }
    auto parent = std::filesystem::path{ rows.front().path }.parent_path().string();
    return parent.empty() ? std::string{ k_unknown_value_placeholder } : parent;
}

[[nodiscard]] std::string
format_header(const run_metadata& meta, std::span<const artifact> rows,
              std::size_t process_count)
{
    const auto sizes = rows | std::views::transform(&artifact::size_bytes);
    const auto total = std::accumulate(sizes.begin(), sizes.end(), std::uint64_t{ 0 });
    return fmt::format("  Run: {}   Duration: {}   Processes: {}   Total output: {}\n"
                       "  Output dir: {}\n",
                       meta.run_label.empty() ? std::string{ k_unknown_value_placeholder }
                                              : meta.run_label,
                       format_duration(meta.duration), process_count,
                       format_data_size(total), derive_output_dir(rows));
}

[[nodiscard]] std::string
build_legend(std::span<const artifact> rows)
{
    std::set<output_format> formats;
    for(const auto& row : rows)
    {
        formats.insert(row.format);
    }
    std::string legend;
    for(const output_format format : formats)
    {
        const auto badge = badge_for(format);
        if(badge.viewer_hint.empty())
        {
            continue;
        }
        if(!legend.empty())
        {
            legend += "    ";
        }
        legend += fmt::format("{} → {}", badge.name, badge.viewer_hint);
    }
    return legend;
}
}  // namespace

std::string
registry::format_summary(std::chrono::steady_clock::time_point load_baseline) const
{
    const auto rows_snapshot      = rows();
    const auto processes_snapshot = processes();
    if(rows_snapshot.empty())
    {
        return {};
    }

    const auto tree = process_tree{ rows_snapshot, processes_snapshot };
    const auto meta = run_metadata::capture(load_baseline);

    report_diagnostics(tree.diagnostics());

    std::error_code cwd_error;
    const auto      cwd = std::filesystem::current_path(cwd_error);

    std::string out =
        fmt::format("\nOutput Summary\n{}\nProcess tree\n",
                    format_header(meta, rows_snapshot, processes_snapshot.size()));
    for(const auto& root : tree.roots())
    {
        out += format_process_subtree(root, "", "  ", getpid(), cwd);
    }
    if(const auto legend = build_legend(rows_snapshot); !legend.empty())
    {
        out += fmt::format("\n  {}\n", legend);
    }
    return out;
}

}  // namespace rocprofsys::output
