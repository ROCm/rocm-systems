// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/output/output_summary.hpp"
#include "core/state.hpp"

#include "common/units/data_size.hpp"
#include "logger/debug.hpp"

#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/fmt.h>

#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <numeric>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
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
    static registry s_inst{};
    return s_inst;
}

void
registry::register_file(std::string path)
{
    artifact entry{};
    entry.pid        = getpid();
    entry.size_bytes = get_file_size(path);
    entry.path       = std::move(path);

    const auto thread_state_guard = state::thread::scoped(state::thread::Internal);
    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.push_back(std::move(entry));
}

void
registry::record_process(process_metadata meta)
{
    const auto thread_state_guard = state::thread::scoped(state::thread::Internal);
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
    const auto thread_state_guard = state::thread::scoped(state::thread::Internal);
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_files;
}

std::vector<process_metadata>
registry::processes() const
{
    const auto thread_state_guard = state::thread::scoped(state::thread::Internal);
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
    const auto thread_state_guard = state::thread::scoped(state::thread::Internal);
    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.clear();
    m_processes.clear();
}

namespace
{
inline constexpr std::string_view k_unknown_value_placeholder = "?";

inline constexpr std::size_t k_format_name_width = 9;
inline constexpr std::size_t k_file_size_width   = 10;

struct process_node
{
    process_metadata          meta;
    std::vector<artifact>     rows;
    std::vector<process_node> children;
};

// Metadata is attached by pid; a pid with rows but no metadata keeps the
// default k_no_pid and is logged as missing.
[[nodiscard]] std::map<pid_t, process_node>
build_nodes(std::span<const artifact> rows, std::span<const process_metadata> processes)
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
            LOG_WARNING("Output Summary: missing process metadata for pid {}; it "
                        "renders at root depth without role/parent",
                        pid);
        }
        std::ranges::sort(node.rows, std::greater{}, &artifact::size_bytes);
    }
    return nodes;
}

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

run_metadata::run_metadata(std::chrono::nanoseconds elapsed)
: duration{ elapsed }
{
    const auto start_time =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    run_label = fmt::format("{:%FT%TZ}", fmt::gmtime(start_time));
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

struct file_type
{
    std::string_view extension;
    std::string_view name;
    std::string_view viewer_hint;
};

[[nodiscard]] const file_type&
extract_file_type(const std::string& path)
{
    static constexpr std::array<file_type, 4> k_file_types{ {
        { ".pftrace", "perfetto", "https://ui.perfetto.dev" },
        { ".db", "rocpd", "sqlite3 / ROCm Optiq" },
        { ".txt", "text", "cat" },
        { ".json", "json", "jq" },
    } };

    const auto ext = std::filesystem::path{ path }.extension().string();
    for(const auto& type : k_file_types)
    {
        if(type.extension == ext)
        {
            return type;
        }
    }
    throw std::runtime_error(fmt::format(
        "Unrecognized output file type! Missing implementation for: '{}'", ext));
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
    const auto& type = extract_file_type(file.path);
    return fmt::format("{}{:<{}} {:>{}}  {}\n", branch, type.name, k_format_name_width,
                       format_data_size(file.size_bytes), k_file_size_width,
                       display_path(file.path, cwd));
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
    std::set<std::string_view> seen;
    std::string                legend;
    for(const auto& row : rows)
    {
        const auto& type = extract_file_type(row.path);
        if(!seen.insert(type.name).second)
        {
            continue;
        }
        if(!legend.empty())
        {
            legend += "    ";
        }
        legend += fmt::format("{} → {}", type.name, type.viewer_hint);
    }
    return legend;
}
}  // namespace

std::string
registry::format_summary() const
{
    const auto rows_snapshot      = rows();
    const auto processes_snapshot = processes();
    if(rows_snapshot.empty())
    {
        return {};
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - m_start_time);
    const run_metadata meta{ elapsed };

    // Extracting every reachable root leaves whatever remains in `nodes`
    // unreachable, i.e. sitting on (or hanging off) a ppid cycle.
    auto nodes = build_nodes(rows_snapshot, processes_snapshot);

    std::map<pid_t, std::vector<pid_t>> children_of;
    std::vector<pid_t>                  root_pids;
    for(const auto& [pid, node] : nodes)
    {
        (nodes.contains(node.meta.ppid) ? children_of[node.meta.ppid] : root_pids)
            .push_back(pid);
    }

    std::vector<process_node> roots;
    roots.reserve(root_pids.size());
    for(const pid_t pid : root_pids)
    {
        roots.push_back(extract_subtree(pid, nodes, children_of));
    }

    for(const pid_t pid : std::views::keys(nodes))
    {
        LOG_WARNING("Output Summary: pid {} excluded — its parent-process chain "
                    "forms a cycle (corrupted metadata) instead of reaching a real "
                    "root",
                    pid);
    }

    std::error_code cwd_error;
    const auto      cwd = std::filesystem::current_path(cwd_error);

    std::string out =
        fmt::format("\nOutput Summary\n{}\nProcess tree\n",
                    format_header(meta, rows_snapshot, processes_snapshot.size()));
    for(const auto& root : roots)
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
