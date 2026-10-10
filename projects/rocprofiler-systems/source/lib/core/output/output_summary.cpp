// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/output/output_summary.hpp"
#include "core/state.hpp"

#include "common/units/data_size.hpp"
#include "logger/debug.hpp"

#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/fmt.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iterator>
#include <map>
#include <ranges>
#include <set>
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
inline constexpr std::size_t k_format_name_width = 9;
inline constexpr std::size_t k_file_size_width   = 7;

struct file_type
{
    std::string_view extension;
    std::string_view name;
    std::string_view viewer_hint;
};

inline constexpr std::array<file_type, 4> k_file_types{ {
    { .extension   = ".pftrace",
      .name        = "perfetto",
      .viewer_hint = "https://ui.perfetto.dev" },
    { .extension = ".db", .name = "rocpd", .viewer_hint = "sqlite3 / ROCm Optiq" },
    { .extension = ".txt", .name = "text", .viewer_hint = "cat" },
    { .extension = ".json", .name = "json", .viewer_hint = "jq" },
} };

using rows_by_pid_t = std::map<pid_t, std::vector<std::string>>;
using processes_t   = std::unordered_map<pid_t, process_metadata>;

[[nodiscard]] megabytes
to_megabytes(std::uint64_t size_bytes)
{
    return data_size_cast<megabytes>(bytes{ static_cast<double>(size_bytes) });
}

[[nodiscard]] std::string
strip_control_chars(std::string text)
{
    std::erase_if(text, [](char character) {
        return std::iscntrl(static_cast<unsigned char>(character)) != 0;
    });
    return text;
}

struct tree_context
{
    const rows_by_pid_t&                rows_by_pid;
    const processes_t&                  processes;
    std::map<pid_t, std::vector<pid_t>> children;
    std::set<pid_t>                     visited;
    pid_t                               main_pid;
    std::string                         out;
};

[[nodiscard]] std::string
format_process_label(const tree_context& ctx, pid_t pid)
{
    std::string label = fmt::format("[{}]", pid);
    const auto  meta  = ctx.processes.find(pid);
    if(meta != ctx.processes.end())
    {
        const std::string_view command = meta->second.command;
        const auto             program =
            std::filesystem::path{ command.substr(0, command.find_first_of(" \t")) };
        if(!program.empty())
        {
            label += ' ' + strip_control_chars(program.filename().string());
        }
    }

    if(pid == ctx.main_pid)
    {
        label += "  main";
    }
    else if(meta != ctx.processes.end() && !ctx.rows_by_pid.contains(meta->second.ppid))
    {
        label += fmt::format("  (child of [{}])", meta->second.ppid);
    }
    return label;
}

void
append_process_node(tree_context& ctx, pid_t pid, const std::string& line_prefix,
                    const std::string& child_prefix)
{
    if(!ctx.visited.insert(pid).second)
    {
        return;
    }
    ctx.out += fmt::format("{}● {}\n", line_prefix, format_process_label(ctx, pid));

    const auto& rows       = ctx.rows_by_pid.at(pid);
    const auto& children   = ctx.children[pid];
    const auto  item_count = rows.size() + children.size();
    std::size_t index      = 0;
    for(const auto& row : rows)
    {
        const bool is_last = ++index == item_count;
        ctx.out += fmt::format("{}{} {}\n", child_prefix, is_last ? "└─" : "├─", row);
    }
    for(const pid_t child : children)
    {
        const bool is_last = ++index == item_count;
        append_process_node(ctx, child, child_prefix + (is_last ? "└─" : "├─"),
                            child_prefix + (is_last ? "    " : "│   "));
    }
}

[[nodiscard]] std::string
format_process_tree(const rows_by_pid_t& rows_by_pid, const processes_t& processes,
                    pid_t main_pid)
{
    tree_context       ctx{ .rows_by_pid = rows_by_pid,
                            .processes   = processes,
                            .main_pid    = main_pid };
    std::vector<pid_t> roots;
    for(const pid_t pid : std::views::keys(rows_by_pid))
    {
        const auto meta = processes.find(pid);
        auto&      siblings =
            meta != processes.end() && rows_by_pid.contains(meta->second.ppid)
                     ? ctx.children[meta->second.ppid]
                     : roots;
        siblings.push_back(pid);
    }
    if(const auto main = std::ranges::find(roots, main_pid); main != roots.end())
    {
        std::rotate(roots.begin(), main, std::next(main));
    }

    for(const pid_t root : roots)
    {
        append_process_node(ctx, root, "", "  ");
    }
    // Processes on a ppid cycle have no root to hang from.
    for(const pid_t pid : std::views::keys(rows_by_pid))
    {
        append_process_node(ctx, pid, "", "  ");
    }
    return ctx.out;
}
}  // namespace

registry&
registry::instance()
{
    static registry s_inst{};
    return s_inst;
}

void
registry::register_file(std::string path, pid_t pid)
{
    std::error_code error;
    std::uint64_t   size_bytes = std::filesystem::file_size(path, error);
    if(error)
    {
        LOG_WARNING("registry: failed to read size of '{}' ({}); reporting size as 0",
                    path, error.message());
        size_bytes = 0;
    }
    if(const auto absolute = std::filesystem::absolute(path, error); !error)
    {
        path = absolute.string();
    }
    path = strip_control_chars(std::move(path));

    const auto thread_state_guard = state::thread::scoped(state::thread::Internal);
    const std::lock_guard<std::mutex> lock(m_mutex);
    m_files.push_back(
        artifact{ .path = std::move(path), .pid = pid, .size_bytes = size_bytes });
}

void
registry::record_process(process_metadata meta)
{
    const auto thread_state_guard = state::thread::scoped(state::thread::Internal);
    const std::lock_guard<std::mutex> lock(m_mutex);
    const pid_t                       pid = meta.pid;
    m_processes.insert_or_assign(pid, std::move(meta));
}

void
registry::start_new_session()
{
    const auto thread_state_guard = state::thread::scoped(state::thread::Internal);
    const std::lock_guard<std::mutex> lock(m_mutex);
    m_files.clear();
    m_processes.clear();
    m_start_time = std::chrono::steady_clock::now();
}

std::string
registry::format_summary() const
{
    std::vector<artifact>                                files;
    processes_t                                          processes;
    std::optional<std::chrono::steady_clock::time_point> start_time;
    {
        const auto thread_state_guard = state::thread::scoped(state::thread::Internal);
        const std::lock_guard<std::mutex> lock(m_mutex);
        files      = m_files;
        processes  = m_processes;
        start_time = m_start_time;
    }
    if(files.empty())
    {
        return {};
    }

    std::ranges::sort(files, std::greater{}, &artifact::size_bytes);

    rows_by_pid_t              rows_by_pid;
    std::uint64_t              total_bytes = 0;
    std::set<std::string_view> seen_types;
    std::string                legend;
    for(const auto& file : files)
    {
        const auto ext  = std::filesystem::path{ file.path }.extension().string();
        const auto type = std::ranges::find(k_file_types, std::string_view{ ext },
                                            &file_type::extension);
        if(type == k_file_types.end())
        {
            throw std::runtime_error(fmt::format(
                "Unrecognized output file type! Missing implementation for: '{}'", ext));
        }
        rows_by_pid[file.pid].push_back(
            fmt::format("{:<{}} {:>{}.2f}  {}", type->name, k_format_name_width,
                        to_megabytes(file.size_bytes), k_file_size_width, file.path));
        total_bytes += file.size_bytes;
        if(seen_types.insert(type->name).second)
        {
            legend += fmt::format("{}{} → {}", legend.empty() ? "" : "    ", type->name,
                                  type->viewer_hint);
        }
    }

    std::string out = fmt::format("\nOutput Summary\n  Run: {:%FT%TZ}",
                                  fmt::gmtime(std::time(nullptr)));
    if(start_time)
    {
        const std::chrono::duration<double> elapsed =
            std::chrono::steady_clock::now() - *start_time;
        out += fmt::format("   Duration: {:.2f}s", elapsed.count());
    }
    out += fmt::format(
        "   Processes: {}   Total output: {:.2f}\n  Output dir: {}\n\nProcess tree\n",
        rows_by_pid.size(), to_megabytes(total_bytes),
        std::filesystem::path{ files.front().path }.parent_path().string());
    out += format_process_tree(rows_by_pid, processes, getpid());
    out += fmt::format("\n  {}\n", legend);
    return out;
}

}  // namespace rocprofsys::output
