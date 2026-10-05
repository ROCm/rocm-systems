// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/output/output_summary.hpp"

#include "logger/debug.hpp"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

#include <sys/types.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <ostream>
#include <set>
#include <system_error>

namespace rocprofsys::output
{

namespace
{
inline constexpr std::size_t ISO_8601_BUFFER_BYTES = 32;

inline constexpr std::string_view UNKNOWN_VALUE_PLACEHOLDER = "?";

inline constexpr double BYTES_PER_KILOBYTE = 1000.0;
inline constexpr double BYTES_PER_MEGABYTE = 1000.0 * BYTES_PER_KILOBYTE;
inline constexpr double BYTES_PER_GIGABYTE = 1000.0 * BYTES_PER_MEGABYTE;

inline constexpr std::size_t FORMAT_NAME_WIDTH = 9;
inline constexpr std::size_t FILE_SIZE_WIDTH   = 10;

inline constexpr std::string_view GLYPH_NODE_MARKER       = "● ";
inline constexpr std::string_view GLYPH_SEPARATOR         = "│";
inline constexpr std::string_view GLYPH_FILE_BRANCH_LAST  = "└─ ";
inline constexpr std::string_view GLYPH_FILE_BRANCH_MID   = "├─ ";
inline constexpr std::string_view GLYPH_CHILD_CONN_LAST   = "└─";
inline constexpr std::string_view GLYPH_CHILD_CONN_MID    = "├─";
inline constexpr std::string_view GLYPH_CHILD_INDENT_LAST = "    ";
inline constexpr std::string_view GLYPH_CHILD_INDENT_MID  = "│   ";
inline constexpr std::string_view GLYPH_ROOT_INDENT       = "  ";
}  // namespace

run_metadata
run_metadata::capture(std::chrono::steady_clock::time_point load_baseline)
{
    run_metadata meta{};

    const auto now = std::chrono::system_clock::now();
    const auto tt  = std::chrono::system_clock::to_time_t(now);
    std::tm    utc{};
    if(::gmtime_r(&tt, &utc) != nullptr)
    {
        std::array<char, ISO_8601_BUFFER_BYTES> buf{};
        if(std::strftime(buf.data(), buf.size(), "%Y-%m-%dT%H:%M:%SZ", &utc) > 0)
            meta.run_label = buf.data();
    }

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
}  // namespace

std::string
summarize_command(std::string_view command)
{
    const std::string cleaned = strip_terminal_control_chars(command);
    if(cleaned.empty()) return {};

    const auto  token_end = cleaned.find_first_of(" \t");
    std::string program =
        (token_end == std::string::npos) ? cleaned : cleaned.substr(0, token_end);

    const auto slash = program.find_last_of('/');
    if(slash != std::string::npos) program = program.substr(slash + 1);
    return program;
}

namespace
{
std::string
format_duration(std::chrono::nanoseconds dur)
{
    if(dur.count() <= 0) return std::string{ UNKNOWN_VALUE_PLACEHOLDER };
    const double seconds = std::chrono::duration<double>(dur).count();
    return fmt::format("{:.2f}s", seconds);
}

std::string
datasize_to_string(std::uint64_t size_bytes)
{
    if(size_bytes < BYTES_PER_KILOBYTE) return fmt::format("{} B", size_bytes);
    if(size_bytes < BYTES_PER_MEGABYTE)
        return fmt::format("{:.2f} KB",
                           static_cast<double>(size_bytes) / BYTES_PER_KILOBYTE);
    if(size_bytes < BYTES_PER_GIGABYTE)
        return fmt::format("{:.2f} MB",
                           static_cast<double>(size_bytes) / BYTES_PER_MEGABYTE);
    return fmt::format("{:.2f} GB", static_cast<double>(size_bytes) / BYTES_PER_GIGABYTE);
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

void
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
process_label(const process_node& node, pid_t main_pid)
{
    const std::string program = summarize_command(node.meta.command);
    std::string       label   = program.empty() ? fmt::format("[{}]", node.meta.pid)
                                                : fmt::format("[{}] {}", node.meta.pid, program);
    if(node.meta.pid == main_pid) label += "  main";
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
    return fmt::format("{}{} {:<{}} {:>{}}  {}", branch, badge.glyph, badge.name,
                       FORMAT_NAME_WIDTH, datasize_to_string(file.size_bytes),
                       FILE_SIZE_WIDTH, display_path(file.path, cwd));
}

struct render_task
{
    const process_node* node = nullptr;  // nullptr => separator task
    std::string         connector;
    std::string         child_prefix;
};

[[nodiscard]] std::size_t
count_nodes(const std::vector<process_node>& roots)
{
    std::size_t                      count = 0;
    std::vector<const process_node*> stack;
    stack.reserve(roots.size());
    for(const auto& root : roots)
        stack.push_back(&root);
    while(!stack.empty())
    {
        const process_node* node = stack.back();
        stack.pop_back();
        ++count;
        for(const auto& child : node->children)
            stack.push_back(&child);
    }
    return count;
}

[[nodiscard]] std::string
derive_output_dir(const run_metadata& meta, std::span<const artifact> rows)
{
    if(!meta.output_dir_abs.empty()) return meta.output_dir_abs;
    if(rows.empty()) return std::string{ UNKNOWN_VALUE_PLACEHOLDER };
    auto parent = std::filesystem::path{ rows.front().path }.parent_path().string();
    return parent.empty() ? std::string{ UNKNOWN_VALUE_PLACEHOLDER } : parent;
}

void
push_root_tasks(std::vector<render_task>& stack, const process_tree& tree)
{
    for(auto it = tree.roots().rbegin(); it != tree.roots().rend(); ++it)
        stack.push_back({ &*it, std::string{}, std::string{ GLYPH_ROOT_INDENT } });
}

void
emit_file_rows(std::vector<std::string>& lines, const render_task& task,
               const process_node& node, const std::filesystem::path& cwd)
{
    const std::size_t file_count  = node.rows.size();
    const std::size_t child_count = node.children.size();
    for(std::size_t index = 0; index < file_count; ++index)
    {
        const bool last_entry = (index + 1 == file_count) && child_count == 0;
        const auto branch =
            task.child_prefix +
            std::string{ last_entry ? GLYPH_FILE_BRANCH_LAST : GLYPH_FILE_BRANCH_MID };
        lines.push_back(file_row_line(branch, node.rows[index], cwd));
    }
    if(file_count > 0 && child_count > 0)
        lines.push_back(task.child_prefix + std::string{ GLYPH_SEPARATOR });
}

// Pushes a node's children directly onto the DFS stack in reverse order (so
// popping restores left-to-right order), including the separator rows
// between them — no intermediate vector needed.
void
push_child_tasks(std::vector<render_task>& stack, const render_task& task,
                 const process_node& node)
{
    const std::size_t child_count = node.children.size();
    for(std::size_t ri = child_count; ri-- > 0;)
    {
        const bool  last_child = (ri + 1 == child_count);
        std::string child_conn =
            task.child_prefix +
            std::string{ last_child ? GLYPH_CHILD_CONN_LAST : GLYPH_CHILD_CONN_MID };
        std::string next_prefix =
            task.child_prefix +
            std::string{ last_child ? GLYPH_CHILD_INDENT_LAST : GLYPH_CHILD_INDENT_MID };
        stack.push_back(
            { &node.children[ri], std::move(child_conn), std::move(next_prefix) });
        if(ri > 0) stack.push_back({ nullptr, {}, task.child_prefix });
    }
}

std::vector<std::string>
render_header(const run_metadata& meta, const process_tree& tree,
              std::span<const artifact> rows)
{
    std::string run_line =
        fmt::format("Run: {}   Duration: {}   Processes: {}",
                    meta.run_label.empty() ? std::string{ UNKNOWN_VALUE_PLACEHOLDER }
                                           : meta.run_label,
                    format_duration(meta.duration), count_nodes(tree.roots()));
    run_line += fmt::format("   Total output: {}", datasize_to_string(sum_sizes(rows)));

    std::string dir_line = fmt::format("Output dir: {}", derive_output_dir(meta, rows));

    return { std::move(run_line), std::move(dir_line) };
}

std::vector<std::string>
render_tree(const process_tree& tree, pid_t main_pid)
{
    std::vector<std::string> lines;
    std::vector<render_task> stack;
    push_root_tasks(stack, tree);

    // Resolved once for the whole render
    std::error_code       cwd_error;
    std::filesystem::path cwd = std::filesystem::current_path(cwd_error);

    while(!stack.empty())
    {
        render_task task = std::move(stack.back());
        stack.pop_back();

        if(task.node == nullptr)
        {
            lines.push_back(task.child_prefix + std::string{ GLYPH_SEPARATOR });
            continue;
        }

        const process_node& node = *task.node;
        lines.push_back(task.connector + std::string{ GLYPH_NODE_MARKER } +
                        process_label(node, main_pid));
        emit_file_rows(lines, task, node, cwd);
        push_child_tasks(stack, task, node);
    }

    return lines;
}

[[nodiscard]] std::vector<artifact>
collect_rows(const process_tree& tree)
{
    std::vector<artifact>            rows;
    std::vector<const process_node*> stack;
    for(const auto& root : tree.roots())
        stack.push_back(&root);
    while(!stack.empty())
    {
        const process_node* node = stack.back();
        stack.pop_back();
        rows.insert(rows.end(), node->rows.begin(), node->rows.end());
        for(const auto& child : node->children)
            stack.push_back(&child);
    }
    return rows;
}

[[nodiscard]] std::string
build_legend(std::span<const artifact> rows)
{
    std::set<output_format> formats;
    for(const auto& row : rows)
        formats.insert(row.format);

    std::string legend;
    for(output_format format : formats)
    {
        const auto badge = badge_for(format);
        if(badge.viewer_hint.empty()) continue;
        if(!legend.empty()) legend += "    ";
        legend += fmt::format("{} → {}", badge.name, badge.viewer_hint);
    }
    return legend;
}
}  // namespace

void
write_summary(std::ostream& os, const process_tree& tree, const run_metadata& meta)
{
    const auto rows = collect_rows(tree);
    if(rows.empty()) return;

    report_diagnostics(tree.diagnostics());

    const auto header_lines = render_header(meta, tree, rows);
    const auto tree_lines   = render_tree(tree, getpid());
    const auto legend       = build_legend(rows);

    std::string out = "\nOutput Summary\n";
    for(const auto& line : header_lines)
        out += "  " + line + "\n";
    out += "\nProcess tree\n";
    for(const auto& line : tree_lines)
        out += "  " + line + "\n";
    if(!legend.empty()) out += fmt::format("\n  {}\n", legend);

    os << out;
}

}  // namespace rocprofsys::output
