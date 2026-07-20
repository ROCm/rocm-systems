// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/output/summary_writer.hpp"

#include "common/units.hpp"
#include "logger/debug.hpp"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <ctime>
#include <filesystem>
#include <iterator>
#include <numeric>
#include <ostream>
#include <set>
#include <system_error>

namespace rocprofsys::output
{

namespace
{
inline constexpr std::size_t ISO_8601_BUFFER_BYTES = 32;

inline constexpr unsigned char UTF8_CONTINUATION_MASK = 0xC0;
inline constexpr unsigned char UTF8_CONTINUATION_BITS = 0x80;

inline constexpr std::size_t FORMAT_NAME_WIDTH = 9;
inline constexpr std::size_t FILE_SIZE_WIDTH   = 10;
inline constexpr std::size_t MIN_BOX_WIDTH     = 40;

inline constexpr std::string_view GLYPH_NODE_MARKER       = "● ";
inline constexpr std::string_view GLYPH_SEPARATOR         = "│";
inline constexpr std::string_view GLYPH_FILE_BRANCH_LAST  = "└─ ";
inline constexpr std::string_view GLYPH_FILE_BRANCH_MID   = "├─ ";
inline constexpr std::string_view GLYPH_CHILD_CONN_LAST   = "└─";
inline constexpr std::string_view GLYPH_CHILD_CONN_MID    = "├─";
inline constexpr std::string_view GLYPH_CHILD_INDENT_LAST = "    ";
inline constexpr std::string_view GLYPH_CHILD_INDENT_MID  = "│   ";
inline constexpr std::string_view GLYPH_ROOT_INDENT       = "  ";
inline constexpr std::string_view GLYPH_BOX_TOP_LEFT      = "╭─ ";
inline constexpr std::string_view GLYPH_BOX_BOTTOM_LEFT   = "╰";
inline constexpr std::string_view GLYPH_BOX_LINE          = "─";
inline constexpr std::string_view GLYPH_BOX_LEFT_RAIL     = "│ ";
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

std::size_t
display_width(std::string_view text)
{
    return static_cast<std::size_t>(std::ranges::count_if(text, [](char byte) {
        return (static_cast<unsigned char>(byte) & UTF8_CONTINUATION_MASK) !=
               UTF8_CONTINUATION_BITS;
    }));
}

namespace
{
std::string
repeat_glyph(std::string_view glyph, std::size_t count)
{
    std::string out;
    out.reserve(glyph.size() * count);
    for(std::size_t index = 0; index < count; ++index)
        out.append(glyph);
    return out;
}

std::string
strip_terminal_control_chars(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for(std::size_t i = 0; i < s.size();)
    {
        const auto byte = static_cast<unsigned char>(s[i]);
        // CSI sequence: ESC [ ... <final byte in 0x40..0x7E>
        if(byte == 0x1B && i + 1 < s.size() && s[i + 1] == '[')
        {
            std::size_t j = i + 2;
            while(j < s.size())
            {
                const auto fb = static_cast<unsigned char>(s[j]);
                if(fb >= 0x40 && fb <= 0x7E)
                {
                    ++j;
                    break;
                }
                ++j;
            }
            i = j;
            continue;
        }
        // Drop other C0 controls + DEL; keep tab (0x09) and newline (0x0A)
        // so downstream layout still sees structure.
        if((byte < 0x20 && byte != 0x09 && byte != 0x0A) || byte == 0x7F)
        {
            ++i;
            continue;
        }
        out.push_back(static_cast<char>(byte));
        ++i;
    }
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
    if(dur.count() <= 0) return std::string{ common::UNKNOWN_VALUE_PLACEHOLDER };
    const double seconds = std::chrono::duration<double>(dur).count();
    return fmt::format("{:.2f}s", seconds);
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
                       FORMAT_NAME_WIDTH, common::datasize_to_string(file.size_bytes),
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
    if(rows.empty()) return std::string{ common::UNKNOWN_VALUE_PLACEHOLDER };
    auto parent = std::filesystem::path{ rows.front().path }.parent_path().string();
    return parent.empty() ? std::string{ common::UNKNOWN_VALUE_PLACEHOLDER } : parent;
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
    std::string run_line = fmt::format(
        "Run: {}   Duration: {}   Processes: {}",
        meta.run_label.empty() ? std::string{ common::UNKNOWN_VALUE_PLACEHOLDER }
                               : meta.run_label,
        format_duration(meta.duration), count_nodes(tree.roots()));
    run_line +=
        fmt::format("   Total output: {}", common::datasize_to_string(sum_sizes(rows)));

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
}  // namespace

std::size_t
box_width(std::span<const std::string> header_lines,
          std::span<const std::string> tree_lines)
{
    std::size_t width = MIN_BOX_WIDTH;
    for(const auto& line : header_lines)
        width = std::max(width, display_width(line) + 2);  // + 2 for the "│ " rail
    for(const auto& line : tree_lines)
        width = std::max(width, display_width(line) + 2);
    return width;
}

namespace
{
void
append_box(std::string& out, std::string_view title, std::span<const std::string> lines,
           std::size_t width)
{
    const std::string head      = fmt::format("{}{} ", GLYPH_BOX_TOP_LEFT, title);
    const std::size_t head_cols = display_width(head);

    const std::size_t reserve_hint =
        std::accumulate(lines.begin(), lines.end(), out.size() + head_cols + width + 4,
                        [](std::size_t acc, const std::string& line) {
                            return acc + line.size() + GLYPH_BOX_LEFT_RAIL.size() + 1;
                        });
    out.reserve(reserve_hint);

    out += head;
    if(width > head_cols) out += repeat_glyph(GLYPH_BOX_LINE, width - head_cols);
    out += "\n";
    for(const auto& line : lines)
        fmt::format_to(std::back_inserter(out), "{}{}\n", GLYPH_BOX_LEFT_RAIL, line);
    out += GLYPH_BOX_BOTTOM_LEFT;
    out += repeat_glyph(GLYPH_BOX_LINE, width - 1);
    out += "\n";
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
write_summary(std::ostream& os, const process_tree& tree, const run_metadata& meta,
              std::span<const artifact> rows)
{
    if(rows.empty()) return;

    report_diagnostics(tree.diagnostics());

    const auto header_lines = render_header(meta, tree, rows);
    const auto tree_lines   = render_tree(tree, getpid());
    const auto legend       = build_legend(rows);
    const auto width        = box_width(header_lines, tree_lines);

    std::string out = "\n";
    append_box(out, "Output Summary", header_lines, width);
    out += "\n";
    append_box(out, "Process tree", tree_lines, width);
    if(!legend.empty()) out += fmt::format("\n  {}\n", legend);

    os << out;
}

}  // namespace rocprofsys::output
