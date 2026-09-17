// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/output/summary_writer.hpp"

#include "core/output/artifact.hpp"
#include "core/output/process_tree.hpp"
#include "logger/debug.hpp"

#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iterator>
#include <numeric>
#include <ostream>
#include <set>
#include <span>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace rocprofsys::output
{

namespace
{
inline constexpr std::size_t k_iso8601_buffer_bytes = 32;

inline constexpr unsigned char k_utf8_continuation_mask = 0xC0;
inline constexpr unsigned char k_utf8_continuation_bits = 0x80;

inline constexpr std::string_view k_unknown_value_placeholder = "?";

inline constexpr double k_bytes_per_kilobyte = 1000.0;
inline constexpr double k_bytes_per_megabyte = 1000.0 * k_bytes_per_kilobyte;
inline constexpr double k_bytes_per_gigabyte = 1000.0 * k_bytes_per_megabyte;

inline constexpr std::size_t k_format_name_width = 9;
inline constexpr std::size_t k_file_size_width   = 10;
inline constexpr std::size_t k_min_box_width     = 40;

inline constexpr std::string_view k_glyph_node_marker       = "● ";
inline constexpr std::string_view k_glyph_separator         = "│";
inline constexpr std::string_view k_glyph_file_branch_last  = "└─ ";
inline constexpr std::string_view k_glyph_file_branch_mid   = "├─ ";
inline constexpr std::string_view k_glyph_child_conn_last   = "└─";
inline constexpr std::string_view k_glyph_child_conn_mid    = "├─";
inline constexpr std::string_view k_glyph_child_indent_last = "    ";
inline constexpr std::string_view k_glyph_child_indent_mid  = "│   ";
inline constexpr std::string_view k_glyph_root_indent       = "  ";
inline constexpr std::string_view k_glyph_box_top_left      = "╭─ ";
inline constexpr std::string_view k_glyph_box_bottom_left   = "╰";
inline constexpr std::string_view k_glyph_box_line          = "─";
inline constexpr std::string_view k_glyph_box_left_rail     = "│ ";

inline constexpr unsigned char k_escape_byte        = 0x1B;
inline constexpr unsigned char k_csi_final_byte_min = 0x40;
inline constexpr unsigned char k_csi_final_byte_max = 0x7E;
inline constexpr unsigned char k_c0_control_max     = 0x20;
inline constexpr unsigned char k_tab_byte           = 0x09;
inline constexpr unsigned char k_line_feed_byte     = 0x0A;
inline constexpr unsigned char k_del_byte           = 0x7F;
}  // namespace

run_metadata
run_metadata::capture(std::chrono::steady_clock::time_point load_baseline)
{
    run_metadata meta{};

    const auto now          = std::chrono::system_clock::now();
    const auto time_t_value = std::chrono::system_clock::to_time_t(now);
    std::tm    utc{};
    if(::gmtime_r(&time_t_value, &utc) != nullptr)
    {
        std::array<char, k_iso8601_buffer_bytes> buf{};
        if(std::strftime(buf.data(), buf.size(), "%Y-%m-%dT%H:%M:%SZ", &utc) > 0)
        {
            meta.run_label = buf.data();
        }
    }

    meta.duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - load_baseline);

    return meta;
}

std::size_t
display_width(std::string_view text)
{
    return static_cast<std::size_t>(std::ranges::count_if(text, [](char byte) {
        return (static_cast<unsigned char>(byte) & k_utf8_continuation_mask) !=
               k_utf8_continuation_bits;
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
    {
        out.append(glyph);
    }
    return out;
}

std::string
strip_terminal_control_chars(std::string_view text)
{
    std::string out;
    out.reserve(text.size());
    for(std::size_t i = 0; i < text.size();)
    {
        const auto byte = static_cast<unsigned char>(text[i]);
        // CSI sequence: ESC [ ... <final byte in
        // k_csi_final_byte_min..k_csi_final_byte_max>
        if(byte == k_escape_byte && i + 1 < text.size() && text[i + 1] == '[')
        {
            std::size_t scan_index = i + 2;
            while(scan_index < text.size())
            {
                const auto final_byte = static_cast<unsigned char>(text[scan_index]);
                if(final_byte >= k_csi_final_byte_min &&
                   final_byte <= k_csi_final_byte_max)
                {
                    ++scan_index;
                    break;
                }
                ++scan_index;
            }
            i = scan_index;
            continue;
        }
        // Drop other C0 controls + DEL; keep tab and newline so downstream
        // layout still sees structure.
        if((byte < k_c0_control_max && byte != k_tab_byte && byte != k_line_feed_byte) ||
           byte == k_del_byte)
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
    if(cleaned.empty())
    {
        return {};
    }

    const auto  token_end = cleaned.find_first_of(" \t");
    std::string program =
        (token_end == std::string::npos) ? cleaned : cleaned.substr(0, token_end);

    const auto slash = program.find_last_of('/');
    if(slash != std::string::npos)
    {
        program = program.substr(slash + 1);
    }
    return program;
}

namespace
{
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
datasize_to_string(std::uint64_t size_bytes)
{
    if(static_cast<double>(size_bytes) < k_bytes_per_kilobyte)
    {
        return fmt::format("{} B", size_bytes);
    }
    if(static_cast<double>(size_bytes) < k_bytes_per_megabyte)
    {
        return fmt::format("{:.2f} KB",
                           static_cast<double>(size_bytes) / k_bytes_per_kilobyte);
    }
    if(static_cast<double>(size_bytes) < k_bytes_per_gigabyte)
    {
        return fmt::format("{:.2f} MB",
                           static_cast<double>(size_bytes) / k_bytes_per_megabyte);
    }
    return fmt::format("{:.2f} GB",
                       static_cast<double>(size_bytes) / k_bytes_per_gigabyte);
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
    if(node.meta.pid == main_pid)
    {
        label += "  main";
    }
    return label;
}

[[nodiscard]] std::string
display_path(const std::string& path, const std::filesystem::path& cwd)
{
    std::filesystem::path resolved{ path };
    if(!resolved.is_absolute())
    {
        resolved = cwd / resolved;
    }
    return strip_terminal_control_chars(resolved.string());
}

[[nodiscard]] std::string
file_row_line(std::string_view branch, const artifact& file,
              const std::filesystem::path& cwd)
{
    const auto badge = badge_for(file.format);
    return fmt::format("{}{} {:<{}} {:>{}}  {}", branch, badge.glyph, badge.name,
                       k_format_name_width, datasize_to_string(file.size_bytes),
                       k_file_size_width, display_path(file.path, cwd));
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
    {
        stack.push_back(&root);
    }
    while(!stack.empty())
    {
        const process_node* node = stack.back();
        stack.pop_back();
        ++count;
        for(const auto& child : node->children)
        {
            stack.push_back(&child);
        }
    }
    return count;
}

[[nodiscard]] std::string
derive_output_dir(const run_metadata& meta, std::span<const artifact> rows)
{
    if(!meta.output_dir_abs.empty())
    {
        return meta.output_dir_abs;
    }
    if(rows.empty())
    {
        return std::string{ k_unknown_value_placeholder };
    }
    const auto parent = std::filesystem::path{ rows.front().path }.parent_path().string();
    return parent.empty() ? std::string{ k_unknown_value_placeholder } : parent;
}

void
push_root_tasks(std::vector<render_task>& stack, const process_tree& tree)
{
    for(auto root_it = tree.roots().rbegin(); root_it != tree.roots().rend(); ++root_it)
    {
        stack.push_back({ .node         = &*root_it,
                          .connector    = std::string{},
                          .child_prefix = std::string{ k_glyph_root_indent } });
    }
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
            task.child_prefix + std::string{ last_entry ? k_glyph_file_branch_last
                                                        : k_glyph_file_branch_mid };
        lines.push_back(file_row_line(branch, node.rows[index], cwd));
    }
    if(file_count > 0 && child_count > 0)
    {
        lines.push_back(task.child_prefix + std::string{ k_glyph_separator });
    }
}

// Pushes a node's children directly onto the DFS stack in reverse order (so
// popping restores left-to-right order), including the separator rows
// between them — no intermediate vector needed.
void
push_child_tasks(std::vector<render_task>& stack, const render_task& task,
                 const process_node& node)
{
    const std::size_t child_count = node.children.size();
    for(std::size_t reverse_index = child_count; reverse_index-- > 0;)
    {
        const bool  last_child = (reverse_index + 1 == child_count);
        std::string child_conn =
            task.child_prefix +
            std::string{ last_child ? k_glyph_child_conn_last : k_glyph_child_conn_mid };
        std::string next_prefix =
            task.child_prefix + std::string{ last_child ? k_glyph_child_indent_last
                                                        : k_glyph_child_indent_mid };
        stack.push_back({ .node         = &node.children[reverse_index],
                          .connector    = std::move(child_conn),
                          .child_prefix = std::move(next_prefix) });
        if(reverse_index > 0)
        {
            stack.push_back(
                { .node = nullptr, .connector = {}, .child_prefix = task.child_prefix });
        }
    }
}

std::vector<std::string>
render_header(const run_metadata& meta, const process_tree& tree,
              std::span<const artifact> rows)
{
    std::string run_line =
        fmt::format("Run: {}   Duration: {}   Processes: {}",
                    meta.run_label.empty() ? std::string{ k_unknown_value_placeholder }
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
    std::error_code             cwd_error;
    const std::filesystem::path cwd = std::filesystem::current_path(cwd_error);

    while(!stack.empty())
    {
        const render_task task = std::move(stack.back());
        stack.pop_back();

        if(task.node == nullptr)
        {
            lines.push_back(task.child_prefix + std::string{ k_glyph_separator });
            continue;
        }

        const process_node& node = *task.node;
        lines.push_back(task.connector + std::string{ k_glyph_node_marker } +
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
    std::size_t width = k_min_box_width;
    for(const auto& line : header_lines)
    {
        width = std::max(width, display_width(line) + 2);  // + 2 for the "│ " rail
    }
    for(const auto& line : tree_lines)
    {
        width = std::max(width, display_width(line) + 2);
    }
    return width;
}

namespace
{
void
append_box(std::string& out, std::string_view title, std::span<const std::string> lines,
           std::size_t width)
{
    const std::string head      = fmt::format("{}{} ", k_glyph_box_top_left, title);
    const std::size_t head_cols = display_width(head);

    const std::size_t reserve_hint =
        std::accumulate(lines.begin(), lines.end(), out.size() + head_cols + width + 4,
                        [](std::size_t acc, const std::string& line) {
                            return acc + line.size() + k_glyph_box_left_rail.size() + 1;
                        });
    out.reserve(reserve_hint);

    out += head;
    if(width > head_cols)
    {
        out += repeat_glyph(k_glyph_box_line, width - head_cols);
    }
    out += "\n";
    for(const auto& line : lines)
    {
        fmt::format_to(std::back_inserter(out), "{}{}\n", k_glyph_box_left_rail, line);
    }
    out += k_glyph_box_bottom_left;
    out += repeat_glyph(k_glyph_box_line, width - 1);
    out += "\n";
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

void
write_summary(std::ostream& stream, const process_tree& tree, const run_metadata& meta,
              std::span<const artifact> rows)
{
    if(rows.empty())
    {
        return;
    }

    report_diagnostics(tree.diagnostics());

    const auto header_lines = render_header(meta, tree, rows);
    const auto tree_lines   = render_tree(tree, getpid());
    const auto legend       = build_legend(rows);
    const auto width        = box_width(header_lines, tree_lines);

    std::string out = "\n";
    append_box(out, "Output Summary", header_lines, width);
    out += "\n";
    append_box(out, "Process tree", tree_lines, width);
    if(!legend.empty())
    {
        out += fmt::format("\n  {}\n", legend);
    }

    stream << out;
}

}  // namespace rocprofsys::output
