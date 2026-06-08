// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/summary_renderer.hpp"

#include "common/size_format.hpp"
#include "logger/debug.hpp"
#include "output/process_tree_builder.hpp"
#include "output/run_metadata.hpp"
#include "output/summary_model.hpp"
#include "output/text_layout.hpp"
#include "output_file_registry.hpp"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace rocprofsys::output
{

namespace
{
// Column at which a collapsed process line's "own / cumulative" sizes
// begin, padding the label so sibling summaries roughly align.
constexpr std::size_t COLLAPSED_SIZE_COLUMN = 44;
constexpr std::size_t FORMAT_NAME_WIDTH     = 9;
constexpr std::size_t FILE_SIZE_WIDTH       = 10;
constexpr std::size_t MIN_BOX_WIDTH         = 40;
// File rows are expanded for the main process and its direct children;
// deeper processes collapse to a one-line own/cumulative summary.
constexpr std::size_t MAX_EXPANDED_DEPTH = 1;

struct format_badge
{
    const char* glyph;
    const char* name;
};

[[nodiscard]] constexpr format_badge
badge_for(output_format format) noexcept
{
    switch(format)
    {
        case output_format::perfetto: return { "◈", "perfetto" };
        case output_format::rocpd: return { "◆", "rocpd" };
        case output_format::json: return { "▪", "json" };
        case output_format::text: return { "▪", "text" };
        case output_format::causal_json: return { "▪", "causal-json" };
        case output_format::causal_text: return { "▪", "causal-text" };
    }
    return { "▪", "output" };
}

[[nodiscard]] constexpr const char*
viewer_hint(output_format format) noexcept
{
    switch(format)
    {
        case output_format::perfetto: return "https://ui.perfetto.dev";
        case output_format::rocpd: return "sqlite3 / AMD Visualizer (OPTIQ)";
        case output_format::json:
        case output_format::causal_json: return "jq";
        case output_format::text:
        case output_format::causal_text: return "cat";
    }
    return "";
}

void
emit_diagnostics(const build_diagnostics& diagnostics)
{
    if(!diagnostics.missing_metadata_pids.empty())
    {
        LOG_WARNING("Output Summary: missing process metadata for pid(s) [{}]; "
                    "they render at root depth without role/parent",
                    fmt::join(diagnostics.missing_metadata_pids, ","));
    }
}

[[nodiscard]] std::string
process_label(const process_node& node)
{
    if(node.collapsed)
        return fmt::format("[{}..{}] ({} short-lived helpers)", node.collapsed->min_pid,
                           node.collapsed->max_pid, node.collapsed->count);

    const std::string program = summarize_command(node.meta.command);
    std::string       label   = program.empty() ? fmt::format("[{}]", node.meta.pid)
                                                : fmt::format("[{}] {}", node.meta.pid, program);

    // Only the main process is annotated; GPU assignment is aggregated in
    // the header, so per-process gpu/engine markers are intentionally gone.
    if(node.role && *node.role == role_hint::main) label += "  main";
    return label;
}

[[nodiscard]] std::string
collapsed_sizes(const process_node& node)
{
    return fmt::format("{} / {}", common::format_size_human(node.own_size_bytes),
                       common::format_size_human(node.cumulative_size_bytes));
}

// Absolute path so the rendered row is clickable regardless of how the
// registrar passed it; control characters stripped for terminal safety.
[[nodiscard]] std::string
display_path(const std::string& path)
{
    std::error_code error;
    const auto      absolute = std::filesystem::absolute(path, error);
    return strip_terminal_control_chars(error ? path : absolute.string());
}

[[nodiscard]] std::string
file_row_line(std::string_view branch, const output_file& file)
{
    const auto badge = badge_for(file.format);
    return fmt::format("{}{} {:<{}} {:>{}}  {}", branch, badge.glyph, badge.name,
                       FORMAT_NAME_WIDTH, common::format_size_human(file.size_bytes),
                       FILE_SIZE_WIDTH, display_path(file.path));
}

void
append_process_subtree(std::vector<std::string>& lines, std::set<output_format>& formats,
                       const process_node& node, const std::string& connector,
                       const std::string& child_prefix, std::size_t depth)
{
    const bool expand_files = depth <= MAX_EXPANDED_DEPTH && !node.collapsed;

    std::string line = connector + "● " + process_label(node);
    if(!expand_files && !node.collapsed)
    {
        const std::size_t label_width = display_width(line);
        if(label_width < COLLAPSED_SIZE_COLUMN)
            line.append(COLLAPSED_SIZE_COLUMN - label_width, ' ');
        line += collapsed_sizes(node);
    }
    lines.push_back(std::move(line));

    if(node.collapsed) return;

    const std::size_t file_count        = expand_files ? node.rows.size() : 0;
    const std::size_t child_count       = node.children.size();
    const bool        children_expanded = (depth + 1) <= MAX_EXPANDED_DEPTH;

    for(std::size_t index = 0; index < file_count; ++index)
    {
        const bool last_entry = (index + 1 == file_count) && child_count == 0;
        const auto branch     = child_prefix + (last_entry ? "└─ " : "├─ ");
        lines.push_back(file_row_line(branch, node.rows[index]));
        formats.insert(node.rows[index].format);
    }

    // Blank rail line separates a visually heavy file block from the
    // child processes, and separates expanded child subtrees from each
    // other; collapsed one-line children stay tightly packed.
    if(file_count > 0 && child_count > 0) lines.push_back(child_prefix + "│");

    for(std::size_t index = 0; index < child_count; ++index)
    {
        if(index > 0 && children_expanded) lines.push_back(child_prefix + "│");
        const bool        last_child  = (index + 1 == child_count);
        const std::string child_conn  = child_prefix + (last_child ? "└─" : "├─");
        const std::string next_prefix = child_prefix + (last_child ? "    " : "│   ");
        append_process_subtree(lines, formats, node.children[index], child_conn,
                               next_prefix, depth + 1);
    }
}

[[nodiscard]] std::vector<std::string>
build_tree_lines(const summary_model& model, const std::vector<output_file>& rows,
                 std::set<output_format>& formats)
{
    std::vector<std::string> lines;
    if(model.built.roots.empty())
    {
        // Degenerate fallback: rows present but no tree shape resolved.
        for(const auto& row : rows)
        {
            lines.push_back(file_row_line("", row));
            formats.insert(row.format);
        }
        return lines;
    }
    for(const auto& root : model.built.roots)
        append_process_subtree(lines, formats, root, std::string{}, std::string{ "  " },
                               0);
    return lines;
}

[[nodiscard]] std::vector<std::string>
build_header_lines(const run_metadata& meta, const summary_model& model)
{
    std::string run_line =
        fmt::format("Run: {}   Duration: {}   Processes: {}",
                    meta.run_label.empty() ? std::string{ "?" } : meta.run_label,
                    format_duration(meta.duration), model.process_count);

    if(!model.utilized_gpu_ids.empty())
    {
        std::string gpus = format_gpu_ids(model.utilized_gpu_ids);
        if(!gpus.empty() && gpus.front() == ':') gpus.erase(gpus.begin());

        const bool all = model.node_gpu_count &&
                         *model.node_gpu_count == model.utilized_gpu_ids.size();
        run_line += fmt::format("   GPUs: {}{}", gpus, all ? " (all)" : "");
    }
    run_line += fmt::format("   Total output: {}",
                            common::format_size_human(model.total_output_bytes));

    std::string dir_line =
        fmt::format("Output dir: {}",
                    model.output_dir.empty() ? std::string{ "?" } : model.output_dir);

    return { std::move(run_line), std::move(dir_line) };
}

[[nodiscard]] std::string
build_legend(const std::set<output_format>& formats)
{
    std::string legend;
    for(output_format format : formats)
    {
        const char* hint = viewer_hint(format);
        if(hint[0] == '\0') continue;
        if(!legend.empty()) legend += "    ";
        legend += fmt::format("{} → {}", badge_for(format).name, hint);
    }
    return legend;
}

[[nodiscard]] std::size_t
box_width(const std::vector<std::string>& header_lines,
          const std::vector<std::string>& tree_lines)
{
    std::size_t width = MIN_BOX_WIDTH;
    for(const auto& line : header_lines)
        width = std::max(width, display_width(line) + 2);  // + 2 for the "│ " rail
    for(const auto& line : tree_lines)
        width = std::max(width, display_width(line) + 2);
    return width;
}

void
append_box(std::string& out, std::string_view title,
           const std::vector<std::string>& lines, std::size_t width)
{
    const std::string head      = fmt::format("╭─ {} ", title);
    const std::size_t head_cols = display_width(head);
    out += head;
    if(width > head_cols) out += repeat_glyph("─", width - head_cols);
    out += "\n";
    for(const auto& line : lines)
        out += fmt::format("│ {}\n", line);
    out += "╰";
    out += repeat_glyph("─", width > 0 ? width - 1 : 0);
    out += "\n";
}
}  // namespace

void
print_summary(std::ostream& os, const output_file_registry& registry)
{
    run_metadata empty{};
    print_summary(os, registry, empty);
}

void
print_summary(std::ostream& os, const output_file_registry& registry,
              const run_metadata& meta)
{
    auto rows = registry.rows();
    if(rows.empty()) return;

    auto model = build_summary_model(registry, meta);
    emit_diagnostics(model.built.diagnostics);

    std::set<output_format> formats;
    const auto              header_lines = build_header_lines(meta, model);
    const auto              tree_lines   = build_tree_lines(model, rows, formats);
    const std::string       legend       = build_legend(formats);
    const std::size_t       width        = box_width(header_lines, tree_lines);

    std::string out = "\n";
    append_box(out, "Output Summary", header_lines, width);
    out += "\n";
    append_box(out, "Process tree", tree_lines, width);
    if(!legend.empty()) out += fmt::format("\n  {}\n", legend);

    os << out;
}

}  // namespace rocprofsys::output
