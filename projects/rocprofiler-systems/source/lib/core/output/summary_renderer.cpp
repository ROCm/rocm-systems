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
#include <filesystem>
#include <ostream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace rocprofsys::output
{

namespace
{
constexpr const char*
role_token(role_hint r) noexcept
{
    switch(r)
    {
        case role_hint::main: return "*main";
        case role_hint::gpu: return "*gpu";
        case role_hint::engine: return "*engine";
    }
    return "";
}

const std::vector<int>&
ids_for_role(const process_node& node, role_hint role)
{
    // Engine reports the union over its subtree; main/gpu report only
    // what the process holds directly.
    switch(role)
    {
        case role_hint::main:
        case role_hint::gpu: return node.meta.gpu_ids;
        case role_hint::engine: return node.effective_gpu_ids;
    }
    return node.meta.gpu_ids;
}

void
emit_diagnostics(const build_diagnostics& diag)
{
    if(!diag.missing_metadata_pids.empty())
    {
        LOG_WARNING("Output Summary: missing process metadata for pid(s) [{}]; "
                    "they render at root depth without role/parent",
                    fmt::join(diag.missing_metadata_pids, ","));
    }
}

std::string
basename_of(const std::string& path)
{
    return strip_terminal_control_chars(
        std::filesystem::path{ path }.filename().string());
}

std::string
node_label(const process_node& node)
{
    std::string label;
    if(node.collapsed)
    {
        label = fmt::format("[{}..{}] ({} short-lived helpers)", node.collapsed->min_pid,
                            node.collapsed->max_pid, node.collapsed->count);
    }
    else if(node.meta.command.empty())
    {
        label = fmt::format("[{}]", node.meta.pid);
    }
    else
    {
        label = fmt::format("[{}] {}", node.meta.pid,
                            strip_terminal_control_chars(node.meta.command));
    }
    if(node.role)
    {
        label += fmt::format("  {}{}*", role_token(*node.role),
                             format_gpu_ids(ids_for_role(node, *node.role)));
    }
    return label;
}

// Single source of truth for column-aligned file rows.
void
render_file_row(std::string& msg, std::string_view prefix, const output_file& f)
{
    const std::string left = fmt::format("{}{}", prefix, f.label);
    msg += fmt::format("  {:<{}} {:>{}}  {}\n", left, PROCESS_TREE_COL_WIDTH,
                       common::format_size_human(f.size_bytes), SIZE_COL_WIDTH,
                       basename_of(f.path));
}

struct viewer_entry
{
    std::string label;
    std::string viewer;
    bool        operator<(const viewer_entry& o) const
    {
        return std::tie(label, viewer) < std::tie(o.label, o.viewer);
    }
    bool operator==(const viewer_entry& o) const
    {
        return label == o.label && viewer == o.viewer;
    }
};

void
render_node(std::string& msg, std::vector<viewer_entry>& viewers,
            const process_node& node, const std::string& indent, bool is_last)
{
    const char* branch    = is_last ? "└─" : "├─";
    const char* child_pad = is_last ? "  " : "│ ";

    msg += fmt::format("  {}{} {}\n", indent, branch, node_label(node));

    if(node.collapsed) return;

    const std::string sub_indent = indent + child_pad + " ";

    for(std::size_t i = 0; i < node.rows.size(); ++i)
    {
        const bool last_row = (i + 1 == node.rows.size()) && node.children.empty();
        const auto prefix   = sub_indent + (last_row ? "└─" : "├─") + " ";
        render_file_row(msg, prefix, node.rows[i]);
        viewers.push_back({ node.rows[i].label, node.rows[i].viewer });
    }

    for(std::size_t i = 0; i < node.children.size(); ++i)
    {
        const bool last_child = (i + 1 == node.children.size());
        render_node(msg, viewers, node.children[i], sub_indent, last_child);
    }
}

void
render_single_process_body(std::string& msg, std::vector<viewer_entry>& viewers,
                           const process_node& root)
{
    // Single-process runs only get a header line when a role marker is
    // set, so single-GPU workloads still see their main:<gpu-ids> tag.
    if(root.role) msg += fmt::format("  {}\n", node_label(root));

    for(const auto& row : root.rows)
    {
        render_file_row(msg, "", row);
        viewers.push_back({ row.label, row.viewer });
    }
}

void
render_header(std::string& msg, const run_metadata& meta, std::size_t process_count,
              const std::string& output_dir_for_header)
{
    const std::string title = "Output Summary";
    const std::string run_line =
        fmt::format("Run: {}   Duration: {}   Processes: {}",
                    meta.run_label.empty() ? std::string{ "?" } : meta.run_label,
                    format_duration(meta.duration), process_count);
    const std::string dir_line = fmt::format(
        "Output dir: {}",
        output_dir_for_header.empty() ? std::string{ "?" } : output_dir_for_header);

    const std::size_t longest =
        std::max({ title.size(), run_line.size(), dir_line.size() });
    const std::size_t inner = std::clamp<std::size_t>(longest + 2, MIN_INNER, MAX_INNER);
    const std::size_t content_width = inner - 2;

    // ─ is 3-byte UTF-8.
    std::string h_rule;
    h_rule.reserve(inner * 3);
    for(std::size_t i = 0; i < inner; ++i)
        h_rule += "─";

    auto box_line = [&](const std::string& content) {
        return fmt::format("  │ {:<{}} │\n", content, content_width);
    };

    auto box_wrapped = [&](const std::string& content) {
        for(const auto& chunk : wrap_to_width(content, content_width))
            msg += box_line(chunk);
    };

    msg += "\n";
    msg += fmt::format("  ┌{}┐\n", h_rule);
    box_wrapped(title);
    box_wrapped(run_line);
    box_wrapped(dir_line);
    msg += fmt::format("  └{}┘\n", h_rule);
}

void
render_column_header(std::string& msg)
{
    msg += fmt::format("  {:<{}} {:>{}}  {}\n", "Process tree", PROCESS_TREE_COL_WIDTH,
                       "Size", SIZE_COL_WIDTH, "Trace");
    std::string sep;
    sep.reserve(SEPARATOR_WIDTH * 3);
    for(std::size_t i = 0; i < SEPARATOR_WIDTH; ++i)
        sep += "═";
    msg += fmt::format("  {}\n", sep);
}

void
render_viewer_footer(std::string& msg, const std::vector<viewer_entry>& entries)
{
    if(entries.empty()) return;
    msg += "\n";
    for(const auto& e : entries)
        msg += fmt::format("  Open {}: {}\n", e.label, e.viewer);
}

void
dedup_viewers(std::vector<viewer_entry>& v)
{
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

void
render_body(std::string& msg, std::vector<viewer_entry>& viewers,
            const summary_model& model, const std::vector<output_file>& rows)
{
    if(model.built.roots.empty())
    {
        // Degenerate fallback: rows present, no tree shape.
        for(const auto& r : rows)
        {
            render_file_row(msg, "", r);
            viewers.push_back({ r.label, r.viewer });
        }
        return;
    }
    if(model.process_count == 1 && model.built.roots.size() == 1)
    {
        render_single_process_body(msg, viewers, model.built.roots.front());
        return;
    }
    for(std::size_t i = 0; i < model.built.roots.size(); ++i)
    {
        const bool last = (i + 1 == model.built.roots.size());
        render_node(msg, viewers, model.built.roots[i], std::string{}, last);
    }
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

    std::string               msg;
    std::vector<viewer_entry> viewers;
    render_header(msg, meta, model.process_count, model.output_dir);
    msg += "\n";
    render_column_header(msg);
    render_body(msg, viewers, model, rows);
    dedup_viewers(viewers);
    render_viewer_footer(msg, viewers);

    os << msg;
}

}  // namespace rocprofsys::output
