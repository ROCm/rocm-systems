// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/summary_renderer.hpp"

#include "common/size_format.hpp"
#include "logger/debug.hpp"
#include "output/helper_collapser.hpp"
#include "output/process_tree_builder.hpp"
#include "output/role_classifier.hpp"
#include "output/run_metadata.hpp"
#include "output_file_registry.hpp"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>

namespace rocprofsys::output
{

namespace
{
inline constexpr std::size_t PROCESS_TREE_COL_WIDTH = 56;
inline constexpr std::size_t SIZE_COL_WIDTH         = 10;

const char*
role_token(role_hint r)
{
    switch(r)
    {
        case role_hint::main: return "*main*";
        case role_hint::gpu: return "*gpu*";
        case role_hint::engine: return "*engine*";
    }
    return "";
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
    return std::filesystem::path{ path }.filename().string();
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
        label = fmt::format("[{}] {}", node.meta.pid, node.meta.command);
    }
    if(node.role) label += std::string{ "  " } + role_token(*node.role);
    return label;
}

// Single source of truth for column-aligned file rows. `prefix`
// carries any tree indent + branch glyph (or empty for bare flat
// rows in single-process mode).
void
render_file_row(std::string& msg, std::string_view prefix, const output_file& f)
{
    const std::string left = fmt::format("{}{}", prefix, f.label);
    msg += fmt::format("  {:<{}} {:>{}}  {}\n", left, PROCESS_TREE_COL_WIDTH,
                       common::format_size_human(f.size_bytes), SIZE_COL_WIDTH,
                       basename_of(f.path));
}

void
render_node(std::string& msg, const process_node& node, const std::string& indent,
            bool is_last)
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
    }

    for(std::size_t i = 0; i < node.children.size(); ++i)
    {
        const bool last_child = (i + 1 == node.children.size());
        render_node(msg, node.children[i], sub_indent, last_child);
    }
}

struct viewer_entry
{
    std::string label;
    std::string viewer;
    bool        operator<(const viewer_entry& o) const
    {
        return std::tie(label, viewer) < std::tie(o.label, o.viewer);
    }
};

void
collect_viewer_entries(const process_node& node, std::set<viewer_entry>& out)
{
    for(const auto& r : node.rows)
        out.insert({ r.label, r.viewer });
    for(const auto& c : node.children)
        collect_viewer_entries(c, out);
}

void
render_single_process_body(std::string& msg, const process_node& root)
{
    for(const auto& row : root.rows)
        render_file_row(msg, "", row);
}

std::string
format_duration(std::chrono::nanoseconds dur)
{
    if(dur.count() <= 0) return "?";
    const double seconds = std::chrono::duration<double>(dur).count();
    return fmt::format("{:.2f}s", seconds);
}

}  // namespace

// Wrap content into chunks of at most `width` bytes. Word-boundary
// preferred (last whitespace at or before width); falls back to
// byte chunking when no whitespace exists in the chunk. UTF-8 safe:
// when the byte-chunk fallback would land on a continuation byte
// (10xxxxxx), back off to the preceding code-point boundary so we
// never split a multi-byte sequence mid-character.
std::vector<std::string>
wrap_to_width(std::string_view content, std::size_t width)
{
    std::vector<std::string> out;
    if(content.empty())
    {
        out.emplace_back();
        return out;
    }
    if(width == 0) width = 1;
    while(!content.empty())
    {
        if(content.size() <= width)
        {
            out.emplace_back(content);
            break;
        }
        std::size_t cut = width;
        // UTF-8 backoff: avoid splitting a multi-byte code point.
        while(cut > 0 && (static_cast<unsigned char>(content[cut]) & 0xC0) == 0x80)
        {
            --cut;
        }
        // Prefer the last whitespace at or before `cut` if one exists.
        // ws==0 would yield an empty leading chunk and never advance —
        // treat as "no break" and fall through to the byte chunking.
        const auto ws             = content.rfind(' ', cut);
        bool       broke_on_space = false;
        if(ws != std::string_view::npos && ws > 0)
        {
            cut            = ws;
            broke_on_space = true;
        }
        if(cut == 0) cut = 1;  // pathological: force progress
        out.emplace_back(content.substr(0, cut));
        // Skip the breaking space so it doesn't lead the next chunk.
        std::size_t next_start = cut + (broke_on_space ? 1 : 0);
        content.remove_prefix(std::min(next_start, content.size()));
    }
    return out;
}

namespace
{
// Box width is capped so it fits in a typical terminal; long output
// dirs wrap onto multiple boxed lines instead of letting the
// terminal break mid-content (which destroys the box visual).
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

    constexpr std::size_t MIN_INNER = 60;
    constexpr std::size_t MAX_INNER = 96;
    const std::size_t     longest =
        std::max({ title.size(), run_line.size(), dir_line.size() });
    const std::size_t inner = std::clamp<std::size_t>(longest + 2, MIN_INNER, MAX_INNER);
    const std::size_t content_width = inner - 2;  // " " padding each side

    // ─ is 3-byte UTF-8; reserve accordingly.
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
    constexpr std::size_t SEPARATOR_WIDTH = 80;
    std::string           sep;
    sep.reserve(SEPARATOR_WIDTH * 3);
    for(std::size_t i = 0; i < SEPARATOR_WIDTH; ++i)
        sep += "═";
    msg += fmt::format("  {}\n", sep);
}

void
render_viewer_footer(std::string& msg, const std::set<viewer_entry>& entries)
{
    if(entries.empty()) return;
    msg += "\n";
    for(const auto& e : entries)
        msg += fmt::format("  Open {}: {}\n", e.label, e.viewer);
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

    auto processes = registry.processes();
    auto built     = build_tree(rows, processes);
    built.roots    = collapse_helpers(std::move(built.roots));
    classify(built.roots, getpid());

    emit_diagnostics(built.diagnostics);

    // Count distinct PIDs from rows rather than tree size — collapsed
    // nodes would skew the tree count.
    std::unordered_set<pid_t> pid_set;
    for(const auto& r : rows)
        pid_set.insert(r.pid);
    const std::size_t process_count = pid_set.size();

    std::string output_dir = meta.output_dir_abs;
    if(output_dir.empty())
    {
        output_dir = std::filesystem::path{ rows.front().path }.parent_path().string();
    }
    if(output_dir.empty()) output_dir = "?";

    auto msg = std::string{};
    render_header(msg, meta, process_count, output_dir);
    msg += "\n";
    render_column_header(msg);

    if(built.roots.empty())
    {
        // Degenerate fallback: rows present but no tree shape.
        for(const auto& r : rows)
            msg += fmt::format("  {:<{}} {:>{}}  {}\n", r.label, PROCESS_TREE_COL_WIDTH,
                               common::format_size_human(r.size_bytes), SIZE_COL_WIDTH,
                               basename_of(r.path));
    }
    else if(process_count == 1 && built.roots.size() == 1)
    {
        render_single_process_body(msg, built.roots.front());
    }
    else
    {
        for(std::size_t i = 0; i < built.roots.size(); ++i)
        {
            const bool last = (i + 1 == built.roots.size());
            render_node(msg, built.roots[i], std::string{}, last);
        }
    }

    std::set<viewer_entry> viewers;
    for(const auto& r : built.roots)
        collect_viewer_entries(r, viewers);
    render_viewer_footer(msg, viewers);

    os << msg;
}

}  // namespace rocprofsys::output
