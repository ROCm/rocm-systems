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
#include <string>
#include <string_view>
#include <unordered_set>

namespace rocprofsys::output
{

namespace
{
// All layout widths grouped so the visual relationship is grep-able.
// Inner box width is clamped between MIN_INNER and MAX_INNER so the
// header fits a typical terminal; the body keeps separator and column
// alignment in sync with that range. SEPARATOR_WIDTH should not
// exceed MAX_INNER; layout::PROCESS_TREE_COL_WIDTH + layout::SIZE_COL_WIDTH plus the
// "Trace" basename column should fit comfortably inside the body
// column budget (~80 chars on a typical terminal).
namespace layout
{
inline constexpr std::size_t PROCESS_TREE_COL_WIDTH = 56;
inline constexpr std::size_t SIZE_COL_WIDTH         = 10;
inline constexpr std::size_t MIN_INNER              = 60;
inline constexpr std::size_t MAX_INNER              = 96;
inline constexpr std::size_t SEPARATOR_WIDTH        = 80;
// Cap on GPU-id tokens rendered in a role marker; longer lists get a
// truncation suffix to keep the marker from blowing column width.
inline constexpr std::size_t MAX_RENDERED_GPU_IDS = 16;
}  // namespace layout

const char*
role_token(role_hint r)
{
    switch(r)
    {
        case role_hint::main: return "*main";
        case role_hint::gpu: return "*gpu";
        case role_hint::engine: return "*engine";
    }
    return "";
}

// Returns the GPU-id list relevant to a role marker:
//  - main / gpu: the node's own gpu_ids. Main shows only what this
//    process directly uses; subtree summary belongs to engine.
//  - engine: effective_gpu_ids (union over descendants), so a
//    reader can see at a glance which GPUs the engine orchestrates.
const std::vector<int>&
ids_for_role(const process_node& node, role_hint role)
{
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
    if(node.role)
    {
        label += fmt::format("  {}{}*", role_token(*node.role),
                             format_gpu_ids(ids_for_role(node, *node.role)));
    }
    return label;
}

// Single source of truth for column-aligned file rows. `prefix`
// carries any tree indent + branch glyph (or empty for bare flat
// rows in single-process mode).
void
render_file_row(std::string& msg, std::string_view prefix, const output_file& f)
{
    const std::string left = fmt::format("{}{}", prefix, f.label);
    msg += fmt::format("  {:<{}} {:>{}}  {}\n", left, layout::PROCESS_TREE_COL_WIDTH,
                       common::format_size_human(f.size_bytes), layout::SIZE_COL_WIDTH,
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

// Walks the tree once: emits the rendered text into `msg` and
// accumulates viewer entries into `viewers` as it visits each row.
// Caller is responsible for sort+unique on the viewers vector before
// rendering the footer.
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
    // Single-process runs get a header line carrying the role marker
    // when one is set. This is how the main:<gpu-ids> suffix becomes
    // visible for the most common workload shape: one process
    // driving one or more GPUs directly. Without the header line
    // the GPU-id summary would only appear on multi-process trees.
    if(root.role) msg += fmt::format("  {}\n", node_label(root));

    for(const auto& row : root.rows)
    {
        render_file_row(msg, "", row);
        viewers.push_back({ row.label, row.viewer });
    }
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

// Compress contiguous runs of length >= 3 into "min-max"; render
// shorter runs as comma-separated singletons or pairs. Truncate
// rendered output to layout::MAX_RENDERED_GPU_IDS tokens to keep
// pathological inputs from blowing the column width.
std::string
format_gpu_ids(const std::vector<int>& gpu_ids)
{
    if(gpu_ids.empty()) return "";

    std::vector<int> sorted = gpu_ids;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    constexpr std::size_t MAX_RENDERED = layout::MAX_RENDERED_GPU_IDS;

    std::vector<std::string> tokens;
    for(std::size_t i = 0; i < sorted.size();)
    {
        std::size_t j = i;
        while(j + 1 < sorted.size() && sorted[j + 1] == sorted[j] + 1)
            ++j;
        const std::size_t run_len = j - i + 1;
        if(run_len >= 3)
            tokens.push_back(fmt::format("{}-{}", sorted[i], sorted[j]));
        else
            for(std::size_t k = i; k <= j; ++k)
                tokens.push_back(fmt::format("{}", sorted[k]));
        i = j + 1;
    }

    std::string out = ":";
    if(tokens.size() <= MAX_RENDERED)
    {
        for(std::size_t k = 0; k < tokens.size(); ++k)
        {
            if(k > 0) out += ",";
            out += tokens[k];
        }
    }
    else
    {
        const std::size_t extra = tokens.size() - MAX_RENDERED;
        for(std::size_t k = 0; k < MAX_RENDERED; ++k)
        {
            if(k > 0) out += ",";
            out += tokens[k];
        }
        out += fmt::format(",...(+{} more)", extra);
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

    const std::size_t longest =
        std::max({ title.size(), run_line.size(), dir_line.size() });
    const std::size_t inner =
        std::clamp<std::size_t>(longest + 2, layout::MIN_INNER, layout::MAX_INNER);
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
    msg += fmt::format("  {:<{}} {:>{}}  {}\n", "Process tree",
                       layout::PROCESS_TREE_COL_WIDTH, "Size", layout::SIZE_COL_WIDTH,
                       "Trace");
    std::string sep;
    sep.reserve(layout::SEPARATOR_WIDTH * 3);
    for(std::size_t i = 0; i < layout::SEPARATOR_WIDTH; ++i)
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

// Sort + de-duplicate in place. The renderer accumulates entries in
// tree-walk order; the footer should list each (label, viewer) pair
// at most once, sorted for stable output across runs.
void
dedup_viewers(std::vector<viewer_entry>& v)
{
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
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

    auto                      msg = std::string{};
    std::vector<viewer_entry> viewers;
    render_header(msg, meta, process_count, output_dir);
    msg += "\n";
    render_column_header(msg);

    if(built.roots.empty())
    {
        // Degenerate fallback: rows present but no tree shape.
        for(const auto& r : rows)
        {
            msg += fmt::format("  {:<{}} {:>{}}  {}\n", r.label,
                               layout::PROCESS_TREE_COL_WIDTH,
                               common::format_size_human(r.size_bytes),
                               layout::SIZE_COL_WIDTH, basename_of(r.path));
            viewers.push_back({ r.label, r.viewer });
        }
    }
    else if(process_count == 1 && built.roots.size() == 1)
    {
        render_single_process_body(msg, viewers, built.roots.front());
    }
    else
    {
        for(std::size_t i = 0; i < built.roots.size(); ++i)
        {
            const bool last = (i + 1 == built.roots.size());
            render_node(msg, viewers, built.roots[i], std::string{}, last);
        }
    }

    dedup_viewers(viewers);
    render_viewer_footer(msg, viewers);

    os << msg;
}

}  // namespace rocprofsys::output
