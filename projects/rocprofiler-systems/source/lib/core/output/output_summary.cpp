// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output_summary.hpp"

#include "common/size_format.hpp"
#include "core/config.hpp"
#include "logger/debug.hpp"
#include "output/process_tree_builder.hpp"
#include "output/run_metadata.hpp"
#include "output/text_layout.hpp"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <ostream>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
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

struct summary_model
{
    build_result               built{};
    std::size_t                process_count{ 0 };
    pid_t                      main_pid{ -1 };
    std::string                output_dir{};
    std::uintmax_t             total_output_bytes{ 0 };
    std::vector<int>           utilized_gpu_ids{};
    std::optional<std::size_t> node_gpu_count{};
};

std::size_t
count_distinct_pids(std::span<const output_file> rows)
{
    std::unordered_set<pid_t> pid_set;
    for(const auto& row : rows)
        pid_set.insert(row.pid);
    return pid_set.size();
}

std::string
derive_output_dir(const run_metadata& meta, std::span<const output_file> rows)
{
    if(!meta.output_dir_abs.empty()) return meta.output_dir_abs;
    if(rows.empty()) return "?";
    auto parent = std::filesystem::path{ rows.front().path }.parent_path().string();
    return parent.empty() ? std::string{ "?" } : parent;
}

std::uintmax_t
total_known_output_bytes(std::span<const output_file> rows)
{
    std::uintmax_t total = 0;
    for(const auto& row : rows)
        if(row.size_bytes) total += *row.size_bytes;
    return total;
}

void
append_gpu_ids(const process_node& node, std::set<int>& utilized)
{
    utilized.insert(node.meta.gpu_ids.begin(), node.meta.gpu_ids.end());
    for(const auto& child : node.children)
        append_gpu_ids(child, utilized);
}

std::vector<int>
union_of_utilized_gpus(const std::vector<process_node>& roots)
{
    std::set<int> utilized;
    for(const auto& root : roots)
        append_gpu_ids(root, utilized);

    return { utilized.begin(), utilized.end() };
}

summary_model
build_summary_model(const output_summary& summary, const run_metadata& meta,
                    std::span<const output_file> rows)
{
    summary_model model{};

    auto processes    = summary.processes();
    model.main_pid    = getpid();
    model.built       = build_tree(rows, processes);
    model.built.roots = collapse_small_processes(std::move(model.built.roots));
    compute_subtree_sizes(model.built.roots);

    model.process_count      = count_distinct_pids(rows);
    model.output_dir         = derive_output_dir(meta, rows);
    model.total_output_bytes = total_known_output_bytes(rows);
    model.utilized_gpu_ids   = union_of_utilized_gpus(model.built.roots);
    model.node_gpu_count     = summary.node_gpu_count();
    return model;
}

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
        case output_format::perfetto: return { .glyph = "◈", .name = "perfetto" };
        case output_format::rocpd: return { .glyph = "◆", .name = "rocpd" };
        case output_format::json: return { .glyph = "▪", .name = "json" };
        case output_format::text: return { .glyph = "▪", .name = "text" };
        case output_format::causal_json: return { .glyph = "▪", .name = "causal-json" };
        case output_format::causal_text: return { .glyph = "▪", .name = "causal-text" };
    }
    return { .glyph = "▪", .name = "output" };
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
report_diagnostics(const build_diagnostics& diagnostics)
{
    if(!diagnostics.missing_metadata_pids.empty())
    {
        LOG_WARNING("Output Summary: missing process metadata for pid(s) [{}]; "
                    "they render at root depth without role/parent",
                    fmt::join(diagnostics.missing_metadata_pids, ","));
    }
}

[[nodiscard]] std::string
process_label(const process_node& node, pid_t main_pid)
{
    if(node.collapsed)
        return fmt::format("[{}..{}] ({} small child processes)", node.collapsed->min_pid,
                           node.collapsed->max_pid, node.collapsed->count);

    const std::string program = summarize_command(node.meta.command);
    std::string       label   = program.empty() ? fmt::format("[{}]", node.meta.pid)
                                                : fmt::format("[{}] {}", node.meta.pid, program);

    // GPU assignment is aggregated in the header, so only the main process
    // is annotated per row.
    if(node.meta.pid == main_pid) label += "  main";
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
                       const std::string& child_prefix, std::size_t depth, pid_t main_pid)
{
    const bool expand_files = depth <= MAX_EXPANDED_DEPTH && !node.collapsed;

    std::string line = connector + "● " + process_label(node, main_pid);
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
                               next_prefix, depth + 1, main_pid);
    }
}

[[nodiscard]] std::vector<std::string>
build_tree_lines(const summary_model& model, std::set<output_format>& formats)
{
    std::vector<std::string> lines;
    for(const auto& root : model.built.roots)
        append_process_subtree(lines, formats, root, std::string{}, std::string{ "  " },
                               0, model.main_pid);
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

static void
write_summary(std::ostream& os, const output_summary& summary, const run_metadata& meta)
{
    auto rows = summary.rows();
    if(rows.empty()) return;

    auto model = build_summary_model(summary, meta, rows);
    report_diagnostics(model.built.diagnostics);

    std::set<output_format> formats;
    const auto              header_lines = build_header_lines(meta, model);
    const auto              tree_lines   = build_tree_lines(model, formats);
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

namespace rocprofsys
{

output_file
output_summary::make_entry(std::string path, output_format format)
{
    output_file entry{};
    entry.path   = std::move(path);
    entry.format = format;
    return entry;
}

namespace
{
std::optional<std::uintmax_t>
try_file_size(const std::string& path)
{
    std::error_code ec;
    auto            size = std::filesystem::file_size(path, ec);
    if(ec)
    {
        LOG_WARNING("output_summary: failed to read size of '{}' ({}); row will render "
                    "with size '?'",
                    path, ec.message());
        return std::nullopt;
    }
    return size;
}
}  // namespace

void
output_summary::push_entry(output_file&& entry, std::optional<pid_t> pid)
{
    // stat() runs before the lock so filesystem latency never blocks
    // concurrent register_file calls on the summary mutex.
    entry.pid        = pid.value_or(getpid());
    entry.size_bytes = try_file_size(entry.path);

    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.push_back({ m_session_id, std::move(entry) });
}

void
output_summary::register_file(std::string path, output_format format,
                              std::optional<pid_t> pid)
{
    push_entry(make_entry(std::move(path), format), pid);
}

void
output_summary::print(std::ostream&                         os,
                      std::chrono::steady_clock::time_point load_baseline)
{
    output::process_metadata self{};
    self.pid     = getpid();
    self.ppid    = getppid();
    self.command = config::get_exe_name();
    record_process(std::move(self));

    const auto meta = output::run_metadata::capture(load_baseline);
    output::write_summary(os, *this, meta);
}

std::vector<output_file>
output_summary::rows() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto                  current = m_session_id;
    std::vector<output_file>    out;
    out.reserve(m_files.size());
    for(const auto& v : m_files)
    {
        if(v.session_id == current) out.push_back(v.value);
    }
    return out;
}

void
output_summary::record_process(output::process_metadata meta)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for(auto& rec : m_processes)
    {
        if(rec.session_id == m_session_id && rec.value.pid == meta.pid)
        {
            // Merge, do not replace: a later sparser record (e.g. the
            // finalize-time self-registration, which knows pid/ppid/command
            // but not gpu_ids) must not erase richer fields a prior
            // post-processor record already supplied. Non-empty incoming
            // fields win; empty ones preserve the existing value.
            if(meta.ppid != -1) rec.value.ppid = meta.ppid;
            if(!meta.command.empty()) rec.value.command = std::move(meta.command);
            if(!meta.gpu_ids.empty()) rec.value.gpu_ids = std::move(meta.gpu_ids);
            return;
        }
    }
    m_processes.push_back({ m_session_id, std::move(meta) });
}

std::vector<output::process_metadata>
output_summary::processes() const
{
    std::lock_guard<std::mutex>           lock(m_mutex);
    const auto                            current = m_session_id;
    std::vector<output::process_metadata> out;
    out.reserve(m_processes.size());
    for(const auto& v : m_processes)
    {
        if(v.session_id == current) out.push_back(v.value);
    }
    return out;
}

void
output_summary::set_node_gpu_count(std::size_t count)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_node_gpu_count = count;
}

std::optional<std::size_t>
output_summary::node_gpu_count() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_node_gpu_count;
}

std::uint64_t
output_summary::bump_session()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    ++m_session_id;
    const auto current           = m_session_id;
    auto       not_current_files = [current](const versioned<output_file>& v) {
        return v.session_id != current;
    };
    auto not_current_procs = [current](const versioned<output::process_metadata>& v) {
        return v.session_id != current;
    };
    std::erase_if(m_files, not_current_files);
    std::erase_if(m_processes, not_current_procs);
    return m_session_id;
}

output_summary&
output_summary::instance_for_top_level_attach_finalize()
{
    static output_summary s_instance{};
    return s_instance;
}

}  // namespace rocprofsys
