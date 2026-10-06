// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/output/output_summary.hpp"

#include "common/units/data_size.hpp"
#include "logger/debug.hpp"

#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <numeric>
#include <ostream>
#include <ranges>
#include <set>
#include <system_error>

namespace rocprofsys::output
{

using rocprofsys::common::units::bytes;
using rocprofsys::common::units::data_size_cast;
using rocprofsys::common::units::gigabytes;
using rocprofsys::common::units::kilobytes;
using rocprofsys::common::units::megabytes;

namespace
{
inline constexpr std::string_view UNKNOWN_VALUE_PLACEHOLDER = "?";

inline constexpr std::size_t FORMAT_NAME_WIDTH = 9;
inline constexpr std::size_t FILE_SIZE_WIDTH   = 10;
}  // namespace

run_metadata
run_metadata::capture(std::chrono::steady_clock::time_point load_baseline)
{
    run_metadata meta{};

    const auto tt =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    meta.run_label = fmt::format("{:%FT%TZ}", fmt::gmtime(tt));

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
    const auto value = bytes{ static_cast<double>(size_bytes) };
    if(value < kilobytes{ 1 }) return fmt::format("{}", data_size_cast<bytes>(value));
    if(value < megabytes{ 1 })
        return fmt::format("{:.2f}", data_size_cast<kilobytes>(value));
    if(value < gigabytes{ 1 })
        return fmt::format("{:.2f}", data_size_cast<megabytes>(value));
    return fmt::format("{:.2f}", data_size_cast<gigabytes>(value));
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

void
print_node(std::string& out, const process_node& node, std::string_view connector,
           const std::string& prefix, pid_t main_pid, const std::filesystem::path& cwd)
{
    out += connector;
    out += "● ";
    out += process_label(node, main_pid);
    out += '\n';

    const bool has_children = !node.children.empty();
    for(std::size_t i = 0; i < node.rows.size(); ++i)
    {
        const bool last = (i + 1 == node.rows.size()) && !has_children;
        out += file_row_line(prefix + (last ? "└─ " : "├─ "), node.rows[i], cwd);
        out += '\n';
    }
    if(!node.rows.empty() && has_children) out += prefix + "│\n";

    for(std::size_t i = 0; i < node.children.size(); ++i)
    {
        const bool last = (i + 1 == node.children.size());
        if(i > 0) out += prefix + "│\n";
        print_node(out, node.children[i], last ? "└─" : "├─",
                   prefix + (last ? "    " : "│   "), main_pid, cwd);
    }
}

[[nodiscard]] std::string
derive_output_dir(std::span<const artifact> rows)
{
    if(rows.empty()) return std::string{ UNKNOWN_VALUE_PLACEHOLDER };
    auto parent = std::filesystem::path{ rows.front().path }.parent_path().string();
    return parent.empty() ? std::string{ UNKNOWN_VALUE_PLACEHOLDER } : parent;
}

std::vector<std::string>
render_header(const run_metadata& meta, std::span<const artifact> rows,
              std::size_t process_count)
{
    const auto sizes = rows | std::views::transform(&artifact::size_bytes);

    std::string run_line =
        fmt::format("Run: {}   Duration: {}   Processes: {}",
                    meta.run_label.empty() ? std::string{ UNKNOWN_VALUE_PLACEHOLDER }
                                           : meta.run_label,
                    format_duration(meta.duration), process_count);
    run_line += fmt::format("   Total output: {}",
                            datasize_to_string(std::accumulate(sizes.begin(), sizes.end(),
                                                               std::uint64_t{ 0 })));

    std::string dir_line = fmt::format("Output dir: {}", derive_output_dir(rows));

    return { std::move(run_line), std::move(dir_line) };
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
              std::span<const artifact> rows, std::size_t process_count)
{
    if(rows.empty()) return;

    report_diagnostics(tree.diagnostics());

    const auto header_lines = render_header(meta, rows, process_count);
    const auto legend       = build_legend(rows);

    std::error_code       cwd_error;
    std::filesystem::path cwd      = std::filesystem::current_path(cwd_error);
    const pid_t           main_pid = getpid();

    std::string out = "\nOutput Summary\n";
    for(const auto& line : header_lines)
        out += "  " + line + "\n";
    out += "\nProcess tree\n";
    for(const auto& root : tree.roots())
        print_node(out, root, "", "  ", main_pid, cwd);
    if(!legend.empty()) out += fmt::format("\n  {}\n", legend);

    os << out;
}

}  // namespace rocprofsys::output
