// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/artifact.hpp"
#include "core/output/process_metadata.hpp"
#include "core/output/process_tree.hpp"
#include "core/output/summary_writer.hpp"

#include <spdlog/fmt/fmt.h>

#include <unistd.h>

#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using rocprofsys::output::artifact;
using rocprofsys::output::box_width;
using rocprofsys::output::display_width;
using rocprofsys::output::output_format;
using rocprofsys::output::process_metadata;
using rocprofsys::output::process_tree;
using rocprofsys::output::summarize_command;

TEST(summarize_command, empty_returns_empty) { EXPECT_EQ(summarize_command(""), ""); }

TEST(summarize_command, strips_path_to_basename)
{
    EXPECT_EQ(summarize_command("/usr/bin/python3"), "python3");
}

TEST(summarize_command, takes_first_token_only)
{
    EXPECT_EQ(summarize_command("python -c import_x --flag"), "python");
}

TEST(summarize_command, strips_terminal_control_chars)
{
    EXPECT_EQ(summarize_command("\x1b[31mpython\x1b[0m"), "python");
}

TEST(display_width, ascii_equals_byte_count)
{
    EXPECT_EQ(display_width("hello"), 5u);
    EXPECT_EQ(display_width(""), 0u);
}

TEST(display_width, counts_each_multibyte_codepoint_once)
{
    // Two U+2500 box-drawing chars are 6 bytes but 2 display columns.
    EXPECT_EQ(display_width("──"), 2u);
}

TEST(display_width, mixed_ascii_and_multibyte) { EXPECT_EQ(display_width("a─b"), 3u); }

TEST(box_width, widens_to_fit_the_longest_line_across_both_sets)
{
    const std::vector<std::string> header{ "short" };
    const std::vector<std::string> tree{
        "a much longer line than the header that clearly exceeds forty columns"
    };
    EXPECT_EQ(box_width(header, tree), display_width(tree.front()) + 2);
}

TEST(box_width, never_narrower_than_the_minimum)
{
    const std::vector<std::string> header{ "x" };
    const std::vector<std::string> tree{ "y" };
    EXPECT_GE(box_width(header, tree), 40u);
}

namespace
{
std::string
render(const std::vector<artifact>& rows, const std::vector<process_metadata>& processes)
{
    process_tree                     tree{ rows, processes };
    rocprofsys::output::run_metadata meta{};
    std::ostringstream               oss;
    rocprofsys::output::write_summary(oss, tree, meta, rows);
    return oss.str();
}
}  // namespace

TEST(write_summary, empty_rows_prints_nothing) { EXPECT_TRUE(render({}, {}).empty()); }

TEST(write_summary, single_row_renders_all_header_fields)
{
    std::vector<artifact> rows{ artifact{ "/tmp/rocprofsys-test/perfetto-trace.proto",
                                          getpid(), 0, output_format::perfetto } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("Output Summary"), std::string::npos);
    EXPECT_NE(out.find("Run: "), std::string::npos);
    EXPECT_NE(out.find("Duration: "), std::string::npos);
    EXPECT_NE(out.find("Processes: "), std::string::npos);
    EXPECT_NE(out.find("Output dir: "), std::string::npos);
    EXPECT_NE(out.find("Total output: "), std::string::npos);
}

TEST(write_summary, single_row_renders_full_absolute_path)
{
    std::vector<artifact> rows{ artifact{ "/tmp/rocprofsys-test/perfetto-trace.proto",
                                          getpid(), 0, output_format::perfetto } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("/tmp/rocprofsys-test/perfetto-trace.proto"), std::string::npos);
}

TEST(write_summary, single_row_renders_format_badge_name)
{
    std::vector<artifact> rows{ artifact{ "/tmp/rocprofsys-test/perfetto-trace.proto",
                                          getpid(), 0, output_format::perfetto } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("perfetto"), std::string::npos);
}

TEST(write_summary, single_row_renders_legend_entry)
{
    std::vector<artifact> rows{ artifact{ "/tmp/rocprofsys-test/perfetto-trace.proto",
                                          getpid(), 0, output_format::perfetto } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("perfetto → https://ui.perfetto.dev"), std::string::npos);
}

TEST(write_summary, multiple_formats_render_both_file_names)
{
    std::vector<artifact> rows{ artifact{ "/tmp/rocprofsys-test/perfetto-trace.proto",
                                          getpid(), 0, output_format::perfetto },
                                artifact{ "/tmp/rocprofsys-test/wall_clock.txt", getpid(),
                                          0, output_format::text } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("perfetto-trace.proto"), std::string::npos);
    EXPECT_NE(out.find("wall_clock.txt"), std::string::npos);
}

TEST(write_summary, multiple_formats_render_both_legend_entries)
{
    std::vector<artifact> rows{ artifact{ "/tmp/rocprofsys-test/perfetto-trace.proto",
                                          getpid(), 0, output_format::perfetto },
                                artifact{ "/tmp/rocprofsys-test/wall_clock.txt", getpid(),
                                          0, output_format::text } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("perfetto → https://ui.perfetto.dev"), std::string::npos);
    EXPECT_NE(out.find("text → cat"), std::string::npos);
}

TEST(write_summary, peer_controlled_path_control_chars_are_stripped)
{
    std::vector<artifact>         rows{ artifact{
        "/tmp/rocprofsys-test/\x1b[31mevil\x1b[0m.proto", getpid(), 0,
        output_format::perfetto } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_EQ(out.find('\x1b'), std::string::npos);
    EXPECT_NE(out.find("evil.proto"), std::string::npos);
}

TEST(write_summary, relative_path_renders_as_absolute)
{
    std::vector<artifact> rows{ artifact{ "relative-dir/perfetto-trace.proto", getpid(),
                                          0, output_format::perfetto } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("/relative-dir/perfetto-trace.proto"), std::string::npos);
}

TEST(write_summary, multi_process_tree_renders_parent_and_child)
{
    const pid_t           root  = getpid();
    constexpr pid_t       child = 700;
    std::vector<artifact> rows{
        artifact{ "/tmp/rocprofsys-test/root.proto", root, 0, output_format::perfetto },
        artifact{ "/tmp/rocprofsys-test/child.proto", child, 0, output_format::perfetto }
    };
    std::vector<process_metadata> processes{ process_metadata{ root, -1, "root" },
                                             process_metadata{ child, root, "child" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find(fmt::format("[{}]", root)), std::string::npos);
    EXPECT_NE(out.find(fmt::format("[{}]", child)), std::string::npos);
    EXPECT_NE(out.find("main"), std::string::npos);
}
