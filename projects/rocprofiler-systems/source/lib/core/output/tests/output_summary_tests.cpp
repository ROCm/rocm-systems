// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/output_summary.hpp"

#include <unistd.h>

#include <string>
#include <utility>
#include <vector>

using rocprofsys::output::process_metadata;
using rocprofsys::output::registry;

namespace
{
// Above PID_MAX_LIMIT, so never equal to the test process's own pid.
constexpr pid_t k_child_pid    = 5000001;
constexpr pid_t k_orphan_pid   = 5000002;
constexpr pid_t k_missing_pid  = 5000099;
constexpr pid_t k_cycle_pid_a  = 5000003;
constexpr pid_t k_cycle_pid_b  = 5000004;
constexpr auto  k_perfetto_out = "/tmp/rocprofsys-test/perfetto-trace.pftrace";

using test_file = std::pair<std::string, pid_t>;

std::string
render(const std::vector<test_file>&        files,
       const std::vector<process_metadata>& processes)
{
    registry::instance().start_new_session();
    for(const auto& [path, pid] : files)
    {
        registry::instance().register_file(path, pid);
    }
    for(const auto& proc : processes)
    {
        registry::instance().record_process(proc);
    }
    return registry::instance().format_summary();
}

process_metadata
self_process()
{
    return process_metadata{ .pid = getpid(), .ppid = getppid(), .command = "self" };
}

std::size_t
count_occurrences(const std::string& text, const std::string& needle)
{
    std::size_t count = 0;
    for(auto pos = text.find(needle); pos != std::string::npos;
        pos      = text.find(needle, pos + 1))
    {
        ++count;
    }
    return count;
}
}  // namespace

TEST(format_summary, empty_rows_prints_nothing) { EXPECT_TRUE(render({}, {}).empty()); }

TEST(format_summary, single_row_renders_all_header_fields)
{
    const std::string out = render({ { k_perfetto_out, getpid() } }, { self_process() });
    EXPECT_NE(out.find("Output Summary"), std::string::npos);
    EXPECT_NE(out.find("Run: "), std::string::npos);
    EXPECT_NE(out.find("Duration: "), std::string::npos);
    EXPECT_NE(out.find("Processes: 1"), std::string::npos);
    EXPECT_NE(out.find("Output dir: /tmp/rocprofsys-test"), std::string::npos);
    EXPECT_NE(out.find("Total output: "), std::string::npos);
}

TEST(format_summary, single_row_renders_full_absolute_path)
{
    const std::string out = render({ { k_perfetto_out, getpid() } }, { self_process() });
    EXPECT_NE(out.find(k_perfetto_out), std::string::npos);
}

TEST(format_summary, single_row_renders_format_badge_name)
{
    const std::string out = render({ { k_perfetto_out, getpid() } }, { self_process() });
    EXPECT_NE(out.find("─ perfetto "), std::string::npos);
}

TEST(format_summary, single_row_renders_legend_entry)
{
    const std::string out = render({ { k_perfetto_out, getpid() } }, { self_process() });
    EXPECT_NE(out.find("perfetto → https://ui.perfetto.dev"), std::string::npos);
}

TEST(format_summary, multiple_formats_render_both_file_names)
{
    const std::string out =
        render({ { k_perfetto_out, getpid() },
                 { "/tmp/rocprofsys-test/wall_clock.txt", getpid() } },
               { self_process() });
    EXPECT_NE(out.find("perfetto-trace.pftrace"), std::string::npos);
    EXPECT_NE(out.find("wall_clock.txt"), std::string::npos);
}

TEST(format_summary, multiple_formats_render_both_legend_entries)
{
    const std::string out =
        render({ { k_perfetto_out, getpid() },
                 { "/tmp/rocprofsys-test/wall_clock.txt", getpid() } },
               { self_process() });
    EXPECT_NE(out.find("perfetto → https://ui.perfetto.dev"), std::string::npos);
    EXPECT_NE(out.find("text → cat"), std::string::npos);
}

TEST(format_summary, peer_controlled_path_control_chars_are_stripped)
{
    const std::string out =
        render({ { "/tmp/rocprofsys-test/\x1b[31mevil\x1b[0m.pftrace", getpid() } },
               { self_process() });
    // Only the triggering ESC byte is guaranteed gone; the CSI parameter
    // bytes are printable and are not stripped.
    EXPECT_EQ(out.find('\x1b'), std::string::npos);
    EXPECT_NE(out.find("evil"), std::string::npos);
    EXPECT_NE(out.find(".pftrace"), std::string::npos);
}

TEST(format_summary, relative_path_renders_as_absolute)
{
    const std::string out = render(
        { { "relative-dir/perfetto-trace.pftrace", getpid() } }, { self_process() });
    EXPECT_NE(out.find("/relative-dir/perfetto-trace.pftrace"), std::string::npos);
}

TEST(format_summary, child_process_renders_nested_under_parent)
{
    const std::string out = render(
        { { k_perfetto_out, getpid() },
          { "/tmp/rocprofsys-test/child.pftrace", k_child_pid } },
        { self_process(), process_metadata{ .pid     = k_child_pid,
                                            .ppid    = getpid(),
                                            .command = "/usr/bin/worker --flag" } });
    EXPECT_NE(out.find("\n  └─● [5000001] worker\n      └─ perfetto"), std::string::npos)
        << out;
    EXPECT_NE(out.find("Processes: 2"), std::string::npos);
}

TEST(format_summary, orphan_renders_at_top_level_with_parent_pid)
{
    const std::string out =
        render({ { k_perfetto_out, getpid() },
                 { "/tmp/rocprofsys-test/orphan.pftrace", k_orphan_pid } },
               { self_process(), process_metadata{ .pid     = k_orphan_pid,
                                                   .ppid    = k_missing_pid,
                                                   .command = "worker" } });
    EXPECT_NE(out.find("\n● [5000002] worker  (child of [5000099])\n"), std::string::npos)
        << out;
}

TEST(format_summary, main_process_renders_first)
{
    const std::string out = render(
        { { "/tmp/rocprofsys-test/init.pftrace", 1 }, { k_perfetto_out, getpid() } },
        { self_process(), process_metadata{ .pid = 1, .ppid = 0, .command = "init" } });
    const auto main_pos = out.find("main\n");
    const auto init_pos = out.find("● [1] init");
    ASSERT_NE(main_pos, std::string::npos) << out;
    ASSERT_NE(init_pos, std::string::npos) << out;
    EXPECT_LT(main_pos, init_pos);
}

TEST(format_summary, ppid_cycle_renders_each_process_once)
{
    const std::string out = render(
        { { "/tmp/rocprofsys-test/a.pftrace", k_cycle_pid_a },
          { "/tmp/rocprofsys-test/b.pftrace", k_cycle_pid_b } },
        { process_metadata{ .pid = k_cycle_pid_a, .ppid = k_cycle_pid_b, .command = "a" },
          process_metadata{
              .pid = k_cycle_pid_b, .ppid = k_cycle_pid_a, .command = "b" } });
    EXPECT_EQ(count_occurrences(out, "● [5000003]"), 1u) << out;
    EXPECT_EQ(count_occurrences(out, "● [5000004]"), 1u) << out;
}
