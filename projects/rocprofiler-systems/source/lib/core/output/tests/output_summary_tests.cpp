// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/output_summary.hpp"

#include <unistd.h>

#include <string>
#include <vector>

using rocprofsys::output::artifact;
using rocprofsys::output::process_metadata;
using rocprofsys::output::registry;

// ---------------------------------------------------------------------------
// format_summary
// ---------------------------------------------------------------------------

namespace
{
std::string
render(const std::vector<artifact>& rows, const std::vector<process_metadata>& processes)
{
    registry::instance().start_new_session();
    for(const auto& row : rows)
        registry::instance().register_file(row.path);
    for(const auto& proc : processes)
        registry::instance().record_process(proc);
    return registry::instance().format_summary();
}
}  // namespace

TEST(format_summary, empty_rows_prints_nothing) { EXPECT_TRUE(render({}, {}).empty()); }

TEST(format_summary, single_row_renders_all_header_fields)
{
    std::vector<artifact> rows{ artifact{ "/tmp/rocprofsys-test/perfetto-trace.pftrace",
                                          getpid(), 0 } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("Output Summary"), std::string::npos);
    EXPECT_NE(out.find("Run: "), std::string::npos);
    EXPECT_NE(out.find("Duration: "), std::string::npos);
    EXPECT_NE(out.find("Processes: "), std::string::npos);
    EXPECT_NE(out.find("Output dir: "), std::string::npos);
    EXPECT_NE(out.find("Total output: "), std::string::npos);
}

TEST(format_summary, single_row_renders_full_absolute_path)
{
    std::vector<artifact> rows{ artifact{ "/tmp/rocprofsys-test/perfetto-trace.pftrace",
                                          getpid(), 0 } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("/tmp/rocprofsys-test/perfetto-trace.pftrace"), std::string::npos);
}

TEST(format_summary, single_row_renders_format_badge_name)
{
    std::vector<artifact> rows{ artifact{ "/tmp/rocprofsys-test/perfetto-trace.pftrace",
                                          getpid(), 0 } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("perfetto"), std::string::npos);
}

TEST(format_summary, single_row_renders_legend_entry)
{
    std::vector<artifact> rows{ artifact{ "/tmp/rocprofsys-test/perfetto-trace.pftrace",
                                          getpid(), 0 } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("perfetto → https://ui.perfetto.dev"), std::string::npos);
}

TEST(format_summary, multiple_formats_render_both_file_names)
{
    std::vector<artifact> rows{
        artifact{ "/tmp/rocprofsys-test/perfetto-trace.pftrace", getpid(), 0 },
        artifact{ "/tmp/rocprofsys-test/wall_clock.txt", getpid(), 0 }
    };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("perfetto-trace.pftrace"), std::string::npos);
    EXPECT_NE(out.find("wall_clock.txt"), std::string::npos);
}

TEST(format_summary, multiple_formats_render_both_legend_entries)
{
    std::vector<artifact> rows{
        artifact{ "/tmp/rocprofsys-test/perfetto-trace.pftrace", getpid(), 0 },
        artifact{ "/tmp/rocprofsys-test/wall_clock.txt", getpid(), 0 }
    };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("perfetto → https://ui.perfetto.dev"), std::string::npos);
    EXPECT_NE(out.find("text → cat"), std::string::npos);
}

TEST(format_summary, peer_controlled_path_control_chars_are_stripped)
{
    std::vector<artifact>         rows{ artifact{
        "/tmp/rocprofsys-test/\x1b[31mevil\x1b[0m.pftrace", getpid(), 0 } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    // Only the triggering ESC byte is guaranteed gone; the CSI parameter
    // bytes are printable and are not stripped.
    EXPECT_EQ(out.find('\x1b'), std::string::npos);
    EXPECT_NE(out.find("evil"), std::string::npos);
    EXPECT_NE(out.find(".pftrace"), std::string::npos);
}

TEST(format_summary, relative_path_renders_as_absolute)
{
    std::vector<artifact> rows{ artifact{ "relative-dir/perfetto-trace.pftrace", getpid(),
                                          0 } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("/relative-dir/perfetto-trace.pftrace"), std::string::npos);
}
