// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/summary_renderer.hpp"
#include "core/output_file_registry.hpp"

#include <unistd.h>

#include <sstream>
#include <string>

TEST(summary_renderer, empty_registry_emits_nothing)
{
    rocprofsys::output_file_registry registry;
    std::ostringstream               oss;
    rocprofsys::output::print_summary(oss, registry);
    EXPECT_TRUE(oss.str().empty());
}

TEST(summary_renderer, single_row_renders_header_basename_and_viewer_footer)
{
    rocprofsys::output_file_registry registry;
    registry.register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto);

    std::ostringstream oss;
    rocprofsys::output::print_summary(oss, registry);

    const std::string out = oss.str();
    EXPECT_NE(out.find("Output Summary"), std::string::npos);
    EXPECT_NE(out.find("Run: "), std::string::npos);
    EXPECT_NE(out.find("Duration: "), std::string::npos);
    EXPECT_NE(out.find("Processes: "), std::string::npos);
    EXPECT_NE(out.find("Output dir: "), std::string::npos);
    EXPECT_NE(out.find("perfetto-trace.proto"), std::string::npos);
    EXPECT_NE(out.find("Open Perfetto trace: Open in https://ui.perfetto.dev"),
              std::string::npos);
}

TEST(summary_renderer, multi_row_uses_branch_glyphs_and_dedups_viewer_footer)
{
    rocprofsys::output_file_registry registry;
    registry.register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto);
    registry.register_file("/tmp/rocprofsys-test/wall_clock.txt",
                           rocprofsys::output_format::text, "wall_clock");

    std::ostringstream oss;
    rocprofsys::output::print_summary(oss, registry);

    const std::string out = oss.str();
    EXPECT_NE(out.find("Perfetto trace"), std::string::npos);
    EXPECT_NE(out.find("Profile (wall_clock)"), std::string::npos);
    EXPECT_NE(out.find("perfetto-trace.proto"), std::string::npos);
    EXPECT_NE(out.find("wall_clock.txt"), std::string::npos);
    EXPECT_NE(out.find("Open Perfetto trace:"), std::string::npos);
    EXPECT_NE(out.find("Open Profile (wall_clock):"), std::string::npos);
}

TEST(summary_renderer, main_with_own_gpus_shows_ids)
{
    rocprofsys::output_file_registry registry;
    registry.register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto, getpid());
    rocprofsys::output::process_metadata self{};
    self.pid     = getpid();
    self.ppid    = -1;
    self.command = "single_proc";
    self.gpu_ids = { 0, 1 };
    registry.record_process(std::move(self));

    std::ostringstream oss;
    rocprofsys::output::print_summary(oss, registry);
    const auto out = oss.str();
    EXPECT_NE(out.find("*main:0,1*"), std::string::npos);
}
