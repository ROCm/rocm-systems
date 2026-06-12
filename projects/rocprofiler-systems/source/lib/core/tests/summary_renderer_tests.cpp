// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/summary_renderer.hpp"
#include "core/output_file_registry.hpp"

#include <unistd.h>

#include <sstream>
#include <string>

namespace
{
std::string
render(const rocprofsys::output_file_registry& registry)
{
    std::ostringstream oss;
    rocprofsys::output::print_summary(oss, registry);
    return oss.str();
}
}  // namespace

TEST(summary_renderer, empty_registry_emits_nothing)
{
    rocprofsys::output_file_registry registry;
    EXPECT_TRUE(render(registry).empty());
}

TEST(summary_renderer, single_row_renders_header_full_path_and_legend)
{
    rocprofsys::output_file_registry registry;
    registry.register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto);

    const std::string out = render(registry);
    EXPECT_NE(out.find("Output Summary"), std::string::npos);
    EXPECT_NE(out.find("Run: "), std::string::npos);
    EXPECT_NE(out.find("Duration: "), std::string::npos);
    EXPECT_NE(out.find("Processes: "), std::string::npos);
    EXPECT_NE(out.find("Output dir: "), std::string::npos);
    EXPECT_NE(out.find("Total output: "), std::string::npos);
    // Full absolute path is rendered inline (the clickable-path goal),
    // not just the basename.
    EXPECT_NE(out.find("/tmp/rocprofsys-test/perfetto-trace.proto"), std::string::npos);
    // Typed glyph + format name and the compact legend (no per-row footer).
    EXPECT_NE(out.find("perfetto"), std::string::npos);
    EXPECT_NE(out.find("perfetto → https://ui.perfetto.dev"), std::string::npos);
}

TEST(summary_renderer, multiple_formats_render_names_and_legend_entries)
{
    rocprofsys::output_file_registry registry;
    registry.register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto);
    registry.register_file("/tmp/rocprofsys-test/wall_clock.txt",
                           rocprofsys::output_format::text);

    const std::string out = render(registry);
    EXPECT_NE(out.find("perfetto-trace.proto"), std::string::npos);
    EXPECT_NE(out.find("wall_clock.txt"), std::string::npos);
    EXPECT_NE(out.find("perfetto → https://ui.perfetto.dev"), std::string::npos);
    EXPECT_NE(out.find("text → cat"), std::string::npos);
}

TEST(summary_renderer, utilized_gpus_render_in_header_not_per_process)
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

    const std::string out = render(registry);
    EXPECT_NE(out.find("GPUs: 0,1"), std::string::npos);
    // Per-process *gpu* markers are gone; GPUs live only in the header.
    EXPECT_EQ(out.find("*gpu"), std::string::npos);
    // Without a node GPU count the "(all)" qualifier is omitted.
    EXPECT_EQ(out.find("(all)"), std::string::npos);
}

TEST(summary_renderer, full_gpu_set_marks_all)
{
    rocprofsys::output_file_registry registry;
    registry.register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto, getpid());
    registry.set_node_gpu_count(2);
    rocprofsys::output::process_metadata self{};
    self.pid     = getpid();
    self.ppid    = -1;
    self.gpu_ids = { 0, 1 };
    registry.record_process(std::move(self));

    EXPECT_NE(render(registry).find("GPUs: 0,1 (all)"), std::string::npos);
}

TEST(summary_renderer, peer_controlled_path_control_chars_are_stripped)
{
    rocprofsys::output_file_registry registry;
    // A traced binary could emit a file whose name carries a CSI escape.
    registry.register_file("/tmp/rocprofsys-test/\x1b[31mevil\x1b[0m.proto",
                           rocprofsys::output_format::perfetto);

    const std::string out = render(registry);
    EXPECT_EQ(out.find('\x1b'), std::string::npos);
    EXPECT_NE(out.find("evil.proto"), std::string::npos);
}

TEST(summary_renderer, relative_path_renders_as_absolute)
{
    rocprofsys::output_file_registry registry;
    registry.register_file("relative-dir/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto);

    const std::string out = render(registry);
    // The clickable-path goal requires an absolute path even when the
    // registrar passed a relative one.
    EXPECT_NE(out.find("/relative-dir/perfetto-trace.proto"), std::string::npos);
}

TEST(summary_renderer, deep_process_collapses_to_one_line_with_sizes)
{
    rocprofsys::output_file_registry registry;
    const pid_t                      root  = getpid();
    constexpr pid_t                  child = 700;
    constexpr pid_t                  grand = 800;
    registry.register_file("/tmp/rocprofsys-test/root.proto",
                           rocprofsys::output_format::perfetto, root);
    registry.register_file("/tmp/rocprofsys-test/child.proto",
                           rocprofsys::output_format::perfetto, child);
    registry.register_file("/tmp/rocprofsys-test/grand.proto",
                           rocprofsys::output_format::perfetto, grand);
    auto record = [&](pid_t pid, pid_t ppid) {
        rocprofsys::output::process_metadata meta{};
        meta.pid  = pid;
        meta.ppid = ppid;
        registry.record_process(std::move(meta));
    };
    record(root, -1);
    record(child, root);
    record(grand, child);

    const std::string out = render(registry);
    // The depth-2 process renders as a one-line own/cumulative summary
    // (the " / " separator is unique to collapsed process lines).
    EXPECT_NE(out.find("[800]"), std::string::npos);
    EXPECT_NE(out.find(" / "), std::string::npos);
}
