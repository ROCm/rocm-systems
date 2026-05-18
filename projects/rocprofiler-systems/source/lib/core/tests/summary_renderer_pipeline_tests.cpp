// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/summary_renderer.hpp"
#include "core/output_file_registry.hpp"
#include "test_support/process_tree_builders.hpp"

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

TEST(summary_renderer_pipeline, fork_layout_emits_tree_glyphs_and_range_line)
{
    namespace fs = std::filesystem;
    using rocprofsys::test_support::make_meta;

    // Build real files on disk so size_bytes is populated. Required
    // because is_helper is conservative on unknown sizes (rejects
    // collapse if size_bytes is nullopt).
    const auto base = fs::temp_directory_path() / "rocprofsys-pipeline-test";
    fs::create_directories(base);
    auto write_file = [&](const std::string& name, std::size_t bytes) {
        const auto        p = base / name;
        std::ofstream     o(p, std::ios::binary);
        const std::string buf(bytes, 'x');
        o.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        return p.string();
    };

    // Synthetic: parent + 3 small-trace helper children + 1 GPU child.
    rocprofsys::output_file_registry registry;
    registry.register_file(write_file("main-trace.proto", 4096),
                           rocprofsys::output_format::perfetto, getpid());
    registry.register_file(write_file("helper-1.proto", 1024),
                           rocprofsys::output_format::perfetto, 1001);
    registry.register_file(write_file("helper-2.proto", 1024),
                           rocprofsys::output_format::perfetto, 1002);
    registry.register_file(write_file("helper-3.proto", 1024),
                           rocprofsys::output_format::perfetto, 1003);
    registry.register_file(write_file("worker-tp0.proto",
                                      100ULL * 1024 * 1024),  // 100 MiB
                           rocprofsys::output_format::perfetto, 1100);

    registry.record_process(make_meta(getpid(), -1, "main"));
    registry.record_process(make_meta(1001, getpid()));
    registry.record_process(make_meta(1002, getpid()));
    registry.record_process(make_meta(1003, getpid()));
    registry.record_process(make_meta(1100, getpid(), "worker", { 0 }));

    std::ostringstream oss;
    rocprofsys::output::print_summary(oss, registry);
    const auto out = oss.str();

    EXPECT_NE(out.find("Output Summary"), std::string::npos);
    EXPECT_NE(out.find("├─"), std::string::npos);
    EXPECT_NE(out.find("└─"), std::string::npos);
    // Helpers under 16 KiB AND no GPU agents → collapse into one line.
    EXPECT_NE(out.find("[1001..1003] (3 short-lived helpers)"), std::string::npos);
    // Worker has GPU agent → not collapsed; renders as its own row.
    EXPECT_NE(out.find("worker-tp0.proto"), std::string::npos);

    fs::remove_all(base);
}
