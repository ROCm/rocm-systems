// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/output_summary.hpp"
#include "test_support/process_tree_builders.hpp"

#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

TEST(output_summary_pipeline, fork_layout_prints_tree_glyphs_and_range_line)
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

    // Synthetic: parent + 3 small-trace child processes + 1 GPU child.
    rocprofsys::output_summary summary;
    summary.register_file(write_file("main-trace.proto", 4096),
                          rocprofsys::output_format::perfetto, getpid());
    summary.register_file(write_file("helper-1.proto", 1024),
                          rocprofsys::output_format::perfetto, 1001);
    summary.register_file(write_file("helper-2.proto", 1024),
                          rocprofsys::output_format::perfetto, 1002);
    summary.register_file(write_file("helper-3.proto", 1024),
                          rocprofsys::output_format::perfetto, 1003);
    summary.register_file(write_file("worker-tp0.proto",
                                     100ULL * 1024 * 1024),  // 100 MiB
                          rocprofsys::output_format::perfetto, 1100);

    summary.record_process(make_meta(getpid(), -1, "main"));
    summary.record_process(make_meta(1001, getpid()));
    summary.record_process(make_meta(1002, getpid()));
    summary.record_process(make_meta(1003, getpid()));
    summary.record_process(make_meta(1100, getpid(), "worker", { 0 }));

    std::ostringstream oss;
    summary.print(oss, std::chrono::steady_clock::now());
    const auto out = oss.str();

    EXPECT_NE(out.find("Output Summary"), std::string::npos);
    EXPECT_NE(out.find("├─"), std::string::npos);
    EXPECT_NE(out.find("└─"), std::string::npos);
    // Small child processes under 16 KiB AND no GPU agents collapse into one line.
    EXPECT_NE(out.find("[1001..1003] (3 small child processes)"), std::string::npos);
    // Worker has GPU agent → not collapsed; renders as its own row.
    EXPECT_NE(out.find("worker-tp0.proto"), std::string::npos);

    fs::remove_all(base);
}
