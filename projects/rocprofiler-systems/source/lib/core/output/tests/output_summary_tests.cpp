// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/output_summary.hpp"

#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using rocprofsys::output::artifact;
using rocprofsys::output::output_format;
using rocprofsys::output::process_metadata;
using rocprofsys::output::process_tree;
using rocprofsys::output::registry;

// ---------------------------------------------------------------------------
// registry
// ---------------------------------------------------------------------------

namespace
{
class RegistryTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rocprofsys::output::registry::instance().start_new_session();
    }
};
}  // namespace

TEST_F(RegistryTest, default_pid_resolves_to_getpid)
{
    registry::instance().register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                                       output_format::perfetto);
    const auto rows = registry::instance().rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().pid, getpid());
}

TEST_F(RegistryTest, start_new_session_clears_prior_rows)
{
    registry::instance().register_file("/tmp/rocprofsys-test/session1-a.proto",
                                       output_format::perfetto);
    EXPECT_EQ(registry::instance().rows().size(), 1u);

    registry::instance().start_new_session();

    EXPECT_TRUE(registry::instance().rows().empty());

    registry::instance().register_file("/tmp/rocprofsys-test/session2-a.proto",
                                       output_format::perfetto);
    registry::instance().register_file("/tmp/rocprofsys-test/session2-b.proto",
                                       output_format::perfetto);
    const auto rows_v2 = registry::instance().rows();
    EXPECT_EQ(rows_v2.size(), 2u);
    for(const auto& r : rows_v2)
        EXPECT_FALSE(r.path.empty());
}

TEST_F(RegistryTest, start_new_session_is_race_safe_with_concurrent_register)
{
    constexpr int WRITES_PER_ROUND = 50;

    std::atomic<bool> stop{ false };
    std::thread       writer([&]() {
        int i = 0;
        while(!stop.load(std::memory_order_relaxed))
        {
            registry::instance().register_file("/tmp/rocprofsys-test/stress-" +
                                                         std::to_string(i++) + ".proto",
                                                     output_format::perfetto);
        }
    });

    for(int round = 0; round < 5; ++round)
    {
        for(int i = 0; i < WRITES_PER_ROUND; ++i)
            std::this_thread::yield();
        registry::instance().start_new_session();
    }
    stop.store(true, std::memory_order_relaxed);
    writer.join();

    const auto rows = registry::instance().rows();
    for(const auto& r : rows)
        EXPECT_FALSE(r.path.empty());
}

TEST_F(RegistryTest, missing_file_yields_zero_size)
{
    namespace fs = std::filesystem;
    const auto missing =
        fs::temp_directory_path() / "rocprofsys-no-such-file-9b7c2.proto";
    fs::remove(missing);  // ensure absence

    registry::instance().register_file(missing.string(), output_format::perfetto);
    const auto rows = registry::instance().rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().size_bytes, 0u);
}

TEST_F(RegistryTest, existing_file_size_is_captured)
{
    namespace fs        = std::filesystem;
    const auto base_dir = fs::temp_directory_path() / "rocprofsys-registry-test";
    fs::create_directories(base_dir);
    const auto path = base_dir / "sized.bin";
    {
        std::ofstream     out(path, std::ios::binary);
        const std::string payload(2048, 'x');  // 2 KiB
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    }

    registry::instance().register_file(path.string(), output_format::perfetto);
    const auto rows = registry::instance().rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().size_bytes, 2048u);

    fs::remove(path);
}

TEST_F(RegistryTest, concurrent_register_is_thread_safe)
{
    constexpr int THREAD_COUNT = 4;
    constexpr int PER_THREAD   = 25;

    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);
    for(int t = 0; t < THREAD_COUNT; ++t)
    {
        threads.emplace_back([t]() {
            for(int i = 0; i < PER_THREAD; ++i)
            {
                registry::instance().register_file("/tmp/rocprofsys-test/concurrent-" +
                                                       std::to_string(t) + "-" +
                                                       std::to_string(i) + ".proto",
                                                   output_format::perfetto);
            }
        });
    }
    for(auto& th : threads)
        th.join();

    EXPECT_EQ(registry::instance().rows().size(),
              static_cast<std::size_t>(THREAD_COUNT * PER_THREAD));
}

TEST_F(RegistryTest, record_process_sparse_upsert_preserves_ppid)
{
    constexpr pid_t MAIN_PID = 1000;

    process_metadata rich;
    rich.pid     = MAIN_PID;
    rich.ppid    = 1;
    rich.command = "worker";
    registry::instance().record_process(rich);

    process_metadata sparse;
    sparse.pid     = MAIN_PID;
    sparse.ppid    = -1;  // sentinel: must not overwrite existing ppid
    sparse.command = "worker";
    registry::instance().record_process(sparse);

    const auto procs = registry::instance().processes();
    ASSERT_EQ(procs.size(), 1u);
    EXPECT_EQ(procs.front().ppid, 1);
}

TEST_F(RegistryTest, record_process_non_empty_fields_win_on_upsert)
{
    constexpr pid_t MAIN_PID = 1001;

    process_metadata sparse;
    sparse.pid     = MAIN_PID;
    sparse.ppid    = 1;
    sparse.command = "main";
    registry::instance().record_process(sparse);

    process_metadata rich;
    rich.pid     = MAIN_PID;
    rich.ppid    = 7;
    rich.command = "main-resolved";
    registry::instance().record_process(rich);

    const auto procs = registry::instance().processes();
    ASSERT_EQ(procs.size(), 1u);
    EXPECT_EQ(procs.front().ppid, 7);
    EXPECT_EQ(procs.front().command, "main-resolved");
}

// ---------------------------------------------------------------------------
// process_tree
// ---------------------------------------------------------------------------

namespace
{
artifact
make_row(std::string path, pid_t pid, std::uint64_t size_bytes = 0)
{
    artifact a{};
    a.path       = std::move(path);
    a.pid        = pid;
    a.size_bytes = size_bytes;
    a.format     = output_format::text;
    return a;
}

process_metadata
make_meta(pid_t pid, pid_t ppid, std::string command = "")
{
    process_metadata m{};
    m.pid     = pid;
    m.ppid    = ppid;
    m.command = std::move(command);
    return m;
}
}  // namespace

TEST(process_tree, single_pid_becomes_single_root)
{
    std::vector<artifact>         rows{ make_row("a", 100) };
    std::vector<process_metadata> processes{ make_meta(100, -1) };
    process_tree                  tree{ rows, processes };
    ASSERT_EQ(tree.roots().size(), 1u);
    EXPECT_EQ(tree.roots().front().meta.pid, 100);
    EXPECT_EQ(tree.roots().front().rows.size(), 1u);
    EXPECT_TRUE(tree.roots().front().children.empty());
    EXPECT_TRUE(tree.diagnostics().missing_metadata_pids.empty());
    EXPECT_TRUE(tree.diagnostics().cyclic_ppid_pids.empty());
}

TEST(process_tree, parent_with_two_children_nests_under_parent)
{
    std::vector<artifact>         rows{ make_row("p", 100), make_row("c1", 200),
                                make_row("c2", 201) };
    std::vector<process_metadata> processes{ make_meta(100, -1), make_meta(200, 100),
                                             make_meta(201, 100) };
    process_tree                  tree{ rows, processes };
    ASSERT_EQ(tree.roots().size(), 1u);
    ASSERT_EQ(tree.roots().front().children.size(), 2u);
    EXPECT_EQ(tree.roots().front().children[0].meta.pid, 200);
    EXPECT_EQ(tree.roots().front().children[1].meta.pid, 201);
}

TEST(process_tree, orphan_with_missing_ppid_attaches_at_root)
{
    std::vector<artifact>         rows{ make_row("p", 100), make_row("orphan", 999) };
    std::vector<process_metadata> processes{ make_meta(100, -1),
                                             make_meta(999, 12345 /* unknown ppid */) };
    process_tree                  tree{ rows, processes };
    ASSERT_EQ(tree.roots().size(), 2u);
    EXPECT_EQ(tree.roots()[0].meta.pid, 100);
    EXPECT_EQ(tree.roots()[1].meta.pid, 999);
}

TEST(process_tree, missing_metadata_pid_is_diagnosed)
{
    std::vector<artifact>         rows{ make_row("p", 100), make_row("ghost", 555) };
    std::vector<process_metadata> processes{ make_meta(100, -1) };
    process_tree                  tree{ rows, processes };
    EXPECT_EQ(tree.diagnostics().missing_metadata_pids, (std::vector<pid_t>{ 555 }));
    ASSERT_EQ(tree.roots().size(), 2u);
}

TEST(process_tree, ppid_cycle_excludes_members_and_is_diagnosed)
{
    std::vector<artifact>         rows{ make_row("a", 300), make_row("b", 400) };
    std::vector<process_metadata> processes{ make_meta(300, 400), make_meta(400, 300) };
    process_tree                  tree{ rows, processes };
    EXPECT_TRUE(tree.roots().empty());
    EXPECT_EQ(tree.diagnostics().cyclic_ppid_pids, (std::vector<pid_t>{ 300, 400 }));
}

TEST(process_tree, deep_parent_chain_builds_correctly)
{
    constexpr int                 CHAIN_DEPTH = 5;
    std::vector<process_metadata> processes;
    processes.reserve(CHAIN_DEPTH);
    for(pid_t pid = 1; pid <= CHAIN_DEPTH; ++pid)
        processes.push_back(make_meta(pid, pid == 1 ? -1 : pid - 1));

    std::vector<artifact> rows;
    rows.reserve(CHAIN_DEPTH);
    for(pid_t pid = 1; pid <= CHAIN_DEPTH; ++pid)
        rows.push_back(make_row(std::to_string(pid), pid));

    process_tree tree{ rows, processes };
    ASSERT_EQ(tree.roots().size(), 1u);
    EXPECT_EQ(tree.roots().front().meta.pid, 1);

    const auto* cur   = &tree.roots().front();
    int         depth = 1;
    while(!cur->children.empty())
    {
        ASSERT_EQ(cur->children.size(), 1u);
        cur = &cur->children.front();
        ++depth;
    }
    EXPECT_EQ(depth, CHAIN_DEPTH);
}

TEST(process_tree, rows_sorted_descending_by_size)
{
    std::vector<artifact>         rows{ make_row("small", 100, 1024),
                                make_row("large", 100, 1024ULL * 1024),
                                make_row("medium", 100, 4096) };
    std::vector<process_metadata> processes{ make_meta(100, -1) };
    process_tree                  tree{ rows, processes };
    ASSERT_EQ(tree.roots().size(), 1u);
    const auto& sorted_rows = tree.roots().front().rows;
    ASSERT_EQ(sorted_rows.size(), 3u);
    EXPECT_EQ(sorted_rows[0].path, "large");
    EXPECT_EQ(sorted_rows[1].path, "medium");
    EXPECT_EQ(sorted_rows[2].path, "small");
}

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
        registry::instance().register_file(row.path, row.format);
    for(const auto& proc : processes)
        registry::instance().record_process(proc);
    return registry::instance().format_summary(std::chrono::steady_clock::now());
}
}  // namespace

TEST(format_summary, empty_rows_prints_nothing) { EXPECT_TRUE(render({}, {}).empty()); }

TEST(format_summary, single_row_renders_all_header_fields)
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

TEST(format_summary, single_row_renders_full_absolute_path)
{
    std::vector<artifact> rows{ artifact{ "/tmp/rocprofsys-test/perfetto-trace.proto",
                                          getpid(), 0, output_format::perfetto } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("/tmp/rocprofsys-test/perfetto-trace.proto"), std::string::npos);
}

TEST(format_summary, single_row_renders_format_badge_name)
{
    std::vector<artifact> rows{ artifact{ "/tmp/rocprofsys-test/perfetto-trace.proto",
                                          getpid(), 0, output_format::perfetto } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("perfetto"), std::string::npos);
}

TEST(format_summary, single_row_renders_legend_entry)
{
    std::vector<artifact> rows{ artifact{ "/tmp/rocprofsys-test/perfetto-trace.proto",
                                          getpid(), 0, output_format::perfetto } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("perfetto → https://ui.perfetto.dev"), std::string::npos);
}

TEST(format_summary, multiple_formats_render_both_file_names)
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

TEST(format_summary, multiple_formats_render_both_legend_entries)
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

TEST(format_summary, peer_controlled_path_control_chars_are_stripped)
{
    std::vector<artifact>         rows{ artifact{
        "/tmp/rocprofsys-test/\x1b[31mevil\x1b[0m.proto", getpid(), 0,
        output_format::perfetto } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    // Only the triggering ESC byte is guaranteed gone; the CSI parameter
    // bytes are printable and are not stripped.
    EXPECT_EQ(out.find('\x1b'), std::string::npos);
    EXPECT_NE(out.find("evil"), std::string::npos);
    EXPECT_NE(out.find(".proto"), std::string::npos);
}

TEST(format_summary, relative_path_renders_as_absolute)
{
    std::vector<artifact> rows{ artifact{ "relative-dir/perfetto-trace.proto", getpid(),
                                          0, output_format::perfetto } };
    std::vector<process_metadata> processes{ process_metadata{ getpid(), -1, "self" } };

    const std::string out = render(rows, processes);
    EXPECT_NE(out.find("/relative-dir/perfetto-trace.proto"), std::string::npos);
}
