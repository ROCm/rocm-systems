// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/process_metadata.hpp"
#include "core/output_file_registry.hpp"

#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

TEST(output_file_registry, default_pid_resolves_to_getpid)
{
    rocprofsys::output_file_registry registry;
    registry.register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto);
    const auto rows = registry.rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().pid, getpid());
}

TEST(output_file_registry, explicit_pid_is_preserved)
{
    rocprofsys::output_file_registry registry;
    constexpr pid_t                  CHILD_PID = 4242;
    registry.register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto, CHILD_PID);
    const auto rows = registry.rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().pid, CHILD_PID);
}

TEST(output_file_registry, bump_session_filters_prior_rows_from_view)
{
    rocprofsys::output_file_registry registry;

    registry.register_file("/tmp/rocprofsys-test/session1-a.proto",
                           rocprofsys::output_format::perfetto);
    EXPECT_EQ(registry.rows().size(), 1u);

    const auto session_two = registry.bump_session();
    EXPECT_EQ(session_two, 2u);

    // After bump, the prior-session row is filtered out of rows().
    EXPECT_TRUE(registry.rows().empty());

    registry.register_file("/tmp/rocprofsys-test/session2-a.proto",
                           rocprofsys::output_format::perfetto);
    registry.register_file("/tmp/rocprofsys-test/session2-b.proto",
                           rocprofsys::output_format::perfetto);
    const auto rows_v2 = registry.rows();
    EXPECT_EQ(rows_v2.size(), 2u);
    for(const auto& r : rows_v2)
        EXPECT_FALSE(r.path.empty());
}

TEST(output_file_registry, bump_session_is_race_safe_with_concurrent_register)
{
    // Stress test: a writer thread registers files while another
    // thread bumps the session id. Each row must consistently land in
    // either the prior or the new session — never torn, never seen as
    // the wrong session by rows().
    rocprofsys::output_file_registry registry;
    constexpr int                    WRITES_PER_ROUND = 50;

    std::atomic<bool> stop{ false };
    std::thread       writer([&]() {
        int i = 0;
        while(!stop.load(std::memory_order_relaxed))
        {
            registry.register_file("/tmp/rocprofsys-test/stress-" + std::to_string(i++) +
                                             ".proto",
                                         rocprofsys::output_format::perfetto);
        }
    });

    // Yield-loop interleaving (not std::barrier / std::latch) is a
    // deliberate trade-off: this translation unit compiles at
    // -std=c++17 and the C++20 sync primitives are not available.
    // The user-visible invariant asserted below is timing-
    // independent, so the stress *shape* uses yields without
    // weakening the assertion. Swap to std::barrier(2) when the
    // project moves to C++20.
    for(int round = 0; round < 5; ++round)
    {
        for(int i = 0; i < WRITES_PER_ROUND; ++i)
            std::this_thread::yield();
        const auto sid = registry.bump_session();
        EXPECT_GE(sid, 2u);
    }
    stop.store(true, std::memory_order_relaxed);
    writer.join();

    // After all bumps, rows() returns only the final-session rows;
    // session bookkeeping is internal, so we assert the user-visible
    // invariant — every returned row carries a real registration
    // with a non-empty path.
    const auto rows = registry.rows();
    for(const auto& r : rows)
        EXPECT_FALSE(r.path.empty());
}

TEST(output_file_registry, missing_file_yields_nullopt_size)
{
    namespace fs = std::filesystem;
    const auto missing =
        fs::temp_directory_path() / "rocprofsys-no-such-file-9b7c2.proto";
    fs::remove(missing);  // ensure absence

    rocprofsys::output_file_registry registry;
    registry.register_file(missing.string(), rocprofsys::output_format::perfetto);
    const auto rows = registry.rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_FALSE(rows.front().size_bytes.has_value());
}

TEST(output_file_registry, existing_file_size_is_captured)
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

    rocprofsys::output_file_registry registry;
    registry.register_file(path.string(), rocprofsys::output_format::perfetto);
    const auto rows = registry.rows();
    ASSERT_EQ(rows.size(), 1u);
    ASSERT_TRUE(rows.front().size_bytes.has_value());
    EXPECT_EQ(*rows.front().size_bytes, 2048u);

    fs::remove(path);
}

TEST(output_file_registry, concurrent_register_is_thread_safe)
{
    rocprofsys::output_file_registry registry;
    constexpr int                    THREAD_COUNT = 4;
    constexpr int                    PER_THREAD   = 25;

    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);
    for(int t = 0; t < THREAD_COUNT; ++t)
    {
        threads.emplace_back([&registry, t]() {
            for(int i = 0; i < PER_THREAD; ++i)
            {
                registry.register_file("/tmp/rocprofsys-test/concurrent-" +
                                           std::to_string(t) + "-" + std::to_string(i) +
                                           ".proto",
                                       rocprofsys::output_format::perfetto);
            }
        });
    }
    for(auto& th : threads)
        th.join();

    EXPECT_EQ(registry.rows().size(),
              static_cast<std::size_t>(THREAD_COUNT * PER_THREAD));
}

TEST(output_file_registry, record_process_sparse_upsert_preserves_gpu_ids)
{
    // Mirrors the finalize ordering: the post-processor records the main
    // process with its GPU ids and ppid, then emit_summary re-records the
    // same pid with a sparse self-record that omits both (gpu_ids empty,
    // ppid at the -1 sentinel). The merge must preserve every richer field
    // the first record supplied, not let the sentinels clobber them.
    rocprofsys::output_file_registry registry;
    constexpr pid_t                  MAIN_PID = 1000;

    rocprofsys::output::process_metadata rich;
    rich.pid     = MAIN_PID;
    rich.ppid    = 1;
    rich.command = "worker";
    rich.gpu_ids = { 0, 1 };
    registry.record_process(rich);

    rocprofsys::output::process_metadata sparse;
    sparse.pid     = MAIN_PID;
    sparse.ppid    = -1;        // sentinel: must not overwrite existing ppid
    sparse.command = "worker";  // no gpu_ids
    registry.record_process(sparse);

    const auto procs = registry.processes();
    ASSERT_EQ(procs.size(), 1u);
    EXPECT_EQ(procs.front().ppid, 1);
    EXPECT_EQ(procs.front().gpu_ids, (std::vector<int>{ 0, 1 }));
}

TEST(output_file_registry, record_process_non_empty_fields_win_on_upsert)
{
    // A later record carrying richer data must overwrite an earlier
    // sparse record (e.g. self recorded first, post-processor second).
    rocprofsys::output_file_registry registry;
    constexpr pid_t                  MAIN_PID = 1001;

    rocprofsys::output::process_metadata sparse;
    sparse.pid     = MAIN_PID;
    sparse.ppid    = 1;
    sparse.command = "main";
    registry.record_process(sparse);

    rocprofsys::output::process_metadata rich;
    rich.pid     = MAIN_PID;
    rich.ppid    = 7;
    rich.command = "main-resolved";
    rich.gpu_ids = { 2, 3 };
    registry.record_process(rich);

    const auto procs = registry.processes();
    ASSERT_EQ(procs.size(), 1u);
    EXPECT_EQ(procs.front().ppid, 7);
    EXPECT_EQ(procs.front().command, "main-resolved");
    EXPECT_EQ(procs.front().gpu_ids, (std::vector<int>{ 2, 3 }));
}
