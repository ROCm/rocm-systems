// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/artifact.hpp"
#include "core/output/process_metadata.hpp"
#include "core/output/registry.hpp"

#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

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

using rocprofsys::output::output_format;
using rocprofsys::output::process_metadata;
using rocprofsys::output::registry;

TEST_F(RegistryTest, default_pid_resolves_to_getpid)
{
    registry::instance().register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                                       output_format::perfetto);
    const auto rows = registry::instance().rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().pid, getpid());
}

TEST_F(RegistryTest, explicit_pid_is_preserved)
{
    constexpr pid_t CHILD_PID = 4242;
    registry::instance().register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                                       output_format::perfetto, CHILD_PID);
    const auto rows = registry::instance().rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().pid, CHILD_PID);
}

TEST_F(RegistryTest, start_new_session_filters_prior_rows_from_view)
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

TEST_F(RegistryTest, start_new_session_compacts_entries_older_than_the_ended_session)
{
    registry::instance().register_file("/tmp/rocprofsys-test/gen1.proto",
                                       output_format::perfetto);
    registry::instance()
        .start_new_session();  // gen1 becomes "just-ended", kept one cycle
    registry::instance().register_file("/tmp/rocprofsys-test/gen2.proto",
                                       output_format::perfetto);
    registry::instance().start_new_session();  // gen1 now compacted away; gen2 kept
    registry::instance().register_file("/tmp/rocprofsys-test/gen3.proto",
                                       output_format::perfetto);

    const auto rows = registry::instance().rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().path, "/tmp/rocprofsys-test/gen3.proto");
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
        const auto sid = registry::instance().start_new_session();
        EXPECT_GE(sid, 2u);
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
