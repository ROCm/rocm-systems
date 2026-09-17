// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/artifact.hpp"
#include "core/output/process_metadata.hpp"
#include "core/output/registry.hpp"

// Directly provides pid_t used below; clang-tidy's Include Cleaner flags this
// header as both redundant and required (a known false-positive with glibc's
// sys/types.h).
// NOLINTNEXTLINE(misc-include-cleaner)
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <thread>
#include <vector>

namespace
{
// GTest fixture convention is PascalCase; the AbstractClassCase "_interface"
// naming rule doesn't apply here.
// NOLINTNEXTLINE(readability-identifier-naming)
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
    // NOLINTNEXTLINE(misc-include-cleaner)
    constexpr pid_t k_child_pid = 4242;
    registry::instance().register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                                       output_format::perfetto, k_child_pid);
    const auto rows = registry::instance().rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().pid, k_child_pid);
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
    for(const auto& row : rows_v2)
    {
        EXPECT_FALSE(row.path.empty());
    }
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
    constexpr int k_writes_per_round = 50;
    constexpr int k_session_rounds   = 5;

    std::atomic<bool> stop{ false };
    std::thread       writer([&]() {
        int write_index = 0;
        while(!stop.load(std::memory_order_relaxed))
        {
            registry::instance().register_file(
                "/tmp/rocprofsys-test/stress-" + std::to_string(write_index++) + ".proto",
                output_format::perfetto);
        }
    });

    for(int round = 0; round < k_session_rounds; ++round)
    {
        for(int i = 0; i < k_writes_per_round; ++i)
        {
            std::this_thread::yield();
        }
        const auto sid = registry::instance().start_new_session();
        EXPECT_GE(sid, 2u);
    }
    stop.store(true, std::memory_order_relaxed);
    writer.join();

    const auto rows = registry::instance().rows();
    for(const auto& row : rows)
    {
        EXPECT_FALSE(row.path.empty());
    }
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

namespace
{
void
write_concurrent_files(int thread_index, int file_count)
{
    for(int i = 0; i < file_count; ++i)
    {
        registry::instance().register_file("/tmp/rocprofsys-test/concurrent-" +
                                               std::to_string(thread_index) + "-" +
                                               std::to_string(i) + ".proto",
                                           output_format::perfetto);
    }
}
}  // namespace

TEST_F(RegistryTest, concurrent_register_is_thread_safe)
{
    constexpr int k_thread_count = 4;
    constexpr int k_per_thread   = 25;

    std::vector<std::thread> threads;
    threads.reserve(k_thread_count);
    for(int thread_index = 0; thread_index < k_thread_count; ++thread_index)
    {
        threads.emplace_back(
            [thread_index]() { write_concurrent_files(thread_index, k_per_thread); });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(registry::instance().rows().size(),
              static_cast<std::size_t>(k_thread_count * k_per_thread));
}

TEST_F(RegistryTest, record_process_sparse_upsert_preserves_ppid)
{
    constexpr pid_t k_main_pid = 1000;

    process_metadata rich;
    rich.pid     = k_main_pid;
    rich.ppid    = 1;
    rich.command = "worker";
    registry::instance().record_process(rich);

    process_metadata sparse;
    sparse.pid     = k_main_pid;
    sparse.ppid    = -1;  // sentinel: must not overwrite existing ppid
    sparse.command = "worker";
    registry::instance().record_process(sparse);

    const auto procs = registry::instance().processes();
    ASSERT_EQ(procs.size(), 1u);
    EXPECT_EQ(procs.front().ppid, 1);
}

TEST_F(RegistryTest, record_process_non_empty_fields_win_on_upsert)
{
    constexpr pid_t k_main_pid      = 1001;
    constexpr pid_t k_resolved_ppid = 7;

    process_metadata sparse;
    sparse.pid     = k_main_pid;
    sparse.ppid    = 1;
    sparse.command = "main";
    registry::instance().record_process(sparse);

    process_metadata rich;
    rich.pid     = k_main_pid;
    rich.ppid    = k_resolved_ppid;
    rich.command = "main-resolved";
    registry::instance().record_process(rich);

    const auto procs = registry::instance().processes();
    ASSERT_EQ(procs.size(), 1u);
    EXPECT_EQ(procs.front().ppid, k_resolved_ppid);
    EXPECT_EQ(procs.front().command, "main-resolved");
}
