// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/agent.hpp"
#include "core/trace_cache/metadata_registry.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════
// metadata_registry save_to_file → load_from_file
// ═══════════════════════════════════════════════════════════════════════════

namespace
{
using rocprofsys::agent;
using rocprofsys::trace_cache::metadata_registry;

// Above UINT32_MAX, so 32-bit truncation is caught.
constexpr std::uint64_t k_start = 1'000'000'000'001'000;
constexpr std::uint64_t k_end   = 1'000'000'000'009'000;
static_assert(k_start > std::numeric_limits<std::uint32_t>::max());

// Not a realistic TID (Linux caps TIDs at 2^22). Tests that the JSON keeps 64 bits field
constexpr std::uint64_t k_thread_id = 5'000'000'300;
static_assert(k_thread_id > std::numeric_limits<std::uint32_t>::max());

constexpr pid_t k_pid  = 200;
constexpr pid_t k_ppid = 100;

void
set_test_process(metadata_registry& registry)
{
    registry.set_process({ .pid         = k_pid,
                           .ppid        = k_ppid,
                           .command     = "test_binary",
                           .environment = "{}",
                           .extdata     = "{}",
                           .start       = k_start,
                           .end         = k_end });
}
}  // namespace

class metadata_registry_json_test_interface : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_file_path =
            std::filesystem::temp_directory_path() /
            ("metadata_registry_json_test_" + std::to_string(::getpid()) + ".json");
    }

    void TearDown() override { std::filesystem::remove(m_file_path); }

    [[nodiscard]] bool round_trip(const metadata_registry& saved,
                                  metadata_registry&       loaded) const
    {
        const std::vector<std::shared_ptr<agent>> saved_agents;
        std::vector<std::shared_ptr<agent>>       loaded_agents;
        return saved.save_to_file(m_file_path.string(), saved_agents) &&
               loaded.load_from_file(m_file_path.string(), loaded_agents);
    }

private:
    std::filesystem::path m_file_path;
};

TEST_F(metadata_registry_json_test_interface, process_timestamps_survive_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    auto const process = loaded.get_process_info();
    EXPECT_EQ(process.start, k_start);
    EXPECT_EQ(process.end, k_end);
}

TEST_F(metadata_registry_json_test_interface, thread_timestamps_survive_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    saved.add_thread_info({ .parent_process_id = k_ppid,
                            .process_id        = k_pid,
                            .thread_id         = k_thread_id,
                            .start             = k_start,
                            .end               = k_end,
                            .extdata           = "{}" });

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    auto const threads = loaded.get_thread_info_list();
    ASSERT_EQ(threads.size(), 1U);
    EXPECT_EQ(threads[0].thread_id, k_thread_id);
    EXPECT_EQ(threads[0].start, k_start);
    EXPECT_EQ(threads[0].end, k_end);
}

TEST_F(metadata_registry_json_test_interface, track_thread_id_survives_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    saved.add_track(
        { .track_name = "test_track", .thread_id = k_thread_id, .extdata = "{}" });

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    auto const tracks = loaded.get_track_info_list();
    ASSERT_EQ(tracks.size(), 1U);
    EXPECT_EQ(tracks[0].thread_id, k_thread_id);
}

TEST_F(metadata_registry_json_test_interface, track_without_thread_id_survives_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    saved.add_track(
        { .track_name = "test_track", .thread_id = std::nullopt, .extdata = "{}" });

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    auto const tracks = loaded.get_track_info_list();
    ASSERT_EQ(tracks.size(), 1U);
    EXPECT_FALSE(tracks[0].thread_id.has_value());
}
