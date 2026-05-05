// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/discovery.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace rocprofsys::trace_cache::discovery
{
namespace
{
namespace fs = std::filesystem;

class temp_dir_fixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_dir = fs::temp_directory_path() /
                ("rocprofsys_discovery_test_" +
                 std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                 ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::create_directories(m_dir);
    }

    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(m_dir, ec);  // best effort
    }

    void touch(const std::string& filename)
    {
        std::ofstream{ (m_dir / filename).string() }.put('x');
    }

    [[nodiscard]] std::string path_of(const std::string& filename) const
    {
        return (m_dir / filename).string();
    }

    fs::path m_dir;
};
}  // namespace

TEST(discovery_test, list_dir_files_empty_path_returns_empty)
{
    EXPECT_TRUE(list_dir_files("").empty());
}

TEST(discovery_test, list_dir_files_nonexistent_path_throws)
{
    EXPECT_THROW(list_dir_files("/nonexistent_path_for_rocprofsys_test"),
                 std::runtime_error);
}

TEST_F(temp_dir_fixture, list_dir_files_returns_files_excluding_dot_entries)
{
    touch("a.txt");
    touch("b.bin");
    touch("c.json");

    auto files = list_dir_files(m_dir.string());

    EXPECT_EQ(files.size(), 3U);
    EXPECT_NE(std::find(files.begin(), files.end(), "a.txt"), files.end());
    EXPECT_NE(std::find(files.begin(), files.end(), "b.bin"), files.end());
    EXPECT_NE(std::find(files.begin(), files.end(), "c.json"), files.end());
    EXPECT_EQ(std::find(files.begin(), files.end(), "."), files.end());
    EXPECT_EQ(std::find(files.begin(), files.end(), ".."), files.end());
}

TEST(discovery_test, find_cache_files_empty_input_returns_empty)
{
    EXPECT_TRUE(find_cache_files(1234, {}).empty());
}

TEST(discovery_test, find_cache_files_matches_buffered_storage_pattern)
{
    auto m = find_cache_files(100, { "buffered_storage_100_42.bin" });
    ASSERT_EQ(m.size(), 1U);
    EXPECT_NE(m.find(42), m.end());
    EXPECT_NE(m[42].buff_storage.find("buffered_storage_100_42.bin"), std::string::npos);
}

TEST(discovery_test, find_cache_files_matches_metadata_pattern)
{
    auto m = find_cache_files(100, { "metadata_100_42.json" });
    ASSERT_EQ(m.size(), 1U);
    EXPECT_NE(m.find(42), m.end());
    EXPECT_NE(m[42].metadata.find("metadata_100_42.json"), std::string::npos);
}

TEST(discovery_test, find_cache_files_pairs_buffered_and_metadata_for_same_pid)
{
    auto m =
        find_cache_files(100, { "buffered_storage_100_42.bin", "metadata_100_42.json" });
    ASSERT_EQ(m.size(), 1U);
    EXPECT_FALSE(m[42].empty());
}

TEST(discovery_test, find_cache_files_skips_mismatched_parent_pid)
{
    auto m = find_cache_files(100, { "buffered_storage_999_42.bin" });
    EXPECT_TRUE(m.empty());
}

TEST(discovery_test, find_cache_files_skips_unparsable_pid_without_aborting)
{
    // A PID that overflows int (regression: try/catch around stoi).
    auto m = find_cache_files(100, { "buffered_storage_99999999999999999999_42.bin",
                                     "buffered_storage_100_42.bin" });
    EXPECT_EQ(m.size(), 1U);
    EXPECT_NE(m.find(42), m.end());
}

TEST_F(temp_dir_fixture, clear_removes_existing_files)
{
    touch("buff.bin");
    touch("meta.json");

    data::mapped_cache_files_t cache;
    cache[1].buff_storage = path_of("buff.bin");
    cache[1].metadata     = path_of("meta.json");

    clear(cache);

    EXPECT_FALSE(fs::exists(path_of("buff.bin")));
    EXPECT_FALSE(fs::exists(path_of("meta.json")));
}

TEST(discovery_test, clear_does_not_throw_on_missing_files)
{
    data::mapped_cache_files_t cache;
    cache[1].buff_storage = "/tmp/rocprofsys_does_not_exist_buff.bin";
    cache[1].metadata     = "/tmp/rocprofsys_does_not_exist_meta.json";

    EXPECT_NO_THROW(clear(cache));
}

}  // namespace rocprofsys::trace_cache::discovery
