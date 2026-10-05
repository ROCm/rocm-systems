// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/connection_pool.hpp"
#include "common/thread_pool.hpp"
#include "populate_reader_catalog.hpp"
#include "profiler-hub/cpp/reader.hpp"
#include "reader_catalog.hpp"
#include "trace_fixtures.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sqlite3.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace
{

using namespace profiler_hub;

class populate_reader_catalog_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_db_path = test::temp_trace_path("populate_reader_catalog_test");
        test::write_thread_track_with_two_regions(m_db_path);
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    void drop_table(const std::string& table) const
    {
        sqlite3* db = nullptr;
        ASSERT_EQ(sqlite3_open(m_db_path.c_str(), &db), SQLITE_OK);
        ASSERT_EQ(
            sqlite3_exec(db, ("DROP TABLE " + table).c_str(), nullptr, nullptr, nullptr),
            SQLITE_OK);
        sqlite3_close(db);
    }

    std::string m_db_path;
};

std::vector<std::string>
track_names(const reader_t& reader)
{
    std::vector<std::string> names;
    for(const auto& track : reader.get_all_tracks())
    {
        names.push_back(track->name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

TEST_F(populate_reader_catalog_test, fills_the_shared_catalog_like_a_standalone_reader)
{
    const reader_t standalone{ std::make_unique<storage_t>(m_db_path, "") };

    auto                    catalog = std::make_shared<reader_catalog_t>();
    common::thread_pool     workers{ 4 };
    common::connection_pool connections{ m_db_path, 4, catalog };
    populate_reader_catalog(workers, connections, *catalog);

    std::vector<std::string> populated;
    for(const auto& track : catalog->tracks)
    {
        populated.push_back(track->name);
    }
    std::sort(populated.begin(), populated.end());
    EXPECT_EQ(populated, track_names(standalone));
    EXPECT_EQ(catalog->nodes.size(), standalone.get_all_nodes().size());
    EXPECT_EQ(catalog->processes.size(), standalone.get_all_processes().size());
    EXPECT_EQ(catalog->threads.size(), standalone.get_all_threads().size());
    EXPECT_FALSE(catalog->string_utility.empty());
}

TEST_F(populate_reader_catalog_test, works_with_a_single_worker_and_a_single_connection)
{
    auto                    catalog = std::make_shared<reader_catalog_t>();
    common::thread_pool     workers{ 1 };
    common::connection_pool connections{ m_db_path, 1, catalog };

    populate_reader_catalog(workers, connections, *catalog);

    EXPECT_EQ(catalog->tracks.size(), 1U);
}

TEST_F(populate_reader_catalog_test,
       a_failing_query_is_reported_and_the_pools_stay_usable)
{
    auto                    catalog = std::make_shared<reader_catalog_t>();
    common::thread_pool     workers{ 4 };
    common::connection_pool connections{ m_db_path, 4, catalog };
    drop_table(std::string{ "rocpd_info_node_" } + std::string{ test::trace_uuid });

    EXPECT_ANY_THROW(populate_reader_catalog(workers, connections, *catalog));

    std::atomic<bool> ran{ false };
    workers.submit([&](const std::stop_token&) { ran = true; }).wait();
    EXPECT_TRUE(ran);

    std::vector<std::optional<common::connection_pool::lease>> leases;
    for(int i = 0; i < 4; ++i)
    {
        leases.push_back(connections.try_acquire());
        EXPECT_TRUE(leases.back().has_value())
            << "connection " << i << " was not returned";
    }
}

}  // namespace
