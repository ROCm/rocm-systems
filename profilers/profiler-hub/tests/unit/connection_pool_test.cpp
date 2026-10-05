// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/connection_pool.hpp"
#include "reader_catalog.hpp"
#include "trace_fixtures.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace
{

using namespace profiler_hub;
using namespace std::chrono_literals;

class connection_pool_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_db_path = test::temp_trace_path("connection_pool_test");
        test::write_thread_track_with_two_regions(m_db_path);
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    [[nodiscard]] std::unique_ptr<common::connection_pool> make_pool(size_t size) const
    {
        return std::make_unique<common::connection_pool>(
            m_db_path, size, std::make_shared<reader_catalog_t>());
    }

    std::string m_db_path;
};

TEST_F(connection_pool_test, try_acquire_returns_nothing_when_every_connection_is_leased)
{
    const auto pool = make_pool(2);

    const auto first  = pool->try_acquire();
    const auto second = pool->try_acquire();
    const auto third  = pool->try_acquire();

    EXPECT_TRUE(first.has_value());
    EXPECT_TRUE(second.has_value());
    EXPECT_FALSE(third.has_value());
}

TEST_F(connection_pool_test, a_released_connection_can_be_leased_again)
{
    const auto pool = make_pool(1);

    {
        const auto lease = pool->try_acquire();
        ASSERT_TRUE(lease.has_value());
    }

    EXPECT_TRUE(pool->try_acquire().has_value());
}

TEST_F(connection_pool_test, a_moved_lease_is_released_exactly_once)
{
    const auto pool = make_pool(1);

    {
        auto first  = pool->acquire();
        auto second = std::move(first);
    }

    const auto lease     = pool->try_acquire();
    const auto exhausted = pool->try_acquire();
    EXPECT_TRUE(lease.has_value());
    EXPECT_FALSE(exhausted.has_value());
}

TEST_F(connection_pool_test, different_leases_hold_different_connections)
{
    const auto pool = make_pool(2);

    const auto first  = pool->try_acquire();
    const auto second = pool->try_acquire();

    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_NE(&**first, &**second);
}

TEST_F(connection_pool_test, acquire_blocks_until_a_connection_is_released)
{
    const auto        pool = make_pool(1);
    auto              held = std::make_optional(pool->acquire());
    std::atomic<bool> acquired{ false };

    std::thread waiter{ [&] {
        const auto lease = pool->acquire();
        acquired      = true;
    } };
    std::this_thread::sleep_for(100ms);
    const bool acquired_while_held = acquired;
    held.reset();
    waiter.join();

    EXPECT_FALSE(acquired_while_held);
    EXPECT_TRUE(acquired);
}

TEST_F(connection_pool_test, run_sync_returns_the_result_of_the_callable)
{
    const auto pool = make_pool(1);

    const auto span = pool->run_sync([](common::connection& connection) {
        return connection.reader().get_event_id_span(reader_types::event_type_t::region);
    });

    EXPECT_TRUE(span.has_value());
}

TEST_F(connection_pool_test, run_sync_gives_the_connection_back_after_the_call)
{
    const auto pool = make_pool(1);

    pool->run_sync([](common::connection&) { return 0; });

    EXPECT_TRUE(pool->try_acquire().has_value());
}

TEST_F(connection_pool_test, run_sync_gives_the_connection_back_when_the_callable_throws)
{
    const auto pool = make_pool(1);

    EXPECT_THROW(pool->run_sync([](common::connection&) -> int {
        throw std::runtime_error("read failed");
    }),
                 std::runtime_error);

    EXPECT_TRUE(pool->try_acquire().has_value());
}

TEST_F(connection_pool_test, connections_of_a_pool_share_the_catalog)
{
    auto                    catalog = std::make_shared<reader_catalog_t>();
    common::connection_pool pool{ m_db_path, 2, catalog };

    EXPECT_GE(catalog.use_count(), 2);
}

}  // namespace
