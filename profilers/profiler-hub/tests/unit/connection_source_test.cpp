// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/connection_pool.hpp"
#include "connection_source.hpp"
#include "pooled_connection_source.hpp"
#include "reader_catalog.hpp"

#include "profiler-hub/cpp/storage.hpp"
#include "profiler-hub/cpp/writer.hpp"
#include "profiler-hub/cpp/writer_types.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{

using namespace profiler_hub;

class counting_reader final : public track_row_reader
{
public:
    explicit counting_reader(int& outstanding)
    : m_outstanding{ outstanding }
    {
        ++m_outstanding;
    }

    ~counting_reader() override { --m_outstanding; }

    reader_types::timeline_event_list_t events_for_track(
        const reader_types::track_info_ptr_t&,
        const reader_types::event_filter_t&) override
    {
        return {};
    }

    reader_types::counter_timeline_event_list_t counter_events_for_track(
        const reader_types::track_info_ptr_t&,
        const reader_types::event_filter_t&) override
    {
        return {};
    }

    std::optional<std::pair<size_t, size_t>> event_id_span(
        reader_types::event_type_t) override
    {
        return std::pair<size_t, size_t>{ 3, 9 };
    }

    void visit_events_in_id_range(const reader_types::track_info_ptr_t&,
                                  reader_types::event_type_t,
                                  size_t,
                                  size_t,
                                  reader_t::event_visitor_t,
                                  void*) override
    {}

private:
    int& m_outstanding;
};

class limited_source final : public connection_source
{
public:
    explicit limited_source(int capacity)
    : m_capacity{ capacity }
    {}

    std::unique_ptr<track_row_reader> acquire() override
    {
        return std::make_unique<counting_reader>(m_outstanding);
    }

    std::unique_ptr<track_row_reader> try_acquire() override
    {
        if(m_outstanding >= m_capacity) return nullptr;
        return std::make_unique<counting_reader>(m_outstanding);
    }

    [[nodiscard]] int outstanding() const { return m_outstanding; }

private:
    int m_capacity;
    int m_outstanding{ 0 };
};

TEST(with_reader_test, returns_the_result_of_the_callable)
{
    limited_source source{ 1 };

    const auto span = with_reader(source, [](track_row_reader& reader) {
        return reader.event_id_span(reader_types::event_type_t::region);
    });

    ASSERT_TRUE(span.has_value());
    EXPECT_EQ(span->first, 3U);
    EXPECT_EQ(span->second, 9U);
}

TEST(with_reader_test, holds_the_reader_only_while_the_callable_runs)
{
    limited_source source{ 1 };
    int            outstanding_inside = -1;

    with_reader(source,
                [&](track_row_reader&) { outstanding_inside = source.outstanding(); });

    EXPECT_EQ(outstanding_inside, 1);
    EXPECT_EQ(source.outstanding(), 0);
}

TEST(with_reader_test, releases_the_reader_when_the_callable_throws)
{
    limited_source source{ 1 };

    EXPECT_THROW(
        with_reader(source,
                    [](track_row_reader&) { throw std::runtime_error("read failed"); }),
        std::runtime_error);

    EXPECT_EQ(source.outstanding(), 0);
}

class pooled_connection_source_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_db_path             = (std::filesystem::temp_directory_path() /
                     (std::string{ "pooled_connection_source_test_" } +
                      test_info->name() + ".db"))
                        .string();
        std::filesystem::remove(m_db_path);

        auto writer =
            std::make_unique<writer_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
        seed_one_region(*writer);
        writer->flush_in_memory_data_to_disk();
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    void seed_one_region(writer_t& writer) const
    {
        writer.register_node_info(writer_types::node_info_t{ 1, 42, "machine-1" });

        writer_types::process_info_t process_info;
        process_info.pid     = 100;
        process_info.node_id = 1;
        writer.register_process_info(process_info);

        writer_types::thread_info_t thread_info;
        thread_info.thread_id  = 200;
        thread_info.node_id    = 1;
        thread_info.process_id = 100;
        writer.register_thread_info(thread_info);

        writer_types::track_info_t track_info;
        track_info.node_id    = 1;
        track_info.process_id = 100;
        track_info.thread_id  = 200;
        writer.register_track_info(track_info);

        writer_types::trace_environment_t trace_environment;
        trace_environment.node_id    = 1;
        trace_environment.process_id = 100;
        trace_environment.thread_id  = 200;

        writer_types::event_data_t event_data;
        event_data.stack_id = 1;

        writer_types::region_data_t region_data;
        region_data.name            = "test-region";
        region_data.start_timestamp = 1000;
        region_data.end_timestamp   = 2000;
        region_data.event           = event_data;
        writer.insert_region_data(region_data, trace_environment);
    }

    std::string m_db_path;
    std::string m_uuid = "testuuid0000";
};

TEST_F(pooled_connection_source_test,
       try_acquire_returns_nullptr_when_the_pool_is_exhausted)
{
    common::connection_pool  pool{ m_db_path, 1, std::make_shared<reader_catalog_t>() };
    pooled_connection_source source{ pool };

    const auto held      = source.acquire();
    const auto exhausted = source.try_acquire();

    EXPECT_NE(held, nullptr);
    EXPECT_EQ(exhausted, nullptr);
}

TEST_F(pooled_connection_source_test,
       destroying_a_reader_returns_the_connection_to_the_pool)
{
    common::connection_pool  pool{ m_db_path, 1, std::make_shared<reader_catalog_t>() };
    pooled_connection_source source{ pool };

    auto held = source.acquire();
    held.reset();

    EXPECT_NE(source.try_acquire(), nullptr);
}

TEST_F(pooled_connection_source_test,
       reads_the_event_id_span_through_the_leased_connection)
{
    common::connection_pool  pool{ m_db_path, 1, std::make_shared<reader_catalog_t>() };
    pooled_connection_source source{ pool };

    const auto reader = source.acquire();
    const auto region = reader->event_id_span(reader_types::event_type_t::region);
    const auto copy   = reader->event_id_span(reader_types::event_type_t::memory_copy);

    ASSERT_TRUE(region.has_value());
    EXPECT_LE(region->first, region->second);
    EXPECT_FALSE(copy.has_value());
}

}  // namespace
