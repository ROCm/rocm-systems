// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/reader.hpp"
#include "profiler-hub/storage.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

namespace
{

using namespace profiler_hub;

class reader_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_db_path             = (std::filesystem::temp_directory_path() /
                     (std::string{ "reader_test_" } + test_info->name() + ".db"))
                        .string();
        std::filesystem::remove(m_db_path);
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    [[nodiscard]] std::unique_ptr<writer_t> make_writer() const
    {
        return std::make_unique<writer_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    }

    [[nodiscard]] std::unique_ptr<reader_t> make_reader() const
    {
        return std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    }

    // Registers node/process/thread and inserts one region event with a call stack,
    // one arg, and a correlation stack_id, so the detail/property getters below have
    // real data to resolve.
    void seed_region_with_full_event(writer_t& writer) const
    {
        const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
        writer.register_node_info(node_info);

        writer_types::process_info_t process_info;
        process_info.pid     = 100;
        process_info.node_id = 1;
        writer.register_process_info(process_info);

        writer_types::thread_info_t thread_info;
        thread_info.thread_id  = 200;
        thread_info.node_id    = 1;
        thread_info.process_id = 100;
        writer.register_thread_info(thread_info);

        writer_types::trace_environment_t trace_environment;
        trace_environment.node_id    = 1;
        trace_environment.process_id = 100;
        trace_environment.thread_id  = 200;

        writer_types::event_data_t event_data;
        event_data.stack_id = 1;

        writer_types::arg_data_t arg;
        arg.position = 0;
        arg.type     = "int";
        arg.name     = "x";
        arg.value    = "5";

        writer_types::region_data_t region_data;
        region_data.name            = "test-region";
        region_data.start_timestamp = 1000;
        region_data.end_timestamp   = 2000;
        region_data.event           = event_data;
        region_data.args.push_back(arg);
        writer.insert_region_data(region_data, trace_environment);
    }

    std::string m_db_path;
    // Embedded verbatim into unquoted SQL table names by insert_statements, so it
    // must be a valid identifier fragment - no hyphens.
    std::string m_uuid = "testuuid0000";
};

TEST_F(reader_test, get_all_threads_is_readable_after_flush)
{
    auto writer = make_writer();

    const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
    writer->register_node_info(node_info);

    writer_types::process_info_t process_info;
    process_info.pid     = 100;
    process_info.node_id = 1;
    writer->register_process_info(process_info);

    writer_types::thread_info_t thread_info;
    thread_info.thread_id  = 200;
    thread_info.node_id    = 1;
    thread_info.process_id = 100;
    writer->register_thread_info(thread_info);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader  = make_reader();
    auto threads = reader->get_all_threads();

    ASSERT_EQ(threads.size(), 1);
    EXPECT_EQ(threads[0]->thread_id, 200);
}

TEST_F(reader_test, get_events_for_track_returns_empty_for_null_track)
{
    auto writer = make_writer();
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    EXPECT_TRUE(reader->get_events_for_track(nullptr).empty());
}

TEST_F(reader_test, get_events_for_track_returns_events_for_registered_track)
{
    auto writer = make_writer();

    const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
    writer->register_node_info(node_info);

    writer_types::track_info_t track_info;
    track_info.node_id = 1;
    writer->register_track_info(track_info);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto tracks = reader->get_all_tracks();
    ASSERT_EQ(tracks.size(), 1);

    EXPECT_TRUE(reader->get_events_for_track(tracks[0]).empty());
}

TEST_F(reader_test, get_region_details_returns_matching_data)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    auto region = reader->get_region_details(events[0]);
    ASSERT_TRUE(region.has_value());
    EXPECT_EQ(region->name, "test-region");
    EXPECT_EQ(region->start_timestamp, 1000);
    EXPECT_EQ(region->end_timestamp, 2000);
}

TEST_F(reader_test, get_kernel_dispatch_details_returns_nullopt_for_region_event)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_FALSE(reader->get_kernel_dispatch_details(events[0]).has_value());
}

TEST_F(reader_test, get_memory_copy_details_returns_nullopt_for_region_event)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_FALSE(reader->get_memory_copy_details(events[0]).has_value());
}

TEST_F(reader_test, get_memory_alloc_details_returns_nullopt_for_region_event)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_FALSE(reader->get_memory_alloc_details(events[0]).has_value());
}

TEST_F(reader_test, get_sample_details_returns_nullopt)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_FALSE(reader->get_sample_details(events[0]).has_value());
}

TEST_F(reader_test, get_pmc_event_details_returns_nullopt)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_FALSE(reader->get_pmc_event_details(events[0]).has_value());
}

TEST_F(reader_test, get_arguments_returns_registered_arg)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    auto args = reader->get_arguments(events[0]);
    ASSERT_EQ(args.size(), 1);
    EXPECT_EQ(args[0]->name, "x");
    EXPECT_EQ(args[0]->value, "5");
}

TEST_F(reader_test, get_call_stack_and_source_context_do_not_throw)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_NO_THROW(reader->get_call_stack(events[0]));
    EXPECT_NO_THROW(reader->get_source_context(events[0]));
}

TEST_F(reader_test, get_correlated_events_excludes_self)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_TRUE(reader->get_correlated_events(events[0]).empty());
}

TEST_F(reader_test, get_kernel_summary_and_region_summary_return_empty)
{
    auto writer = make_writer();
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();

    EXPECT_TRUE(reader->get_kernel_summary().empty());
    EXPECT_TRUE(reader->get_region_summary().empty());
}

TEST_F(reader_test, get_data_time_range_reflects_inserted_event)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto window = reader->get_data_time_range();

    ASSERT_TRUE(window.start.has_value());
    ASSERT_TRUE(window.end.has_value());
    EXPECT_EQ(window.start.value(), 1000);
    EXPECT_EQ(window.end.value(), 2000);
}

TEST_F(reader_test, get_event_counts_reflects_inserted_event)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto counts = reader->get_event_counts();

    EXPECT_EQ(counts.at(reader_types::event_type_t::region), 1);
}

}  // namespace
