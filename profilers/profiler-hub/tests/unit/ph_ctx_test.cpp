// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler_hub_ctx.hpp"

#include "profiler-hub/cpp/storage.hpp"
#include "profiler-hub/cpp/writer.hpp"
#include "profiler-hub/cpp/writer_types.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

namespace
{

using namespace profiler_hub;

class ph_ctx_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_db_path             = (std::filesystem::temp_directory_path() /
                     (std::string{ "ph_ctx_test_" } + test_info->name() + ".db"))
                        .string();
        std::filesystem::remove(m_db_path);
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    [[nodiscard]] std::unique_ptr<writer_t> make_writer() const
    {
        return std::make_unique<writer_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    }

    // Registers node/process/thread, a track matching that topology, and one
    // region event on it, so ph_ctx sees exactly one non-empty track.
    void seed_track_with_one_event(writer_t& writer) const
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
    // Embedded verbatim into unquoted SQL table names by insert_statements, so it
    // must be a valid identifier fragment - no hyphens.
    std::string m_uuid = "testuuid0000";
};

TEST_F(ph_ctx_test, construction_populates_schema_version_and_node)
{
    auto writer = make_writer();
    seed_track_with_one_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };

    const auto version = ctx.get_storage_version();
    EXPECT_GT(version.major + version.minor + version.patch, 0U);

    const auto node = ctx.get_node();
    EXPECT_EQ(node.info.id, 1U);
    EXPECT_STREQ(node.info.machine_id, "machine-1");
}

TEST_F(ph_ctx_test, get_track_list_returns_seeded_non_empty_track)
{
    auto writer = make_writer();
    seed_track_with_one_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };

    const auto tracks = ctx.get_track_list();
    ASSERT_EQ(tracks.list_size, 1U);
    EXPECT_EQ(tracks.tracks[0].nid, 1U);
    EXPECT_EQ(tracks.tracks[0].pid, 100U);
    EXPECT_EQ(tracks.tracks[0].tid, 200U);
    EXPECT_EQ(tracks.tracks[0].event_count, 1U);
}

TEST_F(ph_ctx_test, has_track_reflects_seeded_and_unknown_ids)
{
    auto writer = make_writer();
    seed_track_with_one_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };

    const auto tracks = ctx.get_track_list();
    ASSERT_EQ(tracks.list_size, 1U);

    EXPECT_TRUE(ctx.has_track(tracks.tracks[0].id));
    EXPECT_FALSE(ctx.has_track(tracks.tracks[0].id + 1000));
}

TEST_F(ph_ctx_test, get_track_events_returns_event_for_seeded_track)
{
    auto writer = make_writer();
    seed_track_with_one_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };

    const auto tracks = ctx.get_track_list();
    ASSERT_EQ(tracks.list_size, 1U);

    const auto events = ctx.get_track_events(tracks.tracks[0].id, 0, 0);
    ASSERT_EQ(events.list_size, 1U);
    EXPECT_EQ(events.events[0].start, 1000U);
    EXPECT_EQ(events.events[0].end, 2000U);
}

TEST_F(ph_ctx_test, get_track_events_returns_empty_for_unknown_track)
{
    auto writer = make_writer();
    seed_track_with_one_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };

    const auto events = ctx.get_track_events(999999, 0, 0);
    EXPECT_EQ(events.list_size, 0U);
    EXPECT_EQ(events.events, nullptr);
}

TEST_F(ph_ctx_test, get_track_samples_returns_empty_for_non_pmc_track)
{
    auto writer = make_writer();
    seed_track_with_one_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };

    const auto tracks = ctx.get_track_list();
    ASSERT_EQ(tracks.list_size, 1U);

    const auto samples = ctx.get_track_samples(tracks.tracks[0].id, 0, 0);
    EXPECT_EQ(samples.list_size, 0U);
}

}  // namespace
