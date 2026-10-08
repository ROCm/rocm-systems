// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/c/profiler_hub.h"
#include "trace_fixtures.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace
{

using namespace profiler_hub;

class c_api_track_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_db_path = test::temp_trace_path("c_api_track_test");
        test::write_thread_track_with_two_regions(m_db_path);
        ASSERT_EQ(ph_ctx_create(&m_ctx, m_db_path.c_str()), PH_RESULT_SUCCESS);

        ph_track_list_t tracks{};
        ASSERT_EQ(ph_get_track_list(m_ctx, &tracks), PH_RESULT_SUCCESS);
        ASSERT_EQ(tracks.list_size, 1U);
        m_track_id = tracks.tracks[0].id;
    }

    void TearDown() override
    {
        if(m_ctx != nullptr) ph_ctx_free(m_ctx);
        std::filesystem::remove(m_db_path);
    }

    static std::vector<std::string> names_of(const ph_event_list_t& list)
    {
        std::vector<std::string> names;
        for(uint32_t i = 0; i < list.list_size; ++i)
        {
            names.emplace_back(list.events[i].name);
        }
        return names;
    }

    std::string m_db_path;
    ph_ctx_t    m_ctx{ nullptr };
    uint32_t    m_track_id{ 0 };
};

TEST_F(c_api_track_test, the_track_list_describes_the_seeded_thread_track)
{
    ph_track_list_t tracks{};
    ASSERT_EQ(ph_get_track_list(m_ctx, &tracks), PH_RESULT_SUCCESS);

    EXPECT_EQ(tracks.tracks[0].category, PH_TRACK_CATEGORY_THREAD);
    EXPECT_EQ(tracks.tracks[0].event_count, 2U);
    EXPECT_EQ(tracks.tracks[0].start_ts, 1000U);
    EXPECT_EQ(tracks.tracks[0].end_ts, 6000U);
}

TEST_F(c_api_track_test, the_track_list_stays_valid_across_later_calls)
{
    ph_track_list_t first{};
    ph_track_list_t second{};
    ASSERT_EQ(ph_get_track_list(m_ctx, &first), PH_RESULT_SUCCESS);
    ASSERT_EQ(ph_get_track_list(m_ctx, &second), PH_RESULT_SUCCESS);

    EXPECT_EQ(first.tracks, second.tracks);
}

TEST_F(c_api_track_test, whole_track_events_come_in_start_order)
{
    ph_event_list_t events{};
    ASSERT_EQ(ph_get_track_events(m_ctx, m_track_id, 0, 0, &events), PH_RESULT_SUCCESS);

    EXPECT_THAT(names_of(events), ::testing::ElementsAre("region-a", "region-b"));
    EXPECT_EQ(events.events[0].start, 1000U);
    EXPECT_EQ(events.events[0].end, 2000U);
}

TEST_F(c_api_track_test, whole_track_events_share_their_storage_between_calls)
{
    ph_event_list_t first{};
    ph_event_list_t second{};
    ASSERT_EQ(ph_get_track_events(m_ctx, m_track_id, 0, 0, &first), PH_RESULT_SUCCESS);
    ASSERT_EQ(ph_get_track_events(m_ctx, m_track_id, 0, 0, &second), PH_RESULT_SUCCESS);

    EXPECT_EQ(first.events, second.events);
}

TEST_F(c_api_track_test, a_window_returns_the_events_that_overlap_it)
{
    ph_event_list_t events{};

    ASSERT_EQ(ph_get_track_events(m_ctx, m_track_id, 4000, 7000, &events),
              PH_RESULT_SUCCESS);
    EXPECT_THAT(names_of(events), ::testing::ElementsAre("region-b"));

    ASSERT_EQ(ph_get_track_events(m_ctx, m_track_id, 1500, 1800, &events),
              PH_RESULT_SUCCESS);
    EXPECT_THAT(names_of(events), ::testing::ElementsAre("region-a"));

    ASSERT_EQ(ph_get_track_events(m_ctx, m_track_id, 3000, 4000, &events),
              PH_RESULT_SUCCESS);
    EXPECT_EQ(events.list_size, 0U);
}

TEST_F(c_api_track_test, a_window_that_covers_everything_matches_the_whole_track)
{
    ph_event_list_t whole{};
    ph_event_list_t window{};
    ASSERT_EQ(ph_get_track_events(m_ctx, m_track_id, 0, 0, &whole), PH_RESULT_SUCCESS);
    ASSERT_EQ(ph_get_track_events(m_ctx, m_track_id, 1, 100000, &window),
              PH_RESULT_SUCCESS);

    EXPECT_THAT(names_of(window), ::testing::UnorderedElementsAreArray(names_of(whole)));
}

TEST_F(c_api_track_test, a_huge_end_bound_means_no_upper_bound)
{
    ph_event_list_t events{};

    ASSERT_EQ(ph_get_track_events(m_ctx, m_track_id, 1, UINT64_MAX, &events),
              PH_RESULT_SUCCESS);

    EXPECT_EQ(events.list_size, 2U);
}

TEST_F(c_api_track_test, a_window_starting_in_the_future_is_empty)
{
    ph_event_list_t events{};

    ASSERT_EQ(ph_get_track_events(m_ctx, m_track_id, UINT64_MAX, UINT64_MAX, &events),
              PH_RESULT_SUCCESS);

    EXPECT_EQ(events.list_size, 0U);
}

TEST_F(c_api_track_test, a_window_result_stays_valid_after_a_later_window)
{
    ph_event_list_t first{};
    ph_event_list_t second{};
    ASSERT_EQ(ph_get_track_events(m_ctx, m_track_id, 1500, 1800, &first),
              PH_RESULT_SUCCESS);
    ASSERT_EQ(ph_get_track_events(m_ctx, m_track_id, 4000, 7000, &second),
              PH_RESULT_SUCCESS);

    EXPECT_THAT(names_of(first), ::testing::ElementsAre("region-a"));
    EXPECT_THAT(names_of(second), ::testing::ElementsAre("region-b"));
}

TEST_F(c_api_track_test, an_unknown_track_clears_the_event_list_and_reports_the_argument)
{
    ph_event_list_t events{ .list_size = 7,
                            .events    = reinterpret_cast<ph_event_t*>(0x1) };

    EXPECT_EQ(ph_get_track_events(m_ctx, 999999, 0, 0, &events),
              PH_RESULT_INVALID_ARGUMENT);

    EXPECT_EQ(events.list_size, 0U);
    EXPECT_EQ(events.events, nullptr);
}

TEST_F(c_api_track_test, a_null_event_list_is_an_invalid_argument)
{
    EXPECT_EQ(ph_get_track_events(m_ctx, m_track_id, 0, 0, nullptr),
              PH_RESULT_INVALID_ARGUMENT);
}

TEST_F(c_api_track_test, a_null_context_is_rejected_by_the_event_and_sample_getters)
{
    ph_event_list_t  events{};
    ph_sample_list_t samples{};

    EXPECT_EQ(ph_get_track_events(nullptr, m_track_id, 0, 0, &events),
              PH_RESULT_INVALID_CONTEXT);
    EXPECT_EQ(ph_get_track_samples(nullptr, m_track_id, 0, 0, &samples),
              PH_RESULT_INVALID_CONTEXT);
}

TEST_F(c_api_track_test, samples_of_a_thread_track_are_empty)
{
    ph_sample_list_t samples{ .list_size = 5,
                              .samples   = reinterpret_cast<ph_sample_t*>(0x1) };

    ASSERT_EQ(ph_get_track_samples(m_ctx, m_track_id, 0, 0, &samples), PH_RESULT_SUCCESS);

    EXPECT_EQ(samples.list_size, 0U);
}

TEST_F(c_api_track_test, an_unknown_track_clears_the_sample_list_and_reports_the_argument)
{
    ph_sample_list_t samples{ .list_size = 5,
                              .samples   = reinterpret_cast<ph_sample_t*>(0x1) };

    EXPECT_EQ(ph_get_track_samples(m_ctx, 999999, 0, 0, &samples),
              PH_RESULT_INVALID_ARGUMENT);

    EXPECT_EQ(samples.list_size, 0U);
    EXPECT_EQ(samples.samples, nullptr);
}

TEST_F(c_api_track_test, the_node_describes_the_seeded_machine)
{
    ph_node_t node{};

    ASSERT_EQ(ph_get_node(m_ctx, &node), PH_RESULT_SUCCESS);

    EXPECT_EQ(node.info.id, 1U);
    EXPECT_STREQ(node.info.machine_id, "machine-1");
    EXPECT_EQ(node.track_list.list_size, 1U);
}

TEST_F(c_api_track_test, the_node_arguments_are_checked)
{
    ph_node_t node{};

    EXPECT_EQ(ph_get_node(nullptr, &node), PH_RESULT_INVALID_CONTEXT);
    EXPECT_EQ(ph_get_node(m_ctx, nullptr), PH_RESULT_INVALID_ARGUMENT);
}

TEST_F(c_api_track_test, the_schema_version_is_reported)
{
    ph_schema_version_t version{};

    ASSERT_EQ(ph_get_schema_version(m_ctx, &version), PH_RESULT_SUCCESS);

    EXPECT_GT(version.major + version.minor + version.patch, 0U);
    EXPECT_EQ(ph_get_schema_version(nullptr, &version), PH_RESULT_INVALID_CONTEXT);
    EXPECT_EQ(ph_get_schema_version(m_ctx, nullptr), PH_RESULT_INVALID_ARGUMENT);
}

TEST(c_api_create_test, a_null_path_is_an_invalid_argument_and_clears_the_context)
{
    ph_ctx_t ctx = reinterpret_cast<ph_ctx_t>(0x1);

    EXPECT_EQ(ph_ctx_create(&ctx, nullptr), PH_RESULT_INVALID_ARGUMENT);

    EXPECT_EQ(ctx, nullptr);
}

TEST(c_api_create_test, a_missing_trace_fails_without_creating_files)
{
    const auto directory =
        std::filesystem::temp_directory_path() / "c_api_create_test_missing_directory";
    std::filesystem::remove_all(directory);
    ph_ctx_t ctx = reinterpret_cast<ph_ctx_t>(0x1);

    EXPECT_EQ(ph_ctx_create(&ctx, (directory / "trace.db").string().c_str()),
              PH_RESULT_CONTEXT_ALLOCATION_FAILED);

    EXPECT_EQ(ctx, nullptr);
    EXPECT_FALSE(std::filesystem::exists(directory));
}

TEST(c_api_create_test, a_directory_is_not_a_trace)
{
    ph_ctx_t ctx = nullptr;

    EXPECT_EQ(
        ph_ctx_create(&ctx, std::filesystem::temp_directory_path().string().c_str()),
        PH_RESULT_CONTEXT_ALLOCATION_FAILED);

    EXPECT_EQ(ctx, nullptr);
}

}  // namespace
