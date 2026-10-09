// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/c/profiler_hub.h"

#include "profiler-hub/cpp/storage.hpp"
#include "profiler-hub/cpp/writer.hpp"
#include "profiler-hub/cpp/writer_types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>

namespace
{

using namespace profiler_hub;

TEST(c_api_test, ph_ctx_create_null_ctx_returns_invalid_context)
{
    EXPECT_EQ(ph_ctx_create(nullptr, "unused.db", nullptr), PH_RESULT_INVALID_CONTEXT);
}

TEST(c_api_test, ph_ctx_free_null_returns_invalid_context)
{
    EXPECT_EQ(ph_ctx_free(nullptr), PH_RESULT_INVALID_CONTEXT);
}

TEST(c_api_test, ph_get_library_version_null_ctx_returns_invalid_context)
{
    ph_library_version_t version{};
    EXPECT_EQ(ph_get_library_version(nullptr, &version), PH_RESULT_INVALID_CONTEXT);
}

class c_api_real_ctx_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_db_path             = (std::filesystem::temp_directory_path() /
                     (std::string{ "c_api_test_" } + test_info->name() + ".db"))
                        .string();
        std::filesystem::remove(m_db_path);

        auto writer =
            std::make_unique<writer_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
        seed_trace(*writer);
        writer->flush_in_memory_data_to_disk();
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    virtual void seed_trace(writer_t& writer)
    {
        const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
        writer.register_node_info(node_info);
    }

    std::string m_db_path;
    std::string m_uuid = "testuuid0000";
};

class c_api_pmc_ctx_test : public c_api_real_ctx_test
{
protected:
    void seed_trace(writer_t& writer) override
    {
        c_api_real_ctx_test::seed_trace(writer);

        writer_types::process_info_t process_info;
        process_info.pid     = 100;
        process_info.node_id = 1;
        writer.register_process_info(process_info);

        writer_types::agent_info_t agent_info;
        agent_info.unique_id.agent_type = "GPU";
        agent_info.unique_id.type_index = 0;
        agent_info.uuid                 = 456;
        agent_info.node_id              = 1;
        agent_info.process_id           = 100;
        writer.register_agent_info(agent_info);

        writer_types::pmc_info_t pmc_info;
        pmc_info.unique_id.name     = "device_temp";
        pmc_info.unique_id.agent_id = agent_info.unique_id;
        pmc_info.symbol             = "device_temp";
        pmc_info.node_id            = 1;
        pmc_info.process_id         = 100;
        writer.register_pmc_info(pmc_info);

        writer_types::track_info_t track_info;
        track_info.name       = "counter-track";
        track_info.node_id    = 1;
        track_info.process_id = 100;
        writer.register_track_info(track_info);

        size_t timestamp = 1000;
        for(const double value : { 5.0, -2.5, 10.0 })
        {
            writer_types::pmc_event_data_t pmc_event_data;
            pmc_event_data.event.emplace();
            pmc_event_data.value            = value;
            pmc_event_data.sample.timestamp = timestamp;
            pmc_event_data.sample.track     = track_info;
            timestamp += 1000;
            writer.insert_pmc_event_data(pmc_event_data, pmc_info.unique_id);
        }
    }
};

TEST_F(c_api_real_ctx_test, create_and_free_succeeds_on_real_trace)
{
    ph_ctx_t ctx = nullptr;
    ASSERT_EQ(ph_ctx_create(&ctx, m_db_path.c_str(), nullptr), PH_RESULT_SUCCESS);
    ASSERT_NE(ctx, nullptr);

    ph_node_t node{};
    EXPECT_EQ(ph_get_node(ctx, &node, nullptr), PH_RESULT_SUCCESS);
    EXPECT_EQ(node.info.id, 1U);

    ph_track_list_t tracks{};
    EXPECT_EQ(ph_get_track_list(ctx, &tracks, nullptr), PH_RESULT_SUCCESS);
    EXPECT_EQ(tracks.list_size, 0U);

    EXPECT_EQ(ph_ctx_free(ctx), PH_RESULT_SUCCESS);
}

TEST_F(c_api_real_ctx_test, ph_get_library_version_returns_build_version)
{
    ph_ctx_t ctx = nullptr;
    ASSERT_EQ(ph_ctx_create(&ctx, m_db_path.c_str(), nullptr), PH_RESULT_SUCCESS);

    ph_library_version_t version{};
    EXPECT_EQ(ph_get_library_version(ctx, &version), PH_RESULT_SUCCESS);
    EXPECT_EQ(version.major, PROFILER_HUB_VERSION_MAJOR);
    EXPECT_EQ(version.minor, PROFILER_HUB_VERSION_MINOR);
    EXPECT_EQ(version.patch, PROFILER_HUB_VERSION_PATCH);

    ph_ctx_free(ctx);
}

class c_api_thread_ctx_test : public c_api_real_ctx_test
{
protected:
    void seed_trace(writer_t& writer) override
    {
        c_api_real_ctx_test::seed_trace(writer);

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

        writer_types::region_data_t region_data;
        region_data.name            = "test-region";
        region_data.start_timestamp = 1000;
        region_data.end_timestamp   = 2000;
        region_data.event.emplace();
        writer.insert_region_data(region_data, trace_environment);
    }
};

TEST_F(c_api_pmc_ctx_test, ph_get_track_list_pmc_track_exposes_valid_value_range)
{
    ph_ctx_t ctx = nullptr;
    ASSERT_EQ(ph_ctx_create(&ctx, m_db_path.c_str(), nullptr), PH_RESULT_SUCCESS);

    ph_track_list_t tracks{};
    ASSERT_EQ(ph_get_track_list(ctx, &tracks, nullptr), PH_RESULT_SUCCESS);

    const auto* pmc_track = std::find_if(
        tracks.tracks, tracks.tracks + tracks.list_size, [](const ph_track_t& track) {
            return track.category == PH_TRACK_CATEGORY_PMC_AGENT;
        });
    ASSERT_NE(pmc_track, tracks.tracks + tracks.list_size);
    EXPECT_NE(pmc_track->value_range.is_valid, 0U);
    EXPECT_DOUBLE_EQ(pmc_track->value_range.min, -2.5);
    EXPECT_DOUBLE_EQ(pmc_track->value_range.max, 10.0);

    ph_ctx_free(ctx);
}

TEST_F(c_api_pmc_ctx_test, ph_get_track_list_pmc_track_has_nesting_depth_zero)
{
    ph_ctx_t ctx = nullptr;
    ASSERT_EQ(ph_ctx_create(&ctx, m_db_path.c_str(), nullptr), PH_RESULT_SUCCESS);

    ph_track_list_t tracks{};
    ASSERT_EQ(ph_get_track_list(ctx, &tracks, nullptr), PH_RESULT_SUCCESS);

    const auto* pmc_track = std::find_if(
        tracks.tracks, tracks.tracks + tracks.list_size, [](const ph_track_t& track) {
            return track.category == PH_TRACK_CATEGORY_PMC_AGENT;
        });
    ASSERT_NE(pmc_track, tracks.tracks + tracks.list_size);
    EXPECT_EQ(pmc_track->nesting_depth, 0U);

    ph_ctx_free(ctx);
}

TEST_F(c_api_thread_ctx_test, ph_get_track_list_thread_track_value_range_is_not_valid)
{
    ph_ctx_t ctx = nullptr;
    ASSERT_EQ(ph_ctx_create(&ctx, m_db_path.c_str(), nullptr), PH_RESULT_SUCCESS);

    ph_track_list_t tracks{};
    ASSERT_EQ(ph_get_track_list(ctx, &tracks, nullptr), PH_RESULT_SUCCESS);
    ASSERT_EQ(tracks.list_size, 1U);

    ASSERT_EQ(tracks.tracks[0].category, PH_TRACK_CATEGORY_THREAD);
    EXPECT_EQ(tracks.tracks[0].value_range.is_valid, 0U);

    ph_ctx_free(ctx);
}

TEST_F(c_api_real_ctx_test, ph_get_track_list_null_track_list_returns_invalid_argument)
{
    ph_ctx_t ctx = nullptr;
    ASSERT_EQ(ph_ctx_create(&ctx, m_db_path.c_str(), nullptr), PH_RESULT_SUCCESS);

    EXPECT_EQ(ph_get_track_list(ctx, nullptr, nullptr), PH_RESULT_INVALID_ARGUMENT);

    ph_ctx_free(ctx);
}

TEST_F(c_api_real_ctx_test, get_track_events_unknown_track_returns_invalid_argument)
{
    ph_ctx_t ctx = nullptr;
    ASSERT_EQ(ph_ctx_create(&ctx, m_db_path.c_str(), nullptr), PH_RESULT_SUCCESS);

    ph_event_list_t events{};
    EXPECT_EQ(ph_get_track_events(ctx, 999999, 0, 0, &events, nullptr),
              PH_RESULT_INVALID_ARGUMENT);

    ph_ctx_free(ctx);
}

}  // namespace
