// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/c/profiler_hub.h"

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

TEST(c_api_test, ph_ctx_create_null_ctx_returns_invalid_context)
{
    EXPECT_EQ(ph_ctx_create(nullptr, "unused.db"), PH_RESULT_INVALID_CONTEXT);
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
        const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
        writer->register_node_info(node_info);
        writer->flush_in_memory_data_to_disk();
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    std::string m_db_path;
    std::string m_uuid = "testuuid0000";
};

TEST_F(c_api_real_ctx_test, create_and_free_succeeds_on_real_trace)
{
    ph_ctx_t ctx = nullptr;
    ASSERT_EQ(ph_ctx_create(&ctx, m_db_path.c_str()), PH_RESULT_SUCCESS);
    ASSERT_NE(ctx, nullptr);

    ph_node_t node{};
    EXPECT_EQ(ph_get_node(ctx, &node), PH_RESULT_SUCCESS);
    EXPECT_EQ(node.info.id, 1U);

    ph_track_list_t tracks{};
    EXPECT_EQ(ph_get_track_list(ctx, &tracks), PH_RESULT_SUCCESS);
    EXPECT_EQ(tracks.list_size, 0U);

    EXPECT_EQ(ph_ctx_free(ctx), PH_RESULT_SUCCESS);
}

TEST_F(c_api_real_ctx_test, ph_get_track_list_null_track_list_returns_invalid_argument)
{
    ph_ctx_t ctx = nullptr;
    ASSERT_EQ(ph_ctx_create(&ctx, m_db_path.c_str()), PH_RESULT_SUCCESS);

    EXPECT_EQ(ph_get_track_list(ctx, nullptr), PH_RESULT_INVALID_ARGUMENT);

    ph_ctx_free(ctx);
}

TEST_F(c_api_real_ctx_test, get_track_events_unknown_track_returns_invalid_argument)
{
    ph_ctx_t ctx = nullptr;
    ASSERT_EQ(ph_ctx_create(&ctx, m_db_path.c_str()), PH_RESULT_SUCCESS);

    ph_event_list_t events{};
    EXPECT_EQ(ph_get_track_events(ctx, 999999, 0, 0, &events),
              PH_RESULT_INVALID_ARGUMENT);

    ph_ctx_free(ctx);
}

}  // namespace
