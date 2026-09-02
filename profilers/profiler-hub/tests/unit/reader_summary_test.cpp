// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "reader_test_fixture.hpp"

#include "profiler-hub/reader.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

using namespace profiler_hub;
using namespace profiler_hub::test;

// ============================================================================
// get_kernel_summary / get_region_summary — GROUP-BY-name aggregation
//
// Two synthetic summary fixtures share ONE by-construction oracle so both reader
// backends are asserted against identical expected aggregates: rocpd_v3_summary.db
// (durations = inline "end" - start) and rocpd_v4_summary.db (durations via the
// rocpd_timestamp spine). See fixtures/rocpd_v{3,4}_summary_data.sql for the full
// derivation. The oracle checks live in free helpers so v3 and v4 cannot drift.
//
//   KERNELS  kA "kA(int)"   : count 3, total 600, min 100, max 300, avg 200
//              (kd1,kd2 on symbol 1 + kd4 on symbol 3 — two symbol ids, one name,
//               so fold_summary_rows MERGES their GROUP BY rows into one bucket)
//            kB "kB(float)" : count 1, total 50,  min 50,  max 50,  avg 50
//   REGIONS  rX             : count 2, total 400, min 200, max 200, avg 200
//            rY             : count 1, total 600, min 600, max 600, avg 600
//   WINDOW kernels [1500,5000] -> kA count 2 (kd2,kd4), kB count 1.
//   WINDOW regions [0,750]     -> rX count 1 (r1 only), rY absent.
//   FAR-FUTURE window          -> empty list.
// ============================================================================

const profiler_hub::reader_types::event_summary_t*
find_summary(const profiler_hub::reader_types::event_summary_list_t& list,
             const std::string&                                      name)
{
    for(const auto& s : list)
        if(s.name == name) return &s;
    return nullptr;
}

void
expect_kernel_summary_oracle(const profiler_hub::reader_t& reader)
{
    auto list = reader.get_kernel_summary();
    ASSERT_EQ(list.size(), 2U);

    const auto* ka = find_summary(list, "kA(int)");
    const auto* kb = find_summary(list, "kB(float)");
    ASSERT_NE(ka, nullptr);
    ASSERT_NE(kb, nullptr);

    EXPECT_EQ(ka->count, 3U);
    EXPECT_EQ(ka->total_duration, 600U);
    EXPECT_EQ(ka->min_duration, 100U);
    EXPECT_EQ(ka->max_duration, 300U);
    EXPECT_EQ(ka->avg_duration, 200U);  // 600 / 3

    EXPECT_EQ(kb->count, 1U);
    EXPECT_EQ(kb->total_duration, 50U);
    EXPECT_EQ(kb->min_duration, 50U);
    EXPECT_EQ(kb->max_duration, 50U);
    EXPECT_EQ(kb->avg_duration, 50U);
}

void
expect_region_summary_oracle(const profiler_hub::reader_t& reader)
{
    auto list = reader.get_region_summary();
    ASSERT_EQ(list.size(), 2U);

    const auto* rx = find_summary(list, "rX");
    const auto* ry = find_summary(list, "rY");
    ASSERT_NE(rx, nullptr);
    ASSERT_NE(ry, nullptr);

    EXPECT_EQ(rx->count, 2U);
    EXPECT_EQ(rx->total_duration, 400U);
    EXPECT_EQ(rx->min_duration, 200U);
    EXPECT_EQ(rx->max_duration, 200U);
    EXPECT_EQ(rx->avg_duration, 200U);  // 400 / 2

    EXPECT_EQ(ry->count, 1U);
    EXPECT_EQ(ry->total_duration, 600U);
    EXPECT_EQ(ry->min_duration, 600U);
    EXPECT_EQ(ry->max_duration, 600U);
    EXPECT_EQ(ry->avg_duration, 600U);
}

void
expect_windowed_kernel_summary(const profiler_hub::reader_t& reader)
{
    profiler_hub::reader_types::time_window_t window;
    window.start = 1500;
    window.end   = 5000;

    auto list = reader.get_kernel_summary(window);
    ASSERT_EQ(list.size(), 2U);

    const auto* ka = find_summary(list, "kA(int)");
    const auto* kb = find_summary(list, "kB(float)");
    ASSERT_NE(ka, nullptr);
    ASSERT_NE(kb, nullptr);

    EXPECT_EQ(ka->count, 2U);             // kd1 [1000,1100] dropped; kd2 + kd4 kept
    EXPECT_EQ(ka->total_duration, 500U);  // 300 (kd2) + 200 (kd4)
    EXPECT_EQ(kb->count, 1U);             // kd3 [3000,3050] still inside the window

    auto        full    = reader.get_kernel_summary();
    const auto* ka_full = find_summary(full, "kA(int)");
    ASSERT_NE(ka_full, nullptr);
    EXPECT_LT(ka->count, ka_full->count);  // 2 < 3
}

void
expect_windowed_region_summary(const profiler_hub::reader_t& reader)
{
    profiler_hub::reader_types::time_window_t window;
    window.start = 0;
    window.end   = 750;

    auto list = reader.get_region_summary(window);
    ASSERT_EQ(list.size(), 1U);  // rY (r3) fully dropped, so it vanishes from the list

    const auto* rx = find_summary(list, "rX");
    ASSERT_NE(rx, nullptr);
    EXPECT_EQ(rx->count, 1U);             // r2 [800,1000] dropped, only r1 remains
    EXPECT_EQ(rx->total_duration, 200U);  // r1 duration alone
    EXPECT_EQ(find_summary(list, "rY"), nullptr);
}

void
expect_far_future_window_empty(const profiler_hub::reader_t& reader)
{
    profiler_hub::reader_types::time_window_t window;
    window.start = 1000000000;
    window.end   = 1000000001;
    EXPECT_TRUE(reader.get_kernel_summary(window).empty());
    EXPECT_TRUE(reader.get_region_summary(window).empty());
}

class reader_v4_summary_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_V4_SUMMARY_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v4_summary_test, kernel_summary_groups_by_name)
{
    expect_kernel_summary_oracle(*m_reader);
}
TEST_F(reader_v4_summary_test, region_summary_groups_by_name)
{
    expect_region_summary_oracle(*m_reader);
}
TEST_F(reader_v4_summary_test, kernel_summary_honors_time_window)
{
    expect_windowed_kernel_summary(*m_reader);
}
TEST_F(reader_v4_summary_test, region_summary_honors_time_window)
{
    expect_windowed_region_summary(*m_reader);
}
TEST_F(reader_v4_summary_test, far_future_window_yields_empty)
{
    expect_far_future_window_empty(*m_reader);
}

class reader_v3_summary_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_V3_SUMMARY_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_summary_test, kernel_summary_groups_by_name)
{
    expect_kernel_summary_oracle(*m_reader);
}
TEST_F(reader_v3_summary_test, region_summary_groups_by_name)
{
    expect_region_summary_oracle(*m_reader);
}
TEST_F(reader_v3_summary_test, kernel_summary_honors_time_window)
{
    expect_windowed_kernel_summary(*m_reader);
}
TEST_F(reader_v3_summary_test, region_summary_honors_time_window)
{
    expect_windowed_region_summary(*m_reader);
}
TEST_F(reader_v3_summary_test, far_future_window_yields_empty)
{
    expect_far_future_window_empty(*m_reader);
}
