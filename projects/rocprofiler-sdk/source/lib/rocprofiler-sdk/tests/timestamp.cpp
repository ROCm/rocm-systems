// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>

namespace
{
class profiling_time_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        core_backup = *rocprofiler::hsa::get_core_table();
        amd_backup  = *rocprofiler::hsa::get_amd_ext_table();
        query_time  = {};
        future_end  = false;
        rocprofiler::hsa::get_core_table()->hsa_system_get_info_fn = [](hsa_system_info_t attribute,
                                                                        void*             value) {
            if(attribute != HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY)
                return HSA_STATUS_ERROR_INVALID_ARGUMENT;
            *static_cast<uint64_t*>(value) = 1'000'000'000;
            return HSA_STATUS_SUCCESS;
        };
        rocprofiler::hsa::get_amd_ext_table()->hsa_amd_profiling_get_dispatch_time_fn =
            [](hsa_agent_t, hsa_signal_t, hsa_amd_profiling_dispatch_time_t* value) {
                if(future_end) query_time.end = rocprofiler::common::timestamp_ns() + 1'000;
                *value = query_time;
                return HSA_STATUS_SUCCESS;
            };
        rocprofiler::hsa::get_amd_ext_table()->hsa_amd_profiling_get_async_copy_time_fn =
            [](hsa_signal_t, hsa_amd_profiling_async_copy_time_t*) { return HSA_STATUS_SUCCESS; };
    }

    void TearDown() override
    {
        *rocprofiler::hsa::get_core_table()    = core_backup;
        *rocprofiler::hsa::get_amd_ext_table() = amd_backup;
    }

    static inline hsa_amd_profiling_dispatch_time_t query_time  = {};
    static inline bool                              future_end  = false;
    rocprofiler::hsa::hsa_core_table_t              core_backup = {};
    rocprofiler::hsa::hsa_amd_ext_table_t           amd_backup  = {};
};
}  // namespace

TEST_F(profiling_time_test, kernel_upper_bound_preserves_predecessor_order)
{
    const auto now      = rocprofiler::common::timestamp_ns();
    const auto boundary = now - 10'000;
    query_time          = {now - 20'000, boundary};
    const auto previous = rocprofiler::kernel_dispatch::get_dispatch_time(
        hsa_agent_t{}, hsa_signal_t{}, 0, now - 30'000);

    query_time           = {boundary, 0};
    future_end           = true;
    const auto following = rocprofiler::kernel_dispatch::get_dispatch_time(
        hsa_agent_t{}, hsa_signal_t{}, 0, now - 30'000);

    EXPECT_EQ(previous.status, HSA_STATUS_SUCCESS);
    EXPECT_EQ(following.status, HSA_STATUS_SUCCESS);
    EXPECT_EQ(previous.end, boundary);
    EXPECT_EQ(following.start, boundary);
    EXPECT_LE(previous.end, following.start);
    EXPECT_LT(following.start, following.end);
    EXPECT_LT(following.end, query_time.end);
    EXPECT_LE(following.end, rocprofiler::common::timestamp_ns());
}

TEST_F(profiling_time_test, kernel_lower_bound_preserves_successor_order)
{
    const auto enqueue  = rocprofiler::common::timestamp_ns() - 1'000'000;
    const auto boundary = enqueue + 5'000;
    query_time          = {enqueue - 5'000, boundary};
    const auto previous =
        rocprofiler::kernel_dispatch::get_dispatch_time(hsa_agent_t{}, hsa_signal_t{}, 0, enqueue);

    query_time = {boundary, boundary + 10'000};
    const auto following =
        rocprofiler::kernel_dispatch::get_dispatch_time(hsa_agent_t{}, hsa_signal_t{}, 0, enqueue);

    EXPECT_EQ(previous.status, HSA_STATUS_SUCCESS);
    EXPECT_EQ(following.status, HSA_STATUS_SUCCESS);
    EXPECT_EQ(previous.start, enqueue);
    EXPECT_EQ(previous.end, boundary);
    EXPECT_EQ(following.start, boundary);
    EXPECT_LE(previous.end, following.start);
    EXPECT_LT(previous.start, previous.end);
}

TEST_F(profiling_time_test, clipping_preserves_valid_nonempty_intervals)
{
    using namespace rocprofiler::tracing;
    struct test_case
    {
        uint64_t    start;
        uint64_t    end;
        uint64_t    lower;
        uint64_t    upper;
        uint64_t    expected_start;
        uint64_t    expected_end;
        const char* label;
    };
    constexpr auto  maximum = std::numeric_limits<uint64_t>::max();
    const test_case cases[] = {
        {100, 200, 0, 500, 100, 200, "inside"},
        {100, 200, 0, 190, 100, 190, "upper bound"},
        {100, 200, 110, 300, 110, 200, "lower bound"},
        {100, 200, 110, 190, 110, 190, "both bounds"},
        {100, 101, 0, 500, 100, 101, "one nanosecond"},
        {100, 101, 0, 100, 99, 100, "touching upper bound"},
        {100, 101, 101, 500, 101, 102, "touching lower bound"},
        {300, 400, 0, 200, 100, 200, "wholly after window"},
        {100, 200, 300, 500, 300, 400, "wholly before window"},
        {10'000'000'100, 10'000'000'200, 0, 1000, 900, 1000, "large epoch offset"},
        {200, 100, 0, 500, 100, 200, "inverted interval"},
        {1, maximum, 2, maximum - 1, 2, maximum - 1, "unsigned boundaries"},
    };
    for(const auto& entry : cases)
    {
        SCOPED_TRACE(entry.label);
        const auto result = adjust_profiling_time("dispatch",
                                                  "mock",
                                                  {HSA_STATUS_SUCCESS, entry.start, entry.end},
                                                  {HSA_STATUS_SUCCESS, entry.lower, entry.upper},
                                                  profiling_time_bounds_policy::clip);
        EXPECT_EQ(result.status, HSA_STATUS_SUCCESS);
        EXPECT_EQ(result.start, entry.expected_start);
        EXPECT_EQ(result.end, entry.expected_end);
        EXPECT_LT(result.start, result.end);
    }
}

TEST_F(profiling_time_test, clipping_only_shrinks_overlapping_intervals)
{
    using namespace rocprofiler::tracing;
    for(uint64_t start = 1; start < 8; ++start)
    {
        for(uint64_t end = start + 1; end < 10; ++end)
        {
            for(uint64_t lower = 0; lower < 10; ++lower)
            {
                for(uint64_t upper = lower + 1; upper < 12; ++upper)
                {
                    if(std::max(start, lower) >= std::min(end, upper)) continue;
                    const auto result = adjust_profiling_time("dispatch",
                                                              "mock",
                                                              {HSA_STATUS_SUCCESS, start, end},
                                                              {HSA_STATUS_SUCCESS, lower, upper},
                                                              profiling_time_bounds_policy::clip);
                    EXPECT_EQ(result.start, std::max(start, lower));
                    EXPECT_EQ(result.end, std::min(end, upper));
                    EXPECT_LT(result.start, result.end);
                }
            }
        }
    }
}

TEST_F(profiling_time_test, default_policy_preserves_duration)
{
    using namespace rocprofiler::tracing;
    const auto upper = adjust_profiling_time(
        "memcpy", "mock", {HSA_STATUS_SUCCESS, 100, 200}, {HSA_STATUS_SUCCESS, 0, 190});
    EXPECT_EQ(upper.start, 90);
    EXPECT_EQ(upper.end, 190);
    const auto lower = adjust_profiling_time(
        "memcpy", "mock", {HSA_STATUS_SUCCESS, 100, 200}, {HSA_STATUS_SUCCESS, 110, 300});
    EXPECT_EQ(lower.start, 110);
    EXPECT_EQ(lower.end, 210);
}

TEST_F(profiling_time_test, clipping_does_not_invent_zero_interval_duration)
{
    using namespace rocprofiler::tracing;
    const auto result = adjust_profiling_time("dispatch",
                                              "mock",
                                              {HSA_STATUS_SUCCESS, 100, 100},
                                              {HSA_STATUS_SUCCESS, 0, 200},
                                              profiling_time_bounds_policy::clip);
    EXPECT_EQ(result.start, 100);
    EXPECT_EQ(result.end, 100);
}

TEST_F(profiling_time_test, strict_mode_still_rejects_out_of_bounds)
{
    const auto original_style               = ::testing::FLAGS_gtest_death_test_style;
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_DEATH(
        {
            setenv("ROCPROFILER_CI_STRICT_TIMESTAMPS", "1", 1);
            rocprofiler::tracing::adjust_profiling_time(
                "dispatch",
                "mock",
                {HSA_STATUS_SUCCESS, 100, 200},
                {HSA_STATUS_SUCCESS, 0, 190},
                rocprofiler::tracing::profiling_time_bounds_policy::clip);
        },
        "end time is greater than");
    ::testing::FLAGS_gtest_death_test_style = original_style;
}

TEST_F(profiling_time_test, normalization_precedes_clipping)
{
    const auto original_style               = ::testing::FLAGS_gtest_death_test_style;
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_EXIT(
        {
            setenv("ROCPROFILER_CI_FREQ_SCALE_TIMESTAMPS", "1", 1);
            rocprofiler::hsa::get_core_table()->hsa_system_get_info_fn = [](hsa_system_info_t,
                                                                            void* value) {
                *static_cast<uint64_t*>(value) = 100'000'000;
                return HSA_STATUS_SUCCESS;
            };
            const auto result = rocprofiler::tracing::adjust_profiling_time(
                "dispatch",
                "mock",
                {HSA_STATUS_SUCCESS, 10, 20},
                {HSA_STATUS_SUCCESS, 90, 190},
                rocprofiler::tracing::profiling_time_bounds_policy::clip);
            std::_Exit(result.start == 100 && result.end == 190 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0),
        "");
    ::testing::FLAGS_gtest_death_test_style = original_style;
}

TEST(rocprofiler_lib, timestamp)
{
    auto beg = rocprofiler::common::timestamp_ns();
    auto mid = rocprofiler_timestamp_t{};
    auto ret = rocprofiler_get_timestamp(&mid);
    auto end = rocprofiler::common::timestamp_ns();

    EXPECT_EQ(ret, ROCPROFILER_STATUS_SUCCESS);
    EXPECT_GT(beg, 0);
    EXPECT_GT(mid, beg);
    EXPECT_GT(end, mid);
}
