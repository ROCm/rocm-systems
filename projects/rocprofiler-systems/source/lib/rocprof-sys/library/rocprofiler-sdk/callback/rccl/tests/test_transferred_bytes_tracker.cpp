// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/rccl/transferred_bytes_tracker.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"

#include <cstdint>
#include <gtest/gtest.h>

namespace rocprofsys::domains::callback
{
namespace
{

using test_support::externals;
using tracker = rccl::transferred_bytes_tracker<externals>;

}  // namespace

TEST(rccl_transferred_bytes_tracker_test,
     first_call_for_a_device_reports_first_registration)
{
    constexpr std::uint32_t k_device_id = 100;
    constexpr std::size_t   k_bytes     = 64;

    const auto result = tracker::add_bytes(k_device_id, k_bytes);

    EXPECT_TRUE(result.is_first_registration);
    EXPECT_EQ(result.cumulative_bytes, k_bytes);
}

TEST(rccl_transferred_bytes_tracker_test,
     subsequent_calls_accumulate_without_re_registering)
{
    constexpr std::uint32_t k_device_id = 101;

    const auto first  = tracker::add_bytes(k_device_id, 10);
    const auto second = tracker::add_bytes(k_device_id, 20);

    EXPECT_TRUE(first.is_first_registration);
    EXPECT_FALSE(second.is_first_registration);
    EXPECT_EQ(second.cumulative_bytes, 30U);
}

TEST(rccl_transferred_bytes_tracker_test, tracks_devices_independently)
{
    constexpr std::uint32_t k_device_a       = 102;
    constexpr std::uint32_t k_device_b       = 103;
    constexpr std::size_t   k_bytes_per_call = 5;

    const auto result_a_first = tracker::add_bytes(k_device_a, k_bytes_per_call);
    const auto result_a       = tracker::add_bytes(k_device_a, k_bytes_per_call);
    const auto result_b       = tracker::add_bytes(k_device_b, 42);

    EXPECT_TRUE(result_a_first.is_first_registration);
    EXPECT_EQ(result_a.cumulative_bytes, 2 * k_bytes_per_call);
    EXPECT_TRUE(result_b.is_first_registration);
    EXPECT_EQ(result_b.cumulative_bytes, 42U);
}

}  // namespace rocprofsys::domains::callback
