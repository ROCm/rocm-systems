// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys/library/rocprofiler-sdk/service_compatibility.hpp"

#include <gtest/gtest.h>

namespace
{
namespace service_compatibility = rocprofsys::rocprofiler_sdk::service_compatibility;

constexpr auto k_spm_not_requested = false;
constexpr auto k_spm_requested     = true;

TEST(service_compatibility_test, accepts_when_neither_service_is_requested)
{
    EXPECT_FALSE(service_compatibility::has_spm_gpu_perf_counter_conflict(
        k_spm_not_requested, {}));
}

TEST(service_compatibility_test, accepts_when_only_gpu_perf_counters_are_requested)
{
    EXPECT_FALSE(service_compatibility::has_spm_gpu_perf_counter_conflict(
        k_spm_not_requested, "SQ_WAVES:device=0"));
}

TEST(service_compatibility_test, accepts_when_only_spm_is_requested)
{
    EXPECT_FALSE(
        service_compatibility::has_spm_gpu_perf_counter_conflict(k_spm_requested, {}));
}

TEST(service_compatibility_test, accepts_whitespace_only_gpu_perf_counter_setting)
{
    EXPECT_FALSE(service_compatibility::has_spm_gpu_perf_counter_conflict(k_spm_requested,
                                                                          " \t\n"));
}

TEST(service_compatibility_test, rejects_when_both_services_are_requested)
{
    EXPECT_TRUE(service_compatibility::has_spm_gpu_perf_counter_conflict(
        k_spm_requested, "SQ_WAVES:device=0"));
}
}  // namespace
