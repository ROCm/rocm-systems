// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/gpu/types.hpp"

#include <gtest/gtest.h>

#include <set>
#include <string>

namespace rocprofsys::pmc::collectors::gpu::testing
{

// The runtime-visibility filter (enumerate_devices) keeps a GPU only when
// is_runtime_visible() returns true for the device's PCIe BDF. These tests pin the
// three behaviors that decide whether a physical GPU enumerated by AMD SMI is sampled:
// visible device kept, non-visible device excluded, and unknown/empty BDF excluded
// (fail-closed).

TEST(gpu_runtime_visibility, visible_device_is_included)
{
    const std::set<std::string> visible{ "0000:05:00.0", "0000:26:00.0" };
    EXPECT_TRUE(is_runtime_visible("0000:26:00.0", visible));
}

TEST(gpu_runtime_visibility, non_visible_device_is_excluded)
{
    const std::set<std::string> visible{ "0000:26:00.0" };
    // A different physical GPU (masked by ROCR/HIP_VISIBLE_DEVICES) must not match.
    EXPECT_FALSE(is_runtime_visible("0000:05:00.0", visible));
}

TEST(gpu_runtime_visibility, empty_bdf_is_excluded_fail_closed)
{
    const std::set<std::string> visible{ "0000:05:00.0" };
    // Device whose BDF could not be determined: never sample it.
    EXPECT_FALSE(is_runtime_visible("", visible));
}

TEST(gpu_runtime_visibility, empty_visible_set_excludes_everything)
{
    // No runtime-visible GPUs (e.g. all masked): nothing is sampled.
    const std::set<std::string> visible{};
    EXPECT_FALSE(is_runtime_visible("0000:05:00.0", visible));
    EXPECT_FALSE(is_runtime_visible("", visible));
}

TEST(gpu_runtime_visibility, match_is_exact_not_substring)
{
    const std::set<std::string> visible{ "0000:05:00.0" };
    // Correlation must be an exact BDF match, not a prefix/substring.
    EXPECT_FALSE(is_runtime_visible("0000:05:00.1", visible));
    EXPECT_FALSE(is_runtime_visible("0000:05:00", visible));
}

}  // namespace rocprofsys::pmc::collectors::gpu::testing
