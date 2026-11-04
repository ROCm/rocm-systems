// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/version.hpp"

#include <gtest/gtest.h>

using namespace rocprofsys::common;

TEST(version_mismatch_test, equal_versions_match)
{
    const version current{ .major = 1, .minor = 2, .patch = 3 };
    EXPECT_FALSE(describe_version_mismatch("lib", current, current).has_value());
}

TEST(version_mismatch_test, any_component_difference_is_reported)
{
    const version built{ .major = 1, .minor = 2, .patch = 3 };

    EXPECT_TRUE(describe_version_mismatch("lib", built,
                                          version{ .major = 2, .minor = 2, .patch = 3 })
                    .has_value());
    EXPECT_TRUE(describe_version_mismatch("lib", built,
                                          version{ .major = 1, .minor = 3, .patch = 3 })
                    .has_value());
    EXPECT_TRUE(describe_version_mismatch("lib", built,
                                          version{ .major = 1, .minor = 2, .patch = 4 })
                    .has_value());
}

TEST(version_mismatch_test, message_names_library_and_both_versions)
{
    const auto message = describe_version_mismatch(
        "profiler-hub", version{ .major = 0, .minor = 2, .patch = 0 },
        version{ .major = 0, .minor = 3, .patch = 1 });

    ASSERT_TRUE(message.has_value());
    EXPECT_EQ(*message,
              "profiler-hub version mismatch: built against v0.2.0 but v0.3.1 is loaded");
}
