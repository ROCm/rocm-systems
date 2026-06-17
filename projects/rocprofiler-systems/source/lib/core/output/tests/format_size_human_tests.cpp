// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "common/size_format.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace
{
struct size_format_case
{
    std::optional<std::uintmax_t> input;
    std::string                   expected;
};

class format_size_human_test : public ::testing::TestWithParam<size_format_case>
{};
}  // namespace

TEST_P(format_size_human_test, formats_at_threshold_boundaries)
{
    EXPECT_EQ(rocprofsys::common::format_size_human(GetParam().input),
              GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    boundaries, format_size_human_test,
    ::testing::Values(size_format_case{ std::nullopt, "?" },
                      size_format_case{ 0u, "0 B" }, size_format_case{ 999u, "999 B" },
                      size_format_case{ 1000u, "1.00 KB" },
                      size_format_case{ 1000ULL * 1000 - 1, "1000.00 KB" },
                      size_format_case{ 1000ULL * 1000, "1.00 MB" },
                      size_format_case{ 1000ULL * 1000 * 1000 - 1, "1000.00 MB" },
                      size_format_case{ 1000ULL * 1000 * 1000, "1.00 GB" },
                      size_format_case{ 5ULL * 1000 * 1000 * 1000, "5.00 GB" }));
