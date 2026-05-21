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
                      size_format_case{ 0u, "0 B" },
                      size_format_case{ 1023u, "1023 B" },
                      size_format_case{ 1024u, "1.00 KiB" },
                      size_format_case{ 1024ULL * 1024 - 1, "1024.00 KiB" },
                      size_format_case{ 1024ULL * 1024, "1.00 MiB" },
                      size_format_case{ 1024ULL * 1024 * 1024 - 1, "1024.00 MiB" },
                      size_format_case{ 1024ULL * 1024 * 1024, "1.00 GiB" },
                      size_format_case{ 5ULL * 1024 * 1024 * 1024, "5.00 GiB" }));
