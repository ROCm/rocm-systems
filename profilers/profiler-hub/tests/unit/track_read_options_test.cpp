// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "track_read_options.hpp"

#include <gtest/gtest.h>

namespace
{

using profiler_hub::parse_size;

TEST(parse_size_test, valid_number_returns_value) { EXPECT_EQ(parse_size("42", 7), 42U); }

}  // namespace
