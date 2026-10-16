// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/clocks/read.hpp"

#include <chrono>
#include <gtest/gtest.h>
#include <time.h>

namespace
{
namespace detail = rocprofsys::clocks::detail;

// The kernel rejects positive clock IDs past its last static clock with EINVAL.
constexpr clockid_t k_invalid_clock_id = 1'000;
}  // namespace

// The role clocks cannot reach this path: their clocks always exist.
TEST(clocks_read_test, returns_zero_when_the_clock_fails)
{
    EXPECT_EQ(detail::read(k_invalid_clock_id), std::chrono::nanoseconds::zero());
}
