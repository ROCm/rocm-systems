// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/cpu_snap_decoder.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>

namespace rocprofsys::trace_cache
{
namespace
{
constexpr std::uint64_t k_previous_ns = 1'000;
constexpr std::uint64_t k_current_ns  = 2'000;
constexpr std::uint64_t k_between_ns  = (k_previous_ns + k_current_ns) / 2;
}  // namespace

TEST(sampling_resumes_test, no_resume_starts_at_previous_sample)
{
    auto const start =
        slice_start(std::span<const std::uint64_t>{}, k_previous_ns, k_current_ns);

    EXPECT_EQ(start, k_previous_ns);
}

TEST(sampling_resumes_test, resume_between_samples_starts_at_resume)
{
    auto const resumes = std::array{ k_between_ns };

    auto const start = slice_start(resumes, k_previous_ns, k_current_ns);

    EXPECT_EQ(start, k_between_ns);
}

TEST(sampling_resumes_test, resume_before_previous_sample_starts_at_previous_sample)
{
    auto const resumes = std::array{ k_previous_ns - 1 };

    auto const start = slice_start(resumes, k_previous_ns, k_current_ns);

    EXPECT_EQ(start, k_previous_ns);
}

TEST(sampling_resumes_test, resume_after_sample_starts_at_previous_sample)
{
    auto const resumes = std::array{ k_current_ns + 1 };

    auto const start = slice_start(resumes, k_previous_ns, k_current_ns);

    EXPECT_EQ(start, k_previous_ns);
}

TEST(sampling_resumes_test, resume_at_previous_sample_starts_at_previous_sample)
{
    auto const resumes = std::array{ k_previous_ns };

    auto const start = slice_start(resumes, k_previous_ns, k_current_ns);

    EXPECT_EQ(start, k_previous_ns);
}

TEST(sampling_resumes_test, resume_at_sample_starts_at_sample)
{
    auto const resumes = std::array{ k_current_ns };

    auto const start = slice_start(resumes, k_previous_ns, k_current_ns);

    EXPECT_EQ(start, k_current_ns);
}

TEST(sampling_resumes_test, two_resumes_between_samples_start_at_later_resume)
{
    auto const resumes = std::array{ k_previous_ns + 1, k_current_ns - 1 };

    auto const start = slice_start(resumes, k_previous_ns, k_current_ns);

    EXPECT_EQ(start, k_current_ns - 1);
}
}  // namespace rocprofsys::trace_cache
