// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "kernel_iteration_filter.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace
{
using rocprofiler::tool::is_targeted_kernel_iteration;
using rocprofiler::tool::kernel_iteration_count_map_t;
using rocprofiler::tool::kernel_target_map_t;

constexpr bool PMC = false;  // advanced_thread_trace == false
constexpr bool ATT = true;   // advanced_thread_trace == true

// call is_targeted_kernel_iteration n times for one kernel and return the
// per-launch decisions as a string of '1' (profiled) / '0' (skipped)
std::string
profile_sequence(uint64_t                   kernel_id,
                 const kernel_target_map_t& targets,
                 bool                       advanced_thread_trace,
                 int                        n)
{
    kernel_iteration_count_map_t counter;
    std::string                  out;
    out.reserve(n);
    for(int i = 0; i < n; ++i)
        out.push_back(
            is_targeted_kernel_iteration(kernel_id, targets, counter, advanced_thread_trace) ? '1'
                                                                                             : '0');
    return out;
}
}  // namespace

TEST(kernel_iteration_filter, untargeted_kernel_never_profiled_or_counted)
{
    kernel_target_map_t          targets = {{1, {}}};
    kernel_iteration_count_map_t counter;

    // kernel 2 is not in the target map -> false, and must not be counted
    EXPECT_FALSE(is_targeted_kernel_iteration(2, targets, counter, PMC));
    EXPECT_FALSE(is_targeted_kernel_iteration(2, targets, counter, PMC));
    EXPECT_EQ(counter.find(2), counter.end());
}

TEST(kernel_iteration_filter, empty_range_pmc_profiles_all_iterations)
{
    kernel_target_map_t targets = {{7, {}}};  // empty range == all iterations
    EXPECT_EQ(profile_sequence(7, targets, PMC, 5), "11111");
}

TEST(kernel_iteration_filter, empty_range_att_profiles_only_first_iteration)
{
    kernel_target_map_t targets = {{7, {}}};
    EXPECT_EQ(profile_sequence(7, targets, ATT, 4), "1000");
}

TEST(kernel_iteration_filter, first_dispatch_is_iteration_one)
{
    // range {1} must match the very first dispatch and nothing else
    kernel_target_map_t targets = {{9, {1}}};
    EXPECT_EQ(profile_sequence(9, targets, PMC, 3), "100");
}

TEST(kernel_iteration_filter, range_selects_exact_iterations_both_directions)
{
    // range {2,4} over 5 launches -> only iterations 2 and 4 are profiled
    kernel_target_map_t targets = {{3, {2, 4}}};
    EXPECT_EQ(profile_sequence(3, targets, PMC, 5), "01010");
}

TEST(kernel_iteration_filter, out_of_bounds_range_profiles_nothing)
{
    // range references iterations beyond the number of launches -> no profiling
    kernel_target_map_t targets = {{3, {99}}};
    EXPECT_EQ(profile_sequence(3, targets, PMC, 4), "0000");
}

TEST(kernel_iteration_filter, distinct_kernels_have_independent_counters)
{
    // interleave two kernels; each maintains its own 1-based counter. kernel 10
    // wants iteration 1, kernel 20 wants iteration 2.
    kernel_target_map_t          targets = {{10, {1}}, {20, {2}}};
    kernel_iteration_count_map_t counter;

    EXPECT_TRUE(is_targeted_kernel_iteration(10, targets, counter, PMC));   // 10 -> iter 1 (hit)
    EXPECT_FALSE(is_targeted_kernel_iteration(20, targets, counter, PMC));  // 20 -> iter 1 (miss)
    EXPECT_FALSE(is_targeted_kernel_iteration(10, targets, counter, PMC));  // 10 -> iter 2 (miss)
    EXPECT_TRUE(is_targeted_kernel_iteration(20, targets, counter, PMC));   // 20 -> iter 2 (hit)

    EXPECT_EQ(counter[10], 2u);
    EXPECT_EQ(counter[20], 2u);
}

TEST(kernel_iteration_filter, range_with_gaps_and_multiple_kernels)
{
    // both kernels share range {1,3}; verify per-kernel independence over 4 launches
    kernel_target_map_t targets = {{100, {1, 3}}, {200, {1, 3}}};
    EXPECT_EQ(profile_sequence(100, targets, PMC, 4), "1010");
    EXPECT_EQ(profile_sequence(200, targets, PMC, 4), "1010");
}
