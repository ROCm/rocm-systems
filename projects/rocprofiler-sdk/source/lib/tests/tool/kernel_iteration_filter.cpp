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
#include <optional>
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

namespace
{
using rocprofiler::tool::dispatch_decision_cache_t;
using rocprofiler::tool::replay_pass_count_for_dispatch;
using rocprofiler::tool::targeted_once_per_dispatch;

// Drives the same sequence of questions the tool sees for each launch under kernel replay: the
// CONFIG pass-count callback first, then one counter-dispatch question per executed pass. A
// dispatch that is not replayed gets exactly one counter-dispatch question, on the ordinary path.
// Returns "<passes executed><T|F>" per launch, e.g. "2T" or "1F".
struct replay_driver_t
{
    kernel_target_map_t          targets;
    kernel_iteration_count_map_t counter;
    dispatch_decision_cache_t    cache;
    size_t                       groups   = 0;
    uint64_t                     next_id  = 0;
    size_t                       consults = 0;

    bool targeted(uint64_t kernel_id, uint64_t dispatch_id)
    {
        return targeted_once_per_dispatch(cache, dispatch_id, [&] {
            ++consults;
            return is_targeted_kernel_iteration(kernel_id, targets, counter, PMC);
        });
    }

    std::string launch(uint64_t kernel_id)
    {
        const auto id     = ++next_id;
        const auto passes = replay_pass_count_for_dispatch(targeted(kernel_id, id), groups);

        bool seen = false;
        for(uint64_t pass = 0; pass < passes; ++pass)
            seen = targeted(kernel_id, id);  // counter_dispatch_callback, once per pass
        return std::to_string(passes) + (seen ? "T" : "F");
    }
};
}  // namespace

TEST(kernel_iteration_filter, dispatch_decision_cache_is_keyed_by_dispatch_id)
{
    auto cache = dispatch_decision_cache_t{};
    EXPECT_FALSE(cache.find(5).has_value());

    cache.store(5, true);
    EXPECT_EQ(cache.find(5), std::optional<bool>{true});
    EXPECT_FALSE(cache.find(6).has_value());

    // the next dispatch replaces the slot; the stale id no longer answers
    cache.store(6, false);
    EXPECT_EQ(cache.find(6), std::optional<bool>{false});
    EXPECT_FALSE(cache.find(5).has_value());
}

TEST(kernel_iteration_filter, dispatch_id_zero_is_never_cached)
{
    auto cache = dispatch_decision_cache_t{};
    auto calls = 0;
    for(int i = 0; i < 3; ++i)
        EXPECT_TRUE(targeted_once_per_dispatch(cache, 0, [&] {
            ++calls;
            return true;
        }));
    EXPECT_EQ(calls, 3);
}

TEST(kernel_iteration_filter, replay_pass_count_is_one_unless_targeted_with_groups)
{
    EXPECT_EQ(replay_pass_count_for_dispatch(true, 5), 5u);
    EXPECT_EQ(replay_pass_count_for_dispatch(true, 1), 1u);
    EXPECT_EQ(replay_pass_count_for_dispatch(true, 0), 1u);
    EXPECT_EQ(replay_pass_count_for_dispatch(false, 5), 1u);
    EXPECT_EQ(replay_pass_count_for_dispatch(false, 0), 1u);
}

TEST(kernel_iteration_filter, replay_skips_kernels_outside_the_target_set)
{
    // kernel 1 is a compute kernel the user targets; kernel 2 stands in for a collective kernel
    // that --kernel-exclude-regex / --kernel-include-regex left out.
    auto d    = replay_driver_t{};
    d.targets = {{1, {}}};
    d.groups  = 3;

    EXPECT_EQ(d.launch(1), "3T");
    EXPECT_EQ(d.launch(2), "1F");  // not replayed, no counters
    EXPECT_EQ(d.launch(1), "3T");
    EXPECT_EQ(d.launch(2), "1F");
    EXPECT_EQ(d.counter.find(2), d.counter.end());
}

TEST(kernel_iteration_filter, replay_skips_launches_outside_the_iteration_range)
{
    auto d    = replay_driver_t{};
    d.targets = {{1, {2, 3}}};
    d.groups  = 2;

    // launches 1 and 4 are outside [2, 3]: they run once; 2 and 3 are replayed once per group
    EXPECT_EQ(d.launch(1), "1F");
    EXPECT_EQ(d.launch(1), "2T");
    EXPECT_EQ(d.launch(1), "2T");
    EXPECT_EQ(d.launch(1), "1F");
}

TEST(kernel_iteration_filter, filter_is_consulted_once_per_logical_dispatch)
{
    auto d    = replay_driver_t{};
    d.targets = {{1, {}}};
    d.groups  = 5;

    // one launch, five passes: CONFIG plus five counter-dispatch questions, one consultation
    EXPECT_EQ(d.launch(1), "5T");
    EXPECT_EQ(d.consults, 1u);
    EXPECT_EQ(d.counter[1], 1u);

    // a single-group dispatch is not replayed and still advances the count by exactly one
    d.groups = 1;
    EXPECT_EQ(d.launch(1), "1T");
    EXPECT_EQ(d.counter[1], 2u);
}

TEST(kernel_iteration_filter, interleaved_kernels_keep_independent_iteration_counts)
{
    auto d    = replay_driver_t{};
    d.targets = {{1, {2}}, {2, {1}}};
    d.groups  = 2;

    // MPI-style loop: compute, collective, compute, collective. Only compute launch #2 and
    // collective launch #1 are targeted; every other launch runs once.
    EXPECT_EQ(d.launch(1), "1F");
    EXPECT_EQ(d.launch(2), "2T");
    EXPECT_EQ(d.launch(1), "2T");
    EXPECT_EQ(d.launch(2), "1F");
}
