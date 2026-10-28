// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#include "att_no_intercept_impl.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace
{
using rocprofiler::tool::att_no_intercept::advance_trace_range;
using rocprofiler::tool::att_no_intercept::dispatch_offset_t;
using rocprofiler::tool::att_no_intercept::trace_range_t;

struct range_case
{
    const char*                                name;
    uint64_t                                   consecutive;
    std::vector<dispatch_offset_t>             dispatches;
    std::vector<std::pair<uint64_t, uint64_t>> completed;
    trace_range_t                              tail;
};

class att_trace_range : public ::testing::TestWithParam<range_case>
{};

TEST_P(att_trace_range, dispatch_sequence)
{
    const auto&                                test  = GetParam();
    auto                                       trace = trace_range_t{};
    std::vector<std::pair<uint64_t, uint64_t>> completed;
    for(const auto& dispatch : test.dispatches)
    {
        auto range = advance_trace_range(trace, dispatch, test.consecutive);
        if(!range.active) continue;
        EXPECT_EQ(range.remaining_dispatches, 0);
        EXPECT_GT(range.offset_end, range.offset_begin);
        completed.emplace_back(range.offset_begin, range.offset_end);
    }

    EXPECT_EQ(completed, test.completed);
    EXPECT_EQ(trace.active, test.tail.active);
    EXPECT_EQ(trace.me_id, test.tail.me_id);
    EXPECT_EQ(trace.pipe_id, test.tail.pipe_id);
    EXPECT_EQ(trace.offset_begin, test.tail.offset_begin);
    EXPECT_EQ(trace.offset_end, test.tail.offset_end);
    EXPECT_EQ(trace.remaining_dispatches, test.tail.remaining_dispatches);
}

INSTANTIATE_TEST_SUITE_P(
    dispatches,
    att_trace_range,
    ::testing::Values(
        range_case{"empty", 2, {}, {}, {}},
        range_case{"untargeted", 2, {{0, 0, 10, false}, {1, 1, 20, false}}, {}, {}},
        range_case{"default_count",
                   0,
                   {{0, 0, 10, true}, {0, 0, 20, false}, {0, 0, 30, true}},
                   {{10, 20}},
                   {true, 0, 0, 30, 0, 0}},
        range_case{"selected_boundaries",
                   1,
                   {{0, 0, 10, true}, {0, 0, 20, true}, {0, 1, 30, false}, {0, 0, 40, false}},
                   {{10, 20}, {20, 40}},
                   {}},
        range_case{"dispatch_budget",
                   2,
                   {{0, 0, 10, true}, {0, 1, 20, false}, {0, 0, 30, false}, {0, 1, 40, false}},
                   {{10, 40}},
                   {}},
        range_case{"selected_reset",
                   2,
                   {{0, 0, 10, true}, {0, 0, 20, true}, {0, 0, 30, false}, {0, 0, 40, false}},
                   {{10, 40}},
                   {}},
        range_case{"active_tail",
                   2,
                   {{0, 0, 10, true}, {0, 0, 20, true}, {0, 0, 30, false}},
                   {},
                   {true, 0, 0, 10, 0, 0}},
        range_case{"me_pipe_transitions",
                   3,
                   {{0, 0, 10, true},
                    {0, 1, 20, false},
                    {1, 0, 30, true},
                    {0, 1, 40, false},
                    {1, 1, 50, false},
                    {1, 0, 60, false},
                    {1, 1, 70, false}},
                   {{10, 70}},
                   {}},
        range_case{"selected_other_pipe",
                   1,
                   {{0, 0, 10, true}, {0, 1, 20, true}, {0, 1, 30, false}},
                   {{10, 30}},
                   {}},
        range_case{"completed_then_selected",
                   2,
                   {{0, 0, 10, true},
                    {0, 1, 20, false},
                    {0, 1, 30, true},
                    {0, 0, 40, false},
                    {0, 0, 50, false}},
                   {{10, 30}, {30, 50}},
                   {}}),
    [](const ::testing::TestParamInfo<range_case>& info) { return info.param.name; });
}  // namespace
