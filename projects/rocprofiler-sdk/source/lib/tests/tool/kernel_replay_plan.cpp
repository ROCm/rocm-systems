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

#include "kernel_replay_plan.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace
{
using rocprofiler::tool::replay_pass_layout;
using rocprofiler::tool::thread_trace_pass_index;

// What every pass of a layout collects, as a string: the counter group's digit, or 'T' for the
// thread trace pass.
std::string
pass_sequence(const replay_pass_layout& layout)
{
    std::string out;
    for(uint64_t pass = 0; pass < layout.total_passes(); ++pass)
    {
        auto group = layout.counter_group(pass);
        out.push_back(group ? static_cast<char>('0' + *group) : 'T');
    }
    return out;
}
}  // namespace

TEST(kernel_replay_plan, counter_groups_only)
{
    const auto layout = replay_pass_layout{.counter_passes = 3, .thread_trace_pass = std::nullopt};
    EXPECT_EQ(layout.total_passes(), 3u);
    EXPECT_EQ(pass_sequence(layout), "012");
}

TEST(kernel_replay_plan, thread_trace_first)
{
    const auto layout = replay_pass_layout{.counter_passes = 2, .thread_trace_pass = 0};
    EXPECT_EQ(layout.total_passes(), 3u);
    EXPECT_EQ(pass_sequence(layout), "T01");
}

TEST(kernel_replay_plan, thread_trace_between_counter_groups)
{
    const auto layout = replay_pass_layout{.counter_passes = 3, .thread_trace_pass = 1};
    EXPECT_EQ(pass_sequence(layout), "0T12");
}

TEST(kernel_replay_plan, thread_trace_last)
{
    const auto layout = replay_pass_layout{.counter_passes = 2, .thread_trace_pass = 2};
    EXPECT_EQ(pass_sequence(layout), "01T");
}

// An agent with no counter group to collect still traces; its single pass is the thread trace.
TEST(kernel_replay_plan, thread_trace_without_counter_groups)
{
    const auto layout = replay_pass_layout{.counter_passes = 0, .thread_trace_pass = 0};
    EXPECT_EQ(layout.total_passes(), 1u);
    EXPECT_EQ(pass_sequence(layout), "T");
}

// With every configured group collectable, the thread trace pass index is simply how many groups
// were configured ahead of it.
TEST(kernel_replay_plan, thread_trace_index_follows_the_configured_groups)
{
    const auto all_groups = std::vector<uint64_t>{0, 1, 2};
    EXPECT_EQ(thread_trace_pass_index(all_groups, 0), 0u);
    EXPECT_EQ(thread_trace_pass_index(all_groups, 1), 1u);
    EXPECT_EQ(thread_trace_pass_index(all_groups, 3), 3u);
}

// An agent that cannot collect configured group 1 has profiles built from groups 0 and 2 only. A
// thread trace configured after the first two groups must still follow group 0 and precede group
// 2 on that agent, rather than land after both of the agent's groups.
TEST(kernel_replay_plan, thread_trace_index_skips_groups_the_agent_lacks)
{
    const auto sources = std::vector<uint64_t>{0, 2};
    EXPECT_EQ(thread_trace_pass_index(sources, 0), 0u);
    EXPECT_EQ(thread_trace_pass_index(sources, 1), 1u);
    EXPECT_EQ(thread_trace_pass_index(sources, 2), 1u);
    EXPECT_EQ(thread_trace_pass_index(sources, 3), 2u);

    const auto layout = replay_pass_layout{
        .counter_passes = sources.size(), .thread_trace_pass = thread_trace_pass_index(sources, 2)};
    EXPECT_EQ(pass_sequence(layout), "0T1");
}
