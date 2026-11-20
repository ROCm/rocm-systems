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

#include "kernel_replay_att.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace
{
namespace kr = rocprofiler::tool::kernel_replay_att;

constexpr bool ATT_ON  = true;
constexpr bool ATT_OFF = false;

constexpr size_t kMaxGroups = 8;

// Counter hook and thread trace hook as the tool wires them for one dispatch pass: the counter
// callback stores tid and pass in the dispatch's shared user_data slot, and the thread-trace
// callback stores the dispatch id there unless it yields to the counters.
struct user_data_slot
{
    uint64_t value = 0;
};

constexpr uint64_t
pack(uint32_t tid, uint32_t pass)
{
    return (static_cast<uint64_t>(pass) << 32) | tid;
}

void
run_counter_hook(user_data_slot& slot, uint32_t tid, uint32_t pass)
{
    slot.value = pack(tid, pass);
}

bool
run_thread_trace_hook(user_data_slot& slot,
                      uint64_t        dispatch_id,
                      bool            kernel_replay,
                      bool            in_replay_pass,
                      size_t          groups)
{
    if(kr::thread_trace_yields_to_counters(kernel_replay, in_replay_pass, groups)) return false;
    slot.value = dispatch_id;
    return true;
}
}  // namespace

TEST(kernel_replay_att, pass_count_is_groups_plus_one_with_thread_trace)
{
    for(size_t groups = 1; groups <= kMaxGroups; ++groups)
        EXPECT_EQ(kr::pass_count(groups, true, ATT_ON), groups + 1) << "groups=" << groups;
}

TEST(kernel_replay_att, pass_count_is_groups_without_thread_trace)
{
    // Counters alone are unchanged: N groups are N passes, and one group is not replayed at all.
    for(size_t groups = 1; groups <= kMaxGroups; ++groups)
        EXPECT_EQ(kr::pass_count(groups, true, ATT_OFF), groups) << "groups=" << groups;
    EXPECT_EQ(kr::pass_count(1, true, ATT_OFF), 1u);
}

TEST(kernel_replay_att, single_counter_group_with_thread_trace_is_two_passes)
{
    // Without the extra pass a lone group is one pass, so counters and thread trace would share one
    // execution.
    EXPECT_EQ(kr::pass_count(1, true, ATT_ON), 2u);
    EXPECT_TRUE(kr::services_for_pass(0, 1, ATT_ON).counters);
    EXPECT_FALSE(kr::services_for_pass(0, 1, ATT_ON).thread_trace);
    EXPECT_FALSE(kr::services_for_pass(1, 1, ATT_ON).counters);
    EXPECT_TRUE(kr::services_for_pass(1, 1, ATT_ON).thread_trace);
}

TEST(kernel_replay_att, dispatch_that_is_not_replayed_is_one_pass)
{
    // Counter collection paused or the kernel filters did not select the dispatch.
    for(bool att : {ATT_ON, ATT_OFF})
    {
        for(size_t groups = 0; groups <= kMaxGroups; ++groups)
            EXPECT_EQ(kr::pass_count(groups, false, att), 1u)
                << "groups=" << groups << " att=" << att;
    }

    // An agent with no counter groups has nothing to replay, thread trace or not.
    EXPECT_EQ(kr::pass_count(0, true, ATT_ON), 1u);
    EXPECT_EQ(kr::pass_count(0, true, ATT_OFF), 1u);
}

TEST(kernel_replay_att, exactly_one_service_is_enabled_on_every_pass)
{
    for(size_t groups = 1; groups <= kMaxGroups; ++groups)
    {
        const auto passes = kr::pass_count(groups, true, ATT_ON);
        for(uint64_t pass = 0; pass < passes; ++pass)
        {
            const auto services = kr::services_for_pass(pass, groups, ATT_ON);
            EXPECT_NE(services.counters, services.thread_trace)
                << "groups=" << groups << " pass=" << pass
                << ": counters and thread trace must never share a pass, and no pass may "
                   "collect neither";
        }
    }
}

TEST(kernel_replay_att, thread_trace_is_captured_once_per_dispatch_on_the_last_pass)
{
    for(size_t groups = 1; groups <= kMaxGroups; ++groups)
    {
        const auto passes = kr::pass_count(groups, true, ATT_ON);

        size_t   counter_passes      = 0;
        size_t   thread_trace_passes = 0;
        uint64_t thread_trace_pass   = 0;
        for(uint64_t pass = 0; pass < passes; ++pass)
        {
            const auto services = kr::services_for_pass(pass, groups, ATT_ON);
            if(services.counters) ++counter_passes;
            if(services.thread_trace)
            {
                ++thread_trace_passes;
                thread_trace_pass = pass;
            }
        }

        EXPECT_EQ(counter_passes, groups) << "every counter group is collected exactly once";
        EXPECT_EQ(thread_trace_passes, 1u) << "groups=" << groups;
        EXPECT_EQ(thread_trace_pass, groups) << "thread trace follows the counter groups";
    }
}

TEST(kernel_replay_att, thread_trace_pass_has_no_counter_group)
{
    for(size_t groups = 1; groups <= kMaxGroups; ++groups)
    {
        // Passes 0..groups-1 each select a group, so each produces a counter row.
        for(uint64_t pass = 0; pass < groups; ++pass)
        {
            EXPECT_FALSE(kr::pass_has_no_counter_group(pass, groups))
                << "groups=" << groups << " pass=" << pass;
            EXPECT_FALSE(kr::is_thread_trace_pass(pass, groups, ATT_ON));
        }

        // The pass after them is the thread-trace pass and must not wrap onto group 0.
        EXPECT_TRUE(kr::pass_has_no_counter_group(groups, groups)) << "groups=" << groups;
        EXPECT_TRUE(kr::is_thread_trace_pass(groups, groups, ATT_ON));
    }
}

TEST(kernel_replay_att, counter_only_replay_never_reaches_a_pass_without_a_group)
{
    // Without thread trace the pass count equals the group count, so the replay loop never runs a
    // pass that has no counter group, and no pass is ever a thread-trace pass.
    for(size_t groups = 2; groups <= kMaxGroups; ++groups)
    {
        const auto passes = kr::pass_count(groups, true, ATT_OFF);
        for(uint64_t pass = 0; pass < passes; ++pass)
        {
            EXPECT_FALSE(kr::pass_has_no_counter_group(pass, groups));
            EXPECT_FALSE(kr::is_thread_trace_pass(pass, groups, ATT_OFF));
            EXPECT_TRUE(kr::services_for_pass(pass, groups, ATT_OFF).counters);
            EXPECT_FALSE(kr::services_for_pass(pass, groups, ATT_OFF).thread_trace);
        }
    }
}

TEST(kernel_replay_att, agent_without_counter_groups_has_no_thread_trace_pass)
{
    EXPECT_FALSE(kr::is_thread_trace_pass(0, 0, ATT_ON));
    EXPECT_FALSE(kr::pass_has_no_counter_group(0, 0));
}

TEST(kernel_replay_att, thread_trace_yields_when_a_dispatch_is_outside_a_replay_pass)
{
    // Replay declined, a HIP graph, or a multi-packet submission: no pass callback, both services
    // are enabled, and the agent has counter groups whose record must stay attributable.
    EXPECT_TRUE(kr::thread_trace_yields_to_counters(true, false, 1));
    EXPECT_TRUE(kr::thread_trace_yields_to_counters(true, false, 4));
}

TEST(kernel_replay_att, thread_trace_does_not_yield_inside_a_replay_pass)
{
    // Only one service is enabled in a pass, so the slot is not shared.
    for(size_t groups = 1; groups <= kMaxGroups; ++groups)
        EXPECT_FALSE(kr::thread_trace_yields_to_counters(true, true, groups));
}

TEST(kernel_replay_att, thread_trace_is_untouched_without_kernel_replay)
{
    // Plain --att keeps writing its dispatch id and tracing, whatever the counter configuration.
    for(size_t groups = 0; groups <= kMaxGroups; ++groups)
    {
        EXPECT_FALSE(kr::thread_trace_yields_to_counters(false, false, groups));
        EXPECT_FALSE(kr::thread_trace_yields_to_counters(false, true, groups));
    }
}

TEST(kernel_replay_att, thread_trace_is_untouched_on_an_agent_without_counter_groups)
{
    // No counter record exists for such an agent, so there is nothing to protect.
    EXPECT_FALSE(kr::thread_trace_yields_to_counters(true, false, 0));
}

TEST(kernel_replay_att, fallback_dispatch_keeps_counter_thread_and_pass)
{
    constexpr uint32_t tid         = 4242;
    constexpr uint64_t dispatch_id = 17;
    constexpr size_t   groups      = 3;

    // Not replayed: the counter hook runs first and packs pass 0, then thread trace runs.
    user_data_slot slot;
    run_counter_hook(slot, tid, 0);
    const bool traced = run_thread_trace_hook(slot, dispatch_id, true, false, groups);

    EXPECT_FALSE(traced) << "a dispatch that shares user_data with counters is not traced";
    EXPECT_EQ(slot.value, pack(tid, 0))
        << "the counter record still decodes the enqueuing thread and pass 0";
}

TEST(kernel_replay_att, replayed_dispatch_attributes_each_pass_to_its_own_service)
{
    constexpr uint32_t tid         = 4242;
    constexpr uint64_t dispatch_id = 17;
    constexpr size_t   groups      = 3;

    const auto passes = kr::pass_count(groups, true, ATT_ON);
    ASSERT_EQ(passes, groups + 1);

    for(uint64_t pass = 0; pass < passes; ++pass)
    {
        // Each pass is its own submission with its own slot; only the enabled service's hook runs.
        user_data_slot slot;
        const auto     services = kr::services_for_pass(pass, groups, ATT_ON);
        if(services.counters) run_counter_hook(slot, tid, static_cast<uint32_t>(pass));
        if(services.thread_trace) run_thread_trace_hook(slot, dispatch_id, true, true, groups);

        if(pass < groups)
            EXPECT_EQ(slot.value, pack(tid, static_cast<uint32_t>(pass))) << "pass=" << pass;
        else
            EXPECT_EQ(slot.value, dispatch_id) << "the trace file names the dispatch";
    }
}
