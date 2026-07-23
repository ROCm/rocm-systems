// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Unit tests for the kernel-replay localized context control mechanism. This is pure host logic
// (thread-local override state) with no GPU or rocprofiler runtime dependency, so the tests run
// unconditionally.

#include "lib/rocprofiler-sdk/kernel_replay/local_context.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>

#include <vector>

namespace lc = rocprofiler::kernel_replay;

// Outside any replay loop the query yields nothing and the toggles are illegal (not attached).
TEST(kernel_replay_local_context, inactive_outside_loop)
{
    EXPECT_FALSE(lc::local_context_override({1}).has_value());
    EXPECT_EQ(lc::replay_local_start_context({1}), ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR);
    EXPECT_EQ(lc::replay_local_stop_context({1}), ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR);
    EXPECT_FALSE(lc::local_context_override({1}).has_value());
}

// Inside a loop but before the PASS-enter arm window, toggles still fail and nothing is recorded.
TEST(kernel_replay_local_context, loop_without_arm_rejects_toggles)
{
    lc::scoped_local_context_control loop{};

    EXPECT_EQ(lc::replay_local_start_context({7}), ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR);
    EXPECT_FALSE(lc::local_context_override({7}).has_value());
}

// Armed within a loop: start forces active, stop forces inactive, and the recorded value persists
// past the arm window (sticky) while further toggles outside the window fail.
TEST(kernel_replay_local_context, armed_toggles_record_and_stick)
{
    lc::scoped_local_context_control loop{};

    lc::set_toggles_armed(true);
    EXPECT_EQ(lc::replay_local_start_context({3}), ROCPROFILER_STATUS_SUCCESS);
    lc::set_toggles_armed(false);

    // arm window closed: the override sticks, but a new toggle is rejected.
    ASSERT_TRUE(lc::local_context_override({3}).has_value());
    EXPECT_TRUE(*lc::local_context_override({3}));
    EXPECT_EQ(lc::replay_local_stop_context({3}), ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR);

    lc::set_toggles_armed(true);
    EXPECT_EQ(lc::replay_local_stop_context({3}), ROCPROFILER_STATUS_SUCCESS);
    lc::set_toggles_armed(false);
    ASSERT_TRUE(lc::local_context_override({3}).has_value());
    EXPECT_FALSE(*lc::local_context_override({3}));
}

// Each context carries an independent override; untouched contexts stay unset.
TEST(kernel_replay_local_context, per_context_independent)
{
    lc::scoped_local_context_control loop{};

    lc::set_toggles_armed(true);
    EXPECT_EQ(lc::replay_local_start_context({1}), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(lc::replay_local_stop_context({2}), ROCPROFILER_STATUS_SUCCESS);
    lc::set_toggles_armed(false);

    EXPECT_TRUE(*lc::local_context_override({1}));
    EXPECT_FALSE(*lc::local_context_override({2}));
    EXPECT_FALSE(lc::local_context_override({3}).has_value());
}

// The override map is scoped to the loop: once the loop guard is gone the query is unset again.
TEST(kernel_replay_local_context, override_cleared_after_loop)
{
    {
        lc::scoped_local_context_control loop{};
        lc::set_toggles_armed(true);
        EXPECT_EQ(lc::replay_local_start_context({5}), ROCPROFILER_STATUS_SUCCESS);
        lc::set_toggles_armed(false);
        EXPECT_TRUE(*lc::local_context_override({5}));
    }
    EXPECT_FALSE(lc::local_context_override({5}).has_value());
}

// End-to-end narrative: stand in for the SDK's replay loop plus a tool that drives the toggles from
// its PASS PHASE_ENTER callback, and a service that consults the override at dispatch. Mirrors the
// real control flow -- loop scope -> per-pass arm window (tool decides) -> dispatch (service
// reads). The tool wants counters collected on every pass but kernel timing only on pass 0.
TEST(kernel_replay_local_context, simulated_replay_loop_and_misbehaving_tool)
{
    const rocprofiler_context_id_t counters{10};  // globally active; wanted every pass
    const rocprofiler_context_id_t timing{20};    // globally active; wanted on pass 0 only

    // The tool's PASS PHASE_ENTER callback. It only needs to turn timing off once, after pass 0;
    // pass 0 touches nothing (both default to their global active state) and passes 2+ touch
    // nothing (timing stays off -- sticky).
    const auto tool_pass_enter = [&](int pass) {
        if(pass == 1)
        {
            EXPECT_EQ(lc::replay_local_stop_context(timing), ROCPROFILER_STATUS_SUCCESS);
        }
    };

    std::vector<bool> counters_ran{};
    std::vector<bool> timing_ran{};

    // SDK: one control object for the whole replay loop.
    lc::scoped_local_context_control loop{};

    for(int pass = 0; pass < 4; ++pass)
    {
        // SDK: open the PASS PHASE_ENTER window and invoke the tool callback inside it.
        lc::set_toggles_armed(true);
        tool_pass_enter(pass);
        lc::set_toggles_armed(false);

        // SDK: dispatch the pass -> each service asks "am I active for this pass?"
        // effective_active = local override if set, else the service's global state (active here).
        counters_ran.push_back(lc::local_context_override(counters).value_or(true));
        timing_ran.push_back(lc::local_context_override(timing).value_or(true));
    }

    // counters run every pass; timing runs pass 0 only, then stays off (sticky).
    EXPECT_EQ(counters_ran, (std::vector<bool>{true, true, true, true}));
    EXPECT_EQ(timing_ran, (std::vector<bool>{true, false, false, false}));

    // A misbehaving tool stashes a toggle and fires it mid-dispatch (no arm window open): the call
    // is rejected and changes nothing.
    EXPECT_EQ(lc::replay_local_start_context(timing), ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR);
    EXPECT_FALSE(lc::local_context_override(timing).value_or(true));  // still off
}
