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

#include <deque>
#include <initializer_list>
#include <vector>

namespace lc = rocprofiler::kernel_replay;

namespace
{
// Stand-in for the SDK's active-context set handed to the loop guard. The localized-context logic
// only reads context_idx (the handle), so these carry nothing else; a deque keeps the pointers
// given to the guard valid as more are added.
struct fake_active_contexts
{
    std::deque<rocprofiler::context::context> storage{};
    rocprofiler::context::context_array_t     array{};

    explicit fake_active_contexts(std::initializer_list<uint64_t> handles)
    {
        for(auto handle : handles)
        {
            storage.emplace_back().context_idx = handle;
            array.emplace_back(&storage.back());
        }
    }
};
}  // namespace

// Outside any replay loop the query yields nothing and the toggles are illegal (not attached).
TEST(kernel_replay_local_context, inactive_outside_loop)
{
    EXPECT_FALSE(lc::local_context_override({1}).has_value());
    EXPECT_EQ(lc::replay_local_enable_context({1}), ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR);
    EXPECT_EQ(lc::replay_local_disable_context({1}), ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR);
    EXPECT_FALSE(lc::local_context_override({1}).has_value());
    EXPECT_FALSE(lc::local_context_has_overrides());  // no active loop on this thread
}

// Inside a loop but before the PASS-enter arm window, toggles still fail and nothing is recorded.
TEST(kernel_replay_local_context, loop_without_arm_rejects_toggles)
{
    fake_active_contexts             active{7};
    lc::scoped_local_context_control loop{active.array};

    EXPECT_EQ(lc::replay_local_enable_context({7}), ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR);
    EXPECT_FALSE(lc::local_context_override({7}).has_value());
    EXPECT_FALSE(lc::local_context_has_overrides());  // loop active but nothing recorded
}

// Armed within a loop: start forces active, stop forces inactive, and the recorded value persists
// past the arm window (sticky) while further toggles outside the window fail.
TEST(kernel_replay_local_context, armed_toggles_record_and_stick)
{
    fake_active_contexts             active{3};
    lc::scoped_local_context_control loop{active.array};

    lc::set_toggles_armed(true);
    EXPECT_EQ(lc::replay_local_enable_context({3}), ROCPROFILER_STATUS_SUCCESS);
    lc::set_toggles_armed(false);
    EXPECT_TRUE(lc::local_context_has_overrides());  // an override is now recorded

    // arm window closed: the override sticks, but a new toggle is rejected.
    ASSERT_TRUE(lc::local_context_override({3}).has_value());
    EXPECT_TRUE(*lc::local_context_override({3}));
    EXPECT_EQ(lc::replay_local_disable_context({3}), ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR);

    lc::set_toggles_armed(true);
    EXPECT_EQ(lc::replay_local_disable_context({3}), ROCPROFILER_STATUS_SUCCESS);
    lc::set_toggles_armed(false);
    ASSERT_TRUE(lc::local_context_override({3}).has_value());
    EXPECT_FALSE(*lc::local_context_override({3}));
}

// Each context carries an independent override; untouched contexts stay unset.
TEST(kernel_replay_local_context, per_context_independent)
{
    fake_active_contexts             active{1, 2};
    lc::scoped_local_context_control loop{active.array};

    lc::set_toggles_armed(true);
    EXPECT_EQ(lc::replay_local_enable_context({1}), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(lc::replay_local_disable_context({2}), ROCPROFILER_STATUS_SUCCESS);
    lc::set_toggles_armed(false);

    EXPECT_TRUE(*lc::local_context_override({1}));
    EXPECT_FALSE(*lc::local_context_override({2}));
    EXPECT_FALSE(lc::local_context_override({3}).has_value());
}

// The override map is scoped to the loop: once the loop guard is gone the query is unset again.
TEST(kernel_replay_local_context, override_cleared_after_loop)
{
    {
        fake_active_contexts             active{5};
        lc::scoped_local_context_control loop{active.array};
        lc::set_toggles_armed(true);
        EXPECT_EQ(lc::replay_local_enable_context({5}), ROCPROFILER_STATUS_SUCCESS);
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
            EXPECT_EQ(lc::replay_local_disable_context(timing), ROCPROFILER_STATUS_SUCCESS);
        }
    };

    std::vector<bool> counters_ran{};
    std::vector<bool> timing_ran{};

    // SDK: one control object for the whole replay loop, seeded with the globally-active contexts.
    fake_active_contexts             active{counters.handle, timing.handle};
    lc::scoped_local_context_control loop{active.array};

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
    EXPECT_EQ(lc::replay_local_enable_context(timing), ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR);
    EXPECT_FALSE(lc::local_context_override(timing).value_or(true));  // still off
}

// A local toggle is only valid for a context that was globally active when the loop began, so a
// local start cannot promote a globally-stopped context. Any other context is rejected with
// CONTEXT_NOT_STARTED and records nothing.
TEST(kernel_replay_local_context, toggle_rejects_context_inactive_pre_replay)
{
    fake_active_contexts             active{5};  // only context 5 was globally active pre-replay
    lc::scoped_local_context_control loop{active.array};
    lc::set_toggles_armed(true);

    // 5 was active pre-replay: a local stop and a later re-start are both honored.
    EXPECT_EQ(lc::replay_local_disable_context({5}), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(lc::replay_local_enable_context({5}), ROCPROFILER_STATUS_SUCCESS);

    // 9 was not active pre-replay, so both start and stop are rejected and record nothing.
    EXPECT_EQ(lc::replay_local_enable_context({9}), ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED);
    EXPECT_EQ(lc::replay_local_disable_context({9}), ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED);

    lc::set_toggles_armed(false);
    EXPECT_FALSE(lc::local_context_override({9}).has_value());  // nothing recorded for 9
    EXPECT_TRUE(lc::local_context_override({5}).has_value());   // 5 was toggled
}

// Last write in a single PASS-enter window wins: start then stop records inactive.
TEST(kernel_replay_local_context, last_write_wins_in_arm_window)
{
    fake_active_contexts             active{9};
    lc::scoped_local_context_control loop{active.array};

    lc::set_toggles_armed(true);
    EXPECT_EQ(lc::replay_local_enable_context({9}), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(lc::replay_local_disable_context({9}), ROCPROFILER_STATUS_SUCCESS);
    lc::set_toggles_armed(false);

    ASSERT_TRUE(lc::local_context_override({9}).has_value());
    EXPECT_FALSE(*lc::local_context_override({9}));
}

// Per-service consumer models used at dispatch time. Dispatch counters and SPM AND the override
// with the global enabled flag (local start cannot promote a globally stopped context); ATT skips
// only when forced off; PC sampling is agent-wide and ignores the override (a local stop is a
// recorded no-op).
TEST(kernel_replay_local_context, simulated_service_consumers)
{
    const rocprofiler_context_id_t counters{10};
    const rocprofiler_context_id_t att{20};
    const rocprofiler_context_id_t spm{30};
    const rocprofiler_context_id_t pcs{40};

    // Mirrors counters/spm dispatch_handlers: enabled = enabled && *ov.
    const auto dispatch_enabled = [](rocprofiler_context_id_t id, bool globally_on) {
        bool enabled = globally_on;
        if(auto ov = lc::local_context_override(id)) enabled = enabled && *ov;
        return enabled;
    };
    const auto att_runs = [](rocprofiler_context_id_t id) {
        if(auto ov = lc::local_context_override(id); ov && !*ov) return false;
        return true;
    };
    const auto pcs_runs = [](rocprofiler_context_id_t) {
        return true;  // agent-wide; does not consult the override
    };

    std::vector<bool> counters_ran{};
    std::vector<bool> att_ran{};
    std::vector<bool> spm_ran{};
    std::vector<bool> pcs_ran{};

    fake_active_contexts             active{counters.handle, att.handle, spm.handle, pcs.handle};
    lc::scoped_local_context_control loop{active.array};
    for(int pass = 0; pass < 4; ++pass)
    {
        lc::set_toggles_armed(true);
        if(pass == 1)
        {
            EXPECT_EQ(lc::replay_local_disable_context(att), ROCPROFILER_STATUS_SUCCESS);
            EXPECT_EQ(lc::replay_local_disable_context(spm), ROCPROFILER_STATUS_SUCCESS);
            EXPECT_EQ(lc::replay_local_disable_context(pcs), ROCPROFILER_STATUS_SUCCESS);
        }
        lc::set_toggles_armed(false);

        counters_ran.push_back(dispatch_enabled(counters, true));
        att_ran.push_back(att_runs(att));
        spm_ran.push_back(dispatch_enabled(spm, true));
        pcs_ran.push_back(pcs_runs(pcs));
    }

    EXPECT_EQ(counters_ran, (std::vector<bool>{true, true, true, true}));
    EXPECT_EQ(att_ran, (std::vector<bool>{true, false, false, false}));
    EXPECT_EQ(spm_ran, (std::vector<bool>{true, false, false, false}));
    EXPECT_EQ(pcs_ran, (std::vector<bool>{true, true, true, true}));
}

// Local start must not resurrect a globally-stopped dispatch service: the override is ANDed with
// the global enabled flag at the consumer (mirrors counters/spm dispatch_handlers).
TEST(kernel_replay_local_context, consumer_no_promotion_of_globally_stopped)
{
    const rocprofiler_context_id_t counters{11};

    const auto dispatch_enabled = [](rocprofiler_context_id_t id, bool globally_on) {
        bool enabled = globally_on;
        if(auto ov = lc::local_context_override(id)) enabled = enabled && *ov;
        return enabled;
    };

    fake_active_contexts             active{counters.handle};
    lc::scoped_local_context_control loop{active.array};
    lc::set_toggles_armed(true);
    EXPECT_EQ(lc::replay_local_enable_context(counters), ROCPROFILER_STATUS_SUCCESS);
    lc::set_toggles_armed(false);

    EXPECT_TRUE(*lc::local_context_override(counters));
    EXPECT_FALSE(dispatch_enabled(counters, /*globally_on=*/false))
        << "local start must not promote a globally stopped context";
    EXPECT_TRUE(dispatch_enabled(counters, /*globally_on=*/true))
        << "local start must undo a prior local stop when globally on";
}
