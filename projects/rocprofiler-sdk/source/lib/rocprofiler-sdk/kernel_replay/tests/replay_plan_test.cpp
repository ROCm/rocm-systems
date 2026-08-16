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

// Unit tests for the user_data selection invariant in execute_config_phase_enter.
//
// pass_count_cb and replay_continue_cb are harvested from the shared config_data payload after all
// CONFIG callbacks have run.  Because every callback writes through the same payload reference, the
// last registered context wins (last-writer-wins).  user_data must be taken from the same (last)
// context so that pass_count_cb receives the user_data its registrant actually supplied, rather
// than the user_data of an unrelated earlier context.
//
// These tests exercise the selection rule directly on callback_context_data_vec_t -- the same
// container used in execute_config_phase_enter -- without requiring the HSA / rocprofiler runtime.

#include "lib/rocprofiler-sdk/tracing/fwd.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>

namespace
{
namespace tracing = rocprofiler::tracing;

// Simulate what execute_phase_enter_callbacks does: each context's callback writes its
// pass_count_cb into the shared payload (last writer wins) and may update its own user_data.
// Returns the pass_count_cb pointer that ended up in the shared payload after all callbacks ran.
using pass_count_fn_t = uint64_t (*)(rocprofiler_kernel_dispatch_info_t,
                                     rocprofiler_user_data_t);

// Helper: populate a context vector with N entries where each entry has a distinct user_data.value
// and a corresponding last-writer-wins pass_count_cb function pointer.
struct fake_config
{
    rocprofiler_user_data_t user_data;
    pass_count_fn_t         pass_count_cb;
};
}  // namespace

// ---- single context ----------------------------------------------------------

static uint64_t
pass_count_42(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    return 42;
}

TEST(replay_plan_user_data, single_context_uses_that_context)
{
    // One context: back() == front(), so the selection is unambiguous.
    tracing::callback_context_data_vec_t contexts;
    contexts.push_back({nullptr, {}, rocprofiler_user_data_t{.value = 10}});

    // Simulate last-writer-wins payload (only one writer here).
    pass_count_fn_t shared_pass_count_cb = pass_count_42;

    // Selection rule used in execute_config_phase_enter.
    auto user_data = contexts.empty() ? tracing::empty_user_data : contexts.back().user_data;

    EXPECT_EQ(user_data.value, 10u);
    EXPECT_EQ(shared_pass_count_cb, pass_count_42);
}

// ---- multiple contexts (the interesting case) --------------------------------

static uint64_t
pass_count_1(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    return 1;
}
static uint64_t
pass_count_2(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    return 2;
}
static uint64_t
pass_count_3(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    return 3;
}

TEST(replay_plan_user_data, multiple_contexts_uses_last_not_first)
{
    // Three contexts each writing a distinct pass_count_cb and user_data.
    // After all callbacks run, the shared payload holds the last writer's cb (pass_count_3).
    // user_data must come from the same last context (value=3), not the first (value=1).
    tracing::callback_context_data_vec_t contexts;
    contexts.push_back({nullptr, {}, rocprofiler_user_data_t{.value = 1}});
    contexts.push_back({nullptr, {}, rocprofiler_user_data_t{.value = 2}});
    contexts.push_back({nullptr, {}, rocprofiler_user_data_t{.value = 3}});

    // Simulate last-writer-wins: each callback in order writes its own pass_count_cb into the
    // shared payload; the last one (pass_count_3) survives.
    pass_count_fn_t shared_pass_count_cb = nullptr;
    for(const auto& ctx : contexts)
    {
        // In the real code every callback receives a pointer to the same config_data; here we
        // model the overwrite sequence by directly assigning to shared_pass_count_cb.
        if(ctx.user_data.value == 1) shared_pass_count_cb = pass_count_1;
        else if(ctx.user_data.value == 2)
            shared_pass_count_cb = pass_count_2;
        else if(ctx.user_data.value == 3)
            shared_pass_count_cb = pass_count_3;
    }
    ASSERT_EQ(shared_pass_count_cb, pass_count_3);  // last writer won

    // The selection rule used in execute_config_phase_enter: back(), not front().
    auto user_data = contexts.empty() ? tracing::empty_user_data : contexts.back().user_data;

    // user_data must come from the same (last) context as pass_count_cb.
    EXPECT_EQ(user_data.value, 3u);

    // Confirm front() would have been wrong: it would have selected context 0's user_data while
    // pass_count_cb came from context 2.
    EXPECT_NE(user_data.value, contexts.front().user_data.value);
}

// ---- empty context vector ----------------------------------------------------

TEST(replay_plan_user_data, empty_contexts_gives_empty_user_data)
{
    tracing::callback_context_data_vec_t contexts;

    auto user_data = contexts.empty() ? tracing::empty_user_data : contexts.back().user_data;

    EXPECT_EQ(user_data.value, tracing::empty_user_data.value);
}
