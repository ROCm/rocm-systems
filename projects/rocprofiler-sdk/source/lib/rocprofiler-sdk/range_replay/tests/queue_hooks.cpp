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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/range_replay/queue_hooks.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"

#include <gtest/gtest.h>

namespace range_replay = ::rocprofiler::range_replay;

// Every call below passes a null queue. The hook dereferences its queue only once it has decided
// a submission is range replay's business, so a regression that stops it returning early aborts
// the test instead of passing quietly.

TEST(range_replay_queue_hooks, inert_when_no_range_is_open)
{
    ASSERT_FALSE(range_replay::any_range_open());

    range_replay::submission_hook(nullptr, nullptr, 0, false, nullptr);

    EXPECT_FALSE(range_replay::any_range_open());
    EXPECT_EQ(range_replay::current_range(), nullptr);
}

TEST(range_replay_queue_hooks, ignores_a_thread_that_is_resubmitting_a_replay)
{
    // any_range_open() is process-wide, so an open range here makes the replaying guard the only
    // thing standing between the hook and its queue.
    ASSERT_TRUE(range_replay::open_range(7));
    auto _take = rocprofiler::common::scope_destructor{[]() {
        auto taken = range_replay::range_context_t{};
        range_replay::take_range(taken);
    }};

    range_replay::set_this_thread_replaying(true);
    range_replay::submission_hook(nullptr, nullptr, 1, false, nullptr);
    range_replay::set_this_thread_replaying(false);

    auto* open = range_replay::current_range();
    ASSERT_NE(open, nullptr);
    EXPECT_TRUE(open->record.eligible());
    EXPECT_EQ(open->record.observed_dispatch_count(), 0U)
        << "a replay's own re-submission must not be seen by a range at all";
}
