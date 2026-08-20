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

// Eligibility bookkeeping for range replay. This is the part of the mechanism that decides whether
// a recorded range may be re-executed, and it is deliberately free of GPU dependencies so the
// decision table can be tested directly.

#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace range_replay = ::rocprofiler::range_replay;

namespace
{
constexpr uint64_t queue_a = 0x1001;
constexpr uint64_t queue_b = 0x1002;
constexpr uint64_t agent_a = 0x2001;
constexpr uint64_t agent_b = 0x2002;

range_replay::recorded_dispatch_t
make_dispatch(uint64_t kernel_object)
{
    auto dispatch          = range_replay::recorded_dispatch_t{};
    dispatch.kernel_object = kernel_object;
    dispatch.kernel_id     = kernel_object;
    dispatch.kernarg.assign(64, static_cast<uint8_t>(kernel_object & 0xFF));
    return dispatch;
}
}  // namespace

TEST(range_replay_state, fresh_record_is_eligible)
{
    auto record = range_replay::range_record_t{7};

    EXPECT_EQ(record.range_id(), 7U);
    EXPECT_TRUE(record.eligible());
    EXPECT_EQ(record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED);
    EXPECT_EQ(record.dispatch_count(), 0U);
    EXPECT_EQ(record.observed_dispatch_count(), 0U);
    EXPECT_EQ(record.queue_key(), 0U);
    EXPECT_EQ(record.agent_key(), 0U);
}

TEST(range_replay_state, binds_to_first_queue_and_agent)
{
    auto record = range_replay::range_record_t{1};

    ASSERT_TRUE(record.bind(queue_a, agent_a));
    EXPECT_EQ(record.queue_key(), queue_a);
    EXPECT_EQ(record.agent_key(), agent_a);

    // Re-binding to the same pair is what every subsequent dispatch in the range does.
    EXPECT_TRUE(record.bind(queue_a, agent_a));
    EXPECT_TRUE(record.eligible());
}

TEST(range_replay_state, second_queue_on_same_agent_declines_multi_queue)
{
    auto record = range_replay::range_record_t{1};

    ASSERT_TRUE(record.bind(queue_a, agent_a));
    EXPECT_FALSE(record.bind(queue_b, agent_a));

    EXPECT_FALSE(record.eligible());
    EXPECT_EQ(record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE);
}

TEST(range_replay_state, second_agent_declines_multi_agent)
{
    auto record = range_replay::range_record_t{1};

    ASSERT_TRUE(record.bind(queue_a, agent_a));
    // A different agent is reported as MULTI_AGENT even though the queue differs too: the agent is
    // the more informative reason, because the snapshot is agent-scoped.
    EXPECT_FALSE(record.bind(queue_b, agent_b));

    EXPECT_EQ(record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_AGENT);
}

TEST(range_replay_state, records_dispatches_in_submission_order)
{
    auto record = range_replay::range_record_t{1};
    ASSERT_TRUE(record.bind(queue_a, agent_a));

    for(uint64_t i = 1; i <= 4; ++i)
        ASSERT_TRUE(record.add_dispatch(make_dispatch(i)));

    ASSERT_EQ(record.dispatch_count(), 4U);
    EXPECT_EQ(record.observed_dispatch_count(), 4U);
    for(uint64_t i = 0; i < 4; ++i)
        EXPECT_EQ(record.dispatches()[i].kernel_object, i + 1);
}

TEST(range_replay_state, first_decline_reason_wins)
{
    auto record = range_replay::range_record_t{1};

    record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_GRAPH_LAUNCH);
    record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE);

    EXPECT_EQ(record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_GRAPH_LAUNCH);
}

TEST(range_replay_state, declining_releases_recorded_dispatches)
{
    auto record = range_replay::range_record_t{1};
    ASSERT_TRUE(record.bind(queue_a, agent_a));
    ASSERT_TRUE(record.add_dispatch(make_dispatch(1)));
    ASSERT_TRUE(record.add_dispatch(make_dispatch(2)));

    record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE);

    // The kernarg copies are dropped as soon as the range cannot be replayed, but the range still
    // reports how many dispatches it contained.
    EXPECT_EQ(record.dispatch_count(), 0U);
    EXPECT_EQ(record.observed_dispatch_count(), 2U);
    EXPECT_FALSE(record.add_dispatch(make_dispatch(3)));
    EXPECT_EQ(record.observed_dispatch_count(), 2U);
}

TEST(range_replay_state, declined_range_stops_binding)
{
    auto record = range_replay::range_record_t{1};
    record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_UNSUPPORTED_QUEUE_PATH);

    EXPECT_FALSE(record.bind(queue_a, agent_a));
    EXPECT_EQ(record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_UNSUPPORTED_QUEUE_PATH);
}

TEST(range_replay_state, exceeding_the_dispatch_budget_declines)
{
    auto record = range_replay::range_record_t{1};
    ASSERT_TRUE(record.bind(queue_a, agent_a));

    for(size_t i = 0; i < range_replay::kMaxRecordedDispatches; ++i)
        ASSERT_TRUE(record.add_dispatch(make_dispatch(i))) << "at dispatch " << i;

    EXPECT_FALSE(record.add_dispatch(make_dispatch(0)));
    EXPECT_EQ(record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_PROGRAM_TOO_LARGE);
}

TEST(range_replay_state, external_decline_is_folded_into_the_record)
{
    auto ctx     = range_replay::range_context_t{};
    ctx.record   = range_replay::range_record_t{1};
    ctx.external = std::make_shared<range_replay::external_decline_t>();

    range_replay::fold_external_decline(ctx);
    EXPECT_TRUE(ctx.record.eligible()) << "nothing published, so the range stays eligible";

    // A foreign dispatch and a device-writing copy publish through the same channel; the first one
    // to arrive is the reason reported.
    range_replay::publish_external_decline(*ctx.external,
                                           ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH);
    range_replay::publish_external_decline(*ctx.external,
                                           ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE);

    range_replay::fold_external_decline(ctx);
    EXPECT_FALSE(ctx.record.eligible());
    EXPECT_EQ(ctx.record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH);
}

TEST(range_replay_state, no_range_is_open_by_default)
{
    EXPECT_EQ(range_replay::current_range(), nullptr);
    EXPECT_FALSE(range_replay::any_range_open());
    EXPECT_FALSE(range_replay::this_thread_replaying());

    auto taken = range_replay::range_context_t{};
    EXPECT_FALSE(range_replay::take_range(taken));
}

TEST(range_replay_state, ranges_are_thread_scoped_and_do_not_nest)
{
    ASSERT_TRUE(range_replay::open_range(42));
    EXPECT_TRUE(range_replay::any_range_open());

    auto* open = range_replay::current_range();
    ASSERT_NE(open, nullptr);
    EXPECT_EQ(open->record.range_id(), 42U);

    EXPECT_FALSE(range_replay::open_range(43)) << "a second range on the same thread must fail";

    auto taken = range_replay::range_context_t{};
    ASSERT_TRUE(range_replay::take_range(taken));
    EXPECT_EQ(taken.record.range_id(), 42U);

    EXPECT_EQ(range_replay::current_range(), nullptr);
    EXPECT_FALSE(range_replay::any_range_open());
}

TEST(range_replay_state, foreign_dispatch_declines_an_open_range_on_that_agent)
{
    ASSERT_TRUE(range_replay::open_range(1));
    auto* open = range_replay::current_range();
    ASSERT_NE(open, nullptr);

    // A range only becomes interferable once it binds to an agent, which normally happens when its
    // first dispatch is recorded. There is no queue here, so publish through the channel the queue
    // path would use.
    range_replay::publish_external_decline(*open->external,
                                           ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH);

    // note_foreign_dispatch from this same thread must not decline this thread's own range.
    range_replay::note_foreign_dispatch(agent_a);

    auto taken = range_replay::range_context_t{};
    ASSERT_TRUE(range_replay::take_range(taken));
    EXPECT_EQ(taken.record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH);
}
