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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

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
    EXPECT_EQ(record.dispatch_count(), 0U) << "the recording is released on decline";

    // The dispatch that overran the budget was still in the range, so it is reported to the tool
    // even though it was never recorded.
    EXPECT_EQ(record.observed_dispatch_count(), range_replay::kMaxRecordedDispatches + 1);
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

TEST(range_replay_state, kernels_from_code_objects_loaded_before_the_range_are_admitted)
{
    auto record = range_replay::range_record_t{1};
    ASSERT_TRUE(record.bind(queue_a, agent_a));
    record.set_code_object_watermark(5);

    EXPECT_TRUE(record.admit_code_object(1));
    EXPECT_TRUE(record.admit_code_object(5)) << "the newest code object at bind time is covered";
    EXPECT_TRUE(record.eligible());
}

TEST(range_replay_state, kernel_from_a_code_object_loaded_inside_the_range_declines)
{
    auto record = range_replay::range_record_t{1};
    ASSERT_TRUE(record.bind(queue_a, agent_a));
    record.set_code_object_watermark(5);
    ASSERT_TRUE(record.add_dispatch(make_dispatch(1)));

    // Code object 6 was loaded after the entry snapshot (a lazily loaded module, a JIT-compiled
    // kernel), so its module variables are not in the snapshot.
    EXPECT_FALSE(record.admit_code_object(6));
    EXPECT_EQ(record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_CODE_OBJECT_CHANGED_IN_RANGE);
    EXPECT_EQ(record.dispatch_count(), 0U) << "the recording is released on decline";

    record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE);
    EXPECT_EQ(record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_CODE_OBJECT_CHANGED_IN_RANGE);
}

TEST(range_replay_state, code_object_check_needs_a_watermark_and_an_eligible_range)
{
    auto unset = range_replay::range_record_t{1};
    EXPECT_TRUE(unset.admit_code_object(99)) << "no watermark was taken, so nothing is compared";

    auto declined = range_replay::range_record_t{2};
    declined.set_code_object_watermark(5);
    declined.decline(ROCPROFILER_RANGE_REPLAY_STATUS_GRAPH_LAUNCH);
    EXPECT_FALSE(declined.admit_code_object(1));
    EXPECT_EQ(declined.status(), ROCPROFILER_RANGE_REPLAY_STATUS_GRAPH_LAUNCH);
}

TEST(range_replay_state, code_object_unload_leaves_an_unbound_range_eligible)
{
    ASSERT_TRUE(range_replay::open_range(1));

    // Nothing has been recorded, so no packet of this range can point at the unloaded code. A bound
    // range is declined instead; binding needs a queue, so that half is covered on hardware.
    range_replay::note_code_object_unload();

    auto taken = range_replay::range_context_t{};
    ASSERT_TRUE(range_replay::take_range(taken));
    EXPECT_TRUE(taken.record.eligible());
}

TEST(range_replay_state, code_object_unload_with_no_range_open_is_harmless)
{
    ASSERT_FALSE(range_replay::any_range_open());
    range_replay::note_code_object_unload();
    EXPECT_FALSE(range_replay::any_range_open());
}

// Random operation sequences against an independent model of the decision table: the first decline
// wins and drops the recording, the first bind fixes queue and agent, a later agent or queue
// mismatch declines, the budget declines the dispatch that overruns it, and code objects newer than
// a nonzero watermark decline.
TEST(range_replay_state, record_matches_a_reference_model_under_random_operations)
{
    struct model_t
    {
        rocprofiler_range_replay_status_t status    = ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED;
        uint64_t                          queue     = 0;
        uint64_t                          agent     = 0;
        size_t                            recorded  = 0;
        size_t                            observed  = 0;
        uint64_t                          watermark = 0;

        bool eligible() const { return status == ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED; }
        void decline(rocprofiler_range_replay_status_t reason)
        {
            if(reason == ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED || !eligible()) return;
            status   = reason;
            recorded = 0;
        }
    };

    constexpr rocprofiler_range_replay_status_t reasons[] = {
        ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED,
        ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH,
        ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE,
        ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED};
    constexpr uint64_t queues[] = {queue_a, queue_b};
    constexpr uint64_t agents[] = {agent_a, agent_b};

    auto rng = std::mt19937_64{0x5eed};
    for(int sequence = 0; sequence < 2000; ++sequence)
    {
        auto record = range_replay::range_record_t{static_cast<uint64_t>(sequence)};
        auto model  = model_t{};
        // Some sequences run the recording up to the budget.
        const bool fill = (sequence % 50) == 0;
        for(int step = 0; step < (fill ? 5000 : 64); ++step)
        {
            switch(fill ? 1 : rng() % 6)
            {
                case 0:
                {
                    const auto queue = queues[rng() % 2];
                    const auto agent = agents[rng() % 2];
                    auto       ok    = false;
                    if(model.eligible())
                    {
                        if(model.queue == 0 && model.agent == 0)
                        {
                            model.queue = queue;
                            model.agent = agent;
                            ok          = true;
                        }
                        else if(model.agent != agent)
                            model.decline(ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_AGENT);
                        else if(model.queue != queue)
                            model.decline(ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE);
                        else
                            ok = true;
                    }
                    ASSERT_EQ(record.bind(queue, agent), ok);
                    break;
                }
                case 1:
                {
                    auto ok = false;
                    if(model.eligible())
                    {
                        ++model.observed;
                        if(model.recorded >= range_replay::kMaxRecordedDispatches)
                            model.decline(ROCPROFILER_RANGE_REPLAY_STATUS_PROGRAM_TOO_LARGE);
                        else
                        {
                            ++model.recorded;
                            ok = true;
                        }
                    }
                    ASSERT_EQ(record.add_dispatch(make_dispatch(rng() % 7 + 1)), ok);
                    break;
                }
                case 2:
                {
                    const auto count = static_cast<size_t>(rng() % 4);
                    if(model.eligible()) model.observed += count;
                    record.count_dispatches(count);
                    break;
                }
                case 3:
                {
                    const auto reason = reasons[rng() % 4];
                    model.decline(reason);
                    record.decline(reason);
                    break;
                }
                case 4:
                {
                    model.watermark = rng() % 4;
                    record.set_code_object_watermark(model.watermark);
                    break;
                }
                default:
                {
                    const auto id = rng() % 6;
                    auto       ok = model.eligible();
                    if(ok && model.watermark != 0 && id > model.watermark)
                    {
                        model.decline(ROCPROFILER_RANGE_REPLAY_STATUS_CODE_OBJECT_CHANGED_IN_RANGE);
                        ok = false;
                    }
                    ASSERT_EQ(record.admit_code_object(id), ok);
                    break;
                }
            }
            ASSERT_EQ(record.status(), model.status) << "sequence " << sequence << " step " << step;
            ASSERT_EQ(record.eligible(), model.eligible());
            ASSERT_EQ(record.dispatch_count(), model.recorded);
            ASSERT_EQ(record.observed_dispatch_count(), model.observed);
        }
    }
}

// Owners open host ranges while other threads dispatch. Every operation is stamped on one global
// sequence counter. A dispatch whose foreign-dispatch check ran entirely inside a range's open
// window must decline that range; a range whose window no dispatch overlapped must stay eligible.
TEST(range_replay_state, foreign_dispatches_decline_exactly_the_ranges_they_overlap)
{
    struct span_t
    {
        uint64_t begin = 0;
        uint64_t end   = 0;
    };
    struct range_window_t
    {
        span_t inner    = {};  // after open returned .. before take began
        span_t outer    = {};  // before open began .. after take returned
        bool   declined = false;
    };

    constexpr int owners           = 4;
    constexpr int foreigners       = 3;
    constexpr int ranges_per_owner = 400;

    auto seq        = std::atomic<uint64_t>{1};
    auto owners_out = std::atomic<int>{owners};
    auto windows    = std::vector<std::vector<range_window_t>>(owners);
    auto dispatches = std::vector<std::vector<span_t>>(foreigners);

    auto threads = std::vector<std::thread>{};
    for(int i = 0; i < owners; ++i)
    {
        threads.emplace_back([&, i]() {
            auto rng = std::mt19937{static_cast<uint32_t>(i + 1)};
            for(int r = 0; r < ranges_per_owner; ++r)
            {
                auto window        = range_window_t{};
                window.outer.begin = seq.fetch_add(1);
                if(!range_replay::open_range(static_cast<uint64_t>(r))) return;
                range_replay::set_host_mode(*range_replay::current_range());
                window.inner.begin = seq.fetch_add(1);
                for(auto spin = rng() % 64; spin > 0; --spin)
                    std::this_thread::yield();
                window.inner.end = seq.fetch_add(1);
                auto taken       = range_replay::range_context_t{};
                if(!range_replay::take_range(taken)) return;
                window.outer.end = seq.fetch_add(1);
                window.declined  = !taken.record.eligible();
                if(window.declined)
                {
                    EXPECT_EQ(taken.record.status(),
                              ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH);
                }
                windows[i].emplace_back(window);
                // Leave gaps so that some windows see no dispatch at all.
                for(auto spin = rng() % 256; spin > 0; --spin)
                    std::this_thread::yield();
            }
            owners_out.fetch_sub(1);
        });
    }
    for(int i = 0; i < foreigners; ++i)
    {
        threads.emplace_back([&, i]() {
            auto rng = std::mt19937{static_cast<uint32_t>(100 + i)};
            while(owners_out.load() > 0)
            {
                auto span  = span_t{};
                span.begin = seq.fetch_add(1);
                range_replay::note_foreign_dispatch((rng() % 2) != 0 ? agent_a : agent_b);
                span.end = seq.fetch_add(1);
                dispatches[i].emplace_back(span);
                for(auto spin = rng() % 512; spin > 0; --spin)
                    std::this_thread::yield();
            }
        });
    }
    for(auto& thread : threads)
        thread.join();

    auto all = std::vector<span_t>{};
    for(const auto& list : dispatches)
        all.insert(all.end(), list.begin(), list.end());
    std::sort(all.begin(), all.end(), [](auto lhs, auto rhs) { return lhs.begin < rhs.begin; });
    // min_end_after[k]: smallest end among dispatches k..; max_end_before[k]: largest end among
    // dispatches ..k-1.
    auto min_end_after  = std::vector<uint64_t>(all.size() + 1, UINT64_MAX);
    auto max_end_before = std::vector<uint64_t>(all.size() + 1, 0);
    for(size_t k = all.size(); k > 0; --k)
        min_end_after[k - 1] = std::min(min_end_after[k], all[k - 1].end);
    for(size_t k = 0; k < all.size(); ++k)
        max_end_before[k + 1] = std::max(max_end_before[k], all[k].end);
    auto first_beginning_after = [&](uint64_t stamp) {
        return static_cast<size_t>(std::upper_bound(all.begin(),
                                                    all.end(),
                                                    stamp,
                                                    [](uint64_t value, const span_t& span) {
                                                        return value < span.begin;
                                                    }) -
                                   all.begin());
    };

    size_t must_decline = 0;
    size_t must_keep    = 0;
    for(const auto& list : windows)
    {
        ASSERT_EQ(list.size(), static_cast<size_t>(ranges_per_owner));
        for(const auto& window : list)
        {
            if(min_end_after[first_beginning_after(window.inner.begin)] < window.inner.end)
            {
                ++must_decline;
                EXPECT_TRUE(window.declined) << "a dispatch inside the window was missed";
            }
            if(max_end_before[first_beginning_after(window.outer.end - 1)] <= window.outer.begin)
            {
                ++must_keep;
                EXPECT_FALSE(window.declined) << "declined without an overlapping dispatch";
            }
        }
    }
    EXPECT_GT(must_decline, 0U) << "the stress run never placed a dispatch inside a window";
    EXPECT_GT(must_keep, 0U) << "the stress run never left a window undisturbed";
    EXPECT_FALSE(range_replay::any_range_open());
}
