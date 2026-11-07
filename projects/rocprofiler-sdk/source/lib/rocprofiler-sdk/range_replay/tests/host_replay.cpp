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

// Host replay's GPU-free half: the jump buffer's arming, the public begin macro, the registry rules
// that differ from packet replay, and the capture/restore of registered host memory. The device
// snapshot and the pass loop proper need a GPU and are covered by tests/range-replay-host.

#include "lib/rocprofiler-sdk/range_replay/host_replay.hpp"
#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <gtest/gtest.h>

#include <array>
#include <csetjmp>
#include <cstdint>
#include <thread>

namespace range_replay = ::rocprofiler::range_replay;

namespace
{
constexpr uint64_t agent_a = 0x2001;
constexpr uint64_t agent_b = 0x2002;

range_replay::range_context_t*
open_host_range(uint64_t range_id)
{
    if(!range_replay::open_range(range_id)) return nullptr;
    auto* ctx = range_replay::current_range();
    range_replay::set_host_mode(*ctx);
    return ctx;
}

rocprofiler_range_replay_status_t
close_status()
{
    auto taken = range_replay::range_context_t{};
    if(!range_replay::take_range(taken)) return ROCPROFILER_RANGE_REPLAY_STATUS_LAST;
    return taken.record.status();
}
}  // namespace

TEST(range_replay_host, arming_is_consumed_exactly_once)
{
    EXPECT_FALSE(range_replay::take_host_jmp_armed()) << "nothing armed by default";

    auto* buffer = range_replay::host_jmpbuf();
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(buffer, range_replay::host_jmp_target());

    EXPECT_TRUE(range_replay::take_host_jmp_armed());
    EXPECT_FALSE(range_replay::take_host_jmp_armed())
        << "a second begin must not inherit the first begin's jump target";
}

TEST(range_replay_host, arming_is_thread_local)
{
    range_replay::host_jmpbuf();

    auto other_armed  = true;
    auto other_buffer = static_cast<void*>(nullptr);
    std::thread{[&]() {
        other_armed  = range_replay::take_host_jmp_armed();
        other_buffer = range_replay::host_jmp_target();
    }}.join();

    EXPECT_FALSE(other_armed);
    EXPECT_NE(other_buffer, range_replay::host_jmp_target());
    EXPECT_TRUE(range_replay::take_host_jmp_armed());
}

TEST(range_replay_host, begin_consumes_arming_and_pending_regions_even_when_it_fails)
{
    auto value = 7;
    ASSERT_EQ(rocprofiler_range_replay_host_track_memory(&value, sizeof(value)),
              ROCPROFILER_STATUS_SUCCESS);
    range_replay::host_jmpbuf();

    // No context configured range replay in this process, so begin fails -- after consuming both.
    EXPECT_EQ(rocprofiler_range_replay_begin(1), ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND);
    EXPECT_FALSE(range_replay::take_host_jmp_armed());
    EXPECT_TRUE(range_replay::take_pending_host_regions().empty());
}

TEST(range_replay_host, begin_macro_calls_begin_once_and_reports_success_on_reentry)
{
    // volatile: written after the sigsetjmp, read after the siglongjmp.
    volatile int  entries      = 0;
    volatile auto first_status = ROCPROFILER_STATUS_SUCCESS;
    auto          status       = ROCPROFILER_STATUS_ERROR;

    ROCPROFILER_RANGE_REPLAY_HOST_BEGIN(1, status);
    ++entries;
    if(entries == 1)
    {
        first_status = status;
        // What end() does for a re-executed pass, minus the restore.
        siglongjmp(*static_cast<sigjmp_buf*>(range_replay::host_jmp_target()), 1);
    }

    EXPECT_EQ(entries, 2);
    EXPECT_EQ(first_status, ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND)
        << "the first entry reports begin's own status";
    EXPECT_EQ(status, ROCPROFILER_STATUS_SUCCESS) << "a re-entry must not call begin again";
}

TEST(range_replay_host, track_memory_and_get_host_pass_validate_arguments)
{
    auto value = 1;
    EXPECT_EQ(rocprofiler_range_replay_host_track_memory(nullptr, 4),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(rocprofiler_range_replay_host_track_memory(&value, 0),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(rocprofiler_range_replay_get_host_pass(nullptr),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);

    auto pass = uint64_t{99};
    EXPECT_EQ(rocprofiler_range_replay_get_host_pass(&pass), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(pass, 0U) << "outside a range the host is on the application's own pass";
}

TEST(range_replay_host, regions_registered_before_begin_are_pending_in_order)
{
    auto a = std::array<int, 4>{};
    auto b = 0.0;
    ASSERT_EQ(rocprofiler_range_replay_host_track_memory(a.data(), sizeof(a)),
              ROCPROFILER_STATUS_SUCCESS);
    ASSERT_EQ(rocprofiler_range_replay_host_track_memory(&b, sizeof(b)),
              ROCPROFILER_STATUS_SUCCESS);

    const auto pending = range_replay::take_pending_host_regions();
    ASSERT_EQ(pending.size(), 2U);
    EXPECT_EQ(pending[0].ptr, a.data());
    EXPECT_EQ(pending[0].size, sizeof(a));
    EXPECT_EQ(pending[1].ptr, &b);
    EXPECT_TRUE(range_replay::take_pending_host_regions().empty());
}

TEST(range_replay_host, captured_regions_restore_their_entry_contents)
{
    auto data  = std::array<uint32_t, 8>{1, 2, 3, 4, 5, 6, 7, 8};
    auto other = uint64_t{42};

    auto regions = std::vector<range_replay::host_region_t>{};
    regions.emplace_back(range_replay::host::capture_region(data.data(), sizeof(data)));
    regions.emplace_back(range_replay::host::capture_region(&other, sizeof(other)));

    data.fill(0xDEADBEEF);
    other = 0;
    range_replay::host::restore_regions(regions);

    EXPECT_EQ(data, (std::array<uint32_t, 8>{1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_EQ(other, 42U);
}

TEST(range_replay_host, regions_tracked_inside_a_range_are_captured_on_the_first_pass_only)
{
    auto* ctx = open_host_range(5);
    ASSERT_NE(ctx, nullptr);
    // As if enter_range had captured the entry state.
    ctx->host.snapshot_taken = true;

    auto first = 10;
    ASSERT_EQ(rocprofiler_range_replay_host_track_memory(&first, sizeof(first)),
              ROCPROFILER_STATUS_SUCCESS);
    ASSERT_EQ(ctx->host.regions.size(), 1U);

    ctx->host.pass = 1;
    auto pass      = uint64_t{0};
    ASSERT_EQ(rocprofiler_range_replay_get_host_pass(&pass), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(pass, 1U);

    auto second = 20;
    ASSERT_EQ(rocprofiler_range_replay_host_track_memory(&second, sizeof(second)),
              ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(ctx->host.regions.size(), 1U) << "a re-executed pass must not re-capture";
    EXPECT_TRUE(range_replay::take_pending_host_regions().empty());

    first = 11;
    range_replay::host::restore_regions(ctx->host.regions);
    EXPECT_EQ(first, 10);

    EXPECT_EQ(close_status(), ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED);
}

TEST(range_replay_host, own_device_writes_do_not_decline_a_host_range)
{
    ASSERT_NE(open_host_range(1), nullptr);

    // Copies and memsets are part of the program a host range re-executes.
    range_replay::note_device_write(agent_a);
    range_replay::note_device_write(agent_b);

    EXPECT_EQ(close_status(), ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED);
}

TEST(range_replay_host, another_threads_device_write_declines_a_host_range)
{
    ASSERT_NE(open_host_range(1), nullptr);

    // No agent was bound: a host range snapshots every agent, so it matches any of them.
    std::thread{[]() { range_replay::note_device_write(agent_b); }}.join();

    EXPECT_EQ(close_status(), ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE);
}

TEST(range_replay_host, another_threads_dispatch_on_any_agent_declines_a_host_range)
{
    ASSERT_NE(open_host_range(1), nullptr);

    range_replay::note_foreign_dispatch(agent_a);  // same thread: ignored
    std::thread{[]() { range_replay::note_foreign_dispatch(agent_b); }}.join();

    EXPECT_EQ(close_status(), ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH);
}

TEST(range_replay_host, an_unbound_packet_range_ignores_other_threads)
{
    ASSERT_TRUE(range_replay::open_range(1));

    // The packet rule, unchanged: a range that has recorded nothing has no agent to interfere with.
    std::thread{[]() {
        range_replay::note_foreign_dispatch(agent_a);
        range_replay::note_device_write(agent_a);
    }}.join();

    EXPECT_EQ(close_status(), ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED);
}

TEST(range_replay_host, code_object_unload_does_not_decline_a_host_range)
{
    ASSERT_NE(open_host_range(1), nullptr);

    // Nothing is recorded, so no packet can point at the unloaded code.
    range_replay::note_code_object_unload();

    EXPECT_EQ(close_status(), ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED);
}

TEST(range_replay_host, dispatch_counting_stops_at_a_decline)
{
    auto record = range_replay::range_record_t{1};
    record.count_dispatches(3);
    record.count_dispatches(2);
    EXPECT_EQ(record.observed_dispatch_count(), 5U);
    EXPECT_EQ(record.dispatch_count(), 0U) << "host replay counts, it does not record";

    record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH);
    record.count_dispatches(4);
    EXPECT_EQ(record.observed_dispatch_count(), 5U);
}

namespace
{
// Namespace scope rather than locals of the test: objects the range modifies between the
// sigsetjmp and the siglongjmp have indeterminate values afterwards unless they are volatile or
// have static storage.
std::array<int, 4> jump_state  = {};
std::array<int, 3> jump_sums   = {};
int                jump_passes = 0;
}  // namespace

TEST(range_replay_host, a_jump_re_executes_the_range_over_restored_host_state)
{
    // The control flow end() drives, with only the host half: capture at entry, run, restore, jump.
    jump_state  = {1, 2, 3, 4};
    jump_sums   = {};
    jump_passes = 0;

    auto regions = std::vector<range_replay::host_region_t>{};
    regions.emplace_back(range_replay::host::capture_region(jump_state.data(), sizeof(jump_state)));

    if(sigsetjmp(*static_cast<sigjmp_buf*>(range_replay::host_jmp_target()), 1) != 0)
    {
        // Re-entry: the range runs again from here.
    }

    // The range: host computation that is not idempotent, so a pass over un-restored state differs.
    for(auto& v : jump_state)
        v *= 2;
    jump_sums.at(jump_passes) = jump_state[0] + jump_state[1] + jump_state[2] + jump_state[3];
    ++jump_passes;

    if(jump_passes < 3)
    {
        range_replay::host::restore_regions(regions);
        siglongjmp(*static_cast<sigjmp_buf*>(range_replay::host_jmp_target()), jump_passes);
    }

    EXPECT_EQ(jump_passes, 3);
    EXPECT_EQ(jump_sums, (std::array<int, 3>{20, 20, 20}));
}
