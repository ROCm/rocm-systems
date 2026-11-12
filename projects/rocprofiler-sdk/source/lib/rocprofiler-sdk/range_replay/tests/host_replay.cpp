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
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <gtest/gtest.h>

#include <array>
#include <csetjmp>
#include <cstdint>
#include <future>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <vector>

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
    EXPECT_NE(buffer, range_replay::host_jmp_target()) << "arming writes a staged target";

    EXPECT_TRUE(range_replay::take_host_jmp_armed());
    EXPECT_EQ(buffer, range_replay::host_jmp_target());
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

namespace
{
rocprofiler_status_t
attempt_nested_host_begin()
{
    auto status = ROCPROFILER_STATUS_SUCCESS;
    ROCPROFILER_RANGE_REPLAY_HOST_BEGIN(2, status);
    return status;
}
}  // namespace

TEST(range_replay_host, rejected_nested_begin_preserves_active_jump_target)
{
    volatile bool reentered = false;
    auto*         target    = static_cast<sigjmp_buf*>(range_replay::host_jmpbuf());
    if(sigsetjmp(*target, 1) == 0)
    {
        ASSERT_TRUE(range_replay::take_host_jmp_armed());
        ASSERT_NE(open_host_range(1), nullptr);

        auto* staged = range_replay::host_jmpbuf();
        EXPECT_NE(staged, target);
        EXPECT_NE(attempt_nested_host_begin(), ROCPROFILER_STATUS_SUCCESS);
        EXPECT_FALSE(range_replay::take_host_jmp_armed());
        EXPECT_EQ(range_replay::host_jmp_target(), target);
        // Do not jump to the returned nested frame if staging or promotion was broken.
        if(staged != target && range_replay::host_jmp_target() == target) siglongjmp(*target, 1);
    }
    else
    {
        reentered = true;
    }

    EXPECT_TRUE(reentered);
    EXPECT_EQ(close_status(), ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED);
}

namespace
{
struct unprofiled_body_t
{
    int                   value  = 1;
    bool                  fail   = false;
    std::vector<uint64_t> passes = {};
};

void
unprofiled_body(uint64_t pass, void* user_data)
{
    auto& body = *static_cast<unprofiled_body_t*>(user_data);
    body.passes.emplace_back(pass);
    EXPECT_EQ(rocprofiler_range_replay_host_track_memory(&body.value, sizeof(body.value)),
              ROCPROFILER_STATUS_SUCCESS);
    if(body.fail) throw std::runtime_error{"unprofiled body failed"};
}
}  // namespace

TEST(range_replay_host, host_run_without_a_context_runs_the_body_once)
{
    auto body = unprofiled_body_t{};
    EXPECT_EQ(rocprofiler_range_replay_host_run(1, unprofiled_body, &body),
              ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND);
    EXPECT_EQ(body.passes, (std::vector<uint64_t>{0}));
    EXPECT_EQ(range_replay::current_range(), nullptr);
    EXPECT_TRUE(range_replay::take_pending_host_regions().empty())
        << "a region the body registered must not be captured by a later range";

    body      = unprofiled_body_t{};
    body.fail = true;
    EXPECT_THROW(rocprofiler_range_replay_host_run(1, unprofiled_body, &body), std::runtime_error);
    EXPECT_EQ(body.passes, (std::vector<uint64_t>{0}));
    EXPECT_TRUE(range_replay::take_pending_host_regions().empty());
}

TEST(range_replay_host, host_run_rejects_a_null_body_before_consuming_any_state)
{
    auto value = 7;
    ASSERT_EQ(rocprofiler_range_replay_host_track_memory(&value, sizeof(value)),
              ROCPROFILER_STATUS_SUCCESS);
    range_replay::host_jmpbuf();

    EXPECT_EQ(rocprofiler_range_replay_host_run(1, nullptr, nullptr),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_TRUE(range_replay::take_host_jmp_armed());
    EXPECT_EQ(range_replay::take_pending_host_regions().size(), 1U);
}

TEST(range_replay_host, host_run_inside_a_range_runs_its_body_as_that_ranges_pass)
{
    auto* ctx = open_host_range(1);
    ASSERT_NE(ctx, nullptr);
    ctx->host.pass = 2;

    auto body = unprofiled_body_t{};
    EXPECT_NE(rocprofiler_range_replay_host_run(2, unprofiled_body, &body),
              ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(body.passes, (std::vector<uint64_t>{2}));
    EXPECT_EQ(range_replay::current_range(), ctx);
    EXPECT_EQ(close_status(), ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED);
}

TEST(range_replay_host, end_leaves_a_host_run_range_open)
{
    auto* ctx = open_host_range(1);
    ASSERT_NE(ctx, nullptr);
    ctx->opened_by_host_run = true;

    EXPECT_EQ(rocprofiler_range_replay_end(), ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(range_replay::current_range(), ctx);
    EXPECT_EQ(close_status(), ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED);
}

TEST(range_replay_host, track_memory_and_get_host_pass_validate_arguments)
{
    auto value = 1;
    EXPECT_EQ(rocprofiler_range_replay_host_track_memory(nullptr, 4),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(rocprofiler_range_replay_host_track_memory(&value, 0),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(rocprofiler_range_replay_host_track_memory(&value, UINT64_MAX),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT)
        << "a region that wraps past the end of the address space";
    EXPECT_TRUE(range_replay::take_pending_host_regions().empty());
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

TEST(range_replay_host, overlapping_regions_restore_the_earliest_capture)
{
    auto data = std::array<uint32_t, 4>{1, 2, 3, 4};

    auto regions = std::vector<range_replay::host_region_t>{};
    regions.emplace_back(range_replay::host::capture_region(data.data(), sizeof(data)));
    // The range modifies part of the buffer, then registers that part again.
    data[1] = 20;
    data[2] = 30;
    regions.emplace_back(range_replay::host::capture_region(&data[1], 2 * sizeof(uint32_t)));

    data.fill(0);
    range_replay::host::restore_regions(regions);
    EXPECT_EQ(data, (std::array<uint32_t, 4>{1, 2, 3, 4}));

    // A later region extending past an earlier one still restores its own bytes.
    auto wide = std::array<uint32_t, 4>{1, 2, 3, 4};
    regions.clear();
    regions.emplace_back(range_replay::host::capture_region(wide.data(), 2 * sizeof(uint32_t)));
    wide[1] = 20;
    wide[2] = 30;
    regions.emplace_back(range_replay::host::capture_region(&wide[1], 3 * sizeof(uint32_t)));
    wide.fill(0);
    range_replay::host::restore_regions(regions);
    EXPECT_EQ(wide, (std::array<uint32_t, 4>{1, 2, 30, 4}));
}

// Random registrations over a buffer the range keeps writing: every byte a region covers is
// restored to its value at the earliest registration covering it, and bytes no region covers are
// untouched.
TEST(range_replay_host, random_overlapping_regions_restore_every_byte_from_its_earliest_capture)
{
    constexpr size_t width = 96;
    auto             rng   = std::mt19937{7};
    for(int trial = 0; trial < 500; ++trial)
    {
        auto buffer   = std::vector<uint8_t>(width);
        auto earliest = std::vector<int>(width, -1);
        for(auto& byte : buffer)
            byte = static_cast<uint8_t>(rng());

        auto regions = std::vector<range_replay::host_region_t>{};
        auto expect  = std::vector<uint8_t>(width);
        for(auto count = 1 + rng() % 6; count > 0; --count)
        {
            const auto begin = rng() % width;
            const auto size  = 1 + rng() % (width - begin);
            regions.emplace_back(range_replay::host::capture_region(buffer.data() + begin, size));
            for(size_t i = begin; i < begin + size; ++i)
            {
                if(earliest[i] >= 0) continue;
                earliest[i] = static_cast<int>(regions.size());
                expect[i]   = buffer[i];
            }
            for(auto& byte : buffer)
                byte = static_cast<uint8_t>(rng());
        }

        const auto untouched = buffer;
        range_replay::host::restore_regions(regions);
        for(size_t i = 0; i < width; ++i)
            ASSERT_EQ(buffer[i], earliest[i] >= 0 ? expect[i] : untouched[i])
                << "trial " << trial << " byte " << i;
    }
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

TEST(range_replay_host, dispatch_from_another_range_owner_declines_a_host_range)
{
    ASSERT_NE(open_host_range(1), nullptr);

    std::thread{[]() {
        ASSERT_NE(open_host_range(2), nullptr);
        range_replay::note_foreign_dispatch(agent_b);
        EXPECT_EQ(close_status(), ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED)
            << "a submission does not interfere with its own range";
    }}.join();

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

TEST(range_replay_host, code_object_unload_declines_a_host_range)
{
    auto* ctx = open_host_range(1);
    ASSERT_NE(ctx, nullptr);
    ctx->host.snapshot_taken = true;

    // Executable globals in the entry snapshot must not be restored after their owner unloads.
    range_replay::note_code_object_unload();

    EXPECT_EQ(close_status(), ROCPROFILER_RANGE_REPLAY_STATUS_CODE_OBJECT_CHANGED_IN_RANGE);
}

TEST(range_replay_host, executable_retirement_waits_for_snapshot_restore_guard)
{
    auto* ctx = open_host_range(1);
    ASSERT_NE(ctx, nullptr);
    ctx->host.snapshot_taken = true;

    auto restore_guard = std::shared_lock{rocprofiler::code_object::executable_lifetime_mutex()};
    auto attempted     = std::promise<bool>{};
    auto blocked       = attempted.get_future();
    auto unload        = std::thread{[&]() {
        auto retirement_guard = std::unique_lock{
            rocprofiler::code_object::executable_lifetime_mutex(), std::defer_lock};
        const bool must_wait = !retirement_guard.try_lock();
        attempted.set_value(must_wait);
        if(must_wait) retirement_guard.lock();
        range_replay::note_code_object_unload();
    }};

    EXPECT_TRUE(blocked.get());
    range_replay::fold_external_decline(*ctx);
    EXPECT_TRUE(ctx->record.eligible()) << "the executable cannot retire during restoration";
    restore_guard.unlock();
    unload.join();

    EXPECT_EQ(close_status(), ROCPROFILER_RANGE_REPLAY_STATUS_CODE_OBJECT_CHANGED_IN_RANGE);
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

TEST(range_replay_host, pending_region_capture_failures_decline_without_throwing)
{
    auto value                = 7;
    auto ctx                  = range_replay::range_context_t{};
    ctx.plan.replay_requested = true;
    auto pending              = std::vector<range_replay::pending_host_region_t>{
        {&value, sizeof(value)}, {&value, std::vector<uint8_t>{}.max_size() + 1}};

    EXPECT_NO_THROW(range_replay::host::enter_range(ctx, std::move(pending)));
    EXPECT_EQ(ctx.record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED);
    EXPECT_FALSE(ctx.host.snapshot_taken);
    EXPECT_TRUE(ctx.host.regions.empty());
    EXPECT_EQ(value, 7);
}

TEST(range_replay_host, in_range_region_capture_failures_decline_without_throwing)
{
    auto value              = 7;
    auto ctx                = range_replay::range_context_t{};
    ctx.host.enabled        = true;
    ctx.host.snapshot_taken = true;
    range_replay::host::track_region(ctx, &value, sizeof(value));
    ASSERT_EQ(ctx.host.regions.size(), 1U);

    EXPECT_NO_THROW(
        range_replay::host::track_region(ctx, &value, std::vector<uint8_t>{}.max_size() + 1));
    EXPECT_EQ(ctx.record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED);
    EXPECT_TRUE(ctx.host.regions.empty());
    range_replay::host::track_region(ctx, &value, sizeof(value));
    EXPECT_TRUE(ctx.host.regions.empty()) << "a declined range must not keep capturing";
    EXPECT_EQ(value, 7);
}

TEST(range_replay_host, replacing_or_resizing_an_entry_allocation_declines_before_restore)
{
    namespace mt     = ::rocprofiler::kernel_replay::memory_tracker;
    auto       value = 7;
    const auto agent = hsa_agent_t{.handle = agent_a};
    for(const auto replacement :
        {mt::alloc_info_t{sizeof(value) + 1, agent, 1}, mt::alloc_info_t{sizeof(value), agent, 2}})
    {
        auto ctx                  = range_replay::range_context_t{};
        ctx.plan.replay_requested = true;
        ctx.record.count_dispatches(1);
        ctx.host.snapshot_taken     = true;
        auto agent_state            = range_replay::host_agent_state_t{};
        agent_state.hsa_agent       = agent;
        agent_state.agent_id.handle = agent_a;
        mt::inventory().wlock([&](auto& map) {
            map[&value] = mt::alloc_info_t{sizeof(value), agent, 1};
        });
        agent_state.tracked = mt::snap_allocation_identity(agent);
        EXPECT_TRUE(mt::allocations_match(agent, agent_state.tracked));
        ctx.host.agents.emplace_back(std::move(agent_state));
        ctx.host.regions.emplace_back(range_replay::host::capture_region(&value, sizeof(value)));
        ++value;
        mt::inventory().wlock([&](auto& map) { map[&value] = replacement; });

        const auto result = range_replay::host::finish_pass(ctx);
        EXPECT_EQ(result.next_pass, 0U);
        EXPECT_EQ(result.status, ROCPROFILER_RANGE_REPLAY_STATUS_ALLOCATION_CHANGED_IN_RANGE);
        EXPECT_EQ(value, 8) << "the host snapshot must not be restored after an identity change";
        mt::record_free(&value);
        value = 7;
    }
}

TEST(range_replay_host, allocation_change_after_initial_checks_declines_before_any_agent_restore)
{
    namespace mt                = ::rocprofiler::kernel_replay::memory_tracker;
    auto first                  = 7;
    auto second                 = 11;
    auto ctx                    = range_replay::range_context_t{};
    ctx.plan.replay_requested   = true;
    ctx.plan.indefinite         = true;
    ctx.plan.user_data.ptr      = &second;
    ctx.plan.replay_continue_cb = [](uint64_t, uint64_t, uint64_t, rocprofiler_user_data_t data) {
        mt::inventory().wlock([&](auto& map) { ++map.at(data.ptr).lifetime_id; });
        return 1;
    };
    ctx.host.pass           = 1;
    ctx.host.snapshot_taken = true;

    for(const auto& item : {std::pair<void*, uint64_t>{&first, agent_a}, {&second, agent_b}})
    {
        auto state             = range_replay::host_agent_state_t{};
        state.hsa_agent.handle = item.second;
        state.agent_id.handle  = item.second;
        mt::inventory().wlock([&](auto& map) {
            map[item.first] = mt::alloc_info_t{sizeof(int), state.hsa_agent, 1};
        });
        state.tracked = mt::snap_allocation_identity(state.hsa_agent);
        state.snapshot.blocks.push_back({item.first, std::vector<char>(sizeof(int), 0), true});
        ctx.host.agents.emplace_back(std::move(state));
        ctx.host.regions.emplace_back(range_replay::host::capture_region(item.first, sizeof(int)));
    }
    ++first;
    ++second;

    // The callback changes the second agent's identity after the initial allocation check. No
    // device copy may start on the first agent before the second agent's identity is validated.
    const auto result = range_replay::host::finish_pass(ctx);
    EXPECT_EQ(result.next_pass, 0U);
    EXPECT_EQ(result.status, ROCPROFILER_RANGE_REPLAY_STATUS_ALLOCATION_CHANGED_IN_RANGE);
    EXPECT_EQ(first, 8);
    EXPECT_EQ(second, 12);
    mt::record_free(&first);
    mt::record_free(&second);
}

TEST(range_replay_host, device_write_from_a_continuation_callback_declines_before_restore)
{
    auto value   = 7;
    auto ctx     = range_replay::range_context_t{};
    ctx.external = std::make_shared<range_replay::external_decline_t>();
    // Copies the application made during the pass are already counted and must not decline.
    ctx.external->owning_device_writes.store(3);
    ctx.plan.replay_requested   = true;
    ctx.plan.indefinite         = true;
    ctx.plan.user_data.ptr      = &ctx;
    ctx.plan.replay_continue_cb = [](uint64_t, uint64_t, uint64_t, rocprofiler_user_data_t data) {
        auto& range = *static_cast<range_replay::range_context_t*>(data.ptr);
        range.external->owning_device_writes.fetch_add(1);
        return 1;
    };
    ctx.host.pass           = 1;
    ctx.host.snapshot_taken = true;
    ctx.host.regions.emplace_back(range_replay::host::capture_region(&value, sizeof(value)));
    ++value;

    const auto result = range_replay::host::finish_pass(ctx);
    EXPECT_EQ(result.next_pass, 0U);
    EXPECT_EQ(result.status, ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE);
    EXPECT_EQ(value, 8) << "a callback copy must decline before the entry snapshot is restored";
}

TEST(range_replay_host, continuation_unload_is_folded_after_reacquiring_executable_lifetime)
{
    auto value                  = 7;
    auto ctx                    = range_replay::range_context_t{};
    ctx.external                = std::make_shared<range_replay::external_decline_t>();
    ctx.plan.replay_requested   = true;
    ctx.plan.indefinite         = true;
    ctx.plan.user_data.ptr      = &ctx;
    ctx.plan.replay_continue_cb = [](uint64_t, uint64_t, uint64_t, rocprofiler_user_data_t data) {
        auto& range    = *static_cast<range_replay::range_context_t*>(data.ptr);
        auto  acquired = false;
        std::thread{[&] {
            auto guard = std::unique_lock{::rocprofiler::code_object::executable_lifetime_mutex(),
                                          std::try_to_lock};
            acquired   = guard.owns_lock();
            if(acquired)
                range_replay::publish_external_decline(
                    *range.external, ROCPROFILER_RANGE_REPLAY_STATUS_CODE_OBJECT_CHANGED_IN_RANGE);
        }}.join();
        EXPECT_TRUE(acquired) << "continuation callbacks must not hold executable lifetime locks";
        return 1;
    };
    ctx.host.pass           = 1;
    ctx.host.snapshot_taken = true;
    ctx.host.regions.emplace_back(range_replay::host::capture_region(&value, sizeof(value)));
    ++value;

    const auto result = range_replay::host::finish_pass(ctx);
    EXPECT_EQ(result.next_pass, 0U);
    EXPECT_EQ(result.status, ROCPROFILER_RANGE_REPLAY_STATUS_CODE_OBJECT_CHANGED_IN_RANGE);
    EXPECT_EQ(value, 8);
}
