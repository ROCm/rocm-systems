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

// The parts of the range replay executor that can be checked without a device.
//
// execute_range() decides whether a range is replayable before it touches the GPU, and every one
// of those refusals becomes the status the tool reads at CLOSE. Getting one wrong does not crash
// anything; it reports the wrong reason, or silently proceeds into a replay window that should not
// have opened. The tests below drive those decisions directly.
//
// The rest is the two pieces of packet-level bookkeeping the pass loop depends on, split out of
// the executor so they can be checked here: the kernarg layout, where an offset wrong by less than
// an alignment unit still lands inside the block and hands a kernel a neighbour's arguments; and
// the pass packet list, where a missing barrier bit lets a pass's dispatches overlap when the
// replay protocol requires them serialized. Neither failure mode is visible without inspecting the
// packets, which is exactly what a hardware test cannot do.
//
// What is not covered here is the replay window itself -- the drain, snapshot, restore, kernarg
// staging and ring submission -- which needs a queue and an agent. That is the samples' and the
// integration tests' job.

#include "lib/rocprofiler-sdk/range_replay/executor.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/range_replay/digest.hpp"
#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace rr = rocprofiler::range_replay;

namespace
{
using snapshot_t = rocprofiler::kernel_replay::memory_snapshot::device_snapshot_t;

rr::recorded_dispatch_t
make_dispatch(size_t kernarg_size, uint64_t kernel_object = 1)
{
    auto dispatch                                 = rr::recorded_dispatch_t{};
    dispatch.kernel_object                        = kernel_object;
    dispatch.packet.kernel_dispatch.kernel_object = kernel_object;
    dispatch.kernarg.assign(kernarg_size, static_cast<uint8_t>(kernel_object));
    return dispatch;
}

// A range that would replay: eligible, a plan that asked for passes, a bound queue, and a good
// snapshot. Individual tests knock out one precondition at a time.
//
// `queue` stays null: every check reachable without a device runs before execute_range() binds the
// queue reference, and there is no way to produce a real hsa::Queue here.
rr::range_context_t
make_replayable_context()
{
    auto ctx                  = rr::range_context_t{};
    ctx.record                = rr::range_record_t{7};
    ctx.plan.replay_requested = true;
    ctx.plan.total_passes     = 3;
    ctx.snapshot_taken        = true;
    ctx.snapshot.ok           = true;
    return ctx;
}

snapshot_t
make_snapshot(const std::vector<std::pair<void*, std::string>>& regions)
{
    auto snapshot = snapshot_t{};
    for(const auto& [addr, contents] : regions)
    {
        auto block     = rocprofiler::kernel_replay::memory_snapshot::mem_block_t{};
        block.gpu_addr = addr;
        block.host_copy.assign(contents.begin(), contents.end());
        snapshot.blocks.emplace_back(std::move(block));
    }
    return snapshot;
}

// Arbitrary distinct device addresses. Never dereferenced: the digest reads host_copy and only
// sorts on gpu_addr.
void* const ADDR_LOW  = reinterpret_cast<void*>(uintptr_t{0x1000});
void* const ADDR_MID  = reinterpret_cast<void*>(uintptr_t{0x2000});
void* const ADDR_HIGH = reinterpret_cast<void*>(uintptr_t{0x3000});
}  // namespace

// ---------------------------------------------------------------------------
// execute_range refusals
// ---------------------------------------------------------------------------

// A range declined while it was recording reports the reason it was declined, not a generic
// failure. This is the whole point of the decline machinery: the tool learns that its range
// spanned two queues, rather than that replay "did not happen".
TEST(range_replay_executor, declined_range_reports_its_own_decline_reason)
{
    for(auto reason : {ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE,
                       ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_AGENT,
                       ROCPROFILER_RANGE_REPLAY_STATUS_GRAPH_LAUNCH,
                       ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE,
                       ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH,
                       ROCPROFILER_RANGE_REPLAY_STATUS_UNSUPPORTED_QUEUE_PATH})
    {
        auto ctx = make_replayable_context();
        ctx.record.decline(reason);

        auto divergence = uint64_t{12345};  // must be cleared even on the refusal paths
        EXPECT_EQ(rr::execute_range(ctx, divergence), reason);
        EXPECT_EQ(divergence, 0u);
    }
}

// The decline check comes first. A range that was declined *and* has no snapshot must still report
// why it was declined, not SNAPSHOT_FAILED: the missing snapshot is a consequence of the decline
// (ensure_entry_snapshot skips an ineligible range), so reporting it would name the symptom.
TEST(range_replay_executor, decline_reason_outranks_a_missing_snapshot)
{
    auto ctx           = make_replayable_context();
    ctx.snapshot_taken = false;
    ctx.record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_AGENT);

    auto divergence = uint64_t{0};
    EXPECT_EQ(rr::execute_range(ctx, divergence), ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_AGENT);
}

// A tool that left pass_count_cb null, or returned 1 from it, opted this range out. That is not an
// error; it is the documented per-range opt-out, and it has its own status so the tool can tell it
// apart from a range it wanted replayed that could not be.
TEST(range_replay_executor, range_without_a_pass_count_is_not_replayed)
{
    auto ctx                  = make_replayable_context();
    ctx.plan.replay_requested = false;

    auto divergence = uint64_t{0};
    EXPECT_EQ(rr::execute_range(ctx, divergence), ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT);
}

// An empty range has nothing to re-execute. A tool that bracketed host-only code gets this rather
// than a spurious success.
TEST(range_replay_executor, empty_range_reports_no_dispatch)
{
    auto ctx = make_replayable_context();
    ASSERT_EQ(ctx.record.dispatch_count(), 0u);

    auto divergence = uint64_t{0};
    EXPECT_EQ(rr::execute_range(ctx, divergence), ROCPROFILER_RANGE_REPLAY_STATUS_NO_DISPATCH);
}

// A range that recorded dispatches but never bound a queue cannot be re-submitted anywhere. The
// two conditions share a status because from the tool's side they are the same thing: the SDK has
// nothing it can replay.
TEST(range_replay_executor, unbound_range_reports_no_dispatch)
{
    auto ctx = make_replayable_context();
    ASSERT_TRUE(ctx.record.bind(/*queue_key=*/1, /*agent_key=*/1));
    ASSERT_TRUE(ctx.record.add_dispatch(make_dispatch(64)));
    ASSERT_EQ(ctx.queue, nullptr);

    auto divergence = uint64_t{0};
    EXPECT_EQ(rr::execute_range(ctx, divergence), ROCPROFILER_RANGE_REPLAY_STATUS_NO_DISPATCH);
}

// ---------------------------------------------------------------------------
// Kernarg layout
// ---------------------------------------------------------------------------

TEST(range_replay_executor, empty_recording_needs_no_kernarg_block)
{
    const auto placement = rr::plan_kernarg_layout({});
    EXPECT_TRUE(placement.offsets.empty());
    EXPECT_EQ(placement.total, 0u);
}

// Dispatches that take no arguments still get an offset, so the offset vector stays parallel to the
// recording, but they reserve nothing.
TEST(range_replay_executor, argument_free_dispatches_reserve_nothing)
{
    const auto placement = rr::plan_kernarg_layout({make_dispatch(0), make_dispatch(0)});

    ASSERT_EQ(placement.offsets.size(), 2u);
    EXPECT_EQ(placement.offsets[0], 0u);
    EXPECT_EQ(placement.offsets[1], 0u);
    EXPECT_EQ(placement.total, 0u);
}

// Every dispatch's arguments start at an aligned offset, and no two overlap. An unaligned start
// would hand the packet a kernarg address the hardware rejects; an overlap would hand a kernel the
// tail of its predecessor's arguments, which is silent.
TEST(range_replay_executor, each_dispatch_is_aligned_and_disjoint)
{
    const auto sizes = std::vector<size_t>{1, 8, 255, 256, 257, 1024};

    auto dispatches = std::vector<rr::recorded_dispatch_t>{};
    for(size_t i = 0; i < sizes.size(); ++i)
        dispatches.emplace_back(make_dispatch(sizes[i], i + 1));

    const auto placement = rr::plan_kernarg_layout(dispatches);
    ASSERT_EQ(placement.offsets.size(), sizes.size());

    for(size_t i = 0; i < sizes.size(); ++i)
    {
        EXPECT_EQ(placement.offsets[i] % rr::kKernargAlignment, 0u)
            << "dispatch " << i << " is not aligned";

        // The block must hold this dispatch's whole argument buffer.
        EXPECT_LE(placement.offsets[i] + sizes[i], placement.total) << "dispatch " << i;

        if(i + 1 < sizes.size())
        {
            EXPECT_GE(placement.offsets[i + 1], placement.offsets[i] + sizes[i])
                << "dispatch " << i << " overlaps " << (i + 1);
        }
    }
}

// The exact packing, so a change to the alignment rule is deliberate: each dispatch is rounded up
// to the alignment, and the total is the sum of the rounded sizes.
TEST(range_replay_executor, sizes_round_up_to_the_alignment)
{
    const auto placement =
        rr::plan_kernarg_layout({make_dispatch(1), make_dispatch(256), make_dispatch(257)});

    ASSERT_EQ(placement.offsets.size(), 3u);
    EXPECT_EQ(placement.offsets[0], 0u);
    EXPECT_EQ(placement.offsets[1], 256u);
    EXPECT_EQ(placement.offsets[2], 512u);
    EXPECT_EQ(placement.total, 512u + 512u);
}

// A leading argument-free dispatch must not push the next one off zero: it reserves nothing, so the
// dispatch after it starts where it would have anyway.
TEST(range_replay_executor, argument_free_dispatches_do_not_consume_space)
{
    const auto placement =
        rr::plan_kernarg_layout({make_dispatch(0), make_dispatch(16), make_dispatch(0)});

    ASSERT_EQ(placement.offsets.size(), 3u);
    EXPECT_EQ(placement.offsets[0], 0u);
    EXPECT_EQ(placement.offsets[1], 0u);
    EXPECT_EQ(placement.offsets[2], 256u);
    EXPECT_EQ(placement.total, 256u);
}

// ---------------------------------------------------------------------------
// Pass packets
// ---------------------------------------------------------------------------

// Every packet in a pass carries the barrier bit, so the pass's dispatches run one at a time. The
// recorded packets may not have had it: the application was free to let them overlap.
TEST(range_replay_executor, pass_packets_are_serialized)
{
    auto dispatches = std::vector<rr::recorded_dispatch_t>{};
    for(uint64_t i = 1; i <= 4; ++i)
    {
        auto dispatch                          = make_dispatch(32, i);
        dispatch.packet.kernel_dispatch.header = 0;  // application left them free to overlap
        dispatches.emplace_back(std::move(dispatch));
    }

    const auto packets = rr::build_pass_packets(dispatches);

    ASSERT_EQ(packets.size(), dispatches.size());
    for(size_t i = 0; i < packets.size(); ++i)
    {
        EXPECT_NE(packets[i].kernel_dispatch.header & (1U << HSA_PACKET_HEADER_BARRIER), 0u)
            << "packet " << i << " may overlap its predecessor";
    }
}

// The barrier bit is the only change: everything else the application set in the packet header,
// and the recording's order, must survive into the pass.
TEST(range_replay_executor, pass_packets_preserve_the_recording)
{
    const auto packet_type_header =
        static_cast<uint16_t>(HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);

    auto dispatches = std::vector<rr::recorded_dispatch_t>{};
    for(uint64_t i = 1; i <= 3; ++i)
    {
        auto dispatch                               = make_dispatch(32, i);
        dispatch.packet.kernel_dispatch.header      = packet_type_header;
        dispatch.packet.kernel_dispatch.grid_size_x = static_cast<uint32_t>(i * 64);
        dispatches.emplace_back(std::move(dispatch));
    }

    const auto packets = rr::build_pass_packets(dispatches);

    ASSERT_EQ(packets.size(), 3u);
    for(size_t i = 0; i < packets.size(); ++i)
    {
        EXPECT_EQ(packets[i].kernel_dispatch.kernel_object, i + 1) << "recording order changed";
        EXPECT_EQ(packets[i].kernel_dispatch.grid_size_x, (i + 1) * 64);
        EXPECT_EQ(packets[i].kernel_dispatch.header & ~(1U << HSA_PACKET_HEADER_BARRIER),
                  packet_type_header)
            << "packet " << i << " header changed beyond the barrier bit";
    }
}

// Rebuilding the same recording twice must produce identical packets: the executor builds the list
// once and refills only the kernarg pointers per pass, so a build that depended on prior state
// would make later passes differ from the first.
TEST(range_replay_executor, building_a_pass_does_not_consume_the_recording)
{
    auto dispatches = std::vector<rr::recorded_dispatch_t>{};
    for(uint64_t i = 1; i <= 3; ++i)
        dispatches.emplace_back(make_dispatch(32, i));

    const auto first  = rr::build_pass_packets(dispatches);
    const auto second = rr::build_pass_packets(dispatches);

    ASSERT_EQ(first.size(), second.size());
    for(size_t i = 0; i < first.size(); ++i)
    {
        EXPECT_EQ(first[i].kernel_dispatch.header, second[i].kernel_dispatch.header);
        EXPECT_EQ(first[i].kernel_dispatch.kernel_object, second[i].kernel_dispatch.kernel_object);
    }
}

// ---------------------------------------------------------------------------
// Snapshot digests
// ---------------------------------------------------------------------------

// The snapshot inventory is unordered, so two snapshots of the same regions can enumerate them in
// different orders. Digests are ordered by device address so the comparison is positional; without
// that, a reordered inventory would report every region as divergent.
TEST(range_replay_executor, digests_are_ordered_by_device_address)
{
    const auto forward =
        make_snapshot({{ADDR_LOW, "alpha"}, {ADDR_MID, "beta"}, {ADDR_HIGH, "gamma"}});
    const auto shuffled =
        make_snapshot({{ADDR_HIGH, "gamma"}, {ADDR_LOW, "alpha"}, {ADDR_MID, "beta"}});

    const auto forward_digests  = rr::snapshot_digests(forward);
    const auto shuffled_digests = rr::snapshot_digests(shuffled);

    ASSERT_EQ(forward_digests.size(), 3u);
    EXPECT_EQ(forward_digests, shuffled_digests);
    EXPECT_EQ(rocprofiler::range_replay::digest::count_divergent(forward_digests, shuffled_digests),
              0u);
}

// A region whose contents changed between the application's execution and the last replayed pass is
// reported, and only that region: the count is what CLOSE hands the tool as divergence_count.
TEST(range_replay_executor, only_changed_regions_are_counted_as_divergent)
{
    const auto before =
        make_snapshot({{ADDR_LOW, "alpha"}, {ADDR_MID, "beta"}, {ADDR_HIGH, "gamma"}});
    const auto after =
        make_snapshot({{ADDR_HIGH, "gamma"}, {ADDR_MID, "BETA"}, {ADDR_LOW, "alpha"}});

    EXPECT_EQ(rocprofiler::range_replay::digest::count_divergent(rr::snapshot_digests(before),
                                                                 rr::snapshot_digests(after)),
              1u);
}

TEST(range_replay_executor, an_empty_snapshot_digests_to_nothing)
{
    EXPECT_TRUE(rr::snapshot_digests(snapshot_t{}).empty());
}
