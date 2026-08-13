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

// Host-only tests for the snapshot capture contract that decides whether a dispatch is replayed.
//
// device_snapshot_t::ok is the signal the replay loop keys on: a capture that could not be
// completed must not be restored, because a partial restore corrupts application data. The replay
// path therefore declines replay and runs the dispatch exactly once instead. These tests pin that
// contract, and in particular pin that "captured nothing" is a *successful* snapshot rather than a
// failed one -- conflating the two would silently disable replay for any dispatch that touches no
// tracked memory.
//
// No GPU is required: with an empty allocation inventory snap() performs no device copy, and
// restore() of an empty snapshot performs none either.

#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include <hsa/hsa.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace mt = rocprofiler::kernel_replay::memory_tracker;
namespace ms = rocprofiler::kernel_replay::memory_snapshot;

namespace
{
constexpr auto test_agent = hsa_agent_t{.handle = 0xA};

// The tracker inventory is a process-wide static, so each test starts from an empty map.
struct kernel_replay_memory_snapshot : public ::testing::Test
{
    void SetUp() override { clear_inventory(); }
    void TearDown() override { clear_inventory(); }

    static void clear_inventory()
    {
        mt::inventory().wlock([](mt::tracked_map_t& _map) { _map.clear(); });
    }

    static void seed(void* ptr, size_t size, hsa_agent_t agent = test_agent)
    {
        mt::inventory().wlock(
            [&](mt::tracked_map_t& _map) { _map[ptr] = mt::alloc_info_t{size, agent}; });
    }

    // Fake device addresses. Never dereferenced -- they are only map keys, and in every case below
    // restore() rejects the block before it would copy to them.
    static void* fake_addr(uintptr_t v) { return reinterpret_cast<void*>(v); }

    // A tracked block as snap() would have recorded it.
    static ms::device_snapshot_t tracked_block(void* addr, size_t size)
    {
        auto snapshot = ms::device_snapshot_t{};
        auto& blk     = snapshot.blocks.emplace_back();
        blk.gpu_addr  = addr;
        blk.host_copy.resize(size);
        blk.from_tracker = true;
        return snapshot;
    }
};
}  // namespace

// A default-constructed snapshot is usable: only a failed capture clears ok.
TEST_F(kernel_replay_memory_snapshot, default_snapshot_is_ok_and_empty)
{
    auto snapshot = ms::device_snapshot_t{};
    EXPECT_TRUE(snapshot.ok);
    EXPECT_TRUE(snapshot.empty());
}

// Nothing tracked means nothing to save. That is a successful snapshot with zero blocks -- not a
// failure -- so the replay loop proceeds normally.
TEST_F(kernel_replay_memory_snapshot, snap_with_empty_inventory_succeeds)
{
    const auto snapshot = ms::snap(test_agent);
    EXPECT_TRUE(snapshot.ok);
    EXPECT_TRUE(snapshot.empty());
}

// Restoring an empty snapshot is a no-op rather than an error, so a replay pass over a dispatch
// that touches no tracked memory still runs.
TEST_F(kernel_replay_memory_snapshot, restore_of_empty_snapshot_is_noop)
{
    auto snapshot = ms::device_snapshot_t{};
    EXPECT_EQ(ms::restore(snapshot), 0U);
}

// An incomplete capture is reported through ok, and it stays reported even when some blocks were
// already collected before the failure. This is the state on which the caller must decline replay:
// the partially populated block list is exactly what makes restoring it unsafe.
TEST_F(kernel_replay_memory_snapshot, failed_capture_is_distinguishable_from_empty_capture)
{
    auto failed = ms::device_snapshot_t{};
    failed.ok   = false;
    EXPECT_FALSE(failed.ok);
    EXPECT_TRUE(failed.empty());

    auto partial = ms::device_snapshot_t{};
    partial.ok   = false;
    partial.blocks.emplace_back();
    EXPECT_FALSE(partial.ok);
    EXPECT_FALSE(partial.empty());  // non-empty but must still not be restored
}

// A tracked region that is still live at (at least) its snapshotted size is restored. The lookup
// happens under the inventory read lock, so a concurrent free cannot slip between the liveness
// check and the copy.
TEST_F(kernel_replay_memory_snapshot, restore_requires_a_live_tracked_region)
{
    // Present and large enough: restore() gets past the liveness gate and attempts the copy. The
    // copy itself needs a device, so only the gate is asserted here -- see the skip cases below for
    // the behavior this gate exists to produce.
    seed(fake_addr(0x1000), 64);
    const auto live = mt::snap_inventory(test_agent);
    ASSERT_EQ(live.count(fake_addr(0x1000)), 1U);
    EXPECT_GE(live.at(fake_addr(0x1000)), 64U);
}

// Freed between snap and restore: the address is gone from the inventory, so writing the saved
// bytes back would clobber whatever owns that memory now. The block is skipped.
TEST_F(kernel_replay_memory_snapshot, restore_skips_region_freed_after_snap)
{
    const auto snapshot = tracked_block(fake_addr(0x1000), 64);
    // Inventory is empty: the allocation is gone.
    EXPECT_EQ(ms::restore(snapshot), 0U);
}

// Reallocated smaller at the same address: restoring the full snapshotted length would write past
// the end of the new, smaller allocation, so the block is skipped rather than truncated.
TEST_F(kernel_replay_memory_snapshot, restore_skips_region_shrunk_after_snap)
{
    seed(fake_addr(0x1000), 32);
    const auto snapshot = tracked_block(fake_addr(0x1000), 64);
    EXPECT_EQ(ms::restore(snapshot), 0U);
}

// The liveness re-check applies per block: a dead region is skipped without abandoning the rest of
// the snapshot.
TEST_F(kernel_replay_memory_snapshot, restore_skips_only_the_dead_block)
{
    auto snapshot = ms::device_snapshot_t{};
    for(auto addr : {fake_addr(0x1000), fake_addr(0x2000)})
    {
        auto& blk    = snapshot.blocks.emplace_back();
        blk.gpu_addr = addr;
        blk.host_copy.resize(64);
        blk.from_tracker = true;
    }

    // Neither is live, so both are skipped and the count reflects that rather than aborting on the
    // first dead block.
    EXPECT_EQ(ms::restore(snapshot), 0U);
    EXPECT_EQ(snapshot.blocks.size(), 2U);
}
