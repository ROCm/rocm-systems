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

// Host-only tests for the kernel-replay allocation tracker.
//
// The behavior worth pinning here is the library-instance gate on memory_tracker_init(). The HSA
// table wrappers chain through the pointer that was in the table when they were installed, so
// installing a second time would capture our own wrapper as that "next" pointer and every
// allocation would recurse until the stack overflowed. Only instance 0 may install.
//
// The rest covers the inventory itself: the tracking gate, per-agent scoping of the snapshot view,
// and removal on free. All in-process map operations, so no GPU is required.

#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace kr = rocprofiler::kernel_replay;
namespace mt = rocprofiler::kernel_replay::memory_tracker;

namespace
{
// Distinct sentinel implementations so a table entry can be compared against what was installed.
// They are never called: the tests only inspect which pointer sits in the table.
hsa_status_t
sentinel_memory_allocate(hsa_region_t, size_t, void**)
{
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
sentinel_memory_free(void*)
{
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
sentinel_pool_allocate(hsa_amd_memory_pool_t, size_t, uint32_t, void**)
{
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
sentinel_pool_free(void*)
{
    return HSA_STATUS_SUCCESS;
}

CoreApiTable
make_core_table()
{
    auto tbl                   = CoreApiTable{};
    tbl.hsa_memory_allocate_fn = sentinel_memory_allocate;
    tbl.hsa_memory_free_fn     = sentinel_memory_free;
    return tbl;
}

AmdExtTable
make_amd_ext_table()
{
    auto tbl                            = AmdExtTable{};
    tbl.hsa_amd_memory_pool_allocate_fn = sentinel_pool_allocate;
    tbl.hsa_amd_memory_pool_free_fn     = sentinel_pool_free;
    return tbl;
}

constexpr auto agent_a = hsa_agent_t{.handle = 0xA};
constexpr auto agent_b = hsa_agent_t{.handle = 0xB};

// The tracker inventory is a process-wide static, so each test starts from an empty map.
struct kernel_replay_memory_tracker : public ::testing::Test
{
    void SetUp() override { clear_inventory(); }
    void TearDown() override { clear_inventory(); }

    static void clear_inventory()
    {
        mt::inventory().wlock([](mt::tracked_map_t& _map) { _map.clear(); });
    }

    // Fake device addresses; only ever used as map keys.
    static void* fake_addr(uintptr_t v) { return reinterpret_cast<void*>(v); }

    static void seed(void* ptr, size_t size, hsa_agent_t agent)
    {
        mt::inventory().wlock(
            [&](mt::tracked_map_t& _map) { _map[ptr] = mt::alloc_info_t{size, agent}; });
    }
};
}  // namespace

// Instance 0 is the first install, so the wrappers replace the runtime's entries.
TEST_F(kernel_replay_memory_tracker, init_installs_wrappers_for_first_instance)
{
    auto core = make_core_table();
    kr::memory_tracker_init(&core, 0);
    EXPECT_NE(core.hsa_memory_allocate_fn, sentinel_memory_allocate);
    EXPECT_NE(core.hsa_memory_free_fn, sentinel_memory_free);

    auto amd_ext = make_amd_ext_table();
    kr::memory_tracker_init(&amd_ext, 0);
    EXPECT_NE(amd_ext.hsa_amd_memory_pool_allocate_fn, sentinel_pool_allocate);
    EXPECT_NE(amd_ext.hsa_amd_memory_pool_free_fn, sentinel_pool_free);
}

// A later library instance must be left untouched. Installing again would capture the wrapper from
// instance 0 as the chained "next" pointer, so every allocation would recurse.
TEST_F(kernel_replay_memory_tracker, init_skips_later_instances)
{
    auto core = make_core_table();
    kr::memory_tracker_init(&core, 1);
    EXPECT_EQ(core.hsa_memory_allocate_fn, sentinel_memory_allocate);
    EXPECT_EQ(core.hsa_memory_free_fn, sentinel_memory_free);

    auto amd_ext = make_amd_ext_table();
    kr::memory_tracker_init(&amd_ext, 2);
    EXPECT_EQ(amd_ext.hsa_amd_memory_pool_allocate_fn, sentinel_pool_allocate);
    EXPECT_EQ(amd_ext.hsa_amd_memory_pool_free_fn, sentinel_pool_free);
}

TEST_F(kernel_replay_memory_tracker, init_ignores_null_table)
{
    kr::memory_tracker_init(static_cast<rocprofiler::hsa::hsa_core_table_t*>(nullptr), 0);
    kr::memory_tracker_init(static_cast<rocprofiler::hsa::hsa_amd_ext_table_t*>(nullptr), 0);
}

// Tracking is off until a replay context turns it on, so a non-replay run pays only a relaxed load.
TEST_F(kernel_replay_memory_tracker, tracking_disabled_until_enabled)
{
    EXPECT_FALSE(mt::tracking_enabled());
    EXPECT_TRUE(mt::set_tracking_enabled(true));
    EXPECT_TRUE(mt::tracking_enabled());
    EXPECT_FALSE(mt::set_tracking_enabled(false));
    EXPECT_FALSE(mt::tracking_enabled());
}

// A replay only snapshots its own agent's memory, so the inventory view is filtered by agent and a
// concurrent replay on another agent is unaffected.
TEST_F(kernel_replay_memory_tracker, snap_inventory_scopes_to_agent)
{
    seed(fake_addr(0x1000), 64, agent_a);
    seed(fake_addr(0x2000), 128, agent_a);
    seed(fake_addr(0x3000), 256, agent_b);

    const auto a = mt::snap_inventory(agent_a);
    EXPECT_EQ(a.size(), 2U);
    ASSERT_EQ(a.count(fake_addr(0x1000)), 1U);
    EXPECT_EQ(a.at(fake_addr(0x1000)), 64U);
    EXPECT_EQ(a.at(fake_addr(0x2000)), 128U);
    EXPECT_EQ(a.count(fake_addr(0x3000)), 0U);

    const auto b = mt::snap_inventory(agent_b);
    EXPECT_EQ(b.size(), 1U);
    EXPECT_EQ(b.at(fake_addr(0x3000)), 256U);
}

TEST_F(kernel_replay_memory_tracker, record_free_removes_from_inventory)
{
    seed(fake_addr(0x1000), 64, agent_a);
    ASSERT_EQ(mt::snap_inventory(agent_a).size(), 1U);

    mt::record_free(fake_addr(0x1000));
    EXPECT_TRUE(mt::snap_inventory(agent_a).empty());
}
