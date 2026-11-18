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

// D9 (F7) end-to-end per-GPU cap behaviour. hub_max_pending_per_gpu() reads its env
// once into a function-local static const, so a specific cap can only be pinned for
// a whole binary: this file's CTest target sets
// ROCPROFILER_KFD_DISPATCH_LOG_MAX_PENDING_PER_GPU=1 so every test here runs at
// cap 1. The cap ARITHMETIC (all values, no wrap) is covered deterministically by
// HubCapArithmetic in dispatch_hub_test.cpp; this file proves the register_batch
// wiring: admit/refuse, eligible-only eviction, and map-unmutated-on-refusal.
// Spec 1227-1238.

#include "lib/rocprofiler-sdk/kfd/dispatch_hub.hpp"
#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace
{
using namespace rocprofiler::kfd;

struct tracked_payload
{
    uint64_t id = 0;
};
using hub_t = DispatchHub<tracked_payload>;

void
reset_disable()
{
    signal_less_disable_latch().store(false);
}

correlation_key
key_of(uint32_t slot, uint32_t dispatch_id, uint32_t gpu = 0)
{
    return correlation_key{slot, dispatch_id, gpu};
}

window_ptr
mk_window(uint32_t slot, uint64_t t_open, uint64_t t_close = kWindowOpen)
{
    auto w    = std::make_shared<owner_window>();
    w->slot   = slot;
    w->t_open = t_open;
    w->t_close.store(t_close);
    return w;
}

hub_t::registration
reg_of(correlation_key key, window_ptr window, uint64_t corr_id)
{
    auto r           = hub_t::registration{};
    r.key            = key;
    r.correlation_id = corr_id;
    r.window         = std::move(window);
    return r;
}
}  // namespace

// The binary's cap really is 1 (env pinned by the CTest target); the rest of the
// suite would be meaningless otherwise, so assert it first.
TEST(HubCap, cap_is_one_for_this_binary) { EXPECT_EQ(hub_max_pending_per_gpu(), 1u); }

// cap 1, empty GPU: a one-entry batch is admitted; a two-entry batch is refused
// (no eligible victim exists and it exceeds the cap) with the map unmutated.
TEST(HubCap, admit_one_refuse_two_on_empty_gpu)
{
    reset_disable();
    auto hub = hub_t{};

    auto one = std::vector<hub_t::registration>{};
    one.emplace_back(reg_of(key_of(1, 1), mk_window(1, 100), 11));
    auto ev1 = std::vector<hub_t::leaked>{};
    EXPECT_TRUE(hub.register_batch(std::move(one), ev1));
    EXPECT_TRUE(ev1.empty());
    EXPECT_EQ(hub.pending_count(), 1u);

    auto two = std::vector<hub_t::registration>{};
    two.emplace_back(reg_of(key_of(2, 1), mk_window(2, 100), 21));
    two.emplace_back(reg_of(key_of(2, 2), mk_window(2, 100), 22));
    auto ev2 = std::vector<hub_t::leaked>{};
    EXPECT_FALSE(hub.register_batch(std::move(two), ev2)) << "two entries exceed cap 1";
    EXPECT_TRUE(ev2.empty());
    EXPECT_EQ(hub.pending_count(), 1u) << "map unmutated on refusal";
}

// A batch larger than the whole cap must refuse (the checked arithmetic must not
// wrap into "no shortfall").
TEST(HubCap, batch_larger_than_cap_refuses)
{
    reset_disable();
    auto hub   = hub_t{};
    auto batch = std::vector<hub_t::registration>{};
    for(uint32_t i = 1; i <= 3; ++i)
        batch.emplace_back(reg_of(key_of(3, i), mk_window(3, 100), 30 + i));
    auto ev = std::vector<hub_t::leaked>{};
    EXPECT_FALSE(hub.register_batch(std::move(batch), ev));
    EXPECT_TRUE(ev.empty());
    EXPECT_EQ(hub.pending_count(), 0u);
}

// live == cap with a closed-window (eligible) entry: a new one-entry batch takes
// the shortfall path and evicts the oldest eligible victim.
TEST(HubCap, live_equals_cap_evicts_eligible_victim)
{
    reset_disable();
    auto hub = hub_t{};
    auto w   = mk_window(4, /*t_open=*/100, /*t_close=*/200);  // closed
    // gc_deadline_ns = 0 so any steady_now_ns() the cap check samples is already past
    // the grace -- the entry is eligible deterministically, no sleep or clock race.
    w->gc_deadline_ns = 0;
    auto first        = std::vector<hub_t::registration>{};
    first.emplace_back(reg_of(key_of(4, 1), w, /*corr_id=*/41));
    auto ev0 = std::vector<hub_t::leaked>{};
    ASSERT_TRUE(hub.register_batch(std::move(first), ev0));  // live == cap (1)

    auto next = std::vector<hub_t::registration>{};
    next.emplace_back(reg_of(key_of(5, 1), mk_window(5, 700), /*corr_id=*/51));
    auto ev1 = std::vector<hub_t::leaked>{};
    EXPECT_TRUE(hub.register_batch(std::move(next), ev1))
        << "eligible victim evicted, batch admitted";
    ASSERT_EQ(ev1.size(), 1u);
    EXPECT_EQ(ev1[0].correlation_id, 41u) << "the closed pre-existing entry was evicted";
    EXPECT_EQ(hub.pending_count(), 1u);
    EXPECT_TRUE(hub.is_ledgered(41u)) << "evicted entry ledgered";
}

// insufficient victim: live == cap but the only entry is OPEN-window (not eligible),
// so a new batch is refused and the map is unmutated. This is the fail-closed case.
TEST(HubCap, insufficient_victim_refuses_and_leaves_map_unmutated)
{
    reset_disable();
    auto hub        = hub_t{};
    auto open_batch = std::vector<hub_t::registration>{};
    open_batch.emplace_back(reg_of(key_of(6, 1), mk_window(6, /*t_open=*/100), /*corr_id=*/61));
    auto ev0 = std::vector<hub_t::leaked>{};
    ASSERT_TRUE(hub.register_batch(std::move(open_batch), ev0));  // live == cap, open window

    auto next = std::vector<hub_t::registration>{};
    next.emplace_back(reg_of(key_of(7, 1), mk_window(7, 300), /*corr_id=*/71));
    auto ev1 = std::vector<hub_t::leaked>{};
    EXPECT_FALSE(hub.register_batch(std::move(next), ev1))
        << "no eligible (closed) victim -> refuse";
    EXPECT_TRUE(ev1.empty());
    EXPECT_EQ(hub.pending_count(), 1u) << "map unmutated across the refusal";
    EXPECT_FALSE(hub.is_ledgered(61u)) << "the open-window entry was NOT evicted or ledgered";
}
