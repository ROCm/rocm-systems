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
// wiring: admit/refuse, completion preservation, and map-unmutated-on-refusal.
// Spec 1227-1238.

#include "lib/rocprofiler-sdk/kfd/dispatch_hub.hpp"
#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <optional>
#include <thread>
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
// (it exceeds the cap; admission never evicts) with the map unmutated.
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

// An expired closed window can still have an EOP in the reader backlog.
// Saturated admission must preserve it for completion instead of evicting it.
TEST(HubCap, live_equals_cap_preserves_closed_window_completion)
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
    EXPECT_FALSE(hub.register_batch(std::move(next), ev1));
    EXPECT_TRUE(ev1.empty());
    EXPECT_EQ(hub.pending_count(), 1u);
    EXPECT_FALSE(hub.is_ledgered(41u));
    ASSERT_EQ(next.size(), 1u);
    EXPECT_EQ(next.front().correlation_id, 51u);

    auto completion = hub.record_kernel_end(key_of(4, 1), uint64_t{150}, 250);
    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(completion->correlation_id, 41u);
    EXPECT_EQ(hub.pending_count(), 0u);
    EXPECT_TRUE(hub.register_batch(std::move(next), ev1));
    EXPECT_EQ(hub.pending_count(), 1u);
}

// live == cap with an OPEN-window entry: a new batch is refused and the map is
// unmutated -- the live entry is neither evicted nor ledgered. Fail-closed.
TEST(HubCap, saturated_open_window_refuses_and_leaves_map_unmutated)
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
    EXPECT_FALSE(hub.register_batch(std::move(next), ev1)) << "saturated -> refuse";
    EXPECT_TRUE(ev1.empty());
    EXPECT_EQ(hub.pending_count(), 1u) << "map unmutated across the refusal";
    EXPECT_FALSE(hub.is_ledgered(61u)) << "the open-window entry was NOT evicted or ledgered";
}

TEST(HubCap, advisory_check_observes_retired_capacity)
{
    reset_disable();
    auto hub    = hub_t{};
    auto window = mk_window(8, 100);
    auto batch  = std::vector<hub_t::registration>{};
    auto leaked = std::vector<hub_t::leaked>{};
    batch.emplace_back(reg_of(key_of(8, 1), window, 81));
    ASSERT_TRUE(hub.register_batch(std::move(batch), leaked));

    EXPECT_FALSE(hub.can_register_batch({key_of(8, 2)}));
    EXPECT_FALSE(hub.can_register_batch(0, 8, 1));
    EXPECT_TRUE(hub.can_register_batch({key_of(8, 2, 1)}));
    EXPECT_TRUE(hub.can_register_batch(1, 8, 1));
    EXPECT_FALSE(hub.can_register_batch(1, 8, 2));
    ASSERT_TRUE(hub.record_kernel_end(key_of(8, 1), uint64_t{150}, 200).has_value());
    EXPECT_TRUE(hub.can_register_batch({key_of(8, 2)}));
    EXPECT_TRUE(hub.can_register_batch(0, 8, 1));
}

// A permanently quarantined slot must be refused by the enqueue-side hint, not
// only by register_batch: otherwise every batch on a queue that reuses the slot
// would skip its signals, be refused, and replay them.
TEST(HubCap, capacity_hint_refuses_quarantined_slot)
{
    reset_disable();
    auto hub = hub_t{};
    hub.quarantine_slot(0, 12);
    EXPECT_FALSE(hub.can_register_batch(0, 12, 1)) << "quarantined slot";
    EXPECT_TRUE(hub.can_register_batch(0, 13, 1)) << "other slot on the same GPU";
    EXPECT_TRUE(hub.can_register_batch(1, 12, 1)) << "same slot number on another GPU";
    EXPECT_FALSE(hub.can_register_batch({key_of(12, 1)}));
    auto batch = std::vector<hub_t::registration>{};
    batch.emplace_back(reg_of(key_of(12, 1), mk_window(12, 100), 121));
    auto leaked = std::vector<hub_t::leaked>{};
    EXPECT_FALSE(hub.register_batch(std::move(batch), leaked));
    EXPECT_TRUE(leaked.empty());
    EXPECT_EQ(hub.pending_count(), 0u);
    ASSERT_EQ(batch.size(), 1u);
    EXPECT_EQ(batch.front().correlation_id, 121u);
}

TEST(HubCap, capacity_hint_observes_lifecycle_gates)
{
    reset_disable();
    auto hub = hub_t{};
    EXPECT_TRUE(hub.can_register_batch(0, 1, 1));
    hub.set_mode(session_mode::stopping);
    EXPECT_FALSE(hub.can_register_batch(0, 1, 1));
    hub.set_mode(session_mode::running);
    signal_less_disable_latch().store(true);
    EXPECT_FALSE(hub.can_register_batch(0, 1, 1));
    reset_disable();
    hub.abandon_in_child();
    EXPECT_FALSE(hub.can_register_batch(0, 1, 1));
}

TEST(HubCap, capacity_hint_refreshes_after_gc_and_quarantine)
{
    reset_disable();
    auto hub    = hub_t{};
    auto batch  = std::vector<hub_t::registration>{};
    auto leaked = std::vector<hub_t::leaked>{};
    batch.emplace_back(reg_of(key_of(13, 1), mk_window(13, 100, 200), 131));
    ASSERT_TRUE(hub.register_batch(std::move(batch), leaked));
    EXPECT_FALSE(hub.can_register_batch(0, 13, 1));
    EXPECT_TRUE(hub.can_register_batch(1, 13, 1));
    EXPECT_EQ(hub.gc_closed_windows(UINT64_MAX).second.dispatches, 1u);
    EXPECT_TRUE(hub.can_register_batch(0, 13, 1));
    batch.clear();
    batch.emplace_back(reg_of(key_of(14, 1), mk_window(14, 300), 141));
    ASSERT_TRUE(hub.register_batch(std::move(batch), leaked));
    EXPECT_FALSE(hub.can_register_batch(0, 15, 1));
    EXPECT_EQ(hub.quarantine_slot(0, 14).size(), 1u);
    EXPECT_TRUE(hub.can_register_batch(0, 15, 1)) << "capacity returned by the quarantine";
    EXPECT_FALSE(hub.can_register_batch(0, 14, 1)) << "the quarantined slot itself";
}

TEST(HubCap, saturated_gpu_refuses_multi_gpu_batch_without_mutation)
{
    reset_disable();
    auto hub    = hub_t{};
    auto batch  = std::vector<hub_t::registration>{};
    auto leaked = std::vector<hub_t::leaked>{};
    batch.emplace_back(reg_of(key_of(9, 1, 1), mk_window(9, 100, 200), 91));
    ASSERT_TRUE(hub.register_batch(std::move(batch), leaked));

    auto mixed = std::vector<hub_t::registration>{};
    mixed.emplace_back(reg_of(key_of(10, 1, 0), mk_window(10, 100), 101));
    mixed.emplace_back(reg_of(key_of(10, 1, 1), mk_window(10, 100), 102));
    EXPECT_FALSE(hub.can_register_batch({key_of(10, 1, 0), key_of(10, 1, 1)}));
    EXPECT_FALSE(hub.register_batch(std::move(mixed), leaked));
    EXPECT_TRUE(leaked.empty());
    EXPECT_EQ(hub.pending_count(), 1u);
    EXPECT_FALSE(hub.is_ledgered(91));
    ASSERT_EQ(mixed.size(), 2u);
    EXPECT_EQ(mixed.front().correlation_id, 101u);
    EXPECT_TRUE(hub.can_register_batch({key_of(10, 1, 0)}));
    ASSERT_TRUE(hub.record_kernel_end(key_of(9, 1, 1), uint64_t{150}, 250).has_value());
    EXPECT_TRUE(hub.register_batch(std::move(mixed), leaked));
    EXPECT_EQ(hub.pending_count(), 2u);
}

TEST(HubCap, concurrent_batches_cannot_consume_the_same_capacity)
{
    reset_disable();
    auto             hub          = hub_t{};
    auto             ready        = std::atomic<size_t>{0};
    auto             admitted     = std::atomic<size_t>{0};
    auto             winner       = std::atomic<uint32_t>{0};
    auto             workers      = std::vector<std::thread>{};
    auto             window       = mk_window(11, 100);
    constexpr size_t worker_count = 16;

    for(uint32_t dispatch = 1; dispatch <= worker_count; ++dispatch)
        workers.emplace_back([&, dispatch]() {
            auto batch  = std::vector<hub_t::registration>{};
            auto leaked = std::vector<hub_t::leaked>{};
            batch.emplace_back(reg_of(key_of(11, dispatch), window, dispatch));
            ready.fetch_add(1, std::memory_order_release);
            while(ready.load(std::memory_order_acquire) != worker_count)
                std::this_thread::yield();
            hub.can_register_batch(0, 11, 1);
            if(hub.register_batch(std::move(batch), leaked))
            {
                admitted.fetch_add(1, std::memory_order_relaxed);
                winner.store(dispatch, std::memory_order_relaxed);
            }
        });
    for(auto& worker : workers)
        worker.join();

    EXPECT_EQ(admitted.load(), 1u);
    EXPECT_EQ(hub.pending_count(), 1u);
    EXPECT_TRUE(hub.record_kernel_end(key_of(11, winner.load()), uint64_t{150}, 200).has_value());
    EXPECT_EQ(hub.pending_count(), 0u);
}
