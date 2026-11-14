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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// The hub has no admission cap: batches are admitted regardless of how many
// entries are already pending. This binary's CTest target still pins
// ROCPROFILER_KFD_DISPATCH_LOG_MAX_PENDING_PER_GPU=1 -- the knob the cap used to
// read. On a tree that still honors the knob every admission test below runs at
// cap 1 and FAILS (the fail-first proof of the removal); with the cap removed
// the environment variable is inert and they all pass. The guard-rail tests at
// the end hold on both trees: quarantine and the lifecycle gates must keep
// refusing registration.

#include "lib/rocprofiler-sdk/kfd/dispatch_hub.hpp"
#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"

#include <gtest/gtest.h>

#include <cstdint>
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

bool
register_one(hub_t& hub, correlation_key key, window_ptr window, uint64_t corr_id)
{
    auto batch   = std::vector<hub_t::registration>{};
    auto evicted = std::vector<hub_t::leaked>{};
    batch.emplace_back(reg_of(key, std::move(window), corr_id));
    return hub.register_batch(std::move(batch), evicted);
}
}  // namespace

// Three successive one-entry batches on one GPU all admit: nothing counts
// live entries against new ones. At cap 1 the second and third refuse.
TEST(HubNoCap, sequential_batches_all_admit)
{
    reset_disable();
    auto hub = hub_t{};
    EXPECT_TRUE(register_one(hub, key_of(1, 1), mk_window(1, 100), 11));
    EXPECT_TRUE(register_one(hub, key_of(1, 2), mk_window(1, 100), 12));
    EXPECT_TRUE(register_one(hub, key_of(1, 3), mk_window(1, 100), 13));
    EXPECT_EQ(hub.pending_count(), 3u);
}

// A batch bigger than any former cap admits as one unit. At cap 1 this is the
// checked-arithmetic branch: 0 live + 3 new > 1 -> refuse.
TEST(HubNoCap, oversize_single_batch_admits)
{
    reset_disable();
    auto hub   = hub_t{};
    auto batch = std::vector<hub_t::registration>{};
    for(uint32_t i = 1; i <= 3; ++i)
        batch.emplace_back(reg_of(key_of(2, i), mk_window(2, 100), 20 + i));
    auto evicted = std::vector<hub_t::leaked>{};
    EXPECT_TRUE(hub.register_batch(std::move(batch), evicted));
    EXPECT_TRUE(evicted.empty());
    EXPECT_EQ(hub.pending_count(), 3u);
}

// The keys hint answers for a whole batch at once, with no per-GPU count to
// check: an oversize batch is admissible. At cap 1 the hint refuses 3 new keys.
TEST(HubNoCap, keys_hint_admits_oversize_batch)
{
    reset_disable();
    auto hub  = hub_t{};
    auto keys = std::vector<correlation_key>{};
    for(uint32_t i = 1; i <= 3; ++i)
        keys.emplace_back(key_of(3, i));
    EXPECT_TRUE(hub.can_register_batch(keys));
}

// Live pending entries never gate later batches: the hint keeps answering true
// no matter how much is already pending on the GPU. At cap 1, 1 live + 1 new
// refuses, and so does 1 live + 3 new.
TEST(HubNoCap, admission_ignores_live_pending)
{
    reset_disable();
    auto hub = hub_t{};
    ASSERT_TRUE(register_one(hub, key_of(4, 1), mk_window(4, 100), 41));
    EXPECT_TRUE(hub.can_register_batch({key_of(4, 2)}));
    EXPECT_TRUE(hub.can_register_batch({key_of(4, 3), key_of(4, 4), key_of(4, 5)}));
    EXPECT_TRUE(register_one(hub, key_of(4, 2), mk_window(4, 100), 42));
    EXPECT_EQ(hub.pending_count(), 2u);
}

// Guard-rail (holds with and without a cap): a quarantined slot is refused by
// the keys hint and by register_batch itself.
TEST(HubNoCap, quarantined_slot_still_refuses)
{
    reset_disable();
    auto hub = hub_t{};
    hub.quarantine_slot(0, 5);
    EXPECT_FALSE(hub.can_register_batch({key_of(5, 1)}));
    EXPECT_FALSE(register_one(hub, key_of(5, 1), mk_window(5, 100), 51));
    EXPECT_TRUE(register_one(hub, key_of(6, 1), mk_window(6, 100), 61));
    EXPECT_EQ(hub.pending_count(), 1u);
}

// Guard-rail (holds with and without a cap): the mode and disable-lifecycle
// gates still refuse.
TEST(HubNoCap, lifecycle_gates_still_refuse)
{
    reset_disable();
    {  // stopping mode refuses.
        auto hub = hub_t{};
        EXPECT_TRUE(hub.can_register_batch({key_of(7, 1)}));
        hub.set_mode(session_mode::stopping);
        EXPECT_FALSE(hub.can_register_batch({key_of(7, 1)}));
    }
    {  // the process-wide disable latch refuses.
        auto hub = hub_t{};
        signal_less_disable_latch().store(true);
        EXPECT_FALSE(hub.can_register_batch({key_of(7, 2)}));
        EXPECT_FALSE(register_one(hub, key_of(7, 2), mk_window(7, 100), 72));
        reset_disable();
    }
    {  // a fork child abandons the hub.
        auto hub = hub_t{};
        hub.abandon_in_child();
        EXPECT_FALSE(hub.can_register_batch({key_of(7, 3)}));
        EXPECT_FALSE(register_one(hub, key_of(7, 3), mk_window(7, 100), 73));
    }
}
