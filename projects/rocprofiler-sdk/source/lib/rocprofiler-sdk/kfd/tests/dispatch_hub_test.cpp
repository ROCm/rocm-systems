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

// Unit tests for the signal-less pending-completion hub under time-as-generation.
// Exercised without a GPU, the HSA runtime, or the reader thread: window ticks
// are injected directly.

#include "lib/rocprofiler-sdk/kfd/dispatch_hub.hpp"
#include "lib/rocprofiler-sdk/kfd/complete_signal_less_dispatch.hpp"
#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"
#include "lib/rocprofiler-sdk/kfd/env_parse.hpp"
#include "lib/rocprofiler-sdk/kfd/owner_registry.hpp"
#include "lib/rocprofiler-sdk/kfd/record_pipe.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace
{
using namespace rocprofiler::kfd;

// Payload that tracks its own lifetime, so a test can prove exactly one owner at
// all times and that cleanup runs exactly once.
struct tracked_payload
{
    static std::atomic<int> live;
    uint64_t                id    = 0;
    bool                    armed = false;

    tracked_payload() = default;
    explicit tracked_payload(uint64_t v)
    : id{v}
    , armed{true}
    {
        ++live;
    }
    tracked_payload(tracked_payload&& rhs) noexcept
    : id{rhs.id}
    , armed{rhs.armed}
    {
        rhs.armed = false;
    }
    tracked_payload& operator=(tracked_payload&& rhs) noexcept
    {
        if(this != &rhs)
        {
            if(armed) --live;
            id        = rhs.id;
            armed     = rhs.armed;
            rhs.armed = false;
        }
        return *this;
    }
    tracked_payload(const tracked_payload&) = delete;
    tracked_payload& operator=(const tracked_payload&) = delete;
    ~tracked_payload()
    {
        if(armed) --live;
    }
};

std::atomic<int> tracked_payload::live = {0};

using hub_t = DispatchHub<tracked_payload>;

// The disable latch is a process-wide inline static; reset it so one test's
// disable cannot leak into another's registration.
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
mk_window(uint32_t slot,
          uint64_t t_open,
          uint64_t t_close     = kWindowOpen,
          bool     first_owner = true,
          bool     superseded  = false)
{
    auto w         = std::make_shared<owner_window>();
    w->slot        = slot;
    w->first_owner = first_owner;
    w->t_open      = t_open;
    w->t_close.store(t_close);
    w->superseded.store(superseded);
    return w;
}

hub_t::registration
reg_of(correlation_key key, window_ptr window, uint64_t corr_id = 1, uint64_t payload_id = 0)
{
    auto r           = hub_t::registration{};
    r.key            = key;
    r.correlation_id = corr_id;
    r.window         = std::move(window);
    r.payload        = tracked_payload{payload_id != 0 ? payload_id : key.dispatch_idx_low32};
    return r;
}

bool
register_win(hub_t& hub, correlation_key key, window_ptr window, uint64_t corr_id = 1)
{
    auto batch = std::vector<hub_t::registration>{};
    batch.emplace_back(reg_of(key, std::move(window), corr_id));
    auto evicted = std::vector<hub_t::leaked>{};
    return hub.register_batch(std::move(batch), evicted);
}

std::optional<hub_t::proven>
end_start(hub_t& hub, correlation_key key, uint64_t start, uint64_t end)
{
    return hub.record_kernel_end(key, std::optional<uint64_t>{start}, end);
}

std::optional<hub_t::proven>
end_nostart(hub_t& hub, correlation_key key, uint64_t end)
{
    return hub.record_kernel_end(key, std::nullopt, end);
}
}  // namespace

// the resolution rule -- containment by the window's START tick.
TEST(DispatchHub, resolution_rule)
{
    reset_disable();
    const auto K = key_of(5, 7);

    {  // recycled doorbell, both entries live at once (the multimap). Each
       // record resolves into the window that contains its START tick.
        auto hub = hub_t{};
        ASSERT_TRUE(register_win(hub, K, mk_window(5, 100, 200), /*corr_id=*/10));
        ASSERT_TRUE(register_win(hub,
                                 K,
                                 mk_window(5, 200, kWindowOpen, /*first=*/false),
                                 /*corr_id=*/20));
        auto old = end_start(hub, K, /*start=*/150, /*end=*/210);
        ASSERT_TRUE(old.has_value());
        EXPECT_EQ(old->key.dispatch_idx_low32, 7u);
        auto neu = end_start(hub, K, /*start=*/250, /*end=*/300);
        ASSERT_TRUE(neu.has_value());
        EXPECT_EQ(hub.pending_count(), 0u) << "both completed";
    }
    {  // a sole candidate is STILL time-checked -- a tick outside the only
       // window drops, never accepts (a wrong-dispatch hazard otherwise).
        auto hub = hub_t{};
        ASSERT_TRUE(register_win(hub, K, mk_window(5, 100, 200)));
        EXPECT_FALSE(end_start(hub, K, /*start=*/250, /*end=*/300).has_value())
            << "sole candidate must still be contained";
        EXPECT_EQ(hub.pending_count(), 1u) << "not consumed";
    }
    {  // late START. t_close=200, successor open at t_open=200; a
       // START at 201 resolves to the successor, never the closed predecessor.
        auto hub = hub_t{};
        ASSERT_TRUE(register_win(hub, K, mk_window(5, 100, 200)));
        EXPECT_FALSE(end_start(hub, K, 201, 300).has_value()) << "past a closed window -> drop";
        ASSERT_TRUE(register_win(hub, K, mk_window(5, 200, kWindowOpen, false), /*corr_id=*/2));
        EXPECT_TRUE(end_start(hub, K, 201, 300).has_value()) << "contained by the successor";
    }
    {  // both interval ends are exclusive; a boundary tick is dropped.
        auto hub = hub_t{};
        ASSERT_TRUE(register_win(hub, K, mk_window(5, 100, 200)));
        ASSERT_TRUE(register_win(hub, K, mk_window(5, 200, kWindowOpen, false), /*corr_id=*/2));
        EXPECT_FALSE(end_start(hub, K, /*start=*/100, 300).has_value()) << "t_open edge dropped";
        EXPECT_FALSE(end_start(hub, K, /*start=*/200, 300).has_value())
            << "shared boundary accepted by neither";
        EXPECT_TRUE(end_start(hub, K, /*start=*/199, 300).has_value()) << "inside predecessor";
        EXPECT_TRUE(end_start(hub, K, /*start=*/201, 300).has_value()) << "inside successor";
    }
    {// unknown-START needs sole candidate && first_owner && !superseded.
     {auto hub = hub_t{};
    ASSERT_TRUE(register_win(hub, K, mk_window(5, 100)));  // first_owner, not superseded
    EXPECT_TRUE(end_nostart(hub, K, 300).has_value()) << "sole first-owner accepts";
}
{  // superseded -> drop
    auto hub = hub_t{};
    ASSERT_TRUE(register_win(hub, K, mk_window(5, 100, kWindowOpen, true, /*superseded=*/true)));
    EXPECT_FALSE(end_nostart(hub, K, 300).has_value());
}
{  // not first_owner (predecessor existed, since pruned) -> drop
    auto hub = hub_t{};
    ASSERT_TRUE(register_win(hub, K, mk_window(5, 100, kWindowOpen, /*first=*/false)));
    EXPECT_FALSE(end_nostart(hub, K, 300).has_value());
}
{  // two candidates -> drop
    auto hub = hub_t{};
    ASSERT_TRUE(register_win(hub, K, mk_window(5, 100, 200)));
    ASSERT_TRUE(register_win(hub, K, mk_window(5, 200), /*corr_id=*/2));
    EXPECT_FALSE(end_nostart(hub, K, 300).has_value());
}
{  // a stale EOP (its own START was lost, it predates this window) must NOT
   // consume the newer entry: reject BEFORE take, leaving it PENDING for its own.
    auto hub = hub_t{};
    ASSERT_TRUE(register_win(hub, K, mk_window(5, 100)));
    EXPECT_FALSE(end_nostart(hub, K, 50).has_value()) << "EOP before t_open rejected";
    EXPECT_FALSE(end_nostart(hub, K, 100).has_value()) << "EOP at t_open rejected";
    EXPECT_EQ(hub.pending_count(), 1u) << "a rejected EOP consumes nothing";
    EXPECT_TRUE(end_nostart(hub, K, 300).has_value()) << "its own later EOP still resolves";
}
}
{  // a contained START with end < start is dropped (free sanity check).
    auto hub = hub_t{};
    ASSERT_TRUE(register_win(hub, K, mk_window(5, 100, 200)));
    EXPECT_FALSE(end_start(hub, K, /*start=*/150, /*end=*/140).has_value());
}
{  // An EOP with no candidate at all is rejected, never cached.
    auto hub = hub_t{};
    EXPECT_FALSE(end_start(hub, K, 150, 200).has_value());
    ASSERT_TRUE(register_win(hub, K, mk_window(5, 100, 200)));
    EXPECT_TRUE(end_start(hub, K, 150, 200).has_value()) << "fresh record, not a cached stale";
}
}

// admissibility -- across a closed window yes, onto an open one no, onto
// a quarantined slot never; all-or-none registration.
TEST(DispatchHub, admissibility_and_batch_atomicity)
{
    reset_disable();
    {  // recurrence across windows is admitted; onto an OPEN-window entry refused.
        auto hub = hub_t{};
        ASSERT_TRUE(register_win(hub, key_of(4, 1), mk_window(4, 100)));  // open
        EXPECT_FALSE(register_win(hub, key_of(4, 1), mk_window(4, 100)))
            << "same key on an open window is a low-32 wrap -> refuse";
        // close it, then a successor with the same key3 is admissible.
        ASSERT_TRUE(end_start(hub, key_of(4, 1), 150, 200).has_value());
        EXPECT_TRUE(register_win(hub, key_of(4, 1), mk_window(4, 300, 400)))
            << "closed predecessor -> the recycled key is admitted";
    }
    {  // all-or-none: a last-entry open-window collision inserts none of the batch.
        auto hub = hub_t{};
        ASSERT_TRUE(register_win(hub, key_of(4, 2), mk_window(4, 100)));
        auto batch = std::vector<hub_t::registration>{};
        batch.emplace_back(reg_of(key_of(4, 10), mk_window(4, 100)));
        batch.emplace_back(reg_of(key_of(4, 2), mk_window(4, 100)));  // collides (open)
        auto evicted = std::vector<hub_t::leaked>{};
        EXPECT_FALSE(hub.register_batch(std::move(batch), evicted));
        EXPECT_EQ(hub.pending_count(), 1u) << "no partial registration";
    }
    {  // a duplicate key within one batch is rejected.
        auto hub   = hub_t{};
        auto batch = std::vector<hub_t::registration>{};
        batch.emplace_back(reg_of(key_of(4, 1), mk_window(4, 100)));
        batch.emplace_back(reg_of(key_of(4, 1), mk_window(4, 100)));
        auto evicted = std::vector<hub_t::leaked>{};
        EXPECT_FALSE(hub.register_batch(std::move(batch), evicted));
        EXPECT_EQ(hub.pending_count(), 0u);
    }
    {  // quarantine refuses every future registration and is permanent.
        auto hub = hub_t{};
        hub.quarantine_slot(0, 6);
        EXPECT_FALSE(register_win(hub, key_of(6, 1), mk_window(6, 100)));
        EXPECT_FALSE(hub.can_register_batch({key_of(6, 1)}));
        EXPECT_TRUE(hub.can_register_batch({key_of(7, 1)})) << "unaffected slot works";
    }
    {  // two GPUs, same slot/index = two independent dispatches (no cross-agent tick).
        auto hub = hub_t{};
        ASSERT_TRUE(register_win(hub, key_of(40, 1, /*gpu=*/0), mk_window(40, 100), 10));
        ASSERT_TRUE(register_win(hub, key_of(40, 1, /*gpu=*/1), mk_window(40, 100), 20));
        EXPECT_EQ(hub.pending_count(), 2u);
        auto p0 = end_start(hub, key_of(40, 1, 0), 150, 200);
        ASSERT_TRUE(p0.has_value());
        EXPECT_EQ(p0->key.gpu_id, 0u);
        auto p1 = end_start(hub, key_of(40, 1, 1), 150, 200);
        ASSERT_TRUE(p1.has_value());
        EXPECT_EQ(p1->key.gpu_id, 1u);
    }
}

// deferred window GC leaks+ledgers entries whose window closed >= grace ago.
TEST(DispatchHub, gc_closed_windows)
{
    reset_disable();
    auto hub = hub_t{};
    auto w   = mk_window(5, 100);
    // register two entries on this window, then close it with gc_deadline_ns=1000.
    {
        auto batch = std::vector<hub_t::registration>{};
        batch.emplace_back(reg_of(key_of(5, 1), w, /*corr_id=*/11));
        batch.emplace_back(reg_of(key_of(5, 2), w, /*corr_id=*/12));
        auto evicted = std::vector<hub_t::leaked>{};
        ASSERT_TRUE(hub.register_batch(std::move(batch), evicted));
    }
    w->gc_deadline_ns = 1000;
    w->t_close.store(200, std::memory_order_release);

    EXPECT_TRUE(hub.gc_closed_windows(/*now_ns=*/999).first.empty()) << "before the grace deadline";
    EXPECT_EQ(hub.pending_count(), 2u);

    auto [leaked, stats] = hub.gc_closed_windows(/*now_ns=*/1000);
    EXPECT_EQ(stats.dispatches, 2u);
    EXPECT_EQ(stats.correlation_ids, 2u);
    EXPECT_EQ(hub.pending_count(), 0u);
    EXPECT_TRUE(hub.is_ledgered(11)) << "GC'd entries are ledgered, not force-retired";
    EXPECT_TRUE(hub.is_ledgered(12));
    // A record for a GC'd window now finds no candidate.
    EXPECT_FALSE(end_start(hub, key_of(5, 1), 150, 200).has_value());
}

// Abandon-on-timeout has two race windows the fence must close (item 2 / §2.2),
// exercised here against the REAL register_batch / record_kernel_end /
// ledger_abandoned / is_ledgered and the REAL process-wide disable latch (removing
// either the latch check in register_batch or the insert in ledger_abandoned fails
// this).
TEST(DispatchHub, abandon_ledgers_proven_and_refuses_late_registration)
{
    reset_disable();
    auto hub = hub_t{};

    // (a) Pause AFTER record_kernel_end, BEFORE hand_off: the entry is out of the
    // map, so drain_for_teardown can no longer reach it. The abandon path must
    // ledger it explicitly (what hand_off_proven does on g_abandoned), or
    // correlation_id_finalize would force-retire it as a dangling id.
    const auto K = key_of(5, 7);
    ASSERT_TRUE(register_win(hub, K, mk_window(5, 100, 200), /*corr_id=*/77));
    auto _proven = end_start(hub, K, 150, 190);
    ASSERT_TRUE(_proven.has_value());
    EXPECT_EQ(_proven->correlation_id, 77u) << "record_kernel_end carries the correlation id";
    EXPECT_EQ(hub.pending_count(), 0u) << "the proven entry left the map";
    EXPECT_FALSE(hub.is_ledgered(77)) << "not ledgered until the abandon path acts";

    hub.ledger_abandoned(_proven->correlation_id);  // what hand_off_proven does on g_abandoned
    EXPECT_TRUE(hub.is_ledgered(77))
        << "a dropped proven is ledgered so finalize skips it instead of force-retiring";

    // (b) Pause AFTER eligibility, BEFORE register_batch: eligibility passed while
    // running; the fence abandon then sets the disable latch; the late register_batch
    // must be refused rather than land in the emptied hub.
    const auto K2 = key_of(6, 3);
    EXPECT_TRUE(hub.can_register_batch({K2})) << "eligible before the abandon";
    signal_less_disable_latch().store(true, std::memory_order_release);  // what the fence sets
    EXPECT_FALSE(register_win(hub, K2, mk_window(6, 100, 200)))
        << "register_batch rejects on the disable latch after abandon";
    EXPECT_EQ(hub.pending_count(), 0u) << "nothing landed in the emptied hub";

    reset_disable();
}

// Recycle identity: two live entries on ONE key (a recycled doorbell), each with
// its own window. record_kernel_end must return the payload AND correlation id
// whose window contains the START tick -- swapping old<->new must fail, not merely
// "some pair is returned".
TEST(DispatchHub, recycle_maps_each_start_to_its_own_window)
{
    reset_disable();
    auto       hub = hub_t{};
    const auto K   = key_of(5, 7);

    // Old owner: window [100,200), corr 10, payload 1. New owner reusing the same
    // doorbell: window [200,300), first_owner=false, corr 20, payload 2. The old
    // window is closed (t_close=200), so the recycled registration is admissible and
    // both entries are live at once in the multimap.
    {
        auto batch = std::vector<hub_t::registration>{};
        batch.emplace_back(reg_of(K, mk_window(5, 100, 200), /*corr_id=*/10, /*payload_id=*/1));
        auto evicted = std::vector<hub_t::leaked>{};
        ASSERT_TRUE(hub.register_batch(std::move(batch), evicted));
    }
    {
        auto batch = std::vector<hub_t::registration>{};
        batch.emplace_back(reg_of(K,
                                  mk_window(5, 200, kWindowOpen, /*first=*/false),
                                  /*corr_id=*/20,
                                  /*payload_id=*/2));
        auto evicted = std::vector<hub_t::leaked>{};
        ASSERT_TRUE(hub.register_batch(std::move(batch), evicted));
    }
    EXPECT_EQ(hub.pending_count(), 2u) << "both recycled owners live at once";

    // START 150 lies in the OLD window only -> old payload/correlation.
    auto _old = hub.record_kernel_end(K, std::optional<uint64_t>{150}, 180);
    ASSERT_TRUE(_old.has_value());
    EXPECT_EQ(_old->correlation_id, 10u) << "150 in [100,200) selects the old owner";
    EXPECT_EQ(_old->payload.id, 1u) << "must be the OLD payload, not the new one";

    // START 250 lies in the NEW window only -> new payload/correlation.
    auto _new = hub.record_kernel_end(K, std::optional<uint64_t>{250}, 280);
    ASSERT_TRUE(_new.has_value());
    EXPECT_EQ(_new->correlation_id, 20u) << "250 in [200,300) selects the new owner";
    EXPECT_EQ(_new->payload.id, 2u) << "must be the NEW payload, not the old one";
    EXPECT_EQ(hub.pending_count(), 0u);
}

// state bounds -- register/complete cycles return to zero; the ledger caps.
TEST(DispatchHub, state_bounds_and_ledger)
{
    reset_disable();
    {  // 10k open/register/complete cycles on one slot return pending_count to 0.
        auto hub = hub_t{};
        for(uint32_t i = 0; i < 10000; ++i)
        {
            auto k = key_of(5, i);
            ASSERT_TRUE(register_win(hub, k, mk_window(5, 100, 300)));
            ASSERT_TRUE(end_start(hub, k, 150, 200).has_value());
        }
        EXPECT_EQ(hub.pending_count(), 0u);
    }
    {  // the loss ledger drives the finalize skip; completed ids retire normally.
        auto hub = hub_t{};
        EXPECT_FALSE(hub.is_ledgered(11)) << "empty ledger excludes nothing";
        ASSERT_TRUE(register_win(hub, key_of(4, 1), mk_window(4, 100, 200), /*corr_id=*/11));
        ASSERT_TRUE(register_win(hub, key_of(4, 2), mk_window(4, 100), /*corr_id=*/22));
        ASSERT_TRUE(end_start(hub, key_of(4, 1), 150, 199).has_value());
        ASSERT_EQ(hub.quarantine_slot(0, 4).size(), 1u);
        EXPECT_FALSE(hub.is_ledgered(11)) << "completed -> retires normally";
        EXPECT_TRUE(hub.is_ledgered(22)) << "leaked -> finalize skips";
    }
}

// the process-wide disable latch refuses registration under m_mu, and
// drain_for_teardown() (part 2) leaks+ledgers everything live.
TEST(DispatchHub, disable_latch_refuses_registration_and_drain_ledgers)
{
    reset_disable();
    auto hub = hub_t{};
    ASSERT_TRUE(register_win(hub, key_of(5, 1), mk_window(5, 100), /*corr_id=*/77));
    // Part 2: drain while live -> stopping + leak+ledger.
    auto [leaked, stats] = hub.drain_for_teardown();
    EXPECT_EQ(stats.dispatches, 1u);
    EXPECT_TRUE(hub.is_ledgered(77)) << "drained entry is ledgered, never force-retired";
    // Part 3: with the latch set, register_batch refuses.
    signal_less_disable_latch().store(true);
    EXPECT_FALSE(register_win(hub, key_of(5, 2), mk_window(5, 300)))
        << "register_batch rejects on the disable latch";
    EXPECT_FALSE(hub.can_register_batch({key_of(5, 2)}));
    reset_disable();
}

// The result-vs-loss race under real concurrency: a prover racing a thread that
// keeps quarantining the slot. Each key resolves exactly once.
TEST(DispatchHub, concurrent_prove_vs_leak_resolves_each_key_once)
{
    reset_disable();
    constexpr uint32_t kCount = 512;

    auto hub = hub_t{};
    for(uint32_t i = 0; i < kCount; ++i)
        ASSERT_TRUE(register_win(hub, key_of(4, i), mk_window(4, 100)));  // sole first-owner
    EXPECT_EQ(tracked_payload::live.load(), static_cast<int>(kCount));

    auto proven_n = std::atomic<uint32_t>{0};
    auto leaked_n = std::atomic<uint32_t>{0};

    auto prover = std::thread{[&hub, &proven_n]() {
        for(uint32_t i = 0; i < kCount; ++i)
            if(end_nostart(hub, key_of(4, i), 300).has_value()) ++proven_n;  // startless, sole
    }};
    auto leaker = std::thread{[&hub, &leaked_n]() {
        for(uint32_t i = 0; i < kCount; ++i)
            leaked_n += hub.quarantine_slot(0, 4).size();
    }};
    prover.join();
    leaker.join();

    EXPECT_EQ(proven_n.load() + leaked_n.load(), kCount) << "exactly one winner per key";
    EXPECT_EQ(hub.pending_count(), 0u);
    EXPECT_EQ(tracked_payload::live.load(), 0) << "each payload destroyed once";
}

// Destroy (quarantine) runs concurrently with completion on other slots; run
// under TSan for the ordering. The hub must stay consistent and never deadlock.
TEST(DispatchHub, concurrent_quarantine_and_completion_stay_consistent)
{
    reset_disable();
    constexpr uint32_t kSlots = 64;

    auto hub = hub_t{};
    for(uint32_t s = 0; s < kSlots; ++s)
        ASSERT_TRUE(register_win(hub, key_of(s, 1), mk_window(s, 100), /*corr_id=*/s));

    auto destroyer = std::thread{[&hub]() {
        for(uint32_t s = 0; s < kSlots; ++s)
            hub.quarantine_slot(0, s);
    }};
    auto prover    = std::atomic<uint32_t>{0};
    auto reader    = std::thread{[&hub, &prover]() {
        for(uint32_t s = 0; s < kSlots; ++s)
            if(end_nostart(hub, key_of(s, 1), 300).has_value()) ++prover;
    }};
    destroyer.join();
    reader.join();

    EXPECT_EQ(hub.pending_count(), 0u);
    EXPECT_LE(prover.load(), kSlots);
    EXPECT_EQ(tracked_payload::live.load(), 0);
}

// TSan: the owner_window cross-thread publish. A destroy thread runs
// close_window (plain gc_deadline_ns write, THEN t_close release-store) racing a
// processor thread that record_kernel_end's (t_close acquire-load) and then
// gc_closed_windows (reads gc_deadline only after observing a closed t_close). The
// release/acquire on t_close must publish the plain gc_deadline with no data race.
TEST(DispatchHub, concurrent_close_publish_and_resolve)
{
    reset_disable();
    constexpr uint32_t N   = 256;
    auto               dbm = DoorbellMap{};
    auto               hub = hub_t{};
    for(uint32_t i = 0; i < N; ++i)
    {
        auto o = dbm.open_window(0, rocprofiler_queue_id_t{i + 1}, i, 100);
        ASSERT_TRUE(o.w);
        ASSERT_TRUE(register_win(hub, key_of(i, 1), o.w, /*corr_id=*/i + 1));
    }
    auto destroyer = std::thread{[&]() {
        for(uint32_t i = 0; i < N; ++i)
            dbm.close_window(rocprofiler_queue_id_t{i + 1}, /*t_close=*/200, /*gc_deadline=*/1000);
    }};
    auto processor = std::thread{[&]() {
        for(uint32_t i = 0; i < N; ++i)
            end_start(hub, key_of(i, 1), /*start=*/150, /*end=*/210);  // reads t_close acquire
        hub.gc_closed_windows(/*now_ns=*/2000);                        // reads gc_deadline
    }};
    destroyer.join();
    processor.join();
    SUCCEED() << "clean TSan report is the assertion";
}

// TSan: the disable-drain vs register race. A registrar thread
// runs register_batch (checks the disable latch, inserts under m_mu) racing a
// disabler thread that sets the latch (release) then drain_for_teardown (leaks
// under m_mu). No race on m_entries (one mutex) or the latch (atomic); and every
// registration is either drained-and-ledgered or refused -- none left un-ledgered.
TEST(DispatchHub, concurrent_disable_drain_vs_register)
{
    reset_disable();
    constexpr uint32_t N         = 512;
    auto               hub       = hub_t{};
    auto               registrar = std::thread{[&]() {
        for(uint32_t i = 0; i < N; ++i)
            register_win(hub, key_of(1, i), mk_window(1, 100), /*corr_id=*/i + 1);
    }};
    auto               disabler  = std::thread{[&]() {
        signal_less_disable_latch().store(true, std::memory_order_release);
        hub.drain_for_teardown();
    }};
    registrar.join();
    disabler.join();
    // A registration that raced past the latch is swept by the drain; one that
    // acquired m_mu after it is refused -- nothing survives un-drained.
    EXPECT_EQ(hub.pending_count(), 0u);
    reset_disable();
}

// submission-gate model: a faithful stand-in for the production
// hand_off_proven / flush_deferred_completions / abandon discipline, which cannot
// be unit-tested directly (they need the task group and the reader). A submitter
// takes the gate SHARED over its whole body -- the g_abandoned check, the (failed)
// submit, and the deferred append -- while the abandoner takes it EXCLUSIVE to
// latch g_abandoned, then flushes. The gate closes the check-then-append window:
// no submitter can append after the abandoner flushed. The bare-atomic variant
// (no gate) leaks -- which this test demonstrates -- so it is fails-without-fix.
namespace
{
struct fence_model
{
    std::shared_mutex gate;
    std::atomic<bool> abandoned{false};
    std::mutex        mu;
    std::deque<int>   deferred;
    std::atomic<bool> flush_done{false};
    std::atomic<int>  appended_after_flush{0};  // the leak the gate prevents
    std::atomic<int>  dropped{0};

    // Every call "fails to submit" and defers, so the race is always exercised.
    // hold_us models a submitter descheduled between the abandon check and append.
    void submit(int id, bool use_gate, int hold_us)
    {
        if(use_gate)
        {
            auto lk = std::shared_lock<std::shared_mutex>{gate};
            if(abandoned.load(std::memory_order_acquire))
            {
                ++dropped;
                return;
            }
            std::this_thread::sleep_for(std::chrono::microseconds{hold_us});
            do_append(id);
        }
        else
        {
            if(abandoned.load(std::memory_order_acquire))
            {
                ++dropped;
                return;
            }
            std::this_thread::sleep_for(std::chrono::microseconds{hold_us});
            do_append(id);
        }
    }

    void abandon_and_flush(bool use_gate)
    {
        if(use_gate)
        {
            auto lk = std::unique_lock<std::shared_mutex>{gate};
            abandoned.store(true, std::memory_order_release);
        }
        else
        {
            abandoned.store(true, std::memory_order_release);
        }
        std::deque<int> taken;
        {
            auto lk = std::lock_guard<std::mutex>{mu};
            taken.swap(deferred);
        }
        flush_done.store(true, std::memory_order_release);
    }

private:
    void do_append(int id)
    {
        auto lk = std::lock_guard<std::mutex>{mu};
        if(flush_done.load(std::memory_order_acquire)) ++appended_after_flush;
        deferred.push_back(id);
    }
};

int
run_fence_race(bool use_gate)
{
    constexpr int kSubmitters = 64;
    auto          m           = fence_model{};
    auto          threads     = std::vector<std::thread>{};
    threads.reserve(kSubmitters);
    for(int i = 0; i < kSubmitters; ++i)
        threads.emplace_back([&m, i, use_gate]() { m.submit(i, use_gate, /*hold_us=*/3000); });
    // The abandoner runs while the submitters are mid-window (checked, sleeping).
    std::this_thread::sleep_for(std::chrono::microseconds{500});
    m.abandon_and_flush(use_gate);
    for(auto& t : threads)
        t.join();
    // Under the gate this must be 0; every submitter is either flushed (its append
    // preceded the exclusive) or dropped (it saw the latch). Any leftover in
    // `deferred` is an append the flush missed -- a leaked completion.
    return m.appended_after_flush.load() + static_cast<int>(m.deferred.size());
}
}  // namespace

// The gate closes the abandon-vs-submit race: no completion is appended after the
// abandoner flushed. Run under TSan for the shared/exclusive ordering.
TEST(SubmitGate, abandon_never_races_a_submit)
{
    EXPECT_EQ(run_fence_race(/*use_gate=*/true), 0)
        << "under the shared/exclusive gate, no submit lands after the abandon flush";
}

// Fails-without-fix: the bare-atomic variant (no gate) DOES let a submit land
// after the flush -- the leak the submission gate exists to prevent.
TEST(SubmitGate, bare_atomic_leaks_a_post_flush_submit)
{
    EXPECT_GT(run_fence_race(/*use_gate=*/false), 0)
        << "without the gate a descheduled submitter appends after flush -- a leak";
}

// Live doorbell-owner registry: injectivity and counts. Unchanged by
// time-as-generation.
TEST(OwnerRegistry, injectivity_and_quarantine)
{
    reset_disable();
    {  // An unknown slot is not injective: an unresolved doorbell is never unique.
        auto reg = OwnerRegistry{};
        EXPECT_FALSE(reg.slot_uniquely_owned(/*gpu_id=*/0, /*slot=*/4100));
    }
    {  // A sole owner is unique.
        auto reg = OwnerRegistry{};
        EXPECT_EQ(reg.add_queue(/*token=*/1, /*gpu=*/0, /*slot=*/uint32_t{40}),
                  OwnerRegistry::add_result::sole_owner);
        EXPECT_TRUE(reg.slot_uniquely_owned(0, 40));
        EXPECT_EQ(reg.owners_of(0, 40), 1u);
        EXPECT_EQ(reg.live_queues(), 1u);
    }
    {  // A second live owner collides; the caller quarantines the slot in the hub.
        auto reg = OwnerRegistry{};
        auto hub = hub_t{};
        ASSERT_TRUE(register_win(hub, key_of(40, 1), mk_window(40, 100)));
        ASSERT_TRUE(register_win(hub, key_of(40, 2), mk_window(40, 100)));
        EXPECT_EQ(reg.add_queue(1, 0, uint32_t{40}), OwnerRegistry::add_result::sole_owner);
        EXPECT_EQ(reg.add_queue(2, 0, uint32_t{40}), OwnerRegistry::add_result::collision);
        EXPECT_FALSE(reg.slot_uniquely_owned(0, 40));
        auto stranded = hub.quarantine_slot(0, 40);
        EXPECT_EQ(stranded.size(), 2u);
        EXPECT_FALSE(hub.can_register_batch({key_of(40, 3)}));
    }
    {  // Quarantine outlives the collision even after a co-owner dies.
        auto reg = OwnerRegistry{};
        auto hub = hub_t{};
        reg.add_queue(1, 0, uint32_t{40});
        reg.add_queue(2, 0, uint32_t{40});
        hub.quarantine_slot(0, 40);
        reg.remove_queue(2);
        EXPECT_TRUE(reg.slot_uniquely_owned(0, 40)) << "ownership looks clean again";
        EXPECT_FALSE(hub.can_register_batch({key_of(40, 9)})) << "but slot stays unusable";
    }
    {  // A pre-session queue participates in injectivity, so a later queue collides.
        auto reg = OwnerRegistry{};
        EXPECT_EQ(reg.add_queue(/*pre-session*/ 1, 0, uint32_t{7}),
                  OwnerRegistry::add_result::sole_owner);
        EXPECT_EQ(reg.add_queue(/*post-session*/ 2, 0, uint32_t{7}),
                  OwnerRegistry::add_result::collision);
        EXPECT_FALSE(reg.slot_uniquely_owned(0, 7));
    }
}

// Registry resolution, GPU scoping, and exact reference counting (unchanged).
TEST(OwnerRegistry, resolution_scope_and_counts)
{
    {  // An unresolved queue disables its whole GPU until it dies; others unaffected.
        auto reg = OwnerRegistry{};
        reg.add_queue(1, 0, uint32_t{40});
        reg.add_queue(2, 1, uint32_t{50});
        EXPECT_TRUE(reg.slot_uniquely_owned(0, 40));
        EXPECT_EQ(reg.add_queue(3, 0, std::nullopt), OwnerRegistry::add_result::slot_unknown);
        EXPECT_EQ(reg.unresolved_queues(0), 1u);
        EXPECT_FALSE(reg.slot_uniquely_owned(0, 40)) << "this GPU is out";
        EXPECT_TRUE(reg.slot_uniquely_owned(1, 50)) << "other GPU unaffected";
        reg.remove_queue(3);
        EXPECT_TRUE(reg.slot_uniquely_owned(0, 40));
    }
    {  // Slots are scoped per GPU: the same slot on two GPUs is not a collision.
        auto reg = OwnerRegistry{};
        EXPECT_EQ(reg.add_queue(1, 0, uint32_t{40}), OwnerRegistry::add_result::sole_owner);
        EXPECT_EQ(reg.add_queue(2, 1, uint32_t{40}), OwnerRegistry::add_result::sole_owner);
        EXPECT_TRUE(reg.slot_uniquely_owned(0, 40));
        EXPECT_TRUE(reg.slot_uniquely_owned(1, 40));
    }
    {  // remove releases ownership; re-registering a token replaces, not doubles.
        auto reg = OwnerRegistry{};
        reg.add_queue(1, 0, uint32_t{40});
        reg.add_queue(1, 0, uint32_t{40});
        EXPECT_EQ(reg.owners_of(0, 40), 1u);
        reg.add_queue(1, 0, uint32_t{41});
        EXPECT_EQ(reg.owners_of(0, 40), 0u);
        EXPECT_EQ(reg.owners_of(0, 41), 1u);
        reg.remove_queue(1);
        EXPECT_EQ(reg.owners_of(0, 41), 0u);
        reg.remove_queue(1);  // idempotent
        EXPECT_EQ(reg.live_queues(), 0u);
    }
}

namespace
{
hub_t&
forked_hub()
{
    static auto _v = hub_t{};
    return _v;
}

OwnerRegistry&
forked_registry()
{
    static auto _v = OwnerRegistry{};
    return _v;
}

// Every entry point a child must call without touching an inherited mutex.
bool
all_entry_points_short_circuit()
{
    bool ok = true;
    ok      = ok && !register_win(forked_hub(), key_of(40, 99), mk_window(40, 100));
    ok      = ok && !forked_hub().record_kernel_end(key_of(40, 1), std::nullopt, 1).has_value();
    ok      = ok && forked_hub().pending_count() == 0;
    ok      = ok && !forked_hub().is_ledgered(500);
    ok      = ok && forked_hub().mode() == session_mode::child_stale;
    ok      = ok && forked_hub().quarantine_slot(0, 40).empty();
    ok      = ok && forked_hub().drain_for_teardown().first.empty();
    ok      = ok && !forked_registry().slot_uniquely_owned(0, 40);
    ok      = ok && !forked_registry().slot_of(1).has_value();
    return ok;
}
}  // namespace

// A REAL fork: the child abandons inherited state and exits normally, so the
// inherited statics' destructors run over abandoned state without a double-free.
TEST(fork_safety, forked_child_short_circuits_and_survives_normal_exit)
{
    reset_disable();
    auto parent_only = hub_t{};
    ASSERT_TRUE(register_win(parent_only, key_of(9, 1), mk_window(9, 100)));

    pid_t pid = fork();
    ASSERT_NE(pid, -1);

    if(pid == 0)
    {
        forked_hub().abandon_in_child();
        forked_registry().abandon_in_child();
        const bool ok = all_entry_points_short_circuit();
        std::exit(ok ? 0 : 2);
    }

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status)) << "child did not exit normally";
    if(WIFEXITED(status))
    {
        EXPECT_EQ(WEXITSTATUS(status), 0) << "child did not short-circuit";
    }

    EXPECT_EQ(parent_only.pending_count(), 1u);
    EXPECT_TRUE(end_nostart(parent_only, key_of(9, 1), 300).has_value());
}

// ---------------------------------------------------------------------------
// D9 (F7) per-GPU cap ARITHMETIC seam. hub_cap_need is the pure branch the review
// flagged: it must never form `live + batch` (overflow) or subtract into a wrap on
// a healthy GPU (the underflow that would evict everything). Spec 1191-1202,1227-1230.
// ---------------------------------------------------------------------------
TEST(HubCapArithmetic, checked_shortfall_never_wraps)
{
    // cap 1, empty GPU: a one-entry batch fits; a two-entry batch needs one victim.
    EXPECT_EQ(hub_cap_need(/*live=*/0, /*batch=*/1, /*cap=*/1), 0u);
    EXPECT_EQ(hub_cap_need(0, 2, 1), 1u);

    // A batch LARGER than the whole cap must report the exact shortfall, never wrap
    // into "no shortfall" (the `live + batch - cap` overflow bug).
    EXPECT_EQ(hub_cap_need(0, 5, 2), 3u);

    // live == cap exactly with a one-entry batch takes the shortfall path.
    EXPECT_EQ(hub_cap_need(1, 1, 1), 1u);
    EXPECT_EQ(hub_cap_need(10, 1, 10), 1u);

    // live < cap on a healthy GPU takes ZERO victims -- the underflow regression the
    // `live + batch - cap` form causes (it would wrap to a huge need here).
    EXPECT_EQ(hub_cap_need(1, 1, 10), 0u);
    EXPECT_EQ(hub_cap_need(5, 1, 10), 0u);

    // live > cap: exact overflow-free shortfall.
    EXPECT_EQ(hub_cap_need(12, 3, 10), 5u);
}

// ---------------------------------------------------------------------------
// D2 (F4) GC gate predicate + counter protocol. The predicate is pure (all four
// combos); the reserve->publish counter balance is exercised through the real
// copy/spill helpers. Spec 1324-1329.
// ---------------------------------------------------------------------------
TEST(GcGate, predicate_truth_table)
{
    EXPECT_TRUE(gc_gate_open(/*inflight=*/0, /*pipe_empty=*/true));  // only open state
    EXPECT_FALSE(gc_gate_open(1, true)) << "copied-but-unpublished record in flight";
    EXPECT_FALSE(gc_gate_open(0, false)) << "pipe not drained";
    EXPECT_FALSE(gc_gate_open(3, false));
}

// The reader-published in-flight counter is reserved before a record leaves the
// ring and discharged at publish, so a full copy->publish cycle nets to 0 and the
// gate reopens; while a batch sits unpublished in overflow the counter stays > 0.
TEST(GcGate, spill_publish_balances_the_inflight_counter)
{
    auto pipe     = record_pipe<4>{};
    auto overflow = std::deque<record_batch>{};
    auto inflight = std::atomic<uint64_t>{0};

    // Two batches parked in overflow, reserved by the (simulated) copy: 3 + 2 = 5.
    auto b0 = record_batch{};
    b0.records.resize(3);
    auto b1 = record_batch{};
    b1.records.resize(2);
    overflow.emplace_back(std::move(b0));
    overflow.emplace_back(std::move(b1));
    inflight.fetch_add(5, std::memory_order_release);  // reserve-per-region analogue

    EXPECT_FALSE(gc_gate_open(inflight.load(std::memory_order_acquire), pipe.empty()))
        << "GC gated while copied records sit unpublished in overflow";

    ASSERT_TRUE(spill_overflow_to_pipe(pipe, overflow, &inflight));
    EXPECT_EQ(inflight.load(std::memory_order_acquire), 0u) << "every reservation discharged";
    EXPECT_FALSE(pipe.empty()) << "batches are now visible to the processor";

    // Drain the pipe; only then is the gate open.
    while(pipe.peek() != nullptr)
        pipe.pop();
    EXPECT_TRUE(gc_gate_open(inflight.load(std::memory_order_acquire), pipe.empty()));
}

// ---------------------------------------------------------------------------
// D8 shared env parse contract (env_long_in_range): reject-and-default, never
// clamp; one verdict per case. Spec 1018-1040. Uses a private env var so no other
// accessor's cached static-const read is disturbed.
// ---------------------------------------------------------------------------
TEST(EnvLongInRange, reject_and_default_never_clamp)
{
    const char* var = "ROCPROFILER_KFD_DLOG_TEST_ENVLONG";
    auto        set = [&](const char* v) { ::setenv(var, v, 1); };

    ::unsetenv(var);
    EXPECT_FALSE(env_long_in_range(var, 0, 1000).has_value())
        << "unset -> nullopt (caller default)";

    set("100");
    EXPECT_EQ(env_long_in_range(var, 0, 1000).value_or(-1), 100);  // valid

    set("0");
    EXPECT_EQ(env_long_in_range(var, 0, 1000).value_or(-1), 0) << "0 is in-range here, honored";

    set("100junk");
    EXPECT_FALSE(env_long_in_range(var, 0, 1000).has_value()) << "trailing junk rejected";

    set("100 ");
    EXPECT_FALSE(env_long_in_range(var, 0, 1000).has_value()) << "trailing whitespace rejected";

    set("-5");
    EXPECT_FALSE(env_long_in_range(var, 0, 1000).has_value()) << "below range rejected";

    set("1001");
    EXPECT_FALSE(env_long_in_range(var, 0, 1000).has_value())
        << "above range rejected, not clamped";

    set("4294967296");  // > INT_MAX: must not be narrowed into range
    EXPECT_FALSE(env_long_in_range(var, 0, 1000).has_value())
        << "above INT_MAX rejected, not narrowed";

    set("1000");
    EXPECT_EQ(env_long_in_range(var, 0, 1000).value_or(-1), 1000) << "upper endpoint accepted";

    ::unsetenv(var);
}

// The strict opt-in switch: ONLY an explicit true value enables. Everything else
// -- unset, empty, a false word, a typo, a number -- is DISABLED (fail-closed),
// and nothing throws or fatals.
TEST(EnvBoolOptIn, only_explicit_true_enables)
{
    const char* var = "ROCPROFILER_KFD_DLOG_TEST_ENVBOOL";
    auto        set = [&](const char* v) { ::setenv(var, v, 1); };

    ::unsetenv(var);
    EXPECT_FALSE(env_bool_opt_in(var)) << "unset -> off";

    for(const char* _yes : {"1", "true", "yes", "on"})
    {
        set(_yes);
        EXPECT_TRUE(env_bool_opt_in(var)) << _yes << " enables";
    }
    for(const char* _no :
        {"0", "false", "no", "off", "", "flase", "2", "TRUE", "99999999999999999"})
    {
        set(_no);
        EXPECT_FALSE(env_bool_opt_in(var)) << "'" << _no << "' must not enable";
    }

    ::unsetenv(var);
}
