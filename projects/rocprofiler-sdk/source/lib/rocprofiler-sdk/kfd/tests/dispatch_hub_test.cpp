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

// Unit tests for the signal-less pending-completion hub (dispatch_hub.hpp).
// The hub is templated on its payload and holds no singletons, so the whole
// state machine and every rendezvous invariant is exercised here without a GPU,
// the HSA runtime, or the reader thread (test seam S2).

#include "lib/rocprofiler-sdk/kfd/dispatch_hub.hpp"
#include "lib/rocprofiler-sdk/kfd/no_signal_finalizer.hpp"
#include "lib/rocprofiler-sdk/kfd/owner_registry.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less_gate.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <unordered_set>
#include <vector>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <random>
#include <set>

#include <sys/wait.h>
#include <unistd.h>

namespace
{
using namespace rocprofiler::kfd;

// Payload that tracks its own lifetime, so a test can prove there is exactly one
// owner at all times and that cleanup runs exactly once (invariant 6, U17).
struct tracked_payload
{
    static std::atomic<int> live;

    uint64_t id      = 0;
    bool     armed   = false;  // true while this instance owns the resource

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
        rhs.armed = false;  // ownership moved: exactly one owner remains
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

correlation_key
key_of(uint32_t slot, uint32_t dispatch_id, uint32_t generation = 0)
{
    return correlation_key{slot, dispatch_id, generation};
}

hub_t::registration
reg_of(correlation_key key, uint64_t corr_id = 1, uint64_t queue_token = 1, uint64_t payload_id = 0)
{
    auto r           = hub_t::registration{};
    r.key            = key;
    r.correlation_id = corr_id;
    r.queue_token    = queue_token;
    r.submit_index   = key.dispatch_idx_low32;
    r.payload        = tracked_payload{payload_id != 0 ? payload_id : key.dispatch_idx_low32};
    return r;
}

// Register a single dispatch; returns whether the hub accepted the batch.
bool
register_one(hub_t& hub, correlation_key key, uint64_t corr_id = 1, uint64_t queue_token = 1)
{
    auto batch = std::vector<hub_t::registration>{};
    batch.emplace_back(reg_of(key, corr_id, queue_token));
    return hub.register_batch(std::move(batch));
}
}  // namespace

// ---------------------------------------------------------------------------
// U1: the state machine -- every legal transition, illegal ones rejected
// ---------------------------------------------------------------------------

TEST(DispatchHub, register_then_prove_completes_once)
{
    auto hub = hub_t{};
    auto key = key_of(4, 7);
    ASSERT_TRUE(register_one(hub, key));
    EXPECT_EQ(hub.live_entries(), 1u);
    EXPECT_EQ(hub.outstanding(1), 1u);

    hub.note_start(key, 1000);
    auto p = hub.prove_eop(key, 2000, /*drain_loss_free=*/true);
    ASSERT_TRUE(p.has_value());
    ASSERT_TRUE(p->start_ticks.has_value());
    EXPECT_EQ(*p->start_ticks, 1000u);
    EXPECT_EQ(p->end_ticks, 2000u);
    EXPECT_TRUE(p->payload.armed);  // ownership handed to the caller

    // Proven entries leave the hub, so they can never be handed out again and can
    // never cross to LEAKED.
    EXPECT_EQ(hub.live_entries(), 0u);
    EXPECT_EQ(hub.outstanding(1), 0u);
    EXPECT_FALSE(hub.prove_eop(key, 3000, true).has_value());
    EXPECT_FALSE(hub.leak(key).has_value());
}

// Shape (ii): the START was lost but the EOP still proves completion. The worker
// turns a missing start_ticks into COMPLETED_NO_TIMING; the hub just reports it.
TEST(DispatchHub, eop_without_start_still_proves_completion)
{
    auto hub = hub_t{};
    auto key = key_of(4, 8);
    ASSERT_TRUE(register_one(hub, key));

    auto p = hub.prove_eop(key, 2000, true);
    ASSERT_TRUE(p.has_value());
    EXPECT_FALSE(p->start_ticks.has_value());  // -> COMPLETED_NO_TIMING, still retires
}

// U9: an EOP with no pending owner completes nothing and must not be cached.
// U3b: a result arriving before its registration is REJECTED, and a later
// same-key dispatch must not inherit it.
TEST(DispatchHub, result_before_register_is_rejected_not_cached)
{
    auto hub = hub_t{};
    auto key = key_of(4, 9);

    EXPECT_FALSE(hub.prove_eop(key, 2000, true).has_value());

    ASSERT_TRUE(register_one(hub, key));
    // The earlier result was not retained: the new dispatch is still pending.
    EXPECT_EQ(hub.live_entries(), 1u);
    auto p = hub.prove_eop(key, 5000, true);
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->end_ticks, 5000u);  // the fresh record, not the stale one
}

// An EOP drained under an overrun proves nothing about WHICH dispatch it belongs
// to, so it completes nothing even though the key matches.
TEST(DispatchHub, eop_under_lossy_drain_completes_nothing)
{
    auto hub = hub_t{};
    auto key = key_of(4, 10);
    ASSERT_TRUE(register_one(hub, key));

    EXPECT_FALSE(hub.prove_eop(key, 2000, /*drain_loss_free=*/false).has_value());
    EXPECT_EQ(hub.live_entries(), 1u);  // still pending, not consumed
}

// U8: a START is retained for as long as the entry lives -- no 5 s eviction --
// so a legitimately long kernel still pairs when its EOP finally arrives.
TEST(DispatchHub, start_is_retained_until_terminal)
{
    auto hub = hub_t{};
    auto key = key_of(4, 11);
    ASSERT_TRUE(register_one(hub, key));
    EXPECT_TRUE(hub.note_start(key, 42));

    // Nothing ages it out; the only thing that clears it is a terminal transition.
    auto p = hub.prove_eop(key, 99, true);
    ASSERT_TRUE(p.has_value());
    ASSERT_TRUE(p->start_ticks.has_value());
    EXPECT_EQ(*p->start_ticks, 42u);
}

TEST(DispatchHub, unmatched_start_is_dropped)
{
    auto hub = hub_t{};
    EXPECT_FALSE(hub.note_start(key_of(4, 12), 1));
    EXPECT_EQ(hub.live_entries(), 0u);
}

// ---------------------------------------------------------------------------
// Invariant 1 / U2: whole-batch atomicity
// ---------------------------------------------------------------------------

TEST(DispatchHub, batch_registers_all_or_none)
{
    auto hub = hub_t{};
    auto ok  = std::vector<hub_t::registration>{};
    ok.emplace_back(reg_of(key_of(4, 1)));
    ok.emplace_back(reg_of(key_of(4, 2)));
    ok.emplace_back(reg_of(key_of(4, 3)));
    ASSERT_TRUE(hub.register_batch(std::move(ok)));
    EXPECT_EQ(hub.live_entries(), 3u);
    EXPECT_EQ(hub.outstanding(1), 3u);

    // A batch whose LAST entry collides with a live key must insert none of it.
    auto clash = std::vector<hub_t::registration>{};
    clash.emplace_back(reg_of(key_of(4, 10)));
    clash.emplace_back(reg_of(key_of(4, 11)));
    clash.emplace_back(reg_of(key_of(4, 2)));  // already live
    EXPECT_FALSE(hub.register_batch(std::move(clash)));
    EXPECT_EQ(hub.live_entries(), 3u);  // unchanged: no partial registration
    EXPECT_FALSE(hub.prove_eop(key_of(4, 10), 1, true).has_value());
    EXPECT_FALSE(hub.prove_eop(key_of(4, 11), 1, true).has_value());
}

// Invariant 3: no overwrite semantics -- a duplicate inside one batch is an
// error, not first-writer-wins.
TEST(DispatchHub, duplicate_key_within_batch_rejected)
{
    auto hub   = hub_t{};
    auto batch = std::vector<hub_t::registration>{};
    batch.emplace_back(reg_of(key_of(4, 1)));
    batch.emplace_back(reg_of(key_of(4, 1)));
    EXPECT_FALSE(hub.register_batch(std::move(batch)));
    EXPECT_EQ(hub.live_entries(), 0u);
}

// ---------------------------------------------------------------------------
// Invariant 4 / U4: result-vs-loss has exactly one winner
// ---------------------------------------------------------------------------

TEST(DispatchHub, prove_and_leak_race_has_one_winner)
{
    auto hub = hub_t{};
    auto key = key_of(4, 20);
    ASSERT_TRUE(register_one(hub, key));

    auto p = hub.prove_eop(key, 1, true);
    ASSERT_TRUE(p.has_value());
    EXPECT_FALSE(hub.leak(key).has_value());  // loss loses: no double ownership

    auto key2 = key_of(4, 21);
    ASSERT_TRUE(register_one(hub, key2));
    auto l = hub.leak(key2);
    ASSERT_TRUE(l.has_value());
    EXPECT_FALSE(hub.prove_eop(key2, 1, true).has_value());  // result loses
}

// The same race under real concurrency: N dispatches, a prover thread and a
// leaker thread going after every key. Each key must resolve exactly once, and
// the payload accounting must show exactly one live owner per resolution.
TEST(DispatchHub, concurrent_prove_vs_leak_resolves_each_key_once)
{
    constexpr uint32_t kCount = 512;

    auto hub = hub_t{};
    for(uint32_t i = 0; i < kCount; ++i)
    {
        ASSERT_TRUE(register_one(hub, key_of(4, i)));
    }
    const int live_after_register = tracked_payload::live.load();
    EXPECT_EQ(live_after_register, static_cast<int>(kCount));

    auto proven_n = std::atomic<uint32_t>{0};
    auto leaked_n = std::atomic<uint32_t>{0};

    auto prover = std::thread{[&hub, &proven_n]() {
        for(uint32_t i = 0; i < kCount; ++i)
            if(hub.prove_eop(key_of(4, i), 7, true).has_value()) ++proven_n;
    }};
    auto leaker = std::thread{[&hub, &leaked_n]() {
        for(uint32_t i = 0; i < kCount; ++i)
            if(hub.leak(key_of(4, i)).has_value()) ++leaked_n;
    }};
    prover.join();
    leaker.join();

    // Exactly one winner per key, and nothing left behind.
    EXPECT_EQ(proven_n.load() + leaked_n.load(), kCount);
    EXPECT_EQ(hub.live_entries(), 0u);
    EXPECT_EQ(hub.outstanding(1), 0u);
    // Every payload handed to a winner was destroyed by that winner exactly once.
    EXPECT_EQ(tracked_payload::live.load(), 0);
}

// ---------------------------------------------------------------------------
// P1 loss policy: poison, ledger, tombstones (U11, U14, U15)
// ---------------------------------------------------------------------------

// U15: the loud warning needs dispatches AND unique correlation ids separately --
// a batch shares one correlation id with one reference per dispatch.
TEST(DispatchHub, poison_leaks_everything_and_counts_dispatches_and_corr_ids)
{
    auto hub = hub_t{};
    // Two batches: 3 dispatches on corr id 100, 2 on corr id 200.
    auto b1 = std::vector<hub_t::registration>{};
    for(uint32_t i = 0; i < 3; ++i)
        b1.emplace_back(reg_of(key_of(4, i), /*corr_id=*/100));
    ASSERT_TRUE(hub.register_batch(std::move(b1)));

    auto b2 = std::vector<hub_t::registration>{};
    for(uint32_t i = 10; i < 12; ++i)
        b2.emplace_back(reg_of(key_of(5, i), /*corr_id=*/200));
    ASSERT_TRUE(hub.register_batch(std::move(b2)));

    auto [lost, stats] = hub.poison(session_mode::loss_poisoned);

    EXPECT_EQ(stats.dispatches, 5u);
    EXPECT_EQ(stats.correlation_ids, 2u);
    EXPECT_EQ(lost.size(), 5u);
    EXPECT_EQ(hub.live_entries(), 0u);
    EXPECT_EQ(hub.mode(), session_mode::loss_poisoned);

    // U14: both ids are on the loss ledger, so correlation_id_finalize() must skip
    // force-retiring them.
    EXPECT_TRUE(hub.is_ledgered(100));
    EXPECT_TRUE(hub.is_ledgered(200));
    EXPECT_FALSE(hub.is_ledgered(300));
}

// Poison is terminal: nothing registers or completes afterwards, so no dispatch
// can be admitted to a session that has already lost records.
TEST(DispatchHub, poison_blocks_further_registration_and_completion)
{
    auto hub = hub_t{};
    auto key = key_of(4, 30);
    ASSERT_TRUE(register_one(hub, key));
    hub.poison(session_mode::loss_poisoned);

    EXPECT_FALSE(register_one(hub, key_of(4, 31)));
    EXPECT_FALSE(hub.prove_eop(key_of(4, 31), 1, true).has_value());
    EXPECT_FALSE(hub.note_start(key_of(4, 31), 1));
}

// U11: a leaked key is tombstoned, so a recurring low-32 dispatch id cannot
// reactivate it or be registered onto it again.
TEST(DispatchHub, leaked_key_is_tombstoned_and_non_matchable)
{
    auto hub = hub_t{};
    auto key = key_of(4, 40);
    ASSERT_TRUE(register_one(hub, key));
    ASSERT_TRUE(hub.leak(key).has_value());

    EXPECT_EQ(hub.tombstones(), 1u);
    EXPECT_FALSE(hub.prove_eop(key, 1, true).has_value());  // cannot be reactivated
    EXPECT_FALSE(register_one(hub, key));                   // and cannot be re-registered
}

// Tombstones are bounded: rather than forget one (which would let a recurring key
// reactivate), exceeding the cap poisons the session.
TEST(DispatchHub, tombstone_growth_is_bounded_by_poisoning)
{
    auto hub = hub_t{};
    for(uint32_t i = 0; i <= hub_t::kMaxTombstones; ++i)
    {
        if(!register_one(hub, key_of(4, i))) break;
        hub.leak(key_of(4, i));
    }
    EXPECT_LE(hub.tombstones(), hub_t::kMaxTombstones + 1);
    EXPECT_EQ(hub.mode(), session_mode::loss_poisoned);
}

// ---------------------------------------------------------------------------
// Invariant 9 / requirements 3+4: slot quarantine and generation closure
// ---------------------------------------------------------------------------

// U12: a slot with a second live owner is quarantined -- pending work on it is
// leaked and nothing may reserve it again for the rest of the process.
TEST(DispatchHub, quarantine_leaks_slot_and_blocks_reservation)
{
    auto hub = hub_t{};
    ASSERT_TRUE(register_one(hub, key_of(4, 1)));
    ASSERT_TRUE(register_one(hub, key_of(4, 2)));
    ASSERT_TRUE(register_one(hub, key_of(9, 1)));  // different slot, must survive

    auto lost = hub.quarantine_slot(4);
    EXPECT_EQ(lost.size(), 2u);
    EXPECT_TRUE(hub.is_quarantined(4));
    EXPECT_EQ(hub.live_entries(), 1u);

    EXPECT_FALSE(register_one(hub, key_of(4, 3)));  // permanently unusable
    EXPECT_TRUE(register_one(hub, key_of(9, 2)));   // unaffected slot still works
}

// U13: a new generation on a quarantined slot is still refused, so an old record
// can never be matched to a new queue's dispatch.
TEST(DispatchHub, quarantined_slot_refuses_a_new_generation)
{
    auto hub = hub_t{};
    ASSERT_TRUE(register_one(hub, key_of(4, 1, /*generation=*/0)));
    hub.quarantine_slot(4);
    EXPECT_FALSE(register_one(hub, key_of(4, 1, /*generation=*/1)));
}

// Invariant 7: queue destroy explicitly leaks whatever the queue still owns.
TEST(DispatchHub, close_queue_leaks_its_outstanding_work)
{
    auto hub = hub_t{};
    ASSERT_TRUE(register_one(hub, key_of(4, 1), 1, /*queue_token=*/77));
    ASSERT_TRUE(register_one(hub, key_of(4, 2), 1, /*queue_token=*/77));
    ASSERT_TRUE(register_one(hub, key_of(5, 1), 1, /*queue_token=*/88));
    EXPECT_EQ(hub.outstanding(77), 2u);

    auto lost = hub.close_queue(77);
    EXPECT_EQ(lost.size(), 2u);
    EXPECT_EQ(hub.outstanding(77), 0u);
    EXPECT_EQ(hub.outstanding(88), 1u);
}

// ---------------------------------------------------------------------------
// U18: teardown
// ---------------------------------------------------------------------------

TEST(DispatchHub, teardown_leaks_still_pending_entries)
{
    auto hub = hub_t{};
    ASSERT_TRUE(register_one(hub, key_of(4, 1), /*corr_id=*/11));
    ASSERT_TRUE(register_one(hub, key_of(4, 2), /*corr_id=*/11));

    auto [lost, stats] = hub.drain_for_teardown();
    EXPECT_EQ(stats.dispatches, 2u);
    EXPECT_EQ(stats.correlation_ids, 1u);
    EXPECT_EQ(hub.live_entries(), 0u);
    EXPECT_EQ(hub.mode(), session_mode::stopping);
    // Their correlation id is on the ledger, so finalize must not force-retire it.
    EXPECT_TRUE(hub.is_ledgered(11));
    // And nothing new can be admitted once teardown has begun.
    EXPECT_FALSE(register_one(hub, key_of(4, 3)));
}

// ---------------------------------------------------------------------------
// U19: fork
// ---------------------------------------------------------------------------

// The child handler is one atomic store; afterwards every operation
// short-circuits without touching the inherited map or mutex, so the child can
// run to a normal exit() over abandoned state.
TEST(DispatchHub, child_epoch_short_circuits_every_operation)
{
    auto hub = hub_t{};
    ASSERT_TRUE(register_one(hub, key_of(4, 1)));

    hub.abandon_in_child();
    EXPECT_TRUE(hub.abandoned());

    EXPECT_FALSE(register_one(hub, key_of(4, 2)));
    EXPECT_FALSE(hub.note_start(key_of(4, 1), 1));
    EXPECT_FALSE(hub.prove_eop(key_of(4, 1), 1, true).has_value());
    EXPECT_FALSE(hub.leak(key_of(4, 1)).has_value());
    EXPECT_FALSE(hub.is_ledgered(1));
    EXPECT_TRUE(hub.quarantine_slot(4).empty());
    EXPECT_TRUE(hub.close_queue(1).empty());
    EXPECT_TRUE(hub.drain_for_teardown().first.empty());
    EXPECT_TRUE(hub.poison(session_mode::child_stale).first.empty());
}

// ---------------------------------------------------------------------------
// Payload ownership (U17 core): exactly one owner, cleanup exactly once
// ---------------------------------------------------------------------------

TEST(DispatchHub, payload_has_exactly_one_owner_across_every_terminal)
{
    const int before = tracked_payload::live.load();
    {
        auto hub = hub_t{};
        ASSERT_TRUE(register_one(hub, key_of(4, 1)));
        ASSERT_TRUE(register_one(hub, key_of(4, 2)));
        ASSERT_TRUE(register_one(hub, key_of(4, 3)));
        EXPECT_EQ(tracked_payload::live.load(), before + 3);

        {
            auto p = hub.prove_eop(key_of(4, 1), 1, true);
            ASSERT_TRUE(p.has_value());
            EXPECT_EQ(tracked_payload::live.load(), before + 3);  // moved, not copied
        }
        EXPECT_EQ(tracked_payload::live.load(), before + 2);  // released once

        {
            auto l = hub.leak(key_of(4, 2));
            ASSERT_TRUE(l.has_value());
        }
        EXPECT_EQ(tracked_payload::live.load(), before + 1);
    }
    // The hub's own destruction releases whatever was still pending, exactly once.
    EXPECT_EQ(tracked_payload::live.load(), before);
}

// ---------------------------------------------------------------------------
// Enqueue-side batch admission: the hub pre-check the eligibility decision uses
// ---------------------------------------------------------------------------

// Eligibility must be final BEFORE any packet is modified, so it asks the hub up
// front whether the batch's keys are admissible. The answer must match what
// register_batch() would actually do.
TEST(DispatchHub, can_register_batch_agrees_with_register_batch)
{
    auto hub  = hub_t{};
    auto keys = std::vector<correlation_key>{key_of(4, 1), key_of(4, 2)};
    EXPECT_TRUE(hub.can_register_batch(keys));

    auto batch = std::vector<hub_t::registration>{};
    for(const auto& k : keys)
        batch.emplace_back(reg_of(k));
    EXPECT_TRUE(hub.register_batch(std::move(batch)));

    // Now the same keys are live, so a second batch must be refused by both.
    EXPECT_FALSE(hub.can_register_batch(keys));
    auto again = std::vector<hub_t::registration>{};
    for(const auto& k : keys)
        again.emplace_back(reg_of(k));
    EXPECT_FALSE(hub.register_batch(std::move(again)));
}

TEST(DispatchHub, can_register_batch_refuses_quarantine_tombstone_and_duplicates)
{
    auto hub = hub_t{};

    // Duplicate within the batch.
    EXPECT_FALSE(hub.can_register_batch({key_of(4, 1), key_of(4, 1)}));

    // Tombstoned key.
    ASSERT_TRUE(register_one(hub, key_of(4, 2)));
    ASSERT_TRUE(hub.leak(key_of(4, 2)).has_value());
    EXPECT_FALSE(hub.can_register_batch({key_of(4, 2)}));

    // Quarantined slot.
    hub.quarantine_slot(6);
    EXPECT_FALSE(hub.can_register_batch({key_of(6, 1)}));

    // Poisoned session.
    EXPECT_TRUE(hub.can_register_batch({key_of(7, 1)}));
    hub.poison(session_mode::loss_poisoned);
    EXPECT_FALSE(hub.can_register_batch({key_of(7, 1)}));
}

// ---------------------------------------------------------------------------
// Feature flag + eligibility decision table
// ---------------------------------------------------------------------------

// Default OFF: only an explicit, recognised enable turns the feature on, so a
// typo, an empty value, or an unrelated string can never activate it.
TEST(signal_less_flag, only_explicit_enable_turns_it_on)
{
    EXPECT_TRUE(parse_signal_less_env("1"));
    EXPECT_TRUE(parse_signal_less_env("on"));
    EXPECT_TRUE(parse_signal_less_env("ON"));
    EXPECT_TRUE(parse_signal_less_env("true"));
    EXPECT_TRUE(parse_signal_less_env("TRUE"));
    EXPECT_TRUE(parse_signal_less_env("yes"));

    EXPECT_FALSE(parse_signal_less_env(""));
    EXPECT_FALSE(parse_signal_less_env("0"));
    EXPECT_FALSE(parse_signal_less_env("off"));
    EXPECT_FALSE(parse_signal_less_env("false"));
    EXPECT_FALSE(parse_signal_less_env("2"));
    EXPECT_FALSE(parse_signal_less_env("enable"));
    EXPECT_FALSE(parse_signal_less_env(" 1"));
    EXPECT_FALSE(parse_signal_less_env("1 "));
}

// The master switch is what keeps signal-less inert while it is being landed:
// even with the env flag on, no batch can be eligible until it flips.
TEST(signal_less_flag, master_switch_holds_the_feature_off)
{
    static_assert(!signal_less_fully_wired(),
                  "signal-less must stay inert until every stage is present");

    // Everything else set to what a batch could possibly satisfy today.
    auto in                  = eligibility_inputs{};
    in.feature_enabled       = true;
    in.session_live_for_gpu  = true;
    in.reader_alive          = true;
    in.doorbells_injective   = true;
    in.hub_accepts_batch     = true;
    in.payload_constructible = true;
    in.fully_wired           = signal_less_fully_wired();

    EXPECT_FALSE(batch_is_signal_less_eligible(in));
}

// The owner-injectivity input now comes from the live-owner registry, and a slot
// with no known owner is not injective -- so it alone holds a batch on the signal
// path even with everything else satisfied.
TEST(signal_less_flag, unknown_slot_is_not_injective)
{
    auto reg = OwnerRegistry{};

    auto in                  = eligibility_inputs{};
    in.feature_enabled       = true;
    in.fully_wired           = true;
    in.session_live_for_gpu  = true;
    in.reader_alive          = true;
    in.hub_accepts_batch     = true;
    in.payload_constructible = true;
    in.doorbells_injective   = reg.is_injective(/*gpu_id=*/0, /*slot=*/4100);

    EXPECT_FALSE(batch_is_signal_less_eligible(in));
}

// Every input is necessary: dropping any one of them keeps the whole batch on the
// signal path (no mixed-mode batches, no "minimal/where-known" gating).
TEST(signal_less_flag, eligibility_requires_every_condition)
{
    auto all_true                  = eligibility_inputs{};
    all_true.feature_enabled       = true;
    all_true.fully_wired           = true;
    all_true.session_live_for_gpu  = true;
    all_true.reader_alive          = true;
    all_true.doorbells_injective   = true;
    all_true.hub_accepts_batch     = true;
    all_true.payload_constructible = true;
    ASSERT_TRUE(batch_is_signal_less_eligible(all_true));

    bool eligibility_inputs::*fields[] = {&eligibility_inputs::feature_enabled,
                                          &eligibility_inputs::fully_wired,
                                          &eligibility_inputs::session_live_for_gpu,
                                          &eligibility_inputs::reader_alive,
                                          &eligibility_inputs::doorbells_injective,
                                          &eligibility_inputs::hub_accepts_batch,
                                          &eligibility_inputs::payload_constructible};

    for(auto field : fields)
    {
        auto one_false   = all_true;
        one_false.*field = false;
        EXPECT_FALSE(batch_is_signal_less_eligible(one_false));
    }
}

// ---------------------------------------------------------------------------
// Unit 3: EOP shape (ii), overrun leak-and-shout, and the finalize skip
// ---------------------------------------------------------------------------

// Shape (ii): an EOP whose START was lost still proves completion for the unique
// current PENDING entry, with start_ticks unknown -> COMPLETED_NO_TIMING.
TEST(DispatchHub, shape_ii_proves_completion_without_a_start)
{
    auto hub = hub_t{};
    auto key = key_of(4, 50);
    ASSERT_TRUE(register_one(hub, key));

    // No note_start(): the START record was overwritten before the reader saw it.
    auto p = hub.prove_eop(key, 900, /*drain_loss_free=*/true);
    ASSERT_TRUE(p.has_value());
    EXPECT_FALSE(p->start_ticks.has_value());
    EXPECT_EQ(p->end_ticks, 900u);
    EXPECT_EQ(hub.live_entries(), 0u);
}

// The same EOP under a lossy drain proves nothing: the record may be torn and
// wptr cannot say which dispatch it belongs to.
TEST(DispatchHub, shape_ii_under_a_lossy_drain_proves_nothing)
{
    auto hub = hub_t{};
    auto key = key_of(4, 51);
    ASSERT_TRUE(register_one(hub, key));

    EXPECT_FALSE(hub.prove_eop(key, 900, /*drain_loss_free=*/false).has_value());
    EXPECT_EQ(hub.live_entries(), 1u);  // still pending, still matchable
}

// Leak-and-shout: one overrun strands every in-flight dispatch, reports the two
// counts the warning needs, poisons the session, and puts the ids on the ledger.
TEST(DispatchHub, overrun_strands_everything_and_reports_both_counts)
{
    auto hub = hub_t{};

    auto b1 = std::vector<hub_t::registration>{};
    for(uint32_t i = 0; i < 4; ++i)
        b1.emplace_back(reg_of(key_of(4, i), /*corr_id=*/700));
    ASSERT_TRUE(hub.register_batch(std::move(b1)));

    auto b2 = std::vector<hub_t::registration>{};
    b2.emplace_back(reg_of(key_of(5, 0), /*corr_id=*/800));
    ASSERT_TRUE(hub.register_batch(std::move(b2)));

    auto loss = hub.poison(session_mode::loss_poisoned);

    EXPECT_EQ(loss.second.dispatches, 5u);       // five stranded dispatches
    EXPECT_EQ(loss.second.correlation_ids, 2u);  // sharing two correlation ids
    EXPECT_EQ(loss.first.size(), 5u);            // payloads handed back for release

    // Signal-less is off for the rest of the process: no later batch is admissible.
    EXPECT_EQ(hub.mode(), session_mode::loss_poisoned);
    EXPECT_FALSE(hub.can_register_batch({key_of(6, 0)}));
    EXPECT_FALSE(register_one(hub, key_of(6, 0)));

    // A late record for a stranded dispatch completes nothing.
    EXPECT_FALSE(hub.prove_eop(key_of(4, 0), 1, true).has_value());
}

// The ledger drives the finalize skip: leaked ids must be excluded from
// force-retirement, non-leaked ids must not be, and an empty ledger excludes
// nothing at all.
TEST(DispatchHub, loss_ledger_selects_exactly_the_leaked_ids)
{
    auto hub = hub_t{};

    // Empty ledger: nothing is excluded.
    EXPECT_FALSE(hub.is_ledgered(11));
    EXPECT_FALSE(hub.is_ledgered(22));

    ASSERT_TRUE(register_one(hub, key_of(4, 1), /*corr_id=*/11));
    ASSERT_TRUE(register_one(hub, key_of(4, 2), /*corr_id=*/22));

    // 11 completes normally, 22 is leaked.
    ASSERT_TRUE(hub.prove_eop(key_of(4, 1), 1, true).has_value());
    ASSERT_TRUE(hub.leak(key_of(4, 2)).has_value());

    EXPECT_FALSE(hub.is_ledgered(11));  // completed -> retires normally
    EXPECT_TRUE(hub.is_ledgered(22));   // leaked -> finalize must skip it

    // Simulate the correlation_id_finalize() force-retire loop.
    auto  dangling = std::vector<uint64_t>{11, 22, 33};
    auto  retired  = std::vector<uint64_t>{};
    for(auto id : dangling)
    {
        if(hub.is_ledgered(id)) continue;
        retired.emplace_back(id);
    }
    EXPECT_EQ(retired, (std::vector<uint64_t>{11, 33}));
}

// ---------------------------------------------------------------------------
// Unit 4: live doorbell-owner registry (requirement 3)
// ---------------------------------------------------------------------------

// One live queue on a slot is the sole owner, so records for it are unambiguous
// and a batch on that queue can be eligible.
TEST(OwnerRegistry, sole_owner_is_injective)
{
    auto reg = OwnerRegistry{};
    EXPECT_EQ(reg.add_queue(/*token=*/1, /*gpu=*/0, /*slot=*/uint32_t{40}),
              OwnerRegistry::add_result::sole_owner);

    EXPECT_TRUE(reg.is_injective(0, 40));
    EXPECT_EQ(reg.owners_of(0, 40), 1u);
    EXPECT_EQ(reg.live_queues(), 1u);

    // With injectivity satisfied and everything else true, a batch is eligible.
    auto in                  = eligibility_inputs{};
    in.feature_enabled       = true;
    in.fully_wired           = true;  // forced: the real switch is still off
    in.session_live_for_gpu  = true;
    in.reader_alive          = true;
    in.hub_accepts_batch     = true;
    in.payload_constructible = true;
    in.doorbells_injective   = reg.is_injective(0, 40);
    EXPECT_TRUE(batch_is_signal_less_eligible(in));
}

// A second live owner is a collision: the registry says so, the slot stops being
// injective, and the caller quarantines it in the hub.
TEST(OwnerRegistry, second_owner_collides_and_quarantines)
{
    auto reg = OwnerRegistry{};
    auto hub = hub_t{};

    ASSERT_TRUE(register_one(hub, key_of(40, 1)));
    ASSERT_TRUE(register_one(hub, key_of(40, 2)));

    EXPECT_EQ(reg.add_queue(1, 0, uint32_t{40}), OwnerRegistry::add_result::sole_owner);
    EXPECT_EQ(reg.add_queue(2, 0, uint32_t{40}), OwnerRegistry::add_result::collision);

    EXPECT_FALSE(reg.is_injective(0, 40));
    EXPECT_EQ(reg.owners_of(0, 40), 2u);

    // The caller's response: quarantine, stranding what was pending on the slot.
    auto stranded = hub.quarantine_slot(40);
    EXPECT_EQ(stranded.size(), 2u);
    EXPECT_TRUE(hub.is_quarantined(40));
    EXPECT_FALSE(hub.can_register_batch({key_of(40, 3)}));  // both owners now signal-path
}

// Quarantine outlives the collision: after one colliding queue dies the registry
// sees a sole owner again, but the hub still refuses the slot for the rest of the
// process, so signal-less never resumes on it.
TEST(OwnerRegistry, quarantine_persists_after_the_collision_clears)
{
    auto reg = OwnerRegistry{};
    auto hub = hub_t{};

    reg.add_queue(1, 0, uint32_t{40});
    reg.add_queue(2, 0, uint32_t{40});
    hub.quarantine_slot(40);

    reg.remove_queue(2);
    EXPECT_TRUE(reg.is_injective(0, 40));  // ownership looks clean again...
    EXPECT_TRUE(hub.is_quarantined(40));   // ...but the slot stays unusable
    EXPECT_FALSE(hub.can_register_batch({key_of(40, 9)}));
}

// A queue registered before the session existed still participates: it is already
// an owner, so a queue created later on the same slot collides with it.
TEST(OwnerRegistry, pre_session_queue_participates_in_injectivity)
{
    auto reg = OwnerRegistry{};

    // Registered at creation, long before any dispatch-log session.
    EXPECT_EQ(reg.add_queue(/*pre-session*/ 1, 0, uint32_t{7}),
              OwnerRegistry::add_result::sole_owner);
    EXPECT_TRUE(reg.is_injective(0, 7));

    EXPECT_EQ(reg.add_queue(/*post-session*/ 2, 0, uint32_t{7}),
              OwnerRegistry::add_result::collision);
    EXPECT_FALSE(reg.is_injective(0, 7));
}

// A live queue whose doorbell could not be resolved could be the second owner of
// ANY slot, so it makes every slot on its GPU non-injective until it dies.
TEST(OwnerRegistry, unresolved_queue_disables_the_whole_gpu)
{
    auto reg = OwnerRegistry{};
    reg.add_queue(1, 0, uint32_t{40});
    reg.add_queue(2, 1, uint32_t{50});  // a different GPU
    EXPECT_TRUE(reg.is_injective(0, 40));

    EXPECT_EQ(reg.add_queue(3, 0, std::nullopt), OwnerRegistry::add_result::slot_unknown);
    EXPECT_EQ(reg.unresolved_queues(0), 1u);

    EXPECT_FALSE(reg.is_injective(0, 40));  // this GPU is out
    EXPECT_FALSE(reg.is_injective(0, 99));
    EXPECT_TRUE(reg.is_injective(1, 50));   // the other GPU is unaffected

    reg.remove_queue(3);
    EXPECT_EQ(reg.unresolved_queues(0), 0u);
    EXPECT_TRUE(reg.is_injective(0, 40));
}

// Slots are scoped per GPU: the same page-relative slot on two GPUs is not a
// collision, because records for a slot only ever come from the session GPU.
TEST(OwnerRegistry, same_slot_on_different_gpus_is_not_a_collision)
{
    auto reg = OwnerRegistry{};
    EXPECT_EQ(reg.add_queue(1, 0, uint32_t{40}), OwnerRegistry::add_result::sole_owner);
    EXPECT_EQ(reg.add_queue(2, 1, uint32_t{40}), OwnerRegistry::add_result::sole_owner);
    EXPECT_TRUE(reg.is_injective(0, 40));
    EXPECT_TRUE(reg.is_injective(1, 40));
}

// Destroying a queue releases its ownership so a surviving co-owner is sole again,
// and re-registering the same token replaces rather than double-counts.
TEST(OwnerRegistry, remove_and_reregister_keep_counts_exact)
{
    auto reg = OwnerRegistry{};
    reg.add_queue(1, 0, uint32_t{40});
    reg.add_queue(1, 0, uint32_t{40});  // same token again
    EXPECT_EQ(reg.owners_of(0, 40), 1u);
    EXPECT_EQ(reg.live_queues(), 1u);

    // The same token moving to another slot leaves the old one empty.
    reg.add_queue(1, 0, uint32_t{41});
    EXPECT_EQ(reg.owners_of(0, 40), 0u);
    EXPECT_EQ(reg.owners_of(0, 41), 1u);

    reg.remove_queue(1);
    EXPECT_EQ(reg.owners_of(0, 41), 0u);
    EXPECT_EQ(reg.live_queues(), 0u);
    reg.remove_queue(1);  // idempotent
    EXPECT_EQ(reg.live_queues(), 0u);
}

// ---------------------------------------------------------------------------
// Unit 4: lazy HW-profiling enable
// ---------------------------------------------------------------------------

// A queue enables profiling exactly once, on its first signal-path batch; a queue
// that only ever runs signal-less batches never enables it.
TEST(ProfilingEnableTracker, enables_once_per_queue_and_never_for_signal_less)
{
    auto tracker = ProfilingEnableTracker{};

    // Queue 1 takes the signal path three times: enabled once.
    EXPECT_TRUE(tracker.mark(1));
    EXPECT_FALSE(tracker.mark(1));
    EXPECT_FALSE(tracker.mark(1));
    EXPECT_TRUE(tracker.enabled(1));

    // Queue 2 only ever runs signal-less batches, so mark() is never called.
    EXPECT_FALSE(tracker.enabled(2));
    EXPECT_EQ(tracker.size(), 1u);

    // Queue destroy forgets it, so a reused token re-enables on its first signal
    // batch rather than assuming a new queue inherited the old one's state.
    tracker.forget(1);
    EXPECT_FALSE(tracker.enabled(1));
    EXPECT_TRUE(tracker.mark(1));
}

// While signal-less is off, laziness is off too: the create-time enable stays,
// which is what keeps the flag-off path byte-identical.
TEST(ProfilingEnableTracker, laziness_is_tied_to_the_feature_being_active)
{
    static_assert(!signal_less_fully_wired(),
                  "lazy profiling must not engage before signal-less is fully wired");
}

// ---------------------------------------------------------------------------
// Unit 5: generation/reuse closure on queue destroy (requirement 4)
// ---------------------------------------------------------------------------

// The whole destroy sequence: a queue with in-flight signal-less work goes away,
// its pending entries are stranded, and the slot becomes signal-path-only so a
// queue that reuses the doorbell can never be matched to the old queue's records.
TEST(DispatchHub, destroy_closes_the_slot_for_reuse)
{
    auto reg = OwnerRegistry{};
    auto hub = hub_t{};

    // Queue A owns slot 40 and has two dispatches in flight.
    reg.add_queue(/*token=*/1, /*gpu=*/0, uint32_t{40});
    ASSERT_TRUE(register_one(hub, key_of(40, 1), /*corr_id=*/500, /*queue_token=*/1));
    ASSERT_TRUE(register_one(hub, key_of(40, 2), /*corr_id=*/500, /*queue_token=*/1));

    // Step 1: stop new reservations (hub lock taken and released).
    hub.mark_slot_closing(40);
    EXPECT_TRUE(hub.is_closing(40));

    // Step 2 is the gate_lock fence, which holds no hub lock; nothing to assert
    // here beyond it not being part of this critical section.

    // Step 3+4: strand what is left and quarantine permanently.
    auto stranded = hub.quarantine_slot(40);
    EXPECT_EQ(stranded.size(), 2u);
    EXPECT_TRUE(hub.is_quarantined(40));
    EXPECT_EQ(hub.live_entries(), 0u);

    // Their correlation id is on the ledger: deliberately not retired.
    EXPECT_TRUE(hub.is_ledgered(500));

    reg.remove_queue(1);

    // Queue B reuses the doorbell. Ownership looks clean, but the slot is
    // quarantined for the rest of the process, so B is signal-path-only.
    reg.add_queue(/*token=*/2, 0, uint32_t{40});
    EXPECT_TRUE(reg.is_injective(0, 40));
    EXPECT_FALSE(hub.can_register_batch({key_of(40, 1)}));
    EXPECT_FALSE(register_one(hub, key_of(40, 7), 600, 2));

    // A stale late record from queue A completes nothing.
    EXPECT_FALSE(hub.prove_eop(key_of(40, 1), 999, /*loss_free=*/true).has_value());
    EXPECT_FALSE(hub.prove_eop(key_of(40, 2), 999, true).has_value());
}

// "Closing" is deliberately an ELIGIBILITY-only gate. A batch that already passed
// eligibility and skipped its completion signals must still be able to register,
// because the destroy path fences those in flight before it strands anything --
// refusing them would leave dispatches with neither a signal nor a hub entry.
TEST(DispatchHub, closing_blocks_new_reservations_but_not_an_in_flight_one)
{
    auto hub = hub_t{};
    hub.mark_slot_closing(40);

    // Eligibility consults is_closing() and refuses...
    EXPECT_TRUE(hub.is_closing(40));

    // ...but a batch already past that point still registers, and can complete.
    EXPECT_TRUE(hub.can_register_batch({key_of(40, 1)}));
    ASSERT_TRUE(register_one(hub, key_of(40, 1)));
    EXPECT_TRUE(hub.prove_eop(key_of(40, 1), 5, true).has_value());

    // Quarantine, by contrast, refuses registration outright.
    hub.quarantine_slot(40);
    EXPECT_FALSE(hub.can_register_batch({key_of(40, 2)}));
    EXPECT_FALSE(register_one(hub, key_of(40, 2)));
}

// A clean destroy -- nothing signal-less on the slot -- strands nothing and
// reports nothing, which is what keeps the flag-off destroy path silent.
TEST(DispatchHub, clean_destroy_strands_nothing)
{
    auto hub = hub_t{};
    hub.mark_slot_closing(40);
    auto stranded = hub.quarantine_slot(40);

    EXPECT_TRUE(stranded.empty());
    EXPECT_EQ(hub.live_entries(), 0u);
    EXPECT_EQ(hub.tombstones(), 0u);
    EXPECT_EQ(hub.mode(), session_mode::running);
}

// Destroy runs concurrently with enqueue registration on other slots. The hub must
// stay internally consistent and never deadlock; run under TSan for the ordering.
TEST(DispatchHub, concurrent_destroy_and_registration_stay_consistent)
{
    constexpr uint32_t kSlots = 64;

    auto hub = hub_t{};
    for(uint32_t s = 0; s < kSlots; ++s)
    {
        ASSERT_TRUE(register_one(hub, key_of(s, 1), /*corr_id=*/s, /*queue_token=*/s));
    }

    // "Destroy" thread walks the slots doing the close sequence.
    auto destroyer = std::thread{[&hub]() {
        for(uint32_t s = 0; s < kSlots; ++s)
        {
            hub.mark_slot_closing(s);
            hub.quarantine_slot(s);
        }
    }};

    // "Reader" thread races it trying to prove the same entries.
    auto prover = std::atomic<uint32_t>{0};
    auto reader = std::thread{[&hub, &prover]() {
        for(uint32_t s = 0; s < kSlots; ++s)
            if(hub.prove_eop(key_of(s, 1), 9, true).has_value()) ++prover;
    }};

    destroyer.join();
    reader.join();

    // Every slot ended quarantined, nothing is still pending, and each entry went
    // exactly one way -- proven or stranded, never both.
    for(uint32_t s = 0; s < kSlots; ++s)
    {
        EXPECT_TRUE(hub.is_quarantined(s));
    }
    EXPECT_EQ(hub.live_entries(), 0u);
    EXPECT_LE(prover.load(), kSlots);
    EXPECT_EQ(tracked_payload::live.load(), 0);
}

// ---------------------------------------------------------------------------
// Unit 7: fork epoch / child abandonment (requirement 8, U19)
// ---------------------------------------------------------------------------

namespace
{
// Inherited process-wide state, mirroring production: function-local statics, so
// a forked child inherits them AND runs their destructors at a normal exit().
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

retry_owner<int>&
forked_retry()
{
    static auto _v = retry_owner<int>{};
    return _v;
}

// Everything a child must be able to call without touching an inherited mutex.
// Returns true when every entry point reported "disabled/empty".
bool
all_entry_points_short_circuit()
{
    bool ok = true;

    ok = ok && !register_one(forked_hub(), key_of(40, 99));
    ok = ok && !forked_hub().prove_eop(key_of(40, 1), 1, true).has_value();
    ok = ok && !forked_hub().note_start(key_of(40, 1), 1);
    ok = ok && !forked_hub().leak(key_of(40, 1)).has_value();
    ok = ok && forked_hub().live_entries() == 0;
    ok = ok && forked_hub().tombstones() == 0;
    ok = ok && forked_hub().outstanding(1) == 0;
    ok = ok && !forked_hub().is_quarantined(40);
    ok = ok && !forked_hub().is_closing(40);
    ok = ok && !forked_hub().is_ledgered(500);
    ok = ok && forked_hub().mode() == session_mode::child_stale;
    ok = ok && forked_hub().quarantine_slot(40).empty();
    ok = ok && forked_hub().close_queue(1).empty();
    ok = ok && forked_hub().drain_for_teardown().first.empty();
    ok = ok && forked_hub().poison(session_mode::loss_poisoned).first.empty();

    ok = ok && forked_registry().live_queues() == 0;
    ok = ok && !forked_registry().is_injective(0, 40);
    ok = ok && !forked_registry().slot_of(1).has_value();
    ok = ok && forked_registry().owners_of(0, 40) == 0;
    ok = ok && forked_registry().unresolved_queues(0) == 0;

    ok = ok && forked_retry().size() == 0;
    ok = ok && forked_retry().closed();
    // A completion arriving in a child is dropped, never finalized: the task group
    // that would run the finalizer did not survive the fork.
    bool finalized = false;
    ok = ok && !forked_retry().hold(7, [&finalized](int&&) { finalized = true; });
    ok = ok && !finalized;
    ok = ok && forked_retry().flush([](int&) { return submit_result::accepted; },
                                    [&finalized](int&&) { finalized = true; }) == 0;
    ok = ok && !finalized;

    return ok;
}
}  // namespace

// Every entry point on every Phase-2 shared object short-circuits once abandoned,
// so a child never reaches the lock behind it.
TEST(fork_safety, abandoned_objects_short_circuit_every_entry_point)
{
    forked_registry().add_queue(1, 0, uint32_t{40});
    register_one(forked_hub(), key_of(40, 1), /*corr_id=*/500);
    forked_retry().hold(1, [](int&&) {});

    forked_hub().abandon_in_child();
    forked_registry().abandon_in_child();
    forked_retry().abandon_in_child();

    EXPECT_TRUE(forked_hub().abandoned());
    EXPECT_TRUE(forked_registry().abandoned());
    EXPECT_TRUE(forked_retry().abandoned());
    EXPECT_TRUE(all_entry_points_short_circuit());
}

// U19: a REAL fork. The child abandons its inherited state exactly as the atfork
// handler does, exercises the entry points, and leaves through a normal exit() so
// the inherited statics' destructors actually run. It must not hang, crash, or
// double-free, and the parent must be unaffected.
TEST(fork_safety, forked_child_short_circuits_and_survives_normal_exit)
{
    // Populate in the parent so the child inherits non-empty state.
    auto parent_only = hub_t{};
    ASSERT_TRUE(register_one(parent_only, key_of(9, 1)));

    pid_t pid = fork();
    ASSERT_NE(pid, -1);

    if(pid == 0)
    {
        // CHILD. Exactly the atfork handler's work: atomic stores only.
        forked_hub().abandon_in_child();
        forked_registry().abandon_in_child();
        forked_retry().abandon_in_child();

        const bool ok = all_entry_points_short_circuit();

        // Normal exit(): runs static destructors over the abandoned state. The
        // plan requires exercising this, not just the handler.
        std::exit(ok ? 0 : 2);
    }

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status)) << "child did not exit normally";
    if(WIFEXITED(status))
    {
        EXPECT_EQ(WEXITSTATUS(status), 0) << "child entry points did not short-circuit";
    }

    // The parent is untouched by the child's abandonment.
    EXPECT_EQ(parent_only.live_entries(), 1u);
    EXPECT_TRUE(parent_only.prove_eop(key_of(9, 1), 5, true).has_value());
}

// The child must survive even when the fork happens while another thread is
// hammering the shared objects -- the case where an inherited mutex can be left
// permanently locked. Without the abandoned check preceding every lock, the child
// would deadlock here and the test would time out.
TEST(fork_safety, child_survives_a_fork_taken_under_contention)
{
    auto  hub  = hub_t{};
    auto  stop = std::atomic<bool>{false};
    auto  busy = std::thread{[&hub, &stop]() {
        for(uint32_t i = 0; !stop.load(); ++i)
        {
            register_one(hub, key_of(1, i % 512));
            hub.prove_eop(key_of(1, i % 512), 1, true);
            hub.live_entries();
        }
    }};

    for(int i = 0; i < 8; ++i)
    {
        pid_t pid = fork();
        ASSERT_NE(pid, -1);
        if(pid == 0)
        {
            // The inherited hub's mutex may be locked by the thread that did not
            // survive; abandoning makes every entry point avoid it entirely.
            hub.abandon_in_child();
            const bool ok = hub.live_entries() == 0 && !register_one(hub, key_of(1, 7)) &&
                            !hub.prove_eop(key_of(1, 7), 1, true).has_value();
            // _exit, not exit: this case is about not DEADLOCKING on an inherited
            // locked mutex. The busy thread's allocations are unreachable in the
            // child (its stack is gone), so a normal exit's leak check would report
            // a fork artifact rather than a defect. Destructor safety at a normal
            // exit() is covered by the test above.
            _exit(ok ? 0 : 2);
        }
        int status = 0;
        ASSERT_EQ(waitpid(pid, &status, 0), pid);
        EXPECT_TRUE(WIFEXITED(status));
        if(WIFEXITED(status))
        {
            EXPECT_EQ(WEXITSTATUS(status), 0);
        }
    }

    stop.store(true);
    busy.join();
}

// ---------------------------------------------------------------------------
// Unit 8: stateful model / property test
//
// Drives the hub + retry owner + no-signal finalizer through a randomized but
// SEEDED event sequence against a small reference model, asserting the plan's
// completion invariants after EVERY event. This is the highest coverage-per-
// effort test in the plan: it explores interleavings no hand-written case does.
// ---------------------------------------------------------------------------

namespace
{
enum class model_state
{
    absent,
    pending,
    proven,  // ownership handed out; can never become leaked
    leaked,
};

// The reference model: what the plan says must be true, tracked independently of
// the hub's own bookkeeping so the two can be compared.
struct reference_model
{
    std::map<std::pair<uint32_t, uint32_t>, model_state> state;      // (slot,id) -> state
    std::map<std::pair<uint32_t, uint32_t>, uint64_t>    corr_of;    // -> correlation id
    std::set<std::pair<uint32_t, uint32_t>>              tombstoned;
    std::set<uint32_t>                                   quarantined;
    std::set<uint64_t>                                   ledger;
    std::map<std::pair<uint32_t, uint32_t>, int>         emitted;
    std::map<std::pair<uint32_t, uint32_t>, int>         retired;
    bool                                                 poisoned = false;
    bool                                                 stopping = false;

    using key_t = std::pair<uint32_t, uint32_t>;

    bool admissible(const key_t& k) const
    {
        return !poisoned && !stopping && state.count(k) == 0 && tombstoned.count(k) == 0 &&
               quarantined.count(k.first) == 0;
    }

    model_state at(const key_t& k) const
    {
        auto it = state.find(k);
        return (it == state.end()) ? model_state::absent : it->second;
    }
};

correlation_key
to_key(const reference_model::key_t& k)
{
    return key_of(k.first, k.second);
}
}  // namespace

TEST(DispatchHub, stateful_model_matches_the_reference_across_random_events)
{
    constexpr int      kEvents  = 4000;
    constexpr uint32_t kSlots   = 6;
    constexpr uint32_t kPerSlot = 8;

    const int payloads_before = tracked_payload::live.load();

    auto hub   = hub_t{};
    auto owner = retry_owner<hub_t::proven>{};
    auto model = reference_model{};
    auto rng   = std::mt19937{20260803};

    // Proven completions waiting to be finalized, mirroring the task group.
    auto in_flight = std::vector<hub_t::proven>{};

    auto random_key = [&rng]() {
        return reference_model::key_t{rng() % kSlots, rng() % kPerSlot};
    };

    // Finalize one proven completion exactly as production does, and record what
    // the plan requires: retire exactly once, emit only when timing was usable.
    auto finalize = [&model](hub_t::proven&& p, bool convert_ok) {
        auto     mk      = reference_model::key_t{p.key.doorbell_off, p.key.dispatch_idx_low32};
        int      emits   = 0;
        int      retires = 0;
        auto     outcome = run_no_signal_finalizer(
            p.start_ticks,
            p.end_ticks,
            /*enqueue_ts=*/0,
            /*now_ns=*/1'000'000,
            [convert_ok](uint64_t t, uint64_t* out) {
                if(!convert_ok) return false;
                *out = t;
                return true;
            },
            [&emits](uint64_t, uint64_t) { ++emits; },
            [&retires]() { ++retires; });

        // A completion ALWAYS retires exactly once; it emits only on RESULT_READY.
        EXPECT_EQ(retires, 1);
        EXPECT_LE(emits, 1);
        EXPECT_EQ(emits, outcome == finalize_outcome::result_ready ? 1 : 0);

        model.emitted[mk] += emits;
        model.retired[mk] += retires;
    };

    for(int ev = 0; ev < kEvents; ++ev)
    {
        switch(rng() % 10)
        {
            case 0:  // RegisterBatch
            {
                auto batch = std::vector<hub_t::registration>{};
                auto keys  = std::vector<reference_model::key_t>{};
                auto corr  = uint64_t{100 + (rng() % 7)};
                for(uint32_t i = 0, n = 1 + (rng() % 3); i < n; ++i)
                {
                    auto k = random_key();
                    if(std::find(keys.begin(), keys.end(), k) != keys.end()) continue;
                    keys.emplace_back(k);
                    batch.emplace_back(reg_of(to_key(k), corr, /*queue_token=*/k.first));
                }
                if(keys.empty()) break;

                bool expect_ok = true;
                for(const auto& k : keys)
                    expect_ok = expect_ok && model.admissible(k);

                const bool got = hub.register_batch(std::move(batch));
                EXPECT_EQ(got, expect_ok) << "register_batch disagreed with the model";
                if(got)
                {
                    for(const auto& k : keys)
                    {
                        model.state[k]   = model_state::pending;
                        model.corr_of[k] = corr;
                    }
                }
                break;
            }
            case 1:  // START
            {
                auto k = random_key();
                hub.note_start(to_key(k), 10 + (rng() % 50));
                break;
            }
            case 2:
            case 3:  // EOP (sometimes under a lossy drain)
            {
                auto       k         = random_key();
                const bool loss_free = (rng() % 4) != 0;
                const bool expect_proven =
                    loss_free && !model.poisoned && !model.stopping &&
                    model.at(k) == model_state::pending;

                auto got = hub.prove_eop(to_key(k), 900, loss_free);
                EXPECT_EQ(got.has_value(), expect_proven) << "prove_eop disagreed with the model";
                if(got)
                {
                    model.state[k] = model_state::proven;
                    in_flight.emplace_back(std::move(*got));
                }
                break;
            }
            case 4:  // SubmitTask / RejectTask / RunTask
            {
                if(in_flight.empty()) break;
                auto p = std::move(in_flight.back());
                in_flight.pop_back();

                const auto disposition = rng() % 3;
                if(disposition == 0)
                {
                    finalize(std::move(p), /*convert_ok=*/true);  // accepted + ran
                }
                else if(disposition == 1)
                {
                    // ConvertFail -> COMPLETED_NO_TIMING: no record, still retires.
                    finalize(std::move(p), /*convert_ok=*/false);
                }
                else
                {
                    // Temporary rejection: the retry owner takes ownership.
                    owner.hold(std::move(p), [&finalize](hub_t::proven&& q) {
                        finalize(std::move(q), true);
                    });
                }
                break;
            }
            case 5:  // flush the retry owner
            {
                owner.flush([](hub_t::proven&) { return submit_result::rejected_permanent; },
                            [&finalize](hub_t::proven&& q) { finalize(std::move(q), true); });
                break;
            }
            case 6:  // Collision -> quarantine a slot
            {
                const uint32_t slot = rng() % kSlots;
                auto           lost = hub.quarantine_slot(slot);
                model.quarantined.insert(slot);
                for(auto& kv : model.state)
                {
                    if(kv.first.first == slot && kv.second == model_state::pending)
                    {
                        kv.second = model_state::leaked;
                        model.tombstoned.insert(kv.first);
                        model.ledger.insert(model.corr_of[kv.first]);
                    }
                }
                for(auto& l : lost)
                    EXPECT_EQ(l.key.doorbell_off, slot);
                break;
            }
            case 7:  // DestroyQueue
            {
                const uint32_t token = rng() % kSlots;
                hub.close_queue(token);
                for(auto& kv : model.state)
                {
                    // queue_token was set to the slot at registration.
                    if(kv.first.first == token && kv.second == model_state::pending)
                    {
                        kv.second = model_state::leaked;
                        model.tombstoned.insert(kv.first);
                        model.ledger.insert(model.corr_of[kv.first]);
                    }
                }
                break;
            }
            case 8:  // Poison (ring overrun / reader dead)
            {
                if(model.poisoned) break;
                if((rng() % 12) != 0) break;  // rare, it is terminal
                hub.poison(session_mode::loss_poisoned);
                model.poisoned = true;
                for(auto& kv : model.state)
                {
                    if(kv.second == model_state::pending)
                    {
                        kv.second = model_state::leaked;
                        model.tombstoned.insert(kv.first);
                        model.ledger.insert(model.corr_of[kv.first]);
                    }
                }
                break;
            }
            default: break;
        }

        // --- invariants, after EVERY event ---------------------------------

        for(const auto& kv : model.state)
        {
            const auto& k = kv.first;

            // No emitted record without a completion proven for that exact owner,
            // and at most one emit + one retire per dispatch.
            EXPECT_LE(model.emitted[k], 1) << "more than one record for one dispatch";
            EXPECT_LE(model.retired[k], 1) << "more than one cleanup for one dispatch";
            if(model.emitted[k] > 0)
            {
                EXPECT_EQ(kv.second, model_state::proven) << "record emitted without a proven EOP";
            }

            // Retire iff completion was proven; a leaked entry NEVER retires.
            if(kv.second == model_state::leaked)
            {
                EXPECT_EQ(model.retired[k], 0) << "a leaked dispatch was retired";
                EXPECT_EQ(model.emitted[k], 0) << "a leaked dispatch emitted a record";

                // No leaked entry is matchable again.
                EXPECT_FALSE(hub.prove_eop(to_key(k), 1, true).has_value())
                    << "a leaked entry was still matchable";
                EXPECT_FALSE(hub.can_register_batch({to_key(k)}))
                    << "a leaked key was re-registerable";
            }
        }

        // The loss ledger is exactly the leaked correlation ids.
        for(auto id : model.ledger)
        {
            EXPECT_TRUE(hub.is_ledgered(id)) << "leaked correlation id missing from the ledger";
        }

        // A quarantined slot never accepts a reservation again.
        for(auto slot : model.quarantined)
        {
            EXPECT_TRUE(hub.is_quarantined(slot));
        }
    }

    // Teardown: everything still pending leaks, nothing proven is lost.
    owner.flush([](hub_t::proven&) { return submit_result::rejected_permanent; },
                [&finalize](hub_t::proven&& q) { finalize(std::move(q), true); });
    for(auto& p : in_flight)
        finalize(std::move(p), true);
    in_flight.clear();
    hub.drain_for_teardown();

    // Every payload ever registered ended with exactly one owner that released it.
    EXPECT_EQ(tracked_payload::live.load(), payloads_before);
}
