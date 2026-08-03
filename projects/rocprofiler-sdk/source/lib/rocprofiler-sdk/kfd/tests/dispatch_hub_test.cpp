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
#include "lib/rocprofiler-sdk/kfd/signal_less_gate.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <unordered_set>
#include <vector>

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

// The owner-injectivity input is a placeholder that reports "not trackable" until
// the all-live-queue reverse owner registry lands, so it alone also holds every
// batch on the signal path.
TEST(signal_less_flag, placeholder_owner_check_is_not_yet_trackable)
{
    EXPECT_FALSE(doorbell_owner_is_injective(0));
    EXPECT_FALSE(doorbell_owner_is_injective(4100));

    // fully_wired is set true here to isolate the owner check as the sole failure.
    auto in                  = eligibility_inputs{};
    in.feature_enabled       = true;
    in.fully_wired           = true;
    in.session_live_for_gpu  = true;
    in.reader_alive          = true;
    in.hub_accepts_batch     = true;
    in.payload_constructible = true;
    in.doorbells_injective   = doorbell_owner_is_injective(4100);

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
