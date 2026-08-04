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

// Unit tests for the no-signal finalizer and the bounded retry owner. Both are
// free of the HSA/tracing headers, so the tick converter (seam S3), the record
// emitter, the retirement observer (seam S7) and the executor (seam S2) are all
// injected here and every branch is deterministic.

#include "lib/rocprofiler-sdk/kfd/no_signal_finalizer.hpp"
#include "lib/rocprofiler-sdk/kfd/teardown.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace rocprofiler::kfd;

// Retirement observer (S7): records how many times a correlation id was retired,
// so "exactly once" is checked directly instead of by reading logs.
struct observer
{
    int      retires = 0;
    int      emits   = 0;
    uint64_t start   = 0;
    uint64_t end     = 0;

    auto retire_fn()
    {
        return [this]() { ++retires; };
    }

    auto emit_fn()
    {
        return [this](uint64_t s, uint64_t e) {
            ++emits;
            start = s;
            end   = e;
        };
    }
};

// Injectable tick converter (S3). Adds a fixed epoch so converted values are
// distinguishable from raw ticks; `ok = false` forces the conversion-failure
// branch the way a non-GPU agent would.
struct converter
{
    bool     ok    = true;
    uint64_t epoch = 1'000'000;

    auto fn()
    {
        return [this](uint64_t ticks, uint64_t* out) {
            if(!ok) return false;
            *out = ticks + epoch;
            return true;
        };
    }
};

// A proven completion stand-in for the retry owner tests: move-only, so the
// "exactly one owner" property is enforced by the type system.
struct fake_proven
{
    uint64_t id = 0;

    explicit fake_proven(uint64_t v)
    : id{v}
    {}

    fake_proven(fake_proven&&) noexcept = default;
    fake_proven& operator=(fake_proven&&) noexcept = default;
    fake_proven(const fake_proven&)                = delete;
    fake_proven& operator=(const fake_proven&) = delete;
};
}  // namespace

// ---------------------------------------------------------------------------
// The finalizer's three outcomes
// ---------------------------------------------------------------------------

// start_ticks known + conversion + sanity OK -> RESULT_READY: one record with the
// converted KFD timestamps, correlation id retired exactly once.
TEST(no_signal_finalizer, result_ready_emits_once_and_retires_once)
{
    auto obs  = observer{};
    auto conv = converter{};

    auto outcome = run_no_signal_finalizer(std::optional<uint64_t>{500},
                                           /*end_ticks=*/900,
                                           /*enqueue_ts=*/0,
                                           /*now_ns=*/10'000'000,
                                           conv.fn(),
                                           obs.emit_fn(),
                                           obs.retire_fn());

    EXPECT_EQ(outcome, finalize_outcome::result_ready);
    EXPECT_EQ(obs.emits, 1);
    EXPECT_EQ(obs.retires, 1);
    EXPECT_EQ(obs.start, 500u + conv.epoch);
    EXPECT_EQ(obs.end, 900u + conv.epoch);
}

// Shape (ii): the EOP proved completion but the START was lost. No record, but the
// correlation id still retires -- the kernel is done (G3), so this is NOT a leak.
TEST(no_signal_finalizer, missing_start_is_completed_no_timing)
{
    auto obs  = observer{};
    auto conv = converter{};

    auto outcome = run_no_signal_finalizer(std::optional<uint64_t>{},
                                           900,
                                           0,
                                           10'000'000,
                                           conv.fn(),
                                           obs.emit_fn(),
                                           obs.retire_fn());

    EXPECT_EQ(outcome, finalize_outcome::completed_no_timing);
    EXPECT_EQ(obs.emits, 0);
    EXPECT_EQ(obs.retires, 1);
}

// Tick conversion failure is not a loss either: no record, normal retire, and
// crucially NO HSA fallback (there is nothing in this code path that could).
TEST(no_signal_finalizer, convert_failure_is_completed_no_timing)
{
    auto obs  = observer{};
    auto conv = converter{};
    conv.ok   = false;

    auto outcome = run_no_signal_finalizer(std::optional<uint64_t>{500},
                                           900,
                                           0,
                                           10'000'000,
                                           conv.fn(),
                                           obs.emit_fn(),
                                           obs.retire_fn());

    EXPECT_EQ(outcome, finalize_outcome::completed_no_timing);
    EXPECT_EQ(obs.emits, 0);
    EXPECT_EQ(obs.retires, 1);
}

// The correlation guard rejects a record that does not fall inside this
// dispatch's own CPU window -- it is not this dispatch's record.
TEST(no_signal_finalizer, sanity_failure_is_completed_no_timing)
{
    auto conv = converter{};

    // Converted interval ends beyond now_ns PLUS the conversion slack. A few ms
    // past now is expected and accepted (see accepts_a_converted_end_slightly_
    // past_now); this is far enough out to be a record from somewhere else.
    {
        auto obs        = observer{};
        auto passthru   = converter{true, /*epoch=*/0};
        auto outcome    = run_no_signal_finalizer(std::optional<uint64_t>{1},
                                                  /*end_ticks=*/1000 + kKfdFutureSlackNs + 1,
                                                  /*enqueue_ts=*/0,
                                                  /*now_ns=*/1000,
                                                  passthru.fn(),
                                                  obs.emit_fn(),
                                                  obs.retire_fn());
        EXPECT_EQ(outcome, finalize_outcome::completed_no_timing);
        EXPECT_EQ(obs.emits, 0);
        EXPECT_EQ(obs.retires, 1);
    }

    // Converted interval starts before the dispatch was enqueued.
    {
        auto obs     = observer{};
        auto outcome = run_no_signal_finalizer(std::optional<uint64_t>{500},
                                               900,
                                               /*enqueue_ts=*/9'000'000,
                                               /*now_ns=*/10'000'000,
                                               conv.fn(),
                                               obs.emit_fn(),
                                               obs.retire_fn());
        EXPECT_EQ(outcome, finalize_outcome::completed_no_timing);
        EXPECT_EQ(obs.emits, 0);
        EXPECT_EQ(obs.retires, 1);
    }

    // Non-positive interval (end <= start).
    {
        auto obs     = observer{};
        auto outcome = run_no_signal_finalizer(std::optional<uint64_t>{900},
                                               900,
                                               0,
                                               10'000'000,
                                               conv.fn(),
                                               obs.emit_fn(),
                                               obs.retire_fn());
        EXPECT_EQ(outcome, finalize_outcome::completed_no_timing);
        EXPECT_EQ(obs.emits, 0);
        EXPECT_EQ(obs.retires, 1);
    }
}

// The shape that actually occurs on hardware: the converted firmware end lands a
// couple of milliseconds past the `now` the finalizer sampled, because the tick
// conversion's correlation is only periodically re-synced. This must produce a
// record -- rejecting it discarded every dispatch on gfx1201.
TEST(no_signal_finalizer, accepts_a_converted_end_slightly_past_now)
{
    constexpr uint64_t now   = 1'000'000'000;
    constexpr uint64_t skew  = 2'700'000;  // measured 2.0-2.7 ms
    auto               obs   = observer{};
    // Converts ticks straight through, so end lands at now + skew.
    auto               conv  = converter{true, /*epoch=*/0};

    auto outcome = run_no_signal_finalizer(std::optional<uint64_t>{now - 5'000'000},
                                           now + skew,
                                           /*enqueue_ts=*/0,
                                           now,
                                           conv.fn(),
                                           obs.emit_fn(),
                                           obs.retire_fn());

    EXPECT_EQ(outcome, finalize_outcome::result_ready);
    EXPECT_EQ(obs.emits, 1);
    EXPECT_EQ(obs.retires, 1);
    EXPECT_EQ(obs.end, now + skew);
}

// A throwing client callback must NOT skip cleanup: the scope destructor retires
// the correlation id exactly once on the way out (AGENTS.md RAII requirement).
TEST(no_signal_finalizer, throwing_emit_still_retires_exactly_once)
{
    auto obs  = observer{};
    auto conv = converter{};

    EXPECT_THROW(run_no_signal_finalizer(
                     std::optional<uint64_t>{500},
                     900,
                     0,
                     10'000'000,
                     conv.fn(),
                     [](uint64_t, uint64_t) { throw std::runtime_error{"callback failed"}; },
                     obs.retire_fn()),
                 std::runtime_error);

    EXPECT_EQ(obs.retires, 1);
}

// ---------------------------------------------------------------------------
// Bounded retry owner: an EOP-proven completion is never dropped
// ---------------------------------------------------------------------------

// A temporary rejection parks the entry; a later flush submits it successfully.
TEST(retry_owner, temporary_rejection_is_retried_on_flush)
{
    auto owner     = retry_owner<fake_proven>{};
    auto finalized = std::vector<uint64_t>{};
    auto submitted = std::vector<uint64_t>{};

    auto finalize = [&finalized](fake_proven&& p) { finalized.emplace_back(p.id); };

    EXPECT_TRUE(owner.hold(fake_proven{7}, finalize));
    EXPECT_EQ(owner.size(), 1u);
    EXPECT_TRUE(finalized.empty());  // held, not finalized

    auto accept = [&submitted](fake_proven& p) {
        submitted.emplace_back(p.id);
        return submit_result::accepted;
    };
    EXPECT_EQ(owner.flush(accept, finalize), 1u);

    EXPECT_EQ(submitted, std::vector<uint64_t>{7});
    EXPECT_TRUE(finalized.empty());  // the executor took it
    EXPECT_EQ(owner.size(), 0u);
}

// A permanent rejection means the executor is closing: the entry is finalized
// SYNCHRONOUSLY on the flushing thread rather than retried or dropped.
TEST(retry_owner, closed_owner_finalizes_in_place)
{
    auto owner     = retry_owner<fake_proven>{};
    auto finalized = std::vector<uint64_t>{};
    auto finalize  = [&finalized](fake_proven&& p) { finalized.emplace_back(p.id); };

    EXPECT_TRUE(owner.hold(fake_proven{1}, finalize));
    EXPECT_TRUE(owner.hold(fake_proven{2}, finalize));
    owner.close();
    EXPECT_TRUE(owner.closed());

    auto never_called = [](fake_proven&) {
        ADD_FAILURE() << "a closed owner must not submit";
        return submit_result::rejected_permanent;
    };
    EXPECT_EQ(owner.flush(never_called, finalize), 2u);

    EXPECT_EQ(finalized, (std::vector<uint64_t>{1, 2}));
    EXPECT_EQ(owner.size(), 0u);
}

// A rejection at flush time also finalizes in place, so nothing is left holding
// an EOP-proven completion after a flush.
TEST(retry_owner, rejection_during_flush_finalizes_in_place)
{
    auto owner     = retry_owner<fake_proven>{};
    auto finalized = std::vector<uint64_t>{};
    auto finalize  = [&finalized](fake_proven&& p) { finalized.emplace_back(p.id); };

    EXPECT_TRUE(owner.hold(fake_proven{5}, finalize));

    auto reject = [](fake_proven&) { return submit_result::rejected_permanent; };
    EXPECT_EQ(owner.flush(reject, finalize), 1u);

    EXPECT_EQ(finalized, std::vector<uint64_t>{5});
    EXPECT_EQ(owner.size(), 0u);
}

// The owner is bounded: past capacity it finalizes in place instead of growing
// without limit, so memory is bounded and the completion is still not dropped.
TEST(retry_owner, is_bounded_and_never_drops_a_completion)
{
    auto owner     = retry_owner<fake_proven>{};
    auto finalized = std::vector<uint64_t>{};
    auto finalize  = [&finalized](fake_proven&& p) { finalized.emplace_back(p.id); };

    for(uint64_t i = 0; i < retry_owner<fake_proven>::kMaxHeld; ++i)
    {
        EXPECT_TRUE(owner.hold(fake_proven{i}, finalize));
    }
    EXPECT_EQ(owner.size(), retry_owner<fake_proven>::kMaxHeld);
    EXPECT_TRUE(finalized.empty());

    // One past capacity: held count does not grow, and the extra entry is
    // finalized rather than dropped.
    EXPECT_FALSE(owner.hold(fake_proven{9999}, finalize));
    EXPECT_EQ(owner.size(), retry_owner<fake_proven>::kMaxHeld);
    EXPECT_EQ(finalized, std::vector<uint64_t>{9999});

    // Everything held is still accounted for by a flush.
    auto accept = [](fake_proven&) { return submit_result::accepted; };
    EXPECT_EQ(owner.flush(accept, finalize), retry_owner<fake_proven>::kMaxHeld);
    EXPECT_EQ(owner.size(), 0u);
}

// ---------------------------------------------------------------------------
// Unit 6: the strict teardown order (design requirement 7)
// ---------------------------------------------------------------------------

namespace
{
// Records the order the steps ran in, and lets a step assert what the world looks
// like at that point (e.g. that no producer can still submit).
struct recording_steps
{
    std::vector<std::string> order;

    bool producers_stopped = false;
    bool reader_joined     = false;
    bool retry_flushed     = false;
    bool group_joined      = false;

    // Set if anything tried to submit after the join began (invariant 12).
    bool submitted_after_join = false;

    void stop_new_reservations()
    {
        order.emplace_back("stop_new_reservations");
        producers_stopped = true;
    }

    void quiesce_interceptor() { order.emplace_back("quiesce_interceptor"); }

    void stop_and_join_reader()
    {
        order.emplace_back("stop_and_join_reader");
        reader_joined = true;
    }

    void flush_retry_owner()
    {
        order.emplace_back("flush_retry_owner");
        retry_flushed = true;
    }

    void leak_remaining_pending() { order.emplace_back("leak_remaining_pending"); }

    void join_task_group()
    {
        order.emplace_back("join_task_group");
        group_joined = true;
    }

    // A producer trying to submit at this instant.
    void simulate_producer_submit()
    {
        if(group_joined) submitted_after_join = true;
    }
};
}  // namespace

// The six steps run in exactly the documented order. A reordering here is a
// use-after-free at exit, so it is pinned by a test rather than a comment.
TEST(signal_less_teardown, steps_run_in_the_required_order)
{
    auto steps = recording_steps{};
    run_signal_less_teardown(steps);

    const auto expected = std::vector<std::string>{"stop_new_reservations",
                                                   "quiesce_interceptor",
                                                   "stop_and_join_reader",
                                                   "flush_retry_owner",
                                                   "leak_remaining_pending",
                                                   "join_task_group"};
    EXPECT_EQ(steps.order, expected);
}

// The properties the order exists to guarantee: reservations stop before the
// interceptor is fenced, the reader is joined before the retry owner is flushed
// (so nothing can be added to it afterwards), and the task group is joined last.
TEST(signal_less_teardown, ordering_guarantees_hold_at_each_step)
{
    struct checking_steps
    {
        bool producers_stopped = false;
        bool interceptor_fenced = false;
        bool reader_joined     = false;
        bool retry_flushed     = false;
        bool pending_leaked    = false;
        bool group_joined      = false;

        void stop_new_reservations()
        {
            EXPECT_FALSE(interceptor_fenced) << "reservations must stop first";
            producers_stopped = true;
        }
        void quiesce_interceptor()
        {
            EXPECT_TRUE(producers_stopped) << "fencing before reservations stop races new entries";
            interceptor_fenced = true;
        }
        void stop_and_join_reader()
        {
            EXPECT_TRUE(interceptor_fenced) << "the reader must be stopped after the fence";
            reader_joined = true;
        }
        void flush_retry_owner()
        {
            EXPECT_TRUE(reader_joined) << "flushing before the reader joins is not final";
            retry_flushed = true;
        }
        void leak_remaining_pending()
        {
            EXPECT_TRUE(retry_flushed) << "proven work must be finalized before leaking the rest";
            pending_leaked = true;
        }
        void join_task_group()
        {
            EXPECT_TRUE(retry_flushed) << "joining before the flush strands retry-owned work";
            EXPECT_TRUE(pending_leaked);
            group_joined = true;
        }
    };

    auto steps = checking_steps{};
    run_signal_less_teardown(steps);
    EXPECT_TRUE(steps.group_joined);
}

// Invariant 12: no task may be submitted after the join begins. Steps 1-3 are what
// make that true, so a producer that runs at any point in the sequence is already
// shut out by the time the join happens.
TEST(signal_less_teardown, no_task_is_submitted_after_the_join_begins)
{
    struct producer_steps : recording_steps
    {
        void stop_new_reservations()
        {
            recording_steps::stop_new_reservations();
            simulate_producer_submit();
        }
        void stop_and_join_reader()
        {
            recording_steps::stop_and_join_reader();
            simulate_producer_submit();
        }
        void join_task_group()
        {
            simulate_producer_submit();  // still before the join is recorded
            recording_steps::join_task_group();
        }
    };

    auto steps = producer_steps{};
    run_signal_less_teardown(steps);
    EXPECT_FALSE(steps.submitted_after_join);
}

// Everything empty -- the flag-off shape -- still runs the sequence cleanly and
// finalizes nothing, which is what makes the real call a no-op when signal-less
// is off.
TEST(signal_less_teardown, flush_with_nothing_held_finalizes_nothing)
{
    auto owner     = retry_owner<fake_proven>{};
    auto finalized = 0;
    auto submitted = 0;

    auto n = owner.flush([&submitted](fake_proven&) {
                             ++submitted;
                             return submit_result::accepted;
                         },
                         [&finalized](fake_proven&&) { ++finalized; });

    EXPECT_EQ(n, 0u);
    EXPECT_EQ(submitted, 0);
    EXPECT_EQ(finalized, 0);
    EXPECT_EQ(owner.size(), 0u);
}

// ---------------------------------------------------------------------------
// Rejection-cause reporting
//
// The no-timing breakdown is only useful if each cause is attributed correctly:
// shape-ii and a rejected sanity clause need opposite fixes, so a mislabelled
// counter would send the next investigation the wrong way.
// ---------------------------------------------------------------------------

TEST(no_signal_finalizer, reports_the_exact_rejection_cause)
{
    constexpr uint64_t now = 1'000'000'000;

    auto run = [&](std::optional<uint64_t> start_ticks,
                   uint64_t                end_ticks,
                   uint64_t                enqueue_ts,
                   bool                    convert_ok) {
        auto obs    = observer{};
        auto conv   = converter{convert_ok, /*epoch=*/0};
        auto detail = finalize_detail{};
        auto outcome =
            run_no_signal_finalizer(start_ticks, end_ticks, enqueue_ts, now, conv.fn(),
                                    obs.emit_fn(), obs.retire_fn(), &detail);
        // Whatever the cause, the correlation id retires exactly once.
        EXPECT_EQ(obs.retires, 1);
        return std::make_pair(outcome, detail.reason);
    };

    // Success reports `ready` and emits.
    {
        auto [outcome, reason] = run(now - 5'000'000, now - 1'000'000, 0, true);
        EXPECT_EQ(outcome, finalize_outcome::result_ready);
        EXPECT_EQ(reason, finalize_reason::ready);
    }

    // Shape ii: the EOP proved completion, the START was lost.
    {
        auto [outcome, reason] = run(std::nullopt, now - 1'000'000, 0, true);
        EXPECT_EQ(outcome, finalize_outcome::completed_no_timing);
        EXPECT_EQ(reason, finalize_reason::start_unknown);
    }

    // Conversion refused (non-GPU agent).
    {
        auto [outcome, reason] = run(now - 5'000'000, now - 1'000'000, 0, false);
        EXPECT_EQ(outcome, finalize_outcome::completed_no_timing);
        EXPECT_EQ(reason, finalize_reason::convert_failed);
    }

    // Non-positive interval.
    {
        auto [outcome, reason] = run(now - 1'000'000, now - 5'000'000, 0, true);
        EXPECT_EQ(outcome, finalize_outcome::completed_no_timing);
        EXPECT_EQ(reason, finalize_reason::bad_interval);
    }

    // Starts before this dispatch was enqueued -- the misattribution check.
    {
        auto [outcome, reason] = run(now - 5'000'000, now - 1'000'000, now - 2'000'000, true);
        EXPECT_EQ(outcome, finalize_outcome::completed_no_timing);
        EXPECT_EQ(reason, finalize_reason::before_enqueue);
    }

    // Ends beyond now + the conversion slack.
    {
        auto [outcome, reason] = run(1, now + kKfdFutureSlackNs + 1, 0, true);
        EXPECT_EQ(outcome, finalize_outcome::completed_no_timing);
        EXPECT_EQ(reason, finalize_reason::after_now);
    }
}

// A few ms past now stays inside the slack, so it must be reported as ready and
// NOT counted against after_now -- otherwise the breakdown would blame the guard
// for the very skew it was widened to absorb.
TEST(no_signal_finalizer, conversion_skew_is_not_counted_as_a_rejection)
{
    constexpr uint64_t now  = 1'000'000'000;
    auto               obs  = observer{};
    auto               conv = converter{true, /*epoch=*/0};
    auto               det  = finalize_detail{};

    auto outcome = run_no_signal_finalizer(std::optional<uint64_t>{now - 5'000'000},
                                           now + 2'700'000,
                                           0,
                                           now,
                                           conv.fn(),
                                           obs.emit_fn(),
                                           obs.retire_fn(),
                                           &det);

    EXPECT_EQ(outcome, finalize_outcome::result_ready);
    EXPECT_EQ(det.reason, finalize_reason::ready);
    EXPECT_EQ(obs.emits, 1);
}
