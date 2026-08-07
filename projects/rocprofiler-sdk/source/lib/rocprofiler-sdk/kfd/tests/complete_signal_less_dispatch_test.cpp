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

#include "lib/rocprofiler-sdk/kfd/complete_signal_less_dispatch.hpp"

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
}  // namespace

// The finalizer's three outcomes

// start_ticks known + conversion + sanity OK -> RESULT_READY: one record with the
// converted KFD timestamps, correlation id retired exactly once.
TEST(complete_signal_less_dispatch, result_ready_emits_once_and_retires_once)
{
    auto obs  = observer{};
    auto conv = converter{};

    auto outcome = run_complete_signal_less_dispatch(std::optional<uint64_t>{500},
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
TEST(complete_signal_less_dispatch, missing_start_is_completed_no_timing)
{
    auto obs  = observer{};
    auto conv = converter{};

    auto outcome = run_complete_signal_less_dispatch(
        std::optional<uint64_t>{}, 900, 0, 10'000'000, conv.fn(), obs.emit_fn(), obs.retire_fn());

    EXPECT_EQ(outcome, finalize_outcome::completed_no_timing);
    EXPECT_EQ(obs.emits, 0);
    EXPECT_EQ(obs.retires, 1);
}

// Tick conversion failure is not a loss either: no record, normal retire, and
// crucially NO HSA fallback (there is nothing in this code path that could).
TEST(complete_signal_less_dispatch, convert_failure_is_completed_no_timing)
{
    auto obs  = observer{};
    auto conv = converter{};
    conv.ok   = false;

    auto outcome = run_complete_signal_less_dispatch(std::optional<uint64_t>{500},
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
TEST(complete_signal_less_dispatch, sanity_failure_is_completed_no_timing)
{
    auto conv = converter{};

    // Converted interval ends beyond now_ns PLUS the conversion slack. A few ms
    // past now is expected and accepted (see accepts_a_converted_end_slightly_
    // past_now); this is far enough out to be a record from somewhere else.
    {
        auto obs      = observer{};
        auto passthru = converter{true, /*epoch=*/0};
        auto outcome  = run_complete_signal_less_dispatch(std::optional<uint64_t>{1},
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
        auto outcome = run_complete_signal_less_dispatch(std::optional<uint64_t>{500},
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
        auto outcome = run_complete_signal_less_dispatch(std::optional<uint64_t>{900},
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
TEST(complete_signal_less_dispatch, accepts_a_converted_end_slightly_past_now)
{
    constexpr uint64_t now  = 1'000'000'000;
    constexpr uint64_t skew = 2'700'000;  // measured 2.0-2.7 ms
    auto               obs  = observer{};
    // Converts ticks straight through, so end lands at now + skew.
    auto conv = converter{true, /*epoch=*/0};

    auto outcome = run_complete_signal_less_dispatch(std::optional<uint64_t>{now - 5'000'000},
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
TEST(complete_signal_less_dispatch, throwing_emit_still_retires_exactly_once)
{
    auto obs  = observer{};
    auto conv = converter{};

    EXPECT_THROW(run_complete_signal_less_dispatch(
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

// Bounded retry owner: an EOP-proven completion is never dropped

// Rejection-cause reporting
//
// The no-timing breakdown is only useful if each cause is attributed correctly:
// shape-ii and a rejected sanity clause need opposite fixes, so a mislabelled
// counter would send the next investigation the wrong way.

TEST(complete_signal_less_dispatch, reports_the_exact_rejection_cause)
{
    constexpr uint64_t now = 1'000'000'000;

    auto run = [&](std::optional<uint64_t> start_ticks,
                   uint64_t                end_ticks,
                   uint64_t                enqueue_ts,
                   bool                    convert_ok) {
        auto obs     = observer{};
        auto conv    = converter{convert_ok, /*epoch=*/0};
        auto detail  = finalize_detail{};
        auto outcome = run_complete_signal_less_dispatch(start_ticks,
                                               end_ticks,
                                               enqueue_ts,
                                               now,
                                               conv.fn(),
                                               obs.emit_fn(),
                                               obs.retire_fn(),
                                               &detail);
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
TEST(complete_signal_less_dispatch, conversion_skew_is_not_counted_as_a_rejection)
{
    constexpr uint64_t now  = 1'000'000'000;
    auto               obs  = observer{};
    auto               conv = converter{true, /*epoch=*/0};
    auto               det  = finalize_detail{};

    auto outcome = run_complete_signal_less_dispatch(std::optional<uint64_t>{now - 5'000'000},
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
