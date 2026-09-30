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

// Unit tests for the no-signal finalizer. Every seam (tick converter, record
// emitter, retirement observer, executor) is injected, so each branch is
// deterministic.

#include "lib/rocprofiler-sdk/kfd/complete_signal_less_dispatch.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace
{
using namespace rocprofiler::kfd;

// The retirement observer: counts retires/emits so "exactly once" is checked
// directly, not via logs.
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

// The tick converter: adds a fixed epoch so converted values are distinguishable
// from raw ticks; ok=false forces the conversion-failure branch (as a non-GPU
// agent would).
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

// start known + conversion + sanity OK -> one record with converted KFD
// timestamps, correlation id retired exactly once.
TEST(complete_signal_less_dispatch, result_ready_emits_once_and_retires_once)
{
    auto obs     = observer{};
    auto conv    = converter{};
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

// Only a genuine conversion failure emits nothing: without a system-domain end
// there is nowhere to place the record. It still retires exactly once -- completion
// is proven, not leaked.
TEST(complete_signal_less_dispatch, conversion_failure_emits_nothing_but_retires_once)
{
    auto obs     = observer{};
    auto conv    = converter{/*ok=*/false, /*epoch=*/1'000'000};
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

// A genuinely zero-duration interval -- identical firmware ticks, so START and EOP
// converted to the same nanosecond -- is the one interval shape still dropped:
// widening rescues distinct-but-quantized ticks, not a real zero. Retires once.
TEST(complete_signal_less_dispatch, identical_ticks_zero_duration_is_dropped)
{
    auto obs     = observer{};
    auto conv    = converter{true, /*epoch=*/1'000'000};
    auto detail  = finalize_detail{};
    auto outcome = run_complete_signal_less_dispatch(std::optional<uint64_t>{900},
                                                     900,
                                                     0,
                                                     10'000'000,
                                                     conv.fn(),
                                                     obs.emit_fn(),
                                                     obs.retire_fn(),
                                                     &detail);
    EXPECT_EQ(outcome, finalize_outcome::completed_no_timing);
    EXPECT_EQ(obs.emits, 0);
    EXPECT_EQ(obs.retires, 1);
    EXPECT_EQ(detail.reason, finalize_reason::bad_interval);
}

// Every recoverable shape -- a lost START, or a conversion-skewed interval the
// signal path would clamp -- now EMITS a record and retires exactly once, because
// the EOP proved the dispatch ran and its converted end is a real timestamp that
// anchors the record. Dropping these was the record loss this test guards against.
TEST(complete_signal_less_dispatch, recoverable_shapes_emit_and_retire_once)
{
    struct row
    {
        std::optional<uint64_t> start;
        uint64_t                end, enqueue, now;
        uint64_t                epoch;
        const char*             label;
    };
    const row rows[] = {
        {std::nullopt, 900, 0, 10'000'000, 1'000'000, "start lost (shape ii)"},
        {std::optional<uint64_t>{1},
         1000 + kKfdFutureSlackNs + 1,
         0,
         1000,
         0,
         "end beyond now + slack"},
        {std::optional<uint64_t>{500},
         900,
         9'000'000,
         10'000'000,
         1'000'000,
         "starts before enqueue"},
        {std::optional<uint64_t>{900}, 800, 0, 10'000'000, 1'000'000, "inverted -> swapped"},
    };
    for(const auto& tc : rows)
    {
        auto obs     = observer{};
        auto conv    = converter{true, tc.epoch};
        auto outcome = run_complete_signal_less_dispatch(
            tc.start, tc.end, tc.enqueue, tc.now, conv.fn(), obs.emit_fn(), obs.retire_fn());
        EXPECT_EQ(outcome, finalize_outcome::result_ready) << tc.label;
        EXPECT_EQ(obs.emits, 1) << tc.label;
        EXPECT_EQ(obs.retires, 1) << tc.label;
    }
}

// a converted firmware end a few ms past `now` must be kept but
// clamped to `now` (retirement samples a host clock just after now, so an
// unclamped future end lands after retirement). The whole interval shifts down so
// the duration is preserved -- same as tracing::adjust_profiling_time.
TEST(complete_signal_less_dispatch, converted_end_past_now_is_clamped_to_now)
{
    constexpr uint64_t now      = 1'000'000'000;
    constexpr uint64_t skew     = 2'700'000;  // measured 2.0-2.7 ms
    constexpr uint64_t duration = 5'000'000;
    auto               obs      = observer{};
    auto               conv     = converter{true, /*epoch=*/0};  // passthrough -> end at now + skew
    auto outcome = run_complete_signal_less_dispatch(std::optional<uint64_t>{now + skew - duration},
                                                     now + skew,
                                                     /*enqueue_ts=*/0,
                                                     now,
                                                     conv.fn(),
                                                     obs.emit_fn(),
                                                     obs.retire_fn());
    EXPECT_EQ(outcome, finalize_outcome::result_ready);
    EXPECT_EQ(obs.emits, 1);
    EXPECT_EQ(obs.retires, 1);
    EXPECT_EQ(obs.end, now);  // clamped
    EXPECT_EQ(obs.end - obs.start, duration);
}

// ordering regression: retire() runs from the scope destructor,
// strictly after `now`, and samples a host clock (>= now); the emitted end must
// not exceed retired_ts within the 10us async-copy-tracing tolerance. Pre-clamp,
// a verbatim future end landed tens of us after retirement (+75898 ns on gfx1201).
TEST(complete_signal_less_dispatch, emitted_end_never_lands_after_retirement)
{
    constexpr uint64_t now         = 1'000'000'000;
    constexpr uint64_t tolerance   = 10'000;
    constexpr uint64_t future_skew = 75'898;  // the observed i93 offender skew
    auto               obs         = observer{};
    auto               conv        = converter{true, /*epoch=*/0};
    uint64_t           retired_ts  = 0;
    uint64_t           host_clock  = now;
    auto               retire_fn   = [&obs, &retired_ts, &host_clock]() {
        ++obs.retires;
        retired_ts = ++host_clock;  // any value >= now, mirroring a later sample
    };
    auto outcome = run_complete_signal_less_dispatch(std::optional<uint64_t>{now - 5'000'000},
                                                     now + future_skew,
                                                     /*enqueue_ts=*/0,
                                                     now,
                                                     conv.fn(),
                                                     obs.emit_fn(),
                                                     retire_fn);
    ASSERT_EQ(outcome, finalize_outcome::result_ready);
    ASSERT_EQ(obs.emits, 1);
    ASSERT_EQ(obs.retires, 1);
    EXPECT_LE(obs.end, retired_ts + tolerance);
}

// A throwing client callback must not skip cleanup: the scope destructor retires
// exactly once on the way out.
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

// Each rejection cause must be attributed correctly: mislabelled counters send the
// next investigation the wrong way. The id retires exactly once regardless.
TEST(complete_signal_less_dispatch, reports_the_exact_rejection_cause)
{
    constexpr uint64_t now = 1'000'000'000;
    auto               run = [&](std::optional<uint64_t> start_ticks,
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
        EXPECT_EQ(obs.retires, 1);
        return std::make_pair(outcome, detail.reason);
    };
    struct row
    {
        std::optional<uint64_t> start;
        uint64_t                end, enqueue;
        bool                    convert_ok;
        finalize_outcome        outcome;
        finalize_reason         reason;
        const char*             label;
    };
    const row rows[] = {
        {now - 5'000'000,
         now - 1'000'000,
         0,
         true,
         finalize_outcome::result_ready,
         finalize_reason::ready,
         "success -> ready + emit"},
        {std::nullopt,
         now - 1'000'000,
         0,
         true,
         finalize_outcome::result_ready,
         finalize_reason::start_unknown,
         "shape ii: start lost -> emitted, end-anchored"},
        {now - 5'000'000,
         now - 1'000'000,
         0,
         false,
         finalize_outcome::completed_no_timing,
         finalize_reason::convert_failed,
         "conversion refused"},
        {now - 1'000'000,
         now - 5'000'000,
         0,
         true,
         finalize_outcome::result_ready,
         finalize_reason::bad_interval,
         "inverted interval -> swapped + emitted"},
        // A before-enqueue skew whose 1 ms interval FITS the 2.5 ms window shifts
        // up and emits; the finalizer only rejects when the interval cannot fit.
        {now - 3'000'000,
         now - 2'000'000,
         now - 2'500'000,
         true,
         finalize_outcome::result_ready,
         finalize_reason::before_enqueue,
         "starts before enqueue -> shifted + emitted (fits window)"},
        {1,
         now + kKfdFutureSlackNs + 1,
         0,
         true,
         finalize_outcome::result_ready,
         finalize_reason::after_now,
         "end beyond now + slack -> clamped + emitted"},
    };
    for(const auto& tc : rows)
    {
        auto [outcome, reason] = run(tc.start, tc.end, tc.enqueue, tc.convert_ok);
        EXPECT_EQ(outcome, tc.outcome) << tc.label;
        EXPECT_EQ(reason, tc.reason) << tc.label;
    }
}

// The RAW terminal tick is admitted before the swap/shift.
// A grossly stale or grossly future end is REJECTED (stale_interval), not repaired
// into [enqueue, now] and emitted as fresh; the ms-scale resync skew is untouched.
TEST(complete_signal_less_dispatch, grossly_stale_or_future_raw_terminal_is_rejected)
{
    struct row
    {
        std::optional<uint64_t> start;
        uint64_t                end, enqueue, now;
        finalize_outcome        outcome;
        finalize_reason         reason;
        const char*             label;
    };
    const row rows[] = {
        // raw_end 4 s before enqueue -> stale (subtraction, no wrap).
        {std::nullopt,
         1'000'000'000,
         5'000'000'000,
         6'000'000'000,
         finalize_outcome::completed_no_timing,
         finalize_reason::stale_interval,
         "grossly stale"},
        // raw_end 4 s past now -> stale.
        {std::optional<uint64_t>{5'000'000'000},
         5'000'000'000,
         0,
         1'000'000'000,
         finalize_outcome::completed_no_timing,
         finalize_reason::stale_interval,
         "grossly future"},
        // exactly at the future bound (raw_end - now == kMaxFutureNs) is NOT
        // rejected (strict >), so it clamps in and emits.
        {std::optional<uint64_t>{1},
         2'000'000'000,
         0,
         1'000'000'000,
         finalize_outcome::result_ready,
         finalize_reason::after_now,
         "future bound is inclusive"},
    };
    for(const auto& tc : rows)
    {
        auto obs     = observer{};
        auto conv    = converter{true, /*epoch=*/0};  // passthrough: raw_end == end
        auto detail  = finalize_detail{};
        auto outcome = run_complete_signal_less_dispatch(tc.start,
                                                         tc.end,
                                                         tc.enqueue,
                                                         tc.now,
                                                         conv.fn(),
                                                         obs.emit_fn(),
                                                         obs.retire_fn(),
                                                         &detail);
        EXPECT_EQ(outcome, tc.outcome) << tc.label;
        EXPECT_EQ(detail.reason, tc.reason) << tc.label;
        EXPECT_EQ(obs.retires, 1) << tc.label;  // proven completion always retires once
    }
}

// A few ms past now stays inside the slack: reported ready, NOT counted against
// after_now, else the breakdown blames the guard for the skew it absorbs.
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

// UINT64_MAX-adjacent grid: the finalizer's bounds and shifts are subtraction- and
// checked-add-based, so extreme tick/enqueue/now values must never wrap into a
// bogus emitted interval. Invariants for EVERY combination: retire exactly once; a
// result_ready always satisfies enqueue <= start < end <= now; nothing is emitted
// otherwise. (UBSan additionally guards the arithmetic under the sanitizer build.)
TEST(complete_signal_less_dispatch, uint64_max_adjacent_inputs_never_wrap)
{
    constexpr uint64_t MAX   = UINT64_MAX;
    const uint64_t     pts[] = {0, 1, 2, 1'000, MAX - 2, MAX - 1, MAX};
    auto               conv  = converter{true, /*epoch=*/0};  // passthrough
    auto check = [](const observer& o, finalize_outcome oc, uint64_t enqueue, uint64_t now) {
        EXPECT_EQ(o.retires, 1) << "retire exactly once on every path";
        if(oc == finalize_outcome::result_ready)
        {
            EXPECT_EQ(o.emits, 1);
            EXPECT_LE(enqueue, o.start);
            EXPECT_LT(o.start, o.end);
            EXPECT_LE(o.end, now);
        }
        else
        {
            EXPECT_EQ(o.emits, 0);
        }
    };

    for(uint64_t end : pts)
        for(uint64_t enqueue : pts)
            for(uint64_t now : pts)
            {
                // Known-START (both edges present).
                for(uint64_t start : pts)
                {
                    auto obs = observer{};
                    auto oc  = run_complete_signal_less_dispatch(std::optional<uint64_t>{start},
                                                                end,
                                                                enqueue,
                                                                now,
                                                                conv.fn(),
                                                                obs.emit_fn(),
                                                                obs.retire_fn());
                    check(obs, oc, enqueue, now);
                }
                // Start-unknown (shape ii) at the same boundaries.
                auto obs = observer{};
                auto oc  = run_complete_signal_less_dispatch(
                    std::nullopt, end, enqueue, now, conv.fn(), obs.emit_fn(), obs.retire_fn());
                check(obs, oc, enqueue, now);
            }
}
