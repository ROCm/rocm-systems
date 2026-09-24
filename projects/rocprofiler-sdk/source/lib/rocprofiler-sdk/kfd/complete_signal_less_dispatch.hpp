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

#pragma once

#include "lib/common/scope_destructor.hpp"
#include "lib/rocprofiler-sdk/kfd/correlation_types.hpp"

#include <cstdint>
#include <optional>
#include <utility>

// Completion of a signal-less dispatch, free of the HSA and tracing headers so
// every branch is unit-testable with injected callables. Runs on a task-group
// worker, or on the teardown thread draining deferred completions -- never on
// the reader or processor thread, and never under a hub/queue lock.

namespace rocprofiler
{
namespace kfd
{
// The two terminal outcomes of a proven completion. Both retire the correlation
// id; they differ only in whether a record is emitted.
enum class finalize_outcome
{
    result_ready,         // KFD timestamps emitted
    completed_no_timing,  // no record; start unknown or convert/sanity failed
};

// Why a completion ended up with no timing, reported so a no-timing spike can be
// attributed without a rebuild.
enum class finalize_reason
{
    ready = 0,       // RESULT_READY: record emitted
    start_unknown,   // shape ii -- EOP proved completion but its START was lost
    convert_failed,  // hsa_amd_profiling_convert_tick_to_system_domain said no
    bad_interval,    // start >= end, or the repaired interval cannot fit [enqueue, now]
    before_enqueue,  // converted start precedes this dispatch's own enqueue
    after_now,       // converted end is beyond now + the conversion slack
    stale_interval,  // the RAW terminal tick is grossly older/newer than the run
};

// bounds on the RAW converted EOP terminal tick (before any swap or
// shift), rejecting a grossly stale or grossly future record rather than dragging
// it into [enqueue, now] and emitting it as fresh. Chosen from the tick-conversion
// resync magnitude (2.0-2.7 ms observed) with generous margin -- these are seconds,
// far above the ~100 ms kKfdFutureSlackNs the after_now clamp still absorbs. Both
// tests are written as subtractions so nothing wraps at the UINT64_MAX boundary.
constexpr uint64_t kMaxStaleNs  = 1'000'000'000;  // 1 s before enqueue
constexpr uint64_t kMaxFutureNs = 1'000'000'000;  // 1 s past now

// Everything the finalizer learned, for diagnostics.
struct finalize_detail
{
    finalize_reason reason    = finalize_reason::ready;
    uint64_t        start_ns  = 0;
    uint64_t        end_ns    = 0;
    bool            converted = false;
};

inline const char*
finalize_reason_name(finalize_reason r)
{
    switch(r)
    {
        case finalize_reason::start_unknown: return "start-unknown";
        case finalize_reason::convert_failed: return "convert-failed";
        case finalize_reason::bad_interval: return "bad-interval";
        case finalize_reason::before_enqueue: return "before-enqueue";
        case finalize_reason::after_now: return "after-now";
        case finalize_reason::stale_interval: return "stale-interval";
        case finalize_reason::ready: break;
    }
    return "ready";
}

// Decide the outcome and produce system-domain timestamps.
template <typename ConvertFn>
finalize_outcome
resolve_finalize(const std::optional<uint64_t>& start_ticks,
                 uint64_t                       end_ticks,
                 uint64_t                       enqueue_ts,
                 uint64_t                       now_ns,
                 ConvertFn&&                    convert,
                 uint64_t*                      start_ns_out,
                 uint64_t*                      end_ns_out,
                 finalize_detail*               detail = nullptr)
{
    auto _note = [detail](finalize_reason r) {
        if(detail) detail->reason = r;
        return finalize_outcome::completed_no_timing;
    };
    // Records the first repair reason for diagnostics without changing the outcome.
    auto _flag = [detail](finalize_reason r) {
        if(detail && detail->reason == finalize_reason::ready) detail->reason = r;
    };

    // The EOP end tick is always present and is a real GPU timestamp, so convert it
    // first: it anchors the record on the timeline even when the START was lost.
    if(!convert(end_ticks, end_ns_out)) return _note(finalize_reason::convert_failed);

    // raw-terminal admission, BEFORE the swap and before any shift.
    // A stale end paired with a fresh start would be hidden by the post-swap
    // placement, and the far-future flag alone would drag an arbitrarily old record
    // into [enqueue, now]. Both bounds are subtractions, so nothing wraps.
    const uint64_t raw_end_ns = *end_ns_out;
    if(enqueue_ts > raw_end_ns && enqueue_ts - raw_end_ns > kMaxStaleNs)
        return _note(finalize_reason::stale_interval);
    if(raw_end_ns > now_ns && raw_end_ns - now_ns > kMaxFutureNs)
        return _note(finalize_reason::stale_interval);

    // Shape (ii): the EOP proved completion but its START was lost
    // (a firmware START that lost the HWS-connect race, or one overwritten by a ring
    // lap). The dispatch still ran and we know exactly when it finished, so anchor a
    // record at that end -- start seeded to end, widened to a representable interval
    // just below -- rather than dropping the row. The duration is the one unknown.
    if(!start_ticks)
    {
        _flag(finalize_reason::start_unknown);
        *start_ns_out = *end_ns_out;
    }
    else if(!convert(*start_ticks, start_ns_out))
    {
        return _note(finalize_reason::convert_failed);
    }

    // Distinct firmware ticks can convert to the same nanosecond for a kernel short
    // enough that the conversion quantizes the interval away; widen it to the
    // smallest representable one and emit. A start-unknown record
    // (no real start tick) is widened the same way. Identical raw ticks -- firmware
    // reported no duration at all -- are deliberately left equal and fall through to
    // the zero-duration guard below.
    if(*start_ns_out == *end_ns_out)
    {
        if(!start_ticks)
            // Start-unknown: preserve the real END (the EOP tick) and move the
            // synthesized start back by one ns.
            *start_ns_out = (*end_ns_out > 0) ? *end_ns_out - 1 : 0;
        else if(*start_ticks != end_ticks)
            // Distinct ticks quantized to the same ns: widen forward by one ns.
            *end_ns_out = *start_ns_out + 1;
        // Identical raw ticks are left equal and drop at the zero-duration guard.
    }

    // From here a skewed-but-nonzero interval is REPAIRED exactly the way the signal
    // path repairs its own in tracing::adjust_profiling_time, and emitted -- never
    // dropped, which was the record loss behind the no-timing class. Each repair is
    // noted in `detail` so a spike stays attributable.

    // Clock-skew inversion: the signal path swaps rather than drops.
    if(*start_ns_out > *end_ns_out)
    {
        _flag(finalize_reason::bad_interval);
        std::swap(*start_ns_out, *end_ns_out);
    }

    // Tick-to-system-domain conversion re-syncs periodically, so a just-completed
    // dispatch's converted end can land a little after a `now` sampled right behind
    // it, and its start a little before this dispatch's own enqueue (both observed in
    // the low microseconds on gfx950). Mirror adjust_profiling_time: shift the whole
    // interval so it ends no later than now, then so it starts no earlier than
    // enqueue, preserving the measured duration. An end far past now is flagged stale
    // but treated the same; the correlation key already proved this record is ours.
    if(*end_ns_out > now_ns + kKfdFutureSlackNs) _flag(finalize_reason::after_now);
    if(*end_ns_out > now_ns)
    {
        const uint64_t _shift = *end_ns_out - now_ns;
        *end_ns_out           = now_ns;
        *start_ns_out         = (*start_ns_out > _shift) ? *start_ns_out - _shift : 0;
    }
    if(*start_ns_out < enqueue_ts)
    {
        _flag(finalize_reason::before_enqueue);
        const uint64_t _shift = enqueue_ts - *start_ns_out;
        // The shift-up addition is CHECKED, so nothing wraps.
        if(_shift > UINT64_MAX - *end_ns_out) return _note(finalize_reason::bad_interval);
        *start_ns_out += _shift;
        *end_ns_out += _shift;
    }

    // The single unconditional postcondition that makes the contract
    // true regardless of how the repairs interacted -- a genuinely zero-duration
    // interval, a shift-up that pushed end back past now, or a measured duration
    // exceeding now - enqueue that no shift can satisfy, all drop here.
    if(!(enqueue_ts <= *start_ns_out && *start_ns_out < *end_ns_out && *end_ns_out <= now_ns))
        return _note(finalize_reason::bad_interval);

    if(detail)
    {
        detail->converted = true;
        detail->start_ns  = *start_ns_out;
        detail->end_ns    = *end_ns_out;
    }

    return finalize_outcome::result_ready;
}

// Convert, emit on success, and retire EXACTLY ONCE on every path.
template <typename ConvertFn, typename EmitFn, typename RetireFn>
finalize_outcome
run_complete_signal_less_dispatch(const std::optional<uint64_t>& start_ticks,
                                  uint64_t                       end_ticks,
                                  uint64_t                       enqueue_ts,
                                  uint64_t                       now_ns,
                                  ConvertFn&&                    convert,
                                  EmitFn&&                       emit,
                                  RetireFn&&                     retire,
                                  finalize_detail*               detail = nullptr)
{
    auto _cleanup = common::scope_destructor{[&retire]() { retire(); }};

    uint64_t _start_ns = 0;
    uint64_t _end_ns   = 0;
    auto     _outcome  = resolve_finalize(
        start_ticks, end_ticks, enqueue_ts, now_ns, convert, &_start_ns, &_end_ns, detail);

    // Only result_ready has timestamps to emit. completed_no_timing is equally
    // terminal -- the scope destructor above retires the id either way -- it just
    // has no record to produce, so there is deliberately no branch for it.
    if(_outcome == finalize_outcome::result_ready) emit(_start_ns, _end_ns);
    return _outcome;
}

}  // namespace kfd
}  // namespace rocprofiler
