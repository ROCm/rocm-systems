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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

// The no-signal finalizer core and the bounded retry owner, both free of the HSA
// and tracing headers so every branch is unit-testable with injected converters,
// emitters and retirement observers (test seams S2, S3, S7).
//
// The finalizer runs on a task-group worker -- or, for a permanently rejected
// submission, on the thread that flushes the retry owner. It NEVER runs on the
// reader thread and never under a hub or queue lock.

namespace rocprofiler
{
namespace kfd
{
// The two terminal outcomes of a proven completion. Both retire the correlation
// id exactly once; they differ only in whether a record is emitted. Neither can
// become LEAKED -- the kernel is done (G3).
enum class finalize_outcome
{
    result_ready,         // KFD timestamps emitted
    completed_no_timing,  // no record; start unknown or convert/sanity failed
};

// Why a completion produced no record. Reported so a no-timing spike can be
// attributed without a rebuild: shape-ii (the START was lost) and a rejected
// sanity clause are entirely different bugs with entirely different fixes.
enum class finalize_reason
{
    ready = 0,       // RESULT_READY: record emitted
    start_unknown,   // shape ii -- EOP proved completion but its START was lost
    convert_failed,  // hsa_amd_profiling_convert_tick_to_system_domain said no
    bad_interval,    // converted start >= end
    before_enqueue,  // converted start precedes this dispatch's own enqueue
    after_now,       // converted end is beyond now + the conversion slack
};

// Everything the finalizer learned, for diagnostics.
struct finalize_detail
{
    finalize_reason reason    = finalize_reason::ready;
    uint64_t        start_ns  = 0;
    uint64_t        end_ns    = 0;
    bool            converted = false;
};

enum class submit_result
{
    accepted,
    rejected_temporary,  // executor busy: hold and retry
    rejected_permanent,  // executor closing: never retry, finalize in place
};

// Decide the outcome and produce system-domain timestamps.
//
// `convert(ticks, &ns)` returns false when the tick conversion fails. Conversion
// failure is NOT a loss: the EOP already proved the kernel finished, so the entry
// still retires -- it just emits no record.
template <typename ConvertFn>
finalize_outcome
resolve_finalize(const std::optional<uint64_t>& start_ticks,
                 uint64_t                      end_ticks,
                 uint64_t                      enqueue_ts,
                 uint64_t                      now_ns,
                 ConvertFn&&                   convert,
                 uint64_t*                     start_ns_out,
                 uint64_t*                     end_ns_out,
                 finalize_detail*              detail = nullptr)
{
    auto _note = [detail](finalize_reason r) {
        if(detail) detail->reason = r;
        return finalize_outcome::completed_no_timing;
    };

    // Shape (ii): the EOP proved completion but its START was lost, so there is no
    // interval to report.
    if(!start_ticks) return _note(finalize_reason::start_unknown);

    if(!convert(*start_ticks, start_ns_out) || !convert(end_ticks, end_ns_out))
        return _note(finalize_reason::convert_failed);

    if(detail)
    {
        detail->converted = true;
        detail->start_ns  = *start_ns_out;
        detail->end_ns    = *end_ns_out;
    }

    // Same correlation guard the signal path uses, reported clause by clause: a
    // record that does not fall inside this dispatch's own CPU window is not this
    // dispatch's record.
    if(!(*start_ns_out < *end_ns_out)) return _note(finalize_reason::bad_interval);
    if(*start_ns_out < enqueue_ts) return _note(finalize_reason::before_enqueue);
    if(*end_ns_out > now_ns + kKfdFutureSlackNs) return _note(finalize_reason::after_now);

    return finalize_outcome::result_ready;
}

// Run the finalizer: convert, emit on success, and retire EXACTLY ONCE.
//
// Contains no HSA fallback by design -- the app may already have destroyed its
// completion signal, so there is nothing left to fall back to. `retire` runs from
// a scope destructor so a throwing emit() cannot skip correlation-id and
// queue-counter cleanup.
template <typename ConvertFn, typename EmitFn, typename RetireFn>
finalize_outcome
run_no_signal_finalizer(const std::optional<uint64_t>& start_ticks,
                        uint64_t                      end_ticks,
                        uint64_t                      enqueue_ts,
                        uint64_t                      now_ns,
                        ConvertFn&&                   convert,
                        EmitFn&&                      emit,
                        RetireFn&&                    retire,
                        finalize_detail*              detail = nullptr)
{
    auto _cleanup = common::scope_destructor{[&retire]() { retire(); }};

    uint64_t _start_ns = 0;
    uint64_t _end_ns   = 0;
    auto     _outcome  = resolve_finalize(
        start_ticks, end_ticks, enqueue_ts, now_ns, convert, &_start_ns, &_end_ns, detail);

    if(_outcome == finalize_outcome::result_ready) emit(_start_ns, _end_ns);
    return _outcome;
}

// Bounded holder for completions the task group would not take.
//
// An EOP-proven entry can never revert to PENDING and never becomes LEAKED, so
// once the reader has claimed it, it MUST eventually be finalized. This owner is
// what guarantees that when the executor rejects a submission.
//
// THREADING: the mutex protects the held vector and the closed flag only. submit
// and finalize callables -- which run client callbacks -- are always invoked with
// NO lock held (invariant 11).
template <typename ProvenT>
class retry_owner
{
public:
    // Deliberately generous: the realistic rejection window is a closing executor
    // during teardown, which holds a handful of entries.
    static constexpr size_t kMaxHeld = 1024;

    // Take ownership of a rejected completion.
    //
    // `finalize_in_place(ProvenT&&)` is used ONLY if the owner is already at
    // capacity, which requires kMaxHeld consecutive rejections. Callers on the
    // reader thread must therefore treat that as the pathological case it is: it
    // is the one path where a callback could run on the caller's thread, so it is
    // logged by the caller.
    template <typename FinalizeFn>
    bool hold(ProvenT&& p, FinalizeFn&& finalize_in_place)
    {
        // A forked child drops the completion on the floor rather than touching the
        // inherited mutex or running a finalizer whose task group no longer exists.
        if(m_abandoned.load(std::memory_order_acquire)) return false;

        auto _overflow = std::optional<ProvenT>{};
        {
            auto lk = std::lock_guard<std::mutex>{m_mu};
            if(m_held.size() < kMaxHeld)
            {
                m_held.emplace_back(std::move(p));
                return true;
            }
            _overflow = std::move(p);
        }
        finalize_in_place(std::move(*_overflow));
        return false;
    }

    // Drain the owner on the CALLING thread: re-submit what the executor will
    // still take, finalize the rest in place. Returns how many entries it handled.
    //
    // `submit(ProvenT&)` must move out of its argument only when it returns
    // `accepted`; on any rejection the entry is left intact for finalization here.
    template <typename SubmitFn, typename FinalizeFn>
    size_t flush(SubmitFn&& submit, FinalizeFn&& finalize_in_place)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return 0;

        auto _taken = std::vector<ProvenT>{};
        bool _closed = false;
        {
            auto lk = std::lock_guard<std::mutex>{m_mu};
            _taken.swap(m_held);
            _closed = m_closed;
        }

        for(auto& _p : _taken)
        {
            if(!_closed && submit(_p) == submit_result::accepted) continue;
            finalize_in_place(std::move(_p));
        }
        return _taken.size();
    }

    // No further submissions will ever be accepted; everything held must be
    // finalized in place by the next flush.
    void close()
    {
        if(m_abandoned.load(std::memory_order_acquire)) return;
        auto lk  = std::lock_guard<std::mutex>{m_mu};
        m_closed = true;
    }

    bool closed() const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return true;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_closed;
    }

    size_t size() const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return 0;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_held.size();
    }

    // pthread_atfork child handler (requirement 8). One atomic store, checked
    // before m_mu by every operation above, so a forked child neither locks the
    // inherited mutex nor finalizes a completion into a task group whose workers
    // did not survive the fork.
    void abandon_in_child() { m_abandoned.store(true, std::memory_order_release); }

    bool abandoned() const { return m_abandoned.load(std::memory_order_acquire); }

private:
    std::atomic<bool>    m_abandoned = {false};
    mutable std::mutex   m_mu     = {};
    std::vector<ProvenT> m_held   = {};
    bool                 m_closed = false;
};
}  // namespace kfd
}  // namespace rocprofiler
