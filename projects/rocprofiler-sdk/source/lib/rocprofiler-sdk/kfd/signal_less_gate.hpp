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

#include <cstdint>
#include <functional>
#include <string_view>

// Signal-less kernel-dispatch completion: the feature flag and the per-batch
// eligibility decision. Deliberately free of the SDK tracing/HSA headers so the
// decision table stays unit-testable; the payload and hub live in signal_less.hpp.

namespace rocprofiler
{
namespace kfd
{
// ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS, read once and cached. Defaults OFF,
// so a typo or unrelated value can never activate the feature.
bool
signal_less_feature_enabled();

// True if the loss policy deliberately leaked this id, in which case
// correlation_id_finalize() must NOT force-retire it: its kernel may still be
// running and its references were intentionally not dropped.
bool
signal_less_id_is_leaked(uint64_t correlation_id);

// Record that the loss ledger is now non-empty. Called by the loss paths only.
void
note_signal_less_losses();

// Pure so the accepted spellings are unit-testable.
inline bool
parse_signal_less_env(std::string_view v)
{
    return v == "1" || v == "on" || v == "ON" || v == "true" || v == "TRUE" || v == "yes";
}

// Whether the machinery is COMPLETE in this build -- not the on switch.
// Activation still needs the env opt-in above. Kept separate so a build that
// ships the machinery incomplete can force it off in one place.
constexpr bool
signal_less_fully_wired()
{
    return true;
}

// Per-batch: if ANY packet fails, the WHOLE batch keeps the signal path -- there
// are no mixed-mode batches. App-signal presence is deliberately not an input.
struct eligibility_inputs
{
    bool feature_enabled       = false;  // env flag
    bool fully_wired           = false;  // signal_less_fully_wired()
    bool session_live_for_gpu  = false;  // a dlog session exists for THIS gpu
    bool reader_alive          = false;  // not poisoned / reader-dead
    bool doorbells_injective   = false;  // every packet's slot has one live owner
    bool hub_accepts_batch     = false;  // no live/tombstoned key, no quarantined slot
    bool payload_constructible = false;  // owned payload can be built for every packet
};

inline bool
batch_is_signal_less_eligible(const eligibility_inputs& in)
{
    return in.feature_enabled && in.fully_wired && in.session_live_for_gpu && in.reader_alive &&
           in.doorbells_injective && in.hub_accepts_batch && in.payload_constructible;
}

// Steps 1-6 of the teardown order, called BEFORE the existing
// queue_controller_fini / kfd::finalize / correlation_id_finalize sequence.
// No-op unless signal-less is active.
void
signal_less_teardown();

// Fences interceptor publication, reader drain, task handoff and task execution,
// so a completion cannot run against a queue/context/code object the caller is
// tearing down. Does not stop the reader. Must be called holding NO lock -- in
// particular not a queue gate_lock, which it acquires, and not the hub lock.
void
signal_less_quiesce();

// Every stage of eligible-batch -> registered -> EOP proven -> handed off ->
// finalized bumps a counter, so a break in the chain is visible without a
// rebuild. Skipped entirely unless signal-less is active.
enum class signal_less_counter
{
    batch_eligible = 0,   // a batch qualified and published its packets untouched
    entry_registered,     // pending entries the hub accepted
    register_refused,     // the hub refused a batch eligibility had accepted
    eop_proven,           // firmware EOP claimed a pending entry
    eop_unmatched,        // firmware EOP found no pending entry (key mismatch?)
    handoff_submitted,    // proven completion accepted by the task group
    handoff_retried,      // rejected; parked in the retry owner
    finalizer_emitted,    // RESULT_READY: record emitted with KFD timestamps
    finalizer_no_timing,  // COMPLETED_NO_TIMING: retired, no record
    // These sum to finalizer_no_timing: a lost START and a rejected sanity
    // clause are different bugs needing opposite fixes.
    no_timing_start_unknown,
    no_timing_convert_failed,
    no_timing_bad_interval,
    no_timing_before_enqueue,
    no_timing_after_now,
    kCount
};

struct signal_less_counters
{
    uint64_t batch_eligible      = 0;
    uint64_t entry_registered    = 0;
    uint64_t register_refused    = 0;
    uint64_t eop_proven          = 0;
    uint64_t eop_unmatched       = 0;
    uint64_t handoff_submitted   = 0;
    uint64_t handoff_retried     = 0;
    uint64_t finalizer_emitted        = 0;
    uint64_t finalizer_no_timing      = 0;
    uint64_t no_timing_start_unknown  = 0;
    uint64_t no_timing_convert_failed = 0;
    uint64_t no_timing_bad_interval   = 0;
    uint64_t no_timing_before_enqueue = 0;
    uint64_t no_timing_after_now      = 0;
};

void
note_signal_less(signal_less_counter which, uint64_t n = 1);

signal_less_counters
signal_less_stats();

// pthread_atfork CHILD handler. RESTRICTED CONTEXT: only the forking thread
// survives, so this does atomic scalar stores ONLY -- no mutex, allocation, map
// access, logging or join. It never constructs a shared object, only abandons
// ones that already exist, because construction would allocate.
void
signal_less_abandon_in_child();

// True in a process that inherited signal-less state across a fork.
bool
signal_less_child_stale();

// Bounded wait for the reader to pair firmware records still outstanding on a
// closing queue's slot. Called between begin_ and finish_close, holding NO lock.
// `wait_hw_drained` runs FIRST and must block until the hardware has consumed
// everything submitted: waiting on records the GPU has not produced yet would
// strand dispatches whose kernels are simply still running. Injected to keep
// this free of the HSA layer. Returns how many were still pending at give-up.
size_t
drain_close_signal_less_queue(uint64_t                                  queue_token,
                              const std::function<bool(uint64_t)>&      wait_hw_drained);

// Whether to enable HW profiling lazily (on a queue's first SIGNAL-path batch)
// instead of at queue creation. Tied to the feature being fully active: with
// signal-less off every batch takes the signal path, so deferring would only
// move a call that always happens, and keeping create-time enable untouched
// makes flag-off behavior byte-identical.
bool
signal_less_lazy_profiling();
}  // namespace kfd
}  // namespace rocprofiler
