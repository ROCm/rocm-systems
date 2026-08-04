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
#include <string_view>

// Signal-less kernel-dispatch completion: the feature flag and the per-batch
// eligibility decision.
//
// Deliberately free of the SDK tracing/HSA headers so the decision table stays
// unit-testable on its own; the payload and the hub instance live in
// signal_less.hpp, which includes this.
//
// STATUS: the machinery is being landed in units. signal_less_fully_wired() is
// the master switch and stays false until the last one lands, so nothing here is
// observable yet regardless of the env flag. See the design plan's "Phasing".

namespace rocprofiler
{
namespace kfd
{
// ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS, read once and cached. Defaults OFF:
// only an explicit enable turns it on, so a typo or an unrelated value can never
// activate the feature.
bool
signal_less_feature_enabled();

// Whether this correlation id was deliberately leaked by the signal-less loss
// policy, in which case correlation_id_finalize() must NOT force-retire it: its
// kernel may still be running and its references were intentionally not dropped.
//
// Declared here, in the header with no tracing/HSA dependencies, so the
// correlation-id finalize path can call it without pulling the hub in. It is a
// single acquire load returning false until something is actually leaked, which
// with signal-less off is never -- so the finalize path is untouched.
bool
signal_less_id_is_leaked(uint64_t correlation_id);

// Record that the loss ledger is now non-empty. Called by the loss paths only.
void
note_signal_less_losses();

// Parse helper for the above; pure so the accepted spellings are unit-testable.
inline bool
parse_signal_less_env(std::string_view v)
{
    return v == "1" || v == "on" || v == "ON" || v == "true" || v == "TRUE" || v == "yes";
}

// Whether the signal-less machinery is COMPLETE in this build: the hub, the
// enqueue-side eligibility and P3 packet no-touch, the reader handoff and
// no-signal finalizer, the overrun loss ledger, the live-owner registry and
// collision quarantine, the generation/reuse closure, the strict teardown order,
// and the fork epoch. All of it is present, so this is true.
//
// It is NOT the on switch. Activation still requires the operator to opt in with
// ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS=1 (signal_less_feature_enabled()); with
// the variable unset -- the default -- every dispatch keeps the signal path and
// HSA timestamps exactly as before. This stays separate from the env flag so a
// build that ever ships the machinery incomplete can force it off in one place.
constexpr bool
signal_less_fully_wired()
{
    return true;
}

// Per-BATCH eligibility (design plan "Eligibility"). Every condition must hold;
// if ANY packet in the batch fails, the WHOLE batch keeps the signal path -- there
// are no mixed-mode batches. App-signal presence is deliberately NOT an input (P3).
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

// Signal-less teardown, steps 1-6 of design requirement 7, in the one order that
// leaves nothing stranded. Called at the very start of finalization, BEFORE the
// existing queue_controller_fini / kfd::finalize / correlation_id_finalize
// sequence (which is step 7).
//
// No-op unless signal-less is actually active: with the feature off there are no
// hub entries, no retry-owner work and no reader->task handoff, so the ordering
// constraint does not apply and the finalize path is left exactly as it was.
void
signal_less_teardown();

// Hub-aware synchronization point for code-object unload, queue-controller sync
// and client detach. Fences (a) interceptor registration/publication, (b) reader
// drain/result production, (c) ready-task handoff and (d) task execution -- so a
// completion cannot run against a queue, context or code object that the caller
// is about to tear down. Unlike the teardown above, this does NOT stop the reader
// or the hub: the process keeps running afterwards.
//
// Must be called holding NO lock -- in particular not a queue gate_lock, which it
// acquires, and not the hub lock.
void
signal_less_quiesce();

// Observability for the signal-less path. Every stage of
// eligible-batch -> registered -> EOP proven -> handed off -> finalized bumps a
// counter, so a break anywhere in the chain is visible in one log line instead of
// requiring a rebuild. Bumps are skipped entirely unless signal-less is active,
// so the default path pays one predictable branch.
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
    uint64_t finalizer_emitted   = 0;
    uint64_t finalizer_no_timing = 0;
};

void
note_signal_less(signal_less_counter which, uint64_t n = 1);

signal_less_counters
signal_less_stats();

// pthread_atfork CHILD handler entry point (design requirement 8).
//
// RESTRICTED CONTEXT: this runs in a forked child where only the forking thread
// survives, so it performs atomic scalar stores ONLY -- no mutex, no allocation,
// no map access, no logging, no join. It marks the signal-less epoch stale and
// abandons each Phase-2 shared object that ALREADY EXISTS; it never constructs
// one, because construction would allocate.
void
signal_less_abandon_in_child();

// True in a process that inherited signal-less state across a fork. Every entry
// point below and in signal_less.hpp tests this before touching anything.
bool
signal_less_child_stale();

// Bounded wait for the reader to pair the firmware records still outstanding on a
// closing queue's doorbell slot, so a destroy does not discard dispatches whose
// EOP is merely in flight. Called between begin_ and finish_close, holding NO
// lock. Returns how many were still pending when it gave up.
size_t
drain_close_signal_less_queue(uint64_t queue_token);

// Whether HW profiling should be enabled lazily (only on a queue's first
// SIGNAL-path batch) instead of at queue creation.
//
// Deliberately tied to the feature being fully active: while signal-less is off,
// EVERY batch takes the signal path, so deferring the enable would only move a
// call that always happens anyway -- with a real risk that some path needs
// profiling on before its first completion. Keeping the create-time enable
// untouched in that case makes the flag-off behavior byte-identical.
bool
signal_less_lazy_profiling();
}  // namespace kfd
}  // namespace rocprofiler
