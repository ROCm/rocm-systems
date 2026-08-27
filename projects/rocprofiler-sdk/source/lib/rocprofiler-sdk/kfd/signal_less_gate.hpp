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

#include <array>
#include <cstdint>
#include <functional>

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

// Populate every env-backed knob's first-use cache from ONE serialized thread,
// before any worker/producer exists. Call sites read `environ` via
// common::get_env*, which is unsafe against a concurrent setenv. Idempotent.
void
signal_less_prime_env();

// True if the loss policy deliberately leaked this id, in which case
// correlation_id_finalize() must NOT force-retire it: its kernel may still be
// running and its references were intentionally not dropped.
bool
signal_less_id_is_leaked(uint64_t correlation_id);

// Record that the loss ledger is now non-empty. Called by the loss paths only.
void
note_signal_less_losses();

// Steps 1-6 of the teardown order, called BEFORE the existing
// queue_controller_fini / kfd::finalize / correlation_id_finalize sequence.
// No-op unless signal-less is active.
void
signal_less_teardown();

// The two-stage completion fence: on return, every kernel-dispatch
// completion whose firmware record had reached the ring before this call has been
// emitted or accounted, and no completion task is still executing. Does not stop
// the reader; on timeout it abandons signal-less process-wide. Must be
// called holding NO lock -- in particular not a queue gate_lock, which it
// acquires, and not the hub lock.
void
signal_less_fence_completions();

// Every stage of eligible-batch -> registered -> EOP proven -> handed off ->
// finalized bumps a counter, so a break in the chain is visible without a
// rebuild. Skipped entirely unless signal-less is active.
enum class signal_less_counter
{
    entry_registered = 0,  // pending entries the hub accepted
    eop_proven,            // firmware EOP claimed a pending entry
    eop_unmatched,         // firmware EOP found no pending entry (key mismatch?)
    finalizer_emitted,     // RESULT_READY: record emitted with KFD timestamps
    finalizer_no_timing,   // COMPLETED_NO_TIMING: retired, no record
    register_refused,      // hub refused a batch eligibility had accepted; id retired here
    cap_evicted,           // closed-window entry evicted under the per-GPU hub cap (D9)
    kCount
};

void
note_signal_less(signal_less_counter which, uint64_t n = 1);

// Snapshot indexed by signal_less_counter, so a new counter needs no mirror
// struct and no copy loop -- add an enumerator and a name and it prints.
using signal_less_counter_array =
    std::array<uint64_t, static_cast<size_t>(signal_less_counter::kCount)>;

signal_less_counter_array
signal_less_stats();

const char*
signal_less_counter_name(signal_less_counter which);

// pthread_atfork CHILD handler. RESTRICTED CONTEXT: only the forking thread
// survives, so this does atomic scalar stores ONLY -- no mutex, allocation, map
// access, logging or join. It never constructs a shared object, only abandons
// ones that already exist, because construction would allocate.
void
signal_less_abandon_in_child();

// True in a process that inherited signal-less state across a fork.
bool
signal_less_child_stale();

}  // namespace kfd
}  // namespace rocprofiler
