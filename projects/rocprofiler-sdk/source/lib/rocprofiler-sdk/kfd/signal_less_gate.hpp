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
    entry_registered = 0,  // pending entries the hub accepted
    eop_proven,            // firmware EOP claimed a pending entry
    eop_unmatched,         // firmware EOP found no pending entry (key mismatch?)
    finalizer_emitted,     // RESULT_READY: record emitted with KFD timestamps
    finalizer_no_timing,   // COMPLETED_NO_TIMING: retired, no record
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

// Bounded wait for the reader to pair firmware records still outstanding on a
// closing queue's slot. Called between begin_ and finish_close, holding NO lock.
// `wait_hw_drained` runs FIRST and must block until the hardware has consumed
// everything submitted: waiting on records the GPU has not produced yet would
// strand dispatches whose kernels are simply still running. Injected to keep
// this free of the HSA layer. Returns how many were still pending at give-up.
size_t
drain_close_signal_less_queue(uint64_t                             queue_token,
                              const std::function<bool(uint64_t)>& wait_hw_drained);

// Whether to enable HW profiling lazily (on a queue's first SIGNAL-path batch)
// instead of at queue creation. Tied to the feature being fully active: with
// signal-less off every batch takes the signal path, so deferring would only
// move a call that always happens, and keeping create-time enable untouched
// makes flag-off behavior byte-identical.
bool
signal_less_lazy_profiling();
}  // namespace kfd
}  // namespace rocprofiler
