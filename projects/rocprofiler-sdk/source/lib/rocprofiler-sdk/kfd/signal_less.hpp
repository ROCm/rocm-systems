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

#include "lib/rocprofiler-sdk/kfd/dispatch_hub.hpp"
#include "lib/rocprofiler-sdk/kfd/no_signal_finalizer.hpp"
#include "lib/rocprofiler-sdk/kfd/owner_registry.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less_gate.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

// Signal-less kernel-dispatch completion: the owned payload the hub carries for
// each pending dispatch, plus the process-wide hub instance. The feature flag and
// the eligibility decision live in signal_less_gate.hpp (no tracing/HSA headers,
// so the decision table is unit-testable on its own).

namespace rocprofiler
{
namespace context
{
struct correlation_id;
}  // namespace context

namespace kfd
{
// Everything the no-signal finalizer needs to emit a record and retire the
// correlation id, held BY VALUE. Deliberately holds no raw `Queue&`, no HSA
// signal handle, and no code-object pointer (invariant 10): the queue is
// identified by a stable token and the agent by its rocprofiler id, so the
// payload stays valid even if the queue is destroyed while the dispatch is in
// flight. `correlation_id` is a refcount handle whose reference was already
// taken at enqueue; releasing it is the finalizer's job.
struct pending_payload
{
    using callback_record_t = rocprofiler_callback_tracing_kernel_dispatch_data_t;

    callback_record_t        callback_record = {};
    tracing::tracing_data    tracing_data    = {};
    context::correlation_id* correlation_id  = nullptr;
    rocprofiler_thread_id_t  tid             = 0;
    rocprofiler_agent_id_t   agent_id        = {};
    uint64_t                 enqueue_ts      = 0;
    uint64_t                 queue_token     = 0;
    uint64_t                 submit_index    = 0;
};

using signal_less_hub_t = DispatchHub<pending_payload>;

// Process-wide hub. Backed by common::static_object for ordered teardown.
signal_less_hub_t&
signal_less_hub();

// Live doorbell ownership across every compute queue the SDK knows about, and the
// per-queue lazy-profiling bookkeeping. Both are process-wide.
//
// LOCK ORDERING: the registry lock is never held while the hub lock is taken, and
// vice versa -- callers take one, release it, then take the other.
OwnerRegistry&
owner_registry();

ProfilingEnableTracker&
profiling_tracker();

// Queue lifecycle hooks. add_live_queue() registers ownership and, when it
// discovers a second live owner, quarantines the slot in the hub (leaking that
// slot's pending entries per P1) AFTER releasing the registry lock.
void
add_live_queue(uint64_t queue_token, uint32_t gpu_id, std::optional<uint32_t> doorbell_slot);

void
remove_live_queue(uint64_t queue_token);

// Generation/reuse closure on queue destroy (design requirement 4), split into
// the two halves the lock ordering demands.
//
// STEP 1, before the caller fences the queue's gate_lock: stop new signal-less
// reservations on the slot. Takes and RELEASES the hub lock, so the caller never
// holds it while waiting on gate_lock -- the enqueue path holds gate_lock while
// taking the hub lock, so the reverse nesting would deadlock.
void
begin_close_signal_less_queue(uint64_t queue_token);

// STEP 2, after the fence: strand whatever is still pending on the slot (P1, no
// record and no retire) and quarantine it permanently, so a queue that reuses the
// doorbell is signal-path-only and a stale late record can never be matched to
// it. Must be called with NO lock held.
void
finish_close_signal_less_queue(uint64_t queue_token);

// How the KFD reader reaches the completion machinery without depending on the
// HSA interposition layer (which in turn depends on the hub). The interposition
// layer installs these once at init, before any queue -- and therefore any
// eligible batch -- can exist.
struct signal_less_ops
{
    // Hand a proven completion to the async task group. Must move out of `p` ONLY
    // when returning `accepted`; on rejection the entry is left intact so the
    // retry owner can finalize it later.
    std::function<submit_result(signal_less_hub_t::proven&)> submit = {};

    // Run the no-signal finalizer synchronously on the CALLING thread. Used only
    // by the retry-owner flush, which runs on the teardown thread -- never on the
    // reader thread.
    std::function<void(signal_less_hub_t::proven&&)> finalize_in_place = {};
};

void
install_signal_less_ops(signal_less_ops ops);

// Reader-side handoff for a proven completion. Submits to the task group and,
// if the executor refuses, parks the entry in the bounded retry owner. It never
// runs a client callback on the caller's thread (invariant 11) except in the
// documented retry-owner overflow case.
void
hand_off_proven(signal_less_hub_t::proven&& p);

// Drain the retry owner on the CALLING thread: re-submit what the executor still
// takes, finalize the rest in place. Teardown step 4 calls this; it is defined
// here so no EOP-proven completion can be dropped in the meantime.
size_t
flush_retry_owner();

// Diagnostics / tests.
size_t
retry_owner_size();

}  // namespace kfd
}  // namespace rocprofiler
