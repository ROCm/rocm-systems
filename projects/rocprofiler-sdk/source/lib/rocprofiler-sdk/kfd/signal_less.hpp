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
#include <optional>

// The owned payload the hub carries for each pending dispatch, plus the
// process-wide hub instance. Flag and eligibility live in signal_less_gate.hpp.

namespace rocprofiler
{
namespace context
{
struct correlation_id;
}  // namespace context

namespace kfd
{
// Held BY VALUE, with no raw `Queue&`, HSA signal handle or code-object pointer:
// the queue is a stable token and the agent a rocprofiler id, so the payload
// survives the queue being destroyed mid-flight. `correlation_id`'s reference
// was taken at enqueue; releasing it is the finalizer's job.
struct pending_payload
{
    using callback_record_t = rocprofiler_callback_tracing_kernel_dispatch_data_t;

    callback_record_t        callback_record = {};
    tracing::tracing_data    tracing_data    = {};
    context::correlation_id* correlation_id  = nullptr;
    rocprofiler_thread_id_t  tid             = 0;
    rocprofiler_agent_id_t   agent_id        = {};
    uint64_t                 enqueue_ts      = 0;
};

using signal_less_hub_t = DispatchHub<pending_payload>;

// Process-wide hub. Backed by common::static_object for ordered teardown.
signal_less_hub_t&
signal_less_hub();

// LOCK ORDERING: the registry lock is never held while the hub lock is taken,
// or vice versa -- callers take one, release it, then take the other.
OwnerRegistry&
owner_registry();

// Registers ownership; on discovering a second live owner it quarantines the
// slot in the hub (leaking that slot's pending entries) AFTER releasing the
// registry lock.
void
add_live_queue(uint64_t queue_token, uint32_t gpu_id, std::optional<uint32_t> doorbell_slot);

void
remove_live_queue(uint64_t queue_token);

// Generation/reuse closure on destroy, step 1 of 2: stop new reservations on the
// slot, before the caller fences the queue's gate_lock. Takes and RELEASES the
// hub lock -- the enqueue path holds gate_lock while taking the hub lock, so
// holding it across the fence would invert that order and deadlock.
void
begin_close_signal_less_queue(uint64_t queue_token);

// Step 2, after the fence: strand whatever is still pending (no record, no
// retire) and quarantine the slot permanently, so a queue that reuses the
// doorbell is signal-path-only. Must be called with NO lock held.
void
finish_close_signal_less_queue(uint64_t queue_token);

// Bridge to the HSA interposition layer, defined in queue_interposition.cpp.
// Everything in this file is linked into the same object library, so these are
// direct calls rather than installed function pointers.

// Hand a proven completion to the async task group. Moves out of `p` ONLY on
// true; false means the task group is gone, which only happens once
// finalization has begun.
bool
submit_no_signal_finalize(signal_less_hub_t::proven& p);

// Run the finalizer on the CALLING thread. Only the deferred flush uses this,
// and only from the teardown thread -- never the reader or processor.
void
finalize_no_signal_in_place(signal_less_hub_t::proven&& p);

// Take and release every live queue's gate_lock. Caller holds no other lock.
void
quiesce_signal_less_interceptor();

// Wait for every already-submitted completion to finish executing.
void
join_signal_less_tasks();

// Processor-side handoff for a proven completion. Submits to the task group;
// if it is gone, defers the entry for the teardown thread. Never runs a client
// callback on the caller's thread.
void
hand_off_proven(signal_less_hub_t::proven&& p);

// Finalize every deferred completion on the CALLING thread. Returns how many.
size_t
flush_deferred_completions();

}  // namespace kfd
}  // namespace rocprofiler
