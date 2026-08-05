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
#include "lib/rocprofiler-sdk/kfd/teardown.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstddef>
#include <cstdint>
#include <functional>
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
    uint64_t                 queue_token     = 0;
    uint64_t                 submit_index    = 0;
};

using signal_less_hub_t = DispatchHub<pending_payload>;

// Process-wide hub. Backed by common::static_object for ordered teardown.
signal_less_hub_t&
signal_less_hub();

// LOCK ORDERING: the registry lock is never held while the hub lock is taken,
// or vice versa -- callers take one, release it, then take the other.
OwnerRegistry&
owner_registry();

ProfilingEnableTracker&
profiling_tracker();

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

// How the reader reaches the completion machinery without depending on the HSA
// interposition layer (which depends on the hub). Installed once at init,
// before any queue -- and therefore any eligible batch -- can exist.
struct signal_less_ops
{
    // Must move out of `p` ONLY when returning `accepted`; on rejection the
    // entry is left intact for the retry owner.
    std::function<submit_result(signal_less_hub_t::proven&)> submit = {};

    // Runs on the CALLING thread. Used only by the retry-owner flush, which is
    // the teardown thread -- never the reader.
    std::function<void(signal_less_hub_t::proven&&)> finalize_in_place = {};

    // Takes and releases every live queue's gate_lock. Must be called holding no
    // other lock.
    std::function<void()> quiesce_interceptor = {};

    // Wait for every already-submitted completion to finish executing.
    std::function<void()> join_task_group = {};
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
flush_retry_owner_now();

// Diagnostics / tests.
size_t
retry_owner_size();


}  // namespace kfd
}  // namespace rocprofiler
