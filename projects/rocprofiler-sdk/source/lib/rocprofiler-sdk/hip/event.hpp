// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace rocprofiler
{
namespace context
{
struct correlation_id;
}
namespace hsa
{
class Queue;
}
namespace hip
{
namespace event
{
const char*
name_by_id(uint32_t id);

std::vector<uint32_t>
get_ids();

template <typename TableT>
void
update_table(TableT* table);

// Records whether any tool has configured a HIP_EVENT service. Set when the callback or
// buffer tracing service is configured; gates the fast path in is_active().
void
set_service_configured(bool enabled);

// Single gate for the queue interceptor fast paths. True only when a HIP_EVENT service is
// configured, at least one context tracing it is started, and either a HIP event API call
// is active on this thread or a deferred wait is outstanding. Independent of the gates for
// other tracing features, which remain separately testable.
bool
is_active();

// What the queue interceptor should emit in place of a candidate barrier packet.
// When intercepted is false the caller emits the original packet unchanged.
struct barrier_plan_t
{
    bool                    intercepted    = false;
    hsa::rocprofiler_packet barrier        = {};
    bool                    has_forwarding = false;
    hsa::rocprofiler_packet forwarding     = {};
};

// Decides whether pkt is the GPU barrier produced by a hipEventRecord or
// hipStreamWaitEvent on this thread. When it is, substitutes a pooled completion signal,
// fires the ENTER/EXIT callbacks, and registers the handler that emits the record. All
// event bookkeeping stays inside this call; the caller only emits packets.
//
// is_barrier is supplied by the caller because AQL packet classification (including the
// vendor BARRIER_VALUE header decode) belongs to the queue layer, and duplicating it here
// would risk the two copies drifting apart.
barrier_plan_t
plan_barrier(hsa::Queue&                    queue,
             const hsa::rocprofiler_packet& pkt,
             bool                           is_barrier,
             const tracing::tracing_data&   tracing_data_v,
             context::correlation_id*       corr_id);

// A hipStreamWaitEvent's GPU dependency is often folded by CLR into a later packet's
// barrier rather than emitted as a standalone one. The interceptor reports candidate
// dependency signals here as it walks a packet batch; any wait registered against one is
// claimed and staged.
//
// Staging is thread-local and scoped to a single packet batch: every claim is followed by
// either bind_staged_waits or flush_staged_waits before the interceptor returns, so
// nothing carries over into the next batch on this thread.
void
claim_deferred_wait(uint64_t dep_signal_handle);

// Claims deferred waits carried as dependencies of a barrier packet, reading whichever
// dependency field matches the packet type. No-op when this barrier was itself the one
// intercepted for a record or wait, since that path reports its own completion.
//
// Must be called after plan_barrier for the same packet: the two share the per-thread
// record of whether that barrier was already captured.
void
claim_deferred_waits(const hsa::rocprofiler_packet& pkt, uint32_t packet_type, bool is_barrier);

// Attach the staged waits to a submission that already has a completion signal. The
// caller must invoke submission_complete for that signal before returning it to the
// signal pool, otherwise the pool can hand the same handle to another submission while
// waits are still bound to it.
void
bind_staged_waits(hsa_signal_t completion_signal);

// Used when a batch has staged waits but no kernel packet whose signal they can ride on.
// Allocates a signal, registers a private completion handler, and returns the barrier
// packet the caller should append. Returns nullopt when nothing is staged.
std::optional<hsa::rocprofiler_packet>
flush_staged_waits(hsa::Queue& queue, rocprofiler_timestamp_t enqueue_ts);

// Emit a WAIT record for every wait bound to this completion signal.
void
submission_complete(const hsa::Queue&       queue,
                    hsa_signal_t            completion_signal,
                    rocprofiler_timestamp_t enqueue_ts);

}  // namespace event
}  // namespace hip
}  // namespace rocprofiler
