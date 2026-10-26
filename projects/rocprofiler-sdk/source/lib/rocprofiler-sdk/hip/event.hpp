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

// Decides how to submit a barrier packet. Intercepted when pkt may be the GPU barrier of a
// hipEventRecord on this thread (any barrier with a completion signal it submits; the record's
// is chosen when the call returns), is the barrier of a hipStreamWaitEvent on this thread
// (one depending on the awaited record's signal, or, when that record was not seen, any
// barrier with a completion signal, chosen like a record's), or when one of its dependency
// signals satisfies a hipStreamWaitEvent that CLR folded into it. An intercepted barrier gets
// a pooled completion signal and a handler that reports each of those events with the
// barrier's GPU start and end once it retires. Fires the ENTER/EXIT callbacks of the event
// call on this thread, before pkt is written, when pkt is the first barrier that can report
// the call (see active_event_context_t in event.cpp). All event bookkeeping stays inside
// event.cpp; the caller only emits packets.
//
// packet_type and is_barrier are supplied by the caller because AQL packet classification
// (including the vendor BARRIER_VALUE header decode) belongs to the queue layer, and
// duplicating it here would risk the two copies drifting apart.
barrier_plan_t
plan_barrier(hsa::Queue&                    queue,
             const hsa::rocprofiler_packet& pkt,
             uint32_t                       packet_type,
             bool                           is_barrier);

// For a dependency carried on a dispatch packet itself (the gfx12.5 ext kernel dispatch
// dep_signal) rather than on a barrier. When dep_signal satisfies a traced
// hipStreamWaitEvent, returns a barrier-AND packet on dep_signal, timed like an intercepted
// barrier, that the caller must submit ahead of the dispatch in place of its dep_signal.
// Returns nullopt otherwise, and the dispatch is submitted unchanged.
std::optional<hsa::rocprofiler_packet>
plan_dependency_barrier(hsa::Queue& queue, hsa_signal_t dep_signal);

}  // namespace event
}  // namespace hip
}  // namespace rocprofiler
