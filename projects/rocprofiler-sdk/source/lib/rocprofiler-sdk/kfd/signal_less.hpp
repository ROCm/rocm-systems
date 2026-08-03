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
#include "lib/rocprofiler-sdk/kfd/signal_less_gate.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstdint>

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

}  // namespace kfd
}  // namespace rocprofiler
