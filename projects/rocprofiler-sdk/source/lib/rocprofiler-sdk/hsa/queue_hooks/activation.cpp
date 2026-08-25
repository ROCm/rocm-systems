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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/queue_hooks/activation.hpp"

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/spm/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/thread_trace/queue_hooks.hpp"

namespace rocprofiler
{
namespace hsa
{
namespace queue_hooks
{
namespace
{
bool
kernel_tracing_active()
{
    auto active = context::get_active_contexts([](const context::context* ctx) -> bool {
        if(ctx == nullptr) return false;
        return (ctx->buffered_tracer &&
                ctx->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)) ||
               (ctx->callback_tracer &&
                ctx->callback_tracer->domains(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH));
    });
    return !active.empty();
}
}  // namespace

bool
any_consumer_active(rocprofiler_agent_id_t agent)
{
    return kernel_tracing_active() || counters::is_any_active() ||
           thread_trace::is_any_active() || spm::is_any_active() ||
           pc_sampling::is_configured_on_agent(agent);
}

bool
should_batch_packets()
{
    return !counters::is_any_active() && !thread_trace::is_any_active() && !spm::is_any_active();
}
}  // namespace queue_hooks
}  // namespace hsa
}  // namespace rocprofiler
