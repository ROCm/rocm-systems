// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include "lib/common/container/small_vector.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"

#include <unordered_map>

namespace rocprofiler
{
namespace context
{
struct context;
struct correlation_id;
}  // namespace context

namespace hsa
{
class Queue;

// Internal session information that is used by write interceptor
// to track state of the intercepted kernel.
struct queue_info_session
{
    using context_t              = context::context;
    using user_data_map_t        = std::unordered_map<const context_t*, rocprofiler_user_data_t>;
    using external_corr_id_map_t = user_data_map_t;
    using callback_record_t      = rocprofiler_callback_tracing_kernel_dispatch_data_t;
    using context_array_t        = common::container::small_vector<const context_t*>;

    Queue&                   queue;
    inst_pkt_t               inst_pkt         = {};
    hsa_signal_t             interrupt_signal = {};
    rocprofiler_thread_id_t  tid              = common::get_tid();
    rocprofiler_timestamp_t  enqueue_ts       = 0;
    rocprofiler_user_data_t  user_data        = {.value = 0};
    context::correlation_id* correlation_id   = nullptr;
    rocprofiler_packet       kernel_pkt       = {};
    callback_record_t        callback_record  = {};
    tracing::tracing_data    tracing_data     = {};
    bool                     is_serialized    = false;
};
}  // namespace hsa
}  // namespace rocprofiler
