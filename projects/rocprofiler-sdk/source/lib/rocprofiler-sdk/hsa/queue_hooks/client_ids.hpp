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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <cstdint>

namespace rocprofiler
{
namespace hsa
{
namespace queue_hooks
{
// Tags identifying the producer subsystem of each inst_pkt_t entry, stored in the
// inst_pkt_t pair as an hsa::ClientID.
//
// The underlying type is fixed to int64_t to match hsa::ClientID. The enum is
// deliberately unscoped rather than an enum class: unscoped keeps the implicit
// conversion to ClientID, so the emplace sites need no cast, while still grouping
// the values under one type.
//
// The values are negative because they share the inst_pkt_t tag space with the
// per-queue callback registry, whose QueueController::add_callback hands out
// strictly positive ids counting up from 1. Migrated services (counters, thread
// trace, SPM) select their own packets by comparing against these tags, so a tag
// drawn from the registry's range could match a packet the service does not own.
// Services that have not migrated yet still receive registry ids, so the two
// ranges must stay disjoint until the registry is gone. Any tag added here must
// likewise be negative; -1 is left unused because QueueController::add_callback
// uses it as a local "unassigned" initializer.
enum client_id : int64_t
{
    COUNTERS_CLIENT_ID     = -1001,
    THREAD_TRACE_CLIENT_ID = -1002,
    PC_SAMPLING_CLIENT_ID  = -1003,
    SPM_CLIENT_ID          = -1004,
};

static_assert(COUNTERS_CLIENT_ID < 0 && THREAD_TRACE_CLIENT_ID < 0 && PC_SAMPLING_CLIENT_ID < 0 &&
                  SPM_CLIENT_ID < 0,
              "queue hook tags must stay disjoint from the positive ids handed out by "
              "QueueController::add_callback");
}  // namespace queue_hooks
}  // namespace hsa
}  // namespace rocprofiler
