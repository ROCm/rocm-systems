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

#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"
#include "lib/rocprofiler-sdk/kfd/results_map.hpp"

// Process-wide shared instances of the two KFD dispatch-log maps: DoorbellMap
// (queue <-> doorbell identity, used to build a correlation_key at enqueue) and
// ResultsMap (firmware timing keyed by that correlation_key). Each is a single
// object for the whole process (see DoorbellMap notes): the interceptor paths
// (enqueue) and the KFD reader thread operate on the same instances. Backed by
// common::static_object so teardown is ordered at library unload.

namespace rocprofiler
{
namespace kfd
{
// queue_id <-> doorbell_off (+ generation) translation, populated from SDK
// queue lifecycle. Read on the enqueue path, written on queue create/destroy.
DoorbellMap&
doorbell_map();

// firmware timing keyed by correlation_key. Deposited by the reader thread,
// taken in get_dispatch_time().
ResultsMap&
results_map();

// PHASE 1 OPTION (b) SWITCH -- the single gate for emitting KFD timestamps.
//
// Selecting a firmware record as a dispatch's timestamp is only sound once that
// record is provably from the current, uniquely-owning queue generation (design
// requirements 3 and 4: all-live-queue owner injectivity plus generation-reuse
// closure). Neither exists yet, and the CPU-window sanity guard is too broad to
// separate two colliding queues, so Phase 1 ships with selection OFF: every
// dispatch deterministically reports HSA timestamps and a late or misattributed
// record can no longer flip a dispatch onto a different clock source. The
// rendezvous is skipped entirely while this is false, so it costs nothing.
//
// Phase 2 flips this to true once the owner/generation tracking lands.
constexpr bool
kfd_selection_enabled()
{
    return false;
}
}  // namespace kfd
}  // namespace rocprofiler
