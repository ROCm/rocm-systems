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
// Tags identifying the producer subsystem of each inst_pkt_t entry.
// Only distinctness matters; values are fixed for test-order stability.
// Services are migrated off the per-queue callback registry one at a time; the
// unused ids are reserved so the tag values stay stable across those PRs.
//
// The underlying type is fixed to int64_t to match hsa::ClientID, which is how the
// tag is stored in inst_pkt_t. The enum is deliberately unscoped rather than an
// enum class: unscoped keeps the implicit conversion to ClientID, so the emplace
// sites need no cast, while still grouping the values under one type.
//
// These values numerically overlap the per-queue registry's auto-incrementing
// ClientID, which also starts at 1. On the individual service branches that overlap
// is inert only because no consumer routes on the tag; here all four services are
// migrated, so QueueController::add_callback has no remaining callers and the
// registry can no longer hand out a colliding id. Counters, thread trace and SPM
// each now dispatch on their own tag. Reintroducing a registry client would revive
// the collision, so move this enum to a range the registry cannot produce if that
// ever happens.
enum client_id : int64_t
{
    COUNTERS_CLIENT_ID     = 1,
    THREAD_TRACE_CLIENT_ID = 2,
    PC_SAMPLING_CLIENT_ID  = 3,
    SPM_CLIENT_ID          = 4,
};
}  // namespace queue_hooks
}  // namespace hsa
}  // namespace rocprofiler
