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

#include <cstddef>
#include <cstdint>
#include <functional>

// Shared identity types that bridge an SDK dispatch to a firmware dispatch-log
// record. A firmware record identifies a dispatch as (doorbell_off,
// dispatch_idx_low32); the SDK captures the same pair at enqueue time. The
// generation field disambiguates a doorbell_off that is reused after its queue
// is destroyed and a new queue takes the same doorbell slot.

namespace rocprofiler
{
namespace kfd
{
// The full key that uniquely identifies one in-flight dispatch across the
// SDK/firmware boundary. All three fields are snapshotted at enqueue time so
// completion-time lookup cannot race with queue destroy/recreate.
struct correlation_key
{
    uint32_t doorbell_off       = 0;
    uint32_t dispatch_idx_low32 = 0;
    uint32_t generation         = 0;
    // The GPU whose firmware produced (or will produce) the record. Doorbell
    // slots and dispatch indices are per-GPU and both restart from low values, so
    // without this a record from one GPU could match a dispatch enqueued on
    // another and report its timestamps -- the wrong-dispatch completion the loss
    // policy forbids. Last field so existing three-field construction still means
    // "GPU 0".
    uint32_t gpu_id = 0;

    bool operator==(const correlation_key& rhs) const
    {
        return doorbell_off == rhs.doorbell_off && dispatch_idx_low32 == rhs.dispatch_idx_low32 &&
               generation == rhs.generation && gpu_id == rhs.gpu_id;
    }

    bool operator!=(const correlation_key& rhs) const { return !(*this == rhs); }
};

// How far past a CPU timestamp a converted firmware end may legitimately land.
//
// hsa_amd_profiling_convert_tick_to_system_domain maps GPU ticks onto the system
// domain through a correlation that is re-synced periodically, so the converted
// value and a CPU timestamp sampled at nearly the same instant disagree by that
// correlation's granularity and drift. Measured on gfx1201, a just-completed
// dispatch's converted end lands 2.0-2.7 ms AFTER a `now` sampled right behind
// it -- which a hard `end <= now` rejects outright, discarding every valid
// record. That skew is inherent to the conversion, not a symptom of a bad record.
//
// 100 ms keeps roughly 40x margin over the observed skew while staying orders of
// magnitude below what this bound exists to catch: a stale record from a previous
// generation or a lapped ring slot, which is seconds out, not milliseconds. The
// discrimination that actually separates dispatches is the correlation key
// (doorbell slot + dispatch index + generation) plus the start >= enqueue_ts
// lower bound; this bound only rejects the implausibly-future.
constexpr uint64_t kKfdFutureSlackNs = 100'000'000;  // 100 ms

// Whether a firmware record's converted (system-domain) timestamps are usable:
// they must form a positive interval that starts no earlier than the dispatch's
// own enqueue and ends no later than now plus the conversion slack above. A
// record taken for the wrong dispatch lands outside those bounds, so the caller
// emits no KFD timestamps for it.
inline bool
kfd_time_is_sane(uint64_t start_ns, uint64_t end_ns, uint64_t enqueue_ts, uint64_t now_ns)
{
    // now_ns is a boot-relative nanosecond count, so the addition cannot overflow.
    return start_ns < end_ns && start_ns >= enqueue_ts &&
           end_ns <= now_ns + kKfdFutureSlackNs;
}

// std::hash-compatible functor for correlation_key. Combines the three 32-bit
// fields with the common boost-style hash_combine mix.
struct correlation_key_hash
{
    size_t operator()(const correlation_key& key) const
    {
        auto mix = [](size_t seed, uint32_t value) {
            return seed ^ (std::hash<uint32_t>{}(value) + 0x9e3779b9UL + (seed << 6) + (seed >> 2));
        };
        size_t seed = std::hash<uint32_t>{}(key.doorbell_off);
        seed        = mix(seed, key.dispatch_idx_low32);
        seed        = mix(seed, key.generation);
        seed        = mix(seed, key.gpu_id);
        return seed;
    }
};
}  // namespace kfd
}  // namespace rocprofiler
