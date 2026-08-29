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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>

// Identity types bridging an SDK dispatch to a firmware dispatch-log record.
// Firmware identifies a dispatch as (doorbell_off, dispatch_idx_low32); the SDK
// captures the same pair at enqueue.

namespace rocprofiler
{
namespace kfd
{
// Absolute monotonic nanoseconds; all deadlines in the KFD path are values of
// this. steady_clock (CLOCK_MONOTONIC) is deliberate: the close/GC deadlines are
// elapsed-time budgets, so they must be immune to CLOCK_REALTIME jumps (NTP step,
// settimeofday). It is never compared against a GPU tick or a system-domain
// timestamp -- those live in their own domains and never cross into this one.
inline uint64_t
steady_now_ns()
{
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

// The identity a firmware record can reconstruct: (page-relative doorbell slot,
// low-32 dispatch index, GPU). The owning owner_window is carried BESIDE the key
// (on the registration and the hub entry), never in it, because a record cannot
// reconstruct a window -- time-as-generation resolves the window from the START
// tick. `generation` is deleted: recycled-doorbell collisions are
// now representable (the hub's multimap) and resolved by window containment.
struct correlation_key
{
    uint32_t doorbell_off       = 0;
    uint32_t dispatch_idx_low32 = 0;
    // Doorbell slots and dispatch indices are per-GPU and both restart from low
    // values, so without this a record from one GPU can match a dispatch on
    // another. No cross-agent tick comparison ever happens.
    uint32_t gpu_id = 0;

    bool operator==(const correlation_key& rhs) const
    {
        return doorbell_off == rhs.doorbell_off && dispatch_idx_low32 == rhs.dispatch_idx_low32 &&
               gpu_id == rhs.gpu_id;
    }

    bool operator!=(const correlation_key& rhs) const { return !(*this == rhs); }
};

// How far past a CPU timestamp a converted firmware end may legitimately land, in
// the CLOCK_MONOTONIC SYSTEM domain (common::timestamp_ns(), the domain
// hsa_amd_profiling_convert_tick_to_system_domain targets) -- NOT the GPU tick
// domain the owner windows compare in, which is raw-tick-only and immune to
// resync. Tick-to-system-domain conversion re-syncs periodically, so a
// just-completed dispatch's converted end can measure a few ms AFTER a `now`
// sampled right behind it (low single-digit ms observed in practice); a hard
// `end <= now` would discard every valid record. 100 ms keeps a wide margin over
// that. This is NOT a correctness threshold: it only raises the after_now
// diagnostic flag. Emission is gated by the finalizer's own postcondition, which
// unconditionally shifts end down to `now`, so the exact value is not
// safety-critical -- too small only over-flags, too large only under-flags. The
// hard raw-terminal rejects are kMaxStaleNs/kMaxFutureNs (seconds).
constexpr uint64_t kKfdFutureSlackNs = 100'000'000;  // 100 ms

// kfd_time_is_sane is deleted: its role -- "is this interval usable" -- is
// now the finalizer's own unconditional postcondition (result_ready requires
// enqueue <= start < end <= now), at a tighter bound than this helper's slack.

struct correlation_key_hash
{
    size_t operator()(const correlation_key& key) const
    {
        auto mix = [](size_t seed, uint32_t value) {
            return seed ^ (std::hash<uint32_t>{}(value) + 0x9e3779b9UL + (seed << 6) + (seed >> 2));
        };
        size_t seed = std::hash<uint32_t>{}(key.doorbell_off);
        seed        = mix(seed, key.dispatch_idx_low32);
        seed        = mix(seed, key.gpu_id);
        return seed;
    }
};
}  // namespace kfd
}  // namespace rocprofiler
