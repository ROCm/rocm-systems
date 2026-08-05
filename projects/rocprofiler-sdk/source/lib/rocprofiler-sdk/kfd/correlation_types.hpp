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

// Identity types bridging an SDK dispatch to a firmware dispatch-log record.
// Firmware identifies a dispatch as (doorbell_off, dispatch_idx_low32); the SDK
// captures the same pair at enqueue.

namespace rocprofiler
{
namespace kfd
{
// Every field is snapshotted at enqueue time so completion-time lookup cannot
// race with queue destroy/recreate.
struct correlation_key
{
    uint32_t doorbell_off       = 0;
    uint32_t dispatch_idx_low32 = 0;
    uint32_t generation         = 0;
    // Doorbell slots and dispatch indices are per-GPU and both restart from low
    // values, so without this a record from one GPU can match a dispatch on
    // another. Last field so three-field construction still means "GPU 0".
    uint32_t gpu_id = 0;

    bool operator==(const correlation_key& rhs) const
    {
        return doorbell_off == rhs.doorbell_off && dispatch_idx_low32 == rhs.dispatch_idx_low32 &&
               generation == rhs.generation && gpu_id == rhs.gpu_id;
    }

    bool operator!=(const correlation_key& rhs) const { return !(*this == rhs); }
};

// How far past a CPU timestamp a converted firmware end may legitimately land.
// Tick-to-system-domain conversion re-syncs periodically, so a just-completed
// dispatch's converted end measures 2.0-2.7 ms AFTER a `now` sampled right
// behind it on gfx1201; a hard `end <= now` would discard every valid record.
// 100 ms keeps margin over that while still catching stale/lapped records,
// which are seconds out. The correlation key does the real discrimination.
constexpr uint64_t kKfdFutureSlackNs = 100'000'000;  // 100 ms

// Usable means a positive interval starting no earlier than the dispatch's own
// enqueue and ending no later than now plus the conversion slack above.
inline bool
kfd_time_is_sane(uint64_t start_ns, uint64_t end_ns, uint64_t enqueue_ts, uint64_t now_ns)
{
    // now_ns is a boot-relative nanosecond count, so the addition cannot overflow.
    return start_ns < end_ns && start_ns >= enqueue_ts && end_ns <= now_ns + kKfdFutureSlackNs;
}

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
