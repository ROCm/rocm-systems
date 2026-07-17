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

    bool operator==(const correlation_key& rhs) const
    {
        return doorbell_off == rhs.doorbell_off && dispatch_idx_low32 == rhs.dispatch_idx_low32 &&
               generation == rhs.generation;
    }

    bool operator!=(const correlation_key& rhs) const { return !(*this == rhs); }
};

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
        return seed;
    }
};
}  // namespace kfd
}  // namespace rocprofiler
