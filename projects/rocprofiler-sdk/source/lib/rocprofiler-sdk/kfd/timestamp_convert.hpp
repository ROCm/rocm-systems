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

#include <cstdint>

// Converts raw firmware GPU clock ticks (from dispatch-log records) into
// CLOCK_BOOTTIME nanoseconds — the domain rocprofiler emits and the same domain
// hsa_amd_profiling_get_dispatch_time() produces. The two clocks are not
// guaranteed to share an epoch, so the conversion is anchored (a jointly-sampled
// GPU-tick / boottime / frequency triple) and, crucially, self-validating: KFD
// timestamps are only trusted for emission after enough per-dispatch samples land
// within tolerance of the HSA result (spec Option B — HSA emitted until then).

namespace rocprofiler
{
namespace kfd
{
// Sample a fresh anchor: GPU tick counter, CLOCK_BOOTTIME ns, and tick frequency,
// captured as close together as possible. Called at reader startup and re-sampled
// periodically to bound drift. Safe to call before HSA is up (no-op if the core
// table / frequency is unavailable; conversion stays disabled).
void
calibrate_clock();

// Convert one raw GPU tick value to CLOCK_BOOTTIME ns using the current anchor.
// Returns 0 if no valid anchor exists yet.
uint64_t
convert_gpu_ticks_to_ns(uint64_t gpu_tick);

// Feed one paired (KFD-converted, HSA-measured) sample. While unvalidated, counts
// in-tolerance samples; flips to validated after enough consecutive good samples,
// and resets (re-calibrating) on any out-of-tolerance sample. Returns true once
// conversion is validated. No-op once already validated.
bool
validate_conversion(uint64_t kfd_start_ns,
                    uint64_t hsa_start_ns,
                    uint64_t kfd_end_ns,
                    uint64_t hsa_end_ns);

// True when enough validated samples exist to trust KFD timestamps for emission.
bool
conversion_validated();
}  // namespace kfd
}  // namespace rocprofiler
