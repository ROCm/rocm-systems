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

#include "lib/rocprofiler-sdk-collector/clock.hpp"

#include <time.h>

namespace rocprofiler
{
namespace collector
{
namespace
{
// Offsets that differ by less than this are the same (sampling jitter).
constexpr int64_t suspend_threshold_ns = 1000000;

uint64_t
read_clock(clockid_t id)
{
    auto now = timespec{};
    ::clock_gettime(id, &now);
    return (static_cast<uint64_t>(now.tv_sec) * 1000000000) + static_cast<uint64_t>(now.tv_nsec);
}

int64_t
offset_of(const clock_sample& sample)
{
    return static_cast<int64_t>(sample.boottime - sample.monotonic);
}
}  // namespace

clock_sample
sample_clocks()
{
    // BOOTTIME read between two MONOTONIC reads, paired with their midpoint.
    auto before   = read_clock(CLOCK_MONOTONIC);
    auto boottime = read_clock(CLOCK_BOOTTIME);
    auto after    = read_clock(CLOCK_MONOTONIC);
    return clock_sample{before + ((after - before) / 2), boottime};
}

clock_conversion
get_clock_conversion(const clock_sample& start, const clock_sample& stop)
{
    auto result   = clock_conversion{};
    result.offset = offset_of(start);
    auto drift    = offset_of(stop) - result.offset;
    result.stable = drift < suspend_threshold_ns && drift > -suspend_threshold_ns;
    return result;
}

uint64_t
to_boottime(const clock_conversion& conversion, uint64_t monotonic)
{
    return static_cast<uint64_t>(static_cast<int64_t>(monotonic) + conversion.offset);
}
}  // namespace collector
}  // namespace rocprofiler
