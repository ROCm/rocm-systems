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

#pragma once

#include <cstdint>

namespace rocprofiler
{
namespace collector
{
// LTTng-UST timestamps events with CLOCK_MONOTONIC; rocprofiler-sdk records use
// CLOCK_BOOTTIME. They differ by the time the system was suspended, so the collector
// samples both when tracing starts and stops.
struct clock_sample
{
    uint64_t monotonic = 0;  // ns
    uint64_t boottime  = 0;  // ns
};

struct clock_conversion
{
    int64_t offset = 0;      // boottime - monotonic, ns
    bool    stable = false;  // the offset was the same at start and stop (no suspend)
};

clock_sample
sample_clocks();

// The offset at the start of tracing, and whether it held until the stop. If the system
// was suspended while tracing, the times recorded after the resume are early by the
// suspended time; that is reported, not corrected.
clock_conversion
get_clock_conversion(const clock_sample& start, const clock_sample& stop);

uint64_t
to_boottime(const clock_conversion& conversion, uint64_t monotonic);
}  // namespace collector
}  // namespace rocprofiler
