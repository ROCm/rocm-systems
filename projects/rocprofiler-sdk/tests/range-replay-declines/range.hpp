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

#include <cstddef>
#include <cstdint>

// The contract between main.cpp, which drives a range into a decline, and client.cpp, which checks
// how the range closed.

constexpr uint64_t kRangeId = 0xDEC11E;

// Total passes the tool asks for, the application's own included. A declined range must run none of
// the replayed ones.
constexpr uint64_t kRequestedPasses = 4;

// Dispatches CLOSE reports for a range declined by a device write between its two kernels: only the
// first. The write is folded into the record before the second kernel is counted, and a declined
// range stops counting.
constexpr uint64_t kObservedDispatches = 1;

// Elements the copy inside the range writes. HIP performs a copy of up to GPU_FORCE_BLIT_COPY_SIZE
// (16 KiB by default) with a kernel on the application's own queue, which the range records like
// any other dispatch. Only a larger copy reaches the copy engines through
// hsa_amd_memory_async_copy*, which is the path that declines the range.
constexpr size_t kCopyElements = (1u << 20) / sizeof(int);

// The value the copy writes at element `i`, so a partial or misplaced copy is detectable.
constexpr int
copy_pattern(size_t i)
{
    return static_cast<int>((i * 3) + 1);
}

// The first kernel computes acc = 0*3 + 1; the second reads the copied buffer and computes
// acc = acc*3 + y[0] + y[n-1].
constexpr int kExpectedResult = 3 + copy_pattern(0) + copy_pattern(kCopyElements - 1);

// Set by the application before it exits without opening the range, so the tool reports a skip
// rather than a missing CLOSE. Its value is the reason.
constexpr const char* kSkipEnv = "RR_DECLINE_SKIPPED";
