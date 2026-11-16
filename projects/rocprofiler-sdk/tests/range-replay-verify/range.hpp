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

// The contract between main.cpp, which runs a range whose kernels only a correct replay can
// reproduce, and client.cpp, which replays it with verification on.

constexpr uint64_t kRangeId = 0xD16E57;

// Total passes the tool asks for, the application's own included.
constexpr uint64_t kPasses = 4;

// Elements of the 16 MiB buffer the range's kernels rewrite in place.
constexpr uint32_t kElements = uint32_t{1} << 22;

// In-place steps of the in-place mode, one dispatch each.
constexpr uint32_t kInPlaceSteps = 3;

// Counters of the atomic-histogram mode. With kElements spread over them, thousands of threads
// increment every counter.
constexpr uint32_t kBins = 256;

// Set by the application before it exits without opening the range, so the tool reports a skip
// rather than a missing CLOSE. Its value is the reason.
constexpr const char* kSkipEnv = "RR_VERIFY_SKIPPED";
