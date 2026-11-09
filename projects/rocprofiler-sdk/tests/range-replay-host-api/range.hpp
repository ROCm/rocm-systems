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

// The contract between main.cpp, which runs one range through rocprofiler_range_replay_host_run,
// and client.cpp, which replays it.

constexpr uint64_t kRangeId = 0xA91B0D;

constexpr int kN      = 4096;
constexpr int kBlock  = 256;
constexpr int kBlocks = kN / kBlock;
constexpr int kTable  = 64;
constexpr int kSteps  = 3;

// Upper bound on passes main.cpp keeps a per-pass result for.
constexpr int kMaxPasses = 16;

// Kernels per step: affine update, table mix, block reduction.
constexpr int kKernelsPerStep = 3;
constexpr int kKernelsPerPass = kSteps * kKernelsPerStep;
