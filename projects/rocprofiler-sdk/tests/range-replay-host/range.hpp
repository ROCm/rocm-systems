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

// The contract between main.cpp, which runs a time-stepping solver inside one host-replay range,
// and client.cpp, which replays it and counts what the counter service collected for it.

constexpr uint64_t kRangeId = 0x5E4057;

// Grid is kN x kN. kN is a power of two for the radix-2 FFT, and kLogN is its log2.
constexpr int kN    = 64;
constexpr int kLogN = 6;

constexpr int kTimeSteps    = 4;
constexpr int kJacobiSweeps = 4;

// Upper bound on passes main.cpp keeps a per-pass result for.
constexpr int kMaxPasses = 16;

// Every kernel main.cpp launches directly carries this prefix; the tool counts counter records for
// those kernels only (HIP's own blit kernels for copies and memsets are excluded, and so is the
// graph's kernel, whose name does not carry it).
constexpr const char* kKernelPrefix = "rr_host_";

// Directly launched kernels per time step: pack, forward FFT, spectral filter, inverse FFT, unpack,
// stencil, the Jacobi sweeps, and the residual.
constexpr int kKernelsPerStep = 5 + 1 + kJacobiSweeps + 1;
constexpr int kKernelsPerPass = kTimeSteps * kKernelsPerStep;
