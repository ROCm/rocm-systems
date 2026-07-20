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

// Host-callable wrappers around the snapshot/restore test kernels. The <<<>>> launches live in
// snap_kernels.cpp, which is compiled as HIP; the test translation unit (snap_restore.cpp) only
// declares these and is compiled as plain CXX. Keeping the sole HIP source in a separate library
// lets the test executable link as CXX and inherit the SDK's automatic -rdynamic (the
// rocprofiler-sdk-dl interface adds it gated on LINK_LANGUAGE C/CXX). That export is required
// because ensure_live_tracking() -> rocprofiler_force_configure() runs a register scan that dlopens
// this statically-linked executable looking for rocprofiler_set_api_table.
//
// Each launcher only enqueues its kernel; callers check hipGetLastError()/hipDeviceSynchronize()
// themselves so the gtest assertions stay in the CXX test unit.
namespace kernel_launch
{
// d[i] = val
void
fill(float* d, float val, int n);

// d[i] = base + i : a non-uniform pattern so a stale/partial restore is easy to catch.
void
iota(float* d, float base, int n);

// y[i] = a * x[i] + y[i] : in-place read-write, the canonical case restore must protect.
void
saxpy(float* y, const float* x, float a, int n);

// d[i] = d[i] + delta : another in-place read-write mutation.
void
add(float* d, float delta, int n);
}  // namespace kernel_launch
