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

#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"
#include "lib/rocprofiler-sdk/kfd/results_map.hpp"

namespace rocprofiler
{
namespace kfd
{
// One instance per process: the enqueue paths and the reader thread share it.
DoorbellMap&
doorbell_map();

// Deposited by the reader thread, taken in get_dispatch_time().
ResultsMap&
results_map();

// The single gate for emitting KFD timestamps: selecting a firmware record is
// only sound once it is provably from the current, uniquely-owning queue
// generation. False keeps every dispatch on HSA timestamps, signals retained.
bool
kfd_selection_enabled();
}  // namespace kfd
}  // namespace rocprofiler
