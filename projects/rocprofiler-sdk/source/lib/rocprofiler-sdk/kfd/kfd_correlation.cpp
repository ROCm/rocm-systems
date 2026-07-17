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

#include "lib/rocprofiler-sdk/kfd/kfd_correlation.hpp"

#include "lib/common/static_object.hpp"

// Each accessor lazily constructs a single process-wide instance via
// common::static_object (same idiom as kfd_profiler.cpp state()): placement-new
// into a static buffer with ordered teardown, no heap leak, no unspecified
// static-destructor ordering at library unload.

namespace rocprofiler
{
namespace kfd
{
DoorbellMap&
doorbell_map()
{
    static auto*& _v = common::static_object<DoorbellMap>::construct();
    return *_v;
}

CorrelationTable&
correlation_table()
{
    static auto*& _v = common::static_object<CorrelationTable>::construct();
    return *_v;
}

ResultsMap&
results_map()
{
    static auto*& _v = common::static_object<ResultsMap>::construct();
    return *_v;
}
}  // namespace kfd
}  // namespace rocprofiler
