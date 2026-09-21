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

// Windows stand-in for late.cpp. Late-start profiling replays API tables that runtimes
// handed to rocprofiler-register; Windows tracing consumes ETW events instead, so there
// is no registry to replay and rocprofiler-register is not in the path.

#include "lib/rocprofiler-sdk/registration/late.hpp"

#include <rocprofiler-sdk/fwd.h>

namespace rocprofiler
{
namespace registration
{
namespace late
{
rocprofiler_status_t
invoke_register_propagation()
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}
}  // namespace late
}  // namespace registration
}  // namespace rocprofiler
