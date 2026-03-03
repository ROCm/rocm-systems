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

#include <rocprofiler-sdk-rocattach/types.h>

#include <sys/ptrace.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <variant>

namespace rocprofiler
{
namespace rocattach
{
using ptrace_parameter_t                          = std::variant<std::monostate, uint64_t, void*>;
static constexpr size_t ptrace_default_timeout_ms = 10000;

// Intended to mirror ptrace(), but with parameters for its return value and errno
// A return value of ROCATTACH_STATUS_ERROR indicates a timeout while communicating with this
// module's worker thread.
rocattach_status_t
ptrace_run(pid_t              pid,
           __ptrace_request   op,
           ptrace_parameter_t addr,
           ptrace_parameter_t data,
           uint64_t*          ptrace_retval,
           int*               ptrace_errno,
           size_t             timeout = ptrace_default_timeout_ms);

}  // namespace rocattach
}  // namespace rocprofiler
