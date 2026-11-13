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

#include <rocprofiler-sdk-collector/types.h>

#include <string>
#include <vector>

namespace rocprofiler
{
namespace collector
{
// Start argv (argv[0] looked up in PATH) as a child of this process, inheriting its
// environment plus extra_env ("NAME=value", replacing a variable of the same name). An
// exec failure is reported here (through a close-on-exec pipe), not as a child that exits
// with 127.
rocprofiler_collector_status_t
launch_process(const char* const* argv, const std::vector<std::string>& extra_env, int* pid);

// Wait for a child started by launch_process. exit_status is the exit code, or 128 + the
// signal number when a signal ended it.
rocprofiler_collector_status_t
wait_process(int pid, int* exit_status);
}  // namespace collector
}  // namespace rocprofiler
