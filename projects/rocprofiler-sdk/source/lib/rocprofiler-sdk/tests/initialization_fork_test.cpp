// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <gtest/gtest.h>
#include <rocprofiler-sdk/registration.h>
#include <unistd.h>
#include <cstdlib>

namespace
{
rocprofiler_tool_configure_result_t*
empty_tool(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
{
    return nullptr;
}
}  // namespace

TEST(initialization_fork, first_force_configure_installs_handler_before_lifecycle_lock)
{
    ASSERT_EXIT(
        {
            alarm(10);
            setenv("ROCPROF_INTERNAL_TEST_FORCE_CONFIGURE_FORK", "1", 1);
            const auto status = rocprofiler_force_configure(empty_tool);
            std::_Exit(status == ROCPROFILER_STATUS_SUCCESS ? 0 : 95);
        },
        ::testing::ExitedWithCode(0),
        "");
}
