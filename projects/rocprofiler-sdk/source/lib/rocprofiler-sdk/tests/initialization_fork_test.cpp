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
#include <rocprofiler-sdk-roctx/api_trace.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/registration.h>

#include "lib/rocprofiler-sdk/registration.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>

namespace
{
rocprofiler_tool_configure_result_t*
empty_tool(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
{
    return nullptr;
}

void
fork_during_runtime_registration(rocprofiler_intercept_table_t kind,
                                 uint64_t,
                                 uint64_t,
                                 void**,
                                 uint64_t,
                                 void*)
{
    if(kind != ROCPROFILER_MARKER_CORE_TABLE) std::_Exit(96);

    const auto child = fork();
    if(child == 0)
    {
        int initialized = -1;
        int finalized   = -1;
        if(rocprofiler_is_initialized(&initialized) != ROCPROFILER_STATUS_SUCCESS ||
           rocprofiler_is_finalized(&finalized) != ROCPROFILER_STATUS_SUCCESS || initialized != 0 ||
           finalized != 1)
            std::_Exit(92);
        std::_Exit(0);
    }

    int status = 0;
    if(child < 0 || waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
       WEXITSTATUS(status) != 0)
        std::_Exit(93);
}

rocprofiler_tool_configure_result_t*
runtime_registration_tool(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
{
    if(rocprofiler_at_intercept_table_registration(fork_during_runtime_registration,
                                                   ROCPROFILER_MARKER_CORE_TABLE,
                                                   nullptr) != ROCPROFILER_STATUS_SUCCESS)
        std::_Exit(94);
    return nullptr;
}

[[noreturn]] void
run_runtime_registration_fork_test()
{
    alarm(10);
    if(rocprofiler_force_configure(runtime_registration_tool) != ROCPROFILER_STATUS_SUCCESS)
        std::_Exit(95);

    auto core      = roctxCoreApiTable_t{};
    auto control   = roctxControlApiTable_t{};
    auto name      = roctxNameApiTable_t{};
    core.size      = sizeof(core);
    control.size   = sizeof(control);
    name.size      = sizeof(name);
    void* tables[] = {&core, &control, &name};

    const auto status = rocprofiler_set_api_table("roctx", 0, 0, tables, 3);
    std::_Exit(status == ROCPROFILER_STATUS_SUCCESS ? 0 : 97);
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

TEST(initialization_fork, runtime_registration_remains_guarded_after_sdk_initialize_returns)
{
    ASSERT_EXIT(run_runtime_registration_fork_test(), ::testing::ExitedWithCode(0), "");
}
