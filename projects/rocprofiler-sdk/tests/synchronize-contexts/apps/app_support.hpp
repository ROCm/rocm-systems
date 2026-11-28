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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// Scaffolding shared by the applications the rocprofiler_synchronize_context tools drive: the
// HIP error check, and resolving the tool out of the process. Application side only, and free of
// rocprofiler.
//
// Every including file must define the tag that prefixes its diagnostics:
//
//     namespace { constexpr const char* kAppTag = "my-app"; }

#include "../sync_test_workload.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <dlfcn.h>

// Setup, not the thing under test: nothing further is worth asserting once one of these fails,
// so it exits at the point of failure.
#define HIP_API_CALL(...)                                                                          \
    do                                                                                             \
    {                                                                                              \
        auto _err = (__VA_ARGS__);                                                                 \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[%s] %s:%d: %s failed: %s\n",                                                 \
                    kAppTag,                                                                       \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    #__VA_ARGS__,                                                                  \
                    hipGetErrorString(_err));                                                      \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)

namespace sync_test_app
{
/// Hand the workload to the preloaded tool and return the exit status its verdict earns. Without
/// a tool there is nothing to run, and exiting zero would register as a passing test.
inline int
run_under_tool(const sync_test_workload_t& workload, const char* app_tag)
{
    // RTLD_DEFAULT rather than a handle: the tool is already in the process, preloaded ahead of
    // this binary.
    auto* run = reinterpret_cast<sync_test_run_fn>(::dlsym(RTLD_DEFAULT, SYNC_TEST_RUN_SYMBOL));
    if(run == nullptr)
    {
        fprintf(stderr,
                "[%s] no tool exporting %s is loaded, so this run would exercise the workload "
                "with nothing watching it\n",
                app_tag,
                SYNC_TEST_RUN_SYMBOL);
        return EXIT_FAILURE;
    }

    return (run(&workload) == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
}  // namespace sync_test_app
