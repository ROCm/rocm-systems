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

// Scaffolding shared by the rocprofiler_synchronize_context tests: the struct builders the API
// requires, and the macros each test asserts with.
//
// Every including file must define a tag naming itself, which prefixes every diagnostic:
//
//     namespace { constexpr const char* kTestTag = "my-test"; }

#include <rocprofiler-sdk/experimental/synchronize_contexts.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Setup rather than the thing under test: there is nothing useful to assert once one of these
// fails, so it takes the process down at the point of failure.
//
// The do/while makes the macro a single statement, so it stays correct in an unbraced if/else
// rather than silently attaching the failure branch to the wrong arm.
#define CHECK_ROCP(...)                                                                            \
    do                                                                                             \
    {                                                                                              \
        auto _status = (__VA_ARGS__);                                                              \
        if(_status != ROCPROFILER_STATUS_SUCCESS)                                                  \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[%s] %s:%d: %s failed: %s\n",                                                 \
                    kTestTag,                                                                      \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    #__VA_ARGS__,                                                                  \
                    rocprofiler_get_status_string(_status));                                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)

#define CHECK_HIP(...)                                                                             \
    do                                                                                             \
    {                                                                                              \
        auto _err = (__VA_ARGS__);                                                                 \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[%s] %s:%d: %s failed: %s\n",                                                 \
                    kTestTag,                                                                      \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    #__VA_ARGS__,                                                                  \
                    hipGetErrorString(_err));                                                      \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)

// Report one failed expectation and evaluate to 1, so a check can accumulate failures with
// `failures += EXPECT_STATUS(...)` and still run everything else.
#define EXPECT_STATUS(EXPECTED, ...)                                                               \
    ([&]() -> uint64_t {                                                                           \
        auto _status = (__VA_ARGS__);                                                              \
        if(_status == (EXPECTED)) return 0;                                                        \
        fprintf(stderr,                                                                            \
                "[%s] %s:%d: %s returned %s, expected %s\n",                                       \
                kTestTag,                                                                          \
                __FILE__,                                                                          \
                __LINE__,                                                                          \
                #__VA_ARGS__,                                                                      \
                rocprofiler_get_status_string(_status),                                            \
                rocprofiler_get_status_string(EXPECTED));                                          \
        return 1;                                                                                  \
    }())

namespace sync_test
{
// The caller stamps the size of each struct it hands the API, which is how the API tells an older
// caller's layout from a newer one. The only way the tests build one, so no call site forgets.
inline rocprofiler_synchronize_context_options_t
make_options(uint64_t flags, uint64_t timeout_ns = 0)
{
    auto options       = rocprofiler_synchronize_context_options_t{};
    options.size       = sizeof(rocprofiler_synchronize_context_options_t);
    options.flags      = flags;
    options.timeout_ns = timeout_ns;
    return options;
}

inline rocprofiler_synchronize_context_result_t
make_result()
{
    auto result = rocprofiler_synchronize_context_result_t{};
    result.size = sizeof(rocprofiler_synchronize_context_result_t);
    return result;
}
}  // namespace sync_test
