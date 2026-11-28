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

// Scaffolding shared by the rocprofiler_synchronize_context tools: the struct builders the API
// requires, the assertion macros, and the workload compatibility check. Tool side only, and free
// of HIP.
//
// Every including file must define the tag that prefixes its diagnostics:
//
//     namespace { constexpr const char* kTestTag = "my-test"; }

#include "sync_test_workload.h"

#include <rocprofiler-sdk/experimental/synchronize_contexts.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Setup, not the thing under test: nothing further is worth asserting once one of these fails,
// so it exits at the point of failure.
#define ROCPROFILER_CHECK(...)                                                                     \
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

// Report one failed expectation and evaluate to 1, so a check can accumulate with
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
// caller's layout from a newer one.
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

/// True when the application can do what the tool is about to ask of it. A tool paired with the
/// wrong application would serialize work it needs concurrent, or skip the domain it covers, and
/// report a pass for doing so.
inline bool
workload_is_usable(const sync_test_workload_t* workload,
                   uint64_t                    required_lanes,
                   bool                        requires_copy,
                   const char*                 test_tag)
{
    if(workload == nullptr || workload->size < sizeof(sync_test_workload_t))
    {
        fprintf(stderr,
                "[%s] the application supplied no workload, or one built against an older "
                "layout than this tool was built against\n",
                test_tag);
        return false;
    }

    if(workload->open_lane == nullptr || workload->close_lane == nullptr ||
       workload->launch_kernel == nullptr || workload->wait == nullptr)
    {
        fprintf(stderr, "[%s] the application left a required workload entry unset\n", test_tag);
        return false;
    }

    if(workload->lanes < required_lanes)
    {
        fprintf(stderr,
                "[%s] this tool needs %lu independent lanes and the application offers %lu, so "
                "the work it is meant to overlap would run in sequence\n",
                test_tag,
                static_cast<unsigned long>(required_lanes),
                static_cast<unsigned long>(workload->lanes));
        return false;
    }

    if(requires_copy && workload->launch_copy == nullptr)
    {
        fprintf(stderr,
                "[%s] this tool covers the memory copy domain and the application cannot issue "
                "a copy\n",
                test_tag);
        return false;
    }

    return true;
}

/// Holds a lane open for a scope, so an early return cannot leak the application's stream.
class scoped_lane
{
public:
    explicit scoped_lane(const sync_test_workload_t& workload, uint64_t index = 0)
    : m_workload{&workload}
    , m_lane{workload.open_lane(index)}
    {}

    ~scoped_lane() { m_workload->close_lane(m_lane); }

    scoped_lane(const scoped_lane&) = delete;
    scoped_lane& operator=(const scoped_lane&) = delete;

    void* get() const { return m_lane; }

    void launch_kernel() const { m_workload->launch_kernel(m_lane); }
    void launch_copy() const { m_workload->launch_copy(m_lane); }
    void wait() const { m_workload->wait(m_lane); }

private:
    const sync_test_workload_t* m_workload = nullptr;
    void*                       m_lane     = nullptr;
};
}  // namespace sync_test
