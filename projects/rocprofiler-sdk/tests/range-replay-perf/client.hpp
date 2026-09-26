// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Shared between the perf tool and the perf application. The application brackets its dispatches
// with this id and the tool ignores every other range, so an unrelated range opened by anything
// else in the process cannot be timed as if it were the workload.
constexpr uint64_t kPerfRangeId = 0xEEF00D;

#define RR_CHECK(...)                                                                              \
    do                                                                                             \
    {                                                                                              \
        const auto _status = (__VA_ARGS__);                                                        \
        if(_status != ROCPROFILER_STATUS_SUCCESS)                                                  \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[rr-perf-client] %s failed at %s:%d with status %d\n",                        \
                    #__VA_ARGS__,                                                                  \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    static_cast<int>(_status));                                                    \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)
