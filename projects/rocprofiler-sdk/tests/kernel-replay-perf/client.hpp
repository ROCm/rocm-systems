// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define KR_CHECK(call)                                                                             \
    do                                                                                             \
    {                                                                                              \
        rocprofiler_status_t _s = (call);                                                          \
        if(_s != ROCPROFILER_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[kernel-replay-perf] %s failed: %s\n",                                        \
                    #call,                                                                         \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

constexpr uint32_t kReplayBlockX = 67;
