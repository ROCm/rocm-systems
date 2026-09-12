// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "range.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define RR_CHECK(call)                                                                             \
    do                                                                                             \
    {                                                                                              \
        rocprofiler_status_t _s = (call);                                                          \
        if(_s != ROCPROFILER_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[range-replay] %s failed: %s\n",                                              \
                    #call,                                                                         \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

inline const char*
status_name(rocprofiler_range_replay_status_t status)
{
    switch(status)
    {
        case ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED: return "REPLAYED";
        case ROCPROFILER_RANGE_REPLAY_STATUS_NO_DISPATCH: return "NO_DISPATCH";
        case ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT: return "NO_PASS_COUNT";
        case ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE: return "MULTI_QUEUE";
        case ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_AGENT: return "MULTI_AGENT";
        case ROCPROFILER_RANGE_REPLAY_STATUS_GRAPH_LAUNCH: return "GRAPH_LAUNCH";
        case ROCPROFILER_RANGE_REPLAY_STATUS_UNKNOWN_KERNARG_SIZE: return "UNKNOWN_KERNARG_SIZE";
        case ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE: return "MEMORY_COPY_IN_RANGE";
        case ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH: return "CONCURRENT_DISPATCH";
        case ROCPROFILER_RANGE_REPLAY_STATUS_PROGRAM_TOO_LARGE: return "PROGRAM_TOO_LARGE";
        case ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED: return "SNAPSHOT_FAILED";
        case ROCPROFILER_RANGE_REPLAY_STATUS_STAGING_FAILED: return "STAGING_FAILED";
        case ROCPROFILER_RANGE_REPLAY_STATUS_ALLOCATION_CHANGED_IN_RANGE:
            return "ALLOCATION_CHANGED_IN_RANGE";
        case ROCPROFILER_RANGE_REPLAY_STATUS_UNSUPPORTED_QUEUE_PATH:
            return "UNSUPPORTED_QUEUE_PATH";
        case ROCPROFILER_RANGE_REPLAY_STATUS_LAST: break;
    }
    return "<unknown>";
}
