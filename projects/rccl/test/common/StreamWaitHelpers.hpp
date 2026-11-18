/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include <hip/hip_runtime.h>

#include "TestChecks.hpp"

namespace RcclUnitTesting
{

// Polls so that a kernel spinning on a peer flag that never arrives is reported instead of hanging.
// Returns hipSuccess, the stream's error, or hipErrorNotReady when the timeout expires.
inline hipError_t waitStreamBounded(hipStream_t stream, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while(true)
    {
        const hipError_t query = hipStreamQuery(stream);
        if(query != hipErrorNotReady || std::chrono::steady_clock::now() >= deadline)
        {
            return query;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// Waits for every stream; past `timeout` raises *abortHost so spinning kernels leave. True if all finished in time.
inline bool drainOrAbort(const std::vector<hipStream_t>& streams,
                         std::chrono::milliseconds timeout,
                         uint32_t* abortHost)
{
    bool drained = true;
    for(hipStream_t st : streams)
    {
        drained = drained && waitStreamBounded(st, timeout) == hipSuccess;
    }
    if(!drained)
    {
        *abortHost = 1;
    }
    for(hipStream_t st : streams)
    {
        HIP_EXPECT(hipStreamSynchronize(st));
    }
    return drained;
}

} // namespace RcclUnitTesting
