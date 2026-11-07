/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <hip/hip_runtime.h>

#include <chrono>
#include <thread>

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

} // namespace RcclUnitTesting
