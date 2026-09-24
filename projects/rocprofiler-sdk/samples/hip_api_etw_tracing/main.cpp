// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "client.hpp"

#include <hip/hip_runtime_api.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

// The provider emits an enter/exit pair for every call it wraps, whatever the call returns, so
// a failing HIP runtime is still a valid trace source. Report the failure and keep going.
#define HIP_API_CALL(FUNC, ...)                                                                    \
    {                                                                                              \
        client::expect_operation(#FUNC);                                                           \
        hipError_t error_ = FUNC(__VA_ARGS__);                                                     \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            std::cerr << __FILE__ << ":" << __LINE__ << " :: " << #FUNC << " returned "            \
                      << hipGetErrorString(error_) << "\n";                                        \
        }                                                                                          \
    }

int
main(int argc, char** argv)
{
    // Run as a plain HIP application, for the case where the consumer lives in another process
    // and this one is only the provider. Self-tracing as well would open a second session on the
    // same provider, and the two would then race over which one outlives the trace.
    auto self_trace = true;
    for(int i = 1; i < argc; ++i)
        if(std::strcmp(argv[i], "--no-self-trace") == 0) self_trace = false;

    if(self_trace) client::setup();

    // Every call below crosses the traced HIP runtime entry points, which is all the ETW
    // provider reports on; no kernel is launched because the sample is compiled as plain C++
    // so that it configures under MSVC.
    constexpr size_t byte_count = 1024 * sizeof(int);

    // Warm-up, deliberately not expected: the provider does not observe that a session has
    // enabled it until the first HIP call after the enable, so that one call goes untraced.
    int device_count = 0;
    static_cast<void>(hipGetDeviceCount(&device_count));

    HIP_API_CALL(hipSetDevice, 0);

    int* device_buffer = nullptr;
    HIP_API_CALL(hipMalloc, &device_buffer, byte_count);
    HIP_API_CALL(hipMemset, device_buffer, 0, byte_count);
    HIP_API_CALL(hipStreamSynchronize, nullptr);
    HIP_API_CALL(hipFree, device_buffer);

    if(self_trace) client::shutdown();

    return EXIT_SUCCESS;
}
