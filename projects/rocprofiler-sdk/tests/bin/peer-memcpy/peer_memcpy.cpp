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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error %i: %s\n",                                                 \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    (int) error_,                                                                  \
                    hipGetErrorString(error_));                                                    \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    }

int
main()
{
    // above ROC_P2P_SDMA_SIZE so HIP performs the peer copy with SDMA, not a blit kernel
    constexpr size_t num_elements = 4 * 1024 * 1024;
    constexpr size_t num_bytes    = num_elements * sizeof(int);

    int num_devices = 0;
    HIP_API_CALL(hipGetDeviceCount(&num_devices));

    auto host_src = std::vector<int>(num_elements, 1);
    auto host_dst = std::vector<int>(num_elements, 0);

    int* dev_0 = nullptr;
    HIP_API_CALL(hipSetDevice(0));
    HIP_API_CALL(hipMalloc(&dev_0, num_bytes));
    HIP_API_CALL(hipMemcpy(dev_0, host_src.data(), num_bytes, hipMemcpyHostToDevice));

    int can_access_peer = 0;
    if(num_devices >= 2) HIP_API_CALL(hipDeviceCanAccessPeer(&can_access_peer, 1, 0));

    if(can_access_peer != 0)
    {
        int* dev_1 = nullptr;
        HIP_API_CALL(hipSetDevice(1));
        HIP_API_CALL(hipMalloc(&dev_1, num_bytes));
        HIP_API_CALL(hipDeviceEnablePeerAccess(0, 0));

        HIP_API_CALL(hipMemcpyPeer(dev_1, 1, dev_0, 0, num_bytes));
        HIP_API_CALL(hipDeviceSynchronize());
        HIP_API_CALL(hipMemcpy(host_dst.data(), dev_1, num_bytes, hipMemcpyDeviceToHost));
        HIP_API_CALL(hipFree(dev_1));
    }
    else
    {
        printf("peer-memcpy: %i device(s), no peer access from device 1 to device 0, "
               "skipping peer copy\n",
               num_devices);
        HIP_API_CALL(hipMemcpy(host_dst.data(), dev_0, num_bytes, hipMemcpyDeviceToHost));
    }

    HIP_API_CALL(hipSetDevice(0));
    HIP_API_CALL(hipFree(dev_0));

    if(host_dst != host_src)
    {
        fprintf(stderr, "peer-memcpy: copied data does not match the source\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
