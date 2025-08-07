// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>

template <typename T>
void
TEST_EXPECT(T&& arg, std::string_view message)
{
    if(!arg)
    {
        std::cerr << "Error: " << message << " ("
                  << "\n";
    }
}

#define HIPCHECK(cmd)                                                                              \
    do                                                                                             \
    {                                                                                              \
        hipError_t err = cmd;                                                                      \
        if(err != hipSuccess)                                                                      \
        {                                                                                          \
            printf("Failed: HIP error %s:%d '%s'\n", __FILE__, __LINE__, hipGetErrorString(err));  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)

#define NCCLCHECK(cmd)                                                                             \
    do                                                                                             \
    {                                                                                              \
        ncclResult_t res = cmd;                                                                    \
        if(res != ncclSuccess)                                                                     \
        {                                                                                          \
            printf(                                                                                \
                "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res));   \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)

int
main(int argc, const char* argv[])
{
    if(argc != 2)
    {
        fprintf(stderr, "Usage: %s allocation size in MiB\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const size_t alloc_size = std::atoll(argv[1]) * 1024UL * 1024UL;
    size_t       nelems     = alloc_size / sizeof(float);
    printf("%s: Allocating %lu Bytes (%lu elements)\n", argv[0], alloc_size, nelems);

    int device_count{};
    HIPCHECK(hipGetDeviceCount(&device_count));
    TEST_EXPECT(device_count != 0, "Device count is zero");

    for(int i = 0; i < device_count; ++i)
    {
        hipDeviceProp_t props{};
        HIPCHECK(hipGetDeviceProperties(&props, i));

        printf("GFX arch: '%s'\n", props.gcnArchName);

        if(std::string_view{props.gcnArchName} == std::string_view{"gfx906"})
        {
            printf("SKIP - %s\n", props.gcnArchName);
            return 0;
        }
    }

    std::vector<int> devs(device_count);
    std::iota(devs.begin(), devs.end(), 0);  // 0, 1, 2, 3 ...
    for(uint32_t i = 0; i < devs.size(); ++i)
    {
        printf("dev[%d]: %d\n", i, devs[i]);
    }

    std::vector<ncclComm_t>  comms(device_count);
    std::vector<hipStream_t> streams(device_count);

    // allocating and initializing device buffers
    std::vector<float*> sendbuff(device_count);
    std::vector<float*> recvbuff(device_count);

    for(int i = 0; i < device_count; ++i)
    {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipMalloc(&sendbuff[i], alloc_size));
        HIPCHECK(hipMalloc(&recvbuff[i], alloc_size));
        HIPCHECK(hipMemset(sendbuff[i], 1, alloc_size));
        HIPCHECK(hipMemset(recvbuff[i], 0, alloc_size));
        HIPCHECK(hipStreamCreate(&streams[i]));
    }

    // initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms.data(), device_count, devs.data()));

    // calling NCCL communication API. Group API is required when using
    // multiple devices per thread
    {
        NCCLCHECK(ncclGroupStart());
        for(int i = 0; i < device_count; ++i)
        {
            NCCLCHECK(ncclAllReduce((const void*) sendbuff[i],
                                    (void*) recvbuff[i],
                                    nelems,
                                    ncclFloat,
                                    ncclSum,
                                    comms[i],
                                    streams[i]));
        }
        NCCLCHECK(ncclGroupEnd());
    }

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for(int i = 0; i < device_count; ++i)
    {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipStreamSynchronize(streams[i]));
    }

    // free device buffers
    for(int i = 0; i < device_count; ++i)
    {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipFree(sendbuff[i]));
        HIPCHECK(hipFree(recvbuff[i]));
    }

    // finalizing NCCL
    for(int i = 0; i < device_count; ++i)
    {
        ncclCommDestroy(comms[i]);
    }

    for(int i = 0; i < device_count; ++i)
    {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipStreamSynchronize(streams[i]));
    }

    printf("Success \n");
    return 0;
}
