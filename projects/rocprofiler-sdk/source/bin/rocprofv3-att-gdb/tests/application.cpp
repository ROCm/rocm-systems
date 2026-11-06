// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
// No ROCTx calls or ROCTx link dependency. Anchors are ordinary CPU functions.
#include <hip/hip_runtime.h>
#include <unistd.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

static void
check(hipError_t error)
{
    if(error != hipSuccess)
    {
        std::fprintf(stderr, "%s\n", hipGetErrorString(error));
        std::exit(1);
    }
}
__global__ void
before_kernel(int* data)
{
    data[threadIdx.x + blockIdx.x * blockDim.x] += 1;
}
__global__ void
inside_kernel(int* data)
{
    data[threadIdx.x + blockIdx.x * blockDim.x] += 2;
}
__global__ void
warmup_kernel(int* data)
{
    data[threadIdx.x + blockIdx.x * blockDim.x] += 2;
}
__global__ void
after_kernel(int* data)
{
    data[threadIdx.x + blockIdx.x * blockDim.x] += 4;
}
extern "C" __attribute__((noinline)) void
capture_begin()
{
    asm volatile("" ::: "memory");
}
extern "C" __attribute__((noinline)) void
capture_end()
{
    asm volatile("" ::: "memory");
}
extern "C" __attribute__((noinline)) void
between_captures()
{
    asm volatile("" ::: "memory");
}

int
main(int argc, char** argv)
{
    int* data = nullptr;
    check(hipMalloc(&data, 8192 * sizeof(int)));
    check(hipMemset(data, 0, 8192 * sizeof(int)));
    bool warmups     = argc > 1 && !std::strcmp(argv[1], "warmup");
    int  repetitions = warmups ? 101 : (argc > 1 && !std::strcmp(argv[1], "repeat")) ? 2 : 1;
    for(int i = 0; i < repetitions; ++i)
    {
        before_kernel<<<64, 128>>>(data);
        check(hipDeviceSynchronize());
        if(argc > 1 && !std::strcmp(argv[1], "concurrent"))
        {
            std::atomic<int> ready{0};
            auto             hit = [&]() {
                ready.fetch_add(1);
                while(ready.load() != 2)
                    std::this_thread::yield();
                capture_begin();
            };
            std::thread left(hit), right(hit);
            left.join();
            right.join();
        }
        else
            capture_begin();
        if(warmups && i < 100)
            warmup_kernel<<<64, 128>>>(data);
        else
            inside_kernel<<<64, 128>>>(data);
        check(hipDeviceSynchronize());
        if(argc > 1 && !std::strcmp(argv[1], "timeout")) usleep(200000);
        capture_end();
        after_kernel<<<64, 128>>>(data);
        check(hipDeviceSynchronize());
        between_captures();
    }
    int result = 0;
    check(hipMemcpy(&result, data, sizeof(int), hipMemcpyDeviceToHost));
    check(hipFree(data));
    if(argc > 2) std::printf("ARGUMENT=%s\n", argv[2]);
    std::printf("RESULT=%d\n", result);
    return result == 7 * repetitions ? 0 : 1;
}
