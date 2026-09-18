// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (call);                                                                  \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(                                                                               \
                stderr, "HIP error '%s' at %s:%d\n", hipGetErrorString(_err), __FILE__, __LINE__); \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while(0)

__global__ void
touch(float* buf, int n, int* counter)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += stride)
        buf[i] = buf[i] * 1.0001f + 0.001f;
    if(blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(counter, 1);
}

int
main(int argc, char** argv)
{
    int ballast_mb = 8;
    int launches   = 32;
    if(argc > 1) ballast_mb = std::atoi(argv[1]);
    if(argc > 2) launches = std::atoi(argv[2]);
    if(ballast_mb < 1) ballast_mb = 1;
    if(launches < 1) launches = 1;

    const size_t ballast_bytes = static_cast<size_t>(ballast_mb) * 1024U * 1024U;
    const int    ballast_elems = static_cast<int>(ballast_bytes / sizeof(float));

    float* ballast = nullptr;
    int*   counter = nullptr;
    HIP_CHECK(hipMalloc(&ballast, ballast_bytes));
    HIP_CHECK(hipMalloc(&counter, sizeof(int)));
    HIP_CHECK(hipMemset(ballast, 0, ballast_bytes));
    HIP_CHECK(hipMemset(counter, 0, sizeof(int)));

    using clock      = std::chrono::steady_clock;
    const auto start = clock::now();
    for(int i = 0; i < launches; ++i)
    {
        touch<<<256, 64>>>(ballast, ballast_elems, counter);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }
    const auto   end     = clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();

    int counter_h = 0;
    HIP_CHECK(hipMemcpy(&counter_h, counter, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(ballast));
    HIP_CHECK(hipFree(counter));

    std::printf("[qh-perf] ballast_mb=%d launches=%d wall_ms=%.3f counter=%d\n",
                ballast_mb,
                launches,
                wall_ms,
                counter_h);

    if(counter_h != launches)
    {
        std::fprintf(stderr, "[qh-perf] FAIL counter=%d expected %d\n", counter_h, launches);
        return EXIT_FAILURE;
    }

    std::printf("[qh-perf] PASS\n");
    return EXIT_SUCCESS;
}
