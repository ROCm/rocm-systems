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

// Distinctive block size matched by the perf replay client.
constexpr int kReplayBlock = 67;

__global__ void
touch(float* ballast, int n, int* counter)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += stride)
        ballast[i] = ballast[i] * 1.0001f + 0.001f;
    if(blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(counter, 1);
}

int
main(int argc, char** argv)
{
    // ballast_mb: device memory held live during replay (tracked footprint stress).
    // launches:   number of replayed dispatches timed in one run.
    // warmup:     dispatches run before timing starts. The first replayed dispatch pays for
    //             ballast page faults, code-object load and the profiler's own attach, none of
    //             which repeat, so timing them makes the sample depend on how the run started.
    int ballast_mb = 64;
    int launches   = 8;
    int warmup     = 1;
    if(argc > 1) ballast_mb = std::atoi(argv[1]);
    if(argc > 2) launches = std::atoi(argv[2]);
    if(argc > 3) warmup = std::atoi(argv[3]);
    if(ballast_mb < 1) ballast_mb = 1;
    if(launches < 1) launches = 1;
    if(warmup < 0) warmup = 0;

    const size_t ballast_bytes = static_cast<size_t>(ballast_mb) * 1024U * 1024U;
    const int    ballast_elems = static_cast<int>(ballast_bytes / sizeof(float));

    float* ballast = nullptr;
    int*   counter = nullptr;
    HIP_CHECK(hipMalloc(&ballast, ballast_bytes));
    HIP_CHECK(hipMalloc(&counter, sizeof(int)));
    HIP_CHECK(hipMemset(ballast, 0, ballast_bytes));
    HIP_CHECK(hipMemset(counter, 0, sizeof(int)));

    for(int i = 0; i < warmup; ++i)
    {
        touch<<<256, kReplayBlock>>>(ballast, ballast_elems, counter);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }
    // The counter is the restore check below, so warmup dispatches must not be in it.
    HIP_CHECK(hipMemset(counter, 0, sizeof(int)));

    using clock      = std::chrono::steady_clock;
    const auto start = clock::now();
    for(int i = 0; i < launches; ++i)
    {
        touch<<<256, kReplayBlock>>>(ballast, ballast_elems, counter);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }
    const auto   end     = clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();

    int counter_h = 0;
    HIP_CHECK(hipMemcpy(&counter_h, counter, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(ballast));
    HIP_CHECK(hipFree(counter));

    std::printf("[kr-perf] ballast_mb=%d launches=%d warmup=%d wall_ms=%.3f counter=%d\n",
                ballast_mb,
                launches,
                warmup,
                wall_ms,
                counter_h);

    if(counter_h != launches)
    {
        std::fprintf(stderr,
                     "[kr-perf] FAIL counter=%d expected %d (restore broke kernel side effects)\n",
                     counter_h,
                     launches);
        return EXIT_FAILURE;
    }

    std::printf("[kr-perf] PASS\n");
    return EXIT_SUCCESS;
}
