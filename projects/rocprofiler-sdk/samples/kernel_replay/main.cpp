// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

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

// Replayed kernel: distinctive block size 67. Busy enough for host-trap PC sampling.
constexpr int kReplayBlock = 67;
// Opt-out kernel: distinctive block size 64. Tools leave pass_count_cb NULL for this dispatch.
constexpr int kOptOutBlock = 64;

__global__ void
bump(int* x)
{
    volatile unsigned acc = 0;
    for(int i = 0; i < 64 * 1024; ++i)
        acc += static_cast<unsigned>(i + threadIdx.x);
    if(threadIdx.x == 0 && acc != ~0u) atomicAdd(x, 1);
}

__global__ void
nudge(int* x)
{
    if(threadIdx.x == 0) atomicAdd(x, 1);
}

int
main()
{
    int* replayed = nullptr;
    int* opted    = nullptr;
    HIP_CHECK(hipMalloc(&replayed, sizeof(int)));
    HIP_CHECK(hipMalloc(&opted, sizeof(int)));
    HIP_CHECK(hipMemset(replayed, 0, sizeof(int)));
    HIP_CHECK(hipMemset(opted, 0, sizeof(int)));

    bump<<<1, kReplayBlock>>>(replayed);
    HIP_CHECK(hipGetLastError());
    nudge<<<1, kOptOutBlock>>>(opted);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    int replayed_h = 0;
    int opted_h    = 0;
    HIP_CHECK(hipMemcpy(&replayed_h, replayed, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&opted_h, opted, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(replayed));
    HIP_CHECK(hipFree(opted));

    printf("[app] replayed_bump=%d opted_nudge=%d\n", replayed_h, opted_h);
    if(replayed_h != 1)
    {
        fprintf(stderr,
                "[app] FAIL: replayed_bump=%d (expected 1; snapshot/restore did not isolate "
                "passes)\n",
                replayed_h);
        return EXIT_FAILURE;
    }
    if(opted_h != 1)
    {
        fprintf(stderr, "[app] FAIL: opted_nudge=%d (expected 1)\n", opted_h);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
