// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

// Application half of the range replay samples: a multi-dispatch range bracketed by
// rocprofiler_range_replay_begin / _end.
//
// Range replay is the one replay service the application drives, so unlike the kernel replay
// samples this main has to call into the SDK. It resolves the two entry points with dlsym rather
// than linking rocprofiler-sdk, because linking the SDK into the HIP executable makes HIP report
// hipErrorInvalidDeviceFunction on the first kernel launch on gfx942 (see
// samples/kernel_replay/CMakeLists.txt). The tool library is LD_PRELOADed, so the SDK is already
// in the process and RTLD_DEFAULT finds the symbols.
//
// RR_APP_MODE selects what the application does inside the range, so one application can serve
// both the clients that expect a replay and the ones that expect a decline:
//
//   plain        (default) three dispatches on one stream -- a replayable range
//   multi-queue  the same, plus a dispatch on a second stream, which the SDK declines

#include "range.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <string>

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

namespace
{
// Advance the accumulator. Each dispatch reads what its predecessor wrote, so the three dispatches
// below are order-dependent and not idempotent: a pass that skipped a dispatch, ran them out of
// order, or started from a state the SDK failed to rewind produces a different number.
//
// `add` differs per dispatch, so the three recorded dispatches share a kernel object but not their
// kernargs -- which is what makes the replay stage a distinct kernarg slot for each of them.
__global__ void
step(int* acc, int add)
{
    if(threadIdx.x == 0) *acc = (*acc * 3) + add;
}

}  // namespace

int
main()
{
    auto* begin_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_begin)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_begin"));
    auto* end_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_end)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_end"));

    if(begin_fn == nullptr || end_fn == nullptr)
    {
        fprintf(stderr, "[app] FAIL: the range replay API is not in this process\n");
        return EXIT_FAILURE;
    }

    const auto* mode_env     = getenv("RR_APP_MODE");
    const auto  mode         = std::string{mode_env != nullptr ? mode_env : "plain"};
    const bool  second_queue = (mode == "multi-queue");

    int* acc     = nullptr;
    int* scratch = nullptr;
    HIP_CHECK(hipMalloc(&acc, sizeof(int)));
    HIP_CHECK(hipMalloc(&scratch, sizeof(int)));
    HIP_CHECK(hipMemset(acc, 0, sizeof(int)));
    HIP_CHECK(hipMemset(scratch, 0, sizeof(int)));

    // The second stream is created before the range opens: HIP backs a stream with an HSA queue,
    // and creating one inside the range would be a second thing for the SDK to object to. The
    // sample wants exactly one decline reason in play.
    hipStream_t other = nullptr;
    if(second_queue) HIP_CHECK(hipStreamCreate(&other));

    // Everything the range depends on is allocated and settled before it opens: a device
    // allocation inside a range is itself a decline reason.
    HIP_CHECK(hipDeviceSynchronize());

    if(const auto status = begin_fn(kRangeId); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: rocprofiler_range_replay_begin returned status %d\n",
                static_cast<int>(status));
        return EXIT_FAILURE;
    }

    // 0 -> 1 -> 6 -> 21
    for(uint64_t add = 1; add <= kRangeDispatches; ++add)
    {
        step<<<1, 1>>>(acc, static_cast<int>(add));
        HIP_CHECK(hipGetLastError());
    }

    // A dispatch on a second queue inside the range. It writes `scratch`, not `acc`, so the
    // accumulator chain -- and therefore the check at the end -- is the same in both modes.
    if(second_queue)
    {
        step<<<1, 1, 0, other>>>(scratch, 1);
        HIP_CHECK(hipGetLastError());
    }

    // The range's own work has to be complete before it closes: _end re-executes the recording
    // from the range-entry snapshot, and the host has to have observed the live run first.
    HIP_CHECK(hipDeviceSynchronize());

    if(const auto status = end_fn(); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: rocprofiler_range_replay_end returned status %d\n",
                static_cast<int>(status));
        return EXIT_FAILURE;
    }

    int acc_h = 0;
    HIP_CHECK(hipMemcpy(&acc_h, acc, sizeof(int), hipMemcpyDeviceToHost));
    if(other != nullptr) HIP_CHECK(hipStreamDestroy(other));
    HIP_CHECK(hipFree(acc));
    HIP_CHECK(hipFree(scratch));

    printf("[app] mode=%s acc=%d\n", mode.c_str(), acc_h);

    // The application must see exactly the value its own execution of the range produced, whether
    // the range was replayed or declined. A replayed range that leaks a pass's result would show
    // up here as 64/195/588 (the chain continued from 21) rather than 21.
    if(acc_h != kExpectedResult)
    {
        fprintf(stderr,
                "[app] FAIL: acc=%d (expected %d; the replay did not hand back the "
                "application's own result)\n",
                acc_h,
                kExpectedResult);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
