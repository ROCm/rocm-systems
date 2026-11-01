// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

// Application half of the range replay perf regression tests: time a number of ranges, each
// containing a configurable number of dispatches over a configurable device-memory footprint.
//
// The dispatch count per range is a parameter rather than a constant because it is the axis the
// amortization test sweeps. Range replay's reason to exist is that the fixed cost of a replay
// window -- agent drain, snapshot of the tracked inventory, restore -- is paid once per range
// instead of once per dispatch, so wall time should grow far more slowly than the dispatch count.
// A regression that reintroduced per-dispatch window cost would still pass a pass-count scaling
// test and only shows up when this axis moves.
//
// Like the range replay samples, this resolves the two API entry points with dlsym rather than
// linking rocprofiler-sdk, because linking the SDK into a HIP executable makes HIP report
// hipErrorInvalidDeviceFunction on the first launch on gfx942. The tool is LD_PRELOADed, so the
// symbols are already in the process.

#include "client.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <chrono>
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

namespace
{
// Elements each dispatch actually writes. Deliberately tiny and independent of the ballast size.
//
// The two costs in a replayed range scale on different axes: the snapshot and the between-pass
// restores are proportional to the tracked footprint and are paid once per range, while the
// dispatches are paid per dispatch per pass. Sizing the kernel's working set to the ballast would
// tie them together, and the amortization sweep -- which varies the dispatch count against a fixed
// footprint -- would then measure GPU work growing linearly with the dispatch count and conclude
// nothing about whether the window cost was amortized. Keeping the dispatch cheap and the
// footprint large is what makes the window cost the dominant term and the sweep meaningful.
constexpr int kWorkElems = 16384;

// Bumps a counter so the application can tell afterwards whether the replay handed back its own
// result, and writes a small slice of the ballast so the dispatches are not optimized away.
__global__ void
touch(float* ballast, int n, int* counter)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += stride)
        ballast[i] = ballast[i] * 1.0001f + 0.001f;
    if(blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(counter, 1);
}
}  // namespace

int
main(int argc, char** argv)
{
    // ballast_mb: device memory held live across the range, i.e. the snapshot footprint.
    // dispatches: dispatches inside each range -- the amortization axis.
    // ranges:     ranges timed in one run.
    // warmup:     untimed ranges first. The first range pays for ballast page faults, code-object
    //             load and the profiler's attach, none of which repeat.
    int ballast_mb = 32;
    int dispatches = 4;
    int ranges     = 2;
    int warmup     = 1;
    if(argc > 1) ballast_mb = std::atoi(argv[1]);
    if(argc > 2) dispatches = std::atoi(argv[2]);
    if(argc > 3) ranges = std::atoi(argv[3]);
    if(argc > 4) warmup = std::atoi(argv[4]);
    if(ballast_mb < 1) ballast_mb = 1;
    if(dispatches < 1) dispatches = 1;
    if(ranges < 1) ranges = 1;
    if(warmup < 0) warmup = 0;

    auto* begin_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_begin)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_begin"));
    auto* end_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_end)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_end"));
    if(begin_fn == nullptr || end_fn == nullptr)
    {
        fprintf(stderr, "[rr-perf] FAIL the range replay API is not in this process\n");
        return EXIT_FAILURE;
    }

    const size_t ballast_bytes = static_cast<size_t>(ballast_mb) * 1024U * 1024U;
    const int    ballast_elems = static_cast<int>(ballast_bytes / sizeof(float));
    const int    work_elems    = (ballast_elems < kWorkElems) ? ballast_elems : kWorkElems;

    float* ballast = nullptr;
    int*   counter = nullptr;
    HIP_CHECK(hipMalloc(&ballast, ballast_bytes));
    HIP_CHECK(hipMalloc(&counter, sizeof(int)));
    HIP_CHECK(hipMemset(ballast, 0, ballast_bytes));
    HIP_CHECK(hipMemset(counter, 0, sizeof(int)));

    // One range, start to finish. Allocation inside a range is a decline reason, so everything the
    // range touches is allocated above and only launches happen in here.
    const auto run_range = [&](uint64_t range_id) -> hipError_t {
        if(const auto status = begin_fn(range_id); status != ROCPROFILER_STATUS_SUCCESS)
        {
            // Distinct from a decline, which is reported to the tool at CLOSE and leaves this
            // call successful. Reaching here means the range never opened at all.
            fprintf(stderr,
                    "[rr-perf] FAIL rocprofiler_range_replay_begin returned status %d\n",
                    static_cast<int>(status));
            return hipErrorUnknown;
        }
        for(int i = 0; i < dispatches; ++i)
        {
            touch<<<32, 64>>>(ballast, work_elems, counter);
            if(auto err = hipGetLastError(); err != hipSuccess) return err;
        }
        // The range's own work must complete before it closes: _end re-executes the recording from
        // the entry snapshot, and the host has to have observed the live run first.
        if(auto err = hipDeviceSynchronize(); err != hipSuccess) return err;
        if(const auto status = end_fn(); status != ROCPROFILER_STATUS_SUCCESS)
        {
            fprintf(stderr,
                    "[rr-perf] FAIL rocprofiler_range_replay_end returned status %d\n",
                    static_cast<int>(status));
            return hipErrorUnknown;
        }
        return hipSuccess;
    };

    for(int i = 0; i < warmup; ++i)
        HIP_CHECK(run_range(kPerfRangeId));

    // Warmup dispatches must not be in the counter: it is the transparency check below.
    HIP_CHECK(hipMemset(counter, 0, sizeof(int)));

    using clock      = std::chrono::steady_clock;
    const auto start = clock::now();
    for(int i = 0; i < ranges; ++i)
        HIP_CHECK(run_range(kPerfRangeId));
    const auto   end     = clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();

    int counter_h = 0;
    HIP_CHECK(hipMemcpy(&counter_h, counter, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(ballast));
    HIP_CHECK(hipFree(counter));

    std::printf(
        "[rr-perf] ballast_mb=%d dispatches=%d ranges=%d warmup=%d wall_ms=%.3f counter=%d\n",
        ballast_mb,
        dispatches,
        ranges,
        warmup,
        wall_ms,
        counter_h);

    // Each range's exit restore puts back the state the application's own execution produced, so
    // the counter must read exactly the number of dispatches the application issued -- no more,
    // even though the replay ran them several more times. A replay that leaked a pass's writes
    // past the window, or failed to restore, lands here.
    const int expected = dispatches * ranges;
    if(counter_h != expected)
    {
        std::fprintf(stderr,
                     "[rr-perf] FAIL counter=%d expected %d (the replay did not hand back the "
                     "application's own result)\n",
                     counter_h,
                     expected);
        return EXIT_FAILURE;
    }

    std::printf("[rr-perf] PASS\n");
    return EXIT_SUCCESS;
}
