// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

// Application half of the range replay perf regression tests.
//
// Modes, selected by environment (at most one):
//
// - Default: time a number of ranges, each containing a configurable number of dispatches over a
//   configurable device-memory footprint. The dispatch count per range is a parameter rather than
//   a constant because it is the axis the amortization test sweeps. Range replay's reason to exist
//   is that the fixed cost of a replay window -- agent drain, snapshot of the tracked inventory,
//   restore -- is paid once per range instead of once per dispatch, so wall time should grow far
//   more slowly than the dispatch count. A regression that reintroduced per-dispatch window cost
//   would still pass a pass-count scaling test and only shows up when this axis moves.
//
// - Idle gate (RR_PERF_IDLE_GATE=1, P1): never open a range. Time a burst of dispatches with range
//   replay configured against the same burst with no tool, so the process-wide interception gate
//   is the only cost under test.
//
// - Copies (RR_PERF_COPIES=1, P2): never open a range. Time a burst of hipMemcpyAsync calls. The
//   copy wrappers stay installed while range replay is configured, and each call classifies the
//   pointers before forwarding, whether or not a range is open and whether or not anything traces
//   memory copies.
//
// - Graphs (RR_PERF_GRAPHS=1, P3): never open a range. Time a burst of hipGraphLaunch calls. While
//   range replay is configured, a graph launch does not take the fast path that writes the
//   original packets through: the service has to see every dispatch, replayed or not.
//
// Like the range replay samples, this resolves the two API entry points with dlsym rather than
// linking rocprofiler-sdk, because linking the SDK into a HIP executable makes HIP report
// hipErrorInvalidDeviceFunction on the first launch on gfx942. The tool is LD_PRELOADed, so the
// symbols are already in the process. The idle-gate path does not need those symbols at all: the
// no-tool baseline has nothing to resolve.

#include "client.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

// Above HIP's default shader-blit threshold (GPU_FORCE_BLIT_COPY_SIZE, 16 KiB), so the copy takes
// the async-copy wrapper rather than a dispatch on the application's queue.
constexpr size_t kCopyBytes = 1U << 20;

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

__global__ void
bump(int* counter)
{
    if(blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(counter, 1);
}

bool
env_flag(const char* name)
{
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

int
skip(const char* reason)
{
    std::printf("[rr-perf] SKIP: %s\n", reason);
    return EXIT_SUCCESS;
}
}  // namespace

int
main(int argc, char** argv)
{
    int device_count = 0;
    if(hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0)
        return skip("no GPU device");

    // ballast_mb: device memory held live across the range, i.e. the snapshot footprint.
    // dispatches: dispatches inside each range -- the amortization axis; or, in idle-gate mode,
    //             the number of timed launches.
    // ranges:     ranges timed in one run (ignored in idle-gate mode).
    // warmup:     untimed ranges (or launches) first. The first pays for ballast page faults,
    //             code-object load and the profiler's attach, none of which repeat.
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

    const bool idle_gate = env_flag("RR_PERF_IDLE_GATE");
    const bool copies    = env_flag("RR_PERF_COPIES");
    const bool graphs    = env_flag("RR_PERF_GRAPHS");
    if(static_cast<int>(idle_gate) + static_cast<int>(copies) + static_cast<int>(graphs) > 1)
    {
        fprintf(
            stderr,
            "[rr-perf] FAIL set only one of RR_PERF_IDLE_GATE, RR_PERF_COPIES, RR_PERF_GRAPHS\n");
        return EXIT_FAILURE;
    }
    // P1, P2 and P3 time the cost of having range replay configured. None of them opens a range,
    // so the no-tool baseline has no range-replay symbols to resolve.
    const bool configured_only = idle_gate || copies || graphs;

    decltype(&rocprofiler_range_replay_begin) begin_fn = nullptr;
    decltype(&rocprofiler_range_replay_end)   end_fn   = nullptr;
    if(!configured_only)
    {
        begin_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_begin)>(
            dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_begin"));
        end_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_end)>(
            dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_end"));
        if(begin_fn == nullptr || end_fn == nullptr)
        {
            fprintf(stderr, "[rr-perf] FAIL the range replay API is not in this process\n");
            return EXIT_FAILURE;
        }
    }

    // The configured-only modes do not snapshot anything. Keep a small ballast so the kernel still
    // writes something, and ignore the caller's ballast_mb.
    const size_t ballast_bytes = configured_only ? static_cast<size_t>(kWorkElems) * sizeof(float)
                                                 : static_cast<size_t>(ballast_mb) * 1024U * 1024U;
    const int    ballast_elems = static_cast<int>(ballast_bytes / sizeof(float));
    const int    work_elems    = (ballast_elems < kWorkElems) ? ballast_elems : kWorkElems;

    float* ballast = nullptr;
    int*   counter = nullptr;
    HIP_CHECK(hipMalloc(&ballast, ballast_bytes));
    HIP_CHECK(hipMalloc(&counter, sizeof(int)));
    HIP_CHECK(hipMemset(ballast, 0, ballast_bytes));
    HIP_CHECK(hipMemset(counter, 0, sizeof(int)));

    const auto launch_burst = [&](int count) -> hipError_t {
        for(int i = 0; i < count; ++i)
        {
            touch<<<32, 64>>>(ballast, work_elems, counter);
            if(auto err = hipGetLastError(); err != hipSuccess) return err;
        }
        return hipDeviceSynchronize();
    };

    if(idle_gate)
    {
        // P1: the interception gate is process-wide once range replay is configured. This path
        // never calls begin/end, so the only cost under measurement is that gate on each submit.
        HIP_CHECK(launch_burst(warmup));
        HIP_CHECK(hipMemset(counter, 0, sizeof(int)));

        using clock      = std::chrono::steady_clock;
        const auto start = clock::now();
        HIP_CHECK(launch_burst(dispatches));
        const auto   end     = clock::now();
        const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();

        int counter_h = 0;
        HIP_CHECK(hipMemcpy(&counter_h, counter, sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(ballast));
        HIP_CHECK(hipFree(counter));

        std::printf("[rr-perf] idle_gate=1 dispatches=%d warmup=%d wall_ms=%.3f counter=%d\n",
                    dispatches,
                    warmup,
                    wall_ms,
                    counter_h);

        if(counter_h != dispatches)
        {
            std::fprintf(stderr, "[rr-perf] FAIL counter=%d expected %d\n", counter_h, dispatches);
            return EXIT_FAILURE;
        }

        std::printf("[rr-perf] PASS\n");
        return EXIT_SUCCESS;
    }

    if(copies)
    {
        // P2: each hipMemcpyAsync pays the copy wrapper's pointer classification while range
        // replay is configured, even though no range is open and nothing traces memory copies.
        void*       src    = nullptr;
        void*       dst    = nullptr;
        hipStream_t stream = nullptr;
        HIP_CHECK(hipHostMalloc(&src, kCopyBytes));
        HIP_CHECK(hipMalloc(&dst, kCopyBytes));
        HIP_CHECK(hipStreamCreate(&stream));
        std::memset(src, 0x5a, kCopyBytes);

        const auto burst = [&](int count) -> hipError_t {
            for(int i = 0; i < count; ++i)
            {
                if(auto err = hipMemcpyAsync(dst, src, kCopyBytes, hipMemcpyHostToDevice, stream);
                   err != hipSuccess)
                    return err;
            }
            return hipStreamSynchronize(stream);
        };

        HIP_CHECK(burst(warmup));

        using clock      = std::chrono::steady_clock;
        const auto start = clock::now();
        HIP_CHECK(burst(dispatches));
        const auto   end     = clock::now();
        const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();

        auto* check = static_cast<unsigned char*>(std::malloc(kCopyBytes));
        if(check == nullptr) return EXIT_FAILURE;
        HIP_CHECK(hipMemcpy(check, dst, kCopyBytes, hipMemcpyDeviceToHost));
        const bool copied = std::memcmp(check, src, kCopyBytes) == 0;
        std::free(check);
        HIP_CHECK(hipStreamDestroy(stream));
        HIP_CHECK(hipFree(dst));
        HIP_CHECK(hipHostFree(src));

        std::printf("[rr-perf] copies=1 bytes=%zu count=%d warmup=%d wall_ms=%.3f ok=%d\n",
                    kCopyBytes,
                    dispatches,
                    warmup,
                    wall_ms,
                    static_cast<int>(copied));
        if(!copied)
        {
            std::fprintf(stderr, "[rr-perf] FAIL the copied bytes do not match the source\n");
            return EXIT_FAILURE;
        }
        std::printf("[rr-perf] PASS\n");
        return EXIT_SUCCESS;
    }

    if(graphs)
    {
        // P3: with range replay configured, graph launch does not take the packet-forwarding fast
        // path. No range is opened, so this times that path and not a replay.
        int*           counter = nullptr;
        hipStream_t    stream  = nullptr;
        hipGraph_t     graph   = nullptr;
        hipGraphExec_t exec    = nullptr;
        HIP_CHECK(hipMalloc(&counter, sizeof(int)));
        HIP_CHECK(hipMemset(counter, 0, sizeof(int)));
        HIP_CHECK(hipStreamCreate(&stream));
        HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        bump<<<1, 64, 0, stream>>>(counter);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamEndCapture(stream, &graph));
        HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

        const auto burst = [&](int count) -> hipError_t {
            for(int i = 0; i < count; ++i)
            {
                if(auto err = hipGraphLaunch(exec, stream); err != hipSuccess) return err;
            }
            return hipStreamSynchronize(stream);
        };

        HIP_CHECK(burst(warmup));
        HIP_CHECK(hipMemset(counter, 0, sizeof(int)));

        using clock      = std::chrono::steady_clock;
        const auto start = clock::now();
        HIP_CHECK(burst(dispatches));
        const auto   end     = clock::now();
        const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();

        int counter_h = 0;
        HIP_CHECK(hipMemcpy(&counter_h, counter, sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipGraphExecDestroy(exec));
        HIP_CHECK(hipGraphDestroy(graph));
        HIP_CHECK(hipStreamDestroy(stream));
        HIP_CHECK(hipFree(counter));

        std::printf("[rr-perf] graphs=1 launches=%d warmup=%d wall_ms=%.3f counter=%d\n",
                    dispatches,
                    warmup,
                    wall_ms,
                    counter_h);
        if(counter_h != dispatches)
        {
            std::fprintf(stderr, "[rr-perf] FAIL counter=%d expected %d\n", counter_h, dispatches);
            return EXIT_FAILURE;
        }
        std::printf("[rr-perf] PASS\n");
        return EXIT_SUCCESS;
    }

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
