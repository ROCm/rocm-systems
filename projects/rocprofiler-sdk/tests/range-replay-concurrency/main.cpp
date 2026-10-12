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

// Two threads on one agent. The main thread records a range and replays it in rocprofiler_range_
// replay_end(); a second thread launches HIP graphs into its own stream as soon as the tool reports
// that the range's replay window is open. Afterwards each thread's buffer must hold exactly what
// its own work computed: the range's result, and one increment per graph launch.
//
// Everything both threads use is created before the range opens: a device allocation inside the
// range would decline it, and so would a dispatch from the second thread while it is recording.

#include "range.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (call);                                                                  \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[app] FAIL: HIP error '%s' at %s:%d\n",                                       \
                    hipGetErrorString(_err),                                                       \
                    __FILE__,                                                                      \
                    __LINE__);                                                                     \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while(0)

__global__ void
rr_concurrency_step(int* acc, int add)
{
    if(threadIdx.x == 0) *acc = (*acc * 3) + add;
}

__global__ void
rr_concurrency_count(int* counter)
{
    if(threadIdx.x == 0) *counter += 1;
}

namespace
{
int
skip(const char* reason)
{
    setenv(kSkipEnv, reason, 1);
    printf("[app] SKIP: %s\n", reason);
    return EXIT_SUCCESS;
}

// Bounded, so a range that is never replayed fails the test with a message instead of hanging it.
bool
wait_for_window(const rr_coord_t& coord)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
    while(coord.window_open.load() == 0 && coord.range_closed.load() == 0)
    {
        if(std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds{100});
    }
    return true;
}

// Joins the graph thread on every exit path, including the early returns of HIP_CHECK.
struct thread_joiner
{
    std::thread& thread;
    ~thread_joiner()
    {
        if(thread.joinable()) thread.join();
    }
};
}  // namespace

int
main()
{
    auto device_count = 0;
    if(hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0)
        return skip("no GPU device");

    auto* begin_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_begin)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_begin"));
    auto* end_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_end)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_end"));
    auto* coord_get = reinterpret_cast<rr_coord_t* (*) ()>(dlsym(RTLD_DEFAULT, "rr_coord_get"));
    if(begin_fn == nullptr || end_fn == nullptr || coord_get == nullptr)
    {
        fprintf(stderr, "[app] FAIL: the range replay API or the tool's handshake is missing\n");
        return EXIT_FAILURE;
    }
    auto& coord = *coord_get();

    // The range's thread: its result, and ballast that makes each replay restore take time.
    int*        acc          = nullptr;
    void*       ballast      = nullptr;
    hipStream_t range_stream = nullptr;
    HIP_CHECK(hipMalloc(&acc, sizeof(int)));
    HIP_CHECK(hipMalloc(&ballast, kBallastBytes));
    HIP_CHECK(hipMemset(acc, 0, sizeof(int)));
    HIP_CHECK(hipMemset(ballast, 0, kBallastBytes));
    HIP_CHECK(hipStreamCreate(&range_stream));

    // The graph thread: its counter, its stream, and a one-kernel graph captured from that stream.
    int*           counter      = nullptr;
    hipStream_t    graph_stream = nullptr;
    hipGraph_t     graph        = nullptr;
    hipGraphExec_t graph_exec   = nullptr;
    HIP_CHECK(hipMalloc(&counter, sizeof(int)));
    HIP_CHECK(hipMemset(counter, 0, sizeof(int)));
    HIP_CHECK(hipStreamCreate(&graph_stream));
    HIP_CHECK(hipStreamBeginCapture(graph_stream, hipStreamCaptureModeThreadLocal));
    rr_concurrency_count<<<1, 64, 0, graph_stream>>>(counter);
    HIP_CHECK(hipStreamEndCapture(graph_stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    // One launch before the range opens, so loading the code object and setting up the executable
    // graph happen outside it.
    HIP_CHECK(hipGraphLaunch(graph_exec, graph_stream));
    HIP_CHECK(hipDeviceSynchronize());

    auto graph_failed      = std::atomic<int>{0};
    auto graph_thread      = std::thread{[&]() {
        if(!wait_for_window(coord))
        {
            fprintf(stderr, "[app] FAIL: timed out waiting for the replay window\n");
            graph_failed.store(1);
            return;
        }
        coord.launches_started.store(1);
        for(int i = 0; i < kGraphLaunches; ++i)
        {
            if(hipGraphLaunch(graph_exec, graph_stream) != hipSuccess)
            {
                graph_failed.store(1);
                return;
            }
        }
        if(hipStreamSynchronize(graph_stream) != hipSuccess) graph_failed.store(1);
    }};
    auto join_graph_thread = thread_joiner{graph_thread};

    if(const auto status = begin_fn(kRangeId); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: rocprofiler_range_replay_begin returned status %d\n",
                static_cast<int>(status));
        return EXIT_FAILURE;
    }

    for(uint64_t add = 1; add <= kRangeDispatches; ++add)
    {
        rr_concurrency_step<<<1, 64, 0, range_stream>>>(acc, static_cast<int>(add));
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(range_stream));

    if(const auto status = end_fn(); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: rocprofiler_range_replay_end returned status %d\n",
                static_cast<int>(status));
        return EXIT_FAILURE;
    }

    graph_thread.join();
    if(graph_failed.load() != 0)
    {
        fprintf(stderr, "[app] FAIL: the graph thread did not complete its launches\n");
        return EXIT_FAILURE;
    }

    auto acc_h     = 0;
    auto counter_h = 0;
    HIP_CHECK(hipMemcpy(&acc_h, acc, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&counter_h, counter, sizeof(int), hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(graph_stream));
    HIP_CHECK(hipStreamDestroy(range_stream));
    HIP_CHECK(hipFree(counter));
    HIP_CHECK(hipFree(ballast));
    HIP_CHECK(hipFree(acc));

    printf("[app] acc=%d counter=%d\n", acc_h, counter_h);
    auto ok = true;
    if(acc_h != kExpectedResult)
    {
        fprintf(stderr, "[app] FAIL: acc=%d (expected %d)\n", acc_h, kExpectedResult);
        ok = false;
    }
    // A graph that ran inside the replay window had its increment restored away.
    if(counter_h != kExpectedCounter)
    {
        fprintf(stderr,
                "[app] FAIL: counter=%d (expected %d: one warm-up launch and %d during the "
                "window)\n",
                counter_h,
                kExpectedCounter,
                kGraphLaunches);
        ok = false;
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
