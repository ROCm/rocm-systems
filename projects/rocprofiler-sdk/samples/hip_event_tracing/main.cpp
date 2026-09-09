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

/**
 * @file samples/hip_event_tracing/main.cpp
 *
 * @brief HIP workload exercising the two interesting hipStreamWaitEvent outcomes:
 *        a wait that must be scheduled as a GPU barrier, and a wait that is elided
 *        because the recorded event has already completed. All iterations of the
 *        first case run before any iteration of the second.
 */

#include "client.hpp"

#include "common/defines.hpp"
#include "hip/hip_runtime.h"

#include <libgen.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            auto _hip_api_print_lk = auto_lock_t{print_lock};                                      \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }

namespace
{
using auto_lock_t = std::unique_lock<std::mutex>;
auto print_lock   = std::mutex{};

constexpr int block_size = 64;

// stream B's kernel is deliberately shorter than stream A's, so that in the deferred-wait
// case the long kernel on A, and not the work on B, is what the barrier is waiting for
constexpr uint64_t follower_divisor = 16;

void
check_hip_error(void);

void
say(const std::string& msg)
{
    client::narrate(msg.c_str());
}

// Wait for the tool to observe another HIP_EVENT_WAIT barrier completion. The completion
// callback is delivered on the tool's barrier signal handler thread, so it can lag the
// hipStreamSynchronize that guarantees the barrier itself has run; polling avoids reporting
// a false negative just because the callback has not landed yet.
bool
await_wait_barrier(uint64_t baseline)
{
    constexpr auto max_wait = std::chrono::milliseconds{500};
    constexpr auto interval = std::chrono::milliseconds{1};

    for(auto waited = std::chrono::milliseconds{0}; waited < max_wait; waited += interval)
    {
        if(client::wait_barriers_completed() > baseline) return true;
        std::this_thread::sleep_for(interval);
    }

    return client::wait_barriers_completed() > baseline;
}
}  // namespace

// Deliberately long-running: a dependent FMA chain whose duration is set by the loop count
// rather than by occupancy, so that a record barrier enqueued behind it is still in flight
// when the host reaches the corresponding hipStreamWaitEvent. Runs on stream A.
__global__ void
spin_kernel(float* out, uint64_t iters)
{
    auto  tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    float acc = static_cast<float>(tid) + 1.0f;
    for(uint64_t i = 0; i < iters; ++i)
        acc = __fmaf_rn(acc, 1.0000001f, 1.0f);
    out[tid] = acc;
}

// Stream B's work, run after every wait. Same shape as spin_kernel but much shorter, and a
// distinct symbol so that it is identifiable in the kernel dispatch records.
__global__ void
follower_kernel(float* out, uint64_t iters)
{
    auto  tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    float acc = static_cast<float>(tid) + 2.0f;
    for(uint64_t i = 0; i < iters; ++i)
        acc = __fmaf_rn(acc, 1.0000001f, 1.0f);
    out[tid] = acc;
}

// A short kernel used on stream A in the already-complete case.
__global__ void
scale_kernel(float* data, int n, float factor)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < n) data[tid] = data[tid] * factor;
}

namespace
{
// Case 1: the event is recorded behind a long-running kernel, so it has not completed when
// stream B asks to wait on it. HIP must schedule a real wait barrier on stream B, and the
// tracer reports a HIP_EVENT WAIT whose source_queue_id is stream A's queue.
void
run_deferred_wait(int         iteration,
                  int         iterations,
                  hipStream_t stream_a,
                  hipStream_t stream_b,
                  hipEvent_t  event,
                  float*      spin_out,
                  float*      follower_out,
                  uint64_t    spin_iters)
{
    say("case 1, iteration " + std::to_string(iteration + 1) + " of " + std::to_string(iterations) +
        ".");

    say("launching the long spin_kernel on stream A");
    spin_kernel<<<1, block_size, 0, stream_a>>>(spin_out, spin_iters);
    check_hip_error();

    say("recording the event on stream A while spin_kernel is still running; expect a "
        "HIP_EVENT_RECORD barrier on stream A's queue");
    HIP_API_CALL(hipEventRecord(event, stream_a));

    say("stream B waits on stream A's not-yet-complete event; expect a HIP_EVENT_WAIT "
        "barrier on stream B's queue naming stream A's queue as its source");
    const auto waits_before = client::wait_barriers_completed();
    HIP_API_CALL(hipStreamWaitEvent(stream_b, event, 0));

    say("launching follower_kernel on stream B; it cannot start until the wait barrier clears");
    follower_kernel<<<1, block_size, 0, stream_b>>>(follower_out, spin_iters / follower_divisor);
    check_hip_error();

    say("synchronizing both streams");
    HIP_API_CALL(hipStreamSynchronize(stream_a));
    HIP_API_CALL(hipStreamSynchronize(stream_b));

    // This case only demonstrates what it claims to if spin_kernel was still running when
    // the host reached hipStreamWaitEvent. That is a timing property, not a guarantee, so
    // say so plainly rather than letting the run look like a successful demonstration.
    if(!await_wait_barrier(waits_before))
    {
        say("NOTE: no HIP_EVENT_WAIT barrier completed on stream B this iteration. "
            "spin_kernel finished before the host reached hipStreamWaitEvent, so the event "
            "was already complete and HIP had nothing to wait for. Re-run with a larger "
            "--spin-iters to keep the recorded event pending long enough.");
    }
}

// Case 2: the event is fully complete before stream B waits on it, so there is nothing to
// wait for and HIP schedules no barrier at all. The tracer emits a HIP_EVENT_RECORD for the
// record and nothing whatsoever for the wait.
void
run_completed_wait(int         iteration,
                   int         iterations,
                   hipStream_t stream_a,
                   hipStream_t stream_b,
                   hipEvent_t  event,
                   float*      follower_out,
                   float*      data,
                   int         n,
                   uint64_t    spin_iters)
{
    auto grid = (n + block_size - 1) / block_size;

    say("case 2, iteration " + std::to_string(iteration + 1) + " of " + std::to_string(iterations) +
        ".");

    say("launching the short scale_kernel on stream A");
    scale_kernel<<<grid, block_size, 0, stream_a>>>(data, n, 0.99f);
    check_hip_error();

    say("recording the event on stream A; expect a HIP_EVENT_RECORD barrier");
    HIP_API_CALL(hipEventRecord(event, stream_a));

    say("synchronizing on the event so that it is definitely complete");
    HIP_API_CALL(hipEventSynchronize(event));
    HIP_API_CALL(hipStreamSynchronize(stream_a));

    say("stream B waits on the already-complete event; expect NO HIP_EVENT_WAIT record at "
        "all, because HIP has nothing to wait for and schedules no barrier");
    HIP_API_CALL(hipStreamWaitEvent(stream_b, event, 0));

    say("launching follower_kernel on stream B; nothing gates it");
    follower_kernel<<<1, block_size, 0, stream_b>>>(follower_out, spin_iters / follower_divisor);
    check_hip_error();

    say("synchronizing stream B");
    HIP_API_CALL(hipStreamSynchronize(stream_b));
}
}  // namespace

int
main(int argc, char** argv)
{
    client::setup();  // forces rocprofiler to configure/initialize
    client::start();  // starts context before any API tables are available

    auto* exe_name = basename(argv[0]);

    int      n          = 1024;
    int      iterations = 2;
    uint64_t spin_iters = 5000000;

    for(int i = 1; i < argc; ++i)
    {
        auto arg = std::string{argv[i]};
        auto val = [&]() { return (i + 1 < argc) ? atoll(argv[++i]) : 0; };

        if(arg == "--size")
            n = static_cast<int>(val());
        else if(arg == "--iterations")
            iterations = static_cast<int>(val());
        else if(arg == "--spin-iters")
            spin_iters = static_cast<uint64_t>(val());
        else if(arg == "?" || arg == "-h" || arg == "--help")
        {
            fprintf(stderr,
                    "usage: %s [--size %i] [--iterations %i] [--spin-iters %lu]\n",
                    exe_name,
                    n,
                    iterations,
                    static_cast<unsigned long>(spin_iters));
            fprintf(stderr, "  --size        elements per scale_kernel launch\n");
            fprintf(stderr, "  --iterations  iterations of each case\n");
            fprintf(stderr,
                    "  --spin-iters  loop count of the long kernel that keeps a record "
                    "barrier in flight\n");
            exit(EXIT_SUCCESS);
        }
    }

    if(n <= 0 || iterations <= 0 || spin_iters < follower_divisor)
    {
        fprintf(stderr,
                "error: --size and --iterations must be positive and --spin-iters must be at "
                "least %lu\n",
                static_cast<unsigned long>(follower_divisor));
        return EXIT_FAILURE;
    }

    int ndevice = 0;
    HIP_API_CALL(hipGetDeviceCount(&ndevice));
    if(ndevice < 1)
    {
        fprintf(stderr, "error: no HIP devices found\n");
        return EXIT_FAILURE;
    }

    HIP_API_CALL(hipSetDevice(0));

    auto* stream_a = hipStream_t{};
    auto* stream_b = hipStream_t{};
    HIP_API_CALL(hipStreamCreate(&stream_a));
    HIP_API_CALL(hipStreamCreate(&stream_b));

    // one event per case so that the event handle in the trace identifies which case a
    // barrier belongs to
    auto* deferred_event  = hipEvent_t{};
    auto* completed_event = hipEvent_t{};
    HIP_API_CALL(hipEventCreate(&deferred_event));
    HIP_API_CALL(hipEventCreate(&completed_event));

    float* spin_out     = nullptr;
    float* follower_out = nullptr;
    float* data         = nullptr;
    HIP_API_CALL(hipMalloc(&spin_out, block_size * sizeof(float)));
    HIP_API_CALL(hipMalloc(&follower_out, block_size * sizeof(float)));
    HIP_API_CALL(hipMalloc(&data, n * sizeof(float)));
    HIP_API_CALL(hipMemset(spin_out, 0, block_size * sizeof(float)));
    HIP_API_CALL(hipMemset(follower_out, 0, block_size * sizeof(float)));
    HIP_API_CALL(hipMemset(data, 0, n * sizeof(float)));

    {
        auto msg = std::stringstream{};
        msg << exe_name << ": " << iterations << " iteration(s) of each case, " << n
            << " elements, spin_kernel loop count " << spin_iters;
        say(msg.str());
    }

    // The first touch of each stream and kernel drags in one-time work: the
    // fillBufferAligned blits behind hipMemset, the code object load, queue creation. Its
    // dispatch records would otherwise be delivered in the middle of the first traced
    // iteration. Run one of everything, synchronize, and flush so the cases below start
    // from a quiet state.
    client::banner("WARM-UP: running one of each kernel so that one-time setup work does not "
                   "land in the middle of the first traced iteration");

    spin_kernel<<<1, block_size, 0, stream_a>>>(spin_out, spin_iters / follower_divisor);
    check_hip_error();
    scale_kernel<<<(n + block_size - 1) / block_size, block_size, 0, stream_a>>>(data, n, 1.0f);
    check_hip_error();
    follower_kernel<<<1, block_size, 0, stream_b>>>(follower_out, spin_iters / follower_divisor);
    check_hip_error();

    HIP_API_CALL(hipStreamSynchronize(stream_a));
    HIP_API_CALL(hipStreamSynchronize(stream_b));
    HIP_API_CALL(hipDeviceSynchronize());

    say("flushing the warm-up records");
    client::flush();

    client::banner("CASE 1. DEFERRED WAIT: the recorded event is still pending when stream B "
                   "waits, so HIP schedules a real barrier");

    for(int i = 0; i < iterations; ++i)
        run_deferred_wait(
            i, iterations, stream_a, stream_b, deferred_event, spin_out, follower_out, spin_iters);

    // Nothing above forced the buffer to deliver, so every barrier record from every
    // iteration of this case has been accumulating.
    say("all case 1 iterations are complete; flushing the buffer to take delivery of every "
        "barrier record collected across them at once");
    client::flush();

    client::banner("CASE 2. ALREADY-COMPLETE WAIT: the recorded event has finished before "
                   "stream B waits, so HIP schedules no barrier and nothing is traced");

    for(int i = 0; i < iterations; ++i)
        run_completed_wait(
            i, iterations, stream_a, stream_b, completed_event, follower_out, data, n, spin_iters);

    say("all case 2 iterations are complete; flushing the buffer again");
    client::flush();

    client::banner("TEARDOWN");

    HIP_API_CALL(hipFree(spin_out));
    HIP_API_CALL(hipFree(follower_out));
    HIP_API_CALL(hipFree(data));
    HIP_API_CALL(hipEventDestroy(deferred_event));
    HIP_API_CALL(hipEventDestroy(completed_event));
    HIP_API_CALL(hipStreamDestroy(stream_a));
    HIP_API_CALL(hipStreamDestroy(stream_b));
    HIP_API_CALL(hipDeviceSynchronize());

    client::stop();
    client::shutdown();

    return 0;
}

namespace
{
void
check_hip_error(void)
{
    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        auto_lock_t _lk{print_lock};
        fprintf(stderr, "Error: %s\n", hipGetErrorString(err));
        throw std::runtime_error("hip_api_call");
    }
}
}  // namespace
