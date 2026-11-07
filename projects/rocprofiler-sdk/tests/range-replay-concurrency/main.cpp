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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/**
 * @file tests/range-replay-concurrency/main.cpp
 *
 * @brief Another thread's device writes must survive range replay on the same agent, and neither
 * thread may deadlock.
 *
 * The main thread replays ranges over its own buffer. A second thread writes a new value into its
 * own device buffer -- with hipMemcpyAsync (the copy engine) or with a kernel -- waits for it, and
 * reads a few words back. The write is timed to start just after the main thread calls end(), so
 * it lands while the range's replay window runs: a window that restores the agent's memory after
 * the write landed hands the second thread an older value, and one that holds the replay lock
 * while ringing an intercepted doorbell deadlocks against the second thread's submission. Run with
 * GPU_MAX_HW_QUEUES=1 the two threads share one hardware queue.
 *
 * Environment:
 *   RR_CONC_MODE    packet (default) or host
 *   RR_CONC_WRITER  copy (default) or kernel
 *   RR_CONC_RANGES  ranges to open (default 16)
 */

#include "range.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <setjmp.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#define HIP_OK(call)                                                                               \
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
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while(0)

namespace
{
__global__ void
rr_conc_scale(float* x, size_t n, float s)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if(i < n) x[i] = x[i] * s + 1.0f;
}

__global__ void
rr_conc_fill(uint32_t* x, size_t n, uint32_t v)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if(i < n) x[i] = v;
}

std::atomic<bool>     g_stop{false};
std::atomic<uint64_t> g_end_calls{0};
std::atomic<uint64_t> g_writes{0};
std::atomic<uint64_t> g_lost{0};

void
writer(bool use_kernel)
{
    HIP_OK(hipSetDevice(0));
    constexpr size_t words = kWriterBytes / sizeof(uint32_t);
    uint32_t*        d     = nullptr;
    uint32_t*        h     = nullptr;
    uint32_t*        back  = nullptr;
    HIP_OK(hipMalloc(&d, kWriterBytes));
    HIP_OK(hipHostMalloc(&h, kWriterBytes, hipHostMallocDefault));
    HIP_OK(hipHostMalloc(&back, 3 * sizeof(uint32_t), hipHostMallocDefault));
    hipStream_t s;
    HIP_OK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

    uint64_t seen_end = 0;
    for(uint32_t v = 1; !g_stop.load(); ++v)
    {
        // The first few writes run before any range so the main thread has something to wait for.
        if(v > 3)
        {
            while(!g_stop.load() && g_end_calls.load() == seen_end)
                std::this_thread::yield();
            if(g_stop.load()) break;
            seen_end = g_end_calls.load();
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }

        if(use_kernel)
        {
            rr_conc_fill<<<(words + 255) / 256, 256, 0, s>>>(d, words, v);
            HIP_OK(hipGetLastError());
        }
        else
        {
            for(size_t i = 0; i < words; i += 1024)
                h[i] = v;
            h[words - 1] = v;
            HIP_OK(hipMemcpyAsync(d, h, kWriterBytes, hipMemcpyHostToDevice, s));
        }
        HIP_OK(hipStreamSynchronize(s));

        // Leave a concurrent replay window time to restore over this buffer before reading back.
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        HIP_OK(hipMemcpyAsync(&back[0], d, sizeof(uint32_t), hipMemcpyDeviceToHost, s));
        HIP_OK(hipMemcpyAsync(
            &back[1], d + (words / 2 / 1024) * 1024, sizeof(uint32_t), hipMemcpyDeviceToHost, s));
        HIP_OK(hipMemcpyAsync(&back[2], d + words - 1, sizeof(uint32_t), hipMemcpyDeviceToHost, s));
        HIP_OK(hipStreamSynchronize(s));
        g_writes.fetch_add(1);
        if(back[0] != v || back[1] != v || back[2] != v)
        {
            if(g_lost.fetch_add(1) < 5)
                fprintf(stderr,
                        "[app] lost write: wrote %u, read back %u %u %u\n",
                        v,
                        back[0],
                        back[1],
                        back[2]);
        }
    }

    HIP_OK(hipStreamDestroy(s));
    HIP_OK(hipFree(d));
    HIP_OK(hipHostFree(h));
    HIP_OK(hipHostFree(back));
}

uint64_t
env_u64(const char* name, uint64_t fallback)
{
    const char* v = std::getenv(name);
    return (v && *v) ? std::strtoull(v, nullptr, 10) : fallback;
}
}  // namespace

int
main()
{
    const char* mode_env   = std::getenv("RR_CONC_MODE");
    const char* writer_env = std::getenv("RR_CONC_WRITER");
    const bool  host       = mode_env != nullptr && std::string{mode_env} == "host";
    const bool  use_kernel = writer_env != nullptr && std::string{writer_env} == "kernel";
    const auto  ranges     = env_u64("RR_CONC_RANGES", 16);

    // Resolved at run time rather than linked, as in the range replay samples.
    auto* begin_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_begin)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_begin"));
    auto* end_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_end)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_end"));
    auto* jmp_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_host_jmpbuf)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_host_jmpbuf"));
    if(!begin_fn || !end_fn || (host && !jmp_fn))
    {
        fprintf(stderr, "[app] FAIL: range replay API is not in this process\n");
        return EXIT_FAILURE;
    }

    HIP_OK(hipSetDevice(0));
    constexpr size_t n = kRangeBytes / sizeof(float);
    float*           x = nullptr;
    HIP_OK(hipMalloc(&x, kRangeBytes));
    HIP_OK(hipMemset(x, 0, kRangeBytes));
    rr_conc_scale<<<(n + 255) / 256, 256>>>(x, n, 1.0f);
    HIP_OK(hipDeviceSynchronize());

    std::thread w{writer, use_kernel};
    while(g_writes.load() < 3)
        std::this_thread::yield();

    for(uint64_t r = 0; r < ranges; ++r)
    {
        volatile int st = 0;
        if(host)
        {
            if(sigsetjmp(*static_cast<sigjmp_buf*>(jmp_fn()), 1) == 0)
                st = begin_fn(kRangeIdBase + r);
        }
        else
        {
            st = begin_fn(kRangeIdBase + r);
        }
        if(st != ROCPROFILER_STATUS_SUCCESS)
        {
            fprintf(stderr, "[app] FAIL: begin returned %d\n", st);
            return EXIT_FAILURE;
        }
        rr_conc_scale<<<(n + 255) / 256, 256>>>(x, n, 0.5f);
        HIP_OK(hipDeviceSynchronize());
        g_end_calls.fetch_add(1);
        end_fn();
    }

    g_stop.store(true);
    w.join();
    HIP_OK(hipFree(x));

    fprintf(stderr,
            "[app] mode=%s writer=%s ranges=%lu writes=%lu lost_writes=%lu\n",
            host ? "host" : "packet",
            use_kernel ? "kernel" : "copy",
            static_cast<unsigned long>(ranges),
            static_cast<unsigned long>(g_writes.load()),
            static_cast<unsigned long>(g_lost.load()));
    fprintf(stderr, g_lost.load() == 0 ? "[app] PASS\n" : "[app] FAIL\n");
    return g_lost.load() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
