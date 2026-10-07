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
 * @file tests/kernel-replay-concurrency/main.cpp
 *
 * @brief Deterministic workload for the kernel-replay P0-1 concurrency regression test.
 *
 * Run under client.cpp (LD_PRELOAD), which replays the "hog" (block 256) and exposes a kr_coord_t
 * handshake so the concurrent non-replayed "victim" (block 64) can time its write to land exactly
 * inside a replay window. Two streams on one agent; per cycle k:
 *   victim (main thread): stamp V=OLD; old_ready=k; wait snapshot_done==k; stamp V=NEW;
 *                         wait window_done==k; sync; read V; cycle_done=k
 *   hog (thread A):       wait old_ready==k; launch hog (replayed 5x, snapshot/restore window);
 *                         sync; wait cycle_done==k
 *
 * The replay window snapshots and restores the agent's tracked device memory. If the snapshot
 * over-captures HIP's per-stream kernarg pool (allocated in the coarse-grained segment with the
 * executable flag), restore() tears the victim dispatch's kernel arguments, so its NEW write is
 * lost and the read-back is OLD -- every cycle (100% deterministic). The fix excludes executable
 * allocations from the snapshot, so the kernarg pool is never torn and the victim's write survives.
 *
 *   [repro] PASS => 0 corrupt cycles (fix present)
 *   [repro] FAIL => >0 corrupt cycles (P0-1 reproduced)
 */

#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define HC(call)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        hipError_t _e = (call);                                                                    \
        if(_e != hipSuccess)                                                                       \
        {                                                                                          \
            fprintf(stderr, "HIP '%s' @%d: %s\n", #call, __LINE__, hipGetErrorString(_e));         \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

// Must match the layout exposed by client.cpp's kr_coord_get().
struct kr_coord_t
{
    std::atomic<long> old_ready;
    std::atomic<long> snapshot_done;
    std::atomic<long> window_done;
    std::atomic<long> cycle_done;
};

// block 256 -> replayed "hog" (pure device busy-work; its buffer is never checked).
__global__ void
hog(float* b, int n, int reps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockDim.x * gridDim.x;
    for(int k = 0; k < reps; ++k)
        for(int j = i; j < n; j += s)
            b[j] = b[j] * 1.0000001f + 1.0f;
}

// block 64 -> non-replayed "victim": stamps a sentinel into a tracked device buffer.
__global__ void
stamp(float* v, float val, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockDim.x * gridDim.x;
    for(int j = i; j < n; j += s)
        v[j] = val;
}

// Bounded spin on a phase signal so a mis-wired run (e.g. replay never firing) fails via a clear
// message instead of hanging until the ctest timeout.
static bool
wait_eq(std::atomic<long>& sig, long target)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while(sig.load() != target)
    {
        if(std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

int
main(int argc, char** argv)
{
    const int   cycles   = (argc > 1) ? atoi(argv[1]) : 50;
    const int   n        = (argc > 2) ? atoi(argv[2]) : (1 << 16);
    const int   hog_reps = (argc > 3) ? atoi(argv[3]) : 1000;
    const float OLD = 111.0f, NEW = 222.0f;

    auto getter = reinterpret_cast<kr_coord_t* (*) ()>(dlsym(RTLD_DEFAULT, "kr_coord_get"));
    if(getter == nullptr)
    {
        fprintf(stderr, "[repro] FAIL: no tool handshake (run under LD_PRELOAD=client.so)\n");
        return 3;
    }
    kr_coord_t* c = getter();

    HC(hipSetDevice(0));
    const size_t bytes  = static_cast<size_t>(n) * sizeof(float);
    float *      hogbuf = nullptr, *V = nullptr;
    HC(hipMalloc(&hogbuf, bytes));
    HC(hipMalloc(&V, bytes));
    // NOTE: no hipMemset here on purpose -- HIP's memset fill kernel uses a 256-thread block and
    // would be misclassified as the "hog".
    hipStream_t sa, sb;
    HC(hipStreamCreate(&sa));
    HC(hipStreamCreate(&sb));

    std::atomic<bool> stop{false};
    std::thread       hog_thread([&] {
        for(long k = 1; k <= cycles; ++k)
        {
            while(!stop.load() && c->old_ready.load() != k)
                std::this_thread::yield();
            if(stop.load()) return;
            hipLaunchKernelGGL(hog, dim3(256), dim3(256), 0, sa, hogbuf, n, hog_reps);
            HC(hipGetLastError());
            HC(hipStreamSynchronize(sa));
            while(!stop.load() && c->cycle_done.load() != k)
                std::this_thread::yield();
        }
    });

    long corrupt = 0, ok = 0;
    int  rc = 0;
    for(long k = 1; k <= cycles; ++k)
    {
        hipLaunchKernelGGL(stamp, dim3(64), dim3(64), 0, sb, V, OLD, n);  // V = OLD
        HC(hipStreamSynchronize(sb));
        c->old_ready.store(k);

        if(!wait_eq(c->snapshot_done, k))  // snapshot captured OLD
        {
            fprintf(stderr, "[repro] FAIL: timed out waiting for snapshot_done (replay firing?)\n");
            rc = 2;
            break;
        }
        hipLaunchKernelGGL(stamp, dim3(64), dim3(64), 0, sb, V, NEW, n);  // V = NEW (races window)
        HC(hipStreamSynchronize(sb));

        if(!wait_eq(c->window_done, k))  // all restores done
        {
            fprintf(stderr, "[repro] FAIL: timed out waiting for window_done (replay firing?)\n");
            rc = 2;
            break;
        }
        HC(hipDeviceSynchronize());  // let all device work settle
        float hv = 0.0f;
        HC(hipMemcpy(&hv, V, sizeof(float), hipMemcpyDeviceToHost));

        if(hv == OLD)
            ++corrupt;
        else
            ++ok;
        if(k <= 3) fprintf(stderr, "[dbg] cycle %ld read=%.0f (expect NEW=%.0f)\n", k, hv, NEW);
        c->cycle_done.store(k);
    }

    stop.store(true);
    hog_thread.join();

    if(rc != 0) return rc;

    printf("[repro] cycles=%d corrupt=%ld ok=%ld\n", cycles, corrupt, ok);
    if(corrupt == 0)
        printf("[repro] PASS: no corruption in any cycle\n");
    else
        printf("[repro] FAIL: P0-1 reproduced (%ld/%d cycles reverted)\n", corrupt, cycles);
    fflush(stdout);

    return corrupt > 0 ? 1 : 0;
}
