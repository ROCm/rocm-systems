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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// Reproducer R3: hipFree during a replay window, so restore() writes into freed memory.
//
// memory_snapshot::snap() takes the tracker's inventory lock only while building its list
// of blocks; the device->host and host->device copies then run on raw pointers outside
// that lock. The per-agent reader/writer lock serialises *dispatches*, not allocations, so
// nothing stops another thread from calling hipFree between snapshot and restore. restore()
// re-issues a copy to every captured address with no revalidation.
//
// This program:
//   thread A  launches a replayed kernel (the tool library replays it several passes)
//   thread B  waits until the replay window has certainly opened, then frees a buffer that
//             was tracked at snapshot time
//
// Expected on a correct implementation: either the free is serialised against the window,
// or restore() revalidates and skips the dead block. Either way the process exits 0.
// Observed failure signature: a HIP/HSA fault, an abort inside restore, or silent
// corruption reported by the checksum below. Run under ASAN or with
// AMD_SERIALIZE_KERNEL=3 to sharpen it.
//
//   hipcc -O2 -std=c++17 r3_free_during_replay.cpp -o r3
//   LD_PRELOAD=./librepro_client.so KR_REPRO_PASSES=8 ./r3

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define HC(...)                                                                          \
    do                                                                                    \
    {                                                                                     \
        hipError_t _e = (__VA_ARGS__);                                                     \
        if(_e != hipSuccess)                                                               \
        {                                                                                  \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #__VA_ARGS__,           \
                    hipGetErrorString(_e));                                                 \
            exit(2);                                                                        \
        }                                                                                  \
    } while(0)

namespace
{
constexpr size_t N        = 1u << 22;  // 4 Mi elements, 16 MiB
constexpr int    BLOCK    = 256;
std::atomic<int> g_launched{0};
}  // namespace

__global__ void
grind(float* out, const float* in, size_t n, int iters)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= n) return;
    float v = in[i];
    for(int k = 0; k < iters; ++k)
        v = v * 1.000001f + 1.0f;
    out[i] = v;
}

int
main(int argc, char** argv)
{
    const int iters = (argc > 1) ? atoi(argv[1]) : 4000;
    HC(hipSetDevice(0));

    // Tracked by the replay memory tracker because both come from hipMalloc.
    float* in       = nullptr;
    float* out      = nullptr;
    float* doomed   = nullptr;  // freed mid-window by thread B
    HC(hipMalloc(&in, N * sizeof(float)));
    HC(hipMalloc(&out, N * sizeof(float)));
    HC(hipMalloc(&doomed, N * sizeof(float)));

    std::vector<float> host(N, 1.0f);
    HC(hipMemcpy(in, host.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HC(hipMemcpy(doomed, host.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Thread B: free the tracked buffer shortly after the window opens. The kernel is long
    // enough that the first pass is still running.
    std::thread freer([&] {
        while(g_launched.load(std::memory_order_acquire) == 0)
            std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        fprintf(stderr, "[r3] freeing a tracked buffer mid-replay: %p\n", (void*) doomed);
        hipError_t e = hipFree(doomed);
        fprintf(stderr, "[r3] hipFree -> %s\n", hipGetErrorString(e));
    });

    fprintf(stderr, "[r3] launching the replayed kernel\n");
    const size_t grid = (N + BLOCK - 1) / BLOCK;
    g_launched.store(1, std::memory_order_release);
    hipLaunchKernelGGL(grind, dim3(grid), dim3(BLOCK), 0, 0, out, in, N, iters);
    HC(hipDeviceSynchronize());
    freer.join();

    // The kernel is deterministic, so every replay pass must produce this value.
    std::vector<float> result(N);
    HC(hipMemcpy(result.data(), out, N * sizeof(float), hipMemcpyDeviceToHost));
    float expect = 1.0f;
    for(int k = 0; k < iters; ++k)
        expect = expect * 1.000001f + 1.0f;

    size_t wrong = 0;
    for(size_t i = 0; i < N; ++i)
        if(result[i] != result[i] || __builtin_fabsf(result[i] - expect) > 1e-3f * expect) ++wrong;

    HC(hipFree(in));
    HC(hipFree(out));

    if(wrong != 0)
    {
        fprintf(stderr, "[r3] REPRODUCED (silent corruption): %zu/%zu elements wrong\n", wrong, N);
        return 1;
    }
    fprintf(stderr, "[r3] survived: output consistent. Re-run under ASAN, or raise the pass\n"
                    "[r3] count and shorten the sleep, before concluding the race is absent.\n");
    return 0;
}
