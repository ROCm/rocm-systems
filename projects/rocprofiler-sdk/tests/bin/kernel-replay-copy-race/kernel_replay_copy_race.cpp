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

// kernel-replay-copy-race: does a kernel-replay window corrupt an async copy that is in flight
// when it snapshots?
//
//   kernel-replay-copy-race [iterations] [MiB] [threads|streams]
//
// Thread "copier": for each iteration, fill a host buffer with a per-iteration pattern, copy it to
// device buffer X with hipMemcpyAsync on its own stream, synchronize, copy X back to a second host
// buffer and compare. Thread "launcher": launches a small kernel on its own stream and
// synchronizes, in a loop, until the copier is done. Under kernel replay each launcher dispatch
// opens a replay window that snapshots and restores every tracked allocation of the agent, X
// included. If the window does not fence the copier's in-flight copy, a restore writes back a
// snapshot taken mid-copy and X ends up with stale bytes. Mode "streams" does the same from ONE
// thread: the copy is issued on stream B without waiting, a kernel is launched and synchronized on
// stream A, then B is synchronized and X verified (a prefetch on a side stream). Prints one JSON
// line; exit 1 if any iteration differed.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK(x)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        hipError_t e_ = (x);                                                                       \
        if(e_ != hipSuccess)                                                                       \
        {                                                                                          \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));      \
            std::exit(2);                                                                          \
        }                                                                                          \
    } while(0)

__global__ void
bump(int* k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) k[i] += 1;
}

int
main(int argc, char** argv)
{
    const int    iters      = argc > 1 ? std::atoi(argv[1]) : 40;
    const size_t mib        = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 256;
    const size_t bytes      = mib << 20;
    const size_t words      = bytes / sizeof(uint32_t);
    const bool   one_thread = argc > 3 && std::strcmp(argv[3], "streams") == 0;

    uint32_t* x = nullptr;
    int*      k = nullptr;
    CHECK(hipMalloc(&x, bytes));
    CHECK(hipMalloc(&k, 1 << 20));
    CHECK(hipMemset(x, 0, bytes));
    CHECK(hipMemset(k, 0, 1 << 20));

    std::atomic<bool> done{false};
    std::atomic<long> launches{0};
    int               bad_iters = 0;
    size_t            bad_words = 0;

    if(one_thread)
    {
        hipStream_t sa, sb;
        CHECK(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking));
        CHECK(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking));
        std::vector<uint32_t> src(words), dst(words);
        uint32_t*             pinned = nullptr;
        CHECK(hipHostMalloc(reinterpret_cast<void**>(&pinned), bytes, hipHostMallocDefault));
        long nlaunch = 0;
        for(int it = 0; it < iters; ++it)
        {
            const uint32_t pattern = 0x9e3779b9u * static_cast<uint32_t>(it + 1);
            for(size_t i = 0; i < words; ++i)
                pinned[i] = pattern ^ static_cast<uint32_t>(i);
            CHECK(hipMemcpyAsync(x, pinned, bytes, hipMemcpyHostToDevice, sb));
            for(int j = 0; j < 4; ++j)
            {
                bump<<<256, 256, 0, sa>>>(k, 1 << 16);
                CHECK(hipStreamSynchronize(sa));
                ++nlaunch;
            }
            CHECK(hipStreamSynchronize(sb));
            CHECK(hipMemcpy(dst.data(), x, bytes, hipMemcpyDeviceToHost));
            size_t diff = 0;
            for(size_t i = 0; i < words; ++i)
                diff += (dst[i] != (pattern ^ static_cast<uint32_t>(i)));
            if(diff)
            {
                ++bad_iters;
                bad_words += diff;
            }
        }
        launches = nlaunch;
        CHECK(hipHostFree(pinned));
        CHECK(hipStreamDestroy(sa));
        CHECK(hipStreamDestroy(sb));
    }
    else
    {
        std::thread launcher([&] {
            hipStream_t s;
            CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
            while(!done.load())
            {
                bump<<<256, 256, 0, s>>>(k, 1 << 16);
                CHECK(hipStreamSynchronize(s));
                launches++;
            }
            CHECK(hipStreamDestroy(s));
        });

        std::thread copier([&] {
            hipStream_t s;
            CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
            std::vector<uint32_t> src(words), dst(words);
            for(int it = 0; it < iters; ++it)
            {
                const uint32_t pattern = 0x9e3779b9u * static_cast<uint32_t>(it + 1);
                for(size_t i = 0; i < words; ++i)
                    src[i] = pattern ^ static_cast<uint32_t>(i);
                CHECK(hipMemcpyAsync(x, src.data(), bytes, hipMemcpyHostToDevice, s));
                CHECK(hipStreamSynchronize(s));
                CHECK(hipMemcpyAsync(dst.data(), x, bytes, hipMemcpyDeviceToHost, s));
                CHECK(hipStreamSynchronize(s));
                size_t diff = 0;
                for(size_t i = 0; i < words; ++i)
                    diff += (dst[i] != src[i]);
                if(diff)
                {
                    ++bad_iters;
                    bad_words += diff;
                }
            }
            done = true;
            CHECK(hipStreamDestroy(s));
        });

        copier.join();
        launcher.join();
    }
    printf("{\"app\":\"kernel-replay-copy-race\",\"mode\":\"%s\",\"iters\":%d,\"mib\":%zu,"
           "\"launches\":%ld,\"bad_iters\":%d,"
           "\"bad_words\":%zu,\"pass\":%s}\n",
           one_thread ? "streams" : "threads",
           iters,
           mib,
           launches.load(),
           bad_iters,
           bad_words,
           bad_iters ? "false" : "true");
    CHECK(hipFree(x));
    CHECK(hipFree(k));
    return bad_iters ? 1 : 0;
}
