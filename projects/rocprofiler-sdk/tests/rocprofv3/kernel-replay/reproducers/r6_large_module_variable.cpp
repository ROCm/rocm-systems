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
// Reproducer R6: a module-scope device variable above the snapshot size cap is not restored.
//
// collect_module_variable() skips any HSA_SYMBOL_KIND_VARIABLE whose size exceeds 1 GiB and
// returns HSA_STATUS_SUCCESS, so the symbol is silently absent from the snapshot. A kernel
// that accumulates into such a variable therefore sees pass N-1's leftovers on pass N, and
// the shared-counter checks cannot notice because the wave count is unchanged.
//
// The small variable is the control: it must be restored, so its post-replay value must
// equal one pass worth of accumulation. The large variable is the subject: if it holds more
// than one pass worth, it was never reverted.
//
// Build the two configurations separately, since the cap is a size threshold:
//   hipcc -O2 -std=c++17 -DKR_BIG_ELEMS=$((300*1024*1024/4)) r6_large_module_variable.cpp -o r6_small
//   hipcc -O2 -std=c++17 -DKR_BIG_ELEMS=$((1300*1024*1024/4)) r6_large_module_variable.cpp -o r6_big
//   LD_PRELOAD=./librepro_client.so KR_REPRO_PASSES=4 ./r6_small   # expect restored
//   LD_PRELOAD=./librepro_client.so KR_REPRO_PASSES=4 ./r6_big     # expect NOT restored
//
// KR_BIG_ELEMS is given in floats: 300 MiB is under the cap, 1300 MiB is over it. The large
// build needs a card with enough free VRAM.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef KR_BIG_ELEMS
#    define KR_BIG_ELEMS (300u * 1024u * 1024u / 4u)
#endif

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

// Subject: sized by KR_BIG_ELEMS so the same source covers under- and over-cap.
__device__ float g_big[KR_BIG_ELEMS];
// Control: comfortably under any cap, so it must be snapshotted and restored.
__device__ float g_small[1024];

__global__ void
accumulate(size_t big_elems)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < 1024) g_small[i] += 1.0f;
    if(i < big_elems) g_big[i] += 1.0f;
}

int
main()
{
    HC(hipSetDevice(0));
    const size_t big_elems = KR_BIG_ELEMS;
    printf("[r6] subject g_big = %zu floats = %.0f MiB\n", big_elems,
           double(big_elems * sizeof(float)) / (1024.0 * 1024.0));

    // Both variables start at zero, which is the state a correct restore reverts to.
    std::vector<float> zeros_small(1024, 0.0f);
    HC(hipMemcpyToSymbol(HIP_SYMBOL(g_small), zeros_small.data(), sizeof(float) * 1024));
    {
        std::vector<float> zeros(1u << 20, 0.0f);
        for(size_t off = 0; off < big_elems; off += (1u << 20))
        {
            size_t n = (big_elems - off < (1u << 20)) ? (big_elems - off) : (1u << 20);
            HC(hipMemcpyToSymbol(HIP_SYMBOL(g_big), zeros.data(), n * sizeof(float),
                                 off * sizeof(float)));
        }
    }

    // One launch. The tool library replays it, so the kernel body runs several times; each
    // pass must start from the restored zeros, leaving exactly 1.0 behind.
    const size_t threads = 1024;
    const size_t grid    = (big_elems + threads - 1) / threads;
    hipLaunchKernelGGL(accumulate, dim3(grid), dim3(threads), 0, 0, big_elems);
    HC(hipDeviceSynchronize());

    float small0 = -1.0f, big0 = -1.0f;
    HC(hipMemcpyFromSymbol(&small0, HIP_SYMBOL(g_small), sizeof(float)));
    HC(hipMemcpyFromSymbol(&big0, HIP_SYMBOL(g_big), sizeof(float)));
    printf("[r6] after replay: g_small[0]=%.1f  g_big[0]=%.1f  (1.0 means restored)\n", small0,
           big0);

    if(small0 > 1.5f)
    {
        printf("[r6] control also accumulated -- module variables are not being restored at\n"
               "[r6] all, so this run says nothing about the size cap.\n");
        return 3;
    }
    if(big0 > 1.5f)
    {
        printf("[r6] REPRODUCED: the control was reverted but the large variable accumulated\n"
               "[r6] to %.1f, so it was skipped by the snapshot.\n", big0);
        return 1;
    }
    printf("[r6] both restored: no size cap observed at this size.\n");
    return 0;
}
