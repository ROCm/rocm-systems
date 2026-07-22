// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_ASSERT(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        hipError_t gpuErr = call;                                                                  \
        if(hipSuccess != gpuErr)                                                                   \
        {                                                                                          \
            printf(                                                                                \
                "GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr));   \
            exit(1);                                                                               \
        }                                                                                          \
    } while(0)

__global__ void
dummy_kernel(float* out, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) out[tid] = static_cast<float>(tid);
}

static std::vector<void*> g_stream_ptrs;

static void
use_stream(hipStream_t stream, float* d_buf, size_t n)
{
    g_stream_ptrs.push_back(static_cast<void*>(stream));
    dummy_kernel<<<4, 256, 0, stream>>>(d_buf, n);
    HIP_ASSERT(hipStreamSynchronize(stream));
}

// Total expected kernel dispatches:
//   Phase 1 (sequential):  4
//   Phase 2 (sequential):  4
//   Phase 3 (bulk round1): 4
//   Phase 3 (bulk round2): 4
//   Phase 4 (alternating): 6 (3 pairs x 2)
//   ---------------------------------
//   Total:                 22
int
main(int argc, char** argv)
{
    constexpr size_t n     = 1024;
    constexpr size_t bytes = n * sizeof(float);

    float* d_buf = nullptr;
    HIP_ASSERT(hipMalloc(&d_buf, bytes));

    // Phase 1: sequential create/use/destroy — tests immediate LIFO reuse
    for(int i = 0; i < 4; ++i)
    {
        hipStream_t s = nullptr;
        HIP_ASSERT(hipStreamCreate(&s));
        use_stream(s, d_buf, n);
        HIP_ASSERT(hipStreamDestroy(s));
    }

    // Phase 2: repeat — recycled pointers must get fresh IDs
    for(int i = 0; i < 4; ++i)
    {
        hipStream_t s = nullptr;
        HIP_ASSERT(hipStreamCreate(&s));
        use_stream(s, d_buf, n);
        HIP_ASSERT(hipStreamDestroy(s));
    }

    // Phase 3: bulk create, use all, bulk destroy, then bulk re-create
    {
        constexpr int bulk = 4;
        hipStream_t   round1[bulk];
        hipStream_t   round2[bulk];

        for(int i = 0; i < bulk; ++i)
            HIP_ASSERT(hipStreamCreate(&round1[i]));
        for(int i = 0; i < bulk; ++i)
            use_stream(round1[i], d_buf, n);
        for(int i = 0; i < bulk; ++i)
            HIP_ASSERT(hipStreamDestroy(round1[i]));

        // Round 2 — likely gets same pointers back
        for(int i = 0; i < bulk; ++i)
            HIP_ASSERT(hipStreamCreate(&round2[i]));
        for(int i = 0; i < bulk; ++i)
            use_stream(round2[i], d_buf, n);
        for(int i = 0; i < bulk; ++i)
            HIP_ASSERT(hipStreamDestroy(round2[i]));
    }

    // Phase 4: paired create/destroy with alternating destroy order
    for(int i = 0; i < 3; ++i)
    {
        hipStream_t a = nullptr;
        hipStream_t b = nullptr;
        HIP_ASSERT(hipStreamCreate(&a));
        HIP_ASSERT(hipStreamCreate(&b));

        use_stream(a, d_buf, n);
        use_stream(b, d_buf, n);

        if(i % 2 == 0)
        {
            HIP_ASSERT(hipStreamDestroy(a));
            HIP_ASSERT(hipStreamDestroy(b));
        }
        else
        {
            HIP_ASSERT(hipStreamDestroy(b));
            HIP_ASSERT(hipStreamDestroy(a));
        }
    }

    HIP_ASSERT(hipFree(d_buf));

    // Write raw pointer values to file so the validation script can detect
    // whether pointer recycling actually occurred.
    const char* ptr_file = argc > 1 ? argv[1] : "stream_pointers.txt";

    FILE* fp = fopen(ptr_file, "w");
    if(!fp)
    {
        fprintf(stderr, "Failed to open '%s' for writing: %s\n", ptr_file, strerror(errno));
        return 1;
    }

    for(auto* ptr : g_stream_ptrs)
        fprintf(fp, "%p\n", ptr);
    fclose(fp);

    return 0;
}
