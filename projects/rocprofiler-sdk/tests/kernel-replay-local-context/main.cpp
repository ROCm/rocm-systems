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

#include <hip/hip_runtime.h>

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

// Distinctive workgroup size so the tool can opt only this dispatch into replay.
constexpr int kBlock = 67;

__global__ void
bump(int* x)
{
    if(threadIdx.x == 0) atomicAdd(x, 1);
}

int
main()
{
    int* d = nullptr;
    HIP_CHECK(hipMalloc(&d, sizeof(int)));
    HIP_CHECK(hipMemset(d, 0, sizeof(int)));
    bump<<<1, kBlock>>>(d);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    int h = 0;
    HIP_CHECK(hipMemcpy(&h, d, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d));

    // Replay restore must make this look like a single application launch.
    printf("[app] bump=%d\n", h);
    if(h != 1)
    {
        fprintf(
            stderr, "[app] FAIL: bump=%d (expected 1; restore did not revert between passes)\n", h);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
