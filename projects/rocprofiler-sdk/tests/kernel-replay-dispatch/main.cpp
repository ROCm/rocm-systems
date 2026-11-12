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

// Application for the replay-dispatch test: launches `bump` once. The tool replays it and, on one
// pass, submits `substitute` instead. `substitute` is never launched here; it only has to be in
// this program's code object so the tool can load its own copy of it.
//
// Environment:
//   KR_DISPATCH_CLUSTER=1  launch `bump` as a cluster dispatch (2 workgroups per cluster)

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Distinctive workgroup size so the tool replays only this dispatch.
constexpr int kBlock  = 67;
constexpr int kBlocks = 64;
constexpr int kN      = kBlock * kBlocks;

extern "C" __global__ void
bump(int* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) out[i] += 1;
}

// Runs with bump's kernel arguments on a grid the tool chooses. It reads only its explicit
// arguments and hardware ids, and counts its workgroups in out[n].
extern "C" __global__ void
substitute(int* out, int n)
{
    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int i    = static_cast<int>(__builtin_amdgcn_workgroup_id_x()) * kBlock + lane;
    if(i < n) out[i] = 1000 + i;
    if(lane == 0) atomicAdd(&out[n], 1);
}

int
main()
{
    const char* cluster_env = std::getenv("KR_DISPATCH_CLUSTER");
    const bool  cluster     = cluster_env && std::strcmp(cluster_env, "1") == 0;

    int* d = nullptr;
    HIP_CHECK(hipMalloc(&d, (kN + 1) * sizeof(int)));
    HIP_CHECK(hipMemset(d, 0, (kN + 1) * sizeof(int)));

    if(cluster)
    {
        auto attr               = hipLaunchAttribute{};
        attr.id                 = hipLaunchAttributeClusterDimension;
        attr.val.clusterDim.x   = 2;
        attr.val.clusterDim.y   = 1;
        attr.val.clusterDim.z   = 1;
        auto config             = hipLaunchConfig_t{};
        config.gridDim          = dim3(kBlocks);
        config.blockDim         = dim3(kBlock);
        config.dynamicSmemBytes = 0;
        config.stream           = nullptr;
        config.attrs            = &attr;
        config.numAttrs         = 1;
        int   n                 = kN;
        void* args[]            = {&d, &n};
        if(hipLaunchKernelExC(&config, reinterpret_cast<const void*>(&bump), args) != hipSuccess)
        {
            fprintf(stderr, "cluster dispatch unavailable\n");
            return EXIT_SUCCESS;
        }
    }
    else
    {
        bump<<<kBlocks, kBlock>>>(d, kN);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    auto h = std::vector<int>(kN + 1, -1);
    HIP_CHECK(hipMemcpy(h.data(), d, h.size() * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d));

    // The final replay pass submits the application's dispatch, so the output is one bump.
    for(int i = 0; i < kN; ++i)
    {
        if(h[i] != 1)
        {
            fprintf(stderr, "[app] FAIL: out[%d]=%d (expected 1)\n", i, h[i]);
            return EXIT_FAILURE;
        }
    }
    if(h[kN] != 0)
    {
        fprintf(stderr, "[app] FAIL: out[n]=%d (expected 0; a substitute pass leaked)\n", h[kN]);
        return EXIT_FAILURE;
    }
    printf("[app] output matches one original dispatch\n");
    return EXIT_SUCCESS;
}
