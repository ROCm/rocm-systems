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

// Workload for rocprofv3 --att-remap: a K loop over LDS tiles with two workgroup barriers per
// step, the shape the remapper's merge compares across SIMDs. The host checks every output, so a
// replay loop that left the remapped kernel's results (or none) behind fails the run.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (call);                                                                  \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            std::fprintf(stderr, "%s failed: %s\n", #call, hipGetErrorString(_err));               \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while(0)

namespace
{
constexpr int block_size = 256;

// The remapper does not rewrite gfx1250 kernels compiled for cluster launches, whose workgroup id
// reads the cluster rank: a no_cluster kernel reads the workgroup id alone.
__global__ __attribute__((no_cluster))
__launch_bounds__(block_size) void tiled_k_loop(const float* __restrict__ in,
                                                float* __restrict__ out,
                                                int ktiles)
{
    __shared__ float tile[block_size];
    const int        row = static_cast<int>(blockIdx.x);
    const int        t   = static_cast<int>(threadIdx.x);
    float            acc = 0.0f;
    for(int k = 0; k < ktiles; ++k)
    {
        tile[t] = in[(static_cast<size_t>(row) * ktiles + k) * block_size + t];
        __syncthreads();
        acc += tile[t] + 0.5f * tile[(t + 1) % block_size];
        __syncthreads();
    }
    out[static_cast<size_t>(row) * block_size + t] = acc;
}
}  // namespace

int
main(int argc, char** argv)
{
    const int workgroups = (argc > 1) ? std::atoi(argv[1]) : 2048;
    const int ktiles     = (argc > 2) ? std::atoi(argv[2]) : 16;
    if(workgroups <= 0 || ktiles <= 0)
    {
        std::fprintf(stderr, "usage: %s [workgroups] [ktiles]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const auto n_in  = static_cast<size_t>(workgroups) * ktiles * block_size;
    const auto n_out = static_cast<size_t>(workgroups) * block_size;
    auto       in    = std::vector<float>(n_in);
    for(size_t i = 0; i < n_in; ++i)
        in[i] = static_cast<float>((i * 7) % 13) * 0.25f;

    float* d_in  = nullptr;
    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, n_in * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, n_out * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, in.data(), n_in * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, n_out * sizeof(float)));

    hipLaunchKernelGGL(tiled_k_loop, dim3(workgroups), dim3(block_size), 0, 0, d_in, d_out, ktiles);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    auto out = std::vector<float>(n_out);
    HIP_CHECK(hipMemcpy(out.data(), d_out, n_out * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    size_t bad = 0;
    for(int row = 0; row < workgroups; ++row)
    {
        for(int t = 0; t < block_size; ++t)
        {
            float want = 0.0f;
            for(int k = 0; k < ktiles; ++k)
            {
                const auto* tile = &in[(static_cast<size_t>(row) * ktiles + k) * block_size];
                want += tile[t] + 0.5f * tile[(t + 1) % block_size];
            }
            const float got = out[static_cast<size_t>(row) * block_size + t];
            if(std::fabs(got - want) > 1e-3f * std::fabs(want) + 1e-3f) ++bad;
        }
    }
    if(bad != 0)
    {
        std::fprintf(stderr, "att-remap: %zu of %zu outputs are wrong\n", bad, n_out);
        return EXIT_FAILURE;
    }
    std::printf("att-remap: %d workgroups x %d tiles verified\n", workgroups, ktiles);
    return EXIT_SUCCESS;
}
