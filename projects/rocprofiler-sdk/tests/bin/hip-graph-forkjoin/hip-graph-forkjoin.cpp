// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Multi-stream HIP-graph fork/join workload. Each replay executes a chain of
// "diamond" layers: one root node fans out into WIDTH independent branch kernels
// that each depend on the previous join, then those branches join back into one
// node. Independent branches within a layer become concurrent, barrier-coupled
// segments across streams; a depth greater than four keeps the graph from being
// collapsed onto a single stream. The structure produces many completion signals
// that are in flight and mutually dependent at once, exercising the queue
// interposition completion path under concurrency.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define HIP_CHECK(CALL)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d HIP error (%d): %s\n",                                                  \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    static_cast<int>(error_),                                                      \
                    hipGetErrorString(error_));                                                    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)

namespace
{
__global__ void
spin_kernel(float* buf, int n, int iters)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        float v = buf[idx];
        for(int i = 0; i < iters; ++i)
            v = (v * 1.0000001F) + 0.0000001F;
        buf[idx] = v;
    }
}

hipGraphNode_t
add_kernel_node(hipGraph_t                         graph,
                const std::vector<hipGraphNode_t>& deps,
                float*                             buf,
                int*                               n,
                int*                               iters)
{
    auto           params = hipKernelNodeParams{};
    void*          args[] = {&buf, n, iters};
    hipGraphNode_t node   = nullptr;

    params.func           = reinterpret_cast<void*>(spin_kernel);
    params.gridDim        = dim3(1);
    params.blockDim       = dim3(256);
    params.sharedMemBytes = 0;
    params.kernelParams   = args;
    params.extra          = nullptr;

    HIP_CHECK(hipGraphAddKernelNode(&node, graph, deps.data(), deps.size(), &params));
    return node;
}
}  // namespace

int
main(int argc, char** argv)
{
    auto width   = uint64_t{8};
    auto depth   = uint64_t{8};
    auto replays = uint64_t{50};

    if(argc > 1) width = std::stoul(argv[1]);
    if(argc > 2) depth = std::stoul(argv[2]);
    if(argc > 3) replays = std::stoul(argv[3]);

    std::cout << "[hip-graph-forkjoin] width=" << width << " depth=" << depth
              << " replays=" << replays << std::endl;

    auto n     = 1 << 16;
    auto iters = 2000;

    auto bufs = std::vector<float*>(width, nullptr);
    for(auto& buf : bufs)
        HIP_CHECK(hipMalloc(&buf, n * sizeof(float)));

    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Root node, then `depth` diamond layers hung off it.
    auto prev = std::vector<hipGraphNode_t>{add_kernel_node(graph, {}, bufs[0], &n, &iters)};

    for(uint64_t d = 0; d < depth; ++d)
    {
        auto branches = std::vector<hipGraphNode_t>{};
        branches.reserve(width);
        for(uint64_t w = 0; w < width; ++w)
            branches.emplace_back(add_kernel_node(graph, prev, bufs[w], &n, &iters));

        auto join = add_kernel_node(graph, branches, bufs[0], &n, &iters);
        prev      = std::vector<hipGraphNode_t>{join};
    }

    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));

    for(uint64_t r = 0; r < replays; ++r)
    {
        HIP_CHECK(hipGraphLaunch(exec, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    std::cout << "[hip-graph-forkjoin] complete" << std::endl;

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
    for(auto* buf : bufs)
        HIP_CHECK(hipFree(buf));

    return 0;
}
