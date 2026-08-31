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

// HIP-graph fork/join workload. Each replay runs a chain of "diamond" layers: one root node
// fans out into `width` independent branch kernels, then those branches join back into one
// node, which the next layer forks from. The graph is launched on a single stream; HIP spreads
// a layer's independent branches across hardware queues, so they run at the same time and the
// join that follows couples their completions. Stacking layers keeps many mutually dependent
// completion signals in flight at once, which is what exercises the queue interposition
// completion path.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
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
// One block per kernel node. Buffers are sized to match, so every element is touched;
// kernel cost is set by the spin count, not by the buffer size.
constexpr int block_size = 256;

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
    params.blockDim       = dim3(block_size);
    params.sharedMemBytes = 0;
    params.kernelParams   = args;
    params.extra          = nullptr;

    HIP_CHECK(hipGraphAddKernelNode(&node, graph, deps.data(), deps.size(), &params));
    return node;
}

// Every dimension indexes something, so zero is rejected along with non-numeric input.
uint64_t
parse_dimension(const char* text, const char* name)
{
    try
    {
        const auto value = std::stoul(text);
        if(value > 0) return value;
    } catch(const std::exception&)
    {}

    std::cerr << "[hip-graph-forkjoin] " << name << " must be a positive integer, got '" << text
              << "'" << std::endl;
    exit(EXIT_FAILURE);
}
}  // namespace

int
main(int argc, char** argv)
{
    auto width   = uint64_t{8};
    auto depth   = uint64_t{8};
    auto replays = uint64_t{50};

    if(argc > 1) width = parse_dimension(argv[1], "width");
    if(argc > 2) depth = parse_dimension(argv[2], "depth");
    if(argc > 3) replays = parse_dimension(argv[3], "replays");

    std::cout << "[hip-graph-forkjoin] width=" << width << " depth=" << depth
              << " replays=" << replays << std::endl;

    auto n     = block_size;
    auto iters = 2000;

    // Zeroed before the first kernel reads them: uninitialized device memory can hold
    // denormals, and the spin loop takes longer on those, which moves kernel duration in a
    // test whose failure signal is a timeout.
    auto bufs = std::vector<float*>(width, nullptr);
    for(auto& buf : bufs)
    {
        HIP_CHECK(hipMalloc(&buf, n * sizeof(float)));
        HIP_CHECK(hipMemset(buf, 0, n * sizeof(float)));
    }

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
