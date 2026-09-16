/*
Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

template <typename T>
void
check(T result, char const* const func, const char* const file, int const line)
{
    if(result)
    {
        fprintf(stderr,
                "Hip error at %s:%d code=%d(%s) \"%s\" \n",
                file,
                line,
                static_cast<unsigned int>(result),
                hipGetErrorName(result),
                func);
        exit(EXIT_FAILURE);
    }
}
#define checkHipErrors(val) check((val), #val, __FILE__, __LINE__)

__global__ void
kernel_a(float* x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    x[idx] += 1.0f;
}

__global__ void
kernel_b(float* x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    x[idx] *= 2.0f;
}

// Graph: a0 -> {b1, a2} -> a3 -> memcpy -> b4 (3x kernel_a, 2x kernel_b, 1x memcpy).
int
main(int argc, char** argv)
{
    int iterations = (argc > 1) ? std::atoi(argv[1]) : 50;

    constexpr size_t N     = 1024;
    float*           d_buf = nullptr;
    float*           h_buf = nullptr;
    checkHipErrors(hipMalloc(&d_buf, N * sizeof(float)));
    checkHipErrors(hipHostMalloc(&h_buf, N * sizeof(float)));
    for(size_t i = 0; i < N; ++i)
        h_buf[i] = 1.0f;

    // Out-of-graph H2D memcpy ensures memory_copy CSV exists even if in-graph copy goes via blit.
    checkHipErrors(hipMemcpy(d_buf, h_buf, N * sizeof(float), hipMemcpyHostToDevice));

    hipGraph_t graph;
    checkHipErrors(hipGraphCreate(&graph, 0));

    auto make_kernel_node =
        [&](hipGraphNode_t* node, hipGraphNode_t* deps, size_t ndeps, void (*fn)(float*)) {
            hipKernelNodeParams kp{};
            void*               args[] = {&d_buf};
            kp.func                    = reinterpret_cast<void*>(fn);
            kp.gridDim                 = dim3(8, 1, 1);
            kp.blockDim                = dim3(128, 1, 1);
            kp.kernelParams            = args;
            kp.extra                   = nullptr;
            kp.sharedMemBytes          = 0;
            checkHipErrors(hipGraphAddKernelNode(node, graph, deps, ndeps, &kp));
        };

    hipGraphNode_t a0, b1, a2, a3, m, b4;
    make_kernel_node(&a0, nullptr, 0, kernel_a);
    make_kernel_node(&b1, &a0, 1, kernel_b);
    make_kernel_node(&a2, &a0, 1, kernel_a);
    hipGraphNode_t deps_for_a3[] = {b1, a2};
    make_kernel_node(&a3, deps_for_a3, 2, kernel_a);
    checkHipErrors(hipGraphAddMemcpyNode1D(
        &m, graph, &a3, 1, d_buf, h_buf, N * sizeof(float), hipMemcpyHostToDevice));
    make_kernel_node(&b4, &m, 1, kernel_b);

    // Two executable instantiations to verify per-exec distinctness of graph_exec_id.
    hipGraphExec_t exec_a, exec_b;
    checkHipErrors(hipGraphInstantiate(&exec_a, graph, nullptr, nullptr, 0));
    checkHipErrors(hipGraphInstantiate(&exec_b, graph, nullptr, nullptr, 0));

    hipStream_t stream;
    checkHipErrors(hipStreamCreate(&stream));

    for(int i = 0; i < iterations; ++i)
    {
        checkHipErrors(hipGraphLaunch(exec_a, stream));
        checkHipErrors(hipGraphLaunch(exec_b, stream));
    }
    checkHipErrors(hipStreamSynchronize(stream));

    // Failure-mode: launch on destroyed exec must fail (no record); next launch must still
    // attribute.
    checkHipErrors(hipGraphExecDestroy(exec_a));
    hipError_t failed = hipGraphLaunch(exec_a, stream);
    (void) failed;
    checkHipErrors(hipGraphLaunch(exec_b, stream));
    checkHipErrors(hipStreamSynchronize(stream));

    checkHipErrors(hipGraphExecDestroy(exec_b));
    checkHipErrors(hipGraphDestroy(graph));
    checkHipErrors(hipStreamDestroy(stream));
    checkHipErrors(hipFree(d_buf));
    checkHipErrors(hipHostFree(h_buf));
    return 0;
}
