/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

// Duplicate kernels expose stale capture plans: after the first node changes or
// is disabled, the second must never inherit the first node's capture correlation.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call)                                                                                \
    do                                                                                             \
    {                                                                                              \
        auto e = (call);                                                                           \
        if(e != hipSuccess)                                                                        \
        {                                                                                          \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, hipGetErrorString(e));    \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while(0)

__global__ void
capture_same(int* x)
{
    *x += 1;
}
__global__ void
capture_changed(int* x)
{
    *x += 2;
}

void
run_case(int mode)
{
    hipStream_t    stream{};
    hipGraph_t     graph{};
    hipGraphExec_t exec{};
    int*           data = nullptr;
    CHECK(hipStreamCreate(&stream));
    CHECK(hipMalloc(&data, 2 * sizeof(int)));
    CHECK(hipMemset(data, 0, 2 * sizeof(int)));
    CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    hipLaunchKernelGGL(capture_same, dim3(1), dim3(1), 0, stream, data);
    CHECK(hipGetLastError());
    hipLaunchKernelGGL(capture_same, dim3(1), dim3(1), 0, stream, data + 1);
    CHECK(hipGetLastError());
    CHECK(hipStreamEndCapture(stream, &graph));
    CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    auto replay = [&]() {
        for(int i = 0; i < 3; ++i)
            CHECK(hipGraphLaunch(exec, stream));
        CHECK(hipStreamSynchronize(stream));
    };
    replay();
    size_t n = 0;
    CHECK(hipGraphGetRootNodes(graph, nullptr, &n));
    if(n != 1) std::exit(2);
    hipGraphNode_t first{};
    CHECK(hipGraphGetRootNodes(graph, &first, &n));
    // A failed mutation must not invalidate a still-correct plan.
    if(hipGraphExecKernelNodeSetParams(exec, first, nullptr) == hipSuccess) std::exit(3);
    (void) hipGetLastError();  // clear the expected failure before the next capture
    replay();
    hipKernelNodeParams params{};
    CHECK(hipGraphKernelNodeGetParams(first, &params));
    params.func = reinterpret_cast<void*>(capture_changed);
    if(mode == 0)
        CHECK(hipGraphExecKernelNodeSetParams(exec, first, &params));
    else if(mode == 1)
        CHECK(hipGraphNodeSetEnabled(exec, first, 0));
    else
    {
        CHECK(hipGraphKernelNodeSetParams(first, &params));
        hipGraphNode_t           error_node{};
        hipGraphExecUpdateResult result{};
        CHECK(hipGraphExecUpdate(exec, graph, &error_node, &result));
        if(result != hipGraphExecUpdateSuccess) std::exit(4);
    }
    replay();
    int actual[2]{};
    CHECK(hipMemcpy(actual, data, sizeof(actual), hipMemcpyDeviceToHost));
    const int expected_first = mode == 1 ? 6 : 12;
    if(actual[0] != expected_first || actual[1] != 9) std::exit(5);
    CHECK(hipGraphExecDestroy(exec));
    // A fresh exec must obtain a fresh plan, not remain invalid forever.
    CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    replay();
    CHECK(hipGraphExecDestroy(exec));
    CHECK(hipGraphDestroy(graph));
    CHECK(hipFree(data));
    CHECK(hipStreamDestroy(stream));
}

int
main()
{
    for(int mode = 0; mode < 3; ++mode)
        run_case(mode);
    puts("hip-graph-capture-update: complete");
}
