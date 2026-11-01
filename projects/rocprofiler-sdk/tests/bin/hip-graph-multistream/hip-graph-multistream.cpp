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

// Graphs captured across streams, for capture_correlation_id: each graph forks from its capture
// stream after its first launch, launches alternately on its streams, and joins back before its
// last launch. Every launch in a graph runs a different kernel, so a replayed dispatch's kernel
// names the launch that captured it. The app makes no other kernel launch: its launches, in call
// order, are the two captures' in capture order (validate.py relies on it).

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call)                                                                                \
    do                                                                                             \
    {                                                                                              \
        auto _err = (call);                                                                        \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(                                                                               \
                stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, hipGetErrorString(_err));    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)

#define MS_KERNEL(NAME)                                                                            \
    __global__ void NAME(float* x) { x[threadIdx.x] += 1.0f; }

MS_KERNEL(ms_a0)
MS_KERNEL(ms_b0)
MS_KERNEL(ms_b1)
MS_KERNEL(ms_b2)
MS_KERNEL(ms_c0)
MS_KERNEL(ms_c1)
MS_KERNEL(ms_c2)
MS_KERNEL(ms_e0)
MS_KERNEL(ms_e1)
MS_KERNEL(ms_d0)

namespace
{
using kernel_t = void (*)(float*);

struct launch
{
    size_t   stream = 0;
    kernel_t kernel = nullptr;
};

void
capture_and_replay(const std::vector<launch>& launches, size_t nstreams, int replays)
{
    auto streams = std::vector<hipStream_t>(nstreams, nullptr);
    auto joins   = std::vector<hipEvent_t>(nstreams, nullptr);
    auto fork    = hipEvent_t{nullptr};
    for(size_t i = 0; i < nstreams; ++i)
    {
        CHECK(hipStreamCreate(&streams.at(i)));
        CHECK(hipEventCreateWithFlags(&joins.at(i), hipEventDisableTiming));
    }
    CHECK(hipEventCreateWithFlags(&fork, hipEventDisableTiming));
    float* data = nullptr;
    CHECK(hipMalloc(&data, 64 * sizeof(float) * launches.size()));

    auto run = [&](size_t i) {
        const auto& itr = launches.at(i);
        hipLaunchKernelGGL(itr.kernel, dim3(1), dim3(64), 0, streams.at(itr.stream), data + 64 * i);
        CHECK(hipGetLastError());
    };

    auto graph = hipGraph_t{nullptr};
    auto exec  = hipGraphExec_t{nullptr};
    CHECK(hipStreamBeginCapture(streams.at(0), hipStreamCaptureModeGlobal));
    run(0);
    CHECK(hipEventRecord(fork, streams.at(0)));
    for(size_t i = 1; i < nstreams; ++i)
        CHECK(hipStreamWaitEvent(streams.at(i), fork, 0));
    for(size_t i = 1; i + 1 < launches.size(); ++i)
        run(i);
    for(size_t i = 1; i < nstreams; ++i)
    {
        CHECK(hipEventRecord(joins.at(i), streams.at(i)));
        CHECK(hipStreamWaitEvent(streams.at(0), joins.at(i), 0));
    }
    run(launches.size() - 1);
    CHECK(hipStreamEndCapture(streams.at(0), &graph));
    CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    for(int r = 0; r < replays; ++r)
        CHECK(hipGraphLaunch(exec, streams.at(0)));
    CHECK(hipStreamSynchronize(streams.at(0)));

    CHECK(hipGraphExecDestroy(exec));
    CHECK(hipGraphDestroy(graph));
    CHECK(hipFree(data));
    CHECK(hipEventDestroy(fork));
    for(size_t i = 0; i < nstreams; ++i)
    {
        CHECK(hipEventDestroy(joins.at(i)));
        CHECK(hipStreamDestroy(streams.at(i)));
    }
}
}  // namespace

int
main(int argc, char** argv)
{
    int replays = (argc > 1) ? atoi(argv[1]) : 3;

    // two streams, launches alternating: b on the capture stream, c on the other
    capture_and_replay({{0, ms_a0},
                        {0, ms_b0},
                        {1, ms_c0},
                        {0, ms_b1},
                        {1, ms_c1},
                        {0, ms_b2},
                        {1, ms_c2},
                        {0, ms_d0}},
                       2,
                       replays);

    // three streams, round-robin
    capture_and_replay({{0, ms_a0},
                        {0, ms_b0},
                        {1, ms_c0},
                        {2, ms_e0},
                        {0, ms_b1},
                        {1, ms_c1},
                        {2, ms_e1},
                        {0, ms_d0}},
                       3,
                       replays);

    printf("[hip-graph-multistream] complete\n");
    return 0;
}
