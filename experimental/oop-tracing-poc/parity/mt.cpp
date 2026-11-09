// Parity test app: many threads, each with its own streams, issuing HIP calls concurrently.
//
//   mt [threads=4] [streams=2] [iters=50] [n=65536] [graph=1]
//
// Per thread: streams, events, pinned host memory, device memory, memset, async copies, kernel
// launches, event record/synchronize. With graph=1, each thread also captures one stream into a
// graph and launches the graph each iteration.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

__global__ void scale_add(float* y, const float* x, float a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) y[i] = a * x[i] + y[i];
}

namespace
{
std::atomic<int> failures{0};

#define CHECK(x)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        hipError_t _e = (x);                                                                       \
        if(_e != hipSuccess)                                                                       \
        {                                                                                          \
            std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(_e));                    \
            ++failures;                                                                            \
            return;                                                                                \
        }                                                                                          \
    } while(0)

void
worker(int nstreams, int iters, int n, bool use_graph)
{
    CHECK(hipSetDevice(0));
    auto streams = std::vector<hipStream_t>(nstreams);
    auto events  = std::vector<hipEvent_t>(nstreams);
    auto x       = std::vector<float*>(nstreams, nullptr);
    auto y       = std::vector<float*>(nstreams, nullptr);
    float* host  = nullptr;
    const auto bytes = static_cast<size_t>(n) * sizeof(float);
    CHECK(hipHostMalloc(reinterpret_cast<void**>(&host), bytes, hipHostMallocDefault));
    for(int i = 0; i < n; ++i)
        host[i] = 1.0f;
    for(int s = 0; s < nstreams; ++s)
    {
        CHECK(hipStreamCreateWithFlags(&streams[s], hipStreamNonBlocking));
        CHECK(hipEventCreate(&events[s]));
        CHECK(hipMalloc(&x[s], bytes));
        CHECK(hipMalloc(&y[s], bytes));
        CHECK(hipMemsetAsync(y[s], 0, bytes, streams[s]));
        CHECK(hipMemcpyAsync(x[s], host, bytes, hipMemcpyHostToDevice, streams[s]));
    }

    hipGraphExec_t exec = nullptr;
    hipGraph_t     graph = nullptr;
    if(use_graph)
    {
        CHECK(hipStreamBeginCapture(streams[0], hipStreamCaptureModeThreadLocal));
        hipLaunchKernelGGL(scale_add, dim3((n + 255) / 256), dim3(256), 0, streams[0], y[0], x[0],
                           1.0f, n);
        CHECK(hipStreamEndCapture(streams[0], &graph));
        CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }

    for(int it = 0; it < iters; ++it)
    {
        for(int s = 0; s < nstreams; ++s)
        {
            hipLaunchKernelGGL(scale_add, dim3((n + 255) / 256), dim3(256), 0, streams[s], y[s],
                               x[s], 1.0f, n);
            CHECK(hipGetLastError());
            CHECK(hipEventRecord(events[s], streams[s]));
        }
        if(exec) CHECK(hipGraphLaunch(exec, streams[0]));
        for(int s = 0; s < nstreams; ++s)
            CHECK(hipEventSynchronize(events[s]));
    }

    // Each y = iters launches (+ iters graph launches on stream 0) of y += x, x = 1.
    for(int s = 0; s < nstreams; ++s)
    {
        CHECK(hipMemcpyAsync(host, y[s], bytes, hipMemcpyDeviceToHost, streams[s]));
        CHECK(hipStreamSynchronize(streams[s]));
        const float expect = static_cast<float>(iters * ((exec && s == 0) ? 2 : 1));
        if(host[0] != expect || host[n - 1] != expect)
        {
            std::fprintf(stderr, "stream %d: got %g, expected %g\n", s, host[0], expect);
            ++failures;
        }
    }

    if(exec) CHECK(hipGraphExecDestroy(exec));
    if(graph) CHECK(hipGraphDestroy(graph));
    for(int s = 0; s < nstreams; ++s)
    {
        CHECK(hipFree(x[s]));
        CHECK(hipFree(y[s]));
        CHECK(hipEventDestroy(events[s]));
        CHECK(hipStreamDestroy(streams[s]));
    }
    CHECK(hipHostFree(host));
}
}  // namespace

int
main(int argc, char** argv)
{
    const int  nthreads  = (argc > 1) ? std::atoi(argv[1]) : 4;
    const int  nstreams  = (argc > 2) ? std::atoi(argv[2]) : 2;
    const int  iters     = (argc > 3) ? std::atoi(argv[3]) : 50;
    const int  n         = (argc > 4) ? std::atoi(argv[4]) : 65536;
    const bool use_graph = (argc > 5) ? std::atoi(argv[5]) != 0 : true;

    auto threads = std::vector<std::thread>{};
    threads.reserve(nthreads);
    for(int t = 0; t < nthreads; ++t)
        threads.emplace_back(worker, nstreams, iters, n, use_graph);
    for(auto& t : threads)
        t.join();
    if(hipDeviceSynchronize() != hipSuccess) ++failures;

    std::printf("threads=%d streams=%d iters=%d n=%d graph=%d: %s\n", nthreads, nstreams, iters, n,
                use_graph ? 1 : 0, failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
