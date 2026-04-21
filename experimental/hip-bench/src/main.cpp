// main.cpp — hip-bench driver.
//
// Runs a sweep of "submit N kernels + sync" and "launch graph of N nodes + sync"
// segments, repeated 10 times. Emits CSV on stdout.

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" __global__ void noop();

#define HIP_CHECK(expr)                                                                   \
    do {                                                                                  \
        hipError_t _err = (expr);                                                         \
        if (_err != hipSuccess) {                                                         \
            std::fprintf(stderr, "hip error %d (%s) at %s:%d: %s\n",                      \
                         _err, hipGetErrorString(_err), __FILE__, __LINE__, #expr);       \
            std::exit(2);                                                                 \
        }                                                                                 \
    } while (0)

using clk = std::chrono::steady_clock;

static uint64_t ns_since(clk::time_point t0) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count());
}

// --------------------------------------------------------------------------
// Kernel-mode segment: launch N noop kernels back-to-back, then sync.
// --------------------------------------------------------------------------
static uint64_t time_kernel_segment(int n) {
    auto t0 = clk::now();
    for (int i = 0; i < n; ++i) {
        hipLaunchKernelGGL(noop, dim3(1), dim3(1), 0, 0);
    }
    HIP_CHECK(hipDeviceSynchronize());
    return ns_since(t0);
}

// --------------------------------------------------------------------------
// Graph-mode helpers: build a chained graph of N kernel nodes.
// --------------------------------------------------------------------------
struct GraphHolder {
    hipGraph_t      graph     = nullptr;
    hipGraphExec_t  exec      = nullptr;
    int             node_count = 0;
};

static GraphHolder build_graph(int n) {
    GraphHolder gh;
    gh.node_count = n;
    HIP_CHECK(hipGraphCreate(&gh.graph, 0));

    hipKernelNodeParams params{};
    params.func           = reinterpret_cast<void*>(&noop);
    params.gridDim        = dim3(1);
    params.blockDim       = dim3(1);
    params.sharedMemBytes = 0;
    params.kernelParams   = nullptr;
    params.extra          = nullptr;

    std::vector<hipGraphNode_t> nodes;
    nodes.reserve(n);
    for (int i = 0; i < n; ++i) {
        hipGraphNode_t node = nullptr;
        const hipGraphNode_t* deps = nodes.empty() ? nullptr : &nodes.back();
        size_t              ndeps = nodes.empty() ? 0 : 1;
        HIP_CHECK(hipGraphAddKernelNode(&node, gh.graph, deps, ndeps, &params));
        nodes.push_back(node);
    }
    HIP_CHECK(hipGraphInstantiate(&gh.exec, gh.graph, nullptr, nullptr, 0));
    return gh;
}

static void destroy_graph(GraphHolder& gh) {
    if (gh.exec)  HIP_CHECK(hipGraphExecDestroy(gh.exec));
    if (gh.graph) HIP_CHECK(hipGraphDestroy(gh.graph));
    gh.exec  = nullptr;
    gh.graph = nullptr;
}

static uint64_t time_graph_segment(const GraphHolder& gh) {
    auto t0 = clk::now();
    HIP_CHECK(hipGraphLaunch(gh.exec, 0));
    HIP_CHECK(hipDeviceSynchronize());
    return ns_since(t0);
}

// --------------------------------------------------------------------------
// Driver
// --------------------------------------------------------------------------
int main() {
    constexpr int kKernelCounts[] = {1, 10, 1000};
    constexpr int kGraphCounts[]  = {1, 10, 100, 1000};
    constexpr int kIterations     = 10;

    // Pre-build graphs once.
    std::vector<GraphHolder> graphs;
    graphs.reserve(sizeof(kGraphCounts) / sizeof(kGraphCounts[0]));
    for (int n : kGraphCounts) graphs.push_back(build_graph(n));

    // Warmup: one full pass, discarded.
    std::fprintf(stderr, "hipbench: warmup\n");
    for (int n : kKernelCounts) (void)time_kernel_segment(n);
    for (auto& gh : graphs)     (void)time_graph_segment(gh);

    // CSV header.
    std::printf("iteration,mode,count,ns\n");

    for (int it = 0; it < kIterations; ++it) {
        for (int n : kKernelCounts) {
            uint64_t ns = time_kernel_segment(n);
            std::printf("%d,kernel,%d,%llu\n", it, n, (unsigned long long)ns);
        }
        for (auto& gh : graphs) {
            uint64_t ns = time_graph_segment(gh);
            std::printf("%d,graph,%d,%llu\n", it, gh.node_count, (unsigned long long)ns);
        }
        std::fflush(stdout);
        std::fprintf(stderr, "hipbench: iteration %d/%d done\n", it + 1, kIterations);
    }

    for (auto& gh : graphs) destroy_graph(gh);
    std::fprintf(stderr, "hipbench: complete\n");
    return 0;
}
