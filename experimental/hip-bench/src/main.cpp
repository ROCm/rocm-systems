// main.cpp — hip-bench driver.
//
// Runs a sweep of "call one HIP API N times", "submit N kernels + sync" and
// "launch graph of N nodes + sync" segments, repeated kIterations times. Emits
// per-iteration CSV plus a stats footer (min/max/mean/stddev) on stdout.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
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
// API-mode segment: call a traced HIP API that does no GPU work, N times.
//
// The kernel and graph segments include dispatch and sync time, which is what a
// user feels but also swamps the per-call cost of the trace call site. This
// segment is the isolator: hipGetDevice reads thread-local state and returns, so
// the difference between a traced and an untraced run is almost entirely the
// cost of the call site.
//
// The return value is deliberately ignored: a trace call site wraps the call
// whether or not it succeeds, so this segment still measures the thing we care
// about on a build that cannot enumerate a device.
// --------------------------------------------------------------------------
static uint64_t time_api_segment(int n) {
    int  dev = 0;
    auto t0  = clk::now();
    for (int i = 0; i < n; ++i) {
        (void) hipGetDevice(&dev);
    }
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
int main(int argc, char** argv) {
    constexpr int kApiCounts[]    = {1000, 100000};
    constexpr int kKernelCounts[] = {1, 10, 1000};
    constexpr int kGraphCounts[]  = {1, 10, 100, 1000};

    // Overridable because a traced run can emit an event per API call, and the
    // default sweep would hand the tracer ~10M of them.
    const int kIterations = (argc > 1) ? std::atoi(argv[1]) : 100;
    if (kIterations <= 0) {
        std::fprintf(stderr, "usage: %s [iterations]\n", argv[0]);
        return 1;
    }

    // A runtime that cannot enumerate a device still executes and traces API call
    // sites, so the api segment remains meaningful. Kernel and graph work does not,
    // so run only what the runtime in front of us can actually do.
    int device_count = 0;
    const bool have_device =
        (hipGetDeviceCount(&device_count) == hipSuccess) && device_count > 0;
    std::fprintf(stderr, "hipbench: %d device(s) visible; gpu segments %s\n",
                 device_count, have_device ? "enabled" : "SKIPPED");

    // Pre-build graphs once.
    std::vector<GraphHolder> graphs;
    if (have_device) {
        graphs.reserve(sizeof(kGraphCounts) / sizeof(kGraphCounts[0]));
        for (int n : kGraphCounts) graphs.push_back(build_graph(n));
    }

    // Warmup: one full pass, discarded.
    std::fprintf(stderr, "hipbench: warmup\n");
    for (int n : kApiCounts) (void)time_api_segment(n);
    if (have_device) {
        for (int n : kKernelCounts) (void)time_kernel_segment(n);
        for (auto& gh : graphs)     (void)time_graph_segment(gh);
    }

    // Per-segment sample storage so we can emit a stats footer at the end.
    // Key: "<mode>,<count>" string; value: vector of ns samples (size kIterations).
    std::map<std::string, std::vector<uint64_t>> samples;

    // CSV header.
    std::printf("iteration,mode,count,ns\n");

    for (int it = 0; it < kIterations; ++it) {
        for (int n : kApiCounts) {
            uint64_t ns = time_api_segment(n);
            std::printf("%d,api,%d,%llu\n", it, n, (unsigned long long)ns);
            samples["api," + std::to_string(n)].push_back(ns);
        }
        for (int n : kKernelCounts) {
            if (!have_device) break;
            uint64_t ns = time_kernel_segment(n);
            std::printf("%d,kernel,%d,%llu\n", it, n, (unsigned long long)ns);
            samples["kernel," + std::to_string(n)].push_back(ns);
        }
        for (auto& gh : graphs) {
            uint64_t ns = time_graph_segment(gh);
            std::printf("%d,graph,%d,%llu\n", it, gh.node_count, (unsigned long long)ns);
            samples["graph," + std::to_string(gh.node_count)].push_back(ns);
        }
        std::fflush(stdout);
        if ((it + 1) % 10 == 0 || it + 1 == kIterations)
            std::fprintf(stderr, "hipbench: iteration %d/%d done\n", it + 1, kIterations);
    }

    // Stats footer (separated by blank line + new header so consumers can split
    // on the blank line, or grep for "stats," prefix). Emits min/max/mean/stddev
    // per (mode,count) tuple.
    std::printf("\n");
    std::printf("section,mode,count,n,min_ns,max_ns,mean_ns,stddev_ns\n");
    auto emit = [&](const std::string& key) {
        auto& v = samples[key];
        if (v.empty()) return;
        uint64_t mn = v[0], mx = v[0];
        double sum = 0;
        for (auto x : v) {
            mn = std::min(mn, x);
            mx = std::max(mx, x);
            sum += static_cast<double>(x);
        }
        double mean = sum / v.size();
        double sq = 0;
        for (auto x : v) {
            double d = static_cast<double>(x) - mean;
            sq += d * d;
        }
        double stddev = (v.size() > 1) ? std::sqrt(sq / (v.size() - 1)) : 0;
        std::printf("stats,%s,%zu,%llu,%llu,%llu,%llu\n",
            key.c_str(), v.size(),
            (unsigned long long)mn,
            (unsigned long long)mx,
            (unsigned long long)mean,
            (unsigned long long)stddev);
    };
    for (int n : kApiCounts)    emit("api,"    + std::to_string(n));
    for (int n : kKernelCounts) emit("kernel," + std::to_string(n));
    for (auto& gh : graphs)     emit("graph,"  + std::to_string(gh.node_count));

    for (auto& gh : graphs) destroy_graph(gh);
    std::fprintf(stderr, "hipbench: complete\n");
    return 0;
}
