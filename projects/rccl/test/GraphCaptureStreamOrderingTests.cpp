/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Whitebox tests for how ncclLaunchPrepare/ncclLaunchFinish order the internal
// deviceStream against the user stream while a HIP graph is being captured.
//
// After every captured launch, ncclLaunchFinish fast-forwards deviceStream to the
// launch stream's capture frontier. If that frontier is wrong, the next captured
// launch (which waits on deviceStream in ncclLaunchPrepare) either loses a dependency
// on preceding user work, picks up a redundant edge, or references nodes of another
// graph. These tests capture representative sequences, then inspect the resulting
// graph topology node by node and replay it to check the data.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "common/ResourceGuards.hpp"
#include "common/TestChecks.hpp"

namespace RcclUnitTesting
{

static constexpr int    kTimeoutSeconds = 120;
static constexpr size_t kCount          = 1 << 16;
static constexpr int    kRanks          = 2;

__global__ void graphOrderAddKernel(uint32_t* buf, size_t n, uint32_t v)
{
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        buf[i] += v;
}

namespace
{

using EnvMap = std::unordered_map<std::string, std::string>;

// Per-rank communicators, two user streams and two buffers, all on device r.
struct Harness
{
    int                      nRanks = 0;
    std::vector<ncclComm_t>  comms;
    std::vector<hipStream_t> s, t;
    std::vector<uint32_t*>   a, b;

    std::vector<RCCLTestGuards::NcclCommAutoGuard>     commGuards;
    std::vector<RCCLTestGuards::HipStreamAutoGuard>    streamGuards;
    std::vector<RCCLTestGuards::DeviceBufferAutoGuard> bufGuards;

    ~Harness()
    {
        for(int r = 0; r < nRanks; r++)
        {
            (void)hipSetDevice(r);
            (void)hipDeviceSynchronize();
        }
        // Communicators must go before the streams and buffers they reference.
        commGuards.clear();
    }
};

void initHarness(Harness& h, int nRanks)
{
    int devCount = 0;
    HIP_CHECK(hipGetDeviceCount(&devCount));
    if(devCount < nRanks)
        GTEST_SKIP() << "Requires >= " << nRanks << " GPUs; found " << devCount;

    h.nRanks = nRanks;
    h.comms.assign(nRanks, nullptr);
    h.s.assign(nRanks, nullptr);
    h.t.assign(nRanks, nullptr);
    h.a.assign(nRanks, nullptr);
    h.b.assign(nRanks, nullptr);

    std::vector<int> devices(nRanks);
    std::iota(devices.begin(), devices.end(), 0);
    ASSERT_EQ(ncclCommInitAll(h.comms.data(), nRanks, devices.data()), ncclSuccess);
    for(int r = 0; r < nRanks; r++)
        h.commGuards.push_back(RCCLTestGuards::makeCommAutoGuard(h.comms[r]));

    for(int r = 0; r < nRanks; r++)
    {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipStreamCreateWithFlags(&h.s[r], hipStreamNonBlocking));
        h.streamGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(h.s[r]));
        HIP_CHECK(hipStreamCreateWithFlags(&h.t[r], hipStreamNonBlocking));
        h.streamGuards.push_back(RCCLTestGuards::makeStreamAutoGuard(h.t[r]));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&h.a[r]), kCount * sizeof(uint32_t)));
        h.bufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(h.a[r]));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&h.b[r]), kCount * sizeof(uint32_t)));
        h.bufGuards.push_back(RCCLTestGuards::makeDeviceBufferAutoGuard(h.b[r]));
    }
}

// --- Stream operations, issued for every rank -------------------------------------

using BufSel    = std::vector<uint32_t*> Harness::*;
using StreamSel = std::vector<hipStream_t> Harness::*;

void fillOp(Harness& h, BufSel buf, StreamSel st, const std::function<uint32_t(int)>& value)
{
    for(int r = 0; r < h.nRanks; r++)
    {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>((h.*buf)[r]), value(r), kCount,
                                    (h.*st)[r]));
    }
}

void addOp(Harness& h, BufSel buf, StreamSel st, uint32_t v)
{
    for(int r = 0; r < h.nRanks; r++)
    {
        HIP_CHECK(hipSetDevice(r));
        hipLaunchKernelGGL(graphOrderAddKernel, dim3(64), dim3(256), 0, (h.*st)[r], (h.*buf)[r], kCount, v);
        HIP_CHECK(hipGetLastError());
    }
}

void allReduceOp(Harness& h, BufSel buf, StreamSel st, size_t count = kCount)
{
    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for(int r = 0; r < h.nRanks; r++)
    {
        EXPECT_EQ(ncclAllReduce((h.*buf)[r], (h.*buf)[r], count, ncclUint32, ncclSum, h.comms[r], (h.*st)[r]),
                  ncclSuccess);
    }
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
}

void syncAll(Harness& h)
{
    for(int r = 0; r < h.nRanks; r++)
    {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipStreamSynchronize(h.s[r]));
        HIP_CHECK(hipStreamSynchronize(h.t[r]));
    }
}

void expectBuffer(Harness& h, BufSel buf, uint32_t expected, const std::string& what)
{
    std::vector<uint32_t> host(kCount);
    for(int r = 0; r < h.nRanks; r++)
    {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipMemcpy(host.data(), (h.*buf)[r], kCount * sizeof(uint32_t), hipMemcpyDeviceToHost));
        auto bad = std::find_if(host.begin(), host.end(), [&](uint32_t x) { return x != expected; });
        EXPECT_EQ(bad, host.end()) << what << ": rank " << r << " index " << (bad - host.begin()) << " got "
                                   << (bad == host.end() ? 0 : *bad) << " expected " << expected;
    }
}

// --- Capture, replay -------------------------------------------------------------

struct Graphs
{
    std::vector<hipGraph_t>     graph;
    std::vector<hipGraphExec_t> exec;
    ~Graphs()
    {
        for(size_t r = 0; r < graph.size(); r++)
        {
            (void)hipSetDevice(r);
            if(exec[r]) (void)hipGraphExecDestroy(exec[r]);
            if(graph[r]) (void)hipGraphDestroy(graph[r]);
        }
    }
};

void beginCapture(Harness& h, StreamSel st, hipStreamCaptureMode mode)
{
    for(int r = 0; r < h.nRanks; r++)
    {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipStreamBeginCapture((h.*st)[r], mode));
    }
}

void endCapture(Harness& h, StreamSel st, Graphs& g)
{
    g.graph.assign(h.nRanks, nullptr);
    g.exec.assign(h.nRanks, nullptr);
    for(int r = 0; r < h.nRanks; r++)
    {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipStreamEndCapture((h.*st)[r], &g.graph[r]));
    }
    for(int r = 0; r < h.nRanks; r++)
    {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipGraphInstantiate(&g.exec[r], g.graph[r], nullptr, nullptr, 0));
    }
}

void launchGraphs(Harness& h, Graphs& g, StreamSel st)
{
    for(int r = 0; r < h.nRanks; r++)
    {
        HIP_CHECK(hipSetDevice(r));
        HIP_CHECK(hipGraphLaunch(g.exec[r], (h.*st)[r]));
    }
}

void capture(Harness& h, hipStreamCaptureMode mode, const std::function<void()>& body, Graphs& g)
{
    ASSERT_NO_FATAL_FAILURE(beginCapture(h, &Harness::s, mode));
    body();
    ASSERT_NO_FATAL_FAILURE(endCapture(h, &Harness::s, g));
}

// --- Graph topology --------------------------------------------------------------
//
// Nodes are labelled U (graphOrderAddKernel), M (memset), C (any other kernel, i.e. an
// RCCL kernel) and E (event record). Unlabelled nodes (host callbacks, empty nodes) still
// take part in reachability.

struct GraphShape
{
    size_t                         nodes = 0, edges = 0;
    std::vector<hipGraphNode_t>    node;
    std::vector<char>              label;
    std::vector<std::vector<int>>  succ;
    std::vector<std::vector<bool>> reach; // reach[i][j]: path i -> j of length >= 1

    // Labels of the M/C/U nodes in dependency order, or "" if they are not totally ordered.
    std::string order() const
    {
        std::vector<int> idx;
        for(size_t i = 0; i < node.size(); i++)
            if(label[i] == 'M' || label[i] == 'C' || label[i] == 'U') idx.push_back(i);
        auto ancestors = [&](int j) {
            int n = 0;
            for(size_t i = 0; i < node.size(); i++) n += reach[i][j];
            return n;
        };
        std::sort(idx.begin(), idx.end(), [&](int x, int y) { return ancestors(x) < ancestors(y); });
        std::string out;
        for(size_t k = 0; k < idx.size(); k++)
        {
            if(k > 0 && !reach[idx[k - 1]][idx[k]]) return "";
            out += label[idx[k]];
        }
        return out;
    }

    // Edges u->v that are implied by another path u -> w ->* v.
    size_t redundantEdges() const
    {
        size_t n = 0;
        for(size_t u = 0; u < node.size(); u++)
            for(int v : succ[u])
                for(int w : succ[u])
                    if(w != v && reach[w][v])
                    {
                        n++;
                        break;
                    }
        return n;
    }

    int count(char c) const { return std::count(label.begin(), label.end(), c); }

    size_t maxInDegree() const
    {
        std::vector<size_t> indeg(node.size(), 0);
        for(size_t u = 0; u < node.size(); u++)
            for(int v : succ[u]) indeg[v]++;
        return node.empty() ? 0 : *std::max_element(indeg.begin(), indeg.end());
    }

    // "0M->1C 1C->2U ..." for failure messages.
    std::string describe() const
    {
        std::string out;
        for(size_t u = 0; u < node.size(); u++)
            for(int v : succ[u])
                out += std::to_string(u) + label[u] + "->" + std::to_string(v) + label[v] + " ";
        return out;
    }

    std::vector<int> nodesLabelled(char c) const
    {
        std::vector<int> out;
        for(size_t i = 0; i < label.size(); i++)
            if(label[i] == c) out.push_back(i);
        return out;
    }

    // Every node labelled `from` has a path to every node labelled `to`.
    bool allBefore(char from, char to) const
    {
        for(int x : nodesLabelled(from))
            for(int y : nodesLabelled(to))
                if(!reach[x][y]) return false;
        return true;
    }
};

// Collapse runs of the same label: "MCCCU" -> "MCU".
std::string collapse(const std::string& s)
{
    std::string out;
    for(char c : s)
        if(out.empty() || out.back() != c) out += c;
    return out;
}

void analyze(hipGraph_t g, GraphShape& shape)
{
    HIP_CHECK(hipGraphGetNodes(g, nullptr, &shape.nodes));
    shape.node.resize(shape.nodes);
    HIP_CHECK(hipGraphGetNodes(g, shape.node.data(), &shape.nodes));
    HIP_CHECK(hipGraphGetEdges(g, nullptr, nullptr, &shape.edges));
    std::vector<hipGraphNode_t> from(shape.edges), to(shape.edges);
    HIP_CHECK(hipGraphGetEdges(g, from.data(), to.data(), &shape.edges));

    std::unordered_map<hipGraphNode_t, int> index;
    for(size_t i = 0; i < shape.nodes; i++) index[shape.node[i]] = i;

    shape.label.assign(shape.nodes, '-');
    for(size_t i = 0; i < shape.nodes; i++)
    {
        hipGraphNodeType type;
        HIP_CHECK(hipGraphNodeGetType(shape.node[i], &type));
        if(type == hipGraphNodeTypeMemset)
        {
            shape.label[i] = 'M';
        }
        else if(type == hipGraphNodeTypeEventRecord)
        {
            shape.label[i] = 'E';
        }
        else if(type == hipGraphNodeTypeKernel)
        {
            hipKernelNodeParams p{};
            bool user = hipGraphKernelNodeGetParams(shape.node[i], &p) == hipSuccess &&
                        p.func == reinterpret_cast<void*>(graphOrderAddKernel);
            shape.label[i] = user ? 'U' : 'C';
        }
    }

    shape.succ.assign(shape.nodes, {});
    std::vector<int> indeg(shape.nodes, 0);
    for(size_t e = 0; e < shape.edges; e++)
    {
        shape.succ[index.at(from[e])].push_back(index.at(to[e]));
        indeg[index.at(to[e])]++;
    }

    // Reachability in reverse topological order.
    std::vector<int> topo;
    for(size_t i = 0; i < shape.nodes; i++)
        if(indeg[i] == 0) topo.push_back(i);
    for(size_t k = 0; k < topo.size(); k++)
        for(int v : shape.succ[topo[k]])
            if(--indeg[v] == 0) topo.push_back(v);
    ASSERT_EQ(topo.size(), shape.nodes) << "graph has a cycle";
    shape.reach.assign(shape.nodes, std::vector<bool>(shape.nodes, false));
    for(auto it = topo.rbegin(); it != topo.rend(); ++it)
        for(int v : shape.succ[*it])
        {
            shape.reach[*it][v] = true;
            for(size_t j = 0; j < shape.nodes; j++)
                if(shape.reach[v][j]) shape.reach[*it][j] = true;
        }
}

void analyzeRank(Graphs& g, int r, GraphShape& shape)
{
    HIP_CHECK(hipSetDevice(r));
    ASSERT_NO_FATAL_FAILURE(analyze(g.graph[r], shape));
}

// A captured launch joins the user stream with deviceStream and at most one more internal
// strong stream (hostStream when plans carry proxy ops), each adding one dependency. When user work sits between collectives, the deviceStream dependency
// duplicates the path through that work, so redundant edges are expected; what must not
// happen is a node's dependency count growing with the length of the capture, which is
// what a deviceStream that is not fast-forwarded produces.
static constexpr size_t kMaxDependencies = 3;

void expectBoundedDependencies(const GraphShape& shape, const std::string& what)
{
    EXPECT_LE(shape.maxInDegree(), kMaxDependencies) << what << ": " << shape.describe();
}

// With graph mixing, ncclLaunchFinish releases deviceStream, which records serialEvent on
// deviceStream's capture frontier; uncaptured launches wait on that event. Every
// collective must therefore precede some event record, which holds only if
// deviceStream was fast-forwarded past it.
void expectSerialEventAfterCollectives(const GraphShape& shape, const std::string& what)
{
    std::vector<int> events = shape.nodesLabelled('E');
    ASSERT_FALSE(events.empty()) << what << ": no event record nodes: " << shape.describe();
    for(int c : shape.nodesLabelled('C'))
    {
        bool reached = std::any_of(events.begin(), events.end(), [&](int e) { return shape.reach[c][e]; });
        EXPECT_TRUE(reached) << what << ": collective node " << c << " precedes no event record: "
                             << shape.describe();
    }
}

// --- Test bodies -----------------------------------------------------------------

// Back-to-back captured AllReduces must form a single chain with no redundant edges.
void sequentialCollectives(int n, hipStreamCaptureMode mode)
{
    Harness h;
    ASSERT_NO_FATAL_FAILURE(initHarness(h, kRanks));
    if(h.nRanks == 0) return;

    Graphs g;
    ASSERT_NO_FATAL_FAILURE(capture(h, mode, [&] {
        fillOp(h, &Harness::a, &Harness::s, [](int r) { return r + 1; });
        for(int i = 0; i < n; i++) allReduceOp(h, &Harness::a, &Harness::s);
    }, g));

    for(int r = 0; r < h.nRanks; r++)
    {
        GraphShape shape;
        ASSERT_NO_FATAL_FAILURE(analyzeRank(g, r, shape));
        EXPECT_EQ(shape.count('C'), n) << "rank " << r;
        EXPECT_EQ(shape.order(), "M" + std::string(n, 'C')) << "rank " << r;
        EXPECT_EQ(shape.redundantEdges(), 0u) << "rank " << r << ": " << shape.describe();
    }

    // Sum over ranks of (r+1) is 3, then each further AllReduce doubles it (mod 2^32).
    const uint32_t expected = static_cast<uint32_t>(3ull << (n - 1));
    for(int rep = 0; rep < 3; rep++)
    {
        ASSERT_NO_FATAL_FAILURE(launchGraphs(h, g, &Harness::s));
        ASSERT_NO_FATAL_FAILURE(syncAll(h));
        expectBuffer(h, &Harness::a, expected, "replay " + std::to_string(rep));
    }
}

// User work between captured collectives must stay ordered on both sides of each one.
void interleavedUserWork(hipStreamCaptureMode mode, bool checkDependencyBound, bool checkSerialEvent,
                         int rounds = 12)
{
    Harness h;
    ASSERT_NO_FATAL_FAILURE(initHarness(h, kRanks));
    if(h.nRanks == 0) return;

    Graphs g;
    ASSERT_NO_FATAL_FAILURE(capture(h, mode, [&] {
        fillOp(h, &Harness::a, &Harness::s, [](int r) { return r + 1; });
        for(int i = 0; i < rounds; i++)
        {
            allReduceOp(h, &Harness::a, &Harness::s);
            addOp(h, &Harness::a, &Harness::s, i + 1);
        }
    }, g));

    std::string expectedOrder = "M";
    for(int i = 0; i < rounds; i++) expectedOrder += "CU";
    for(int r = 0; r < h.nRanks; r++)
    {
        GraphShape shape;
        ASSERT_NO_FATAL_FAILURE(analyzeRank(g, r, shape));
        EXPECT_EQ(shape.order(), expectedOrder) << "rank " << r << ": " << shape.describe();
        if(checkDependencyBound) expectBoundedDependencies(shape, "rank " + std::to_string(r));
        if(checkSerialEvent) expectSerialEventAfterCollectives(shape, "rank " + std::to_string(r));
    }

    // Ranks start at 1 and 2; every AllReduce leaves both at twice the common value.
    uint32_t expected = 3;
    for(int i = 0; i < rounds; i++) expected = (i == 0 ? expected : 2 * expected) + (i + 1);
    for(int rep = 0; rep < 2; rep++)
    {
        ASSERT_NO_FATAL_FAILURE(launchGraphs(h, g, &Harness::s));
        ASSERT_NO_FATAL_FAILURE(syncAll(h));
        expectBuffer(h, &Harness::a, expected, "replay " + std::to_string(rep));
    }
}

ProcessIsolatedTestRunner::TestConfig isolated(const std::string& name, std::function<void()> body,
                                               const EnvMap& env = {})
{
    EnvMap full = {{"NCCL_DEBUG", "WARN"}};
    full.insert(env.begin(), env.end());
    return ProcessIsolatedTestRunner::TestConfig(name, std::move(body))
        .withNumGpus(kRanks)
        .withEnvironment(full)
        .withTimeout(std::chrono::seconds(kTimeoutSeconds));
}

ProcessIsolatedTestRunner::ExecutionOptions parallelOptions()
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.maxParallelJobs    = ProcessIsolatedTestRunner::ExecutionOptions::kAutoParallelism;
    return options;
}

const char* modeName(hipStreamCaptureMode mode)
{
    switch(mode)
    {
    case hipStreamCaptureModeGlobal: return "Global";
    case hipStreamCaptureModeThreadLocal: return "ThreadLocal";
    default: return "Relaxed";
    }
}

} // namespace

class GraphCaptureStreamOrdering : public ::testing::Test
{
};

// One captured collective, the smallest chain, under every capture mode.
TEST_F(GraphCaptureStreamOrdering, SingleCollective)
{
    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        parallelOptions(),
        isolated("Global", [] { sequentialCollectives(1, hipStreamCaptureModeGlobal); }),
        isolated("ThreadLocal", [] { sequentialCollectives(1, hipStreamCaptureModeThreadLocal); }),
        isolated("Relaxed", [] { sequentialCollectives(1, hipStreamCaptureModeRelaxed); }));
}

// Many collectives: the fast-forward must not accumulate edges as the capture grows.
TEST_F(GraphCaptureStreamOrdering, LongChainNoEdgeBlowup)
{
    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        parallelOptions(),
        isolated("Chain16", [] { sequentialCollectives(16, hipStreamCaptureModeThreadLocal); }),
        isolated("Chain128", [] { sequentialCollectives(128, hipStreamCaptureModeThreadLocal); }));
}

// User kernels between collectives, across capture modes and the environment settings
// that change which streams ncclLaunchPrepare/ncclLaunchFinish touch.
TEST_F(GraphCaptureStreamOrdering, InterleavedUserWork)
{
    std::vector<ProcessIsolatedTestRunner::TestConfig> configs;
    for(auto mode : {hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed})
        for(const char* mixing : {"0", "1"})
            for(const char* implicitOrder : {"0", "1"})
            {
                std::string name = std::string(modeName(mode)) + "_Mixing" + mixing + "_ImplicitOrder" + implicitOrder;
                // With implicit launch order, ncclLaunchFinish makes the per-device launchOrder
                // stream plain-wait on every launch, so each captured kernel depends on all
                // earlier ones. Only ordering and data are checked in that configuration.
                bool bound  = implicitOrder[0] == '0';
                bool serial = mixing[0] == '1';
                configs.push_back(isolated(name, [mode, bound, serial] { interleavedUserWork(mode, bound, serial); },
                                           {{"NCCL_GRAPH_MIXING_SUPPORT", mixing},
                                            {"NCCL_LAUNCH_ORDER_IMPLICIT", implicitOrder}}));
            }
    for(const auto& c : configs) ProcessIsolatedTestRunner::registerTest(c);
    EXPECT_TRUE(ProcessIsolatedTestRunner::executeAllTests(parallelOptions()));
}

// Network transport: plans carry proxy ops, so every captured launch also adds
// host-callback nodes via hostStream.
TEST_F(GraphCaptureStreamOrdering, InterleavedUserWorkNetTransport)
{
    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        parallelOptions(),
        isolated("Socket", [] { interleavedUserWork(hipStreamCaptureModeThreadLocal, true, false); },
                 {{"NCCL_P2P_DISABLE", "1"}, {"NCCL_SHM_DISABLE", "1"}, {"NCCL_NET", "Socket"}}));
}

// The user stream already depends on two branches (fork/join with plain HIP work on a
// second stream) when RCCL launches; RCCL itself only sees one stream.
TEST_F(GraphCaptureStreamOrdering, ForkJoinBeforeCollective)
{
    RUN_ISOLATED_TESTS_WITH_OPTIONS(parallelOptions(), isolated("ForkJoin", [] {
        Harness h;
        ASSERT_NO_FATAL_FAILURE(initHarness(h, kRanks));
        if(h.nRanks == 0) return;
        std::vector<hipEvent_t> fork(h.nRanks), join(h.nRanks);
        std::vector<RCCLTestGuards::HipEventAutoGuard> eventGuards;
        for(int r = 0; r < h.nRanks; r++)
        {
            HIP_CHECK(hipSetDevice(r));
            HIP_CHECK(hipEventCreateWithFlags(&fork[r], hipEventDisableTiming));
            eventGuards.push_back(RCCLTestGuards::makeEventAutoGuard(fork[r]));
            HIP_CHECK(hipEventCreateWithFlags(&join[r], hipEventDisableTiming));
            eventGuards.push_back(RCCLTestGuards::makeEventAutoGuard(join[r]));
        }

        Graphs g;
        ASSERT_NO_FATAL_FAILURE(capture(h, hipStreamCaptureModeThreadLocal, [&] {
            for(int r = 0; r < h.nRanks; r++)
            {
                HIP_CHECK(hipSetDevice(r));
                HIP_CHECK(hipEventRecord(fork[r], h.s[r]));
                HIP_CHECK(hipStreamWaitEvent(h.t[r], fork[r], 0));
            }
            fillOp(h, &Harness::a, &Harness::s, [](int r) { return r + 1; });
            fillOp(h, &Harness::b, &Harness::t, [](int r) { return 10 * (r + 1); });
            addOp(h, &Harness::b, &Harness::t, 1); // 11, 21
            for(int r = 0; r < h.nRanks; r++)
            {
                HIP_CHECK(hipSetDevice(r));
                HIP_CHECK(hipEventRecord(join[r], h.t[r]));
                HIP_CHECK(hipStreamWaitEvent(h.s[r], join[r], 0));
            }
            ASSERT_EQ(ncclGroupStart(), ncclSuccess);
            allReduceOp(h, &Harness::a, &Harness::s); // 3
            allReduceOp(h, &Harness::b, &Harness::s); // 32
            ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
            addOp(h, &Harness::a, &Harness::s, 4); // 7
            allReduceOp(h, &Harness::a, &Harness::s); // 14
        }, g));

        for(int r = 0; r < h.nRanks; r++)
        {
            GraphShape shape;
            ASSERT_NO_FATAL_FAILURE(analyzeRank(g, r, shape));
            ASSERT_EQ(shape.count('M'), 2);
            ASSERT_EQ(shape.count('U'), 2);
            ASSERT_GE(shape.count('C'), 2);
            // Both branches (each memset, and the add on the side stream) must feed every
            // collective; the trailing add must sit between collectives.
            EXPECT_TRUE(shape.allBefore('M', 'C')) << "rank " << r;
            int feedsAll = 0, between = 0;
            for(int u : shape.nodesLabelled('U'))
            {
                bool toAll = true, fromSome = false, toSome = false;
                for(int c : shape.nodesLabelled('C'))
                {
                    toAll    = toAll && shape.reach[u][c];
                    toSome   = toSome || shape.reach[u][c];
                    fromSome = fromSome || shape.reach[c][u];
                }
                feedsAll += toAll;
                between += fromSome && toSome;
            }
            EXPECT_EQ(feedsAll, 1) << "rank " << r;
            EXPECT_EQ(between, 1) << "rank " << r;
            expectBoundedDependencies(shape, "rank " + std::to_string(r));
        }
        ASSERT_NO_FATAL_FAILURE(launchGraphs(h, g, &Harness::s));
        ASSERT_NO_FATAL_FAILURE(syncAll(h));
        expectBuffer(h, &Harness::a, 14, "a");
        expectBuffer(h, &Harness::b, 32, "b");
    }));
}

// Several collectives in one group on the same stream (one planner, numStreams == 1),
// plus a captured send/recv exchange.
TEST_F(GraphCaptureStreamOrdering, GroupedAndPointToPoint)
{
    RUN_ISOLATED_TESTS_WITH_OPTIONS(parallelOptions(), isolated("GroupAndSendRecv", [] {
        Harness h;
        ASSERT_NO_FATAL_FAILURE(initHarness(h, kRanks));
        if(h.nRanks == 0) return;

        Graphs g;
        ASSERT_NO_FATAL_FAILURE(capture(h, hipStreamCaptureModeThreadLocal, [&] {
            fillOp(h, &Harness::a, &Harness::s, [](int r) { return r + 1; });
            fillOp(h, &Harness::b, &Harness::s, [](int r) { return 100 * (r + 1); });
            ASSERT_EQ(ncclGroupStart(), ncclSuccess);
            for(int r = 0; r < h.nRanks; r++)
            {
                EXPECT_EQ(ncclAllReduce(h.a[r], h.a[r], kCount, ncclUint32, ncclSum, h.comms[r], h.s[r]),
                          ncclSuccess);
                EXPECT_EQ(ncclBroadcast(h.b[r], h.b[r], kCount, ncclUint32, 1, h.comms[r], h.s[r]), ncclSuccess);
            }
            ASSERT_EQ(ncclGroupEnd(), ncclSuccess);         // a = 3, b = 200
            addOp(h, &Harness::a, &Harness::s, 1);          // a = 4
            ASSERT_EQ(ncclGroupStart(), ncclSuccess);
            for(int r = 0; r < h.nRanks; r++)
            {
                int peer = (r + 1) % h.nRanks;
                EXPECT_EQ(ncclSend(h.a[r], kCount, ncclUint32, peer, h.comms[r], h.s[r]), ncclSuccess);
                EXPECT_EQ(ncclRecv(h.b[r], kCount, ncclUint32, peer, h.comms[r], h.s[r]), ncclSuccess);
            }
            ASSERT_EQ(ncclGroupEnd(), ncclSuccess);         // b = 4
            addOp(h, &Harness::b, &Harness::s, 3);          // b = 7
        }, g));

        for(int r = 0; r < h.nRanks; r++)
        {
            GraphShape shape;
            ASSERT_NO_FATAL_FAILURE(analyzeRank(g, r, shape));
            EXPECT_EQ(collapse(shape.order()), "MCUCU") << "rank " << r;
            expectBoundedDependencies(shape, "rank " + std::to_string(r));
        }
        for(int rep = 0; rep < 2; rep++)
        {
            ASSERT_NO_FATAL_FAILURE(launchGraphs(h, g, &Harness::s));
            ASSERT_NO_FATAL_FAILURE(syncAll(h));
            expectBuffer(h, &Harness::a, 4, "a");
            expectBuffer(h, &Harness::b, 7, "b");
        }
    }));
}

// Eager launches before, between (after EndCapture, before replay) and after graph
// launches on the same stream: the capture must not leave deviceStream inconsistent
// for uncaptured work. Run with implicit launch order too, which sends eager
// single-stream launches through the ncclLaunchFinish fast-forward as well.
// NCCL_GRAPH_STREAM_ORDERING=0 is only exercised here, where the collective is the first
// captured node: on ROCm 7.0 an external event wait after any captured node segfaults in HIP.
TEST_F(GraphCaptureStreamOrdering, EagerAroundCapture)
{
    auto body = [] {
        Harness h;
        ASSERT_NO_FATAL_FAILURE(initHarness(h, kRanks));
        if(h.nRanks == 0) return;
        ASSERT_NO_FATAL_FAILURE(fillOp(h, &Harness::a, &Harness::s, [](int r) { return r + 1; }));
        ASSERT_NO_FATAL_FAILURE(allReduceOp(h, &Harness::a, &Harness::s)); // 3

        Graphs g;
        ASSERT_NO_FATAL_FAILURE(capture(h, hipStreamCaptureModeThreadLocal, [&] {
            allReduceOp(h, &Harness::a, &Harness::s);
            addOp(h, &Harness::a, &Harness::s, 1);
        }, g));

        ASSERT_NO_FATAL_FAILURE(allReduceOp(h, &Harness::a, &Harness::s)); // 6
        ASSERT_NO_FATAL_FAILURE(launchGraphs(h, g, &Harness::s));          // 13
        ASSERT_NO_FATAL_FAILURE(allReduceOp(h, &Harness::a, &Harness::s)); // 26
        ASSERT_NO_FATAL_FAILURE(launchGraphs(h, g, &Harness::s));          // 53
        ASSERT_NO_FATAL_FAILURE(syncAll(h));
        expectBuffer(h, &Harness::a, 53, "a");
    };
    RUN_ISOLATED_TESTS_WITH_OPTIONS(parallelOptions(), isolated("Default", body),
                                    isolated("ImplicitOrder", body, {{"NCCL_LAUNCH_ORDER_IMPLICIT", "1"}}),
                                    isolated("Mixing", body, {{"NCCL_GRAPH_MIXING_SUPPORT", "1"}}),
                                    isolated("StreamOrdering0", body, {{"NCCL_GRAPH_STREAM_ORDERING", "0"}}));
}

// Eager group spanning two user streams (numStreams == 2): exercises the uncaptured
// multi-stream join in ncclLaunchPrepare/ncclLaunchFinish next to captured launches.
TEST_F(GraphCaptureStreamOrdering, EagerMultiStreamAroundCapture)
{
    RUN_ISOLATED_TESTS_WITH_OPTIONS(parallelOptions(), isolated("MultiStream", [] {
        Harness h;
        ASSERT_NO_FATAL_FAILURE(initHarness(h, kRanks));
        if(h.nRanks == 0) return;
        ASSERT_NO_FATAL_FAILURE(fillOp(h, &Harness::a, &Harness::s, [](int r) { return r + 1; }));
        ASSERT_NO_FATAL_FAILURE(fillOp(h, &Harness::b, &Harness::t, [](int r) { return 10 * (r + 1); }));

        auto twoStreamGroup = [&] {
            ASSERT_EQ(ncclGroupStart(), ncclSuccess);
            allReduceOp(h, &Harness::a, &Harness::s);
            allReduceOp(h, &Harness::b, &Harness::t);
            ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
        };
        ASSERT_NO_FATAL_FAILURE(twoStreamGroup()); // a = 3, b = 30

        Graphs g;
        ASSERT_NO_FATAL_FAILURE(capture(h, hipStreamCaptureModeThreadLocal, [&] {
            allReduceOp(h, &Harness::a, &Harness::s);
        }, g));

        ASSERT_NO_FATAL_FAILURE(launchGraphs(h, g, &Harness::s)); // a = 6
        ASSERT_NO_FATAL_FAILURE(twoStreamGroup());                 // a = 12, b = 60
        ASSERT_NO_FATAL_FAILURE(launchGraphs(h, g, &Harness::s)); // a = 24
        ASSERT_NO_FATAL_FAILURE(syncAll(h));
        expectBuffer(h, &Harness::a, 24, "a");
        expectBuffer(h, &Harness::b, 60, "b");
    }));
}

// Two graphs captured from the same communicators at the same time on different
// streams, with launches interleaved. Each graph's deviceStream capture must only ever
// see that graph's nodes.
TEST_F(GraphCaptureStreamOrdering, ConcurrentCapturesSameComm)
{
    RUN_ISOLATED_TESTS_WITH_OPTIONS(parallelOptions(), isolated("Concurrent", [] {
        Harness h;
        ASSERT_NO_FATAL_FAILURE(initHarness(h, kRanks));
        if(h.nRanks == 0) return;

        Graphs g1, g2;
        ASSERT_NO_FATAL_FAILURE(beginCapture(h, &Harness::s, hipStreamCaptureModeThreadLocal));
        ASSERT_NO_FATAL_FAILURE(fillOp(h, &Harness::a, &Harness::s, [](int r) { return r + 1; }));
        ASSERT_NO_FATAL_FAILURE(allReduceOp(h, &Harness::a, &Harness::s)); // a = 3
        ASSERT_NO_FATAL_FAILURE(beginCapture(h, &Harness::t, hipStreamCaptureModeThreadLocal));
        ASSERT_NO_FATAL_FAILURE(fillOp(h, &Harness::b, &Harness::t, [](int r) { return 10 * (r + 1); }));
        ASSERT_NO_FATAL_FAILURE(allReduceOp(h, &Harness::b, &Harness::t)); // b = 30
        ASSERT_NO_FATAL_FAILURE(addOp(h, &Harness::a, &Harness::s, 1));    // a = 4
        ASSERT_NO_FATAL_FAILURE(allReduceOp(h, &Harness::a, &Harness::s)); // a = 8
        ASSERT_NO_FATAL_FAILURE(addOp(h, &Harness::b, &Harness::t, 2));    // b = 32
        ASSERT_NO_FATAL_FAILURE(endCapture(h, &Harness::t, g2));
        ASSERT_NO_FATAL_FAILURE(endCapture(h, &Harness::s, g1));

        for(int r = 0; r < h.nRanks; r++)
        {
            GraphShape s1, s2;
            ASSERT_NO_FATAL_FAILURE(analyzeRank(g1, r, s1));
            ASSERT_NO_FATAL_FAILURE(analyzeRank(g2, r, s2));
            EXPECT_EQ(collapse(s1.order()), "MCUC") << "rank " << r;
            EXPECT_EQ(s1.count('C'), 2) << "rank " << r;
            EXPECT_EQ(collapse(s2.order()), "MCU") << "rank " << r;
            EXPECT_EQ(s2.count('C'), 1) << "rank " << r;
            expectBoundedDependencies(s1, "graph 1 rank " + std::to_string(r));
            expectBoundedDependencies(s2, "graph 2 rank " + std::to_string(r));
        }
        // The graphs share communicators, so they must not run concurrently.
        ASSERT_NO_FATAL_FAILURE(launchGraphs(h, g1, &Harness::s));
        ASSERT_NO_FATAL_FAILURE(syncAll(h));
        ASSERT_NO_FATAL_FAILURE(launchGraphs(h, g2, &Harness::t));
        ASSERT_NO_FATAL_FAILURE(syncAll(h));
        expectBuffer(h, &Harness::a, 8, "a");
        expectBuffer(h, &Harness::b, 32, "b");
    }));
}

// Repeated capture/instantiate/launch/destroy on the same stream, then on the other
// stream: each new graph starts from a fresh deviceStream capture.
TEST_F(GraphCaptureStreamOrdering, RecaptureAcrossStreams)
{
    RUN_ISOLATED_TESTS_WITH_OPTIONS(parallelOptions(), isolated("Recapture", [] {
        Harness h;
        ASSERT_NO_FATAL_FAILURE(initHarness(h, kRanks));
        if(h.nRanks == 0) return;
        for(int iter = 0; iter < 6; iter++)
        {
            StreamSel st = (iter < 3) ? &Harness::s : &Harness::t;
            Graphs    g;
            ASSERT_NO_FATAL_FAILURE(beginCapture(h, st, hipStreamCaptureModeThreadLocal));
            fillOp(h, &Harness::a, st, [iter](int r) { return iter + r; }); // sum 2*iter+1
            allReduceOp(h, &Harness::a, st);
            addOp(h, &Harness::a, st, 1);
            allReduceOp(h, &Harness::a, st);
            ASSERT_NO_FATAL_FAILURE(endCapture(h, st, g));
            for(int r = 0; r < h.nRanks; r++)
            {
                GraphShape shape;
                ASSERT_NO_FATAL_FAILURE(analyzeRank(g, r, shape));
                EXPECT_EQ(collapse(shape.order()), "MCUC") << "iter " << iter << " rank " << r;
                expectBoundedDependencies(shape, "iter " + std::to_string(iter) + " rank " + std::to_string(r));
            }
            ASSERT_NO_FATAL_FAILURE(launchGraphs(h, g, st));
            ASSERT_NO_FATAL_FAILURE(syncAll(h));
            expectBuffer(h, &Harness::a, 2 * (2 * iter + 2), "iter " + std::to_string(iter));
        }
    }));
}

// Edge cases where a captured RCCL call produces no kernel plan: a single-rank
// communicator, and a zero-count collective between real ones.
TEST_F(GraphCaptureStreamOrdering, CallsWithoutKernelPlan)
{
    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        parallelOptions(),
        isolated("SingleRankComm", [] {
            Harness h;
            ASSERT_NO_FATAL_FAILURE(initHarness(h, 1));
            if(h.nRanks == 0) return;
            Graphs g;
            ASSERT_NO_FATAL_FAILURE(capture(h, hipStreamCaptureModeThreadLocal, [&] {
                fillOp(h, &Harness::a, &Harness::s, [](int) { return 7; });
                fillOp(h, &Harness::b, &Harness::s, [](int) { return 0; });
                EXPECT_EQ(ncclAllReduce(h.a[0], h.b[0], kCount, ncclUint32, ncclSum, h.comms[0], h.s[0]),
                          ncclSuccess);
                addOp(h, &Harness::b, &Harness::s, 1);
                EXPECT_EQ(ncclAllReduce(h.b[0], h.b[0], kCount, ncclUint32, ncclSum, h.comms[0], h.s[0]),
                          ncclSuccess);
                addOp(h, &Harness::b, &Harness::s, 1);
            }, g));
            GraphShape shape;
            ASSERT_NO_FATAL_FAILURE(analyzeRank(g, 0, shape));
            // Everything was issued on one stream, so the graph must be a total order that
            // ends with the two adds, whatever the single-rank copy turned into.
            std::string order = shape.order();
            EXPECT_EQ(shape.count('U'), 2);
            EXPECT_TRUE(shape.allBefore('M', 'U'));
            EXPECT_EQ(order.substr(order.size() < 2 ? 0 : order.size() - 2), "UU") << order;
            expectBoundedDependencies(shape, "rank 0");
            ASSERT_NO_FATAL_FAILURE(launchGraphs(h, g, &Harness::s));
            ASSERT_NO_FATAL_FAILURE(syncAll(h));
            expectBuffer(h, &Harness::b, 9, "b");
        }),
        isolated("ZeroCount", [] {
            Harness h;
            ASSERT_NO_FATAL_FAILURE(initHarness(h, kRanks));
            if(h.nRanks == 0) return;
            Graphs g;
            ASSERT_NO_FATAL_FAILURE(capture(h, hipStreamCaptureModeThreadLocal, [&] {
                fillOp(h, &Harness::a, &Harness::s, [](int r) { return r + 1; });
                allReduceOp(h, &Harness::a, &Harness::s, 0);
                addOp(h, &Harness::a, &Harness::s, 1); // 2, 3
                allReduceOp(h, &Harness::a, &Harness::s, 0);
                allReduceOp(h, &Harness::a, &Harness::s); // 5
                addOp(h, &Harness::a, &Harness::s, 1);    // 6
            }, g));
            for(int r = 0; r < h.nRanks; r++)
            {
                GraphShape shape;
                ASSERT_NO_FATAL_FAILURE(analyzeRank(g, r, shape));
                EXPECT_EQ(collapse(shape.order()), "MUCU") << "rank " << r;
                expectBoundedDependencies(shape, "rank " + std::to_string(r));
            }
            ASSERT_NO_FATAL_FAILURE(launchGraphs(h, g, &Harness::s));
            ASSERT_NO_FATAL_FAILURE(syncAll(h));
            expectBuffer(h, &Harness::a, 6, "a");
        }));
}

} // namespace RcclUnitTesting
