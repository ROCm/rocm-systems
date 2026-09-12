/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Helper executable for Unit_hipGraphNodeSetEnabled_Functional_DependencyOrdering.
//
// A disabled node is documented to behave as an empty node until it is
// reenabled, and an empty node performs no work but still orders its
// predecessors ahead of its successors.  This checks that a disabled node does
// the same when the edge crosses a stream boundary, which is the case the graph
// code has to track explicitly rather than getting for free from a stream's
// in-order execution.
//
// It lives in its own executable because only the classic graph execution path
// is affected, and selecting that path means setting DEBUG_HIP_GRAPH_CLASSIC_PATH
// before the runtime reads its flags, which a test case running inside an
// already initialised process cannot do.  The parent spawns this with the
// variable set, in the same way unit/env spawns hipSetEnv_helper for
// GPU_ENABLE_PAL.
//
// Usage: hipGraphNodeSetEnabledOrdering_Exe <shape> <variant> [reps]
//
//   shapes    fanout  many chains, the disabled node's successor on a foreign
//                     stream, which catches a node that reports no commands
//             inedge  one chain, the disabled node's predecessor on a foreign
//                     stream, which catches a node whose commands are reported
//                     but never submitted
//   variants  empty | kernel | memcpy | enabled
//
// Exit status: 0 no ordering violation, 1 violations observed, 2 usage or API
// failure.  Both shapes are correct on either execution path; they can only
// fail on the classic one.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr)                                                                                \
  do {                                                                                             \
    hipError_t err = (expr);                                                                       \
    if (err != hipSuccess) {                                                                       \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,                         \
                   hipGetErrorString(err));                                                        \
      return -1;                                                                                   \
    }                                                                                              \
  } while (0)

namespace {

constexpr int kChains = 6;
constexpr long kSliceInts = 2L << 20;
constexpr int kPasses = 20;

// Writes -1, -2, ... into the slice and only 1 on the final pass, so a reader
// that overlaps this kernel sees a value it can tell apart from the finished
// state rather than racing on a single store.
__global__ void slowFill(int* buf, long n, int passes) {
  long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (int p = 0; p < passes; ++p) {
    int v = (p == passes - 1) ? 1 : -(p + 1);
    for (long j = i; j < n; j += stride) buf[j] = v;
  }
}

__global__ void countNotOne(const int* buf, long n, int* errs) {
  long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  long stride = static_cast<long>(gridDim.x) * blockDim.x;
  int local = 0;
  for (long j = i; j < n; j += stride)
    if (buf[j] != 1) ++local;
  if (local) atomicAdd(errs, local);
}

__global__ void touch(int* p) { *p = *p + 0; }

enum class Variant { Empty, Kernel, Memcpy, Enabled };

// The node under test.  A memset node cannot appear here: a disabled memset
// keeps a real command, so its edges are enforced whatever the graph tracks,
// and there is no ordering for this to observe.
// side is the address of the caller's pointer variable, not the device pointer:
// a kernel node's parameter list holds a pointer to each argument's value, and
// that value has to outlive the launches.
int addVariantNode(hipGraphNode_t* node, hipGraph_t graph, Variant variant, int** side,
                   int* scratch, hipKernelNodeParams* params, void** args) {
  switch (variant) {
    case Variant::Empty:
      CHECK(hipGraphAddEmptyNode(node, graph, nullptr, 0));
      return 0;
    case Variant::Kernel:
      *args = side;
      *params = {};
      params->func = reinterpret_cast<void*>(touch);
      params->gridDim = dim3(1);
      params->blockDim = dim3(1);
      params->kernelParams = args;
      CHECK(hipGraphAddKernelNode(node, graph, nullptr, 0, params));
      return 0;
    case Variant::Memcpy:
    case Variant::Enabled:
      CHECK(hipGraphAddMemcpyNode1D(node, graph, nullptr, 0, scratch, scratch + 1, sizeof(int),
                                    hipMemcpyDeviceToDevice));
      return 0;
  }
  return -1;
}

// kChains independent copies of
//
//     S_i --------------------> R_i     reads slice i, counts elements != 1
//     W_i --> D_i(variant) --> /        writes 1 into slice i
//
// Three properties of this graph are load bearing.  kChains must not be a
// multiple of the graph stream pool size (DEBUG_HIP_FORCE_GRAPH_QUEUES, default
// 4), or D_i and R_i share a stream and that stream's in-order execution
// enforces the edge whether or not the graph tracked it.  Every S_i is added
// before any W_i, so the topological walk reaches R_i from its S_i root and
// assigns its stream there.  And the writer is narrow while the reader is wide,
// since a writer sized to fill the device would hold every CU for its duration
// and even a completely unordered reader could not overlap it.
int runFanout(Variant variant, int reps, int* violations) {
  const size_t bytes = static_cast<size_t>(kSliceInts) * kChains * sizeof(int);
  hipStream_t stream;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int *buf = nullptr, *errs = nullptr, *side = nullptr, *scratch = nullptr;

  CHECK(hipStreamCreate(&stream));
  CHECK(hipMalloc(&buf, bytes));
  CHECK(hipMalloc(&errs, sizeof(int)));
  CHECK(hipMalloc(&side, sizeof(int)));
  CHECK(hipMalloc(&scratch, 2 * sizeof(int)));
  CHECK(hipMemset(side, 0, sizeof(int)));
  CHECK(hipMemset(scratch, 0, 2 * sizeof(int)));
  CHECK(hipGraphCreate(&graph, 0));

  std::vector<hipGraphNode_t> S(kChains), W(kChains), D(kChains), R(kChains);
  std::vector<int*> slice(kChains);
  std::vector<long> nArg(kChains, kSliceInts);
  std::vector<int> passesArg(kChains, kPasses);
  for (int i = 0; i < kChains; ++i) slice[i] = buf + static_cast<size_t>(i) * kSliceInts;

  std::vector<void*> sArgs(kChains);
  std::vector<hipKernelNodeParams> sParams(kChains);
  for (int i = 0; i < kChains; ++i) {
    sArgs[i] = &side;
    sParams[i] = {};
    sParams[i].func = reinterpret_cast<void*>(touch);
    sParams[i].gridDim = dim3(1);
    sParams[i].blockDim = dim3(1);
    sParams[i].kernelParams = &sArgs[i];
    CHECK(hipGraphAddKernelNode(&S[i], graph, nullptr, 0, &sParams[i]));
  }

  std::vector<std::vector<void*>> wArgs(kChains);
  std::vector<hipKernelNodeParams> wParams(kChains);
  for (int i = 0; i < kChains; ++i) {
    wArgs[i] = {&slice[i], &nArg[i], &passesArg[i]};
    wParams[i] = {};
    wParams[i].func = reinterpret_cast<void*>(slowFill);
    wParams[i].gridDim = dim3(4);
    wParams[i].blockDim = dim3(64);
    wParams[i].kernelParams = wArgs[i].data();
    CHECK(hipGraphAddKernelNode(&W[i], graph, nullptr, 0, &wParams[i]));
  }

  std::vector<void*> dArgs(kChains);
  std::vector<hipKernelNodeParams> dParams(kChains);
  for (int i = 0; i < kChains; ++i)
    if (addVariantNode(&D[i], graph, variant, &side, scratch, &dParams[i], &dArgs[i]) != 0) return -1;

  std::vector<std::vector<void*>> rArgs(kChains);
  std::vector<hipKernelNodeParams> rParams(kChains);
  for (int i = 0; i < kChains; ++i) {
    rArgs[i] = {&slice[i], &nArg[i], &errs};
    rParams[i] = {};
    rParams[i].func = reinterpret_cast<void*>(countNotOne);
    rParams[i].gridDim = dim3(128);
    rParams[i].blockDim = dim3(256);
    rParams[i].kernelParams = rArgs[i].data();
    CHECK(hipGraphAddKernelNode(&R[i], graph, nullptr, 0, &rParams[i]));
  }

  for (int i = 0; i < kChains; ++i) {
    CHECK(hipGraphAddDependencies(graph, &S[i], &R[i], 1));
    CHECK(hipGraphAddDependencies(graph, &W[i], &D[i], 1));
    CHECK(hipGraphAddDependencies(graph, &D[i], &R[i], 1));
  }

  CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  if (variant == Variant::Kernel || variant == Variant::Memcpy) {
    for (int i = 0; i < kChains; ++i) {
      unsigned int enabled = 1;
      CHECK(hipGraphNodeSetEnabled(graphExec, D[i], 0));
      CHECK(hipGraphNodeGetEnabled(graphExec, D[i], &enabled));
      if (enabled != 0) {
        std::fprintf(stderr, "node %d did not report itself disabled\n", i);
        return -1;
      }
    }
  }

  int bad = 0;
  for (int rep = 0; rep < reps; ++rep) {
    CHECK(hipMemset(buf, 0, bytes));
    CHECK(hipMemset(errs, 0, sizeof(int)));
    CHECK(hipDeviceSynchronize());

    CHECK(hipGraphLaunch(graphExec, stream));
    CHECK(hipStreamSynchronize(stream));

    int hostErrs = 0;
    CHECK(hipMemcpy(&hostErrs, errs, sizeof(int), hipMemcpyDeviceToHost));
    if (hostErrs != 0) ++bad;
  }

  CHECK(hipGraphExecDestroy(graphExec));
  CHECK(hipGraphDestroy(graph));
  CHECK(hipFree(buf));
  CHECK(hipFree(errs));
  CHECK(hipFree(side));
  CHECK(hipFree(scratch));
  CHECK(hipStreamDestroy(stream));
  *violations = bad;
  return 0;
}

// One chain, arranged so the disabled node's *predecessor* is the cross-stream
// edge:
//
//     S --------------------> R     fast root, wide reader          stream 0
//     T ----------> D -------/      fast root, node under test      stream 1
//     W -----------/                the one slow writer             stream 2
//
// Node creation order is S, T, W, then D, R.  Roots are handed streams round
// robin in creation order and a non-root takes the stream of the root whose
// branch reaches it first, so D lands on T's stream rather than inheriting W's,
// leaving W -> D cross-stream.  The fanout shape cannot express this, because it
// reaches its disabled node by walking W_i -> D_i.
//
// The graph holds exactly one slow kernel, and R's stream does not hold it, so
// nothing but the W -> D -> R chain can order R after W.  Both parts matter: a
// second writer of similar duration on R's stream would mask a lost edge by
// timing, since waiting for that one leaves W finished as well.
int runInedge(Variant variant, int reps, int* violations) {
  const size_t bytes = static_cast<size_t>(kSliceInts) * sizeof(int);
  hipStream_t stream;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int *buf = nullptr, *errs = nullptr, *side = nullptr, *scratch = nullptr;

  CHECK(hipStreamCreate(&stream));
  CHECK(hipMalloc(&buf, bytes));
  CHECK(hipMalloc(&errs, sizeof(int)));
  CHECK(hipMalloc(&side, sizeof(int)));
  CHECK(hipMalloc(&scratch, 2 * sizeof(int)));
  CHECK(hipMemset(side, 0, sizeof(int)));
  CHECK(hipMemset(scratch, 0, 2 * sizeof(int)));
  CHECK(hipGraphCreate(&graph, 0));

  long nArg = kSliceInts;
  int passesArg = kPasses;

  void* sArgs = &side;
  hipKernelNodeParams fastParams{};
  fastParams.func = reinterpret_cast<void*>(touch);
  fastParams.gridDim = dim3(1);
  fastParams.blockDim = dim3(1);
  fastParams.kernelParams = &sArgs;

  hipGraphNode_t S, T, W, D, R;
  CHECK(hipGraphAddKernelNode(&S, graph, nullptr, 0, &fastParams));
  CHECK(hipGraphAddKernelNode(&T, graph, nullptr, 0, &fastParams));

  void* wArgs[] = {&buf, &nArg, &passesArg};
  hipKernelNodeParams wParams{};
  wParams.func = reinterpret_cast<void*>(slowFill);
  wParams.gridDim = dim3(4);
  wParams.blockDim = dim3(64);
  wParams.kernelParams = wArgs;
  CHECK(hipGraphAddKernelNode(&W, graph, nullptr, 0, &wParams));

  hipKernelNodeParams dParams{};
  void* dArgs = nullptr;
  if (addVariantNode(&D, graph, variant, &side, scratch, &dParams, &dArgs) != 0) return -1;

  void* rArgs[] = {&buf, &nArg, &errs};
  hipKernelNodeParams rParams{};
  rParams.func = reinterpret_cast<void*>(countNotOne);
  rParams.gridDim = dim3(128);
  rParams.blockDim = dim3(256);
  rParams.kernelParams = rArgs;
  CHECK(hipGraphAddKernelNode(&R, graph, nullptr, 0, &rParams));

  CHECK(hipGraphAddDependencies(graph, &S, &R, 1));
  CHECK(hipGraphAddDependencies(graph, &T, &D, 1));
  CHECK(hipGraphAddDependencies(graph, &W, &D, 1));
  CHECK(hipGraphAddDependencies(graph, &D, &R, 1));

  CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  if (variant == Variant::Kernel || variant == Variant::Memcpy) {
    unsigned int enabled = 1;
    CHECK(hipGraphNodeSetEnabled(graphExec, D, 0));
    CHECK(hipGraphNodeGetEnabled(graphExec, D, &enabled));
    if (enabled != 0) {
      std::fprintf(stderr, "node did not report itself disabled\n");
      return -1;
    }
  }

  int bad = 0;
  for (int rep = 0; rep < reps; ++rep) {
    CHECK(hipMemset(buf, 0, bytes));
    CHECK(hipMemset(errs, 0, sizeof(int)));
    CHECK(hipDeviceSynchronize());

    CHECK(hipGraphLaunch(graphExec, stream));
    CHECK(hipStreamSynchronize(stream));

    int hostErrs = 0;
    CHECK(hipMemcpy(&hostErrs, errs, sizeof(int), hipMemcpyDeviceToHost));
    if (hostErrs != 0) ++bad;
  }

  CHECK(hipGraphExecDestroy(graphExec));
  CHECK(hipGraphDestroy(graph));
  CHECK(hipFree(buf));
  CHECK(hipFree(errs));
  CHECK(hipFree(side));
  CHECK(hipFree(scratch));
  CHECK(hipStreamDestroy(stream));
  *violations = bad;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <fanout|inedge> <empty|kernel|memcpy|enabled> [reps]\n",
                 argv[0]);
    return 2;
  }
  const std::string shape = argv[1];
  const std::string variantName = argv[2];
  const int reps = argc > 3 ? std::atoi(argv[3]) : 20;

  Variant variant;
  if (variantName == "empty") {
    variant = Variant::Empty;
  } else if (variantName == "kernel") {
    variant = Variant::Kernel;
  } else if (variantName == "memcpy") {
    variant = Variant::Memcpy;
  } else if (variantName == "enabled") {
    variant = Variant::Enabled;
  } else {
    std::fprintf(stderr, "unknown variant '%s'\n", variantName.c_str());
    return 2;
  }

  int violations = 0;
  int status;
  if (shape == "fanout") {
    status = runFanout(variant, reps, &violations);
  } else if (shape == "inedge") {
    status = runInedge(variant, reps, &violations);
  } else {
    std::fprintf(stderr, "unknown shape '%s'\n", shape.c_str());
    return 2;
  }
  if (status != 0) return 2;

  std::printf("shape=%s variant=%s violations=%d of %d : %s\n", shape.c_str(),
              variantName.c_str(), violations, reps, violations ? "FAIL" : "PASS");
  return violations ? 1 : 0;
}
