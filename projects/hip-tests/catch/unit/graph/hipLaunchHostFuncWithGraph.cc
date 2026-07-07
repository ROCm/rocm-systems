/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include "stream_capture_common.hh"

/**
 * @addtogroup hipLaunchHostFunc hipLaunchHostFunc
 * @{
 * @ingroup GraphTest
 * `hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void *userData)` -
 * enqueues a host function call in a stream
 */

#if HT_NVIDIA
static void hostNodeCallbackDummy(void* data) { REQUIRE(data == nullptr); }
#endif

static void hostNodeCallback(void* data) {
  float** userData = static_cast<float**>(data);

  float input_data = *(userData[0]);
  float output_data = *(userData[1]);
  REQUIRE(input_data == output_data);
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify API behavior with invalid arguments:
 *        -# Stream is legacy/nullptr stream
 *        -# Function is nullptr
 *        -# Stream is uninitialized
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipLaunchHostFunc.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
HIP_TEST_CASE(Unit_hipLaunchHostFunc_Negative_Parameters) {
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();
  hipGraph_t graph{nullptr};
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
#if HT_NVIDIA  // EXSWHTEC-228
  SECTION("Pass stream as nullptr") {
    hipHostFn_t fn = hostNodeCallbackDummy;
    HIP_CHECK_ERROR(hipLaunchHostFunc(nullptr, fn, nullptr), hipErrorStreamCaptureImplicit);
  }
#endif
  SECTION("Pass functions as nullptr") {
    HIP_CHECK_ERROR(hipLaunchHostFunc(stream, nullptr, nullptr), hipErrorInvalidValue);
  }
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify enquing a host function into a stream, which checks if
 * the captured computation result is correct
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipLaunchHostFunc.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
HIP_TEST_CASE(Unit_hipLaunchHostFunc_Positive_Functional) {
  LinearAllocGuard<float> A_h(LinearAllocs::malloc, sizeof(float));
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, sizeof(float));
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, sizeof(float));

  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = hipStreamCaptureModeGlobal;

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  captureSequenceSimple(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), 1, stream);

  hipHostFn_t fn = hostNodeCallback;
  float* data[2] = {A_h.host_ptr(), B_h.host_ptr()};
  HIP_CHECK(hipLaunchHostFunc(stream, fn, static_cast<void*>(data)));

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  // Validate end capture is successful
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Replay the recorded sequence multiple times
  for (size_t i = 0; i < kLaunchIters; i++) {
    std::fill_n(A_h.host_ptr(), 1, static_cast<float>(i));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    ArrayFindIfNot(B_h.host_ptr(), static_cast<float>(i), 1);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

static void thread_func_pos(hipStream_t* stream, hipHostFn_t fn, float** data){

    HIP_CHECK(hipLaunchHostFunc(*stream, fn, static_cast<void*>(data)))}

/**
 * Test Description
 * ------------------------
 *    - Test to verify enquing a host function into a stream on a different
 * thread, which checks if the captured computation result is correct
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipLaunchHostFunc.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
HIP_TEST_CASE(Unit_hipLaunchHostFunc_Positive_Thread) {
  LinearAllocGuard<float> A_h(LinearAllocs::malloc, sizeof(float));
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, sizeof(float));
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, sizeof(float));

  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = hipStreamCaptureModeGlobal;

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  captureSequenceSimple(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), 1, stream);

  hipHostFn_t fn = hostNodeCallback;
  float* data[2] = {A_h.host_ptr(), B_h.host_ptr()};
  std::thread t(thread_func_pos, &stream, fn, data);
  t.join();

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  // Validate end capture is successful
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Replay the recorded sequence multiple times
  for (size_t i = 0; i < kLaunchIters; i++) {
    std::fill_n(A_h.host_ptr(), 1, static_cast<float>(i));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    ArrayFindIfNot(B_h.host_ptr(), static_cast<float>(i), 1);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}
namespace {
__global__ void kernelA(double* arrayA, size_t size) {
  const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  if (x < size) {
    arrayA[x] *= 2.0;
  }
}

struct set_vector_args {
  std::vector<double>& h_array;
  double value;
};

static void set_vector(void* args) {
  set_vector_args h_args{*(reinterpret_cast<set_vector_args*>(args))};
  std::vector<double>& vec{h_args.h_array};
  vec.assign(vec.size(), h_args.value);
}
}  // namespace

HIP_TEST_CASE(Unit_hipLaunchHostFunc_H2D_Kernel_D2H_Capture) {
  constexpr int numOfBlocks = 1024;
  constexpr int threadsPerBlock = 1024;
  constexpr size_t arraySize = 1U << 20;  // 1,048,576
  constexpr double initValue = 2.0;

  double* d_arrayA = nullptr;
  std::vector<double> h_array(arraySize);

  hipStream_t captureStream{};
  HIP_CHECK(hipStreamCreate(&captureStream));

  // Begin stream capture
  HIP_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));

  // Device alloc (async so it belongs to the captured stream)
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&d_arrayA), arraySize * sizeof(double),
                           captureStream));

  // Initialize host data via a host function in the stream
  set_vector_args args{h_array, initValue};
  HIP_CHECK(hipLaunchHostFunc(captureStream, set_vector, &args));

  // HtoD copy
  HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array.data(), arraySize * sizeof(double),
                           hipMemcpyHostToDevice, captureStream));

  // KernelA only
  kernelA<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, arraySize);
  HIP_CHECK(hipGetLastError());

  // DtoH copy
  HIP_CHECK(hipMemcpyAsync(h_array.data(), d_arrayA, arraySize * sizeof(double),
                           hipMemcpyDeviceToHost, captureStream));

  // Free device memory inside the graph
  HIP_CHECK(hipFreeAsync(d_arrayA, captureStream));

  // End capture -> graph
  hipGraph_t graph{};
  HIP_CHECK(hipStreamEndCapture(captureStream, &graph));

  // Instantiate and launch
  hipGraphExec_t graphExec{};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphLaunch(graphExec, captureStream));
  HIP_CHECK(hipStreamSynchronize(captureStream));

  // Validate: each element should be initValue * 2.0
  const double expected = initValue * 2.0;
  for (size_t i = 0; i < arraySize; ++i) {
    REQUIRE(h_array[i] == expected);
  }

  // Cleanup
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(captureStream));
}


namespace {
__global__ void spin_then_set(int* flag, long long spin_ticks) {
#if HT_NVIDIA
  long long start = clock64();
  while ((clock64() - start) < spin_ticks) { /* spin */ }
#else
  unsigned long long start = wall_clock64();
  while ((wall_clock64() - start) < static_cast<unsigned long long>(spin_ticks)) {
    __builtin_amdgcn_s_sleep(10);
  }
#endif
  *flag = 1;
}
__global__ void noop() {}

constexpr int kSdmaCopyElements = 1 << 20;
constexpr size_t kSdmaCopyBytes = kSdmaCopyElements * sizeof(int);
}  // namespace

/**
 * Test Description
 * ------------------------
 *    - Regression test: leading-PacketBatch cross-dep barrier must be dispatched
 *      before an uncaptured first node in a consumer segment.
 *
 *      Graph (direct API):
 *
 *        K_dummy  [root, stream 0]  ─────────────────────────┐
 *        K_prod   [root, stream 1]  spin then set *d_flag=1  │
 *        DtoH     [SDMA, seg1]      d_flag -> h_flag           │
 *                                        │ cross-stream dep  │ join
 *                                        ▼                   ▼
 *                         host_node  [UNCAPTURED, JOIN] reads h_flag[0] -> h_cb_saw
 *                         DtoH       [captured]         d_flag → h_out
 *
 *      K_dummy forces 2 level-0 segments → 2 HW queues.  K_prod runs on HW
 *      queue 1; the consumer segment on HW queue 0.  host_node is always
 *      non-capturable (hipGraphAddHostNode).  BuildSyncPlan parks the cross-dep
 *      BARRIER_AND in the leading empty batch[0] of the consumer segment.
 *
 *      With the fix:    batch[0] dispatched before host_node markers → callback
 *                       runs after K_prod's SDMA DtoH is visible -> h_cb_saw==1
 *      Without the fix: callback can observe stale host memory -> h_cb_saw==0
 *
 * Test source
 * ------------------------
 *    - catch/unit/graph/hipLaunchHostFuncWithGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
static void RunCrossStreamDepUncapturedFirstNode(bool use_sdma_dtoh) {
  int ticks_per_ms = 0;
#if HT_NVIDIA
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeClockRate, 0));
#else
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeWallClockRate, 0));
#endif
  if (ticks_per_ms == 0) {
    ticks_per_ms = 1000;
  }
  const long long kSpin = static_cast<long long>(ticks_per_ms) * 500;
  constexpr int kExpected = 1;

  std::vector<int> h_flag_sdma;
  int* h_flag_pinned{};
  int* h_flag{};
  size_t copy_bytes = sizeof(int);
  if (use_sdma_dtoh) {
    h_flag_sdma.resize(kSdmaCopyElements);
    h_flag = h_flag_sdma.data();
    copy_bytes = kSdmaCopyBytes;
  } else {
    HIP_CHECK(hipHostMalloc(&h_flag_pinned, sizeof(int), hipHostMallocDefault));
    h_flag = h_flag_pinned;
  }

  int* h_cb_saw{};
  int* h_out{};
  HIP_CHECK(hipHostMalloc(&h_cb_saw, sizeof(int), hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&h_out, sizeof(int), hipHostMallocDefault));

  int* d_flag{};
  HIP_CHECK(hipMalloc(&d_flag, copy_bytes));

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t dummy_node{};
  {
    hipKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(noop);
    p.gridDim = dim3(1);
    p.blockDim = dim3(1);
    HIP_CHECK(hipGraphAddKernelNode(&dummy_node, graph, nullptr, 0, &p));
  }

  hipGraphNode_t prod_node{};
  {
    long long spin = kSpin;
    void* args[] = {&d_flag, &spin};
    hipKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(spin_then_set);
    p.gridDim = dim3(1);
    p.blockDim = dim3(1);
    p.kernelParams = args;
    HIP_CHECK(hipGraphAddKernelNode(&prod_node, graph, nullptr, 0, &p));
  }

  hipGraphNode_t dtoh_flag_node{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&dtoh_flag_node, graph, &prod_node, 1,
                                    h_flag, d_flag, copy_bytes, hipMemcpyDeviceToHost));

  struct CbArgs {
    int* h_flag;
    int* h_cb_saw;
  };
  CbArgs cb_args{h_flag, h_cb_saw};
  hipGraphNode_t host_node{};
  hipHostNodeParams host_params{};
  host_params.fn = [](void* ud) {
    auto* a = static_cast<CbArgs*>(ud);
    *a->h_cb_saw = *a->h_flag;
  };
  host_params.userData = &cb_args;
  {
    hipGraphNode_t deps[] = {dummy_node, dtoh_flag_node};
    HIP_CHECK(hipGraphAddHostNode(&host_node, graph, deps, 2, &host_params));
  }

  hipGraphNode_t check_node{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&check_node, graph, &host_node, 1,
                                    h_out, d_flag, sizeof(int), hipMemcpyDeviceToHost));

  hipGraphExec_t graphExec{};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  hipStream_t launch_stream{};
  HIP_CHECK(hipStreamCreate(&launch_stream));

  for (int iter = 0; iter < 10; ++iter) {
    if (use_sdma_dtoh) {
      std::fill(h_flag_sdma.begin(), h_flag_sdma.end(), 0);
    } else {
      *h_flag = 0;
    }
    *h_cb_saw = 0;
    *h_out = 0;
    HIP_CHECK(hipMemset(d_flag, 0, copy_bytes));
    HIP_CHECK(hipGraphLaunch(graphExec, launch_stream));
    HIP_CHECK(hipStreamSynchronize(launch_stream));

    INFO("iter=" << iter << " h_cb_saw=" << *h_cb_saw << " h_out=" << *h_out);
    REQUIRE(*h_cb_saw == kExpected);
    REQUIRE(*h_out == kExpected);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(launch_stream));
  HIP_CHECK(hipFree(d_flag));
  HIP_CHECK(hipHostFree(h_out));
  HIP_CHECK(hipHostFree(h_cb_saw));
  if (h_flag_pinned != nullptr) {
    HIP_CHECK(hipHostFree(h_flag_pinned));
  }
}

HIP_TEST_CASE(Unit_hipLaunchHostFunc_CrossStreamDep_UncapturedFirstNode) {
  RunCrossStreamDepUncapturedFirstNode(false);
}

HIP_TEST_CASE(Unit_hipLaunchHostFunc_CrossStreamDepSdmaDtoH_UncapturedFirstNode) {
  RunCrossStreamDepUncapturedFirstNode(true);
}

HIP_TEST_CASE(Unit_hipLaunchHostFunc_BlitDtoHHostCallbackVisibility) {
  int ticks_per_ms = 0;
#if HT_NVIDIA
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeClockRate, 0));
#else
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeWallClockRate, 0));
#endif
  if (ticks_per_ms == 0) {
    ticks_per_ms = 1000;
  }
  const long long kSpin = static_cast<long long>(ticks_per_ms) * 500;
  constexpr int kExpected = 1;

  int* h_flag{};
  int* h_cb_first{};
  int* h_cb_last{};
  int* h_out{};
  HIP_CHECK(hipHostMalloc(&h_flag, sizeof(int), hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&h_cb_first, sizeof(int), hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&h_cb_last, sizeof(int), hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&h_out, sizeof(int), hipHostMallocDefault));

  int* d_flag{};
  HIP_CHECK(hipMalloc(&d_flag, sizeof(int)));

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t dummy_node{};
  {
    hipKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(noop);
    p.gridDim = dim3(1);
    p.blockDim = dim3(1);
    HIP_CHECK(hipGraphAddKernelNode(&dummy_node, graph, nullptr, 0, &p));
  }

  hipGraphNode_t prod_node{};
  {
    long long spin = kSpin;
    void* args[] = {&d_flag, &spin};
    hipKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(spin_then_set);
    p.gridDim = dim3(1);
    p.blockDim = dim3(1);
    p.kernelParams = args;
    HIP_CHECK(hipGraphAddKernelNode(&prod_node, graph, nullptr, 0, &p));
  }

  hipGraphNode_t dtoh_flag_node{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&dtoh_flag_node, graph, &prod_node, 1,
                                    h_flag, d_flag, sizeof(int), hipMemcpyDeviceToHost));

  struct CbArgs {
    int* h_flag;
    int* h_cb_first;
    int* h_cb_last;
  };
  CbArgs cb_args{h_flag, h_cb_first, h_cb_last};
  hipGraphNode_t host_node{};
  hipHostNodeParams host_params{};
  host_params.fn = [](void* ud) {
    auto* a = static_cast<CbArgs*>(ud);
    auto* flag = reinterpret_cast<volatile int*>(a->h_flag);
    int last = *flag;
    *a->h_cb_first = last;
    for (int spin = 0; spin < 1000000 && last != 1; ++spin) {
      last = *flag;
    }
    *a->h_cb_last = last;
  };
  host_params.userData = &cb_args;
  {
    hipGraphNode_t deps[] = {dummy_node, dtoh_flag_node};
    HIP_CHECK(hipGraphAddHostNode(&host_node, graph, deps, 2, &host_params));
  }

  hipGraphNode_t check_node{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&check_node, graph, &host_node, 1,
                                    h_out, d_flag, sizeof(int), hipMemcpyDeviceToHost));

  hipGraphExec_t graphExec{};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  hipStream_t launch_stream{};
  HIP_CHECK(hipStreamCreate(&launch_stream));

  for (int iter = 0; iter < 10; ++iter) {
    *h_flag = 0;
    *h_cb_first = 0;
    *h_cb_last = 0;
    *h_out = 0;
    HIP_CHECK(hipMemset(d_flag, 0, sizeof(int)));
    HIP_CHECK(hipGraphLaunch(graphExec, launch_stream));
    HIP_CHECK(hipStreamSynchronize(launch_stream));

    const int after_sync_h_flag = *h_flag;
    INFO("iter=" << iter << " h_cb_first=" << *h_cb_first
                 << " h_cb_last=" << *h_cb_last
                 << " after_sync_h_flag=" << after_sync_h_flag
                 << " h_out=" << *h_out);
    REQUIRE(*h_out == kExpected);
    REQUIRE(after_sync_h_flag == kExpected);
    REQUIRE(*h_cb_first == kExpected);
    REQUIRE(*h_cb_last == kExpected);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(launch_stream));
  HIP_CHECK(hipFree(d_flag));
  HIP_CHECK(hipHostFree(h_out));
  HIP_CHECK(hipHostFree(h_cb_last));
  HIP_CHECK(hipHostFree(h_cb_first));
  HIP_CHECK(hipHostFree(h_flag));
}

HIP_TEST_CASE(Unit_hipLaunchHostFunc_SdmaDtoHHostCallbackVisibilityNoProducer) {
  constexpr int kExpected = 1;

  std::vector<int> h_src(kSdmaCopyElements, kExpected);
  std::vector<int> h_flag(kSdmaCopyElements, 0);
  int* h_cb_first{};
  int* h_cb_last{};
  int* h_out{};
  HIP_CHECK(hipHostMalloc(&h_cb_first, sizeof(int), hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&h_cb_last, sizeof(int), hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&h_out, sizeof(int), hipHostMallocDefault));

  int* d_flag{};
  HIP_CHECK(hipMalloc(&d_flag, kSdmaCopyBytes));

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t dtoh_flag_node{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&dtoh_flag_node, graph, nullptr, 0,
                                    h_flag.data(), d_flag, kSdmaCopyBytes,
                                    hipMemcpyDeviceToHost));

  struct CbArgs {
    int* h_flag;
    int* h_cb_first;
    int* h_cb_last;
  };
  CbArgs cb_args{h_flag.data(), h_cb_first, h_cb_last};
  hipGraphNode_t host_node{};
  hipHostNodeParams host_params{};
  host_params.fn = [](void* ud) {
    auto* a = static_cast<CbArgs*>(ud);
    auto* flag = reinterpret_cast<volatile int*>(a->h_flag);
    int last = *flag;
    *a->h_cb_first = last;
    for (int spin = 0; spin < 1000000 && last != 1; ++spin) {
      last = *flag;
    }
    *a->h_cb_last = last;
  };
  host_params.userData = &cb_args;
  HIP_CHECK(hipGraphAddHostNode(&host_node, graph, &dtoh_flag_node, 1, &host_params));

  hipGraphNode_t check_node{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&check_node, graph, &host_node, 1,
                                    h_out, d_flag, sizeof(int), hipMemcpyDeviceToHost));

  hipGraphExec_t graphExec{};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  hipStream_t launch_stream{};
  HIP_CHECK(hipStreamCreate(&launch_stream));

  for (int iter = 0; iter < 10; ++iter) {
    std::fill(h_flag.begin(), h_flag.end(), 0);
    *h_cb_first = 0;
    *h_cb_last = 0;
    *h_out = 0;
    HIP_CHECK(hipMemcpy(d_flag, h_src.data(), kSdmaCopyBytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphLaunch(graphExec, launch_stream));
    HIP_CHECK(hipStreamSynchronize(launch_stream));

    const int after_sync_h_flag = h_flag[0];
    INFO("iter=" << iter << " h_cb_first=" << *h_cb_first
                 << " h_cb_last=" << *h_cb_last
                 << " after_sync_h_flag=" << after_sync_h_flag
                 << " h_out=" << *h_out);
    REQUIRE(*h_out == kExpected);
    REQUIRE(after_sync_h_flag == kExpected);
    REQUIRE(*h_cb_first == kExpected);
    REQUIRE(*h_cb_last == kExpected);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(launch_stream));
  HIP_CHECK(hipFree(d_flag));
  HIP_CHECK(hipHostFree(h_out));
  HIP_CHECK(hipHostFree(h_cb_last));
  HIP_CHECK(hipHostFree(h_cb_first));
}



/**
 * Test Description
 * ------------------------
 *    - Regression test: an uncaptured host node that immediately follows a
 *      kernel -> SDMA DtoH handoff in the same graph segment must see host
 *      memory written by that SDMA copy.
 *
 *      Graph (direct API, single path / same stream):
 *
 *        K_prod -> SDMA DtoH(d_flag -> h_flag) -> host_node(reads h_flag[0]) -> DtoH sanity
 *
 *      The host node has a single dependency, so it remains on the same
 *      graph segment and stream. This covers the same-stream path that the
 *      cross-stream host-first-consumer test does not exercise.
 *
 * Test source
 * ------------------------
 *    - catch/unit/graph/hipLaunchHostFuncWithGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
HIP_TEST_CASE(Unit_hipLaunchHostFunc_SameStreamSdmaDtoH_UncapturedHost) {
  int ticks_per_ms = 0;
#if HT_NVIDIA
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeClockRate, 0));
#else
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeWallClockRate, 0));
#endif
  if (ticks_per_ms == 0) {
    ticks_per_ms = 1000;
  }
  const long long kSpin = static_cast<long long>(ticks_per_ms) * 50;  // ~50 ms
  constexpr int kExpected = 1;

  std::vector<int> h_flag(kSdmaCopyElements);
  int* h_cb_saw{};
  int* h_out{};
  HIP_CHECK(hipHostMalloc(&h_cb_saw, sizeof(int), hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&h_out, sizeof(int), hipHostMallocDefault));

  int* d_flag{};
  HIP_CHECK(hipMalloc(&d_flag, kSdmaCopyBytes));

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t prod_node{};
  {
    long long spin = kSpin;
    void* args[] = {&d_flag, &spin};
    hipKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(spin_then_set);
    p.gridDim = dim3(1);
    p.blockDim = dim3(1);
    p.kernelParams = args;
    HIP_CHECK(hipGraphAddKernelNode(&prod_node, graph, nullptr, 0, &p));
  }

  hipGraphNode_t dtoh_flag_node{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&dtoh_flag_node, graph, &prod_node, 1,
                                    h_flag.data(), d_flag, kSdmaCopyBytes,
                                    hipMemcpyDeviceToHost));

  struct CbArgs {
    int* h_flag;
    int* h_cb_saw;
  };
  CbArgs cb_args{h_flag.data(), h_cb_saw};
  hipGraphNode_t host_node{};
  hipHostNodeParams host_params{};
  host_params.fn = [](void* ud) {
    auto* a = static_cast<CbArgs*>(ud);
    *a->h_cb_saw = *a->h_flag;
  };
  host_params.userData = &cb_args;
  HIP_CHECK(hipGraphAddHostNode(&host_node, graph, &dtoh_flag_node, 1, &host_params));

  hipGraphNode_t check_node{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&check_node, graph, &host_node, 1,
                                    h_out, d_flag, sizeof(int), hipMemcpyDeviceToHost));

  hipGraphExec_t graphExec{};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  hipStream_t launch_stream{};
  HIP_CHECK(hipStreamCreate(&launch_stream));

  for (int iter = 0; iter < 10; ++iter) {
    std::fill(h_flag.begin(), h_flag.end(), 0);
    *h_cb_saw = 0;
    *h_out = 0;
    HIP_CHECK(hipMemset(d_flag, 0, kSdmaCopyBytes));
    HIP_CHECK(hipGraphLaunch(graphExec, launch_stream));
    HIP_CHECK(hipStreamSynchronize(launch_stream));

    INFO("iter=" << iter << " h_cb_saw=" << *h_cb_saw << " h_out=" << *h_out);
    REQUIRE(*h_cb_saw == kExpected);
    REQUIRE(*h_out == kExpected);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(launch_stream));
  HIP_CHECK(hipFree(d_flag));
  HIP_CHECK(hipHostFree(h_out));
  HIP_CHECK(hipHostFree(h_cb_saw));
}

/**
 * Test Description
 * ------------------------
 *    - GPU kernel writes to hipMallocManaged memory; host node reads directly.
 *      Compare with Unit_hipLaunchHostFunc_BlitDtoHHostCallbackVisibility which
 *      uses a blit D2H copy. Both require GPU L2 flush on CPX before host read.
 *
 * Test source
 * ------------------------
 *    - catch/unit/graph/hipLaunchHostFuncWithGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
HIP_TEST_CASE(Unit_hipLaunchHostFunc_KernelWritesManagedMem_HostNodeReads) {
  int ticks_per_ms = 0;
#if HT_NVIDIA
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeClockRate, 0));
#else
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeWallClockRate, 0));
#endif
  if (ticks_per_ms == 0) {
    ticks_per_ms = 1000;
  }
  const long long kSpin = static_cast<long long>(ticks_per_ms) * 500;
  constexpr int kExpected = 1;

  int* d_managed{};
  HIP_CHECK(hipMallocManaged(&d_managed, sizeof(int)));

  int* h_cb_first{};
  int* h_cb_last{};
  int* h_out{};
  HIP_CHECK(hipHostMalloc(&h_cb_first, sizeof(int), hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&h_cb_last,  sizeof(int), hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&h_out,      sizeof(int), hipHostMallocDefault));

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t prod_node{};
  {
    long long spin = kSpin;
    void* args[] = {&d_managed, &spin};
    hipKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(spin_then_set);
    p.gridDim = dim3(1);
    p.blockDim = dim3(1);
    p.kernelParams = args;
    HIP_CHECK(hipGraphAddKernelNode(&prod_node, graph, nullptr, 0, &p));
  }

  struct CbArgsMgd {
    int* d_managed;
    int* h_cb_first;
    int* h_cb_last;
  };
  CbArgsMgd cb_args{d_managed, h_cb_first, h_cb_last};
  hipGraphNode_t host_node{};
  hipHostNodeParams host_params{};
  host_params.fn = [](void* ud) {
    auto* a = static_cast<CbArgsMgd*>(ud);
    auto* managed = reinterpret_cast<volatile int*>(a->d_managed);
    int first = *managed;
    *a->h_cb_first = first;
    int last = first;
    for (int spin = 0; spin < 1000000 && last != 1; ++spin) {
      last = *managed;
    }
    *a->h_cb_last = last;
  };
  host_params.userData = &cb_args;
  {
    hipGraphNode_t deps[] = {prod_node};
    HIP_CHECK(hipGraphAddHostNode(&host_node, graph, deps, 1, &host_params));
  }

  hipGraphNode_t check_node{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&check_node, graph, &host_node, 1,
                                    h_out, d_managed, sizeof(int),
                                    hipMemcpyDefault));

  hipGraphExec_t graphExec{};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  hipStream_t launch_stream{};
  HIP_CHECK(hipStreamCreate(&launch_stream));

  for (int iter = 0; iter < 10; ++iter) {
    *d_managed   = 0;
    *h_cb_first  = 0;
    *h_cb_last   = 0;
    *h_out       = 0;
    HIP_CHECK(hipGraphLaunch(graphExec, launch_stream));
    HIP_CHECK(hipStreamSynchronize(launch_stream));

    const int after_sync = *d_managed;
    INFO("iter=" << iter
         << " h_cb_first=" << *h_cb_first
         << " h_cb_last="  << *h_cb_last
         << " after_sync=" << after_sync
         << " h_out="      << *h_out);
    REQUIRE(*h_out      == kExpected);
    REQUIRE(after_sync  == kExpected);
    REQUIRE(*h_cb_first == kExpected);
    REQUIRE(*h_cb_last  == kExpected);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(launch_stream));
  HIP_CHECK(hipFree(d_managed));
  HIP_CHECK(hipHostFree(h_out));
  HIP_CHECK(hipHostFree(h_cb_last));
  HIP_CHECK(hipHostFree(h_cb_first));
}

/**
 * Test Description
 * ------------------------
 *    - GPU kernel writes DIRECTLY to hipHostMalloc (pinned) memory; host node reads
 *      without a D2H copy. Compare with KernelWritesManagedMem variant.
 *
 * Test source
 * ------------------------
 *    - catch/unit/graph/hipLaunchHostFuncWithGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
HIP_TEST_CASE(Unit_hipLaunchHostFunc_KernelWritesPinnedMem_HostNodeReads) {
  int ticks_per_ms = 0;
#if HT_NVIDIA
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeClockRate, 0));
#else
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeWallClockRate, 0));
#endif
  if (ticks_per_ms == 0) {
    ticks_per_ms = 1000;
  }
  const long long kSpin = static_cast<long long>(ticks_per_ms) * 500;
  constexpr int kExpected = 1;

  int* h_pinned{};
  HIP_CHECK(hipHostMalloc(&h_pinned, sizeof(int), hipHostMallocDefault));

  int* h_cb_first{};
  int* h_cb_last{};
  int* h_out{};
  HIP_CHECK(hipHostMalloc(&h_cb_first, sizeof(int), hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&h_cb_last,  sizeof(int), hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&h_out,      sizeof(int), hipHostMallocDefault));

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t prod_node{};
  {
    long long spin = kSpin;
    void* args[] = {&h_pinned, &spin};
    hipKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(spin_then_set);
    p.gridDim = dim3(1);
    p.blockDim = dim3(1);
    p.kernelParams = args;
    HIP_CHECK(hipGraphAddKernelNode(&prod_node, graph, nullptr, 0, &p));
  }

  struct CbArgsPin {
    int* h_pinned;
    int* h_cb_first;
    int* h_cb_last;
  };
  CbArgsPin cb_args{h_pinned, h_cb_first, h_cb_last};
  hipGraphNode_t host_node{};
  hipHostNodeParams host_params{};
  host_params.fn = [](void* ud) {
    auto* a = static_cast<CbArgsPin*>(ud);
    auto* pinned = reinterpret_cast<volatile int*>(a->h_pinned);
    int first = *pinned;
    *a->h_cb_first = first;
    int last = first;
    for (int spin = 0; spin < 1000000 && last != 1; ++spin) {
      last = *pinned;
    }
    *a->h_cb_last = last;
  };
  host_params.userData = &cb_args;
  {
    hipGraphNode_t deps[] = {prod_node};
    HIP_CHECK(hipGraphAddHostNode(&host_node, graph, deps, 1, &host_params));
  }

  hipGraphNode_t check_node{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&check_node, graph, &host_node, 1,
                                    h_out, h_pinned, sizeof(int),
                                    hipMemcpyHostToHost));

  hipGraphExec_t graphExec{};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  hipStream_t launch_stream{};
  HIP_CHECK(hipStreamCreate(&launch_stream));

  for (int iter = 0; iter < 10; ++iter) {
    *h_pinned   = 0;
    *h_cb_first = 0;
    *h_cb_last  = 0;
    *h_out      = 0;
    HIP_CHECK(hipGraphLaunch(graphExec, launch_stream));
    HIP_CHECK(hipStreamSynchronize(launch_stream));

    const int after_sync = *h_pinned;
    INFO("iter=" << iter
         << " h_cb_first=" << *h_cb_first
         << " h_cb_last="  << *h_cb_last
         << " after_sync=" << after_sync
         << " h_out="      << *h_out);
    REQUIRE(*h_out      == kExpected);
    REQUIRE(after_sync  == kExpected);
    REQUIRE(*h_cb_first == kExpected);
    REQUIRE(*h_cb_last  == kExpected);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(launch_stream));
  HIP_CHECK(hipHostFree(h_pinned));
  HIP_CHECK(hipHostFree(h_out));
  HIP_CHECK(hipHostFree(h_cb_last));
  HIP_CHECK(hipHostFree(h_cb_first));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
