/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipExecutionCtxStreamDetached hipExecutionCtxStreamDetached
 * @{
 * @ingroup ExecutionContextTest
 * `hipExecutionCtxDestroy` stream-detach behavioral tests.
 *
 * After hipExecutionCtxDestroy, every stream that was created on that ctx
 * must:
 *   (a) outlive the ctx (the stream handle stays valid and destroyable),
 *   (b) reject every guarded work / sync / capture API with
 *       hipErrorStreamDetached (CHECK_STREAM_DETACHED in hip_internal.hpp),
 *   (c) keep all metadata getters (flags / priority / device / id / CU mask /
 *       attributes) succeeding with the pre-detach values,
 *   (d) invalidate any active stream capture on the stream and on every
 *       parallel-capture branch forked off of it.
 * Default / legacy / per-thread streams are never owned by an ExecutionCtx
 * and must never detach.
 *
 * The existing PR-3253 test `Unit_hipExecutionCtxStreamDestroy_Negative`
 * already covers hipStreamSynchronize / hipStreamQuery / hipStreamDestroy on
 * the detached stream; this file adds coverage for everything else.
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include "hip_executionctx_common.hh"

#include <vector>

namespace {

// Trivial kernel used by KernelLaunch_Negative. The body is intentionally
// empty: the test only asserts the launch entry point's detach guard, never
// any device-side behavior.
__global__ void NoOpKernel() {}

// Device symbol used by the hipMemcpy{To,From}SymbolAsync sections of
// MemcpyAsync_Negative.
__device__ int kSymbol = 0;

// Host-function callback used by StreamApis_Negative. Never actually invoked
// (the surrounding hipLaunchHostFunc call is expected to fail with detach).
void HostFnNoop(void* /*userData*/) {}

// Stream callback used by StreamApis_Negative. Same reasoning as HostFnNoop.
void StreamCallbackNoop(hipStream_t /*stream*/, hipError_t /*status*/, void* /*userData*/) {}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Verifies that the per-stream work / wait / host-func entrypoints in
 *    hip_stream.cpp guarded by CHECK_STREAM_DETACHED return
 *    hipErrorStreamDetached after the owning ctx is destroyed.
 *    Skips hipStreamSynchronize / hipStreamQuery (covered by the existing
 *    Unit_hipExecutionCtxStreamDestroy_Negative test).
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_StreamApis_Negative) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  // Pre-record an event on a separate non-detached stream so the
  // hipStreamWaitEvent section isolates the detach check on `detached`
  // (otherwise an unrecorded event would also fail).
  hipStream_t helper = nullptr;
  HIP_CHECK(hipStreamCreate(&helper));
  hipEvent_t event = nullptr;
  HIP_CHECK(hipEventCreate(&event));
  HIP_CHECK(hipEventRecord(event, helper));
  HIP_CHECK(hipStreamSynchronize(helper));

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  SECTION("hipStreamWaitEvent") {
    HIP_CHECK_ERROR(hipStreamWaitEvent(detached, event, 0), hipErrorStreamDetached);
  }

  SECTION("hipStreamAddCallback") {
    HIP_CHECK_ERROR(hipStreamAddCallback(detached, StreamCallbackNoop, nullptr, 0),
                    hipErrorStreamDetached);
  }

  SECTION("hipLaunchHostFunc") {
    HIP_CHECK_ERROR(hipLaunchHostFunc(detached, HostFnNoop, nullptr), hipErrorStreamDetached);
  }

  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(helper));
  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * Test Description
 * ------------------------
 *  - hipEventRecord and hipEventRecordWithFlags on a detached stream return
 *    hipErrorStreamDetached.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_EventRecord_Negative) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  hipEvent_t event = nullptr;
  HIP_CHECK(hipEventCreate(&event));

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  SECTION("hipEventRecord") {
    HIP_CHECK_ERROR(hipEventRecord(event, detached), hipErrorStreamDetached);
  }

  SECTION("hipEventRecordWithFlags") {
    HIP_CHECK_ERROR(hipEventRecordWithFlags(event, detached, 0), hipErrorStreamDetached);
  }

  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * Test Description
 * ------------------------
 *  - Runtime kernel launch (<<<>>>) and hipLaunchKernelExC on a detached
 *    stream return hipErrorStreamDetached. The module / cooperative /
 *    multi-device launch entry points share the exact same
 *    CHECK_STREAM_DETACHED macro and aren't separately exercised here.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_KernelLaunch_Negative) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  SECTION("triple-chevron launch (hipLaunchKernel)") {
    NoOpKernel<<<dim3(1), dim3(1), 0, detached>>>();
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorStreamDetached);
  }

  SECTION("hipLaunchKernelExC") {
    hipLaunchConfig_t config{};
    config.gridDim = dim3(1);
    config.blockDim = dim3(1);
    config.dynamicSmemBytes = 0;
    config.stream = detached;
    config.attrs = nullptr;
    config.numAttrs = 0;
    HIP_CHECK_ERROR(
        hipLaunchKernelExC(&config, reinterpret_cast<const void*>(NoOpKernel), nullptr),
        hipErrorStreamDetached);
  }

  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * Test Description
 * ------------------------
 *  - Every async memcpy entry point in hip_memory.cpp guarded by
 *    CHECK_STREAM_DETACHED returns hipErrorStreamDetached on a detached
 *    stream. Buffers are allocated before detach and freed after; the test
 *    only checks the return code, not data correctness.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_MemcpyAsync_Negative) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  constexpr size_t kBytes = 256;
  void* h_src = nullptr;
  void* h_dst = nullptr;
  void* d_src = nullptr;
  void* d_dst = nullptr;
  HIP_CHECK(hipHostMalloc(&h_src, kBytes));
  HIP_CHECK(hipHostMalloc(&h_dst, kBytes));
  HIP_CHECK(hipMalloc(&d_src, kBytes));
  HIP_CHECK(hipMalloc(&d_dst, kBytes));

  // 2D / 3D scratch buffers, all pre-detach.
  constexpr size_t k2DPitch = 64;
  constexpr size_t k2DRows = 4;
  void* d_2d_src = nullptr;
  void* d_2d_dst = nullptr;
  HIP_CHECK(hipMalloc(&d_2d_src, k2DPitch * k2DRows));
  HIP_CHECK(hipMalloc(&d_2d_dst, k2DPitch * k2DRows));

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  SECTION("hipMemcpyAsync") {
    HIP_CHECK_ERROR(
        hipMemcpyAsync(d_dst, d_src, kBytes, hipMemcpyDeviceToDevice, detached),
        hipErrorStreamDetached);
  }

  SECTION("hipMemcpyHtoDAsync") {
    HIP_CHECK_ERROR(
        hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(d_dst), h_src, kBytes, detached),
        hipErrorStreamDetached);
  }

  SECTION("hipMemcpyDtoHAsync") {
    HIP_CHECK_ERROR(
        hipMemcpyDtoHAsync(h_dst, reinterpret_cast<hipDeviceptr_t>(d_src), kBytes, detached),
        hipErrorStreamDetached);
  }

  SECTION("hipMemcpyDtoDAsync") {
    HIP_CHECK_ERROR(
        hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(d_dst),
                           reinterpret_cast<hipDeviceptr_t>(d_src), kBytes, detached),
        hipErrorStreamDetached);
  }

  SECTION("hipMemcpy2DAsync") {
    HIP_CHECK_ERROR(hipMemcpy2DAsync(d_2d_dst, k2DPitch, d_2d_src, k2DPitch, k2DPitch, k2DRows,
                                     hipMemcpyDeviceToDevice, detached),
                    hipErrorStreamDetached);
  }

  SECTION("hipMemcpy3DAsync") {
    // CHECK_STREAM_DETACHED runs before any param validation; an empty params
    // struct is sufficient for the detach assertion.
    hipMemcpy3DParms params{};
    HIP_CHECK_ERROR(hipMemcpy3DAsync(&params, detached), hipErrorStreamDetached);
  }

  SECTION("hipDrvMemcpy3DAsync") {
    HIP_MEMCPY3D params{};
    HIP_CHECK_ERROR(hipDrvMemcpy3DAsync(&params, detached), hipErrorStreamDetached);
  }

  SECTION("hipMemcpyToSymbolAsync") {
    int value = 42;
    HIP_CHECK_ERROR(hipMemcpyToSymbolAsync(HIP_SYMBOL(kSymbol), &value, sizeof(value), 0,
                                           hipMemcpyHostToDevice, detached),
                    hipErrorStreamDetached);
  }

  SECTION("hipMemcpyFromSymbolAsync") {
    int value = 0;
    HIP_CHECK_ERROR(hipMemcpyFromSymbolAsync(&value, HIP_SYMBOL(kSymbol), sizeof(value), 0,
                                             hipMemcpyDeviceToHost, detached),
                    hipErrorStreamDetached);
  }

  HIP_CHECK(hipFree(d_2d_src));
  HIP_CHECK(hipFree(d_2d_dst));
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipHostFree(h_src));
  HIP_CHECK(hipHostFree(h_dst));
  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * Test Description
 * ------------------------
 *  - Every async memset entry point in hip_memory.cpp guarded by
 *    CHECK_STREAM_DETACHED returns hipErrorStreamDetached on a detached
 *    stream.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_MemsetAsync_Negative) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  constexpr size_t kBytes = 256;
  void* d_buf = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, kBytes));

  constexpr size_t k2DPitch = 64;
  constexpr size_t k2DRows = 4;
  void* d_2d = nullptr;
  HIP_CHECK(hipMalloc(&d_2d, k2DPitch * k2DRows));

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  SECTION("hipMemsetAsync") {
    HIP_CHECK_ERROR(hipMemsetAsync(d_buf, 0, kBytes, detached), hipErrorStreamDetached);
  }

  SECTION("hipMemsetD8Async") {
    HIP_CHECK_ERROR(hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(d_buf), 0, kBytes, detached),
                    hipErrorStreamDetached);
  }

  SECTION("hipMemsetD16Async") {
    HIP_CHECK_ERROR(
        hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(d_buf), 0, kBytes / 2, detached),
        hipErrorStreamDetached);
  }

  SECTION("hipMemsetD32Async") {
    HIP_CHECK_ERROR(
        hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d_buf), 0, kBytes / 4, detached),
        hipErrorStreamDetached);
  }

  SECTION("hipMemset2DAsync") {
    HIP_CHECK_ERROR(hipMemset2DAsync(d_2d, k2DPitch, 0, k2DPitch, k2DRows, detached),
                    hipErrorStreamDetached);
  }

  SECTION("hipMemset3DAsync") {
    hipPitchedPtr pitched{d_2d, k2DPitch, k2DPitch, k2DRows};
    hipExtent extent{k2DPitch, k2DRows, 1};
    HIP_CHECK_ERROR(hipMemset3DAsync(pitched, 0, extent, detached), hipErrorStreamDetached);
  }

  HIP_CHECK(hipFree(d_2d));
  HIP_CHECK(hipFree(d_buf));
  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * Test Description
 * ------------------------
 *  - Stream-ordered allocator entry points (hipMallocAsync / hipFreeAsync /
 *    hipMallocFromPoolAsync) on a detached stream return
 *    hipErrorStreamDetached.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_MempoolAsync_Negative) {
  int mempool_supported = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&mempool_supported, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (mempool_supported == 0) {
    SKIP("Device does not support stream-ordered memory pools");
  }

  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  hipMemPool_t pool = nullptr;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&pool, 0));
  REQUIRE(pool != nullptr);

  // Pre-allocate a buffer through the pool on a non-detached stream so the
  // hipFreeAsync section has a valid pointer to free (the detach assertion
  // fires before the actual free, so no leak occurs).
  hipStream_t helper = nullptr;
  HIP_CHECK(hipStreamCreate(&helper));
  void* d_pool_buf = nullptr;
  HIP_CHECK(hipMallocFromPoolAsync(&d_pool_buf, 256, pool, helper));
  HIP_CHECK(hipStreamSynchronize(helper));

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  SECTION("hipMallocAsync") {
    void* d_ptr = nullptr;
    HIP_CHECK_ERROR(hipMallocAsync(&d_ptr, 256, detached), hipErrorStreamDetached);
  }

  SECTION("hipFreeAsync") {
    HIP_CHECK_ERROR(hipFreeAsync(d_pool_buf, detached), hipErrorStreamDetached);
  }

  SECTION("hipMallocFromPoolAsync") {
    void* d_ptr = nullptr;
    HIP_CHECK_ERROR(hipMallocFromPoolAsync(&d_ptr, 256, pool, detached),
                    hipErrorStreamDetached);
  }

  // Free the pre-allocated buffer cleanly via the helper stream.
  HIP_CHECK(hipFreeAsync(d_pool_buf, helper));
  HIP_CHECK(hipStreamSynchronize(helper));
  HIP_CHECK(hipStreamDestroy(helper));
  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * Test Description
 * ------------------------
 *  - hipMemPrefetchAsync, hipMemPrefetchAsync_v2, and hipStreamAttachMemAsync
 *    on a detached stream return hipErrorStreamDetached. Skipped on devices
 *    that do not support managed memory (or, for the prefetch variants,
 *    concurrent managed access).
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 *  - hipDeviceAttributeManagedMemory != 0 (and ConcurrentManagedAccess for
 *    the prefetch variants)
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_HmmAsync_Negative) {
  int managed_supported = 0;
  HIP_CHECK(hipDeviceGetAttribute(&managed_supported, hipDeviceAttributeManagedMemory, 0));
  if (managed_supported == 0) {
    SKIP("Device does not support managed memory; skipping HMM detach checks");
  }

  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  constexpr size_t kBytes = 256;
  int* managed_ptr = nullptr;
  HIP_CHECK(hipMallocManaged(reinterpret_cast<void**>(&managed_ptr), kBytes));

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  SECTION("hipMemPrefetchAsync") {
    int concurrent = 0;
    HIP_CHECK(
        hipDeviceGetAttribute(&concurrent, hipDeviceAttributeConcurrentManagedAccess, 0));
    if (concurrent == 0) {
      SKIP("Device does not support ConcurrentManagedAccess; skipping prefetch detach check");
    }
    HIP_CHECK_ERROR(hipMemPrefetchAsync(managed_ptr, kBytes, 0, detached),
                    hipErrorStreamDetached);
  }

  SECTION("hipMemPrefetchAsync_v2") {
    int concurrent = 0;
    HIP_CHECK(
        hipDeviceGetAttribute(&concurrent, hipDeviceAttributeConcurrentManagedAccess, 0));
    if (concurrent == 0) {
      SKIP("Device does not support ConcurrentManagedAccess; skipping prefetch_v2 detach check");
    }
    hipMemLocation location{};
    location.type = hipMemLocationTypeDevice;
    location.id = 0;
    HIP_CHECK_ERROR(hipMemPrefetchAsync_v2(managed_ptr, kBytes, location, 0, detached),
                    hipErrorStreamDetached);
  }

  SECTION("hipStreamAttachMemAsync") {
    HIP_CHECK_ERROR(
        hipStreamAttachMemAsync(detached, reinterpret_cast<hipDeviceptr_t*>(managed_ptr), kBytes, hipMemAttachSingle),
        hipErrorStreamDetached);
  }

  HIP_CHECK(hipFree(managed_ptr));
  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * Test Description
 * ------------------------
 *  - hipGraphLaunch on a detached stream returns hipErrorStreamDetached. The
 *    graph and exec are built BEFORE detaching so the launch path itself is
 *    the only thing under test.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_GraphLaunch_Negative) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t empty_node = nullptr;
  HIP_CHECK(hipGraphAddEmptyNode(&empty_node, graph, nullptr, 0));
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  REQUIRE(exec != nullptr);

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  HIP_CHECK_ERROR(hipGraphLaunch(exec, detached), hipErrorStreamDetached);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * Test Description
 * ------------------------
 *  - hipStreamBeginCapture, hipStreamBeginCaptureToGraph, and
 *    hipStreamEndCapture on a detached stream return hipErrorStreamDetached.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_StreamCapture_Negative) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  SECTION("hipStreamBeginCapture") {
    HIP_CHECK_ERROR(hipStreamBeginCapture(detached, hipStreamCaptureModeRelaxed),
                    hipErrorStreamDetached);
  }

  SECTION("hipStreamBeginCaptureToGraph") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK_ERROR(hipStreamBeginCaptureToGraph(detached, graph, nullptr, nullptr, 0,
                                                 hipStreamCaptureModeRelaxed),
                    hipErrorStreamDetached);
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("hipStreamEndCapture") {
    hipGraph_t out_graph = nullptr;
    HIP_CHECK_ERROR(hipStreamEndCapture(detached, &out_graph), hipErrorStreamDetached);
  }

  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * Test Description
 * ------------------------
 *  - Begin a stream capture on a ctx stream, then destroy the ctx. The
 *    capture must be invalidated: hipStreamGetCaptureInfo (which is NOT
 *    guarded - it reads captureStatus_ directly) reports
 *    hipStreamCaptureStatusInvalidated, while hipStreamEndCapture (which IS
 *    guarded) returns hipErrorStreamDetached. Covers requirement (d) for
 *    the single-stream case. A regular non-ctx stream is begun-and-ended
 *    in parallel as a control to confirm the API itself still functions.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_ActiveCapture_Invalidated) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  hipStream_t regular = nullptr;
  HIP_CHECK(hipStreamCreate(&regular));

  HIP_CHECK(hipStreamBeginCapture(detached, hipStreamCaptureModeRelaxed));
  HIP_CHECK(hipStreamBeginCapture(regular, hipStreamCaptureModeRelaxed));

  hipStreamCaptureStatus status_before = hipStreamCaptureStatusNone;
  unsigned long long capture_id = 0;
  HIP_CHECK(hipStreamGetCaptureInfo(detached, &status_before, &capture_id));
  REQUIRE(status_before == hipStreamCaptureStatusActive);

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  hipStreamCaptureStatus status_after = hipStreamCaptureStatusNone;
  HIP_CHECK(hipStreamGetCaptureInfo(detached, &status_after, &capture_id));
  REQUIRE(status_after == hipStreamCaptureStatusInvalidated);

  hipGraph_t out_graph = nullptr;
  HIP_CHECK_ERROR(hipStreamEndCapture(detached, &out_graph), hipErrorStreamDetached);

  // Control: the regular stream's capture is unaffected and ends cleanly.
  hipGraph_t regular_graph = nullptr;
  HIP_CHECK(hipStreamEndCapture(regular, &regular_graph));
  if (regular_graph != nullptr) {
    HIP_CHECK(hipGraphDestroy(regular_graph));
  }

  HIP_CHECK(hipStreamDestroy(regular));
  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * Test Description
 * ------------------------
 *  - Standard fork-via-event capture pattern: begin capture on the origin
 *    stream, record an event on origin, wait that event on the forked
 *    stream. Both streams are owned by the same ctx. After
 *    hipExecutionCtxDestroy, both captureStatus_ slots must flip to
 *    hipStreamCaptureStatusInvalidated. Covers requirement (d) for
 *    parallelCaptureStreams_.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_ParallelCapture_Invalidated) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t origin = nullptr;
  MakeCtxAndStream(ctx, origin);
  hipStream_t fork = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&fork, ctx, hipStreamNonBlocking, 0));
  REQUIRE(fork != nullptr);

  hipEvent_t fork_event = nullptr;
  HIP_CHECK(hipEventCreate(&fork_event));

  HIP_CHECK(hipStreamBeginCapture(origin, hipStreamCaptureModeRelaxed));
  HIP_CHECK(hipEventRecord(fork_event, origin));
  HIP_CHECK(hipStreamWaitEvent(fork, fork_event, 0));

  hipStreamCaptureStatus origin_status = hipStreamCaptureStatusNone;
  hipStreamCaptureStatus fork_status = hipStreamCaptureStatusNone;
  unsigned long long capture_id = 0;
  HIP_CHECK(hipStreamGetCaptureInfo(origin, &origin_status, &capture_id));
  HIP_CHECK(hipStreamGetCaptureInfo(fork, &fork_status, &capture_id));
  REQUIRE(origin_status == hipStreamCaptureStatusActive);
  REQUIRE(fork_status == hipStreamCaptureStatusActive);

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  HIP_CHECK(hipStreamGetCaptureInfo(origin, &origin_status, &capture_id));
  HIP_CHECK(hipStreamGetCaptureInfo(fork, &fork_status, &capture_id));
  REQUIRE(origin_status == hipStreamCaptureStatusInvalidated);
  REQUIRE(fork_status == hipStreamCaptureStatusInvalidated);

  HIP_CHECK(hipEventDestroy(fork_event));
  HIP_CHECK(hipStreamDestroy(fork));
  HIP_CHECK(hipStreamDestroy(origin));
}

/**
 * Test Description
 * ------------------------
 *  - Metadata getters on a detached stream succeed and return the same
 *    values they returned before detach. Covers requirement (c): the
 *    deliberately-unguarded getters in hip_stream.cpp
 *    (hipStreamGetFlags / Priority / Device / Id / hipExtStreamGetCUMask /
 *    hipStreamGetAttribute / hipStreamSetAttribute / hipStreamCopyAttributes),
 *    which must continue to report the values that were valid for the
 *    stream prior to detach.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_MetadataGetters_Succeed) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  unsigned int flags_before = 0;
  int priority_before = 0;
  hipDevice_t device_before = -1;
  unsigned long long id_before = 0;
  HIP_CHECK(hipStreamGetFlags(detached, &flags_before));
  HIP_CHECK(hipStreamGetPriority(detached, &priority_before));
  HIP_CHECK(hipStreamGetDevice(detached, &device_before));
  HIP_CHECK(hipStreamGetId(detached, &id_before));

#if HT_AMD
  // hipExtStreamGetCUMask is an AMD-only extension; skip this leg when
  // targeting the NVIDIA backend, which does not expose an analogous getter.
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  const uint32_t cu_mask_size =
      (static_cast<uint32_t>(props.multiProcessorCount) / 32) + 1;
  std::vector<uint32_t> cu_mask_before(cu_mask_size, 0);
  HIP_CHECK(hipExtStreamGetCUMask(detached, cu_mask_before.size(), cu_mask_before.data()));
#endif

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  unsigned int flags_after = 0;
  int priority_after = 0;
  hipDevice_t device_after = -1;
  unsigned long long id_after = 0;
  HIP_CHECK(hipStreamGetFlags(detached, &flags_after));
  HIP_CHECK(hipStreamGetPriority(detached, &priority_after));
  HIP_CHECK(hipStreamGetDevice(detached, &device_after));
  HIP_CHECK(hipStreamGetId(detached, &id_after));

  REQUIRE(flags_after == flags_before);
  REQUIRE(priority_after == priority_before);
  REQUIRE(device_after == device_before);
  REQUIRE(id_after == id_before);

#if HT_AMD
  std::vector<uint32_t> cu_mask_after(cu_mask_size, 0);
  HIP_CHECK(hipExtStreamGetCUMask(detached, cu_mask_after.size(), cu_mask_after.data()));
  for (size_t i = 0; i < cu_mask_before.size(); ++i) {
    REQUIRE(cu_mask_after[i] == cu_mask_before[i]);
  }
#endif

  // Attribute get/set/copy must also still work post-detach.
  hipStreamAttrValue attr_value{};
  HIP_CHECK(hipStreamGetAttribute(detached, hipStreamAttributeSynchronizationPolicy, &attr_value));
  HIP_CHECK(hipStreamSetAttribute(detached, hipStreamAttributeSynchronizationPolicy, &attr_value));

  hipStream_t regular = nullptr;
  HIP_CHECK(hipStreamCreate(&regular));
  HIP_CHECK(hipStreamCopyAttributes(regular, detached));
  HIP_CHECK(hipStreamDestroy(regular));

  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * Test Description
 * ------------------------
 *  - Reverse-order teardown: destroy the ctx first, then destroy the
 *    stream. Both calls must succeed. Exercises the
 *    "Stream::Destroy with executionCtx_ already cleared" branch.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_StreamOutlivesCtx) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t stream = nullptr;
  MakeCtxAndStream(ctx, stream);

  HIP_CHECK(hipExecutionCtxDestroy(ctx));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Normal-order teardown: destroy the stream first, then the ctx.
 *    Verifies the ctx->removeStream(stream) path so ~ExecutionCtx finds
 *    the stream set already empty and the per-survivor Detach loop is a
 *    no-op.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_StreamDestroyedBeforeCtx) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t stream = nullptr;
  MakeCtxAndStream(ctx, stream);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipExecutionCtxDestroy(ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Multiple streams under one ctx: destroying the ctx detaches all of
 *    them, all of them remain destroyable, and each rejects post-detach
 *    work with hipErrorStreamDetached.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_MultipleStreams) {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  hipExecutionCtx_t ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&ctx, desc, 0, 0));
  REQUIRE(ctx != nullptr);

  constexpr int kNumStreams = 4;
  hipStream_t streams[kNumStreams] = {nullptr, nullptr, nullptr, nullptr};
  for (int i = 0; i < kNumStreams; ++i) {
    HIP_CHECK(hipExecutionCtxStreamCreate(&streams[i], ctx, hipStreamNonBlocking, 0));
    REQUIRE(streams[i] != nullptr);
  }

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  for (int i = 0; i < kNumStreams; ++i) {
    HIP_CHECK_ERROR(hipStreamSynchronize(streams[i]), hipErrorStreamDetached);
  }
  for (int i = 0; i < kNumStreams; ++i) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
}

/**
 * Test Description
 * ------------------------
 *  - A ctx with zero streams destroys cleanly. Validates that the
 *    ~ExecutionCtx detach loop tolerates an empty streams_ set.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_EmptyCtx) {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  hipExecutionCtx_t ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&ctx, desc, 0, 0));
  REQUIRE(ctx != nullptr);

  HIP_CHECK(hipExecutionCtxDestroy(ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Two independent ctxs built on disjoint SM partitions, each owning one
 *    stream. Destroying ctx1 detaches only stream1; stream2 (owned by
 *    ctx2) remains fully usable. Then destroying ctx2 detaches stream2.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_MultiCtx_Isolation) {
  HIP_CHECK(hipSetDevice(0));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource input{};
  HIP_CHECK(hipDeviceGetDevResource(device, &input, hipDevResourceTypeSm));

  unsigned int totalSMs = input.sm.smCount;
  unsigned int alignment = input.sm.smCoscheduledAlignment;
  unsigned int halfSMs = (totalSMs / 2 / alignment) * alignment;
  if (halfSMs < input.sm.minSmPartitionSize) {
    SKIP("Device does not have enough SMs to split into two non-overlapping ctxs");
  }

  hipDevSmResourceGroupParams params[2] = {};
  params[0].smCount = halfSMs;
  params[1].smCount = halfSMs;

  hipDevResource result[2] = {};
  hipDevResource remainder{};
  HIP_CHECK(hipDevSmResourceSplit(result, 2, &input, &remainder, 0, params));

  hipDevResourceDesc_t desc1{};
  hipDevResourceDesc_t desc2{};
  HIP_CHECK(hipDevResourceGenerateDesc(&desc1, &result[0], 1));
  HIP_CHECK(hipDevResourceGenerateDesc(&desc2, &result[1], 1));

  hipExecutionCtx_t ctx1 = nullptr;
  hipExecutionCtx_t ctx2 = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&ctx1, desc1, 0, 0));
  HIP_CHECK(hipGreenCtxCreate(&ctx2, desc2, 0, 0));
  REQUIRE(ctx1 != nullptr);
  REQUIRE(ctx2 != nullptr);

  hipStream_t stream1 = nullptr;
  hipStream_t stream2 = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream1, ctx1, hipStreamNonBlocking, 0));
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream2, ctx2, hipStreamNonBlocking, 0));

  // Destroy ctx1 only - stream1 should detach, stream2 should remain alive.
  HIP_CHECK(hipExecutionCtxDestroy(ctx1));

  HIP_CHECK_ERROR(hipStreamSynchronize(stream1), hipErrorStreamDetached);
  HIP_CHECK(hipStreamSynchronize(stream2));

  // Destroy ctx2 - stream2 detaches now too.
  HIP_CHECK(hipExecutionCtxDestroy(ctx2));
  HIP_CHECK_ERROR(hipStreamSynchronize(stream2), hipErrorStreamDetached);

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
}

/**
 * Test Description
 * ------------------------
 *  - A plain hipStreamCreate stream coexisting with a ctx-owned stream is
 *    not affected by hipExecutionCtxDestroy. The regular stream still
 *    accepts and completes work after the unrelated ctx is destroyed.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_RegularStream_NotAffected) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t ctx_stream = nullptr;
  MakeCtxAndStream(ctx, ctx_stream);

  hipStream_t regular = nullptr;
  HIP_CHECK(hipStreamCreate(&regular));

  constexpr size_t kBytes = 256;
  void* h_src = nullptr;
  void* h_dst = nullptr;
  void* d_buf = nullptr;
  HIP_CHECK(hipHostMalloc(&h_src, kBytes));
  HIP_CHECK(hipHostMalloc(&h_dst, kBytes));
  HIP_CHECK(hipMalloc(&d_buf, kBytes));
  for (size_t i = 0; i < kBytes; ++i) {
    static_cast<unsigned char*>(h_src)[i] = static_cast<unsigned char>(i & 0xFF);
    static_cast<unsigned char*>(h_dst)[i] = 0;
  }

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  // The regular stream is unaffected: round-trip a memcpy through it.
  HIP_CHECK(hipMemcpyAsync(d_buf, h_src, kBytes, hipMemcpyHostToDevice, regular));
  HIP_CHECK(hipMemcpyAsync(h_dst, d_buf, kBytes, hipMemcpyDeviceToHost, regular));
  HIP_CHECK(hipStreamSynchronize(regular));
  for (size_t i = 0; i < kBytes; ++i) {
    REQUIRE(static_cast<unsigned char*>(h_dst)[i] == static_cast<unsigned char>(i & 0xFF));
  }

  // The ctx-owned stream should be detached.
  HIP_CHECK_ERROR(hipStreamSynchronize(ctx_stream), hipErrorStreamDetached);

  HIP_CHECK(hipFree(d_buf));
  HIP_CHECK(hipHostFree(h_src));
  HIP_CHECK(hipHostFree(h_dst));
  HIP_CHECK(hipStreamDestroy(regular));
  HIP_CHECK(hipStreamDestroy(ctx_stream));
}

/**
 * Test Description
 * ------------------------
 *  - The default streams (nullptr / hipStreamLegacy / hipStreamPerThread)
 *    are never owned by an ExecutionCtx and must never detach. After
 *    destroying an unrelated ctx, hipStreamSynchronize and hipStreamQuery
 *    on each default stream still return hipSuccess, matching the
 *    nullptr / hipStreamLegacy / hipStreamPerThread skip clause in
 *    CHECK_STREAM_DETACHED.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_DefaultStreams_NeverDetach) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t scratch = nullptr;
  MakeCtxAndStream(ctx, scratch);
  HIP_CHECK(hipExecutionCtxDestroy(ctx));
  HIP_CHECK(hipStreamDestroy(scratch));

  // hipStreamQuery may legitimately return hipErrorNotReady if work is in
  // flight; both hipSuccess and hipErrorNotReady are acceptable as
  // not-detached responses.
  SECTION("hipStreamSynchronize on default streams") {
    HIP_CHECK(hipStreamSynchronize(nullptr));
    HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
  }

  SECTION("hipStreamQuery on default streams") {
    hipError_t q0 = hipStreamQuery(nullptr);
    REQUIRE((q0 == hipSuccess || q0 == hipErrorNotReady));
    hipError_t q1 = hipStreamQuery(hipStreamLegacy);
    REQUIRE((q1 == hipSuccess || q1 == hipErrorNotReady));
    hipError_t q2 = hipStreamQuery(hipStreamPerThread);
    REQUIRE((q2 == hipSuccess || q2 == hipErrorNotReady));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Stream value operations (hipStreamWaitValue32 / hipStreamWaitValue64 /
 *    hipStreamWriteValue32 / hipStreamWriteValue64) and the batch entry
 *    point hipStreamBatchMemOp on a detached stream must return
 *    hipErrorStreamDetached. These are work-submitting APIs that fall under
 *    the general detach contract.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_StreamMemOps_Negative) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  // Allocate scratch device memory. The CHECK_STREAM_DETACHED guard runs
  // before getMemoryObject() in ihipStreamOperation, so this buffer only
  // exists so the ptr argument is non-null and a meaningful target if the
  // detach guard were ever (incorrectly) bypassed.
  uint64_t* d_signal64 = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_signal64), sizeof(uint64_t)));
  uint32_t* d_signal32 = reinterpret_cast<uint32_t*>(d_signal64);

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  SECTION("hipStreamWaitValue32") {
    HIP_CHECK_ERROR(hipStreamWaitValue32(detached, d_signal32, 0u, hipStreamWaitValueEq, 0xFFFFFFFFu),
                    hipErrorStreamDetached);
  }

  SECTION("hipStreamWaitValue64") {
    HIP_CHECK_ERROR(
        hipStreamWaitValue64(detached, d_signal64, 0ull, hipStreamWaitValueEq, 0xFFFFFFFFFFFFFFFFull),
        hipErrorStreamDetached);
  }

  SECTION("hipStreamWriteValue32") {
    HIP_CHECK_ERROR(
        hipStreamWriteValue32(detached, d_signal32, 0u, hipStreamWriteValueDefault),
        hipErrorStreamDetached);
  }

  SECTION("hipStreamWriteValue64") {
    HIP_CHECK_ERROR(
        hipStreamWriteValue64(detached, d_signal64, 0ull, hipStreamWriteValueDefault),
        hipErrorStreamDetached);
  }

  SECTION("hipStreamBatchMemOp") {
    // A single WaitValue32 op is sufficient to drive the batch path far
    // enough to hit CHECK_STREAM_DETACHED. The op never actually runs.
    hipStreamBatchMemOpParams op{};
    op.operation = hipStreamMemOpWaitValue32;
    op.waitValue.operation = hipStreamMemOpWaitValue32;
    op.waitValue.address = reinterpret_cast<hipDeviceptr_t>(d_signal32);
    op.waitValue.value = 0u;
    op.waitValue.flags = hipStreamWaitValueEq;
    HIP_CHECK_ERROR(hipStreamBatchMemOp(detached, 1, &op, 0), hipErrorStreamDetached);
  }

  HIP_CHECK(hipFree(d_signal64));
  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * Test Description
 * ------------------------
 *  - hipMemcpyBatchAsync and hipMemcpy3DBatchAsync on a detached stream
 *    return hipErrorStreamDetached. Both APIs are guarded in
 *    hip_memory.cpp but were not covered by the initial detach test
 *    suite; adding them closes the work-submission coverage gap.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_MemcpyBatch_Negative) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  constexpr size_t kBytes = 256;
  void* h_buf = nullptr;
  void* d_buf = nullptr;
  HIP_CHECK(hipHostMalloc(&h_buf, kBytes));
  HIP_CHECK(hipMalloc(&d_buf, kBytes));

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  SECTION("hipMemcpyBatchAsync") {
    void* dsts[1] = {d_buf};
    void* srcs[1] = {h_buf};
    size_t sizes[1] = {kBytes};
    size_t fail_idx = 0;
    HIP_CHECK_ERROR(
        hipMemcpyBatchAsync(dsts, srcs, sizes, 1, /*attrs=*/nullptr,
                            /*attrsIdxs=*/nullptr, /*numAttrs=*/0, &fail_idx, detached),
        hipErrorStreamDetached);
  }

  SECTION("hipMemcpy3DBatchAsync") {
    // The detach guard fires before any deep inspection of the op list;
    // a one-element list with a stub op is enough to clear parameter
    // validation in hipMemcpy3DBatchAsync.
    hipMemcpy3DBatchOp op{};
    size_t fail_idx = 0;
    HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0ull, detached),
                    hipErrorStreamDetached);
  }

  HIP_CHECK(hipFree(d_buf));
  HIP_CHECK(hipHostFree(h_buf));
  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * Test Description
 * ------------------------
 *  - An event that was recorded on a ctx-owned stream BEFORE the ctx is
 *    destroyed must remain valid and queryable after detach. The detach
 *    contract scopes hipErrorStreamDetached to subsequent API calls on
 *    the stream; it does not invalidate completed events.
 *    Concretely:
 *      - hipEventQuery on the recorded event returns hipSuccess.
 *      - hipEventSynchronize on the recorded event succeeds.
 *      - hipStreamWaitEvent from a regular (non-detached) stream on
 *        that event succeeds — and the regular stream still completes
 *        work normally.
 *      - hipEventElapsedTime between two pre-detach events succeeds.
 *    This validates that Stream::Detach() does not invalidate the
 *    event itself (it only marks the stream detached and invalidates
 *    captures).
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDetached_EventRecordedBeforeDetach_StillUsable) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t detached = nullptr;
  MakeCtxAndStream(ctx, detached);

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  // Drive a tiny bit of real work on the ctx stream and record both events
  // around it so the events have a well-defined completion ordering.
  HIP_CHECK(hipEventRecord(start, detached));
  constexpr size_t kBytes = 64;
  void* d_buf = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, kBytes));
  HIP_CHECK(hipMemsetAsync(d_buf, 0, kBytes, detached));
  HIP_CHECK(hipEventRecord(stop, detached));

  // Make sure both events are completed BEFORE we destroy the ctx, so the
  // post-detach assertions below are about event lifetime, not about
  // pending work on a now-detached stream.
  HIP_CHECK(hipStreamSynchronize(detached));

  HIP_CHECK(hipExecutionCtxDestroy(ctx));

  SECTION("hipEventQuery survives detach") {
    HIP_CHECK(hipEventQuery(start));
    HIP_CHECK(hipEventQuery(stop));
  }

  SECTION("hipEventSynchronize survives detach") {
    HIP_CHECK(hipEventSynchronize(start));
    HIP_CHECK(hipEventSynchronize(stop));
  }

  SECTION("hipEventElapsedTime survives detach") {
    float ms = -1.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    REQUIRE(ms >= 0.0f);
  }

  SECTION("hipStreamWaitEvent on regular stream still works") {
    hipStream_t regular = nullptr;
    HIP_CHECK(hipStreamCreate(&regular));
    // Waiting on a completed pre-detach event from a non-ctx stream must
    // succeed and not poison the regular stream.
    HIP_CHECK(hipStreamWaitEvent(regular, stop, 0));
    // The regular stream still accepts and completes work afterwards.
    HIP_CHECK(hipMemsetAsync(d_buf, 0, kBytes, regular));
    HIP_CHECK(hipStreamSynchronize(regular));
    HIP_CHECK(hipStreamDestroy(regular));
  }

  HIP_CHECK(hipFree(d_buf));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipStreamDestroy(detached));
}

/**
 * End doxygen group hipExecutionCtxStreamDetached.
 * @}
 */
