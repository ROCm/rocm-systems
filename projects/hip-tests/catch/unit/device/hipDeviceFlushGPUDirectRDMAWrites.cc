/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <atomic>
#include <thread>  // NOLINT
#include <vector>

#include <hip_test_common.hh>

/**
 * @addtogroup hipDeviceFlushGPUDirectRDMAWrites hipDeviceFlushGPUDirectRDMAWrites
 * @{
 * @ingroup DeviceTest
 * `hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTarget target,
 *                              hipFlushGPUDirectRDMAWritesScope scope)` -
 * Blocks until remote writes that were made visible to the target by a third-party
 * device (typically an RDMA-capable NIC writing directly into device memory) are
 * visible at the requested scope.
 *
 * The call is a host-ordered, blocking visibility barrier on inbound remote writes.
 * It is not a synchronization primitive: it must not wait on any stream or kernel.
 *
 * Capability is reported through three device attributes, mirrored in
 * hipDeviceProp_t:
 *  - hipDeviceAttributeGPUDirectRDMASupported
 *  - hipDeviceAttributeGPUDirectRDMAFlushWritesOptions (hipFlushGPUDirectRDMAWritesOptions mask)
 *  - hipDeviceAttributeGPUDirectRDMAWritesOrdering (hipGPUDirectRDMAWritesOrdering value)
 */

namespace {

constexpr int kFlushOptionsMask =
    hipFlushGPUDirectRDMAWritesOptionHost | hipFlushGPUDirectRDMAWritesOptionMemOps;

struct RdmaCaps {
  int supported;
  int flushOptions;
  int writesOrdering;
};

RdmaCaps GetRdmaCaps(int device) {
  RdmaCaps caps{};
  HIP_CHECK(hipDeviceGetAttribute(&caps.supported, hipDeviceAttributeGPUDirectRDMASupported,
                                  device));
  HIP_CHECK(hipDeviceGetAttribute(&caps.flushOptions,
                                  hipDeviceAttributeGPUDirectRDMAFlushWritesOptions, device));
  HIP_CHECK(hipDeviceGetAttribute(&caps.writesOrdering,
                                  hipDeviceAttributeGPUDirectRDMAWritesOrdering, device));
  return caps;
}

bool HostFlushSupported(int device) {
  return (GetRdmaCaps(device).flushOptions & hipFlushGPUDirectRDMAWritesOptionHost) != 0;
}

/**
 * Expected return value of hipDeviceFlushGPUDirectRDMAWrites on a given device.
 *
 * Contract being pinned: the host-issued flush is available if and only if the
 * device advertises hipFlushGPUDirectRDMAWritesOptionHost. When the bit is clear
 * the call must fail loudly rather than silently succeed, otherwise a caller has
 * no way to distinguish "flushed" from "cannot flush".
 *
 * The hipErrorNotSupported branch could not be verified against CUDA. Both parts
 * measured (an RTX PRO 4000 Blackwell and a GeForce RTX 5090) advertise the Host
 * option, including the GeForce which reports GPUDirectRDMASupported = 0. The
 * flush-writes options and writes-ordering attributes describe the hardware write
 * path and are apparently independent of whether GPUDirect RDMA itself is enabled.
 */
hipError_t ExpectedFlushStatus(int device) {
  return HostFlushSupported(device) ? hipSuccess : hipErrorNotSupported;
}

__global__ void SpinUntilFlag(volatile int* flag) {
  while (*flag == 0) {
  }
}

__global__ void CheckPattern(const int* buf, size_t n, int* mismatches) {
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    if (buf[i] != static_cast<int>(i)) {
      atomicAdd(mismatches, 1);
    }
  }
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Validates that the three GPUDirect RDMA capability attributes are queryable on
 *    every device and that each reported value is inside its defined domain:
 *    -# hipDeviceAttributeGPUDirectRDMASupported is a boolean
 *    -# hipDeviceAttributeGPUDirectRDMAFlushWritesOptions is a subset of the
 *       hipFlushGPUDirectRDMAWritesOptions mask
 *    -# hipDeviceAttributeGPUDirectRDMAWritesOrdering is one of None/Owner/AllDevices
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_AttributeDomains) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
    return;
  }

  for (int device = 0; device < numDevices; ++device) {
    const RdmaCaps caps = GetRdmaCaps(device);
    INFO("device " << device << " supported=" << caps.supported << " flushOptions="
                   << caps.flushOptions << " writesOrdering=" << caps.writesOrdering);

    REQUIRE((caps.supported == 0 || caps.supported == 1));
    REQUIRE((caps.flushOptions & ~kFlushOptionsMask) == 0);
    REQUIRE((caps.writesOrdering == hipGPUDirectRDMAWritesOrderingNone ||
             caps.writesOrdering == hipGPUDirectRDMAWritesOrderingOwner ||
             caps.writesOrdering == hipGPUDirectRDMAWritesOrderingAllDevices));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates that hipDeviceGetAttribute and hipGetDeviceProperties report identical
 *    values for the three GPUDirect RDMA capabilities. Callers use them
 *    interchangeably, so a divergence is a defect regardless of which value is right.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_AttributesMatchProperties) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
    return;
  }

  for (int device = 0; device < numDevices; ++device) {
    const RdmaCaps caps = GetRdmaCaps(device);
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    INFO("device " << device);
    REQUIRE(caps.supported == props.gpuDirectRDMASupported);
    REQUIRE(static_cast<unsigned int>(caps.flushOptions) == props.gpuDirectRDMAFlushWritesOptions);
    REQUIRE(caps.writesOrdering == props.gpuDirectRDMAWritesOrdering);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates the internal consistency of the advertised capabilities. A device that
 *    claims GPUDirect RDMA support must offer some way to make an inbound remote write
 *    visible to the owning device, which is either:
 *    -# native ordering at owner scope or stronger, or
 *    -# an explicit flush path (host and/or memory-operation based)
 *  - A device reporting supported=1, ordering=None and flushOptions=0 advertises RDMA
 *    that no caller can ever safely consume.
 *  - The converse is legal and observed: a GeForce RTX 5090 reports supported=0 with
 *    flushOptions=Host and ordering=Owner. The flush attributes describe the write path,
 *    not the RDMA feature, so this test only constrains the supported=1 case.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_CapabilityConsistency) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
    return;
  }

  for (int device = 0; device < numDevices; ++device) {
    const RdmaCaps caps = GetRdmaCaps(device);
    if (caps.supported == 0) {
      continue;
    }
    INFO("device " << device << " claims GPUDirect RDMA support with flushOptions="
                   << caps.flushOptions << " writesOrdering=" << caps.writesOrdering);
    REQUIRE((caps.writesOrdering != hipGPUDirectRDMAWritesOrderingNone || caps.flushOptions != 0));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates the basic flush call against the current device for both scopes:
 *    -# hipFlushGPUDirectRDMAWritesToOwner
 *    -# hipFlushGPUDirectRDMAWritesToAllDevices
 *  - Expected output: `hipSuccess` when the device advertises the host flush option,
 *    `hipErrorNotSupported` otherwise.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_Positive_Basic) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  const hipError_t expected = ExpectedFlushStatus(device);

  SECTION("scope is ToOwner") {
    HIP_CHECK_ERROR(hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                                hipFlushGPUDirectRDMAWritesToOwner),
                    expected);
  }

  SECTION("scope is ToAllDevices") {
    HIP_CHECK_ERROR(hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                                hipFlushGPUDirectRDMAWritesToAllDevices),
                    expected);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates that the flush is repeatable and leaves no residual state. The call is
 *    issued in a tight loop and normal device work is then exercised to confirm the
 *    context is still healthy.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_Positive_Repeated) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  if (!HostFlushSupported(device)) {
    HIP_SKIP_TEST("host GPUDirect RDMA flush is not supported on this device.");
    return;
  }

  constexpr int kIterations = 1024;
  for (int i = 0; i < kIterations; ++i) {
    HIP_CHECK(hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                          hipFlushGPUDirectRDMAWritesToOwner));
  }

  constexpr size_t kElements = 4096;
  int* deviceBuffer = nullptr;
  HIP_CHECK(hipMalloc(&deviceBuffer, kElements * sizeof(int)));
  std::vector<int> hostBuffer(kElements);
  for (size_t i = 0; i < kElements; ++i) {
    hostBuffer[i] = static_cast<int>(i);
  }
  HIP_CHECK(hipMemcpy(deviceBuffer, hostBuffer.data(), kElements * sizeof(int),
                      hipMemcpyHostToDevice));
  std::vector<int> readBack(kElements, -1);
  HIP_CHECK(
      hipMemcpy(readBack.data(), deviceBuffer, kElements * sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(deviceBuffer));

  REQUIRE(readBack == hostBuffer);
}

/**
 * Test Description
 * ------------------------
 *  - Validates that the flush applies to the current device and succeeds after every
 *    hipSetDevice on a multi-GPU system.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - Multi-GPU system
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_Positive_MultiDevice) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
    return;
  }

  int originalDevice = 0;
  HIP_CHECK(hipGetDevice(&originalDevice));

  for (int device = 0; device < numDevices; ++device) {
    HIP_CHECK(hipSetDevice(device));
    INFO("current device " << device);
    HIP_CHECK_ERROR(hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                                hipFlushGPUDirectRDMAWritesToOwner),
                    ExpectedFlushStatus(device));
  }

  HIP_CHECK(hipSetDevice(originalDevice));
}

/**
 * Test Description
 * ------------------------
 *  - Validates that the flush is safe to call concurrently from multiple host threads
 *    sharing a device, as an RDMA receive path routinely would.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_Positive_MultiThreaded) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  if (!HostFlushSupported(device)) {
    HIP_SKIP_TEST("host GPUDirect RDMA flush is not supported on this device.");
    return;
  }

  constexpr int kThreads = 8;
  constexpr int kIterationsPerThread = 256;
  std::vector<hipError_t> results(kThreads, hipSuccess);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([device, t, &results]() {
      hipError_t status = hipSetDevice(device);
      for (int i = 0; i < kIterationsPerThread && status == hipSuccess; ++i) {
        status = hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                             hipFlushGPUDirectRDMAWritesToOwner);
      }
      results[t] = status;
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  for (int t = 0; t < kThreads; ++t) {
    INFO("thread " << t << " returned " << hipGetErrorString(results[t]));
    REQUIRE(results[t] == hipSuccess);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates that the flush is a memory-visibility barrier and not a synchronization
 *    point. A kernel is kept resident by spinning on a host-written flag; the flush must
 *    return while that kernel is still executing and must leave the stream incomplete.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - Host memory mapping support
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_Positive_DoesNotSynchronize) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  if (!HostFlushSupported(device)) {
    HIP_SKIP_TEST("host GPUDirect RDMA flush is not supported on this device.");
    return;
  }

  int* flag = nullptr;
  HIP_CHECK(hipHostMalloc(&flag, sizeof(int), hipHostMallocMapped));
  *flag = 0;

  int* deviceFlag = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&deviceFlag), flag, 0));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  SpinUntilFlag<<<1, 1, 0, stream>>>(deviceFlag);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                        hipFlushGPUDirectRDMAWritesToOwner));

  // The kernel cannot have retired: nothing has released it yet.
  HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);

  std::atomic_thread_fence(std::memory_order_release);
  *static_cast<volatile int*>(flag) = 1;
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipHostFree(flag));
}

/**
 * Test Description
 * ------------------------
 *  - Validates that data already resident in device memory stays intact and readable by
 *    a kernel across a flush. This is a portable proxy for the RDMA receive sequence
 *    (write payload, flush, consume) that does not require an RDMA-capable NIC; it does
 *    not prove inbound remote-write visibility, only that the flush is non-destructive
 *    and correctly ordered before subsequent kernel launches.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_Positive_PayloadVisibleToKernel) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  if (!HostFlushSupported(device)) {
    HIP_SKIP_TEST("host GPUDirect RDMA flush is not supported on this device.");
    return;
  }

  constexpr size_t kElements = 1 << 16;
  std::vector<int> hostBuffer(kElements);
  for (size_t i = 0; i < kElements; ++i) {
    hostBuffer[i] = static_cast<int>(i);
  }

  int* payload = nullptr;
  int* mismatches = nullptr;
  HIP_CHECK(hipMalloc(&payload, kElements * sizeof(int)));
  HIP_CHECK(hipMalloc(&mismatches, sizeof(int)));
  HIP_CHECK(hipMemset(mismatches, 0, sizeof(int)));
  HIP_CHECK(
      hipMemcpy(payload, hostBuffer.data(), kElements * sizeof(int), hipMemcpyHostToDevice));

  HIP_CHECK(hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                        hipFlushGPUDirectRDMAWritesToOwner));

  CheckPattern<<<64, 256>>>(payload, kElements, mismatches);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int mismatchCount = -1;
  HIP_CHECK(hipMemcpy(&mismatchCount, mismatches, sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(mismatches));
  HIP_CHECK(hipFree(payload));

  REQUIRE(mismatchCount == 0);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments. Every enumerator outside the defined set
 *    must be rejected with `hipErrorInvalidValue`, including the small values adjacent
 *    to the valid ones. The scope enumerators are 100 and 200, so a range check written
 *    as `scope < 0 || scope > 200` would wrongly accept 0, 1 and 99.
 *  - Argument validation is required to precede the capability check, so an out-of-range
 *    enumerator never reports `hipErrorNotSupported`.
 *  - Matches observed CUDA 13.1 behaviour on an RDMA-capable device: every one of these
 *    combinations returns `CUDA_ERROR_INVALID_VALUE`.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_Negative_Parameters) {
  using Target = hipFlushGPUDirectRDMAWritesTarget;
  using Scope = hipFlushGPUDirectRDMAWritesScope;

  const Target goodTarget = hipFlushGPUDirectRDMAWritesTargetCurrentDevice;
  const Scope goodScope = hipFlushGPUDirectRDMAWritesToOwner;

  SECTION("scope is zero") {
    HIP_CHECK_ERROR(hipDeviceFlushGPUDirectRDMAWrites(goodTarget, static_cast<Scope>(0)),
                    hipErrorInvalidValue);
    (void)hipGetLastError();
  }

  SECTION("scope is one") {
    HIP_CHECK_ERROR(hipDeviceFlushGPUDirectRDMAWrites(goodTarget, static_cast<Scope>(1)),
                    hipErrorInvalidValue);
    (void)hipGetLastError();
  }

  SECTION("scope is just below ToOwner") {
    HIP_CHECK_ERROR(hipDeviceFlushGPUDirectRDMAWrites(goodTarget, static_cast<Scope>(99)),
                    hipErrorInvalidValue);
    (void)hipGetLastError();
  }

  SECTION("scope is negative") {
    HIP_CHECK_ERROR(hipDeviceFlushGPUDirectRDMAWrites(goodTarget, static_cast<Scope>(-1)),
                    hipErrorInvalidValue);
    (void)hipGetLastError();
  }

  SECTION("scope is far out of range") {
    HIP_CHECK_ERROR(hipDeviceFlushGPUDirectRDMAWrites(goodTarget, static_cast<Scope>(0x7fff)),
                    hipErrorInvalidValue);
    (void)hipGetLastError();
  }

  SECTION("target is one past the only valid enumerator") {
    HIP_CHECK_ERROR(hipDeviceFlushGPUDirectRDMAWrites(static_cast<Target>(1), goodScope),
                    hipErrorInvalidValue);
    (void)hipGetLastError();
  }

  SECTION("target is negative") {
    HIP_CHECK_ERROR(hipDeviceFlushGPUDirectRDMAWrites(static_cast<Target>(-1), goodScope),
                    hipErrorInvalidValue);
    (void)hipGetLastError();
  }

  SECTION("target is far out of range") {
    HIP_CHECK_ERROR(hipDeviceFlushGPUDirectRDMAWrites(static_cast<Target>(0x7fff), goodScope),
                    hipErrorInvalidValue);
    (void)hipGetLastError();
  }

  SECTION("target and scope are out of range") {
    HIP_CHECK_ERROR(
        hipDeviceFlushGPUDirectRDMAWrites(static_cast<Target>(0x7fff), static_cast<Scope>(0x7fff)),
        hipErrorInvalidValue);
    (void)hipGetLastError();
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates the error accounting of a rejected flush:
 *    -# the error is returned to the caller
 *    -# it is recorded in the per-thread last-error state, as every runtime entry point does
 *    -# reading the last error clears it
 *    -# the context is otherwise undamaged and subsequent device work succeeds
 *  - Matches observed CUDA 13.1 behaviour: `cudaGetLastError` reports
 *    `cudaErrorInvalidValue` after a rejected flush, while the following `cudaMalloc`
 *    still returns `cudaSuccess`.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_Negative_FailureSetsLastErrorOnly) {
  // Discard, not HIP_CHECK: a sibling test may have left a sticky error and this case must
  // start from a known-clean state rather than assert on what it inherited.
  (void)hipGetLastError();

  HIP_CHECK_ERROR(
      hipDeviceFlushGPUDirectRDMAWrites(static_cast<hipFlushGPUDirectRDMAWritesTarget>(0x7fff),
                                        hipFlushGPUDirectRDMAWritesToOwner),
      hipErrorInvalidValue);

  // The failure is recorded, and reading it clears it.
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());

  int* buffer = nullptr;
  HIP_CHECK(hipMalloc(&buffer, 256));
  HIP_CHECK(hipMemset(buffer, 0, 256));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipFree(buffer));
}

/**
 * Test Description
 * ------------------------
 *  - Validates that the flush works against a freshly recreated primary context. After
 *    `hipDeviceReset` destroys the context, the next flush must transparently reinitialise
 *    rather than fail.
 *  - Matches observed CUDA 13.1 behaviour: `cudaSuccess` immediately after
 *    `cudaDeviceReset`. Note that the driver-level entry point instead reports
 *    `CUDA_ERROR_CONTEXT_IS_DESTROYED` in the same situation; the runtime-level contract
 *    is the one HIP mirrors.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_Positive_AfterDeviceReset) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  const hipError_t expected = ExpectedFlushStatus(device);

  HIP_CHECK(hipDeviceReset());

  HIP_CHECK_ERROR(hipDeviceFlushGPUDirectRDMAWrites(
                      hipFlushGPUDirectRDMAWritesTargetCurrentDevice,
                      hipFlushGPUDirectRDMAWritesToOwner),
                  expected);
  // Discard rather than assert: on a device with no flush path `expected` is an error and
  // would legitimately be sticky here.
  (void)hipGetLastError();

  // The recreated context must be usable.
  int* buffer = nullptr;
  HIP_CHECK(hipMalloc(&buffer, 256));
  HIP_CHECK(hipFree(buffer));
}

/**
 * Test Description
 * ------------------------
 *  - Validates behaviour inside a stream capture region. The flush is a host-side
 *    visibility barrier with no stream ordering, so it must:
 *    -# succeed while a capture is active
 *    -# not invalidate the capture
 *    -# contribute no node to the resulting graph
 *  - Matches observed CUDA 13.1 behaviour: success during capture, successful
 *    `cudaStreamEndCapture`, and a graph with zero nodes.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_Positive_NotCapturedByGraph) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  if (!HostFlushSupported(device)) {
    HIP_SKIP_TEST("host GPUDirect RDMA flush is not supported on this device.");
    return;
  }

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));

  HIP_CHECK(hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                        hipFlushGPUDirectRDMAWritesToOwner));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);

  size_t numNodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  INFO("captured graph node count " << numNodes);
  REQUIRE(numNodes == 0);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid device ordinals when querying the capability
 *    attributes:
 *    -# When the device ordinal is negative
 *    -# When the device ordinal is out of bounds
 *    -# When the output pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue` / `hipErrorInvalidDevice`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipDeviceFlushGPUDirectRDMAWrites_Negative_AttributeQuery) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
    return;
  }

  int value = 0;
  const hipDeviceAttribute_t attributes[] = {hipDeviceAttributeGPUDirectRDMASupported,
                                             hipDeviceAttributeGPUDirectRDMAFlushWritesOptions,
                                             hipDeviceAttributeGPUDirectRDMAWritesOrdering};

  for (const auto attribute : attributes) {
    INFO("attribute " << static_cast<int>(attribute));

    HIP_CHECK_ERROR(hipDeviceGetAttribute(&value, attribute, -1), hipErrorInvalidDevice);
    HIP_CHECK_ERROR(hipDeviceGetAttribute(&value, attribute, numDevices), hipErrorInvalidDevice);
    HIP_CHECK_ERROR(hipDeviceGetAttribute(nullptr, attribute, 0), hipErrorInvalidValue);
    (void)hipGetLastError();
  }
}

/**
 * End doxygen group DeviceTest.
 * @}
 */
