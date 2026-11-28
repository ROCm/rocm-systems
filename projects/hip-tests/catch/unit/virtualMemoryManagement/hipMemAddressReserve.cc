/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemAddressReserve hipMemAddressReserve
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemAddressReserve (void** ptr,
 *                                   size_t size,
 *                                   size_t alignment,
 *                                   void* addr,
 *                                   unsigned long long flags)` -
 * Reserves an address range.
 */

#include <hip_test_common.hh>

#include "hip_vmm_common.hh"

#define DATA_SIZE (1 << 13)

/**
 * Test Description
 * ------------------------
 *    - Verify if reserved address returned by hipMemAddressReserve
 * for different alignment values are correctly aligned.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemAddressReserve.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemAddressReserve_AlignmentTest) {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  constexpr int initializer = 0;
  int deviceId = 0;
  hipDevice_t device;
  CTX_CREATE();
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  // Allocate virtual address range
  void* ptrA;
  size_t alignmnt = 1;
  hipMemGenericAllocationHandle_t handle;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  // check for address alignment fron 2 to 1024
  for (int iter = 0; iter < 12; iter++) {
    alignmnt = alignmnt * 2;
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, alignmnt, 0, 0));
    REQUIRE((reinterpret_cast<unsigned long long>(ptrA) % alignmnt) == 0);
    std::fill(B_h.begin(), B_h.end(), initializer);
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
    // Set access
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = device;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    // Make the address accessible to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  }
  HIP_CHECK(hipMemRelease(handle));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Negative Tests
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemAddressReserve.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemAddressReserve_Negative) {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
  hipDevice_t device;
  CTX_CREATE();
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  // Allocate virtual address range
  void* ptrA;

  SECTION("Nullptr to ptr") {
    REQUIRE(hipMemAddressReserve(nullptr, size_mem, 0, 0, 0) == hipErrorInvalidValue);
  }

  SECTION("pass size as 0") {
    REQUIRE(hipMemAddressReserve(&ptrA, 0, 0, 0, 0) == hipErrorInvalidValue);
  }

  SECTION("pass non power of two for alignment") {
    REQUIRE(hipMemAddressReserve(&ptrA, size_mem, 3, 0, 0) == hipErrorInvalidValue);
  }

  SECTION("pass size as non multiple of host page size") {
    REQUIRE(hipMemAddressReserve(&ptrA, (size_mem - 1), 0, 0, 0) == hipErrorInvalidValue);
  }

  CTX_DESTROY();
}

HIP_TEST_CASE(Unit_hipMemAddressReserve_Capture) {
  hipMemGenericAllocationHandle_t allocation_handle;
  size_t granularity = 0;
  constexpr size_t kAlignment = 2;
  constexpr int kDeviceId = 0;
  hipDevice_t device = 0;
  void* device_ptr = nullptr;

  HIP_TEST_DRIVER_INIT();
  checkVMMSupported(kDeviceId);
  CTX_CREATE();
  HIP_CHECK(hipDeviceGet(&device, kDeviceId));

  hipMemAllocationProp allocation_prop{};
  allocation_prop.type = hipMemAllocationTypePinned;
  allocation_prop.location.type = hipMemLocationTypeDevice;
  allocation_prop.location.id = device;

  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &allocation_prop,
                                           hipMemAllocationGranularityMinimum));
  HIP_CHECK(hipMemCreate(&allocation_handle, granularity, &allocation_prop, 0));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemAddressReserve(&device_ptr, granularity, kAlignment, nullptr, 0));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemAddressFree(device_ptr, granularity));
  HIP_CHECK(hipMemRelease(allocation_handle));
  CTX_DESTROY();
}

#if HT_AMD
namespace {
constexpr size_t kExhaustChunkSize = 256ull << 30;  // 256 GiB
constexpr int kExhaustMaxChunks = 1024;             // 256 TiB, more than any VA aperture

// The kExhaustChunkSize ranges held while the address space is exhausted. Free() releases them
// and checks each call; the destructor releases what is left if a check threw before that.
class ReservedRanges {
 public:
  ReservedRanges() { ptrs_.reserve(kExhaustMaxChunks); }
  ReservedRanges(const ReservedRanges&) = delete;
  ReservedRanges& operator=(const ReservedRanges&) = delete;
  ~ReservedRanges() {
    for (void* p : ptrs_) {
      (void)hipMemAddressFree(p, kExhaustChunkSize);
    }
  }
  void Add(void* p) { ptrs_.push_back(p); }
  size_t Count() const { return ptrs_.size(); }
  void Free() {
    while (!ptrs_.empty()) {
      void* p = ptrs_.back();
      ptrs_.pop_back();
      HIP_CHECK(hipMemAddressFree(p, kExhaustChunkSize));
    }
  }

 private:
  std::vector<void*> ptrs_;
};

// Reserves kExhaustChunkSize ranges until hipMemAddressReserve fails or kExhaustMaxChunks are
// held. Returns the status of the last call and sets *last_ptr to its pointer; the ranges
// reserved so far stay in reserved. After a failure no kExhaustChunkSize range fits any more.
hipError_t ExhaustDeviceVa(ReservedRanges& reserved, void** last_ptr) {
  // No allocation may be needed while the address space is exhausted (Add() does not allocate)
  hipError_t status = hipSuccess;
  for (int i = 0; i < kExhaustMaxChunks; ++i) {
    *last_ptr = reinterpret_cast<void*>(0x1);
    status = hipMemAddressReserve(last_ptr, kExhaustChunkSize, 0, nullptr, 0);
    if (status != hipSuccess) {
      break;
    }
    reserved.Add(*last_ptr);
  }
  return status;
}

// The stream, graph and graph exec a test holds. The destructor releases what is still held and
// trims graph memory, so a check that throws does not leak them into later tests in the process.
class GraphResources {
 public:
  GraphResources() { HIP_CHECK(hipStreamCreate(&stream)); }
  GraphResources(const GraphResources&) = delete;
  GraphResources& operator=(const GraphResources&) = delete;
  ~GraphResources() {
    if (exec != nullptr) {
      (void)hipGraphExecDestroy(exec);
    }
    if (graph != nullptr) {
      (void)hipGraphDestroy(graph);
    }
    (void)hipStreamDestroy(stream);
    (void)hipDeviceGraphMemTrim(0);
  }
  // Destroys the graph and its exec, checking both calls.
  void DestroyGraph() {
    if (exec != nullptr) {
      HIP_CHECK(hipGraphExecDestroy(exec));
      exec = nullptr;
    }
    HIP_CHECK(hipGraphDestroy(graph));
    graph = nullptr;
  }

  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
};

bool MemPoolsSupported() {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  return mem_pool_support != 0;
}
}  // namespace
#endif

/**
 * Test Description
 * ------------------------
 *    - Reserve large address ranges until the device virtual address space
 *      is exhausted. The failing call must return hipErrorOutOfMemory and
 *      leave *ptr as nullptr instead of crashing, and the ranges reserved
 *      before it must still be freeable.
 *    - AMD on Windows only: there the range is the PAL SVM range (4 TiB), or
 *      for ROCr on Windows a device heap sized from the GPU's memory, which
 *      may be smaller than one 256 GiB reservation. On
 *      Linux the ROCr reservation also takes CPU address space, so exhausting
 *      it exhausts the whole process address space, and CUDA does not promise
 *      to clear *ptr on failure.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemAddressReserve.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipMemAddressReserve_Exhaustion) {
#if !(HT_AMD && HT_WIN)
  HIP_SKIP_TEST("needs the AMD Windows (PAL) runtime");
#else
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  ReservedRanges reserved;
  void* ptr = nullptr;
  const hipError_t status = ExhaustDeviceVa(reserved, &ptr);
  const size_t num_reserved = reserved.Count();
  reserved.Free();
  INFO("Reserved " << num_reserved << " x 256 GiB before the first failure");
  if (status == hipSuccess) {
    HIP_SKIP_TEST("virtual address space was not exhausted");
  }
  REQUIRE(status == hipErrorOutOfMemory);
  REQUIRE(ptr == nullptr);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorOutOfMemory);

  // Once the ranges are freed the address space is usable again. When even the first range did
  // not fit (ROCr on Windows reserves from a device heap sized from the GPU's memory, 200 GiB on
  // some GPUs), there is nothing to take back.
  if (num_reserved > 0) {
    HIP_CHECK(hipMemAddressReserve(&ptr, kExhaustChunkSize, 0, nullptr, 0));
    HIP_CHECK(hipMemAddressFree(ptr, kExhaustChunkSize));
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - With the device virtual address space exhausted, hipMallocAsync captured
 *      in a stream capture must return hipErrorOutOfMemory and a null pointer,
 *      not hipSuccess with a null pointer. The capture stays active and ends
 *      with an empty graph, and once the address space is freed the stream
 *      captures and runs a graph allocation again.
 *    - AMD only (Windows and Linux).
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemAddressReserve.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipMemAddressReserve_Exhaustion_CaptureMallocAsync) {
#if !HT_AMD
  HIP_SKIP_TEST("needs the AMD runtime");
#else
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);
  if (!MemPoolsSupported()) {
    HIP_SKIP_TEST("Runtime doesn't support Memory Pool. Skip the test case.");
  }
  GraphResources res;

  ReservedRanges reserved;
  void* ptr = nullptr;
  if (ExhaustDeviceVa(reserved, &ptr) == hipSuccess) {
    reserved.Free();
    HIP_SKIP_TEST("virtual address space was not exhausted");
  }
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorOutOfMemory);  // clear the failed reserve's error
  HIP_CHECK(hipStreamBeginCapture(res.stream, hipStreamCaptureModeGlobal));
  void* dev_ptr = reinterpret_cast<void*>(0x1);
  const hipError_t alloc_status = hipMallocAsync(&dev_ptr, kExhaustChunkSize, res.stream);
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  HIP_CHECK(hipStreamIsCapturing(res.stream, &capture_status));
  HIP_CHECK(hipStreamEndCapture(res.stream, &res.graph));
  reserved.Free();

  REQUIRE(alloc_status == hipErrorOutOfMemory);
  REQUIRE(dev_ptr == nullptr);
  REQUIRE(capture_status == hipStreamCaptureStatusActive);
  REQUIRE(res.graph != nullptr);
  size_t num_nodes = 1;
  HIP_CHECK(hipGraphGetNodes(res.graph, nullptr, &num_nodes));
  REQUIRE(num_nodes == 0);
  res.DestroyGraph();

  // Once the ranges are freed the stream captures and runs a graph allocation again
  constexpr size_t kSmallSize = 1 << 20;
  HIP_CHECK(hipStreamBeginCapture(res.stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMallocAsync(&dev_ptr, kSmallSize, res.stream));
  REQUIRE(dev_ptr != nullptr);
  HIP_CHECK(hipMemsetAsync(dev_ptr, 0, kSmallSize, res.stream));
  HIP_CHECK(hipFreeAsync(dev_ptr, res.stream));
  HIP_CHECK(hipStreamEndCapture(res.stream, &res.graph));
  HIP_CHECK(hipGraphInstantiate(&res.exec, res.graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(res.exec, res.stream));
  HIP_CHECK(hipStreamSynchronize(res.stream));
  res.DestroyGraph();
#endif
}

/**
 * Test Description
 * ------------------------
 *    - With the device virtual address space exhausted, adding a graph memory
 *      allocation node with hipGraphAddNode (hipGraphNodeTypeMemAlloc) or with
 *      hipGraphAddMemAllocNode must return hipErrorOutOfMemory and a null
 *      pointer, not hipSuccess with a null pointer, and leave no node in the
 *      graph. Once the address space is freed the graph takes and runs an
 *      allocation again.
 *    - AMD only (Windows and Linux).
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemAddressReserve.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipMemAddressReserve_Exhaustion_GraphMemAllocNode) {
#if !HT_AMD
  HIP_SKIP_TEST("needs the AMD runtime");
#else
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);
  if (!MemPoolsSupported()) {
    HIP_SKIP_TEST("Runtime doesn't support Memory Pool. Skip the test case.");
  }
  GraphResources res;
  HIP_CHECK(hipGraphCreate(&res.graph, 0));

  hipMemAllocNodeParams alloc_params{};
  alloc_params.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_params.poolProps.location.type = hipMemLocationTypeDevice;
  alloc_params.poolProps.location.id = 0;
  alloc_params.bytesize = kExhaustChunkSize;

  ReservedRanges reserved;
  void* ptr = nullptr;
  if (ExhaustDeviceVa(reserved, &ptr) == hipSuccess) {
    reserved.Free();
    HIP_SKIP_TEST("virtual address space was not exhausted");
  }
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorOutOfMemory);  // clear the failed reserve's error
  hipGraphNode_t node = nullptr;
  hipError_t status = hipSuccess;
  void* dev_ptr = nullptr;
  SECTION("hipGraphAddNode") {
    hipGraphNodeParams node_params{};
    node_params.type = hipGraphNodeTypeMemAlloc;
    node_params.alloc = alloc_params;
    node_params.alloc.dptr = reinterpret_cast<void*>(0x1);
    status = hipGraphAddNode(&node, res.graph, nullptr, 0, &node_params);
    dev_ptr = node_params.alloc.dptr;
  }
  SECTION("hipGraphAddMemAllocNode") {
    hipMemAllocNodeParams params = alloc_params;
    params.dptr = reinterpret_cast<void*>(0x1);
    status = hipGraphAddMemAllocNode(&node, res.graph, nullptr, 0, &params);
    dev_ptr = params.dptr;
  }
  reserved.Free();

  REQUIRE(status == hipErrorOutOfMemory);
  REQUIRE(dev_ptr == nullptr);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorOutOfMemory);
  size_t num_nodes = 1;
  HIP_CHECK(hipGraphGetNodes(res.graph, nullptr, &num_nodes));
  REQUIRE(num_nodes == 0);

  // Once the ranges are freed the graph takes and runs an allocation again
  alloc_params.bytesize = 1 << 20;
  HIP_CHECK(hipGraphAddMemAllocNode(&node, res.graph, nullptr, 0, &alloc_params));
  REQUIRE(alloc_params.dptr != nullptr);
  hipGraphNode_t free_node = nullptr;
  HIP_CHECK(hipGraphAddMemFreeNode(&free_node, res.graph, &node, 1, alloc_params.dptr));
  HIP_CHECK(hipGraphInstantiate(&res.exec, res.graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(res.exec, res.stream));
  HIP_CHECK(hipStreamSynchronize(res.stream));
  res.DestroyGraph();
#endif
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
