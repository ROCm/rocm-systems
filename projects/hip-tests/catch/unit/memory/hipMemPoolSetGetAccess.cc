/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <utility>
#include <vector>
#include <resource_guards.hh>
#include "mempool_common.hh"

/**
 * @addtogroup hipMemPoolSetAccess hipMemPoolSetAccess
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc* desc_list, size_t count)`
 * - Controls visibility of the specified pool between devices
 */

__global__ void copyP2PAndScale(int* dst, const int* src, size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    // scale & store src vector.
    dst[idx] = 2 * src[idx];
  }
}

static void MemPoolSetGetAccess(const MemPools mempool_type, int src_device, int dst_device,
                                hipMemAccessFlags access_flags) {
  MemPoolGuard mempool(mempool_type, src_device);

  hipMemAccessDesc desc;
  memset(&desc, 0, sizeof(hipMemAccessDesc));
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = dst_device;
  desc.flags = access_flags;
  HIP_CHECK(hipMemPoolSetAccess(mempool.mempool(), &desc, 1));

  hipMemAccessFlags flags = hipMemAccessFlagsProtNone;
  HIP_CHECK(hipMemPoolGetAccess(&flags, mempool.mempool(), &desc.location));
  REQUIRE(flags == access_flags);
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify hipMemPoolSetAccess/hipMemPoolGetAccess on a single device.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolSetGetAccess_Positive_Basic) {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));

  checkMempoolSupported(device)

      const auto mempool_type = GENERATE(MemPools::dev_default, MemPools::created);

  MemPoolSetGetAccess(mempool_type, device, device, hipMemAccessFlagsProtReadWrite);
}

int CheckP2PMemPoolSupport(int src_device, int dst_device) {
  int mem_pool_support = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, src_device));
  if (mem_pool_support) {
    HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported,
                                    dst_device));
  }
  return mem_pool_support;
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify hipMemPoolSetAccess/hipMemPoolGetAccess on multiple devices.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolSetGetAccess_Positive_MultipleGPU) {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  const auto src_device = GENERATE(range(0, HipTest::getDeviceCount()));
  const auto dst_device = GENERATE(range(0, HipTest::getDeviceCount()));
  INFO("Src device: " << src_device << ", Dst device: " << dst_device);

  int mem_pool_support = CheckP2PMemPoolSupport(src_device, dst_device);
  if (!mem_pool_support) {
    HIP_SKIP_TEST(HipTest::SkipReason::kMemoryPoolUnsupported);
  }

  const auto mempool_type = GENERATE(MemPools::dev_default, MemPools::created);
  const auto access_flag = hipMemAccessFlagsProtReadWrite;

  int can_access_peer = 0;
  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    MemPoolSetGetAccess(mempool_type, src_device, dst_device, access_flag);
  }
}

void MemPoolSetGetAccess_P2P(const MemPools mempool_type) {
  const auto src_device = GENERATE(range(0, HipTest::getDeviceCount()));
  const auto dst_device = GENERATE(range(0, HipTest::getDeviceCount()));
  INFO("Src device: " << src_device << ", Dst device: " << dst_device);

  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);

  int mem_pool_support = CheckP2PMemPoolSupport(src_device, dst_device);
  if (!mem_pool_support) {
    HIP_SKIP_TEST(HipTest::SkipReason::kMemoryPoolUnsupported);
  }

  int *alloc_mem1, *alloc_mem2;
  int can_access_peer = 0;
  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    hipEvent_t waitOnStream1;

    LinearAllocGuard<int> host_alloc(LinearAllocs::malloc, allocation_size);
    HIP_CHECK(hipEventCreate(&waitOnStream1))
    StreamGuard stream1(Streams::withFlags, hipStreamNonBlocking);
    // Get/create mempool for src_device
    MemPoolGuard mempool(mempool_type, src_device);

    // Allocate memory in a stream from the pool set above
    if (mempool_type == MemPools::dev_default) {
      HIP_CHECK(
          hipMallocAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size, stream1.stream()));
    } else {
      HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                       mempool.mempool(), stream1.stream()));
    }

    const auto element_count = allocation_size / sizeof(int);
    constexpr auto thread_count = 1024;
    const auto block_count = element_count / thread_count + 1;
    constexpr int expected_value = 15;
    VectorSet<<<block_count, thread_count, 0, stream1.stream()>>>(alloc_mem1, expected_value,
                                                                  element_count);
    HIP_CHECK(hipEventRecord(waitOnStream1, stream1.stream()));

    HIP_CHECK(hipSetDevice(dst_device));
    StreamGuard stream2(Streams::withFlags, hipStreamNonBlocking);

    // Allocate memory in dst device
    HIP_CHECK(
        hipMallocAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size, stream2.stream()));

    // Setup peer mappings for dst device
    hipMemAccessDesc desc;
    memset(&desc, 0, sizeof(hipMemAccessDesc));
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id = dst_device;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemPoolSetAccess(mempool.mempool(), &desc, 1));

    hipMemAccessFlags flags = hipMemAccessFlagsProtNone;
    HIP_CHECK(hipMemPoolGetAccess(&flags, mempool.mempool(), &desc.location));
    REQUIRE(flags == hipMemAccessFlagsProtReadWrite);

    HIP_CHECK(hipStreamWaitEvent(stream2.stream(), waitOnStream1, 0));
    copyP2PAndScale<<<block_count, thread_count, 0, stream2.stream()>>>(alloc_mem2, alloc_mem1,
                                                                        element_count);

    HIP_CHECK(hipMemcpyAsync(host_alloc.host_ptr(), alloc_mem2, allocation_size,
                             hipMemcpyDeviceToHost, stream2.stream()));
    HIP_CHECK(hipFreeAsync(alloc_mem1, stream2.stream()));
    HIP_CHECK(hipFreeAsync(alloc_mem2, stream2.stream()));
    HIP_CHECK(hipStreamSynchronize(stream2.stream()));
    HIP_CHECK(hipEventDestroy(waitOnStream1));

    ArrayFindIfNot(host_alloc.host_ptr(), 2 * expected_value, element_count);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify peer-to-peer access of stream ordered memory with hipMemPoolSetAccess.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolSetGetAccess_Positive_P2P) {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  SECTION("Default MemPool") { MemPoolSetGetAccess_P2P(MemPools::dev_default); }

  SECTION("Created MemPool") { MemPoolSetGetAccess_P2P(MemPools::created); }
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMemPoolSetAccess behavior with invalid arguments:
 *    -# Nullptr mem_pool
 *    -# Desc is nullptr and count is > 0
 *    -# Count > num_device
 *    -# Invalid desc location type
 *    -# Invalid desc location id
 *    -# Revoking access to own memory pool
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolSetAccess_Negative_Parameters) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));
  checkMempoolSupported(device_id) MemPoolGuard mempool(MemPools::dev_default, device_id);

  int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev));

  hipMemAccessDesc desc;
  memset(&desc, 0, sizeof(hipMemAccessDesc));
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = device_id;
  desc.flags = hipMemAccessFlagsProtReadWrite;

  SECTION("Mempool is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolSetAccess(nullptr, &desc, 1), hipErrorInvalidValue);
  }

  // Cuda segfaults here!
#if HT_AMD
  SECTION("Desc is nullptr and count is > 0") {
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), nullptr, 1), hipErrorInvalidValue);
  }
#endif

  SECTION("Count > num_device") {
#if HT_AMD
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), &desc, (num_dev + 1)),
                    hipErrorInvalidDevice);
#else
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), &desc, (num_dev + 1)),
                    hipErrorNotSupported);
#endif
  }

  SECTION("Passing invalid desc location type") {
    desc.location.type = hipMemLocationTypeInvalid;
#if HT_AMD
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), &desc, 1), hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), &desc, 1), hipErrorNotSupported);
#endif
  }

  SECTION("Passing invalid desc location id") {
    desc.location.id = num_dev;
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), &desc, 1), hipErrorInvalidDevice);
  }

  SECTION("Revoking access to own memory pool") {
    desc.flags = hipMemAccessFlagsProtNone;
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), &desc, 1), hipErrorInvalidDevice);
  }
}

/**
 * Local function to test hipMemPoolSetAccess function.
 */
static bool checkMempoolSetAccess(int N, int dev0, int dev1) {
  // Set the current device context to dev0
  HIP_CHECK(hipSetDevice(dev0));
  // Create mempool in current device
  hipMemPool_t mem_pool;
  hipMemPoolProps pool_props{};
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = dev0;
  pool_props.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));

  int *A_h, *B_h, *C_h;
  size_t byte_size = N * sizeof(int);
  // assign memory to host pointers
  A_h = reinterpret_cast<int*>(malloc(byte_size));
  REQUIRE(A_h != nullptr);
  B_h = reinterpret_cast<int*>(malloc(byte_size));
  REQUIRE(B_h != nullptr);
  C_h = reinterpret_cast<int*>(malloc(byte_size));
  REQUIRE(C_h != nullptr);
  // set data to host
  for (int i = 0; i < N; i++) {
    A_h[i] = 2 * i + 1;  // Odd
    B_h[i] = 2 * i;      // Even
    C_h[i] = 0;
  }
  // create multiple streams
  hipStream_t stream0;
  HIP_CHECK(hipStreamCreate(&stream0));
  int *A_d0, *B_d0;
  // Allocate memory on dev0 and initialize it on stream0
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d0), byte_size, mem_pool, stream0));
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B_d0), byte_size, mem_pool, stream0));
  HIP_CHECK(hipMemcpyAsync(A_d0, A_h, byte_size, hipMemcpyHostToDevice, stream0));
  HIP_CHECK(hipMemcpyAsync(B_d0, B_h, byte_size, hipMemcpyHostToDevice, stream0));
  HIP_CHECK(hipStreamSynchronize(stream0));
  HIP_CHECK(hipStreamDestroy(stream0));
  // Set the current device context to dev1
  HIP_CHECK(hipSetDevice(dev1));
  // if withSetAccess is true set the access of mem_pool
  // to both dev0 and dev1.
  hipMemAccessDesc accessDesc;
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = dev1;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemPoolSetAccess(mem_pool, &accessDesc, 1));

  int *A_d1, *B_d1, *C_d1;
  hipStream_t stream1;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d1), byte_size, mem_pool, stream1));
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B_d1), byte_size, mem_pool, stream1));
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&C_d1), byte_size, mem_pool, stream1));
  HIP_CHECK(hipMemcpyAsync(A_d1, A_d0, byte_size, hipMemcpyDeviceToDevice, stream1));
  HIP_CHECK(hipMemcpyAsync(B_d1, B_d0, byte_size, hipMemcpyDeviceToDevice, stream1));
  // Launch Kernel on stream1
  hipLaunchKernelGGL(HipTest::vectorADD, dim3(N / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0,
                     stream1, static_cast<const int*>(A_d1), static_cast<const int*>(B_d1), C_d1,
                     N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d1, byte_size, hipMemcpyDeviceToHost, stream1));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d1), stream1));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B_d1), stream1));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C_d1), stream1));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d0), stream1));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B_d0), stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipStreamDestroy(stream1));
  // Set the current device context back to dev0
  HIP_CHECK(hipSetDevice(dev0));
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  // verify and validate
  for (int i = 0; i < N; i++) {
    REQUIRE(C_h[i] == (A_h[i] + B_h[i]));
  }
  free(A_h);
  free(B_h);
  free(C_h);
  return true;
}

/**
 * Local function to get pairs of devices.
 */
static void getDevicePairs(std::vector<std::pair<int, int>>* p2p_pairs, int numDevices) {
  for (int i = 0; i < (numDevices - 1); i++) {
    for (int j = i + 1; j < numDevices; j++) {
      std::pair<int, int> p2p_pair = std::make_pair(i, j);
      p2p_pairs->push_back(p2p_pair);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - P2P Access Scenario for mempool: Precondition: NUM OF GPUs >= 2
 * and P2P is enabled. Create explicit memory pool (mempool) on default GPU.
 * Allocate memory on device 0 and initialize it with data. Set current GPU
 * to device 1. Set the access of mempool to device 1. Allocate memory on
 * device 1 and transfer data from device 0 to device 1. Launch kernel to
 * perform vector add on the data. Validate the data. Destroy the mempool.
 * ------------------------
 *    - catch\unit\memory\hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolSetAccess_SetAccess) {
  constexpr int N = 1 << 14;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  checkIfMultiDev(numDevices) for (int dev = 0; dev < numDevices; dev++){
      checkMempoolSupported(dev)} std::vector<std::pair<int, int>>
      p2p_pairs;
  getDevicePairs(&p2p_pairs, numDevices);
  for (auto pair : p2p_pairs) {
    int canAccessPeer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, pair.first, pair.second));
    if (canAccessPeer) {
      REQUIRE(true == checkMempoolSetAccess(N, pair.first, pair.second));
    } else {
      WARN("P2P access not enabled between " << pair.first << " and " << pair.second << " .");
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - hipMemPoolSetAccess negative tests
 * ------------------------
 *    - catch\unit\memory\hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolSetAccess_NegTst) {
  checkMempoolSupported(0) hipMemPool_t mem_pool;
  hipMemPoolProps pool_props{};
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
  constexpr size_t count = 1;
  hipMemAccessDesc descList, descListNeg;
  descList.flags = hipMemAccessFlagsProtReadWrite;
  descList.location.type = hipMemLocationTypeDevice;
  descList.location.id = 0;
  // Scenario1
  SECTION("memPool NULL check") {
    REQUIRE(hipMemPoolSetAccess(nullptr, &descList, count) == hipErrorInvalidValue);
  }
  // Scenario2
  SECTION("Invalid Flag") {
    descListNeg.flags = static_cast<hipMemAccessFlags>(0xffff);
    descListNeg.location.type = hipMemLocationTypeDevice;
    descListNeg.location.id = 0;
    REQUIRE(hipMemPoolSetAccess(mem_pool, &descListNeg, count) == hipErrorInvalidValue);
  }
  // Scenario3
#if HT_AMD
  SECTION("Invalid location type") {
    descListNeg.flags = hipMemAccessFlagsProtReadWrite;
    descListNeg.location.type = hipMemLocationTypeInvalid;
    descListNeg.location.id = 0;
    REQUIRE(hipMemPoolSetAccess(mem_pool, &descListNeg, count) == hipErrorInvalidValue);
  }
#endif
  // Scenario4
  SECTION("Invalid device number") {
    descListNeg.flags = hipMemAccessFlagsProtReadWrite;
    descListNeg.location.type = hipMemLocationTypeDevice;
    descListNeg.location.id = -1;
    REQUIRE(hipMemPoolSetAccess(mem_pool, &descListNeg, count) == hipErrorInvalidDevice);
  }
  // Scenario5
  SECTION("Unavailable device number") {
    int num_devices = 0;
    HIP_CHECK(hipGetDeviceCount(&num_devices));
    descListNeg.flags = hipMemAccessFlagsProtReadWrite;
    descListNeg.location.type = hipMemLocationTypeDevice;
    descListNeg.location.id = num_devices;
    REQUIRE(hipMemPoolSetAccess(mem_pool, &descListNeg, count) == hipErrorInvalidDevice);
  }
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}

/**
 * End doxygen group hipMemPoolSetAccess.
 * @}
 */

/**
 * @addtogroup hipMemPoolGetAccess hipMemPoolGetAccess
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolGetAccess(hipMemAccessFlags* flags, hipMemPool_t mem_pool, hipMemLocation* location)`
 * - Returns the accessibility of a pool from a device
 */

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMemPoolGetAccess behavior with invalid arguments:
 *    -# Nullptr mem_pool
 *    -# Flags is nullptr
 *    -# Invalid location type
 *    -# Invalid location id
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolGetAccess_Negative_Parameters) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));
  checkMempoolSupported(device_id) MemPoolGuard mempool(MemPools::dev_default, device_id);

  int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev));

  hipMemAccessFlags flags = hipMemAccessFlagsProtNone;
  hipMemLocation location = {hipMemLocationTypeDevice, device_id};

  SECTION("Mempool is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolGetAccess(&flags, nullptr, &location), hipErrorInvalidValue);
  }
#if HT_AMD
  SECTION("Flags is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolGetAccess(nullptr, mempool.mempool(), &location),
                    hipErrorInvalidValue);
  }
#endif
  SECTION("Passing invalid location type") {
    location.type = hipMemLocationTypeInvalid;
    HIP_CHECK_ERROR(hipMemPoolGetAccess(&flags, mempool.mempool(), &location),
                    hipErrorInvalidValue);
    location.type = hipMemLocationTypeDevice;
  }

  SECTION("Passing invalid location id") {
    location.id = num_dev;
    HIP_CHECK_ERROR(hipMemPoolGetAccess(&flags, mempool.mempool(), &location),
                    hipErrorInvalidValue);
    location.id = device_id;
  }
}

/**
 * Local function to test hipMemPoolSetAccess/hipMemPoolGetAccess
 * function.
 */
static bool checkMempoolSetAccessWithGet(int dev0, int dev1) {
  // Create mempool in current device
  hipMemPool_t mem_pool;
  hipMemPoolProps pool_props{};
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = dev0;
  pool_props.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
  // Set access to dev1
  hipMemAccessDesc accessDesc;
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = dev1;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemPoolSetAccess(mem_pool, &accessDesc, 1));
  // Validate access for dev1
  hipMemAccessFlags flags;
  hipMemLocation location;
  location.type = hipMemLocationTypeDevice;
  location.id = dev1;
  HIP_CHECK(hipMemPoolGetAccess(&flags, mem_pool, &location));
  REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
  // Validate access for dev0
  location.id = dev0;
  HIP_CHECK(hipMemPoolGetAccess(&flags, mem_pool, &location));
  REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  return true;
}

static bool checkMempoolSetAccessWithGetUsingArray(int dev0, int dev1) {
  // Create mempool in current device
  hipMemPool_t mem_pool;
  hipMemPoolProps pool_props{};
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = dev0;
  pool_props.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
  // Set access of dev0 and dev1
  hipMemAccessDesc accessDesc[2];
  accessDesc[0].location.type = hipMemLocationTypeDevice;
  accessDesc[0].location.id = dev0;
  accessDesc[0].flags = hipMemAccessFlagsProtReadWrite;
  accessDesc[1].location.type = hipMemLocationTypeDevice;
  accessDesc[1].location.id = dev1;
  accessDesc[1].flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemPoolSetAccess(mem_pool, accessDesc, 2));
  // Validate access for dev0 and dev1
  hipMemAccessFlags flags;
  hipMemLocation location;
  location.type = hipMemLocationTypeDevice;
  location.id = dev0;
  HIP_CHECK(hipMemPoolGetAccess(&flags, mem_pool, &location));
  REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
  location.id = dev1;
  HIP_CHECK(hipMemPoolGetAccess(&flags, mem_pool, &location));
  REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  return true;
}

/**
 * Test Description
 * ------------------------
 *    - Validate hipMemPoolSetAccess with hipMemPoolGetAccess for all
 * devices on the system.
 * ------------------------
 *    - catch\unit\memory\hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolGetAccess_SetGet) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  checkIfMultiDev(numDevices) for (int dev = 0; dev < numDevices; dev++){
      checkMempoolSupported(dev)} std::vector<std::pair<int, int>>
      p2p_pairs;
  getDevicePairs(&p2p_pairs, numDevices);
  for (auto pair : p2p_pairs) {
    int canAccessPeer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, pair.first, pair.second));
    if (canAccessPeer) {
      REQUIRE(true == checkMempoolSetAccessWithGet(pair.first, pair.second));
      REQUIRE(true == checkMempoolSetAccessWithGetUsingArray(pair.first, pair.second));
    } else {
      WARN("P2P access not enabled between " << pair.first << " and " << pair.second << " .");
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Get the access of the default mempool of each device and verify
 * its value.
 * ------------------------
 *    - catch\unit\memory\hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolGetAccess_GetDefMempoolOfEachDevice) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int dev = 0; dev < numDevices; dev++) {
    checkMempoolSupported(dev) hipMemAccessFlags flags;
    hipMemLocation location;
    hipMemPool_t mem_pool;
    HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, dev));
    location.id = dev;
    location.type = hipMemLocationTypeDevice;
    HIP_CHECK(hipMemPoolGetAccess(&flags, mem_pool, &location));
    REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validation matrix for hipMemPoolSetAccess mirroring CUDA's observed
 *    behavior. Covers:
 *      target / flags  | ProtNone        | ProtRead        | ProtReadWrite
 *      own-device      | InvalidDevice   | NotSupported    | apply (no-op)
 *      peer            | apply           | apply           | apply
 *      invalid id      | InvalidDevice   | InvalidDevice   | InvalidDevice
 *    Also verifies batch validate-up-front semantics: a bad descriptor
 *    anywhere in the batch must leave earlier descriptors un-applied.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolSetAccess_ValidationMatrix) {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
    return;
  }

  const auto device_supports_mempool = [](int device) {
    int mempool_supported = 0;
    HIP_CHECK(hipDeviceGetAttribute(&mempool_supported, hipDeviceAttributeMemoryPoolsSupported,
                                    device));
    return mempool_supported != 0;
  };
  const auto can_access_peer = [](int src, int dst) {
    int can_access = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can_access, src, dst));
    return can_access != 0;
  };

  // Check all devices support mempool and can access each others memory
  if (!device_supports_mempool(0) || !device_supports_mempool(1) ||
      !device_supports_mempool(2) ||
      !can_access_peer(0, 1) || !can_access_peer(0, 2)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
    return;
  }

  SECTION("Peer ProtNone after enable transitions state") {
    MemPoolGuard mempool(MemPools::created, 0);

    hipMemAccessDesc enable_desc;
    memset(&enable_desc, 0, sizeof(hipMemAccessDesc));
    enable_desc.location.type = hipMemLocationTypeDevice;
    enable_desc.location.id = 1;
    enable_desc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemPoolSetAccess(mempool.mempool(), &enable_desc, 1));

    hipMemAccessDesc disable_desc;
    memset(&disable_desc, 0, sizeof(hipMemAccessDesc));
    disable_desc.location.type = hipMemLocationTypeDevice;
    disable_desc.location.id = 1;
    disable_desc.flags = hipMemAccessFlagsProtNone;
    HIP_CHECK(hipMemPoolSetAccess(mempool.mempool(), &disable_desc, 1));

    hipMemLocation loc;
    loc.type = hipMemLocationTypeDevice;
    loc.id = 1;
    hipMemAccessFlags flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemPoolGetAccess(&flags, mempool.mempool(), &loc));
    REQUIRE(flags == hipMemAccessFlagsProtNone);
  }

  SECTION("Batch of N peers ProtNone applies to all") {
    if (device_count < 3) {
      HIP_SKIP_TEST("Skipping: needs >=3 GPUs");
      return;
    }
    MemPoolGuard mempool(MemPools::created, 0);

    std::vector<hipMemAccessDesc> enable_batch(device_count - 1);
    for (int i = 1; i < device_count; ++i) {
      memset(&enable_batch[i - 1], 0, sizeof(hipMemAccessDesc));
      enable_batch[i - 1].location.type = hipMemLocationTypeDevice;
      enable_batch[i - 1].location.id = i;
      enable_batch[i - 1].flags = hipMemAccessFlagsProtReadWrite;
    }
    HIP_CHECK(hipMemPoolSetAccess(mempool.mempool(), enable_batch.data(),
                                  enable_batch.size()));

    std::vector<hipMemAccessDesc> disable_batch(device_count - 1);
    for (int i = 1; i < device_count; ++i) {
      memset(&disable_batch[i - 1], 0, sizeof(hipMemAccessDesc));
      disable_batch[i - 1].location.type = hipMemLocationTypeDevice;
      disable_batch[i - 1].location.id = i;
      disable_batch[i - 1].flags = hipMemAccessFlagsProtNone;
    }
    HIP_CHECK(hipMemPoolSetAccess(mempool.mempool(), disable_batch.data(),
                                  disable_batch.size()));

    for (int i = 1; i < device_count; ++i) {
      hipMemLocation loc;
      loc.type = hipMemLocationTypeDevice;
      loc.id = i;
      hipMemAccessFlags flags = hipMemAccessFlagsProtReadWrite;
      HIP_CHECK(hipMemPoolGetAccess(&flags, mempool.mempool(), &loc));
      REQUIRE(flags == hipMemAccessFlagsProtNone);
    }
  }

  SECTION("Own-device ProtNone returns InvalidDevice") {
    MemPoolGuard mempool(MemPools::created, 0);

    hipMemAccessDesc desc;
    memset(&desc, 0, sizeof(hipMemAccessDesc));
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id = 0;
    desc.flags = hipMemAccessFlagsProtNone;
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), &desc, 1),
                    hipErrorInvalidDevice);

    hipMemLocation loc;
    loc.type = hipMemLocationTypeDevice;
    loc.id = 0;
    hipMemAccessFlags flags = hipMemAccessFlagsProtNone;
    HIP_CHECK(hipMemPoolGetAccess(&flags, mempool.mempool(), &loc));
    REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
  }

  SECTION("Own-device ProtRead returns NotSupported") {
    MemPoolGuard mempool(MemPools::created, 0);

    hipMemAccessDesc desc;
    memset(&desc, 0, sizeof(hipMemAccessDesc));
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id = 0;
    desc.flags = hipMemAccessFlagsProtRead;
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), &desc, 1),
                    hipErrorNotSupported);

    hipMemLocation loc;
    loc.type = hipMemLocationTypeDevice;
    loc.id = 0;
    hipMemAccessFlags flags = hipMemAccessFlagsProtNone;
    HIP_CHECK(hipMemPoolGetAccess(&flags, mempool.mempool(), &loc));
    REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
  }

  SECTION("Own-device ProtReadWrite is accepted as no-op") {
    MemPoolGuard mempool(MemPools::created, 0);

    hipMemAccessDesc desc;
    memset(&desc, 0, sizeof(hipMemAccessDesc));
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id = 0;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemPoolSetAccess(mempool.mempool(), &desc, 1));

    hipMemLocation loc;
    loc.type = hipMemLocationTypeDevice;
    loc.id = 0;
    hipMemAccessFlags flags = hipMemAccessFlagsProtNone;
    HIP_CHECK(hipMemPoolGetAccess(&flags, mempool.mempool(), &loc));
    REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
  }

  SECTION("Batch with invalid trailing descriptor leaves earlier state unchanged") {
    MemPoolGuard mempool(MemPools::created, 0);

    hipMemAccessDesc disable_peer;
    memset(&disable_peer, 0, sizeof(hipMemAccessDesc));
    disable_peer.location.type = hipMemLocationTypeDevice;
    disable_peer.location.id = 1;
    disable_peer.flags = hipMemAccessFlagsProtNone;
    HIP_CHECK(hipMemPoolSetAccess(mempool.mempool(), &disable_peer, 1));

    hipMemLocation peer_loc;
    peer_loc.type = hipMemLocationTypeDevice;
    peer_loc.id = 1;
    hipMemAccessFlags pre_flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemPoolGetAccess(&pre_flags, mempool.mempool(), &peer_loc));
    REQUIRE(pre_flags == hipMemAccessFlagsProtNone);

    std::vector<hipMemAccessDesc> batch(2);
    memset(&batch[0], 0, sizeof(hipMemAccessDesc));
    batch[0].location.type = hipMemLocationTypeDevice;
    batch[0].location.id = 1;
    batch[0].flags = hipMemAccessFlagsProtReadWrite;
    memset(&batch[1], 0, sizeof(hipMemAccessDesc));
    batch[1].location.type = hipMemLocationTypeDevice;
    batch[1].location.id = device_count;  // invalid id
    batch[1].flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), batch.data(), 2),
                    hipErrorInvalidDevice);

    hipMemAccessFlags post_flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemPoolGetAccess(&post_flags, mempool.mempool(), &peer_loc));
    REQUIRE(post_flags == hipMemAccessFlagsProtNone);
  }

  SECTION("Mixed-flag batch applies each descriptor") {
    if (device_count < 3) {
      HIP_SKIP_TEST("Skipping: needs >=3 GPUs");
      return;
    }
    MemPoolGuard mempool(MemPools::created, 0);

    std::vector<hipMemAccessDesc> batch(2);
    memset(&batch[0], 0, sizeof(hipMemAccessDesc));
    batch[0].location.type = hipMemLocationTypeDevice;
    batch[0].location.id = 1;
    batch[0].flags = hipMemAccessFlagsProtReadWrite;
    memset(&batch[1], 0, sizeof(hipMemAccessDesc));
    batch[1].location.type = hipMemLocationTypeDevice;
    batch[1].location.id = 2;
    batch[1].flags = hipMemAccessFlagsProtNone;
    HIP_CHECK(hipMemPoolSetAccess(mempool.mempool(), batch.data(), 2));

    hipMemLocation loc1;
    loc1.type = hipMemLocationTypeDevice;
    loc1.id = 1;
    hipMemAccessFlags flags1 = hipMemAccessFlagsProtNone;
    HIP_CHECK(hipMemPoolGetAccess(&flags1, mempool.mempool(), &loc1));
    REQUIRE(flags1 == hipMemAccessFlagsProtReadWrite);

    hipMemLocation loc2;
    loc2.type = hipMemLocationTypeDevice;
    loc2.id = 2;
    hipMemAccessFlags flags2 = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemPoolGetAccess(&flags2, mempool.mempool(), &loc2));
    REQUIRE(flags2 == hipMemAccessFlagsProtNone);
  }
}
