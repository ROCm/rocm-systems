/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipDeviceGetMemPool hipDeviceGetMemPool
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool,
 *                                 int device)` -
 *  Gets the current memory pool for the specified device.
 */

#include "mempool_common.hh"  // NOLINT

#define THREADS_PER_BLOCK 512
static constexpr auto NUM_ELM{1024 * 1024};

/**
 * Common function to allocate memory using hipMallocAsync API through a stream,
 * launch kernel and perform vectorADD and validate results. Free memory using
 * hipFreeAsync.
 */
static bool checkMallocAsync() {
  streamMemAllocTest testObj(NUM_ELM);
  // create a stream
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  // Create host buffer with test data.
  testObj.createHostBufferWithData();
  // Allocate device memory and transfer data to it asyncronously on stream.
  testObj.allocFromDefMempool(stream);
  testObj.transferToMempool(stream);
  // Execute kernel and transfer result back to host asynchronously on stream.
  testObj.runKernel(stream);
  testObj.transferFromMempool(stream);
  // Free Buffer Asynchronously on stream.
  testObj.freeDevBuf(stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  // verify and validate
  REQUIRE(true == testObj.validateResult());
  // Destroy resources
  HIP_CHECK(hipStreamDestroy(stream));
  testObj.freeHostBuf();
  return true;
}
/**
 * Test Description
 * ------------------------
 *    - Test case to perform basic scenario, get device mem pool
 * and default mem pool and validate both are same.
 * ------------------------
 *    - catch\unit\memory\hipDeviceGetMemPool.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE(Unit_hipDeviceGetMemPool_Basic) {
  checkMempoolSupported(0) hipMemPool_t mem_pool_device = nullptr, mem_pool_default = nullptr;
  SECTION("Check current mempool is default mempool") {
    // assign default mem pool to device
    HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool_default, 0));
    // assign device mem pool to device
    HIP_CHECK(hipDeviceGetMemPool(&mem_pool_device, 0));
    // validate both are same
    REQUIRE(mem_pool_device == mem_pool_default);
  }
  SECTION("Allocating a mempool does not impact default mempool ctx") {
    hipMemPoolProps PoolProps{};
    PoolProps.allocType = hipMemAllocationTypePinned;
    PoolProps.location.id = 0;
    PoolProps.location.type = hipMemLocationTypeDevice;
    // assign default mem pool to device
    HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool_default, 0));
    // create explicit mem pool
    hipMemPool_t user_mempool;
    HIP_CHECK(hipMemPoolCreate(&user_mempool, &PoolProps));
    // assign device mem pool to device
    HIP_CHECK(hipDeviceGetMemPool(&mem_pool_device, 0));
    // validate both are same
    REQUIRE(mem_pool_device == mem_pool_default);
    HIP_CHECK(hipMemPoolDestroy(user_mempool));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test case to check functional scenario, Get the current mempool using
 * hipDeviceGetMempool. Set attribute hipMemPoolAttrReleaseThreshold to
 * UINT64_MAX. call checkMallocAsync().
 * ------------------------
 *    - catch\unit\memory\hipDeviceGetMemPool.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE(Unit_hipDeviceGetMemPool_Functional) {
  hipMemPool_t mem_pool = nullptr;
  checkMempoolSupported(0)
      // assign current mem pool to device
      HIP_CHECK(hipDeviceGetMemPool(&mem_pool, 0));
  // set attribute hipMemPoolAttrReleaseThreshold as UINT64_MAX
  hipMemPoolAttr attr = hipMemPoolAttrReleaseThreshold;
  std::uint64_t value = UINT64_MAX;
  HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));
  // call checkMallocAsync() and validate
  REQUIRE(true == checkMallocAsync());
}

/**
 * Test Description
 * ------------------------
 *    - Test case to verify multi device, get number of devices available
 * and verify device mem pool and default mem pool are same.
 * ------------------------
 *    - catch\unit\memory\hipDeviceGetMemPool.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE(Unit_hipDeviceGetMemPool_Multidevice) {
  int num_devices;
  HIP_CHECK(hipGetDeviceCount(&num_devices));

  for (int i = 0; i < num_devices; i++) {
    checkMempoolSupported(i) HIP_CHECK(hipSetDevice(i));
    hipMemPool_t mem_pool_device = nullptr, mem_pool_default = nullptr;
    // assign default mem pool to device
    HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool_default, i));
    // assign device mem pool to device
    HIP_CHECK(hipDeviceGetMemPool(&mem_pool_device, i));
    // validate both are same
    REQUIRE(mem_pool_device == mem_pool_default);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test case to check functional scenario, Get the current mempool using
 * hipDeviceGetDefaultMemPool. Set attribute hipMemPoolAttrReleaseThreshold
 * to UINT64_MAX. call checkMallocAsync().
 * ------------------------
 *    - catch\unit\memory\hipDeviceGetMemPool.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE(Unit_hipDeviceGetDefaultMemPool_Functional) {
  hipMemPool_t mem_pool = nullptr;
  checkMempoolSupported(0)
      // assign current mem pool to device
      HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, 0));
  // set attribute hipMemPoolAttrReleaseThreshold as UINT64_MAX
  hipMemPoolAttr attr = hipMemPoolAttrReleaseThreshold;
  std::uint64_t value = UINT64_MAX;
  HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));
  // call checkMallocAsync() and validate
  REQUIRE(true == checkMallocAsync());
}
