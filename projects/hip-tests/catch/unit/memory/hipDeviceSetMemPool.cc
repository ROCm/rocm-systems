/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipDeviceSetMemPool hipDeviceSetMemPool
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipDeviceSetMemPool(int device,
 *                                 hipMemPool_t mem_pool)` -
 *  Sets the current memory pool for the specified device.
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
 *    - Test case to verify Basic scenario, create an explicit mem pool
 * and validate current pool is same as created mem pool.
 * ------------------------
 *    - catch\unit\memory\hipDeviceSetMemPool.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipDeviceSetMemPool_Basic") {
  int num_devices;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  for (int dev = 0; dev < num_devices; dev++) {
    checkMempoolSupported(dev) hipMemPool_t mem_pool_device = nullptr, curr_mem_pool = nullptr;
    // create explicit mem pool
    hipMemPoolProps prop{};
    prop.allocType = hipMemAllocationTypePinned;
    prop.location.id = dev;
    prop.location.type = hipMemLocationTypeDevice;
    HIP_CHECK(hipMemPoolCreate(&mem_pool_device, &prop));
    HIP_CHECK(hipDeviceSetMemPool(dev, mem_pool_device));
    // get current mem pool
    HIP_CHECK(hipDeviceGetMemPool(&curr_mem_pool, dev));
    // validate both memory are same.
    REQUIRE(curr_mem_pool == mem_pool_device);
    // free mem pool
    HIP_CHECK(hipMemPoolDestroy(mem_pool_device));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Create a mempool and set it as the current mempool of the
 * device. Validate that destroying the current mempool of a device
 * sets the default mempool of that device as the current mempool
 * for that device.
 * ------------------------
 *    - catch\unit\memory\hipDeviceSetMemPool.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipDeviceSetMemPool_DestroyCurrentMempool") {
  int num_devices;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  for (int dev = 0; dev < num_devices; dev++) {
    checkMempoolSupported(dev) HIP_CHECK(hipSetDevice(dev));
    hipMemPool_t mem_pool_device, curr_mem_pool, def_mem_pool;
    hipMemPoolProps prop{};
    prop.allocType = hipMemAllocationTypePinned;
    prop.location.id = dev;
    prop.location.type = hipMemLocationTypeDevice;
    // Create explicit mempool
    HIP_CHECK(hipMemPoolCreate(&mem_pool_device, &prop));
    // Set mempool
    HIP_CHECK(hipDeviceSetMemPool(dev, mem_pool_device));
    // Destroy mem pool
    HIP_CHECK(hipMemPoolDestroy(mem_pool_device));
    // Get current mem pool
    HIP_CHECK(hipDeviceGetMemPool(&curr_mem_pool, dev));
    // Get default mempool
    HIP_CHECK(hipDeviceGetDefaultMemPool(&def_mem_pool, dev));
    // validate the mempool is the default mempool
    REQUIRE(curr_mem_pool == def_mem_pool);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Create explicit memory pool on default GPU. Set this as the current mempool
 * call checkMallocAsync() and destroy the mem pool.
 * ------------------------
 *    - catch\unit\memory\hipDeviceSetMemPool.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipDeviceSetMemPool_functional") {
  checkMempoolSupported(0) hipMemPool_t mem_pool = nullptr;
  // create explicit mem pool
  hipMemPoolProps PoolProps{};
  PoolProps.allocType = hipMemAllocationTypePinned;
  PoolProps.location.id = 0;
  PoolProps.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &PoolProps));
  HIP_CHECK(hipDeviceSetMemPool(0, mem_pool));
  // call checkMallocAsync function
  REQUIRE(true == checkMallocAsync());
  // destroy the mem pool.
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}

/**
 * Test Description
 * ------------------------
 *    - Create explicit memory pool on default GPU. Set this as the current mempool
 * Set attribute hipMemPoolAttrReleaseThreshold to UINT64_MAX. call checkMallocAsync()
 * and destroy the mem pool.
 * ------------------------
 *    - catch\unit\memory\hipDeviceSetMemPool.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipDeviceSetMemPool_functionalAttribute") {
  checkMempoolSupported(0) hipMemPool_t mem_pool = nullptr;
  // create explicit mem pool
  hipMemPoolProps PoolProps{};
  PoolProps.allocType = hipMemAllocationTypePinned;
  PoolProps.location.id = 0;
  PoolProps.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &PoolProps));
  HIP_CHECK(hipDeviceSetMemPool(0, mem_pool));
  // set attribute hipMemPoolAttrReleaseThreshold as UINT64_MAX
  hipMemPoolAttr attr = hipMemPoolAttrReleaseThreshold;
  std::uint64_t value = UINT64_MAX;
  HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));
  // call checkMallocAsync function
  REQUIRE(true == checkMallocAsync());
  // destroy the mem pool.
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}
