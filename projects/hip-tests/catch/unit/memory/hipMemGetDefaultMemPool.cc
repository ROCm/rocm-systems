/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include "mempool_common.hh"

/**
 * @addtogroup hipMemGetDefaultMemPool hipMemGetDefaultMemPool
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemGetDefaultMemPool(hipMemPool_t* memPool, hipMemLocation* location,
                                       hipMemAllocationType type)` -
 *  Gets the default memory pool for the location and allocation type.
 */

/**
 * Test Description
 * ------------------------
 *    - Test negative cases: null memPool, null location, invalid location id,
 * invalid location type (None and Host), and invalid allocation type.
 * ------------------------
 *    - catch\unit\memory\hipMemGetDefaultMemPool.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipMemGetDefaultMemPool_Negative) {
  int dev;
  HIP_CHECK(hipGetDevice(&dev));
  checkMempoolSupported(dev);

  hipMemPool_t memPool;
  hipMemLocation location{};
  location.id = dev;
  location.type = hipMemLocationTypeDevice;
  hipMemAllocationType allocationType = hipMemAllocationTypePinned;

  SECTION("Invalid memPool") {
    HIP_CHECK_ERROR(hipMemGetDefaultMemPool(nullptr, &location, allocationType),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid location - null") {
    HIP_CHECK_ERROR(hipMemGetDefaultMemPool(&memPool, nullptr, allocationType),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid location - negative device id") {
    location.id = -1;
    HIP_CHECK_ERROR(hipMemGetDefaultMemPool(&memPool, &location, allocationType),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid location - out of range device id") {
    int dev_count = 0;
    HIP_CHECK(hipGetDeviceCount(&dev_count));
    location.id = dev_count;
    HIP_CHECK_ERROR(hipMemGetDefaultMemPool(&memPool, &location, allocationType),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid location - type None") {
    location.type = hipMemLocationTypeNone;
    HIP_CHECK_ERROR(hipMemGetDefaultMemPool(&memPool, &location, allocationType),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid location - type Host") {
    location.type = hipMemLocationTypeHost;
#if HT_NVIDIA
    HIP_CHECK(hipMemGetDefaultMemPool(&memPool, &location, allocationType));
#else
    HIP_CHECK_ERROR(hipMemGetDefaultMemPool(&memPool, &location, allocationType),
                    hipErrorInvalidValue);
#endif
  }

  SECTION("Invalid allocation type") {
    HIP_CHECK_ERROR(hipMemGetDefaultMemPool(&memPool, &location, hipMemAllocationTypeInvalid),
                    hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test basic functionality: verify the returned pool is non-null, matches
 * hipDeviceGetDefaultMemPool for Pinned type, and that repeated calls return
 * the same pool handle (idempotency).
 * ------------------------
 *    - catch\unit\memory\hipMemGetDefaultMemPool.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipMemGetDefaultMemPool_Basic) {
  int dev;
  HIP_CHECK(hipGetDevice(&dev));
  checkMempoolSupported(dev);

  hipMemLocation location{};
  location.id = dev;
  location.type = hipMemLocationTypeDevice;

  SECTION("Pinned - matches hipDeviceGetDefaultMemPool") {
    hipMemPool_t memPool1, memPool2, deviceMemPool;
    HIP_CHECK(hipMemGetDefaultMemPool(&memPool1, &location, hipMemAllocationTypePinned));
    REQUIRE(memPool1 != nullptr);
    HIP_CHECK(hipDeviceGetDefaultMemPool(&deviceMemPool, dev));
    REQUIRE(deviceMemPool != nullptr);
    REQUIRE(memPool1 == deviceMemPool);

    // Idempotency: second call returns the same pool
    HIP_CHECK(hipMemGetDefaultMemPool(&memPool2, &location, hipMemAllocationTypePinned));
    REQUIRE(memPool1 == memPool2);
  }

  SECTION("Managed - returns non-null pool") {
    hipMemPool_t memPool1, memPool2;
    HIP_CHECK(hipMemGetDefaultMemPool(&memPool1, &location, hipMemAllocationTypeManaged));
    REQUIRE(memPool1 != nullptr);

    // Idempotency: second call returns the same pool
    HIP_CHECK(hipMemGetDefaultMemPool(&memPool2, &location, hipMemAllocationTypeManaged));
    REQUIRE(memPool1 == memPool2);
  }

  SECTION("Pinned and Managed pools are distinct") {
    hipMemPool_t pinnedPool, managedPool;
    HIP_CHECK(hipMemGetDefaultMemPool(&pinnedPool, &location, hipMemAllocationTypePinned));
    HIP_CHECK(hipMemGetDefaultMemPool(&managedPool, &location, hipMemAllocationTypeManaged));
    REQUIRE(pinnedPool != nullptr);
    REQUIRE(managedPool != nullptr);
    REQUIRE(pinnedPool != managedPool);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test functional usage: retrieve the default pinned pool and use it to
 * allocate memory via hipMallocFromPoolAsync, copies data to Host, and validate
 * results.
 * ------------------------
 *    - catch\unit\memory\hipMemGetDefaultMemPool.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipMemGetDefaultMemPool_Functional) {
  int dev;
  HIP_CHECK(hipGetDevice(&dev));
  checkMempoolSupported(dev);

  hipMemLocation location{};
  location.id = dev;
  location.type = hipMemLocationTypeDevice;

  hipMemPool_t memPool;
  HIP_CHECK(hipMemGetDefaultMemPool(&memPool, &location, hipMemAllocationTypePinned));
  REQUIRE(memPool != nullptr);

  // Set release threshold to keep memory reserved across syncs
  hipMemPoolAttr attr = hipMemPoolAttrReleaseThreshold;
  std::uint64_t threshold = UINT64_MAX;
  HIP_CHECK(hipMemPoolSetAttribute(memPool, attr, &threshold));

  constexpr size_t num_elems = 1024;
  constexpr size_t byte_size = num_elems * sizeof(int);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  int* d_buf = nullptr;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&d_buf), byte_size, memPool, stream));
  REQUIRE(d_buf != nullptr);

  // Initialize device memory and copy back to verify
  HIP_CHECK(hipMemsetAsync(d_buf, 0, byte_size, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> host_buf(num_elems, -1);
  HIP_CHECK(hipMemcpyAsync(host_buf.data(), d_buf, byte_size, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (size_t i = 0; i < num_elems; ++i) {
    REQUIRE(host_buf[i] == 0);
  }

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(d_buf), stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *    - Test multi-device: for each device, verify that the default pool returned
 * by hipMemGetDefaultMemPool matches hipDeviceGetDefaultMemPool, and that pools
 * from different devices are distinct.
 * ------------------------
 *    - catch\unit\memory\hipMemGetDefaultMemPool.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipMemGetDefaultMemPool_Multidevice) {
  int num_devices;
  HIP_CHECK(hipGetDeviceCount(&num_devices));

  std::vector<hipMemPool_t> pools(num_devices, nullptr);

  for (int i = 0; i < num_devices; i++) {
    checkMempoolSupported(i);
    HIP_CHECK(hipSetDevice(i));

    hipMemLocation location{};
    location.id = i;
    location.type = hipMemLocationTypeDevice;

    hipMemPool_t pool, defaultPool;
    HIP_CHECK(hipMemGetDefaultMemPool(&pool, &location, hipMemAllocationTypePinned));
    REQUIRE(pool != nullptr);

    HIP_CHECK(hipDeviceGetDefaultMemPool(&defaultPool, i));
    REQUIRE(pool == defaultPool);

    pools[i] = pool;
  }

  // Verify each device's default pool is distinct
  for (int i = 0; i < num_devices; i++) {
    for (int j = i + 1; j < num_devices; j++) {
      REQUIRE(pools[i] != pools[j]);
    }
  }
}
