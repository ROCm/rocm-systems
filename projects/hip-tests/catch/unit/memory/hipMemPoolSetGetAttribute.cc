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
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */
#include "mempool_common.hh"
#include <resource_guards.hh>
#include <utils.hh>
/**
 * @addtogroup hipMemPoolSetAttribute hipMemPoolSetAttribute
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value)`
 * - Sets attributes of a memory pool
 */

template <typename T> static void MemPoolSetGetAttribute(const hipMemPool_t mempool,
                                                         const hipMemPoolAttr attr, T& set_value) {
  T get_value = 100;
  HIP_CHECK(hipMemPoolSetAttribute(mempool, attr, &set_value));
  HIP_CHECK(hipMemPoolGetAttribute(mempool, attr, &get_value));
  REQUIRE(get_value == set_value);
}


/**
 * Test Description
 * ------------------------
 *  - Basic test to verify that default attribute values are correct.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolSetGetAttribute_Positive_Default") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));

  checkMempoolSupported(device)

      const auto mempool_type = GENERATE(MemPools::dev_default, MemPools::created);
  MemPoolGuard mempool(mempool_type, device);

  const auto attr_type =
      GENERATE(hipMemPoolReuseFollowEventDependencies, hipMemPoolReuseAllowOpportunistic,
               hipMemPoolReuseAllowInternalDependencies);

  // Check default value
  int def_value = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr_type, &def_value));
  REQUIRE(def_value == 1);

  // Check if attribute can be disabled
  int set_value = 0;
  MemPoolSetGetAttribute(mempool.mempool(), attr_type, set_value);
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify hipMemPoolSetAttribute/hipMemPoolGetAttribute functionality.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolSetGetAttribute_Positive_MemBasic") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));

  checkMempoolSupported(device)

      const auto mempool_type = GENERATE(MemPools::dev_default, MemPools::created);
  MemPoolGuard mempool(mempool_type, device);

  // Check hipMemPoolAttrReleaseThreshold default value
  hipMemPoolAttr attr = hipMemPoolAttrReleaseThreshold;
  std::uint64_t value64 = 100;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  REQUIRE(value64 == 0);

  // Check setting hipMemPoolAttrReleaseThreshold to a value
  std::uint64_t set_value64 = kPageSize;
  MemPoolSetGetAttribute(mempool.mempool(), hipMemPoolAttrReleaseThreshold, set_value64);

  // Check reset of hipMemPoolAttrReservedMemHigh and hipMemPoolAttrUsedMemHigh
  set_value64 = 0;
  MemPoolSetGetAttribute(mempool.mempool(), hipMemPoolAttrReservedMemHigh, set_value64);
  MemPoolSetGetAttribute(mempool.mempool(), hipMemPoolAttrUsedMemHigh, set_value64);
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify correct behavior of the Opportunistic attribute.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolSetAttribute_Opportunistic") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));
  checkMempoolSupported(device_id)

      unsigned int *notified1 = nullptr,
                   *notified2 = nullptr;
  HIP_CHECK(hipHostMalloc(&notified1, sizeof(unsigned int)));
  HIP_CHECK(hipHostMalloc(&notified2, sizeof(unsigned int)));
  *notified1 = 0;
  *notified2 = 0;

  MemPoolGuard mempool(MemPools::created, device_id);

  hipMemPoolAttr attr;
  int blocks = 2;
  int *alloc_mem1, *alloc_mem2, *alloc_mem3;

  // Create 2 async non-blocking streams
  StreamGuard stream1(Streams::withFlags, hipStreamNonBlocking);
  StreamGuard stream2(Streams::withFlags, hipStreamNonBlocking);

  size_t allocation_size = kPageSize;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem3), allocation_size,
                                   mempool.mempool(), stream1.stream()));
  int value = 0;

  SECTION("Disallow Opportunistic - No Reuse") {
    allocation_size = kPageSize * kPageSize * 2;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                     mempool.mempool(), stream1.stream()));

    // Disable all default pool states
    attr = hipMemPoolReuseFollowEventDependencies;
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));
    attr = hipMemPoolReuseAllowOpportunistic;
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));
    attr = hipMemPoolReuseAllowInternalDependencies;
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));

    // Run kernel in the first stream
    notifiedKernel<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, notified1);
    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream1.stream()));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified1 = 1;
    // Sleep for 1 second GPU should be idle by now
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size,
                                     mempool.mempool(), stream2.stream()));
    // Without Opportunistic state runtime must allocate another buffer
    REQUIRE(alloc_mem1 != alloc_mem2);

    // Run kernel with the new memory in the second stream
    notifiedKernel<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, notified2);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified2 = 1;
    HIP_CHECK(hipStreamSynchronize(stream1.stream()));
    HIP_CHECK(hipStreamSynchronize(stream2.stream()));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream2.stream()));
  }

  SECTION("Disallow Opportunistic - Reuse") {
    allocation_size = kPageSize * kPageSize * 2;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                     mempool.mempool(), stream1.stream()));

    // Disable all default pool states
    attr = hipMemPoolReuseFollowEventDependencies;
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));
    attr = hipMemPoolReuseAllowOpportunistic;
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));
    attr = hipMemPoolReuseAllowInternalDependencies;
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));

    // Run kernel in the first stream
    notifiedKernel<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, notified1);

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream1.stream()));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified1 = 1;
    // Sleep for 1 second GPU should be idle by now
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Allocate memory for the first stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size,
                                     mempool.mempool(), stream1.stream()));
    // Without Opportunistic state runtime must reuse freed buffer
    REQUIRE(alloc_mem1 == alloc_mem2);

    // Run kernel with the new memory in the first stream
    notifiedKernel<<<32, blocks, 0, stream1.stream()>>>(alloc_mem2, notified2);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified2 = 1;
    HIP_CHECK(hipStreamSynchronize(stream1.stream()));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream1.stream()));
  }

  SECTION("Allow Opportunistic - Reuse") {
    allocation_size = kPageSize * kPageSize * 2;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                     mempool.mempool(), stream1.stream()));

    value = 1;
    attr = hipMemPoolReuseAllowOpportunistic;
    // Enable Opportunistic
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));
    // Run kernel in the first stream
    notifiedKernel<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, notified1);

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream1.stream()));

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified1 = 1;  // Notifiy kernel to exit after 500 ms

    // Sleep for 1 second GPU should be idle by now
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size,
                                     mempool.mempool(), stream2.stream()));
    // With Opportunistic state runtime will reuse freed buffer A
    REQUIRE(alloc_mem1 == alloc_mem2);

    // Run kernel with the new memory in the second stream
    notifiedKernel<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, notified2);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified2 = 1;  // Notifiy kernel to exit after 500 ms

    HIP_CHECK(hipStreamSynchronize(stream1.stream()));
    HIP_CHECK(hipStreamSynchronize(stream2.stream()));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream2.stream()));
  }

  SECTION("Allow Opportunistic - No Reuse") {
    allocation_size = kPageSize * kPageSize * 2;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                     mempool.mempool(), stream1.stream()));

    value = 1;
    attr = hipMemPoolReuseAllowOpportunistic;
    // Enable Opportunistic
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));

    // Run kernel in the first stream
    notifiedKernel<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, notified1);

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream1.stream()));

    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size,
                                     mempool.mempool(), stream2.stream()));
    // With Opportunistic state runtime can't reuse freed buffer A, because it's still busy with the
    // kernel
    REQUIRE(alloc_mem1 != alloc_mem2);

    // Run kernel with the new memory in the second stream
    notifiedKernel<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, notified2);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified1 = 1;
    *notified2 = 1;
    HIP_CHECK(hipStreamSynchronize(stream1.stream()));
    HIP_CHECK(hipStreamSynchronize(stream2.stream()));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream2.stream()));
  }

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem3), stream1.stream()));
  HIP_CHECK(hipHostFree(notified1));
  HIP_CHECK(hipHostFree(notified2));
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify correct behavior of the EventDependencies attribute.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolSetAttribute_EventDependencies") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  checkMempoolSupported(device_id)

      MemPoolGuard mempool(MemPools::created, device_id);

  hipMemPoolAttr attr;
  int blocks = 2;

  unsigned int *notified1 = nullptr, *notified2 = nullptr;
  HIP_CHECK(hipHostMalloc(&notified1, sizeof(unsigned int)));
  HIP_CHECK(hipHostMalloc(&notified2, sizeof(unsigned int)));
  *notified1 = 0;
  *notified2 = 0;

  int *alloc_mem1, *alloc_mem2, *alloc_mem3;

  // Create 2 async non-blocking streams
  StreamGuard stream1(Streams::withFlags, hipStreamNonBlocking);
  StreamGuard stream2(Streams::withFlags, hipStreamNonBlocking);

  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));

  size_t allocation_size = kPageSize;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem3), allocation_size,
                                   mempool.mempool(), stream1.stream()));
  int value = 0;

  SECTION("Allow Event Dependencies - Reuse") {
    allocation_size = kPageSize * kPageSize * 2;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                     mempool.mempool(), stream1.stream()));

    value = 1;
    attr = hipMemPoolReuseFollowEventDependencies;
    // Enable Opportunistic-
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));

    // Run kernel in the first stream
    notifiedKernel<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, notified1);

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream1.stream()));

    HIP_CHECK(hipEventRecord(event, stream1.stream()));
    HIP_CHECK(hipStreamWaitEvent(stream2.stream(), event, 0));

    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size,
                                     mempool.mempool(), stream2.stream()));
    // With Opportunistic state runtime will reuse freed buffer A
    REQUIRE(alloc_mem1 == alloc_mem2);

    // Run kernel with the new memory in the second stream
    notifiedKernel<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, notified2);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified1 = 1;
    *notified2 = 1;
    HIP_CHECK(hipStreamSynchronize(stream1.stream()));
    HIP_CHECK(hipStreamSynchronize(stream2.stream()));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream2.stream()));
  }

  SECTION("Disallow Event Dependencies - No Reuse") {
    allocation_size = kPageSize * kPageSize * 2;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                     mempool.mempool(), stream1.stream()));

    value = 0;
    attr = hipMemPoolReuseFollowEventDependencies;
    // Enable Opportunistic
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));

    // Run kernel in the first stream
    notifiedKernel<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, notified1);
    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream1.stream()));
    HIP_CHECK(hipEventRecord(event, stream1.stream()));
    HIP_CHECK(hipStreamWaitEvent(stream2.stream(), event, 0));

    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size,
                                     mempool.mempool(), stream2.stream()));
    // With Opportunistic state runtime can't reuse freed buffer A, because it's still busy with the
    // kernel
    REQUIRE(alloc_mem1 != alloc_mem2);

    // Run kernel with the new memory in the second stream
    notifiedKernel<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, notified2);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified1 = 1;
    *notified2 = 1;
    HIP_CHECK(hipStreamSynchronize(stream1.stream()));
    HIP_CHECK(hipStreamSynchronize(stream2.stream()));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream2.stream()));
  }

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem3), stream1.stream()));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipHostFree(notified1));
  HIP_CHECK(hipHostFree(notified2));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMemPoolSetAttribute behavior with invalid arguments:
 *    -# Nullptr mem_pool
 *    -# Attribute value is not valid
 *    -# Nullptr value
 *    -# hipMemPoolAttrReservedMemHigh set to non-zero
 *    -# IhipMemPoolAttrUsedMemHigh set to non-zero
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolSetAttribute_Negative_Parameters") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));
  checkMempoolSupported(device_id) MemPoolGuard mempool(MemPools::dev_default, device_id);

  hipMemPoolAttr attr = hipMemPoolReuseFollowEventDependencies;
  int set_value = 0;
  std::uint64_t set_value64 = 0;

  SECTION("Mempool is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolSetAttribute(nullptr, attr, &set_value), hipErrorInvalidValue);
  }

  SECTION("Attribute value is not valid") {
    HIP_CHECK_ERROR(
        hipMemPoolSetAttribute(mempool.mempool(), static_cast<hipMemPoolAttr>(0x9), &set_value),
        hipErrorInvalidValue);
  }
#if HT_AMD
  SECTION("Set values is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolSetAttribute(mempool.mempool(), attr, nullptr), hipErrorInvalidValue);
  }
#endif

  SECTION("Set hipMemPoolAttrReservedMemHigh to non-zero") {
    hipMemPoolAttr attr = hipMemPoolAttrReservedMemHigh;
    set_value64 = 1;
    HIP_CHECK_ERROR((hipMemPoolSetAttribute(mempool.mempool(), attr, &set_value64)),
                    hipErrorInvalidValue);
  }

  SECTION("Set hipMemPoolAttrUsedMemHigh to non-zero") {
    hipMemPoolAttr attr = hipMemPoolAttrUsedMemHigh;
    set_value64 = 1;
    HIP_CHECK_ERROR((hipMemPoolSetAttribute(mempool.mempool(), attr, &set_value64)),
                    hipErrorInvalidValue);
  }
}

/**
 * Local function to reset hipMemPoolAttrReservedMemHigh and hipMemPoolAttrUsedMemHigh.
 */
static void resetHighValue(hipMemPool_t& memPool) {
  uint64_t value = 0;
  HIP_CHECK(hipMemPoolSetAttribute(memPool, hipMemPoolAttrReservedMemHigh, &value));
  HIP_CHECK(hipMemPoolSetAttribute(memPool, hipMemPoolAttrUsedMemHigh, &value));
}

/**
 * Test Description
 * ------------------------
 *    - Reset hipMemPoolAttrReservedMemHigh and hipMemPoolAttrUsedMemHigh values
 * and validate their values.
 * ------------------------
 *    - catch\unit\memory\hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolSetAttribute_ResetMemHighAttr") {
  checkMempoolSupported(0)
      // Create mempool
      hipMemPool_t mem_pool;
  hipMemPoolProps pool_props{};
  constexpr int N = 1 << 14;
  size_t byte_size = (N * sizeof(int));
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
  // Reset High Attributes
  resetHighValue(mem_pool);

  // Allocate from mempool
  int* A_d;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size, mem_pool, 0));
  // Deallocate
  HIP_CHECK(hipFreeAsync(A_d, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  // Reset High Attributes
  resetHighValue(mem_pool);
  // Validate usage statistics
  uint64_t valueReservedHighAfterReset = 0, valueUsedHighAfterReset = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, hipMemPoolAttrReservedMemHigh,
                                   &valueReservedHighAfterReset));
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, hipMemPoolAttrUsedMemHigh, &valueUsedHighAfterReset));
  REQUIRE(valueReservedHighAfterReset == 0);
  REQUIRE(valueUsedHighAfterReset == 0);
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}

/**
 * End doxygen group hipMemPoolSetAttribute.
 * @}
 */

/**
 * @addtogroup hipMemPoolGetAttribute hipMemPoolGetAttribute
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value)`
 * - 	Gets attributes of a memory pool
 */

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMemPoolGetAttribute behavior with invalid arguments:
 *    -# Nullptr mem_pool
 *    -# Attribute value is not valid
 *    -# Nullptr value
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolGetAttribute_Negative_Parameters") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));
  checkMempoolSupported(device_id) MemPoolGuard mempool(MemPools::dev_default, device_id);


  hipMemPoolAttr attr = hipMemPoolReuseFollowEventDependencies;
  int get_value = 0;

  SECTION("Mempool is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolGetAttribute(nullptr, attr, &get_value), hipErrorInvalidValue);
  }

  SECTION("Attribute value is not valid") {
    HIP_CHECK_ERROR(
        hipMemPoolGetAttribute(mempool.mempool(), static_cast<hipMemPoolAttr>(0x9), &get_value),
        hipErrorInvalidValue);
  }

  SECTION("Get values is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolGetAttribute(mempool.mempool(), attr, nullptr), hipErrorInvalidValue);
  }
}

constexpr int iterations = 20;
static int reservedHighExp = 0;
static int usedHighExp = 0;

struct mempoolUsgStat {
  uint64_t reservedMem;
  uint64_t reservedMemHigh;
  uint64_t usedMem;
  uint64_t usedMemHigh;
};

/**
 * Local function to fetch usage statistics.
 */
static void getUsageStatistics(hipMemPool_t& memPool, struct mempoolUsgStat* stat) {
  HIP_CHECK(
      hipMemPoolGetAttribute(memPool, hipMemPoolAttrReservedMemCurrent, &(stat->reservedMem)));
  HIP_CHECK(
      hipMemPoolGetAttribute(memPool, hipMemPoolAttrReservedMemHigh, &(stat->reservedMemHigh)));
  HIP_CHECK(hipMemPoolGetAttribute(memPool, hipMemPoolAttrUsedMemCurrent, &(stat->usedMem)));
  HIP_CHECK(hipMemPoolGetAttribute(memPool, hipMemPoolAttrUsedMemHigh, &(stat->usedMemHigh)));
}

/**
 * Local function to get default mempool attribute values.
 */
static bool checkDefaultAttributeValues(hipMemPoolAttr attr, int dev) {
  // Create mempool in current device
  uint64_t ui64_setValue = 0;
  int i32_setValue = 0;
  hipMemPool_t mem_pool;
  hipMemPoolProps pool_props{};
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = dev;
  pool_props.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
  if (attr == hipMemPoolAttrReleaseThreshold) {
    HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &ui64_setValue));
    REQUIRE(ui64_setValue == 0);
  } else {
    HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &i32_setValue));
    REQUIRE(i32_setValue == 1);
  }
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  return true;
}

/**
 * Local function to set mempool attribute values and validate
 * by getting the values.
 */
static bool checkhipMemPoolSetAttribute(hipMemPoolAttr attr, int dev) {
  // Create mempool in current device
  uint64_t ui64_setValue = 0;
  int i32_setValue = 0;
  hipMemPool_t mem_pool;
  hipMemPoolProps pool_props{};
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = dev;
  pool_props.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
  if (attr == hipMemPoolAttrReleaseThreshold) {
    uint64_t val = UINT64_MAX;
    HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &val));
    HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &ui64_setValue));
    REQUIRE(ui64_setValue == val);
  } else {
    int val = 0;
    HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &val));
    HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &i32_setValue));
    REQUIRE(i32_setValue == val);
  }
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  return true;
}

/**
 * Test Description
 * ------------------------
 *    - Validate hipMemPoolGetAttribute() by setting hipMemPoolSetAttribute().
 * ------------------------
 *    - catch\unit\memory\hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolGetAttribute_SetGet") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int dev = 0; dev < numDevices; dev++) {
    checkMempoolSupported(dev)
        REQUIRE(true == checkhipMemPoolSetAttribute(hipMemPoolAttrReleaseThreshold, dev));
    REQUIRE(true == checkhipMemPoolSetAttribute(hipMemPoolReuseFollowEventDependencies, dev));
    REQUIRE(true == checkhipMemPoolSetAttribute(hipMemPoolReuseAllowOpportunistic, dev));
    REQUIRE(true == checkhipMemPoolSetAttribute(hipMemPoolReuseAllowInternalDependencies, dev));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Validate hipMemPoolAttrUsedMemCurrent value.
 * ------------------------
 *    - catch\unit\memory\hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolGetAttribute_UsedMem") {
  checkMempoolSupported(0) constexpr int N = 1 << 14;
  hipMemPool_t mem_pool;
  hipMemPoolProps pool_props{};
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
  size_t byte_size = (N * sizeof(int));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  // Get hipMemPoolAttrUsedMemCurrent value for mem_pool when no memory
  // is allocated from this pool.
  SECTION("Check created mempool") {
    uint64_t val = 0;
    HIP_CHECK(hipMemPoolGetAttribute(mem_pool, hipMemPoolAttrUsedMemCurrent, &val));
    REQUIRE(val == 0);
    int* A_d;
    // Allocate memory on dev0 from mem_pool.
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size, mem_pool, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    // Get hipMemPoolAttrUsedMemCurrent value for mem_pool and validate
    // its value.
    HIP_CHECK(hipMemPoolGetAttribute(mem_pool, hipMemPoolAttrUsedMemCurrent, &val));
    REQUIRE(val == byte_size);
    // Free memory back to memory pool.
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d), stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    // Again get hipMemPoolAttrUsedMemCurrent value for mem_pool and validate
    // its value.
    HIP_CHECK(hipMemPoolGetAttribute(mem_pool, hipMemPoolAttrUsedMemCurrent, &val));
    REQUIRE(val == 0);
  }
  SECTION("Check default mempool") {
    hipMemPool_t mem_pool_default = nullptr;
    // assign default mem pool to device
    HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool_default, 0));
    uint64_t valInitital = 0, valPostAlloc = 0, valPostFree = 0;
    HIP_CHECK(hipMemPoolGetAttribute(mem_pool_default, hipMemPoolAttrUsedMemCurrent, &valInitital));
    int* A_d;
    // Allocate memory on dev0 from mem_pool.
    HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&A_d), byte_size, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    // Get hipMemPoolAttrUsedMemCurrent value for mem_pool and validate
    // its value.
    HIP_CHECK(
        hipMemPoolGetAttribute(mem_pool_default, hipMemPoolAttrUsedMemCurrent, &valPostAlloc));
    uint64_t expVal = byte_size;
    expVal = expVal + valInitital;
    REQUIRE(valPostAlloc == expVal);
    // Free memory back to memory pool.
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d), stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    // Again get hipMemPoolAttrUsedMemCurrent value for mem_pool and validate
    // its value.
    HIP_CHECK(hipMemPoolGetAttribute(mem_pool_default, hipMemPoolAttrUsedMemCurrent, &valPostFree));
    REQUIRE(valPostFree == valInitital);
  }
  SECTION("Default memory pool allocation") {
    uint64_t val = 0;
    HIP_CHECK(hipMemPoolGetAttribute(mem_pool, hipMemPoolAttrUsedMemCurrent, &val));
    REQUIRE(val == 0);
    int* A_d;
    // Allocate memory on dev0 from mem_pool.
    HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&A_d), byte_size, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    // Get hipMemPoolAttrUsedMemCurrent value for mem_pool and validate
    // its value.
    HIP_CHECK(hipMemPoolGetAttribute(mem_pool, hipMemPoolAttrUsedMemCurrent, &val));
    REQUIRE(val == 0);
    // Free memory back to memory pool.
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d), stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}

/**
 * Test Description
 * ------------------------
 *    - Validate hipMemPoolAttrReservedMemCurrent value.
 * ------------------------
 *    - catch\unit\memory\hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolGetAttribute_ReservedMem") {
  checkMempoolSupported(0) constexpr int N = 1 << 14;
  hipMemPool_t mem_pool;
  hipMemPoolProps pool_props{};
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
  uint64_t val = 0;
  // Verify that at the beginning mempool contains at least
  // 0 memory reserved.
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, hipMemPoolAttrReservedMemCurrent, &val));
  REQUIRE(val >= 0);
  size_t byte_size = (N * sizeof(int));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  int* A_d;
  // Allocate memory on dev0 from mem_pool.
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size, mem_pool, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, hipMemPoolAttrReservedMemCurrent, &val));
  REQUIRE(val >= byte_size);
  // Free memory back to memory pool.
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d), stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  // Again get hipMemPoolAttrReservedMemCurrent value for mem_pool and validate
  // its value.
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, hipMemPoolAttrReservedMemCurrent, &val));
  REQUIRE(val >= 0);
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}

/**
 * Test Description
 * ------------------------
 *    - Validate hipMemPoolAttrReservedMemHigh and hipMemPoolAttrUsedMemHigh value.
 * ------------------------
 *    - catch\unit\memory\hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolGetAttribute_UsageStatistics") {
  checkMempoolSupported(0) struct mempoolUsgStat stats;
  // Create mempool
  hipMemPool_t mem_pool;
  hipMemPoolProps pool_props{};
  constexpr int N = 1 << 14;
  size_t byte_size = (N * sizeof(int));
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
  // Reset and Take Usage Statistics
  resetHighValue(mem_pool);
  getUsageStatistics(mem_pool, &stats);
  // Validate usage statistics
  REQUIRE(stats.reservedMem == stats.reservedMemHigh);
  REQUIRE(stats.usedMem == 0);
  REQUIRE(stats.usedMemHigh == 0);

  // Allocate from mempool
  int* A_d[iterations];
  for (int i = 0; i < iterations; i++) {
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d[i]), byte_size, mem_pool, 0));
  }
  HIP_CHECK(hipStreamSynchronize(0));
  // Take Usage Statistics
  getUsageStatistics(mem_pool, &stats);
  // Validate usage statistics
  REQUIRE(stats.reservedMem == stats.reservedMemHigh);
  REQUIRE(stats.usedMem == (iterations * byte_size));
  REQUIRE(stats.usedMemHigh == (iterations * byte_size));
  reservedHighExp = stats.reservedMemHigh;
  usedHighExp = (iterations * byte_size);

  // Deallocate half of the allocations
  for (int i = 0; i < iterations / 2; i++) {
    HIP_CHECK(hipFreeAsync(A_d[i], 0));
  }
  HIP_CHECK(hipStreamSynchronize(0));
  // Take Usage Statistics
  getUsageStatistics(mem_pool, &stats);
  // Validate usage statistics
  REQUIRE(stats.reservedMemHigh == reservedHighExp);
  REQUIRE(stats.usedMem == (iterations * byte_size - (iterations / 2) * byte_size));
  REQUIRE(stats.usedMemHigh == usedHighExp);

  // Deallocate remaining allocations
  for (int i = (iterations / 2); i < iterations; i++) {
    HIP_CHECK(hipFreeAsync(A_d[i], 0));
  }
  HIP_CHECK(hipStreamSynchronize(0));
  // Take Usage Statistics
  getUsageStatistics(mem_pool, &stats);
  // Validate usage statistics
  REQUIRE(stats.reservedMemHigh == reservedHighExp);
  REQUIRE(stats.usedMem == 0);
  REQUIRE(stats.usedMemHigh == usedHighExp);

  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}

/**
 * Test Description
 * ------------------------
 *    - Validate hipMalloc does not affect default mempool
 * statistics.
 * ------------------------
 *    - catch\unit\memory\hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolGetAttribute_hipMalloc_DefMempool") {
  checkMempoolSupported(0) struct mempoolUsgStat stats;
  // Create mempool
  hipMemPool_t mem_pool;
  constexpr int N = 1 << 14;
  size_t byte_size = (N * sizeof(int));
  // Get default mempool
  HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, 0));
  // Reset and Take Usage Statistics
  resetHighValue(mem_pool);
  getUsageStatistics(mem_pool, &stats);
  uint64_t reservedMemStart, reservedMemHighStart, usedMemStart, usedMemHighStart;
  reservedMemStart = stats.reservedMem;
  reservedMemHighStart = stats.reservedMemHigh;
  usedMemStart = stats.usedMem;
  usedMemHighStart = stats.usedMemHigh;
  // Allocate using hipMalloc
  int* Ad;
  HIP_CHECK(hipMalloc(&Ad, byte_size));
  getUsageStatistics(mem_pool, &stats);
  REQUIRE(reservedMemStart == stats.reservedMem);
  REQUIRE(reservedMemHighStart == stats.reservedMemHigh);
  REQUIRE(usedMemStart == stats.usedMem);
  REQUIRE(usedMemHighStart == stats.usedMemHigh);
  HIP_CHECK(hipFree(Ad));
}
/**
 * Test Description
 * ------------------------
 *    - Validate default attribute values.
 * ------------------------
 *    - catch\unit\memory\hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolGetAttribute_CheckDefaultValues") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int dev = 0; dev < numDevices; dev++) {
    checkMempoolSupported(dev)
        REQUIRE(true == checkDefaultAttributeValues(hipMemPoolAttrReleaseThreshold, dev));
    REQUIRE(true == checkDefaultAttributeValues(hipMemPoolReuseFollowEventDependencies, dev));
    REQUIRE(true == checkDefaultAttributeValues(hipMemPoolReuseAllowOpportunistic, dev));
    REQUIRE(true == checkDefaultAttributeValues(hipMemPoolReuseAllowInternalDependencies, dev));
  }
}
