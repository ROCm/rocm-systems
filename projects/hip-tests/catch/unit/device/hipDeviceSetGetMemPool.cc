/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <hip/hip_runtime_api.h>
#include <threaded_zig_zag_test.hh>

/**
 * @addtogroup hipDeviceSetMemPool hipDeviceSetMemPool
 * @{
 * @ingroup DeviceTest
 * `hipDeviceSetMemPool(int device, hipMemPool_t mem_pool)` -
 * Sets the current memory pool of a device.
 *
 * The memory pool must be local to the specified device.
 */

static inline bool CheckMemPoolSupport(const int device) {
  int mem_pool_support = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, device));
  if (!mem_pool_support) {
    HipTest::HIP_SKIP_TEST("Test only runs on devices with memory pool support");
    return false;
  }
  return true;
}

static inline hipMemPool_t CreateMemPool(const int device) {
  hipMemPoolProps kPoolProps;
  memset(&kPoolProps, 0, sizeof(kPoolProps));
  kPoolProps.allocType = hipMemAllocationTypePinned;
  kPoolProps.handleTypes = hipMemHandleTypeNone;
  kPoolProps.location.type = hipMemLocationTypeDevice;
  kPoolProps.location.id = device;
  kPoolProps.win32SecurityAttributes = nullptr;

  hipMemPool_t mem_pool;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));

  return mem_pool;
}

/**
 * Test Description
 * ------------------------
 *  - Validate behaviour when the valid MemPool is used, for multiple devices.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetMemPool.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceSetMemPool_Positive_Basic") {
  const int device = GENERATE(range(0, HipTest::getDeviceCount()));

  if (!CheckMemPoolSupport(device)) {
    return;
  }

  hipMemPool_t mem_pool = CreateMemPool(device);
  HIP_CHECK(hipDeviceSetMemPool(device, mem_pool));

  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}

/**
 * Test Description
 * ------------------------
 *  - Validate handling of invalid arguments:
 *    -# When pointer to MemPool is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When device ID is equal to -1
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When device ID is out of bounds
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetMemPool.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceSetMemPool_Negative_Parameters") {
  hipMemPool_t mem_pool;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, 0));

  SECTION("mem_pool == nullptr") {
    HIP_CHECK_ERROR(hipDeviceSetMemPool(0, nullptr), hipErrorInvalidValue);
  }

  SECTION("device < 0") {
    HIP_CHECK_ERROR(hipDeviceSetMemPool(-1, mem_pool), hipErrorInvalidValue);
  }

  SECTION("device ordinance too large") {
    HIP_CHECK_ERROR(hipDeviceSetMemPool(HipTest::getDeviceCount(), mem_pool), hipErrorInvalidValue);
  }
}

/**
 * End doxygen group hipDeviceSetMemPool.
 * @}
 */

/**
 * @addtogroup hipDeviceGetMemPool hipDeviceGetMemPool
 * @{
 * @ingroup DeviceTest
 * `hipDeviceGetMemPool(hipMemPool_t* mem_pool, int device)` -
 * Gets the current memory pool for the specified device
 */

/**
 * Test Description
 * ------------------------
 *  - Checks that returned MemPool is the default MemPool.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetMemPool.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetMemPool_Positive_Default") {
  const int device = GENERATE(range(0, HipTest::getDeviceCount()));

  if (!CheckMemPoolSupport(device)) {
    return;
  }

  hipMemPool_t default_mem_pool;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&default_mem_pool, device));

  hipMemPool_t mem_pool;
  HIP_CHECK(hipDeviceGetMemPool(&mem_pool, device));

  REQUIRE(mem_pool == default_mem_pool);
}

/**
 * Test Description
 * ------------------------
 *  - Checks that returned MemPool is equal to the mempool that is set.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetMemPool.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetMemPool_Positive_Basic") {
  const int device = GENERATE(range(0, HipTest::getDeviceCount()));

  if (!CheckMemPoolSupport(device)) {
    return;
  }

  hipMemPool_t mem_pool = CreateMemPool(device);
  HIP_CHECK(hipDeviceSetMemPool(device, mem_pool));

  hipMemPool_t returned_mem_pool;
  HIP_CHECK(hipDeviceGetMemPool(&returned_mem_pool, device));

  REQUIRE(returned_mem_pool == mem_pool);

  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}

/**
 * Test Description
 * ------------------------
 *  - Checks that returned MemPool is equal to the mempool that is set from a different thread.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetMemPool.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetMemPool_Positive_Threaded") {
  class HipDeviceGetMemPoolTest : public ThreadedZigZagTest<HipDeviceGetMemPoolTest> {
   public:
    void TestPart2() {
      mem_pool_ = CreateMemPool(0);
      HIP_CHECK_THREAD(hipDeviceSetMemPool(0, mem_pool_));
    }
    void TestPart3() {
      hipMemPool_t returned_mem_pool;
      HIP_CHECK(hipDeviceGetMemPool(&returned_mem_pool, 0));

      REQUIRE(returned_mem_pool == mem_pool_);
    }
    void TestPart4() { HIP_CHECK_THREAD(hipMemPoolDestroy(mem_pool_)); }

   private:
    hipMemPool_t mem_pool_;
  };

  if (!CheckMemPoolSupport(0)) {
    return;
  }

  HipDeviceGetMemPoolTest test;
  test.run();
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the MemPool is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When device ID is equal to -1
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When device ID is out of bounds
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetMemPool.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetMemPool_Negative_Parameters") {
  hipMemPool_t mem_pool;

  SECTION("mem_pool == nullptr") {
    HIP_CHECK_ERROR(hipDeviceGetMemPool(nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("device < 0") {
    HIP_CHECK_ERROR(hipDeviceGetMemPool(&mem_pool, -1), hipErrorInvalidValue);
  }

  SECTION("device ordinance too large") {
    HIP_CHECK_ERROR(hipDeviceGetMemPool(&mem_pool, HipTest::getDeviceCount()),
                    hipErrorInvalidValue);
  }
}
