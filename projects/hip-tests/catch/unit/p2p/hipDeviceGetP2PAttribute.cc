/*
Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdlib>
#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include "hip/hip_runtime_api.h"
#include <hip_test_process.hh>
#include <string>


/**
 * @addtogroup hipDeviceGetP2PAttribute hipDeviceGetP2PAttribute
 * @{
 * @ingroup DriverTest
 * `hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr,
 * int srcDevice, int dstDevice)` -
 * Returns a value for attr of link between two devices.
 */

/**
 * Test Description
 * ------------------------
 *  - Get all possible combinations of attributes between all pairs of devices.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetP2PAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */

/**
 * Test Description
 * ------------------------
 *    - Test all possible combination of attributes and devices for hipDeviceGetP2PAttribute
 * Verify that the output is within the range of acceptable values.

 * Test source
 * ------------------------
 *    - catch/unit/p2p/hipDeviceGetP2PAttribute.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

TEST_CASE("Unit_hipDeviceGetP2PAttribute_Basic") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-119");
  return;
#else

  int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  hipDeviceP2PAttr attribute =
      GENERATE(hipDevP2PAttrPerformanceRank, hipDevP2PAttrAccessSupported,
               hipDevP2PAttrNativeAtomicSupported, hipDevP2PAttrHipArrayAccessSupported);

  /* Test all combinations of devices in the system */
  for (int srcDevice = 0; srcDevice < deviceCount; ++srcDevice) {
    for (int dstDevice = 0; dstDevice < deviceCount; ++dstDevice) {
      if (srcDevice != dstDevice) {
        int value{-1};
        HIP_CHECK(hipDeviceGetP2PAttribute(&value, attribute, srcDevice, dstDevice));
        INFO("hipDeviceP2PAttr: " << attribute << "\nsrcDevice: " << srcDevice
                                  << "\ndstDevice: " << dstDevice << "\nValue: " << value);
        if (attribute == hipDevP2PAttrPerformanceRank) {
          REQUIRE(value >= 0);
        } else {
          REQUIRE((value == 0 || value == 1));
        }
      }
    }
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Verifies handling of invalid arguments:
 *    -# When output pointer to the value is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When attribute is invalid, out of bounds
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When device ordinal is negative (-1)
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When device ordinal is out of bounds
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When the src and dst devices are the same one
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When some devices are hidden using environment variables
 *      - Expected output: different scenarios produce different return value
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetP2PAttribute.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (NVIDIA)
 *  - HIP_VERSION >= 5.2
 */

TEST_CASE("Unit_hipDeviceGetP2PAttribute_Negative") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-122");
  return;
#else

  int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  int value;
  int validSrcDevice = 0;
  int validDstDevice = 1;
  hipDeviceP2PAttr validAttr = hipDevP2PAttrAccessSupported;

  SECTION("Nullptr value") {
    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(nullptr, validAttr, validSrcDevice, validDstDevice),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid attribute") {
    hipDeviceP2PAttr invalidAttr = static_cast<hipDeviceP2PAttr>(10);
    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(&value, invalidAttr, validSrcDevice, validDstDevice),
                    hipErrorInvalidValue);
  }

  SECTION("Device is -1") {
    int invalidDevice = -1;
    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(&value, validAttr, invalidDevice, validDstDevice),
                    hipErrorInvalidDevice);
    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(&value, validAttr, validSrcDevice, invalidDevice),
                    hipErrorInvalidDevice);
  }

  SECTION("Device is out of bounds") {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    REQUIRE_FALSE(deviceCount == 0);

    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(&value, validAttr, deviceCount, validDstDevice),
                    hipErrorInvalidDevice);
    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(&value, validAttr, validSrcDevice, deviceCount),
                    hipErrorInvalidDevice);
  }

  SECTION("Source and destination devices are the same") {
    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(&value, validAttr, validSrcDevice, validSrcDevice),
                    hipErrorInvalidDevice);
  }

  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
  SECTION("Hidden devices using environment variables") {
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute_exe").run("") == hipSuccess);  // NOLINT
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute_exe").run("0") ==
            hipErrorInvalidDevice);  // NOLINT
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute_exe").run("1") ==
            hipErrorInvalidDevice);                                                    // NOLINT
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute_exe").run("0,1") == hipSuccess);  // NOLINT
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute_exe").run("-1,0") ==
            hipErrorNoDevice);  // NOLINT
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute_exe").run("0,-1") ==
            hipErrorInvalidDevice);                                                       // NOLINT
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute_exe").run("0,1,-1") == hipSuccess);  // NOLINT
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute_exe").run("0,-1,1") ==
            hipErrorInvalidDevice);  // NOLINT

    if (deviceCount > 2) {
      REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute_exe").run("2,1") == hipSuccess);  // NOLINT
      REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute_exe").run("2") ==
              hipErrorInvalidDevice);  // NOLINT
    } else {
      REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute_exe").run("2,1") ==
              hipErrorNoDevice);  // NOLINT
    }
  }
#endif
}

/**
 * End doxygen group DriverTest.
 * @}
 */
