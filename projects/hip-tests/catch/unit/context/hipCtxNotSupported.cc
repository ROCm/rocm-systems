/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When the device id passed is greater than available
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# else
 *      - Expected output: return `hipErrorContextAlreadyInUse`
 * Test source
 * ------------------------
 *  - unit/context/hipCtxNotSupported.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipDevicePrimaryCtxSetFlags_Negative") {
  hipDevice_t dev;
  unsigned int flags = 0;
  SECTION("Negative device index") {
    dev = static_cast<hipDevice_t>(-1);
    auto res = hipDevicePrimaryCtxSetFlags(dev, flags);
    REQUIRE(res == hipErrorInvalidDevice);
  }
  SECTION("Valid device index") {
    dev = static_cast<hipDevice_t>(0);
    auto res = hipDevicePrimaryCtxSetFlags(dev, flags);
    REQUIRE(res == hipErrorContextAlreadyInUse);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When nullptr passed to hipCtxGetDevice
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When a non-nullptr is passed
 *      - Expected output: returned device ID, within [0, numDevices]
 * Test source
 * ------------------------
 *  - unit/context/hipCtxNotSupported.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipDeviceAPIs_not_supported") {
  hipDevice_t device;
  int numDevices = -1;
  HIP_CHECK(hipDeviceGet(&device, 0));
  auto res = hipGetDeviceCount(&numDevices);
  REQUIRE(res == hipSuccess);
  REQUIRE(numDevices > 0);

  SECTION("hipDevicePrimaryCtxReset_not_supported") { HIP_CHECK(hipDevicePrimaryCtxReset(device)); }

  SECTION("hipCtxGetDevice_not_supported") {
    SECTION("hipCtxGetDevice") {
      auto res = hipCtxGetDevice(nullptr);
      REQUIRE(res == hipErrorInvalidValue);
    }
    SECTION("hipCtxGetDevice_deviceCount") {
      hipDevice_t dev = static_cast<hipDevice_t>(-1);
      HIP_CHECK(hipCtxGetDevice(&dev));
      // Ensure the returned device ID is within [0, numDevices]
      REQUIRE(dev >= 0);
      REQUIRE(dev < numDevices);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# any value not equal to the four valid hipFuncCache_t constants
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When valid enum values are passed
 *      - Expected output: return `hipErrorNotSupported`
 * Test source
 * ------------------------
 *  - unit/context/hipCtxNotSupported.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipCtxGetSetCacheConfig_not_supported") {
  hipFuncCache_t cacheConfig;
  SECTION("hipCtxSetCacheConfig_not_supported") {
    SECTION("Invalid enum value") {
      // any value not equal to the four valid hipFuncCache_t constants
      cacheConfig = static_cast<hipFuncCache_t>(0x100);
      auto res = hipCtxSetCacheConfig(cacheConfig);
      REQUIRE(res == hipErrorInvalidValue);
    }

    SECTION("Valid enum values") {
      std::array<hipFuncCache_t, 4> validCfgs = {hipFuncCachePreferNone, hipFuncCachePreferShared,
                                                 hipFuncCachePreferL1, hipFuncCachePreferEqual};

      for (auto cfg : validCfgs) {
        auto res = hipCtxSetCacheConfig(cfg);
        REQUIRE(res == hipErrorNotSupported);
      }
    }
  }
  SECTION("hipCtxGetCacheConfig_not_supported") {
    auto res = hipCtxGetCacheConfig(&cacheConfig);
    REQUIRE(res == hipErrorNotSupported);
  }
}

/**
 * Test Description
 * ------------------------
 *  - hipCtxGetSetSharedMemConfig APIs are verified to be unsupported
 *    or return an empty hipSuccess
 * Test source
 * ------------------------
 *  - unit/context/hipCtxNotSupported.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipCtxGetSetSharedMemConfig_not_supported") {
  hipSharedMemConfig config;
  config = hipSharedMemBankSizeEightByte;
  SECTION("hipCtxSetSharedMemConfig_not_supported") {
    auto res = hipCtxGetSharedMemConfig(&config);
    REQUIRE(res == hipSuccess);
    REQUIRE(config == hipSharedMemBankSizeFourByte);
  }
  SECTION("hipCtxSetSharedMemConfig_not_supported") {
    auto res = hipCtxSetSharedMemConfig(config);
    REQUIRE(res == hipErrorNotSupported);
  }
}

/**
 * Test Description
 * ------------------------
 *  - hipCtxEnable/DisablePeerAccess APIs are verified to return hipSuccess:
 * Test source
 * ------------------------
 *  - unit/context/hipCtxNotSupported.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipCtxPeerAccess_not_supported") {
  hipCtx_t peerCtx = nullptr;
  unsigned int flags = 0;
  SECTION("hipCtxEnablePeerAccess_not_supported") {
    auto res = hipCtxEnablePeerAccess(peerCtx, flags);
    REQUIRE(res == hipSuccess);
  }
  SECTION("hipCtxDisablePeerAccess_not_supported") {
    auto res = hipCtxDisablePeerAccess(peerCtx);
    REQUIRE(res == hipSuccess);
  }
}

/**
 * Test Description
 * ------------------------
 *  - hipCtx APIs are verified to be unsupported:
 * Test source
 * ------------------------
 *  - unit/texture/hipCtxNotSupported.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipCtxAPIs_not_supported") {
  SECTION("hipCtxGetFlags_not_supported") {
    unsigned int flags = 0x100;
    auto res = hipCtxGetFlags(&flags);
    REQUIRE(res == hipErrorNotSupported);
    // In release builds (asserts disabled), flags should remain unchanged
    REQUIRE(flags == 0x100);
  }

  SECTION("hipCtxSynchronize_not_supported") {
    auto res = hipCtxSynchronize();
    REQUIRE(res == hipErrorNotSupported);
  }

  SECTION("hipCtxGetApiVersion_not_supported") {
    hipCtx_t ctx = nullptr;
    unsigned int apiVersion;
    auto res = hipCtxGetApiVersion(ctx, &apiVersion);
    REQUIRE(res == hipErrorNotSupported);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Goes through the retain-reset-release cycle on a valid and invalid device:
 *    Verifies
 *    - a valid primary context is returned
 *    - an active state is returned
 *    - an invalidDevice is returned
 * Test source
 * ------------------------
 *  - unit/context/hipCtxNotSupported.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("hipDevicePrimaryCtxGetState_Negative") {
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  hipCtx_t primaryCtx = nullptr;

  SECTION("Valid device") {
    HIP_CHECK(hipDevicePrimaryCtxRetain(&primaryCtx, device));
    REQUIRE(primaryCtx != nullptr);
    // Make it current
    HIP_CHECK(hipCtxSetCurrent(primaryCtx));
    unsigned int flags = 0;
    int active = 0;
    HIP_CHECK(hipDevicePrimaryCtxGetState(device, &flags, &active));
    // Reset the primary context
    HIP_CHECK(hipDevicePrimaryCtxReset(device));
    // Release our retain-handle
    HIP_CHECK(hipDevicePrimaryCtxRelease(device));
  }
  SECTION("Invalid device") {
    device = -1;
    // Retain the primary context
    auto res = hipDevicePrimaryCtxRetain(&primaryCtx, device);
    REQUIRE(res == hipErrorInvalidDevice);
    unsigned int flags = 0;
    int active = 0;
    res = hipDevicePrimaryCtxGetState(device, &flags, &active);
    REQUIRE(res == hipErrorInvalidDevice);
    // Release our retain-handle
    res = hipDevicePrimaryCtxRelease(device);
    REQUIRE(res == hipErrorInvalidDevice);
  }
}
