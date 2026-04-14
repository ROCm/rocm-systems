/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "execution_control_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

/**
 * @addtogroup hipFuncSetSharedMemConfig hipFuncSetSharedMemConfig
 * @{
 * @ingroup ExecutionTest
 * `hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config)` -
 * Set shared memory configuation for a specific function.
 */

namespace {
constexpr std::array<hipSharedMemConfig, 3> kSharedMemConfigs{
    hipSharedMemBankSizeDefault, hipSharedMemBankSizeFourByte, hipSharedMemBankSizeEightByte};
}  // anonymous namespace

/**
 * Test Description
 * ------------------------
 *  - Sets cache config for every shared memory config enumeration.
 * Test source
 * ------------------------
 *  - unit/executionControl/hipFuncSetSharedMemConfig.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipFuncSetSharedMemConfig_Positive_Basic) {
  const auto shared_mem_config =
      GENERATE(from_range(begin(kSharedMemConfigs), end(kSharedMemConfigs)));

  HIP_CHECK(hipFuncSetSharedMemConfig(reinterpret_cast<void*>(kernel), shared_mem_config));

  kernel<<<1, 1>>>();
  HIP_CHECK(hipDeviceSynchronize());
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When pointer to the kernel function is `nullptr`
 *      - Expected output: return `hipErrorInvalidDeviceFunction`
 *    -# When shared memory config enumeration is invalid
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/executionControl/hipFuncSetSharedMemConfig.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipFuncSetSharedMemConfig_Negative_Parameters) {
  SECTION("func == nullptr") {
    HIP_CHECK_ERROR(hipFuncSetSharedMemConfig(nullptr, hipSharedMemBankSizeDefault),
                    hipErrorInvalidDeviceFunction);
  }
  SECTION("invalid shared mem config") {
    HIP_CHECK_ERROR(hipFuncSetSharedMemConfig(reinterpret_cast<void*>(kernel),
                                              static_cast<hipSharedMemConfig>(-1)),
                    hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Sets shared memory config that is not supported.
 *    - Expected output: return `hipErrorNotSupported`
 * Test source
 * ------------------------
 *  - unit/executionControl/hipFuncSetSharedMemConfig.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipFuncSetSharedMemConfig_Negative_Not_Supported) {
#if HT_NVIDIA
  HipTest::HIP_SKIP_TEST("This is an AMD specific test");
  return;
#endif

  HIP_CHECK_ERROR(
      hipFuncSetSharedMemConfig(reinterpret_cast<void*>(kernel), hipSharedMemBankSizeDefault),
      hipErrorNotSupported);
}

/**
 * End doxygen group ExecutionTest.
 * @}
 */
