/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipApiName hipApiName
 * @{
 * @ingroup CallbackTest
 * `hipApiName(uint32_t id)` -
 * returns the name of API with passed ID
 */

const char* kUnknownApi{"unknown"};
const uint32_t kApiNumber{1024};

/**
 * Test Description
 * ------------------------
 *  - Acquires HIP API names and checks that they are valid
 * Test source
 * ------------------------
 *  - unit/callback/hipApiName.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Platform specific (AMD)
 */
HIP_TEST_CASE(Unit_hipApiName_Positive_Basic) {
  int valid_api_count = 0;
  for (uint32_t i = 0; i < kApiNumber; ++i) {
    const char* api_name = hipApiName(i);
    REQUIRE(api_name != nullptr);
    REQUIRE(strlen(api_name) > 0);

    if(strcmp(hipApiName(i), kUnknownApi)) {
      ++valid_api_count;
    }
  }
  REQUIRE(valid_api_count > 0);
}

/**
 * Test Description
 * ------------------------
 *  - Checks that upper and lower limit IDs are mapped to unknown APIs:
 *    -# When the `uint32_t` upper limit is passed
 *      - Expected output: return "unknown"
 *    -# When the `uint32_t` lower limit is passed
 *      - Expected output: return "unknown"
 * Test source
 * ------------------------
 *  - unit/callback/hipApiName.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Platform specific (AMD)
 */
HIP_TEST_CASE(Unit_hipApiName_Negative_ReservedIds) {
  REQUIRE_THAT(hipApiName(std::numeric_limits<uint32_t>::min()),
               Catch::Matchers::Equals(kUnknownApi));
  REQUIRE_THAT(hipApiName(std::numeric_limits<uint32_t>::max()),
               Catch::Matchers::Equals(kUnknownApi));
}

/**
 * End doxygen group CallbackTest.
 * @}
 */
