/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipEventCreate_NullCheck - Test unsuccessful event creation when event passed as nullptr
Unit_hipEventCreateWithFlags_NullCheck - Test unsuccessful event creation with flags when event
passed as nullptr Unit_hipEventSynchronize_NullCheck - Test unsuccessful event synchronization when
event passed as nullptr Unit_hipEventQuery_NullCheck - Test unsuccessful event query when event
passed as nullptr Unit_hipEventDestroy_NullCheck - Test unsuccessful event destruction when event
passed as nullptr Unit_hipEventCreate_IncompatibleFlags - Test unsuccessful event creation when
incompatible flags are passed
*/

#include <hip_test_common.hh>

/**
 * @addtogroup hipEventCreate hipEventCreate
 * @{
 * @ingroup EventTest
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the event is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEvent_Negative.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipEventCreate_NullCheck) {
  auto res = hipEventCreate(nullptr);
  REQUIRE(res != hipSuccess);
}
/**
 * End doxygen group hipEventCreate.
 * @}
 */

/**
 * @addtogroup hipEventCreateWithFlags hipEventCreateWithFlags
 * @{
 * @ingroup EventTest
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of `nullptr` arguments:
 *    -# When output pointer to the event is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEvent_Negative.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipEventCreateWithFlags_NullCheck) {
  auto res = hipEventCreateWithFlags(nullptr, 0);
  REQUIRE(res != hipSuccess);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of incompatible flags:
 *    -# When the flags are not supported on the device
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEvent_Negative.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipEventCreate_IncompatibleFlags) {
  hipEvent_t event;
  HIP_CHECK_ERROR(hipEventCreateWithFlags(&event, hipEventInterprocess), hipErrorInvalidValue);

#if HT_AMD
  HIP_CHECK_ERROR(
      hipEventCreateWithFlags(&event, hipEventReleaseToDevice | hipEventReleaseToSystem),
      hipErrorInvalidValue);
#endif

  unsigned allFlags{hipEventReleaseToDevice | hipEventReleaseToSystem | hipEventBlockingSync |
                    hipEventDisableTiming | hipEventDefault | hipEventInterprocess};

#if HT_AMD
  HIP_CHECK_ERROR(hipEventCreateWithFlags(&event, allFlags), hipErrorInvalidValue);
#else
  /* Works on Non-AMD because hipEventReleaseToDevice / hipEventReleaseToSystem have no meaning in
   * that case */
  HIP_CHECK(hipEventCreateWithFlags(&event, allFlags));
#endif

  unsigned invalidFlag{0x08000000};
  HIP_CHECK_ERROR(hipEventCreateWithFlags(&event, invalidFlag), hipErrorInvalidValue);
}
/**
 * End doxygen group hipEventCreateWithFlags.
 * @}
 */

/**
 * @addtogroup hipEventSynchronize hipEventSynchronize
 * @{
 * @ingroup EventTest
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When event is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEvent_Negative.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipEventSynchronize_NullCheck) {
  auto res = hipEventSynchronize(nullptr);
  REQUIRE(res != hipSuccess);
}
/**
 * End doxygen group hipEventSynchronize.
 * @}
 */

/**
 * @addtogroup hipEventQuery hipEventQuery
 * @{
 * @ingroup EventTest
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When event is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEvent_Negative.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipEventQuery_NullCheck) {
  auto res = hipEventQuery(nullptr);
  REQUIRE(res != hipSuccess);
}
