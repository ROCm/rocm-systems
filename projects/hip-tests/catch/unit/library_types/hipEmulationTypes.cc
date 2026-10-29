/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/library_types.h>

#include <type_traits>

/**
 * @addtogroup hipEmulationTypes hipEmulationTypes
 * @{
 * @ingroup LibraryTypesTest
 * Floating-point emulation control types in `hip/library_types.h`:
 * `hipEmulationStrategy`, `hipEmulationMantissaControl` and
 * `hipEmulationSpecialValuesSupport`. Math libraries pass these values straight
 * through to their backends, so the enumerator values must match CUDA.
 *
 * Most checks are static_asserts, so a mismatch fails the build. The test cases
 * additionally exercise the types at runtime so the coverage shows up in CTest.
 */

/*
 * Availability of the types on each platform:
 * - AMD: always defined by hip/library_types.h.
 * - NVIDIA: aliases of the CUDA types. hipEmulationStrategy needs
 *   CUDART_VERSION >= 13000; the other two need CUDART_VERSION >= 13010.
 */
#if defined(__HIP_PLATFORM_AMD__)
#define HIP_TEST_HAS_EMULATION_STRATEGY 1
#define HIP_TEST_HAS_EMULATION_MANTISSA_SPECIAL 1
#elif defined(__HIP_PLATFORM_NVIDIA__) && defined(CUDART_VERSION)
#define HIP_TEST_HAS_EMULATION_STRATEGY (CUDART_VERSION >= 13000)
#define HIP_TEST_HAS_EMULATION_MANTISSA_SPECIAL (CUDART_VERSION >= 13010)
#else
#define HIP_TEST_HAS_EMULATION_STRATEGY 0
#define HIP_TEST_HAS_EMULATION_MANTISSA_SPECIAL 0
#endif

#if HIP_TEST_HAS_EMULATION_STRATEGY
// Documented values; these match CUDA's cudaEmulationStrategy.
static_assert(HIP_EMULATION_STRATEGY_DEFAULT == 0, "HIP_EMULATION_STRATEGY_DEFAULT");
static_assert(HIP_EMULATION_STRATEGY_PERFORMANT == 1, "HIP_EMULATION_STRATEGY_PERFORMANT");
static_assert(HIP_EMULATION_STRATEGY_EAGER == 2, "HIP_EMULATION_STRATEGY_EAGER");
static_assert(std::is_same<hipEmulationStrategy, hipEmulationStrategy_t>::value ||
                  std::is_same<hipEmulationStrategy, enum hipEmulationStrategy_t>::value,
              "hipEmulationStrategy and hipEmulationStrategy_t name the same type");

#if defined(__HIP_PLATFORM_NVIDIA__)
static_assert(std::is_same<hipEmulationStrategy, cudaEmulationStrategy>::value,
              "hipEmulationStrategy is cudaEmulationStrategy");
static_assert(std::is_same<hipEmulationStrategy_t, cudaEmulationStrategy_t>::value,
              "hipEmulationStrategy_t is cudaEmulationStrategy_t");
static_assert(HIP_EMULATION_STRATEGY_DEFAULT == CUDA_EMULATION_STRATEGY_DEFAULT, "");
static_assert(HIP_EMULATION_STRATEGY_PERFORMANT == CUDA_EMULATION_STRATEGY_PERFORMANT, "");
static_assert(HIP_EMULATION_STRATEGY_EAGER == CUDA_EMULATION_STRATEGY_EAGER, "");
#endif
#endif  // HIP_TEST_HAS_EMULATION_STRATEGY

#if HIP_TEST_HAS_EMULATION_MANTISSA_SPECIAL
// Documented values; these match CUDA's cudaEmulationMantissaControl.
static_assert(HIP_EMULATION_MANTISSA_CONTROL_DYNAMIC == 0,
              "HIP_EMULATION_MANTISSA_CONTROL_DYNAMIC");
static_assert(HIP_EMULATION_MANTISSA_CONTROL_FIXED == 1, "HIP_EMULATION_MANTISSA_CONTROL_FIXED");

// Documented values; these match CUDA's cudaEmulationSpecialValuesSupport.
static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NONE == 0,
              "HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NONE");
static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY == 1,
              "HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY");
static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN == 2,
              "HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN");
static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT == 0xFFFF,
              "HIP_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT");

#if defined(__HIP_PLATFORM_NVIDIA__)
static_assert(std::is_same<hipEmulationMantissaControl, cudaEmulationMantissaControl>::value,
              "hipEmulationMantissaControl is cudaEmulationMantissaControl");
static_assert(std::is_same<hipEmulationMantissaControl_t, cudaEmulationMantissaControl_t>::value,
              "hipEmulationMantissaControl_t is cudaEmulationMantissaControl_t");
static_assert(HIP_EMULATION_MANTISSA_CONTROL_DYNAMIC == CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC,
              "");
static_assert(HIP_EMULATION_MANTISSA_CONTROL_FIXED == CUDA_EMULATION_MANTISSA_CONTROL_FIXED, "");

static_assert(
    std::is_same<hipEmulationSpecialValuesSupport, cudaEmulationSpecialValuesSupport>::value,
    "hipEmulationSpecialValuesSupport is cudaEmulationSpecialValuesSupport");
static_assert(
    std::is_same<hipEmulationSpecialValuesSupport_t, cudaEmulationSpecialValuesSupport_t>::value,
    "hipEmulationSpecialValuesSupport_t is cudaEmulationSpecialValuesSupport_t");
static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NONE ==
                  CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE,
              "");
static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY ==
                  CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY,
              "");
static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN == CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NAN,
              "");
static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT ==
                  CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT,
              "");
#endif
#endif  // HIP_TEST_HAS_EMULATION_MANTISSA_SPECIAL

/**
 * Test Description
 * ------------------------
 *    - Assigns each hipEmulationStrategy enumerator to a variable of type
 *      hipEmulationStrategy_t and checks the documented value. On NVIDIA the
 *      values are also checked at compile time against CUDA_EMULATION_STRATEGY_*.
 * Test source
 * ------------------------
 *    - catch/unit/library_types/hipEmulationTypes.cc
 * Test requirements
 * ------------------------
 *    - AMD, or NVIDIA with CUDA 13.0 or later
 */
// @asserts: hipEmulationStrategy - enumerator values are 0/1/2 and match CUDA
HIP_TEST_CASE(Unit_hipEmulationStrategy_Values) {
#if HIP_TEST_HAS_EMULATION_STRATEGY
  hipEmulationStrategy_t strategy = HIP_EMULATION_STRATEGY_DEFAULT;
  REQUIRE(static_cast<int>(strategy) == 0);
  strategy = HIP_EMULATION_STRATEGY_PERFORMANT;
  REQUIRE(static_cast<int>(strategy) == 1);
  hipEmulationStrategy alias = HIP_EMULATION_STRATEGY_EAGER;
  REQUIRE(static_cast<int>(alias) == 2);
#else
  HIP_SKIP_TEST("hipEmulationStrategy requires CUDA 13.0 or later on NVIDIA");
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Assigns each hipEmulationMantissaControl enumerator to a variable of type
 *      hipEmulationMantissaControl_t and checks the documented value. On NVIDIA
 *      the values are also checked at compile time against
 *      CUDA_EMULATION_MANTISSA_CONTROL_*.
 * Test source
 * ------------------------
 *    - catch/unit/library_types/hipEmulationTypes.cc
 * Test requirements
 * ------------------------
 *    - AMD, or NVIDIA with CUDA 13.1 or later
 */
// @asserts: hipEmulationMantissaControl - enumerator values are 0/1 and match CUDA
HIP_TEST_CASE(Unit_hipEmulationMantissaControl_Values) {
#if HIP_TEST_HAS_EMULATION_MANTISSA_SPECIAL
  hipEmulationMantissaControl_t control = HIP_EMULATION_MANTISSA_CONTROL_DYNAMIC;
  REQUIRE(static_cast<int>(control) == 0);
  hipEmulationMantissaControl alias = HIP_EMULATION_MANTISSA_CONTROL_FIXED;
  REQUIRE(static_cast<int>(alias) == 1);
#else
  HIP_SKIP_TEST("hipEmulationMantissaControl requires CUDA 13.1 or later on NVIDIA");
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Checks the documented hipEmulationSpecialValuesSupport values and that the
 *      INFINITY and NAN flags combine with bitwise OR into a valid
 *      hipEmulationSpecialValuesSupport_t. On NVIDIA the values are also checked
 *      at compile time against CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_*.
 * Test source
 * ------------------------
 *    - catch/unit/library_types/hipEmulationTypes.cc
 * Test requirements
 * ------------------------
 *    - AMD, or NVIDIA with CUDA 13.1 or later
 */
// @asserts: hipEmulationSpecialValuesSupport - values are 0/1/2/0xFFFF, OR-combine, match CUDA
HIP_TEST_CASE(Unit_hipEmulationSpecialValuesSupport_Values) {
#if HIP_TEST_HAS_EMULATION_MANTISSA_SPECIAL
  hipEmulationSpecialValuesSupport_t support = HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NONE;
  REQUIRE(static_cast<int>(support) == 0);

  support = static_cast<hipEmulationSpecialValuesSupport_t>(
      HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY | HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN);
  REQUIRE(static_cast<int>(support) == 3);
  REQUIRE((static_cast<int>(support) & HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY) != 0);
  REQUIRE((static_cast<int>(support) & HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN) != 0);

  hipEmulationSpecialValuesSupport alias = HIP_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT;
  REQUIRE(static_cast<int>(alias) == 0xFFFF);
#else
  HIP_SKIP_TEST("hipEmulationSpecialValuesSupport requires CUDA 13.1 or later on NVIDIA");
#endif
}

/**
 * End doxygen group LibraryTypesTest.
 * @}
 */
