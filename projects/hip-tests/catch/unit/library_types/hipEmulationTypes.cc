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
 * All checks are static_asserts, so a mismatch fails the build. The single test
 * case only registers the check in CTest and reports a skip on NVIDIA when the
 * CUDA version is too old to provide the types.
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

// Both names must be usable as variable types.
constexpr hipEmulationStrategy_t kStrategy = HIP_EMULATION_STRATEGY_EAGER;
constexpr hipEmulationStrategy kStrategyAlias = HIP_EMULATION_STRATEGY_PERFORMANT;
static_assert(kStrategy == 2 && kStrategyAlias == 1, "hipEmulationStrategy variables");

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

constexpr hipEmulationMantissaControl_t kMantissa = HIP_EMULATION_MANTISSA_CONTROL_FIXED;
constexpr hipEmulationMantissaControl kMantissaAlias = HIP_EMULATION_MANTISSA_CONTROL_DYNAMIC;
static_assert(kMantissa == 1 && kMantissaAlias == 0, "hipEmulationMantissaControl variables");

// Documented values; these match CUDA's cudaEmulationSpecialValuesSupport.
static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NONE == 0,
              "HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NONE");
static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY == 1,
              "HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY");
static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN == 2,
              "HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN");
static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT == 0xFFFF,
              "HIP_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT");

// The INFINITY and NAN flags form a bitmask and combine with bitwise OR.
constexpr hipEmulationSpecialValuesSupport_t kSpecial =
    static_cast<hipEmulationSpecialValuesSupport_t>(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY |
                                                    HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN);
constexpr hipEmulationSpecialValuesSupport kSpecialAlias =
    HIP_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT;
static_assert(kSpecial == 3 && kSpecialAlias == 0xFFFF,
              "hipEmulationSpecialValuesSupport variables");

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
 *    - The checks in this file are static_asserts on the documented enumerator
 *      values (and, on NVIDIA, against the CUDA_EMULATION_* values and types).
 *      Reaching this test case means they all passed at compile time. On NVIDIA
 *      the test is skipped if the CUDA version does not provide all three types.
 * Test source
 * ------------------------
 *    - catch/unit/library_types/hipEmulationTypes.cc
 * Test requirements
 * ------------------------
 *    - AMD, or NVIDIA with CUDA 13.1 or later (CUDA 13.0 checks hipEmulationStrategy only)
 */
HIP_TEST_CASE(Unit_hipEmulationTypes_CompileChecks) {
#if HIP_TEST_HAS_EMULATION_STRATEGY && HIP_TEST_HAS_EMULATION_MANTISSA_SPECIAL
  SUCCEED("hipEmulation* type and value checks passed at compile time");
#elif HIP_TEST_HAS_EMULATION_STRATEGY
  HIP_SKIP_TEST(
      "hipEmulationMantissaControl and hipEmulationSpecialValuesSupport require CUDA 13.1 or "
      "later on NVIDIA; only hipEmulationStrategy was checked");
#else
  HIP_SKIP_TEST("hipEmulation* types require CUDA 13.0 or later on NVIDIA");
#endif
}

/**
 * End doxygen group LibraryTypesTest.
 * @}
 */
