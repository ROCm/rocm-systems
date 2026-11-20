/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "half_precision_common.hh"
#include "casting_common.hh"
#include "half_cast_reference.hh"

#include <array>

/**
 * @addtogroup HalfPrecisionCastingFloat HalfPrecisionCastingFloat
 * @{
 * @ingroup MathTest
 */

namespace {

Float16 HalfFromBits(uint16_t bits) {
  __half_raw raw{};
  raw.x = bits;
  return Float16{__half{raw}};
}

struct DirectedHalfCase {
  float input;
  uint16_t downward;
  uint16_t upward;
  uint16_t toward_zero;
};

// Literal binary16 endpoints keep these checks independent of the reference
// algorithm. Include both signs at zero, subnormal/normal boundaries, ties and
// overflow, as well as the original small positive regression values.
constexpr DirectedHalfCase kDirectedHalfCases[] = {
    {0.0f, 0x0000, 0x0000, 0x0000},
    {-0.0f, 0x8000, 0x8000, 0x8000},
    {0x1p-149f, 0x0000, 0x0001, 0x0000},
    {-0x1p-149f, 0x8001, 0x8000, 0x8000},
    {0x1p-126f, 0x0000, 0x0001, 0x0000},
    {-0x1p-126f, 0x8001, 0x8000, 0x8000},
    {0x1p-25f, 0x0000, 0x0001, 0x0000},
    {-0x1p-25f, 0x8001, 0x8000, 0x8000},
    {0x1p-24f, 0x0001, 0x0001, 0x0001},
    {-0x1p-24f, 0x8001, 0x8001, 0x8001},
    {0x1.8p-24f, 0x0001, 0x0002, 0x0001},
    {-0x1.8p-24f, 0x8002, 0x8001, 0x8001},
    {0x1.ff8p-15f, 0x03ff, 0x03ff, 0x03ff},
    {-0x1.ff8p-15f, 0x83ff, 0x83ff, 0x83ff},
    {0x1.ffcp-15f, 0x03ff, 0x0400, 0x03ff},
    {-0x1.ffcp-15f, 0x8400, 0x83ff, 0x83ff},
    {0x1p-14f, 0x0400, 0x0400, 0x0400},
    {-0x1p-14f, 0x8400, 0x8400, 0x8400},
    {0x1.002p-14f, 0x0400, 0x0401, 0x0400},
    {-0x1.002p-14f, 0x8401, 0x8400, 0x8400},
    {0x1.fffffep-1f, 0x3bff, 0x3c00, 0x3bff},
    {-0x1.fffffep-1f, 0xbc00, 0xbbff, 0xbbff},
    {1.0f, 0x3c00, 0x3c00, 0x3c00},
    {-1.0f, 0xbc00, 0xbc00, 0xbc00},
    {0x1.000002p0f, 0x3c00, 0x3c01, 0x3c00},
    {-0x1.000002p0f, 0xbc01, 0xbc00, 0xbc00},
    {0x1.002p0f, 0x3c00, 0x3c01, 0x3c00},
    {-0x1.002p0f, 0xbc01, 0xbc00, 0xbc00},
    {0x1.004p0f, 0x3c01, 0x3c01, 0x3c01},
    {-0x1.004p0f, 0xbc01, 0xbc01, 0xbc01},
    {0x1.006p0f, 0x3c01, 0x3c02, 0x3c01},
    {-0x1.006p0f, 0xbc02, 0xbc01, 0xbc01},
    {0x1.ffbffep15f, 0x7bfe, 0x7bff, 0x7bfe},
    {-0x1.ffbffep15f, 0xfbff, 0xfbfe, 0xfbfe},
    {65504.0f, 0x7bff, 0x7bff, 0x7bff},
    {-65504.0f, 0xfbff, 0xfbff, 0xfbff},
    {0x1.ffc002p15f, 0x7bff, 0x7c00, 0x7bff},
    {-0x1.ffc002p15f, 0xfc00, 0xfbff, 0xfbff},
    {65520.0f, 0x7bff, 0x7c00, 0x7bff},
    {-65520.0f, 0xfc00, 0xfbff, 0xfbff},
    {65536.0f, 0x7bff, 0x7c00, 0x7bff},
    {-65536.0f, 0xfc00, 0xfbff, 0xfbff},
    {std::numeric_limits<float>::max(), 0x7bff, 0x7c00, 0x7bff},
    {std::numeric_limits<float>::lowest(), 0xfc00, 0xfbff, 0xfbff},
    {std::numeric_limits<float>::infinity(), 0x7c00, 0x7c00, 0x7c00},
    {-std::numeric_limits<float>::infinity(), 0xfc00, 0xfc00, 0xfc00},
    {0.8859e-06f, 0x000e, 0x000f, 0x000e},
    {1.5454e-07f, 0x0002, 0x0003, 0x0002},
    {6.5955e-08f, 0x0001, 0x0002, 0x0001},
    {2.7955e-08f, 0x0000, 0x0001, 0x0000},
    {3.7956e-09f, 0x0000, 0x0001, 0x0000},
    {4.8995e-10f, 0x0000, 0x0001, 0x0000},
    {5.7997e-15f, 0x0000, 0x0001, 0x0000},
    {6.2117e-20f, 0x0000, 0x0001, 0x0000},
    {7.4999e-25f, 0x0000, 0x0001, 0x0000},
    {8.9999e-30f, 0x0000, 0x0001, 0x0000},
    {9.0001e-35f, 0x0000, 0x0001, 0x0000},
};

template <HalfRoundingMode Mode> void DirectedHalfBoundaryTest(kernel_sig<Float16, float> kernel) {
  constexpr size_t count = std::size(kDirectedHalfCases);
  std::array<float, count> input{};
  std::array<uint16_t, count> expected{};
  for (size_t i = 0; i < count; ++i) {
    const auto& test = kDirectedHalfCases[i];
    input[i] = test.input;
    expected[i] = Mode == HalfRoundingMode::Downward ? test.downward
                  : Mode == HalfRoundingMode::Upward ? test.upward
                                                     : test.toward_zero;
    INFO("Reference input: " << input[i]);
    REQUIRE(RoundFloatToHalfBits(input[i], Mode) == expected[i]);
  }

  LinearAllocGuard<float> input_dev{LinearAllocs::hipMalloc, sizeof(input)};
  LinearAllocGuard<Float16> output{LinearAllocs::hipMallocManaged, count * sizeof(Float16)};
  HIP_CHECK(hipMemcpy(input_dev.ptr(), input.data(), sizeof(input), hipMemcpyHostToDevice));
  kernel<<<1, 64>>>(output.ptr(), count, input_dev.ptr());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  for (size_t i = 0; i < count; ++i) {
    INFO("Device input: " << input[i]);
    REQUIRE(output.ptr()[i] == HalfFromBits(expected[i]));
  }
}

}  // namespace

#define CAST_FLOAT2HALF_TEST_DEF(kern_name, round_dir)                                             \
  CAST_KERNEL_DEF(kern_name, Float16, float)                                                       \
  Float16 kern_name##_ref(float arg) {                                                             \
    return HalfFromBits(RoundFloatToHalfBits(arg, round_dir));                                     \
  }                                                                                                \
                                                                                                   \
  HIP_TEST_CASE(Unit_Device_##kern_name##_Accuracy_Limited_Positive) {                             \
    Float16 (*ref)(float) = kern_name##_ref;                                                       \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<Float16>(),   \
                                  std::numeric_limits<float>::lowest(),                            \
                                  std::numeric_limits<float>::max());                              \
  }

#define CAST_FLOAT2HALF_RN_TEST_DEF(kern_name)                                                     \
  CAST_KERNEL_DEF(kern_name, Float16, float)                                                       \
  CAST_REF_DEF(kern_name, Float16, float)                                                          \
                                                                                                   \
  HIP_TEST_CASE(Unit_Device_##kern_name##_Accuracy_Positive) {                                     \
    Float16 (*ref)(float) = kern_name##_ref;                                                       \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<Float16>(),   \
                                  std::numeric_limits<float>::lowest(),                            \
                                  std::numeric_limits<float>::max());                              \
  }

/**
 * Test Description
 * ------------------------
 *    - Checks `__float2half_rd` over the full finite float range against an independent
 * binary16 downward rounding reference. Results must match exactly.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2HALF_TEST_DEF(__float2half_rd, HalfRoundingMode::Downward)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2half_rn` for all possible inputs. The results are compared
 * against reference function which performs float cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2HALF_RN_TEST_DEF(__float2half_rn)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__float2half` for all possible inputs. The results are compared against
 * reference function which performs float cast to __half.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2HALF_RN_TEST_DEF(__float2half)

/**
 * Test Description
 * ------------------------
 *    - Checks `__float2half_ru` over the full finite float range against an independent
 * binary16 upward rounding reference. Results must match exactly.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2HALF_TEST_DEF(__float2half_ru, HalfRoundingMode::Upward)

/**
 * Test Description
 * ------------------------
 *    - Checks `__float2half_rz` over the full finite float range against an independent
 * binary16 toward-zero rounding reference. Results must match exactly.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
CAST_FLOAT2HALF_TEST_DEF(__float2half_rz, HalfRoundingMode::TowardZero)

/**
 * Test Description
 * ------------------------
 *    - Checks `__float2half_rd` against exact binary16 endpoints for both signs at
 * zero, subnormal/normal boundaries, ties, overflow and the original small positive inputs.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Device___float2half_rd_SmallVals_Sanity_Positive) {
  DirectedHalfBoundaryTest<HalfRoundingMode::Downward>(__float2half_rd_kernel);
}

/**
 * Test Description
 * ------------------------
 *    - Checks `__float2half_ru` against exact binary16 endpoints for both signs at
 * zero, subnormal/normal boundaries, ties, overflow and the original small positive inputs.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Device___float2half_ru_SmallVals_Sanity_Positive) {
  DirectedHalfBoundaryTest<HalfRoundingMode::Upward>(__float2half_ru_kernel);
}

/**
 * Test Description
 * ------------------------
 *    - Checks `__float2half_rz` against exact binary16 endpoints for both signs at
 * zero, subnormal/normal boundaries, ties, overflow and the original small positive inputs.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Device___float2half_rz_SmallVals_Sanity_Positive) {
  DirectedHalfBoundaryTest<HalfRoundingMode::TowardZero>(__float2half_rz_kernel);
}

CAST_KERNEL_DEF(__half2float, float, Float16)
CAST_REF_DEF(__half2float, float, Float16)

/**
 * Test Description
 * ------------------------
 *    - Tests that checks `__half2float` for all possible inputs. The results are compared against
 * reference function which performs __half cast to float.
 *
 * Test source
 * ------------------------
 *    - unit/math/casting_half_float_funcs.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Device___half2float_Accuracy_Positive) {
  float (*ref)(Float16) = __half2float_ref;
  UnaryHalfPrecisionTest(__half2float_kernel, ref, EqValidatorBuilderFactory<float>());
}

/**
 * End doxygen group MathTest.
 * @}
 */
