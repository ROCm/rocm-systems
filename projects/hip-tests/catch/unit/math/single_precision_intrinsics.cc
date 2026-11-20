/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include "unary_common.hh"
#include "binary_common.hh"
#include "ternary_common.hh"

/********** Unary Functions **********/

#define MATH_UNARY_SP_KERNEL_DEF(func_name)                                                        \
  __global__ void func_name##_kernel(float* const ys, const size_t num_xs, float* const xs) {      \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (size_t i = tid; i < num_xs; i += stride) {                                                \
      ys[i] = func_name(xs[i]);                                                                    \
    }                                                                                              \
  }

#define MATH_UNARY_SP_TEST_DEF_IMPL(func_name, ref_func, validator_builder)                        \
  HIP_TEST_CASE(Unit_Device_##func_name##_Accuracy_Positive) {                                     \
    UnarySinglePrecisionTest(func_name##_kernel, ref_func, validator_builder);                     \
  }

#define MATH_UNARY_SP_TEST_DEF(func_name, ref_func)                                                \
  MATH_UNARY_SP_TEST_DEF_IMPL(func_name, ref_func, func_name##_validator_builder)

#define MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(func_name)                                             \
  static std::unique_ptr<MatcherBase<float>> func_name##_validator_builder(float target, float x)


static float __frcp_rn_ref(float x) { return 1.0f / x; }

MATH_UNARY_SP_KERNEL_DEF(__frcp_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__frcp_rn(x)` for all possible inputs. The error bounds are
 * IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF_IMPL(__frcp_rn, __frcp_rn_ref, EqValidatorBuilderFactory<float>());


MATH_UNARY_SP_KERNEL_DEF(__fsqrt_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fsqrt_rn(x)` for all possible inputs. The results are
 * compared against reference function `float std::sqrt(float)`. RDNA4 permits 1 ULP;
 * other targets retain exact comparison.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Device___fsqrt_rn_Accuracy_Positive) {
  const auto validator_builder = [rdna4 = IsNavi4X()](
                                     float target, float) -> std::unique_ptr<MatcherBase<float>> {
  // RDNA4 ISA, V_SQRT_F32: 1 ULP. The compiler scales subnormal inputs
  // for __ocml_native_sqrt_f32, so retain the unflushed reference here.
#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
    // This opt-in selects the explicitly correctly rounded OCML operation.
    (void)rdna4;
#else
    if (rdna4 && std::isfinite(target) && target != 0.0f) {
      return ULPValidatorBuilderFactory<float>(1)(target);
    }
#endif
    return EqValidatorBuilderFactory<float>()(target);
  };
  UnarySinglePrecisionTest(__fsqrt_rn_kernel, static_cast<float (*)(float)>(std::sqrt),
                           validator_builder);
}


static float __frsqrt_rn_ref(float x) {
  // Round once to float after computing the reciprocal square root in double.
  return static_cast<float>(1.0 / std::sqrt(static_cast<double>(x)));
}

MATH_UNARY_SP_KERNEL_DEF(__frsqrt_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__frsqrt_rn(x)` for all possible inputs. The results are
 * compared against a double-precision reciprocal square root rounded to float. RDNA4
 * permits 1 ULP and flushes subnormal inputs to signed zero; other targets retain exact comparison.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Device___frsqrt_rn_Accuracy_Positive) {
  const auto validator_builder = [rdna4 = IsNavi4X()](
                                     float target, float x) -> std::unique_ptr<MatcherBase<float>> {
    if (rdna4) {
      // __frsqrt_rn directly uses V_RSQ_F32: 1 ULP and input denormals flushed
      // to signed zero (RDNA4 ISA, VOP1 instructions).
      if (std::fpclassify(x) == FP_SUBNORMAL) {
        target = std::copysign(std::numeric_limits<float>::infinity(), x);
      }
      if (std::isfinite(target) && target != 0.0f) {
        return ULPValidatorBuilderFactory<float>(1)(target);
      }
    }
    return EqValidatorBuilderFactory<float>()(target);
  };
  UnarySinglePrecisionTest(__frsqrt_rn_kernel, __frsqrt_rn_ref, validator_builder);
}


class ExponentialUlpValidator : public MatcherBase<float> {
 public:
  ExponentialUlpValidator(float target, int64_t ulps, bool flush_denormals)
      : target_{target},
        ulps_{ulps},
        flush_denormals_{flush_denormals},
        validator_{ULPValidatorBuilderFactory<float>(ulps)(target)} {}

  bool match(const float& value) const override {
    // Exponentials cannot produce a negative result, including negative zero.
    // A saturated ULP allowance can otherwise span both signs.
    if (std::signbit(value)) return false;
    if (flush_denormals_) {
      if (std::fpclassify(value) == FP_SUBNORMAL) return false;
      if (value == 0.0f) {
        // Flush the permitted result interval, not just the reference. A
        // near-boundary approximation can cross in either direction.
        const float largest_subnormal = std::nextafter(std::numeric_limits<float>::min(), 0.0f);
        if (std::fpclassify(target_) == FP_SUBNORMAL || validator_->match(largest_subnormal)) {
          return true;
        }
      }
    }
    return validator_->match(value);
  }

  std::string describe() const override {
    // Catch builds its displayed range one nextafter step per ULP. Large
    // exponential error budgets must not make failure reporting take billions of steps.
    std::ostringstream message;
    message << "is within " << ulps_ << " ULPs of "
            << std::setprecision(std::numeric_limits<float>::max_digits10) << target_;
    if (flush_denormals_) message << " after flushing subnormal results to zero";
    return message.str();
  }

 private:
  float target_;
  int64_t ulps_;
  bool flush_denormals_;
  std::unique_ptr<MatcherBase<float>> validator_;
};

static std::unique_ptr<MatcherBase<float>> ExponentialValidator(float target, float x,
                                                                float error_scale,
                                                                bool flush_denormals = false) {
  // Non-finite inputs have exact special results, not a finite error budget.
  // Finite inputs keep their ULP allowance even when the reference overflows
  // or rounds to zero: an approximate result can cross either boundary.
  if (!std::isfinite(x)) {
    return EqValidatorBuilderFactory<float>()(target);
  }

  // Clamp before converting to an integer: even finite x can overflow the
  // float product. Catch requires a ULP budget strictly below UINT32_MAX.
  const double ulp_err = 2.0 + std::floor(std::abs(error_scale * x));
  constexpr auto max_ulps = std::numeric_limits<uint32_t>::max() - 1;
  const auto bounded_ulps = static_cast<int64_t>(std::min(ulp_err, static_cast<double>(max_ulps)));
  return std::make_unique<ExponentialUlpValidator>(target, bounded_ulps, flush_denormals);
}

HIP_TEST_CASE(Unit_ExponentialValidator_ResultBounds) {
  const bool flush_denormals = GENERATE(false, true);
  const float error_scale = GENERATE(1.16f, 2.95f);
  const float infinity = std::numeric_limits<float>::infinity();
  const float smallest_normal = std::numeric_limits<float>::min();
  const float smallest_subnormal = std::numeric_limits<float>::denorm_min();

  SECTION("Saturated allowances reject every negative result") {
    const float x = GENERATE(-1.0e10f, 1.0e10f);
    const auto validator =
        ExponentialValidator(x < 0.0f ? 0.0f : infinity, x, error_scale, flush_denormals);
    for (const float value :
         {-0.0f, -smallest_subnormal, -1.0f, -std::numeric_limits<float>::max(), -infinity}) {
      CAPTURE(x, value, flush_denormals, error_scale);
      REQUIRE_FALSE(validator->match(value));
    }
    REQUIRE(validator->match(0.0f));
    REQUIRE(validator->match(std::numeric_limits<float>::max()));
    REQUIRE(validator->match(infinity));
    REQUIRE_FALSE(validator->match(std::numeric_limits<float>::quiet_NaN()));
  }

  SECTION("Positive overflow and underflow retain their boundary allowances") {
    const auto overflow = ExponentialValidator(infinity, 100.0f, error_scale, flush_denormals);
    REQUIRE(overflow->match(infinity));
    REQUIRE(overflow->match(std::numeric_limits<float>::max()));

    const auto underflow = ExponentialValidator(0.0f, -110.0f, error_scale, flush_denormals);
    REQUIRE(underflow->match(0.0f));
    REQUIRE(underflow->match(smallest_subnormal) == !flush_denormals);

    const auto normal_boundary =
        ExponentialValidator(smallest_normal, -87.0f, error_scale, flush_denormals);
    REQUIRE(normal_boundary->match(smallest_normal));
    REQUIRE(normal_boundary->match(0.0f) == flush_denormals);
  }

  SECTION("Ordinary results still obey their ULP allowance") {
    const auto validator = ExponentialValidator(1.0f, 0.0f, error_scale, flush_denormals);
    float value = std::nextafter(std::nextafter(1.0f, infinity), infinity);
    REQUIRE(validator->match(value));
    REQUIRE_FALSE(validator->match(std::nextafter(value, infinity)));
  }
}

static auto ExponentialValidatorBuilder(float error_scale) {
  return [error_scale, rdna4 = IsNavi4X()](float target, float x) {
    // The AMD wrappers use V_EXP_F32, whose RDNA4 contract flushes denormals.
    return ExponentialValidator(target, x, error_scale, rdna4);
  };
}

MATH_UNARY_SP_KERNEL_DEF(__expf);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__expf(x)` for all possible inputs. The results are
 * compared against reference function `double std::exp(double)`. The maximum ulp error is `2 +
 * floor(abs(1.16 * x))`. RDNA4 flushes subnormal outputs to zero.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF_IMPL(__expf, static_cast<double (*)(double)>(std::exp),
                            ExponentialValidatorBuilder(1.16f));

MATH_UNARY_SP_KERNEL_DEF(__exp10f);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__exp10f(x)` for all possible inputs. The results are
 * compared against reference function `double exp10(double)`. The maximum ulp error is `2 +
 * floor(abs(2.95 * x))`. RDNA4 flushes subnormal outputs to zero.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF_IMPL(__exp10f, static_cast<double (*)(double)>(exp10),
                            ExponentialValidatorBuilder(2.95f));


MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(__logf) {
  if (0.5f <= x && x <= 2.0f) {
    const auto abs_err = std::pow(2.0, -21.41);
    return AbsValidatorBuilderFactory<float>(abs_err)(target);
  } else {
    return ULPValidatorBuilderFactory<float>(3)(target);
  }
}

MATH_UNARY_SP_KERNEL_DEF(__logf);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__logf(x)` for all possible inputs. The results are
 * compared against reference function `double std::log(double)`. For `x` in [0.5, 2], the maximum
 * absolute error is 2^-21.41, otherwise, the maximum ulp error is 3.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF(__logf, static_cast<double (*)(double)>(std::log));


MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(__log2f) {
  if (0.5f <= x && x <= 2.0f) {
    const auto abs_err = std::pow(2.0, -22.0);
    return AbsValidatorBuilderFactory<float>(abs_err)(target);
  } else {
    return ULPValidatorBuilderFactory<float>(2)(target);
  }
}

MATH_UNARY_SP_KERNEL_DEF(__log2f);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__log2f(x)` for all possible inputs. The results are
 * compared against reference function `double std::log2(double)`. For `x` in [0.5, 2], the maximum
 * absolute error is 2^-22, otherwise, the maximum ulp error is 2. RDNA4 flushes subnormal
 * inputs to signed zero before taking the logarithm.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Device___log2f_Accuracy_Positive) {
  const auto validator_builder = [rdna4 = IsNavi4X()](
                                     float target, float x) -> std::unique_ptr<MatcherBase<float>> {
    // __log2f directly uses V_LOG_F32. Both signs of subnormal input flush
    // to zero and produce -infinity under the RDNA4 instruction contract.
    if (rdna4 && std::fpclassify(x) == FP_SUBNORMAL) {
      return EqValidatorBuilderFactory<float>()(-std::numeric_limits<float>::infinity());
    }
    return __log2f_validator_builder(target, x);
  };
  UnarySinglePrecisionTest(__log2f_kernel, static_cast<double (*)(double)>(std::log2),
                           validator_builder);
}


MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(__log10f) {
  if (0.5f <= x && x <= 2.0f) {
    const auto abs_err = std::pow(2.0, -24.0);
    return AbsValidatorBuilderFactory<float>(abs_err)(target);
  } else {
    return ULPValidatorBuilderFactory<float>(3)(target);
  }
}

MATH_UNARY_SP_KERNEL_DEF(__log10f);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__log10f(x)` for all possible inputs. The results are
 * compared against reference function `double std::log10(double)`. For `x` in [0.5, 2], the maximum
 * absolute error is 2^-24, otherwise, the maximum ulp error is 3.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF(__log10f, static_cast<double (*)(double)>(std::log10));


MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(__sinf) {
  if (-M_PI <= x && x <= M_PI) {
    const auto abs_err = std::pow(2.0, -21.41);
    return AbsValidatorBuilderFactory<float>(abs_err)(target);
  } else {
    return NopValidatorBuilderFactory<float>()();
  }
}

MATH_UNARY_SP_KERNEL_DEF(__sinf);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__sinf(x)` for all possible inputs. The results are
 * compared against reference function `double std::sin(double)`. For `x` in [-PI, PI], the maximum
 * absolute error is 2^-21.41, and larger otherwise.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF(__sinf, static_cast<double (*)(double)>(std::sin));


__device__ float __sincosf_sin(float x) {
  float sin, cos;
  __sincosf(x, &sin, &cos);
  return sin;
}

MATH_UNARY_SP_KERNEL_DEF(__sincosf_sin);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__sincosf(x, sptr, cptr)` for all possible inputs. The
 * results in `sptr` are compared against reference function `double std::sin(double)`. For `x` in
 * [-PI, PI], the maximum absolute error is 2^-21.41, and larger otherwise.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF_IMPL(__sincosf_sin, static_cast<double (*)(double)>(std::sin),
                            __sinf_validator_builder);


MATH_UNARY_SP_VALIDATOR_BUILDER_DEF(__cosf) {
  if (-M_PI <= x && x <= M_PI) {
    const auto abs_err = std::pow(2.0, -21.19);
    return AbsValidatorBuilderFactory<float>(abs_err)(target);
  } else {
    return NopValidatorBuilderFactory<float>()();
  }
}

MATH_UNARY_SP_KERNEL_DEF(__cosf);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__cosf(x)` for all possible inputs. The results are
 * compared against reference function `double std::cos(double)`. For `x` in [-PI, PI], the maximum
 * absolute error is 2^-21.19, and larger otherwise.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF(__cosf, static_cast<double (*)(double)>(std::cos));


__device__ float __sincosf_cos(float x) {
  float sin, cos;
  __sincosf(x, &sin, &cos);
  return cos;
}

MATH_UNARY_SP_KERNEL_DEF(__sincosf_cos);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__sincosf(x, sptr, cptr)` for all possible inputs. The
 * results in `cptr` are compared against reference function `double std::cos(double)`. For `x` in
 * [-PI, PI], the maximum absolute error is 2^-21.19, and larger otherwise.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_UNARY_SP_TEST_DEF_IMPL(__sincosf_cos, static_cast<double (*)(double)>(std::cos),
                            __cosf_validator_builder);


/********** Binary Functions **********/

#define MATH_BINARY_SP_KERNEL_DEF(func_name)                                                       \
  __global__ void func_name##_kernel(float* const ys, const size_t num_xs, float* const x1s,       \
                                     float* const x2s) {                                           \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (size_t i = tid; i < num_xs; i += stride) {                                                \
      ys[i] = func_name(x1s[i], x2s[i]);                                                           \
    }                                                                                              \
  }

#define MATH_BINARY_SP_TEST_DEF_IMPL(func_name, ref_func, validator_builder)                       \
  HIP_TEST_CASE(Unit_Device_##func_name##_Accuracy_Positive) {                                     \
    BinaryFloatingPointTest(func_name##_kernel, ref_func, validator_builder);                      \
  }

#define MATH_BINARY_SP_TEST_DEF(func_name, ref_func)                                               \
  MATH_BINARY_SP_TEST_DEF_IMPL(func_name, ref_func, func_name##_validator_builder)

#define MATH_BINARY_SP_VALIDATOR_BUILDER_DEF(func_name)                                            \
  static std::unique_ptr<MatcherBase<float>> func_name##_validator_builder(float target, float x1, \
                                                                           float x2)

static float __fadd_rn_ref(float x1, float x2) { return x1 + x2; }

MATH_BINARY_SP_KERNEL_DEF(__fadd_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fadd_rn(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_SP_TEST_DEF_IMPL(__fadd_rn, __fadd_rn_ref, EqValidatorBuilderFactory<float>());


static float __fsub_rn_ref(float x1, float x2) { return x1 - x2; }

MATH_BINARY_SP_KERNEL_DEF(__fsub_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fsub_rn(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_SP_TEST_DEF_IMPL(__fsub_rn, __fsub_rn_ref, EqValidatorBuilderFactory<float>());


static float __fmul_rn_ref(float x1, float x2) { return x1 * x2; }

MATH_BINARY_SP_KERNEL_DEF(__fmul_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fmul_rn(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_SP_TEST_DEF_IMPL(__fmul_rn, __fmul_rn_ref, EqValidatorBuilderFactory<float>());


static float __fdiv_rn_ref(float x1, float x2) { return x1 / x2; }

MATH_BINARY_SP_KERNEL_DEF(__fdiv_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fdiv_rn(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_SP_TEST_DEF_IMPL(__fdiv_rn, __fdiv_rn_ref, EqValidatorBuilderFactory<float>());


MATH_BINARY_SP_VALIDATOR_BUILDER_DEF(__fdividef) {
  x1 = 2.0f;
  const auto abs_x2 = std::abs(x2);
  if (std::pow(x1, -126.0f) <= abs_x2 && abs_x2 <= std::pow(x1, 126.0f)) {
    return ULPValidatorBuilderFactory<float>(2)(target);
  } else {
    return NopValidatorBuilderFactory<float>()();
  }
}

MATH_BINARY_SP_KERNEL_DEF(__fdividef);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fdividef(x,y)` against a table of difficult values,
 * followed by a large number of randomly generated values. For `|y|` in [2^-126, 2^126], the
 * maximum ulp error is 2.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_BINARY_SP_TEST_DEF(__fdividef, __fdiv_rn_ref);


/********** Ternary Functions **********/

#define MATH_TERNARY_SP_KERNEL_DEF(func_name)                                                      \
  __global__ void func_name##_kernel(float* const ys, const size_t num_xs, float* const x1s,       \
                                     float* const x2s, float* const x3s) {                         \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (size_t i = tid; i < num_xs; i += stride) {                                                \
      ys[i] = func_name(x1s[i], x2s[i], x3s[i]);                                                   \
    }                                                                                              \
  }

#define MATH_TERNARY_SP_TEST_DEF_IMPL(func_name, ref_func, validator_builder)                      \
  HIP_TEST_CASE(Unit_Device_##func_name##_Accuracy_Positive) {                                     \
    TernaryFloatingPointTest(func_name##_kernel, ref_func, validator_builder);                     \
  }

#define MATH_TERNARY_SP_TEST_DEF(func_name, ref_func, validator_builder)                           \
  MATH_TERNARY_SP_TEST_DEF_IMPL(func_name, ref_func, func_name##_validator_builder)

#define MATH_TERNARY_SP_VALIDATOR_BUILDER_DEF(func_name)                                           \
  static std::unique_ptr<MatcherBase<float>> func_name##_validator_builder(float target, float x1, \
                                                                           float x2, float x3)


MATH_TERNARY_SP_KERNEL_DEF(__fmaf_rn);

/**
 * Test Description
 * ------------------------
 *    - Tests the numerical accuracy of `__fmaf(x,y,z)` against a table of difficult values,
 * followed by a large number of randomly generated values. The error bounds are IEEE-compliant.
 *
 * Test source
 * ------------------------
 *    - unit/math/single_precision_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
MATH_TERNARY_SP_TEST_DEF_IMPL(__fmaf_rn, static_cast<float (*)(float, float, float)>(std::fma),
                              EqValidatorBuilderFactory<float>());
