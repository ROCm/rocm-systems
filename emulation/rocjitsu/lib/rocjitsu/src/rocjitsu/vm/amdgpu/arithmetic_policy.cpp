// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rj_arith/rj_arith.h"
#include "util/simd.h"

#include <stdexcept>

namespace rocjitsu::amdgpu {
namespace {

static_assert((uint64_t(ArithmeticSemantics::Loose) << 16) == RJ_ARITH_SEMANTICS_LOOSE);
static_assert((uint64_t(ArithmeticSemantics::BitExactEmpiricalAllowLoose) << 16) ==
              RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
static_assert((uint64_t(ArithmeticSemantics::BitExactEmpiricalNoFpStatus) << 16) ==
              RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
static_assert((uint64_t(ArithmeticSemantics::BitExactEmpirical) << 16) ==
              RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);

static_assert((uint64_t(ArithmeticSemantics::Compatibility) << 16) ==
              RJ_ARITH_SEMANTICS_COMPATIBILITY);

// Independent of other dynamic initializers. CU construction reads this only
// after module initialization; instruction execution uses its own cached flags.
const uint64_t detected_cpu_flags = rj_arith_init_cpu_flags();

} // namespace

ArithmeticPolicy parse_arithmetic_policy(std::string_view semantics) {
  for (auto mode : {ArithmeticSemantics::Loose, ArithmeticSemantics::BitExactEmpiricalAllowLoose,
                    ArithmeticSemantics::BitExactEmpiricalNoFpStatus,
                    ArithmeticSemantics::BitExactEmpirical, ArithmeticSemantics::Compatibility}) {
    if (semantics == arithmetic_semantics_name(mode))
      return {mode};
  }
  throw std::invalid_argument(
      "arithmetic.semantics must be loose, bit_exact_empirical_allow_loose, "
      "bit_exact_empirical_no_fp_status, bit_exact_empirical, or compatibility");
}

const char *arithmetic_semantics_name(ArithmeticSemantics semantics) {
  switch (semantics) {
  case ArithmeticSemantics::Compatibility:
    return "compatibility";
  case ArithmeticSemantics::Loose:
    return "loose";
  case ArithmeticSemantics::BitExactEmpiricalAllowLoose:
    return "bit_exact_empirical_allow_loose";
  case ArithmeticSemantics::BitExactEmpiricalNoFpStatus:
    return "bit_exact_empirical_no_fp_status";
  case ArithmeticSemantics::BitExactEmpirical:
    return "bit_exact_empirical";
  }
  throw std::invalid_argument("Invalid arithmetic semantics");
}

ArithmeticExecution::ArithmeticExecution(ArithmeticPolicy policy)
    : flags_(detected_cpu_flags | (static_cast<uint64_t>(policy.semantics) << 16)) {}

uint64_t ArithmeticExecution::flags() const {
  return util::force_scalar() ? flags_ & ~(RJ_ARITH_CPU_MASK | RJ_ARITH_CPU_FEATURES_MASK) : flags_;
}

} // namespace rocjitsu::amdgpu
