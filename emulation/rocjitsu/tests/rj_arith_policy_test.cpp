// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_arith/rj_arith.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "util/simd.h"

#include <gtest/gtest.h>
#include <stdexcept>

namespace rocjitsu::amdgpu {
namespace {

TEST(ArithPolicyTest, DefaultForwardsTemporaryCompatibility) {
  ArithmeticExecution execution;
  const uint64_t expected_cpu =
      util::force_scalar() ? RJ_ARITH_CPU_BASELINE : rj_arith_init_cpu_flags();
  EXPECT_EQ(execution.flags() & RJ_ARITH_CPU_MASK, expected_cpu & RJ_ARITH_CPU_MASK);
  EXPECT_EQ(execution.flags() & RJ_ARITH_SEMANTICS_MASK, RJ_ARITH_SEMANTICS_COMPATIBILITY);
}

TEST(ArithPolicyTest, ConfigurationNamesMapDirectlyToArithSemantics) {
  const struct {
    const char *name;
    uint64_t mode;
  } cases[] = {
      {"loose", RJ_ARITH_SEMANTICS_LOOSE},
      {"bit_exact_empirical_allow_loose", RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE},
      {"bit_exact_empirical_no_fp_status", RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS},
      {"bit_exact_empirical", RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL},
      {"compatibility", RJ_ARITH_SEMANTICS_COMPATIBILITY},
  };
  for (const auto &sample : cases) {
    SCOPED_TRACE(sample.name);
    auto policy = parse_arithmetic_policy(sample.name);
    ArithmeticExecution execution(policy);
    EXPECT_EQ((static_cast<uint64_t>(policy.semantics) << 16), sample.mode);
    EXPECT_EQ(execution.flags() & RJ_ARITH_SEMANTICS_MASK, sample.mode);
    EXPECT_STREQ(arithmetic_semantics_name(policy.semantics), sample.name);
    EXPECT_EQ(execution.flags() & RJ_ARITH_CPU_MASK,
              ArithmeticExecution().flags() & RJ_ARITH_CPU_MASK);
  }
}

TEST(ArithPolicyTest, RejectsUnknownReservedAndRemovedNames) {
  for (auto name : {"typo", "exact_empirical", "bit_exact_empirical_fp_status", "4", "7", ""})
    EXPECT_THROW(parse_arithmetic_policy(name), std::invalid_argument);
}

} // namespace
} // namespace rocjitsu::amdgpu
