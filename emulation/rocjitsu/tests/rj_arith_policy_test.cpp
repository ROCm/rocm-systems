// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_arith/rj_arith.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"

#include <gtest/gtest.h>
#include <stdexcept>

namespace rocjitsu::amdgpu {
namespace {

TEST(ArithPolicyTest, DefaultForwardsTemporaryCompatibility) {
  ArithmeticExecution execution;
  EXPECT_EQ(execution.flags() & RJ_ARITH_SEMANTICS_MASK, RJ_ARITH_SEMANTICS_COMPATIBILITY);
  EXPECT_TRUE(execution.reports_fp_status());
}

TEST(ArithPolicyTest, ConfigurationNamesMapDirectlyToArithSemantics) {
  const char *names[] = {"loose", "bit_exact_empirical_allow_loose",
                         "bit_exact_empirical_no_fp_status", "bit_exact_empirical",
                         "compatibility"};
  const uint64_t modes[] = {
      RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
      RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL,
      RJ_ARITH_SEMANTICS_COMPATIBILITY};
  for (unsigned i = 0; i < 5; ++i) {
    auto policy = parse_arithmetic_policy(names[i]);
    ArithmeticExecution execution(policy);
    EXPECT_EQ((static_cast<uint64_t>(policy.semantics) << 16), modes[i]);
    EXPECT_EQ(execution.flags() & RJ_ARITH_SEMANTICS_MASK, modes[i]);
    EXPECT_STREQ(arithmetic_semantics_name(policy.semantics), names[i]);
    EXPECT_EQ(execution.reports_fp_status(), i >= 3);
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
