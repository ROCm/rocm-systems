// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The encoders in scalar_builders.h.
//
// Expected encodings are ground truth captured from
// `llvm-mc -triple=amdgcn -mcpu=<gfx942|gfx950|gfx1200> -show-encoding`.

#include "scalar_builders.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/rj_code.h"

#include <gtest/gtest.h>

namespace rocjitsu {
namespace test {
namespace {

// v_readfirstlane_b32 s4, v0; s_bcnt1_i32_b64 s8, s[6:7];
// s_ff1_i32_b64 s9, s[6:7] (s_ctz_i32_b64 on gfx1200); s_and_b32 s12, s10, 3;
// s_cbranch_scc0 5; s_cbranch_scc1 -4. Identical on gfx942 and gfx950.
TEST(ScalarBuilders, MatchLlvmOnCdna) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    EXPECT_EQ(build_v_readfirstlane_b32(4, 0, arch), 0x7e080500u);
    util::FailureOr<uint32_t> maybe_bcnt1_i32_b64 = build_s_bcnt1_i32_b64(8, 6, arch);
    ASSERT_TRUE(maybe_bcnt1_i32_b64.succeeded());
    EXPECT_EQ(maybe_bcnt1_i32_b64.value(), 0xbe880d06u);
    util::FailureOr<uint32_t> maybe_first_set_bit_b64 = build_s_first_set_bit_b64(9, 6, arch);
    ASSERT_TRUE(maybe_first_set_bit_b64.succeeded());
    EXPECT_EQ(maybe_first_set_bit_b64.value(), 0xbe891106u);
    util::FailureOr<uint32_t> maybe_and_b32 =
        build_s_and_b32(12, 10, scalar_positive_inline_u32(3), arch);
    ASSERT_TRUE(maybe_and_b32.succeeded());
    EXPECT_EQ(maybe_and_b32.value(), 0x860c830au);
    util::FailureOr<uint32_t> maybe_branch_if_scc_clear =
        build_s_cbranch_scc(/*scc1=*/false, 5, arch);
    ASSERT_TRUE(maybe_branch_if_scc_clear.succeeded());
    EXPECT_EQ(maybe_branch_if_scc_clear.value(), 0xbf840005u);
    util::FailureOr<uint32_t> maybe_branch_if_scc_set =
        build_s_cbranch_scc(/*scc1=*/true, -4, arch);
    ASSERT_TRUE(maybe_branch_if_scc_set.succeeded());
    EXPECT_EQ(maybe_branch_if_scc_set.value(), 0xbf85fffcu);
  }
}

TEST(ScalarBuilders, MatchLlvmOnRdna4) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  EXPECT_EQ(build_v_readfirstlane_b32(4, 0, kArch), 0x7e080500u);
  util::FailureOr<uint32_t> maybe_bcnt1_i32_b64 = build_s_bcnt1_i32_b64(8, 6, kArch);
  ASSERT_TRUE(maybe_bcnt1_i32_b64.succeeded());
  EXPECT_EQ(maybe_bcnt1_i32_b64.value(), 0xbe881906u);
  util::FailureOr<uint32_t> maybe_first_set_bit_b64 = build_s_first_set_bit_b64(9, 6, kArch);
  ASSERT_TRUE(maybe_first_set_bit_b64.succeeded());
  EXPECT_EQ(maybe_first_set_bit_b64.value(), 0xbe890906u);
  util::FailureOr<uint32_t> maybe_and_b32 =
      build_s_and_b32(12, 10, scalar_positive_inline_u32(3), kArch);
  ASSERT_TRUE(maybe_and_b32.succeeded());
  EXPECT_EQ(maybe_and_b32.value(), 0x8b0c830au);
  util::FailureOr<uint32_t> maybe_branch_if_scc_clear =
      build_s_cbranch_scc(/*scc1=*/false, 5, kArch);
  ASSERT_TRUE(maybe_branch_if_scc_clear.succeeded());
  EXPECT_EQ(maybe_branch_if_scc_clear.value(), 0xbfa10005u);
  util::FailureOr<uint32_t> maybe_branch_if_scc_set = build_s_cbranch_scc(/*scc1=*/true, -4, kArch);
  ASSERT_TRUE(maybe_branch_if_scc_set.succeeded());
  EXPECT_EQ(maybe_branch_if_scc_set.value(), 0xbfa2fffcu);
}

TEST(ScalarBuilders, UnsupportedArchitecturesReturnFailure) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_INVALID, ROCJITSU_CODE_ARCH_RV32I, ROCJITSU_CODE_ARCH_CDNA1}) {
    SCOPED_TRACE(static_cast<int>(arch));
    EXPECT_TRUE(build_s_bcnt1_i32_b64(8, 6, arch).failed());
    EXPECT_TRUE(build_s_first_set_bit_b64(9, 6, arch).failed());
    EXPECT_TRUE(build_s_and_b32(12, 10, 131, arch).failed());
    EXPECT_TRUE(build_s_cbranch_scc(false, 5, arch).failed());
  }
}

} // namespace
} // namespace test
} // namespace rocjitsu
