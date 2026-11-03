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
    EXPECT_EQ(build_s_bcnt1_i32_b64(8, 6, arch), 0xbe880d06u);
    EXPECT_EQ(build_s_first_set_bit_b64(9, 6, arch), 0xbe891106u);
    EXPECT_EQ(build_s_and_b32(12, 10, scalar_positive_inline_u32(3), arch), 0x860c830au);
    EXPECT_EQ(build_s_cbranch_scc(/*scc1=*/false, 5, arch), 0xbf840005u);
    EXPECT_EQ(build_s_cbranch_scc(/*scc1=*/true, -4, arch), 0xbf85fffcu);
  }
}

TEST(ScalarBuilders, MatchLlvmOnRdna4) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  EXPECT_EQ(build_v_readfirstlane_b32(4, 0, kArch), 0x7e080500u);
  EXPECT_EQ(build_s_bcnt1_i32_b64(8, 6, kArch), 0xbe881906u);
  EXPECT_EQ(build_s_first_set_bit_b64(9, 6, kArch), 0xbe890906u);
  EXPECT_EQ(build_s_and_b32(12, 10, scalar_positive_inline_u32(3), kArch), 0x8b0c830au);
  EXPECT_EQ(build_s_cbranch_scc(/*scc1=*/false, 5, kArch), 0xbfa10005u);
  EXPECT_EQ(build_s_cbranch_scc(/*scc1=*/true, -4, kArch), 0xbfa2fffcu);
}

} // namespace
} // namespace test
} // namespace rocjitsu
