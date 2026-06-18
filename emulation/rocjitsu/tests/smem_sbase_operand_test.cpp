// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file smem_sbase_operand_test.cpp
/// @brief Regression test for the SMEM SBASE operand-model ×2 scale.
///
/// SMEM encodes its SBASE field in units of 2 SGPRs: a raw field value of N
/// names the SGPR pair s[2N : 2N+1].
///
/// This test crafts an SMEM load with raw sbase = 2 and asserts that both
/// `to_register_ref()` and `InstDefUse` (the dataflow bridge that feeds
/// liveness) resolve the base to s[4:5]. It covers both encoding families:
///   - CDNA1-4 share one SMEM encoding (machine_insts_cdna.h), s_load_dwordx2.
///     gfx90a (CDNA2) is the target the bug was originally found on.
///   - RDNA4 has a distinct SMEM encoding (machine_insts.h), s_load_b64.
///
/// CDNA SMEM word[0]: sbase[5:0], sdata[12:6], imm[17], op[25:18],
///   encoding[31:26]=0x30; op=1 is s_load_dwordx2.
/// RDNA4 SMEM word[0]: sbase[5:0], sdata[12:6], op[18:13], encoding[31:26]=0x3D;
///   op=1 is s_load_b64.

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace {

using namespace rocjitsu;

/// @brief CDNA s_load_dwordx2 (raw sbase=2, sdata=s0), immediate-offset form.
constexpr std::array<uint32_t, 2> kCdnaSLoadDwordx2 = {
    /*lo=*/(2u & 0x3Fu) | (0u << 6) | (1u << 17) /*imm*/ | (1u << 18) /*op*/ |
        (0x30u << 26) /*encoding*/,
    /*hi=*/0u};

/// @brief RDNA4 s_load_b64 (raw sbase=2, sdata=s0).
constexpr std::array<uint32_t, 2> kRdna4SLoadB64 = {
    /*lo=*/(2u & 0x3Fu) | (0u << 6) | (1u << 13) /*op*/ | (0x3Du << 26) /*encoding*/,
    /*hi=*/0u};

struct SmemCase {
  rj_code_arch_t arch;
  const char *arch_name;
  std::array<uint32_t, 2> word;
  const char *mnemonic;
};

class SmemSbaseOperandTest : public ::testing::TestWithParam<SmemCase> {};

// SMEM load with raw sbase = 2 -> base register must be s[4:5].
TEST_P(SmemSbaseOperandTest, SbaseResolvesToScaledSgprPair) {
  const SmemCase &tc = GetParam();

  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr) << "Decoder::create() failed for " << tc.arch_name;

  std::unique_ptr<Instruction> inst(decoder->decode(tc.word.data()));
  ASSERT_NE(inst, nullptr) << "decode() returned nullptr for " << tc.arch_name;
  ASSERT_EQ(inst->mnemonic(), tc.mnemonic) << "unexpected mnemonic for " << tc.arch_name;

  // The SBASE is the SGPR-pair source operand (sdata is s0, width 1).
  std::optional<RegisterRef> base;
  for (int i = 0; i < inst->num_src_operands(); ++i) {
    auto ref = inst->src_operand(i)->to_register_ref();
    if (ref && ref->cls == RegClass::SGPR && ref->width == 2)
      base = ref;
  }
  ASSERT_TRUE(base.has_value()) << "no SGPR-pair source operand found for " << tc.arch_name;

  // Raw field 2 names s[4:5]; the unscaled (incorrect) operand model yields
  // s[2:3].
  EXPECT_EQ(base->index, 4u) << "SBASE operand resolved to s[" << base->index << ":"
                             << base->index + 1 << "] (expected s[4:5]) for " << tc.arch_name;

  // The dataflow bridge that feeds liveness must see s[4:5] as a use. This is
  // the exact path whose gap let the probe-call planner clobber the live base.
  InstDefUse du(*inst);
  EXPECT_TRUE(du.uses.contains(RegisterRef{RegClass::SGPR, 4, 2}))
      << "InstDefUse.uses is missing s[4:5] for " << tc.arch_name;
}

INSTANTIATE_TEST_SUITE_P(
    SmemFamilies, SmemSbaseOperandTest,
    ::testing::Values(
        SmemCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1", kCdnaSLoadDwordx2, "s_load_dwordx2"},
        SmemCase{ROCJITSU_CODE_ARCH_CDNA2, "cdna2", kCdnaSLoadDwordx2, "s_load_dwordx2"},
        SmemCase{ROCJITSU_CODE_ARCH_CDNA3, "cdna3", kCdnaSLoadDwordx2, "s_load_dwordx2"},
        SmemCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4", kCdnaSLoadDwordx2, "s_load_dwordx2"},
        SmemCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", kRdna4SLoadB64, "s_load_b64"}),
    [](const ::testing::TestParamInfo<SmemCase> &info) {
      return std::string(info.param.arch_name);
    });

} // namespace
