// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/spill_builders.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

TEST(InstructionBuilder, ScalarFrontendsRejectUnsupportedArchitecturesWithoutThrowing) {
  struct EncodingCase {
    std::string_view name;
    util::FailureOr<uint32_t> (*encode)(rj_code_arch_t, const util::DiagnosticEmitter *);
  };
  const EncodingCase cases[] = {
      {"s_branch",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_branch(0, arch, error);
       }},
      {"s_endpgm",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_endpgm(arch, error);
       }},
      {"s_trap", [](rj_code_arch_t arch,
                    const util::DiagnosticEmitter *error) { return build_s_trap(arch, 0, error); }},
      {"s_nop", [](rj_code_arch_t arch,
                   const util::DiagnosticEmitter *error) { return build_s_nop(0, arch, error); }},
      {"s_getpc",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_getpc_b64(0, arch, error);
       }},
      {"s_setpc",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_setpc_b64(0, arch, error);
       }},
      {"s_swappc",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_swappc_b64(0, 0, arch, error);
       }},
      {"s_call_b64",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_call_b64(0, 0, arch, error);
       }},
      {"s_mov_b32",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_mov_b32(0, 0, arch, error);
       }},
      {"s_mov_b64",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_mov_b64(0, 0, arch, error);
       }},
      {"s_lshl_b32",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_lshl_b32(0, 0, 0, arch, error);
       }},
      {"s_lshr_b32",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_lshr_b32(0, 0, 0, arch, error);
       }},
      {"s_cselect_b32",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_cselect_b32(0, 0, 0, arch, error);
       }},
      {"s_cmp_lg_u32",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_cmp_lg_u32(0, 0, arch, error);
       }},
      {"s_add_u32",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_add_u32(0, 0, 0, arch, error);
       }},
      {"s_addc_u32",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_s_addc_u32(0, 0, 0, arch, error);
       }},
      {"SOP1",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_sop1_encoding(arch, 0, 0, 0, error);
       }},
      {"SOP2",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_sop2_encoding(arch, 0, 0, 0, 0, error);
       }},
      {"SOPK",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_sopk_encoding(arch, 0, 0, 0, error);
       }},
      {"SOPC",
       [](rj_code_arch_t arch, const util::DiagnosticEmitter *error) {
         return build_sopc_encoding(arch, 0, 0, 0, error);
       }},
  };
  for (const EncodingCase &entry : cases) {
    SCOPED_TRACE(entry.name);
    for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_INVALID, ROCJITSU_CODE_ARCH_RV32I,
                                ROCJITSU_CODE_ARCH_RV64I, static_cast<rj_code_arch_t>(15)}) {
      SCOPED_TRACE(static_cast<int>(arch));
      unsigned messages = 0;
      std::string message;
      auto record = [&](std::string_view text) {
        ++messages;
        message = text;
      };
      const util::DiagnosticEmitter error(record);
      util::FailureOr<uint32_t> encoded = util::Result::failure();
      ASSERT_NO_THROW(encoded = entry.encode(arch, &error));
      EXPECT_TRUE(encoded.failed());
      EXPECT_EQ(messages, 1u);
      EXPECT_NE(message.find(entry.name), std::string::npos) << message;
      ASSERT_NO_THROW(encoded = entry.encode(arch, nullptr));
      EXPECT_TRUE(encoded.failed());
      EXPECT_EQ(messages, 1u);
    }
  }
}

TEST(InstructionBuilder, ScalarOperandLookupsRejectUnsupportedArchitecturesWithoutThrowing) {
  struct OperandCase {
    std::string_view name;
    util::FailureOr<uint16_t> (*lookup)(rj_code_arch_t, const util::DiagnosticEmitter *);
  };
  const OperandCase cases[] = {
      {"M0", scalar_operand_m0},
      {"VCC_LO", scalar_operand_vcc_lo},
      {"EXEC_LO", scalar_operand_exec_lo},
      {"inline -1", scalar_inline_neg_one},
  };
  for (const OperandCase &entry : cases) {
    SCOPED_TRACE(entry.name);
    for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_INVALID, ROCJITSU_CODE_ARCH_RV32I,
                                ROCJITSU_CODE_ARCH_RV64I, static_cast<rj_code_arch_t>(15)}) {
      unsigned messages = 0;
      std::string message;
      auto record = [&](std::string_view text) {
        ++messages;
        message = text;
      };
      const util::DiagnosticEmitter error(record);
      util::FailureOr<uint16_t> operand = util::Result::failure();
      ASSERT_NO_THROW(operand = entry.lookup(arch, &error));
      EXPECT_TRUE(operand.failed());
      EXPECT_EQ(messages, 1u);
      EXPECT_NE(message.find(entry.name), std::string::npos) << message;
    }
  }
}

TEST(InstructionBuilder, ScalarLayoutsRejectEveryOversizedFieldBeforeNarrowing) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
        ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA1,
        ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
        ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(static_cast<int>(arch));
    EXPECT_TRUE(build_sop1_encoding(arch, 256, 0, 0).failed());
    EXPECT_TRUE(build_sop1_encoding(arch, 0, 128, 0).failed());
    EXPECT_TRUE(build_sop1_encoding(arch, 0, 0, 256).failed());
    EXPECT_TRUE(build_sop2_encoding(arch, 128, 0, 0, 0).failed());
    EXPECT_TRUE(build_sop2_encoding(arch, 0, 128, 0, 0).failed());
    EXPECT_TRUE(build_sop2_encoding(arch, 0, 0, 256, 0).failed());
    EXPECT_TRUE(build_sop2_encoding(arch, 0, 0, 0, 256).failed());
    EXPECT_TRUE(build_sopk_encoding(arch, 32, 0, 0).failed());
    EXPECT_TRUE(build_sopk_encoding(arch, 0, 128, 0).failed());
    EXPECT_TRUE(build_sopc_encoding(arch, 128, 0, 0).failed());
    EXPECT_TRUE(build_sopc_encoding(arch, 0, 256, 0).failed());
    EXPECT_TRUE(build_sopc_encoding(arch, 0, 0, 256).failed());

    util::FailureOr<uint32_t> sop1 = build_sop1_encoding(arch, 255, 127, 255);
    ASSERT_TRUE(sop1.succeeded());
    EXPECT_EQ(sop1.value(), pack_sop1(255, 127, 255));
    util::FailureOr<uint32_t> sop2 = build_sop2_encoding(arch, 127, 127, 255, 255);
    ASSERT_TRUE(sop2.succeeded());
    EXPECT_EQ(sop2.value(), pack_sop2(127, 127, 255, 255));
    util::FailureOr<uint32_t> sopk = build_sopk_encoding(arch, 31, 127, 65535);
    ASSERT_TRUE(sopk.succeeded());
    EXPECT_EQ(sopk.value(), pack_sopk(31, 127, 65535));
    util::FailureOr<uint32_t> sopc = build_sopc_encoding(arch, 127, 255, 255);
    ASSERT_TRUE(sopc.succeeded());
    EXPECT_EQ(sopc.value(), pack_sopc(127, 255, 255));
  }
}

TEST(InstructionBuilder, ScalarFrontendFieldFailureEmitsOneInstructionSpecificDiagnostic) {
  unsigned messages = 0;
  std::string message;
  auto record = [&](std::string_view text) {
    ++messages;
    message = text;
  };
  const util::DiagnosticEmitter error(record);
  EXPECT_TRUE(build_s_mov_b32(128, 0, ROCJITSU_CODE_ARCH_CDNA4, &error).failed());
  EXPECT_EQ(messages, 1u);
  EXPECT_EQ(message, "s_mov_b32 encoding field out of range");
  messages = 0;
  message.clear();
  EXPECT_TRUE(build_s_cmp_lg_u32(0, 256, ROCJITSU_CODE_ARCH_RDNA4, &error).failed());
  EXPECT_EQ(messages, 1u);
  EXPECT_EQ(message, "s_cmp_lg_u32 encoding field out of range");
  messages = 0;
  message.clear();
  EXPECT_TRUE(build_s_mov_b32(1, 2, ROCJITSU_CODE_ARCH_CDNA4, &error).succeeded());
  EXPECT_EQ(messages, 0u);
  EXPECT_TRUE(message.empty());
}

TEST(InstructionBuilder, UnsupportedPaddingAndIslandPoolLeaveOutputsUnchanged) {
  const std::vector<uint8_t> original{1, 2, 3, 4};
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RV32I, ROCJITSU_CODE_ARCH_INVALID}) {
    std::vector<uint8_t> text = original;
    ASSERT_NO_THROW(EXPECT_TRUE(append_nop_padding(text, 8, arch).failed()));
    EXPECT_EQ(text, original);
    KernelTextLayout layout;
    layout.branch_island_slots = {42};
    ASSERT_NO_THROW(EXPECT_TRUE(append_direct_branch_island_pool(text, layout, arch).failed()));
    EXPECT_EQ(text, original);
    EXPECT_EQ(layout.branch_island_slots, std::vector<uint64_t>{42});
    ASSERT_NO_THROW(EXPECT_FALSE(append_skipped_kernel_stub(text, {}, arch).ok));
    EXPECT_EQ(text, original);
  }
}

TEST(InstructionBuilder, Sop2SetsEncodingPrefix) {
  util::FailureOr<uint32_t> maybe_s_lshl_b32 = build_s_lshl_b32(1, 2, 3, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_s_lshl_b32.succeeded());
  const uint32_t word = maybe_s_lshl_b32.value();
  EXPECT_EQ((word >> 30) & 0x3u, 0x2u);
}

TEST(InstructionBuilder, SoppEncodingReturnsEncodingOrFailure) {
  constexpr auto invalid = build_sopp_encoding(ROCJITSU_CODE_ARCH_INVALID, 0, 0);
  static_assert(invalid.failed());

  for (int value = 0; value < ROCJITSU_CODE_ARCH_NUM_ARCHS; ++value) {
    const auto arch = static_cast<rj_code_arch_t>(value);
    SCOPED_TRACE(value);
    if (arch == ROCJITSU_CODE_ARCH_RV32I || arch == ROCJITSU_CODE_ARCH_RV64I)
      continue;
    util::FailureOr<uint32_t> maybe_sopp_op_nop = sopp_op_nop(arch);
    ASSERT_TRUE(maybe_sopp_op_nop.succeeded());
    EXPECT_TRUE(build_sopp_encoding(arch, maybe_sopp_op_nop.value(), 0).succeeded());
  }

  for (auto arch : {ROCJITSU_CODE_ARCH_INVALID, ROCJITSU_CODE_ARCH_RV32I, ROCJITSU_CODE_ARCH_RV64I,
                    static_cast<rj_code_arch_t>(15)}) {
    SCOPED_TRACE(static_cast<int>(arch));
    EXPECT_TRUE(build_sopp_encoding(arch, 0, 0).failed());
  }
}

TEST(InstructionBuilder, SoppEncodingEmitsOnlyRequestedFailureDiagnostic) {
  unsigned messages = 0;
  std::string message;
  auto callback = [&](std::string_view text) {
    ++messages;
    message = text;
  };
  const util::DiagnosticEmitter error(callback);
  EXPECT_TRUE(build_sopp_encoding(ROCJITSU_CODE_ARCH_CDNA5, 0, 0, &error).succeeded());
  EXPECT_EQ(messages, 0u);
  EXPECT_TRUE(build_sopp_encoding(ROCJITSU_CODE_ARCH_INVALID, 0, 0, &error).failed());
  EXPECT_EQ(messages, 1u);
  EXPECT_EQ(message, "unsupported target architecture for SOPP encoding");
}

TEST(InstructionBuilder, DelayAluReturnsEncodingOrFailure) {
  constexpr auto rdna4 = build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4);
  static_assert(rdna4.succeeded());
  static_assert(rdna4.value() == 0xBF870009u);
  constexpr auto cdna4 = build_s_delay_alu(0, ROCJITSU_CODE_ARCH_CDNA4);
  static_assert(cdna4.failed());

  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4,
                    ROCJITSU_CODE_ARCH_CDNA5}) {
    const auto word = build_s_delay_alu(0xFFFF, arch);
    ASSERT_TRUE(word.succeeded());
    EXPECT_EQ(word.value(), 0xBF87FFFFu);
  }
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                    ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                    ROCJITSU_CODE_ARCH_RV32I, ROCJITSU_CODE_ARCH_RV64I, ROCJITSU_CODE_ARCH_INVALID})
    EXPECT_TRUE(build_s_delay_alu(0, arch).failed());
}

TEST(InstructionBuilder, DelayAluEmitsOnlyRequestedFailureDiagnostic) {
  unsigned messages = 0;
  std::string message;
  auto callback = [&](std::string_view text) {
    ++messages;
    message = text;
  };
  const util::DiagnosticEmitter error(callback);
  EXPECT_TRUE(build_s_delay_alu(0, ROCJITSU_CODE_ARCH_RDNA4, &error).succeeded());
  EXPECT_EQ(messages, 0u);
  EXPECT_TRUE(build_s_delay_alu(0, ROCJITSU_CODE_ARCH_CDNA4, &error).failed());
  EXPECT_EQ(messages, 1u);
  EXPECT_EQ(message, "s_delay_alu for target architecture");
}

// SOPP semantics under test:
//   target = branch_pc + 4 + simm16 * 4
// Inverted:
//   simm16 = (target - (branch_pc + 4)) / 4
//
// The helper must return nullopt when the delta is not dword-aligned, when it
// would not fit in a signed 16-bit dword field, or when branch_pc/target are so
// large that the signed int64 intermediate (branch_pc + 4) would overflow.

TEST(ComputeSoppBranchSimm16, SelfBranchIsMinusOne) {
  auto simm = compute_sopp_branch_simm16(/*branch_pc=*/0x100, /*target=*/0x100);
  ASSERT_TRUE(simm.has_value());
  EXPECT_EQ(*simm, -1);
}

TEST(ComputeSoppBranchSimm16, FallThroughIsZero) {
  auto simm = compute_sopp_branch_simm16(/*branch_pc=*/0x100, /*target=*/0x104);
  ASSERT_TRUE(simm.has_value());
  EXPECT_EQ(*simm, 0);
}

TEST(ComputeSoppBranchSimm16, SmallForwardBranch) {
  auto simm = compute_sopp_branch_simm16(0x1000, 0x1100);
  ASSERT_TRUE(simm.has_value());
  EXPECT_EQ(*simm, 63);
}

TEST(ComputeSoppBranchSimm16, SmallBackwardBranch) {
  auto simm = compute_sopp_branch_simm16(0x1100, 0x1000);
  ASSERT_TRUE(simm.has_value());
  EXPECT_EQ(*simm, -65);
}

TEST(ComputeSoppBranchSimm16, MaxPositiveSimm16) {
  constexpr uint64_t pc = 0x10000;
  constexpr int64_t kMaxDelta = static_cast<int64_t>(std::numeric_limits<int16_t>::max()) * 4;
  auto simm = compute_sopp_branch_simm16(pc, pc + 4 + kMaxDelta);
  ASSERT_TRUE(simm.has_value());
  EXPECT_EQ(*simm, std::numeric_limits<int16_t>::max());
}

TEST(ComputeSoppBranchSimm16, MaxNegativeSimm16) {
  constexpr uint64_t pc = 0x10'0000;
  constexpr int64_t kMinDelta = static_cast<int64_t>(std::numeric_limits<int16_t>::min()) * 4;
  auto simm = compute_sopp_branch_simm16(
      pc, static_cast<uint64_t>(static_cast<int64_t>(pc) + 4 + kMinDelta));
  ASSERT_TRUE(simm.has_value());
  EXPECT_EQ(*simm, std::numeric_limits<int16_t>::min());
}

TEST(ComputeSoppBranchSimm16, PositiveOverflowFails) {
  constexpr uint64_t pc = 0x10000;
  constexpr int64_t kJustOver = (static_cast<int64_t>(std::numeric_limits<int16_t>::max()) + 1) * 4;
  EXPECT_FALSE(compute_sopp_branch_simm16(pc, pc + 4 + kJustOver).has_value());
}

TEST(ComputeSoppBranchSimm16, NegativeOverflowFails) {
  constexpr uint64_t pc = 0x10'0000;
  constexpr int64_t kJustUnder =
      (static_cast<int64_t>(std::numeric_limits<int16_t>::min()) - 1) * 4;
  EXPECT_FALSE(compute_sopp_branch_simm16(
                   pc, static_cast<uint64_t>(static_cast<int64_t>(pc) + 4 + kJustUnder))
                   .has_value());
}

TEST(ComputeSoppBranchSimm16, NonDwordAlignedTargetFails) {
  EXPECT_FALSE(compute_sopp_branch_simm16(0x1000, 0x1002).has_value());
}

TEST(ComputeSoppBranchSimm16, NonDwordAlignedBranchPcFails) {
  EXPECT_FALSE(compute_sopp_branch_simm16(0x1002, 0x1100).has_value());
}

// branch_pc and target are misaligned by the same amount, so the delta is
// dword-aligned (0 and 4 here). A delta-only check would accept these; the
// branch_pc/target alignment checks must still reject them.
TEST(ComputeSoppBranchSimm16, EquallyMisalignedPcsFailEvenWhenDeltaAligned) {
  EXPECT_FALSE(compute_sopp_branch_simm16(0x1002, 0x1006).has_value()); // delta 0
  EXPECT_FALSE(compute_sopp_branch_simm16(0x1002, 0x100a).has_value()); // delta 4
}

// C++20 specifies truncated-toward-zero integer division/modulo, so
// `(-258) % 4 == -2 != 0`. This pins that semantic: a negative delta that
// is not a multiple of 4 must be rejected (not silently rounded).
TEST(ComputeSoppBranchSimm16, NegativeUnalignedDeltaFails) {
  // branch_pc = 0x1100, target = 0x1002 →
  //   delta = 0x1002 - 0x1100 - 4 = -0x102 = -258 bytes (not /4).
  EXPECT_FALSE(compute_sopp_branch_simm16(0x1100, 0x1002).has_value());
}

TEST(ComputeSoppBranchSimm16, BranchPcNearInt64MaxFails) {
  constexpr uint64_t kHugePc = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  EXPECT_FALSE(compute_sopp_branch_simm16(kHugePc, kHugePc).has_value());
}

TEST(ComputeSoppBranchSimm16, TargetNearUint64MaxFails) {
  constexpr uint64_t kHugeTarget = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1;
  EXPECT_FALSE(compute_sopp_branch_simm16(0x1000, kHugeTarget).has_value());
}

TEST(InstructionBuilder, BuildSEndpgm) {
  // calculate with SOPP prefix (0x17F) << 23 | opcode 0x1 << 16
  constexpr uint32_t SOPP_S_ENDPGM_CDNA4 = 0xBF810000u;
  util::FailureOr<uint32_t> maybe_s_endpgm = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(maybe_s_endpgm.succeeded());
  EXPECT_EQ(maybe_s_endpgm.value(), SOPP_S_ENDPGM_CDNA4);
  // calculate with SOPP prefix (0x17F) << 23 | opcode 0x30 << 16
  constexpr uint32_t SOPP_S_ENDPGM_RDNA4 = 0xBFB00000u;
  util::FailureOr<uint32_t> maybe_s_endpgm_2 = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_s_endpgm_2.succeeded());
  EXPECT_EQ(maybe_s_endpgm_2.value(), SOPP_S_ENDPGM_RDNA4);
}

TEST(InstructionBuilder, BuildSMovB32UsesRdna1AndRdna2Opcodes) {
  constexpr uint16_t kDst = 4;
  constexpr uint16_t kSrc = 8;

  util::FailureOr<uint32_t> maybe_s_mov_b32 = build_s_mov_b32(kDst, kSrc, ROCJITSU_CODE_ARCH_RDNA1);
  ASSERT_TRUE(maybe_s_mov_b32.succeeded());
  const uint32_t rdna1_word = maybe_s_mov_b32.value();
  util::FailureOr<uint32_t> maybe_s_mov_b32_2 =
      build_s_mov_b32(kDst, kSrc, ROCJITSU_CODE_ARCH_RDNA2);
  ASSERT_TRUE(maybe_s_mov_b32_2.succeeded());
  const uint32_t rdna2_word = maybe_s_mov_b32_2.value();

  // RDNA1/2 assign s_mov_b32 opcode 3 rather than the opcode 0 used by CDNA
  // and newer RDNA targets. This is an intentional correctness fix over the
  // old architecture-agnostic builder and must not be treated as NFC.
  EXPECT_EQ((rdna1_word >> 8) & 0xFFu, rdna1::kSMovB32Sop1);
  EXPECT_EQ((rdna2_word >> 8) & 0xFFu, rdna2::kSMovB32Sop1);
  EXPECT_EQ(rdna1::kSMovB32Sop1, 3u);
  EXPECT_EQ(rdna2::kSMovB32Sop1, 3u);
}

TEST(GeneratedInstructionBuilder, PacksCdna3FormatsFromXmlLayouts) {
  // Pin both a one-word scalar format and representative two-word VALU/LDS
  // formats. These exact words were previously produced by local MachineInst
  // bitfield initialization in the CDNA4-to-CDNA3 semantic rules.
  constexpr auto sopp = cdna3::build_sopp(/*op=*/12, {.simm16 = 0xC07F});
  EXPECT_EQ(sopp, (std::array<uint32_t, 1>{0xBF8CC07Fu}));

  constexpr auto vop3 = cdna3::build_vop3(
      /*op=*/321, {.vdst = 7, .src0 = 256 + 8, .src1 = 256 + 9, .src2 = 128});
  EXPECT_EQ(vop3, (std::array<uint32_t, 2>{0xD1410007u, 0x02021308u}));

  constexpr auto ds = cdna3::build_ds(
      /*op=*/54, {.offset0 = 3, .offset1 = 5, .addr = 10, .data0 = 11, .data1 = 12, .vdst = 13});
  EXPECT_EQ(ds, (std::array<uint32_t, 2>{0xD86C0503u, 0x0D0C0B0Au}));
}

TEST(GeneratedInstructionBuilder, Gfx1250ScalarPathsUseGeneratedLayouts) {
  constexpr uint16_t kSimm16 = 0xC07F;
  constexpr uint16_t kSdst = 7;
  constexpr uint16_t kSsrc0 = 8;
  constexpr uint16_t kSsrc1 = 9;

  constexpr auto sopp = cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = kSimm16});
  constexpr auto encoded_sopp =
      build_sopp_encoding(ROCJITSU_CODE_ARCH_CDNA5, cdna5::kSBranchSopp, kSimm16);
  static_assert(encoded_sopp.succeeded());
  EXPECT_EQ(encoded_sopp.value(), sopp[0]);

  constexpr auto sop1 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = kSsrc0, .sdst = kSdst});
  util::FailureOr<uint32_t> maybe_sop1_encoding =
      build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA5, cdna5::kSMovB32Sop1, kSdst, kSsrc0);
  ASSERT_TRUE(maybe_sop1_encoding.succeeded());
  EXPECT_EQ(maybe_sop1_encoding.value(), sop1[0]);

  constexpr auto sop2 =
      cdna5::build_sop2(cdna5::kSLshlB32Sop2, {.ssrc0 = kSsrc0, .ssrc1 = kSsrc1, .sdst = kSdst});
  util::FailureOr<uint32_t> maybe_sop2_encoding =
      build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, cdna5::kSLshlB32Sop2, kSdst, kSsrc0, kSsrc1);
  ASSERT_TRUE(maybe_sop2_encoding.succeeded());
  EXPECT_EQ(maybe_sop2_encoding.value(), sop2[0]);
}

TEST(InstructionBuilder, BuildSGetpcB64) {
  // SOP1; opcode is arch-specific: 0x1C (GFX9), 0x1F (GFX10), 0x47 (GFX11+).
  util::FailureOr<uint32_t> maybe_s_getpc_b64 =
      build_s_getpc_b64(/*sdst_pair_base=*/0, ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_TRUE(maybe_s_getpc_b64.succeeded());
  EXPECT_EQ(maybe_s_getpc_b64.value(), 0xBE801C00u);
  util::FailureOr<uint32_t> maybe_s_getpc_b64_2 = build_s_getpc_b64(0, ROCJITSU_CODE_ARCH_RDNA2);
  ASSERT_TRUE(maybe_s_getpc_b64_2.succeeded());
  EXPECT_EQ(maybe_s_getpc_b64_2.value(), 0xBE801F00u);
  util::FailureOr<uint32_t> maybe_s_getpc_b64_3 = build_s_getpc_b64(0, ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_TRUE(maybe_s_getpc_b64_3.succeeded());
  EXPECT_EQ(maybe_s_getpc_b64_3.value(), 0xBE804700u);
  util::FailureOr<uint32_t> maybe_s_getpc_b64_4 = build_s_getpc_b64(0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_s_getpc_b64_4.succeeded());
  EXPECT_EQ(maybe_s_getpc_b64_4.value(), 0xBE804700u);
  // gfx1250 renames the family (s_get_pc_i64) but keeps opcode 0x47.
  util::FailureOr<uint32_t> maybe_s_getpc_b64_5 = build_s_getpc_b64(0, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(maybe_s_getpc_b64_5.succeeded());
  EXPECT_EQ(maybe_s_getpc_b64_5.value(), 0xBE804700u);
}

TEST(InstructionBuilder, BuildSAddU32) {
  // SOP2 opcode 0 (all gens); ssrc1 = inline literal 16.
  util::FailureOr<uint32_t> maybe_s_add_u32 = build_s_add_u32(
      /*sdst=*/0, /*ssrc0=*/0, scalar_positive_inline_u32(16), ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_TRUE(maybe_s_add_u32.succeeded());
  EXPECT_EQ(maybe_s_add_u32.value(), 0x80009000u);
}

TEST(InstructionBuilder, BuildSAddcU32) {
  // SOP2 opcode 4 (all gens); ssrc1 = inline literal 0.
  util::FailureOr<uint32_t> maybe_s_addc_u32 = build_s_addc_u32(
      /*sdst=*/1, /*ssrc0=*/1, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_TRUE(maybe_s_addc_u32.succeeded());
  EXPECT_EQ(maybe_s_addc_u32.value(), 0x82018001u);
}

TEST(InstructionBuilder, BuildSSwappcB64) {
  // SOP1; opcode is arch-specific: 0x1E (GFX9), 0x21 (GFX10), 0x49 (GFX11+).
  // link pair s[30:31], target pair s[0:1].
  util::FailureOr<uint32_t> maybe_s_swappc_b64 =
      build_s_swappc_b64(/*sdst_link_base=*/30, /*ssrc0_target_base=*/0, ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_TRUE(maybe_s_swappc_b64.succeeded());
  EXPECT_EQ(maybe_s_swappc_b64.value(), 0xBE9E1E00u);
  util::FailureOr<uint32_t> maybe_s_swappc_b64_2 =
      build_s_swappc_b64(30, 0, ROCJITSU_CODE_ARCH_RDNA2);
  ASSERT_TRUE(maybe_s_swappc_b64_2.succeeded());
  EXPECT_EQ(maybe_s_swappc_b64_2.value(), 0xBE9E2100u);
  util::FailureOr<uint32_t> maybe_s_swappc_b64_3 =
      build_s_swappc_b64(30, 0, ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_TRUE(maybe_s_swappc_b64_3.succeeded());
  EXPECT_EQ(maybe_s_swappc_b64_3.value(), 0xBE9E4900u);
  util::FailureOr<uint32_t> maybe_s_swappc_b64_4 =
      build_s_swappc_b64(30, 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_s_swappc_b64_4.succeeded());
  EXPECT_EQ(maybe_s_swappc_b64_4.value(), 0xBE9E4900u);
  // gfx1250: s_swap_pc_i64, opcode 0x49
  util::FailureOr<uint32_t> maybe_s_swappc_b64_5 =
      build_s_swappc_b64(30, 0, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(maybe_s_swappc_b64_5.succeeded());
  EXPECT_EQ(maybe_s_swappc_b64_5.value(), 0xBE9E4900u);
}

TEST(InstructionBuilder, BuildSCallB64UsesGfx1250SCallI64Opcode) {
  constexpr uint16_t kReturnSreg = 30;
  constexpr int16_t kOffsetDwords = -2;
  constexpr auto expected =
      cdna5::build_sopk(cdna5::kSCallI64Sopk, {.simm16 = static_cast<uint16_t>(kOffsetDwords),
                                               .sdst = static_cast<uint8_t>(kReturnSreg)});

  util::FailureOr<uint32_t> maybe_s_call_b64 =
      build_s_call_b64(kReturnSreg, kOffsetDwords, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(maybe_s_call_b64.succeeded());
  EXPECT_EQ(maybe_s_call_b64.value(), expected[0]);
  EXPECT_EQ(expected[0], 0xBA1EFFFEu);
}

TEST(InstructionBuilder, BuildSCselectB32) {
  // SOP2; opcode 0x0A on GFX9/GFX10, 0x30 on GFX11+. ssrc0 = inline 1, ssrc1 = inline 0.
  util::FailureOr<uint32_t> maybe_s_cselect_b32 =
      build_s_cselect_b32(/*sdst=*/2, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
                          ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_TRUE(maybe_s_cselect_b32.succeeded());
  EXPECT_EQ(maybe_s_cselect_b32.value(), 0x85028081u);
  util::FailureOr<uint32_t> maybe_s_cselect_b32_2 = build_s_cselect_b32(
      2, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_TRUE(maybe_s_cselect_b32_2.succeeded());
  EXPECT_EQ(maybe_s_cselect_b32_2.value(), 0x98028081u);
  util::FailureOr<uint32_t> maybe_s_cselect_b32_3 = build_s_cselect_b32(
      2, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_s_cselect_b32_3.succeeded());
  EXPECT_EQ(maybe_s_cselect_b32_3.value(), 0x98028081u);
  // gfx1250 uses the GFX11+ opcode 0x30.
  util::FailureOr<uint32_t> maybe_s_cselect_b32_4 = build_s_cselect_b32(
      2, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(maybe_s_cselect_b32_4.succeeded());
  EXPECT_EQ(maybe_s_cselect_b32_4.value(), 0x98028081u);
}

TEST(InstructionBuilder, BuildSCmpLgU32) {
  // SOPC prefix (0x17E) << 23 | opcode 7 << 16; ssrc1 = inline literal 0.
  util::FailureOr<uint32_t> maybe_s_cmp_lg_u32 =
      build_s_cmp_lg_u32(/*ssrc0=*/2, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_TRUE(maybe_s_cmp_lg_u32.succeeded());
  EXPECT_EQ(maybe_s_cmp_lg_u32.value(), 0xBF078002u);
  // Opcode 7 is invariant across gens, including gfx1250.
  util::FailureOr<uint32_t> maybe_s_cmp_lg_u32_2 =
      build_s_cmp_lg_u32(2, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_s_cmp_lg_u32_2.succeeded());
  EXPECT_EQ(maybe_s_cmp_lg_u32_2.value(), 0xBF078002u);
  util::FailureOr<uint32_t> maybe_s_cmp_lg_u32_3 =
      build_s_cmp_lg_u32(2, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(maybe_s_cmp_lg_u32_3.succeeded());
  EXPECT_EQ(maybe_s_cmp_lg_u32_3.value(), 0xBF078002u);
}

TEST(InstructionBuilder, BuildVWritelaneB32) {
  // VOP3: base v_writelane_b32 is 0xD28A0000 (cdna4) / 0xD7610000 (rdna4); vdst
  // in word0 bits[7:0]; word1 = src0 | src1<<9 with lane 0 as inline const 128.
  // CDNA3 and CDNA4 share the GFX9 VOP3 encoding.
  EXPECT_EQ(
      build_v_writelane_b32(/*vgpr_dst=*/2, /*sgpr_src=*/5, /*lane=*/0, ROCJITSU_CODE_ARCH_CDNA3),
      (std::array<uint32_t, 2>{0xD28A0002u, 0x00010005u}));
  EXPECT_EQ(build_v_writelane_b32(2, 5, 0, ROCJITSU_CODE_ARCH_CDNA4),
            (std::array<uint32_t, 2>{0xD28A0002u, 0x00010005u}));
  EXPECT_EQ(build_v_writelane_b32(2, 5, 0, ROCJITSU_CODE_ARCH_RDNA4),
            (std::array<uint32_t, 2>{0xD7610002u, 0x00010005u}));
}

TEST(InstructionBuilder, BuildVReadlaneB32) {
  // VOP3: base v_readlane_b32 is 0xD2890000 (cdna4) / 0xD7600000 (rdna4); sdst in
  // word0 bits[7:0]; src0 is a VGPR (256 + index), so v3 -> 0x103.
  // CDNA3 and CDNA4 share the GFX9 VOP3 encoding.
  EXPECT_EQ(
      build_v_readlane_b32(/*sgpr_dst=*/6, /*vgpr_src=*/3, /*lane=*/0, ROCJITSU_CODE_ARCH_CDNA3),
      (std::array<uint32_t, 2>{0xD2890006u, 0x00010103u}));
  EXPECT_EQ(build_v_readlane_b32(6, 3, 0, ROCJITSU_CODE_ARCH_CDNA4),
            (std::array<uint32_t, 2>{0xD2890006u, 0x00010103u}));
  EXPECT_EQ(build_v_readlane_b32(6, 3, 0, ROCJITSU_CODE_ARCH_RDNA4),
            (std::array<uint32_t, 2>{0xD7600006u, 0x00010103u}));
}

TEST(InstructionBuilder, BuildScratchStoreDword) {
  // CDNA3 and CDNA4 (GFX9): FLAT flat_store_dword base 0xDC700000 | seg=1<<14;
  // word1 = data<<8 | saddr(0x7F)<<16. 2 words.
  const auto cdna3 =
      build_scratch_store_dword(/*vdata=*/3, /*byte_offset=*/64, ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_EQ(cdna3.size(), 2u);
  EXPECT_EQ(cdna3[0], 0xDC704040u);
  EXPECT_EQ(cdna3[1], 0x007F0300u);

  const auto cdna4 = build_scratch_store_dword(3, 64, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_EQ(cdna4.size(), 2u);
  EXPECT_EQ(cdna4[0], 0xDC704040u);
  EXPECT_EQ(cdna4[1], 0x007F0300u);

  // RDNA4: VSCRATCH scratch_store_b32 base 0xED068000 | saddr(0x7C); word1 =
  // vsrc<<23; word2 = ioffset<<8. 3 words.
  const auto rdna4 = build_scratch_store_dword(3, 64, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_EQ(rdna4.size(), 3u);
  EXPECT_EQ(rdna4[0], 0xED06807Cu);
  EXPECT_EQ(rdna4[1], 0x01800000u);
  EXPECT_EQ(rdna4[2], 0x00004000u);
}

TEST(InstructionBuilder, BuildScratchLoadDword) {
  // CDNA3 and CDNA4 (GFX9): FLAT flat_load_dword base 0xDC500000 | seg=1<<14;
  // word1 = saddr<<16 | vdst<<24. 2 words.
  const auto cdna3 =
      build_scratch_load_dword(/*vdst=*/5, /*byte_offset=*/64, ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_EQ(cdna3.size(), 2u);
  EXPECT_EQ(cdna3[0], 0xDC504040u);
  EXPECT_EQ(cdna3[1], 0x057F0000u);

  const auto cdna4 = build_scratch_load_dword(5, 64, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_EQ(cdna4.size(), 2u);
  EXPECT_EQ(cdna4[0], 0xDC504040u);
  EXPECT_EQ(cdna4[1], 0x057F0000u);

  // RDNA4: VSCRATCH scratch_load_b32 base 0xED050000 | saddr(0x7C); word1 = vdst;
  // word2 = ioffset<<8. 3 words.
  const auto rdna4 = build_scratch_load_dword(5, 64, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_EQ(rdna4.size(), 3u);
  EXPECT_EQ(rdna4[0], 0xED05007Cu);
  EXPECT_EQ(rdna4[1], 0x00000005u);
  EXPECT_EQ(rdna4[2], 0x00004000u);
}

TEST(InstructionBuilder, BuildScratchStoreDwordAccVgpr) {
  // acc=1 sets the CDNA FLAT acc bit (word1 bit 23 = 0x00800000), addressing the
  // AccVGPR file for the data operand. Same as BuildScratchStoreDword otherwise.
  const auto cdna3 = build_scratch_store_dword(/*vdata=*/3, /*byte_offset=*/64,
                                               ROCJITSU_CODE_ARCH_CDNA3, /*acc=*/true);
  ASSERT_EQ(cdna3.size(), 2u);
  EXPECT_EQ(cdna3[0], 0xDC704040u);
  EXPECT_EQ(cdna3[1], 0x00FF0300u);

  const auto cdna4 = build_scratch_store_dword(3, 64, ROCJITSU_CODE_ARCH_CDNA4, /*acc=*/true);
  ASSERT_EQ(cdna4.size(), 2u);
  EXPECT_EQ(cdna4[0], 0xDC704040u);
  EXPECT_EQ(cdna4[1], 0x00FF0300u);

  // RDNA has no AccVGPRs: requesting acc is rejected.
  EXPECT_THROW((void)build_scratch_store_dword(3, 64, ROCJITSU_CODE_ARCH_RDNA4, /*acc=*/true),
               util::UnimplementedInst);
}

TEST(InstructionBuilder, BuildScratchLoadDwordAccVgpr) {
  // acc=1 sets the CDNA FLAT acc bit (word1 bit 23 = 0x00800000), writing the load
  // result into the AccVGPR file. Same as BuildScratchLoadDword otherwise.
  const auto cdna3 = build_scratch_load_dword(/*vdst=*/5, /*byte_offset=*/64,
                                              ROCJITSU_CODE_ARCH_CDNA3, /*acc=*/true);
  ASSERT_EQ(cdna3.size(), 2u);
  EXPECT_EQ(cdna3[0], 0xDC504040u);
  EXPECT_EQ(cdna3[1], 0x05FF0000u);

  const auto cdna4 = build_scratch_load_dword(5, 64, ROCJITSU_CODE_ARCH_CDNA4, /*acc=*/true);
  ASSERT_EQ(cdna4.size(), 2u);
  EXPECT_EQ(cdna4[0], 0xDC504040u);
  EXPECT_EQ(cdna4[1], 0x05FF0000u);

  EXPECT_THROW((void)build_scratch_load_dword(5, 64, ROCJITSU_CODE_ARCH_RDNA4, /*acc=*/true),
               util::UnimplementedInst);
}

TEST(InstructionBuilder, BuildWaitLoadsComplete) {
  // CDNA3 and CDNA4: s_waitcnt 0 (all counters).
  EXPECT_EQ(build_wait_loads_complete(ROCJITSU_CODE_ARCH_CDNA3), 0xBF8C0000u);
  EXPECT_EQ(build_wait_loads_complete(ROCJITSU_CODE_ARCH_CDNA4), 0xBF8C0000u);
  // RDNA4: s_wait_loadcnt 0 (split counter).
  EXPECT_EQ(build_wait_loads_complete(ROCJITSU_CODE_ARCH_RDNA4), 0xBFC00000u);
}

TEST(InstructionBuilder, BuildWaitStoresComplete) {
  // CDNA3 and CDNA4: s_waitcnt 0 (unified vmcnt covers stores).
  EXPECT_EQ(build_wait_stores_complete(ROCJITSU_CODE_ARCH_CDNA3), 0xBF8C0000u);
  EXPECT_EQ(build_wait_stores_complete(ROCJITSU_CODE_ARCH_CDNA4), 0xBF8C0000u);
  // RDNA4: s_wait_storecnt 0 (split counter, distinct from s_wait_loadcnt).
  EXPECT_EQ(build_wait_stores_complete(ROCJITSU_CODE_ARCH_RDNA4), 0xBFC10000u);
}

TEST(InstructionBuilder, BuildWaitAllLoadsComplete) {
  // CDNA3 and CDNA4: one s_waitcnt 0 drains vmcnt+lgkmcnt (VMEM, LDS, scalar).
  EXPECT_EQ(build_wait_all_loads_complete(ROCJITSU_CODE_ARCH_CDNA3),
            (std::vector<uint32_t>{0xBF8C0000u}));
  EXPECT_EQ(build_wait_all_loads_complete(ROCJITSU_CODE_ARCH_CDNA4),
            (std::vector<uint32_t>{0xBF8C0000u}));
  // RDNA4 splits the counters and every VGPR-targeting load counter must drain:
  // s_wait_loadcnt_dscnt 0 (VMEM + LDS -> VGPRs), s_wait_samplecnt 0 (image
  // sample/gather -> VGPRs), s_wait_bvhcnt 0 (BVH -> VGPRs), then s_wait_kmcnt 0
  // (scalar -> SGPRs).
  EXPECT_EQ(build_wait_all_loads_complete(ROCJITSU_CODE_ARCH_RDNA4),
            (std::vector<uint32_t>{0xBFC80000u, 0xBFC20000u, 0xBFC30000u, 0xBFC70000u}));
}

// VCC_LO/EXEC_LO scalar-operand codes are resolved per-arch: each case returns
// its generation's generated table entry. This guards that every generation's
// table still agrees (and that the well-known 106/126 values have not moved).
TEST(InstructionBuilder, ScalarOperandCodesMatchGeneratedTables) {
  util::FailureOr<uint16_t> maybe_scalar_operand_vcc_lo =
      scalar_operand_vcc_lo(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(maybe_scalar_operand_vcc_lo.succeeded());
  EXPECT_EQ(maybe_scalar_operand_vcc_lo.value(), 106);
  util::FailureOr<uint16_t> maybe_scalar_operand_exec_lo =
      scalar_operand_exec_lo(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(maybe_scalar_operand_exec_lo.succeeded());
  EXPECT_EQ(maybe_scalar_operand_exec_lo.value(), 126);

  util::FailureOr<uint16_t> maybe_scalar_operand_vcc_lo_2 =
      scalar_operand_vcc_lo(ROCJITSU_CODE_ARCH_CDNA1);
  ASSERT_TRUE(maybe_scalar_operand_vcc_lo_2.succeeded());
  EXPECT_EQ(maybe_scalar_operand_vcc_lo_2.value(), cdna1::OPR_SDST_VCC_LO);
  util::FailureOr<uint16_t> maybe_scalar_operand_vcc_lo_3 =
      scalar_operand_vcc_lo(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(maybe_scalar_operand_vcc_lo_3.succeeded());
  EXPECT_EQ(maybe_scalar_operand_vcc_lo_3.value(), cdna4::OPR_SDST_VCC_LO);
  util::FailureOr<uint16_t> maybe_scalar_operand_vcc_lo_4 =
      scalar_operand_vcc_lo(ROCJITSU_CODE_ARCH_RDNA2);
  ASSERT_TRUE(maybe_scalar_operand_vcc_lo_4.succeeded());
  EXPECT_EQ(maybe_scalar_operand_vcc_lo_4.value(), rdna2::OPR_SDST_VCC_LO);
  util::FailureOr<uint16_t> maybe_scalar_operand_vcc_lo_5 =
      scalar_operand_vcc_lo(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_scalar_operand_vcc_lo_5.succeeded());
  EXPECT_EQ(maybe_scalar_operand_vcc_lo_5.value(), rdna4::OPR_SDST_VCC_LO);
  util::FailureOr<uint16_t> maybe_scalar_operand_vcc_lo_6 =
      scalar_operand_vcc_lo(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(maybe_scalar_operand_vcc_lo_6.succeeded());
  EXPECT_EQ(maybe_scalar_operand_vcc_lo_6.value(), cdna5::OPR_SDST_VCC_LO);

  util::FailureOr<uint16_t> maybe_scalar_operand_exec_lo_2 =
      scalar_operand_exec_lo(ROCJITSU_CODE_ARCH_CDNA1);
  ASSERT_TRUE(maybe_scalar_operand_exec_lo_2.succeeded());
  EXPECT_EQ(maybe_scalar_operand_exec_lo_2.value(), cdna1::OPR_SDST_EXEC_LO);
  util::FailureOr<uint16_t> maybe_scalar_operand_exec_lo_3 =
      scalar_operand_exec_lo(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(maybe_scalar_operand_exec_lo_3.succeeded());
  EXPECT_EQ(maybe_scalar_operand_exec_lo_3.value(), cdna4::OPR_SDST_EXEC_LO);
  util::FailureOr<uint16_t> maybe_scalar_operand_exec_lo_4 =
      scalar_operand_exec_lo(ROCJITSU_CODE_ARCH_RDNA2);
  ASSERT_TRUE(maybe_scalar_operand_exec_lo_4.succeeded());
  EXPECT_EQ(maybe_scalar_operand_exec_lo_4.value(), rdna2::OPR_SDST_EXEC_LO);
  util::FailureOr<uint16_t> maybe_scalar_operand_exec_lo_5 =
      scalar_operand_exec_lo(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_scalar_operand_exec_lo_5.succeeded());
  EXPECT_EQ(maybe_scalar_operand_exec_lo_5.value(), rdna4::OPR_SDST_EXEC_LO);
  util::FailureOr<uint16_t> maybe_scalar_operand_exec_lo_6 =
      scalar_operand_exec_lo(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(maybe_scalar_operand_exec_lo_6.succeeded());
  EXPECT_EQ(maybe_scalar_operand_exec_lo_6.value(), cdna5::OPR_SDST_EXEC_LO);

  // Inline-constant source for -1 (193), also generation-stable.
  util::FailureOr<uint16_t> maybe_scalar_inline_neg_one =
      scalar_inline_neg_one(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(maybe_scalar_inline_neg_one.succeeded());
  EXPECT_EQ(maybe_scalar_inline_neg_one.value(), 193);
  util::FailureOr<uint16_t> maybe_scalar_inline_neg_one_2 =
      scalar_inline_neg_one(ROCJITSU_CODE_ARCH_CDNA1);
  ASSERT_TRUE(maybe_scalar_inline_neg_one_2.succeeded());
  EXPECT_EQ(maybe_scalar_inline_neg_one_2.value(), cdna1::OPR_SRC_NEG_INT_MIN);
  util::FailureOr<uint16_t> maybe_scalar_inline_neg_one_3 =
      scalar_inline_neg_one(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(maybe_scalar_inline_neg_one_3.succeeded());
  EXPECT_EQ(maybe_scalar_inline_neg_one_3.value(), cdna4::OPR_SRC_NEG_INT_MIN);
  util::FailureOr<uint16_t> maybe_scalar_inline_neg_one_4 =
      scalar_inline_neg_one(ROCJITSU_CODE_ARCH_RDNA2);
  ASSERT_TRUE(maybe_scalar_inline_neg_one_4.succeeded());
  EXPECT_EQ(maybe_scalar_inline_neg_one_4.value(), rdna2::OPR_SRC_NEG_INT_MIN);
  util::FailureOr<uint16_t> maybe_scalar_inline_neg_one_5 =
      scalar_inline_neg_one(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_scalar_inline_neg_one_5.succeeded());
  EXPECT_EQ(maybe_scalar_inline_neg_one_5.value(), rdna4::OPR_SRC_NEG_INT_MIN);
  util::FailureOr<uint16_t> maybe_scalar_inline_neg_one_6 =
      scalar_inline_neg_one(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(maybe_scalar_inline_neg_one_6.succeeded());
  EXPECT_EQ(maybe_scalar_inline_neg_one_6.value(), cdna5::OPR_SRC_NEG_INT_MIN);

  // Base for non-negative inline integers (128 = 0), also generation-stable.
  EXPECT_EQ(kScalarPositiveInlineBase, 128);
  EXPECT_EQ(kScalarPositiveInlineBase, cdna1::OPR_SRC_POS_INT_MIN);
  EXPECT_EQ(kScalarPositiveInlineBase, cdna4::OPR_SRC_POS_INT_MIN);
  EXPECT_EQ(kScalarPositiveInlineBase, rdna2::OPR_SRC_POS_INT_MIN);
  EXPECT_EQ(kScalarPositiveInlineBase, rdna4::OPR_SRC_POS_INT_MIN);
  EXPECT_EQ(kScalarPositiveInlineBase, cdna5::OPR_SRC_POS_INT_MIN);
}

// M0 moved from 124 (gfx9/gfx10.x) to 125 (gfx11+); each arch resolves to its
// own generated OPR_SDST_M0.
TEST(InstructionBuilder, ScalarOperandM0IsPerArch) {
  util::FailureOr<uint16_t> maybe_scalar_operand_m0 = scalar_operand_m0(ROCJITSU_CODE_ARCH_CDNA1);
  ASSERT_TRUE(maybe_scalar_operand_m0.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0.value(), cdna1::OPR_SDST_M0);
  util::FailureOr<uint16_t> maybe_scalar_operand_m0_2 = scalar_operand_m0(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_TRUE(maybe_scalar_operand_m0_2.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0_2.value(), cdna2::OPR_SDST_M0);
  util::FailureOr<uint16_t> maybe_scalar_operand_m0_3 = scalar_operand_m0(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_TRUE(maybe_scalar_operand_m0_3.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0_3.value(), cdna3::OPR_SDST_M0);
  util::FailureOr<uint16_t> maybe_scalar_operand_m0_4 = scalar_operand_m0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(maybe_scalar_operand_m0_4.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0_4.value(), cdna4::OPR_SDST_M0);
  util::FailureOr<uint16_t> maybe_scalar_operand_m0_5 = scalar_operand_m0(ROCJITSU_CODE_ARCH_RDNA1);
  ASSERT_TRUE(maybe_scalar_operand_m0_5.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0_5.value(), rdna1::OPR_SDST_M0);
  util::FailureOr<uint16_t> maybe_scalar_operand_m0_6 = scalar_operand_m0(ROCJITSU_CODE_ARCH_RDNA2);
  ASSERT_TRUE(maybe_scalar_operand_m0_6.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0_6.value(), rdna2::OPR_SDST_M0);
  util::FailureOr<uint16_t> maybe_scalar_operand_m0_7 = scalar_operand_m0(ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_TRUE(maybe_scalar_operand_m0_7.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0_7.value(), rdna3::OPR_SDST_M0);
  util::FailureOr<uint16_t> maybe_scalar_operand_m0_8 =
      scalar_operand_m0(ROCJITSU_CODE_ARCH_RDNA3_5);
  ASSERT_TRUE(maybe_scalar_operand_m0_8.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0_8.value(), rdna3_5::OPR_SDST_M0);
  util::FailureOr<uint16_t> maybe_scalar_operand_m0_9 = scalar_operand_m0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_scalar_operand_m0_9.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0_9.value(), rdna4::OPR_SDST_M0);
  util::FailureOr<uint16_t> maybe_scalar_operand_m0_10 =
      scalar_operand_m0(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(maybe_scalar_operand_m0_10.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0_10.value(), cdna5::OPR_SDST_M0);

  // gfx9 / gfx10.x = 124; gfx11+ = 125.
  util::FailureOr<uint16_t> maybe_scalar_operand_m0_11 =
      scalar_operand_m0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(maybe_scalar_operand_m0_11.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0_11.value(), 124);
  util::FailureOr<uint16_t> maybe_scalar_operand_m0_12 =
      scalar_operand_m0(ROCJITSU_CODE_ARCH_RDNA2);
  ASSERT_TRUE(maybe_scalar_operand_m0_12.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0_12.value(), 124);
  util::FailureOr<uint16_t> maybe_scalar_operand_m0_13 =
      scalar_operand_m0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_scalar_operand_m0_13.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0_13.value(), 125);
  util::FailureOr<uint16_t> maybe_scalar_operand_m0_14 =
      scalar_operand_m0(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(maybe_scalar_operand_m0_14.succeeded());
  EXPECT_EQ(maybe_scalar_operand_m0_14.value(), 125);
}

} // namespace
} // namespace rocjitsu
