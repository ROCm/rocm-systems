// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file decode_exec_modify_test.cpp
/// @brief Verifies the EXEC_MODIFY instruction flag for AMD GPU SOP1 / VOPC /
/// SMEM / VOP3 encodings. The same case table is run against every supported
/// GFX9-family ISA (CDNA3, CDNA4); the encodings tested are binary-compatible
/// across these archs, so identical words must produce identical flag values.
/// EXEC_MODIFY must be set when an instruction modifies the EXEC mask via
/// either path:
///   (a) Opcode-implied: the MR ISA spec attaches an implicit OPR_SDST_EXEC
///       output (s_*_saveexec_*, s_*_wrexec_*, v_cmpx_*).
///   (b) Operand-encoded: the destination operand's encoding value names
///       EXEC_LO (126) or EXEC_HI (127), e.g. `s_mov_b64 exec, s[0:1]`.
///
/// NOTE: requires regenerated SOP1/VOPC sources (codegen emits both the
/// unconditional opcode-implied set and the runtime operand-encoded check).

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

namespace {

using namespace rocjitsu;

// ---------------------------------------------------------------------------
// SOP1 encoding (32-bit):
//   bits [7:0]   = ssrc0
//   bits [15:8]  = op
//   bits [22:16] = sdst
//   bits [31:23] = encoding (0x17D for SOP1, => prefix 0xBE800000)
//
// Special SDST encoding values: 126 = EXEC_LO, 127 = EXEC_HI. SDST=126 with
// a B64 destination spans EXEC_LO+EXEC_HI (full EXEC).
// ---------------------------------------------------------------------------

constexpr uint32_t sop1_word(uint32_t op, uint32_t sdst, uint32_t ssrc0) {
  return 0xBE800000u | (sdst << 16) | (op << 8) | ssrc0;
}

// SOP1 opcodes (CDNA3, GFX9 family).
constexpr uint32_t SOP1_S_MOV_B32 = 0;
constexpr uint32_t SOP1_S_MOV_B64 = 1;
constexpr uint32_t SOP1_S_AND_SAVEEXEC_B64 = 32;
constexpr uint32_t SOP1_S_ANDN1_WREXEC_B64 = 53;

// SDST encoding values for the EXEC mask registers.
constexpr uint32_t SDST_EXEC_LO = 126;
constexpr uint32_t SDST_EXEC_HI = 127;

// ---------------------------------------------------------------------------
// VOPC encoding (32-bit):
//   bits [8:0]   = src0
//   bits [16:9]  = vsrc1
//   bits [24:17] = op
//   bits [31:25] = encoding (0x3E for VOPC, => prefix 0x7C000000)
//
// VOPC writes a fixed implicit dst (VCC for v_cmp_*, EXEC for v_cmpx_*) — the
// destination is not encoded. EXEC_MODIFY for v_cmpx_* comes from the
// implicit OPR_SDST_EXEC operand in the MR ISA spec.
// ---------------------------------------------------------------------------

constexpr uint32_t vopc_word(uint32_t op, uint32_t vsrc1, uint32_t src0) {
  return 0x7C000000u | (op << 17) | (vsrc1 << 9) | src0;
}

// VOPC opcodes (CDNA3).
constexpr uint32_t VOPC_V_CMP_F_F32 = 0x40;
constexpr uint32_t VOPC_V_CMPX_F_F32 = 0x50;

// ---------------------------------------------------------------------------
// SMEM encoding (64-bit, CDNA3):
//   dword 0: [5:0]=sbase  [12:6]=sdata  [13]=pad  [14]=soffset_en
//            [15]=nv  [16]=glc  [17]=imm  [25:18]=op  [31:26]=encoding (0x30)
//   dword 1: [20:0]=offset  [26:21]=soffset
//
// Exercises OPR_SREG injection — `sdata` is OPR_SREG-typed, and the spec
// originally omits exec_lo/hi from its predefined values.
// ---------------------------------------------------------------------------

struct Smem64 {
  uint32_t words[2];
};

constexpr Smem64 smem_word(uint32_t op, uint32_t sdata, uint32_t sbase, uint32_t offset = 0) {
  return {{0xC0000000u | (op << 18) | (sdata << 6) | sbase, offset & 0x1FFFFFu}};
}

// SMEM opcodes (CDNA3).
constexpr uint32_t SMEM_S_LOAD_DWORD = 0;

// ---------------------------------------------------------------------------
// VOP3 encoding (64-bit, CDNA3):
//   dword 0: [7:0]=vdst  [10:8]=abs  [14:11]=op_sel  [15]=clamp
//            [25:16]=op  [31:26]=encoding (0x34)
//   dword 1: [8:0]=src0  [17:9]=src1  [26:18]=src2  [28:27]=omod  [31:29]=neg
//
// VOP3 v_cmp_*'s vdst is OPR_SREG-typed; encoding 126 names exec_lo even
// though the spec's OPR_SREG predefined values omit it. Tests the parser
// injection on a non-SOP1 encoding.
// ---------------------------------------------------------------------------

struct Vop3 {
  uint32_t words[2];
};

constexpr Vop3 vop3_word(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1) {
  return {{0xD0000000u | (op << 16) | vdst, (src1 << 9) | src0}};
}

// VOP3 opcodes for v_cmp_eq_u32 family (CDNA3).
constexpr uint32_t VOP3_V_CMP_EQ_U32 = 202;

// VGPR encoding values for VOP3 src fields.
constexpr uint32_t VGPR_V0 = 256;
constexpr uint32_t VGPR_V1 = 257;

// SREG encoding for vcc_lo — used as a near-miss negative.
constexpr uint32_t SDST_VCC_LO = 106;

struct ExecModifyCase {
  const char *label;
  uint32_t words[2]; ///< Up to two encoding words; second is 0 for 32-bit instructions.
  const char *expected_mnemonic;
  bool expect_exec_modify;
};

constexpr ExecModifyCase make_case(const char *label, uint32_t word, const char *mnemonic,
                                   bool expect) {
  return {label, {word, 0}, mnemonic, expect};
}

constexpr ExecModifyCase make_case(const char *label, Smem64 enc, const char *mnemonic,
                                   bool expect) {
  return {label, {enc.words[0], enc.words[1]}, mnemonic, expect};
}

constexpr ExecModifyCase make_case(const char *label, Vop3 enc, const char *mnemonic, bool expect) {
  return {label, {enc.words[0], enc.words[1]}, mnemonic, expect};
}

struct ArchEntry {
  rj_code_arch_t arch;
  const char *name;
};

class ExecModifyFlagTest : public ::testing::TestWithParam<std::tuple<ArchEntry, ExecModifyCase>> {
};

TEST_P(ExecModifyFlagTest, FlagMatchesExpectation) {
  const auto &[arch, tc] = GetParam();
  auto decoder = Decoder::create(arch.arch);
  ASSERT_NE(decoder, nullptr) << "Decoder::create() returned nullptr for " << arch.name;

  std::unique_ptr<Instruction> inst(decoder->decode(tc.words));
  ASSERT_NE(inst, nullptr) << "decode() returned nullptr for " << arch.name << "/" << tc.label;
  EXPECT_EQ(inst->mnemonic(), tc.expected_mnemonic)
      << "Wrong mnemonic for " << arch.name << "/" << tc.label;
  EXPECT_EQ(inst->is_exec_modify(), tc.expect_exec_modify)
      << "Wrong EXEC_MODIFY flag for " << arch.name << "/" << tc.label
      << " (disasm: " << inst->disassemble() << ")";
}

INSTANTIATE_TEST_SUITE_P(
    AllCdna, ExecModifyFlagTest,
    ::testing::Combine(
        ::testing::Values(ArchEntry{ROCJITSU_CODE_ARCH_CDNA3, "cdna3"},
                          ArchEntry{ROCJITSU_CODE_ARCH_CDNA4, "cdna4"}),
        ::testing::Values(
            // (a) Opcode-implied EXEC writes — flag set unconditionally.
            make_case("s_and_saveexec_b64_s23_s01",
                      sop1_word(SOP1_S_AND_SAVEEXEC_B64, /*sdst=*/2, /*ssrc0=*/0),
                      "s_and_saveexec_b64", true),
            make_case("s_andn1_wrexec_b64_s23_s01",
                      sop1_word(SOP1_S_ANDN1_WREXEC_B64, /*sdst=*/2, /*ssrc0=*/0),
                      "s_andn1_wrexec_b64", true),
            make_case("v_cmpx_f_f32_v1_v0", vopc_word(VOPC_V_CMPX_F_F32, /*vsrc1=*/1, /*src0=*/0),
                      "v_cmpx_f_f32_e32", true),

            // (b.1) Operand-encoded EXEC writes via OPR_SDST (s_mov_b32/64).
            make_case("s_mov_b64_exec_s01", sop1_word(SOP1_S_MOV_B64, SDST_EXEC_LO, /*ssrc0=*/0),
                      "s_mov_b64", true),
            make_case("s_mov_b32_exec_lo_s0", sop1_word(SOP1_S_MOV_B32, SDST_EXEC_LO, /*ssrc0=*/0),
                      "s_mov_b32", true),
            make_case("s_mov_b32_exec_hi_s0", sop1_word(SOP1_S_MOV_B32, SDST_EXEC_HI, /*ssrc0=*/0),
                      "s_mov_b32", true),

            // (b.2) Operand-encoded EXEC writes via OPR_SREG — exercises the
            // parser injection (spec omits exec_lo/hi for OPR_SREG).
            // SMEM s_load_dword exec_lo, s[0:1], 0 — sdata=126 (OPR_SREG).
            make_case("s_load_dword_exec_lo_s01",
                      smem_word(SMEM_S_LOAD_DWORD, /*sdata=*/SDST_EXEC_LO,
                                /*sbase=*/0),
                      "s_load_dword", true),
            // VOP3 v_cmp_eq_u32 exec, v0, v1 — vdst=126 (OPR_SREG).
            make_case("v_cmp_eq_u32_exec_v0_v1",
                      vop3_word(VOP3_V_CMP_EQ_U32, /*vdst=*/SDST_EXEC_LO,
                                /*src0=*/VGPR_V0, /*src1=*/VGPR_V1),
                      "v_cmp_eq_u32", true),

            // Negatives — flag must NOT be set.
            make_case("s_mov_b64_s23_s01", sop1_word(SOP1_S_MOV_B64, /*sdst=*/2, /*ssrc0=*/0),
                      "s_mov_b64", false),
            make_case("s_mov_b32_s5_s0", sop1_word(SOP1_S_MOV_B32, /*sdst=*/5, /*ssrc0=*/0),
                      "s_mov_b32", false),
            make_case("v_cmp_f_f32_v1_v0", vopc_word(VOPC_V_CMP_F_F32, /*vsrc1=*/1, /*src0=*/0),
                      "v_cmp_f_f32_e32", false),
            // Baseline SMEM with non-EXEC sdata.
            make_case("s_load_dword_s4_s01", smem_word(SMEM_S_LOAD_DWORD, /*sdata=*/4, /*sbase=*/0),
                      "s_load_dword", false),
            // Near-miss: vdst=vcc_lo (106) — close to EXEC_LO (126) but a
            // different special register; flag must not fire.
            make_case("v_cmp_eq_u32_vcc_v0_v1",
                      vop3_word(VOP3_V_CMP_EQ_U32, /*vdst=*/SDST_VCC_LO,
                                /*src0=*/VGPR_V0, /*src1=*/VGPR_V1),
                      "v_cmp_eq_u32", false))),
    [](const auto &info) {
      return std::string(std::get<0>(info.param).name) + "_" + std::get<1>(info.param).label;
    });

} // namespace
