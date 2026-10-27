// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Synthetic, always-on coverage for the callee-body clobber summary.
// ProbeCallable is a plain struct, so these tests construct it directly from
// body words instead of parsing an ELF.
//
// The original valid fixtures were captured with
// `llvm-mc -arch=amdgcn -mcpu=<gfx90a|gfx1200> -show-encoding`.
// Malformed-tuple cases below deliberately replace their register selectors.

#include "../dbi_test_util.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_clobber.h"
#include "rocjitsu/code/patch/probe_symbol.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

namespace rocjitsu {
namespace {

// gfx90a instruction encodings (single 32-bit word).
constexpr uint32_t kSMovS5_0 = 0xbe850080;     // s_mov_b32 s5, 0
constexpr uint32_t kSWaitcnt0 = 0xbf8c0000;    // s_waitcnt 0
constexpr uint32_t kSNop0 = 0xbf800000;        // s_nop 0
constexpr uint32_t kSSetpcS30S31 = 0xbe801d1e; // s_setpc_b64 s[30:31]

// gfx1200 encodings. RDNA4 global memory instructions (VGLOBAL) are 96-bit.
constexpr uint32_t kGfx12GlobalLoadW0 = 0xee050002; // global_load_b32 v1, v0, s[2:3] (word 0)
constexpr uint32_t kGfx12GlobalLoadW1 = 0x00000001; // ... (word 1)
constexpr uint32_t kGfx12GlobalLoadW2 = 0x00000000; // ... (word 2)
constexpr uint32_t kGfx12SSetpcS30S31 = 0xbe80481e; // s_setpc_b64 s[30:31]

// Explicit special-state writes. Generated selector references and fieldless
// special effects must identify these structurally, without display-name parsing.
constexpr uint32_t kSMovExecLo_0 = 0xbefe0080;    // s_mov_b32 exec_lo, 0
constexpr uint32_t kSMovM0_0 = 0xbefc0080;        // s_mov_b32 m0, 0
constexpr uint32_t kSMovVccLo_0 = 0xbeea0080;     // s_mov_b32 vcc_lo, 0
constexpr uint32_t kSMovFlatScrLo_0 = 0xbee60080; // s_mov_b32 flat_scratch_lo, 0

// VOPC compares model their special-state write as a named destination operand:
// v_cmp_eq_u32 exposes "vcc"; v_cmpx_eq_u32 exposes "vcc" and "exec". The summary's
// structural operand scan therefore sees these defs (see the *WriteFromCompareIsDetected
// tests), even though the write carries no explicitly-encoded operand field.
constexpr uint32_t kVCmpEqU32 = 0x7D940000;  // v_cmp_eq_u32 vcc, s0, v0
constexpr uint32_t kVCmpxEqU32 = 0x7DB40000; // v_cmpx_eq_u32 s0, v0

constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA2;

ProbeCallable make_callable(std::vector<uint32_t> body) {
  ProbeCallable callable;
  callable.symbol = "rj_nop_probe";
  callable.arch = kArch;
  callable.body_words = std::move(body);
  callable.abi = *derive_probe_abi(ProbeCallingConvention::AmdGpuFuncReturnS30S31);
  return callable;
}

bool has_sgpr(const RegisterSet &set, uint16_t index) {
  return set.contains(RegisterRef{RegClass::SGPR, index, 1});
}

// A no-op probe body's clobber summary is empty: no ordinary registers, no
// special state, no private-segment use.
TEST(ProbeClobber, NopProbeSummaryIsEmpty) {
  const auto callable = make_callable({kSWaitcnt0, kSSetpcS30S31});
  std::string err;
  const auto summary = build_probe_clobber_summary(callable, &err);
  ASSERT_TRUE(summary.has_value()) << err;
  EXPECT_TRUE(summary->ordinary_clobbers.none());
  EXPECT_FALSE(summary->touches_exec);
  EXPECT_FALSE(summary->touches_vcc);
  EXPECT_FALSE(summary->touches_scc);
  EXPECT_FALSE(summary->touches_m0);
  EXPECT_FALSE(summary->touches_flat_scratch);
  EXPECT_FALSE(summary->uses_private_segment);
}

TEST(ProbeClobber, ReportsRejectedEncodingAndWordOffset) {
  const auto callable = make_callable({0xffffffffu, kSSetpcS30S31});
  std::string err;
  EXPECT_FALSE(build_probe_clobber_summary(callable, &err).has_value());
  EXPECT_NE(err.find("word 0"), std::string::npos) << err;
  EXPECT_NE(err.find("Invalid instruction opcode"), std::string::npos) << err;
}

// The summary is decode-derived, not declared: a body that writes s5 reports s5
// (and only s5) as an ordinary clobber.
TEST(ProbeClobber, DerivesOrdinaryClobberFromBody) {
  const auto callable = make_callable({kSMovS5_0, kSSetpcS30S31});
  std::string err;
  const auto summary = build_probe_clobber_summary(callable, &err);
  ASSERT_TRUE(summary.has_value()) << err;
  EXPECT_TRUE(has_sgpr(summary->ordinary_clobbers, 5));
  EXPECT_FALSE(has_sgpr(summary->ordinary_clobbers, 6));
}

// A 12-byte RDNA4 global load is summarized like any other instruction: its
// destination is a clobber, and the address registers it only reads are not.
TEST(ProbeClobber, DerivesClobberFromATwelveByteInstruction) {
  ProbeCallable callable = make_callable(
      {kGfx12GlobalLoadW0, kGfx12GlobalLoadW1, kGfx12GlobalLoadW2, kGfx12SSetpcS30S31});
  callable.arch = ROCJITSU_CODE_ARCH_RDNA4;
  std::string err;
  const auto summary = build_probe_clobber_summary(callable, &err);
  ASSERT_TRUE(summary.has_value()) << err;
  EXPECT_TRUE(summary->ordinary_clobbers.contains(RegisterRef{RegClass::VGPR, 1, 1}));
  EXPECT_FALSE(summary->ordinary_clobbers.contains(RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_FALSE(has_sgpr(summary->ordinary_clobbers, 2));
}

// The body ends one word into a 12-byte load. The summary decodes independently
// of build_probe_callable, so it must bound its own reads to the body.
TEST(ProbeClobber, RejectsAnInstructionRunningPastTheBody) {
  ProbeCallable callable = make_callable({kGfx12GlobalLoadW0});
  callable.arch = ROCJITSU_CODE_ARCH_RDNA4;
  std::string err;
  EXPECT_FALSE(build_probe_clobber_summary(callable, &err).has_value());
  EXPECT_NE(err.find("word 0"), std::string::npos) << err;
  EXPECT_NE(err.find("truncated"), std::string::npos) << err;
}

// The return-link use (s_setpc_b64 s[30:31]) is a use, not a def, so it must not
// appear in the probe body's ordinary clobbers; the call envelope owns the link
// pair instead.
TEST(ProbeClobber, ReturnLinkIsNotAProbeClobber) {
  const auto callable = make_callable({kSNop0, kSSetpcS30S31});
  std::string err;
  const auto summary = build_probe_clobber_summary(callable, &err);
  ASSERT_TRUE(summary.has_value()) << err;
  EXPECT_FALSE(has_sgpr(summary->ordinary_clobbers, 30));
  EXPECT_FALSE(has_sgpr(summary->ordinary_clobbers, 31));
}

// An explicit EXEC destination must be visible through the generated
// structural operand model and set touches_exec.
TEST(ProbeClobber, DetectsExplicitExecWrite) {
  const auto callable = make_callable({kSMovExecLo_0, kSSetpcS30S31});
  std::string err;
  const auto summary = build_probe_clobber_summary(callable, &err);
  ASSERT_TRUE(summary.has_value()) << err;
  EXPECT_TRUE(summary->touches_exec);
  EXPECT_FALSE(summary->touches_vcc);
  EXPECT_FALSE(summary->touches_m0);
  EXPECT_FALSE(summary->touches_flat_scratch);
}

// A body that writes M0 via an explicit operand must set touches_m0.
TEST(ProbeClobber, DetectsExplicitM0Write) {
  const auto callable = make_callable({kSMovM0_0, kSSetpcS30S31});
  std::string err;
  const auto summary = build_probe_clobber_summary(callable, &err);
  ASSERT_TRUE(summary.has_value()) << err;
  EXPECT_TRUE(summary->touches_m0);
  EXPECT_FALSE(summary->touches_exec);
  EXPECT_FALSE(summary->touches_vcc);
  EXPECT_FALSE(summary->touches_flat_scratch);
}

// A body that writes VCC via an explicit operand must set touches_vcc.
TEST(ProbeClobber, DetectsExplicitVccWrite) {
  const auto callable = make_callable({kSMovVccLo_0, kSSetpcS30S31});
  std::string err;
  const auto summary = build_probe_clobber_summary(callable, &err);
  ASSERT_TRUE(summary.has_value()) << err;
  EXPECT_TRUE(summary->touches_vcc);
  EXPECT_FALSE(summary->touches_exec);
  EXPECT_FALSE(summary->touches_m0);
  EXPECT_FALSE(summary->touches_flat_scratch);
}

// VOPC compares write VCC (v_cmp) / EXEC (v_cmpx) through a named destination
// operand the decoder exposes ("vcc", "exec"), so the summary's structural scan
// catches them even though the write carries no explicitly-encoded operand field.
// Missing effects remain Unknown and require preservation.
TEST(ProbeClobber, VccWriteFromCompareIsDetected) {
  const auto callable = make_callable({kVCmpEqU32, kSSetpcS30S31});
  std::string err;
  const auto summary = build_probe_clobber_summary(callable, &err);
  ASSERT_TRUE(summary.has_value()) << err;
  EXPECT_TRUE(summary->touches_vcc) << "v_cmp's VCC def is exposed as a named 'vcc' operand";
}

TEST(ProbeClobber, ExecWriteFromCmpxIsDetected) {
  const auto callable = make_callable({kVCmpxEqU32, kSSetpcS30S31});
  std::string err;
  const auto summary = build_probe_clobber_summary(callable, &err);
  ASSERT_TRUE(summary.has_value()) << err;
  EXPECT_TRUE(summary->touches_exec) << "v_cmpx's EXEC def is exposed as a named 'exec' operand";
}

// A body that writes FLAT_SCRATCH via an explicit operand must set
// touches_flat_scratch.
TEST(ProbeClobber, DetectsExplicitFlatScratchWrite) {
  const auto callable = make_callable({kSMovFlatScrLo_0, kSSetpcS30S31});
  std::string err;
  const auto summary = build_probe_clobber_summary(callable, &err);
  ASSERT_TRUE(summary.has_value()) << err;
  EXPECT_TRUE(summary->touches_flat_scratch);
  EXPECT_FALSE(summary->touches_exec);
  EXPECT_FALSE(summary->touches_vcc);
  EXPECT_FALSE(summary->touches_m0);
}

// A body that writes both an ordinary SGPR (s5) and EXEC (via the s_mov_b32
// exec_lo selector). ordinary_clobbers is built from InstDefUse's ordinary
// projection, so s5 appears there while no special member does. The
// selector-encoded EXEC write is not surfaced by InstDefUse; it is caught
// separately by scan_special_state and reported via touches_exec.
TEST(ProbeClobber, SpecialWritesDoNotLeakIntoOrdinaryClobbers) {
  const auto callable = make_callable({kSMovS5_0, kSMovExecLo_0, kSSetpcS30S31});
  std::string err;
  const auto summary = build_probe_clobber_summary(callable, &err);
  ASSERT_TRUE(summary.has_value()) << err;
  EXPECT_TRUE(has_sgpr(summary->ordinary_clobbers, 5));
  EXPECT_FALSE(summary->ordinary_clobbers.has_specials());
  EXPECT_TRUE(summary->touches_exec);
}

TEST(ProbeClobber, WideSgprTupleCannotHideSpecialSelectorOverlap) {
  // llvm-mc's s_load_dwordx16 s[0:15], s[0:1], 0 with SDATA changed to
  // 96. A decoder may reject the out-of-range tuple; if it accepts it, the
  // interval crosses VCC even though the first selector names an ordinary SGPR.
  ProbeCallable probe = make_callable({0xc0121800u, 0, kSSetpcS30S31});
  probe.arch = ROCJITSU_CODE_ARCH_CDNA3;
  probe.target = ROCJITSU_CODE_TARGET_GFX942;
  std::string error;
  const std::optional<ProbeClobberSummary> summary = build_probe_clobber_summary(probe, &error);
  if (summary)
    EXPECT_NE(summary->vcc_write, ProbeWriteProof::NoWrite);
  else
    EXPECT_FALSE(error.empty());
}

TEST(ProbeClobber, RejectsTtmpTuplesPastTheRegisterFile) {
  struct Target {
    rj_code_arch_t arch;
    rj_code_target_id_t target;
    std::vector<uint32_t> body;
  };
  for (const Target &target : {Target{ROCJITSU_CODE_ARCH_CDNA3,
                                      ROCJITSU_CODE_TARGET_GFX942,
                                      {0xc00e1e00u, 0x00000000u, kSSetpcS30S31}},
                               Target{ROCJITSU_CODE_ARCH_RDNA4,
                                      ROCJITSU_CODE_TARGET_GFX1200,
                                      {0xf4007e00u, 0xf8000000u, kGfx12SSetpcS30S31}}}) {
    SCOPED_TRACE(target.target);
    // Both encodings name ttmp[12:19], beyond the sixteen-register TTMP file.
    // Reject at decode rather than infer how invalid selectors affect EXEC.
    std::unique_ptr<Decoder> decoder =
        Decoder::create(default_isa_target_registry(), target.target);
    ASSERT_NE(decoder, nullptr);
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(decoder->decode_window(target.body, 0, diagnostic.emitter()).failed());
    EXPECT_NE(diagnostic.message().find("invalid scalar register selector"), std::string::npos);

    ProbeCallable probe = make_callable(target.body);
    probe.arch = target.arch;
    probe.target = target.target;
    std::string error;
    EXPECT_FALSE(build_probe_clobber_summary(probe, &error));
    EXPECT_NE(error.find("invalid scalar register selector"), std::string::npos);
  }
}

TEST(ProbeClobber, TtmpTupleMayEndAtTheLastRegister) {
  for (const auto &[arch, target, body] :
       {std::tuple{ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_TARGET_GFX942,
                   std::vector<uint32_t>{0xc00e1d00u, 0, kSSetpcS30S31}},
        std::tuple{ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1200,
                   std::vector<uint32_t>{0xf4007d00u, 0xf8000000u, kGfx12SSetpcS30S31}}}) {
    SCOPED_TRACE(target);
    std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
    ASSERT_NE(decoder, nullptr);
    DecodeResult decoded = decoder->decode_window(body);
    ASSERT_TRUE(decoded.succeeded());
    const std::unique_ptr<Instruction> instruction = std::move(decoded).value();
    const std::optional<RegisterRef> destination = instruction->dst_operand(0)->to_register_ref();
    ASSERT_TRUE(destination);
    EXPECT_EQ(*destination, (RegisterRef{RegClass::TTMP, 8, 8}));
    ProbeCallable probe = make_callable(body);
    probe.arch = arch;
    probe.target = target;
    const std::optional<ProbeClobberSummary> summary = build_probe_clobber_summary(probe);
    ASSERT_TRUE(summary);
    EXPECT_EQ(summary->exec_write, ProbeWriteProof::NoWrite);
    EXPECT_EQ(summary->vcc_write, ProbeWriteProof::NoWrite);
  }
}

TEST(ProbeClobber, WaveDependentTtmpExtentNeedsWaveContext) {
  for (uint8_t destination : {122, 123}) {
    SCOPED_TRACE(unsigned(destination));
    for (bool source_mask : {false, true}) {
      SCOPED_TRACE(source_mask);
      const auto words =
          source_mask
              ? rdna4::build_vop3(rdna4::kVCndmaskB32Vop3,
                                  {.vdst = 0, .src0 = 129, .src1 = 128, .src2 = destination})
              : rdna4::build_vop3(rdna4::kVCmpGtI32Vop3,
                                  {.vdst = destination, .src0 = 129, .src1 = 128});
      ProbeCallable probe = make_callable({words[0], words[1], kGfx12SSetpcS30S31});
      probe.arch = ROCJITSU_CODE_ARCH_RDNA4;
      probe.target = ROCJITSU_CODE_TARGET_GFX1200;
      const auto summary = build_probe_clobber_summary(probe);
      ASSERT_TRUE(summary);
      // TTMP14 fits either wave size. TTMP15 is valid on Wave32, but this
      // wave-independent proof cannot establish legality for Wave64.
      const ProbeWriteProof expected =
          destination == 122 ? ProbeWriteProof::NoWrite : ProbeWriteProof::Unknown;
      EXPECT_EQ(summary->exec_write, expected);
      EXPECT_EQ(summary->vcc_write, expected);
    }
  }
}

struct ProofTarget {
  rj_code_arch_t arch;
  rj_code_target_id_t target;
  uint32_t wait;
  uint32_t ret;
};

void PrintTo(const ProofTarget &target, std::ostream *out) {
  *out << "Target" << static_cast<int>(target.target);
}

class ProbeWriteProofTest : public ::testing::TestWithParam<ProofTarget> {
protected:
  ProbeCallable callable(std::vector<uint32_t> words) const {
    ProbeCallable probe = make_callable(std::move(words));
    probe.arch = GetParam().arch;
    probe.target = GetParam().target;
    return probe;
  }
};

TEST_P(ProbeWriteProofTest, ExactAuditedBodiesHaveNoWrites) {
  for (const std::vector<uint32_t> &words :
       {std::vector<uint32_t>{GetParam().ret},
        std::vector<uint32_t>{kSNop0, GetParam().wait, GetParam().ret}}) {
    const std::optional<ProbeClobberSummary> summary = build_probe_clobber_summary(callable(words));
    ASSERT_TRUE(summary);
    EXPECT_EQ(summary->exec_write, ProbeWriteProof::NoWrite);
    EXPECT_EQ(summary->vcc_write, ProbeWriteProof::NoWrite);
  }
}

TEST_P(ProbeWriteProofTest, ConcreteTargetProbeExtractionRecognizesSemanticReturn) {
  const IsaGpuTargetDescription *target =
      default_isa_target_registry().find_gpu_target(GetParam().target);
  ASSERT_NE(target, nullptr);
  const std::vector<uint8_t> image =
      test::make_amdgpu_probe_elf("proof", {kSNop0, GetParam().ret}, target->elf_machine);
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  std::string error;
  const std::optional<ResolvedProbeSymbol> symbol = resolve_probe_symbol(object, "proof", &error);
  ASSERT_TRUE(symbol) << error;
  const std::optional<ProbeCallable> extracted =
      build_probe_callable(object, *symbol, GetParam().arch, 0, &error);
  ASSERT_TRUE(extracted) << error;
  EXPECT_EQ(extracted->target, GetParam().target);
  const std::optional<ProbeClobberSummary> summary = build_probe_clobber_summary(*extracted);
  ASSERT_TRUE(summary);
  EXPECT_EQ(summary->exec_write, ProbeWriteProof::NoWrite);
  EXPECT_EQ(summary->vcc_write, ProbeWriteProof::NoWrite);
}

TEST_P(ProbeWriteProofTest, AbiReturnPredicateChecksSemanticsAndLinkPair) {
  const rj_code_arch_t arch = GetParam().arch;
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), GetParam().target);
  ASSERT_NE(decoder, nullptr);
  const ProbeAbi abi = callable({}).abi;
  for (const auto &[word, expected] :
       {std::pair{GetParam().ret, true}, std::pair{build_s_setpc_b64(28, arch), false},
        std::pair{build_s_swappc_b64(28, 30, arch), false},
        std::pair{build_s_mov_b64(8, 30, arch), false}}) {
    SCOPED_TRACE(word);
    DecodeResult decoded = decoder->decode_window(std::span(&word, 1));
    ASSERT_TRUE(decoded.succeeded());
    const std::unique_ptr<Instruction> instruction = std::move(decoded).value();
    EXPECT_EQ(is_probe_abi_return(*instruction, abi), expected);
    ProbeAbi invalid_abi = abi;
    invalid_abi.link_pair_base = 28;
    EXPECT_FALSE(is_probe_abi_return(*instruction, invalid_abi));
  }
}

TEST_P(ProbeWriteProofTest, OrdinaryAndReadOnlyOperationsHaveIndependentProofs) {
  const rj_code_arch_t arch = GetParam().arch;
  const uint32_t ret = GetParam().ret;
  for (uint32_t instruction : {build_s_nop(1, arch), build_s_mov_b32(5, 128, arch),
                               build_s_mov_b32(5, scalar_operand_exec_lo(arch), arch),
                               build_s_mov_b32(5, scalar_operand_vcc_lo(arch), arch)}) {
    const std::optional<ProbeClobberSummary> summary =
        build_probe_clobber_summary(callable({instruction, ret}));
    ASSERT_TRUE(summary);
    EXPECT_EQ(summary->exec_write, ProbeWriteProof::NoWrite);
    EXPECT_EQ(summary->vcc_write, ProbeWriteProof::NoWrite);
  }
  const std::optional<ProbeClobberSummary> exec = build_probe_clobber_summary(
      callable({build_s_mov_b32(scalar_operand_exec_lo(arch), 128, arch), ret}));
  ASSERT_TRUE(exec);
  EXPECT_EQ(exec->exec_write, ProbeWriteProof::MayWrite);
  EXPECT_EQ(exec->vcc_write, ProbeWriteProof::NoWrite);
  const std::optional<ProbeClobberSummary> vcc = build_probe_clobber_summary(
      callable({build_s_mov_b32(scalar_operand_vcc_lo(arch), 128, arch), ret}));
  ASSERT_TRUE(vcc);
  EXPECT_EQ(vcc->exec_write, ProbeWriteProof::NoWrite);
  EXPECT_EQ(vcc->vcc_write, ProbeWriteProof::MayWrite);
}

TEST_P(ProbeWriteProofTest, InternalDirectBranchesHaveNoWrites) {
  const std::optional<ProbeClobberSummary> summary =
      build_probe_clobber_summary(callable({build_s_branch(0, GetParam().arch), GetParam().ret}));
  ASSERT_TRUE(summary);
  EXPECT_EQ(summary->exec_write, ProbeWriteProof::NoWrite);
  EXPECT_EQ(summary->vcc_write, ProbeWriteProof::NoWrite);
}

TEST_P(ProbeWriteProofTest, UncoveredEffectsAndControlTransfersStayUnknown) {
  const rj_code_arch_t arch = GetParam().arch;
  const uint32_t ret = GetParam().ret;
  const std::vector<std::vector<uint32_t>> bodies{
      {},
      {kSNop0},
      {arch == ROCJITSU_CODE_ARCH_CDNA5 || arch == ROCJITSU_CODE_ARCH_RDNA4 ? 0xbfb50000u
                                                                            : 0xbf8f0000u,
       ret},

      {build_s_branch(-2, arch), ret},
      {build_s_branch(1, arch), ret},
      {build_s_branch(1, arch), build_s_mov_b32(5, 255, arch), 0, ret},
      {ret, ret},
      {build_s_setpc_b64(28, arch), ret},
      {build_s_swappc_b64(28, 30, arch), ret}};
  for (const std::vector<uint32_t> &body : bodies) {
    SCOPED_TRACE(::testing::PrintToString(body));
    const std::optional<ProbeClobberSummary> summary = build_probe_clobber_summary(callable(body));
    ASSERT_TRUE(summary);
    EXPECT_EQ(summary->exec_write, ProbeWriteProof::Unknown);
    EXPECT_EQ(summary->vcc_write, ProbeWriteProof::Unknown);
  }
}

TEST_P(ProbeWriteProofTest, SelectorWritesIncludingBothHalvesCannotBecomeNoWrite) {
  const rj_code_arch_t arch = GetParam().arch;
  for (uint16_t half : {0, 1}) {
    for (bool exec : {false, true}) {
      const uint16_t selector =
          (exec ? scalar_operand_exec_lo(arch) : scalar_operand_vcc_lo(arch)) + half;
      const std::optional<ProbeClobberSummary> summary = build_probe_clobber_summary(
          callable({build_s_mov_b32(selector, 128, arch), GetParam().ret}));
      ASSERT_TRUE(summary);
      EXPECT_EQ(exec ? summary->exec_write : summary->vcc_write, ProbeWriteProof::MayWrite);
      EXPECT_EQ(exec ? summary->vcc_write : summary->exec_write, ProbeWriteProof::NoWrite);
    }
  }
}

TEST_P(ProbeWriteProofTest, CompareAndCmpxCannotBecomeNoWrite) {
  const bool rdna =
      GetParam().arch == ROCJITSU_CODE_ARCH_RDNA4 || GetParam().arch == ROCJITSU_CODE_ARCH_CDNA5;
  const uint32_t cmp = rdna ? 0x7c940080u : 0x7d940080u;
  const uint32_t cmpx = rdna ? 0x7d940080u : 0x7db40080u;
  const std::optional<ProbeClobberSummary> vcc =
      build_probe_clobber_summary(callable({cmp, GetParam().ret}));
  ASSERT_TRUE(vcc);
  EXPECT_EQ(vcc->vcc_write, ProbeWriteProof::MayWrite);
  const std::optional<ProbeClobberSummary> exec =
      build_probe_clobber_summary(callable({cmpx, GetParam().ret}));
  ASSERT_TRUE(exec);
  EXPECT_EQ(exec->exec_write, ProbeWriteProof::MayWrite);
  EXPECT_EQ(exec->vcc_write, rdna ? ProbeWriteProof::NoWrite : ProbeWriteProof::MayWrite);
}

TEST_P(ProbeWriteProofTest, Vop3CompareUsesActualScalarDestination) {
  const bool gfx12 =
      GetParam().arch == ROCJITSU_CODE_ARCH_RDNA4 || GetParam().arch == ROCJITSU_CODE_ARCH_CDNA5;
  // llvm-mc: v_cmp_eq_u32_e64 s[8:9], v0, v1 (GFX9), s8 (GFX12).
  const uint32_t first = gfx12 ? 0xd44a0008u : 0xd0ca0008u;
  const uint32_t second = gfx12 ? 0x02020300u : 0x00020300u;
  const std::optional<ProbeClobberSummary> summary =
      build_probe_clobber_summary(callable({first, second, GetParam().ret}));
  ASSERT_TRUE(summary);
  EXPECT_EQ(summary->exec_write, ProbeWriteProof::NoWrite);
  EXPECT_EQ(summary->vcc_write, ProbeWriteProof::NoWrite);
  EXPECT_TRUE(summary->ordinary_clobbers.contains(RegisterRef{RegClass::SGPR, 8, 1}));
}

TEST_P(ProbeWriteProofTest, MissingTargetOrInvalidAbiCannotProveNoWrite) {
  ProbeCallable probe = callable({kSNop0, GetParam().ret});
  probe.target = ROCJITSU_CODE_TARGET_INVALID;
  const std::optional<ProbeClobberSummary> missing_target = build_probe_clobber_summary(probe);
  ASSERT_TRUE(missing_target);
  EXPECT_EQ(missing_target->exec_write, ProbeWriteProof::Unknown);
  EXPECT_EQ(missing_target->vcc_write, ProbeWriteProof::Unknown);
  probe = callable({kSNop0, GetParam().ret});
  probe.abi.link_pair_base = 28;
  const std::optional<ProbeClobberSummary> invalid_abi = build_probe_clobber_summary(probe);
  ASSERT_TRUE(invalid_abi);
  EXPECT_EQ(invalid_abi->exec_write, ProbeWriteProof::Unknown);
  EXPECT_EQ(invalid_abi->vcc_write, ProbeWriteProof::Unknown);
}

TEST(ProbeClobber, Gfx1201UsesItsGeneratedEffectModel) {
  ProbeCallable probe = make_callable({kSNop0, kGfx12SSetpcS30S31});
  probe.arch = ROCJITSU_CODE_ARCH_RDNA4;
  probe.target = ROCJITSU_CODE_TARGET_GFX1201;
  const std::optional<ProbeClobberSummary> summary = build_probe_clobber_summary(probe);
  ASSERT_TRUE(summary);
  EXPECT_EQ(summary->exec_write, ProbeWriteProof::NoWrite);
  EXPECT_EQ(summary->vcc_write, ProbeWriteProof::NoWrite);
}

INSTANTIATE_TEST_SUITE_P(
    AuditedTargets, ProbeWriteProofTest,
    ::testing::Values(
        ProofTarget{ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_TARGET_GFX90A, 0xbf8c0000, 0xbe801d1e},
        ProofTarget{ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_TARGET_GFX942, 0xbf8c0000, 0xbe801d1e},
        ProofTarget{ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_TARGET_GFX950, 0xbf8c0000, 0xbe801d1e},
        ProofTarget{ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1200, 0xbf890000, 0xbe80481e},
        ProofTarget{ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201, 0xbf890000, 0xbe80481e},
        ProofTarget{ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1250, 0xbfc00000, 0xbe80481e},
        ProofTarget{ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1251, 0xbfc00000,
                    0xbe80481e}));

} // namespace
} // namespace rocjitsu
