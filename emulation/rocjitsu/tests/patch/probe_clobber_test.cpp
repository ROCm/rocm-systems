// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Synthetic, always-on coverage for the callee-body clobber summary.
// ProbeCallable is a plain struct, so these tests construct it directly from
// body words instead of parsing an ELF.
//
// Instruction encodings below are gfx90a ground truth captured from
// `llvm-mc -arch=amdgcn -mcpu=gfx90a -show-encoding`.

#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_clobber.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

// gfx90a instruction encodings (single 32-bit word).
constexpr uint32_t kSMovS5_0 = 0xbe850080;     // s_mov_b32 s5, 0
constexpr uint32_t kSWaitcnt0 = 0xbf8c0000;    // s_waitcnt 0
constexpr uint32_t kSNop0 = 0xbf800000;        // s_nop 0
constexpr uint32_t kSSetpcS30S31 = 0xbe801d1e; // s_setpc_b64 s[30:31]

// Explicit special-state writes. to_register_ref() currently returns nullopt
// for these operand forms, so the summary must recognize them via the operand
// name.
constexpr uint32_t kSMovExecLo_0 = 0xbefe0080;    // s_mov_b32 exec_lo, 0
constexpr uint32_t kSMovM0_0 = 0xbefc0080;        // s_mov_b32 m0, 0
constexpr uint32_t kSMovVccLo_0 = 0xbeea0080;     // s_mov_b32 vcc_lo, 0
constexpr uint32_t kSMovFlatScrLo_0 = 0xbee60080; // s_mov_b32 flat_scratch_lo, 0

constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA2;

ProbeCallable make_callable(std::vector<uint32_t> body) {
  ProbeCallable callable;
  callable.symbol = "rj_nop_probe";
  callable.arch = kArch;
  callable.body_words = std::move(body);
  callable.cc = ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31;
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

// A body that writes EXEC via an explicit operand must set touches_exec, even
// though to_register_ref() does not map the OPR_SDST_EXEC operand form. This is
// the decoder-gap regression guard: if the name fallback ever silently breaks,
// touches_exec goes false and this fails.
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

} // namespace
} // namespace rocjitsu
