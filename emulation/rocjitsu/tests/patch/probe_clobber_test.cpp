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

} // namespace
} // namespace rocjitsu
