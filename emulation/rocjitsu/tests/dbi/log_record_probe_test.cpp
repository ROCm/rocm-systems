// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The record-writing test probe (log_record_probe.h) without running it: its
// RDNA4 hazard waits, and the register footprint it declares.
// dbi_log_record_sim_test.cpp runs it; scalar_builders_test.cpp pins the
// encoders it is built from.
//
// The wait's encoding is from `llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding`.

#include "log_record_probe.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_clobber.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu {
namespace test {
namespace {

constexpr uint32_t kRecordType = 7;
constexpr uint32_t kSlotCount = 4;

// The header's footprint is the one the body has. Tests rely on it to keep the
// probe's registers dead at their anchors, and nothing checks a probe's
// registers against the kernel's allocation.
void expect_declared_footprint(rj_code_arch_t arch) {
  ProbeCallable callable;
  callable.symbol = "rj_log_record_probe";
  callable.arch = arch;
  callable.body_words = build_log_record_probe_body(arch, kRecordType, kSlotCount);
  callable.abi =
      *derive_probe_abi(ProbeCallingConvention::AmdGpuFuncReturnS30S31, kLogRecordProbeArgDwords);
  std::string err;
  const auto summary = build_probe_clobber_summary(callable, &err);
  ASSERT_TRUE(summary.has_value()) << err;
  const RegisterSet &clobbers = summary->ordinary_clobbers;
  for (uint16_t sgpr = 0; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr)
    EXPECT_EQ(clobbers.contains(RegisterRef{RegClass::SGPR, sgpr, 1}),
              sgpr >= 4 && sgpr <= kLogRecordProbeLastSgpr)
        << "s" << sgpr;
  // The written VGPRs are checked at their bounds: no argument register, the
  // last one, and nothing above it. Unlike the SGPRs they are not contiguous,
  // since the body skips registers to keep its tuples even-aligned.
  for (uint16_t vgpr = 0; vgpr < kLogRecordProbeArgDwords; ++vgpr)
    EXPECT_FALSE(clobbers.contains(RegisterRef{RegClass::VGPR, vgpr, 1})) << "argument v" << vgpr;
  EXPECT_TRUE(clobbers.contains(RegisterRef{RegClass::VGPR, kLogRecordProbeLastVgpr, 1}));
  for (uint16_t vgpr = kLogRecordProbeLastVgpr + 1; vgpr < REGISTER_SET_MAX_VGPRS; ++vgpr)
    EXPECT_FALSE(clobbers.contains(RegisterRef{RegClass::VGPR, vgpr, 1})) << "v" << vgpr;
  for (uint16_t acc = 0; acc < REGISTER_SET_MAX_ACC_VGPRS; ++acc)
    EXPECT_FALSE(clobbers.contains(RegisterRef{RegClass::ACC_VGPR, acc, 1})) << "acc" << acc;
  // The forced full mask is the caller's; the probe must not narrow it.
  EXPECT_FALSE(summary->touches_exec);
}

// LLVM's amdgpu-wait-sgpr-hazards pass, run on this body for gfx1200, places
// `s_wait_alu 0xfffe` after s_and_b32 s12, s_lshl_b32 s12, s_add_u32 s12 and
// s_addc_u32 s11. The simulator models no hazards, so the positions are what
// can be checked.
constexpr uint32_t kSWaitAluSaSdst = 0xbf88fffe; // s_wait_alu 0xfffe on gfx1200

TEST(LogRecordProbe, Rdna4WaitsAfterEachSaluWriteLlvmFlags) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  const std::vector<uint32_t> body = build_log_record_probe_body(kArch, kRecordType, kSlotCount);
  EXPECT_EQ(std::ranges::count(body, kSWaitAluSaSdst), 4);

  // The wait must be the word after @p inst, or after its literal when it has one.
  const auto expect_wait_after = [&](uint32_t inst, size_t literal_words, const char *what) {
    const auto it = std::ranges::find(body, inst);
    ASSERT_NE(it, body.end()) << what << " not found";
    const auto at = static_cast<size_t>(it - body.begin()) + 1 + literal_words;
    ASSERT_LT(at, body.size()) << what;
    EXPECT_EQ(body[at], kSWaitAluSaSdst) << "no wait after " << what;
  };
  expect_wait_after(build_s_and_b32(12, 10, scalar_positive_inline_u32(kSlotCount - 1), kArch), 0,
                    "s_and_b32 s12");
  expect_wait_after(build_s_lshl_b32(12, 12, scalar_positive_inline_u32(6), kArch), 0,
                    "s_lshl_b32 s12");
  expect_wait_after(build_s_add_u32(12, 12, kScalarSrcLiteral, kArch), 1, "s_add_u32 s12");
  expect_wait_after(build_s_addc_u32(11, 11, scalar_positive_inline_u32(0), kArch), 0,
                    "s_addc_u32 s11");
}

TEST(LogRecordProbe, CdnaBodiesCarryNoRdna4Wait) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    const std::vector<uint32_t> body = build_log_record_probe_body(arch, kRecordType, kSlotCount);
    EXPECT_EQ(std::ranges::count(body, kSWaitAluSaSdst), 0);
  }
}

TEST(LogRecordProbe, DeclaredFootprintMatchesTheBodyOnCdna3) {
  expect_declared_footprint(ROCJITSU_CODE_ARCH_CDNA3);
}
TEST(LogRecordProbe, DeclaredFootprintMatchesTheBodyOnCdna4) {
  expect_declared_footprint(ROCJITSU_CODE_ARCH_CDNA4);
}
TEST(LogRecordProbe, DeclaredFootprintMatchesTheBodyOnRdna4) {
  expect_declared_footprint(ROCJITSU_CODE_ARCH_RDNA4);
}

} // namespace
} // namespace test
} // namespace rocjitsu
