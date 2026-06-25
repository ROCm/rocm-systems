// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/trampoline_builder.h"

#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/rj_code.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

uint32_t word_at(const std::vector<uint8_t> &bytes, size_t byte_off) {
  uint32_t w = 0;
  std::memcpy(&w, bytes.data() + byte_off, sizeof(w));
  return w;
}

int16_t decode_sopp_simm16(uint32_t word) { return static_cast<int16_t>(word & 0xFFFFu); }

uint64_t resolve_sopp_target(uint64_t branch_pc, uint32_t branch_word) {
  return branch_pc + 4 + static_cast<int64_t>(decode_sopp_simm16(branch_word)) * 4;
}

//==============================================================================
// Permanent contract: byte layout, branch math, arch honoring
//
// These tests describe what TrampolineBuilder::build() must always produce
// from a valid plan.
//==============================================================================

TEST(TrampolineBuilder, Emits4ByteRelocationAnchorPatch) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint64_t kAnchor = 0x100;
  constexpr uint64_t kTrampoline = 0x200;
  constexpr uint32_t kOriginalWord = 0xDEADBEEFu;

  TrampolinePlan plan;
  plan.arch = kArch;
  plan.anchor_offset = kAnchor;
  plan.original_size = 4;
  plan.trampoline_offset = kTrampoline;
  plan.return_target = kAnchor + 4;
  plan.original_words = {kOriginalWord};
  plan.before_items = {InlineAsmItem{{build_s_nop(0, kArch)}}};
  plan.emit_original = true;

  auto bytes = TrampolineBuilder::build(plan);
  ASSERT_TRUE(bytes.has_value());

  // Patched anchor is one s_branch covering the forward delta.
  // forward_simm16 = (0x200 - (0x100 + 4)) / 4 = 63.
  ASSERT_EQ(bytes->patched_anchor_bytes.size(), 4u);
  EXPECT_EQ(word_at(bytes->patched_anchor_bytes, 0), build_s_branch(63, kArch));

  // Trampoline: [before s_nop, original word, return s_branch].
  // return_branch_offset = 0x200 + 4 + 4 = 0x208.
  // return_simm16 = (0x104 - (0x208 + 4)) / 4 = -66.
  ASSERT_EQ(bytes->trampoline_words.size(), 3u);
  EXPECT_EQ(bytes->trampoline_words[0], build_s_nop(0, kArch));
  EXPECT_EQ(bytes->trampoline_words[1], kOriginalWord);
  EXPECT_EQ(bytes->trampoline_words[2], build_s_branch(-66, kArch));
}

TEST(TrampolineBuilder, Emits8ByteRelocationAnchorPatchWithNopTail) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint64_t kAnchor = 0x100;
  constexpr uint64_t kTrampoline = 0x200;
  constexpr uint32_t kW0 = 0xAAAA1111u;
  constexpr uint32_t kW1 = 0xBBBB2222u;

  TrampolinePlan plan;
  plan.arch = kArch;
  plan.anchor_offset = kAnchor;
  plan.original_size = 8;
  plan.trampoline_offset = kTrampoline;
  plan.return_target = kAnchor + 8;
  plan.original_words = {kW0, kW1};
  plan.before_items = {InlineAsmItem{{build_s_nop(0, kArch)}}};
  plan.emit_original = true;

  auto bytes = TrampolineBuilder::build(plan);
  ASSERT_TRUE(bytes.has_value());

  // Patched anchor is s_branch + s_nop 0 tail (preserves the 8-byte slot).
  ASSERT_EQ(bytes->patched_anchor_bytes.size(), 8u);
  EXPECT_EQ(word_at(bytes->patched_anchor_bytes, 0), build_s_branch(63, kArch));
  EXPECT_EQ(word_at(bytes->patched_anchor_bytes, 4), build_s_nop(0, kArch));

  // Trampoline: [before s_nop, w0, w1, return s_branch].
  // return_branch_offset = 0x200 + 4 + 8 = 0x20C.
  // return_simm16 = (0x108 - (0x20C + 4)) / 4 = -66.
  ASSERT_EQ(bytes->trampoline_words.size(), 4u);
  EXPECT_EQ(bytes->trampoline_words[0], build_s_nop(0, kArch));
  EXPECT_EQ(bytes->trampoline_words[1], kW0);
  EXPECT_EQ(bytes->trampoline_words[2], kW1);
  EXPECT_EQ(bytes->trampoline_words[3], build_s_branch(-66, kArch));
}

// build_s_branch uses opcode 32 on RDNA3/3.5/4 and opcode 2 on CDNA1-4. If the
// builder hard-coded one of those, this test would catch it.
TEST(TrampolineBuilder, RespectsTargetArchForBranchEncoding) {
  constexpr rj_code_arch_t kRdna = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr rj_code_arch_t kCdna = ROCJITSU_CODE_ARCH_CDNA4;

  TrampolinePlan plan;
  plan.arch = kRdna;
  plan.anchor_offset = 0x100;
  plan.original_size = 4;
  plan.trampoline_offset = 0x200;
  plan.return_target = 0x104;
  plan.original_words = {0xCAFEF00Du};
  plan.before_items = {InlineAsmItem{{build_s_nop(0, kRdna)}}};
  plan.emit_original = true;

  auto bytes = TrampolineBuilder::build(plan);
  ASSERT_TRUE(bytes.has_value());
  EXPECT_EQ(word_at(bytes->patched_anchor_bytes, 0), build_s_branch(63, kRdna));
  EXPECT_NE(word_at(bytes->patched_anchor_bytes, 0), build_s_branch(63, kCdna))
      << "Builder must use plan.arch, not a hard-coded opcode";
  EXPECT_EQ(bytes->trampoline_words.back(), build_s_branch(-66, kRdna));
}

TEST(TrampolineBuilder, ForwardBranchOverflowFails) {
  // forward_simm16 = (trampoline - (anchor + 4)) / 4. Place trampoline one
  // dword past the positive INT16 limit so the forward branch cannot fit.
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr int64_t kJustOver = (static_cast<int64_t>(std::numeric_limits<int16_t>::max()) + 1) * 4;
  constexpr uint64_t kAnchor = 0x100;
  const uint64_t kTrampoline = kAnchor + 4 + static_cast<uint64_t>(kJustOver);

  TrampolinePlan plan;
  plan.arch = kArch;
  plan.anchor_offset = kAnchor;
  plan.original_size = 4;
  plan.trampoline_offset = kTrampoline;
  plan.return_target = kAnchor + 4;
  plan.original_words = {0xDEADBEEFu};
  plan.before_items = {InlineAsmItem{{build_s_nop(0, kArch)}}};
  plan.emit_original = true;

  std::string err;
  EXPECT_FALSE(TrampolineBuilder::build(plan, &err).has_value());
  EXPECT_FALSE(err.empty()) << "Builder must explain the rejection";
  EXPECT_NE(err.find("forward"), std::string::npos)
      << "Diagnostic must identify the forward branch, got: " << err;
}

// original_size and original_words.size()*4 must agree. The builder rejects
// inconsistent plans rather than silently using one or the other.
TEST(TrampolineBuilder, RejectsOriginalWordsSizeMismatch) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  TrampolinePlan plan;
  plan.arch = kArch;
  plan.anchor_offset = 0x100;
  plan.original_size = 8; // expects 2 words ...
  plan.trampoline_offset = 0x200;
  plan.return_target = 0x108;
  plan.original_words = {0xDEADBEEFu}; // ... but only one provided.
  plan.before_items = {InlineAsmItem{{build_s_nop(0, kArch)}}};
  plan.emit_original = true;

  std::string err;
  EXPECT_FALSE(TrampolineBuilder::build(plan, &err).has_value());
  EXPECT_FALSE(err.empty()) << "Builder must explain the mismatch";
}

// arch defaults to ROCJITSU_CODE_ARCH_INVALID; a caller who forgets to set it
// must be rejected loudly rather than silently emitting a wrong-ISA encoding.
TEST(TrampolineBuilder, RejectsUnsetArch) {
  TrampolinePlan plan; // arch left at its ROCJITSU_CODE_ARCH_INVALID default.
  plan.anchor_offset = 0x100;
  plan.original_size = 4;
  plan.trampoline_offset = 0x200;
  plan.return_target = 0x104;
  plan.original_words = {0xDEADBEEFu};
  plan.emit_original = true;

  std::string err;
  EXPECT_FALSE(TrampolineBuilder::build(plan, &err).has_value());
  EXPECT_NE(err.find("arch"), std::string::npos)
      << "Diagnostic must identify the unset arch, got: " << err;
}

TEST(TrampolineBuilder, ReturnBranchOverflowFails) {
  // With forward_simm16 = INT16_MAX = 32767 (just in range) and
  // original_size = 4, the return branch needs simm16 = -32770 (one past
  // INT16_MIN). Asymmetric layout — trampoline placed exactly at the forward
  // limit forces the return out of range.
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint64_t kAnchor = 0;
  constexpr uint64_t kTrampoline = 4 + 32767ull * 4;

  TrampolinePlan plan;
  plan.arch = kArch;
  plan.anchor_offset = kAnchor;
  plan.original_size = 4;
  plan.trampoline_offset = kTrampoline;
  plan.return_target = kAnchor + 4;
  plan.original_words = {0xDEADBEEFu};
  plan.before_items = {InlineAsmItem{{build_s_nop(0, kArch)}}};
  plan.emit_original = true;

  std::string err;
  EXPECT_FALSE(TrampolineBuilder::build(plan, &err).has_value());
  EXPECT_FALSE(err.empty());
  EXPECT_NE(err.find("return"), std::string::npos)
      << "Diagnostic must identify the return branch, got: " << err;
}

// Decode each emitted s_branch back through SOPP semantics and confirm it
// lands at the plan-specified target. Pins the negative-immediate path:
// build_s_branch packs a signed int16 into a uint16 field, and a wrong
// sign-extension on decode would not be caught by the byte-equality
// assertions in the earlier tests.
TEST(TrampolineBuilder, EncodedBranchesRoundTripToPlanCoordinates) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint64_t kAnchor = 0x100;
  constexpr uint64_t kTrampoline = 0x200;

  TrampolinePlan plan;
  plan.arch = kArch;
  plan.anchor_offset = kAnchor;
  plan.original_size = 4;
  plan.trampoline_offset = kTrampoline;
  plan.return_target = kAnchor + 4;
  plan.original_words = {0xDEADBEEFu};
  plan.before_items = {InlineAsmItem{{build_s_nop(0, kArch)}}};
  plan.emit_original = true;

  auto bytes = TrampolineBuilder::build(plan);
  ASSERT_TRUE(bytes.has_value());

  // Forward: the anchor word decodes to the trampoline offset.
  const uint32_t fwd_word = word_at(bytes->patched_anchor_bytes, 0);
  EXPECT_EQ(resolve_sopp_target(kAnchor, fwd_word), kTrampoline);

  // Return: the last trampoline word (negative immediate) decodes back to
  // return_target. This is the only assertion in the file that exercises the
  // negative-immediate sign-extension path semantically rather than by byte
  // equality against build_s_branch(-66, ...).
  const uint64_t ret_pc = kTrampoline + (bytes->trampoline_words.size() - 1) * sizeof(uint32_t);
  EXPECT_EQ(resolve_sopp_target(ret_pc, bytes->trampoline_words.back()), plan.return_target);
}

// Under the inline-nop smoke body (1 nop + 1-2 original words = 8-12 bytes
// between forward and return branches), the asymmetry forces a layout where
// forward = INT16_MAX implies return = -32770 and vice versa. Positive-limit
// success cases at the builder level are therefore not constructible without
// pathological return_target divergence; the math-level positive-limit cases
// are covered by ComputeSoppBranchSimm16.MaxPositiveSimm16 in
// instruction_builder_test.cpp.

TEST(TrampolineBuilder, ForwardSimm16AtNegativeLimitSucceeds) {
  // Trampoline placed before the anchor so the forward branch goes backward.
  // forward_simm16 = (trampoline - (anchor + 4)) / 4 = INT16_MIN = -32768
  //   → trampoline = anchor + 4 + (-32768)*4
  //   With anchor = 131068, trampoline = 0.
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint64_t kTrampoline = 0;
  constexpr uint64_t kAnchor = 131068;

  TrampolinePlan plan;
  plan.arch = kArch;
  plan.anchor_offset = kAnchor;
  plan.original_size = 4;
  plan.trampoline_offset = kTrampoline;
  plan.return_target = kAnchor + 4;
  plan.original_words = {0xDEADBEEFu};
  plan.before_items = {InlineAsmItem{{build_s_nop(0, kArch)}}};
  plan.emit_original = true;

  auto bytes = TrampolineBuilder::build(plan);
  ASSERT_TRUE(bytes.has_value());
  EXPECT_EQ(word_at(bytes->patched_anchor_bytes, 0),
            build_s_branch(std::numeric_limits<int16_t>::min(), kArch));
}

TEST(TrampolineBuilder, ReturnSimm16AtNegativeLimitSucceeds) {
  // Trampoline placed far ahead of the anchor so the return branch goes
  // backward at exactly INT16_MIN.
  //   return_branch_pc = trampoline + 4 + original_size
  //   return_simm16    = (return_target - return_branch_pc - 4) / 4 = -32768
  //   With anchor = 0, original_size = 4, return_target = 4:
  //     trampoline + 8 + 4 = 4 + 131072  →  trampoline = 131064
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint64_t kAnchor = 0;
  constexpr uint64_t kTrampoline = 131064;

  TrampolinePlan plan;
  plan.arch = kArch;
  plan.anchor_offset = kAnchor;
  plan.original_size = 4;
  plan.trampoline_offset = kTrampoline;
  plan.return_target = kAnchor + 4;
  plan.original_words = {0xDEADBEEFu};
  plan.before_items = {InlineAsmItem{{build_s_nop(0, kArch)}}};
  plan.emit_original = true;

  auto bytes = TrampolineBuilder::build(plan);
  ASSERT_TRUE(bytes.has_value());
  EXPECT_EQ(bytes->trampoline_words.back(),
            build_s_branch(std::numeric_limits<int16_t>::min(), kArch));
}

// NOTE: the inline-nop guardrail used to live in TrampolineBuilder and was
// tested here. It has been moved to the orchestrator boundary as
// validate_inline_nop_plan() in instrumentor.h, and the test moved with it
// (see InlineNopGuardrail.* in instrumentor_test.cpp). The builder is now
// generic and accepts any well-formed plan; milestone-scoped restrictions
// are the orchestrator's responsibility.

} // namespace
} // namespace rocjitsu
