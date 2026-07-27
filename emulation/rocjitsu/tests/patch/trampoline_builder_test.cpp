// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/trampoline_builder.h"

#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/rj_code.h"

#include <gtest/gtest.h>

#include <algorithm>
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
  plan.original_words.assign(1, kOriginalWord);
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
  plan.original_words.assign(1, 0xCAFEF00Du);
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
  plan.original_words.assign(1, 0xDEADBEEFu);
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
  plan.original_words.assign(1, 0xDEADBEEFu); // ... but only one provided.
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
  plan.original_words.assign(1, 0xDEADBEEFu);
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
  plan.original_words.assign(1, 0xDEADBEEFu);
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
  plan.original_words.assign(1, 0xDEADBEEFu);
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
  plan.original_words.assign(1, 0xDEADBEEFu);
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
  plan.original_words.assign(1, 0xDEADBEEFu);
  plan.before_items = {InlineAsmItem{{build_s_nop(0, kArch)}}};
  plan.emit_original = true;

  auto bytes = TrampolineBuilder::build(plan);
  ASSERT_TRUE(bytes.has_value());
  EXPECT_EQ(bytes->trampoline_words.back(),
            build_s_branch(std::numeric_limits<int16_t>::min(), kArch));
}

TEST(DbiPatchPlacementPlanner, ReservesSequentialAppendedCavesWithExplicitMappings) {
  DbiPatchPlacementPlanner planner(ROCJITSU_CODE_ARCH_RDNA4, /*original_text_size=*/256);
  DbiPatchPlacementRequest first;
  first.anchor_offset = 16;
  first.original_size = 8;
  first.body_size = 40;
  first.inline_capacity = 8;
  DbiPatchPlacementRequest second = first;
  second.anchor_offset = 32;
  second.body_size = 24;

  const auto first_placement = planner.plan(first);
  const auto second_placement = planner.plan(second);

  ASSERT_TRUE(first_placement);
  ASSERT_TRUE(second_placement);
  EXPECT_EQ(first_placement->kind, DbiPatchPlacementKind::AppendedCave);
  EXPECT_EQ(first_placement->body_offset, 256u);
  EXPECT_EQ(first_placement->return_branch_offset, 296u);
  EXPECT_EQ(first_placement->return_target, 24u);
  EXPECT_EQ(second_placement->body_offset, 300u);
  EXPECT_EQ(second_placement->return_branch_offset, 324u);
  EXPECT_EQ(second_placement->return_target, 40u);
  EXPECT_EQ(planner.appended_end(), 328u);
}

TEST(DbiPatchPlacementPlanner, PrefersInlineThenLocalAndRejectsOverlapTransactionally) {
  DbiPatchPlacementPlanner planner(ROCJITSU_CODE_ARCH_RDNA4, /*original_text_size=*/512);
  DbiPatchPlacementRequest inline_request;
  inline_request.anchor_offset = 32;
  inline_request.original_size = 8;
  inline_request.body_size = 32;
  inline_request.inline_capacity = 40;
  inline_request.local_cave = DbiPatchLocalCave{256, 64};
  ASSERT_EQ(planner.plan(inline_request)->kind, DbiPatchPlacementKind::Inline);

  DbiPatchPlacementRequest local_request;
  local_request.anchor_offset = 96;
  local_request.original_size = 8;
  local_request.body_size = 40;
  local_request.inline_capacity = 8;
  local_request.local_cave = DbiPatchLocalCave{256, 64};
  const auto local = planner.plan(local_request);
  ASSERT_TRUE(local);
  EXPECT_EQ(local->kind, DbiPatchPlacementKind::LocalCave);
  EXPECT_EQ(local->body_offset, 256u);

  const uint64_t appended_before_failure = planner.appended_end();
  std::string error;
  DbiPatchPlacementRequest overlap = local_request;
  overlap.allow_appended_cave = false;
  EXPECT_FALSE(planner.plan(overlap, &error));
  EXPECT_NE(error.find("no nonoverlapping"), std::string::npos);
  EXPECT_EQ(planner.appended_end(), appended_before_failure);
}

TEST(DbiPatchPlacementPlanner, SeedsComposedRangesBeforeLaterFamilySelection) {
  DbiPatchPlacementPlanner planner(ROCJITSU_CODE_ARCH_RDNA4, /*original_text_size=*/512);
  EXPECT_TRUE(planner.reserve_existing_range(/*begin=*/32, /*size=*/48));
  EXPECT_TRUE(planner.reserve_existing_range(/*begin=*/256, /*size=*/64));
  std::string error;
  EXPECT_FALSE(planner.reserve_existing_range(/*begin=*/64, /*size=*/8, &error));
  EXPECT_NE(error.find("overlapping"), std::string::npos);

  DbiPatchPlacementRequest request;
  request.anchor_offset = 96;
  request.original_size = 4;
  request.body_size = 8;
  request.inline_capacity = 0;
  request.local_cave = DbiPatchLocalCave{/*offset=*/256, /*capacity=*/128};
  const auto placement = planner.plan(request);
  ASSERT_TRUE(placement);
  EXPECT_EQ(placement->kind, DbiPatchPlacementKind::AppendedCave);
  EXPECT_EQ(placement->body_offset, 512u);
}

TEST(DbiPatchPlacementPlanner, FailsBeforeReservingUnreachableAppendedMapping) {
  constexpr uint64_t kTextSize = 1u << 20u;
  DbiPatchPlacementPlanner planner(ROCJITSU_CODE_ARCH_RDNA4, kTextSize);
  DbiPatchPlacementRequest request;
  request.anchor_offset = 0;
  request.original_size = 4;
  request.body_size = 16;
  request.inline_capacity = 4;
  std::string error;

  EXPECT_FALSE(planner.plan(request, &error));
  EXPECT_NE(error.find("no nonoverlapping"), std::string::npos);
  EXPECT_TRUE(planner.occupied_ranges().empty());
  EXPECT_EQ(planner.appended_end(), kTextSize);
}

TEST(DbiPatchPlacementPlanner, ReservesIndirectAppendedBodiesWithoutSoppReachability) {
  constexpr uint64_t kTextSize = 1u << 20u;
  DbiPatchPlacementPlanner planner(ROCJITSU_CODE_ARCH_RDNA4, kTextSize);

  const auto first = planner.plan_indirect_appended(
      /*anchor_offset=*/0, /*original_size=*/4, /*body_size=*/28);
  const auto second = planner.plan_indirect_appended(
      /*anchor_offset=*/16, /*original_size=*/4, /*body_size=*/12);

  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first->body_offset, kTextSize);
  EXPECT_EQ(first->body_size, 28u);
  EXPECT_EQ(second->body_offset, kTextSize + 28u);
  EXPECT_EQ(planner.appended_end(), kTextSize + 40u);
  EXPECT_FALSE(planner.plan_indirect_appended(
      /*anchor_offset=*/0, /*original_size=*/4, /*body_size=*/4));
}

TEST(DbiPatchPlacementPlanner, ReservesStableAppendedPrefixBeforeIndirectBodies) {
  constexpr uint64_t kTextSize = 4096u;
  DbiPatchPlacementPlanner planner(ROCJITSU_CODE_ARCH_RDNA4, kTextSize);

  EXPECT_TRUE(planner.reserve_appended_prefix(/*size=*/56));
  EXPECT_EQ(planner.appended_end(), kTextSize + 56u);
  const auto body = planner.plan_indirect_appended(
      /*anchor_offset=*/16, /*original_size=*/8, /*body_size=*/80);

  ASSERT_TRUE(body);
  EXPECT_EQ(body->body_offset, kTextSize + 56u);
  EXPECT_EQ(planner.appended_end(), kTextSize + 136u);
}

TEST(SoppBranchRelayPlanner, AssignsInterchangeableIslandsAtMaximumCardinality) {
  const std::vector<uint64_t> sources = {16, 0, 32};
  const std::vector<uint64_t> islands = {96, 80};
  const auto plan = plan_forward_sopp_branch_relays(sources, {}, islands);

  ASSERT_TRUE(plan);
  ASSERT_EQ(plan->routes.size(), 2u);
  EXPECT_EQ(plan->routes[0].source_index, 0u);
  EXPECT_EQ(plan->routes[0].island_offset, 96u);
  EXPECT_TRUE(plan->routes[0].relay_offsets.empty());
  EXPECT_EQ(plan->routes[1].source_index, 1u);
  EXPECT_EQ(plan->routes[1].island_offset, 80u);
  EXPECT_EQ(plan->rejected_source_indices, std::vector<size_t>({2}));
}

TEST(SoppBranchRelayPlanner, ReachesLongTextThroughCapacityOneRelayWords) {
  constexpr uint64_t kMaximumForwardHop = 4u + 32767u * 4u;
  const std::vector<uint64_t> sources = {0, 16};
  const std::vector<uint64_t> relays = {kMaximumForwardHop, kMaximumForwardHop + 16};
  const std::vector<uint64_t> islands = {2 * kMaximumForwardHop, 2 * kMaximumForwardHop + 16};
  const auto plan = plan_forward_sopp_branch_relays(sources, relays, islands);

  ASSERT_TRUE(plan);
  ASSERT_EQ(plan->routes.size(), 2u);
  EXPECT_TRUE(plan->rejected_source_indices.empty());
  for (const SoppBranchRelayRoute &route : plan->routes) {
    ASSERT_EQ(route.relay_offsets.size(), 1u);
    ASSERT_TRUE(compute_sopp_branch_simm16(sources[route.source_index], route.relay_offsets[0]));
    EXPECT_TRUE(compute_sopp_branch_simm16(route.relay_offsets[0], route.island_offset));
  }
  EXPECT_NE(plan->routes[0].relay_offsets[0], plan->routes[1].relay_offsets[0]);
  EXPECT_NE(plan->routes[0].island_offset, plan->routes[1].island_offset);
}

TEST(SoppBranchRelayPlanner, RoutesBackwardThroughCapacityOneRelayWords) {
  constexpr uint64_t kMaximumBackwardHop = 32768u * 4u - 4u;
  const std::vector<uint64_t> sources = {2u * kMaximumBackwardHop + 16u, 2u * kMaximumBackwardHop};
  const std::vector<uint64_t> relays = {kMaximumBackwardHop + 16u, kMaximumBackwardHop};
  const std::vector<uint64_t> islands = {16u, 0u};
  const auto plan = plan_backward_sopp_branch_relays(sources, relays, islands);

  ASSERT_TRUE(plan);
  ASSERT_EQ(plan->routes.size(), 2u);
  EXPECT_TRUE(plan->rejected_source_indices.empty());
  for (const SoppBranchRelayRoute &route : plan->routes) {
    ASSERT_EQ(route.relay_offsets.size(), 1u);
    ASSERT_TRUE(compute_sopp_branch_simm16(sources[route.source_index], route.relay_offsets[0]));
    EXPECT_TRUE(compute_sopp_branch_simm16(route.relay_offsets[0], route.island_offset));
  }
  EXPECT_NE(plan->routes[0].relay_offsets[0], plan->routes[1].relay_offsets[0]);
}

TEST(SoppBranchRelayPlanner, HonorsAsymmetricBackwardReachBoundary) {
  constexpr uint64_t kMaximumBackwardHop = 32768u * 4u - 4u;
  const auto accepted = plan_backward_sopp_branch_relays(std::vector<uint64_t>{kMaximumBackwardHop},
                                                         {}, std::vector<uint64_t>{0});
  const auto rejected = plan_backward_sopp_branch_relays(
      std::vector<uint64_t>{kMaximumBackwardHop + sizeof(uint32_t)}, {}, std::vector<uint64_t>{0});

  ASSERT_TRUE(accepted);
  EXPECT_EQ(accepted->routes.size(), 1u);
  ASSERT_TRUE(rejected);
  EXPECT_TRUE(rejected->routes.empty());
  EXPECT_EQ(rejected->rejected_source_indices, std::vector<size_t>({0}));
}

TEST(SoppBranchRelayPlanner, UsesDistinctRelaysToAdmitAllFeasibleSources) {
  constexpr uint64_t kHop = 4u + 32767u * 4u;
  const std::vector<uint64_t> sources = {0, 8};
  const std::vector<uint64_t> relays = {kHop, kHop + 8};
  const std::vector<uint64_t> islands = {2 * kHop, 2 * kHop + 8};
  const auto plan = plan_forward_sopp_branch_relays(sources, relays, islands);

  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->routes.size(), 2u);
  EXPECT_TRUE(plan->rejected_source_indices.empty());
}

TEST(SoppBranchRelayPlanner, ScalesAcrossLargeGeneratedCoordinateSets) {
  constexpr size_t kRouteCount = 1400u;
  constexpr uint64_t kMaximumForwardHop = 4u + 32767u * 4u;
  std::vector<uint64_t> sources;
  std::vector<uint64_t> relays;
  std::vector<uint64_t> islands;
  sources.reserve(kRouteCount);
  relays.reserve(kRouteCount);
  islands.reserve(kRouteCount);
  for (size_t index = 0; index < kRouteCount; ++index) {
    sources.push_back(index * sizeof(uint32_t));
    relays.push_back(kMaximumForwardHop + index * sizeof(uint32_t));
    islands.push_back(2u * kMaximumForwardHop + index * sizeof(uint32_t));
  }

  const auto plan = plan_forward_sopp_branch_relays(sources, relays, islands);

  ASSERT_TRUE(plan);
  ASSERT_EQ(plan->routes.size(), kRouteCount);
  EXPECT_TRUE(plan->rejected_source_indices.empty());
  for (size_t index = 0; index < kRouteCount; ++index) {
    EXPECT_EQ(plan->routes[index].source_index, index);
    EXPECT_EQ(plan->routes[index].relay_offsets, std::vector<uint64_t>({relays[index]}));
    EXPECT_EQ(plan->routes[index].island_offset, islands[index]);
  }
}

TEST(SoppBranchRelayPlanner, SharesNeitherRelayNorIslandAcrossRoutes) {
  constexpr uint64_t kHop = 4u + 32767u * 4u;
  const std::vector<uint64_t> sources = {0, 8};
  const std::vector<uint64_t> relays = {kHop};
  const std::vector<uint64_t> islands = {2 * kHop, 2 * kHop + 4};
  const auto plan = plan_forward_sopp_branch_relays(sources, relays, islands);

  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->routes.size(), 1u);
  EXPECT_EQ(plan->rejected_source_indices.size(), 1u);
}

TEST(SoppBranchRelayPlanner, RejectsUnalignedOrAliasedCoordinates) {
  std::string error;
  EXPECT_FALSE(plan_forward_sopp_branch_relays(std::vector<uint64_t>{2}, {},
                                               std::vector<uint64_t>{16}, &error));
  EXPECT_NE(error.find("aligned"), std::string::npos);

  error.clear();
  EXPECT_FALSE(plan_forward_sopp_branch_relays(std::vector<uint64_t>{0}, std::vector<uint64_t>{16},
                                               std::vector<uint64_t>{16}, &error));
  EXPECT_EQ(error,
            "SOPP relay planner: coordinates must be globally unique; offset=16 first=relay[0] "
            "second=island[0]");

  error.clear();
  EXPECT_FALSE(plan_forward_sopp_branch_relays(std::vector<uint64_t>{16}, std::vector<uint64_t>{16},
                                               std::vector<uint64_t>{32}, &error));
  EXPECT_EQ(error,
            "SOPP relay planner: coordinates must be globally unique; offset=16 first=source[0] "
            "second=relay[0]");
}

// NOTE: the inline-nop guardrail used to live in TrampolineBuilder and was
// tested here. It has been moved to the orchestrator boundary as
// validate_inline_nop_plan() in instrumentor.h, and the test moved with it
// (see InlineNopGuardrail.* in instrumentor_test.cpp). The builder is now
// generic and accepts any well-formed plan; milestone-scoped restrictions
// are the orchestrator's responsibility.

//==============================================================================
// Probe-call resource planning (plan_probe_call)
//
// Resource selection only: which envelope registers, how many envelope words.
// No layout, no bytes. Exercised here on synthetic RegisterSets.
//==============================================================================

// The only verified convention today; its link pair is s[30:31].
constexpr ProbeCallingConvention kNoArgsCc = ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31;

RegisterSet make_sgpr_set(std::initializer_list<uint16_t> indices) {
  RegisterSet set;
  for (uint16_t i : indices)
    set.expand(RegisterRef{RegClass::SGPR, i, 1});
  return set;
}

// All allocatable SGPRs marked live, except the listed indices left dead.
RegisterSet all_sgprs_live_except(std::initializer_list<uint16_t> dead) {
  RegisterSet set;
  for (uint16_t i = 0; i < REGISTER_SET_ALLOCATABLE_SGPRS; ++i)
    set.expand(RegisterRef{RegClass::SGPR, i, 1});
  for (uint16_t i : dead)
    set.erase(RegisterRef{RegClass::SGPR, i, 1});
  return set;
}

bool has_sgpr(const RegisterSet &set, uint16_t index) {
  return set.contains(RegisterRef{RegClass::SGPR, index, 1});
}

// A live link pair s[30:31] fails closed (until this is supported)
TEST(TrampolineBuilderPlan, LiveLinkPairFails) {
  TrampolinePlan plan;
  std::string err;
  EXPECT_FALSE(TrampolineBuilder::plan_probe_call(plan, kNoArgsCc, make_sgpr_set({30}),
                                                  /*probe_body_clobbers=*/{}, &err));
  EXPECT_NE(err.find("s[30:31]"), std::string::npos);
  EXPECT_FALSE(plan.is_probe_call); // plan left unmodified on failure.
}

// No dead even SGPR pair (everything live but the excluded link pair) fails and
// names the target resource.
TEST(TrampolineBuilderPlan, NoDeadTargetPairFails) {
  TrampolinePlan plan;
  std::string err;
  EXPECT_FALSE(TrampolineBuilder::plan_probe_call(plan, kNoArgsCc, all_sgprs_live_except({30, 31}),
                                                  /*probe_body_clobbers=*/{}, &err));
  EXPECT_NE(err.find("target"), std::string::npos);
}

// A target pair is available but nothing else is, so the SCC temp search fails
// and names the SCC resource.
TEST(TrampolineBuilderPlan, NoSccTempFails) {
  TrampolinePlan plan;
  std::string err;
  // s[0:1] is a dead even pair (the target); s30/s31 are the reserved link pair;
  // every other SGPR is live, so no SCC temp remains.
  EXPECT_FALSE(TrampolineBuilder::plan_probe_call(plan, kNoArgsCc,
                                                  all_sgprs_live_except({0, 1, 30, 31}),
                                                  /*probe_body_clobbers=*/{}, &err));
  EXPECT_NE(err.find("SCC"), std::string::npos);
}

// Mirror of NoSccTempFails with SCC preservation disabled: no SCC temp is
// needed, so the same register-starved kernel that fails closed above now plans
// successfully and reserves only the link + target pairs. Guards the regression
// where the SCC temp was searched/reserved even when preserve_scc was false.
TEST(TrampolineBuilderPlan, NoSccPreserveSkipsSccTemp) {
  TrampolinePlan plan;
  plan.preserve_scc = false;
  std::string err;
  // Same dead set as NoSccTempFails: only the link pair s[30:31] and the target
  // pair s[0:1] are dead; nothing remains for an SCC temp.
  ASSERT_TRUE(TrampolineBuilder::plan_probe_call(plan, kNoArgsCc,
                                                 all_sgprs_live_except({0, 1, 30, 31}),
                                                 /*probe_body_clobbers=*/{}, &err))
      << err;
  EXPECT_TRUE(plan.is_probe_call);
  // builder_clobbers = {link pair} | {target pair} only -- no SCC temp reserved
  EXPECT_TRUE(has_sgpr(plan.builder_clobbers, 30));
  EXPECT_TRUE(has_sgpr(plan.builder_clobbers, 31));
  EXPECT_TRUE(has_sgpr(plan.builder_clobbers, plan.target_pair_base));
  EXPECT_TRUE(has_sgpr(plan.builder_clobbers, plan.target_pair_base + 1));
  EXPECT_EQ(plan.builder_clobbers.size(), 4u);
}

// Happy path: dead resources selected, distinct, and reported as builder clobbers.
TEST(TrampolineBuilderPlan, SelectsDeadResourcesAndReportsClobbers) {
  TrampolinePlan plan;
  std::string err;
  // s4 live; everything else dead. Target pair and SCC temp must avoid s4 and the
  // link pair s[30:31].
  ASSERT_TRUE(TrampolineBuilder::plan_probe_call(plan, kNoArgsCc, make_sgpr_set({4}),
                                                 /*probe_body_clobbers=*/{}, &err));
  EXPECT_TRUE(plan.is_probe_call);
  EXPECT_EQ(plan.link_pair_base, 30u);

  // Target pair: even-aligned, not live, not the link pair.
  EXPECT_EQ(plan.target_pair_base % 2u, 0u);
  EXPECT_FALSE(has_sgpr(make_sgpr_set({4}), plan.target_pair_base));
  EXPECT_NE(plan.target_pair_base, 30u);

  // SCC temp: not live, not the link pair, outside the target pair.
  EXPECT_NE(plan.scc_temp, 4u);
  EXPECT_NE(plan.scc_temp, 30u);
  EXPECT_NE(plan.scc_temp, 31u);
  EXPECT_NE(plan.scc_temp, plan.target_pair_base);
  EXPECT_NE(plan.scc_temp, plan.target_pair_base + 1);

  // builder_clobbers = {link pair} | {target pair} | {scc temp}.
  EXPECT_TRUE(has_sgpr(plan.builder_clobbers, 30));
  EXPECT_TRUE(has_sgpr(plan.builder_clobbers, 31));
  EXPECT_TRUE(has_sgpr(plan.builder_clobbers, plan.target_pair_base));
  EXPECT_TRUE(has_sgpr(plan.builder_clobbers, plan.target_pair_base + 1));
  EXPECT_TRUE(has_sgpr(plan.builder_clobbers, plan.scc_temp));
}

// The SCC temp lives across the call, so it must avoid the probe body clobbers
// even when those registers are dead at the anchor. The target pair, consumed
// before the call, may overlap them.
TEST(TrampolineBuilderPlan, SccTempAvoidsProbeBodyClobbers) {
  TrampolinePlan plan;
  std::string err;
  // Dead SGPRs are {0,1,2,3,4}: the target pair takes s[0:1], and {2,3} are
  // probe-clobbered. The SCC temp must skip the dead-but-clobbered {2,3} and land
  // on s4, the only dead SGPR that survives the call.
  RegisterSet live = all_sgprs_live_except({0, 1, 2, 3, 4, 30, 31});
  RegisterSet probe_clobbers = make_sgpr_set({2, 3});
  ASSERT_TRUE(TrampolineBuilder::plan_probe_call(plan, kNoArgsCc, live, probe_clobbers, &err));
  EXPECT_EQ(plan.target_pair_base, 0u);
  EXPECT_EQ(plan.scc_temp, 4u);
}

// Word count is derived from the chosen envelope: getpc(1) + add/addc with
// literals(4) + swappc(1) = 6, plus SCC save/restore(2) when preserving SCC.
TEST(TrampolineBuilderPlan, BeforeWordCountReflectsEnvelope) {
  TrampolinePlan with_scc;
  std::string err;
  ASSERT_TRUE(TrampolineBuilder::plan_probe_call(with_scc, kNoArgsCc, make_sgpr_set({4}),
                                                 /*probe_body_clobbers=*/{}, &err));
  EXPECT_TRUE(with_scc.preserve_scc);
  EXPECT_EQ(with_scc.before_word_count, 8u);

  TrampolinePlan no_scc;
  no_scc.preserve_scc = false;
  ASSERT_TRUE(TrampolineBuilder::plan_probe_call(no_scc, kNoArgsCc, make_sgpr_set({4}),
                                                 /*probe_body_clobbers=*/{}, &err));
  EXPECT_EQ(no_scc.before_word_count, 6u);
}

// An unknown calling convention has no link pair, so planning fails closed.
TEST(TrampolineBuilderPlan, UnknownCcFails) {
  TrampolinePlan plan;
  std::string err;
  EXPECT_FALSE(TrampolineBuilder::plan_probe_call(plan, ProbeCallingConvention::Unknown,
                                                  /*live_at_anchor=*/{}, /*probe_body_clobbers=*/{},
                                                  &err));
  EXPECT_NE(err.find("calling convention"), std::string::npos);
  EXPECT_FALSE(plan.is_probe_call);
}

//==============================================================================
// Probe-call emission (emit_probe_call)
//
// Plans, then lowers, a probe call and checks the emitted trampoline words:
// the target-address materialization, the call through the cc-derived link
// pair, the single relocated original, and the return branch.
//==============================================================================

// SOP1 field decoders (matching pack_sop1 in instruction_builder.h).
uint16_t decode_sop1_op(uint32_t word) { return static_cast<uint16_t>((word >> 8) & 0xFFu); }
uint16_t decode_sop1_sdst(uint32_t word) { return static_cast<uint16_t>((word >> 16) & 0x7Fu); }
uint16_t decode_sop1_ssrc0(uint32_t word) { return static_cast<uint16_t>(word & 0xFFu); }

// A valid probe-call plan over a gfx90a/CDNA2 layout. Resources are planned with
// only s4 live so the envelope picks low, dead SGPRs.
TrampolinePlan make_probe_plan(rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA2) {
  TrampolinePlan plan;
  plan.arch = arch;
  plan.anchor_offset = 0x1000;
  plan.original_size = 4;
  plan.original_words.assign(1, 0xDEADBEEFu);
  plan.trampoline_offset = 0x2000;
  plan.return_target = plan.anchor_offset + plan.original_size;
  plan.probe_target_offset = 0x3000;
  std::string err;
  EXPECT_TRUE(TrampolineBuilder::plan_probe_call(plan, kNoArgsCc, make_sgpr_set({4}),
                                                 /*probe_body_clobbers=*/{}, &err))
      << err;
  return plan;
}

// The envelope materializes the target address with getpc + add + addc.
TEST(TrampolineBuilderEmit, ContainsTargetMaterialization) {
  const TrampolinePlan plan = make_probe_plan();
  std::string err;
  const auto bytes = TrampolineBuilder::emit_probe_call(plan, &err);
  ASSERT_TRUE(bytes.has_value()) << err;

  const std::vector<uint32_t> &w = bytes->trampoline_words;
  // preserve_scc default: [cselect, getpc, add (2 words), addc (2 words), swappc, cmp_lg, original,
  // branch].
  EXPECT_EQ(w[1], build_s_getpc_b64(plan.target_pair_base, plan.arch));
  EXPECT_EQ(w[2], build_s_add_u32(plan.target_pair_base, plan.target_pair_base, 0xFF, plan.arch));
  EXPECT_EQ(w[4], build_s_addc_u32(plan.target_pair_base + 1, plan.target_pair_base + 1, 0xFF,
                                   plan.arch));
}

// The envelope calls the probe via s_swappc_b64 through the cc-derived link pair,
// from the chosen target pair.
TEST(TrampolineBuilderEmit, SwappcUsesCcLinkPairAndTargetPair) {
  const TrampolinePlan plan = make_probe_plan();
  std::string err;
  const auto bytes = TrampolineBuilder::emit_probe_call(plan, &err);
  ASSERT_TRUE(bytes.has_value()) << err;

  // The swappc precedes the SCC restore, which precedes the relocated original.
  const std::vector<uint32_t> &w = bytes->trampoline_words;
  const uint32_t swappc = w[6];
  EXPECT_EQ(decode_sop1_op(swappc), sop1_op_swappc_b64(plan.arch));

  // sdst is the link pair; it must equal the pair link_pair_for(cc) reports, the
  // same pair the probe's s_setpc_b64 returns through.
  const std::optional<uint16_t> cc_link = link_pair_for(kNoArgsCc);
  ASSERT_TRUE(cc_link.has_value());
  EXPECT_EQ(decode_sop1_sdst(swappc), *cc_link);
  EXPECT_EQ(decode_sop1_sdst(swappc), plan.link_pair_base);
  // ssrc0 is the materialized target pair.
  EXPECT_EQ(decode_sop1_ssrc0(swappc), plan.target_pair_base);
}

// The relocated original appears exactly once, after the call.
TEST(TrampolineBuilderEmit, OriginalAppearsOnceAfterCall) {
  const TrampolinePlan plan = make_probe_plan();
  std::string err;
  const auto bytes = TrampolineBuilder::emit_probe_call(plan, &err);
  ASSERT_TRUE(bytes.has_value()) << err;

  const std::vector<uint32_t> &w = bytes->trampoline_words;
  const auto first = std::find(w.begin(), w.end(), plan.original_words[0]);
  ASSERT_NE(first, w.end());
  // Exactly one occurrence.
  EXPECT_EQ(std::count(w.begin(), w.end(), plan.original_words[0]), 1);
  // It sits after the swappc (the call is at index before_word_count - 1 when
  // SCC is preserved, but the original is always after the whole envelope).
  const size_t original_index = static_cast<size_t>(first - w.begin());
  EXPECT_EQ(original_index, plan.before_word_count);
}

// The return branch (trailing word) targets anchor + original_size.
TEST(TrampolineBuilderEmit, ReturnBranchTargetsAnchorPlusOriginalSize) {
  const TrampolinePlan plan = make_probe_plan();
  std::string err;
  const auto bytes = TrampolineBuilder::emit_probe_call(plan, &err);
  ASSERT_TRUE(bytes.has_value()) << err;

  const std::vector<uint32_t> &w = bytes->trampoline_words;
  const uint32_t return_branch = w.back();
  const uint64_t return_branch_pc = plan.trampoline_offset + (w.size() - 1) * sizeof(uint32_t);
  EXPECT_EQ(resolve_sopp_target(return_branch_pc, return_branch),
            plan.anchor_offset + plan.original_size);
}

// Dropping SCC preservation removes the cselect/cmp_lg pair (6 envelope words).
TEST(TrampolineBuilderEmit, NoSccPreserveShrinksEnvelope) {
  TrampolinePlan plan = make_probe_plan();
  // Re-plan without SCC preservation so before_word_count is consistent.
  std::string err;
  plan.preserve_scc = false;
  ASSERT_TRUE(TrampolineBuilder::plan_probe_call(plan, kNoArgsCc, make_sgpr_set({4}),
                                                 /*probe_body_clobbers=*/{}, &err));
  ASSERT_EQ(plan.before_word_count, 6u);

  const auto bytes = TrampolineBuilder::emit_probe_call(plan, &err);
  ASSERT_TRUE(bytes.has_value()) << err;

  // getpc is now the first word; swappc is the last envelope word.
  const std::vector<uint32_t> &w = bytes->trampoline_words;
  EXPECT_EQ(decode_sop1_op(w[0]), sop1_op_getpc_b64(plan.arch));
  EXPECT_EQ(decode_sop1_op(w[5]), sop1_op_swappc_b64(plan.arch));
  // Envelope (6) + original (1) + return branch (1).
  EXPECT_EQ(w.size(), plan.before_word_count + 1u + 1u);
}

// A plan that was never planned as a probe call cannot be emitted.
TEST(TrampolineBuilderEmit, RejectsNonProbeCallPlan) {
  TrampolinePlan plan;
  plan.arch = ROCJITSU_CODE_ARCH_CDNA2;
  plan.original_size = 4;
  plan.original_words.assign(1, 0xDEADBEEFu);
  std::string err;
  EXPECT_FALSE(TrampolineBuilder::emit_probe_call(plan, &err).has_value());
  EXPECT_NE(err.find("not a probe call"), std::string::npos);
}

// A planned word count that disagrees with the synthesized envelope fails closed.
TEST(TrampolineBuilderEmit, DetectsBeforeWordCountDrift) {
  TrampolinePlan plan = make_probe_plan();
  plan.before_word_count += 1; // Tamper after planning.
  std::string err;
  EXPECT_FALSE(TrampolineBuilder::emit_probe_call(plan, &err).has_value());
  EXPECT_NE(err.find("before_word_count"), std::string::npos);
}

// The forward (anchor -> trampoline) and return branch ranges are still checked
// via build(): an out-of-range trampoline placement is reported, not emitted.
TEST(TrampolineBuilderEmit, ForwardBranchRangeFailureReported) {
  TrampolinePlan plan = make_probe_plan();
  // Push the trampoline far past the anchor so the forward s_branch overflows.
  plan.trampoline_offset = plan.anchor_offset + (static_cast<uint64_t>(0x10000) * 4);
  std::string err;
  EXPECT_FALSE(TrampolineBuilder::emit_probe_call(plan, &err).has_value());
  EXPECT_NE(err.find("forward branch"), std::string::npos);
}

} // namespace
} // namespace rocjitsu
