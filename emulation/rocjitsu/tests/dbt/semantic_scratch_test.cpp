// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/analysis/exec_state.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/dbt/semantic/cdna3_scratch.h"
#include "rocjitsu/code/dbt/semantic_scratch.h"
#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <span>
#include <vector>

namespace rocjitsu {
namespace {

// TODO: Consolidate this in-memory code-object fixture with the equivalent
// liveness and spill-manager test scaffolding.
class ScratchTestTextSection : public Section {
public:
  ScratchTestTextSection(std::unique_ptr<char[]> data, std::size_t size)
      : Section(".text", std::move(data)), size_(size) {}

  std::size_t size() const override { return size_; }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return 0; }

private:
  std::size_t size_;
};

class ScratchTestCodeObject : public CodeObject {
public:
  explicit ScratchTestCodeObject(std::span<const uint32_t> words) {
    const auto byte_size = words.size_bytes();
    image_.resize(byte_size);
    std::memcpy(image_.data(), words.data(), byte_size);

    auto data = std::make_unique<char[]>(byte_size);
    std::memcpy(data.get(), words.data(), byte_size);
    sections_.push_back(std::make_unique<ScratchTestTextSection>(std::move(data), byte_size));
    text_sections_.push_back(sections_.back().get());
  }
};

std::vector<std::unique_ptr<BasicBlock>> build_scratch_test_blocks() {
  constexpr auto move = cdna3::build_vop1(cdna3::kVMovB32Vop1, {.src0 = 256, .vdst = 0});
  constexpr auto end = cdna3::build_sopp(cdna3::kSEndpgmSopp);
  constexpr std::array words = {move[0], end[0]};
  ScratchTestCodeObject code(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  return build_valid_blocks(code, *decoder, ROCJITSU_CODE_ARCH_CDNA3);
}

std::vector<BasicBlock *>
scratch_test_scope(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  std::vector<BasicBlock *> scope;
  scope.reserve(blocks.size());
  for (const auto &block : blocks)
    scope.push_back(block.get());
  return scope;
}

std::span<const uint8_t> scratch_test_text(const CodeObject &code) {
  const Section *text = code.text_sections().front();
  return {reinterpret_cast<const uint8_t *>(text->data()), text->size()};
}

TEST(SemanticSpillFrame, SeparatesPersistentAndTransientRanges) {
  TranslationContext context(/*vgprs=*/8, /*agprs=*/0, /*accum_base=*/0,
                             /*sgprs=*/8, /*private_bytes=*/20);
  const auto persistent = context.reserve_persistent_semantic_spill_dwords(2);
  ASSERT_TRUE(persistent);
  ASSERT_EQ(*persistent, 32u);

  SemanticSpillFrame frame(context);
  const auto first = frame.allocate_dwords(3, /*byte_alignment=*/4);
  const auto second = frame.allocate_dwords(2, /*byte_alignment=*/8);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);

  // Persistent storage ends at byte 40 and the transient frame is aligned to
  // byte 48. The second allocation follows the first rather than reusing it.
  EXPECT_EQ(first->byte_offset, 48u);
  EXPECT_EQ(second->byte_offset, 64u);
  EXPECT_EQ(context.required_private_segment_fixed_size, 72u);
}

TEST(SemanticSpillFrame, PersistentReservationRejects32BitOverflow) {
  // A guest kernel can advertise a private size near UINT32_MAX. Aligning that up
  // and extending it for a spill must fail closed rather than wrap to a low
  // offset that would corrupt guest scratch.
  TranslationContext context(/*vgprs=*/8, /*agprs=*/0, /*accum_base=*/0,
                             /*sgprs=*/8, /*private_bytes=*/0xfffffff8u);
  const auto persistent = context.reserve_persistent_semantic_spill_dwords(2);
  EXPECT_FALSE(persistent);
}

TEST(SemanticSpillFrame, TransientReservationRejects32BitOverflow) {
  TranslationContext context(/*vgprs=*/8, /*agprs=*/0, /*accum_base=*/0,
                             /*sgprs=*/8, /*private_bytes=*/0xfffffff8u);
  const auto transient = context.reserve_semantic_spill_dwords(2);
  EXPECT_FALSE(transient);
}

TEST(SemanticSpillFrame, NewInstructionsReuseTransientFrameBase) {
  TranslationContext context(/*vgprs=*/8, /*agprs=*/0, /*accum_base=*/0,
                             /*sgprs=*/8, /*private_bytes=*/12);
  SemanticSpillFrame first_instruction(context);
  const auto wide = first_instruction.allocate_dwords(4, /*byte_alignment=*/4);
  ASSERT_TRUE(wide);
  EXPECT_EQ(wide->byte_offset, 16u);
  EXPECT_EQ(context.required_private_segment_fixed_size, 32u);

  SemanticSpillFrame second_instruction(context);
  const auto narrow = second_instruction.allocate_dwords(1, /*byte_alignment=*/4);
  ASSERT_TRUE(narrow);
  EXPECT_EQ(narrow->byte_offset, 16u);
  EXPECT_EQ(context.required_private_segment_fixed_size, 32u);
}

TEST(SemanticScratchAllocator, FallsBackToNonForbiddenSpillVictim) {
  // An instruction absent from the liveness snapshot has no proven-dead
  // registers, which deliberately drives the allocator through its spill tier.
  Instruction inst("scratch_test", nullptr);
  std::vector<BasicBlock *> blocks;
  const ExecMaskAnalysis exec(KernelBlockScope(blocks), /*wave_size=*/64);
  LivenessAnalysis liveness(blocks, std::make_unique<ExecMaskAnalysis>(exec));
  TranslationContext context(/*vgprs=*/8, /*agprs=*/0, /*accum_base=*/0,
                             /*sgprs=*/8, /*private_bytes=*/20);
  SemanticScratchAllocator allocator(inst, liveness, context,
                                     Cdna3ScratchEmitter::allocation_policy());

  SemanticScratchRequest request;
  request.count = 2;
  request.alignment = 2;
  request.forbidden.expand({RegClass::VGPR, 0, 2});
  request.preferred_victim_base = 0;
  const SemanticScratchResult result = allocator.acquire_vgprs(request);

  ASSERT_TRUE(result);
  ASSERT_TRUE(result.lease->spilled);
  EXPECT_EQ(result.lease->base, 2u);
  EXPECT_EQ(result.lease->spill_offset, 32u);
  EXPECT_EQ(context.required_private_segment_fixed_size, 40u);
}

TEST(SemanticScratchAllocator, PrefersKernelUnusedOverSiteDeadVgpr) {
  auto blocks = build_scratch_test_blocks();
  auto scope = scratch_test_scope(blocks);
  const ExecMaskAnalysis exec{KernelBlockScope(scope), /*wave_size=*/64};
  const LivenessAnalysis liveness{KernelBlockScope(scope),
                                  std::make_unique<ExecMaskAnalysis>(exec)};
  auto site = blocks.front()->instructions().begin();
  ++site;
  ASSERT_NE(site, blocks.front()->instructions().end());

  TranslationContext context(/*vgprs=*/8, /*agprs=*/0, /*accum_base=*/0,
                             /*sgprs=*/8, /*private_bytes=*/64);
  SemanticScratchAllocator allocator(*site, liveness, context,
                                     Cdna3ScratchEmitter::allocation_policy());
  SemanticScratchRequest request;
  request.count = 1;
  const SemanticScratchResult result = allocator.acquire_vgprs(request);

  ASSERT_TRUE(result);
  EXPECT_FALSE(result.lease->spilled);
  EXPECT_EQ(result.lease->base, 1u);
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness())
      << "kernel-unused allocation must not force backward CFG liveness";
}

TEST(SemanticScratchAllocator, KernelUnusedSearchSkipsForbiddenWindow) {
  auto blocks = build_scratch_test_blocks();
  auto scope = scratch_test_scope(blocks);
  const ExecMaskAnalysis exec{KernelBlockScope(scope), /*wave_size=*/64};
  const LivenessAnalysis liveness{KernelBlockScope(scope),
                                  std::make_unique<ExecMaskAnalysis>(exec)};
  auto site = blocks.front()->instructions().begin();
  ++site;
  ASSERT_NE(site, blocks.front()->instructions().end());

  TranslationContext context(/*vgprs=*/8, /*agprs=*/0, /*accum_base=*/0,
                             /*sgprs=*/8, /*private_bytes=*/64);
  SemanticScratchAllocator allocator(*site, liveness, context,
                                     Cdna3ScratchEmitter::allocation_policy());
  SemanticScratchRequest request;
  request.count = 1;
  request.forbidden.expand({RegClass::VGPR, 1, 3});
  const SemanticScratchResult result = allocator.acquire_vgprs(request);

  ASSERT_TRUE(result);
  EXPECT_FALSE(result.lease->spilled);
  EXPECT_EQ(result.lease->base, 4u);
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
}

TEST(SemanticScratchAllocator, GprIndexModeWriteBypassesKernelUnusedTier) {
  constexpr uint16_t kModeGprIdxEnableHwreg = 1u | (27u << 6);
  constexpr auto setreg =
      cdna4::build_sopk(cdna4::kSSetregB32Sopk, {.simm16 = kModeGprIdxEnableHwreg, .sdst = 0});
  constexpr auto move = cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 256, .vdst = 0});
  constexpr auto end = cdna4::build_sopp(cdna4::kSEndpgmSopp);
  constexpr std::array words = {setreg[0], move[0], end[0]};
  ScratchTestCodeObject code(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(code, *decoder, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = scratch_test_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA4;
  options.text = scratch_test_text(code);
  const ExecMaskAnalysis exec{KernelBlockScope(scope), /*wave_size=*/64};
  const LivenessAnalysis liveness{KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                                  options};
  auto site = blocks.front()->instructions().begin();
  std::advance(site, 2);
  ASSERT_NE(site, blocks.front()->instructions().end());

  TranslationContext context(/*vgprs=*/2, /*agprs=*/0, /*accum_base=*/0,
                             /*sgprs=*/8, /*private_bytes=*/64);
  SemanticScratchAllocator allocator(*site, liveness, context,
                                     Cdna3ScratchEmitter::allocation_policy());
  SemanticScratchRequest request;
  request.count = 1;
  request.allow_spill = false;
  const SemanticScratchResult result = allocator.acquire_vgprs(request);

  ASSERT_TRUE(result);
  EXPECT_FALSE(result.lease->spilled);
  EXPECT_EQ(result.lease->base, 0u);
  EXPECT_TRUE(liveness.has_materialized_cfg_liveness())
      << "GPR indexing must bypass the kernel-unused tier";
}

TEST(SemanticScratchAllocator, SiteDeadTierCanGrowDescriptor) {
  auto blocks = build_scratch_test_blocks();
  auto scope = scratch_test_scope(blocks);
  const ExecMaskAnalysis exec{KernelBlockScope(scope), /*wave_size=*/64};
  const LivenessAnalysis liveness{KernelBlockScope(scope),
                                  std::make_unique<ExecMaskAnalysis>(exec)};
  const Instruction &site = *blocks.front()->instructions().begin();

  // The sole descriptor-allocated register v0 is read at this site, so the
  // kernel-unused tier has no in-descriptor choice and point liveness grows the
  // allocation to v1.
  TranslationContext context(/*vgprs=*/1, /*agprs=*/0, /*accum_base=*/0,
                             /*sgprs=*/8, /*private_bytes=*/64);
  SemanticScratchAllocator allocator(site, liveness, context,
                                     Cdna3ScratchEmitter::allocation_policy());
  SemanticScratchRequest request;
  request.count = 1;
  request.allow_spill = false;
  const SemanticScratchResult result = allocator.acquire_vgprs(request);

  ASSERT_TRUE(result);
  EXPECT_FALSE(result.lease->spilled);
  EXPECT_EQ(result.lease->base, 1u);
  EXPECT_EQ(context.required_vgpr_count, 2u);
  EXPECT_TRUE(liveness.has_materialized_cfg_liveness());
}

TEST(SemanticScratchAllocator, ReportsTargetSpillOffsetLimit) {
  Instruction inst("scratch_test", nullptr);
  std::vector<BasicBlock *> blocks;
  const ExecMaskAnalysis exec(KernelBlockScope(blocks), /*wave_size=*/64);
  LivenessAnalysis liveness(blocks, std::make_unique<ExecMaskAnalysis>(exec));
  TranslationContext context(/*vgprs=*/8, /*agprs=*/0, /*accum_base=*/0,
                             /*sgprs=*/8, /*private_bytes=*/32);
  SemanticScratchAllocator allocator(
      inst, liveness, context,
      SemanticScratchPolicy{.max_vgprs = 256, .max_spill_dword_offset = 31});

  SemanticScratchRequest request;
  request.count = 1;
  const SemanticScratchResult result = allocator.acquire_vgprs(request);

  EXPECT_FALSE(result);
  EXPECT_EQ(result.failure, SemanticScratchFailure::SpillOffsetUnencodable);
  EXPECT_EQ(context.required_private_segment_fixed_size, 32u);
}

TEST(SemanticScratchAllocator, RejectsSpillVictimInDynamicStackKernel) {
  Instruction inst("scratch_test", nullptr);
  std::vector<BasicBlock *> blocks;
  const ExecMaskAnalysis exec(KernelBlockScope(blocks), /*wave_size=*/64);
  LivenessAnalysis liveness(blocks, std::make_unique<ExecMaskAnalysis>(exec));
  // With no live-before snapshot, no register window is proven free, so the
  // allocator must borrow a victim and exercise the spill policy.
  ASSERT_FALSE(liveness.has_live_before(inst));
  TranslationContext context(/*vgprs=*/8, /*agprs=*/0, /*accum_base=*/0,
                             /*sgprs=*/8, /*private_bytes=*/32);
  context.uses_dynamic_stack = true;
  SemanticScratchAllocator allocator(
      inst, liveness, context, SemanticScratchPolicy{.max_vgprs = 8, .max_spill_dword_offset = 0});

  SemanticScratchRequest request;
  request.count = 1;
  const SemanticScratchResult result = allocator.acquire_vgprs(request);

  EXPECT_FALSE(result);
  EXPECT_EQ(result.failure, SemanticScratchFailure::DynamicStackUnsupported);
  EXPECT_EQ(context.required_private_segment_fixed_size, 32u);
}

TEST(SemanticScratchAllocator, AllowsFreeWindowInDynamicStackKernel) {
  auto blocks = build_scratch_test_blocks();
  std::vector<BasicBlock *> scope;
  for (const auto &block : blocks)
    scope.push_back(block.get());
  const Instruction &inst = *blocks.front()->instructions().begin();
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(scope, std::make_unique<ExecMaskAnalysis>(exec));
  TranslationContext context(/*vgprs=*/8, /*agprs=*/0, /*accum_base=*/0,
                             /*sgprs=*/8, /*private_bytes=*/32);
  context.uses_dynamic_stack = true;
  SemanticScratchAllocator allocator(
      inst, liveness, context, SemanticScratchPolicy{.max_vgprs = 8, .max_spill_dword_offset = 0});

  SemanticScratchRequest request;
  request.count = 1;
  const SemanticScratchResult result = allocator.acquire_vgprs(request);

  ASSERT_TRUE(result);
  EXPECT_EQ(result.failure, SemanticScratchFailure::None);
  EXPECT_FALSE(result.lease->spilled);
  EXPECT_EQ(result.lease->base, 1u);
  EXPECT_EQ(context.required_private_segment_fixed_size, 32u);
}

TEST(Cdna3ScratchEmitter, MaterializesSaveAndRestoreSequences) {
  const SemanticScratchLease lease{
      .reg_class = RegClass::VGPR, .base = 6, .count = 2, .spilled = true, .spill_offset = 48};
  std::vector<uint32_t> save;
  std::vector<uint32_t> restore;
  ASSERT_TRUE(Cdna3ScratchEmitter::append_save(save, lease));
  ASSERT_TRUE(Cdna3ScratchEmitter::append_restore(restore, lease));
  ASSERT_EQ(save.size(), 5u);
  ASSERT_EQ(restore.size(), 5u);

  cdna3::FlatScratchMachineInst first_save{};
  cdna3::FlatScratchMachineInst second_save{};
  cdna3::FlatScratchMachineInst first_restore{};
  std::memcpy(&first_save, save.data(), sizeof(first_save));
  std::memcpy(&second_save, save.data() + 2, sizeof(second_save));
  std::memcpy(&first_restore, restore.data(), sizeof(first_restore));

  EXPECT_EQ(first_save.op, 28u);
  EXPECT_EQ(first_save.data, 6u);
  EXPECT_EQ(first_save.offset, 48u);
  EXPECT_EQ(second_save.data, 7u);
  EXPECT_EQ(second_save.offset, 52u);
  EXPECT_EQ(first_restore.op, 20u);
  EXPECT_EQ(first_restore.vdst, 6u);
  EXPECT_EQ(first_restore.offset, 48u);
}

} // namespace
} // namespace rocjitsu
