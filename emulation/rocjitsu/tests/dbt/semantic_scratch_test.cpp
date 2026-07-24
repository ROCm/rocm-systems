// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/dbt/semantic/cdna3_scratch.h"
#include "rocjitsu/code/dbt/semantic_scratch.h"
#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace rocjitsu {
namespace {

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
  LivenessAnalysis liveness(blocks);
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

TEST(SemanticScratchAllocator, ReportsTargetSpillOffsetLimit) {
  Instruction inst("scratch_test", nullptr);
  std::vector<BasicBlock *> blocks;
  LivenessAnalysis liveness(blocks);
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
