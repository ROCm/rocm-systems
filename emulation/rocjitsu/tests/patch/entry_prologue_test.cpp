// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/smem_builders.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/entry_prologue.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/register_set.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

namespace kd = rocr::llvm::amdhsa;
using KD = kd::kernel_descriptor_t;

constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;

// The probe-call return-link pair. Passed as `reserved` so a run may sit below
// it; the planner must not treat it as a floor.
constexpr uint16_t kLinkPairBase = 30;

class TestTextSection : public Section {
public:
  TestTextSection(std::unique_ptr<char[]> data, std::size_t size)
      : Section(".text", std::move(data)), size_(size) {}

  std::size_t size() const override { return size_; }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return 0; }

private:
  std::size_t size_;
};

class TestCodeObject : public CodeObject {
public:
  explicit TestCodeObject(const std::vector<uint32_t> &words) {
    const auto byte_size = words.size() * sizeof(uint32_t);
    image_.resize(byte_size);
    std::memcpy(image_.data(), words.data(), byte_size);

    auto data = std::make_unique<char[]>(byte_size);
    std::memcpy(data.get(), words.data(), byte_size);
    sections_.push_back(std::make_unique<TestTextSection>(std::move(data), byte_size));
    text_sections_.push_back(sections_.back().get());
  }
};

/// Decoded blocks plus the objects they borrow from, so a scope outlives the
/// expression that built it.
class Kernel {
public:
  explicit Kernel(const std::vector<uint32_t> &words) : co_(words) {
    decoder_ = Decoder::create(kArch);
    if (decoder_ == nullptr) {
      ADD_FAILURE() << "no decoder for the test arch";
      return;
    }
    blocks_ = build_valid_blocks(co_, *decoder_, kArch);
    for (const auto &block : blocks_)
      scope_.push_back(block.get());
  }

  [[nodiscard]] KernelBlockScope scope() const { return KernelBlockScope(scope_); }

private:
  TestCodeObject co_;
  std::unique_ptr<Decoder> decoder_;
  std::vector<std::unique_ptr<BasicBlock>> blocks_;
  std::vector<BasicBlock *> scope_;
};

/// A descriptor that enables the kernarg segment pointer, placing it at s[0:1]
/// when no earlier user-SGPR property is set.
[[nodiscard]] KD kernarg_descriptor(uint32_t kernarg_size = 0) {
  KD desc{};
  set_kernel_descriptor_user_sgpr_count(kArch, desc, 2);
  AMDHSA_BITS_SET(desc.kernel_code_properties,
                  kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
  desc.kernarg_size = kernarg_size;
  return desc;
}

/// A descriptor whose kernarg pointer follows an earlier user-SGPR property, so
/// the pair lands above s[0:1].
[[nodiscard]] KD kernarg_descriptor_at_slot(uint16_t expected_slot) {
  KD desc = kernarg_descriptor();
  AMDHSA_BITS_SET(desc.kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR,
                  1);
  set_kernel_descriptor_user_sgpr_count(kArch, desc, 4);
  EXPECT_EQ(kernarg_segment_ptr_sgpr(desc).value_or(0xFFFF), expected_slot);
  return desc;
}

[[nodiscard]] KD descriptor(uint32_t user_sgpr_count, bool workgroup_id_x = false,
                            bool workgroup_id_y = false, bool workgroup_id_z = false) {
  KD desc{};
  set_kernel_descriptor_user_sgpr_count(kArch, desc, user_sgpr_count);
  // Braces are load-bearing: AMDHSA_BITS_SET expands to two statements with no
  // do-while wrapper, so an unbraced `if` would set the bit unconditionally. The
  // field also has to be named literally at each call, since the macro pastes
  // _SHIFT onto its argument.
  if (workgroup_id_x) {
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1);
  }
  if (workgroup_id_y) {
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y, 1);
  }
  if (workgroup_id_z) {
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z, 1);
  }
  return desc;
}

[[nodiscard]] RegisterSet link_pair() {
  RegisterSet set;
  set.expand({RegClass::SGPR, kLinkPairBase, 2});
  return set;
}

// Inline constant zero. Naming s0 as the source instead would put a second
// register in every fixture for no reason.
constexpr uint16_t kInlineZero = 128;

/// A kernel whose only explicitly named ordinary SGPR is `sgpr`.
[[nodiscard]] std::vector<uint32_t> kernel_naming_sgpr(uint16_t sgpr) {
  return {build_s_mov_b32(sgpr, kInlineZero, kArch), build_s_endpgm(kArch)};
}

// The floor is the higher of the two inputs. Here the kernel's own registers
// reach past its ABI block.
TEST(DbiEntryStorageFloor, TakesTheOperandScanWhenItExceedsTheAbiBlock) {
  const Kernel kernel(kernel_naming_sgpr(24));
  EXPECT_EQ(dbi_entry_storage_floor(kernel.scope(), descriptor(/*user_sgpr_count=*/0), kArch), 25u);
}

TEST(DbiEntryStorageFloor, FoldsInUserSgprsAndEnabledWorkgroupIds) {
  // Nothing named by the body, so the ABI block alone sets the floor: six user
  // SGPRs then two enabled workgroup dimensions, ending at s8.
  const Kernel kernel({build_s_endpgm(kArch)});
  const KD desc = descriptor(/*user_sgpr_count=*/6, /*workgroup_id_x=*/true,
                             /*workgroup_id_y=*/true, /*workgroup_id_z=*/false);
  EXPECT_EQ(dbi_entry_storage_floor(kernel.scope(), desc, kArch), 8u);
}

TEST(DbiEntryStorageFloor, DisabledWorkgroupDimensionsConsumeNoSgpr) {
  const Kernel kernel({build_s_endpgm(kArch)});
  const KD none = descriptor(/*user_sgpr_count=*/6);
  const KD all = descriptor(/*user_sgpr_count=*/6, /*workgroup_id_x=*/true,
                            /*workgroup_id_y=*/true, /*workgroup_id_z=*/true);
  EXPECT_EQ(dbi_entry_storage_floor(kernel.scope(), none, kArch), 6u);
  EXPECT_EQ(dbi_entry_storage_floor(kernel.scope(), all, kArch), 9u);
}

TEST(PlanDbiEntryStorage, PlacesTheRunAtTheFirstAlignedIndexAboveTheFloor) {
  const Kernel kernel(kernel_naming_sgpr(24));
  const auto storage = plan_dbi_entry_storage(kernel.scope(), descriptor(/*user_sgpr_count=*/0),
                                              kArch, /*kernel_sgpr_count=*/40, link_pair());
  ASSERT_TRUE(storage.has_value());
  EXPECT_EQ(storage->persistent_base, 26u);
  EXPECT_EQ(storage->entry_temp_base, 28u);
}

// The floor is 25, so the run could start at 26 and the link pair sits above it
// untouched. Treating s[30:31] as a floor instead would push the run to 32 and
// need eight more SGPRs than the kernel has reason to allocate.
TEST(PlanDbiEntryStorage, PlacesTheRunBelowTheReservedLinkPair) {
  const Kernel kernel(kernel_naming_sgpr(24));
  const auto storage = plan_dbi_entry_storage(kernel.scope(), descriptor(/*user_sgpr_count=*/0),
                                              kArch, /*kernel_sgpr_count=*/32, link_pair());
  ASSERT_TRUE(storage.has_value());
  EXPECT_EQ(storage->persistent_base, 26u);
  EXPECT_EQ(storage->entry_temp_base, 28u);
}

// A floor of 29 aligns to 30, where the run would cover the link pair. The
// planner must step past it rather than hand back overlapping storage.
TEST(PlanDbiEntryStorage, StepsPastAReservedPairTheRunWouldCover) {
  const Kernel kernel(kernel_naming_sgpr(28));
  const auto storage = plan_dbi_entry_storage(kernel.scope(), descriptor(/*user_sgpr_count=*/0),
                                              kArch, /*kernel_sgpr_count=*/40, link_pair());
  ASSERT_TRUE(storage.has_value());
  EXPECT_EQ(storage->persistent_base, 32u);
  EXPECT_EQ(storage->entry_temp_base, 34u);
}

TEST(PlanDbiEntryStorage, FailsClosedWhenTheAllocationCannotHoldTheRun) {
  const Kernel kernel(kernel_naming_sgpr(28));
  std::string error;
  const auto storage =
      plan_dbi_entry_storage(kernel.scope(), descriptor(/*user_sgpr_count=*/0), kArch,
                             /*kernel_sgpr_count=*/34, link_pair(), &error);
  EXPECT_FALSE(storage.has_value());
  EXPECT_FALSE(error.empty());
}

TEST(PlanDbiEntryStorage, FailsClosedWhenTheFloorIsAtOrPastTheAllocation) {
  const Kernel kernel(kernel_naming_sgpr(33));
  std::string error;
  const auto storage =
      plan_dbi_entry_storage(kernel.scope(), descriptor(/*user_sgpr_count=*/0), kArch,
                             /*kernel_sgpr_count=*/32, link_pair(), &error);
  EXPECT_FALSE(storage.has_value());
  EXPECT_FALSE(error.empty());
}

// The bound is the lower of the kernel's own allocation and the cross-ISA
// allocatable maximum, so a kernel advertising more SGPRs than any target
// allocates still cannot place storage past REGISTER_SET_ALLOCATABLE_SGPRS.
TEST(PlanDbiEntryStorage, ClampsTheBoundToTheAllocatableSgprMaximum) {
  const Kernel kernel(kernel_naming_sgpr(100));
  std::string error;
  const auto storage =
      plan_dbi_entry_storage(kernel.scope(), descriptor(/*user_sgpr_count=*/0), kArch,
                             /*kernel_sgpr_count=*/256, link_pair(), &error);
  EXPECT_FALSE(storage.has_value());
  EXPECT_FALSE(error.empty());
}

// The prologue is two loads through the kernarg pair, one wait, then the two
// moves that restore the guest's pointer over it. Asserted against the same
// builders rather than against hand-computed words, so this pins the order and
// the operands; smem_builder_test pins the encodings.
TEST(BuildDbiEntryPrologue, EmitsLoadsThenWaitThenRestore) {
  const KD desc = kernarg_descriptor(/*kernarg_size=*/16);
  constexpr DbiEntryStorage kStorage{.persistent_base = 26, .entry_temp_base = 28};

  std::string error;
  const auto prologue = build_dbi_entry_prologue(desc, kArch, kStorage, &error);
  ASSERT_TRUE(prologue.has_value()) << error;

  const auto payload =
      build_s_load_dwordx2(kStorage.persistent_base, 0, prologue->payload_byte_offset, kArch);
  const auto original = build_s_load_dwordx2(kStorage.entry_temp_base, 0,
                                             prologue->original_kernarg_pointer_offset, kArch);
  const std::vector<uint32_t> expected{
      payload[0],
      payload[1],
      original[0],
      original[1],
      build_wait_scalar_loads_complete(kArch),
      build_s_mov_b32(0, kStorage.entry_temp_base, kArch),
      build_s_mov_b32(1, static_cast<uint16_t>(kStorage.entry_temp_base + 1), kArch),
  };
  EXPECT_EQ(prologue->words, expected);
}

// Every other test here, and every fixture the simulator suites build, puts the
// kernarg pair at s[0:1]. At slot 0 the SMEM SBASE field's halving is the
// identity and base+1 is trivially 1, so a regression that dropped the halving,
// applied it twice, or confused the slot with a register index would be
// invisible. One descriptor with an earlier user-SGPR property closes that.
TEST(BuildDbiEntryPrologue, AddressesThroughTheKernargPairWhereverTheAbiPutsIt) {
  const KD desc = kernarg_descriptor_at_slot(2);
  constexpr DbiEntryStorage kStorage{.persistent_base = 26, .entry_temp_base = 28};

  std::string error;
  const auto prologue = build_dbi_entry_prologue(desc, kArch, kStorage, &error);
  ASSERT_TRUE(prologue.has_value()) << error;

  const auto payload = build_s_load_dwordx2(kStorage.persistent_base, /*sbase=*/2,
                                            prologue->payload_byte_offset, kArch);
  const auto original = build_s_load_dwordx2(kStorage.entry_temp_base, /*sbase=*/2,
                                             prologue->original_kernarg_pointer_offset, kArch);
  const std::vector<uint32_t> expected{
      payload[0],
      payload[1],
      original[0],
      original[1],
      build_wait_scalar_loads_complete(kArch),
      build_s_mov_b32(2, kStorage.entry_temp_base, kArch),
      build_s_mov_b32(3, static_cast<uint16_t>(kStorage.entry_temp_base + 1), kArch),
  };
  EXPECT_EQ(prologue->words, expected);
}

// The two offsets must differ, and the payload must sit past the copied kernarg
// prefix. Equal offsets would load the same pointer twice and leave the payload
// unread; a payload inside the prefix would alias a guest kernarg.
TEST(BuildDbiEntryPrologue, PlacesThePayloadPastTheCopiedKernargPrefix) {
  constexpr uint32_t kKernargSize = 24;
  const auto prologue = build_dbi_entry_prologue(kernarg_descriptor(kKernargSize), kArch,
                                                 {.persistent_base = 26, .entry_temp_base = 28});
  ASSERT_TRUE(prologue.has_value());
  EXPECT_GE(prologue->original_kernarg_pointer_offset, kKernargSize);
  EXPECT_GT(prologue->payload_byte_offset, prologue->original_kernarg_pointer_offset);
  EXPECT_EQ(prologue->payload_byte_offset % kDbiEntryPayloadLayout.alignment, 0u);
}

// The reported offsets are read out of the emitted words, not from a second call
// to the layout helper, so a consumer re-deriving the layout is checking the
// prologue rather than the helper against itself.
TEST(BuildDbiEntryPrologue, ReportsTheOffsetsItEncoded) {
  const auto prologue = build_dbi_entry_prologue(kernarg_descriptor(/*kernarg_size=*/16), kArch,
                                                 {.persistent_base = 26, .entry_temp_base = 28});
  ASSERT_TRUE(prologue.has_value());
  ASSERT_GE(prologue->words.size(), 4u);

  auto decoder = Decoder::create(kArch);
  ASSERT_NE(decoder, nullptr);
  const auto offset_of = [&](size_t word_index) {
    std::array<rj_code_binary_inst_t, 2> words{prologue->words[word_index],
                                               prologue->words[word_index + 1]};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    EXPECT_NE(inst, nullptr);
    return inst == nullptr ? -1 : inst->src_operand(1)->encoding_value();
  };
  EXPECT_EQ(offset_of(0), static_cast<int>(prologue->payload_byte_offset));
  EXPECT_EQ(offset_of(2), static_cast<int>(prologue->original_kernarg_pointer_offset));
}

TEST(BuildDbiEntryPrologue, FailsClosedWithoutAKernargSegmentPointer) {
  std::string error;
  const auto prologue =
      build_dbi_entry_prologue(descriptor(/*user_sgpr_count=*/0), kArch,
                               {.persistent_base = 26, .entry_temp_base = 28}, &error);
  EXPECT_FALSE(prologue.has_value());
  EXPECT_FALSE(error.empty());
}

// Naming one pair twice makes the second load overwrite the payload pointer with
// the guest's. The value-initialized case is also caught by the kernarg-overlap
// check; the second case needs this check specifically.
TEST(BuildDbiEntryPrologue, FailsClosedOnStoragePairsThatAreNotOneRun) {
  const KD desc = kernarg_descriptor();
  std::string error;
  EXPECT_FALSE(build_dbi_entry_prologue(desc, kArch, DbiEntryStorage{}, &error).has_value());
  EXPECT_FALSE(error.empty());

  // Same pair named twice, clear of the kernarg pair, so only the run check can
  // reject it.
  EXPECT_FALSE(build_dbi_entry_prologue(desc, kArch, {.persistent_base = 26, .entry_temp_base = 26})
                   .has_value());
  // Right registers, wrong order.
  EXPECT_FALSE(build_dbi_entry_prologue(desc, kArch, {.persistent_base = 28, .entry_temp_base = 26})
                   .has_value());
  // Contiguous but not adjacent.
  EXPECT_FALSE(build_dbi_entry_prologue(desc, kArch, {.persistent_base = 26, .entry_temp_base = 30})
                   .has_value());
}

// Storage overlapping the kernarg pair would have the restore overwrite the
// payload pointer, or a load corrupt its own address.
TEST(BuildDbiEntryPrologue, FailsClosedOnStorageOverlappingTheKernargPair) {
  std::string error;
  // s[0:3] covers the kernarg pair at s[0:1] with its persistent half.
  EXPECT_FALSE(build_dbi_entry_prologue(kernarg_descriptor(), kArch,
                                        {.persistent_base = 0, .entry_temp_base = 2}, &error)
                   .has_value());
  EXPECT_FALSE(error.empty());
  // A kernarg pair inside the run's temp half is equally fatal, and is what a
  // per-pair check on the persistent half alone would miss.
  EXPECT_FALSE(build_dbi_entry_prologue(kernarg_descriptor_at_slot(2), kArch,
                                        {.persistent_base = 0, .entry_temp_base = 2})
                   .has_value());
}

// Not reachable from plan_dbi_entry_storage, which bounds the run. This guards
// the hand-built path.
TEST(BuildDbiEntryPrologue, FailsClosedOnStoragePastTheSdataField) {
  std::string error;
  EXPECT_FALSE(build_dbi_entry_prologue(
                   kernarg_descriptor(), kArch,
                   {.persistent_base = kMaxSmemSdata + 1, .entry_temp_base = kMaxSmemSdata + 3},
                   &error)
                   .has_value());
  EXPECT_FALSE(error.empty());
}

TEST(BuildDbiEntryPrologue, FailsClosedOnAKernargSizeThatOverflowsTheWrapper) {
  std::string error;
  EXPECT_FALSE(build_dbi_entry_prologue(kernarg_descriptor(0xFFFFFFF8u), kArch,
                                        {.persistent_base = 26, .entry_temp_base = 28}, &error)
                   .has_value());
  EXPECT_FALSE(error.empty());
}

TEST(BuildDbiEntryPrologue, FailsClosedOnAnOddStoragePair) {
  std::string error;
  EXPECT_FALSE(build_dbi_entry_prologue(kernarg_descriptor(), kArch,
                                        {.persistent_base = 27, .entry_temp_base = 29}, &error)
                   .has_value());
  EXPECT_FALSE(error.empty());
}

// A kernel with more kernarg bytes than the SMEM immediate can reach pushes the
// payload out of range. The field is signed, so the limit is one bit below the
// field width.
TEST(BuildDbiEntryPrologue, FailsClosedOnAWrapperPastTheSmemImmediateRange) {
  std::string error;
  const auto prologue =
      build_dbi_entry_prologue(kernarg_descriptor(max_smem_byte_offset(kArch)), kArch,
                               {.persistent_base = 26, .entry_temp_base = 28}, &error);
  EXPECT_FALSE(prologue.has_value());
  EXPECT_FALSE(error.empty());
}

// The composed planner hands back storage choice and words in one answer.
TEST(PlanDbiEntryPrologue, ReturnsTheStorageAndTheWordsItEncoded) {
  const Kernel kernel(kernel_naming_sgpr(4));
  std::string error;
  const auto plan =
      plan_dbi_entry_prologue(kernel.scope(), kernarg_descriptor(/*kernarg_size=*/16), kArch,
                              /*kernel_sgpr_count=*/32, link_pair(), &error);
  ASSERT_TRUE(plan.has_value()) << error;

  // Floor 5 (the kernel's s4) aligns to 6, and the run is contiguous.
  EXPECT_EQ(plan->storage.persistent_base, 6u);
  EXPECT_EQ(plan->storage.entry_temp_base, 8u);

  // The words are build_dbi_entry_prologue's, not a second encoding of them.
  const auto direct = build_dbi_entry_prologue(kernarg_descriptor(/*kernarg_size=*/16), kArch,
                                               plan->storage, &error);
  ASSERT_TRUE(direct.has_value()) << error;
  EXPECT_EQ(plan->prologue.words, direct->words);
  EXPECT_EQ(plan->prologue.payload_byte_offset, direct->payload_byte_offset);
  EXPECT_EQ(plan->prologue.original_kernarg_pointer_offset,
            direct->original_kernarg_pointer_offset);
}

// A preloading kernel has a second hardware entry 256 bytes past the
// descriptor's, which a prologue at the descriptor entry would not cover.
TEST(PlanDbiEntryPrologue, FailsClosedOnAKernargPreloadingKernel) {
  const Kernel kernel(kernel_naming_sgpr(4));
  KD desc = kernarg_descriptor(/*kernarg_size=*/16);
  AMDHSA_BITS_SET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 1);

  std::string error;
  const auto plan = plan_dbi_entry_prologue(kernel.scope(), desc, kArch,
                                            /*kernel_sgpr_count=*/32, link_pair(), &error);
  EXPECT_FALSE(plan.has_value());
  EXPECT_NE(error.find("preloads"), std::string::npos) << error;
}

// The reserved set reaches storage selection rather than being dropped on the
// way through. Without it the run would start at s6.
TEST(PlanDbiEntryPrologue, StepsPastTheReservedRegisters) {
  const Kernel kernel(kernel_naming_sgpr(4));
  RegisterSet reserved = link_pair();
  reserved.expand({RegClass::SGPR, 6, 4});

  std::string error;
  const auto plan =
      plan_dbi_entry_prologue(kernel.scope(), kernarg_descriptor(/*kernarg_size=*/16), kArch,
                              /*kernel_sgpr_count=*/32, reserved, &error);
  ASSERT_TRUE(plan.has_value()) << error;
  EXPECT_EQ(plan->storage.persistent_base, 10u);
  EXPECT_EQ(plan->storage.entry_temp_base, 12u);
}

// Either composed step failing fails the planner, not a partially-filled plan.
TEST(PlanDbiEntryPrologue, PropagatesStorageAndWordFailures) {
  const Kernel kernel(kernel_naming_sgpr(4));
  std::string error;

  // No room for the run inside the allocation.
  EXPECT_FALSE(plan_dbi_entry_prologue(kernel.scope(), kernarg_descriptor(), kArch,
                                       /*kernel_sgpr_count=*/6, link_pair(), &error)
                   .has_value());
  EXPECT_FALSE(error.empty());

  // Storage is available, but the descriptor has no kernarg pointer to load through.
  error.clear();
  EXPECT_FALSE(plan_dbi_entry_prologue(kernel.scope(), descriptor(/*user_sgpr_count=*/2), kArch,
                                       /*kernel_sgpr_count=*/32, link_pair(), &error)
                   .has_value());
  EXPECT_FALSE(error.empty());
}

} // namespace
} // namespace rocjitsu
