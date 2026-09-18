// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/entry_prologue.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/register_set.h"

#include <gtest/gtest.h>

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

// The allocation bound is the kernel's own, not the cross-ISA allocatable
// maximum: a kernel that advertises more SGPRs than any target allocates must
// still not place storage past REGISTER_SET_ALLOCATABLE_SGPRS.
TEST(PlanDbiEntryStorage, ClampsTheBoundToTheAllocatableSgprMaximum) {
  const Kernel kernel(kernel_naming_sgpr(100));
  std::string error;
  const auto storage =
      plan_dbi_entry_storage(kernel.scope(), descriptor(/*user_sgpr_count=*/0), kArch,
                             /*kernel_sgpr_count=*/256, link_pair(), &error);
  EXPECT_FALSE(storage.has_value());
  EXPECT_FALSE(error.empty());
}

} // namespace
} // namespace rocjitsu
