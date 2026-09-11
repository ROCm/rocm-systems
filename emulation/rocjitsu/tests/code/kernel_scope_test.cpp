// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Coverage for the shared kernel reachability walk (code/kernel_scope.h), which
// DBT and DBI both use to decide which blocks belong to one kernel.

#include "decode_test_util.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/kernel_scope.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace rocjitsu {
namespace {

constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;

// s_cbranch_scc0 is SOPP opcode 5 on CDNA; simm16 is the dword delta from the
// next instruction.
[[nodiscard]] constexpr uint32_t build_s_cbranch_scc0(int16_t offset_dwords) {
  return pack_sopp(5, static_cast<uint16_t>(offset_dwords));
}

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
  explicit TestCodeObject(std::vector<uint32_t> words) {
    const auto byte_size = words.size() * sizeof(uint32_t);
    image_.resize(byte_size);
    std::memcpy(image_.data(), words.data(), byte_size);

    auto data = std::make_unique<char[]>(byte_size);
    std::memcpy(data.get(), words.data(), byte_size);
    sections_.push_back(std::make_unique<TestTextSection>(std::move(data), byte_size));
    text_sections_.push_back(sections_.back().get());
  }
};

// Two kernels in one .text. Kernel A branches into kernel B's entry, which is
// the edge the kernel_entries stop exists to cut.
//
//   0x00  A entry   s_cbranch_scc0 -> 0x08     (fallthrough 0x04)
//   0x04  A mid     s_branch       -> 0x10     (into B's entry)
//   0x08  A tail    s_nop
//   0x0c            s_endpgm
//   0x10  B entry   s_cbranch_scc0 -> 0x18     (fallthrough 0x14)
//   0x14  B mid     s_branch       -> 0x18
//   0x18  B tail    s_endpgm
constexpr uint64_t kEntryA = 0x00;
constexpr uint64_t kMidA = 0x04;
constexpr uint64_t kTailA = 0x08;
constexpr uint64_t kEntryB = 0x10;
constexpr uint64_t kMidB = 0x14;
constexpr uint64_t kTailB = 0x18;

[[nodiscard]] std::vector<uint32_t> two_kernel_words() {
  return {
      build_s_cbranch_scc0(1),  // 0x00: -> A tail at 0x08.
      build_s_branch(2, kArch), // 0x04: -> B entry at 0x10.
      build_s_nop(0, kArch),    // 0x08: A tail, first of two instructions.
      build_s_endpgm(kArch),    // 0x0c.
      build_s_cbranch_scc0(1),  // 0x10: -> B tail at 0x18.
      build_s_branch(0, kArch), // 0x14: -> B tail at 0x18.
      build_s_endpgm(kArch),    // 0x18.
  };
}

[[nodiscard]] std::vector<uint64_t> start_offsets(const std::vector<BasicBlock *> &blocks) {
  std::vector<uint64_t> offsets;
  offsets.reserve(blocks.size());
  for (const BasicBlock *block : blocks)
    offsets.push_back(block->start_offset());
  return offsets;
}

class KernelScopeTest : public ::testing::Test {
protected:
  void SetUp() override {
    decoder_ = Decoder::create(kArch);
    ASSERT_NE(decoder_, nullptr);
    object_ = std::make_unique<TestCodeObject>(two_kernel_words());
    blocks_ = build_valid_blocks(*object_, *decoder_, kArch);
    ASSERT_EQ(blocks_.size(), 6u);
    offset_index_ = build_block_offset_index(blocks_);
    position_index_ = build_block_position_index(blocks_);

    // Without this edge the kernel_entries stop would have nothing to cut and
    // ScopeStopsAtAnotherKernelEntry would pass vacuously.
    const BasicBlock *mid_a = block_for_offset(offset_index_, kMidA);
    ASSERT_NE(mid_a, nullptr);
    ASSERT_TRUE(std::ranges::any_of(mid_a->successors(), [](const BasicBlock *succ) {
      return succ->start_offset() == kEntryB;
    })) << "kernel A must branch into kernel B's entry";
  }

  [[nodiscard]] std::vector<BasicBlock *> scope_from(uint64_t entry_offset,
                                                     const KernelScopeSpec &spec) {
    BasicBlock *entry = block_for_offset(offset_index_, entry_offset);
    EXPECT_NE(entry, nullptr);
    if (entry == nullptr)
      return {};
    return reachable_kernel_blocks(blocks_, offset_index_, position_index_, *entry, spec);
  }

  std::unique_ptr<Decoder> decoder_;
  std::unique_ptr<TestCodeObject> object_;
  std::vector<std::unique_ptr<BasicBlock>> blocks_;
  BlockOffsetIndex offset_index_;
  BlockPositionIndex position_index_;
};

TEST_F(KernelScopeTest, ScopeStopsAtAnotherKernelEntry) {
  const KernelScopeSpec spec{
      .kernel_entries = {kEntryA, kEntryB},
      .own_entries = {kEntryA},
      .address_taken_entries = {},
  };
  // A branches straight into B's entry; that edge must not pull B's body in.
  EXPECT_EQ(start_offsets(scope_from(kEntryA, spec)),
            (std::vector<uint64_t>{kEntryA, kMidA, kTailA}));
}

TEST_F(KernelScopeTest, EachKernelGetsOnlyItsOwnBlocks) {
  const KernelScopeSpec spec{
      .kernel_entries = {kEntryA, kEntryB},
      .own_entries = {kEntryB},
      .address_taken_entries = {},
  };
  // B is entered from A, so it is not predecessorless, yet its scope is still
  // just its own body -- the walk starts at the entry, it does not run backward.
  EXPECT_EQ(start_offsets(scope_from(kEntryB, spec)),
            (std::vector<uint64_t>{kEntryB, kMidB, kTailB}));
}

TEST_F(KernelScopeTest, OwnEntriesOverrideTheKernelEntryStop) {
  const KernelScopeSpec spec{
      .kernel_entries = {kEntryA, kEntryB},
      .own_entries = {kEntryA, kEntryB},
      .address_taken_entries = {},
  };
  // This is the shape a kernarg-preload firmware entry or an adopted root takes:
  // an entry that belongs to this scope despite being in kernel_entries.
  EXPECT_EQ(start_offsets(scope_from(kEntryA, spec)),
            (std::vector<uint64_t>{kEntryA, kMidA, kTailA, kEntryB, kMidB, kTailB}));
}

TEST_F(KernelScopeTest, ScopeIsOrderedBySourceOffset) {
  const KernelScopeSpec spec{
      .kernel_entries = {kEntryA, kEntryB},
      .own_entries = {kEntryA, kEntryB},
      .address_taken_entries = {},
  };
  // Emission relies on source order so fallthrough is preserved.
  const std::vector<uint64_t> offsets = start_offsets(scope_from(kEntryA, spec));
  EXPECT_TRUE(std::ranges::is_sorted(offsets));
}

TEST_F(KernelScopeTest, UnreachedEntryYieldsOnlyItself) {
  // An own_entry with no decoded block is ignored rather than faulting.
  const KernelScopeSpec spec{
      .kernel_entries = {kEntryA, kEntryB},
      .own_entries = {kEntryB, 0x1000},
      .address_taken_entries = {},
  };
  EXPECT_EQ(start_offsets(scope_from(kEntryB, spec)),
            (std::vector<uint64_t>{kEntryB, kMidB, kTailB}));
}

TEST_F(KernelScopeTest, BlockForOffsetResolvesInteriorOffsets) {
  // A tail spans [0x08, 0x10): the s_endpgm at 0x0c is interior to it.
  const BasicBlock *tail = block_for_offset(offset_index_, kTailA);
  ASSERT_NE(tail, nullptr);
  EXPECT_EQ(tail->start_offset(), kTailA);
  EXPECT_EQ(block_for_offset(offset_index_, kTailA + 4), tail);

  EXPECT_EQ(block_for_offset(offset_index_, 0x1000), nullptr);
}

} // namespace
} // namespace rocjitsu
