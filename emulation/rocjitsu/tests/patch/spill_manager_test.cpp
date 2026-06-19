// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/spill_manager.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

constexpr uint32_t kBigLimit = 16384;

RegisterRef sgpr(uint16_t i) { return RegisterRef{RegClass::SGPR, i, 1}; }
RegisterRef vgpr(uint16_t i) { return RegisterRef{RegClass::VGPR, i, 1}; }
RegisterRef accvgpr(uint16_t i) { return RegisterRef{RegClass::ACC_VGPR, i, 1}; }

// Minimal synthetic Operand/Instruction/CodeObject/Decoder for integration
// tests. Mirrors the pattern in tests/analysis/liveness_test.cpp;
// duplicated here rather than shared to keep this test self-contained.
class TestOperand : public Operand {
public:
  TestOperand() = default;
  explicit TestOperand(RegisterRef ref) : Operand(ref.width * 32, ref.index), ref_(ref) {}
  std::optional<RegisterRef> to_register_ref() const override { return ref_; }

private:
  std::optional<RegisterRef> ref_;
};

class TestInstruction : public Instruction {
public:
  TestInstruction(std::string_view mnemonic, std::initializer_list<RegisterRef> defs,
                  std::initializer_list<RegisterRef> uses, uint64_t flags = 0)
      : Instruction(mnemonic, nullptr) {
    size_ = 4;
    flags_ = flags;
    for (RegisterRef ref : defs) {
      dst_storage_[num_dst_] = TestOperand(ref);
      dst_operands_[num_dst_] = &dst_storage_[num_dst_];
      ++num_dst_;
    }
    for (RegisterRef ref : uses) {
      src_storage_[num_src_] = TestOperand(ref);
      src_operands_[num_src_] = &src_storage_[num_src_];
      ++num_src_;
    }
  }

private:
  std::array<TestOperand, 2> dst_storage_{};
  std::array<TestOperand, 4> src_storage_{};
};

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

// Tiny opcode set just for the integration test. Each opcode maps to one
// TestInstruction with a fixed def/use pattern.
enum class IntegOpcode : uint32_t {
  Def_Vgpr0_Sgpr4 = 0,
  Def_Vgpr1 = 1,
  Use_Vgpr0_Sgpr4 = 2, // probe site
  Use_Vgpr1 = 3,
  End = 4,
};

class IntegDecoder : public Decoder {
public:
  Instruction *decode(const rj_code_binary_inst_t *inst) override {
    auto op = static_cast<IntegOpcode>(*inst);
    switch (op) {
    case IntegOpcode::Def_Vgpr0_Sgpr4:
      return new TestInstruction("i0_def", {vgpr(0), sgpr(4)}, {});
    case IntegOpcode::Def_Vgpr1:
      return new TestInstruction("i1_def", {vgpr(1)}, {});
    case IntegOpcode::Use_Vgpr0_Sgpr4:
      return new TestInstruction("i2_probe_site", {}, {vgpr(0), sgpr(4)});
    case IntegOpcode::Use_Vgpr1:
      return new TestInstruction("i3_use", {}, {vgpr(1)});
    case IntegOpcode::End:
      return new TestInstruction("end", {}, {}, PROGRAM_TERMINATOR);
    }
    return new TestInstruction("end", {}, {}, PROGRAM_TERMINATOR);
  }
};

// Start SpillManager Tests

TEST(SpillManager, BaseOffsetIsRoundedUpTo16) {
  {
    SpillManager m(0, kBigLimit);
    EXPECT_EQ(m.total_private_bytes(), 0u);
  }
  {
    SpillManager m(1, kBigLimit);
    auto off = m.allocate_slot(sgpr(0));
    ASSERT_TRUE(off.has_value());
    EXPECT_EQ(*off, 16u);
    EXPECT_EQ(m.total_private_bytes(), 20u);
  }
  {
    SpillManager m(20, kBigLimit);
    auto off = m.allocate_slot(sgpr(0));
    ASSERT_TRUE(off.has_value());
    EXPECT_EQ(*off, 32u);
    EXPECT_EQ(m.total_private_bytes(), 36u);
  }
}

TEST(SpillManager, AllocateSgprReturnsAscendingOffsets) {
  SpillManager m(0, kBigLimit);
  auto a = m.allocate_slot(sgpr(0));
  auto b = m.allocate_slot(sgpr(1));
  auto c = m.allocate_slot(sgpr(2));
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  ASSERT_TRUE(c.has_value());
  EXPECT_EQ(*a, 0u);
  EXPECT_EQ(*b, 4u);
  EXPECT_EQ(*c, 8u);
  EXPECT_EQ(m.total_private_bytes(), 12u);
}

TEST(SpillManager, AllocateIsIdempotentForSameRegister) {
  SpillManager m(0, kBigLimit);
  auto first = m.allocate_slot(sgpr(7));
  ASSERT_TRUE(first.has_value());
  const uint32_t bytes_after_first = m.total_private_bytes();

  auto second = m.allocate_slot(sgpr(7));
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(*first, *second);
  EXPECT_EQ(m.total_private_bytes(), bytes_after_first);
}

TEST(SpillManager, OffsetForReturnsNulloptForUnknown) {
  SpillManager m(0, kBigLimit);
  EXPECT_EQ(m.offset_for(sgpr(0)), std::nullopt);
}

TEST(SpillManager, OffsetForReturnsAllocatedOffset) {
  SpillManager m(0, kBigLimit);
  auto allocated = m.allocate_slot(sgpr(3));
  ASSERT_TRUE(allocated.has_value());
  auto looked_up = m.offset_for(sgpr(3));
  ASSERT_TRUE(looked_up.has_value());
  EXPECT_EQ(*looked_up, *allocated);
}

TEST(SpillManager, AllocateSlotsReturnsContiguousOffsets) {
  SpillManager m(0, kBigLimit);
  auto first = m.allocate_slots(sgpr(4), 2);
  ASSERT_TRUE(first.has_value());
  auto s4 = m.offset_for(sgpr(4));
  auto s5 = m.offset_for(sgpr(5));
  ASSERT_TRUE(s4.has_value());
  ASSERT_TRUE(s5.has_value());
  EXPECT_EQ(*s4, *first);
  EXPECT_EQ(*s5, *first + 4u);
  EXPECT_EQ(m.total_private_bytes(), 8u);
}

TEST(SpillManager, AllocateSlotsThenSingleSlotInterleaves) {
  SpillManager m(0, kBigLimit);
  auto pair1 = m.allocate_slots(sgpr(4), 2);
  auto single = m.allocate_slot(sgpr(7));
  auto pair2 = m.allocate_slots(vgpr(10), 2);
  ASSERT_TRUE(pair1.has_value());
  ASSERT_TRUE(single.has_value());
  ASSERT_TRUE(pair2.has_value());
  EXPECT_EQ(*pair1, 0u);
  EXPECT_EQ(*m.offset_for(sgpr(5)), 4u);
  EXPECT_EQ(*single, 8u);
  EXPECT_EQ(*pair2, 12u);
  EXPECT_EQ(*m.offset_for(vgpr(11)), 16u);
  EXPECT_EQ(m.total_private_bytes(), 20u);
}

TEST(SpillManager, AllocateSlotsIsIdempotent) {
  SpillManager m(0, kBigLimit);
  auto first = m.allocate_slots(sgpr(4), 2);
  ASSERT_TRUE(first.has_value());
  const uint32_t bytes_after_first = m.total_private_bytes();

  auto second = m.allocate_slots(sgpr(4), 2);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(*first, *second);
  EXPECT_EQ(m.total_private_bytes(), bytes_after_first);
}

TEST(SpillManager, TestSyntheticSet) {
  SpillManager m(0, kBigLimit);
  const std::vector<RegisterRef> regs = {
      sgpr(2), vgpr(7), sgpr(0), sgpr(11), vgpr(3), sgpr(5),
  };

  std::vector<uint32_t> offsets;
  for (auto reg : regs) {
    auto off = m.allocate_slot(reg);
    ASSERT_TRUE(off.has_value());
    offsets.push_back(*off);
  }

  // Each slot occupies [offset, offset+4); assert no two slot ranges overlap.
  for (size_t i = 0; i < offsets.size(); ++i) {
    for (size_t j = i + 1; j < offsets.size(); ++j) {
      const uint32_t lo = std::min(offsets[i], offsets[j]);
      const uint32_t hi = std::max(offsets[i], offsets[j]);
      EXPECT_GE(hi - lo, 4u) << "slots " << i << " (offset " << offsets[i] << ") and " << j
                             << " (offset " << offsets[j] << ") overlap";
    }
  }

  // Every slot's full [offset, offset+4) range lies within the reported
  // per-lane scratch bytes — no slot extends past the cursor.
  for (size_t i = 0; i < offsets.size(); ++i) {
    EXPECT_LE(offsets[i] + 4u, m.total_private_bytes())
        << "slot " << i << " (offset " << offsets[i] << ") past total_private_bytes "
        << m.total_private_bytes();
  }

  // 6 non-overlapping 4-byte slots fit in exactly 24 bytes — implies tight
  // packing with each slot exactly 4 bytes wide and no padding.
  EXPECT_EQ(m.total_private_bytes(), 24u);
}

TEST(SpillManager, VgprSlotIsFourBytesNotWaveSizeTimesFour) {
  SpillManager m(0, kBigLimit);
  const uint32_t before = m.total_private_bytes();
  ASSERT_TRUE(m.allocate_slot(vgpr(0)).has_value());
  EXPECT_EQ(m.total_private_bytes() - before, 4u);
}

TEST(SpillManager, AccVgprAndVgprSameIndexHaveDistinctSlots) {
  SpillManager m(0, kBigLimit);
  auto v_off = m.allocate_slot(vgpr(5));
  auto a_off = m.allocate_slot(accvgpr(5));
  ASSERT_TRUE(v_off.has_value());
  ASSERT_TRUE(a_off.has_value());
  EXPECT_NE(*v_off, *a_off);
  EXPECT_EQ(m.offset_for(vgpr(5)), v_off);
}

TEST(SpillManager, AccVgprIs4Bytes) {
  SpillManager m(0, kBigLimit);
  const uint32_t before = m.total_private_bytes();
  ASSERT_TRUE(m.allocate_slot(accvgpr(0)).has_value());
  EXPECT_EQ(m.total_private_bytes() - before, 4u);
}

// Pins the contract that allocate_slot returns a per-lane *byte offset* that
// composes additively with the per-lane flat-scratch addressing formula
//   flat_scratch_base_byte_addr[lane] = FLAT_SCRATCH_INIT
//                                     + lane * private_segment_fixed_size
// If this test ever needs editing, the trampoline emitter (Phase 6) is also
// broken. We use a *non-zero* slot so the `+ slot_offset` term is exercised
// (the lane=0/slot=0 case alone is a tautology).
TEST(SpillManager, SlotAddressMatchesFlatScratchFormula) {
  SpillManager m(/*orig=*/0, /*limit=*/16384);
  ASSERT_TRUE(m.allocate_slot(sgpr(0)).has_value());
  auto slot1 = m.allocate_slot(sgpr(1));
  ASSERT_TRUE(slot1.has_value());
  EXPECT_EQ(*slot1, 4u);

  constexpr uint64_t kFlatScratchInit = 0xCAFEBA5Eull;
  const uint32_t per_lane_bytes = m.total_private_bytes();

  {
    constexpr uint32_t lane = 0;
    const uint64_t addr = kFlatScratchInit + lane * per_lane_bytes + *slot1;
    EXPECT_EQ(addr, kFlatScratchInit + 4u);
  }
  {
    constexpr uint32_t lane = 1;
    const uint64_t addr = kFlatScratchInit + lane * per_lane_bytes + *slot1;
    EXPECT_EQ(addr, kFlatScratchInit + per_lane_bytes + 4u);
  }
}

TEST(SpillManager, ReserveSet_AddsAllRegisters) {
  RegisterSet set;
  for (uint16_t i : {0, 2, 5, 11})
    set.expand(sgpr(i));
  for (uint16_t i : {3, 7})
    set.expand(vgpr(i));

  SpillManager m(0, kBigLimit);
  EXPECT_TRUE(m.reserve(set));
  EXPECT_EQ(m.total_private_bytes(), 24u);

  for (uint16_t i : {0, 2, 5, 11})
    EXPECT_TRUE(m.offset_for(sgpr(i)).has_value());
  for (uint16_t i : {3, 7})
    EXPECT_TRUE(m.offset_for(vgpr(i)).has_value());
}

TEST(SpillManager, ReserveSet_IsIdempotent) {
  RegisterSet set;
  for (uint16_t i : {0, 2, 5, 11})
    set.expand(sgpr(i));
  for (uint16_t i : {3, 7})
    set.expand(vgpr(i));

  SpillManager m(0, kBigLimit);
  ASSERT_TRUE(m.reserve(set));
  const uint32_t bytes_after_first = m.total_private_bytes();

  EXPECT_TRUE(m.reserve(set));
  EXPECT_EQ(m.total_private_bytes(), bytes_after_first);
}

TEST(SpillManager, ReserveSet_HandlesEmpty) {
  SpillManager m(0, kBigLimit);
  const uint32_t before = m.total_private_bytes();
  EXPECT_TRUE(m.reserve(RegisterSet{}));
  EXPECT_EQ(m.total_private_bytes(), before);
}

TEST(SpillManager, AllocateSlotReturnsNulloptOnOverflow) {
  SpillManager m(0, /*limit=*/8);
  EXPECT_TRUE(m.allocate_slot(sgpr(0)).has_value());
  EXPECT_TRUE(m.allocate_slot(sgpr(1)).has_value());

  EXPECT_EQ(m.allocate_slot(sgpr(2)), std::nullopt);
  EXPECT_EQ(m.total_private_bytes(), 8u);
  EXPECT_EQ(m.offset_for(sgpr(2)), std::nullopt);
}

// Limit chosen so the first sub-slot fits at offset 8 (8+4 = 12 <= 12) but the
// second fails at offset 12 (12+4 = 16 > 12) — exercises the partial-rollback
// path.
TEST(SpillManager, AllocateSlotsRollsBackOnPartialOverflow) {
  SpillManager m(0, /*limit=*/12);
  ASSERT_TRUE(m.allocate_slots(sgpr(4), 2).has_value());
  ASSERT_EQ(m.total_private_bytes(), 8u);

  EXPECT_EQ(m.allocate_slots(sgpr(6), 2), std::nullopt);
  EXPECT_EQ(m.total_private_bytes(), 8u);
  EXPECT_EQ(m.offset_for(sgpr(6)), std::nullopt);
  EXPECT_EQ(m.offset_for(sgpr(7)), std::nullopt);

  // Proves next_offset_ was actually rewound (not just total_bytes_): a
  // subsequent single-slot allocation lands at the rolled-back cursor.
  auto recovered = m.allocate_slot(sgpr(6));
  ASSERT_TRUE(recovered.has_value());
  EXPECT_EQ(*recovered, 8u);
}

TEST(SpillManager, ReserveRollsBackOnOverflow) {
  RegisterSet set;
  for (uint16_t i : {0, 1, 2})
    set.expand(sgpr(i));

  SpillManager m(0, /*limit=*/8);
  EXPECT_FALSE(m.reserve(set));
  EXPECT_EQ(m.total_private_bytes(), 0u);
  for (uint16_t i : {0, 1, 2})
    EXPECT_EQ(m.offset_for(sgpr(i)), std::nullopt);
}

TEST(SpillManager, LimitBoundaryFitsExactly) {
  SpillManager m(0, /*limit=*/8);
  auto a = m.allocate_slot(sgpr(0));
  auto b = m.allocate_slot(sgpr(1));
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  EXPECT_EQ(*a, 0u);
  EXPECT_EQ(*b, 4u);
  EXPECT_EQ(m.total_private_bytes(), 8u);

  EXPECT_EQ(m.allocate_slot(sgpr(2)), std::nullopt);
}

// At limit=4, exactly one slot fits. A second NEW allocation would fail, but
// re-allocating the same register must hit the cache before the limit check.
TEST(SpillManager, IdempotentReallocationUnderTightLimit) {
  SpillManager m(0, /*limit=*/4);
  auto first = m.allocate_slot(sgpr(0));
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(*first, 0u);
  EXPECT_EQ(m.total_private_bytes(), 4u);

  // A new register would fail.
  EXPECT_EQ(m.allocate_slot(sgpr(1)), std::nullopt);

  // But re-alloc of the same register hits the cache and succeeds.
  auto second = m.allocate_slot(sgpr(0));
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(*second, *first);
}

TEST(SpillManager, AllocateSlotsWidth1EqualsAllocateSlot) {
  SpillManager m(0, kBigLimit);
  auto via_slots = m.allocate_slots(sgpr(0), 1);
  ASSERT_TRUE(via_slots.has_value());
  EXPECT_EQ(*via_slots, 0u);
  EXPECT_EQ(m.total_private_bytes(), 4u);
  EXPECT_EQ(m.offset_for(sgpr(0)), via_slots);
}

TEST(SpillManager, AllocateSlotsWidth0ReturnsNullopt) {
  SpillManager m(0, kBigLimit);
  EXPECT_EQ(m.allocate_slots(sgpr(0), 0), std::nullopt);
  EXPECT_EQ(m.total_private_bytes(), 0u);
  EXPECT_EQ(m.offset_for(sgpr(0)), std::nullopt);
}

TEST(SpillManager, AllocateSlotsRejectsIndexOverflow) {
  SpillManager m(0, kBigLimit);
  // reg.index + width - 1 = 0xFFFE + 4 - 1 = 0x10001 > UINT16_MAX.
  EXPECT_EQ(m.allocate_slots(sgpr(0xFFFE), 4), std::nullopt);
  EXPECT_EQ(m.total_private_bytes(), 0u);
  EXPECT_EQ(m.offset_for(sgpr(0xFFFE)), std::nullopt);
}

// `allocate_slots` rejects ranges that would extend past the per-class
// hardware bound, rather than silently truncating via RegisterSet::expand.
TEST(SpillManager, AllocateSlotsRejectsPerClassOverflow) {
  SpillManager m(0, kBigLimit);
  // base + width = REGISTER_SET_MAX_SGPRS + 1 → past bound.
  const uint16_t base = static_cast<uint16_t>(REGISTER_SET_MAX_SGPRS - 1);
  EXPECT_EQ(m.allocate_slots(sgpr(base), 2), std::nullopt);
  EXPECT_EQ(m.total_private_bytes(), 0u);
  EXPECT_EQ(m.offset_for(sgpr(base)), std::nullopt);

  // A within-bound width-1 allocation at the last index still works.
  auto last = m.allocate_slot(sgpr(base));
  ASSERT_TRUE(last.has_value());
  EXPECT_EQ(*last, 0u);

  // And a width-1 allocate_slots at the last index is also fine (within bound).
  // Width 2 is the smallest that overflows here.
  auto last_via_slots = m.allocate_slots(sgpr(base), 1);
  ASSERT_TRUE(last_via_slots.has_value());
  EXPECT_EQ(*last_via_slots, *last); // idempotent
}

// reserve() called twice with overlapping sets only consumes new bytes for
// registers that weren't already cached.
TEST(SpillManager, ReserveSet_PartialOverlap) {
  SpillManager m(0, kBigLimit);

  RegisterSet first;
  for (uint16_t i : {0, 1})
    first.expand(sgpr(i));
  ASSERT_TRUE(m.reserve(first));
  EXPECT_EQ(m.total_private_bytes(), 8u);

  // Superset: 0 and 1 are cached; 2 and 3 are new. Only 8 more bytes expected.
  RegisterSet second;
  for (uint16_t i : {0, 1, 2, 3})
    second.expand(sgpr(i));
  EXPECT_TRUE(m.reserve(second));
  EXPECT_EQ(m.total_private_bytes(), 16u);

  for (uint16_t i : {0, 1, 2, 3})
    EXPECT_TRUE(m.offset_for(sgpr(i)).has_value());
  // Cached registers kept their original offsets.
  EXPECT_EQ(m.offset_for(sgpr(0)), 0u);
  EXPECT_EQ(m.offset_for(sgpr(1)), 4u);
}

// If the original private segment already exceeds the per-lane cap (rounded up
// to the DBI alignment), the manager is constructible but every allocation
// must fail — there's no room to append a DBI zone.
TEST(SpillManager, ConstructorOverLimitFailsAllAllocations) {
  SpillManager m(/*orig=*/100, /*limit=*/8);
  EXPECT_EQ(m.allocate_slot(sgpr(0)), std::nullopt);
  EXPECT_EQ(m.allocate_slots(sgpr(1), 2), std::nullopt);

  RegisterSet set;
  set.expand(sgpr(2));
  EXPECT_FALSE(m.reserve(set));
}

// End-to-end smoke test: feed a real LivenessAnalysis result into SpillManager.
TEST(SpillManager, IntegrationFromLiveBefore) {
  // Build a tiny single-block kernel:
  //   I0: def vgpr 0, def sgpr 4
  //   I1: def vgpr 1
  //   I2: use vgpr 0, use sgpr 4   <-- probe site (vgpr 1 stays live across)
  //   I3: use vgpr 1
  //   I4: end
  std::vector<uint32_t> words = {
      static_cast<uint32_t>(IntegOpcode::Def_Vgpr0_Sgpr4),
      static_cast<uint32_t>(IntegOpcode::Def_Vgpr1),
      static_cast<uint32_t>(IntegOpcode::Use_Vgpr0_Sgpr4),
      static_cast<uint32_t>(IntegOpcode::Use_Vgpr1),
      static_cast<uint32_t>(IntegOpcode::End),
  };
  TestCodeObject co(std::move(words));
  IntegDecoder decoder;
  auto blocks = BasicBlock::build(co, decoder, std::span<const uint64_t>{});
  ASSERT_FALSE(blocks.empty());

  // Pick the probe-site instruction by mnemonic match.
  const Instruction *probe = nullptr;
  for (const auto &block : blocks) {
    for (const auto &inst : block->instructions()) {
      if (inst.disassemble().find("i2_probe_site") != std::string::npos) {
        probe = &inst;
        break;
      }
    }
    if (probe)
      break;
  }
  ASSERT_NE(probe, nullptr);

  std::vector<BasicBlock *> scope;
  for (const auto &b : blocks)
    scope.push_back(b.get());
  LivenessAnalysis liveness{KernelBlockScope(scope)};
  const RegisterSet &live = liveness.live_before(*probe);
  ASSERT_FALSE(live.none()) << "expected at least one live register at probe site";

  // Make up the private scratch size
  constexpr uint32_t kOrigPrivateBytes = 100;
  SpillManager m(kOrigPrivateBytes, /*limit=*/16384);
  EXPECT_TRUE(m.reserve(live));

  // total = align_up(orig, 16) + kSlotBytes * |live|. align_up(100, 16) == 112.
  constexpr size_t kExpectedBase = 112;
  EXPECT_EQ(static_cast<size_t>(m.total_private_bytes()),
            kExpectedBase + SpillManager::kSlotBytes * live.size());

  // Every live register has an offset; collect them to check pairwise distinct.
  std::vector<uint32_t> offsets;
  live.for_each([&](RegisterRef r) {
    auto off = m.offset_for(r);
    EXPECT_TRUE(off.has_value()) << "missing offset for live register";
    if (off.has_value())
      offsets.push_back(*off);
  });
  for (size_t i = 0; i < offsets.size(); ++i) {
    for (size_t j = i + 1; j < offsets.size(); ++j) {
      EXPECT_NE(offsets[i], offsets[j]) << "alias: live registers at indices " << i << " and " << j
                                        << " share offset " << offsets[i];
    }
  }
}

} // namespace
} // namespace rocjitsu
