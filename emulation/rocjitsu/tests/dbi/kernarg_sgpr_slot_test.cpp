// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernarg_sgpr_slot_test.cpp
/// @brief Pins the CP's user-SGPR placement against kernarg_segment_ptr_slot().
///
/// Two independent walks of the AMDHSA user-SGPR enable-bit order decide where a
/// dispatch's kernarg pointer ends up. CommandProcessor::init_wavefront_regs
/// writes it, and
/// kernarg_segment_ptr_slot() is what the DBI entry prologue reads it from --
/// baked into an s_load_dwordx2's SBASE at patch time. The CP interleaves its
/// register writes with the walk, so it cannot call the helper, and nothing else
/// makes the two agree.
///
/// A drift between them is silent: the prologue would load from whatever pair
/// the stale walk named, the loaded value would be an unrelated register's
/// contents rather than a fault, and every unit test would stay green because
/// none of them dispatch. So this test dispatches.
///
/// Each case states both sides as literals rather than deriving one from the
/// other: the expected slot index is written out, the helper is asserted to
/// return it, and the CP is asserted to have written the pointer there. A walk
/// reimplemented in the test would drift along with whichever side it copied.

#include "dbi_sim.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

namespace kd = rocr::llvm::amdhsa;

// The enable bits ahead of the kernarg pointer in the user-SGPR order. Nothing
// past the kernarg pointer can move it. Each is a one-bit mask, so the mask
// doubles as the value to OR in.
constexpr uint32_t kPrivateSegmentBuffer =
    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER;
constexpr uint32_t kDispatchPtr = kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR;
constexpr uint32_t kQueuePtr = kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR;
constexpr uint32_t kKernargSegmentPtr = kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;

// The properties that follow the kernarg pointer. They cannot move it, which is
// the point: a CP edit that hoisted one of them above the kernarg block would be
// invisible to a sweep that never sets them.
constexpr uint32_t kTrailingProperties = kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID |
                                         kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT |
                                         kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE;

struct SlotCase {
  const char *name;
  uint32_t preceding;       ///< Enabled properties ahead of the kernarg pointer.
  uint32_t expected_slot;   ///< Architectural SGPR index the pointer must land at.
  uint32_t user_sgpr_count; ///< Total the enabled properties imply.
};

// Every combination of the three properties that precede the kernarg pointer.
// The slot is written out rather than summed from a table, so a change to the
// ABI order has to be restated here to pass. The count is weaker: it only feeds
// pkt.num_user_sgprs, so a count that is too small is caught (the workgroup IDs
// land on the kernarg slot) while one that is too large is not.
constexpr SlotCase kCases[] = {
    {"None", 0u, 0u, 2u},
    {"Buffer", kPrivateSegmentBuffer, 4u, 6u},
    {"Dispatch", kDispatchPtr, 2u, 4u},
    {"Queue", kQueuePtr, 2u, 4u},
    {"BufferDispatch", kPrivateSegmentBuffer | kDispatchPtr, 6u, 8u},
    {"BufferQueue", kPrivateSegmentBuffer | kQueuePtr, 6u, 8u},
    {"DispatchQueue", kDispatchPtr | kQueuePtr, 4u, 6u},
    {"BufferDispatchQueue", kPrivateSegmentBuffer | kDispatchPtr | kQueuePtr, 8u, 10u},
    // Trailing properties enabled as well: the slot is unchanged, and the walk
    // has to keep placing them after the kernarg pointer for that to hold.
    // dispatch_id (2) + flat_scratch_init (2) + private_segment_size (1) on top
    // of the eight preceding, so 15 user SGPRs.
    {"TrailingPropertiesOnly", kTrailingProperties, 0u, 7u},
    {"AllProperties", kPrivateSegmentBuffer | kDispatchPtr | kQueuePtr | kTrailingProperties, 8u,
     15u},
};

class KernargSgprSlot : public ::testing::TestWithParam<SlotCase> {};

TEST_P(KernargSgprSlot, CommandProcessorWritesThePointerWhereTheHelperNamesIt) {
  const SlotCase &c = GetParam();
  const uint32_t properties = c.preceding | kKernargSegmentPtr;

  kd::kernel_descriptor_t desc{};
  desc.kernel_code_properties = properties;
  EXPECT_EQ(kernarg_segment_ptr_slot(desc), c.expected_slot);
  EXPECT_TRUE(has_kernarg_segment_ptr(desc));

  // The placement walk reads only enable bits, so one arch exercises it. The
  // USER_SGPR_COUNT field it is paired with is not arch-neutral (gfx1250 widens
  // it), but DbiSim writes the generic field and has no gfx1250 config. The
  // kernarg bytes are never read; only the pointer's placement is under test.
  test::DbiSim sim("cdna3", /*wave_size=*/64);
  sim.set_kernarg(std::vector<uint8_t>(8, 0), properties, c.user_sgpr_count);

  const std::vector<uint32_t> code{build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3)};
  const std::optional<uint64_t> pair =
      sim.run_and_read_sgpr64(code, /*private_bytes=*/0, c.expected_slot);

  ASSERT_TRUE(pair.has_value()) << "no wave halted";
  EXPECT_EQ(*pair, test::DbiSim::KERNARG_ADDR);
}

// A pointer that happened to be in every slot would pass the case above without
// the walks agreeing. The neighbouring pairs must be clear of it.
TEST(KernargSgprSlotNeighbours, ThePointerIsNotAlsoInTheAdjacentPairs) {
  const uint32_t properties = kPrivateSegmentBuffer | kDispatchPtr | kKernargSegmentPtr;
  constexpr uint32_t kSlot = 6;

  kd::kernel_descriptor_t desc{};
  desc.kernel_code_properties = properties;
  ASSERT_EQ(kernarg_segment_ptr_slot(desc), kSlot);

  const std::vector<uint32_t> code{build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3)};
  for (uint32_t slot : {kSlot - 2, kSlot + 2}) {
    test::DbiSim sim("cdna3", /*wave_size=*/64);
    sim.set_kernarg(std::vector<uint8_t>(8, 0), properties, /*user_sgpr_count=*/8);
    const std::optional<uint64_t> pair = sim.run_and_read_sgpr64(code, /*private_bytes=*/0, slot);
    ASSERT_TRUE(pair.has_value()) << "no wave halted";
    EXPECT_NE(*pair, test::DbiSim::KERNARG_ADDR) << "slot " << slot;
  }
}

INSTANTIATE_TEST_SUITE_P(Properties, KernargSgprSlot, ::testing::ValuesIn(kCases),
                         [](const ::testing::TestParamInfo<SlotCase> &info) {
                           return std::string(info.param.name);
                         });

} // namespace
} // namespace rocjitsu
