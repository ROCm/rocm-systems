// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "concurrent_dispatch_fuzz_core.h"
#include "execution_fuzz_core.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand_types.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace rocjitsu::concurrent_dispatch_fuzz {
namespace {

// Fixed raw inputs that leave completion fences pending while host-dependent
// dispatch progress differs between scheduling modes.
constexpr std::array<std::array<uint8_t, 64>, 2> kIncompleteDispatchInputs{{
    {
        0x0b, 0x30, 0x80, 0x23, 0x7d, 0x64, 0xf5, 0x50, 0xa1, 0x13, 0x6b, 0x7a, 0xd2,
        0x5c, 0x2a, 0x43, 0x00, 0x02, 0x00, 0x7e, 0x01, 0x02, 0x02, 0x7e, 0x81, 0x02,
        0x04, 0x7e, 0x7c, 0x40, 0x0d, 0xec, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0xc1, 0xbf, 0x00, 0x00, 0xc6, 0xbf, 0x00, 0x31, 0x80, 0xbe,
        0x00, 0x00, 0x2f, 0xd5, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x15, 0xee,
    },
    {
        0x0a, 0x30, 0x22, 0x2e, 0xf3, 0x56, 0xee, 0xba, 0x15, 0x20, 0x3c, 0xb8, 0x8c,
        0x93, 0x68, 0x2d, 0x00, 0x02, 0x00, 0x7e, 0x01, 0x02, 0x02, 0x7e, 0x81, 0x02,
        0x04, 0x7e, 0x7c, 0x40, 0x0d, 0xec, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0xc1, 0xbf, 0x00, 0x00, 0xc6, 0xbf, 0x00, 0x00, 0x52, 0x7c,
        0x00, 0x42, 0x10, 0xec, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    },
}};
constexpr uint8_t kCdnaFlatSaddrOff = 0; // Canonical CDNA FLAT vector-address-only encoding.

std::vector<uint8_t> faulting_flat_store_input(uint8_t control = 0,
                                               uint8_t packet_count_selector = 0) {
  std::vector<uint8_t> input(kHeaderSize);
  input[0] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx908);
  input[1] = control;
  input[4] = packet_count_selector;
  const auto append = [&](const auto &words) {
    for (uint32_t word : words)
      for (unsigned byte = 0; byte < sizeof(word); ++byte)
        input.push_back(static_cast<uint8_t>(word >> (8 * byte)));
  };
  // Inline zero makes the FLAT address unmapped in the bounded fuzz address space.
  constexpr uint16_t kInlineConstantZero = 128;
  append(cdna1::build_vop1(cdna1::kVMovB32Vop1, {.src0 = kInlineConstantZero, .vdst = 0}));
  append(cdna1::build_vop1(cdna1::kVMovB32Vop1, {.src0 = kInlineConstantZero, .vdst = 1}));
  append(cdna1::build_vop1(cdna1::kVMovB32Vop1, {.src0 = kInlineConstantZero, .vdst = 2}));
  append(cdna1::build_flat(cdna1::kFlatStoreDwordFlat,
                           {.addr = 0, .data = 2, .saddr = kCdnaFlatSaddrOff}));
  append(cdna1::build_sopp(cdna1::kSWaitcntSopp));
  return input;
}

std::vector<uint8_t> halt_then_fault_input() {
  std::vector<uint8_t> input(kHeaderSize);
  input[0] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx942);
  input[4] = 1; // Two packets: the first halts and the second faults.
  constexpr std::array<uint32_t, 10> kProgram{
      0xb8840904, 0xbf068004, 0xbf840001, 0xbf910000, 0x7e000280,
      0x7e020280, 0x7e040280, 0xdc700000, 0x00000200, 0xbf8c0000,
  };
  for (uint32_t word : kProgram)
    for (unsigned byte = 0; byte < sizeof(word); ++byte)
      input.push_back(static_cast<uint8_t>(word >> (8 * byte)));
  return input;
}

TEST(ConcurrentDispatchFuzzTest, CanonicalProgramsUseTargetSpecificFlatBaseAndWaits) {
  const auto seeds = seed_corpus();
  ASSERT_EQ(seeds.size(), static_cast<std::size_t>(execution_fuzz::FuzzTarget::Count));
  for (std::size_t index = 0; index < seeds.size(); ++index) {
    SCOPED_TRACE(index);
    const auto target = static_cast<execution_fuzz::FuzzTarget>(index);
    ASSERT_GT(seeds[index].size(), kHeaderSize + 3 * sizeof(uint32_t));
    EXPECT_EQ(seeds[index][0], index);

    // LLVM 23.1.4 llvm-mc encodings, kept independent of the builders under test.
    std::vector<uint32_t> expected_words;
    switch (target) {
    case execution_fuzz::FuzzTarget::Gfx908:
    case execution_fuzz::FuzzTarget::Gfx90a:
    case execution_fuzz::FuzzTarget::Gfx942:
    case execution_fuzz::FuzzTarget::Gfx950:
      expected_words = {0xdd080000, 0x00000200, 0xbf8c0000};
      break;
    case execution_fuzz::FuzzTarget::Gfx1010:
    case execution_fuzz::FuzzTarget::Gfx1030:
      expected_words = {0xdcc80000, 0x007d0200, 0xbf8c0000, 0xbbfd0000};
      break;
    case execution_fuzz::FuzzTarget::Gfx1100:
    case execution_fuzz::FuzzTarget::Gfx1151:
      expected_words = {0xdcd40000, 0x007c0200, 0xbf890000, 0xbc7c0000};
      break;
    case execution_fuzz::FuzzTarget::Gfx1200:
    case execution_fuzz::FuzzTarget::Gfx1201:
      expected_words = {0xec0d407c, 0x01080000, 0x00000000, 0xbfc60000, 0xbfc10000};
      break;
    case execution_fuzz::FuzzTarget::Gfx1250:
    case execution_fuzz::FuzzTarget::Gfx1251:
      expected_words = {0xec0d407c, 0x01080000, 0x00000000, 0xbfc10000, 0xbfc60000};
      break;
    case execution_fuzz::FuzzTarget::Count:
      FAIL() << "invalid target in seed corpus";
    }

    std::vector<uint8_t> expected_suffix;
    for (uint32_t word : expected_words)
      for (unsigned byte = 0; byte < sizeof(word); ++byte)
        expected_suffix.push_back(static_cast<uint8_t>(word >> (8 * byte)));

    const std::span<const uint8_t> program(seeds[index].data() + kHeaderSize,
                                           seeds[index].size() - kHeaderSize);
    constexpr std::size_t kAddressSetupBytes = 3 * sizeof(uint32_t);
    ASSERT_EQ(program.size(), kAddressSetupBytes + expected_suffix.size());
    EXPECT_TRUE(std::equal(expected_suffix.begin(), expected_suffix.end(),
                           program.begin() + kAddressSetupBytes));
  }
}

TEST(ConcurrentDispatchFuzzTest, SeedCorpusCoversEveryExecutionTarget) {
  const auto seeds = seed_corpus();
  ASSERT_EQ(seeds.size(), static_cast<std::size_t>(execution_fuzz::FuzzTarget::Count));
  uint64_t primed = 0;
  uint64_t multi_xcd = 0;
  uint64_t fanout = 0;
  for (const auto &seed : seeds) {
    const RunStats stats = run_input(seed);
    EXPECT_EQ(stats.inputs, 1u);
    EXPECT_EQ(stats.targets, 1u);
    EXPECT_EQ(stats.modes, stats.multi_xcd_inputs ? 3u : 2u);
    EXPECT_GT(stats.packets, 0u);
    EXPECT_GT(stats.workgroups, 0u);
    EXPECT_GT(stats.workitems, 0u);
    EXPECT_GE(stats.kernel_bytes, 6 * sizeof(uint32_t));
    EXPECT_LE(stats.kernel_bytes, 8 * sizeof(uint32_t));
    EXPECT_EQ(stats.completed_inputs, 1u);
    EXPECT_EQ(stats.budget_limited_inputs, 0u);
    EXPECT_EQ(stats.differential_inputs, 1u);
    EXPECT_EQ(stats.pooled_dispatch_threads, 3u);
    primed += stats.cache_primed_inputs;
    multi_xcd += stats.multi_xcd_inputs;
    fanout += stats.fanout_inputs;
  }
  EXPECT_GT(primed, 0u);
  EXPECT_GT(multi_xcd, 0u);
  EXPECT_GT(fanout, 0u);
  EXPECT_LT(fanout, multi_xcd);
}

TEST(ConcurrentDispatchFuzzTest, RandomInputIsStableAndIncludesBothKernelClasses) {
  EXPECT_EQ(random_input(17), random_input(17));
  EXPECT_NE(random_input(17), random_input(18));
  uint64_t differential = 0;
  uint64_t raw = 0;
  std::vector<uint8_t> differential_input;
  for (uint64_t seed = 0; seed < 64; ++seed) {
    const std::vector<uint8_t> input = random_input(seed);
    EXPECT_LE(input.size(), kGeneratedInputSize);
    if (input.size() == kGeneratedInputSize) {
      ++raw;
    } else {
      ++differential;
      differential_input = input;
    }
  }
  EXPECT_GT(differential, 0u);
  EXPECT_GT(raw, 0u);
  ASSERT_FALSE(differential_input.empty());
  EXPECT_EQ(run_input(differential_input).differential_inputs, 1u);
}

TEST(ConcurrentDispatchFuzzTest, RawKernelMutationUsesSchedulingOutcomeOracle) {
  std::vector<uint8_t> input = seed_corpus().front();
  const uint32_t end = execution_fuzz::terminating_instruction(execution_fuzz::FuzzTarget::Gfx908);
  for (unsigned byte = 0; byte < sizeof(end); ++byte)
    input.push_back(static_cast<uint8_t>(end >> (8 * byte)));
  EXPECT_EQ(run_input(input).differential_inputs, 0u);
}

TEST(ConcurrentDispatchFuzzTest, RawKernelCannotOverwriteHarnessState) {
  std::vector<uint8_t> input(kHeaderSize);
  const auto append = [&](const auto &words) {
    for (uint32_t word : words)
      for (unsigned byte = 0; byte < sizeof(word); ++byte)
        input.push_back(static_cast<uint8_t>(word >> (8 * byte)));
  };
  // The completion-signal value is deliberately outside the wave's bounded
  // address space even though the command processor can access it.
  constexpr uint32_t kCompletionSignalValueAddress = 0x300008;
  append(cdna1::build_vop1(cdna1::kVMovB32Vop1, {.src0 = 255, .vdst = 0}));
  append(std::array<uint32_t, 1>{kCompletionSignalValueAddress});
  append(cdna1::build_vop1(cdna1::kVMovB32Vop1, {.src0 = 128, .vdst = 1}));
  append(cdna1::build_vop1(cdna1::kVMovB32Vop1, {.src0 = 128, .vdst = 2}));
  append(cdna1::build_flat(cdna1::kFlatStoreDwordFlat,
                           {.addr = 0, .data = 2, .saddr = kCdnaFlatSaddrOff}));
  append(cdna1::build_sopp(cdna1::kSWaitcntSopp));

  EXPECT_NO_THROW((void)run_input(input));
}

TEST(ConcurrentDispatchFuzzTest, StoppedRawKernelIsOrdinaryAcrossWorkerModes) {
  std::vector<uint8_t> input(kHeaderSize);
  input[0] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx942);
  constexpr uint32_t kSendMessageAndHalt = 0xbf910000;
  for (unsigned byte = 0; byte < sizeof(kSendMessageAndHalt); ++byte)
    input.push_back(static_cast<uint8_t>(kSendMessageAndHalt >> (8 * byte)));

  const RunStats stats = run_input(input);
  EXPECT_EQ(stats.differential_inputs, 0u);
  EXPECT_EQ(stats.completed_modes, 0u);
}

TEST(ConcurrentDispatchFuzzTest, FatalRawKernelIgnoresPostExitProgress) {
  std::vector<uint8_t> input(kHeaderSize);
  input[0] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx908);
  constexpr auto kUnimplementedScalarAtomic = cdna1::build_smem(cdna1::kSBufferAtomicDecX2Smem);
  for (uint32_t word : kUnimplementedScalarAtomic)
    for (unsigned byte = 0; byte < sizeof(word); ++byte)
      input.push_back(static_cast<uint8_t>(word >> (8 * byte)));

  const RunStats stats = run_input(input);
  EXPECT_EQ(stats.differential_inputs, 0u);
  EXPECT_EQ(stats.completed_modes, 0u);
}

TEST(ConcurrentDispatchFuzzTest, InitialRandomRangeRemainsCampaignViable) {
  for (uint64_t seed = 0; seed < 64; ++seed) {
    SCOPED_TRACE(seed);
    EXPECT_NO_THROW((void)run_input(random_input(seed)));
  }
}

TEST(ConcurrentDispatchFuzzTest, IncompleteRawKernelsIgnoreHostSchedulingProgress) {
  for (std::size_t index = 0; index < kIncompleteDispatchInputs.size(); ++index) {
    SCOPED_TRACE(index);
    for (unsigned repetition = 0; repetition < 20; ++repetition) {
      const RunStats stats = run_input(kIncompleteDispatchInputs[index]);
      EXPECT_EQ(stats.completed_modes, 0u);
      EXPECT_EQ(stats.incomplete_inputs, 1u);
    }
  }
}

TEST(ConcurrentDispatchFuzzTest, TerminallyFaultedRawKernelIgnoresRetirementProgress) {
  for (unsigned repetition = 0; repetition < 20; ++repetition) {
    const RunStats stats = run_input(halt_then_fault_input());
    EXPECT_EQ(stats.terminal_faulted_modes, stats.modes);
    EXPECT_EQ(stats.incomplete_inputs, 1u);
  }
}

TEST(ConcurrentDispatchFuzzTest, TerminalFaultReleasesPrimaryAcrossWorkerModes) {
  const RunStats stats = run_input(faulting_flat_store_input());
  EXPECT_EQ(stats.differential_inputs, 0u);
  EXPECT_EQ(stats.completed_modes, 0u);
  EXPECT_EQ(stats.budget_limited_modes, 0u);
  EXPECT_EQ(stats.terminal_faulted_modes, stats.modes);
  EXPECT_EQ(stats.budget_limited_inputs, 0u);
}

TEST(ConcurrentDispatchFuzzTest, PrimaryLifecycleOracleRespectsCancellationBoundary) {
  EXPECT_FALSE(detail::primary_lifecycle_is_valid(
      /*published_all_completions=*/true, /*terminal_faulted=*/false,
      /*older_work_may_remain=*/false, /*primary_stalled=*/true));
  EXPECT_FALSE(detail::primary_lifecycle_is_valid(
      /*published_all_completions=*/false, /*terminal_faulted=*/true,
      /*older_work_may_remain=*/false, /*primary_stalled=*/true));
  EXPECT_TRUE(detail::primary_lifecycle_is_valid(
      /*published_all_completions=*/false, /*terminal_faulted=*/true,
      /*older_work_may_remain=*/true, /*primary_stalled=*/true));
  EXPECT_TRUE(detail::primary_lifecycle_is_valid(
      /*published_all_completions=*/false, /*terminal_faulted=*/false,
      /*older_work_may_remain=*/false, /*primary_stalled=*/true));
  EXPECT_TRUE(detail::primary_lifecycle_is_valid(
      /*published_all_completions=*/true, /*terminal_faulted=*/false,
      /*older_work_may_remain=*/false, /*primary_stalled=*/false));
}

TEST(ConcurrentDispatchFuzzTest, BudgetCountersRequireNonFaultTickExhaustion) {
  std::vector<uint8_t> input(kHeaderSize);
  input[0] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx908);
  constexpr auto kInfiniteLoop = cdna1::build_sopp(cdna1::kSBranchSopp, {.simm16 = 0xffff});
  for (uint32_t word : kInfiniteLoop)
    for (unsigned byte = 0; byte < sizeof(word); ++byte)
      input.push_back(static_cast<uint8_t>(word >> (8 * byte)));

  const RunStats stats = run_input(input);
  EXPECT_EQ(stats.differential_inputs, 0u);
  EXPECT_EQ(stats.completed_modes, 0u);
  EXPECT_EQ(stats.budget_limited_modes, stats.modes);
  EXPECT_EQ(stats.budget_limited_inputs, 1u);
}

TEST(ConcurrentDispatchFuzzTest, TerminallyFaultedPacketDoesNotPublishCompletion) {
  const RunStats stats = run_input(faulting_flat_store_input());
  EXPECT_EQ(stats.differential_inputs, 0u);
  EXPECT_EQ(stats.completed_modes, 0u);
  EXPECT_EQ(stats.budget_limited_modes, 0u);
  EXPECT_EQ(stats.terminal_faulted_modes, stats.modes);
  EXPECT_EQ(stats.incomplete_inputs, 1u);
}

TEST(ConcurrentDispatchFuzzTest, TerminalFaultDropsBlockedSuccessorAndReleasesPrimary) {
  const RunStats stats = run_input(faulting_flat_store_input(kBarrierBit, 1));
  EXPECT_EQ(stats.packets, 2u);
  EXPECT_EQ(stats.differential_inputs, 0u);
  EXPECT_EQ(stats.completed_modes, 0u);
  EXPECT_EQ(stats.budget_limited_modes, 0u);
  EXPECT_EQ(stats.terminal_faulted_modes, stats.modes);
  EXPECT_EQ(stats.incomplete_inputs, 1u);
}

TEST(ConcurrentDispatchFuzzTest, RejectsInputsOutsideTheSizeContract) {
  EXPECT_THROW((void)run_input({}), std::invalid_argument);
  const std::vector<uint8_t> oversized(kMaxInputSize + 1);
  EXPECT_THROW((void)run_input(oversized), std::invalid_argument);
}

TEST(ConcurrentDispatchFuzzTest, UnassignedTargetIsAnOrdinaryNoOp) {
  const RunStats stats = run_input(std::array<uint8_t, 1>{0xff});
  EXPECT_EQ(stats.inputs, 1u);
  EXPECT_EQ(stats.targets, 0u);
  EXPECT_EQ(stats.modes, 0u);
}

} // namespace
} // namespace rocjitsu::concurrent_dispatch_fuzz
