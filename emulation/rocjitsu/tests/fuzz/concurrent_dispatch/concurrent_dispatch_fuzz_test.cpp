// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "concurrent_dispatch_fuzz_core.h"
#include "execution_fuzz_core.h"

#include "rocjitsu/code/builders/vector_builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "simdojo/sim/sim_types.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace rocjitsu::concurrent_dispatch_fuzz {
namespace {

constexpr uint8_t kCdnaFlatSaddrOff = 0; // Canonical CDNA FLAT vector-address-only encoding.
constexpr uint16_t kInlineZero = cdna1::OPR_SRC_POS_INT_MIN;
constexpr uint16_t kHwIdSimdId = 0x0904; // hwreg(HW_REG_HW_ID, 4, 2)
constexpr std::array<uint32_t, 2> kInvalidCdna3Instruction{0xe0ffebf5, 0x00020000};

template <typename Words> void append_words(std::vector<uint8_t> &input, const Words &words) {
  for (uint32_t word : words)
    for (unsigned byte = 0; byte < sizeof(word); ++byte)
      input.push_back(static_cast<uint8_t>(word >> (8 * byte)));
}

void append_xcc_zero_test(std::vector<uint8_t> &input) {
  constexpr uint16_t kHwRegXccId = 0x1814; // hwreg(HW_REG_XCC_ID, 0, 4)
  append_words(input,
               cdna3::build_sopk(cdna3::kSGetregB32Sopk, {.simm16 = kHwRegXccId, .sdst = 4}));
  append_words(input, cdna3::build_sopc(cdna3::kSCmpEqU32Sopc,
                                        {.ssrc0 = 4, .ssrc1 = cdna3::OPR_SSRC_POS_INT_MIN}));
}

std::vector<uint8_t> flat_store_input(uint32_t address, uint8_t control = 0,
                                      uint8_t packet_count_selector = 0) {
  std::vector<uint8_t> input(kHeaderSize);
  input[0] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx908);
  input[1] = control;
  input[4] = packet_count_selector;
  append_words(input, build_v_mov_b32_imm(0, address, ROCJITSU_CODE_ARCH_CDNA1).value());
  append_words(input, cdna1::build_vop1(cdna1::kVMovB32Vop1, {.src0 = kInlineZero, .vdst = 1}));
  append_words(input, cdna1::build_vop1(cdna1::kVMovB32Vop1, {.src0 = kInlineZero, .vdst = 2}));
  append_words(input, cdna1::build_flat(cdna1::kFlatStoreDwordFlat,
                                        {.addr = 0, .data = 2, .saddr = kCdnaFlatSaddrOff}));
  append_words(input, cdna1::build_sopp(cdna1::kSWaitcntSopp));
  return input;
}

std::vector<uint8_t> simd_id_fault_input(bool halt_on_simd_zero, uint8_t control = 0) {
  std::vector<uint8_t> input(kHeaderSize);
  input[0] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx942);
  input[1] = control;
  input[2] = 2; // Five workgroups, so fan-out can put two on the same peer CU.

  // SIMD 0 halts or retires. Other SIMD slots branch over that instruction
  // and store to an unmapped address. The pool can retire more SIMD-0 waves
  // than serial execution before observing the fault.
  append_words(input,
               cdna3::build_sopk(cdna3::kSGetregB32Sopk, {.simm16 = kHwIdSimdId, .sdst = 4}));
  append_words(input, cdna3::build_sopc(cdna3::kSCmpEqU32Sopc,
                                        {.ssrc0 = 4, .ssrc1 = cdna3::OPR_SSRC_POS_INT_MIN}));
  append_words(input, cdna3::build_sopp(cdna3::kSCbranchScc0Sopp, {.simm16 = 1}));
  append_words(
      input, cdna3::build_sopp(halt_on_simd_zero ? cdna3::kSSendmsghaltSopp : cdna3::kSEndpgmSopp));
  append_words(input, cdna3::build_vop1(cdna3::kVMovB32Vop1, {.src0 = kInlineZero, .vdst = 0}));
  append_words(input, cdna3::build_vop1(cdna3::kVMovB32Vop1, {.src0 = kInlineZero, .vdst = 1}));
  append_words(input, cdna3::build_vop1(cdna3::kVMovB32Vop1, {.src0 = kInlineZero, .vdst = 2}));
  append_words(input, cdna3::build_flat(cdna3::kFlatStoreDwordFlat,
                                        {.addr = 0, .data = 2, .saddr = kCdnaFlatSaddrOff}));
  append_words(input, cdna3::build_sopp(cdna3::kSWaitcntSopp));
  return input;
}

std::vector<uint8_t> racy_stall_input() {
  std::vector<uint8_t> input(kHeaderSize);
  input[0] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx942);
  input[2] = 2; // Five workgroups.
  input[4] = 1; // Two unbarriered packets.

  // Load the shared dword and then store one. The first wave on each CU
  // always halts. Other waves halt only if they read one, so scheduling can
  // change how many workgroups retire while the input still stalls.
  constexpr uint16_t kInlineOne = cdna3::OPR_SRC_POS_INT_MIN + 1;
  append_words(input, cdna3::build_vop1(cdna3::kVMovB32Vop1, {.src0 = 0, .vdst = 0}));
  append_words(input, cdna3::build_vop1(cdna3::kVMovB32Vop1, {.src0 = 1, .vdst = 1}));
  append_words(input, cdna3::build_vop1(cdna3::kVMovB32Vop1, {.src0 = kInlineOne, .vdst = 2}));
  append_words(input, cdna3::build_flat(cdna3::kFlatLoadDwordFlat,
                                        {.addr = 0, .saddr = kCdnaFlatSaddrOff, .vdst = 3}));
  append_words(input, cdna3::build_sopp(cdna3::kSWaitcntSopp));
  append_words(input, cdna3::build_flat(cdna3::kFlatStoreDwordFlat,
                                        {.addr = 0, .data = 2, .saddr = kCdnaFlatSaddrOff}));
  append_words(input, cdna3::build_sopp(cdna3::kSWaitcntSopp));
  append_words(input,
               cdna3::build_sopk(cdna3::kSGetregB32Sopk, {.simm16 = kHwIdSimdId, .sdst = 4}));
  append_words(input, cdna3::build_sopc(cdna3::kSCmpEqU32Sopc,
                                        {.ssrc0 = 4, .ssrc1 = cdna3::OPR_SSRC_POS_INT_MIN}));
  append_words(input, cdna3::build_sopp(cdna3::kSCbranchScc1Sopp, {.simm16 = 3}));
  append_words(input, cdna3::build_vopc(cdna3::kVCmpEqU32Vopc, {.src0 = kInlineOne, .vsrc1 = 3}));
  append_words(input, cdna3::build_sopp(cdna3::kSCbranchVccnzSopp, {.simm16 = 1}));
  append_words(input, cdna3::build_sopp(cdna3::kSEndpgmSopp));
  append_words(input, cdna3::build_sopp(cdna3::kSSendmsghaltSopp));
  return input;
}

TEST(ConcurrentDispatchFuzzTest, CanonicalProgramsUseTargetSpecificFlatBaseAndWaits) {
  const std::vector<std::vector<uint8_t>> seeds = seed_corpus();
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
    case execution_fuzz::FuzzTarget::Gfx1250:
    case execution_fuzz::FuzzTarget::Gfx1251:
      expected_words = {0xec0d407c, 0x01080000, 0x00000000, 0xbfc90000};
      break;
    case execution_fuzz::FuzzTarget::Count:
      FAIL() << "invalid target in seed corpus";
    }

    std::vector<uint8_t> expected_suffix;
    append_words(expected_suffix, expected_words);

    const std::span<const uint8_t> program(seeds[index].data() + kHeaderSize,
                                           seeds[index].size() - kHeaderSize);
    constexpr std::size_t kAddressSetupBytes = 3 * sizeof(uint32_t);
    ASSERT_EQ(program.size(), kAddressSetupBytes + expected_suffix.size());
    EXPECT_TRUE(std::equal(expected_suffix.begin(), expected_suffix.end(),
                           program.begin() + kAddressSetupBytes));
  }
}

TEST(ConcurrentDispatchFuzzTest, SeedCorpusCoversEveryExecutionTarget) {
  const std::vector<std::vector<uint8_t>> seeds = seed_corpus();
  ASSERT_EQ(seeds.size(), static_cast<std::size_t>(execution_fuzz::FuzzTarget::Count));
  uint64_t primed = 0;
  uint64_t multi_xcd = 0;
  uint64_t fanout = 0;
  uint64_t barriered = 0;
  bool uses_second_owner_without_fanout = false;
  for (const std::vector<uint8_t> &seed : seeds) {
    const RunStats stats = run_input(seed);
    EXPECT_EQ(stats.inputs, 1u);
    EXPECT_EQ(stats.targets, 1u);
    EXPECT_EQ(stats.modes, stats.multi_xcd_inputs ? 4u : 2u);
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
    barriered += (seed[1] & kBarrierBit) != 0 && stats.packets > 1;
    uses_second_owner_without_fanout |=
        stats.multi_xcd_inputs != 0 && stats.fanout_inputs == 0 && (seed[5] & 1) != 0;
  }
  EXPECT_GT(primed, 0u);
  EXPECT_GT(multi_xcd, 0u);
  EXPECT_GT(fanout, 0u);
  EXPECT_LT(fanout, multi_xcd);
  EXPECT_GT(barriered, 0u);
  EXPECT_TRUE(uses_second_owner_without_fanout);
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

TEST(ConcurrentDispatchFuzzTest, BarrieredFanoutCompletesFromEitherQueueOwner) {
  const std::vector<std::vector<uint8_t>> seeds = seed_corpus();
  std::vector<uint8_t> input = seeds[static_cast<std::size_t>(execution_fuzz::FuzzTarget::Gfx950)];
  for (uint8_t owner : {uint8_t{0}, uint8_t{1}}) {
    SCOPED_TRACE(owner);
    input[5] = owner;
    const RunStats stats = run_input(input);
    EXPECT_EQ(stats.fanout_inputs, 1u);
    EXPECT_EQ(stats.packets, 2u);
    EXPECT_EQ(stats.differential_inputs, 1u);
    EXPECT_EQ(stats.modes, 4u);
    EXPECT_EQ(stats.completed_modes, stats.modes);
  }
}

TEST(ConcurrentDispatchFuzzTest, RawKernelMutationUsesSchedulingOutcomeOracle) {
  std::vector<uint8_t> input = seed_corpus().front();
  const uint32_t end = execution_fuzz::terminating_instruction(execution_fuzz::FuzzTarget::Gfx908);
  append_words(input, std::array{end});
  EXPECT_EQ(run_input(input).differential_inputs, 0u);
}

TEST(ConcurrentDispatchFuzzTest, RawKernelCannotOverwriteHarnessState) {
  // The completion-signal value is deliberately outside the wave's bounded
  // address space even though the command processor can access it.
  constexpr uint32_t kCompletionSignalValueAddress = 0x300008;
  const RunStats stats = run_input(flat_store_input(kCompletionSignalValueAddress));
  EXPECT_EQ(stats.terminal_faulted_modes, stats.modes);
}

TEST(ConcurrentDispatchFuzzTest, KernelRangeIsNotWritable) {
  constexpr uint32_t kKernelAddress = 0x10000;
  const RunStats stats = run_input(flat_store_input(kKernelAddress));
  EXPECT_EQ(stats.terminal_faulted_modes, stats.modes);
  EXPECT_EQ(stats.completed_modes, 0u);
}

TEST(ConcurrentDispatchFuzzTest, DataIslandCannotBecomeInstructionStream) {
  constexpr auto kTarget = execution_fuzz::FuzzTarget::Gfx1010;
  std::vector<uint8_t> input = seed_corpus()[static_cast<std::size_t>(kTarget)];
  input[1] = kPrimeCachesBit | kBarrierBit | kAgentAcquireBit | kAgentReleaseBit;
  input[2] = 1;  // Four workgroups.
  input[3] = 30; // 31 work-items per workgroup.
  input[4] = 1;  // Two packets.
  input[5] = 0;  // XCD 0 owns the queue.
  input[7] = 1;  // Prime two cache lines.

  // The canonical seed writes the data island. Jump to its kernarg address in s[0:1].
  append_words(input, rdna1::build_sop1(rdna1::kSSwapPcB64Sop1, {.ssrc0 = 0, .sdst = 0}));
  for (unsigned repetition = 0; repetition < 20; ++repetition) {
    const RunStats stats = run_input(input);
    ASSERT_EQ(stats.modes, 2u);
    EXPECT_EQ(stats.terminal_faulted_modes, stats.modes);
    EXPECT_EQ(stats.fatal_exit_modes, 0u);
    EXPECT_EQ(stats.completed_modes, 0u);
  }
}

TEST(ConcurrentDispatchFuzzTest, StoppedRawKernelIsOrdinaryAcrossWorkerModes) {
  std::vector<uint8_t> input(kHeaderSize);
  input[0] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx942);
  append_words(input, cdna3::build_sopp(cdna3::kSSendmsghaltSopp));

  const RunStats stats = run_input(input);
  EXPECT_EQ(stats.differential_inputs, 0u);
  EXPECT_EQ(stats.completed_modes, 0u);
  EXPECT_EQ(stats.terminal_faulted_modes, 0u);
  EXPECT_EQ(stats.primary_stalled_modes, stats.modes);
}

TEST(ConcurrentDispatchFuzzTest, TrailingLiteralCannotConsumeTheTerminatingSuffix) {
  for (const execution_fuzz::FuzzTarget target :
       {execution_fuzz::FuzzTarget::Gfx942, execution_fuzz::FuzzTarget::Gfx1100,
        execution_fuzz::FuzzTarget::Gfx1250}) {
    SCOPED_TRACE(execution_fuzz::target_name(target));
    std::vector<uint8_t> input(kHeaderSize);
    input[0] = static_cast<uint8_t>(target);
    // s_mov_b32 s0, literal. The raw body contains only the opcode word.
    append_words(input, std::array<uint32_t, 1>{0xbe8000ff});
    const RunStats stats = run_input(input);
    EXPECT_EQ(stats.completed_modes, stats.modes);
    EXPECT_EQ(stats.fatal_exit_modes, 0u);
  }
}

TEST(ConcurrentDispatchFuzzTest, FatalRawKernelIgnoresPostExitProgress) {
  std::vector<uint8_t> input(kHeaderSize);
  input[0] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx908);
  constexpr uint16_t kReservedSoppOpcode = 0x7f;
  append_words(input, cdna1::build_sopp(kReservedSoppOpcode));

  const RunStats stats = run_input(input);
  EXPECT_EQ(stats.differential_inputs, 0u);
  EXPECT_EQ(stats.completed_modes, 0u);
  EXPECT_EQ(stats.fatal_exit_modes, stats.modes);
}

TEST(ConcurrentDispatchFuzzTest, FatalPeerFaultReachesQueueOwner) {
  std::vector<uint8_t> input(kHeaderSize);
  input[0] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx942);
  input[1] = kXcdFanoutBit;
  // The owner on XCD 0 skips the invalid instruction. Only XCD 1 faults.
  append_xcc_zero_test(input);
  append_words(input, cdna3::build_sopp(cdna3::kSCbranchScc1Sopp,
                                        {.simm16 = kInvalidCdna3Instruction.size()}));
  append_words(input, kInvalidCdna3Instruction);

  const RunStats stats = run_input(input);
  ASSERT_EQ(stats.modes, 4u);
  EXPECT_EQ(stats.fatal_exit_modes, stats.modes);
  EXPECT_EQ(stats.terminal_faulted_modes, stats.modes);
}

TEST(ConcurrentDispatchFuzzTest, MatchingQueueFaultsAllowFatalInFlightInstructions) {
  std::vector<uint8_t> input(kHeaderSize);
  input[0] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx942);
  input[1] = kXcdFanoutBit;
  input[2] = 2; // Five workgroups.
  input[4] = 1; // Two unbarriered packets.

  // The owner faults on an unmapped store. The peer may reach its invalid
  // instruction before cancellation depending on how far its workers ran.
  append_xcc_zero_test(input);
  constexpr uint16_t kOwnerPathWords = 7; // Three moves, two store words, wait, end.
  append_words(input, cdna3::build_sopp(cdna3::kSCbranchScc0Sopp, {.simm16 = kOwnerPathWords}));
  for (uint8_t reg : {uint8_t{0}, uint8_t{1}, uint8_t{2}})
    append_words(input, cdna3::build_vop1(cdna3::kVMovB32Vop1, {.src0 = kInlineZero, .vdst = reg}));
  append_words(input, cdna3::build_flat(cdna3::kFlatStoreDwordFlat,
                                        {.addr = 0, .data = 2, .saddr = kCdnaFlatSaddrOff}));
  append_words(input, cdna3::build_sopp(cdna3::kSWaitcntSopp));
  append_words(input, cdna3::build_sopp(cdna3::kSEndpgmSopp));
  for (unsigned delay = 0; delay < 8; ++delay)
    append_words(input, cdna3::build_sopp(cdna3::kSNopSopp));
  append_words(input, kInvalidCdna3Instruction);

  for (unsigned repetition = 0; repetition < 20; ++repetition) {
    const RunStats stats = run_input(input);
    ASSERT_EQ(stats.modes, 4u);
    EXPECT_EQ(stats.terminal_faulted_modes, stats.modes);
    EXPECT_EQ(stats.primary_stalled_modes, 0u);
    EXPECT_EQ(stats.published_completions, 0u);
  }
}

TEST(ConcurrentDispatchFuzzTest, InitialRandomRangeRemainsCampaignViable) {
  for (uint64_t seed = 0; seed < 64; ++seed) {
    SCOPED_TRACE(seed);
    EXPECT_NO_THROW((void)run_input(random_input(seed)));
  }
}

TEST(ConcurrentDispatchFuzzTest, IncompleteRawKernelsIgnoreHostSchedulingProgress) {
  const std::vector<uint8_t> input = racy_stall_input();
  for (unsigned repetition = 0; repetition < 20; ++repetition) {
    const RunStats stats = run_input(input);
    EXPECT_EQ(stats.completed_modes, 0u);
    EXPECT_EQ(stats.terminal_faulted_modes, 0u);
    EXPECT_EQ(stats.primary_stalled_modes, stats.modes);
  }
}

TEST(ConcurrentDispatchFuzzTest, TerminallyFaultedRawKernelIgnoresRetirementProgress) {
  const std::vector<uint8_t> input = simd_id_fault_input(/*halt_on_simd_zero=*/false);
  for (unsigned repetition = 0; repetition < 20; ++repetition) {
    const RunStats stats = run_input(input);
    EXPECT_EQ(stats.terminal_faulted_modes, stats.modes);
    EXPECT_EQ(stats.incomplete_inputs, 1u);
  }
}

TEST(ConcurrentDispatchFuzzTest, TerminalFaultCancelsOlderUnbarrieredPacket) {
  for (uint8_t control : {uint8_t{0}, kXcdFanoutBit}) {
    SCOPED_TRACE(control);
    std::vector<uint8_t> input = simd_id_fault_input(/*halt_on_simd_zero=*/true, control);
    if (control == 0)
      input[2] = 0; // Three workgroups, one first-wave slot per CU.
    const RunStats one_packet = run_input(input);
    ASSERT_EQ(one_packet.packets, 1u);
    ASSERT_EQ(one_packet.terminal_faulted_modes, 0u);
    ASSERT_EQ(one_packet.primary_stalled_modes, one_packet.modes);

    input[4] = 1; // A newer packet uses SIMD 1 and faults while the older one is halted.
    const RunStats stats = run_input(input);
    ASSERT_EQ(stats.modes, 4u);
    EXPECT_EQ(stats.packets, 2u);
    EXPECT_EQ(stats.fanout_inputs, control == kXcdFanoutBit ? 1u : 0u);
    EXPECT_EQ(stats.terminal_faulted_modes, stats.modes);
    EXPECT_EQ(stats.primary_stalled_modes, 0u);
  }
}

TEST(ConcurrentDispatchFuzzTest, TerminalFaultReleasesPrimaryWithoutPublishingCompletion) {
  const RunStats stats = run_input(flat_store_input(/*address=*/0));
  EXPECT_EQ(stats.differential_inputs, 0u);
  EXPECT_EQ(stats.completed_modes, 0u);
  EXPECT_EQ(stats.budget_limited_modes, 0u);
  EXPECT_EQ(stats.terminal_faulted_modes, stats.modes);
  EXPECT_EQ(stats.primary_stalled_modes, 0u);
  EXPECT_EQ(stats.budget_limited_inputs, 0u);
  EXPECT_EQ(stats.published_completions, 0u);
  EXPECT_EQ(stats.incomplete_inputs, 1u);
}

TEST(ConcurrentDispatchFuzzTest, PrimaryLifecycleOracleChecksStallsAndReleasedPrimaries) {
  using simdojo::ExitCause;
  EXPECT_FALSE(detail::primary_lifecycle_is_valid(
      /*published_all_completions=*/true, /*terminal_faulted=*/false,
      ExitCause::ACTIVE_PRIMARY_STALL));
  EXPECT_FALSE(detail::primary_lifecycle_is_valid(
      /*published_all_completions=*/false, /*terminal_faulted=*/true,
      ExitCause::ACTIVE_PRIMARY_STALL));
  EXPECT_TRUE(detail::primary_lifecycle_is_valid(
      /*published_all_completions=*/false, /*terminal_faulted=*/false,
      ExitCause::ACTIVE_PRIMARY_STALL));
  EXPECT_FALSE(detail::primary_lifecycle_is_valid(
      /*published_all_completions=*/false, /*terminal_faulted=*/false,
      ExitCause::ALL_PRIMARIES_COMPLETED));
  EXPECT_TRUE(detail::primary_lifecycle_is_valid(
      /*published_all_completions=*/true, /*terminal_faulted=*/false,
      ExitCause::ALL_PRIMARIES_COMPLETED));
  EXPECT_TRUE(detail::primary_lifecycle_is_valid(
      /*published_all_completions=*/false, /*terminal_faulted=*/true,
      ExitCause::ALL_PRIMARIES_COMPLETED));
  EXPECT_TRUE(detail::primary_lifecycle_is_valid(
      /*published_all_completions=*/false, /*terminal_faulted=*/false, ExitCause::MAX_TICKS));
}

TEST(ConcurrentDispatchFuzzTest, BudgetCountersRequireNonFaultTickExhaustion) {
  std::vector<uint8_t> input(kHeaderSize);
  input[0] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx908);
  constexpr auto kInfiniteLoop = cdna1::build_sopp(cdna1::kSBranchSopp, {.simm16 = 0xffff});
  append_words(input, kInfiniteLoop);

  const RunStats stats = run_input(input);
  EXPECT_EQ(stats.differential_inputs, 0u);
  EXPECT_EQ(stats.completed_modes, 0u);
  EXPECT_EQ(stats.budget_limited_modes, stats.modes);
  EXPECT_EQ(stats.budget_limited_inputs, 1u);
}

TEST(ConcurrentDispatchFuzzTest, TerminalFaultDropsBlockedSuccessorAndReleasesPrimary) {
  const RunStats stats =
      run_input(flat_store_input(/*address=*/0, kBarrierBit, /*packet_count_selector=*/1));
  EXPECT_EQ(stats.packets, 2u);
  EXPECT_EQ(stats.differential_inputs, 0u);
  EXPECT_EQ(stats.completed_modes, 0u);
  EXPECT_EQ(stats.budget_limited_modes, 0u);
  EXPECT_EQ(stats.terminal_faulted_modes, stats.modes);
  EXPECT_EQ(stats.primary_stalled_modes, 0u);
  EXPECT_EQ(stats.published_completions, 0u);
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
