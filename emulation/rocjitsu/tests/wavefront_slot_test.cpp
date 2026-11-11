// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file wavefront_slot_test.cpp
/// @brief InstructionTiming and WavefrontSlots: the per-slot sequencer state a
/// cycle-level driver keeps for a compute unit.

#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/instruction_timing.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/amdgpu/wavefront_slot.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

// gfx1201 encodings (llvm-mc -mcpu=gfx1201).
constexpr uint32_t kMovS5Seven = 0xBE850087u;    // s_mov_b32 s5, 7
constexpr uint32_t kAddV2V1V0 = 0x4A040101u;     // v_add_nc_u32 v2, v1, v0 (reads v1)
constexpr uint32_t kSqrtV6V0 = 0x7E0C6700u;      // v_sqrt_f32 v6, v0
constexpr uint32_t kBranchSkipOne = 0xBFA00001u; // s_branch 1
constexpr uint32_t kWaitLoadcnt0 = 0xBFC00000u;  // s_wait_loadcnt 0
constexpr uint32_t kEndpgm = 0xBFB00000u;        // s_endpgm
constexpr uint32_t kSLoadS7 = 0xF40001C0u;       // s_load_b32 s7, s[0:1], 0x0
constexpr uint32_t kSLoadS7Hi = 0xF8000000u;
constexpr uint32_t kDsLoadV1 = 0xD8D80000u; // ds_load_b32 v1, v0
constexpr uint32_t kDsLoadV1Hi = 0x01000000u;
constexpr std::array<uint32_t, 3> kGlobalLoadV4{0xEE050000u, 4u,
                                                0u}; // global_load_b32 v4, v0, s[0:1]
constexpr std::array<uint32_t, 3> kFlatLoadV4{0xEC05007Cu, 4u, 0u}; // flat_load_b32 v4, v[0:1]
// s_sendmsg_rtn_b32 s4, sendmsg(MSG_RTN_GET_DOORBELL)
constexpr uint32_t kSendmsgRtnS4 = 0xBE844C80u;
// v_cndmask_b32_e64 v0, v1, v2, s4 (Wave32) / s[4:5] (Wave64): one encoding.
constexpr std::array<uint32_t, 2> kCndmaskMaskS4{0xD5010000u, 0x00120501u};

constexpr uint64_t kProgram = 0x1000;
constexpr uint64_t kOtherProgram = 0x2000;

/// Decode @p encoding as gfx1201 and classify it for a wave of @p wave_size lanes.
InstructionTiming classify_rdna4(Decoder &decoder, std::array<uint32_t, 4> encoding,
                                 uint8_t wave_size = 32) {
  auto decoded = decoder.decode(encoding.data());
  EXPECT_TRUE(decoded.succeeded());
  if (!decoded.succeeded())
    return {};
  auto timing = InstructionTiming::classify(std::move(decoded.value()), 0x100, wave_size);
  EXPECT_TRUE(timing.has_value());
  return timing ? std::move(*timing) : InstructionTiming{};
}

/// The counters the CU charges for a FLAT request once it has routed its
/// lanes, as MemoryPipeline::issue_counters() derives them.
MemoryPipeline::WaitCounterTokens routed_flat_counters(const InstructionTiming &flat, bool global,
                                                       bool lds) {
  MemoryPipeline::WaitCounterTokens tokens;
  const auto *issue = flat.instruction().amdgpu_memory_issue_info();
  EXPECT_NE(issue, nullptr);
  if (!issue)
    return tokens;
  // Hold the routed info: counter_obligations() is a view into it.
  const MemoryIssueInfo routed = issue->for_flat_memory_domains(global, lds);
  for (const auto obligation : routed.counter_obligations())
    for (uint8_t i = 0; i < obligation.counter_increment(); ++i)
      tokens.types[tokens.size++] = obligation.wait_counter_type();
  return tokens;
}

class InstructionTimingTest : public ::testing::Test {
protected:
  InstructionTiming classify(std::array<uint32_t, 4> encoding, uint8_t wave_size = 32) {
    return classify_rdna4(*decoder, encoding, wave_size);
  }

  std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
};

TEST_F(InstructionTimingTest, ClassifiesBlocksAndSerialization) {
  ASSERT_NE(decoder, nullptr);
  const auto salu = classify({kMovS5Seven, 0, 0, 0});
  ASSERT_NE(salu.inst, nullptr);
  EXPECT_EQ(salu.block(), ExecutionBlock::SALU);
  EXPECT_EQ(salu.pc, 0x100u);
  EXPECT_EQ(salu.size(), 4u);
  EXPECT_FALSE(salu.serializing());

  const auto valu = classify({kAddV2V1V0, 0, 0, 0});
  EXPECT_EQ(valu.block(), ExecutionBlock::VALU);
  EXPECT_TRUE(valu.hazards.contains({RegClass::VGPR, 1, 1}));
  EXPECT_TRUE(valu.writes.contains({RegClass::VGPR, 2, 1}));
  EXPECT_FALSE(valu.writes.contains({RegClass::VGPR, 1, 1}));
  EXPECT_FALSE(valu.instruction().is_transcendental());
  EXPECT_TRUE(classify({kSqrtV6V0, 0, 0, 0}).instruction().is_transcendental());

  const auto branch = classify({kBranchSkipOne, 0, 0, 0});
  EXPECT_EQ(branch.block(), ExecutionBlock::BRANCH);
  EXPECT_TRUE(branch.redirects_fetch());
  EXPECT_TRUE(branch.serializing());

  EXPECT_TRUE(classify({kEndpgm, 0, 0, 0}).instruction().is_program_terminator());

  const auto wait = classify({kWaitLoadcnt0, 0, 0, 0});
  EXPECT_TRUE(wait.instruction().is_waitcnt());
  EXPECT_TRUE(wait.serializing());
}

TEST_F(InstructionTimingTest, WaveMaskOperandsFollowTheWaveSize) {
  ASSERT_NE(decoder, nullptr);
  // The mask is s4 in Wave32 and s[4:5] in Wave64: a fixed width would add a
  // false dependency on s5 in Wave32 or miss the real one in Wave64.
  const auto wave32 = classify({kCndmaskMaskS4[0], kCndmaskMaskS4[1], 0, 0}, 32);
  EXPECT_TRUE(wave32.hazards.contains({RegClass::SGPR, 4, 1}));
  EXPECT_FALSE(wave32.hazards.contains({RegClass::SGPR, 5, 1}));

  const auto wave64 = classify({kCndmaskMaskS4[0], kCndmaskMaskS4[1], 0, 0}, 64);
  EXPECT_TRUE(wave64.hazards.contains({RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(wave64.hazards.contains({RegClass::SGPR, 5, 1}));
}

TEST_F(InstructionTimingTest, CountsMemoryCounterIncrements) {
  ASSERT_NE(decoder, nullptr);
  const auto smem = classify({kSLoadS7, kSLoadS7Hi, 0, 0});
  EXPECT_EQ(smem.block(), ExecutionBlock::SMEM);
  ASSERT_EQ(smem.counters.size, 1u);

  const auto lds = classify({kDsLoadV1, kDsLoadV1Hi, 0, 0});
  EXPECT_EQ(lds.block(), ExecutionBlock::LDS);
  ASSERT_EQ(lds.counters.size, 1u);
  EXPECT_EQ(lds.counters.types[0], WaitCounterType::DSCNT);

  const auto load = classify({kGlobalLoadV4[0], kGlobalLoadV4[1], kGlobalLoadV4[2], 0});
  EXPECT_EQ(load.block(), ExecutionBlock::VMEM);
  EXPECT_EQ(load.size(), 12u);
  ASSERT_EQ(load.counters.size, 1u);
  EXPECT_EQ(load.counters.types[0], WaitCounterType::LOADCNT);
  EXPECT_TRUE(load.writes.contains({RegClass::VGPR, 4, 1}));

  // Before routing, a FLAT load may reach either domain: it reserves both.
  const auto flat = classify({kFlatLoadV4[0], kFlatLoadV4[1], kFlatLoadV4[2], 0});
  ASSERT_EQ(flat.counters.size, 2u);
  EXPECT_EQ(flat.counters.types[0], WaitCounterType::LOADCNT);
  EXPECT_EQ(flat.counters.types[1], WaitCounterType::DSCNT);

  EXPECT_EQ(classify({kMovS5Seven, 0, 0, 0}).counters.size, 0u);
}

TEST_F(InstructionTimingTest, CountsScalarMessageReturns) {
  ASSERT_NE(decoder, nullptr);
  // RDNA4 counts one KMCNT unit for the send and one for the returned value,
  // though the message issues through SALU.
  const auto message = classify({kSendmsgRtnS4, 0, 0, 0});
  EXPECT_EQ(message.block(), ExecutionBlock::SALU);
  ASSERT_EQ(message.counters.size, 2u);
  EXPECT_EQ(message.counters.types[0], WaitCounterType::KMCNT);
  EXPECT_EQ(message.counters.types[1], WaitCounterType::KMCNT);
  EXPECT_TRUE(message.writes.contains({RegClass::SGPR, 4, 1}));
}

class WavefrontSlotsTest : public ::testing::Test {
protected:
  void SetUp() override {
    l2.set_backing_memory(&memory);
    ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_RDNA4;
    config.num_wf_slots = 2;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cu = ComputeUnitCore::create("wavefront_slot_cu", config, &memory, &l2);
    ASSERT_NE(cu, nullptr);
    for (size_t i = 0; i < 4; ++i) {
      memory.write32(kProgram + i * 4, kEndpgm);
      memory.write32(kOtherProgram + i * 4, kEndpgm);
    }
    slots = std::make_unique<WavefrontSlots>(*cu, 3);
  }

  void TearDown() override {
    for (uint32_t slot = 0; cu && slot < cu->num_wf_slots(); ++slot)
      if (Wavefront *wave = cu->wf(slot); wave && !wave->is_halted())
        wave->halt();
  }

  Wavefront *dispatch(uint64_t pc = kProgram) { return cu->dispatch_wf(0, pc, 104, 256); }

  InstructionTiming classify(std::array<uint32_t, 4> encoding) {
    return classify_rdna4(*decoder, encoding);
  }

  /// A VALU instruction writing @p dst and reading @p src. Built by hand: the
  /// slot reads only the register sets and counters, never the instruction.
  static InstructionTiming valu(uint16_t dst, uint16_t src) {
    InstructionTiming inst;
    inst.writes.expand({RegClass::VGPR, dst, 1});
    inst.hazards.expand({RegClass::VGPR, src, 1});
    inst.hazards.expand({RegClass::VGPR, dst, 1});
    return inst;
  }

  GpuMemory memory{"wavefront_slot_memory"};
  L2Cache l2{"wavefront_slot_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  std::unique_ptr<WavefrontSlots> slots;
  std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
};

TEST_F(WavefrontSlotsTest, LaunchFillsTheReservedSlotInArrivalOrder) {
  ASSERT_EQ(slots->size(), 2u);
  EXPECT_FALSE((*slots)[0].active());
  Wavefront *first = dispatch();
  Wavefront *second = dispatch();
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  WavefrontSlot &slot0 = (*slots)[first->wf_id()];
  WavefrontSlot &slot1 = (*slots)[second->wf_id()];
  EXPECT_TRUE(slot0.active());
  EXPECT_EQ(slot0.wf, first);
  EXPECT_EQ(slot0.generation, first->dispatch_generation());
  EXPECT_TRUE(slot0.frontend_empty());
  EXPECT_LT(slot0.age, slot1.age);

  const WavefrontKey key = slots->key(first->wf_id());
  EXPECT_EQ(key.cu, 3u);
  EXPECT_EQ(key.slot, first->wf_id());
  EXPECT_EQ(key.generation, first->dispatch_generation());
  EXPECT_EQ(slots->live(key), &slot0);
  EXPECT_EQ(slots->live({.cu = 0, .slot = key.slot, .generation = key.generation}), nullptr);
}

TEST_F(WavefrontSlotsTest, LaunchStartsFetchAtTheEntryPc) {
  Wavefront *wave = dispatch(kProgram);
  ASSERT_NE(wave, nullptr);
  WavefrontSlot &slot = (*slots)[wave->wf_id()];
  EXPECT_EQ(slot.fetch_pc, kProgram);

  // The previous occupant's cursor does not survive into the slot's next wave.
  slot.fetch_pc = kProgram + 8;
  wave->halt();
  ASSERT_EQ(dispatch(kOtherProgram), wave);
  EXPECT_TRUE(slot.active());
  EXPECT_EQ(slot.fetch_pc, kOtherProgram);
}

TEST_F(WavefrontSlotsTest, ARolledBackDispatchLeavesTheSlotInactive) {
  Wavefront *wave = dispatch();
  ASSERT_NE(wave, nullptr);
  const WavefrontKey key = slots->key(wave->wf_id());
  ASSERT_NE(slots->live(key), nullptr);

  // The CU frees the wave without the slot hearing of it; active() sees it.
  cu->abort_dispatch(wave->dispatch_id());
  EXPECT_FALSE((*slots)[key.slot].active());
  EXPECT_EQ(slots->live(key), nullptr);
}

TEST_F(WavefrontSlotsTest, KeysGoStaleWhenTheWaveLeavesOrIsSquashed) {
  Wavefront *wave = dispatch();
  ASSERT_NE(wave, nullptr);
  const WavefrontKey key = slots->key(wave->wf_id());
  WavefrontSlot &slot = (*slots)[key.slot];
  const uint64_t epoch = slot.epoch;
  EXPECT_EQ(slots->current(key, epoch), &slot);

  // A squash leaves the wave live but discards work that precedes issue.
  slot.squash(/*flush=*/false);
  EXPECT_EQ(slots->live(key), &slot);
  EXPECT_EQ(slots->current(key, epoch), nullptr);

  // A halted wave, or a new dispatch into its slot, no longer matches.
  wave->halt();
  EXPECT_EQ(slots->live(key), nullptr);
  ASSERT_EQ(dispatch(), wave);
  EXPECT_EQ(slots->live(key), nullptr);
  EXPECT_EQ(slots->live(slots->key(key.slot)), &slot);
}

TEST_F(WavefrontSlotsTest, RegisterWritesBecomeVisibleAtWriteback) {
  ASSERT_NE(dispatch(), nullptr);
  WavefrontSlot &slot = (*slots)[0];
  const InstructionTiming producer = valu(2, 1);
  const InstructionTiming consumer = valu(3, 2);
  const InstructionTiming independent = valu(5, 4);

  slot.reserve(producer, 1);
  EXPECT_TRUE(slot.has_hazard(consumer, 10));
  EXPECT_FALSE(slot.has_hazard(independent, 10));

  slot.issued(1);
  slot.written_back(1, 12);
  EXPECT_TRUE(slot.has_hazard(consumer, 11));
  EXPECT_FALSE(slot.has_hazard(consumer, 12));
}

TEST_F(WavefrontSlotsTest, SquashThenCancelDiscardsABufferedReservation) {
  ASSERT_NE(dispatch(), nullptr);
  WavefrontSlot &slot = (*slots)[0];
  slot.buffer.push_back(classify({kGlobalLoadV4[0], kGlobalLoadV4[1], kGlobalLoadV4[2], 0}));
  const InstructionTiming &load = slot.buffer.front();
  const InstructionTiming reader = valu(5, 4);

  // Reserving takes the entry's registers and counter capacity; it stays buffered.
  ASSERT_EQ(slot.next_to_reserve(), &load);
  slot.reserve(load, 1);
  EXPECT_EQ(slot.reserved, 1u);
  EXPECT_EQ(slot.next_to_reserve(), nullptr);
  EXPECT_EQ(slot.memory_tokens, 1u);
  EXPECT_TRUE(slot.has_hazard(reader, 0));

  // The squash rewinds the buffer and frees the registers at once; the
  // instruction may still be on its way to issue, so its capacity stays.
  slot.squash(/*flush=*/false);
  EXPECT_EQ(slot.reserved, 0u);
  EXPECT_EQ(slot.next_to_reserve(), &load);
  EXPECT_FALSE(slot.has_hazard(reader, 0));
  EXPECT_EQ(slot.memory_tokens, 1u);

  // When it reaches issue with a stale epoch, its consumer cancels it.
  slot.cancel(1, load.counters.size);
  EXPECT_EQ(slot.memory_tokens, 0u);
  EXPECT_EQ(slot.next_to_reserve(), &load);

  // The entry is reserved again, as a new instruction.
  slot.reserve(load, 2);
  EXPECT_EQ(slot.next_to_reserve(), nullptr);
  EXPECT_EQ(slot.memory_tokens, 1u);
  EXPECT_TRUE(slot.has_hazard(reader, 0));
}

TEST_F(WavefrontSlotsTest, SquashKeepsIssuedWrites) {
  ASSERT_NE(dispatch(), nullptr);
  WavefrontSlot &slot = (*slots)[0];
  const InstructionTiming consumer = valu(3, 2);

  slot.reserve(valu(2, 1), 2);
  slot.issued(2);
  slot.reserve(valu(3, 0), 3);
  slot.squash(/*flush=*/false);
  EXPECT_EQ(slot.reserved, 0u);
  // The issued write survives the squash; the unissued one is gone.
  EXPECT_TRUE(slot.has_hazard(consumer, 0));
  EXPECT_FALSE(slot.has_hazard(valu(6, 3), 0));
}

TEST_F(WavefrontSlotsTest, CounterWaitHoldsUntilTheMemoryResponseReturns) {
  ASSERT_NE(dispatch(), nullptr);
  WavefrontSlot &slot = (*slots)[0];
  InstructionTiming load;
  load.writes.expand({RegClass::VGPR, 4, 1});
  load.counters.types[load.counters.size++] = WaitCounterType::LOADCNT;

  slot.reserve(load, 7);
  EXPECT_EQ(slot.memory_tokens, 1u);
  slot.issued(7);
  slot.issued_memory(load, load.counters);
  EXPECT_EQ(slot.outstanding.vmcnt, 1u);

  WaitTarget wait;
  wait.vmcnt = 0;
  slot.waiting = wait;
  EXPECT_TRUE(slot.waiting_on_counters());

  slot.memory_returned(7, load.counters, 20);
  EXPECT_EQ(slot.outstanding.vmcnt, 0u);
  EXPECT_EQ(slot.memory_tokens, 0u);
  EXPECT_FALSE(slot.waiting_on_counters());
  EXPECT_FALSE(slot.waiting.has_value());
  EXPECT_FALSE(slot.has_hazard(valu(5, 4), 20));
}

/// Issue a FLAT load routed to @p global and/or @p lds, and check that only the
/// counters of the routed domain go outstanding and return.
void expect_flat_charges_only_its_route(WavefrontSlot &slot, const InstructionTiming &flat,
                                        bool global, bool lds) {
  const auto charged = routed_flat_counters(flat, global, lds);
  ASSERT_EQ(charged.size, 1u);

  slot.reserve(flat, 1);
  EXPECT_EQ(slot.memory_tokens, 2u);
  slot.issued(1);
  slot.issued_memory(flat, charged);
  // The unused domain's reservation is released at issue.
  EXPECT_EQ(slot.memory_tokens, 1u);
  EXPECT_EQ(slot.outstanding.vmcnt, global ? 1u : 0u);
  EXPECT_EQ(slot.outstanding.dscnt, lds ? 1u : 0u);

  // A wait on the other domain's counter does not stall behind this load.
  WaitTarget other;
  if (global)
    other.dscnt = 0;
  else
    other.vmcnt = 0;
  slot.waiting = other;
  EXPECT_FALSE(slot.waiting_on_counters());

  WaitTarget own;
  if (global)
    own.vmcnt = 0;
  else
    own.dscnt = 0;
  slot.waiting = own;
  EXPECT_TRUE(slot.waiting_on_counters());

  // The response returns exactly what was charged.
  slot.memory_returned(1, charged, 30);
  EXPECT_EQ(slot.memory_tokens, 0u);
  EXPECT_TRUE(slot.outstanding.empty());
  EXPECT_FALSE(slot.waiting_on_counters());
}

TEST_F(WavefrontSlotsTest, GlobalOnlyFlatChargesOnlyLoadcnt) {
  ASSERT_NE(dispatch(), nullptr);
  const auto flat = classify({kFlatLoadV4[0], kFlatLoadV4[1], kFlatLoadV4[2], 0});
  expect_flat_charges_only_its_route((*slots)[0], flat, /*global=*/true, /*lds=*/false);
}

TEST_F(WavefrontSlotsTest, LdsOnlyFlatChargesOnlyDscnt) {
  ASSERT_NE(dispatch(), nullptr);
  const auto flat = classify({kFlatLoadV4[0], kFlatLoadV4[1], kFlatLoadV4[2], 0});
  expect_flat_charges_only_its_route((*slots)[0], flat, /*global=*/false, /*lds=*/true);
}

TEST_F(WavefrontSlotsTest, ScalarMessageReturnHoldsAKmcntWait) {
  ASSERT_NE(dispatch(), nullptr);
  WavefrontSlot &slot = (*slots)[0];
  const auto message = classify({kSendmsgRtnS4, 0, 0, 0});
  ASSERT_EQ(message.counters.size, 2u);

  slot.reserve(message, 1);
  slot.issued(1);
  slot.issued_memory(message, message.counters);
  EXPECT_EQ(slot.outstanding.kmcnt, 2u);

  // s_wait_kmcnt 0 holds until the returned value arrives.
  WaitTarget wait;
  wait.kmcnt = 0;
  slot.waiting = wait;
  EXPECT_TRUE(slot.waiting_on_counters());
  // The returned value is s4's write.
  InstructionTiming reads_s4;
  reads_s4.hazards.expand({RegClass::SGPR, 4, 1});
  EXPECT_TRUE(slot.has_hazard(reads_s4, 39));

  slot.memory_returned(1, message.counters, 40);
  EXPECT_EQ(slot.outstanding.kmcnt, 0u);
  EXPECT_EQ(slot.memory_tokens, 0u);
  EXPECT_FALSE(slot.waiting_on_counters());
  EXPECT_FALSE(slot.has_hazard(reads_s4, 40));
}

TEST_F(WavefrontSlotsTest, ReleasesTheReservationHandler) {
  slots.reset();
  Wavefront *wave = dispatch();
  ASSERT_NE(wave, nullptr);
  WavefrontSlots fresh(*cu, 0);
  EXPECT_FALSE(fresh[wave->wf_id()].active());
}

} // namespace
