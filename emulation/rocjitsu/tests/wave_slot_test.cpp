// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file wave_slot_test.cpp
/// @brief InstructionTiming and WaveSlots: the per-slot sequencer state a
/// cycle-level driver keeps for a compute unit.

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/instruction_timing.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wave_slot.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>

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

constexpr uint64_t kProgram = 0x1000;

class InstructionTimingTest : public ::testing::Test {
protected:
  InstructionTiming classify(std::array<uint32_t, 4> encoding) {
    auto decoded = decoder->decode(encoding.data());
    EXPECT_TRUE(decoded.succeeded());
    auto timing = InstructionTiming::classify(*decoded.value(), 0x100);
    EXPECT_TRUE(timing.has_value());
    return timing.value_or(InstructionTiming{});
  }

  std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
};

TEST_F(InstructionTimingTest, ClassifiesBlocksAndSerialization) {
  ASSERT_NE(decoder, nullptr);
  const auto salu = classify({kMovS5Seven, 0, 0, 0});
  EXPECT_EQ(salu.block, ExecutionBlock::SALU);
  EXPECT_EQ(salu.pc, 0x100u);
  EXPECT_EQ(salu.size, 4u);
  EXPECT_FALSE(salu.serializing);

  const auto valu = classify({kAddV2V1V0, 0, 0, 0});
  EXPECT_EQ(valu.block, ExecutionBlock::VALU);
  EXPECT_TRUE(valu.hazards.contains({RegClass::VGPR, 1, 1}));
  EXPECT_TRUE(valu.writes.contains({RegClass::VGPR, 2, 1}));
  EXPECT_FALSE(valu.writes.contains({RegClass::VGPR, 1, 1}));
  EXPECT_FALSE(valu.transcendental);
  EXPECT_TRUE(classify({kSqrtV6V0, 0, 0, 0}).transcendental);

  const auto branch = classify({kBranchSkipOne, 0, 0, 0});
  EXPECT_EQ(branch.block, ExecutionBlock::BRANCH);
  EXPECT_TRUE(branch.redirects_fetch);
  EXPECT_TRUE(branch.serializing);

  EXPECT_TRUE(classify({kEndpgm, 0, 0, 0}).ends_program);

  const auto wait = classify({kWaitLoadcnt0, 0, 0, 0});
  EXPECT_TRUE(wait.waitcnt);
  EXPECT_TRUE(wait.serializing);
}

TEST_F(InstructionTimingTest, CountsMemoryCounterIncrements) {
  ASSERT_NE(decoder, nullptr);
  const auto smem = classify({kSLoadS7, kSLoadS7Hi, 0, 0});
  EXPECT_EQ(smem.block, ExecutionBlock::SMEM);
  ASSERT_EQ(smem.counters.size, 1u);

  const auto lds = classify({kDsLoadV1, kDsLoadV1Hi, 0, 0});
  EXPECT_EQ(lds.block, ExecutionBlock::LDS);
  ASSERT_EQ(lds.counters.size, 1u);
  EXPECT_EQ(lds.counters.types[0], WaitCounterType::DSCNT);

  const auto load = classify({kGlobalLoadV4[0], kGlobalLoadV4[1], kGlobalLoadV4[2], 0});
  EXPECT_EQ(load.block, ExecutionBlock::VMEM);
  EXPECT_EQ(load.size, 12u);
  ASSERT_EQ(load.counters.size, 1u);
  EXPECT_EQ(load.counters.types[0], WaitCounterType::LOADCNT);
  EXPECT_TRUE(load.writes.contains({RegClass::VGPR, 4, 1}));

  EXPECT_EQ(classify({kMovS5Seven, 0, 0, 0}).counters.size, 0u);
}

class WaveSlotsTest : public ::testing::Test {
protected:
  void SetUp() override {
    l2.set_backing_memory(&memory);
    ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_RDNA4;
    config.num_wf_slots = 2;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cu = ComputeUnitCore::create("wave_slot_cu", config, &memory, &l2);
    ASSERT_NE(cu, nullptr);
    for (size_t i = 0; i < 4; ++i)
      memory.write32(kProgram + i * 4, kEndpgm);
    slots = std::make_unique<WaveSlots>(*cu, 3);
  }

  void TearDown() override {
    for (uint32_t slot = 0; cu && slot < cu->num_wf_slots(); ++slot)
      if (Wavefront *wave = cu->wf(slot); wave && !wave->is_halted())
        wave->halt();
  }

  Wavefront *dispatch() { return cu->dispatch_wf(0, kProgram, 104, 256); }

  /// A VALU instruction writing @p dst and reading @p src.
  static InstructionTiming valu(uint16_t dst, uint16_t src) {
    InstructionTiming inst;
    inst.block = ExecutionBlock::VALU;
    inst.writes.expand({RegClass::VGPR, dst, 1});
    inst.hazards.expand({RegClass::VGPR, src, 1});
    inst.hazards.expand({RegClass::VGPR, dst, 1});
    return inst;
  }

  GpuMemory memory{"wave_slot_memory"};
  L2Cache l2{"wave_slot_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  std::unique_ptr<WaveSlots> slots;
};

TEST_F(WaveSlotsTest, LaunchFillsTheDispatchedSlotInArrivalOrder) {
  ASSERT_EQ(slots->size(), 2u);
  EXPECT_FALSE((*slots)[0].active());
  Wavefront *first = dispatch();
  Wavefront *second = dispatch();
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  WaveSlot &slot0 = (*slots)[first->wf_id()];
  WaveSlot &slot1 = (*slots)[second->wf_id()];
  EXPECT_TRUE(slot0.active());
  EXPECT_EQ(slot0.wf, first);
  EXPECT_EQ(slot0.generation, first->dispatch_generation());
  EXPECT_TRUE(slot0.frontend_empty());
  EXPECT_LT(slot0.age, slot1.age);

  const WaveKey key = slots->key(first->wf_id());
  EXPECT_EQ(key.cu, 3u);
  EXPECT_EQ(key.slot, first->wf_id());
  EXPECT_EQ(key.generation, first->dispatch_generation());
  EXPECT_EQ(slots->live(key), &slot0);
  EXPECT_EQ(slots->live({.cu = 0, .slot = key.slot, .generation = key.generation}), nullptr);
}

TEST_F(WaveSlotsTest, KeysGoStaleWhenTheWaveLeavesOrIsSquashed) {
  Wavefront *wave = dispatch();
  ASSERT_NE(wave, nullptr);
  const WaveKey key = slots->key(wave->wf_id());
  WaveSlot &slot = (*slots)[key.slot];
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

TEST_F(WaveSlotsTest, RegisterWritesBecomeVisibleAtWriteback) {
  ASSERT_NE(dispatch(), nullptr);
  WaveSlot &slot = (*slots)[0];
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

TEST_F(WaveSlotsTest, CancelAndSquashReleaseOnlyUnissuedReservations) {
  ASSERT_NE(dispatch(), nullptr);
  WaveSlot &slot = (*slots)[0];
  const InstructionTiming consumer = valu(3, 2);

  slot.reserve(valu(2, 1), 1);
  slot.cancel(1, 0);
  EXPECT_FALSE(slot.has_hazard(consumer, 0));

  slot.reserve(valu(2, 1), 2);
  slot.issued(2);
  slot.reserve(valu(3, 0), 3);
  slot.squash(/*flush=*/false);
  EXPECT_EQ(slot.reserved, 0u);
  // The issued write survives the squash; the unissued one is gone.
  EXPECT_TRUE(slot.has_hazard(consumer, 0));
  EXPECT_FALSE(slot.has_hazard(valu(6, 3), 0));
}

TEST_F(WaveSlotsTest, CounterWaitHoldsUntilTheMemoryResponseReturns) {
  ASSERT_NE(dispatch(), nullptr);
  WaveSlot &slot = (*slots)[0];
  InstructionTiming load;
  load.block = ExecutionBlock::VMEM;
  load.writes.expand({RegClass::VGPR, 4, 1});
  load.counters.types[load.counters.size++] = WaitCounterType::LOADCNT;

  slot.reserve(load, 7);
  EXPECT_EQ(slot.memory_tokens, 1u);
  slot.issued(7);
  slot.issued_memory(load);
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

TEST_F(WaveSlotsTest, ReleasesTheLaunchHandler) {
  slots.reset();
  Wavefront *wave = dispatch();
  ASSERT_NE(wave, nullptr);
  WaveSlots fresh(*cu, 0);
  EXPECT_FALSE(fresh[wave->wf_id()].active());
}

} // namespace
