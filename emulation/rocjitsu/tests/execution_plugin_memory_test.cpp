// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin_memory_test.cpp
/// @brief Memory completion, instruction metadata, and routed-access observation tests.

#include "execution_plugin_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ds_transpose.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
using namespace rocjitsu::test::execution_plugin;

class TestGlobalMemPipeline : public GlobalMemPipeline {
public:
  using GlobalMemPipeline::GlobalMemPipeline;
  using GlobalMemPipeline::initiate_access;
};

class TestLocalMemPipeline : public LocalMemPipeline {
public:
  using LocalMemPipeline::initiate_access;
};

class TestScalarMemPipeline : public ScalarMemPipeline {
public:
  using ScalarMemPipeline::ScalarMemPipeline;
};

TEST(ExecutionPluginTest, MemoryPipelineCompletionDoesNotObserveInstructionWrite) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kAddress = 0x8000;
  constexpr uint32_t kLoadedValue = 0x12345678u;
  constexpr uint32_t kDst = 3;
  f.mem->load_image(reinterpret_cast<const uint8_t *>(&kLoadedValue), sizeof(kLoadedValue),
                    kAddress);

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->elem_size = sizeof(kLoadedValue);
  state->num_elems = 1;
  state->is_load = true;
  state->wf_size = wf->wf_size();
  state->exec_mask = 1;
  state->lane_mask = 1;
  state->dst_reg_base = wf->vgpr_alloc().base + kDst;
  state->per_lane_addr[0] = kAddress;

  plugin->events.clear();
  GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_TRUE(vgpr_write_events(*plugin).empty());
  EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + kDst, 0), kLoadedValue);
}

TEST(ExecutionPluginTest, MemoryPipelineWideDwordCompletionWritesSparseLanesDirectly) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kLaneMask = (uint64_t{1} << 1) | (uint64_t{1} << 5);
  constexpr uint64_t kLane1Address = 0x9000;
  constexpr uint64_t kLane5Address = 0xA000;
  constexpr uint32_t kDst = 7;
  constexpr uint32_t kSentinel = 0xA5A5A5A5u;
  constexpr uint32_t kInactiveLane = 2;
  constexpr std::array<uint32_t, 4> kLane1Values = {0x11111111u, 0x22222222u, 0x33333333u,
                                                    0x44444444u};
  constexpr std::array<uint32_t, 4> kLane5Values = {0x55555555u, 0x66666666u, 0x77777777u,
                                                    0x88888888u};
  f.mem->load_image(reinterpret_cast<const uint8_t *>(kLane1Values.data()), sizeof(kLane1Values),
                    kLane1Address);
  f.mem->load_image(reinterpret_cast<const uint8_t *>(kLane5Values.data()), sizeof(kLane5Values),
                    kLane5Address);
  for (uint32_t reg = 0; reg < 4; ++reg) {
    cu->write_vgpr(wf->vgpr_alloc().base + kDst + reg, 0, kSentinel);
    cu->write_vgpr(wf->vgpr_alloc().base + kDst + reg, kInactiveLane, kSentinel);
  }

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->elem_size = sizeof(uint32_t);
  state->num_elems = 4;
  state->is_load = true;
  state->wf_size = wf->wf_size();
  state->exec_mask = kLaneMask;
  state->lane_mask = kLaneMask;
  state->dst_reg_base = wf->vgpr_alloc().base + kDst;
  state->per_lane_addr[1] = kLane1Address;
  state->per_lane_addr[5] = kLane5Address;

  plugin->events.clear();
  GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_TRUE(vgpr_write_events(*plugin).empty());
  for (uint32_t reg = 0; reg < 4; ++reg) {
    EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + kDst + reg, 0), kSentinel);
    EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + kDst + reg, 1), kLane1Values[reg]);
    EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + kDst + reg, kInactiveLane), kSentinel);
    EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + kDst + reg, 5), kLane5Values[reg]);
  }
}

TEST(ExecutionPluginTest, FlatStoreDwordx4ReportsEverySourceRegisterAndActiveLane) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kLaneMask = (uint64_t{1} << 1) | (uint64_t{1} << 5);
  constexpr uint32_t kAddress = 0;
  constexpr uint32_t kData = 4;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  wf->set_exec(kLaneMask);
  for (uint32_t lane : {1u, 5u}) {
    cu->write_vgpr(vgpr_base + kAddress, lane, 0x1000u + lane * 16);
    cu->write_vgpr(vgpr_base + kAddress + 1, lane, 0);
    for (uint32_t reg = 0; reg < 4; ++reg)
      cu->write_vgpr(vgpr_base + kData + reg, lane, lane * 0x100u + reg);
  }

  const auto words =
      cdna4::build_flat(cdna4::kFlatStoreDwordx4Flat, {.addr = kAddress, .data = kData});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> store(decode_valid(*decoder, words.data()));
  ASSERT_NE(store, nullptr);

  plugin->events.clear();
  EXPECT_TRUE(cu->execute_instruction(store.get(), *wf).succeeded());

  expect_vgpr_read_set(vgpr_read_events(*plugin), vgpr_base, {0, 1, 4, 5, 6, 7}, kLaneMask);
  const auto *state = store->data_as<VectorMemState>();
  ASSERT_NE(state, nullptr);
  for (uint32_t lane : {1u, 5u}) {
    for (uint32_t reg = 0; reg < 4; ++reg) {
      uint32_t stored = 0;
      std::memcpy(&stored, state->store_data.data() + (lane * 4 + reg) * sizeof(uint32_t),
                  sizeof(stored));
      EXPECT_EQ(stored, lane * 0x100u + reg);
    }
  }
}

TEST(ExecutionPluginTest, MemoryPipelineHoldsDecodedCountersUntilCompletion) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *wf = f.cu()->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->wait_counter_type = WaitCounterType::VMCNT;
  CounterObservingPipeline pipeline;
  pipeline.issue(new TestMemoryInstruction(std::move(state), "test_mem", WaitCounterType::LGKMCNT),
                 *wf);

  EXPECT_EQ(pipeline.counters_at_access.vmcnt, 1);
  EXPECT_EQ(pipeline.counters_at_access.lgkmcnt, 1);
  EXPECT_EQ(wf->wait_counters().vmcnt, 0);
  EXPECT_EQ(wf->wait_counters().lgkmcnt, 0);
}

TEST(ExecutionPluginTest, MemoryPipelineHoldsDecodedStoreCountersUntilCompletion) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *wf = f.cu()->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->wait_counter_type = WaitCounterType::VMCNT;
  CounterObservingPipeline pipeline;
  pipeline.issue(
      new TestMemoryInstruction(std::move(state),
                                {{WaitCounterType::VMCNT, MemoryCompletionClass::VMEM},
                                 {WaitCounterType::LGKMCNT, MemoryCompletionClass::LDS},
                                 {WaitCounterType::EXPCNT, MemoryCompletionClass::UNORDERED}}),
      *wf);

  EXPECT_EQ(pipeline.counters_at_access.vmcnt, 1);
  EXPECT_EQ(pipeline.counters_at_access.lgkmcnt, 1);
  EXPECT_EQ(pipeline.counters_at_access.expcnt, 1);
  EXPECT_EQ(wf->wait_counters().vmcnt, 0);
  EXPECT_EQ(wf->wait_counters().lgkmcnt, 0);
  EXPECT_EQ(wf->wait_counters().expcnt, 0);
}

TEST(ExecutionPluginTest, MemoryPipelineHonorsWideScalarCounterIncrement) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *wf = f.cu()->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  auto state = std::make_unique<ScalarMemState>();
  state->wait_counter_type = WaitCounterType::LGKMCNT;
  CounterObservingPipeline pipeline;
  pipeline.issue(new TestMemoryInstruction(
                     std::move(state),
                     {{WaitCounterType::LGKMCNT, MemoryCompletionClass::UNORDERED, 2}}, false),
                 *wf);

  EXPECT_EQ(pipeline.counters_at_access.lgkmcnt, 2);
  EXPECT_EQ(wf->wait_counters().lgkmcnt, 0);
}

TEST(ExecutionPluginTest, MemoryPipelineCompletionDoesNotCrossWaveVgprBlock) {
  constexpr uint32_t kVgprsPerWave = 16;
  PluginFixture f(/*num_wf_slots=*/2, /*arch=*/"rdna4", /*wavefront_size=*/32,
                  /*sgprs_per_wf=*/128, kVgprsPerWave);
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/128, kVgprsPerWave, 32);
  auto *adjacent = cu->dispatch_wf(1, 0, /*sgprs=*/128, kVgprsPerWave, 32);
  ASSERT_NE(wf, nullptr);
  ASSERT_NE(adjacent, nullptr);
  ASSERT_EQ(adjacent->vgpr_alloc().base, wf->vgpr_alloc().base + kVgprsPerWave);

  constexpr uint64_t kAddress = 0x8100;
  constexpr std::array<uint32_t, 2> kLoaded = {0x11112222u, 0x33334444u};
  constexpr uint32_t kLastSentinel = 0xA5A5A5A5u;
  constexpr uint32_t kAdjacentSentinel = 0x5A5A5A5Au;
  f.mem->load_image(reinterpret_cast<const uint8_t *>(kLoaded.data()), sizeof(kLoaded), kAddress);
  cu->write_vgpr(wf->vgpr_alloc().base + kVgprsPerWave - 1, 0, kLastSentinel);
  cu->write_vgpr(adjacent->vgpr_alloc().base, 0, kAdjacentSentinel);

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->elem_size = sizeof(uint32_t);
  state->num_elems = 2;
  state->is_load = true;
  state->wf_size = wf->wf_size();
  state->exec_mask = 1;
  state->lane_mask = 1;
  state->dst_reg_base = wf->vgpr_alloc().base + kVgprsPerWave - 1;
  state->per_lane_addr[0] = kAddress;

  GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + kVgprsPerWave - 1, 0), kLastSentinel);
  EXPECT_EQ(cu->read_vgpr_storage(adjacent->vgpr_alloc().base, 0), kAdjacentSentinel);
}

TEST(ExecutionPluginTest, DualOffsetCompletionRejectsBothDestinationsBeforeWriteback) {
  constexpr uint32_t kVgprsPerWave = 16;
  PluginFixture f(/*num_wf_slots=*/2, /*arch=*/"rdna4", /*wavefront_size=*/32,
                  /*sgprs_per_wf=*/128, kVgprsPerWave);
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/128, kVgprsPerWave, 32);
  auto *adjacent = cu->dispatch_wf(1, 0, /*sgprs=*/128, kVgprsPerWave, 32);
  ASSERT_NE(wf, nullptr);
  ASSERT_NE(adjacent, nullptr);
  ASSERT_EQ(adjacent->vgpr_alloc().base, wf->vgpr_alloc().base + kVgprsPerWave);

  constexpr std::array<uint64_t, 2> kLoaded = {0x1111222233334444ULL, 0x5555666677778888ULL};
  constexpr uint32_t kFirstDst = 3;
  constexpr uint32_t kFirstSentinelLo = 0xA5A5A5A5u;
  constexpr uint32_t kFirstSentinelHi = 0xB6B6B6B6u;
  constexpr uint32_t kSecondSentinel = 0xC7C7C7C7u;
  constexpr uint32_t kAdjacentSentinel = 0xD8D8D8D8u;
  wf->lds().write(uint32_t{0}, reinterpret_cast<const uint8_t *>(kLoaded.data()),
                  static_cast<uint32_t>(sizeof(kLoaded)));
  cu->write_vgpr(wf->vgpr_alloc().base + kFirstDst, 0, kFirstSentinelLo);
  cu->write_vgpr(wf->vgpr_alloc().base + kFirstDst + 1, 0, kFirstSentinelHi);
  cu->write_vgpr(wf->vgpr_alloc().base + kVgprsPerWave - 1, 0, kSecondSentinel);
  cu->write_vgpr(adjacent->vgpr_alloc().base, 0, kAdjacentSentinel);

  auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
  state->elem_size = sizeof(uint64_t);
  state->num_elems = 1;
  state->is_load = true;
  state->wf_size = wf->wf_size();
  state->exec_mask = 1;
  state->lane_mask = 1;
  state->dst_reg_base = wf->vgpr_alloc().base + kFirstDst;
  state->per_lane_addr[0] = 0;
  state->ds2_active = true;
  state->ds2_dst_reg_base = wf->vgpr_alloc().base + kVgprsPerWave - 1;
  state->ds2_per_lane_addr[0] = sizeof(uint64_t);

  LocalMemPipeline pipeline;
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + kFirstDst, 0), kFirstSentinelLo);
  EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + kFirstDst + 1, 0), kFirstSentinelHi);
  EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + kVgprsPerWave - 1, 0), kSecondSentinel);
  EXPECT_EQ(cu->read_vgpr_storage(adjacent->vgpr_alloc().base, 0), kAdjacentSentinel);
}

TEST(ExecutionPluginTest, DualOffsetLoadsPreserveSparsePhysicalDestinations) {
  constexpr uint32_t kSentinel = 0xa55a3cc3;
  for (const auto arch : {std::string_view{"cdna4"}, std::string_view{"cdna5"}}) {
    const uint32_t wave = arch == "cdna4" ? 64 : 32;
    const uint64_t wave_mask = wave == 64 ? ~uint64_t{0} : 0xffffffffull;
    PluginFixture f(/*num_wf_slots=*/1, arch, wave, /*sgprs_per_wf=*/104,
                    /*vgprs_per_wf=*/256);
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256, wave);
    ASSERT_NE(wf, nullptr);
    ASSERT_EQ(wf->vgpr_alloc().count, 256u);
    ASSERT_EQ(cu->vgpr_allocation_block_size(), arch == "cdna4" ? 512u : 256u);
    auto *plugin = f.attach_ordering_plugin();
    std::array<uint32_t, 64 * 4> payload;
    for (uint32_t lane = 0; lane < wave; ++lane)
      for (uint32_t word = 0; word < 4; ++word)
        payload[lane * 4 + word] = 0x12340000 + lane * 4 + word;
    wf->lds().write(uint32_t{0}, reinterpret_cast<const uint8_t *>(payload.data()), wave * 16);
    for (uint32_t bytes : {4u, 8u})
      for (uint32_t bank : {0u, 256u}) {
        if (arch != "cdna4" && bank != 0)
          continue;
        for (uint64_t mask : {uint64_t{0}, uint64_t{0xa55aa55aa55aa55a} & wave_mask, wave_mask}) {
          SCOPED_TRACE(testing::Message()
                       << arch << " bytes=" << bytes << " bank=" << bank << " mask=" << mask);
          const uint32_t first_destination = wf->vgpr_alloc().base + bank + 4;
          const uint32_t second_destination = wf->vgpr_alloc().base + bank + 12;
          for (uint32_t lane = 0; lane < wave; ++lane)
            for (uint32_t word = 0; word < 2; ++word) {
              cu->write_vgpr(first_destination + word, lane, kSentinel);
              cu->write_vgpr(second_destination + word, lane, kSentinel);
            }
          auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
          state->elem_size = bytes;
          state->num_elems = 1;
          state->is_load = true;
          state->wf_size = wave;
          state->exec_mask = mask;
          state->lane_mask = mask;
          state->dst_reg_base = first_destination;
          state->ds2_active = true;
          state->ds2_dst_reg_base = second_destination;
          for (uint32_t lane = 0; lane < wave; ++lane) {
            state->per_lane_addr[lane] = lane * 16;
            state->ds2_per_lane_addr[lane] = lane * 16 + 8;
          }
          plugin->events.clear();
          LocalMemPipeline pipeline;
          EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf),
                    VmAccessOutcome::Complete);
          for (uint32_t lane = 0; lane < wave; ++lane)
            for (uint32_t word = 0; word < 2; ++word) {
              const bool written = ((mask >> lane) & 1) && word < bytes / 4;
              EXPECT_EQ(cu->read_vgpr_storage(first_destination + word, lane),
                        written ? payload[lane * 4 + word] : kSentinel);
              EXPECT_EQ(cu->read_vgpr_storage(second_destination + word, lane),
                        written ? payload[lane * 4 + 2 + word] : kSentinel);
            }
          EXPECT_TRUE(std::ranges::none_of(plugin->events, [](const HookEvent &event) {
            return event.kind == HookEvent::WRITE_VGPR;
          }));
        }
      }
  }
}

TEST(ExecutionPluginTest, DualOffsetLoadsRejectThePhysicalBlockBoundary) {
  constexpr uint32_t kSentinel = 0xa55a'3cc3u;
  for (const auto arch : {std::string_view{"cdna4"}, std::string_view{"cdna5"}}) {
    const uint32_t wave = arch == "cdna4" ? 64 : 32;
    PluginFixture f(/*num_wf_slots=*/2, arch, wave, /*sgprs_per_wf=*/104,
                    /*vgprs_per_wf=*/256);
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256, wave);
    auto *adjacent = cu->dispatch_wf(1, 0, /*sgprs=*/104, /*vgprs=*/256, wave);
    ASSERT_NE(wf, nullptr);
    ASSERT_NE(adjacent, nullptr);
    const uint32_t block_size = cu->vgpr_allocation_block_size();
    ASSERT_EQ(block_size, arch == "cdna4" ? 512u : 256u);
    ASSERT_EQ(adjacent->vgpr_alloc().base, wf->vgpr_alloc().base + block_size);
    const uint32_t first = wf->vgpr_alloc().base + (arch == "cdna4" ? 256 : 0) + 4;
    for (uint32_t bytes : {4u, 8u}) {
      // B32 starts in the adjacent wave; B64 straddles the physical boundary.
      const uint32_t second = adjacent->vgpr_alloc().base - bytes / 4 + 1;
      for (uint32_t reg : {first, first + 1, second, second + 1})
        cu->write_vgpr(reg, 0, kSentinel);
      auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
      state->elem_size = bytes;
      state->num_elems = 1;
      state->is_load = true;
      state->wf_size = wave;
      state->exec_mask = 1;
      state->lane_mask = 1;
      state->dst_reg_base = first;
      state->ds2_active = true;
      state->ds2_dst_reg_base = second;
      state->per_lane_addr[0] = 0;
      state->ds2_per_lane_addr[0] = 0;
      LocalMemPipeline pipeline;
      pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);
      for (uint32_t reg : {first, first + 1, second, second + 1})
        EXPECT_EQ(cu->read_vgpr_storage(reg, 0), kSentinel)
            << arch << " bytes=" << bytes << " register=" << reg;
    }
  }
}

TEST(ExecutionPluginTest, EmptyDualLoadDoesNotMaterializeSecondDestination) {
  auto check = []<typename Isa>(std::string_view arch, uint32_t wave) {
    for (uint32_t bank : {0u, 256u}) {
      if (arch != "cdna4" && bank != 0)
        continue;
      PluginFixture f(/*num_wf_slots=*/1, arch, wave, /*sgprs_per_wf=*/104,
                      /*vgprs_per_wf=*/256);
      using Cu = IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, Isa>;
      auto *cu = static_cast<Cu *>(f.cu());
      auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256, wave);
      ASSERT_NE(wf, nullptr);
      const uint32_t first = wf->vgpr_alloc().base + 4;
      const uint32_t second = wf->vgpr_alloc().base + bank + 64;
      // Isolate the new second-destination path from the existing first-path
      // allocation behavior by placing it in a separate, untouched chunk.
      cu->write_vgpr(first, 0, 7);
      const size_t initial_chunks = cu->vgpr_file().materialized_chunk_count();
      for (uint64_t mask : {uint64_t{0}, uint64_t{1}}) {
        auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
        state->elem_size = 4;
        state->num_elems = 1;
        state->is_load = true;
        state->wf_size = wave;
        state->exec_mask = mask;
        state->lane_mask = mask;
        state->dst_reg_base = first;
        state->ds2_active = true;
        state->ds2_dst_reg_base = second;
        state->per_lane_addr[0] = 0;
        state->ds2_per_lane_addr[0] = 0;
        LocalMemPipeline pipeline;
        ASSERT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf),
                  VmAccessOutcome::Complete);
        EXPECT_EQ(cu->vgpr_file().materialized_chunk_count(), initial_chunks + (mask != 0));
        EXPECT_EQ(cu->read_vgpr_storage(first, 0), mask == 0 ? 7u : 0u);
        EXPECT_EQ(cu->read_vgpr_storage(second, 0), 0u);
      }
    }
  };
  check.operator()<cdna4::Isa>("cdna4", 64);
  check.operator()<cdna5::Isa>("cdna5", 32);
}

TEST(ExecutionPluginTest, ScalarMemoryCompletionDoesNotObserveInstructionWrite) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kAddress = 0x8200;
  constexpr uint32_t kLoadedValue = 0x12345678u;
  constexpr uint32_t kDst = 3;
  f.mem->load_image(reinterpret_cast<const uint8_t *>(&kLoadedValue), sizeof(kLoadedValue),
                    kAddress);

  auto state = std::make_unique<ScalarMemState>();
  state->addr = kAddress;
  state->dst_register = {ScalarRegisterStorage::SGPR, kDst, 1};
  state->num_dwords = 1;
  state->is_load = true;

  plugin->events.clear();
  TestScalarMemPipeline pipeline(&cu->l1_scalar());
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_EQ(cu->read_sgpr_storage(wf->sgpr_alloc().base + kDst), kLoadedValue);
  EXPECT_TRUE(std::ranges::none_of(
      plugin->events, [](const HookEvent &event) { return event.kind == HookEvent::WRITE_SGPR; }));

  constexpr uint64_t kTtmpAddress = 0x8300;
  constexpr uint32_t kTtmpValue = 0xAABBCCDDu;
  f.mem->load_image(reinterpret_cast<const uint8_t *>(&kTtmpValue), sizeof(kTtmpValue),
                    kTtmpAddress);
  auto ttmp_state = std::make_unique<ScalarMemState>();
  ttmp_state->addr = kTtmpAddress;
  ttmp_state->dst_register = {ScalarRegisterStorage::TTMP, 0, 1};
  ttmp_state->num_dwords = 1;
  ttmp_state->is_load = true;
  pipeline.issue(new TestMemoryInstruction(std::move(ttmp_state)), *wf);

  EXPECT_EQ(wf->ttmp(0), kTtmpValue);

  constexpr uint64_t kVccAddress = 0x8400;
  constexpr uint32_t kVccValue = 0x55667788u;
  f.mem->load_image(reinterpret_cast<const uint8_t *>(&kVccValue), sizeof(kVccValue), kVccAddress);
  wf->set_vcc_raw(0xAABBCCDD00000000ull);
  auto vcc_state = std::make_unique<ScalarMemState>();
  vcc_state->addr = kVccAddress;
  vcc_state->dst_register = {ScalarRegisterStorage::VCC, 0, 1};
  vcc_state->num_dwords = 1;
  vcc_state->is_load = true;
  pipeline.issue(new TestMemoryInstruction(std::move(vcc_state)), *wf);

  EXPECT_EQ(wf->vcc(), 0xAABBCCDD55667788ull);
  EXPECT_TRUE(std::ranges::none_of(
      plugin->events, [](const HookEvent &event) { return event.kind == HookEvent::WRITE_SGPR; }));
}

TEST(ExecutionPluginTest, Gfx1250ScalarMemoryRoutesSpecialSelectorsAtNonzeroSgprBase) {
  constexpr uint32_t kSgprsPerWave = 128;
  PluginFixture f(/*num_wf_slots=*/2, /*arch=*/"cdna5", /*wavefront_size=*/32, kSgprsPerWave);
  auto *cu = f.cu();
  auto *first = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, kSgprsPerWave, /*vgprs=*/32);
  auto *wf = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0, kSgprsPerWave, /*vgprs=*/32);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(first->sgpr_alloc().base, 0u);
  ASSERT_EQ(wf->sgpr_alloc().base, kSgprsPerWave);

  constexpr uint32_t kVccLoSlotSentinel = 0x10610610u;
  constexpr uint32_t kVccHiSlotSentinel = 0x10710710u;
  constexpr uint32_t kNullSlotSentinel = 0x12412410u;
  const uint32_t base = wf->sgpr_alloc().base;
  cu->write_sgpr(base + kVccSelectorFirst, kVccLoSlotSentinel);
  cu->write_sgpr(base + kVccSelectorLast, kVccHiSlotSentinel);
  cu->write_sgpr(base + kModernNullSelector, kNullSlotSentinel);

  constexpr uint64_t kDataAddress = 0x8800;
  constexpr std::array<uint32_t, 7> kData = {
      0x12345678u, 0xA5B6C7D8u, 0x89ABCDEFu, 0x01234567u, 0xCAFEBABEu, 0x0BADF00Du, 0xDEADBEEFu,
  };
  f.mem->load_image(reinterpret_cast<const uint8_t *>(kData.data()), sizeof(kData), kDataAddress);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  TestScalarMemPipeline pipeline(&cu->l1_scalar());
  auto issue_load = [&](uint16_t op, uint8_t sdata, uint64_t address, uint8_t sbase = 0) {
    cu->write_sgpr(base, static_cast<uint32_t>(address));
    cu->write_sgpr(base + 1, static_cast<uint32_t>(address >> 32));
    const auto words =
        cdna5::build_smem(op, {.sbase = sbase,
                               .sdata = sdata,
                               .scale_offset = 1,
                               .soffset = static_cast<uint8_t>(kModernNullSelector)});
    std::unique_ptr<Instruction> load(decode_valid(*decoder, words.data()));
    EXPECT_NE(load, nullptr);
    if (!load)
      return;
    EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
    EXPECT_NE(load->data(), nullptr);
    if (!load->data())
      return;
    pipeline.issue(load.release(), *wf);
  };

  wf->set_vcc_raw(0x1122334455667788ull);
  issue_load(cdna5::kSLoadB32Smem, kVccSelectorFirst, kDataAddress);
  EXPECT_EQ(wf->vcc(), 0x1122334412345678ull);
  issue_load(cdna5::kSLoadB32Smem, kVccSelectorLast, kDataAddress + 4);
  EXPECT_EQ(wf->vcc(), 0xA5B6C7D812345678ull);
  issue_load(cdna5::kSLoadB64Smem, kVccSelectorFirst, kDataAddress + 8);
  EXPECT_EQ(wf->vcc(), 0x0123456789ABCDEFull);

  issue_load(cdna5::kSLoadB32Smem, kTtmpSelectorFirst, kDataAddress + 16);
  EXPECT_EQ(wf->ttmp(0), kData[4]);
  issue_load(cdna5::kSLoadB32Smem, 4, kDataAddress + 20);
  EXPECT_EQ(cu->read_sgpr_storage(base + 4), kData[5]);
  issue_load(cdna5::kSLoadB32Smem, kModernNullSelector, kDataAddress + 24);

  EXPECT_EQ(cu->read_sgpr_storage(base + kVccSelectorFirst), kVccLoSlotSentinel);
  EXPECT_EQ(cu->read_sgpr_storage(base + kVccSelectorLast), kVccHiSlotSentinel);
  EXPECT_EQ(cu->read_sgpr_storage(base + kModernNullSelector), kNullSlotSentinel);

  constexpr uint64_t kVccBaseAddress = 0x8900;
  constexpr uint64_t kTtmpBaseAddress = 0x8A00;
  constexpr uint32_t kVccBaseValue = 0x13579BDFu;
  constexpr uint32_t kTtmpBaseValue = 0x2468ACE0u;
  f.mem->load_image(reinterpret_cast<const uint8_t *>(&kVccBaseValue), sizeof(kVccBaseValue),
                    kVccBaseAddress);
  f.mem->load_image(reinterpret_cast<const uint8_t *>(&kTtmpBaseValue), sizeof(kTtmpBaseValue),
                    kTtmpBaseAddress);

  wf->set_vcc_raw(kVccBaseAddress);
  issue_load(cdna5::kSLoadB32Smem, 5, /*address=*/0, static_cast<uint8_t>(kVccSelectorFirst / 2));
  EXPECT_EQ(cu->read_sgpr_storage(base + 5), kVccBaseValue);

  wf->set_ttmp(0, static_cast<uint32_t>(kTtmpBaseAddress));
  wf->set_ttmp(1, static_cast<uint32_t>(kTtmpBaseAddress >> 32));
  issue_load(cdna5::kSLoadB32Smem, 6, /*address=*/0, static_cast<uint8_t>(kTtmpSelectorFirst / 2));
  EXPECT_EQ(cu->read_sgpr_storage(base + 6), kTtmpBaseValue);
}

TEST(ExecutionPluginTest, ScalarMemoryRejectsDestinationThatCrossesTtmpFile) {
  PluginFixture f(/*num_wf_slots=*/2);
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  auto *adjacent = cu->dispatch_wf(1, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_NE(adjacent, nullptr);

  constexpr uint32_t kAdjacentOffset = 20;
  constexpr uint32_t kSentinel = 0xA5A5A5A5u;
  cu->write_sgpr(adjacent->sgpr_alloc().base + kAdjacentOffset, kSentinel);
  cu->write_sgpr(wf->sgpr_alloc().base, 0x8000u);
  cu->write_sgpr(wf->sgpr_alloc().base + 1, 0u);

  const auto words = cdna4::build_smem(cdna4::kSLoadDwordx2Smem,
                                       {.sbase = 0, .sdata = amdgpu::kTtmpSelectorLast, .imm = 1});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> load(decode_valid(*decoder, words.data()));
  ASSERT_NE(load, nullptr);

  EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());

  EXPECT_EQ(load->data(), nullptr);
  EXPECT_EQ(cu->read_sgpr_storage(adjacent->sgpr_alloc().base + kAdjacentOffset), kSentinel);
}

TEST(ExecutionPluginTest, MemoryPipelinesSnapshotDispatchIdentityAtIssue) {
  PluginFixture f(/*num_wf_slots=*/4);
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf_at(/*wf_id=*/3, /*wg_id=*/17, /*pc=*/0, /*num_sgprs=*/104,
                                /*num_vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  auto global_state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  global_state->wg_id = 101;
  global_state->wf_id = 202;
  TestMemoryInstruction global_inst(std::move(global_state));
  TestGlobalMemPipeline global_pipeline(&cu->l1_vector(), cu->l2());
  global_pipeline.initiate_access(global_inst, *wf);
  EXPECT_EQ(global_inst.data_as<VectorMemState>()->wg_id, 17u);
  EXPECT_EQ(global_inst.data_as<VectorMemState>()->wf_id, 3u);

  auto local_state = std::make_unique<VectorMemState>(LOCAL_MEM);
  local_state->wg_id = 101;
  local_state->wf_id = 202;
  TestMemoryInstruction local_inst(std::move(local_state));
  TestLocalMemPipeline local_pipeline;
  local_pipeline.initiate_access(local_inst, *wf);
  EXPECT_EQ(local_inst.data_as<VectorMemState>()->wg_id, 17u);
  EXPECT_EQ(local_inst.data_as<VectorMemState>()->wf_id, 3u);
}

TEST(ExecutionPluginTest, D16MemoryCompletionPreservesHalfWithoutObservation) {
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"rdna4", /*wavefront_size=*/32);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_FALSE(cu->sram_ecc());

  constexpr uint64_t kAddress = 0x8100;
  constexpr uint16_t kLoadedValue = 0x1122u;
  constexpr uint32_t kOldDst = 0xAABBCCDDu;
  constexpr uint32_t kDst = 4;
  const uint32_t physical_dst = wf->vgpr_alloc().base + kDst;
  f.mem->load_image(reinterpret_cast<const uint8_t *>(&kLoadedValue), sizeof(kLoadedValue),
                    kAddress);

  GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  for (bool high_half : {false, true}) {
    SCOPED_TRACE(high_half ? "high half" : "low half");
    cu->write_vgpr(physical_dst, 0, kOldDst);

    auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
    state->elem_size = sizeof(kLoadedValue);
    state->num_elems = 1;
    state->is_load = true;
    state->wf_size = wf->wf_size();
    state->exec_mask = 1;
    state->lane_mask = 1;
    state->dst_reg_base = physical_dst;
    state->per_lane_addr[0] = kAddress;
    state->d16_lo = !high_half;
    state->d16_hi = high_half;

    plugin->events.clear();
    pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

    EXPECT_TRUE(vgpr_read_events(*plugin).empty());
    EXPECT_TRUE(vgpr_write_events(*plugin).empty());
    const uint32_t expected = high_half ? 0x1122CCDDu : 0xAABB1122u;
    EXPECT_EQ(cu->read_vgpr_storage(physical_dst, 0), expected);
  }
}

// Routed memory observation.

TEST(RoutedMemoryObservationTest, AScalarAccessCarriesItsFullIdentity) {
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  auto *wave = cu->dispatch_wf_at(/*wf_id=*/3, /*wg_id=*/17, /*pc=*/0x240,
                                  /*num_sgprs=*/104, /*num_vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  wave->set_dispatch_id(23);
  wave->set_queue_id(6);
  wave->set_process_id(9);

  auto state = std::make_unique<ScalarMemState>();
  state->addr = 0x4000;
  state->num_dwords = 4;
  state->elem_size = 4;
  state->is_load = true;
  state->wait_counter_type = WaitCounterType::LGKMCNT;
  test::ComputeUnitTestAccess::route_memory_inst(*cu, new TestMemoryInstruction(std::move(state)),
                                                 *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  const auto &access = plugin->accesses.front();
  EXPECT_EQ(access.mnemonic, "test_mem");
  EXPECT_EQ(access.pc, 0x240u);
  EXPECT_EQ(access.compute_unit_id, cu->id());
  EXPECT_EQ(access.dispatch_id, 23u);
  EXPECT_EQ(access.workgroup_id, 17u);
  EXPECT_EQ(access.wavefront_id, 3u);
  EXPECT_EQ(access.queue_id, 6u);
  // The address space the addresses live in, without which two guests' traffic
  // is indistinguishable.
  EXPECT_EQ(access.process_id, 9u);
  EXPECT_EQ(access.route, MemoryRoute::SCALAR);
  EXPECT_EQ(access.decoded_space, DecodedMemorySpace::SCALAR);
  EXPECT_EQ(std::string(decoded_memory_space_name(access.decoded_space)), "scalar");
  EXPECT_EQ(access.wait_counter, WaitCounterType::LGKMCNT);
  EXPECT_TRUE(access.is_load);
  // A scalar access is one address, reported as a one-lane wavefront so that
  // both routes take the same per-lane arithmetic.
  EXPECT_EQ(access.wavefront_size, 1u);
  EXPECT_EQ(access.active_lane_mask, 1u);
  EXPECT_EQ(access.valid_lane_mask, 1u);
  EXPECT_EQ(access.request_lane_mask, 1u);
  EXPECT_EQ(access.inactive_lane_mask, 0u);
  EXPECT_EQ(access.bytes_per_lane, 16u);
  ASSERT_EQ(access.addresses.size(), 1u);
  EXPECT_EQ(access.addresses[0], 0x4000u);
  EXPECT_TRUE(access.pre_routing_addresses.empty());
}

TEST(RoutedMemoryObservationTest, AFlatAccessToTheSharedApertureIsSeenAsTheLdsAccessItBecame) {
  // The reason the observation is taken after routing rather than before. This
  // instruction decodes as global and is issued to the local pipeline with its
  // addresses rewritten into the workgroup's LDS allocation and its wait
  // counter changed. An observer that looked before routing would charge it
  // against the vector cache, at an address the memory system never uses.
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  constexpr uint64_t kSharedBase = 0x1000'0000;
  cu->set_apertures(kSharedBase, kSharedBase + 0xffff, 0, 0);
  auto *wave = cu->dispatch_wf_at(/*wf_id=*/1, /*wg_id=*/7, /*pc=*/0x300,
                                  /*num_sgprs=*/104, /*num_vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  wave->set_dispatch_id(31);
  wave->set_exec(0b1111);
  wave->set_lds_base(0x400);

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->wf_size = wave->wf_size();
  state->elem_size = 4;
  state->num_elems = 1;
  state->is_load = false;
  state->exec_mask = 0b1111;
  state->lane_mask = 0b0101;
  state->per_lane_addr[0] = kSharedBase + 0x20;
  state->per_lane_addr[2] = kSharedBase + 0x28;
  state->store_data.resize(static_cast<size_t>(state->wf_size) * state->elem_size);
  test::ComputeUnitTestAccess::route_memory_inst(
      *cu, new TestMemoryInstruction(std::move(state), "flat_store_b32"), *wave);

  // What the pre-routing hook was shown.
  ASSERT_EQ(plugin->before_tag.size(), 1u);
  EXPECT_EQ(plugin->before_tag[0], GLOBAL_MEM);
  EXPECT_EQ(plugin->before_first_address[0], kSharedBase + 0x20);
  EXPECT_EQ(plugin->before_wait_counter[0], WaitCounterType::VMCNT);

  // What the memory system is actually asked for.
  ASSERT_EQ(plugin->accesses.size(), 1u);
  const auto &access = plugin->accesses.front();
  EXPECT_EQ(access.route, MemoryRoute::LOCAL);
  EXPECT_EQ(access.decoded_space, DecodedMemorySpace::FLAT);
  EXPECT_EQ(std::string(decoded_memory_space_name(access.decoded_space)), "flat");
  EXPECT_TRUE(access.normalized_to_local);
  EXPECT_EQ(access.wait_counter, WaitCounterType::LGKMCNT);
  EXPECT_EQ(access.addresses[0], 0x420u);
  EXPECT_EQ(access.addresses[2], 0x428u);
  ASSERT_EQ(access.pre_routing_addresses.size(), wave->wf_size());
  EXPECT_EQ(access.pre_routing_addresses[0], kSharedBase + 0x20);
  EXPECT_EQ(access.pre_routing_addresses[2], kSharedBase + 0x28);
  EXPECT_EQ(access.active_lane_mask, 0b1111u);
  EXPECT_EQ(access.valid_lane_mask, 0b0101u);
  EXPECT_EQ(access.flat_local_lane_mask, 0b0101u);
  EXPECT_EQ(access.flat_dds_lane_mask, 0u);
  EXPECT_EQ(access.unknown_lane_mask, 0b1010u);
}

TEST(RoutedMemoryObservationTest, AFlatAccessSeparatesDdsFromLdsLanes) {
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  constexpr uint64_t kSharedBase = 0x1234'0000'0000'0000;
  constexpr uint64_t kDdsAddress = kSharedBase | 0x8000'0040;
  constexpr uint64_t kLdsAddress = kSharedBase | 0x0000'0020;
  cu->set_apertures(kSharedBase, kSharedBase | 0xffff'ffff, 0, 0);
  auto *wave = cu->dispatch_wf_at(/*wf_id=*/1, /*wg_id=*/7, /*pc=*/0x320,
                                  /*num_sgprs=*/104, /*num_vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0b111);
  wave->set_lds_base(0x400);

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->wf_size = wave->wf_size();
  state->elem_size = 4;
  state->num_elems = 1;
  state->is_load = true;
  state->dst_reg_base = wave->vgpr_alloc().base + 2;
  state->exec_mask = 0b111;
  state->lane_mask = 0b111;
  state->per_lane_addr[0] = kDdsAddress;
  state->per_lane_addr[1] = kLdsAddress;
  state->per_lane_addr[2] = 0x2000;
  auto load = prepare_cdna4_flat_load_for_routing(*cu, *wave, std::move(state));
  ASSERT_NE(load, nullptr);
  test::ComputeUnitTestAccess::route_memory_inst(*cu, load.release(), *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  const auto &access = plugin->accesses.front();
  EXPECT_EQ(access.route, MemoryRoute::LOCAL);
  EXPECT_EQ(access.decoded_space, DecodedMemorySpace::FLAT);
  EXPECT_TRUE(access.normalized_to_local);
  EXPECT_EQ(access.flat_local_lane_mask, 0b010u);
  EXPECT_EQ(access.flat_dds_lane_mask, 0b001u);
  EXPECT_EQ(access.scratch_lane_mask, 0u);
  EXPECT_EQ(access.flat_local_lane_mask & access.flat_dds_lane_mask, 0u);
  ASSERT_EQ(access.pre_routing_addresses.size(), wave->wf_size());
  EXPECT_EQ(access.pre_routing_addresses[0], kDdsAddress);
  EXPECT_EQ(access.pre_routing_addresses[1], kLdsAddress);
  EXPECT_EQ(access.pre_routing_addresses[2], 0x2000u);
  auto *scoreboard = wave->memory_wait_scoreboard();
  ASSERT_NE(scoreboard, nullptr);
  EXPECT_EQ(scoreboard->outstanding(WaitCounterKind::Load), 1u);
  EXPECT_EQ(scoreboard->outstanding(WaitCounterKind::Ds), 1u);
  scoreboard->wait(WaitCounterKind::Load, 0);
  const auto shared = access.flat_local_lane_mask | access.flat_dds_lane_mask;
  scoreboard->access({RegClass::VGPR, 2, 1}, access.active_lane_mask & ~shared, 0xf, false);
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 0u);
  scoreboard->access({RegClass::VGPR, 2, 1}, shared, 0xf, false);
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 1u);
}

TEST(RoutedMemoryObservationTest, AFlatAccessRetainsPerLaneLdsRoutingWhenFirstLaneIsGlobal) {
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  constexpr uint64_t kSharedBase = 0x1000'0000;
  cu->set_apertures(kSharedBase, kSharedBase + 0xffff, 0, 0);
  auto *wave = cu->dispatch_wf_at(/*wf_id=*/1, /*wg_id=*/7, /*pc=*/0x340,
                                  /*num_sgprs=*/104, /*num_vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0b1111);

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->wf_size = wave->wf_size();
  state->elem_size = 4;
  state->num_elems = 1;
  state->is_load = true;
  state->dst_reg_base = wave->vgpr_alloc().base + 2;
  state->exec_mask = 0b1111;
  state->lane_mask = 0b0111;
  state->per_lane_addr[0] = 0x2000;
  state->per_lane_addr[1] = kSharedBase + 0x20;
  state->per_lane_addr[2] = kSharedBase + 0x28;
  auto load = prepare_cdna4_flat_load_for_routing(*cu, *wave, std::move(state));
  ASSERT_NE(load, nullptr);
  test::ComputeUnitTestAccess::route_memory_inst(*cu, load.release(), *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  const auto &access = plugin->accesses.front();
  EXPECT_EQ(access.route, MemoryRoute::GLOBAL);
  EXPECT_EQ(access.decoded_space, DecodedMemorySpace::FLAT);
  EXPECT_FALSE(access.normalized_to_local);
  EXPECT_EQ(access.flat_local_lane_mask, 0b0110u);
  EXPECT_TRUE(access.pre_routing_addresses.empty());
  EXPECT_EQ(access.addresses[0], 0x2000u);
  EXPECT_EQ(access.addresses[1], kSharedBase + 0x20);
  EXPECT_EQ(access.addresses[2], kSharedBase + 0x28);
  auto *scoreboard = wave->memory_wait_scoreboard();
  ASSERT_NE(scoreboard, nullptr);
  EXPECT_EQ(scoreboard->outstanding(WaitCounterKind::Load), 1u);
  EXPECT_EQ(scoreboard->outstanding(WaitCounterKind::Ds), 1u);
  scoreboard->wait(WaitCounterKind::Load, 0);
  const auto shared = access.flat_local_lane_mask | access.flat_dds_lane_mask;
  scoreboard->access({RegClass::VGPR, 2, 1}, access.active_lane_mask & ~shared, 0xf, false);
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 0u);
  scoreboard->access({RegClass::VGPR, 2, 1}, shared, 0xf, false);
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 1u);
}

TEST(RoutedMemoryObservationTest, AnExplicitGlobalAccessIgnoresTheSharedAperture) {
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  constexpr uint64_t kSharedBase = 0x1000'0000;
  cu->set_apertures(kSharedBase, kSharedBase + 0xffff, 0, 0);
  auto *wave = cu->dispatch_wf(/*wg_id=*/7, /*pc=*/0x380, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(1);

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->wf_size = wave->wf_size();
  state->elem_size = 4;
  state->num_elems = 1;
  state->is_load = true;
  state->exec_mask = 1;
  state->lane_mask = 1;
  state->per_lane_addr[0] = kSharedBase + 0x20;
  test::ComputeUnitTestAccess::route_memory_inst(
      *cu, new TestMemoryInstruction(std::move(state), "global_load_b32"), *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  const auto &access = plugin->accesses.front();
  EXPECT_EQ(access.route, MemoryRoute::GLOBAL);
  EXPECT_EQ(access.decoded_space, DecodedMemorySpace::GLOBAL);
  EXPECT_FALSE(access.normalized_to_local);
  EXPECT_EQ(access.flat_local_lane_mask, 0u);
  EXPECT_EQ(access.flat_dds_lane_mask, 0u);
  EXPECT_TRUE(access.pre_routing_addresses.empty());
  EXPECT_EQ(access.addresses[0], kSharedBase + 0x20);
}

TEST(RoutedMemoryObservationTest, DecodedDirectLdsLoadsReportBroadcastMaskAndWidth) {
  for (auto arch : {"rdna3", "rdna3_5", "rdna4"}) {
    PluginFixture fixture(1, arch, 32);
    auto *plugin = fixture.attach_memory_observation_plugin();
    auto *cu = fixture.cu();
    // Seed the whole stream before the CU fills its instruction cache.
    for (uint32_t i = 0; i < 10; ++i)
      fixture.mem->write32(0x1000 + 4 * i, 0xce100006); // direct LDS load to v6
    auto *wave = cu->dispatch_wf(0, 0x1000, 104, 256, 32);
    ASSERT_NE(wave, nullptr);
    wave->set_lds_base(4096);
    wave->lds().write32(4096 + 16, 0x918293e4);
    for (uint32_t type : {0u, 1u, 2u, 4u, 5u}) {
      for (uint64_t mask : {uint64_t{0}, uint64_t{0x101}}) {
        SCOPED_TRACE(testing::Message() << arch << ',' << type << ',' << mask);
        const auto pc = wave->pc;
        wave->set_exec(mask);
        wave->set_m0((type << 16) | 16);
        plugin->accesses.clear();
        cu->step();
        ASSERT_EQ(wave->pc, pc + 4);
        ASSERT_EQ(plugin->accesses.size(), mask ? 1u : 0u);
        EXPECT_EQ(wave->exec(), mask);
        EXPECT_TRUE(wave->wait_counters().empty());
        if (!mask)
          continue;
        const auto &access = plugin->accesses.front();
        EXPECT_EQ(access.route, MemoryRoute::LOCAL);
        EXPECT_EQ(access.wait_counter, WaitCounterType::EXPCNT);
        EXPECT_TRUE(access.is_load);
        EXPECT_EQ(access.elements_per_lane, 1u);
        EXPECT_EQ(access.element_size_bytes, type == 2 ? 4u : (type & 1 ? 2u : 1u));
        EXPECT_EQ(access.architectural_exec_lane_mask, mask);
        EXPECT_EQ(access.active_lane_mask, 0xf0fu);
        EXPECT_EQ(access.valid_lane_mask, 0xf0fu);
        EXPECT_EQ(access.request_lane_mask, 0xf0fu);
        EXPECT_EQ(access.addresses[0], 4096u + 16);
        EXPECT_EQ(access.addresses[11], 4096u + 16);
      }
    }
  }
}

TEST(RoutedMemoryObservationTest, AnLdsAccessThatWasAlwaysLdsIsNotMarkedNormalized) {
  // The counterpart to the case above: a consumer separating "traffic that is
  // really LDS" from "traffic rewritten into LDS" needs the two to differ.
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  auto *wave = cu->dispatch_wf(/*wg_id=*/5, /*pc=*/0x400, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0b11);

  auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
  state->wf_size = wave->wf_size();
  state->elem_size = 4;
  state->num_elems = 2;
  state->is_load = false;
  state->exec_mask = 0b11;
  state->lane_mask = 0b01;
  state->per_lane_addr[0] = 0x40;
  state->store_data.resize(static_cast<size_t>(state->wf_size) * state->elem_size * 2);
  state->ds2_active = true;
  state->ds2_per_lane_addr[0] = 0x80;
  state->ds2_store_data.resize(static_cast<size_t>(state->wf_size) * state->elem_size * 2);
  test::ComputeUnitTestAccess::route_memory_inst(*cu, new TestMemoryInstruction(std::move(state)),
                                                 *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  const auto &access = plugin->accesses.front();
  EXPECT_EQ(access.route, MemoryRoute::LOCAL);
  EXPECT_EQ(access.decoded_space, DecodedMemorySpace::LOCAL);
  EXPECT_FALSE(access.normalized_to_local);
  EXPECT_TRUE(access.pre_routing_addresses.empty());
  EXPECT_EQ(access.wait_counter, WaitCounterType::LGKMCNT);
  EXPECT_EQ(access.bytes_per_lane, 8u);
  // A wave64 with two lanes on: counting the other sixty-two would report a
  // full-width access for something that moves two lanes' worth of bytes.
  EXPECT_EQ(access.inactive_lane_mask, ~uint64_t{0} << 2);
  EXPECT_EQ(access.unknown_lane_mask, 0b10u);
  // The second half of a DS dual access is a second set of addresses, not a
  // second instruction.
  ASSERT_EQ(access.secondary_addresses.size(), wave->wf_size());
  EXPECT_EQ(access.secondary_addresses[0], 0x80u);
}

TEST(RoutedMemoryObservationTest, AGlobalAtomicReportsItsOperationScratchAndPolicy) {
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  auto *wave = cu->dispatch_wf_at(/*wf_id=*/2, /*wg_id=*/11, /*pc=*/0x380,
                                  /*num_sgprs=*/104, /*num_vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0b1111);

  constexpr uint64_t kAddress = 0x9000;
  uint32_t initial = 7;
  fixture.mem->load_image(reinterpret_cast<const uint8_t *>(&initial), sizeof(initial), kAddress);
  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->wf_size = wave->wf_size();
  state->elem_size = sizeof(initial);
  state->num_elems = 1;
  state->is_load = false;
  state->atomic_op = AtomicOp::ADD;
  state->non_temporal = true;
  state->exec_mask = 0b1111;
  state->lane_mask = 0b0001;
  state->scratch_swizzle = true;
  state->scratch_lane_mask = 0b0001;
  state->scratch_addr_stride = 256;
  state->per_lane_addr[0] = kAddress;
  state->store_data.resize(static_cast<size_t>(state->wf_size) * sizeof(initial));
  uint32_t increment = 3;
  std::memcpy(state->store_data.data(), &increment, sizeof(increment));
  test::ComputeUnitTestAccess::route_memory_inst(
      *cu, new TestMemoryInstruction(std::move(state), "flat_atomic_add"), *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  const auto &access = plugin->accesses.front();
  EXPECT_EQ(access.route, MemoryRoute::GLOBAL);
  EXPECT_EQ(access.decoded_space, DecodedMemorySpace::FLAT);
  EXPECT_FALSE(access.normalized_to_local);
  EXPECT_TRUE(access.pre_routing_addresses.empty());
  // An atomic is neither a plain load nor a plain store, and a model that
  // treated it as either would put it on the wrong side of a wait counter.
  EXPECT_EQ(access.atomic_op, AtomicOp::ADD);
  EXPECT_FALSE(access.is_load);
  EXPECT_TRUE(access.non_temporal);
  EXPECT_EQ(access.active_lane_mask, 0b1111u);
  EXPECT_EQ(access.valid_lane_mask, 0b0001u);
  EXPECT_EQ(access.unknown_lane_mask, 0b1110u);
  // Swizzled scratch is not contiguous, so the stride has to travel with it.
  EXPECT_EQ(access.scratch_lane_mask, 0b0001u);
  EXPECT_EQ(access.scratch_element_stride_bytes, 256u);
  EXPECT_EQ(access.addresses[0], kAddress);
  // The atomic really ran; the observation describes an access that happened.
  EXPECT_EQ(fixture.mem->read32(kAddress), initial + increment);
}

TEST(RoutedMemoryObservationTest, DedicatedScratchIsDistinctFromFlatScratch) {
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  auto *wave = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0x3C0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0b1);

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->wf_size = wave->wf_size();
  state->elem_size = 4;
  state->num_elems = 1;
  state->is_load = false;
  state->exec_mask = 0b1;
  state->lane_mask = 0b1;
  state->scratch_swizzle = true;
  state->scratch_lane_mask = 0b1;
  state->scratch_addr_stride = wave->wf_size() * sizeof(uint32_t);
  state->per_lane_addr[0] = 0x9800;
  state->store_data.resize(static_cast<size_t>(state->wf_size) * state->elem_size);
  test::ComputeUnitTestAccess::route_memory_inst(
      *cu, new TestMemoryInstruction(std::move(state), "scratch_store_dword"), *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  const auto &access = plugin->accesses.front();
  EXPECT_EQ(access.route, MemoryRoute::GLOBAL);
  EXPECT_EQ(access.decoded_space, DecodedMemorySpace::SCRATCH);
  EXPECT_EQ(std::string(decoded_memory_space_name(access.decoded_space)), "scratch");
  EXPECT_EQ(access.scratch_lane_mask, 0b1u);
  EXPECT_TRUE(access.pre_routing_addresses.empty());
}

TEST(RoutedMemoryObservationTest, ANonScratchAccessReportsNoSwizzleStride) {
  // scratch_addr_stride is left set by whatever last used the state, so the
  // observation has to gate it on the swizzle flag rather than copy it.
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  auto *wave = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0x500, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0b1);

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->wf_size = wave->wf_size();
  state->elem_size = 4;
  state->num_elems = 1;
  state->exec_mask = 0b1;
  state->lane_mask = 0b1;
  state->scratch_swizzle = false;
  state->scratch_lane_mask = 0b1;
  state->scratch_addr_stride = 256;
  state->per_lane_addr[0] = 0xA000;
  test::ComputeUnitTestAccess::route_memory_inst(
      *cu, new TestMemoryInstruction(std::move(state), "buffer_load_dword"), *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  EXPECT_EQ(plugin->accesses.front().decoded_space, DecodedMemorySpace::GLOBAL);
  EXPECT_EQ(plugin->accesses.front().scratch_lane_mask, 0u);
  EXPECT_EQ(plugin->accesses.front().scratch_element_stride_bytes, 0u);
}

TEST(RoutedMemoryObservationTest, TheLaneCountIsTheWavefrontsOwn) {
  // VectorMemState's wf_size defaults to 64 whatever the target, so a
  // wave32 access would otherwise be reported with thirty-two lanes of
  // whatever the array happened to hold.
  PluginFixture fixture(/*num_wf_slots=*/1, /*arch=*/"rdna4", /*wavefront_size=*/32);
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  auto *wave = cu->dispatch_wf(/*wg_id=*/5, /*pc=*/0x440, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  ASSERT_EQ(wave->wf_size(), 32u);
  wave->set_exec(0b11);

  auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
  ASSERT_EQ(state->wf_size, 64u) << "the default this test exists to catch has changed";
  state->elem_size = 4;
  state->num_elems = 1;
  state->exec_mask = 0b11;
  state->lane_mask = 0b11;
  state->per_lane_addr[0] = 0x10;
  state->per_lane_addr[1] = 0x14;
  test::ComputeUnitTestAccess::route_memory_inst(*cu, new TestMemoryInstruction(std::move(state)),
                                                 *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  const auto &access = plugin->accesses.front();
  EXPECT_EQ(access.wavefront_size, 32u);
  EXPECT_EQ(access.addresses.size(), 32u);
  // Thirty of thirty-two, not thirty of sixty-four.
  EXPECT_EQ(access.inactive_lane_mask, 0xFFFF'FFFCu);
}

TEST(RoutedMemoryObservationTest, ATransposeLoadRequestsFromFewerLanesThanItFills) {
  // A wave64 B8 transpose load fills every valid lane but issues its requests
  // through the low half, so a model that charged traffic per valid lane would
  // double it. Its effective all-lane execution must not erase incoming EXEC.
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  auto *wave = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0x700, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  ASSERT_EQ(wave->wf_size(), 64u);
  wave->set_exec(0b0101);

  auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
  state->wf_size = wave->wf_size();
  state->elem_size = 4;
  state->num_elems = 1;
  state->exec_mask = ~uint64_t{0};
  state->lane_mask = ~uint64_t{0};
  state->transpose = static_cast<uint8_t>(TransposeKind::WMMA_TR_B8);
  test::ComputeUnitTestAccess::route_memory_inst(*cu, new TestMemoryInstruction(std::move(state)),
                                                 *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  EXPECT_EQ(plugin->accesses.front().active_lane_mask, ~uint64_t{0});
  EXPECT_EQ(plugin->accesses.front().architectural_exec_lane_mask, 0b0101u);
  EXPECT_EQ(plugin->accesses.front().valid_lane_mask, ~uint64_t{0});
  EXPECT_EQ(plugin->accesses.front().request_lane_mask, 0xFFFF'FFFFu);
}

TEST(RoutedMemoryObservationTest, TheTransposeRequestMaskUsesTheWavefrontsWidth) {
  // The half-issue rule keys on the wavefront's width, and the width recorded
  // in VectorMemState is the pipeline's -- set on some paths only after
  // routing, so it can still hold whatever the last user left. The wavefront
  // is the authority; the state is not.
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  auto *wave = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0x740, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  ASSERT_EQ(wave->wf_size(), 64u);
  wave->set_exec(~uint64_t{0});

  auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
  state->wf_size = 32; // not yet the pipeline's, and not this wavefront's
  state->elem_size = 4;
  state->num_elems = 1;
  state->exec_mask = ~uint64_t{0};
  state->lane_mask = ~uint64_t{0};
  state->transpose = static_cast<uint8_t>(TransposeKind::WMMA_TR_B8);
  test::ComputeUnitTestAccess::route_memory_inst(*cu, new TestMemoryInstruction(std::move(state)),
                                                 *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  EXPECT_EQ(plugin->accesses.front().request_lane_mask, 0xFFFF'FFFFu)
      << "the request mask followed the state's width rather than the wavefront's";
}

TEST(RoutedMemoryObservationTest, PerElementBoundsAndCachePolicyAreCarried) {
  // An access whose later elements go out of bounds while its earlier ones do
  // not moves fewer bytes than its element count suggests, and the cache
  // policy decides which levels it even touches. Both travel with the access.
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  auto *wave = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0x780, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0b1111);

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->wf_size = wave->wf_size();
  state->elem_size = 4;
  state->num_elems = 2;
  state->exec_mask = 0b1111;
  state->lane_mask = 0b0111;
  state->element_lane_masks.assign(2, 0);
  state->element_lane_masks[0] = 0b0111;
  state->element_lane_masks[1] = 0b0011;
  state->mtype = Mtype::UC;
  state->request_force_l1_bypass = true;
  state->lds_dst = true;
  state->per_lane_addr[0] = 0xC000;
  test::ComputeUnitTestAccess::route_memory_inst(*cu, new TestMemoryInstruction(std::move(state)),
                                                 *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  const auto &access = plugin->accesses.front();
  ASSERT_EQ(access.element_lane_masks.size(), 2u);
  EXPECT_EQ(access.element_lane_masks[0], 0b0111u);
  EXPECT_EQ(access.element_lane_masks[1], 0b0011u);
  EXPECT_EQ(access.mtype, Mtype::UC);
  EXPECT_TRUE(access.force_l1_bypass);
  EXPECT_TRUE(access.lds_destination);
}

TEST(RoutedMemoryObservationTest, ASingleAccessCarriesNoSecondAddressSet) {
  // The counterpart to the DS dual-access case: an ordinary access must report
  // an empty second set rather than a stale array of zeroes, which a consumer
  // would otherwise take for sixty-four accesses to address zero.
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  auto *wave = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0x7C0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0b1);

  auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
  state->wf_size = wave->wf_size();
  state->elem_size = 4;
  state->num_elems = 1;
  state->exec_mask = 0b1;
  state->lane_mask = 0b1;
  state->per_lane_addr[0] = 0x20;
  ASSERT_FALSE(state->ds2_active);
  test::ComputeUnitTestAccess::route_memory_inst(*cu, new TestMemoryInstruction(std::move(state)),
                                                 *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  EXPECT_TRUE(plugin->accesses.front().secondary_addresses.empty());
  EXPECT_TRUE(plugin->accesses.front().element_lane_masks.empty());
}

TEST(RoutedMemoryObservationTest, APluginThatDoesNotWantTheHookDoesNotPayForIt) {
  // Building the observation is real work on the per-instruction path. A
  // plugin that never implements the hook -- which is every plugin in the tree
  // but one -- should not be charged for it.
  PluginFixture fixture;
  auto *ordering = fixture.attach_ordering_plugin();
  ASSERT_NE(ordering, nullptr);
  EXPECT_FALSE(fixture.plugin_group().observes_memory_routing());

  ExecutionPluginGroup empty{PluginSinkConfig{}};
  EXPECT_FALSE(empty.observes_memory_routing());

  ExecutionPluginGroup wanting{PluginSinkConfig{}};
  wanting.add(std::make_unique<MemoryObservationPlugin>());
  EXPECT_TRUE(wanting.observes_memory_routing());

  // And the guard itself, not just the flag driving it: a plugin that
  // implements the hook but does not opt in must receive nothing. Without
  // this, deleting the guard and building an observation for every memory
  // instruction would pass the whole suite.
  PluginFixture declining;
  auto *quiet = declining.attach_memory_observation_plugin(/*wants_hook=*/false);
  EXPECT_FALSE(declining.plugin_group().observes_memory_routing());
  auto *cu = declining.cu();
  auto *wave = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0x800, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0b1);
  auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
  state->wf_size = wave->wf_size();
  state->elem_size = 4;
  state->num_elems = 1;
  state->exec_mask = 0b1;
  state->lane_mask = 0b1;
  state->per_lane_addr[0] = 0x40;
  test::ComputeUnitTestAccess::route_memory_inst(*cu, new TestMemoryInstruction(std::move(state)),
                                                 *wave);
  EXPECT_TRUE(quiet->accesses.empty()) << "the observation was built for a plugin that declined it";
  // The pre-routing hook is not gated, so it still fired.
  EXPECT_EQ(quiet->before_tag.size(), 1u);

  // A group-wide subscription is only the collection gate. Once another
  // plugin enables collection, fanout must still exclude a plugin whose own
  // policy was sampled as false when it was added.
  ExecutionPluginGroup mixed{PluginSinkConfig{}};
  auto quiet_member = std::make_unique<MemoryObservationPlugin>(false, "quiet_memory_observation");
  auto *quiet_member_ptr = quiet_member.get();
  auto wanting_member =
      std::make_unique<MemoryObservationPlugin>(true, "wanting_memory_observation");
  auto *wanting_member_ptr = wanting_member.get();
  ASSERT_TRUE(mixed.add(std::move(quiet_member)));
  ASSERT_TRUE(mixed.add(std::move(wanting_member)));
  ASSERT_TRUE(mixed.observes_memory_routing());

  MemoryAccessObservation observation;
  mixed.onAmdgpuMemoryAccessRouted(observation);
  EXPECT_TRUE(quiet_member_ptr->accesses.empty());
  ASSERT_EQ(wanting_member_ptr->accesses.size(), 1u);
}

TEST(RoutedMemoryObservationTest, AnUnroutableAccessIsReportedRatherThanDropped) {
  // No pipeline takes this, so nothing downstream will ever mention it. A
  // model counting a kernel's memory traffic has to be able to see that there
  // was an access it cannot account for.
  PluginFixture fixture;
  auto *plugin = fixture.attach_memory_observation_plugin();
  auto *cu = fixture.cu();
  auto *wave = cu->dispatch_wf(/*wg_id=*/2, /*pc=*/0x600, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wave, nullptr);

  class UnroutedState : public DynamicInstState {
  public:
    UnroutedState() { tag_ = 0; }
  };
  // Built before the new-expression, not inside it. An argument that can throw
  // makes the compiler emit the cleanup that frees the raw storage again, and
  // GCC reads that pairing of Instruction's own operator new with the plain
  // ::operator delete its operator delete falls through to as a mismatch.
  auto state = std::make_unique<UnroutedState>();
  test::ComputeUnitTestAccess::route_memory_inst(*cu, new TestMemoryInstruction(std::move(state)),
                                                 *wave);

  ASSERT_EQ(plugin->accesses.size(), 1u);
  const auto &access = plugin->accesses.front();
  EXPECT_EQ(access.route, MemoryRoute::UNKNOWN);
  EXPECT_EQ(std::string(memory_route_name(access.route)), "unknown");
  EXPECT_EQ(access.pc, 0x600u);
  EXPECT_TRUE(access.addresses.empty());
}

TEST(ExecutionPluginTest, MemoryPipelineFlatCountersFollowRoutedRequests) {
  enum class Operation { Load, Store, ReturningAtomic, Atomic };
  enum class Space { Global, Lds, Scratch, GlobalAndScratch };
  for (const std::string_view arch : {"cdna2", "cdna4", "rdna2", "rdna4", "cdna5"}) {
    for (const auto operation :
         {Operation::Load, Operation::Store, Operation::ReturningAtomic, Operation::Atomic}) {
      for (const auto space :
           {Space::Global, Space::Lds, Space::Scratch, Space::GlobalAndScratch}) {
        // RDNA2's FLAT address calculator does not yet translate private pointers.
        if (arch == "rdna2" && (space == Space::Scratch || space == Space::GlobalAndScratch))
          continue;
        for (const uint64_t exec : {0u, 2u, 3u}) {
          SCOPED_TRACE(arch);
          SCOPED_TRACE(static_cast<int>(operation));
          SCOPED_TRACE(static_cast<int>(space));
          SCOPED_TRACE(exec);
          const bool cdna_legacy = arch == "cdna2" || arch == "cdna4";
          const bool split = arch == "rdna4" || arch == "cdna5";
          const uint32_t wave_size = cdna_legacy ? 64 : 32;
          PluginFixture f(1, std::string(arch), wave_size, cdna_legacy ? 104 : 128);
          auto *observer = f.attach_memory_observation_plugin();
          auto *cu = f.cu();
          constexpr uint64_t shared = 0x0001'0000'0000'0000ULL;
          constexpr uint64_t private_base = 0x0002'0000'0000'0000ULL;
          constexpr uint64_t global = 0x400000;
          constexpr uint64_t scratch = 0x800000;
          cu->set_apertures(shared, shared + 0xFFFF, private_base, private_base + 0xFFFF);
          auto *wf = cu->dispatch_wf(0, 0x100, cdna_legacy ? 104 : 128, 256, wave_size);
          ASSERT_NE(wf, nullptr);
          wf->set_exec(exec);
          wf->set_scratch_base(scratch);
          wf->set_scratch_lane_size(256);
          const uint32_t vb = wf->vgpr_alloc().base;
          for (uint32_t lane = 0; lane < 2; ++lane) {
            const bool private_lane =
                space == Space::Scratch || (space == Space::GlobalAndScratch && lane == 1);
            const uint64_t private_address =
                arch == "cdna5" ? scratch + (uint64_t{lane} << 52) + 0x20 : private_base + 0x20;
            const uint64_t address = space == Space::Lds ? shared + lane * 4
                                     : private_lane      ? private_address
                                                         : global + lane * 4;
            cu->write_vgpr(vb, lane, static_cast<uint32_t>(address));
            cu->write_vgpr(vb + 1, lane, address >> 32);
            cu->write_vgpr(vb + 4, lane, 1);
            const uint32_t value = 17;
            cu->lds().write32(wf->lds_base() + lane * 4, value);
            const uint64_t backing = !private_lane     ? global + lane * 4
                                     : arch == "rdna4" ? scratch + lane * 256 + 0x20
                                                       : scratch + 8 * wave_size * 4 + lane * 4;
            f.mem->load_image(reinterpret_cast<const uint8_t *>(&value), 4, backing);
          }
          auto decoder = Decoder::create(cu->arch());
          std::unique_ptr<Instruction> inst;
          const bool load = operation == Operation::Load;
          const bool store = operation == Operation::Store;
          const bool returns = operation == Operation::ReturningAtomic;
          const auto decode = [&](const auto &words) {
            inst.reset(decode_valid(*decoder, words.data()));
          };
          if (arch == "cdna2")
            decode(cdna2::build_flat(
                load    ? cdna2::kFlatLoadDwordFlat
                : store ? cdna2::kFlatStoreDwordFlat
                        : cdna2::kFlatAtomicAddFlat,
                {.glc = returns, .addr = 0, .data = 4, .saddr = 0x7F, .vdst = 8}));
          else if (arch == "cdna4")
            decode(cdna4::build_flat(
                load    ? cdna4::kFlatLoadDwordFlat
                : store ? cdna4::kFlatStoreDwordFlat
                        : cdna4::kFlatAtomicAddFlat,
                {.sc0 = returns, .addr = 0, .data = 4, .saddr = 0x7F, .vdst = 8}));
          else if (arch == "rdna2")
            decode(rdna2::build_flat(
                load    ? rdna2::kFlatLoadDwordFlat
                : store ? rdna2::kFlatStoreDwordFlat
                        : rdna2::kFlatAtomicAddFlat,
                {.glc = returns, .addr = 0, .data = 4, .saddr = 0x7F, .vdst = 8}));
          else if (arch == "rdna4")
            decode(rdna4::build_vflat(load    ? rdna4::kFlatLoadB32Vflat
                                      : store ? rdna4::kFlatStoreB32Vflat
                                              : rdna4::kFlatAtomicAddU32Vflat,
                                      {.saddr = amdgpu::kModernNullSelector,
                                       .vdst = 8,
                                       .th = returns,
                                       .vsrc = 4,
                                       .vaddr = 0}));
          else
            decode(cdna5::build_vflat(load    ? cdna5::kFlatLoadB32Vflat
                                      : store ? cdna5::kFlatStoreB32Vflat
                                              : cdna5::kFlatAtomicAddU32Vflat,
                                      {.saddr = amdgpu::kModernNullSelector,
                                       .vdst = 8,
                                       .th = returns,
                                       .vsrc = 4,
                                       .vaddr = 0}));
          ASSERT_NE(inst, nullptr);
          unsigned exports = 0;
          for (const auto obligation : inst->amdgpu_memory_issue_info()->counter_obligations())
            if (obligation.wait_counter_type() == WaitCounterType::EXPCNT)
              exports += obligation.counter_increment();
          ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
          test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), *wf);
          ASSERT_EQ(observer->accesses.size(), 1u);
          const auto &counters = observer->accesses.front().issued_counters;
          const bool result = load || returns;
          const auto vm_counter =
              split ? (result ? WaitCounterType::LOADCNT : WaitCounterType::STORECNT)
              : arch == "rdna2" && !result ? WaitCounterType::VSCNT
                                           : WaitCounterType::VMCNT;
          const auto ds_counter = split ? WaitCounterType::DSCNT : WaitCounterType::LGKMCNT;
          EXPECT_EQ(std::ranges::count(counters, vm_counter), exec && space != Space::Lds ? 1 : 0);
          EXPECT_EQ(std::ranges::count(counters, ds_counter), exec && space == Space::Lds ? 1 : 0);
          EXPECT_EQ(std::ranges::count(counters, WaitCounterType::EXPCNT), exports);
        }
      }
    }
  }
}

TEST(InstructionMetadataTest, GenericFlatHasTwoCounterObligations) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);

  const auto generic_words =
      cdna4::build_flat(cdna4::kFlatLoadDwordFlat, {.seg = 0, .addr = 0, .saddr = 0x7F, .vdst = 1});
  std::unique_ptr<Instruction> generic_flat(decode_valid(*decoder, generic_words.data()));
  ASSERT_NE(generic_flat, nullptr);
  const auto *generic_issue = generic_flat->amdgpu_memory_issue_info();
  ASSERT_NE(generic_issue, nullptr);
  const auto generic_obligations = generic_issue->counter_obligations();
  ASSERT_EQ(generic_obligations.size(), 2u);
  EXPECT_EQ(generic_obligations[0].wait_counter_type(), WaitCounterType::VMCNT);
  EXPECT_EQ(generic_obligations[0].completion_class(), MemoryCompletionClass::UNORDERED);
  EXPECT_EQ(generic_obligations[1].wait_counter_type(), WaitCounterType::LGKMCNT);
  EXPECT_EQ(generic_obligations[1].completion_class(), MemoryCompletionClass::UNORDERED);

  for (const uint8_t fixed_segment : {uint8_t{1}, uint8_t{2}}) {
    const auto fixed_words = cdna4::build_flat(
        cdna4::kFlatLoadDwordFlat, {.seg = fixed_segment, .addr = 0, .saddr = 0x7F, .vdst = 1});
    std::unique_ptr<Instruction> fixed(decode_valid(*decoder, fixed_words.data()));
    ASSERT_NE(fixed, nullptr);
    const auto *fixed_issue = fixed->amdgpu_memory_issue_info();
    ASSERT_NE(fixed_issue, nullptr);
    const auto fixed_obligations = fixed_issue->counter_obligations();
    ASSERT_EQ(fixed_obligations.size(), 1u);
    EXPECT_EQ(fixed_obligations[0].wait_counter_type(), WaitCounterType::VMCNT);
    EXPECT_EQ(fixed_obligations[0].completion_class(), MemoryCompletionClass::VMEM);
  }
}

TEST(InstructionMetadataTest, StoresAndGdsExposeEveryCounterObligation) {
  for (const auto &[arch, words] : std::array{
           std::pair{ROCJITSU_CODE_ARCH_CDNA2, cdna2::build_mubuf(cdna2::kBufferStoreDwordMubuf)},
           std::pair{ROCJITSU_CODE_ARCH_RDNA2,
                     rdna2::build_mubuf(rdna2::kBufferStoreDwordMubuf)}}) {
    auto decoder = Decoder::create(arch);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> store(decode_valid(*decoder, words.data()));
    ASSERT_NE(store, nullptr);
    const auto *issue = store->amdgpu_memory_issue_info();
    ASSERT_NE(issue, nullptr);
    const auto obligations = issue->counter_obligations();
    ASSERT_EQ(obligations.size(), 2u);
    EXPECT_EQ(obligations[1].wait_counter_type(), WaitCounterType::EXPCNT);
    EXPECT_EQ(obligations[1].completion_class(), MemoryCompletionClass::UNORDERED);
  }

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_NE(decoder, nullptr);
  const auto lds_words = cdna2::build_ds(cdna2::kDsReadB32Ds, {.gds = 0});
  const auto gds_words = cdna2::build_ds(cdna2::kDsReadB32Ds, {.gds = 1});
  std::unique_ptr<Instruction> lds(decode_valid(*decoder, lds_words.data()));
  std::unique_ptr<Instruction> gds(decode_valid(*decoder, gds_words.data()));
  ASSERT_NE(lds, nullptr);
  ASSERT_NE(gds, nullptr);
  const auto lds_obligations = lds->amdgpu_memory_issue_info()->counter_obligations();
  const auto gds_obligations = gds->amdgpu_memory_issue_info()->counter_obligations();
  ASSERT_EQ(lds_obligations.size(), 1u);
  EXPECT_EQ(lds_obligations[0].completion_class(), MemoryCompletionClass::LDS);
  ASSERT_EQ(gds_obligations.size(), 2u);
  EXPECT_EQ(gds_obligations[0].completion_class(), MemoryCompletionClass::GDS);
  EXPECT_EQ(gds_obligations[1].wait_counter_type(), WaitCounterType::EXPCNT);

  const auto flat_store_words = cdna2::build_flat(cdna2::kFlatStoreDwordFlat,
                                                  {.seg = 0, .addr = 0, .data = 1, .saddr = 0x7F});
  std::unique_ptr<Instruction> flat_store(decode_valid(*decoder, flat_store_words.data()));
  ASSERT_NE(flat_store, nullptr);
  const auto flat_store_obligations = flat_store->amdgpu_memory_issue_info()->counter_obligations();
  ASSERT_EQ(flat_store_obligations.size(), 3u);
  EXPECT_EQ(flat_store_obligations[0].wait_counter_type(), WaitCounterType::VMCNT);
  EXPECT_EQ(flat_store_obligations[1].wait_counter_type(), WaitCounterType::LGKMCNT);
  EXPECT_EQ(flat_store_obligations[2].wait_counter_type(), WaitCounterType::EXPCNT);
}

TEST(InstructionMetadataTest, Cdna5AsyncOperationsExposeDistinctCompletionDomains) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  const auto load_words = cdna5::build_vglobal(cdna5::kGlobalLoadAsyncToLdsB32Vglobal);
  const auto store_words = cdna5::build_vglobal(cdna5::kGlobalStoreAsyncFromLdsB32Vglobal);
  const auto barrier_words = cdna5::build_vds(cdna5::kDsAtomicAsyncBarrierArriveB64Vds);

  const auto expect = [&](const auto &words, MemoryCompletionClass completion) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    const auto obligations = inst->amdgpu_memory_issue_info()->counter_obligations();
    ASSERT_EQ(obligations.size(), 1u);
    EXPECT_EQ(obligations[0].wait_counter_type(), WaitCounterType::ASYNCCNT);
    EXPECT_EQ(obligations[0].completion_class(), completion);
  };
  expect(load_words, MemoryCompletionClass::ASYNC_LOAD);
  expect(store_words, MemoryCompletionClass::ASYNC_STORE);
  expect(barrier_words, MemoryCompletionClass::ASYNC_LOAD);
}

TEST(InstructionMetadataTest, WideScalarLoadContributesTwoCounterTokens) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  const auto words =
      cdna4::build_smem(cdna4::kSLoadDwordx2Smem, {.sbase = 0, .sdata = 4, .imm = 1, .offset = 0});
  std::unique_ptr<Instruction> load(decode_valid(*decoder, words.data()));
  ASSERT_NE(load, nullptr);
  const auto obligations = load->amdgpu_memory_issue_info()->counter_obligations();
  ASSERT_EQ(obligations.size(), 1u);
  EXPECT_EQ(obligations[0].wait_counter_type(), WaitCounterType::LGKMCNT);
  EXPECT_EQ(obligations[0].counter_increment(), 2u);
}

TEST(InstructionMetadataTest, CounterObligationPackingRoundTrips) {
  constexpr MemoryCounterObligation obligation{WaitCounterType::ASYNCCNT,
                                               MemoryCompletionClass::ASYNC_STORE, 2};
  static_assert(sizeof(MemoryCounterObligation) == 1);
  EXPECT_TRUE(obligation.valid());
  EXPECT_EQ(obligation.wait_counter_type(), WaitCounterType::ASYNCCNT);
  EXPECT_EQ(obligation.completion_class(), MemoryCompletionClass::ASYNC_STORE);
  EXPECT_EQ(obligation.counter_increment(), 2u);
  EXPECT_FALSE(MemoryCounterObligation{}.valid());
}

} // namespace
