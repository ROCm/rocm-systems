// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin_race_test.cpp
/// @brief Execution plugin race detection and diagnostic output tests.

#include "execution_plugin_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"

#include "mma_test_util.h"
#include "race-detector/race_test_instruction.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/race_detector/plugin.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

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
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand_types.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
using namespace rocjitsu::plugins::race_detector;
using namespace rocjitsu::test::execution_plugin;
using mma_test::make_cdna4_mfma_scale_words;

enum class InlineImageQuery { ResourceInfo, Lod };
constexpr InlineImageQuery kInlineImageQueries[]{
    InlineImageQuery::ResourceInfo,
};

void record_register_probe(ExecutionPluginGroup &plugins, Wavefront &wave) {
  auto decoder = Decoder::create(wave.cu().arch());
  const uint32_t word = S_NOP;
  std::unique_ptr<Instruction> instruction(decode_valid(*decoder, &word));
  ASSERT_NE(instruction, nullptr);
  plugins.onAmdgpuBeforeExecuteInstruction(wave.pc, *instruction, wave);
}

void register_pending_event(WaveRaceState &race, uint64_t pc, MemoryEventType type,
                            std::vector<uint32_t> registers, WaitCounterType counter,
                            MemoryOrderClass order) {
  const std::array obligations{MemoryCounterObligation{counter, order}};
  race.registerEvent(testInstruction(pc), type, std::move(registers), 1, 0xf, obligations, order);
}

void execute_wait(ExecutionPluginGroup &plugins, Wavefront &wave, uint32_t word) {
  auto decoder = Decoder::create(wave.cu().arch());
  ASSERT_NE(decoder, nullptr);
  util::StringDiagnostic error;
  auto decoded = decoder->decode_window(std::span<const uint32_t>(&word, 1), 0, error.emitter());
  ASSERT_TRUE(decoded.succeeded()) << error.message();
  auto &inst = *decoded.value();
  ASSERT_TRUE(inst.is_waitcnt());
  inst.execute(inst, &wave);
  plugins.onAmdgpuAfterExecuteInstruction(wave.pc, inst, wave);
}

uint32_t legacy_wait_word(rj_code_arch_t arch, uint8_t vmcnt, uint8_t lgkmcnt) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
    return 0xbf890007u | (uint32_t{vmcnt} << 10) | (uint32_t{lgkmcnt} << 4);
  return 0xbf8c0070u | (vmcnt & 0xfu) | ((vmcnt & 0x30u) << 10) | (uint32_t{lgkmcnt} << 8);
}

/// Applies the race detector's register checks to the VGPR reads and writes of one wave.
class MfmaRacePlugin : public ExecutionPlugin {
public:
  MfmaRacePlugin() : ExecutionPlugin("mfma_race_probe") {}

  void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t wg_id, uint32_t physical_vgpr_count,
                                   uint32_t physical_sgpr_count,
                                   std::span<amdgpu::Wavefront *> wavefronts) override {
    detector_ = std::make_unique<RaceDetector>(
        static_cast<int>(wavefronts.size()), static_cast<int>(physical_vgpr_count),
        static_cast<int>(physical_sgpr_count), Dim3d(static_cast<int>(wg_id)),
        [this](RaceViolation v) { violations.push_back(v); },
        counterCapacitiesForArch(wavefronts.front()->cu().arch()),
        wavefronts.front()->lds().size_bytes());
    wf_ = wavefronts.front();
    state_ = &detector_->getWaveRaceState(0);
  }

  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask) override {
    if (wf != wf_ || !state_)
      return;
    uint32_t logical_reg = physical_reg - wf->vgpr_alloc().base;
    state_->checkVgprReadLanes(static_cast<int>(logical_reg), lane_mask, byte_mask);
  }

  void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                              uint64_t lane_mask, uint8_t byte_mask) override {
    if (wf != wf_ || !state_)
      return;
    uint32_t logical_reg = physical_reg - wf->vgpr_alloc().base;
    state_->checkVgprWriteLanes(static_cast<int>(logical_reg), lane_mask, byte_mask);
  }

  void registerOutstandingLoad(uint32_t logical_reg, uint64_t exec_mask, uint8_t byte_mask = 0xF) {
    ASSERT_NE(state_, nullptr);
    state_->registerEvent(testInstruction(/*pc=*/0x1000), MemoryEventType::GLOBAL_TO_VGPR,
                          {logical_reg}, exec_mask, byte_mask);
  }

  [[nodiscard]] bool raced_on(uint32_t logical_reg) const {
    return std::ranges::any_of(violations, [&](const RaceViolation &v) {
      return v.space == RaceViolation::Space::VGPR && v.index == static_cast<int>(logical_reg);
    });
  }

  std::vector<RaceViolation> violations;

private:
  std::unique_ptr<RaceDetector> detector_;
  amdgpu::Wavefront *wf_ = nullptr;
  WaveRaceState *state_ = nullptr;
};

class FormattedLoadRaceTest : public ::testing::Test {
protected:
  PluginFixture fixture;
  StringSink *sink = nullptr;
  Wavefront *wave = nullptr;
  RaceWavefrontState *plugin_state = nullptr;
  std::unique_ptr<Decoder> decoder;

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(fixture.initialize(1, "rdna4", 32, 128));
    PluginSinkConfig sink_config;
    sink = &sink_config.emplace<StringSink>();
    fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *plugin_ptr = plugin.get();
    ASSERT_TRUE(fixture.plugin_group_->add(std::move(plugin)));
    fixture.soc->set_plugin_group(fixture.plugin_group_);
    fixture.plugin_group_->onInit();
    wave = fixture.cu()->dispatch_wf(0, 0, 128, 256, 32);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(1);
    std::array<Wavefront *, 1> waves{wave};
    fixture.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, 128, waves);
    plugin_state = static_cast<RaceWavefrontState *>(wave->plugin_state(plugin_ptr->slot_index()));
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_NE(decoder, nullptr);
  }

  void issue(uint16_t opcode, uint32_t format, uint8_t destination, uint32_t components) {
    auto *cu = fixture.cu();
    const uint32_t scalar_base = wave->sgpr_alloc().base;
    cu->write_sgpr(scalar_base + 4, 0x1000);
    cu->write_sgpr(scalar_base + 5, 0);
    cu->write_sgpr(scalar_base + 6, 64);
    cu->write_sgpr(scalar_base + 7,
                   4 | (5 << 3) | (6 << 6) | (7 << 9) | (format << 12) | (1u << 28));
    const auto words = rdna4::build_vbuffer(opcode, {.soffset = rdna4::OPR_SREG_M0_NULL,
                                                     .vdata = destination,
                                                     .rsrc = 4,
                                                     .format = static_cast<uint8_t>(format)});
    std::unique_ptr<Instruction> load(decode_valid(*decoder, words.data()));
    ASSERT_NE(load, nullptr);
    wave->pc += 16; // Give each probe a distinct PC so report deduplication cannot hide it.
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave->pc, *load, *wave);
    ASSERT_TRUE(cu->execute_instruction(load.get(), *wave).succeeded());
    ASSERT_NE(load->data(), nullptr);
    const auto &state = *load->data_as<VectorMemState>();
    EXPECT_EQ(state.buffer_components, components);
    EXPECT_EQ(state.num_elems, 1u);
    EXPECT_EQ(state.elem_size, format == 46 ? 4u : 16u);
    fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, *load, *wave);
  }

  void wait(uint32_t count) { execute_wait(*fixture.plugin_group_, *wave, 0xbfc00000u | count); }

  bool probe(uint32_t destination, uint8_t byte_mask, bool write, uint64_t lanes = 1) {
    record_register_probe(*fixture.plugin_group_, *wave);
    const size_t previous = sink->str().size();
    const uint32_t physical = wave->vgpr_alloc().base + destination;
    if (write)
      fixture.plugin_group_->onAmdgpuWriteVgprLanes(wave, physical, lanes, byte_mask);
    else
      fixture.plugin_group_->onAmdgpuReadVgprLanes(wave, physical, lanes, byte_mask);
    return sink->str().find("RACE ", previous) != std::string::npos;
  }
};

TEST_F(FormattedLoadRaceTest, TracksEveryDestinationAndWrittenHalf) {
  struct Case {
    uint16_t opcode;
    uint32_t components;
    std::array<uint8_t, 4> masks;
  };
  const Case cases[] = {
      {rdna4::kBufferLoadFormatXVbuffer, 1, {0xf, 0, 0, 0}},
      {rdna4::kBufferLoadFormatXyzwVbuffer, 4, {0xf, 0xf, 0xf, 0xf}},
      {rdna4::kTbufferLoadFormatXyzwVbuffer, 4, {0xf, 0xf, 0xf, 0xf}},
      {rdna4::kBufferLoadD16FormatXVbuffer, 1, {0x3, 0, 0, 0}},
      {rdna4::kBufferLoadD16HiFormatXVbuffer, 1, {0xc, 0, 0, 0}},
      {rdna4::kBufferLoadD16FormatXyVbuffer, 2, {0xf, 0, 0, 0}},
      {rdna4::kBufferLoadD16FormatXyzVbuffer, 3, {0xf, 0x3, 0, 0}},
      {rdna4::kBufferLoadD16FormatXyzwVbuffer, 4, {0xf, 0xf, 0, 0}},
  };
  for (const auto &test : cases) {
    for (uint32_t format : {46u, 62u}) { // Four-byte and sixteen-byte memory footprints.
      for (uint32_t offset = 0; offset < 5; ++offset) {
        for (uint8_t mask : {0x3, 0xc}) {
          for (bool write : {false, true}) {
            SCOPED_TRACE(std::format("opcode={} format={} offset={} mask={} write={}", test.opcode,
                                     format, offset, mask, write));
            issue(test.opcode, format, 8, test.components);
            ASSERT_EQ(plugin_state->race_state->getWaveMemoryEvents().size(), 1u);
            EXPECT_FALSE(probe(8 + offset, mask, write, /*lanes=*/2));
            const bool expected = offset < test.masks.size() && (test.masks[offset] & mask);
            EXPECT_EQ(probe(8 + offset, mask, write), expected);
            wait(0);
            EXPECT_TRUE(plugin_state->race_state->getWaveMemoryEvents().empty());
            EXPECT_FALSE(probe(8 + offset, mask, write));
          }
        }
      }
    }
  }
}

TEST_F(FormattedLoadRaceTest, PartialWaitRetiresOneLoadWithAllItsDestinations) {
  issue(rdna4::kBufferLoadD16FormatXyzVbuffer, 46, 8, 3);
  issue(rdna4::kBufferLoadD16FormatXyzVbuffer, 46, 12, 3);
  ASSERT_EQ(plugin_state->race_state->getWaveMemoryEvents().size(), 2u);
  wait(1);
  ASSERT_EQ(plugin_state->race_state->getWaveMemoryEvents().size(), 1u);
  EXPECT_FALSE(probe(8, 0xf, false));
  EXPECT_FALSE(probe(9, 0xf, false));
  EXPECT_FALSE(probe(13, 0xc, false));
  EXPECT_TRUE(probe(13, 0x3, false));
  wait(0);
  EXPECT_FALSE(probe(12, 0xf, false));
  EXPECT_FALSE(probe(13, 0xf, false));
}

TEST_F(FormattedLoadRaceTest, RecordingBytesDoesNotDisassemble) {
  class ObservableInstruction final : public Instruction {
  public:
    ObservableInstruction() : Instruction("s_nop", nullptr) {
      size_ = sizeof(word);
      raw_encoding_ = &word;
    }
    uint32_t word = S_NOP | 7;
    std::string disassemble() const override {
      was_disassembled = true;
      return Instruction::disassemble();
    }
    mutable bool was_disassembled = false;
  } instruction;
  for (uint64_t pc : {0x540024b100ULL, 0x100002a100ULL})
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(pc, instruction, *wave);
  ASSERT_EQ(plugin_state->trace.size(), 2u);
  EXPECT_EQ(plugin_state->trace[0], testInstruction(0x540024b100ULL, S_NOP | 7));
  instruction.word = S_NOP | 9;
  EXPECT_EQ(plugin_state->trace[0], testInstruction(0x540024b100ULL, S_NOP | 7));
  EXPECT_EQ(plugin_state->trace[1].pc(), 0x100002a100ULL);
  EXPECT_FALSE(instruction.was_disassembled);
}

TEST_F(FormattedLoadRaceTest, RejectsInvalidEncodingAtCapture) {
  class InvalidInstruction final : public Instruction {
  public:
    InvalidInstruction(const uint32_t *words, int size) : Instruction("s_nop", nullptr) {
      raw_encoding_ = words;
      size_ = size;
    }
  };
  const std::array<uint32_t, InstructionSnapshot::kMaxWords + 1> encoding{S_NOP};
  for (const auto &[words, size] : {std::pair<const uint32_t *, int>{nullptr, 4},
                                    {encoding.data(), 0},
                                    {encoding.data(), -4},
                                    {encoding.data(), 2},
                                    {encoding.data(), static_cast<int>(sizeof(encoding))}}) {
    SCOPED_TRACE(size);
    InvalidInstruction instruction(words, size);
    EXPECT_THROW(
        fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(0x1000, instruction, *wave),
        std::logic_error);
    EXPECT_EQ(plugin_state->trace.size(), 0u);
    EXPECT_THROW(fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, instruction, *wave),
                 std::logic_error);
    EXPECT_TRUE(plugin_state->race_state->getWaveMemoryEvents().empty());
  }
}

TEST_F(FormattedLoadRaceTest, RejectsMissingInstructionHistoryWhenReporting) {
  issue(rdna4::kBufferLoadFormatXVbuffer, 46, 8, 1);
  ASSERT_EQ(plugin_state->race_state->getWaveMemoryEvents().size(), 1u);
  plugin_state->trace = {};
  EXPECT_THROW(fixture.plugin_group_->onAmdgpuReadVgprLanes(wave, wave->vgpr_alloc().base + 8, 1),
               std::logic_error);
  EXPECT_EQ(sink->str().find("RACE "), std::string::npos);
}

TEST_F(FormattedLoadRaceTest, RejectsWholeDestinationRangeWhenItExceedsTheWaveAllocation) {
  issue(rdna4::kBufferLoadFormatXyzwVbuffer, 46, 255, 4);
  EXPECT_TRUE(plugin_state->race_state->getWaveMemoryEvents().empty());
  EXPECT_FALSE(probe(255, 0xf, false));
  issue(rdna4::kBufferLoadD16FormatXyzVbuffer, 46, 255, 3);
  EXPECT_TRUE(plugin_state->race_state->getWaveMemoryEvents().empty());
  EXPECT_FALSE(probe(255, 0xf, false));
}

struct BlockLoadCase {
  const char *arch;
  bool scratch;
};

class BlockLoadRaceTest : public ::testing::TestWithParam<BlockLoadCase> {
protected:
  std::unique_ptr<PluginFixture> fixture;
  StringSink *sink = nullptr;
  Wavefront *wave = nullptr;
  RaceWavefrontState *plugin_state = nullptr;
  std::unique_ptr<Decoder> decoder;

  void SetUp() override {
    fixture = std::make_unique<PluginFixture>();
    ASSERT_NO_FATAL_FAILURE(fixture->initialize(/*num_wf_slots=*/1, GetParam().arch,
                                                /*wavefront_size=*/32, /*sgprs_per_wf=*/128));
    PluginSinkConfig sink_config;
    sink = &sink_config.emplace<StringSink>();
    fixture->plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *plugin_ptr = plugin.get();
    ASSERT_TRUE(fixture->plugin_group_->add(std::move(plugin)));
    fixture->soc->set_plugin_group(fixture->plugin_group_);
    fixture->plugin_group_->onInit();
    wave = fixture->cu()->dispatch_wf(0, 0x100, 128, 256, 32);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(1);
    std::array<Wavefront *, 1> waves{wave};
    fixture->plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, 128, waves);
    plugin_state = static_cast<RaceWavefrontState *>(wave->plugin_state(plugin_ptr->slot_index()));
    decoder = Decoder::create(fixture->cu()->arch());
    ASSERT_NE(decoder, nullptr);
    fixture->cu()->write_vgpr(wave->vgpr_alloc().base, 0, 0);
    fixture->cu()->write_vgpr(wave->vgpr_alloc().base + 1, 0, 0);
  }

  void issue(uint32_t mask, uint8_t destination = 32) {
    const bool scratch = GetParam().scratch;
    std::array<uint32_t, 3> words;
    if (fixture->cu()->arch() == ROCJITSU_CODE_ARCH_CDNA5) {
      words =
          scratch
              ? cdna5::build_vscratch(cdna5::kScratchLoadBlock,
                                      {.saddr = 124, .vdst = destination, .sve = 1, .vaddr = 0})
              : cdna5::build_vglobal(cdna5::kGlobalLoadBlock, {.saddr = 124, .vdst = destination});
    } else {
      words =
          scratch
              ? rdna4::build_vscratch(rdna4::kScratchLoadBlock,
                                      {.saddr = 124, .vdst = destination, .sve = 1, .vaddr = 0})
              : rdna4::build_vglobal(rdna4::kGlobalLoadBlock, {.saddr = 124, .vdst = destination});
    }
    std::unique_ptr<Instruction> load(decode_valid(*decoder, words.data()));
    ASSERT_NE(load, nullptr);
    wave->pc += 16;
    wave->set_m0(mask);
    fixture->plugin_group_->onAmdgpuBeforeExecuteInstruction(wave->pc, *load, *wave);
    ASSERT_TRUE(fixture->cu()->execute_instruction(load.get(), *wave).succeeded());
    ASSERT_NE(load->data(), nullptr);
    wave->set_m0(~mask); // The plugin must use the issue-time snapshot.
    fixture->plugin_group_->onAmdgpuMemoryAccessRouted({}, *load, *wave);
  }

  void wait() {
    execute_wait(*fixture->plugin_group_, *wave, 0xbfc00000u); // s_wait_loadcnt 0.
    execute_wait(*fixture->plugin_group_, *wave, 0xbfc60000u); // s_wait_dscnt 0.
  }

  bool probe(uint32_t destination, bool write, uint64_t lanes = 1) {
    wave->pc += 16;
    record_register_probe(*fixture->plugin_group_, *wave);
    const size_t previous = sink->str().size();
    const uint32_t physical = wave->vgpr_alloc().base + destination;
    if (write)
      fixture->plugin_group_->onAmdgpuWriteVgprLanes(wave, physical, lanes);
    else
      fixture->plugin_group_->onAmdgpuReadVgprLanes(wave, physical, lanes);
    return sink->str().find("RACE ", previous) != std::string::npos;
  }
};

TEST_P(BlockLoadRaceTest, TracksOnlyEnabledDestinationsAndKeepsEmptyMaskCounterEntry) {
  for (uint32_t mask : {0u, 1u, 0x80000001u, 0x55555555u, 0xffffffffu}) {
    SCOPED_TRACE(mask);
    for (uint32_t word : {0u, 1u, 15u, 31u}) {
      SCOPED_TRACE(word);
      for (bool write : {false, true}) {
        // Reissue for each consumer so report deduplication at the producing
        // instruction cannot hide the enabled-word positive controls.
        ASSERT_NO_FATAL_FAILURE(issue(mask));
        ASSERT_EQ(plugin_state->race_state->getWaveMemoryEvents().size(), 1u);
        EXPECT_FALSE(probe(32 + word, write, /*lanes=*/2));
        EXPECT_EQ(probe(32 + word, write), (mask & (uint32_t{1} << word)) != 0);
        wait();
        EXPECT_TRUE(plugin_state->race_state->getWaveMemoryEvents().empty());
        EXPECT_FALSE(probe(32 + word, write));
      }
    }
  }
}

TEST_P(BlockLoadRaceTest, IncomingLoadChecksOnlyEnabledWordsForWaw) {
  for (uint32_t mask : {0u, 1u, 0x80000001u}) {
    for (uint32_t word : {0u, 1u, 31u}) {
      SCOPED_TRACE(testing::Message() << "mask=" << mask << " word=" << word);
      // LDS and global loads are not ordered with each other. A pending LDS
      // destination exposes spurious WAW checks on a masked global/scratch load.
      wave->pc += 16;
      register_pending_event(*plugin_state->race_state, wave->pc, MemoryEventType::LDS_TO_VGPR,
                             {32 + word}, WaitCounterType::DSCNT, MemoryOrderClass::LDS);
      const size_t previous = sink->str().size();
      ASSERT_NO_FATAL_FAILURE(issue(mask));
      EXPECT_EQ(sink->str().find("RACE ", previous) != std::string::npos,
                (mask & (uint32_t{1} << word)) != 0);
      wait();
      EXPECT_TRUE(plugin_state->race_state->getWaveMemoryEvents().empty());
    }
  }
}

TEST_P(BlockLoadRaceTest, ChecksOwnedWordsForWawInPartialDestinationSpan) {
  for (uint32_t mask : {0u, 1u, 0x80000000u, 0xffffffffu}) {
    SCOPED_TRACE(mask);
    // v240 belongs to this wave, but the full v[240:271] destination does not.
    // Block destinations are checked per DWORD, so enabled v240 still writes.
    wave->pc += 16;
    register_pending_event(*plugin_state->race_state, wave->pc, MemoryEventType::LDS_TO_VGPR, {240},
                           WaitCounterType::DSCNT, MemoryOrderClass::LDS);
    const size_t previous = sink->str().size();
    ASSERT_NO_FATAL_FAILURE(issue(mask, 240));
    EXPECT_EQ(sink->str().find("RACE ", previous) != std::string::npos, (mask & 1u) != 0);
    wait();
    EXPECT_TRUE(plugin_state->race_state->getWaveMemoryEvents().empty());
  }
}

TEST_P(BlockLoadRaceTest, TracksOwnedWordsAndCounterInPartialDestinationSpan) {
  for (uint32_t mask : {0u, 1u, 0x80000000u, 0x80008001u, 0xffffffffu}) {
    for (uint32_t word : {0u, 1u, 15u}) {
      for (bool write : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "mask=" << mask << " word=" << word << " write=" << write);
        ASSERT_NO_FATAL_FAILURE(issue(mask, 240));
        ASSERT_EQ(plugin_state->race_state->getWaveMemoryEvents().size(), 1u);
        EXPECT_FALSE(probe(240 + word, write, /*lanes=*/2));
        EXPECT_EQ(probe(240 + word, write), (mask & (uint32_t{1} << word)) != 0);
        wait();
        EXPECT_TRUE(plugin_state->race_state->getWaveMemoryEvents().empty());
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(BlockLoads, BlockLoadRaceTest,
                         ::testing::Values(BlockLoadCase{"cdna5", false},
                                           BlockLoadCase{"cdna5", true},
                                           BlockLoadCase{"rdna4", false},
                                           BlockLoadCase{"rdna4", true}));

/// Drives the race detector's tensor DMA hook for two gfx1250 waves of one
/// workgroup, probing LDS with ordinary DS accesses. The Dtl tests also drive
/// the routed-memory hook with direct-to-LDS writes, checked against a pending
/// tensor store or, without any tensor instruction, a pending DS read. Race
/// semantics are covered in race_detector_tests.cpp; these tests cover the
/// hooks' translation.
class TensorDmaRaceTest : public ::testing::Test {
protected:
  static constexpr uint64_t kTensorPc = 0x2010;

  PluginFixture fixture;
  StringSink *sink = nullptr;
  std::array<Wavefront *, 2> waves{};
  std::unique_ptr<Decoder> decoder;

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(fixture.initialize(/*num_wf_slots=*/2, "cdna5",
                                               /*wavefront_size=*/32, /*sgprs_per_wf=*/128));
    PluginSinkConfig sink_config;
    sink = &sink_config.emplace<StringSink>();
    fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    ASSERT_TRUE(fixture.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    fixture.soc->set_plugin_group(fixture.plugin_group_);
    fixture.plugin_group_->onInit();
    for (auto &wave : waves) {
      wave = fixture.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0x100, /*sgprs=*/128, /*vgprs=*/256,
                                       /*wf_size=*/32);
      ASSERT_NE(wave, nullptr);
      wave->set_exec(1);
    }
    fixture.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0, 256, 128,
                                                       waves);
    decoder = Decoder::create(fixture.cu()->arch());
    ASSERT_NE(decoder, nullptr);
  }

  /// Record a decoded tensor DMA instruction in the wave's trace, as execution
  /// does before running it. It does not execute the instruction; transfer()
  /// delivers the issue observation separately.
  void record_tensor_instruction(int wave, bool is_load) {
    const std::array<uint32_t, 3> words{is_load ? 0xd0710001u : 0xd0714001u, 0x7c000000u,
                                        0x7c7c0c00u};
    util::StringDiagnostic error;
    auto inst = decoder->decode_window(words, waves[wave]->pc, error.emitter());
    ASSERT_TRUE(inst.succeeded()) << error.message();
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[wave]->pc, *inst.value(),
                                                            *waves[wave]);
  }

  /// Issue one tensor DMA instruction, as execution does, and return whether it
  /// produced a race.
  bool transfer(int wave, bool is_load, std::span<const TensorDmaLdsRange> ranges) {
    // Each access gets a distinct PC so reports are not deduplicated.
    waves[wave]->pc += 16;
    record_tensor_instruction(wave, is_load);
    const size_t previous = sink->str().size();
    fixture.plugin_group_->onAmdgpuTensorDmaIssued({.is_load = is_load, .lds_ranges = ranges},
                                                   *waves[wave]);
    return sink->str().find("RACE ", previous) != std::string::npos;
  }

  bool tensor_load(int wave, std::initializer_list<TensorDmaLdsRange> ranges) {
    return transfer(wave, true, std::span(ranges.begin(), ranges.size()));
  }

  bool tensor_store(int wave, std::initializer_list<TensorDmaLdsRange> ranges) {
    return transfer(wave, false, std::span(ranges.begin(), ranges.size()));
  }

  /// Issue a one-lane DS access and report whether it produced a race.
  bool ds_access(int wave, uint32_t address, bool is_load) {
    waves[wave]->pc += 16;
    const size_t previous = sink->str().size();
    auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
    state->elem_size = sizeof(uint32_t);
    state->num_elems = 1;
    state->is_load = is_load;
    state->wf_size = waves[wave]->wf_size();
    state->exec_mask = 1;
    state->lane_mask = 1;
    state->dst_reg_base = waves[wave]->vgpr_alloc().base;
    state->per_lane_addr[0] = address;
    TestMemoryInstruction inst(std::move(state),
                               {{WaitCounterType::DSCNT, MemoryCompletionClass::LDS}});
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[wave]->pc, inst, *waves[wave]);
    fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, inst, *waves[wave]);
    return sink->str().find("RACE ", previous) != std::string::npos;
  }

  bool ds_read(int wave, uint32_t address) { return ds_access(wave, address, true); }
  bool ds_write(int wave, uint32_t address) { return ds_access(wave, address, false); }

  /// Issue a one-lane Direct-to-LDS write of one dword, as an async load to LDS
  /// does, and report whether it produced a race. A nonzero `cluster_mcast_mask`
  /// makes it a cluster load with that M0 multicast mask.
  bool dtl_write(int wave, uint32_t address, uint32_t cluster_mcast_mask = 0) {
    waves[wave]->pc += 16;
    const size_t previous = sink->str().size();
    auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
    state->elem_size = sizeof(uint32_t);
    state->num_elems = 1;
    state->is_load = true;
    state->lds_dst = true;
    state->lds_per_lane_addr = true;
    state->wf_size = waves[wave]->wf_size();
    state->exec_mask = 1;
    state->lane_mask = 1;
    state->per_lane_lds_addr[0] = address;
    state->cluster_multicast = cluster_mcast_mask != 0;
    state->cluster_mcast_mask = cluster_mcast_mask;
    TestMemoryInstruction inst(std::move(state),
                               {{WaitCounterType::ASYNCCNT, MemoryCompletionClass::ASYNC_LOAD}});
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[wave]->pc, inst, *waves[wave]);
    fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, inst, *waves[wave]);
    return sink->str().find("RACE ", previous) != std::string::npos;
  }

  void wait_tensorcnt(int wave, uint16_t count) {
    execute_wait(*fixture.plugin_group_, *waves[wave], 0xbfcb0000u | count);
  }

  void wait_asynccnt(int wave, uint16_t count) {
    execute_wait(*fixture.plugin_group_, *waves[wave], 0xbfca0000u | count);
  }

  void wait_dscnt(int wave) {
    execute_wait(*fixture.plugin_group_, *waves[wave], 0xbfc60000u); // s_wait_dscnt 0.
  }

  void barrier() { fixture.plugin_group_->onAmdgpuBarrierResolved(waves, BarrierKind::Workgroup); }

  /// Wave 0 issues a tensor load at 0x2010 that overwrites bytes wave 1 is
  /// still reading. Returns the race report.
  std::string tensor_load_racing_an_older_read() {
    EXPECT_FALSE(ds_read(1, 4));
    waves[0]->pc = kTensorPc - 16; // transfer() advances the PC by 16.
    const size_t previous = sink->str().size();
    EXPECT_TRUE(tensor_load(0, {{.address = 0, .bytes = 16}}));
    return sink->str().substr(previous);
  }
};

// A reported race suppresses later reports involving either instruction, so
// each retirement check uses a transfer that has not raced yet; the unwaited
// case is its own test.
TEST_F(TensorDmaRaceTest, SixtyFourthTransferRetiresTheOldestLdsWrite) {
  for (uint32_t i = 0; i < 64; ++i)
    EXPECT_FALSE(tensor_load(0, {{.address = 16 * i, .bytes = 16}}));
  EXPECT_FALSE(ds_read(0, 0));
  EXPECT_TRUE(ds_read(0, 16));
}

TEST_F(TensorDmaRaceTest, LoadIsAnLdsWriteUntilRetired) {
  tensor_load(0, {{.address = 0, .bytes = 16}});
  EXPECT_TRUE(ds_read(1, 4));
}

TEST_F(TensorDmaRaceTest, LoadIsRetiredByTensorcntAndBarrier) {
  tensor_load(0, {{.address = 0, .bytes = 16}});
  wait_tensorcnt(0, 0);
  barrier();
  EXPECT_FALSE(ds_read(1, 4));
}

TEST_F(TensorDmaRaceTest, StoreIsAnLdsReadUntilRetired) {
  tensor_store(0, {{.address = 32, .bytes = 16}});
  EXPECT_TRUE(ds_write(1, 40));
}

TEST_F(TensorDmaRaceTest, StoreIsRetiredByTensorcntAndBarrier) {
  tensor_store(0, {{.address = 32, .bytes = 16}});
  wait_tensorcnt(0, 0);
  barrier();
  EXPECT_FALSE(ds_write(1, 40));
}

TEST_F(TensorDmaRaceTest, EveryRangeOfATransferIsTracked) {
  // Probe the gap and the second range; a race is reported once per producer.
  tensor_load(0, {{.address = 0, .bytes = 8}, {.address = 12, .bytes = 8}});
  EXPECT_FALSE(ds_read(1, 8));
  EXPECT_TRUE(ds_read(1, 16));
}

TEST_F(TensorDmaRaceTest, SameWaveLoadOverwritingAPendingStoreSourceRaces) {
  // Transfers from one wave complete in order; the detector conservatively
  // treats their LDS accesses as unordered.
  tensor_store(0, {{.address = 0, .bytes = 16}});
  EXPECT_TRUE(tensor_load(0, {{.address = 4, .bytes = 8}}));
}

TEST_F(TensorDmaRaceTest, SameWaveLoadAfterTensorcntIsSafe) {
  tensor_store(0, {{.address = 0, .bytes = 16}});
  wait_tensorcnt(0, 0);
  EXPECT_FALSE(tensor_load(0, {{.address = 0, .bytes = 16}}));
}

TEST_F(TensorDmaRaceTest, TransferIsCheckedAgainstPendingLdsAccesses) {
  EXPECT_FALSE(ds_read(1, 4));
  const size_t previous = sink->str().size();
  EXPECT_TRUE(tensor_load(0, {{.address = 0, .bytes = 16}}));
  // A transfer has no lanes.
  const std::string report = sink->str().substr(previous);
  EXPECT_NE(report.find("type=LDS access=write"), std::string::npos) << report;
  EXPECT_NE(report.find("wave=0 lane=-1"), std::string::npos) << report;
}

TEST_F(TensorDmaRaceTest, StoreIsCheckedAgainstAPendingDsWrite) {
  // DS writes and tensor transfers complete independently, so the store may
  // read the bytes before the earlier write lands.
  EXPECT_FALSE(ds_write(0, 4));
  waves[0]->pc = kTensorPc - 16; // transfer() advances the PC by 16.
  const size_t previous = sink->str().size();
  EXPECT_TRUE(tensor_store(0, {{.address = 0, .bytes = 16}}));
  const std::string report = sink->str().substr(previous);
  EXPECT_NE(report.find("type=LDS access=read"), std::string::npos) << report;
  EXPECT_NE(report.find("wave=0 lane=-1"), std::string::npos) << report;
  EXPECT_NE(report.find(std::format("==>  0x{:x}  tensor_store_from_lds", kTensorPc)),
            std::string::npos)
      << report;
}

TEST_F(TensorDmaRaceTest, RaceIsReportedAtAByteBothAccessesTouch) {
  // The store reads bytes 0-127 but shares only bytes 120-123 with the DS write.
  EXPECT_FALSE(ds_write(0, 120));
  const size_t previous = sink->str().size();
  EXPECT_TRUE(tensor_store(0, {{.address = 0, .bytes = 128}}));
  const std::string report = sink->str().substr(previous);
  EXPECT_NE(report.find("Race on LDS byte 120 ["), std::string::npos) << report;
  EXPECT_NE(report.find(" reg=120 "), std::string::npos) << report;
}

// A Direct-to-LDS write is checked against pending LDS reads when it issues.
TEST_F(TensorDmaRaceTest, DtlOverwritingAPendingStoreSourceRaces) {
  tensor_store(0, {{.address = 0, .bytes = 16}});
  EXPECT_TRUE(dtl_write(0, 4));
}

// A cluster load issued outside a cluster is an ordinary global load (CDNA5 ISA
// 10.7), so it writes the wave's own LDS whatever its M0 mask says.
TEST_F(TensorDmaRaceTest, ClusterLoadOutsideAClusterOverwritingAPendingStoreSourceRaces) {
  ASSERT_EQ(waves[0]->cluster_size(), 1u);
  ASSERT_EQ(waves[0]->cluster_rank(), 0u);
  tensor_store(0, {{.address = 0, .bytes = 16}});
  EXPECT_TRUE(dtl_write(0, 4, /*cluster_mcast_mask=*/2));
}

TEST_F(TensorDmaRaceTest, ClusterLoadToItsOwnRankOverwritingAPendingStoreSourceRaces) {
  // Inside a cluster, so the mask applies and M0 = 1 selects the wave's rank.
  waves[0]->set_cluster_info(/*rank=*/0, /*size=*/2);
  tensor_store(0, {{.address = 0, .bytes = 16}});
  EXPECT_TRUE(dtl_write(0, 4, /*cluster_mcast_mask=*/1));
}

// Inside a cluster, a multicast that excludes the wave's rank does not write
// its LDS.
TEST_F(TensorDmaRaceTest, ClusterLoadToOnlyAPeerLeavesTheLocalStoreSourceAlone) {
  waves[0]->set_cluster_info(/*rank=*/0, /*size=*/2);
  tensor_store(0, {{.address = 0, .bytes = 16}});
  EXPECT_FALSE(dtl_write(0, 4, /*cluster_mcast_mask=*/2));
}

// A peer-only cluster load writes no local LDS but still takes the issuing
// wave's next ASYNCcnt slot. After an ordinary async load to byte 0 and a
// cluster load to rank 0 only, waiting for all but one completes the first load,
// so a tensor store reading byte 0 is safe; waiting for all but two is not.
TEST_F(TensorDmaRaceTest, PeerOnlyClusterLoadKeepsItsAsynccntSlot) {
  waves[0]->set_cluster_info(/*rank=*/1, /*size=*/2);
  EXPECT_FALSE(dtl_write(0, 0));
  EXPECT_FALSE(dtl_write(0, 64, /*cluster_mcast_mask=*/1));
  wait_asynccnt(0, 1);
  EXPECT_FALSE(tensor_store(0, {{.address = 0, .bytes = 16}}));
}

TEST_F(TensorDmaRaceTest, PeerOnlyClusterLoadLeavesTheEarlierLoadPendingUntilTheWaitCoversIt) {
  waves[0]->set_cluster_info(/*rank=*/1, /*size=*/2);
  EXPECT_FALSE(dtl_write(0, 0));
  EXPECT_FALSE(dtl_write(0, 64, /*cluster_mcast_mask=*/1));
  wait_asynccnt(0, 2);
  EXPECT_TRUE(tensor_store(0, {{.address = 0, .bytes = 16}}));
}

TEST_F(TensorDmaRaceTest, DtlAfterTensorcntIsSafe) {
  tensor_store(0, {{.address = 0, .bytes = 16}});
  wait_tensorcnt(0, 0);
  EXPECT_FALSE(dtl_write(0, 4));
}

// A direct-to-LDS write over the same wave's pending DS read is reported as a
// conservative rule: the ISA does not establish that the write waits for the
// read.
TEST_F(TensorDmaRaceTest, DtlOverwritingAPendingDsReadRaces) {
  EXPECT_FALSE(ds_read(0, 4));
  EXPECT_TRUE(dtl_write(0, 4));
}

TEST_F(TensorDmaRaceTest, DtlAfterDscntIsSafe) {
  EXPECT_FALSE(ds_read(0, 4));
  wait_dscnt(0);
  EXPECT_FALSE(dtl_write(0, 4));
}

TEST_F(TensorDmaRaceTest, StoreAfterDscntIsSafe) {
  EXPECT_FALSE(ds_write(0, 4));
  wait_dscnt(0);
  EXPECT_FALSE(tensor_store(0, {{.address = 0, .bytes = 16}}));
}

TEST_F(TensorDmaRaceTest, RaceFoundByTheTransferIsReportedAtItsInstruction) {
  const std::string report = tensor_load_racing_an_older_read();
  EXPECT_NE(report.find(std::format("==>  0x{:x}  tensor_load_to_lds", kTensorPc)),
            std::string::npos)
      << report;
  EXPECT_NE(report.find("; <-- wave 0"), std::string::npos) << report;
}

TEST_F(TensorDmaRaceTest, RaceFoundLaterIsReportedAtTheTensorInstruction) {
  // The event is registered at issue, at the tensor instruction. Keep wave 0's
  // PCs distinct from wave 1's so the report identifies the event.
  waves[0]->pc = 0x2000;
  const uint64_t instruction_pc = waves[0]->pc + 16;
  tensor_load(0, {{.address = 0, .bytes = 16}});
  const size_t previous = sink->str().size();
  EXPECT_TRUE(ds_read(1, 4));
  const std::string report = sink->str().substr(previous);
  EXPECT_NE(report.find(std::format("==>  0x{:x}  tensor_load_to_lds", instruction_pc)),
            std::string::npos)
      << report;
}

TEST_F(TensorDmaRaceTest, ExecutedTensorLoadIsRegisteredWhenItIssues) {
  // Execute a decoded load of one i32 element into LDS offset 0, so the race
  // detector hears of it through the execution plugin group, as in a run.
  const uint32_t lds_base = fixture.cu()->allocate_lds(256);
  for (auto *wave : waves) {
    wave->set_lds_base(lds_base);
    wave->set_lds_size(256);
  }
  const uint32_t sgprs = waves[0]->sgpr_alloc().base;
  const std::array<uint32_t, 4> group0{1, 0, 0x10000u, 0x80000000u};
  const std::array<uint32_t, 8> group1{2u << 16, 1u << 16, 0, 1u << 16, 0, 0, 0, 0};
  for (uint32_t reg = 0; reg < group0.size(); ++reg)
    fixture.cu()->write_sgpr(sgprs + reg, group0[reg]);
  for (uint32_t reg = 0; reg < group1.size(); ++reg)
    fixture.cu()->write_sgpr(sgprs + 12 + reg, group1[reg]);
  const std::array<uint32_t, 3> words{0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  // Tensor instructions ignore EXEC (CDNA5 ISA 10.11.1), so the load is still
  // registered with every lane of the issuing wave disabled. Wave 1 keeps lane
  // 0 enabled for its DS reads.
  waves[0]->set_exec(0);
  fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[0]->pc, *inst, *waves[0]);
  ASSERT_TRUE(fixture.cu()->execute_instruction(inst.get(), *waves[0]).succeeded());

  EXPECT_FALSE(ds_read(1, lds_base + 4));
  EXPECT_TRUE(ds_read(1, lds_base));
}

TEST_F(TensorDmaRaceTest, TransferWithoutLdsBytesStillTakesATensorcntSlot) {
  // A NULL descriptor or a store wholly outside the tensor moves no data but
  // occupies a TENSORcnt slot, so waiting for all but one transfer completes
  // the load issued before it.
  tensor_load(0, {{.address = 0, .bytes = 16}});
  tensor_store(0, {});
  wait_tensorcnt(0, 1);
  EXPECT_FALSE(ds_read(0, 4));
}

TEST_F(TensorDmaRaceTest, TransferWithoutLdsBytesLeavesTheLoadPendingUntilTheWaitCoversIt) {
  tensor_load(0, {{.address = 0, .bytes = 16}});
  tensor_store(0, {});
  wait_tensorcnt(0, 2);
  EXPECT_TRUE(ds_read(0, 4));
}

// Wave 1 writes LDS and may end; wave 0 then passes a barrier and reads the
// bytes. The fixture's DS helpers are not specific to tensor DMA.
class ExitedWaveRaceTest : public TensorDmaRaceTest {
protected:
  void write_and_exit(bool wait) {
    EXPECT_FALSE(ds_write(1, 64));
    if (wait)
      wait_dscnt(1);
    waves[1]->end();
  }

  void barrier_without_wave_1(BarrierKind kind) {
    std::array<Wavefront *, 1> remaining{waves[0]};
    fixture.plugin_group_->onAmdgpuBarrierResolved(remaining, kind);
  }
};

// A workgroup barrier among the remaining waves cannot resolve until wave 1
// has ended, so it orders the write completed by its explicit or implicit wait.
TEST_F(ExitedWaveRaceTest, WorkgroupBarrierRetiresCompletedWrite) {
  write_and_exit(/*wait=*/true);
  barrier_without_wave_1(BarrierKind::Workgroup);
  EXPECT_FALSE(ds_read(0, 64));
}

TEST_F(ExitedWaveRaceTest, WorkgroupBarrierRetiresImplicitlyWaitedWrite) {
  write_and_exit(/*wait=*/false);
  barrier_without_wave_1(BarrierKind::Workgroup);
  EXPECT_FALSE(ds_read(0, 64));
}

TEST_F(ExitedWaveRaceTest, ExitAloneDoesNotOrderTheWritersLdsAccess) {
  write_and_exit(/*wait=*/false);
  EXPECT_TRUE(ds_read(0, 64));
}

TEST_F(ExitedWaveRaceTest, ExitWaitAlsoCompletesTensorDmaBeforeTheWorkgroupBarrier) {
  EXPECT_FALSE(tensor_load(1, {{.address = 64, .bytes = 16}}));
  waves[1]->end();
  barrier_without_wave_1(BarrierKind::Workgroup);
  EXPECT_FALSE(ds_read(0, 64));
}

TEST_F(ExitedWaveRaceTest, NamedBarrierDoesNotRetireAnUnjoinedLiveWave) {
  EXPECT_FALSE(ds_write(1, 64));
  wait_dscnt(1);
  barrier_without_wave_1(BarrierKind::Named);
  EXPECT_TRUE(ds_read(0, 64));
}

// Named and cluster barriers do not wait for every wave of the workgroup, so
// they could have resolved before wave 1 ended.
TEST_F(ExitedWaveRaceTest, NamedAndClusterBarriersDoNotRetireCompletedWrite) {
  write_and_exit(/*wait=*/false);
  barrier_without_wave_1(BarrierKind::Named);
  barrier_without_wave_1(BarrierKind::Cluster);
  EXPECT_TRUE(ds_read(0, 64));
}

// Run a two-wave CDNA4 kernel: wave 1 writes LDS, optionally waits for the
// write, and ends; wave 0 passes s_barrier and reads the same bytes. Returns
// the race report.
std::string run_exited_writer_kernel(bool wait) {
  PluginFixture f;
  EXPECT_NO_FATAL_FAILURE(f.initialize());
  PluginSinkConfig config;
  auto &sink = config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
  EXPECT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  constexpr uint16_t kVgprBase = 256;
  constexpr uint16_t kInline2 = 130;
  constexpr uint8_t kInline64 = 192;
  const auto first_id =
      cdna4::build_vop1(cdna4::kVReadfirstlaneB32Vop1, {.src0 = kVgprBase + 0, .vdst = 4});
  const auto address =
      cdna4::build_vop2(cdna4::kVLshlrevB32Vop2, {.src0 = kInline2, .vsrc1 = 0, .vdst = 1});
  const auto is_wave_0 = cdna4::build_sopc(cdna4::kSCmpLtU32Sopc, {.ssrc0 = 4, .ssrc1 = kInline64});
  const auto to_reader = cdna4::build_sopp(cdna4::kSCbranchScc1Sopp, {.simm16 = 4});
  const auto write = cdna4::build_ds(cdna4::kDsWriteB32Ds, {.addr = 1, .data0 = 0});
  // offset1 is the high byte of the 16-bit offset, so the read is at v1 + 256.
  const auto read = cdna4::build_ds(cdna4::kDsReadB32Ds, {.offset1 = 1, .addr = 1, .vdst = 2});
  const uint32_t lgkm_wait = legacy_wait_word(ROCJITSU_CODE_ARCH_CDNA4, 63, 0);
  const uint32_t code[] = {
      first_id[0],
      address[0],
      is_wave_0[0],
      to_reader[0], // Waves with work-item IDs below 64 read.
      write[0],
      write[1], // Wave 1 writes bytes [256, 512).
      wait ? lgkm_wait : S_NOP,
      S_ENDPGM,
      S_BARRIER,
      read[0],
      read[1], // Wave 0 reads bytes [256, 512).
      lgkm_wait,
      S_ENDPGM,
  };
  const uint64_t kernel = f.write_kernel(0x1000, code, std::size(code), /*sgprs=*/12,
                                         /*group_segment_fixed_size=*/512);
  test::AqlQueue queue(f.mem, f.cp());
  queue.dispatch(kernel, /*grid=*/128, /*workgroup=*/128);
  f.run_until_idle();
  return sink.str();
}

TEST(RaceDetectorPluginTest, BarrierAfterWriterEndsOrdersItsWaitedLdsWrite) {
  const std::string report = run_exited_writer_kernel(/*wait=*/true);
  EXPECT_EQ(report.find("RACE "), std::string::npos) << report;
}

TEST(RaceDetectorPluginTest, BarrierAfterWriterEndsOrdersItsImplicitlyWaitedLdsWrite) {
  const std::string report = run_exited_writer_kernel(/*wait=*/false);
  EXPECT_EQ(report.find("RACE "), std::string::npos) << report;
}

/// Executes decoded tensor DMA instructions, so the emulator itself reports
/// them to the race detector. Both waves share one 256-byte LDS allocation.
class DecodedTensorDmaRaceTest : public TensorDmaRaceTest {
protected:
  static constexpr uint64_t kGlobal = 0x10000;
  static constexpr uint32_t kTileOffset = 0;
  static constexpr uint32_t kEmptyOffset = 64;
  /// The descriptor's data_size encoding for eight-byte elements.
  static constexpr uint32_t kI64DataSizeEncoding = 3;
  /// An i32 element at offset 254 straddles the end of the 256-byte
  /// allocation: the transfer moves only its first two bytes, so its LDS range
  /// ends mid-element. The race detector tracks those bytes at byte
  /// granularity, so a peer DWORD access at kPeerDwordOffset overlaps them;
  /// DWORD granularity would drop them.
  static constexpr uint32_t kPartialI32Offset = 254;
  static constexpr uint32_t kPeerDwordOffset = 252;

  struct Descriptor {
    uint32_t lds_offset = kTileOffset;
    uint32_t tensor_dim0 = 4; ///< Zero places the whole tile outside the tensor.
    uint32_t tile_dim0 = 4;   ///< Zero makes the tile a NOP.
    uint32_t tile_dim1 = 0;
    uint32_t count = 1;     ///< Zero makes this a NULL tensor.
    uint32_t data_size = 2; ///< D# data_size encoding: elements are 1 << data_size bytes.
  };

  void SetUp() override {
    TensorDmaRaceTest::SetUp();
    const uint32_t lds_base = fixture.cu()->allocate_lds(256);
    for (auto *wave : waves) {
      wave->set_lds_base(lds_base);
      wave->set_lds_size(256);
    }
  }

  /// Decode and execute one tensor DMA instruction on `wave`: one row. Unlike
  /// record_tensor_instruction(), this runs the instruction, so the race
  /// detector hears of it through the issue hook as in a run.
  void execute_decoded_tensor(int wave, bool is_load, const Descriptor &desc) {
    waves[wave]->pc += 16;
    const uint32_t sgprs = waves[wave]->sgpr_alloc().base;
    const std::array<uint32_t, 4> group0{desc.count, desc.lds_offset,
                                         static_cast<uint32_t>(kGlobal), 0x80000000u};
    const std::array<uint32_t, 8> group1{desc.data_size << 16,
                                         desc.tensor_dim0 << 16,
                                         1u << 16,
                                         desc.tile_dim0 << 16,
                                         desc.tile_dim1,
                                         4,
                                         0,
                                         0};
    for (uint32_t reg = 0; reg < group0.size(); ++reg)
      fixture.cu()->write_sgpr(sgprs + reg, group0[reg]);
    for (uint32_t reg = 0; reg < group1.size(); ++reg)
      fixture.cu()->write_sgpr(sgprs + 12 + reg, group1[reg]);
    const std::array<uint32_t, 3> words{is_load ? 0xd0710001u : 0xd0714001u, 0x7c000000u,
                                        0x7c7c0c00u};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(inst->mnemonic(), is_load ? "tensor_load_to_lds" : "tensor_store_from_lds");
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[wave]->pc, *inst, *waves[wave]);
    EXPECT_TRUE(fixture.cu()->execute_instruction(inst.get(), *waves[wave]).succeeded());
  }

  /// Load a tile, issue a transfer that moves no data, wait until at most
  /// `remaining` transfers are outstanding, and read the loaded tile.
  bool load_then_empty_then_read(uint16_t remaining, bool empty_is_load, const Descriptor &empty) {
    execute_decoded_tensor(0, /*is_load=*/true, {});
    execute_decoded_tensor(0, empty_is_load, empty);
    wait_tensorcnt(0, remaining);
    return ds_read(0, waves[0]->lds_base() + kTileOffset + 4);
  }
};

// Each tensor instruction completes on TENSORcnt once, in issue order with the
// others, including one that moves no data. Waiting for all but one therefore
// completes the load issued before it.
TEST_F(DecodedTensorDmaRaceTest, StoreOutsideTheTensorTakesATensorcntSlot) {
  EXPECT_FALSE(load_then_empty_then_read(/*remaining=*/1, /*empty_is_load=*/false,
                                         {.lds_offset = kEmptyOffset, .tensor_dim0 = 0}));
}

TEST_F(DecodedTensorDmaRaceTest, NullTensorStoreTakesATensorcntSlot) {
  EXPECT_FALSE(load_then_empty_then_read(/*remaining=*/1, /*empty_is_load=*/false,
                                         {.lds_offset = kEmptyOffset, .count = 0}));
}

TEST_F(DecodedTensorDmaRaceTest, NullTensorLoadTakesATensorcntSlot) {
  EXPECT_FALSE(load_then_empty_then_read(/*remaining=*/1, /*empty_is_load=*/true,
                                         {.lds_offset = kEmptyOffset, .count = 0}));
}

TEST_F(DecodedTensorDmaRaceTest, NopTileLoadTakesATensorcntSlot) {
  // With tile_dim1 = 0 the zero-width tile has rank zero, a shape accepted
  // before NOP tiles were; tile_dim1 = 1 exercises the newly accepted {0,1}.
  EXPECT_FALSE(load_then_empty_then_read(
      /*remaining=*/1,
      /*empty_is_load=*/true, {.lds_offset = kEmptyOffset, .tile_dim0 = 0, .tile_dim1 = 1}));
}

// With both transfers outstanding, the load is still pending, so the read races.
// Had the NOP tile taken more than one slot, waiting for two would complete the
// load.
TEST_F(DecodedTensorDmaRaceTest, NopTileLoadStaysPendingUntilTheWaitCoversIt) {
  EXPECT_TRUE(load_then_empty_then_read(
      /*remaining=*/2, /*empty_is_load=*/true,
      {.lds_offset = kEmptyOffset, .tile_dim0 = 0, .tile_dim1 = 1}));
}

// Likewise for a store outside the tensor.
TEST_F(DecodedTensorDmaRaceTest, LoadStaysPendingUntilTheWaitCoversIt) {
  EXPECT_TRUE(load_then_empty_then_read(/*remaining=*/2, /*empty_is_load=*/false,
                                        {.lds_offset = kEmptyOffset, .tensor_dim0 = 0}));
}

// Partial elements at the end of the allocation; see kPartialI32Offset.

TEST_F(DecodedTensorDmaRaceTest, PartialElementLoadIsAnLdsWrite) {
  execute_decoded_tensor(0, /*is_load=*/true, {.lds_offset = kPartialI32Offset, .tile_dim0 = 1});
  EXPECT_TRUE(ds_read(1, waves[1]->lds_base() + kPeerDwordOffset));
}

TEST_F(DecodedTensorDmaRaceTest, MergedLoadKeepsItsPartialTail) {
  // Elements at 244 and 252 merge into one range of twelve bytes.
  execute_decoded_tensor(0, /*is_load=*/true,
                         {.lds_offset = 244, .tile_dim0 = 2, .data_size = kI64DataSizeEncoding});
  EXPECT_TRUE(ds_read(1, waves[1]->lds_base() + 252));
}

TEST_F(DecodedTensorDmaRaceTest, PartialElementStoreIsAnLdsRead) {
  execute_decoded_tensor(0, /*is_load=*/false, {.lds_offset = kPartialI32Offset, .tile_dim0 = 1});
  EXPECT_TRUE(ds_write(1, waves[1]->lds_base() + kPeerDwordOffset));
}

TEST_F(DecodedTensorDmaRaceTest, PartialElementLoadIsCheckedAgainstAnOlderRead) {
  EXPECT_FALSE(ds_read(1, waves[1]->lds_base() + kPeerDwordOffset));
  const size_t previous = sink->str().size();
  execute_decoded_tensor(0, /*is_load=*/true, {.lds_offset = kPartialI32Offset, .tile_dim0 = 1});
  EXPECT_NE(sink->str().find("RACE ", previous), std::string::npos) << sink->str();
}

// Controls: whole 8-byte elements inside the allocation.
TEST_F(DecodedTensorDmaRaceTest, WholeElementLoadIsAnLdsWrite) {
  execute_decoded_tensor(0, /*is_load=*/true,
                         {.lds_offset = 240, .tile_dim0 = 1, .data_size = kI64DataSizeEncoding});
  EXPECT_TRUE(ds_read(1, waves[1]->lds_base() + 244));
}

TEST_F(DecodedTensorDmaRaceTest, WholeElementStoreIsAnLdsRead) {
  execute_decoded_tensor(0, /*is_load=*/false,
                         {.lds_offset = 240, .tile_dim0 = 1, .data_size = kI64DataSizeEncoding});
  EXPECT_TRUE(ds_write(1, waves[1]->lds_base() + 244));
}

class FaultingLdsStorePage final : public AddressSpaceTranslator {
public:
  VmTranslationResult translate(uint64_t address, size_t size, VmAccessKind access) const override {
    if (address == 0x4000 && access == VmAccessKind::Write)
      return {};
    return IdentityAddressSpaceTranslator{}.translate(address, size, access);
  }
};

class FeedbackImagePages final : public AddressSpaceTranslator {
public:
  bool unavailable = false;
  bool sparse = true;

  std::shared_ptr<const PrtRanges> nonresident_prt_ranges() const override {
    static const auto holes = std::make_shared<const PrtRanges>(PrtRanges{{0x11000, 0x12000}});
    return sparse ? holes : nullptr;
  }

  VmTranslationResult translate(uint64_t addr, size_t size, VmAccessKind kind) const override {
    if (addr >= 0x10000 && unavailable)
      return {.outcome = VmAccessOutcome::Unavailable, .translation = {}};
    if (sparse && (addr >> 12) == 0x11)
      return {};
    return IdentityAddressSpaceTranslator{}.translate(addr, size, kind);
  }
};

TEST(RaceDetectorPluginTest, Gfx11ImageFeedbackCountsEmptyExecOnceWhenVmcntIsPending) {
  for (std::string_view arch : {"rdna3", "rdna3_5"}) {
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 32, 96, 32));
    PluginSinkConfig config;
    auto &sink = config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    auto race = std::make_unique<RaceDetectorPlugin>();
    auto *plugin = race.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(race)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    uint64_t pc = 0x8000;
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, pc, 96, 32, 32);
    ASSERT_NE(wf, nullptr);
    std::array waves{wf};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 96, waves);
    const std::array<uint32_t, 8> resource{0x100, 20u << 20, 0, (9u << 28) | 0x204, 0, 256u << 8};
    for (unsigned i = 0; i < resource.size(); ++i)
      cu->write_sgpr(wf->sgpr_alloc().base + i, resource[i]);
    f.mem->write32(0x10000, 42);
    const auto older = rdna3_5::build_mimg(rdna3_5::kImageLoadMimg,
                                           {.dim = 1, .unorm = 1, .dmask = 1, .vdata = 12});
    const auto drain = rdna3_5::build_sopp(rdna3_5::kSWaitcntSopp)[0];
    for (bool pending : {false, true}) {
      for (bool tfe : {false, true}) {
        for (uint64_t exec : {0u, 1u}) {
          for (int threshold : {-1, 0, 1, 2}) {
            SCOPED_TRACE(testing::Message() << arch << " pending=" << pending << " tfe=" << tfe
                                            << " exec=" << exec << " wait=" << threshold);
            const auto image = rdna3_5::build_mimg(
                rdna3_5::kImageLoadMimg,
                {.dim = 1, .unorm = 1, .dmask = 1, .vdata = 8, .tfe = uint8_t(tfe)});
            std::vector<uint32_t> code(older.begin(), older.end());
            if (!pending)
              code.push_back(drain);
            code.insert(code.end(), image.begin(), image.end());
            if (threshold >= 0)
              code.push_back(rdna3_5::build_sopp(
                  rdna3_5::kSWaitcntSopp,
                  {.simm16 = uint16_t((unsigned(threshold) << 10) | (0x3fu << 4) | 7u)})[0]);
            code.push_back(
                rdna3_5::build_vop1(rdna3_5::kVMovB32Vop1, {.src0 = 268, .vdst = 13})[0]);
            code.push_back(drain);
            f.mem->load_image(reinterpret_cast<const uint8_t *>(code.data()), code.size() * 4, pc);
            wf->pc = pc;
            wf->set_exec(1);
            ASSERT_TRUE(cu->step());
            if (!pending) {
              ASSERT_TRUE(cu->step());
            }
            wf->set_exec(exec);
            ASSERT_TRUE(cu->step());
            const auto &state =
                *static_cast<RaceWavefrontState *>(wf->plugin_state(plugin->slot_index()))
                     ->race_state;
            unsigned slots = 0;
            for (auto event : state.getWaveMemoryEvents())
              slots += state.getDetector()->events().pendingCounterIncrement(
                  event, WaitCounterType::VMCNT);
            EXPECT_EQ(slots, unsigned(pending) + unsigned(pending || exec));
            wf->set_exec(1);
            if (threshold >= 0) {
              ASSERT_TRUE(cu->step());
            }
            const auto core = cu->memory_wait_diagnostic_count(), observer = sink.str().size();
            ASSERT_TRUE(cu->step());
            const bool hazard = pending && (threshold < 0 || threshold >= 2);
            EXPECT_EQ(cu->memory_wait_diagnostic_count() != core, hazard);
            EXPECT_EQ(sink.str().substr(observer).find("Race on VGPR v12") != std::string::npos,
                      hazard);
            EXPECT_EQ(wf->debug_read_vgpr(13, 0), 42u);
            ASSERT_TRUE(cu->step());
            pc += 64;
          }
        }
      }
    }
    wf->halt();
  }
}

TEST(RaceDetectorPluginTest, Gfx11ConditionalImageFeedbackSharesThePayloadCounter) {
  constexpr uint32_t sentinel = 0xabcd1234;
  constexpr uint64_t pc = 0x8000;
  for (std::string_view arch : {"rdna3", "rdna3_5"}) {
    for (bool warning : {false, true}) {
      for (bool wait : {false, true}) {
        for (unsigned lane : {0u, 1u}) {
          for (bool write : {false, true}) {
            SCOPED_TRACE(testing::Message() << arch << " warning=" << warning << " wait=" << wait
                                            << " lane=" << lane << " write=" << write);
            GpuVm vm;
            PluginFixture f;
            ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 32, 96, 32));
            PluginSinkConfig config;
            auto &sink = config.emplace<StringSink>();
            f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
            auto race = std::make_unique<RaceDetectorPlugin>();
            auto *plugin = race.get();
            ASSERT_TRUE(f.plugin_group_->add(std::move(race)));
            f.soc->set_plugin_group(f.plugin_group_);
            f.plugin_group_->onInit();
            auto pages = std::make_shared<FeedbackImagePages>();
            pages->sparse = !warning;
            auto handle = vm.register_address_space(
                1, pages, std::make_shared<GpuMemoryPhysicalAccess>(*f.mem));
            auto *cu = f.cu();
            cu->set_gpu_vm(&vm);
            auto *wf = cu->dispatch_wf(0, pc, 96, 32, 32);
            ASSERT_NE(wf, nullptr);
            wf->set_address_space(handle);
            wf->set_exec(3);
            std::array waves{wf};
            f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 96, waves);

            // Lane 0 misses the second page or requests a LOD below 1.5; lane 1 succeeds.
            const std::array<uint32_t, 8> resource =
                warning ? std::array<uint32_t, 8>{0x100,
                                                  (22u << 20) | (3u << 30) | (3u << 16),
                                                  15u | (63u << 14),
                                                  (9u << 28) | (3u << 16) | 0x204,
                                                  0,
                                                  384u << 8}
                        : std::array<uint32_t, 8>{0x100, (20u << 20) | (3u << 30),
                                                  511,   (9u << 28) | 0x204,
                                                  2047,  256u << 8};
            const auto sb = wf->sgpr_alloc().base;
            for (unsigned i = 0; i < resource.size(); ++i)
              cu->write_sgpr(sb + i, resource[i]);
            cu->write_sgpr(sb + 8, 2u | (2u << 3));
            cu->write_sgpr(sb + 9, 0xfffu << 12);
            cu->write_sgpr(sb + 10, 1u << 26);
            for (unsigned i : {0u, 1u}) {
              wf->debug_write_vgpr(0, i, !warning && i == 0 ? 1024u : 0u);
              wf->debug_write_vgpr(2, i, std::bit_cast<uint32_t>(2.0f * i));
              wf->debug_write_vgpr(10, i, sentinel);
            }
            const auto image =
                rdna3_5::build_mimg(warning ? rdna3_5::kImageSampleLMimg : rdna3_5::kImageLoadMimg,
                                    {.dim = 1,
                                     .unorm = uint8_t(!warning),
                                     .dmask = 7,
                                     .d16 = 1,
                                     .vdata = 8,
                                     .tfe = uint8_t(!warning),
                                     .lwe = uint8_t(warning),
                                     .ssamp = uint8_t(warning ? 2 : 0)});
            std::vector<uint32_t> code(image.begin(), image.end());
            const bool retry = !warning && wait;
            if (retry)
              code.push_back(rdna3_5::build_sopp(rdna3_5::kSNopSopp)[0]);
            if (wait)
              code.push_back(rdna3_5::build_sopp(rdna3_5::kSWaitcntSopp)[0]);
            code.push_back(
                rdna3_5::build_vop1(rdna3_5::kVMovB32Vop1, {.src0 = uint16_t(write ? 128u : 266u),
                                                            .vdst = uint8_t(write ? 10 : 12)})[0]);
            f.mem->load_image(reinterpret_cast<const uint8_t *>(code.data()), 4 * code.size(), pc);
            pages->unavailable = retry;
            ASSERT_TRUE(cu->step());
            auto &state = *static_cast<RaceWavefrontState *>(wf->plugin_state(plugin->slot_index()))
                               ->race_state;
            if (pages->unavailable) {
              ASSERT_EQ(wf->state(), WfState::VM_RETRY);
              EXPECT_TRUE(state.getWaveMemoryEvents().empty());
              pages->unavailable = false;
              for (unsigned i = 0; i < 8 && wf->pc != pc + 4 * (image.size() + 1); ++i)
                cu->step();
            }
            const auto &events = state.getWaveMemoryEvents();
            ASSERT_EQ(events.size(), 1u);
            const auto &registry = state.getDetector()->events();
            EXPECT_EQ(registry.pendingCounterIncrement(events.front(), WaitCounterType::VMCNT), 1u);
            EXPECT_EQ(registry.registerByteMask(events.front(), 9),
                      ExecutionPlugin::kLowHalfByteMask);
            EXPECT_EQ(registry.registerByteMask(events.front(), 10),
                      ExecutionPlugin::kFullByteMask);
            ASSERT_EQ(wf->debug_read_vgpr(10, 0), warning ? 2u : 1u);
            ASSERT_EQ(wf->debug_read_vgpr(10, 1), sentinel);
            const auto vb = wf->vgpr_alloc().base;
            f.plugin_group_->onAmdgpuReadVgprLanes(wf, vb + 9, 3,
                                                   ExecutionPlugin::kHighHalfByteMask);
            EXPECT_EQ(sink.str().find("Race on VGPR v9"), std::string::npos);
            wf->set_exec(uint64_t{1} << lane);
            if (wait) {
              ASSERT_TRUE(cu->step());
            }
            ASSERT_TRUE(cu->step());
            ASSERT_EQ(wf->pc, pc + 4 * code.size());
            EXPECT_EQ(sink.str().find("Race on VGPR v10") != std::string::npos, !wait && lane == 0);
            wf->set_address_space({});
            cu->set_gpu_vm(nullptr);
            wf->halt();
          }
        }
      }
    }
  }
}

TEST(RaceDetectorPluginTest, Cdna2ClaimedLdsToBufferFaultLeavesNoOutstandingLdsRead) {
  for (const bool fault : {false, true}) {
    for (const bool wait : {false, true}) {
      SCOPED_TRACE(testing::Message() << "fault=" << fault << " wait=" << wait);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(1, "cdna2", 64, 102));
      PluginSinkConfig config;
      auto &sink = config.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      auto *cu = f.cu();
      GpuVm vm;
      std::shared_ptr<AddressSpaceTranslator> translator;
      if (fault)
        translator = std::make_shared<FaultingLdsStorePage>();
      else
        translator = std::make_shared<IdentityAddressSpaceTranslator>();
      auto space = vm.register_address_space(5, std::move(translator),
                                             std::make_shared<GpuMemoryPhysicalAccess>(*f.mem));
      ASSERT_TRUE(space);
      cu->set_gpu_vm(&vm);
      constexpr uint64_t pc = 0x8000;
      auto *wave = cu->dispatch_wf(0, pc, 32, 32);
      ASSERT_NE(wave, nullptr);
      wave->set_process_id(5);
      wave->set_address_space(space);
      wave->set_lds_size(512);
      wave->set_exec(1);
      wave->set_m0(0);
      const auto sb = wave->sgpr_alloc().base;
      cu->write_sgpr(sb, 0x4000);
      cu->write_sgpr(sb + 1, 4u << 16);
      cu->write_sgpr(sb + 2, 256);
      cu->write_sgpr(sb + 3, 1u << 23);
      cu->write_vgpr(wave->vgpr_alloc().base + 2, 0, 99);
      const auto buffer =
          cdna2::build_mubuf(cdna2::kBufferStoreLdsDword, {.lds = 1, .soffset = 128});
      const auto ds = cdna2::build_ds(cdna2::kDsWriteB32, {.addr = 0, .data0 = 2});
      const uint32_t code[] = {buffer[0], buffer[1], wait ? 0xbf8c0000u : 0xbf800000u, ds[0],
                               ds[1]};
      f.mem->load_image(reinterpret_cast<const uint8_t *>(code), sizeof(code), pc);
      std::array<Wavefront *, 1> waves{wave};
      cu->begin_workgroup(0, 0, 1);
      f.plugin_group_->onAmdgpuWorkgroupDispatched(0, 0, 32, 32, waves);
      bool claimed = false;
      cu->set_memory_violation_handler([&](Wavefront &, uint64_t, bool, uint64_t issue_pc) {
        EXPECT_EQ(issue_pc, pc);
        claimed = true;
        return true;
      });
      for (unsigned tick = 0; tick < 32 && wave->pc != pc + sizeof(code); ++tick)
        cu->step();
      ASSERT_EQ(wave->pc, pc + sizeof(code));
      EXPECT_EQ(claimed, fault);
      EXPECT_EQ(cu->lds().read32(0), 99u);
      const bool race = sink.str().find("type=LDS access=write reg=0") != std::string::npos;
      EXPECT_EQ(race, !fault && !wait) << sink.str();
      wave->set_address_space({});
      cu->set_gpu_vm(nullptr);
      wave->halt();
    }
  }
}

TEST(MemoryWaitExecutionTest, Cdna2FaultingLdsToBufferKeepsOlderDsScoreboardPosition) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(1, "cdna2", 64, 102));
  auto *cu = f.cu();
  GpuVm vm;
  auto space = vm.register_address_space(5, std::make_shared<FaultingLdsStorePage>(),
                                         std::make_shared<GpuMemoryPhysicalAccess>(*f.mem));
  ASSERT_TRUE(space);
  cu->set_gpu_vm(&vm);
  constexpr uint64_t pc = 0x8000;
  auto *wave = cu->dispatch_wf(0, pc, 32, 32);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(5);
  wave->set_address_space(space);
  wave->set_lds_size(512);
  wave->set_exec(1);
  wave->set_m0(0);
  const auto sb = wave->sgpr_alloc().base;
  cu->write_sgpr(sb, 0x4000);
  cu->write_sgpr(sb + 1, 4u << 16);
  cu->write_sgpr(sb + 2, 256);
  cu->write_sgpr(sb + 3, 1u << 23);
  cu->write_vgpr(wave->vgpr_alloc().base, 0, 0);
  const auto ds = cdna2::build_ds(cdna2::kDsReadB32, {.addr = 0, .vdst = 8});
  const auto store = cdna2::build_mubuf(cdna2::kBufferStoreLdsDword, {.lds = 1, .soffset = 128});
  const uint32_t code[] = {ds[0], ds[1], store[0], store[1]};
  f.mem->load_image(reinterpret_cast<const uint8_t *>(code), sizeof(code), pc);
  bool claimed = false;
  cu->set_memory_violation_handler([&](Wavefront &, uint64_t, bool, uint64_t issue_pc) {
    claimed = true;
    EXPECT_EQ(issue_pc, pc + 8);
    return true;
  });
  ASSERT_TRUE(cu->step());
  ASSERT_EQ(wave->pc, pc + 8);
  const auto *scoreboard = wave->memory_wait_scoreboard();
  ASSERT_NE(scoreboard, nullptr);
  ASSERT_EQ(scoreboard->outstanding(WaitCounterKind::Ds), 1u);
  ASSERT_TRUE(cu->step());
  ASSERT_TRUE(claimed);
  EXPECT_EQ(scoreboard->outstanding(WaitCounterKind::Ds), 1u);
  EXPECT_EQ(scoreboard->outstanding(WaitCounterKind::Load), 0u);
  wave->set_address_space({});
  cu->set_gpu_vm(nullptr);
  wave->halt();
}

TEST(RaceDetectorPluginTest, Cdna2DsChecksAndRecordsOnlyCompleteElementsInWorkgroupLds) {
  struct Case {
    uint16_t store_opcode;
    uint32_t store_offset;
    uint8_t second_offset;
    uint32_t read_offset;
    unsigned stored_bytes;
    bool race;
  };
  for (const Case test : {Case{cdna2::kDsWriteB32, 1024, 0, 1024, 0, false},
                          Case{cdna2::kDsWriteB32, 1020, 0, 1020, 4, true},
                          Case{cdna2::kDsWriteB128, 1016, 0, 1024, 0, false},
                          Case{cdna2::kDsWrite2B32, 1020, 1, 1024, 4, false},
                          Case{cdna2::kDsWrite2B32, 1016, 1, 1020, 8, true}}) {
    SCOPED_TRACE(testing::Message() << test.store_opcode << " store=" << test.store_offset
                                    << " read=" << test.read_offset);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(2, "cdna2", 64, 102));
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *plugin_ptr = plugin.get();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    std::array<Wavefront *, 2> waves{cu->dispatch_wf(0, 0x100, 102, 256, 64),
                                     cu->dispatch_wf(0, 0x200, 102, 256, 64)};
    constexpr uint32_t base = 1024;
    for (auto *wave : waves) {
      ASSERT_NE(wave, nullptr);
      wave->set_exec(1);
      wave->set_lds_base(base);
      wave->set_lds_size(1024);
    }
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 512, 204, waves);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
    ASSERT_NE(decoder, nullptr);
    const auto route = [&](unsigned wave_idx, uint16_t op, uint32_t offset, uint8_t offset1) {
      auto *wave = waves[wave_idx];
      cu->write_vgpr(wave->vgpr_alloc().base, 0, offset);
      const auto words =
          cdna2::build_ds(op, {.offset1 = offset1, .addr = 0, .data0 = 4, .data1 = 8, .vdst = 12});
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      ASSERT_NE(inst, nullptr);
      f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave->pc, *inst, *wave);
      ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
      test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), *wave);
    };
    ASSERT_NO_FATAL_FAILURE(route(0, test.store_opcode, test.store_offset, test.second_offset));
    ASSERT_NO_FATAL_FAILURE(route(1, cdna2::kDsReadB32, test.read_offset, 0));
    auto *state =
        static_cast<RaceWavefrontState *>(waves[0]->plugin_state(plugin_ptr->slot_index()));
    auto &events = state->race_state->getDetector()->events();
    ASSERT_EQ(events.totalAllocated(), 2);
    EXPECT_EQ(events.ldsIntervals(EventId{0}).getTotalBytes(), test.stored_bytes);
    EXPECT_FALSE(events.ldsIntervals(EventId{0}).contains(base + 1024));
    EXPECT_EQ(events.ldsIntervals(EventId{1}).getTotalBytes(), test.read_offset < 1024 ? 4u : 0u);
    EXPECT_EQ(events.execMask(EventId{1}), 1u);
    EXPECT_EQ(events.registers(EventId{1}).size(), 1u);
    EXPECT_EQ(sink.str().find("RACE ") != std::string::npos, test.race);
  }
}

TEST(RaceDetectorPluginTest, D16LoadTracksFullDwordWhenSramEccEnabled) {
  auto opposite_half_read_reports_race = [](std::string_view arch, uint32_t wavefront_size,
                                            bool formatted) {
    PluginFixture f;
    f.initialize(/*num_wf_slots=*/1, arch, wavefront_size);
    if (testing::Test::HasFatalFailure())
      return false;
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    if (!f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>())) {
      ADD_FAILURE() << "failed to attach race detector";
      return false;
    }
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
    EXPECT_NE(wf, nullptr);
    if (wf == nullptr)
      return false;
    wf->set_exec(1u);
    std::array<amdgpu::Wavefront *, 1> waves{wf};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(
        /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256, /*sgpr_count=*/104, waves);

    constexpr uint32_t kDst = 4;
    auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
    state->elem_size = 1;
    state->num_elems = 1;
    state->is_load = true;
    state->wf_size = wf->wf_size();
    state->exec_mask = 1;
    state->lane_mask = 1;
    state->dst_reg_base = wf->vgpr_alloc().base + kDst;
    state->d16_lo = !formatted;
    state->buffer_d16 = formatted;
    state->buffer_components = formatted ? 3 : 0;
    TestMemoryInstruction load(std::move(state));
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, load, *wf);
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

    f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kDst + (formatted ? 1 : 0),
                                           /*lane_mask=*/1, ExecutionPlugin::kHighHalfByteMask);
    return sink.str().find("RACE ") != std::string::npos;
  };

  for (bool formatted : {false, true}) {
    SCOPED_TRACE(formatted);
    EXPECT_TRUE(opposite_half_read_reports_race("cdna4", /*wavefront_size=*/64, formatted));
    EXPECT_TRUE(opposite_half_read_reports_race("cdna5", /*wavefront_size=*/32, formatted));
    EXPECT_FALSE(opposite_half_read_reports_race("rdna4", /*wavefront_size=*/32, formatted));
  }
}

TEST(RaceDetectorPluginTest, CompletedWorkgroupReleasesOnlyItsDetector) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/3));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  auto plugin = std::make_unique<RaceDetectorPlugin>();
  RaceDetectorPlugin *race = plugin.get();
  ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *first = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  auto *second = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  auto *next_dispatch = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(next_dispatch, nullptr);
  first->set_exec(1u);
  second->set_exec(1u);
  next_dispatch->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> first_waves{first};
  std::array<amdgpu::Wavefront *, 1> second_waves{second};
  std::array<amdgpu::Wavefront *, 1> next_dispatch_waves{next_dispatch};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0,
                                               /*physical_vgpr_count=*/256,
                                               /*physical_sgpr_count=*/104, first_waves);
  f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/1,
                                               /*physical_vgpr_count=*/256,
                                               /*physical_sgpr_count=*/104, second_waves);
  f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/2, /*wg_id=*/0,
                                               /*physical_vgpr_count=*/256,
                                               /*physical_sgpr_count=*/104, next_dispatch_waves);
  EXPECT_EQ(race->retainedWorkgroupCount(), 3u);

  // Workgroup 0 of dispatch 2 keeps its detector.
  f.plugin_group_->onAmdgpuWorkgroupCompleted(/*dispatch_id=*/1, /*wg_id=*/0);
  EXPECT_EQ(race->retainedWorkgroupCount(), 2u);

  // The other workgroup of the dispatch still detects a read before its load
  // completes.
  constexpr uint32_t kDst = 4;
  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->elem_size = 4;
  state->num_elems = 1;
  state->is_load = true;
  state->wf_size = second->wf_size();
  state->exec_mask = 1;
  state->lane_mask = 1;
  state->dst_reg_base = second->vgpr_alloc().base + kDst;
  TestMemoryInstruction load(std::move(state));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(second->pc, load, *second);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *second);
  f.plugin_group_->onAmdgpuReadVgprLanes(second, second->vgpr_alloc().base + kDst,
                                         /*lane_mask=*/1);
  EXPECT_NE(sink.str().find("RACE "), std::string::npos);

  const std::string summary = race->getSummary();
  f.plugin_group_->onAmdgpuWorkgroupCompleted(/*dispatch_id=*/1, /*wg_id=*/1);
  f.plugin_group_->onAmdgpuWorkgroupCompleted(/*dispatch_id=*/2, /*wg_id=*/0);
  EXPECT_EQ(race->retainedWorkgroupCount(), 0u);
  EXPECT_EQ(race->getSummary(), summary);
}

TEST(RaceDetectorPluginTest, SchedulerReleasesDetectorsAcrossWaveSlotReuse) {
  // Reads registers from the workgroup-dispatch callback, which runs before the
  // race plugin installs state for the new waves.
  class RegisterProbe : public ExecutionPlugin {
  public:
    RegisterProbe() : ExecutionPlugin("register_probe") {}

    void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, uint32_t, uint32_t,
                                     std::span<Wavefront *> waves) override {
      for (auto *wave : waves) {
        const RegisterAccess registers(*wave);
        static_cast<void>(registers.read_vgpr(wave->vgpr_alloc().base, /*lane=*/0));
        static_cast<void>(registers.read_sgpr(wave->sgpr_alloc().base));
        ++probed_waves;
      }
    }

    uint32_t probed_waves = 0;
  };

  class LifetimeObserver : public ExecutionPlugin {
  public:
    explicit LifetimeObserver(RaceDetectorPlugin &race)
        : ExecutionPlugin("detector_lifetime"), race_(race) {}

    void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, uint32_t, uint32_t,
                                     std::span<Wavefront *> waves) override {
      ++dispatched;
      const size_t retained = race_.retainedWorkgroupCount();
      EXPECT_EQ(retained, dispatched - completed);
      peak_retained = std::max(peak_retained, retained);
      for (auto *wave : waves)
        ++slot_uses[wave];
    }

    void onAmdgpuWorkgroupCompleted(uint32_t, uint32_t) override {
      ++completed;
      EXPECT_EQ(race_.retainedWorkgroupCount(), dispatched - completed);
    }

    uint32_t dispatched = 0;
    uint32_t completed = 0;
    size_t peak_retained = 0;
    std::map<Wavefront *, uint32_t> slot_uses;

  private:
    RaceDetectorPlugin &race_;
  };

  for (uint32_t dispatch_threads : {1u, 2u}) {
    SCOPED_TRACE(dispatch_threads);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/2, "cdna4", /*wavefront_size=*/64,
                                         /*sgprs_per_wf=*/104, /*vgprs_per_wf=*/256,
                                         /*num_cus=*/2));
    f.cp()->set_dispatch_threads(dispatch_threads);
    PluginSinkConfig sink_config;
    auto &sink = sink_config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    auto probe = std::make_unique<RegisterProbe>();
    RegisterProbe *register_probe = probe.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(probe)));
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    RaceDetectorPlugin *race = plugin.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    auto observer = std::make_unique<LifetimeObserver>(*race);
    LifetimeObserver *lifetime = observer.get();
    // Observe counts after the race plugin processes each lifecycle callback.
    ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    const auto load = cdna4::build_ds(cdna4::kDsReadB32Ds, {.addr = 0, .vdst = 8});
    const auto read = cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 264, .vdst = 9});
    // Reading v8 without a wait must still report a race after slots are reused.
    const uint32_t code[] = {S_NOP, load[0], load[1], read[0], S_ENDPGM};
    const uint64_t kernel = f.write_kernel(0x1000, code, std::size(code));
    test::AqlQueue queue(f.mem, f.cp());
    EXPECT_EQ(race->retainedWorkgroupCount(), 0u);

    std::string previous_output;
    std::string previous_summary;
    std::map<Wavefront *, uint32_t> previous_slot_uses;
    for (uint32_t dispatch = 1; dispatch <= 2; ++dispatch) {
      SCOPED_TRACE(dispatch);
      // Each workgroup fills one CU's two slots, so four groups require refills.
      queue.dispatch(kernel, /*grid_size_x=*/512, /*workgroup_size_x=*/128);
      for (uint32_t step = 0;
           step < 100000 && lifetime->completed < dispatch * 4 && f.engine->step(); ++step) {
      }

      EXPECT_EQ(lifetime->dispatched, dispatch * 4);
      ASSERT_EQ(lifetime->completed, dispatch * 4);
      EXPECT_EQ(register_probe->probed_waves, dispatch * 8);
      EXPECT_EQ(lifetime->peak_retained, 2u);
      EXPECT_EQ(race->retainedWorkgroupCount(), 0u);
      EXPECT_TRUE(f.cu(0)->is_idle());
      EXPECT_TRUE(f.cu(1)->is_idle());
      ASSERT_EQ(lifetime->slot_uses.size(), 4u);
      for (const auto &[slot, uses] : lifetime->slot_uses)
        EXPECT_GT(uses, previous_slot_uses[slot]);
      previous_slot_uses = lifetime->slot_uses;

      const std::string output = sink.str();
      EXPECT_TRUE(output.starts_with(previous_output));
      const size_t report = output.find("RACE ", previous_output.size());
      ASSERT_NE(report, std::string::npos) << output;
      EXPECT_EQ(output.find("RACE ", report + 5), std::string::npos) << output;
      previous_output = output;

      const std::string summary = race->getSummary();
      EXPECT_NE(summary.find(" race(s) detected"), std::string::npos) << summary;
      EXPECT_NE(summary, previous_summary);
      previous_summary = summary;
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna2BarrierOrdersDirectReadsFromTerminatedWaves) {
  class Observer : public ExecutionPlugin {
  public:
    Observer() : ExecutionPlugin("barrier_trace") {}
    void onAmdgpuAfterExecuteInstruction(uint64_t, const Instruction &, Wavefront &wave) override {
      if (wave.lds_source_read().bytes) {
        reader = &wave;
        reader_halted = false;
        ++reads;
      }
    }
    void onAmdgpuWavefrontHalted(Wavefront &wave) override {
      if (&wave == reader)
        reader_halted = true;
    }
    void onAmdgpuBarrierResolved(std::span<Wavefront *> waves, BarrierKind kind) override {
      EXPECT_EQ(kind, BarrierKind::Workgroup);
      EXPECT_EQ(waves.size(), 1u);
      EXPECT_TRUE(reader_halted);
      ++barriers;
    }
    unsigned reads = 0;
    unsigned barriers = 0;
    bool reader_halted = false;
    Wavefront *reader = nullptr;
  };

  for (const unsigned barrier_count : {0u, 1u, 2u}) {
    SCOPED_TRACE(barrier_count);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(2, "rdna2", 64, 106));
    PluginSinkConfig config;
    auto &sink = config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    auto observer = std::make_unique<Observer>();
    auto *trace = observer.get();
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    constexpr uint32_t barrier = 0xbf8a0000u;
    const uint32_t code[] = {
        0x7e000500,
        0xbf068000,
        0xbf840004, // Branch to the writer if the first lane's TID is nonzero.
        0xbefc03ff,
        0x00020040, // s_mov_b32 m0, 0x20040 (DWORD at LDS byte 64).
        0x7e0802fe,
        0xbf810000, // v_mov_b32 v4, src_lds_direct then s_endpgm.
        0xbf8e0007,
        0xbf8c0000, // s_sleep 7 followed by s_waitcnt 0.
        barrier_count >= 1 ? barrier : 0xbf800000u,
        barrier_count >= 2 ? barrier : 0xbf800000u,
        0x7e0002c0,
        0x7e100292, // v_mov_b32 v0, 64 and v_mov_b32 v8, 0x12.
        0xd8780000,
        0x00000800, // ds_write_b8 v0, v8.
        0xbf8c0000,
        0xbf810000,
    };
    const auto kernel = f.write_kernel(0x1000, code, std::size(code), 12, 1024);
    test::AqlQueue queue(f.mem, f.cp());
    // Each workgroup fills both slots, so the second one reuses the first group's waves.
    queue.dispatch(kernel, /*grid_size_x=*/256, /*workgroup_size_x=*/128);
    f.run_until_idle();
    ASSERT_TRUE(f.cu()->is_idle());
    EXPECT_EQ(trace->reads, 2u);
    EXPECT_EQ(trace->barriers, 2 * barrier_count);
    EXPECT_EQ(sink.str().find("Race on LDS byte 64") != std::string::npos, barrier_count == 0)
        << sink.str();
  }
}

TEST(RaceDetectorPluginTest, Rdna2GwsProtectsTheConsumedVgprAndExecUntilExpcnt) {
  for (bool exec_write : {false, true})
    for (bool waited : {false, true}) {
      SCOPED_TRACE(waited);
      SCOPED_TRACE(exec_write);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna2", 64, 106));
      PluginSinkConfig config;
      auto &sink = config.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      const uint32_t code[] = {
          0xbefc0380, // s_mov_b32 m0, 0
          0x7e100281, // v_mov_b32_e32 v8, 1
          0xd8660000,
          8,                                      // ds_gws_init v8 gds
          waited ? 0xbf8cff0fu : 0xbf800000u,     // s_waitcnt expcnt(0) or nop
          exec_write ? 0xbf800000u : 0x7e100282u, // nop or v_mov_b32 v8, 2
          exec_write ? 0xbefe0381u : 0xbf800000u, // s_mov_b32 exec_lo, 1 or nop
          0xbf810000,
      };
      const auto kernel = f.write_kernel(0x1000, code, std::size(code));
      test::AqlQueue queue(f.mem, f.cp());
      queue.dispatch(kernel, 64, 64);
      f.run_until_idle();
      ASSERT_TRUE(f.cu()->is_idle());
      const auto output = sink.str();
      EXPECT_EQ(f.cu()->memory_wait_diagnostic_count(), waited ? 0 : 1);
      EXPECT_EQ(output.find("Race on VGPR v8 [") != std::string::npos, !waited && !exec_write);
      EXPECT_EQ(output.find("Race on EXEC") != std::string::npos, !waited && exec_write);
    }
}

TEST(RaceDetectorPluginTest, Rdna4InterpolationRetiresItsEmbeddedWaitBeforeReading) {
  struct Case {
    uint32_t interpolation;
    uint32_t prior_wait;
    bool expected_race;
  };
  for (auto c : {Case{0xcd000009, 0xbf800000, false}, Case{0xcd000709, 0xbf800000, true},
                 Case{0xcd000709, 0xbfc40000, false}}) {
    SCOPED_TRACE(c.interpolation);
    SCOPED_TRACE(c.prior_wait);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna4", 32, 104));
    PluginSinkConfig config;
    auto &sink = config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    const uint32_t code[] = {0xce100006, c.prior_wait, c.interpolation, 0x040e0506, 0xbfb00000};
    constexpr uint64_t pc = 0x1000;
    f.mem->load_image(reinterpret_cast<const uint8_t *>(code), sizeof(code), pc);
    auto *cu = f.cu();
    auto *wave = cu->dispatch_wf(0, pc, 104, 32, 32);
    ASSERT_NE(wave, nullptr);
    std::array<Wavefront *, 1> waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 104, waves);
    wave->set_exec(1);
    wave->set_lds_base(4096);
    wave->set_lds_size(1024);
    wave->set_m0((2u << 16) | 16u);
    wave->lds().write32(4096 + 16, 0x3f800000);
    cu->step();
    ASSERT_EQ(wave->pc, pc + 4);
    EXPECT_EQ(wave->debug_read_vgpr(6, 0), 0x3f800000u);
    cu->step();
    ASSERT_EQ(wave->pc, pc + 8);
    cu->step();
    ASSERT_EQ(wave->pc, pc + 16);
    const bool observed = sink.str().find("Race on VGPR v6 [") != std::string::npos;
    EXPECT_EQ(cu->memory_wait_diagnostic_count(), static_cast<unsigned>(c.expected_race));
    EXPECT_EQ(observed, c.expected_race);
  }
}

TEST_F(TensorDmaRaceTest, OnlyWholeWorkgroupBarriersPublishAnAbsentWritersCompletedBytes) {
  EXPECT_FALSE(ds_write(0, 64));
  wait_dscnt(0);
  const auto reader = std::span(waves).subspan(1);
  fixture.plugin_group_->onAmdgpuBarrierResolved(reader, BarrierKind::Named);
  EXPECT_TRUE(ds_read(1, 64));

  EXPECT_FALSE(ds_write(0, 128));
  wait_dscnt(0);
  fixture.plugin_group_->onAmdgpuBarrierResolved(reader, BarrierKind::Workgroup);
  EXPECT_FALSE(ds_read(1, 128));
}

class RoutedRaceControl {
public:
  void initialize(const char *target, rj_code_arch_t arch) {
    ASSERT_NO_FATAL_FAILURE(fixture.initialize(2, target, 32, 106));
    PluginSinkConfig config;
    sink = &config.emplace<StringSink>();
    fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    ASSERT_TRUE(fixture.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    fixture.soc->set_plugin_group(fixture.plugin_group_);
    fixture.plugin_group_->onInit();
    for (size_t i = 0; i < waves.size(); ++i) {
      waves[i] = fixture.cu()->dispatch_wf(0, 0x1000 + i * 0x100, 106, 32, 32);
      ASSERT_NE(waves[i], nullptr);
      waves[i]->set_lds_size(1024);
      waves[i]->set_exec(1);
    }
    fixture.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 64, 212, waves);
    decoder = Decoder::create(arch);
  }

  void set(unsigned wave, unsigned reg, uint32_t value) {
    fixture.cu()->write_vgpr(waves[wave]->vgpr_alloc().base + reg, 0, value);
  }

  void issue(unsigned index, std::span<const uint32_t> words,
             VmAccessOutcome expected = VmAccessOutcome::Complete) {
    auto &wave = *waves[index];
    auto decoded = decoder->decode_window(words);
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave.pc, *inst, wave);
    ASSERT_TRUE(fixture.cu()->execute_instruction(inst.get(), wave).succeeded());
    fixture.plugin_group_->onAmdgpuAfterExecuteInstruction(wave.pc, *inst, wave);
    const auto bytes = inst->size();
    if (inst->is_memory_op() && inst->data()) {
      ASSERT_EQ(test::ComputeUnitTestAccess::route_memory_inst(*fixture.cu(), inst.release(), wave),
                expected);
    }
    wave.pc += bytes;
  }

  PluginFixture fixture;
  StringSink *sink = nullptr;
  std::array<Wavefront *, 2> waves{};
  std::unique_ptr<Decoder> decoder;
};

TEST(RaceDetectorPluginTest, UnorderedGlobalWritebackRequiresWaitOnTheSplitLoadCounter) {
  for (bool cdna : {false, true}) {
    for (bool waited : {false, true}) {
      SCOPED_TRACE(testing::Message() << cdna << ' ' << waited);
      RoutedRaceControl r;
      ASSERT_NO_FATAL_FAILURE(r.initialize(
          cdna ? "cdna5" : "rdna4", cdna ? ROCJITSU_CODE_ARCH_CDNA5 : ROCJITSU_CODE_ARCH_RDNA4));
      r.fixture.mem->write32(0x9000, 11);
      r.set(0, 0, 0x9000);
      r.set(0, 1, 0);
      const auto flat = cdna ? cdna5::build_vflat(cdna5::kFlatLoadB32Vflat,
                                                  {.saddr = kModernNullSelector, .vdst = 8})
                             : rdna4::build_vflat(rdna4::kFlatLoadB32Vflat,
                                                  {.saddr = kModernNullSelector, .vdst = 8});
      ASSERT_NO_FATAL_FAILURE(r.issue(0, flat));
      if (waited) {
        ASSERT_NO_FATAL_FAILURE(r.issue(0, std::array{0xbfc00000u})); // S_WAIT_LOADCNT 0
      }
      ASSERT_NO_FATAL_FAILURE(r.issue(0, flat));
      EXPECT_EQ(r.sink->str().find("Race on VGPR v8 [") != std::string::npos, !waited);
    }
  }
}

TEST(RaceDetectorPluginTest, FaultingNativeAtomicDoesNotClaimPeerLdsBytes) {
  for (auto [name, arch] :
       {std::pair{"rdna3", ROCJITSU_CODE_ARCH_RDNA3}, std::pair{"rdna4", ROCJITSU_CODE_ARCH_RDNA4},
        std::pair{"cdna5", ROCJITSU_CODE_ARCH_CDNA5}}) {
    for (bool aligned : {false, true}) {
      SCOPED_TRACE(testing::Message() << name << ' ' << aligned);
      RoutedRaceControl r;
      ASSERT_NO_FATAL_FAILURE(r.initialize(name, arch));
      r.set(0, 0, 64);
      r.set(0, 1, 11);
      r.set(1, 0, aligned ? 64 : 65);
      r.set(1, 1, 11);
      ASSERT_NO_FATAL_FAILURE(r.issue(0, std::array{0xd8340000u, 0x00000100u})); // DS_STORE_B32
      ASSERT_NO_FATAL_FAILURE(
          r.issue(1, std::array{0xd8800000u, 0x08000100u}, // DS_ADD_RTN_U32
                  aligned ? VmAccessOutcome::Complete : VmAccessOutcome::Faulted));
      EXPECT_EQ(r.waves[0]->lds().read32(64), aligned ? 22u : 11u);
      EXPECT_EQ(r.sink->str().find("Race on LDS") != std::string::npos, aligned);
    }
  }
}

TEST(RaceDetectorPluginTest, SharedFlatStorePublishesAfterItsLdsCounterRetires) {
  for (bool split : {false, true}) {
    for (bool waited : {false, true}) {
      RoutedRaceControl r;
      ASSERT_NO_FATAL_FAILURE(r.initialize(
          split ? "rdna4" : "rdna3", split ? ROCJITSU_CODE_ARCH_RDNA4 : ROCJITSU_CODE_ARCH_RDNA3));
      constexpr uint64_t shared = uint64_t{16} << 32;
      r.waves[0]->set_apertures(shared, shared + 0xffff, 0, 0, /*override_launch=*/true);
      r.set(0, 0, 64);
      r.set(0, 1, shared >> 32);
      r.set(0, 2, 11);
      if (split) {
        ASSERT_NO_FATAL_FAILURE(
            r.issue(0, rdna4::build_vflat(rdna4::kFlatStoreB32Vflat,
                                          {.saddr = kModernNullSelector, .vsrc = 2, .vaddr = 0})));
      } else {
        ASSERT_NO_FATAL_FAILURE(
            r.issue(0, rdna3::build_flat(rdna3::kFlatStoreB32Flat,
                                         {.addr = 0, .data = 2, .saddr = kModernNullSelector})));
      }
      if (waited) {
        ASSERT_NO_FATAL_FAILURE(r.issue(0, std::array{split ? 0xbfc60000u : 0xbf89fc07u}));
      }
      r.fixture.plugin_group_->onAmdgpuBarrierResolved(r.waves, BarrierKind::Workgroup);
      r.set(1, 4, 64);
      ASSERT_NO_FATAL_FAILURE(r.issue(1, std::array{0xd8d80000u, 0x08000004u})); // DS_LOAD_B32
      EXPECT_EQ(r.waves[1]->debug_read_vgpr(8, 0), 11u);
      EXPECT_EQ(r.sink->str().find("Race on LDS") != std::string::npos, !waited);
    }
  }
}

TEST(RaceDetectorPluginTest, EndedPeerPublishesCompletedLdsOnlyAfterWorkgroupBarrier) {
  struct Code {
    const char *arch;
    uint32_t branch, exec, store, load, wait, sleep, barrier, end;
  };
  const Code gfx11{"rdna3",    0xbfa10007, 0xbefe0181, 0xd8780000, 0xd8e80000,
                   0xbf89fc07, 0xbf830007, 0xbfbd0000, 0xbfb00000};
  auto gfx115 = gfx11;
  gfx115.arch = "rdna3.5";
  const Code gfx9{"cdna3",    0xbf840007, 0xbefe0181, 0xd83c0000, 0xd8740000,
                  0xbf8cc07f, 0xbf8e0007, 0xbf8a0000, 0xbf810000};
  for (const auto &code : {gfx11, gfx115, gfx9}) {
    for (const bool barrier : {false, true}) {
      for (const bool writer_wait : {false, true}) {
        SCOPED_TRACE(testing::Message() << code.arch << ' ' << barrier << ' ' << writer_wait);
        PluginFixture f;
        ASSERT_NO_FATAL_FAILURE(f.initialize(2, code.arch, 64, 106));
        PluginSinkConfig config;
        auto &sink = config.emplace<StringSink>();
        f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
        ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
        f.soc->set_plugin_group(f.plugin_group_);
        f.plugin_group_->onInit();
        // The writer ends after storing one byte. The surviving reader optionally
        // synchronizes the workgroup before consuming that same LDS byte.
        const uint32_t words[]{0x7e000500, 0xbf068000, code.branch,
                               code.exec,  0x7e0002c0, 0x7e100292,
                               code.store, 0x00000800, writer_wait ? code.wait : 0xbf800000u,
                               code.end,   code.sleep, barrier ? code.barrier : 0xbf800000u,
                               0x7e0002c0, code.load,  0x04000000,
                               code.wait,  code.end};
        const auto kernel = f.write_kernel(0x1000, words, std::size(words), 12, 1024);
        test::AqlQueue queue(f.mem, f.cp());
        queue.dispatch(kernel, 128, 128);
        f.run_until_idle();
        ASSERT_TRUE(f.cu()->is_idle());
        EXPECT_EQ(sink.str().find("Race on LDS byte 64") != std::string::npos, !barrier);
      }
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna2EndpgmCompletesDsWritesBeforeWorkgroupBarrier) {
  class Observer : public ExecutionPlugin {
  public:
    Observer() : ExecutionPlugin("endpgm_trace") {}
    void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &inst,
                                          Wavefront &wave) override {
      if (inst.mnemonic() == "ds_write_b8") {
        writer = &wave;
        writer_halted = false;
      }
    }
    void onAmdgpuAfterExecuteInstruction(uint64_t, const Instruction &, Wavefront &wave) override {
      if (wave.lds_source_read().bytes) {
        ++reads;
        EXPECT_TRUE(writer_halted);
        EXPECT_EQ(cu->read_vgpr(wave.vgpr_alloc().base + 4, 0) & 0xffu, 0x12u);
      }
    }
    void onAmdgpuWavefrontHalted(Wavefront &wave) override {
      if (&wave == writer)
        writer_halted = true;
    }
    void onAmdgpuBarrierResolved(std::span<Wavefront *> waves, BarrierKind kind) override {
      EXPECT_EQ(kind, BarrierKind::Workgroup);
      EXPECT_EQ(waves.size(), 1u);
      EXPECT_TRUE(writer_halted);
      ++barriers;
    }
    Wavefront *writer = nullptr;
    ComputeUnitCore *cu = nullptr;
    bool writer_halted = false;
    unsigned reads = 0;
    unsigned barriers = 0;
  };
  // S_ENDPGM, S_ENDPGM_SAVED, S_ENDPGM_ORDERED_PS_DONE.
  for (const uint32_t endpgm : {0xbf810000u, 0xbf9b0000u, 0xbf9e0000u}) {
    for (const bool barrier : {true, false}) {
      for (const bool writer_wait : {true, false}) {
        SCOPED_TRACE(::testing::Message() << "endpgm=" << std::hex << endpgm << " barrier="
                                          << barrier << " explicit_writer_wait=" << writer_wait);
        PluginFixture f;
        ASSERT_NO_FATAL_FAILURE(f.initialize(2, "rdna2", 64, 106));
        PluginSinkConfig config;
        auto &sink = config.emplace<StringSink>();
        f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
        ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
        auto observer = std::make_unique<Observer>();
        auto *trace = observer.get();
        trace->cu = f.cu();
        f.cu()->set_sendmsg_handler([](Wavefront &, uint32_t msg) { return msg == 7; });
        ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
        f.soc->set_plugin_group(f.plugin_group_);
        f.plugin_group_->onInit();

        // Wave 0 writes byte 64 with only lane 0 active, then ends. Wave 1
        // optionally synchronizes and consumes that DWORD through src_lds_direct.
        const uint32_t code[] = {
            0x7e000500, 0xbf068000, 0xbf840007,
            0xbefe0481, 0x7e0002c0, 0x7e100292,
            0xd8780000, 0x00000800, writer_wait ? 0xbf8c0000u : 0xbf800000u,
            endpgm,     0xbf8e0007, barrier ? 0xbf8a0000u : 0xbf800000u,
            0xbefc03ff, 0x00020040, 0x7e0802fe,
            0xbf810000,
        };
        const auto kernel = f.write_kernel(0x1000, code, std::size(code), 12, 1024);
        test::AqlQueue queue(f.mem, f.cp());
        // Two workgroups reuse the same pair of physical wave slots.
        queue.dispatch(kernel, 256, 128);
        f.run_until_idle();
        ASSERT_TRUE(f.cu()->is_idle());
        EXPECT_TRUE(trace->writer_halted);
        EXPECT_EQ(trace->reads, 2u);
        EXPECT_EQ(trace->barriers, 2u * unsigned{barrier});
        EXPECT_EQ(sink.str().find("Race on LDS byte 64") != std::string::npos, !barrier)
            << sink.str();
      }
    }
  }
}

TEST(RaceDetectorPluginTest, OnlySuccessfulRdna2EndpgmCompletesTheLegacyWaitCounters) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(3, "rdna2", 32, 106));
  PluginSinkConfig config;
  auto plugin = std::make_unique<RaceDetectorPlugin>();
  auto *race = plugin.get();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
  ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();
  auto *cu = f.cu();
  std::array waves{cu->dispatch_wf(0, 0x100, 106, 256, 32), cu->dispatch_wf(0, 0x200, 106, 256, 32),
                   cu->dispatch_wf(0, 0x300, 106, 256, 32)};
  ASSERT_NE(waves[0], nullptr);
  ASSERT_NE(waves[1], nullptr);
  ASSERT_NE(waves[2], nullptr);
  f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 768, 318, waves);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA2);
  using Counter = amdgpu::WaitCounterType;
  constexpr std::array counters{Counter::VMCNT, Counter::LGKMCNT, Counter::EXPCNT, Counter::VSCNT};
  for (int i : {0, 1, 2}) {
    auto &wave = *waves[i];
    auto *state = static_cast<RaceWavefrontState *>(wave.plugin_state(race->slot_index()));
    auto *logical = state->race_state;
    const std::array<uint32_t, 1> nop{0xbf800000};
    for (auto counter : counters)
      logical->registerEvent(InstructionSnapshot{wave.pc, nop}, MemoryEventType::GLOBAL_TO_VGPR,
                             {0}, 1, 0xf, CounterObligations(counter, MemoryOrderClass::UNORDERED),
                             MemoryOrderClass::UNORDERED);
    // One wave is cancelled after the before hook for ENDPGM. The second
    // is forcibly halted after SETKILL, another PROGRAM_TERMINATOR. The
    // third actually executes ENDPGM and must complete all four counters.
    const uint32_t word = i == 1 ? 0xbf8b0000u : 0xbf810000u;
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &word));
    ASSERT_NE(inst, nullptr);
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave.pc + 4, *inst, wave);
    if (i == 0) {
      f.plugin_group_->onAmdgpuWavefrontCancelled(wave);
    } else if (i == 1) {
      f.plugin_group_->onAmdgpuWavefrontHalted(wave);
    } else {
      ASSERT_TRUE(cu->execute_instruction(inst.get(), wave).succeeded());
    }
    EXPECT_FALSE(wave.has_plugin_state(race->slot_index()));
    EXPECT_EQ(logical->getWaveMemoryEvents().size(), i == 2 ? 0u : counters.size());
    for (int n = 0; n < static_cast<int>(counters.size()); ++n) {
      const EventId event{static_cast<int>(counters.size()) * i + n};
      EXPECT_EQ(logical->getDetector()->events().status(event),
                i == 2 ? EventStatus::WAVE_COMPLETE : EventStatus::ACTIVE);
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna2BufferLoadsWithEmptyRequestsPreserveVmcntOrder) {
  enum class Second { Valid, TfeOutsideAllocation, EmptyExec, DescriptorBounds, Nop };
  for (auto second : {Second::Valid, Second::TfeOutsideAllocation, Second::EmptyExec,
                      Second::DescriptorBounds, Second::Nop}) {
    for (uint32_t width : {32u, 64u}) {
      SCOPED_TRACE(testing::Message() << "wave=" << width << " second=" << int(second));
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna2", width, 106, 256));
      PluginSinkConfig config;
      auto &sink = config.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();

      const auto first = rdna2::build_mubuf(rdna2::kBufferLoadDword, {.vdata = 0, .soffset = 128});
      const auto next = rdna2::build_mubuf(
          rdna2::kBufferLoadDword,
          {.offset = static_cast<uint16_t>(second == Second::DescriptorBounds ? 256 : 0),
           .vdata = 15,
           .tfe = second == Second::TfeOutsideAllocation,
           .soffset = 128});
      const uint32_t nop = 0xbf800000;
      const uint32_t code[] = {0xbe8003ff,
                               0x4000,
                               0xbe810380,
                               0xbe8203ff,
                               0x100,
                               0xbe8303ff,
                               0x3103c000,
                               first[0],
                               first[1],
                               second == Second::EmptyExec ? 0xbefe0480u : nop, // s_mov_b64 exec,0
                               second == Second::Nop ? nop : next[0],
                               second == Second::Nop ? nop : next[1],
                               second == Second::EmptyExec ? 0xbefe04c1u : nop, // s_mov_b64 exec,-1
                               0xbf8c3f71, // s_waitcnt vmcnt(1)
                               0x7e040300,
                               0x7e06030f, // read v0 and v15 at distinct PCs
                               0xbf810000};
      const auto kernel = f.write_kernel(0x1000, code, std::size(code));
      using namespace amdhsa;
      auto count = f.mem->read32(kernel + offsetof(kernel_descriptor_t, compute_pgm_rsrc1));
      RJ_AMDHSA_BITS_SET(count, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                         width == 32 ? 1 : 3);
      f.mem->write32(kernel + offsetof(kernel_descriptor_t, compute_pgm_rsrc1), count);
      auto properties =
          f.mem->read32(kernel + offsetof(kernel_descriptor_t, kernel_code_properties));
      RJ_AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, width == 32);
      f.mem->write16(kernel + offsetof(kernel_descriptor_t, kernel_code_properties), properties);
      f.mem->write32(0x4000, 0x76543210);
      test::AqlQueue queue(f.mem, f.cp());
      queue.dispatch(kernel, width, width);
      f.run_until_idle();
      ASSERT_TRUE(f.cu()->is_idle());
      const bool older_pending = second == Second::Nop;
      // A descriptor-OOB load writes zero to v15. Only the null wave requests
      // and the omitted instruction leave its bytes available immediately.
      const bool younger_pending = second == Second::Valid || second == Second::DescriptorBounds;
      EXPECT_EQ(sink.str().find("Race on VGPR v0 [") != std::string::npos, older_pending);
      EXPECT_EQ(sink.str().find("Race on VGPR v15 [") != std::string::npos, younger_pending);
      EXPECT_EQ(f.cu()->memory_wait_diagnostic_count(),
                unsigned(older_pending) + unsigned(younger_pending));
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna2SingleCounterWaitReadsItsThresholdBeforeRetiring) {
  constexpr uint32_t nop = 0xbf800000;
  constexpr uint32_t wait_lgkm = 0xbf8cc07f;      // s_waitcnt lgkmcnt(0)
  constexpr uint32_t wait_lgkm_null = 0xbd7d0000; // s_waitcnt_lgkmcnt null,0
  struct Control {
    uint8_t selector;
    uint32_t prior_wait;
    bool read_after;
    bool empty_exec;
  };
  for (const auto op : {rdna2::kSWaitcntVmcnt, rdna2::kSWaitcntVscnt, rdna2::kSWaitcntExpcnt,
                        rdna2::kSWaitcntLgkmcnt}) {
    for (auto c : {Control{8, nop, false, false}, Control{8, nop, false, true},
                   Control{8, wait_lgkm, false, false}, Control{8, wait_lgkm_null, false, false},
                   Control{9, nop, false, false}, Control{125, nop, true, false}}) {
      SCOPED_TRACE(testing::Message()
                   << "opcode=" << op << " selector=" << unsigned(c.selector)
                   << " prior=" << std::hex << c.prior_wait << " EXEC=0 " << c.empty_exec);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna2", 64, 106));
      PluginSinkConfig config;
      auto &sink = config.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      // The loaded threshold wraps to zero when added to one in any counter field.
      const auto wait = rdna2::build_sopk(
          op, {.simm16 = static_cast<uint16_t>(c.selector != 125), .sdst = c.selector});
      const uint32_t code[] = {0xbe8003ff,
                               0x4000,
                               0xbe810380,
                               0xbe8903bf,                       // s[0:1]=address, s9=63
                               c.empty_exec ? 0xbefe0480u : nop, // s_mov_b64 exec,0
                               0xf4000200,
                               0xfa000000, // s_load_dword s8,s[0:1],0
                               c.prior_wait,
                               wait[0],
                               c.read_after ? 0xbe8b0308u : nop, // s_mov_b32 s11,s8
                               0xbf810000};
      f.mem->write32(0x4000, 63);
      auto kernel = f.write_kernel(0x1000, code, std::size(code));
      test::AqlQueue queue(f.mem, f.cp());
      queue.dispatch(kernel, 64, 64);
      f.run_until_idle();
      ASSERT_TRUE(f.cu()->is_idle());
      const bool threshold_hazard = c.selector == 8 && c.prior_wait == nop;
      const bool subsequent_hazard = c.read_after && op != rdna2::kSWaitcntLgkmcnt;
      const bool hazard = threshold_hazard || subsequent_hazard;
      EXPECT_EQ(f.cu()->memory_wait_diagnostic_count(), unsigned(hazard));
      EXPECT_EQ(sink.str().find("Race on SGPR s8 [") != std::string::npos, hazard) << sink.str();
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna2EmptyReturningDsAtomicHasAnLgkmcntPosition) {
  for (const bool gds : {false, true}) {
    for (const bool issue_atomic : {true, false}) {
      SCOPED_TRACE(testing::Message() << "gds=" << gds << " returning_atomic=" << issue_atomic);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna2", 32, 106, 256));
      PluginSinkConfig config;
      auto &sink = config.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      const auto first = rdna2::build_ds(rdna2::kDsReadB32, {.gds = gds, .addr = 8, .vdst = 0});
      const auto second =
          rdna2::build_ds(rdna2::kDsAddRtnU64, {.gds = gds, .addr = 8, .data0 = 4, .vdst = 15});
      const uint32_t nop = 0xbf800000;
      const uint32_t code[] = {
          0xbefc03ff,
          0x00800000, // s_mov_b32 m0, (128 << 16): GDS window
          first[0],    first[1], issue_atomic ? second[0] : nop, issue_atomic ? second[1] : nop,
          0xbf8cc17fu, // lgkmcnt(1) retires the older LDS/GDS result.
          0x7e040300,
          0x7e06030f, // consume the older v0 and the suppressed v15
          0xbf810000};
      const auto kernel = f.write_kernel(0x1000, code, std::size(code), 12, 1024);
      using namespace amdhsa;
      auto count = f.mem->read32(kernel + offsetof(kernel_descriptor_t, compute_pgm_rsrc1));
      RJ_AMDHSA_BITS_SET(count, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 1);
      f.mem->write32(kernel + offsetof(kernel_descriptor_t, compute_pgm_rsrc1), count);
      auto properties =
          f.mem->read32(kernel + offsetof(kernel_descriptor_t, kernel_code_properties));
      RJ_AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 1);
      f.mem->write16(kernel + offsetof(kernel_descriptor_t, kernel_code_properties), properties);
      test::AqlQueue queue(f.mem, f.cp());
      queue.dispatch(kernel, 32, 32);
      f.run_until_idle();
      ASSERT_TRUE(f.cu()->is_idle());
      EXPECT_EQ(sink.str().find("Race on VGPR v0 [") != std::string::npos, !issue_atomic);
      EXPECT_EQ(sink.str().find("Race on VGPR v15 ["), std::string::npos);
      EXPECT_EQ(f.cu()->memory_wait_diagnostic_count(), unsigned(!issue_atomic));
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna2FlatIssuesBothCountersAcrossRoutedDomains) {
  static constexpr uint64_t shared = uint64_t{2} << 32;
  static constexpr uint32_t address = 0x4000, global_value = 0x76543210, lds_value = 0x12345678;
  class ObserveWave : public ExecutionPlugin {
  public:
    ObserveWave(ComputeUnitCore &cu) : ExecutionPlugin("flat_issue_wave"), cu(cu) {}
    void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, uint32_t, uint32_t,
                                     std::span<Wavefront *> waves) override {
      cu.set_apertures(shared, shared + UINT32_MAX, 0, 0);
      for (auto *wf : waves) {
        wf->set_apertures(shared, shared + UINT32_MAX, 0, 0, /*override_launch=*/true);
        wf->lds().write32(wf->lds_base(), lds_value);
      }
    }
    void onAmdgpuWavefrontHalted(Wavefront &wf) override {
      EXPECT_EQ(cu.read_vgpr_storage(wf.vgpr_alloc().base + 6, 0), global_value);
      EXPECT_EQ(cu.read_vgpr_storage(wf.vgpr_alloc().base + 7, 0), lds_value);
    }
    ComputeUnitCore &cu;
  };
  enum class Form { Global, Local, Mixed, ZeroExec, InvalidReturn, Nop };
  for (uint32_t width : {32u, 64u}) {
    for (bool store : {false, true}) {
      for (Form form : {Form::Global, Form::Local, Form::Mixed, Form::ZeroExec, Form::InvalidReturn,
                        Form::Nop}) {
        if (store && form == Form::InvalidReturn)
          continue;
        SCOPED_TRACE(testing::Message() << "width=" << width << " store=" << store
                                        << " form=" << static_cast<int>(form));
        PluginFixture f;
        ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna2", width, 106, 256));
        f.cu()->set_apertures(shared, shared + UINT32_MAX, 0, 0);
        PluginSinkConfig config;
        auto &sink = config.emplace<StringSink>();
        f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
        ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
        auto memory = std::make_unique<MemoryObservationPlugin>();
        auto *routes = memory.get();
        ASSERT_TRUE(f.plugin_group_->add(std::move(memory)));
        ASSERT_TRUE(f.plugin_group_->add(std::make_unique<ObserveWave>(*f.cu())));
        f.soc->set_plugin_group(f.plugin_group_);
        f.plugin_group_->onInit();
        const auto vm = rdna2::build_mubuf(rdna2::kBufferLoadDword, {.vdata = 2, .soffset = 128});
        const auto ds = rdna2::build_ds(rdna2::kDsReadB32, {.addr = 12, .vdst = 3});
        const auto flat =
            rdna2::build_flat(store ? rdna2::kFlatStoreDword : rdna2::kFlatLoadDwordx2,
                              {.addr = 8,
                               .data = static_cast<uint8_t>(store ? 10 : 0),
                               .saddr = 125,
                               .vdst = static_cast<uint8_t>(store                         ? 0
                                                            : form == Form::InvalidReturn ? 15
                                                                                          : 4)});
        constexpr uint32_t nop = 0xbf800000;
        const uint32_t code[] = {
            0xbe8003ff, address, 0xbe810380, 0xbe8203ff, 0x100, 0xbe8303ff, 0x3103c000, 0x7e1002ff,
            address, // both halves of the FLAT address share the low word
            form == Form::Mixed   ? 0x36120081u
            : form == Form::Local ? 0x7e120282u
                                  : 0x7e120280u,
            form == Form::Mixed ? 0x34121281u : nop, // alternating high DWORDs of 0 and 2
            0x7e180280, // v12=0, the address for the older indexed LDS read
            vm[0], vm[1], ds[0], ds[1], form == Form::ZeroExec ? 0xbefe0480u : nop,
            form == Form::Nop ? nop : flat[0], form == Form::Nop ? nop : flat[1],
            form == Form::ZeroExec ? 0xbefe04c1u : nop,
            0xbf8c0171,             // s_waitcnt vmcnt(1) lgkmcnt(1)
            0x7e0c0302, 0x7e0e0303, // consume the older v2 and v3
            form == Form::Global  ? 0xbf8c3f70u
            : form == Form::Local ? 0xbf8cc07fu
                                  : 0xbf8c0000u,
            !store && (form == Form::Global || form == Form::Local) ? 0x7e160304u : nop,
            // A uniform GFX10 FLAT load still requires both acknowledgments.
            0xbf8c0000, 0xbf810000};
        const auto kernel = f.write_kernel(0x1000, code, std::size(code), 12, address + 8);
        using namespace amdhsa;
        auto n = f.mem->read32(kernel + offsetof(kernel_descriptor_t, compute_pgm_rsrc1));
        RJ_AMDHSA_BITS_SET(n, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                           width == 32 ? 1 : 3);
        f.mem->write32(kernel + offsetof(kernel_descriptor_t, compute_pgm_rsrc1), n);
        auto props = f.mem->read32(kernel + offsetof(kernel_descriptor_t, kernel_code_properties));
        RJ_AMDHSA_BITS_SET(props, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, width == 32);
        f.mem->write16(kernel + offsetof(kernel_descriptor_t, kernel_code_properties), props);
        f.mem->write32(address, global_value);
        test::AqlQueue queue(f.mem, f.cp());
        queue.dispatch(kernel, width, width);
        f.run_until_idle();
        ASSERT_TRUE(f.cu()->is_idle());
        const bool vm_pending = store || form == Form::Nop;
        const bool ds_pending = form == Form::Nop;
        EXPECT_EQ(sink.str().find("Race on VGPR v2 [") != std::string::npos, vm_pending);
        EXPECT_EQ(sink.str().find("Race on VGPR v3 [") != std::string::npos, ds_pending);
        const bool flat_pending = !store && (form == Form::Global || form == Form::Local);
        EXPECT_EQ(sink.str().find("Race on VGPR v4 [") != std::string::npos, flat_pending);
        EXPECT_EQ(f.cu()->memory_wait_diagnostic_count(),
                  unsigned(vm_pending) + unsigned(ds_pending) + unsigned(flat_pending));
        ASSERT_EQ(routes->accesses.size(), form == Form::Nop ? 2u : 3u);
        if (form == Form::Nop)
          continue;
        const auto &access = routes->accesses.back();
        const uint64_t full = width == 32 ? UINT32_MAX : UINT64_MAX;
        const bool empty = form == Form::ZeroExec || form == Form::InvalidReturn;
        EXPECT_EQ(access.active_lane_mask, empty ? 0u : full);
        EXPECT_EQ(access.request_lane_mask, empty ? 0u : full);
        const uint64_t local = form == Form::Local   ? full
                               : form == Form::Mixed ? full & 0xaaaaaaaaaaaaaaaaULL
                                                     : 0;
        EXPECT_EQ(access.flat_local_lane_mask, local);
        using Counter = amdgpu::WaitCounterType;
        EXPECT_EQ(access.issued_counters.size(), 2u);
        EXPECT_EQ(
            std::ranges::count(access.issued_counters, store ? Counter::VSCNT : Counter::VMCNT), 1);
        EXPECT_EQ(std::ranges::count(access.issued_counters, Counter::LGKMCNT), 1);
      }
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna2CounterOnlyIssuesKeepBoundedHistory) {
  enum class Loop { FlatGlobal, FlatEmpty, BufferNull, GdsEmpty };
  constexpr uint32_t count = 1100, address = 0x4000, value = 0x76543210;
  class ObserveLoop : public ExecutionPlugin {
  public:
    explicit ObserveLoop(const RaceDetectorPlugin &race)
        : ExecutionPlugin("counter_only_loop"), race(race) {}
    void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &inst,
                                          Wavefront &wf) override {
      if (inst.mnemonic() != "s_nop")
        return;
      auto *wave =
          static_cast<RaceWavefrontState *>(wf.plugin_state(race.slot_index()))->race_state;
      EXPECT_TRUE(wave->getWaveMemoryEvents().empty());
      EXPECT_LE(wave->getDetector()->events().size(), EventRegistry::kTrimAttemptInterval);
      saw_marker = true;
    }
    void onAmdgpuWavefrontHalted(Wavefront &wf) override { result = wf.debug_read_vgpr(5, 0); }
    const RaceDetectorPlugin &race;
    bool saw_marker = false;
    uint32_t result = 0;
  };
  for (const auto form : {Loop::FlatGlobal, Loop::FlatEmpty, Loop::BufferNull, Loop::GdsEmpty}) {
    SCOPED_TRACE(static_cast<int>(form));
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna2", 64, 106, 256));
    PluginSinkConfig config;
    auto &sink = config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    auto race = std::make_unique<RaceDetectorPlugin>();
    auto observer = std::make_unique<ObserveLoop>(*race);
    auto *stats = observer.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(race)));
    ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    const bool empty = form == Loop::FlatEmpty || form == Loop::GdsEmpty;
    auto load = rdna2::build_flat(rdna2::kFlatLoadDword, {.addr = 8, .saddr = 125, .vdst = 4});
    if (form == Loop::BufferNull)
      load = rdna2::build_mubuf(rdna2::kBufferLoadDwordx2, {.vdata = 15, .soffset = 128});
    if (form == Loop::GdsEmpty)
      load = rdna2::build_ds(rdna2::kDsReadB32, {.gds = 1, .addr = 8, .vdst = 4});
    const uint32_t code[] = {0x7e1002ff,
                             address,
                             0x7e120280,
                             0x7e080280,
                             0x7e0a0280,
                             0xbe8803ff,
                             count,
                             empty ? 0xbefe0480u : 0xbefe0381u,
                             load[0],
                             load[1],
                             form == Loop::FlatGlobal ? 0xbf8c0070u : 0xbe870307u,
                             0x7e0a0304, // consume v4 only on active lanes
                             0x80888108,
                             0xbf078008,
                             0xbf85fff9, // decrement, compare, branch to the load
                             0xbf800003,
                             0xbf810000};
    const auto kernel = f.write_kernel(0x1000, code, std::size(code));
    using namespace amdhsa;
    auto rsrc = f.mem->read32(kernel + offsetof(kernel_descriptor_t, compute_pgm_rsrc1));
    RJ_AMDHSA_BITS_SET(rsrc, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 3);
    f.mem->write32(kernel + offsetof(kernel_descriptor_t, compute_pgm_rsrc1), rsrc);
    f.mem->write32(address, value);
    test::AqlQueue queue(f.mem, f.cp());
    queue.dispatch(kernel, 64, 64);
    f.run_until_idle();
    ASSERT_TRUE(f.cu()->is_idle());
    EXPECT_TRUE(stats->saw_marker);
    EXPECT_EQ(stats->result, form == Loop::FlatGlobal ? value : 0u);
    EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
    EXPECT_EQ(f.cu()->memory_wait_diagnostic_count(), 0u);
  }
}

TEST(RaceDetectorPluginTest, Rdna4ResinfoSharesLoadcntButCanOverwriteAnOutstandingVgpr) {
  enum class Wait { None, DsZero, LoadOne, LoadZero };
  for (bool query_first : {false, true}) {
    for (bool overlap : {false, true}) {
      for (Wait wait : {Wait::None, Wait::DsZero, Wait::LoadOne, Wait::LoadZero}) {
        SCOPED_TRACE(testing::Message() << "query_first=" << query_first << " overlap=" << overlap
                                        << " wait=" << static_cast<int>(wait));
        PluginFixture f;
        ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna4", 32, 96, 128));
        PluginSinkConfig config;
        auto &sink = config.emplace<StringSink>();
        f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
        ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
        f.soc->set_plugin_group(f.plugin_group_);
        f.plugin_group_->onInit();

        constexpr uint64_t pc = 0x8000;
        constexpr uint32_t address = 0x4000;
        auto *cu = f.cu();
        auto *wf = cu->dispatch_wf(0, pc, 96, 128, 32);
        ASSERT_NE(wf, nullptr);
        wf->set_exec(1);
        std::array waves{wf};
        f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 128, 96, waves);
        f.mem->write32(address, 42);
        wf->debug_write_vgpr(0, 0, address);
        wf->debug_write_vgpr(2, 0, 0); // Independent mip selector.
        const auto sb = wf->sgpr_alloc().base;
        cu->write_sgpr(sb, address >> 8);
        cu->write_sgpr(sb + 1, 22u << 20);
        cu->write_sgpr(sb + 3, 9u << 28);
        const uint8_t data = overlap ? 8 : 9;
        const auto load = rdna4::build_vglobal(rdna4::kGlobalLoadB32Vglobal,
                                               {.saddr = amdgpu::kModernNullSelector, .vdst = 8});
        const auto query =
            rdna4::build_vimage(rdna4::kImageGetResinfoVimage,
                                {.dim = kImageDim2d, .dmask = 1, .vdata = data, .vaddr0 = 2});
        std::vector<uint32_t> code;
        auto append = [&](const auto &op) { code.insert(code.end(), op.begin(), op.end()); };
        if (query_first)
          append(query);
        else
          append(load);
        if (wait == Wait::DsZero)
          append(rdna4::build_sopp(rdna4::kSWaitDscntSopp));
        if (wait == Wait::LoadOne || wait == Wait::LoadZero)
          append(rdna4::build_sopp(rdna4::kSWaitLoadcntSopp,
                                   {.simm16 = uint16_t(wait == Wait::LoadOne)}));
        if (query_first)
          append(load);
        else
          append(query);
        f.mem->load_image(reinterpret_cast<const uint8_t *>(code.data()), code.size() * 4, pc);
        ASSERT_TRUE(cu->step());
        if (wait != Wait::None) {
          ASSERT_TRUE(cu->step());
        }
        ASSERT_TRUE(cu->step());
        ASSERT_EQ(wf->pc, pc + code.size() * 4);
        EXPECT_EQ(wf->debug_read_vgpr(8, 0), overlap && !query_first ? 1u : 42u);
        if (!overlap) {
          EXPECT_EQ(wf->debug_read_vgpr(data, 0), 1u);
        }
        const bool hazard = overlap && wait != Wait::LoadZero;
        EXPECT_EQ(cu->memory_wait_diagnostic_count() != 0, hazard);
        EXPECT_EQ(sink.str().find("Race on VGPR v8") != std::string::npos, hazard);
        wf->halt();
      }
    }
  }
}

TEST(RaceDetectorPluginTest, InlineImageQueryAdmitsItsCounterBeforeReadingOperands) {
  for (std::string_view profile :
       {"cdna1", "cdna2", "rdna1", "rdna2", "rdna3", "rdna3.5", "rdna4"}) {
    const bool cdna = profile.starts_with("cdna");
    const bool gfx10 = profile == "rdna1" || profile == "rdna2";
    const bool split = profile == "rdna4";
    const uint32_t lanes = cdna ? 64 : 32;
    SCOPED_TRACE(profile);
    for (const auto query_kind : kInlineImageQueries) {
      const bool consume_lod = query_kind == InlineImageQuery::Lod;
      if (consume_lod && (cdna || split))
        continue;
      SCOPED_TRACE(consume_lod);
      for (bool oldest_query : {false, true}) {
        for (const unsigned pending : {62u, 63u}) {
          SCOPED_TRACE(testing::Message() << oldest_query << " " << pending);
          PluginFixture f;
          ASSERT_NO_FATAL_FAILURE(f.initialize(1, profile, lanes, 96, 128));
          PluginSinkConfig config;
          auto &sink = config.emplace<StringSink>();
          f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
          ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
          f.soc->set_plugin_group(f.plugin_group_);
          f.plugin_group_->onInit();
          constexpr uint64_t pc = 0x8000;
          auto *cu = f.cu();
          auto *wf = cu->dispatch_wf(0, pc, 96, 128, lanes);
          ASSERT_NE(wf, nullptr);
          wf->set_exec(consume_lod ? 15 : 1);
          std::array waves{wf};
          f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 128, 96, waves);
          wf->debug_write_vgpr(0, 0, 0x4000);
          f.mem->write32(0x4000, 0);
          const auto sb = wf->sgpr_alloc().base;
          cu->write_sgpr(sb, 0x4000 >> 8);
          cu->write_sgpr(sb + 1, 22u << 20);
          cu->write_sgpr(sb + 3, 9u << 28);
          std::vector<uint32_t> code;
          auto append = [&](const auto &words) {
            code.insert(code.end(), words.begin(), words.end());
          };
          auto query = [&](uint8_t src, uint8_t dst, bool lod) {
            if (split)
              append(rdna4::build_vimage(
                  rdna4::kImageGetResinfoVimage,
                  {.dim = kImageDim2d, .dmask = 1, .vdata = dst, .vaddr0 = src}));
            else if (gfx10)
              append(rdna2::build_mimg(lod ? rdna2::kImageGetLodMimg : rdna2::kImageGetResinfoMimg,
                                       {.dim = kImageDim2d,
                                        .dmask = 1,
                                        .unorm = 1,
                                        .vaddr = src,
                                        .vdata = dst,
                                        .ssamp = 2}));
            else if (cdna)
              append(cdna1::build_mimg(
                  cdna1::kImageGetResinfoMimg,
                  {.dmask = 1, .unorm = 1, .vaddr = src, .vdata = dst, .ssamp = 2}));
            else
              append(rdna3::build_mimg(lod ? rdna3::kImageGetLodMimg : rdna3::kImageGetResinfoMimg,
                                       {.dim = kImageDim2d,
                                        .unorm = 1,
                                        .dmask = 1,
                                        .vaddr = src,
                                        .vdata = dst,
                                        .ssamp = 2}));
          };
          if (oldest_query)
            query(1, 8, false);
          for (unsigned i = oldest_query ? 1u : 0u; i < pending; ++i) {
            const uint8_t dst = i == 0 ? 8 : 16 + i;
            if (split)
              append(rdna4::build_vglobal(rdna4::kGlobalLoadB32Vglobal,
                                          {.saddr = amdgpu::kModernNullSelector, .vdst = dst}));
            else if (gfx10)
              append(rdna2::build_flat(rdna2::kFlatLoadDword,
                                       {.seg = 2, .addr = 0, .saddr = 125, .vdst = dst}));
            else if (cdna)
              append(cdna1::build_flat(cdna1::kFlatLoadDword,
                                       {.seg = 2, .addr = 0, .saddr = 127, .vdst = dst}));
            else
              append(rdna3::build_flat(rdna3::kFlatLoadB32Flat,
                                       {.seg = 2, .addr = 0, .saddr = 124, .vdst = dst}));
          }
          query(8, 4, consume_lod);
          f.mem->load_image(reinterpret_cast<const uint8_t *>(code.data()), code.size() * 4, pc);
          for (unsigned i = 0; i <= pending; ++i)
            ASSERT_TRUE(cu->step());
          ASSERT_EQ(wf->pc, pc + code.size() * 4);
          EXPECT_EQ(cu->memory_wait_diagnostic_count(), unsigned(pending == 62));
          EXPECT_EQ(sink.str().find("Race on VGPR v8") != std::string::npos, pending == 62);
          wf->halt();
        }
      }
    }
  }
}

TEST(RaceDetectorPluginTest, GenericFlatReservesBothCountersBeforeReadingAddress) {
  constexpr uint32_t global = 0x400000, result = 0x1234;
  for (std::string_view arch : {"rdna1", "rdna2", "rdna3", "rdna3.5", "rdna4", "cdna5"}) {
    const bool gfx10 = arch == "rdna1" || arch == "rdna2";
    const bool split = arch == "rdna4" || arch == "cdna5";
    for (bool ds : {false, true}) {
      for (unsigned pending : {62u, 63u}) {
        SCOPED_TRACE(testing::Message() << arch << " ds=" << ds << " pending=" << pending);
        PluginFixture f;
        ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 32, 96, 128));
        PluginSinkConfig config;
        auto &sink = config.emplace<StringSink>();
        f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
        ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
        f.soc->set_plugin_group(f.plugin_group_);
        f.plugin_group_->onInit();
        constexpr uint64_t pc = 0x8000;
        auto *cu = f.cu();
        auto *wf = cu->dispatch_wf(0, pc, 96, 128, 32);
        ASSERT_NE(wf, nullptr);
        wf->set_exec(1);
        wf->set_lds_size(256);
        std::array waves{wf};
        f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 128, 96, waves);
        wf->lds().write32(wf->lds_base(), global);
        wf->debug_write_vgpr(0, 0, ds ? 0 : global + 4);
        f.mem->write32(global + 4, global);
        f.mem->write32(global, result);
        std::vector<uint32_t> code;
        const auto append = [&](const auto &words) {
          code.insert(code.end(), words.begin(), words.end());
        };
        const auto append_global_or_flat = [&](uint8_t addr, uint8_t dst, bool flat) {
          if (split) {
            const auto null = amdgpu::kModernNullSelector;
            if (flat)
              append(rdna4::build_vflat(rdna4::kFlatLoadB32Vflat,
                                        {.saddr = null, .vdst = dst, .vaddr = addr}));
            else
              append(rdna4::build_vglobal(rdna4::kGlobalLoadB32Vglobal,
                                          {.saddr = null, .vdst = dst, .vaddr = addr}));
          } else {
            const uint8_t seg = flat ? 0 : 2;
            append(gfx10
                       ? rdna1::build_flat(rdna1::kFlatLoadDword,
                                           {.seg = seg, .addr = addr, .saddr = 125, .vdst = dst})
                       : rdna3::build_flat(rdna3::kFlatLoadB32Flat,
                                           {.seg = seg, .addr = addr, .saddr = 124, .vdst = dst}));
          }
        };
        const auto lds = [&](uint8_t dst) {
          if (split)
            return rdna4::build_vds(rdna4::kDsLoadB32Vds, {.addr = 0, .vdst = dst});
          return gfx10 ? rdna1::build_ds(rdna1::kDsReadB32, {.addr = 0, .vdst = dst})
                       : rdna3::build_ds(rdna3::kDsLoadB32Ds, {.addr = 0, .vdst = dst});
        };
        for (unsigned i = 0; i < pending; ++i) {
          const auto dst = static_cast<uint8_t>(i == 0 ? 8 : 16 + i);
          if (ds)
            append(lds(dst));
          else
            append_global_or_flat(0, dst, false);
        }
        append_global_or_flat(8, 4, true);
        f.mem->load_image(reinterpret_cast<const uint8_t *>(code.data()), code.size() * 4, pc);
        for (unsigned i = 0; i <= pending; ++i) {
          ASSERT_TRUE(cu->step());
        }
        ASSERT_EQ(wf->pc, pc + code.size() * 4);
        EXPECT_EQ(wf->debug_read_vgpr(4, 0), result);
        EXPECT_EQ(cu->memory_wait_diagnostic_count(), unsigned(pending == 62));
        EXPECT_EQ(sink.str().find("Race on VGPR v8") != std::string::npos, pending == 62);
        wf->halt();
      }
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna2FlatAdmitsBothCountersBeforeReadingAddress) {
  static constexpr uint32_t address = 0x4000, value = 0x13572468;
  class InitializeLds : public ExecutionPlugin {
  public:
    InitializeLds() : ExecutionPlugin("flat_address_from_lds") {}
    void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, uint32_t, uint32_t,
                                     std::span<Wavefront *> waves) override {
      for (auto *wf : waves)
        wf->lds().write32(wf->lds_base(), address);
    }
    void onAmdgpuWavefrontHalted(Wavefront &wf) override {
      EXPECT_EQ(wf.debug_read_vgpr(4, 0), value);
    }
  };
  for (uint32_t younger : {61u, 62u}) {
    SCOPED_TRACE(younger);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna2", 64, 106, 256));
    PluginSinkConfig config;
    auto &sink = config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<InitializeLds>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    const auto old = rdna2::build_ds(rdna2::kDsReadB32, {.addr = 12, .vdst = 8});
    const auto flat =
        rdna2::build_flat(rdna2::kFlatLoadDword, {.addr = 8, .saddr = 125, .vdst = 4});
    const auto null =
        rdna2::build_flat(rdna2::kFlatLoadDword, {.addr = 6, .saddr = 125, .vdst = 5});
    std::vector<uint32_t> code{0x7e180280, 0x7e120280, old[0], old[1], 0xbefe0480};
    for (uint32_t i = 0; i < younger; ++i)
      code.insert(code.end(), null.begin(), null.end());
    code.insert(code.end(), {0xbefe0481u, flat[0], flat[1], 0xbf8c0000u, 0xbf810000u});
    const auto kernel = f.write_kernel(0x1000, code.data(), code.size(), 12, 4);
    f.mem->write32(address, value);
    test::AqlQueue queue(f.mem, f.cp());
    queue.dispatch(kernel, 64, 64);
    f.run_until_idle();
    ASSERT_TRUE(f.cu()->is_idle());
    EXPECT_EQ(sink.str().find("Race on VGPR v8 [") != std::string::npos, younger == 61);
    EXPECT_EQ(f.cu()->memory_wait_diagnostic_count(), younger == 61 ? 1u : 0u);
  }
}

static void check_gfx10_gds_exec_locks(const char *arch) {
  constexpr uint32_t clear_scc = 0xbf070000, set_scc = 0xbf060000, nop = 0xbf800000;
  constexpr uint32_t mov_lo = 0xbefe0381, cmov_lo = 0xbefe0580, cmov_hi = 0xbeff0580;
  constexpr uint32_t cmov_pair = 0xbefe06c1, cmovk_lo = 0xb17e0001;
  constexpr uint32_t bad_source = 0xbefe05d1; // s_cmov_b32 exec_lo, reserved selector 209
  constexpr uint32_t loop_begin = 0xbd840000, loop_end = 0xbe040000;
  constexpr uint32_t lgkm0 = 0xbf8cc07f, exp0 = 0xbf8cff0f;
  struct Case {
    uint32_t condition;
    uint32_t probe;
    bool wave32;
    bool wave64;
    uint32_t next = nop;
    uint32_t low = 1;
    uint32_t high = 0;
  };
  for (const auto &c : {
           Case{clear_scc, cmov_lo, false, false}, Case{set_scc, cmov_lo, true, true},
           Case{clear_scc, cmov_hi, false, false}, Case{set_scc, cmov_hi, false, true},
           Case{clear_scc, cmov_pair, false, false}, Case{set_scc, cmov_pair, true, true},
           Case{clear_scc, cmovk_lo, false, false}, Case{set_scc, cmovk_lo, true, true},
           Case{clear_scc, cmov_lo, true, true,
                mov_lo},                            // The following real write still sees the lock.
           Case{set_scc, bad_source, false, false}, // Decode rejects the reserved source.
           Case{nop, loop_begin, false, false, nop, 0, 1}, // High pass saves zero.
           Case{nop, loop_end, true, true, nop, 0, 1},     // High pass writes EXEC_LO.
           Case{nop, mov_lo, true, true},                  // EXEC_LO already equals 1.
           Case{nop, 0x7da40100, true, true},      // v_cmpx_eq_u32 v0,v0 leaves EXEC unchanged.
           Case{nop, loop_begin, false, true},     // Low pass clears only EXEC_HI.
           Case{nop, loop_end, false, false},      // Saved s4 and EXEC_HI are zero.
           Case{0xbe840381, loop_end, true, true}, // s_mov_b32 s4, 1 starts a high pass.
           Case{nop, 0xbefe2104, true, true},      // s_swappc_b64 writes the return PC to EXEC.
       }) {
    for (uint32_t width : {32u, 64u}) {
      for (uint32_t wait : {lgkm0, exp0}) {
        SCOPED_TRACE(testing::Message() << std::hex << c.condition << "," << c.probe << " wave="
                                        << std::dec << width << " wait=" << std::hex << wait);
        PluginFixture f;
        ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 64, 106, 256));
        PluginSinkConfig config;
        auto &sink = config.emplace<StringSink>();
        f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
        ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
        f.soc->set_plugin_group(f.plugin_group_);
        f.plugin_group_->onInit();
        const auto ds = rdna2::build_ds(rdna2::kDsReadB32, {.gds = 1, .addr = 8, .vdst = 4});
        const uint32_t code[] = {0xbefc03ff,
                                 0x00800004, // s_mov_b32 m0, GDS segment
                                 0xbe840480, // s_mov_b64 s[4:5],0
                                 0xbefe0380 + c.low,
                                 0xbeff0380 + c.high, // s_mov_b32 exec_lo/hi, inline constant
                                 ds[0],
                                 ds[1],
                                 wait,
                                 c.condition,
                                 c.probe,
                                 c.next,
                                 0xbf8c0000,
                                 0xbf810000};
        f.mem->write32(0, 0xbf810000); // S_ENDPGM at the SWAPPC target.
        const auto kernel = f.write_kernel(0x1000, code, std::size(code), 12, 1024);
        using namespace amdhsa;
        auto properties =
            f.mem->read32(kernel + offsetof(kernel_descriptor_t, kernel_code_properties));
        RJ_AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, width == 32);
        f.mem->write16(kernel + offsetof(kernel_descriptor_t, kernel_code_properties), properties);
        test::AqlQueue queue(f.mem, f.cp());
        queue.dispatch(kernel, width, width);
        f.run_until_idle();
        ASSERT_TRUE(f.cu()->is_idle());
        EXPECT_EQ(f.engine->exit_code(), c.probe == bad_source ? 1 : 0);
        const bool hazard = wait != exp0 && (width == 32 ? c.wave32 : c.wave64);
        EXPECT_EQ(sink.str().find("Race on EXEC [") != std::string::npos, hazard);
        EXPECT_EQ(f.cu()->memory_wait_diagnostic_count(), unsigned(hazard));
      }
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna1GdsExecLockChecksOnlyWrittenActiveWords) {
  check_gfx10_gds_exec_locks("rdna1");
}

TEST(RaceDetectorPluginTest, Rdna2GdsExecLockChecksOnlyWrittenActiveWords) {
  check_gfx10_gds_exec_locks("rdna2");
}

TEST(RaceDetectorPluginTest, Gfx10GdsRangeViolationIsIndependentOfRoutedObservers) {
  class HaltStatus final : public ExecutionPlugin {
  public:
    HaltStatus() : ExecutionPlugin("halt_status") {}
    void onAmdgpuWavefrontHalted(Wavefront &wave) override { trapsts = wave.trapsts(); }
    std::optional<uint32_t> trapsts;
  };

  for (const char *arch : {"rdna1", "rdna2"}) {
    for (bool observe : {false, true}) {
      for (uint32_t offset : {0u, 16u}) {
        SCOPED_TRACE(testing::Message() << arch << " observe=" << observe << " offset=" << offset);
        PluginFixture f;
        ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 64, 106, 256));
        PluginSinkConfig config;
        config.emplace<StringSink>();
        f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
        if (observe) {
          ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
        }
        auto status = std::make_unique<HaltStatus>();
        auto *halt = status.get();
        ASSERT_TRUE(f.plugin_group_->add(std::move(status)));
        f.soc->set_plugin_group(f.plugin_group_);
        f.plugin_group_->onInit();

        // On RDNA2 a range violation enters the trap handler before the normal end instruction.
        constexpr uint64_t trap_entry = 0x9000;
        f.cu()->set_trap_handler_resolver([](const Wavefront &) {
          return ComputeUnitCore::TrapHandlerConfig{trap_entry, 0, false};
        });
        f.mem->write32(trap_entry, 0xbf810000); // s_endpgm

        const auto ds = rdna2::build_ds(rdna2::kDsReadB32, {.gds = 1, .addr = 8, .vdst = 4});
        const uint32_t code[] = {0xba800501, 1,          // s_setreg_imm32_b32 MODE[20], 1
                                 0xbefc03ff, 0x00800004, // M0 selects four bytes at 128.
                                 0x7e1002ff, offset,     // v_mov_b32 v8, offset
                                 0xbefe0481,             // s_mov_b64 exec, 1
                                 ds[0],      ds[1],      0xbf8c0000, 0xbf810000};
        const auto kernel = f.write_kernel(0x1000, code, std::size(code), 12, 1024);
        test::AqlQueue queue(f.mem, f.cp());
        queue.dispatch(kernel, 64, 64);
        f.run_until_idle();
        ASSERT_TRUE(halt->trapsts.has_value());
        const bool mem_viol = (*halt->trapsts & (1u << 8)) != 0;
        EXPECT_EQ(mem_viol, std::string_view(arch) == "rdna2" && offset >= 4);
      }
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna2GdsViolationHandlerRetiresOnlyTheFaultingSourceLock) {
  constexpr uint32_t nop = 0xbf800000, exp0 = 0xbf8cff0f;
  constexpr uint32_t write_addr = 0x7e100280, write_exec = 0xbefe0481;
  constexpr uint64_t tba = 0x9000;
  struct Case {
    uint8_t offset;
    bool enable_trap;
    bool older_gds;
    bool wait_exp;
    bool source_hazard;
  };
  const Case cases[] = {
      {8, true, false, false, false}, {8, true, true, false, true},  {8, true, true, true, false},
      {8, false, false, false, true}, {0, true, false, false, true}, {0, true, false, true, false},
  };
  for (uint32_t width : {32u, 64u}) {
    for (const auto &c : cases) {
      for (const bool probe_vgpr : {false, true}) {
        SCOPED_TRACE(testing::Message() << "width=" << width << " offset=" << unsigned(c.offset)
                                        << " trap=" << c.enable_trap << " older=" << c.older_gds
                                        << " wait=" << c.wait_exp << " probe_vgpr=" << probe_vgpr);
        PluginFixture f;
        ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna2", width, 106, 256));
        PluginSinkConfig config;
        auto &sink = config.emplace<StringSink>();
        f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
        ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
        f.soc->set_plugin_group(f.plugin_group_);
        f.plugin_group_->onInit();

        unsigned entries = 0;
        f.cu()->set_trap_handler_resolver([&](const Wavefront &) {
          ++entries;
          return ComputeUnitCore::TrapHandlerConfig{tba, 0, false};
        });
        const bool takes_trap = c.enable_trap && c.offset == 8;
        f.mem->write32(tba, c.wait_exp ? exp0 : nop);
        f.mem->write32(tba + 4, probe_vgpr ? write_addr : write_exec);
        f.mem->write32(tba + 8, 0xbe80226c); // s_rfe_b64 ttmp[0:1] restores PC and PRIV.
        const auto older = rdna2::build_ds(rdna2::kDsReadB32, {.gds = 1, .addr = 8, .vdst = 3});
        const auto current = rdna2::build_ds(rdna2::kDsReadB32,
                                             {.offset0 = c.offset, .gds = 1, .addr = 8, .vdst = 4});
        const uint32_t code[] = {0xbefc03ff,
                                 0x00800008, // M0 selects eight bytes starting at 128.
                                 write_addr, write_exec, 0xba800501,
                                 c.enable_trap, // s_setreg_imm32_b32 MODE[20]
                                 c.older_gds ? older[0] : nop, c.older_gds ? older[1] : nop,
                                 current[0], current[1], !takes_trap && c.wait_exp ? exp0 : nop,
                                 // The checker reports each source once. Probe the other
                                 // source after RFE to verify its lock survived the handler.
                                 (takes_trap != probe_vgpr) ? write_addr : write_exec, 0xbf8c0000,
                                 0xbf810000};
        const auto kernel = f.write_kernel(0x1000, code, std::size(code));
        using namespace amdhsa;
        auto props = f.mem->read32(kernel + offsetof(kernel_descriptor_t, kernel_code_properties));
        RJ_AMDHSA_BITS_SET(props, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, width == 32);
        f.mem->write16(kernel + offsetof(kernel_descriptor_t, kernel_code_properties), props);
        test::AqlQueue queue(f.mem, f.cp());
        queue.dispatch(kernel, 1, width);
        f.run_until_idle();
        ASSERT_TRUE(f.cu()->is_idle());
        EXPECT_EQ(entries, unsigned(takes_trap));
        const auto race = probe_vgpr ? "Race on VGPR v8 [" : "Race on EXEC [";
        EXPECT_EQ(sink.str().find(race) != std::string::npos, c.source_hazard) << sink.str();
        // The plugin emits one report per conflicting producer PC. The default
        // checker separately diagnoses its EXEC and VGPR source obligations.
        const unsigned writes = takes_trap ? 2u : 1u;
        EXPECT_EQ(f.cu()->memory_wait_diagnostic_count(), c.source_hazard ? writes : 0u);
      }
    }
  }
}

static void check_gfx10_gds_waits(const char *arch) {
  enum class Probe { ReadResult, ReadAddress, WriteAddress, WriteHighData, CompareExec, WriteExec };
  enum class Form { Read, AliasedRead, AddReturn64, EmptyRead, SuppressedReturn };
  constexpr uint32_t no_wait = 0xbf800000, lgkm0 = 0xbf8cc07f, exp0 = 0xbf8cff0f;
  constexpr uint32_t exp1 = 0xbf8cff1f, both = 0xbf8c0000, barrier = 0xbf8a0000;
  struct Case {
    Form form;
    uint32_t wait;
    Probe probe;
    bool hazard;
  };
  for (const auto &c : {Case{Form::Read, no_wait, Probe::ReadResult, true},
                        Case{Form::Read, lgkm0, Probe::ReadResult, false},
                        Case{Form::Read, exp0, Probe::ReadResult, true},
                        Case{Form::Read, both, Probe::ReadResult, false},
                        Case{Form::Read, no_wait, Probe::ReadAddress, false},
                        Case{Form::Read, lgkm0, Probe::WriteAddress, true},
                        Case{Form::Read, exp0, Probe::WriteAddress, false},
                        Case{Form::Read, exp1, Probe::WriteAddress, true},
                        Case{Form::Read, barrier, Probe::WriteAddress, true},
                        Case{Form::AliasedRead, lgkm0, Probe::ReadResult, false},
                        Case{Form::AliasedRead, lgkm0, Probe::WriteAddress, true},
                        Case{Form::AddReturn64, lgkm0, Probe::ReadResult, false},
                        Case{Form::AddReturn64, exp0, Probe::ReadResult, true},
                        Case{Form::AddReturn64, lgkm0, Probe::WriteHighData, true},
                        Case{Form::AddReturn64, exp0, Probe::WriteHighData, false},
                        Case{Form::Read, lgkm0, Probe::WriteExec, true},
                        Case{Form::Read, exp0, Probe::WriteExec, false},
                        Case{Form::Read, lgkm0, Probe::CompareExec, true},
                        Case{Form::Read, exp0, Probe::CompareExec, false},
                        Case{Form::EmptyRead, no_wait, Probe::WriteExec, true},
                        Case{Form::SuppressedReturn, no_wait, Probe::WriteExec, false},
                        Case{Form::SuppressedReturn, no_wait, Probe::WriteHighData, false}}) {
    SCOPED_TRACE(testing::Message() << "form=" << int(c.form) << " wait=" << std::hex << c.wait
                                    << " probe=" << int(c.probe));
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 64, 106, 256));
    PluginSinkConfig config;
    auto &sink = config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    const bool atomic = c.form == Form::AddReturn64 || c.form == Form::SuppressedReturn;
    const uint8_t dst = c.form == Form::AliasedRead ? 8 : c.form == Form::SuppressedReturn ? 15 : 4;
    const auto ds = rdna2::build_ds(atomic ? rdna2::kDsAddRtnU64 : rdna2::kDsReadB32,
                                    {.gds = 1, .addr = 8, .data0 = 12, .vdst = dst});
    const uint32_t probe = c.probe == Probe::ReadResult ? 0x7e140300u + dst // v_mov_b32 v10, vdst
                           : c.probe == Probe::ReadAddress   ? 0x7e140308u
                           : c.probe == Probe::WriteAddress  ? 0x7e100280u
                           : c.probe == Probe::WriteHighData ? 0x7e1a0280u
                           : c.probe == Probe::CompareExec   ? 0x7da40d06u
                                                             : 0xbefe0481u;
    const uint32_t code[] = {
        0xbefc03ff, 0x00800004, 0x7e100280,
        0x7e180280, 0x7e1a0280, c.form == Form::EmptyRead ? 0xbefe0480u : 0xbefe0481u,
        ds[0],      ds[1],      c.wait,
        probe,      both,       0xbf810000};
    const auto kernel = f.write_kernel(0x1000, code, std::size(code), 12, 1024);
    using namespace amdhsa;
    auto fields = f.mem->read32(kernel + offsetof(kernel_descriptor_t, compute_pgm_rsrc1));
    RJ_AMDHSA_BITS_SET(fields, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 3);
    f.mem->write32(kernel + offsetof(kernel_descriptor_t, compute_pgm_rsrc1), fields);
    test::AqlQueue queue(f.mem, f.cp());
    queue.dispatch(kernel, 64, 64);
    f.run_until_idle();
    ASSERT_TRUE(f.cu()->is_idle());
    const auto reg = c.probe == Probe::WriteExec || c.probe == Probe::CompareExec
                         ? std::string("EXEC")
                         : "VGPR v" + std::to_string(c.probe == Probe::ReadResult      ? dst
                                                     : c.probe == Probe::WriteHighData ? 13
                                                                                       : 8);
    EXPECT_EQ(sink.str().find("Race on " + reg + " [") != std::string::npos, c.hazard);
    EXPECT_EQ(f.cu()->memory_wait_diagnostic_count(), unsigned(c.hazard));
  }
}

TEST(RaceDetectorPluginTest, Rdna2GdsResultsAndSourceLocksHaveIndependentWaits) {
  check_gfx10_gds_waits("rdna2");
}

TEST(RaceDetectorPluginTest, Rdna1GdsResultsAndSourceLocksHaveIndependentWaits) {
  check_gfx10_gds_waits("rdna1");
}

TEST(RaceDetectorPluginTest, Rdna2DualDsStoresReadAndLockOnlyEffectiveData) {
  struct Form {
    uint16_t write, read;
    uint8_t words;
    bool exchange;
  };
  const Form forms[] = {
      {rdna2::kDsWrite2B32, rdna2::kDsRead2B32, 1, false},
      {rdna2::kDsWrite2B64, rdna2::kDsRead2B64, 2, false},
      {rdna2::kDsWrite2st64B32, rdna2::kDsRead2st64B32, 1, false},
      {rdna2::kDsWrite2st64B64, rdna2::kDsRead2st64B64, 2, false},
      {rdna2::kDsWrxchg2RtnB32, rdna2::kDsRead2B32, 1, true},
      {rdna2::kDsWrxchg2RtnB64, rdna2::kDsRead2B64, 2, true},
      {rdna2::kDsWrxchg2st64RtnB32, rdna2::kDsRead2st64B32, 1, true},
      {rdna2::kDsWrxchg2st64RtnB64, rdna2::kDsRead2st64B64, 2, true},
  };
  class Registers final : public ExecutionPlugin {
  public:
    explicit Registers(uint8_t pending) : ExecutionPlugin("dual_ds_data"), pending_(pending) {}
    void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, uint32_t, uint32_t,
                                     std::span<Wavefront *> waves) override {
      auto &wave = *waves.front();
      wave.set_m0(4096);
      wave.debug_write_vgpr(4, 0, 0);    // Dual access base
      wave.debug_write_vgpr(5, 0, 3072); // Prior load reads zero, outside both target ranges.
      for (unsigned i = 0; i < 2; ++i) {
        wave.debug_write_vgpr(8 + i, 0, input(8 + i));
        wave.debug_write_vgpr(12 + i, 0, input(12 + i));
      }
      for (unsigned i = 0; i < 4; ++i)
        wave.debug_write_vgpr(20 + i, 0, ~0u);
    }
    void onAmdgpuWavefrontHalted(Wavefront &wave) override {
      for (unsigned i = 0; i < 4; ++i) {
        returned[i] = wave.debug_read_vgpr(20 + i, 0);
        stored[i] = wave.debug_read_vgpr(24 + i, 0);
      }
    }
    uint32_t input(unsigned reg) const { return reg == pending_ ? 0u : 0x76540000u + reg; }
    std::array<uint32_t, 4> returned{}, stored{};

  private:
    uint8_t pending_;
  };
  enum Probe {
    ReadData0,
    ReadData1,
    OverwriteData0,
    OverwriteData1,
    WaitThenOverwriteData1,
    OverwriteAlias
  };
  constexpr uint32_t nop = 0xbf800000, lgkm0 = 0xbf8cc07f, exp0 = 0xbf8cff0f, both = 0xbf8c0000;
  for (const auto &f : forms)
    for (uint32_t width : {32u, 64u})
      for (uint8_t gds : {0, 1})
        for (bool equal : {false, true})
          for (auto probe : {ReadData0, ReadData1, OverwriteData0, OverwriteData1,
                             WaitThenOverwriteData1, OverwriteAlias}) {
            if (probe == OverwriteAlias && !equal)
              continue;
            SCOPED_TRACE(testing::Message() << f.write << " wave" << width << " gds=" << int(gds)
                                            << " equal=" << equal << " probe=" << probe);
            PluginFixture fixture;
            ASSERT_NO_FATAL_FAILURE(fixture.initialize(1, "rdna2", 64, 106, 256));
            PluginSinkConfig config;
            auto &sink = config.emplace<StringSink>();
            fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
            ASSERT_TRUE(fixture.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
            const bool reads = probe == ReadData0 || probe == ReadData1;
            const bool data0 =
                probe == ReadData0 || probe == OverwriteData0 || probe == OverwriteAlias;
            const uint8_t reg = (data0 ? 8 : 12) + f.words - 1;
            auto observer = std::make_unique<Registers>(reads ? reg : 255);
            auto *captured = observer.get();
            ASSERT_TRUE(fixture.plugin_group_->add(std::move(observer)));
            fixture.soc->set_plugin_group(fixture.plugin_group_);
            fixture.plugin_group_->onInit();

            const uint8_t first = width == 32 ? 0 : 1, second = first + !equal;
            const auto prior =
                rdna2::build_ds(rdna2::kDsReadB32, {.gds = gds, .addr = 5, .vdst = reg});
            // An unused DATA1 may alias an active DATA0 word.
            const uint8_t second_data = probe == OverwriteAlias ? reg : 12;
            const auto write =
                rdna2::build_ds(f.write, {.offset0 = first,
                                          .offset1 = second,
                                          .gds = gds,
                                          .addr = 4,
                                          .data0 = 8,
                                          .data1 = second_data,
                                          .vdst = f.exchange ? uint8_t{20} : uint8_t{0}});
            const auto read = rdna2::build_ds(
                f.read, {.offset0 = first, .offset1 = second, .gds = gds, .addr = 4, .vdst = 24});
            const uint32_t code[] = {reads ? prior[0] : nop,
                                     reads ? prior[1] : nop,
                                     write[0],
                                     write[1],
                                     lgkm0,
                                     probe == WaitThenOverwriteData1 ? exp0 : nop,
                                     reads ? nop : 0x7e000280u | (uint32_t{reg} << 17),
                                     both,
                                     read[0],
                                     read[1],
                                     both,
                                     0xbf810000};
            const auto kernel = fixture.write_kernel(0x1000, code, std::size(code), 12, 4096);
            using namespace amdhsa;
            auto properties =
                fixture.mem->read32(kernel + offsetof(kernel_descriptor_t, kernel_code_properties));
            RJ_AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                               width == 32);
            fixture.mem->write16(kernel + offsetof(kernel_descriptor_t, kernel_code_properties),
                                 properties);
            test::AqlQueue queue(fixture.mem, fixture.cp());
            queue.dispatch(kernel, 1, 1);
            fixture.run_until_idle();
            ASSERT_TRUE(fixture.cu()->is_idle());
            const bool hazard =
                (data0 || !equal) && (reads || (gds && probe != WaitThenOverwriteData1));
            EXPECT_EQ(sink.str().find("Race on VGPR v" + std::to_string(reg) + " [") !=
                          std::string::npos,
                      hazard);
            EXPECT_EQ(fixture.cu()->memory_wait_diagnostic_count(), unsigned(hazard));
            for (unsigned i = 0; i < f.words; ++i) {
              EXPECT_EQ(captured->stored[i], captured->input(8 + i));
              EXPECT_EQ(captured->stored[f.words + i], captured->input((equal ? 8 : 12) + i));
              if (f.exchange) {
                EXPECT_EQ(captured->returned[i], 0u);
                EXPECT_EQ(captured->returned[f.words + i], 0u);
              }
            }
          }
}

TEST(RaceDetectorPluginTest, Rdna2GdsStoreLocksTheSourceWordsItReadsAtTheVgprBoundary) {
  constexpr uint32_t low = 0x76543210, high = 0xfedcba98, nop = 0xbf800000;
  enum class Probe { PendingSource, OverwriteLow, OverwriteHigh, OverwriteUnrelated };
  class Data : public ExecutionPlugin {
  public:
    Data() : ExecutionPlugin("gds_source_data") {}
    void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, uint32_t, uint32_t,
                                     std::span<Wavefront *> waves) override {
      for (auto *wf : waves)
        wf->lds().write32(wf->lds_base(), 0xfedcba98);
    }
    void onAmdgpuWavefrontHalted(Wavefront &wf) override {
      lo = wf.debug_read_vgpr(4, 0);
      hi = wf.debug_read_vgpr(5, 0);
    }
    uint32_t lo = 0, hi = 0;
  };
  for (const auto probe : {Probe::PendingSource, Probe::OverwriteLow, Probe::OverwriteHigh,
                           Probe::OverwriteUnrelated}) {
    for (const bool wait : {false, true}) {
      SCOPED_TRACE(testing::Message() << "probe=" << int(probe) << " wait=" << wait);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna2", 64, 106, 256));
      PluginSinkConfig config;
      auto &sink = config.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      auto data = std::make_unique<Data>();
      auto *result = data.get();
      ASSERT_TRUE(f.plugin_group_->add(std::move(data)));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      const auto prior = rdna2::build_ds(rdna2::kDsReadB32, {.addr = 8, .vdst = 0});
      const auto store = rdna2::build_ds(rdna2::kDsWriteB64, {.gds = 1, .addr = 8, .data0 = 15});
      const auto read = rdna2::build_ds(rdna2::kDsReadB64, {.gds = 1, .addr = 8, .vdst = 4});
      const bool pending = probe == Probe::PendingSource;
      const uint32_t write = probe == Probe::OverwriteLow         ? 0x7e1e0280u
                             : probe == Probe::OverwriteHigh      ? 0x7e000280u
                             : probe == Probe::OverwriteUnrelated ? 0x7e020280u
                                                                  : nop;
      const uint32_t code[] = {0xbefc03ff,
                               0x00800010,
                               0xbefe0481,
                               0x7e100280,
                               0x7e0002ff,
                               high,
                               0x7e1e02ff,
                               low,
                               pending ? prior[0] : nop,
                               pending ? prior[1] : nop,
                               pending && wait ? 0xbf8cc07fu : nop,
                               store[0],
                               store[1],
                               !pending && wait ? 0xbf8cff0fu : nop,
                               write,
                               0xbf8c0000u,
                               read[0],
                               read[1],
                               0xbf8c0000u,
                               0xbf810000u};
      const auto kernel = f.write_kernel(0x1000, code, std::size(code), 12, 1024);
      using namespace amdhsa;
      auto fields = f.mem->read32(kernel + offsetof(kernel_descriptor_t, compute_pgm_rsrc1));
      RJ_AMDHSA_BITS_SET(fields, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 3);
      f.mem->write32(kernel + offsetof(kernel_descriptor_t, compute_pgm_rsrc1), fields);
      test::AqlQueue queue(f.mem, f.cp());
      queue.dispatch(kernel, 64, 64);
      f.run_until_idle();
      ASSERT_TRUE(f.cu()->is_idle());
      EXPECT_EQ(result->lo, low);
      EXPECT_EQ(result->hi, high);
      const bool hazard = !wait && probe != Probe::OverwriteUnrelated;
      const char *reg = probe == Probe::OverwriteLow         ? "v15"
                        : probe == Probe::OverwriteUnrelated ? "v1"
                                                             : "v0";
      EXPECT_EQ(sink.str().find(std::string("Race on VGPR ") + reg + " [") != std::string::npos,
                hazard);
      EXPECT_EQ(f.cu()->memory_wait_diagnostic_count(), unsigned(hazard));
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna2GdsDestinationWriteOrderKeepsSourceLocksIndependent) {
  enum class Op { GdsRead, GdsAdd, LdsRead };
  enum class Alias { Result, AddressAndResult, Data, Distinct };
  constexpr uint32_t none = 0xbf800000, lgkm0 = 0xbf8cc07f, exp0 = 0xbf8cff0f;
  struct Case {
    Op first;
    Op second;
    Alias alias;
    uint32_t wait;
    bool hazard;
  };
  const Case cases[] = {
      {Op::GdsRead, Op::GdsRead, Alias::Result, none, false},
      {Op::GdsRead, Op::GdsAdd, Alias::Result, none, false},
      {Op::GdsAdd, Op::GdsRead, Alias::Result, none, false},
      {Op::GdsAdd, Op::GdsAdd, Alias::Result, none, false},
      {Op::GdsRead, Op::GdsRead, Alias::Distinct, none, false},
      {Op::GdsRead, Op::GdsRead, Alias::AddressAndResult, none, true},
      {Op::GdsRead, Op::GdsRead, Alias::AddressAndResult, lgkm0, true},
      {Op::GdsRead, Op::GdsRead, Alias::AddressAndResult, exp0, false},
      {Op::GdsAdd, Op::GdsRead, Alias::Data, none, true},
      {Op::GdsAdd, Op::GdsRead, Alias::Data, exp0, false},
      {Op::LdsRead, Op::GdsRead, Alias::Result, none, true},
      {Op::GdsRead, Op::LdsRead, Alias::Result, exp0, true},
      {Op::LdsRead, Op::GdsRead, Alias::Result, lgkm0, false},
      {Op::GdsRead, Op::LdsRead, Alias::Result, lgkm0, false},
  };
  class ObserveResult : public ExecutionPlugin {
  public:
    ObserveResult() : ExecutionPlugin("gds_waw_result") {}
    void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, uint32_t, uint32_t,
                                     std::span<Wavefront *> waves) override {
      for (auto *wf : waves)
        wf->lds().write32(wf->lds_base() + 4, 0x33333333);
    }
    void onAmdgpuWavefrontHalted(Wavefront &wf) override { value = wf.debug_read_vgpr(6, 0); }
    uint32_t value = 0;
  };
  for (uint32_t width : {32u, 64u}) {
    for (const auto &c : cases) {
      SCOPED_TRACE(testing::Message()
                   << "width=" << width << " first=" << int(c.first) << " second=" << int(c.second)
                   << " alias=" << int(c.alias) << " wait=" << std::hex << c.wait);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna2", width, 106, 256));
      PluginSinkConfig config;
      auto &sink = config.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      auto observer = std::make_unique<ObserveResult>();
      auto *result = observer.get();
      ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      const uint8_t destination = c.alias == Alias::Distinct ? 5 : c.alias == Alias::Data ? 9 : 4;
      const auto first =
          rdna2::build_ds(c.first == Op::GdsAdd ? rdna2::kDsAddRtnU32 : rdna2::kDsReadB32,
                          {.gds = c.first != Op::LdsRead,
                           .addr = static_cast<uint8_t>(c.alias == Alias::AddressAndResult ? 4 : 8),
                           .data0 = 9,
                           .vdst = 4});
      const auto second =
          rdna2::build_ds(c.second == Op::GdsAdd ? rdna2::kDsAddRtnU32 : rdna2::kDsReadB32,
                          {.offset0 = 4,
                           .gds = c.second != Op::LdsRead,
                           .addr = 8,
                           .data0 = 9,
                           .vdst = destination});
      const auto init =
          rdna2::build_ds(rdna2::kDsWriteB32, {.offset0 = 4, .gds = 1, .addr = 8, .data0 = 10});
      const uint32_t code[] = {
          0xbefc03ff,
          0x00800010,
          0xbefe0481,
          0x7e080280,
          0x7e100280,
          0x7e120281,
          0x7e1402ff,
          0x22222222,
          init[0],
          init[1],
          0xbf8c0000,
          first[0],
          first[1],
          c.wait,
          second[0],
          second[1],
          0xbf8c0000,
          0x7e0c0300u + destination, // consume the younger result after both counters drain
          0xbf810000};
      const auto kernel = f.write_kernel(0x1000, code, std::size(code), 12, 16);
      using namespace amdhsa;
      auto props = f.mem->read32(kernel + offsetof(kernel_descriptor_t, kernel_code_properties));
      RJ_AMDHSA_BITS_SET(props, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, width == 32);
      f.mem->write16(kernel + offsetof(kernel_descriptor_t, kernel_code_properties), props);
      test::AqlQueue queue(f.mem, f.cp());
      queue.dispatch(kernel, width, width);
      f.run_until_idle();
      ASSERT_TRUE(f.cu()->is_idle());
      EXPECT_EQ(result->value, c.second == Op::LdsRead ? 0x33333333u : 0x22222222u);
      EXPECT_EQ(sink.str().find("Race on VGPR v" + std::to_string(destination) + " [") !=
                    std::string::npos,
                c.hazard);
      EXPECT_EQ(f.cu()->memory_wait_diagnostic_count(), unsigned(c.hazard));
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna2ClippedFlatResultRequiresBothWaits) {
  enum Form { Valid, Partial, Clipped, PrivateOnly, LdsOob, NoActive, Nop };
  static constexpr uint64_t shared = uint64_t{2} << 32, priv = uint64_t{3} << 32, scratch = 0x8000;
  static constexpr uint32_t lds_lo = 0x11111111, lds_hi = 0x22222222;
  static constexpr uint32_t vm_lo = 0x33333333, vm_hi = 0x44444444, sentinel = 0xdeadbeef;
  constexpr uint32_t lgkm0 = 0xbf8cc07f, vm0 = 0xbf8c3f70, both = 0xbf8c0000, nop = 0xbf800000;
  class Inputs : public ExecutionPlugin {
  public:
    explicit Inputs(Form form) : ExecutionPlugin("flat_result_domains"), form(form) {}
    void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, uint32_t, uint32_t,
                                     std::span<Wavefront *> waves) override {
      for (auto *wf : waves) {
        wf->set_apertures(shared, shared + UINT32_MAX, priv, priv + UINT32_MAX,
                          /*override_launch=*/true);
        wf->set_scratch_base(scratch);
        wf->set_scratch_lane_size(form == Valid ? 12 : form == Partial ? 8 : 4);
        wf->lds().write32(wf->lds_base() + 4, lds_lo);
        wf->lds().write32(wf->lds_base() + 8, lds_hi);
        for (unsigned lane = 0; lane < 2; ++lane) {
          wf->debug_write_vgpr(8, lane, !lane && form == LdsOob ? 1024 : 4);
          wf->debug_write_vgpr(9, lane, (!lane && form != PrivateOnly ? shared : priv) >> 32);
          wf->debug_write_vgpr(4, lane, sentinel);
          wf->debug_write_vgpr(5, lane, sentinel);
        }
      }
    }
    void onAmdgpuWavefrontHalted(Wavefront &wf) override {
      for (unsigned lane = 0; lane < 2; ++lane) {
        const bool empty = form == Nop || form == NoActive;
        const bool from_lds = !lane && form != PrivateOnly && form != LdsOob;
        EXPECT_EQ(wf.debug_read_vgpr(4, lane), empty                              ? sentinel
                                               : from_lds                         ? lds_lo
                                               : form == Valid || form == Partial ? vm_lo
                                                                                  : 0u);
        EXPECT_EQ(wf.debug_read_vgpr(5, lane), empty           ? sentinel
                                               : from_lds      ? lds_hi
                                               : form == Valid ? vm_hi
                                                               : 0u);
      }
    }
    Form form;
  };
  for (const auto width : {32u, 64u}) {
    for (const auto form : {Valid, Partial, Clipped, PrivateOnly, LdsOob, NoActive, Nop}) {
      for (const auto wait : {lgkm0, vm0, both, nop}) {
        for (unsigned lane : {0u, 1u}) {
          SCOPED_TRACE(testing::Message() << "width=" << width << " form=" << form
                                          << " wait=" << std::hex << wait << " lane=" << lane);
          PluginFixture f;
          ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna2", width, 106, 256));
          f.cu()->set_apertures(shared, shared + UINT32_MAX, priv, priv + UINT32_MAX);
          PluginSinkConfig config;
          auto &sink = config.emplace<StringSink>();
          f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
          ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
          ASSERT_TRUE(f.plugin_group_->add(std::make_unique<Inputs>(form)));
          auto observer = std::make_unique<MemoryObservationPlugin>();
          auto *routes = observer.get();
          ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
          f.soc->set_plugin_group(f.plugin_group_);
          f.plugin_group_->onInit();
          const auto flat =
              rdna2::build_flat(rdna2::kFlatLoadDwordx2, {.addr = 8, .saddr = 125, .vdst = 4});
          const uint32_t reg = form == Partial ? 5 : 4;
          const uint32_t code[] = {form == NoActive ? 0xbefe0480u : 0xbefe0483u,
                                   form == Nop ? nop : flat[0],
                                   form == Nop ? nop : flat[1],
                                   wait,
                                   0xbefe0480u + (1u << lane),
                                   0x7e0c0300u + reg,
                                   both,
                                   0xbf810000};
          const auto kernel = f.write_kernel(0x1000, code, std::size(code), 12, 1024);
          using namespace amdhsa;
          auto props =
              f.mem->read32(kernel + offsetof(kernel_descriptor_t, kernel_code_properties));
          RJ_AMDHSA_BITS_SET(props, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, width == 32);
          f.mem->write16(kernel + offsetof(kernel_descriptor_t, kernel_code_properties), props);
          f.mem->write32(scratch + width * 4 + 4, vm_lo);
          f.mem->write32(scratch + width * 8 + 4, vm_hi);
          test::AqlQueue queue(f.mem, f.cp());
          queue.dispatch(kernel, width, width);
          f.run_until_idle();
          ASSERT_TRUE(f.cu()->is_idle());
          const bool pending = form != NoActive && form != Nop && wait != both;
          EXPECT_EQ(f.cu()->memory_wait_diagnostic_count(), pending ? (wait == nop ? 2u : 1u) : 0u);
          EXPECT_EQ(sink.str().find("Race on VGPR v" + std::to_string(reg) + " [") !=
                        std::string::npos,
                    pending);
          ASSERT_EQ(routes->accesses.size(), form == Nop ? 0u : 1u);
          if (form != Nop) {
            const auto &route = routes->accesses.front();
            EXPECT_EQ(route.flat_local_lane_mask,
                      form == NoActive || form == PrivateOnly ? 0u : 1u);
            EXPECT_EQ(route.request_lane_mask, form == NoActive || form == PrivateOnly ? 0u
                                               : form == Valid || form == Partial      ? 3u
                                                                                       : 1u);
          }
        }
      }
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna2FlatNormalizedToLdsRecordsOnlySharedRequests) {
  static constexpr uint64_t shared = uint64_t{1} << 48;
  static constexpr uint64_t priv = uint64_t{2} << 48;

  class Probe final : public ExecutionPlugin {
  public:
    Probe(ComputeUnitCore &cu, bool overlap)
        : ExecutionPlugin("mixed_private_probe"), cu(cu), overlap(overlap) {}
    void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, uint32_t, uint32_t,
                                     std::span<Wavefront *> waves) override {
      for (auto *wf : waves) {
        cu.set_apertures(shared, shared + UINT32_MAX, priv, priv + UINT32_MAX);
        wf->set_apertures(shared, shared + UINT32_MAX, priv, priv + UINT32_MAX,
                          /*override_launch=*/true);
        EXPECT_EQ(wf->scratch_base(), 0u);
        EXPECT_EQ(wf->scratch_lane_size(), 0u);
        EXPECT_EQ(wf->lds_base(), 0u);
        wf->lds().write32(0, 0xfeedbeef);
        wf->lds().write32(64, 0x11111111);
        const auto base = wf->vgpr_alloc().base;
        cu.write_vgpr(base + 8, 0, 0);
        cu.write_vgpr(base + 9, 0, (overlap ? shared : priv) >> 32);
        cu.write_vgpr(base + 8, 1, 64);
        cu.write_vgpr(base + 9, 1, shared >> 32);
        cu.write_vgpr(base + 10, 0, 0x87654321);
        cu.write_vgpr(base + 10, 1, 0x12345678);
      }
    }
    void onAmdgpuAfterExecuteInstruction(uint64_t, const Instruction &, Wavefront &wf) override {
      if (wf.lds_source_read().bytes) {
        read_value = cu.read_vgpr_storage(wf.vgpr_alloc().base + 4, 0);
        lds64 = wf.lds().read32(64);
      }
    }
    ComputeUnitCore &cu;
    bool overlap;
    uint32_t read_value = 0, lds64 = 0;
  };

  for (uint32_t width : {32u, 64u}) {
    for (unsigned trial : {0u, 1u, 2u}) {
      // Trial 0 uses an active, OOB private lane. Trial 1 masks it in EXEC.
      // Trial 2 genuinely writes LDS byte zero as a positive race control.
      SCOPED_TRACE(testing::Message() << width << " lanes, trial " << trial);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(2, "rdna2", width, 106));
      PluginSinkConfig config;
      auto &sink = config.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
      auto probe = std::make_unique<Probe>(*f.cu(), trial == 2);
      auto *observed = probe.get();
      auto memory = std::make_unique<MemoryObservationPlugin>();
      auto *routes = memory.get();
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      ASSERT_TRUE(f.plugin_group_->add(std::move(probe)));
      ASSERT_TRUE(f.plugin_group_->add(std::move(memory)));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();

      const auto flat =
          rdna2::build_flat(rdna2::kFlatStoreDword, {.addr = 8, .data = 10, .saddr = 125});
      const uint32_t code[] = {0x7e000500,
                               0xbf068000,
                               0xbf840005, // Wave 1 branches to the reader.
                               trial == 1 ? 0xbefe0482u : 0xbefe0483u,
                               flat[0],
                               flat[1],
                               0xbf8c0000,
                               0xbf810000,
                               0xbf8e0007,
                               0xbefe0481,
                               0xbefc03ff,
                               0x00020000,
                               0x7e0802fe,
                               0xbf810000};
      const auto kernel = f.write_kernel(0x1000, code, std::size(code), 12, 1024);
      using namespace amdhsa;
      auto props = f.mem->read32(kernel + offsetof(kernel_descriptor_t, kernel_code_properties));
      RJ_AMDHSA_BITS_SET(props, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, width == 32);
      f.mem->write16(kernel + offsetof(kernel_descriptor_t, kernel_code_properties), props);
      test::AqlQueue queue(f.mem, f.cp());
      queue.dispatch(kernel, width * 2, width * 2);
      f.run_until_idle();
      ASSERT_TRUE(f.cu()->is_idle());
      ASSERT_EQ(routes->accesses.size(), 1u);
      const auto &access = routes->accesses.front();
      EXPECT_EQ(access.route, MemoryRoute::LOCAL);
      EXPECT_EQ(access.active_lane_mask, trial == 1 ? 2u : 3u);
      EXPECT_EQ(access.request_lane_mask, trial == 2 ? 3u : 2u);
      EXPECT_EQ(access.flat_local_lane_mask, trial == 2 ? 3u : 2u);
      EXPECT_EQ(observed->read_value, trial == 2 ? 0x87654321u : 0xfeedbeefu);
      EXPECT_EQ(observed->lds64, 0x12345678u);
      EXPECT_TRUE(access.normalized_to_local);
      const bool reported = sink.str().find("Race on LDS byte 0") != std::string::npos;
      EXPECT_EQ(reported, trial == 2) << sink.str();
    }
  }
}

TEST(RaceDetectorPluginTest, InvalidVectorLoadDestinationIsRejectedBeforeWawChecks) {
  auto rejects_without_access = [](MemPipelineTag tag) {
    constexpr uint32_t kVgprsPerWave = 16;
    PluginFixture f;
    f.initialize(/*num_wf_slots=*/1, /*arch=*/"rdna4", /*wavefront_size=*/32,
                 /*sgprs_per_wf=*/128, kVgprsPerWave);
    if (testing::Test::HasFatalFailure())
      return false;
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    EXPECT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    auto *wf = f.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/128, kVgprsPerWave, 32);
    EXPECT_NE(wf, nullptr);
    if (!wf)
      return false;
    wf->set_exec(1u);
    std::array<amdgpu::Wavefront *, 1> waves{wf};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(
        /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/kVgprsPerWave,
        /*physical_sgpr_count=*/128, waves);

    auto state = std::make_unique<VectorMemState>(tag);
    state->elem_size = sizeof(uint32_t);
    state->num_elems = 2;
    state->is_load = true;
    state->wf_size = wf->wf_size();
    state->exec_mask = 1;
    state->lane_mask = 1;
    state->dst_reg_base = wf->vgpr_alloc().base + kVgprsPerWave - 1;
    TestMemoryInstruction load(std::move(state));
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, load, *wf);
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

    f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kVgprsPerWave - 1,
                                           /*lane_mask=*/1);
    return sink.str().find("RACE ") == std::string::npos;
  };

  EXPECT_TRUE(rejects_without_access(GLOBAL_MEM));
  EXPECT_TRUE(rejects_without_access(LOCAL_MEM));
}

TEST(RaceDetectorPluginTest, WideVectorLoadTracksEveryDestinationVgpr) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *wf = f.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/104, waves);

  constexpr uint32_t kDst = 4;
  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->elem_size = sizeof(uint64_t);
  state->num_elems = 1;
  state->is_load = true;
  state->wf_size = wf->wf_size();
  state->exec_mask = 1;
  state->lane_mask = 1;
  state->dst_reg_base = wf->vgpr_alloc().base + kDst;
  TestMemoryInstruction load(std::move(state));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, load, *wf);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kDst + 1,
                                         /*lane_mask=*/1);
  EXPECT_NE(sink.str().find("RACE "), std::string::npos);
}

TEST(RaceDetectorPluginTest, DualOffsetLoadTracksBothDestinationRanges) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *wf = f.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/104, waves);

  constexpr uint32_t kFirstDst = 4;
  constexpr uint32_t kSecondDst = 7;
  auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
  state->elem_size = sizeof(uint32_t);
  state->num_elems = 1;
  state->is_load = true;
  state->wf_size = wf->wf_size();
  state->exec_mask = 1;
  state->lane_mask = 1;
  state->dst_reg_base = wf->vgpr_alloc().base + kFirstDst;
  state->ds2_active = true;
  state->ds2_dst_reg_base = wf->vgpr_alloc().base + kSecondDst;
  TestMemoryInstruction load(std::move(state));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, load, *wf);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kSecondDst,
                                         /*lane_mask=*/1);
  EXPECT_NE(sink.str().find("RACE "), std::string::npos);
}

TEST(RaceDetectorPluginTest, Rdna2LdsAtomicDirectReadRequiresWorkgroupOrdering) {
  class ReadValue : public ExecutionPlugin {
  public:
    explicit ReadValue(ComputeUnitCore &cu) : ExecutionPlugin("direct_read_value"), cu(cu) {}
    void onAmdgpuAfterExecuteInstruction(uint64_t, const Instruction &, Wavefront &wave) override {
      if (wave.lds_source_read().bytes)
        value = cu.read_vgpr_storage(wave.vgpr_alloc().base + 4, 0);
    }
    ComputeUnitCore &cu;
    std::optional<uint32_t> value;
  };
  for (bool returning : {false, true}) {
    for (bool barrier : {false, true}) {
      SCOPED_TRACE(testing::Message() << "returning=" << returning << " barrier=" << barrier);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(2, "rdna2", 64, 106));
      PluginSinkConfig config;
      auto &sink = config.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      auto observer = std::make_unique<ReadValue>(*f.cu());
      auto *read = observer.get();
      ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      const auto ds = rdna2::build_ds(returning ? rdna2::kDsAddRtnU32 : rdna2::kDsAddU32,
                                      {.addr = 0, .data0 = 8, .vdst = 9});
      // Wave 0 adds 18 to LDS byte 64, waits and exits. Wave 1 reads that DWORD
      // after a delay, optionally crossing a barrier before the direct read.
      const uint32_t code[] = {
          0x7e000500, 0xbf068000, 0xbf840007, 0xbefe0481,
          0x7e0002c0, 0x7e100292, ds[0],      ds[1],
          0xbf8c0000, 0xbf810000, 0xbf8e0007, barrier ? 0xbf8a0000u : 0xbf800000u,
          0xbefc03ff, 0x00020040, 0x7e0802fe, 0xbf810000,
      };
      const auto kernel = f.write_kernel(0x1000, code, std::size(code), 12, 1024);
      test::AqlQueue queue(f.mem, f.cp());
      queue.dispatch(kernel, 128, 128);
      f.run_until_idle();
      ASSERT_TRUE(f.cu()->is_idle());
      EXPECT_EQ(read->value, 18u);
      EXPECT_EQ(sink.str().find("Race on LDS byte 64") != std::string::npos, !barrier)
          << sink.str();
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna2LdsAtomicSharesOneWaitWithItsReturnedRegister) {
  constexpr int direct = -1;
  const int add = rdna2::kDsAddU32, returning = rdna2::kDsAddRtnU32;
  struct Trial {
    int first, second;
    bool same_wave, wait, gds;
    uint32_t second_address;
    bool lds_race;
  };
  for (Trial trial : {Trial{returning, direct, true, false, false, 64, true},
                      Trial{returning, direct, true, true, false, 64, false},
                      Trial{returning, direct, false, true, true, 64, false},
                      Trial{direct, returning, false, false, false, 64, true},
                      Trial{direct, returning, true, false, false, 64, false},
                      Trial{add, rdna2::kDsWriteB32, false, true, false, 64, true},
                      Trial{returning, add, false, true, false, 64, false},
                      Trial{add, returning, false, false, false, 64, false},
                      Trial{returning, rdna2::kDsAddRtnU64, false, false, false, 64, true},
                      Trial{returning, add, false, false, false, 66, true},
                      Trial{add, rdna2::kDsWriteB32, false, false, false, 68, false}}) {
    SCOPED_TRACE(testing::Message() << trial.first << " then " << trial.second
                                    << " same=" << trial.same_wave << " wait=" << trial.wait
                                    << " GDS=" << trial.gds << " addr=" << trial.second_address);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(2, "rdna2", 32, 106));
    PluginSinkConfig config;
    auto &sink = config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *detector = plugin.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    std::array<Wavefront *, 2> waves{cu->dispatch_wf(0, 0x100, 106, 256, 32),
                                     cu->dispatch_wf(0, 0x200, 106, 256, 32)};
    for (auto *w : waves) {
      ASSERT_NE(w, nullptr);
      w->set_exec(1);
      w->set_lds_size(1024);
    }
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 512, 212, waves);
    auto dec = Decoder::create(ROCJITSU_CODE_ARCH_RDNA2);
    auto issue = [&](Wavefront &w, int op, uint32_t address, bool gds, uint64_t pc) {
      w.set_m0(gds ? 128 : 0x20040);
      cu->write_vgpr(w.vgpr_alloc().base, 0, address);
      auto words = op == direct
                       ? std::array<uint32_t, 2>{0x7e0802fe, 0}
                       : rdna2::build_ds(op, {.gds = gds, .addr = 0, .data0 = 8, .vdst = 12});
      auto inst = dec->decode_window(words);
      ASSERT_TRUE(inst.succeeded());
      f.plugin_group_->onAmdgpuBeforeExecuteInstruction(pc, *inst.value(), w);
      ASSERT_TRUE(cu->execute_instruction(inst.value().get(), w).succeeded());
      if (op == direct)
        f.plugin_group_->onAmdgpuAfterExecuteInstruction(pc, *inst.value(), w);
      else
        test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.value().release(), w);
    };
    auto &first = *waves[0];
    auto *state = static_cast<RaceWavefrontState *>(first.plugin_state(detector->slot_index()));
    auto &race = *state->race_state;
    ASSERT_NO_FATAL_FAILURE(issue(first, trial.first, 64, trial.gds, 0x100));
    EXPECT_EQ(race.getWaveMemoryEvents().size(), trial.first == direct ? 0u : 1u);
    EXPECT_EQ(race.getVgprMemoryEvents(12).size(), trial.first == returning ? 1u : 0u);
    if (trial.wait) {
      execute_wait(*f.plugin_group_, first, 0xbf8c0000);
      EXPECT_TRUE(race.getWaveMemoryEvents().empty());
      EXPECT_TRUE(race.getVgprMemoryEvents(12).empty());
    }
    auto &second = *waves[trial.same_wave ? 0 : 1];
    ASSERT_NO_FATAL_FAILURE(issue(second, trial.second, trial.second_address, false, 0x200));
    EXPECT_EQ(sink.str().find("Race on LDS byte") != std::string::npos, trial.lds_race)
        << sink.str();
  }
}

TEST(RaceDetectorPluginTest, Rdna2DsChecksAndRecordsOnlyCompleteElementsInWorkgroupLds) {
  struct Case {
    uint16_t store_opcode;
    uint32_t store_offset;
    uint8_t second_offset;
    uint32_t read_offset;
    unsigned stored_bytes;
    bool race;
  };
  for (const Case test : {Case{rdna2::kDsWriteB32, 1024, 0, 1024, 0, false},
                          Case{rdna2::kDsWriteB32, 1020, 0, 1020, 4, true},
                          Case{rdna2::kDsWriteB128, 1016, 0, 1024, 0, false},
                          Case{rdna2::kDsWriteB128, 1008, 0, 1020, 16, true},
                          Case{rdna2::kDsWrite2B32, 1020, 1, 1024, 4, false},
                          Case{rdna2::kDsWrite2B32, 1016, 1, 1020, 8, true}}) {
    SCOPED_TRACE(testing::Message() << test.store_opcode << " store=" << test.store_offset
                                    << " read=" << test.read_offset);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(2, "rdna2", 32, 106));
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *plugin_ptr = plugin.get();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    std::array<Wavefront *, 2> waves{cu->dispatch_wf(0, 0x100, 106, 256, 32),
                                     cu->dispatch_wf(0, 0x200, 106, 256, 32)};
    constexpr uint32_t base = 1024;
    for (auto *wave : waves) {
      ASSERT_NE(wave, nullptr);
      wave->set_exec(1);
      wave->set_lds_base(base);
      wave->set_lds_size(1024);
    }
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 512, 212, waves);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA2);
    ASSERT_NE(decoder, nullptr);
    const auto route = [&](unsigned wave_idx, uint16_t op, uint32_t offset, uint8_t offset1) {
      auto *wave = waves[wave_idx];
      cu->write_vgpr(wave->vgpr_alloc().base, 0, offset);
      const auto words =
          rdna2::build_ds(op, {.offset1 = offset1, .addr = 0, .data0 = 4, .data1 = 8, .vdst = 12});
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      ASSERT_NE(inst, nullptr);
      f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave->pc, *inst, *wave);
      ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
      test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), *wave);
    };
    ASSERT_NO_FATAL_FAILURE(route(0, test.store_opcode, test.store_offset, test.second_offset));
    ASSERT_NO_FATAL_FAILURE(route(1, rdna2::kDsReadB32, test.read_offset, 0));
    auto *state =
        static_cast<RaceWavefrontState *>(waves[0]->plugin_state(plugin_ptr->slot_index()));
    auto &events = state->race_state->getDetector()->events();
    const bool stored = test.stored_bytes != 0;
    ASSERT_EQ(events.totalAllocated(), stored ? 2 : 1);
    if (stored) {
      EXPECT_EQ(events.ldsIntervals(EventId{0}).getTotalBytes(), test.stored_bytes);
      EXPECT_FALSE(events.ldsIntervals(EventId{0}).contains(base + 1024));
    }
    const EventId read{stored ? 1 : 0};
    EXPECT_EQ(events.ldsIntervals(read).getTotalBytes(), test.read_offset < 1024 ? 4u : 0u);
    EXPECT_EQ(events.execMask(read), 1u);
    EXPECT_EQ(events.registers(read).size(), 1u);
    EXPECT_EQ(sink.str().find("RACE ") != std::string::npos, test.race);
  }
}

TEST(RaceDetectorPluginTest, Rdna2SynchronousLdsSourceChecksOnlyTheBytesRead) {
  struct Case {
    uint32_t write_offset;
    uint32_t m0;
    uint64_t exec;
    uint32_t value;
    bool same_wave;
    bool race;
  };
  constexpr uint32_t word = 0x44332211, sentinel = 0xdeadbeef;
  constexpr uint64_t active = 1u << 7;
  for (auto test : {Case{67, 0x20040, active, word, false, true},   // DWORD includes byte 67.
                    Case{65, 0x00040, active, 0x11, false, false},  // BYTE excludes byte 65.
                    Case{65, 0x10040, active, 0x2211, false, true}, // WORD includes byte 65.
                    Case{67, 0x20040, 0, sentinel, false, false},   // No active VALU lane.
                    Case{67, 0x20400, active, 0, false, false},     // Outside the allocation.
                    Case{67, 0x20040, active, word, true, true}}) { // No wait after own DS write.
    SCOPED_TRACE(testing::Message() << test.write_offset << " M0=" << test.m0
                                    << " EXEC=" << test.exec << " same wave=" << test.same_wave);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(2, "rdna2", 32, 106));
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *plugin_ptr = plugin.get();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    std::array<Wavefront *, 2> waves{cu->dispatch_wf(0, 0x100, 106, 256, 32),
                                     cu->dispatch_wf(0, 0x200, 106, 256, 32)};
    constexpr uint32_t base = 1024;
    for (auto *wave : waves) {
      ASSERT_NE(wave, nullptr);
      wave->set_lds_base(base);
      wave->set_lds_size(1024);
    }
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 512, 212, waves);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA2);
    auto &writer = *waves[0];
    auto &reader = *waves[test.same_wave ? 0 : 1];
    writer.lds().write32(base + 64, word);
    writer.lds().write32(base + 1024, word);
    writer.set_exec(1);
    cu->write_vgpr(writer.vgpr_alloc().base, 0, test.write_offset);
    cu->write_vgpr(writer.vgpr_alloc().base + 8, 0, word >> ((test.write_offset & 3) * 8));
    const auto store = rdna2::build_ds(rdna2::kDsWriteB8, {.addr = 0, .data0 = 8});
    std::unique_ptr<Instruction> ds(decode_valid(*decoder, store.data()));
    ASSERT_NE(ds, nullptr);
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(writer.pc, *ds, writer);
    ASSERT_TRUE(cu->execute_instruction(ds.get(), writer).succeeded());
    test::ComputeUnitTestAccess::route_memory_inst(*cu, ds.release(), writer);

    cu->write_vgpr(reader.vgpr_alloc().base + 4, 7, sentinel);
    reader.set_m0(test.m0);
    reader.set_exec(test.exec);
    const uint32_t mov = 0x7e0802fe; // v_mov_b32 v4,src_lds_direct
    std::unique_ptr<Instruction> valu(decode_valid(*decoder, &mov));
    ASSERT_NE(valu, nullptr);
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(reader.pc + 8, *valu, reader);
    ASSERT_TRUE(cu->execute_instruction(valu.get(), reader).succeeded());
    f.plugin_group_->onAmdgpuAfterExecuteInstruction(reader.pc + 8, *valu, reader);
    EXPECT_EQ(cu->read_vgpr_storage(reader.vgpr_alloc().base + 4, 7), test.value);
    EXPECT_EQ(sink.str().find("Race on LDS byte " + std::to_string(base + test.write_offset)) !=
                  std::string::npos,
              test.race);
    // A subsequent register-only MOV at a different PC must not replay this LDS read.
    const std::string reports = sink.str();
    const uint32_t mov_reg = 0x7e080300; // v_mov_b32 v4,v0
    std::unique_ptr<Instruction> reg(decode_valid(*decoder, &mov_reg));
    ASSERT_NE(reg, nullptr);
    reader.set_exec(active);
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(reader.pc + 12, *reg, reader);
    ASSERT_TRUE(cu->execute_instruction(reg.get(), reader).succeeded());
    f.plugin_group_->onAmdgpuAfterExecuteInstruction(reader.pc + 12, *reg, reader);
    EXPECT_EQ(reader.lds_source_read().bytes, 0u);
    EXPECT_EQ(sink.str(), reports);
    auto *state = static_cast<RaceWavefrontState *>(reader.plugin_state(plugin_ptr->slot_index()));
    const bool read_lds = test.exec && (test.m0 & 0xffff) < reader.lds_size();
    EXPECT_EQ(state->race_state->getDetector()->events().totalAllocated(), read_lds ? 2 : 1);
  }
}

TEST(RaceDetectorPluginTest, DirectLdsReadBlocksOnlyOtherWavesUntilBarrier) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(2, "rdna2", 32, 106));
  PluginSinkConfig config;
  StringSink &sink = config.emplace<StringSink>();
  auto plugin = std::make_unique<RaceDetectorPlugin>();
  auto *plugin_ptr = plugin.get();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
  ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  std::array<Wavefront *, 2> waves{cu->dispatch_wf(0, 0x100, 106, 256, 32),
                                   cu->dispatch_wf(0, 0x200, 106, 256, 32)};
  constexpr uint32_t base = 1024;
  for (auto *wave : waves) {
    ASSERT_NE(wave, nullptr);
    wave->set_exec(1);
    wave->set_lds_base(base);
    wave->set_lds_size(1024);
  }
  f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 512, 212, waves);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA2);
  auto &reader = *waves[0];
  auto &other = *waves[1];
  const uint32_t mov = 0x7e0802fe; // v_mov_b32 v4,src_lds_direct
  auto read = [&](uint32_t type, uint64_t pc) {
    reader.set_m0((type << 16) | 64);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &mov));
    ASSERT_NE(inst, nullptr);
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(pc, *inst, reader);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), reader).succeeded());
    f.plugin_group_->onAmdgpuAfterExecuteInstruction(pc, *inst, reader);
  };
  auto write = [&](Wavefront &wave, uint32_t address, uint64_t pc) {
    cu->write_vgpr(wave.vgpr_alloc().base, 0, address);
    const auto words = rdna2::build_ds(rdna2::kDsWriteB8, {.addr = 0, .data0 = 8});
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(pc, *inst, wave);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), wave).succeeded());
    test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), wave);
  };
  auto *state = static_cast<RaceWavefrontState *>(reader.plugin_state(plugin_ptr->slot_index()));
  auto &race = *state->race_state;
  auto &events = race.getDetector()->events();
  for (unsigned repeat = 0; repeat != 64; ++repeat) {
    ASSERT_NO_FATAL_FAILURE(read(0, 0x100));
    ASSERT_NO_FATAL_FAILURE(read(1, 0x104));
    ASSERT_NO_FATAL_FAILURE(read(2, 0x108));
  }
  EXPECT_TRUE(race.getWaveMemoryEvents().empty());
  ASSERT_EQ(race.getBarrierPendingEvents().size(), 3u);
  ASSERT_EQ(events.totalAllocated(), 3);
  for (int i = 0; i != 3; ++i) {
    const EventId id{i};
    EXPECT_EQ(events.status(id), EventStatus::WAVE_COMPLETE);
    EXPECT_TRUE(events.counterObligations(id).empty());
    EXPECT_TRUE(events.registers(id).empty());
    EXPECT_EQ(events.pc(id), 0x100u + 4u * i);
  }
  EXPECT_EQ(events.ldsIntervals(EventId{0}).getTotalBytes(), 1);
  EXPECT_EQ(events.ldsIntervals(EventId{1}).getTotalBytes(), 1);
  EXPECT_EQ(events.ldsIntervals(EventId{2}).getTotalBytes(), 2);
  ASSERT_NO_FATAL_FAILURE(write(reader, 65, 0x10c));
  ASSERT_NO_FATAL_FAILURE(write(other, 68, 0x200));
  EXPECT_EQ(sink.str().find("Race on LDS byte"), std::string::npos);
  ASSERT_NO_FATAL_FAILURE(write(other, 67, 0x204));
  EXPECT_NE(sink.str().find("Race on LDS byte " + std::to_string(base + 67)), std::string::npos);
  const std::string reports = sink.str();
  f.plugin_group_->onAmdgpuBarrierResolved(waves, BarrierKind::Workgroup);
  EXPECT_TRUE(race.getBarrierPendingEvents().empty());
  EXPECT_EQ(events.status(EventId{2}), EventStatus::RETIRED);
  ASSERT_NO_FATAL_FAILURE(write(other, 67, 0x208));
  EXPECT_EQ(sink.str(), reports);
  // The next epoch must claim the same DWORD again.
  ASSERT_NO_FATAL_FAILURE(read(0, 0x110));
  ASSERT_EQ(race.getBarrierPendingEvents().size(), 1u);
  EXPECT_EQ(events.ldsIntervals(race.getBarrierPendingEvents().front()).getTotalBytes(), 1);
}

TEST(RaceDetectorPluginTest, Cdna3AqlTracksNamedAndNumberedSgprAliases) {
  enum Use { Read, Write, VccBranch, ScratchLoad, SdwaVcc, SdwaSgpr };
  struct Case {
    uint8_t producer;
    bool wide;
    uint8_t consumer;
    Use use;
    uint8_t physical;
    bool conflict;
    bool cdna4 = false;
    uint32_t descriptor = 1; // Sixteen physical SGPRs.
  };
  const Case cases[] = {{14, true, 0, SdwaVcc, 14, true},
                        {14, true, 6, SdwaSgpr, 6, false},
                        {14, true, 14, SdwaSgpr, 14, true},
                        {14, true, 106, Read, 14, true},
                        {15, false, 107, Read, 15, true},
                        {15, false, 106, Read, 14, false},
                        {106, true, 15, Read, 15, true},
                        {106, true, 107, Write, 15, true},
                        {14, true, 251, Read, 14, true},
                        {10, true, 102, Read, 10, true},
                        {102, true, 11, Read, 11, true},
                        {11, false, 102, Read, 10, false},
                        {11, false, 103, Write, 11, true},
                        {15, false, 0, VccBranch, 15, true},
                        {10, true, 0, ScratchLoad, 10, true},
                        {14, true, 106, Read, 14, false, true},
                        {106, true, 107, Read, 107, true, false, 12}};
  for (auto c : cases) {
    for (bool wait : {false, true}) {
      SCOPED_TRACE(testing::Message() << "producer=" << +c.producer << " consumer=" << +c.consumer
                                      << " wait=" << wait);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(1, c.cdna4 ? "cdna4" : "cdna3", 64, 104));
      PluginSinkConfig config;
      StringSink &sink = config.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      f.soc->set_plugin_group(f.plugin_group_);
      const auto load = cdna3::build_smem(c.wide ? cdna3::kSLoadDwordx2 : cdna3::kSLoadDword,
                                          {.sdata = c.producer, .imm = 1});
      f.mem->write32(0x4000, 0x9000); // Valid scratch base for the implicit use.
      f.mem->write32(0x4004, 0);
      std::vector<uint32_t> code{0xbe8000ffu, 0x4000u, 0xbe810080u,
                                 load[0],     load[1], wait ? 0xbf8c0000u : S_NOP};
      if (c.use == SdwaVcc || c.use == SdwaSgpr) {
        // v_cmp_eq_u32_sdwa vcc/sdst, v0, v1. SD selects the explicit SGPR pair.
        const uint32_t sdst = c.use == SdwaSgpr ? 0x8000u | (uint32_t{c.consumer} << 8) : 0;
        code.insert(code.end(), {0x7d9402f9u, 0x06060000u | sdst});
      } else if (c.use == ScratchLoad) {
        const auto scratch =
            cdna3::build_flat(cdna3::kFlatLoadDword, {.seg = 1, .saddr = 127, .vdst = 0});
        code.insert(code.end(), scratch.begin(), scratch.end());
      } else if (c.use == VccBranch) {
        code.push_back(0xbf860000u); // s_cbranch_vccz with identical successors.
      } else {
        code.push_back(cdna3::build_sop1(
            cdna3::kSMovB32, {.ssrc0 = static_cast<uint8_t>(c.use == Write ? 128 : c.consumer),
                              .sdst = static_cast<uint8_t>(c.use == Write ? c.consumer : 4)})[0]);
      }
      code.push_back(S_ENDPGM);
      f.run_kernel(code.data(), code.size(), 64, 64, c.descriptor);
      const auto message = sink.str();
      const bool expected = !wait && c.conflict;
      EXPECT_EQ(message.find("RACE ") != std::string::npos, expected) << message;
      if (expected) {
        EXPECT_NE(message.find(std::format(
                      "type=SGPR access={} reg={}",
                      (c.use == Write || c.use == SdwaVcc || c.use == SdwaSgpr) ? "write" : "read",
                      c.physical)),
                  std::string::npos)
            << message;
      }
    }
  }
}

TEST(RaceDetectorPluginTest, Cdna3ClipsLdsRacesAndIntervalsToTheWorkgroup) {
  auto ds = [](uint16_t op, uint8_t first = 0, uint8_t second = 0) {
    return cdna3::build_ds(
        op, {.offset0 = first, .offset1 = second, .data0 = 2, .data1 = 4, .vdst = 8});
  };
  const auto flat = cdna3::build_flat(cdna3::kFlatLoadDwordx2, {.saddr = 127, .vdst = 8});
  struct Case {
    std::array<uint32_t, 2> words;
    uint32_t offset;
    unsigned bytes;
    bool secondary;
    bool probe_load;
    bool race_at_508;
  };
  const Case cases[] = {{ds(cdna3::kDsWriteB32), 508, 4, false, true, true},
                        {ds(cdna3::kDsWriteB32), 512, 0, false, true, false},
                        {ds(cdna3::kDsRead2B32, 0, 1), 508, 4, true, false, true},
                        {ds(cdna3::kDsWrite2B32, 1, 0), 508, 4, true, true, true},
                        {ds(cdna3::kDsRead2B32, 0, 1), 504, 8, true, false, true},
                        {ds(cdna3::kDsWrxchg2RtnB32), 508, 4, false, false, true},
                        {ds(cdna3::kDsAddRtnU64), 508, 0, false, false, false},
                        {ds(cdna3::kDsAddRtnU64), 504, 8, false, false, true},
                        {ds(cdna3::kDsAddU32), 512, 0, false, true, false},
                        {flat, 508, 4, false, false, true},
                        {flat, 512, 0, false, false, false}};
  for (auto c : cases) {
    SCOPED_TRACE(testing::Message() << std::hex << c.words[0] << " offset=" << c.offset);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(2, "cdna3", 64, 104));
    PluginSinkConfig config;
    StringSink &sink = config.emplace<StringSink>();
    auto detector = std::make_unique<RaceDetectorPlugin>();
    auto *race = detector.get();
    auto capture = std::make_unique<MemoryObservationPlugin>();
    auto *observer = capture.get();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    ASSERT_TRUE(f.plugin_group_->add(std::move(detector)));
    ASSERT_TRUE(f.plugin_group_->add(std::move(capture)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    constexpr uint64_t aperture = uint64_t{1} << 48;
    cu->set_apertures(aperture, aperture + 0xffffu, 0, 0);
    auto *first = cu->dispatch_wf(0, 0x100, 104, 256, 64);
    auto *second = cu->dispatch_wf(0, 0x200, 104, 256, 64);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    std::array<amdgpu::Wavefront *, 2> waves{first, second};
    cu->allocate_lds(512);
    const uint32_t base = cu->allocate_lds(512);
    for (auto *wave : waves) {
      wave->set_lds_base(base);
      wave->set_lds_size(512);
      wave->set_exec(1);
    }
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 512, 208, waves);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
    auto issue = [&](const std::array<uint32_t, 2> &words, amdgpu::Wavefront &wf) {
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      ASSERT_NE(inst, nullptr);
      f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf.pc, *inst, wf);
      ASSERT_TRUE(cu->execute_instruction(inst.get(), wf).succeeded());
      test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), wf);
    };
    cu->write_vgpr(first->vgpr_alloc().base, 0, c.offset);
    if (c.words == flat) {
      first->set_exec(3);
      cu->write_vgpr(first->vgpr_alloc().base + 1, 0, aperture >> 32);
      cu->write_vgpr(first->vgpr_alloc().base, 1, 0x2000);
    }
    ASSERT_NO_FATAL_FAILURE(issue(c.words, *first));
    ASSERT_EQ(observer->accesses.size(), 1u);
    EXPECT_EQ(!observer->accesses[0].secondary_addresses.empty(), c.secondary);
    auto *state = static_cast<RaceWavefrontState *>(first->plugin_state(race->slot_index()));
    auto &events = state->race_state->getDetector()->events();
    const bool mixed = c.words == flat;
    ASSERT_EQ(events.totalAllocated(), mixed ? 2 : 1);
    const EventId local{int(mixed)};
    EXPECT_EQ(events.ldsIntervals(local).getTotalBytes(), c.bytes);
    EXPECT_FALSE(events.ldsIntervals(local).contains(base + 512));
    if (mixed) {
      EXPECT_EQ(events.execMask(EventId{0}), 2u);
      EXPECT_EQ(events.execMask(local), 1u);
    }
    for (uint32_t offset : {512u, 508u}) {
      cu->write_vgpr(second->vgpr_alloc().base, 0, offset);
      const uint16_t probe = c.probe_load ? cdna3::kDsReadB32 : cdna3::kDsWriteB32;
      ASSERT_NO_FATAL_FAILURE(issue(cdna3::build_ds(probe, {.data0 = 2, .vdst = 8}), *second));
      EXPECT_EQ(sink.str().find("RACE ") != std::string::npos, offset == 508 && c.race_at_508);
    }
  }
}

TEST(RaceDetectorPluginTest, DualOffsetStoreTracksSecondLdsRange) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/2));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *writer = f.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  auto *reader = f.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(writer, nullptr);
  ASSERT_NE(reader, nullptr);
  writer->set_lds_size(256);
  reader->set_lds_size(256);
  writer->set_exec(1u);
  reader->set_exec(1u);
  std::array<amdgpu::Wavefront *, 2> waves{writer, reader};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/512,
      /*physical_sgpr_count=*/208, waves);

  auto store_state = std::make_unique<VectorMemState>(LOCAL_MEM);
  store_state->elem_size = sizeof(uint32_t);
  store_state->num_elems = 1;
  store_state->is_load = false;
  store_state->wf_size = writer->wf_size();
  store_state->exec_mask = 1;
  store_state->lane_mask = 1;
  store_state->per_lane_addr[0] = 0;
  store_state->ds2_active = true;
  store_state->ds2_per_lane_addr[0] = 16;
  TestMemoryInstruction store(std::move(store_state));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(writer->pc, store, *writer);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, store, *writer);

  auto load_state = std::make_unique<VectorMemState>(LOCAL_MEM);
  load_state->elem_size = sizeof(uint32_t);
  load_state->num_elems = 1;
  load_state->is_load = true;
  load_state->wf_size = reader->wf_size();
  load_state->exec_mask = 1;
  load_state->lane_mask = 1;
  load_state->dst_reg_base = reader->vgpr_alloc().base;
  load_state->per_lane_addr[0] = 16;
  TestMemoryInstruction load(std::move(load_state));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(reader->pc, load, *reader);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *reader);

  EXPECT_NE(sink.str().find("RACE "), std::string::npos);
}

TEST(RaceDetectorPluginTest, Cdna1Src2TracksValidSourceReadsAndAtomicDestinations) {
  constexpr auto src2 = cdna1::kDsAddSrc2U32;
  constexpr auto exchange = cdna1::kDsWrxchg2RtnB32;
  struct Access {
    uint16_t opcode;
    uint32_t address;
    uint8_t offset = 0;
    uint8_t offset1 = 0;
  };
  const Access update{src2, 4, 1}; // update byte 4 using byte 8
  struct Case {
    Access other;
    bool race;
    bool barrier = false;
  };
  const Case cases[] = {
      {{cdna1::kDsReadB32, 4}, true},
      {{cdna1::kDsWriteB32, 4}, true},
      {{cdna1::kDsReadB32, 8}, false},
      {{cdna1::kDsWriteB32, 8}, true},
      {{cdna1::kDsAddU32, 4}, false},
      {{cdna1::kDsAddRtnU32, 4}, false},
      {{cdna1::kDsAddU32, 8}, true},
      {{cdna1::kDsAddRtnU32, 8}, true},
      // Each returned value comes from an atomic exchange. Exercise the
      // overlap in each address slot, in both issue orders.
      {{exchange, 4, 0, 2}, false},
      {{exchange, 4, 2, 0}, false},
      {{exchange, 8, 0, 2}, true},
      {{exchange, 8, 2, 0}, true},
      {{src2, 4, 2}, false},
      {{src2, 12, 0}, false},
      {{src2, 8, 1}, true},
      {{cdna1::kDsWriteB32, 8}, false, true},
  };
  auto run = [&](Access first, Access second, bool barrier, bool race, uint32_t conflictByte,
                 bool sameWave = false) {
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(2, "cdna1"));
    PluginSinkConfig sink_config;
    auto &sink = sink_config.emplace<StringSink>();
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *race_plugin = plugin.get();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    std::array waves{cu->dispatch_wf(0, 0x1000, 104, 256), cu->dispatch_wf(0, 0x2000, 104, 256)};
    for (auto *wave : waves) {
      ASSERT_NE(wave, nullptr);
      wave->set_m0(~0u);
      wave->set_lds_size(256);
      wave->set_exec(1);
    }
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 512, 208, waves);
    auto *state =
        static_cast<RaceWavefrontState *>(waves[0]->plugin_state(race_plugin->slot_index()));
    ASSERT_NE(state, nullptr);
    auto decoder = Decoder::create(cu->arch());
    auto issue = [&](unsigned waveIndex, Access op) {
      auto &wave = *waves[waveIndex];
      cu->write_vgpr(wave.vgpr_alloc().base, 0, op.address);
      cu->write_vgpr(wave.vgpr_alloc().base + 1, 0, 7);
      auto words = cdna1::build_ds(
          op.opcode,
          {.offset0 = op.offset, .offset1 = op.offset1, .addr = 0, .data0 = 1, .vdst = 2});
      auto decoded = decoder->decode_window(words);
      ASSERT_TRUE(decoded.succeeded());
      auto &inst = decoded.value();
      f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave.pc, *inst, wave);
      ASSERT_TRUE(cu->execute_instruction(inst.get(), wave).succeeded());
      test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), wave);
      const auto &events = state->race_state->getDetector()->events();
      const EventId event{static_cast<int>(events.totalAllocated()) - 1};
      ASSERT_TRUE(events.contains(event));
      if (op.opcode == exchange) {
        EXPECT_TRUE(events.ldsAtomic(event));
        const auto registers = events.registers(event);
        ASSERT_EQ(registers.size(), 2u);
        EXPECT_EQ(registers[0], 2u);
        EXPECT_EQ(registers[1], 3u);
      }
      if (op.opcode == cdna1::kDsWrite2B32) {
        const auto &intervals = events.ldsIntervals(event);
        EXPECT_TRUE(intervals.contains(252));
        EXPECT_FALSE(intervals.contains(256));
      }
    };
    ASSERT_NO_FATAL_FAILURE(issue(0, first));
    EXPECT_EQ(state->race_state->getDetector()->events().totalAllocated(), 1);
    if (barrier) {
      execute_wait(*f.plugin_group_, *waves[0], legacy_wait_word(ROCJITSU_CODE_ARCH_CDNA1, 0, 0));
      f.plugin_group_->onAmdgpuBarrierResolved(waves, BarrierKind::Workgroup);
    }
    ASSERT_NO_FATAL_FAILURE(issue(sameWave ? 0 : 1, second));
    EXPECT_EQ(state->race_state->getDetector()->events().totalAllocated(), 2);
    EXPECT_EQ(sink.str().find("RACE ") != std::string::npos, race) << sink.str();
    if (race) {
      EXPECT_NE(sink.str().find("Race on LDS byte " + std::to_string(conflictByte) + " ["),
                std::string::npos)
          << sink.str();
    }
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(testing::Message() << test.other.opcode << ',' << test.other.address << ','
                                    << unsigned(test.other.offset) << ','
                                    << unsigned(test.other.offset1) << ',' << test.barrier);
    run(update, test.other, test.barrier, test.race, test.other.address);
    run(test.other, update, test.barrier, test.race, test.other.address);
  }
  // The same native LDS stream orders the store before the SRC2 source read.
  run({cdna1::kDsWriteB32, 8}, update, false, false, 8, /*sameWave=*/true);
  const Access dualStore{cdna1::kDsWrite2B32, 252, 0, 1};
  for (auto address : {252u, 256u}) {
    const Access load{cdna1::kDsReadB32, address};
    const Access store{cdna1::kDsWriteB32, address};
    const bool inRange = address == 252;
    run(store, load, false, inRange, address);
    run(load, store, false, inRange, address);
    run(dualStore, load, false, inRange, address);
    run(load, dualStore, false, inRange, address);
  }
  for (const Access dropped : {Access{src2, 256, 1}, Access{src2, 252, 1}}) {
    // An invalid destination suppresses the RMW. An invalid source contributes zero:
    // the valid destination still conflicts, while neither case reads byte zero.
    for (unsigned addr : {0u, 252u}) {
      const bool writes = dropped.address == 252 && addr == 252;
      run({cdna1::kDsReadB32, addr}, dropped, false, writes, addr);
      run(dropped, {cdna1::kDsWriteB32, addr}, false, writes, addr);
    }
  }
}

TEST(RaceDetectorPluginTest, DualOffsetAccessValidatesSecondLdsRange) {
  const auto second_range_reports_race = [](bool dual_is_load) {
    PluginFixture f;
    f.initialize(/*num_wf_slots=*/2);
    if (testing::Test::HasFatalFailure())
      return false;
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    EXPECT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    auto *first = f.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
    auto *second = f.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
    EXPECT_NE(first, nullptr);
    EXPECT_NE(second, nullptr);
    if (!first || !second)
      return false;
    first->set_lds_size(256);
    second->set_lds_size(256);
    first->set_exec(1u);
    second->set_exec(1u);
    std::array<amdgpu::Wavefront *, 2> waves{first, second};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(
        /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/512,
        /*physical_sgpr_count=*/208, waves);

    auto first_state = std::make_unique<VectorMemState>(LOCAL_MEM);
    first_state->elem_size = sizeof(uint32_t);
    first_state->num_elems = 1;
    first_state->is_load = !dual_is_load;
    first_state->wf_size = first->wf_size();
    first_state->exec_mask = 1;
    first_state->lane_mask = 1;
    first_state->dst_reg_base = first->vgpr_alloc().base;
    first_state->per_lane_addr[0] = 16;
    TestMemoryInstruction first_access(std::move(first_state));
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(first->pc, first_access, *first);
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, first_access, *first);

    auto dual_state = std::make_unique<VectorMemState>(LOCAL_MEM);
    dual_state->elem_size = sizeof(uint32_t);
    dual_state->num_elems = 1;
    dual_state->is_load = dual_is_load;
    dual_state->wf_size = second->wf_size();
    dual_state->exec_mask = 1;
    dual_state->lane_mask = 1;
    dual_state->dst_reg_base = second->vgpr_alloc().base;
    dual_state->ds2_active = true;
    dual_state->ds2_dst_reg_base = second->vgpr_alloc().base + 1;
    dual_state->per_lane_addr[0] = 0;
    dual_state->ds2_per_lane_addr[0] = 16;
    TestMemoryInstruction dual_access(std::move(dual_state));
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(second->pc, dual_access, *second);
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, dual_access, *second);

    return sink.str().find("RACE ") != std::string::npos;
  };

  EXPECT_TRUE(second_range_reports_race(/*dual_is_load=*/true));
  EXPECT_TRUE(second_range_reports_race(/*dual_is_load=*/false));
}

TEST(RaceDetectorPluginTest, LocalMemoryUsesEffectiveIssueMask) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1, /*arch=*/"cdna5", /*wavefront_size=*/32,
                                       /*sgprs_per_wf=*/128));
  PluginSinkConfig sink_config;
  sink_config.emplace<StringSink>();
  auto plugin = std::make_unique<RaceDetectorPlugin>();
  auto *plugin_ptr = plugin.get();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/128, /*vgprs=*/256, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0);
  wf->set_lds_base(cu->allocate_lds(512));
  wf->set_lds_size(512);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/128, waves);

  constexpr uint32_t kAddressVgpr = 4;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
    cu->write_vgpr(wf->vgpr_alloc().base + kAddressVgpr, lane, lane * sizeof(uint64_t));
  const auto words = cdna5::build_vds(cdna5::kDsLoadTr8B64Vds, {.addr = kAddressVgpr, .vdst = 0});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> load(decode_valid(*decoder, words.data()));
  ASSERT_NE(load, nullptr);
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, *load, *wf);
  ASSERT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
  const auto *state = load->data_as<VectorMemState>();
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->exec_mask, 0xFFFF'FFFFu);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, *load, *wf);

  auto *plugin_state =
      static_cast<RaceWavefrontState *>(wf->plugin_state(plugin_ptr->slot_index()));
  ASSERT_NE(plugin_state, nullptr);
  auto &events = plugin_state->race_state->getDetector()->events();
  ASSERT_EQ(events.totalAllocated(), 1);
  const EventId event{events.totalAllocated() - 1};
  EXPECT_EQ(events.execMask(event), 0xFFFF'FFFFu);
}

TEST(RaceDetectorPluginTest, MixedCounterClassesUseUnorderedEventOrdering) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  PluginSinkConfig sink_config;
  sink_config.emplace<StringSink>();
  auto plugin = std::make_unique<RaceDetectorPlugin>();
  auto *plugin_ptr = plugin.get();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *wf = f.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/104, waves);

  auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
  state->elem_size = sizeof(uint32_t);
  state->num_elems = 1;
  state->is_load = true;
  state->wf_size = wf->wf_size();
  state->exec_mask = 1;
  state->lane_mask = 1;
  state->dst_reg_base = wf->vgpr_alloc().base;
  state->per_lane_addr[0] = 0;
  state->wait_counter_type = WaitCounterType::LGKMCNT;
  TestMemoryInstruction load(std::move(state),
                             {{WaitCounterType::VMCNT, MemoryCompletionClass::VMEM},
                              {WaitCounterType::LGKMCNT, MemoryCompletionClass::LDS}});
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, load, *wf);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

  auto *plugin_state =
      static_cast<RaceWavefrontState *>(wf->plugin_state(plugin_ptr->slot_index()));
  ASSERT_NE(plugin_state, nullptr);
  auto &events = plugin_state->race_state->getDetector()->events();
  ASSERT_EQ(events.totalAllocated(), 1);
  const EventId event{events.totalAllocated() - 1};
  EXPECT_EQ(events.memoryOrder(event), MemoryOrderClass::UNORDERED);
}

TEST(RaceDetectorPluginTest, CdnaWholeDsStoreTracksOnlyWrittenLdsBytes) {
  for (auto arch : {"cdna2", "cdna3"}) {
    for (bool wide : {false, true}) {
      SCOPED_TRACE(testing::Message() << arch << " wide=" << wide);
      PluginFixture fixture;
      ASSERT_NO_FATAL_FAILURE(fixture.initialize(2, arch));
      PluginSinkConfig config;
      auto &sink = config.emplace<StringSink>();
      auto plugin = std::make_unique<RaceDetectorPlugin>();
      auto *race = plugin.get();
      fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
      ASSERT_TRUE(fixture.plugin_group_->add(std::move(plugin)));
      fixture.soc->set_plugin_group(fixture.plugin_group_);
      fixture.plugin_group_->onInit();

      auto &cu = *fixture.cu();
      std::array waves{cu.dispatch_wf(0, 0x1000, 104, 16), cu.dispatch_wf(0, 0x2000, 104, 16)};
      constexpr uint32_t size = 64 * 1024;
      for (auto *wave : waves) {
        ASSERT_NE(wave, nullptr);
        wave->set_exec(1);
        wave->set_lds_size(size);
        wave->set_m0(UINT32_MAX);
        cu.write_vgpr(wave->vgpr_alloc().base, 0, size - 4);
      }
      auto &[writer, reader] = waves;
      writer->lds().write32(size - 4, 41);
      cu.write_vgpr(writer->vgpr_alloc().base + 4, 0, 42);
      fixture.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 208, waves);
      auto decoder = Decoder::create(cu.arch());
      auto issue = [&](Wavefront &wave, const auto &words) {
        auto inst = decoder->decode_window(words);
        ASSERT_TRUE(inst.succeeded());
        fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave.pc, *inst.value(), wave);
        ASSERT_TRUE(cu.execute_instruction(inst.value().get(), wave).succeeded());
        ASSERT_EQ(test::ComputeUnitTestAccess::route_memory_inst(cu, inst.value().release(), wave),
                  VmAccessOutcome::Complete);
      };
      ASSERT_NO_FATAL_FAILURE(
          issue(*writer, cdna2::build_ds(wide ? cdna2::kDsWriteB128Ds : cdna2::kDsWriteB32Ds,
                                         {.addr = 0, .data0 = 4})));
      auto *state = static_cast<RaceWavefrontState *>(writer->plugin_state(race->slot_index()));
      ASSERT_NE(state, nullptr);
      const auto &pending = state->race_state->getWaveMemoryEvents();
      ASSERT_EQ(pending.size(), 1u);
      EXPECT_EQ(
          state->race_state->getDetector()->events().ldsIntervals(pending.front()).getTotalBytes(),
          wide ? 0u : 4u);
      ASSERT_NO_FATAL_FAILURE(
          issue(*reader, cdna2::build_ds(cdna2::kDsReadB32Ds, {.addr = 0, .vdst = 8})));
      EXPECT_EQ(cu.read_vgpr_storage(reader->vgpr_alloc().base + 8, 0), wide ? 41u : 42u);
      EXPECT_EQ(sink.str().find("type=LDS access=read") != std::string::npos, !wide) << sink.str();
    }
  }
}

TEST(RaceDetectorPluginTest, LdsRoutedFlatStoreDwordx2TracksTrailingDword) {
  auto trailing_access_reports_race = [](bool flat_store_first) {
    PluginFixture f;
    f.initialize(/*num_wf_slots=*/2);
    if (testing::Test::HasFatalFailure())
      return false;
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    EXPECT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    constexpr uint64_t kSharedBase = 0x0001'0000'0000'0000ULL;
    auto *cu = f.cu();
    cu->set_apertures(kSharedBase, kSharedBase + 0xFFFF, /*private_base=*/0,
                      /*private_limit=*/0);
    auto *writer = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
    auto *reader = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
    EXPECT_NE(writer, nullptr);
    EXPECT_NE(reader, nullptr);
    if (!writer || !reader)
      return false;
    const auto lds_base = cu->allocate_lds(8);
    writer->set_lds_base(lds_base);
    writer->set_lds_size(8);
    reader->set_lds_base(lds_base);
    reader->set_lds_size(8);
    writer->set_exec(1u);
    reader->set_exec(1u);
    EXPECT_EQ(writer->lds_base(), reader->lds_base());
    std::array<amdgpu::Wavefront *, 2> waves{writer, reader};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(
        /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/512,
        /*physical_sgpr_count=*/208, waves);

    auto route_flat_store = [&] {
      const uint32_t vb = writer->vgpr_alloc().base;
      cu->write_vgpr(vb, /*lane=*/0, static_cast<uint32_t>(kSharedBase));
      cu->write_vgpr(vb + 1, /*lane=*/0, static_cast<uint32_t>(kSharedBase >> 32));
      cu->write_vgpr(vb + 2, /*lane=*/0, 0x11112222u);
      cu->write_vgpr(vb + 3, /*lane=*/0, 0x33334444u);

      const auto words = cdna4::build_flat(cdna4::kFlatStoreDwordx2Flat,
                                           {.seg = 0, .addr = 0, .data = 2, .saddr = 0x7F});
      auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
      EXPECT_NE(decoder, nullptr);
      if (!decoder)
        return false;
      std::unique_ptr<Instruction> store(decode_valid(*decoder, words.data()));
      EXPECT_NE(store, nullptr);
      if (!store)
        return false;
      f.plugin_group_->onAmdgpuBeforeExecuteInstruction(writer->pc, *store, *writer);
      EXPECT_TRUE(cu->execute_instruction(store.get(), *writer).succeeded());
      EXPECT_NE(store->data(), nullptr);
      if (!store->data())
        return false;
      EXPECT_EQ(store->data()->tag(), GLOBAL_MEM);
      test::ComputeUnitTestAccess::route_memory_inst(*cu, store.release(), *writer);
      return true;
    };

    auto route_trailing_load = [&] {
      auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
      state->elem_size = sizeof(uint32_t);
      state->num_elems = 1;
      state->is_load = true;
      state->wf_size = reader->wf_size();
      state->exec_mask = 1;
      state->lane_mask = 1;
      state->dst_reg_base = reader->vgpr_alloc().base;
      state->per_lane_addr[0] = reader->lds_base() + sizeof(uint32_t);
      TestMemoryInstruction load(std::move(state));
      f.plugin_group_->onAmdgpuBeforeExecuteInstruction(reader->pc, load, *reader);
      f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *reader);
    };

    if (flat_store_first) {
      if (!route_flat_store())
        return false;
      route_trailing_load();
    } else {
      route_trailing_load();
      if (!route_flat_store())
        return false;
    }
    return sink.str().find("RACE ") != std::string::npos;
  };

  EXPECT_TRUE(trailing_access_reports_race(/*flat_store_first=*/true));
  EXPECT_TRUE(trailing_access_reports_race(/*flat_store_first=*/false));
}

TEST(RaceDetectorPluginTest, MixedFlatChecksAndRecordsOnlySharedLaneIntervals) {
  for (uint32_t local_lane : {0u, 1u}) {
    for (bool load : {false, true}) {
      for (bool flat_first : {false, true}) {
        for (uint32_t probe : {0u, 1u, 2u}) {
          SCOPED_TRACE(testing::Message() << "local_lane=" << local_lane << " load=" << load
                                          << " flat_first=" << flat_first << " probe=" << probe);
          PluginFixture f;
          ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/2, /*arch=*/"cdna5",
                                               /*wavefront_size=*/32,
                                               /*sgprs_per_wf=*/128));
          PluginSinkConfig sink_config;
          StringSink &sink = sink_config.emplace<StringSink>();
          auto plugin = std::make_unique<RaceDetectorPlugin>();
          auto *plugin_ptr = plugin.get();
          f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
          ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
          f.soc->set_plugin_group(f.plugin_group_);
          f.plugin_group_->onInit();
          auto *cu = f.cu();
          constexpr uint64_t shared_base = 0x0001'0000'0000'0000ULL;
          cu->set_apertures(shared_base, shared_base + 0xffff, 0, 0);
          auto *flat_wave = cu->dispatch_wf(0, 0x100, 128, 256, 32);
          auto *ds_wave = cu->dispatch_wf(0, 0x200, 128, 256, 32);
          ASSERT_NE(flat_wave, nullptr);
          ASSERT_NE(ds_wave, nullptr);
          flat_wave->set_exec(0x3);
          ds_wave->set_exec(1);
          cu->allocate_lds(256); // Exercise aperture translation with a nonzero allocation base.
          const uint32_t lds_base = cu->allocate_lds(512);
          flat_wave->set_lds_base(lds_base);
          flat_wave->set_lds_size(512);
          ds_wave->set_lds_base(lds_base);
          ds_wave->set_lds_size(512);
          std::array<amdgpu::Wavefront *, 2> waves{flat_wave, ds_wave};
          f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 512, 256, waves);

          const std::array<uint32_t, 3> probe_addresses{lds_base + 64 + 12, lds_base + 160 + 12,
                                                        lds_base + 256 + 12};
          auto route_ds = [&] {
            auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
            state->elem_size = 4;
            state->num_elems = 1;
            state->is_load = !load;
            state->exec_mask = state->lane_mask = 1;
            state->wf_size = 32;
            state->dst_reg_base = ds_wave->vgpr_alloc().base;
            state->per_lane_addr[0] = probe_addresses[probe];
            TestMemoryInstruction inst(std::move(state));
            f.plugin_group_->onAmdgpuBeforeExecuteInstruction(ds_wave->pc, inst, *ds_wave);
            f.plugin_group_->onAmdgpuMemoryAccessRouted({}, inst, *ds_wave);
          };
          if (!flat_first)
            route_ds();
          const uint32_t vb = flat_wave->vgpr_alloc().base;
          for (uint32_t lane = 0; lane < 3; ++lane) {
            // The global address is also a valid LDS offset numerically. It
            // must not enter the LDS interval set, nor may inactive lane 2.
            const uint64_t addr = lane == local_lane ? shared_base + 64
                                  : lane == 2        ? shared_base + 256
                                                     : lds_base + 160;
            cu->write_vgpr(vb, lane, static_cast<uint32_t>(addr));
            cu->write_vgpr(vb + 1, lane, static_cast<uint32_t>(addr >> 32));
            for (uint32_t reg = 8; reg < 12; ++reg)
              cu->write_vgpr(vb + reg, lane, 0);
          }
          const auto words =
              cdna5::build_vflat(load ? cdna5::kFlatLoadB128Vflat : cdna5::kFlatStoreB128Vflat,
                                 {.saddr = 124,
                                  .vdst = uint8_t(load ? 8 : 0),
                                  .vsrc = uint8_t(load ? 0 : 8),
                                  .vaddr = 0});
          auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
          std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
          ASSERT_NE(inst, nullptr);
          f.plugin_group_->onAmdgpuBeforeExecuteInstruction(flat_wave->pc, *inst, *flat_wave);
          ASSERT_TRUE(cu->execute_instruction(inst.get(), *flat_wave).succeeded());
          test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), *flat_wave);

          auto *plugin_state =
              static_cast<RaceWavefrontState *>(flat_wave->plugin_state(plugin_ptr->slot_index()));
          auto &events = plugin_state->race_state->getDetector()->events();
          const EventId global_event{flat_first ? 0 : 1};
          const EventId flat_event{global_event.value + int(load)};
          ASSERT_EQ(events.totalAllocated(), flat_event.value + 1);
          const auto local_mask = 1u << local_lane;
          EXPECT_EQ(events.execMask(flat_event), load ? local_mask : 3u);
          if (load) {
            EXPECT_EQ(events.execMask(global_event), 3u ^ local_mask);
            EXPECT_TRUE(events.ldsIntervals(global_event).empty());
            EXPECT_EQ(events.registers(global_event).size(), 4u);
          }
          EXPECT_EQ(events.ldsIntervals(flat_event).getTotalBytes(), 16);
          EXPECT_TRUE(events.ldsIntervals(flat_event).contains(probe_addresses[0]));
          EXPECT_FALSE(events.ldsIntervals(flat_event).contains(probe_addresses[1]));
          EXPECT_FALSE(events.ldsIntervals(flat_event).contains(probe_addresses[2]));
          EXPECT_EQ(events.registers(flat_event).size(), load ? 4u : 0u);
          if (flat_first)
            route_ds();
          EXPECT_EQ(sink.str().find("RACE ") != std::string::npos, probe == 0);
        }
      }
    }
  }
}

TEST(RaceDetectorPluginTest, PrivateB64FlatAtomicRejectsAllRoutedLanes) {
  for (bool private_lane : {false, true}) {
    PluginFixture fixture;
    ASSERT_NO_FATAL_FAILURE(fixture.initialize(1, "rdna4", 32, 128));
    fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto observation = std::make_unique<MemoryObservationPlugin>();
    auto *observer = observation.get();
    auto race = std::make_unique<RaceDetectorPlugin>();
    auto *race_plugin = race.get();
    ASSERT_TRUE(fixture.plugin_group_->add(std::move(observation)));
    ASSERT_TRUE(fixture.plugin_group_->add(std::move(race)));
    fixture.soc->set_plugin_group(fixture.plugin_group_);
    fixture.plugin_group_->onInit();
    auto *cu = fixture.cu();
    constexpr uint64_t shared = uint64_t{3} << 32;
    constexpr uint64_t private_base = uint64_t{2} << 32;
    constexpr uint64_t global = 0x4000;
    constexpr uint64_t scratch = 0x8000;
    cu->set_apertures(shared, shared + UINT32_MAX, private_base, private_base + UINT32_MAX);
    auto *wave = cu->dispatch_wf(0, 0x320, 128, 64, 32);
    ASSERT_NE(wave, nullptr);
    ASSERT_EQ(cu->allocate_lds(1024), 0u);
    wave->set_lds_size(1024);
    wave->set_scratch_base(scratch);
    wave->set_scratch_lane_size(256);
    wave->set_exec(7);
    wave->lds().write32(8, 43);
    const std::array<uint32_t, 4> data{37u, 0u, 50u, 0u};
    fixture.mem->load_image(reinterpret_cast<const uint8_t *>(data.data()), sizeof(data), global);
    const std::array<uint64_t, 3> addresses{private_lane ? private_base : global + 8, global,
                                            shared + 8};
    const uint32_t base = wave->vgpr_alloc().base;
    for (unsigned lane = 0; lane < addresses.size(); ++lane) {
      cu->write_vgpr(base + 2, lane, uint32_t(addresses[lane]));
      cu->write_vgpr(base + 3, lane, uint32_t(addresses[lane] >> 32));
      cu->write_vgpr(base + 4, lane, lane + 1);
      cu->write_vgpr(base + 5, lane, 0);
    }
    std::array<Wavefront *, 1> waves{wave};
    fixture.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 64, 128, waves);
    constexpr std::array words{0xec10c07cu, 0x02100000u, 0x00000002u};
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave->pc, *inst, *wave);
    cu->track_memory_wait(*inst, *wave);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
    const auto outcome = test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), *wave);
    EXPECT_EQ(outcome, VmAccessOutcome::Complete);
    EXPECT_EQ(bool(wave->trapsts() & (1u << 8)), private_lane);
    EXPECT_EQ(wave->lds().read32(8), private_lane ? 43u : 46u);
    ASSERT_EQ(observer->accesses.size(), 1u);
    const auto &access = observer->accesses.front();
    auto *state = static_cast<RaceWavefrontState *>(wave->plugin_state(race_plugin->slot_index()));
    ASSERT_NE(state, nullptr);
    const auto &events = state->race_state->getDetector()->events();
    EXPECT_EQ(access.active_lane_mask, 7u);
    EXPECT_EQ(access.valid_lane_mask, private_lane ? 0u : 7u);
    EXPECT_EQ(access.request_lane_mask, private_lane ? 0u : 7u);
    EXPECT_EQ(access.flat_local_lane_mask, private_lane ? 0u : 4u);
    if (private_lane) {
      EXPECT_EQ(events.totalAllocated(), 0);
    } else {
      unsigned lds_bytes = 0;
      for (int id = 0; id < events.totalAllocated(); ++id)
        lds_bytes += events.ldsIntervals(EventId{id}).getTotalBytes();
      EXPECT_EQ(lds_bytes, 8u);
    }
    wave->halt();
  }
}

TEST(RaceDetectorPluginTest, MixedFlatAtLdsBackingEndTracksExecutedElements) {
  for (const auto &[load, dds] :
       {std::pair{false, false}, {true, false}, {false, true}, {true, true}}) {
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/2, /*arch=*/"cdna5",
                                         /*wavefront_size=*/32,
                                         /*sgprs_per_wf=*/128));
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *plugin_ptr = plugin.get();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    constexpr uint64_t shared_base = uint64_t{1} << 48;
    cu->set_apertures(shared_base, shared_base + UINT32_MAX, 0, 0);
    auto *flat_wave = cu->dispatch_wf(0, 0x100, 128, 256, 32);
    auto *probe_wave = cu->dispatch_wf(0, 0x200, 128, 256, 32);
    ASSERT_NE(flat_wave, nullptr);
    ASSERT_NE(probe_wave, nullptr);
    flat_wave->set_exec(dds ? 7 : 3);
    probe_wave->set_exec(1);
    std::array<amdgpu::Wavefront *, 2> waves{flat_wave, probe_wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 512, 256, waves);
    const uint32_t capacity = flat_wave->lds().size_bytes();
    flat_wave->set_lds_size(capacity);
    probe_wave->set_lds_size(capacity);
    const uint32_t vb = flat_wave->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < 2; ++lane) {
      const uint64_t address = lane == 0 ? 0x2000 : shared_base + capacity - 8;
      cu->write_vgpr(vb, lane, static_cast<uint32_t>(address));
      cu->write_vgpr(vb + 1, lane, static_cast<uint32_t>(address >> 32));
      for (uint32_t reg = 8; reg < 12; ++reg)
        cu->write_vgpr(vb + reg, lane, 0x12345678);
    }
    cu->write_vgpr(vb, 2, 0x80000040);
    cu->write_vgpr(vb + 1, 2, uint32_t(shared_base >> 32));
    const auto words = cdna5::build_vflat(
        load ? cdna5::kFlatLoadB128Vflat : cdna5::kFlatStoreB128Vflat,
        {.saddr = 124, .vdst = uint8_t(load ? 8 : 0), .vsrc = uint8_t(load ? 0 : 8), .vaddr = 0});
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(flat_wave->pc, *inst, *flat_wave);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *flat_wave).succeeded());
    EXPECT_EQ(test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), *flat_wave),
              dds ? VmAccessOutcome::Faulted : VmAccessOutcome::Complete);
    auto *state =
        static_cast<RaceWavefrontState *>(flat_wave->plugin_state(plugin_ptr->slot_index()));
    const auto &events = state->race_state->getDetector()->events();
    if (dds) {
      EXPECT_EQ(events.totalAllocated(), 0);
    } else {
      const EventId local_event{int(load)};
      EXPECT_EQ(events.ldsIntervals(local_event).getTotalBytes(), 8);
      EXPECT_TRUE(events.ldsIntervals(local_event).contains(capacity - 1));
      EXPECT_FALSE(events.ldsIntervals(local_event).contains(capacity));
    }
    auto probe = std::make_unique<VectorMemState>(LOCAL_MEM);
    probe->elem_size = 4;
    probe->num_elems = 1;
    probe->is_load = !load;
    probe->exec_mask = probe->lane_mask = 1;
    probe->wf_size = 32;
    probe->dst_reg_base = probe_wave->vgpr_alloc().base;
    probe->per_lane_addr[0] = capacity - 4;
    TestMemoryInstruction probe_inst(std::move(probe));
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(probe_wave->pc, probe_inst, *probe_wave);
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, probe_inst, *probe_wave);
    EXPECT_EQ(sink.str().find("RACE ") != std::string::npos, !dds);
  }
}

TEST(RaceDetectorPluginTest, Cdna4DsFootprintsRespectEachAllocationBound) {
  struct Case {
    uint16_t opcode;
    uint8_t offset0;
    uint8_t offset1;
    bool first_valid;
    bool second_valid;
  };
  for (const auto test : {Case{cdna4::kDsWriteB96Ds, 0, 0, false, false},
                          Case{cdna4::kDsRead2B32Ds, 1, 2, true, false},
                          Case{cdna4::kDsRead2B32Ds, 2, 1, false, true},
                          Case{cdna4::kDsRead2B32Ds, 2, 3, false, false},
                          Case{cdna4::kDsRead2B32Ds, 1, 1, true, false}}) {
    SCOPED_TRACE(::testing::Message()
                 << test.opcode << " " << +test.offset0 << " " << +test.offset1);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, "cdna4", 64, 104));
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *plugin_ptr = plugin.get();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    auto memory_plugin = std::make_unique<MemoryObservationPlugin>();
    auto *observations = memory_plugin.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(memory_plugin)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    auto *wave = cu->dispatch_wf(0, 0x100, 104, 32, 64);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(1);
    cu->allocate_lds(256);
    const uint32_t base = cu->allocate_lds(256);
    wave->set_lds_base(base);
    wave->set_lds_size(256);
    wave->lds().write32(base + 252, 41);
    std::array<Wavefront *, 1> waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 104, waves);
    cu->write_vgpr(wave->vgpr_alloc().base, 0, 248);
    const bool load = test.opcode == cdna4::kDsRead2B32Ds;
    const auto words = cdna4::build_ds(test.opcode, {.offset0 = test.offset0,
                                                     .offset1 = test.offset1,
                                                     .addr = 0,
                                                     .data0 = uint8_t(load ? 0 : 1),
                                                     .vdst = uint8_t(load ? 8 : 0)});
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave->pc, *inst, *wave);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
    test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), *wave);
    ASSERT_EQ(observations->accesses.size(), 1u);
    const auto &observed = observations->accesses.front();
    EXPECT_EQ(observed.ds_suppressed_lanes, test.first_valid ? 0u : 1u);
    EXPECT_EQ(observed.ds2_suppressed_lanes, load && !test.second_valid ? 1u : 0u);
    const auto *state =
        static_cast<RaceWavefrontState *>(wave->plugin_state(plugin_ptr->slot_index()));
    const auto &events = state->race_state->getDetector()->events();
    const EventId event{0};
    ASSERT_EQ(events.totalAllocated(), 1);
    EXPECT_EQ(events.ldsIntervals(event).getTotalBytes(),
              test.first_valid || test.second_valid ? 4 : 0);
    EXPECT_EQ(events.ldsIntervals(event).contains(base + 252),
              test.first_valid || test.second_valid);
    EXPECT_EQ(events.execMask(event), 1u);
    EXPECT_EQ(events.registers(event).size(), load ? 2u : 0u);
    EXPECT_EQ(events.pendingCounterIncrement(event, WaitCounterType::LGKMCNT), 1);
    EXPECT_EQ(wave->lds().read32(base + 252), 41u);
    if (load) {
      EXPECT_EQ(wave->debug_read_vgpr(8, 0), test.first_valid ? 41u : 0u);
      EXPECT_EQ(wave->debug_read_vgpr(9, 0),
                test.second_valid || (test.first_valid && test.offset0 == test.offset1) ? 41u : 0u);
    }
    EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
  }
}

TEST(RaceDetectorPluginTest, FlatInvalidSharedLanesKeepDependenciesWithoutLdsIntervals) {
  constexpr uint64_t shared_base = uint64_t{1} << 48;
  constexpr uint64_t dds_address = shared_base + 0x80000040;
  constexpr uint64_t lds_address = shared_base + 64;
  struct Case {
    std::array<uint64_t, 3> addresses;
    uint64_t exec;
    bool has_global;
    bool has_lds;
  };
  const std::array cases = {Case{{0x2000, dds_address, 0}, 3, true, false},
                            Case{{dds_address, 0x2000, 0}, 3, true, false},
                            Case{{0x2000, dds_address, lds_address}, 7, true, true},
                            Case{{dds_address, 0, 0}, 1, false, false},
                            Case{{0x2000, shared_base + 0x100000, 0}, 3, true, false},
                            Case{{0x2000, shared_base + 0x70000000, 0}, 3, true, false},
                            Case{{shared_base + 0x100000, 0, 0}, 1, false, false},
                            Case{{0x2000, shared_base + 64 * 1024 - 256, 0}, 3, true, false}};
  for (const auto &test : cases) {
    for (bool load : {false, true}) {
      SCOPED_TRACE(testing::Message() << "first_address=" << test.addresses[0]
                                      << " exec=" << test.exec << " load=" << load);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1, /*arch=*/"cdna5",
                                           /*wavefront_size=*/32,
                                           /*sgprs_per_wf=*/128));
      PluginSinkConfig sink_config;
      StringSink &sink = sink_config.emplace<StringSink>();
      auto plugin = std::make_unique<RaceDetectorPlugin>();
      auto *plugin_ptr = plugin.get();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
      ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      auto *cu = f.cu();
      cu->set_apertures(shared_base, shared_base + 0xffffffff, 0, 0);
      auto *wave = cu->dispatch_wf(0, 0x100, 128, 256, 32);
      ASSERT_NE(wave, nullptr);
      wave->set_exec(test.exec);
      cu->allocate_lds(256);
      const uint32_t lds_base = cu->allocate_lds(256);
      wave->set_lds_base(lds_base);
      wave->set_lds_size(256);
      std::array<amdgpu::Wavefront *, 1> waves{wave};
      f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, 128, waves);
      const uint32_t vb = wave->vgpr_alloc().base;
      for (uint32_t lane = 0; lane < test.addresses.size(); ++lane) {
        cu->write_vgpr(vb, lane, static_cast<uint32_t>(test.addresses[lane]));
        cu->write_vgpr(vb + 1, lane, static_cast<uint32_t>(test.addresses[lane] >> 32));
        cu->write_vgpr(vb + 8, lane, 1);
      }
      const auto words = cdna5::build_vflat(
          load ? cdna5::kFlatLoadB32 : cdna5::kFlatStoreB32,
          {.saddr = 124, .vdst = uint8_t(load ? 8 : 0), .vsrc = uint8_t(load ? 0 : 8), .vaddr = 0});
      auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      ASSERT_NE(inst, nullptr);
      f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave->pc, *inst, *wave);
      ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
      test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), *wave);
      auto *state = static_cast<RaceWavefrontState *>(wave->plugin_state(plugin_ptr->slot_index()));
      auto &race = *state->race_state;
      const auto &events = race.getDetector()->events();
      if (std::ranges::find(test.addresses, dds_address) != test.addresses.end()) {
        EXPECT_EQ(events.totalAllocated(), 0);
        continue;
      }
      uint64_t local_lanes = 0;
      for (uint32_t lane = 0; lane < test.addresses.size(); ++lane)
        if (test.addresses[lane] >= shared_base)
          local_lanes |= test.exec & (1u << lane);
      const bool separate = load;
      const EventId global{0}, event{int(separate)};
      EXPECT_EQ((test.exec & ~local_lanes) != 0, test.has_global);
      ASSERT_EQ(events.totalAllocated(), separate ? 2 : 1);
      const auto global_counter = load ? WaitCounterType::LOADCNT : WaitCounterType::STORECNT;
      EXPECT_EQ(events.pendingCounterIncrement(global, global_counter), 1);
      if (separate) {
        EXPECT_TRUE(events.ldsIntervals(global).empty());
        EXPECT_EQ(events.execMask(global), test.exec & ~local_lanes);
      }
      EXPECT_EQ(events.execMask(event), load ? local_lanes : test.exec);
      EXPECT_EQ(events.ldsIntervals(event).getTotalBytes(), test.has_lds ? 4 : 0);
      EXPECT_EQ(events.ldsIntervals(event).contains(lds_base + 64), test.has_lds);
      EXPECT_EQ(events.registers(event).size(), load ? 1u : 0u);
      EXPECT_EQ(events.pendingCounterIncrement(event, WaitCounterType::DSCNT), 1);
      EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
      PendingWaitCount wait;
      wait.add(global_counter, 0);
      race.dispatch(wait);
      EXPECT_EQ(events.status(event), EventStatus::ACTIVE);
      wait.updates.clear();
      wait.add(WaitCounterType::DSCNT, 0);
      race.dispatch(wait);
      // A single-wave workgroup retires a completed event with LDS bytes at once;
      // one without LDS bytes stays wave-complete until the registry trims it.
      EXPECT_EQ(events.status(event),
                test.has_lds ? EventStatus::RETIRED : EventStatus::WAVE_COMPLETE);
    }
  }
}

TEST(RaceDetectorPluginTest, Rdna4PartialLoadWaitRetiresOnlyOldestOrderedEvent) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1, /*arch=*/"rdna4", /*wavefront_size=*/32,
                                       /*sgprs_per_wf=*/128));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/128, /*vgprs=*/256, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/128, waves);

  constexpr uint32_t kAddressVgpr = 4;
  cu->write_vgpr(wf->vgpr_alloc().base + kAddressVgpr, /*lane=*/0, 0);
  cu->write_vgpr(wf->vgpr_alloc().base + kAddressVgpr + 1, /*lane=*/0, 0);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  for (uint32_t dst : {0u, 1u}) {
    const auto words =
        rdna4::build_vglobal(rdna4::kGlobalLoadB32Vglobal, {.saddr = amdgpu::kModernNullSelector,
                                                            .vdst = static_cast<uint8_t>(dst),
                                                            .vaddr = kAddressVgpr});
    std::unique_ptr<Instruction> load(decode_valid(*decoder, words.data()));
    ASSERT_NE(load, nullptr);
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, *load, *wf);
    EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, *load, *wf);
  }

  execute_wait(*f.plugin_group_, *wf, 0xbfc00001u); // s_wait_loadcnt 1.

  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base, /*lane_mask=*/1);
  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + 1, /*lane_mask=*/1);
  EXPECT_NE(sink.str().find("RACE "), std::string::npos);
}

TEST(RaceDetectorPluginTest, Rdna4GenericFlatPartialWaitRetiresOldestEvent) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1, /*arch=*/"rdna4", /*wavefront_size=*/32,
                                       /*sgprs_per_wf=*/128));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  auto plugin = std::make_unique<RaceDetectorPlugin>();
  auto *plugin_ptr = plugin.get();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/128, /*vgprs=*/256, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/128, waves);

  constexpr uint32_t kAddressVgpr = 4;
  cu->write_vgpr(wf->vgpr_alloc().base + kAddressVgpr, /*lane=*/0, 0);
  cu->write_vgpr(wf->vgpr_alloc().base + kAddressVgpr + 1, /*lane=*/0, 0);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  for (uint32_t dst : {0u, 1u}) {
    const auto words =
        rdna4::build_vflat(rdna4::kFlatLoadB32Vflat, {.saddr = amdgpu::kModernNullSelector,
                                                      .vdst = static_cast<uint8_t>(dst),
                                                      .vaddr = kAddressVgpr});
    std::unique_ptr<Instruction> load(decode_valid(*decoder, words.data()));
    ASSERT_NE(load, nullptr);
    const auto *issue = load->amdgpu_memory_issue_info();
    ASSERT_NE(issue, nullptr);
    const auto obligations = issue->counter_obligations();
    ASSERT_EQ(obligations.size(), 2u);
    EXPECT_EQ(obligations[0].wait_counter_type(), WaitCounterType::LOADCNT);
    EXPECT_EQ(obligations[0].completion_class(), MemoryCompletionClass::VMEM);
    EXPECT_EQ(obligations[1].wait_counter_type(), WaitCounterType::DSCNT);
    EXPECT_EQ(obligations[1].completion_class(), MemoryCompletionClass::LDS);

    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, *load, *wf);
    EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, *load, *wf);
  }

  auto *plugin_state =
      static_cast<RaceWavefrontState *>(wf->plugin_state(plugin_ptr->slot_index()));
  ASSERT_NE(plugin_state, nullptr);
  ASSERT_NE(plugin_state->race_state, nullptr);
  const auto &events = plugin_state->race_state->getDetector()->events();
  ASSERT_EQ(events.totalAllocated(), 2);
  EXPECT_EQ(events.memoryOrder(EventId{0}), MemoryOrderClass::UNORDERED);
  EXPECT_EQ(events.memoryOrder(EventId{1}), MemoryOrderClass::UNORDERED);

  execute_wait(*f.plugin_group_, *wf, 0xbfc80101u); // s_wait_loadcnt_dscnt 0x101.

  // Both ordered counter domains prove the oldest mixed-order event complete.
  EXPECT_EQ(events.status(EventId{0}), EventStatus::WAVE_COMPLETE);
  EXPECT_EQ(events.status(EventId{1}), EventStatus::ACTIVE);
  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base, /*lane_mask=*/1);
  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + 1, /*lane_mask=*/1);
  EXPECT_NE(sink.str().find("RACE "), std::string::npos);
}

TEST(RaceDetectorPluginTest, Rdna4GenericFlatStoreRequiresBothCounterWaits) {
  auto verify_wait_order = [](bool storecnt_first) {
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1, /*arch=*/"rdna4",
                                         /*wavefront_size=*/32,
                                         /*sgprs_per_wf=*/128));
    PluginSinkConfig sink_config;
    sink_config.emplace<StringSink>();
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *plugin_ptr = plugin.get();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/128, /*vgprs=*/256, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1u);
    std::array<amdgpu::Wavefront *, 1> waves{wf};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(
        /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
        /*physical_sgpr_count=*/128, waves);

    constexpr uint32_t kAddressVgpr = 0;
    constexpr uint32_t kDataVgpr = 2;
    cu->write_vgpr(wf->vgpr_alloc().base + kAddressVgpr, /*lane=*/0, 0);
    cu->write_vgpr(wf->vgpr_alloc().base + kAddressVgpr + 1, /*lane=*/0, 0);
    cu->write_vgpr(wf->vgpr_alloc().base + kDataVgpr, /*lane=*/0, 0x12345678u);

    const auto words = rdna4::build_vflat(
        rdna4::kFlatStoreB32Vflat,
        {.saddr = amdgpu::kModernNullSelector, .vsrc = kDataVgpr, .vaddr = kAddressVgpr});
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> store(decode_valid(*decoder, words.data()));
    ASSERT_NE(store, nullptr);
    const auto *issue = store->amdgpu_memory_issue_info();
    ASSERT_NE(issue, nullptr);
    const auto obligations = issue->counter_obligations();
    ASSERT_EQ(obligations.size(), 2u);
    EXPECT_EQ(obligations[0].wait_counter_type(), WaitCounterType::STORECNT);
    EXPECT_EQ(obligations[0].completion_class(), MemoryCompletionClass::UNORDERED);
    EXPECT_EQ(obligations[1].wait_counter_type(), WaitCounterType::DSCNT);
    EXPECT_EQ(obligations[1].completion_class(), MemoryCompletionClass::LDS);

    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, *store, *wf);
    EXPECT_TRUE(cu->execute_instruction(store.get(), *wf).succeeded());
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, *store, *wf);

    auto *plugin_state =
        static_cast<RaceWavefrontState *>(wf->plugin_state(plugin_ptr->slot_index()));
    ASSERT_NE(plugin_state, nullptr);
    ASSERT_NE(plugin_state->race_state, nullptr);
    auto &events = plugin_state->race_state->getDetector()->events();
    ASSERT_EQ(events.totalAllocated(), 1);
    const EventId event{events.totalAllocated() - 1};
    ASSERT_TRUE(events.contains(event));
    EXPECT_EQ(events.type(event), MemoryEventType::VGPR_TO_GLOBAL);
    EXPECT_EQ(events.status(event), EventStatus::ACTIVE);

    const auto apply_wait = [&](WaitCounterType counter) {
      if (counter == WaitCounterType::STORECNT) {
        execute_wait(*f.plugin_group_, *wf, 0xbfc10000u); // s_wait_storecnt 0.
      } else {
        execute_wait(*f.plugin_group_, *wf, 0xbfc60000u); // s_wait_dscnt 0.
      }
    };

    const WaitCounterType first =
        storecnt_first ? WaitCounterType::STORECNT : WaitCounterType::DSCNT;
    const WaitCounterType second =
        storecnt_first ? WaitCounterType::DSCNT : WaitCounterType::STORECNT;
    apply_wait(first);
    EXPECT_EQ(events.status(event), EventStatus::ACTIVE);
    EXPECT_FALSE(events.hasPendingWaitCounter(event, first));
    EXPECT_TRUE(events.hasPendingWaitCounter(event, second));
    apply_wait(second);
    EXPECT_EQ(events.status(event), EventStatus::WAVE_COMPLETE);
  };

  verify_wait_order(/*storecnt_first=*/true);
  verify_wait_order(/*storecnt_first=*/false);
}

TEST(RaceDetectorPluginTest, Cdna1TrapCompletesOnlyTheEnteringWavesMemoryObligations) {
  constexpr uint64_t program = 0x1000, handler = 0x2000, data = 0x3000;
  for (bool install_handler : {false, true}) {
    SCOPED_TRACE(install_handler);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(2, "cdna1"));
    PluginSinkConfig config;
    StringSink &sink = config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *race = plugin.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    auto callbacks = std::make_unique<OrderingPlugin>();
    const auto *trace = callbacks.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(callbacks)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    cu->set_functional_quantum(1);
    auto *wf = cu->dispatch_wf(0, program, 104, 256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1);
    auto *peer = cu->dispatch_wf(0, program, 104, 256);
    ASSERT_NE(peer, nullptr);
    peer->set_debug_suspended(true);
    std::array waves{wf, peer};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, 104, waves);

    // A completed wave-local wait must retain this LDS footprint for peers.
    auto *state = static_cast<RaceWavefrontState *>(wf->plugin_state(race->slot_index()));
    const std::array ranges{LdsByteRange{.address = 0, .bytes = 4}};
    const std::array obligations{
        MemoryCounterObligation{WaitCounterType::VMCNT, MemoryOrderClass::UNORDERED}};
    state->race_state->registerLdsRangeEvent(testInstruction(program - 4),
                                             MemoryEventType::GLOBAL_TO_LDS, ranges, obligations,
                                             MemoryOrderClass::UNORDERED);
    auto *peer_state = static_cast<RaceWavefrontState *>(peer->plugin_state(race->slot_index()));
    const std::array peer_ranges{LdsByteRange{.address = 4, .bytes = 4}};
    peer_state->race_state->registerLdsRangeEvent(testInstruction(program - 4),
                                                  MemoryEventType::GLOBAL_TO_LDS, peer_ranges,
                                                  obligations, MemoryOrderClass::UNORDERED);
    const auto &events = state->race_state->getDetector()->events();
    ASSERT_EQ(events.status(EventId{0}), EventStatus::ACTIVE);

    f.mem->write32(data, 123);
    f.mem->write32(program, 0xc0020200u); // s_load_dword s8, s[0:1], 0
    f.mem->write32(program + 4, 0);
    f.mem->write32(program + 8, 0xbf920007u);  // s_trap 7
    f.mem->write32(program + 12, 0xbe890008u); // s_mov_b32 s9, s8
    f.mem->write32(handler, 0xbe890008u);
    if (install_handler) {
      cu->set_trap_handler_resolver(
          [](const Wavefront &) { return ComputeUnitCore::TrapHandlerConfig{handler, 0, false}; });
    }
    cu->write_sgpr(wf->sgpr_alloc().base, data);
    cu->write_sgpr(wf->sgpr_alloc().base + 1, 0);
    cu->step();
    ASSERT_EQ(wf->pc, program + 8);
    if (install_handler) {
      wf->wait_counters().vmcnt = 1;
      cu->step();
      EXPECT_EQ(wf->pc, program + 8);
      wf->release_wait_counter(WaitCounterType::VMCNT);
    }
    const uint64_t consumer = install_handler ? handler : program + 12;
    for (unsigned i = 0; i < 24 && wf->pc != consumer; ++i)
      cu->step();
    ASSERT_EQ(wf->pc, consumer);
    cu->step();
    EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 9), 123u);
    EXPECT_EQ(sink.str().find("type=SGPR access=read reg=8") == std::string::npos, install_handler)
        << sink.str();
    EXPECT_EQ(events.status(EventId{0}),
              install_handler ? EventStatus::WAVE_COMPLETE : EventStatus::ACTIVE);
    EXPECT_EQ(events.status(EventId{1}), EventStatus::ACTIVE);
    const auto trap_callbacks = [&](HookEvent::Kind kind) {
      return std::ranges::count_if(trace->events, [&](const HookEvent &event) {
        return event.kind == kind && event.pc == program + 8 && event.mnemonic == "s_trap";
      });
    };
    EXPECT_EQ(trap_callbacks(HookEvent::BEFORE_INSTRUCTION), 1);
    EXPECT_EQ(trap_callbacks(HookEvent::AFTER_INSTRUCTION), install_handler ? 1 : 0);
    wf->halt();
    peer->halt();
  }
}

TEST(RaceDetectorPluginTest, Cdna2ScalarLoadsFollowPrivilegeAndTakenTrapDrain) {
  struct Case {
    bool ttmp;
    bool handler_installed;
    bool privileged;
  };
  for (const Case test :
       {Case{true, false, false}, Case{true, false, true}, Case{false, false, false},
        Case{false, true, false}, Case{false, true, true}}) {
    SCOPED_TRACE(testing::Message() << "ttmp=" << test.ttmp << " installed="
                                    << test.handler_installed << " privileged=" << test.privileged);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, "cdna2", 64, 102));
    PluginSinkConfig config;
    auto &sink = config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<ExecutionPlugin>("base-only")));
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    const auto *detector = plugin.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    constexpr uint32_t pc = 0x1000, handler = 0x2000, data = 0x4000;
    constexpr uint32_t value = 0x11223344, sentinel = 0x55667788;
    auto *cu = f.cu();
    if (test.handler_installed)
      cu->set_trap_handler_resolver(
          [](const Wavefront &) { return ComputeUnitCore::TrapHandlerConfig{handler, 0, false}; });
    auto *wave = cu->dispatch_wf(0, pc, 32, 32);
    ASSERT_NE(wave, nullptr);
    std::array waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 32, waves);
    wave->set_in_trap_handler(test.privileged);
    wave->set_ttmp(4, sentinel);
    const uint32_t base = wave->sgpr_alloc().base;
    cu->write_sgpr(base, data);
    cu->write_sgpr(base + 1, 0);
    f.mem->write32(data, value);
    // s_load_dword {ttmp4,s2}, s[0:1], 0
    f.mem->write32(pc, test.ttmp ? 0xc0021c00 : 0xc0020080);
    f.mem->write32(pc + 4, 0);
    f.mem->write32(pc + 8, 0xbf920007);                        // s_trap 7
    const uint32_t read = test.ttmp ? 0xbe830070 : 0xbe830002; // s_mov_b32 s3, {ttmp4,s2}
    f.mem->write32(pc + 12, read);
    f.mem->write32(handler, read);
    cu->step();
    ASSERT_EQ(wave->pc, pc + 8);
    auto *state = static_cast<RaceWavefrontState *>(wave->plugin_state(detector->slot_index()));
    ASSERT_NE(state, nullptr);
    auto &race = *state->race_state;
    ASSERT_EQ(race.getWaveMemoryEvents().size(), 1u);
    const auto event = race.getWaveMemoryEvents().front();
    const auto &events = race.getDetector()->events();
    ASSERT_TRUE(events.hasPendingWaitCounter(event, WaitCounterType::LGKMCNT));
    if (test.ttmp) {
      EXPECT_EQ(events.registers(event).empty(), !test.privileged);
    }
    cu->step();
    const bool entered = test.handler_installed && !test.privileged;
    ASSERT_EQ(wave->pc, entered ? handler : pc + 12);
    EXPECT_EQ(events.hasPendingWaitCounter(event, WaitCounterType::LGKMCNT), !entered);
    cu->step();
    EXPECT_EQ(cu->read_sgpr(base + 3), test.ttmp && !test.privileged ? sentinel : value);
    const bool expect_race = !entered && (!test.ttmp || test.privileged);
    EXPECT_EQ(sink.str().find(test.ttmp ? "type=TTMP" : "type=SGPR") != std::string::npos,
              expect_race);
    wave->halt();
  }
}

TEST(RaceDetectorPluginTest, Cdna4TrapObservesOnlyIssuedAttemptsAndProvesOnlyHandlerEntry) {
  enum class Entry { Taken, RevokedDuringWait, AlreadyInHandler };
  for (const Entry action : {Entry::Taken, Entry::RevokedDuringWait, Entry::AlreadyInHandler}) {
    SCOPED_TRACE(static_cast<int>(action));
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
    PluginSinkConfig sinks;
    auto &sink = sinks.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
    auto ordering = std::make_unique<OrderingPlugin>();
    const auto &events = ordering->events;
    ASSERT_TRUE(f.plugin_group_->add(std::move(ordering)));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    constexpr uint64_t start = 0x2000;
    constexpr uint64_t trap_pc = start + 8;
    constexpr uint64_t handler = 0x4000;
    constexpr uint32_t consume = 0x7e040301; // v_mov_b32 v2, v1
    f.mem->write32(start, 0xd86c0000);       // ds_read_b32 v1, v0
    f.mem->write32(start + 4, 0x01000000);
    f.mem->write32(trap_pc, 0xbf920001); // s_trap 1
    f.mem->write32(trap_pc + 4, consume);
    f.mem->write32(handler, consume);

    bool configured = true;
    auto *cu = f.cu();
    cu->set_trap_handler_resolver([&](const Wavefront &) {
      return ComputeUnitCore::TrapHandlerConfig{configured ? handler : 0, 0, false};
    });
    auto *wave = cu->dispatch_wf(0, start, /*sgprs=*/32, /*vgprs=*/8);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(1);
    wave->set_m0(256);
    wave->set_lds_size(256);
    std::array waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(0, 0, 8, 32, waves);
    cu->step();
    ASSERT_EQ(wave->pc, trap_pc);
    const auto trap_issues = [&] {
      return std::ranges::count_if(events, [&](const HookEvent &event) {
        return event.kind == HookEvent::BEFORE_INSTRUCTION && event.pc == trap_pc;
      });
    };
    wave->wait_counters().increment(WaitCounterType::VMCNT);
    if (action == Entry::AlreadyInHandler)
      wave->set_in_trap_handler(true);
    cu->step();
    if (action != Entry::AlreadyInHandler) {
      EXPECT_EQ(trap_issues(), 0);
      ASSERT_EQ(wave->pc, trap_pc);
      configured = action != Entry::RevokedDuringWait;
      wave->release_wait_counter(WaitCounterType::VMCNT);
      cu->step();
    }
    EXPECT_EQ(trap_issues(), 1);
    ASSERT_EQ(wave->pc, action == Entry::Taken ? handler : trap_pc + 4);
    cu->step(); // Read the result of the LDS load at the actual successor.
    EXPECT_EQ(sink.str().find("RACE ") == std::string::npos, action == Entry::Taken) << sink.str();
  }
}

TEST(RaceDetectorPluginTest, ScalarLoadToTtmpReportsReadBeforeWait) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_in_trap_handler(true); // TTMP accesses are observable in privileged code.
  wf->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/104, waves);

  constexpr uint32_t kTtmp0Selector = amdgpu::kTtmpSelectorFirst;
  const auto words =
      cdna4::build_smem(cdna4::kSLoadDwordSmem, {.sbase = 0, .sdata = kTtmp0Selector, .imm = 1});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> load(decode_valid(*decoder, words.data()));
  ASSERT_NE(load, nullptr);
  ASSERT_NE(load->dst_operand(0), nullptr);
  EXPECT_EQ(load->dst_operand(0)->to_register_ref(), (RegisterRef{RegClass::TTMP, 0, 1}));

  cu->write_sgpr(wf->sgpr_alloc().base, 0x1000u);
  cu->write_sgpr(wf->sgpr_alloc().base + 1, 0u);
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, *load, *wf);
  EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
  ASSERT_NE(load->data(), nullptr);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, *load, *wf);

  static_cast<void>(RegisterAccess(*wf).read_scalar(*load->dst_operand(0)));
  EXPECT_NE(sink.str().find("type=TTMP"), std::string::npos);
}

TEST(RaceDetectorPluginTest, DebuggerReadsIgnorePendingLoads) {
  for (bool scalar : {false, true}) {
    for (bool advance_pc : {false, true}) {
      SCOPED_TRACE(std::format("scalar={}, advance_pc={}", scalar, advance_pc));
      PluginFixture fixture;
      ASSERT_NO_FATAL_FAILURE(fixture.initialize(1, "rdna4", 32, 128));
      PluginSinkConfig sink_config;
      auto &sink = sink_config.emplace<StringSink>();
      fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
      auto race_plugin = std::make_unique<RaceDetectorPlugin>();
      auto *race = race_plugin.get();
      ASSERT_TRUE(fixture.plugin_group_->add(std::move(race_plugin)));
      auto ordering_plugin = std::make_unique<OrderingPlugin>();
      auto *ordering = ordering_plugin.get();
      ASSERT_TRUE(fixture.plugin_group_->add(std::move(ordering_plugin)));
      fixture.soc->set_plugin_group(fixture.plugin_group_);
      fixture.plugin_group_->onInit();

      auto *cu = fixture.cu();
      auto *wave = cu->dispatch_wf(0, 0x1000, 128, 256, 32);
      ASSERT_NE(wave, nullptr);
      wave->set_exec(1);
      std::array<Wavefront *, 1> waves{wave};
      fixture.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, 128, waves);
      auto *state = static_cast<RaceWavefrontState *>(wave->plugin_state(race->slot_index()));

      const auto sbase = wave->sgpr_alloc().base;
      const auto vbase = wave->vgpr_alloc().base;
      cu->write_sgpr(sbase, 0x2000);
      cu->write_sgpr(sbase + 1, 0);
      cu->write_sgpr(sbase + 4, 0x2000);
      cu->write_sgpr(sbase + 5, 0);
      cu->write_sgpr(sbase + 6, 64);
      cu->write_sgpr(sbase + 7, 4 | (5 << 3) | (6 << 6) | (7 << 9) | (46 << 12) | (1u << 28));
      cu->write_sgpr(sbase + 8, 42);
      cu->write_vgpr(vbase + 8, 0, 43);
      cu->write_vgpr(vbase + 8, 31, 44);
      auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
      ASSERT_NE(decoder, nullptr);
      util::StringDiagnostic error;
      auto load =
          scalar
              ? decoder->decode_window(rdna4::build_smem(rdna4::kSLoadB32Smem,
                                                         {.sbase = 0, .sdata = 8, .soffset = 124}),
                                       wave->pc, error.emitter())
              : decoder->decode_window(
                    rdna4::build_vbuffer(
                        rdna4::kBufferLoadFormatXVbuffer,
                        {.soffset = rdna4::OPR_SREG_M0_NULL, .vdata = 8, .rsrc = 4, .format = 46}),
                    wave->pc, error.emitter());
      ASSERT_TRUE(load.succeeded()) << error.message();
      fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave->pc, *load.value(), *wave);
      ASSERT_TRUE(cu->execute_instruction(load.value().get(), *wave).succeeded());
      fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, *load.value(), *wave);
      ASSERT_EQ(state->race_state->getWaveMemoryEvents().size(), 1u);
      if (advance_pc)
        wave->pc += 4;

      ordering->events.clear();
      const auto output = sink.str();
      EXPECT_NO_THROW({
        EXPECT_EQ(wave->debug_read_sgpr(8), 42u);
        EXPECT_EQ(wave->debug_read_vgpr(8, 0), 43u);
        EXPECT_EQ(wave->debug_read_vgpr(8, 31), 44u);
      });
      EXPECT_TRUE(ordering->events.empty());
      EXPECT_EQ(sink.str(), output);
      EXPECT_EQ(state->race_state->getWaveMemoryEvents().size(), 1u);

      RegisterAccess registers(*wave);
      if (scalar)
        EXPECT_EQ(registers.read_sgpr(sbase + 8), 42u);
      else
        EXPECT_EQ(registers.read_vgpr(vbase + 8, 0), 43u);
      EXPECT_EQ(ordering->events.size(), 1u);
      EXPECT_NE(sink.str().find(scalar ? "type=SGPR" : "type=VGPR"), std::string::npos);
    }
  }
}

TEST(RaceDetectorPluginTest, ScalarLoadToTtmpReportsWriteBeforeWait) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_in_trap_handler(true); // The modeled producer and consumer use TTMP0.
  wf->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/104, waves);

  auto state = std::make_unique<ScalarMemState>();
  state->dst_register = {ScalarRegisterStorage::TTMP, 0, 1};
  state->is_load = true;
  TestMemoryInstruction load(std::move(state));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, load, *wf);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

  RegisterAccess(*wf).write_ttmp(0, 0x12345678u);
  EXPECT_NE(sink.str().find("type=TTMP"), std::string::npos);
  EXPECT_NE(sink.str().find("access=write"), std::string::npos);
}

TEST(RaceDetectorPluginTest, ScalarLoadToTtmpHonorsSplitKmcntWait) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1, /*arch=*/"rdna4", /*wavefront_size=*/32,
                                       /*sgprs_per_wf=*/128));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/128, /*vgprs=*/256, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/128, waves);

  auto state = std::make_unique<ScalarMemState>();
  state->dst_register = {ScalarRegisterStorage::TTMP, 0, 1};
  state->is_load = true;
  state->wait_counter_type = WaitCounterType::KMCNT;
  TestMemoryInstruction load(std::move(state));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, load, *wf);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

  execute_wait(*f.plugin_group_, *wf, 0xbfc70000u); // s_wait_kmcnt 0.
  static_cast<void>(RegisterAccess(*wf).read_ttmp(0));
  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

class TexelStatusSparsePage final : public AddressSpaceTranslator {
public:
  VmTranslationResult translate(uint64_t addr, size_t size, VmAccessKind access) const override {
    return IdentityAddressSpaceTranslator{}.translate(addr, size, access);
  }
  std::shared_ptr<const PrtRanges> nonresident_prt_ranges() const override {
    return std::make_shared<const PrtRanges>(PrtRanges{{0x5000, 0x6000}});
  }
};

TEST(RaceDetectorPluginTest, Gfx11BufferTfeTracksOnlyWrittenStatusLanes) {
  for (const char *arch : {"rdna3", "rdna3_5"}) {
    for (bool typed : {false, true}) {
      SCOPED_TRACE(testing::Message() << arch << " typed=" << typed);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 32, 106, 32));
      PluginSinkConfig sinks;
      auto &sink = sinks.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      auto *cu = f.cu();
      auto *wave = cu->dispatch_wf(0, 0x8000, 106, 32, 32);
      ASSERT_NE(wave, nullptr);
      std::array<Wavefront *, 1> waves{wave};
      f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
      GpuVm vm;
      const auto space =
          vm.register_address_space(5, std::make_shared<TexelStatusSparsePage>(),
                                    std::make_shared<GpuMemoryPhysicalAccess>(*f.mem));
      cu->set_gpu_vm(&vm);
      wave->set_address_space(space);
      wave->set_exec(3);
      wave->debug_write_sgpr(0, 0x4000);
      wave->debug_write_sgpr(1, 0);
      wave->debug_write_sgpr(2, 0x2000);
      wave->debug_write_sgpr(3, (2u << 28) | (20u << 12) | 4u);
      f.mem->write32(0x4000, 42);
      for (unsigned lane : {0u, 1u}) {
        wave->debug_write_vgpr(0, lane, lane * 0x1000);
        wave->debug_write_vgpr(9, lane, 0xdeadbeef);
      }
      std::vector<uint32_t> code;
      if (typed) {
        rdna3::MtbufBuilderFields fields{};
        fields.offen = fields.tfe = 1;
        fields.vdata = 8;
        fields.soffset = 128;
        fields.format = 20;
        const auto load = rdna3::build_mtbuf(rdna3::kTbufferLoadFormatXMtbuf, fields);
        code.assign(load.begin(), load.end());
      } else {
        rdna3::MubufBuilderFields fields{};
        fields.offen = fields.tfe = 1;
        fields.vdata = 8;
        fields.soffset = 128;
        const auto load = rdna3::build_mubuf(rdna3::kBufferLoadB32Mubuf, fields);
        code.assign(load.begin(), load.end());
      }
      const auto read = rdna3::build_vop1(rdna3::kVMovB32Vop1, {.src0 = 265, .vdst = 3})[0];
      const uint32_t wait = rdna3::build_sopp(rdna3::kSWaitcntSopp, {.simm16 = 0x0007})[0];
      code.insert(code.end(), {read, read, wait, read});
      f.mem->load_image(reinterpret_cast<const uint8_t *>(code.data()), code.size() * 4, wave->pc);
      ASSERT_TRUE(cu->step());
      for (unsigned lane : {0u, 1u}) {
        wave->set_exec(uint64_t{1} << lane);
        const auto count = cu->memory_wait_diagnostic_count();
        const auto size = sink.str().size();
        ASSERT_TRUE(cu->step());
        EXPECT_EQ(wave->debug_read_vgpr(3, lane), lane ? 1u : 0xdeadbeefu);
        EXPECT_EQ(cu->memory_wait_diagnostic_count() - count, lane);
        EXPECT_EQ(sink.str().find("RACE ", size) != std::string::npos, lane != 0);
      }
      const auto count = cu->memory_wait_diagnostic_count();
      const auto size = sink.str().size();
      ASSERT_TRUE(cu->step());
      ASSERT_TRUE(cu->step());
      EXPECT_EQ(cu->memory_wait_diagnostic_count(), count);
      EXPECT_EQ(sink.str().find("RACE ", size), std::string::npos);
      wave->set_address_space({});
      cu->set_gpu_vm(nullptr);
      wave->halt();
    }
  }
}

TEST(RaceDetectorPluginTest, Gfx10ImageTfeTracksTheLanesThatReceiveStatus) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(
        f.initialize(1, arch == ROCJITSU_CODE_ARCH_RDNA1 ? "rdna1" : "rdna2", 32, 106, 32));
    PluginSinkConfig sinks;
    auto &sink = sinks.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    constexpr uint64_t pc = 0x8000;
    auto *wave = cu->dispatch_wf(0, pc, 106, 32, 32);
    ASSERT_NE(wave, nullptr);
    std::array<Wavefront *, 1> waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
    GpuVm vm;
    auto space = vm.register_address_space(5, std::make_shared<TexelStatusSparsePage>(),
                                           std::make_shared<GpuMemoryPhysicalAccess>(*f.mem));
    cu->set_gpu_vm(&vm);
    wave->set_process_id(5);
    wave->set_address_space(space);
    // 2048x1 linear R32_UINT. Texel 1024 is in the sparse page at 0x5000.
    const std::array descriptor{0x40u,
                                (20u << 20) | (3u << 30),
                                511u,
                                (9u << 28) | 4u | (1u << 9),
                                arch == ROCJITSU_CODE_ARCH_RDNA2 ? 2047u : 0u,
                                0u,
                                0u,
                                0u};
    for (unsigned i = 0; i < descriptor.size(); ++i)
      wave->debug_write_sgpr(i, descriptor[i]);
    wave->set_exec(3);
    for (unsigned lane : {0u, 1u}) {
      wave->debug_write_vgpr(0, lane, lane * 1024);
      wave->debug_write_vgpr(1, lane, 0);
      wave->debug_write_vgpr(9, lane, 0xdeadbeef);
    }
    rdna1::MimgBuilderFields fields{};
    fields.dim = fields.unorm = fields.dmask = fields.tfe = 1;
    fields.vdata = 8;
    const auto fetch = rdna1::build_mimg(rdna1::kImageLoadMimg, fields);
    const auto read = rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = 265, .vdst = 3})[0];
    const uint32_t code[]{fetch[0], fetch[1], read, read, legacy_wait_word(arch, 0, 63), read};
    f.mem->load_image(reinterpret_cast<const uint8_t *>(code), sizeof(code), pc);
    ASSERT_TRUE(cu->step());
    ASSERT_EQ(wave->pc, pc + 8);
    for (unsigned lane : {0u, 1u}) {
      const auto before = sink.str().size();
      const auto count = cu->memory_wait_diagnostic_count();
      wave->set_exec(uint64_t{1} << lane);
      ASSERT_TRUE(cu->step());
      EXPECT_EQ(wave->debug_read_vgpr(3, lane), lane ? 1u : 0xdeadbeefu);
      EXPECT_EQ(sink.str().find("RACE ", before) != std::string::npos, lane != 0);
      EXPECT_EQ(cu->memory_wait_diagnostic_count() - count, lane);
    }
    const auto before = sink.str().size();
    const auto count = cu->memory_wait_diagnostic_count();
    ASSERT_TRUE(cu->step());
    ASSERT_TRUE(cu->step());
    EXPECT_EQ(sink.str().find("RACE ", before), std::string::npos);
    EXPECT_EQ(cu->memory_wait_diagnostic_count(), count);
    // An older texture sample and an ordinary fetch have distinct GFX10 writeback classes.
    const auto sample = rdna1::build_mimg(
        rdna1::kImageSampleLz, {.dim = 1, .dmask = 1, .vaddr = 5, .vdata = 9, .ssamp = 2});
    const auto unrelated = rdna1::build_mimg(
        rdna1::kImageSampleLz, {.dim = 1, .dmask = 1, .vaddr = 5, .vdata = 6, .ssamp = 2});
    wave->set_exec(2);
    wave->debug_write_vgpr(5, 1, 0);
    for (unsigned intervening : {0u, 1u}) {
      for (unsigned threshold : {0u, 1u, 2u}) {
        SCOPED_TRACE(testing::Message() << "intervening=" << intervening << " wait=" << threshold);
        std::vector<uint32_t> program{legacy_wait_word(arch, 0, 0), sample[0], sample[1]};
        if (intervening)
          program.insert(program.end(), {unrelated[0], unrelated[1]});
        program.insert(program.end(), {legacy_wait_word(arch, threshold, 0), fetch[0], fetch[1]});
        const uint64_t start = 0x9000 + (3 * intervening + threshold) * 256;
        wave->pc = start;
        f.mem->load_image(reinterpret_cast<const uint8_t *>(program.data()),
                          program.size() * sizeof(uint32_t), start);
        const auto fetch_pc = start + (program.size() - 2) * sizeof(uint32_t);
        for (unsigned step = 0; step < 32 && wave->pc < fetch_pc; ++step) {
          ASSERT_TRUE(cu->step());
        }
        ASSERT_EQ(wave->pc, fetch_pc);
        const auto count = cu->memory_wait_diagnostic_count();
        testing::internal::CaptureStderr();
        const bool stepped = cu->step();
        const auto diagnostic = testing::internal::GetCapturedStderr();
        ASSERT_TRUE(stepped);
        EXPECT_EQ(wave->debug_read_vgpr(9, 1), 1u);
        const bool hazard = threshold > intervening;
        EXPECT_EQ(cu->memory_wait_diagnostic_count() - count, unsigned(hazard));
        if (hazard) {
          EXPECT_NE(diagnostic.find("A wait threshold <= " + std::to_string(intervening) +
                                    " is required"),
                    std::string::npos);
        }
      }
    }
    wave->set_address_space({});
    cu->set_gpu_vm(nullptr);
    wave->halt();
  }
}

TEST(RaceDetectorPluginTest, Rdna1TexelStatusIsAFullDwordResultOnActiveLanes) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna1", 32, 106, 32));
  PluginSinkConfig sinks;
  auto &sink = sinks.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
  auto race = std::make_unique<RaceDetectorPlugin>();
  auto *plugin = race.get();
  ASSERT_TRUE(f.plugin_group_->add(std::move(race)));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();
  auto *cu = f.cu();
  auto *wave = cu->dispatch_wf(0, 0x8000, 106, 16, 32);
  ASSERT_NE(wave, nullptr);
  std::array<Wavefront *, 1> waves{wave};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
  auto &race_state =
      *static_cast<RaceWavefrontState *>(wave->plugin_state(plugin->slot_index()))->race_state;
  GpuVm vm;
  const auto as = vm.register_address_space(5, std::make_shared<TexelStatusSparsePage>(),
                                            std::make_shared<GpuMemoryPhysicalAccess>(*f.mem));
  cu->set_gpu_vm(&vm);
  wave->set_address_space(as);
  wave->set_process_id(5);
  const auto sb = wave->sgpr_alloc().base;
  const auto vb = wave->vgpr_alloc().base;
  for (uint32_t i = 0; i < 16; ++i)
    cu->write_sgpr(sb + i, 0);
  cu->write_sgpr(sb, 0x4000 >> 8);
  cu->write_sgpr(sb + 1, (20u << 20) | (1u << 29)); // 1x1 R32_UINT.
  cu->write_sgpr(sb + 3, (9u << 28) | 4);
  cu->write_sgpr(sb + 5, 256u << 8); // Warn for LOD below 1.0.
  cu->write_sgpr(sb + 12, 0x4000);
  cu->write_sgpr(sb + 14, 0x3000);
  cu->write_sgpr(sb + 15, (3u << 28) | (20u << 12));
  f.mem->write32(0x4000, 12345);
  for (unsigned lane : {1u, 2u}) {
    cu->write_vgpr(vb, lane, 0);
    cu->write_vgpr(vb + 1, lane, 0);
  }
  for (const auto &[image, d16, flag, dst] :
       std::array<std::tuple<bool, bool, bool, uint8_t>, 6>{{{false, false, true, 8},
                                                             {true, false, true, 8},
                                                             {true, true, true, 8},
                                                             {false, false, false, 8},
                                                             {true, false, false, 8},
                                                             {false, false, true, 15}}}) {
    SCOPED_TRACE(testing::Message() << image << "/" << d16 << "/" << flag << "/v" << unsigned(dst));
    const uint8_t payload = d16 ? 2 : 1;
    const uint8_t status = dst + payload;
    const bool writes_status = flag && status < wave->num_vgprs();
    const uint32_t sentinel = 0xfeedbeefu;
    for (unsigned lane = 0; lane <= 2; ++lane)
      cu->write_vgpr(vb + status, lane, sentinel);
    // Buffer VADDR aliases its status slot. Lane 1 is resident, lane 2 is in the hole.
    if (!image) {
      cu->write_vgpr(vb + status, 1, 0);
      cu->write_vgpr(vb + status, 2, 0x1000);
    }
    wave->set_exec(6);
    const auto fetch =
        image ? rdna1::build_mimg(rdna1::kImageSampleLzMimg, {.dim = 1,
                                                              .dmask = uint8_t(d16 ? 7 : 1),
                                                              .lwe = uint8_t(flag),
                                                              .vdata = dst,
                                                              .ssamp = 2,
                                                              .d16 = uint8_t(d16)})
              : rdna1::build_mubuf(rdna1::kBufferLoadDword, {.offen = 1,
                                                             .vaddr = status,
                                                             .vdata = dst,
                                                             .srsrc = 3,
                                                             .tfe = uint8_t(flag),
                                                             .soffset = 128});
    const uint64_t pc = wave->pc + 0x100;
    wave->pc = pc; // Distinct PCs avoid race-report deduplication between cases.
    const auto read =
        rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = uint16_t(256 + status), .vdst = 3})[0];
    const uint32_t code[] = {fetch[0], fetch[1], read, read, legacy_wait_word(cu->arch(), 0, 63),
                             read,     S_ENDPGM};
    f.mem->load_image(reinterpret_cast<const uint8_t *>(code), sizeof(code), pc);
    const auto advance = [&](unsigned word) {
      for (unsigned tick = 0; tick < 32 && wave->pc < pc + 4 * word; ++tick)
        ASSERT_TRUE(cu->step());
      ASSERT_EQ(wave->pc, pc + 4 * word);
    };
    ASSERT_NO_FATAL_FAILURE(advance(2));
    EXPECT_EQ(cu->read_vgpr_storage(vb + status, 0), sentinel);
    EXPECT_EQ(cu->read_vgpr_storage(vb + status, 2),
              writes_status ? (image ? 2u : 1u) : (image ? sentinel : 0x1000u));
    const auto pending_loads = [&](uint32_t reg) {
      const auto &events = race_state.getDetector()->events();
      return std::ranges::count_if(race_state.getVgprMemoryEvents(reg), [&](EventId id) {
        return events.type(id) == MemoryEventType::GLOBAL_TO_VGPR && events.registerLanes(id, reg);
      });
    };
    EXPECT_EQ(pending_loads(dst), unsigned(!flag || writes_status));
    EXPECT_EQ(pending_loads(status), unsigned(writes_status));
    if (d16) {
      ASSERT_EQ(race_state.getWaveMemoryEvents().size(), 1u);
      const auto id = race_state.getWaveMemoryEvents().front();
      const auto &events = race_state.getDetector()->events();
      EXPECT_EQ(events.registerByteMask(id, dst + 1), cu->sram_ecc() ? 0xf : 0x3);
      EXPECT_EQ(events.registerByteMask(id, status), 0xf);
    }
    const auto before = sink.str().size();
    // The buffer preserves the resident status. Both active texture lanes have a LOD warning.
    wave->set_exec(image ? 1 : 2);
    ASSERT_NO_FATAL_FAILURE(advance(3));
    EXPECT_EQ(sink.str().find("RACE ", before), std::string::npos);
    wave->set_exec(4);
    ASSERT_NO_FATAL_FAILURE(advance(4));
    EXPECT_EQ(sink.str().find("RACE ", before) != std::string::npos, writes_status)
        << sink.str().substr(before);
    const auto after = sink.str().size();
    ASSERT_NO_FATAL_FAILURE(advance(6));
    EXPECT_EQ(pending_loads(status), 0);
    EXPECT_EQ(sink.str().find("RACE ", after), std::string::npos);
  }
  wave->set_address_space({});
  cu->set_gpu_vm(nullptr);
  wave->halt();
}

TEST(RaceDetectorPluginTest, Gfx10PackedImageOperandsMatchWaitChecksAndRegisterReads) {
  struct Case {
    uint8_t dim;
    uint16_t opcode;
    bool a16;
    bool nsa;
    // Prefix (offset, bias, compare or directional gradients), followed by coordinates.
    std::vector<uint8_t> masks;
    bool compare = false;
  };
  const Case cases[] = {
      {0, rdna1::kImageSampleDMimg, true, false, {0xf, 0xf, 0x3}},
      {2, rdna1::kImageSampleDMimg, false, false, {0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf}},
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    const bool gfx1010 = arch == ROCJITSU_CODE_ARCH_RDNA1;
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, gfx1010 ? "rdna1" : "rdna2", 32, 106, 32));
    PluginSinkConfig sinks;
    auto &sink = sinks.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    auto observed = std::make_unique<OrderingPlugin>();
    auto &events = observed->events;
    ASSERT_TRUE(f.plugin_group_->add(std::move(observed)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    auto *wave = cu->dispatch_wf(0, 0x8000, 106, 32, 32);
    ASSERT_NE(wave, nullptr);
    std::array<Wavefront *, 1> waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(0, 0, 32, 106, waves);
    wave->set_lds_size(16);
    wave->set_m0(16);
    cu->lds().write32(wave->lds_base(), 0);
    const auto vb = wave->vgpr_alloc().base;
    const auto sb = wave->sgpr_alloc().base;
    constexpr uint32_t texel = 0x40e00000; // R32_FLOAT 7.0.
    // A point on a cube seam can select an adjacent face. Each 1x1 R32 face occupies one aligned
    // row.
    for (unsigned face = 0; face < 6; ++face)
      f.mem->write32(0x4000 + 256 * face, texel);
    cu->write_sgpr(sb, 0x4000 >> 8);
    cu->write_sgpr(sb + 1, 22u << 20);
    cu->write_sgpr(sb + 8, 7u << 12); // Compare ALWAYS.
    const auto issue = [&](const auto &words, uint64_t exec) {
      const uint64_t pc = wave->pc + 0x80;
      wave->pc = pc;
      wave->set_exec(exec);
      f.mem->load_image(reinterpret_cast<const uint8_t *>(words.data()), words.size() * 4, pc);
      for (unsigned tick = 0; tick < 32 && wave->pc == pc; ++tick)
        ASSERT_TRUE(cu->step());
      ASSERT_EQ(wave->pc, pc + words.size() * 4);
    };
    const auto check_input = [&](const auto &instruction, uint8_t reg, uint8_t mask,
                                 const auto &prepare, const auto &verify) {
      for (const bool high : {false, true}) {
        const auto producer = rdna1::build_ds(high ? rdna1::kDsReadU16D16Hi : rdna1::kDsReadU16D16,
                                              {.addr = 30, .vdst = reg});
        for (const auto &[lanes, waited] : {std::pair{4u, false}, {4u, true}, {2u, false}}) {
          SCOPED_TRACE(testing::Message() << "v" << unsigned(reg) << " high=" << high
                                          << " lanes=" << lanes << " waited=" << waited);
          prepare(high);
          ASSERT_NO_FATAL_FAILURE(issue(producer, lanes));
          if (waited) {
            ASSERT_NO_FATAL_FAILURE(issue(std::array{legacy_wait_word(arch, 63, 0)}, 1));
          }
          events.clear();
          const auto output = sink.str().size();
          const auto warnings = cu->memory_wait_diagnostic_count();
          ASSERT_NO_FATAL_FAILURE(issue(instruction, 5));
          const bool hazard = !waited && (lanes & 5) && (mask & (high ? 0xc : 0x3));
          EXPECT_EQ(cu->memory_wait_diagnostic_count() != warnings, hazard);
          EXPECT_EQ(sink.str().find("type=VGPR access=read reg=" + std::to_string(reg) + " ",
                                    output) != std::string::npos,
                    hazard);
          std::array<uint8_t, 32> bytes{};
          for (const auto &event : events)
            if (event.kind == HookEvent::READ_VGPR && event.physical_reg == vb + reg)
              for (unsigned lane = 0; lane < bytes.size(); ++lane)
                if ((event.lane_mask >> lane) & 1)
                  bytes[lane] |= event.byte_mask;
          for (unsigned lane = 0; lane < bytes.size(); ++lane)
            EXPECT_EQ(bytes[lane], lane == 0 || lane == 2 ? mask : 0) << lane;
          verify();
          ASSERT_NO_FATAL_FAILURE(issue(std::array{legacy_wait_word(arch, 0, 0)}, 1));
        }
      }
    };
    for (const auto &test : cases) {
      SCOPED_TRACE(testing::Message() << int(arch) << " dim=" << unsigned(test.dim)
                                      << " op=" << test.opcode << " a16=" << test.a16);
      cu->write_sgpr(sb + 3, ((8u + test.dim) << 28) | 4u);
      cu->write_sgpr(sb + 4, test.dim == 3 ? 5 : test.dim < 2 && !gfx1010 ? 63 : 0);
      const auto reg_at = [&](unsigned i) { return uint8_t(test.nsa ? 2 * i + 1 : i); };
      const unsigned nsa = test.nsa ? (test.masks.size() + 2) / 4 : 0;
      const auto base = rdna1::build_mimg(test.opcode, {.nsa = uint8_t(nsa),
                                                        .dim = test.dim,
                                                        .dmask = 1,
                                                        .vaddr = reg_at(0),
                                                        .vdata = 24,
                                                        .ssamp = 2,
                                                        .a16 = uint8_t(test.a16)});
      std::vector<uint32_t> sample(base.begin(), base.end());
      sample.resize(2 + nsa);
      if (test.nsa)
        for (unsigned i = 1; i < test.masks.size(); ++i)
          sample[2 + (i - 1) / 4] |= uint32_t(reg_at(i)) << (8 * ((i - 1) % 4));
      for (unsigned word = 0; word < test.masks.size(); ++word) {
        ASSERT_NO_FATAL_FAILURE(check_input(
            sample, reg_at(word), test.masks[word], [](bool) {},
            [&] {
              EXPECT_EQ(cu->read_vgpr_storage(vb + 24, 2), test.compare ? 0x3f800000u : texel);
            }));
      }
    }

    struct Transfer {
      uint16_t op;
      uint8_t dim;
      std::vector<uint32_t> coords;
      unsigned offset; // Byte offset for lane 2.
      bool a16 = true;
      bool nsa = false;
      bool returns = false; // GLC on an atomic.
    };
    const unsigned msaa_offset = gfx1010 ? 32892 : 32956; // (3,7), fragment 2, 64KB_R_X.
    const Transfer transfers[] = {
        {rdna1::kImageLoadMimg, 0, {1}, 4},
        {rdna1::kImageStoreMimg, 0, {1}, 4},
        {rdna1::kImageLoadMimg, 4, {1, 1}, 260},
        {rdna1::kImageLoadMipMimg, 1, {1, 1, 1}, 260, true, true},
        {rdna1::kImageLoadMimg, 6, {3, 7, 2}, msaa_offset, true, true},
        {rdna1::kImageStoreMimg, 6, {3, 7, 2}, msaa_offset, true, true},
        {rdna1::kImageLoadMimg, 0, {1}, 4, false},
        {rdna1::kImageGetResinfoMimg, 1, {1}, 0},
        {rdna1::kImageGetResinfoMimg, 1, {0x7e000001}, 0, false},
    };
    unsigned base_address = 0x200000;
    for (const auto &test : transfers) {
      SCOPED_TRACE(testing::Message()
                   << int(arch) << " transfer=" << test.op << " dim=" << unsigned(test.dim)
                   << " a16=" << test.a16 << " return=" << test.returns);
      const bool query = test.op == rdna1::kImageGetResinfoMimg;
      const bool atomic = test.op == rdna1::kImageAtomicAddMimg;
      const bool store = test.op == rdna1::kImageStoreMimg ||
                         test.op == rdna1::kImageStorePckMimg ||
                         test.op == rdna1::kImageStoreMipPckMimg;
      const bool mip =
          query || test.op == rdna1::kImageLoadMipMimg || test.op == rdna1::kImageStoreMipPckMimg;
      const bool msaa = test.dim == 6;
      const unsigned height = msaa ? 8 : mip ? 4 : 2;
      const unsigned levels = msaa ? 2 : unsigned(mip);
      base_address += 0x10000;
      cu->write_sgpr(sb, base_address >> 8);
      cu->write_sgpr(sb + 1, (20u << 20) | (3u << 30)); // R32_UINT, width 8.
      cu->write_sgpr(sb + 2, 1u | ((height - 1) << 14));
      cu->write_sgpr(sb + 3,
                     ((8u + test.dim) << 28) | (msaa ? 27u << 20 : 0) | (levels << 16) | 4u);
      cu->write_sgpr(sb + 4, test.dim == 2 || test.dim == 4 || test.dim == 5 ? 1
                             : gfx1010                                       ? 0
                                                                             : 63);
      cu->write_sgpr(sb + 5, unsigned(mip) << 4);
      const unsigned count = test.a16 ? (test.coords.size() + 1) / 2 : test.coords.size();
      const auto reg_at = [&](unsigned word) { return uint8_t(test.nsa ? 2 * word + 1 : word); };
      const auto encoded = rdna1::build_mimg(test.op, {.nsa = uint8_t(test.nsa),
                                                       .dim = test.dim,
                                                       .dmask = 1,
                                                       .unorm = 1,
                                                       .glc = uint8_t(test.returns),
                                                       .r128 = uint8_t(msaa),
                                                       .vaddr = reg_at(0),
                                                       .vdata = 24,
                                                       .a16 = uint8_t(test.a16)});
      std::vector<uint32_t> words(encoded.begin(), encoded.end());
      if (test.nsa) {
        words.push_back(0);
        for (unsigned i = 1; i < count; ++i)
          words[2] |= uint32_t(reg_at(i)) << (8 * (i - 1));
      }
      for (unsigned index = 0; index < count; ++index) {
        const bool odd_tail = test.a16 && index * 2 + 1 == test.coords.size();
        const auto prepare = [&](bool high) {
          cu->l1_vector().flush_all();
          cu->l2()->flush_all();
          f.mem->write32(base_address, 17);
          f.mem->write32(base_address + test.offset, 99);
          for (unsigned lane = 0; lane <= 2; ++lane) {
            // Lane 0 uses x=0, leaving lane 2 as the sole writer at the tested texel.
            const auto coord = [&](unsigned i) {
              return lane == 0 && i == 0 && !query ? 0u : test.coords[i];
            };
            for (unsigned i = 0; i < count; ++i) {
              uint32_t value = test.a16 ? coord(2 * i) : coord(i);
              if (test.a16)
                value |= (2 * i + 1 < test.coords.size() ? coord(2 * i + 1) : 0x7e00u) << 16;
              cu->write_vgpr(vb + reg_at(i), lane, value);
              if (i == index)
                cu->lds().write32(wave->lds_base() + 4 * lane, high ? value >> 16 : value & 0xffff);
            }
            cu->write_vgpr(vb + 30, lane, 4 * lane);
            cu->write_vgpr(vb + 24, lane, store || atomic ? 7u : 0xdeadbeefu);
          }
        };
        const auto verify = [&] {
          const unsigned result = query                                ? (test.a16 ? 4 : 0)
                                  : store || (atomic && !test.returns) ? 7
                                                                       : 99;
          EXPECT_EQ(cu->read_vgpr_storage(vb + 24, 2), result);
          cu->l1_vector().flush_all();
          cu->l2()->flush_all();
          EXPECT_EQ(f.mem->read32(base_address + test.offset), atomic ? 106u : store ? 7u : 99u);
        };
        ASSERT_NO_FATAL_FAILURE(
            check_input(words, reg_at(index), odd_tail ? 0x3 : 0xf, prepare, verify));
        if (store || (atomic && !test.returns)) {
          ASSERT_NO_FATAL_FAILURE(
              issue(rdna1::build_sopk(rdna1::kSWaitcntVscnt, {.sdst = 125}), 1));
        }
      }
    }
    wave->halt();
  }
}

TEST(RaceDetectorPluginTest, Gfx10InlineImageQueriesTrackVmcntAndVgprWriteback) {
  for (const auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    SCOPED_TRACE(arch);
    const bool gfx1010 = arch == ROCJITSU_CODE_ARCH_RDNA1;
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, gfx1010 ? "rdna1" : "rdna2", 32, 106, 32));
    PluginSinkConfig sinks;
    auto &sink = sinks.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *observer = plugin.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    auto routing = std::make_unique<MemoryObservationPlugin>();
    const auto &routed = routing->accesses;
    ASSERT_TRUE(f.plugin_group_->add(std::move(routing)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    auto *wave = cu->dispatch_wf(0, 0x8000, 106, 16, 32);
    ASSERT_NE(wave, nullptr);
    std::array<Wavefront *, 1> waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
    auto &race =
        *static_cast<RaceWavefrontState *>(wave->plugin_state(observer->slot_index()))->race_state;
    const auto sb = wave->sgpr_alloc().base;
    const auto vb = wave->vgpr_alloc().base;
    cu->write_sgpr(sb, 0x4000 >> 8);
    cu->write_sgpr(sb + 1, 20u << 20);
    cu->write_sgpr(sb + 3, (9u << 28) | 4u);
    cu->write_sgpr(sb + 4, gfx1010 ? 0 : 63);
    cu->write_sgpr(sb + 12, 0x4000);
    cu->write_sgpr(sb + 14, 256);
    cu->write_sgpr(sb + 15, (3u << 28) | (20u << 12));
    f.mem->write32(0x4000, 123);
    const auto step = [&](const auto &words, uint64_t exec) {
      const uint64_t pc = wave->pc + 0x100;
      wave->pc = pc;
      wave->set_exec(exec);
      f.mem->load_image(reinterpret_cast<const uint8_t *>(words.data()), sizeof(words), pc);
      for (unsigned tick = 0; tick < 32 && wave->pc == pc; ++tick)
        ASSERT_TRUE(cu->step());
      ASSERT_EQ(wave->pc, pc + sizeof(words));
    };
    const auto wait = [&](uint8_t threshold) {
      ASSERT_NO_FATAL_FAILURE(step(std::array{legacy_wait_word(arch, threshold, 63)}, 1));
    };
    const auto read = [&](uint16_t reg, uint64_t exec, bool hazard) {
      SCOPED_TRACE(testing::Message() << "read v" << reg << " exec=" << exec);
      const auto output = sink.str().size();
      ASSERT_NO_FATAL_FAILURE(step(
          rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = uint16_t(256 + reg), .vdst = 3}), exec));
      EXPECT_EQ(sink.str().find("type=VGPR access=read reg=" + std::to_string(reg), output) !=
                    std::string::npos,
                hazard);
    };
    const auto fetch = [](bool sample, uint8_t dst) {
      return sample ? rdna1::build_mimg(rdna1::kImageSampleLz,
                                        {.dim = 1, .dmask = 1, .vdata = dst, .ssamp = 2})
                    : rdna1::build_mubuf(rdna1::kBufferLoadDword,
                                         {.vdata = dst, .srsrc = 3, .soffset = 128});
    };
    for (const auto query_kind : kInlineImageQueries) {
      const bool lod = query_kind == InlineImageQuery::Lod;
      SCOPED_TRACE(lod ? "get_lod" : "get_resinfo");
      const auto query = [lod](uint8_t dst, uint8_t addr = 0, uint8_t mask = 3) {
        return rdna1::build_mimg(
            lod ? rdna1::kImageGetLod : rdna1::kImageGetResinfo,
            {.dim = 1, .dmask = mask, .vaddr = addr, .vdata = dst, .ssamp = uint8_t(lod ? 2 : 0)});
      };
      for (uint32_t reg : {8u, 9u}) {
        for (unsigned lane : {1u, 2u})
          cu->write_vgpr(vb + reg, lane, 0xdeadbeef);
        ASSERT_NO_FATAL_FAILURE(step(query(8), 5));
        EXPECT_EQ(cu->read_vgpr_storage(vb + reg, 1), 0xdeadbeefu);
        EXPECT_NE(cu->read_vgpr_storage(vb + reg, 2), 0xdeadbeefu);
        ASSERT_NO_FATAL_FAILURE(read(reg, 2, false));
        ASSERT_NO_FATAL_FAILURE(read(reg, 4, true));
        ASSERT_NO_FATAL_FAILURE(wait(0));
        ASSERT_NO_FATAL_FAILURE(read(reg, 4, false));
      }

      for (bool waited : {false, true}) {
        SCOPED_TRACE(testing::Message() << "query VADDR waited=" << waited);
        ASSERT_NO_FATAL_FAILURE(step(query(8), 1));
        if (waited) {
          ASSERT_NO_FATAL_FAILURE(wait(0));
        }
        const auto output = sink.str().size();
        ASSERT_NO_FATAL_FAILURE(step(query(12, 8), 1));
        EXPECT_EQ(sink.str().find("type=VGPR access=read reg=8", output) != std::string::npos,
                  !waited);
        ASSERT_NO_FATAL_FAILURE(wait(0));
      }
      const auto output = sink.str().size();
      ASSERT_NO_FATAL_FAILURE(step(query(8, 8), 1));
      EXPECT_EQ(sink.str().find("RACE ", output), std::string::npos);
      ASSERT_NO_FATAL_FAILURE(wait(0));
      cu->write_vgpr(vb + 8, 0, 0xdeadbeefu);
      cu->write_vgpr(vb + 9, 0, 0xbad0c0deu);
      ASSERT_NO_FATAL_FAILURE(step(query(8, 0, 0), 1));
      EXPECT_EQ(cu->read_vgpr_storage(vb + 8, 0), 0u);
      EXPECT_EQ(cu->read_vgpr_storage(vb + 9, 0), 0xbad0c0deu);
      ASSERT_NO_FATAL_FAILURE(read(8, 1, true));
      ASSERT_NO_FATAL_FAILURE(wait(0));

      cu->write_sgpr(sb + 3, 0); // An unbound image still writes zero results.
      cu->write_vgpr(vb + 8, 0, 0xdeadbeefu);
      ASSERT_NO_FATAL_FAILURE(step(query(8), 1));
      EXPECT_EQ(cu->read_vgpr_storage(vb + 8, 0), 0u);
      ASSERT_NO_FATAL_FAILURE(read(8, 1, true));
      ASSERT_NO_FATAL_FAILURE(wait(0));
      cu->write_sgpr(sb + 3, (9u << 28) | 4u);

      // Even an empty or suppressed younger query moves the older load's VMCNT position.
      for (const auto &[exec, dst] : {std::pair{1u, 8u}, {0u, 8u}, {1u, 15u}}) {
        SCOPED_TRACE(testing::Message() << "exec=" << exec << " dst=" << dst);
        const auto routes = routed.size();
        ASSERT_NO_FATAL_FAILURE(step(fetch(false, 12), 1));
        ASSERT_EQ(routed.size(), routes + 1);
        ASSERT_NO_FATAL_FAILURE(step(query(dst), exec));
        EXPECT_EQ(routed.size(), routes + 1);
        const auto &events = race.getWaveMemoryEvents();
        ASSERT_EQ(events.size(), exec && dst == 8 ? 2u : 1u);
        EXPECT_EQ(race.getDetector()->events().pendingCounterIncrement(events.front(),
                                                                       WaitCounterType::VMCNT),
                  exec && dst == 8 ? 1 : 2);
        ASSERT_NO_FATAL_FAILURE(wait(1));
        ASSERT_NO_FATAL_FAILURE(read(12, 1, false));
        EXPECT_EQ(cu->read_vgpr_storage(vb + 3, 0), 123u);
        ASSERT_NO_FATAL_FAILURE(read(dst, 1, exec && dst == 8));
        ASSERT_NO_FATAL_FAILURE(wait(0));
      }
      ASSERT_NO_FATAL_FAILURE(step(query(8), 4));
      ASSERT_NO_FATAL_FAILURE(wait(0));
      const auto query_value = cu->read_vgpr_storage(vb + 9, 2);
      ASSERT_NE(query_value, 123u);
      for (bool sample : {false, true}) {
        for (bool query_first : {false, true}) {
          SCOPED_TRACE(testing::Message() << "sample=" << sample << " query_first=" << query_first);
          const auto q = query(8);
          const auto peer = fetch(sample, 9); // Only the query's second result overlaps.
          ASSERT_NO_FATAL_FAILURE(step(query_first ? q : peer, 4));
          const auto output = sink.str().size();
          ASSERT_NO_FATAL_FAILURE(step(query_first ? peer : q, 4));
          EXPECT_EQ(sink.str().find("type=VGPR access=write reg=9", output) != std::string::npos,
                    lod != sample);
          ASSERT_NO_FATAL_FAILURE(wait(0));
          if (lod == sample) {
            EXPECT_EQ(cu->read_vgpr_storage(vb + 9, 2), query_first ? 123u : query_value);
          }
        }
      }
    }
    wave->halt();
  }
}

TEST(RaceDetectorPluginTest, Gfx10IssuedEmptyFetchCountsTowardsPartialVmcnt) {
  enum class Younger {
    Active,
    Empty,
    Suppressed,
    NotIssued,
    Store,
    EmptyFlat,
    SuppressedFlat,
    EmptyFlatStore
  };
  for (const auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    const bool rdna1 = arch == ROCJITSU_CODE_ARCH_RDNA1;
    SCOPED_TRACE(rdna1 ? "rdna1" : "rdna2");
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, rdna1 ? "rdna1" : "rdna2", 32, 106, 32));
    PluginSinkConfig sinks;
    auto &sink = sinks.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
    auto race = std::make_unique<RaceDetectorPlugin>();
    auto *plugin = race.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(race)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    auto *wave = cu->dispatch_wf(0, 0x8000, 106, 16, 32);
    ASSERT_NE(wave, nullptr);
    std::array<Wavefront *, 1> waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
    auto &race_state =
        *static_cast<RaceWavefrontState *>(wave->plugin_state(plugin->slot_index()))->race_state;
    GpuVm vm;
    auto space = vm.register_address_space(5, std::make_shared<TexelStatusSparsePage>(),
                                           std::make_shared<GpuMemoryPhysicalAccess>(*f.mem));
    cu->set_gpu_vm(&vm);
    wave->set_address_space(space);
    wave->set_process_id(5);
    const auto sb = wave->sgpr_alloc().base;
    for (unsigned i = 0; i < 16; ++i)
      cu->write_sgpr(sb + i, 0);
    cu->write_sgpr(sb, 0x4000 >> 8);
    cu->write_sgpr(sb + 1, (20u << 20) | (uint32_t(rdna1) << 29));
    cu->write_sgpr(sb + 3, (9u << 28) | 4u);
    cu->write_sgpr(sb + 4, rdna1 ? 0 : 63);           // GFX10.3 encodes the 256-byte row pitch.
    cu->write_sgpr(sb + 5, rdna1 ? 256u << 8 : 0);    // LWE reports LOD below 1.0.
    cu->write_sgpr(sb + 12, rdna1 ? 0x5000 : 0x4000); // TFE reads the sparse page on RDNA1.
    cu->write_sgpr(sb + 14, 256);
    cu->write_sgpr(sb + 15, (3u << 28) | (20u << 12));
    f.mem->write32(0x4000, 123);
    const uint16_t result = rdna1 ? 9 : 8;
    const auto read =
        rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = uint16_t(256 + result), .vdst = 3});
    for (bool image : {false, true}) {
      const auto fetch = [&](uint8_t dst) {
        return image ? rdna1::build_mimg(
                           rdna1::kImageSampleLzMimg,
                           {.dim = 1, .dmask = 1, .lwe = uint8_t(rdna1), .vdata = dst, .ssamp = 2})
                     : rdna1::build_mubuf(
                           rdna1::kBufferLoadDword,
                           {.vdata = dst, .srsrc = 3, .tfe = uint8_t(rdna1), .soffset = 128});
      };
      for (const auto &[younger, wait, hazard] :
           std::array<std::tuple<Younger, int, bool>, 12>{{{Younger::Active, 1, false},
                                                           {Younger::Empty, 1, false},
                                                           {Younger::Suppressed, 1, false},
                                                           {Younger::Empty, -1, true},
                                                           {Younger::Empty, 0, false},
                                                           {Younger::NotIssued, 1, true},
                                                           {Younger::Store, 1, true},
                                                           {Younger::EmptyFlat, 1, false},
                                                           {Younger::EmptyFlat, -1, true},
                                                           {Younger::EmptyFlat, 0, false},
                                                           {Younger::SuppressedFlat, 1, false},
                                                           {Younger::EmptyFlatStore, 1, true}}}) {
        SCOPED_TRACE(testing::Message() << image << "/" << int(younger) << "/" << wait);
        const auto a = fetch(8);
        auto b = fetch(younger == Younger::Suppressed ? uint8_t(rdna1 ? 15 : 16) : uint8_t(4));
        if (younger == Younger::NotIssued)
          b = {S_NOP, S_NOP};
        if (younger == Younger::Store)
          b = rdna1::build_mubuf(rdna1::kBufferStoreDword,
                                 {.vdata = 4, .srsrc = 3, .soffset = 128});
        const bool flat = younger == Younger::EmptyFlat || younger == Younger::SuppressedFlat ||
                          younger == Younger::EmptyFlatStore;
        if (flat)
          b = younger == Younger::EmptyFlatStore
                  ? rdna1::build_flat(rdna1::kFlatStoreDword, {.saddr = 125})
                  : rdna1::build_flat(
                        rdna1::kFlatLoadDword,
                        {.saddr = 125,
                         .vdst = uint8_t(younger == Younger::SuppressedFlat ? 16 : 4)});
        const uint64_t pc = wave->pc + 0x100;
        wave->pc = pc;
        const auto wait_vs = rdna1::build_sopk(rdna1::kSWaitcntVscnt, {.sdst = 125})[0];
        const uint32_t code[] = {a[0],
                                 a[1],
                                 b[0],
                                 b[1],
                                 wait < 0 ? S_NOP : legacy_wait_word(arch, wait, 63),
                                 read[0],
                                 0xbf8c0000u,
                                 wait_vs,
                                 S_ENDPGM};
        f.mem->load_image(reinterpret_cast<const uint8_t *>(code), sizeof(code), pc);
        const auto advance = [&](unsigned word) {
          for (unsigned tick = 0; tick < 32 && wave->pc < pc + 4 * word; ++tick)
            ASSERT_TRUE(cu->step());
          ASSERT_EQ(wave->pc, pc + 4 * word);
        };
        wave->set_exec(1);
        ASSERT_NO_FATAL_FAILURE(advance(2));
        EXPECT_EQ(cu->read_vgpr_storage(wave->vgpr_alloc().base + result, 0),
                  rdna1 ? (image ? 2u : 1u) : 123u);
        wave->set_exec(younger == Younger::Active || younger == Younger::Suppressed ||
                               younger == Younger::SuppressedFlat
                           ? 1
                           : 0);
        ASSERT_NO_FATAL_FAILURE(advance(4));
        const auto &events = race_state.getWaveMemoryEvents();
        const bool active = younger == Younger::Active;
        const bool issues_fetch = younger != Younger::NotIssued && younger != Younger::Store &&
                                  younger != Younger::EmptyFlatStore;
        ASSERT_EQ(events.size(), active ? 2u : 1u);
        const auto &registry = race_state.getDetector()->events();
        EXPECT_EQ(registry.pendingCounterIncrement(events.front(), WaitCounterType::VMCNT),
                  issues_fetch && !active ? 2 : 1);
        EXPECT_EQ(wave->ensure_memory_wait_scoreboard().outstanding(WaitCounterKind::Load),
                  issues_fetch ? 2u : 1u);
        EXPECT_EQ(wave->ensure_memory_wait_scoreboard().outstanding(WaitCounterKind::Ds),
                  unsigned(flat));
        wave->set_exec(1);
        ASSERT_NO_FATAL_FAILURE(advance(5));
        const uint64_t warnings = cu->memory_wait_diagnostic_count();
        const size_t output = sink.str().size();
        ASSERT_NO_FATAL_FAILURE(advance(6));
        EXPECT_EQ(cu->memory_wait_diagnostic_count() != warnings, hazard);
        EXPECT_EQ(sink.str().find("RACE ", output) != std::string::npos, hazard);
        ASSERT_NO_FATAL_FAILURE(advance(8));
      }
    }

    // Compare overlapping writebacks using the same issued buffer and sampler instructions.
    const auto check_waw = [&](std::array<uint32_t, 2> older, std::array<uint32_t, 2> younger,
                               uint64_t younger_exec, bool waited, bool hazard) {
      SCOPED_TRACE(testing::Message()
                   << std::hex << older[0] << '/' << older[1] << " -> " << younger[0] << '/'
                   << younger[1] << " exec=" << younger_exec << " waited=" << waited);
      const uint64_t pc = wave->pc + 0x100;
      wave->pc = pc;
      const uint32_t vm0 = legacy_wait_word(arch, 0, 63);
      const uint32_t code[] = {older[0], older[1], waited ? vm0 : S_NOP, younger[0], younger[1],
                               vm0,      S_ENDPGM};
      f.mem->load_image(reinterpret_cast<const uint8_t *>(code), sizeof(code), pc);
      const auto advance = [&](unsigned word) {
        for (unsigned tick = 0; tick < 32 && wave->pc < pc + 4 * word; ++tick)
          ASSERT_TRUE(cu->step());
        ASSERT_EQ(wave->pc, pc + 4 * word);
      };
      wave->set_exec(1);
      ASSERT_NO_FATAL_FAILURE(advance(3));
      wave->set_exec(younger_exec);
      const auto before = sink.str().size();
      ASSERT_NO_FATAL_FAILURE(advance(5));
      EXPECT_EQ(sink.str().find("access=write", before) != std::string::npos, hazard)
          << sink.str().substr(before);
      ASSERT_NO_FATAL_FAILURE(advance(6));
    };
    const auto fetch = [&](bool sample, uint8_t dst) {
      return sample ? rdna1::build_mimg(
                          rdna1::kImageSampleLz,
                          {.dim = 1, .dmask = 1, .lwe = uint8_t(rdna1), .vdata = dst, .ssamp = 2})
                    : rdna1::build_mubuf(
                          rdna1::kBufferLoadDword,
                          {.vdata = dst, .srsrc = 3, .tfe = uint8_t(rdna1), .soffset = 128});
    };
    for (bool first_sample : {false, true}) {
      const auto first = fetch(first_sample, 8);
      for (bool second_sample : {false, true}) {
        // The younger payload aliases the older RDNA1 status, or the older RDNA2 payload.
        const auto second = fetch(second_sample, result);
        ASSERT_NO_FATAL_FAILURE(check_waw(first, second, 1, false, first_sample != second_sample));
        if (first_sample != second_sample) {
          ASSERT_NO_FATAL_FAILURE(check_waw(first, second, 1, true, false));
          ASSERT_NO_FATAL_FAILURE(check_waw(first, second, 2, false, false));
          ASSERT_NO_FATAL_FAILURE(check_waw(first, fetch(second_sample, 12), 1, false, false));
        }
      }
    }
    cu->write_sgpr(sb + 12, 0x4000); // The D16 controls read the resident page.
    if (rdna1) {
      // The resident buffer keeps its trailing status word, so the younger sample can write it.
      ASSERT_NO_FATAL_FAILURE(check_waw(fetch(false, 8), fetch(true, result), 1, false, false));
    }
    const auto half_sample = rdna1::build_mimg(
        rdna1::kImageSampleLz, {.dim = 1, .dmask = 1, .vdata = 8, .ssamp = 2, .d16 = 1});
    for (bool high : {false, true}) {
      const auto half_buffer =
          rdna1::build_mubuf(high ? rdna1::kBufferLoadShortD16Hi : rdna1::kBufferLoadShortD16,
                             {.vdata = 8, .srsrc = 3, .soffset = 128});
      ASSERT_NO_FATAL_FAILURE(check_waw(half_sample, half_buffer, 1, false, !high));
      ASSERT_NO_FATAL_FAILURE(check_waw(half_buffer, half_sample, 1, false, !high));
    }
    wave->set_address_space({});
    cu->set_gpu_vm(nullptr);
    wave->halt();
  }
}

TEST(RaceDetectorPluginTest, Gfx10IssuedEmptyDsOperationCountsInItsOwnLgkmOrder) {
  enum class Lanes { Active, Empty, Suppressed };
  for (const auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    const bool rdna1 = arch == ROCJITSU_CODE_ARCH_RDNA1;
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, rdna1 ? "rdna1" : "rdna2", 32, 106, 32));
    PluginSinkConfig sinks;
    auto &sink = sinks.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *race = plugin.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    auto *wave = cu->dispatch_wf(0, 0x8000, 106, 16, 32);
    ASSERT_NE(wave, nullptr);
    wave->set_lds_size(256);
    std::array<Wavefront *, 1> waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
    auto &state =
        *static_cast<RaceWavefrontState *>(wave->plugin_state(race->slot_index()))->race_state;
    cu->write_vgpr(wave->vgpr_alloc().base + 4, 0, 4);
    cu->write_vgpr(wave->vgpr_alloc().base + 5, 0, 4);
    wave->lds().write32(wave->lds_base() + 4, 9);
    for (bool first_gds : {false, true}) {
      enum class Second { Lds, Gds, FlatLoad, FlatStore };
      for (auto second_op : {Second::Lds, Second::Gds, Second::FlatLoad, Second::FlatStore}) {
        const bool second_gds = second_op == Second::Gds;
        const bool flat = second_op == Second::FlatLoad || second_op == Second::FlatStore;
        if (second_gds && !rdna1)
          continue; // Ordinary GDS reads are implemented on RDNA1.
        for (const auto &[lanes, lgkm] :
             std::array<std::pair<Lanes, int>, 5>{{{Lanes::Active, 1},
                                                   {Lanes::Empty, 1},
                                                   {Lanes::Suppressed, 1},
                                                   {Lanes::Empty, -1},
                                                   {Lanes::Empty, 0}}}) {
          if (flat && (lanes == Lanes::Active ||
                       (lanes == Lanes::Suppressed && second_op == Second::FlatStore)))
            continue;
          SCOPED_TRACE(testing::Message() << int(arch) << "/" << first_gds << "/" << int(second_op)
                                          << "/" << int(lanes) << "/" << lgkm);
          cu->gws_device().access_gds(arch, 0, [](Lds &gds) { gds.write32(4, 9); });
          const auto first = rdna2::build_ds(first_gds ? rdna2::kDsOrderedCount : rdna2::kDsReadB32,
                                             {.offset0 = uint8_t(first_gds ? 4 : 0),
                                              .gds = uint8_t(first_gds),
                                              .addr = 4,
                                              .vdst = 2});
          auto second = rdna2::build_ds(rdna2::kDsReadB32,
                                        {.gds = uint8_t(second_gds),
                                         .addr = 5,
                                         .vdst = uint8_t(lanes == Lanes::Suppressed ? 16 : 8)});
          if (flat)
            second = second_op == Second::FlatLoad
                         ? rdna2::build_flat(
                               rdna2::kFlatLoadDwordFlat,
                               {.saddr = 125, .vdst = uint8_t(lanes == Lanes::Suppressed ? 16 : 8)})
                         : rdna2::build_flat(rdna2::kFlatStoreDwordFlat, {.saddr = 125});
          const auto read = rdna2::build_vop1(rdna2::kVMovB32Vop1, {.src0 = 258, .vdst = 3})[0];
          const uint32_t exp0 = legacy_wait_word(arch, 63, 63) & ~(7u << 4);
          const uint32_t code[] = {
              first[0],  first[1],    exp0,
              second[0], second[1],   lgkm < 0 ? S_NOP : legacy_wait_word(arch, 63, lgkm),
              read,      0xbf8c0000u, rdna1::build_sopk(rdna1::kSWaitcntVscnt, {.sdst = 125})[0],
              S_ENDPGM};
          const auto pc = wave->pc + 0x100;
          wave->pc = pc;
          f.mem->load_image(reinterpret_cast<const uint8_t *>(code), sizeof(code), pc);
          const auto advance = [&](unsigned word) {
            for (unsigned tick = 0; tick < 32 && wave->pc < pc + 4 * word; ++tick)
              ASSERT_TRUE(cu->step());
            ASSERT_EQ(wave->pc, pc + 4 * word);
          };
          wave->set_exec(1);
          wave->set_m0(first_gds ? 0 : 256);
          ASSERT_NO_FATAL_FAILURE(advance(3));
          EXPECT_EQ(cu->read_vgpr_storage(wave->vgpr_alloc().base + 2, 0), 9u);
          wave->set_exec(lanes == Lanes::Empty ? 0 : 1);
          wave->set_m0(256);
          ASSERT_NO_FATAL_FAILURE(advance(5));
          const auto &events = state.getWaveMemoryEvents();
          const bool active = lanes == Lanes::Active;
          const unsigned younger_events = active ? (second_gds ? 2 : 1) : 0;
          ASSERT_EQ(events.size(), 1 + younger_events);
          const auto &registry = state.getDetector()->events();
          EXPECT_EQ(registry.pendingCounterIncrement(events.front(), WaitCounterType::LGKMCNT),
                    !active && first_gds == second_gds ? 2 : 1);
          EXPECT_EQ(wave->ensure_memory_wait_scoreboard().outstanding(WaitCounterKind::Ds), 2u);
          wave->set_exec(1);
          ASSERT_NO_FATAL_FAILURE(advance(6));
          const auto warnings = cu->memory_wait_diagnostic_count();
          const auto before = sink.str().size();
          ASSERT_NO_FATAL_FAILURE(advance(7));
          const bool hazard = lgkm < 0 || (lgkm == 1 && first_gds != second_gds);
          EXPECT_EQ(cu->memory_wait_diagnostic_count() != warnings, hazard);
          EXPECT_EQ(sink.str().find("RACE ", before) != std::string::npos, hazard);
          ASSERT_NO_FATAL_FAILURE(advance(9));
        }
      }
    }
    wave->halt();
  }
}

TEST(RaceDetectorPluginTest, OrderedCountTracksRestoredExecAndIndependentGdsCounterEvents) {
  for (const auto arch : {"rdna1", "rdna2"}) {
    SCOPED_TRACE(arch);
    PluginFixture fixture;
    ASSERT_NO_FATAL_FAILURE(fixture.initialize(1, arch, 64, 106, 32));
    PluginSinkConfig sinks;
    auto &sink = sinks.emplace<StringSink>();
    fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
    auto observer = std::make_unique<MemoryObservationPlugin>();
    auto *observed = observer.get();
    ASSERT_TRUE(fixture.plugin_group_->add(std::move(observer)));
    auto race_plugin = std::make_unique<RaceDetectorPlugin>();
    auto *race = race_plugin.get();
    ASSERT_TRUE(fixture.plugin_group_->add(std::move(race_plugin)));
    fixture.soc->set_plugin_group(fixture.plugin_group_);
    fixture.plugin_group_->onInit();
    auto *cu = fixture.cu();
    auto *wave = cu->dispatch_wf(0, 0x100, 106, 16, 64);
    ASSERT_NE(wave, nullptr);
    wave->set_m0(0);
    std::array<Wavefront *, 1> waves{wave};
    fixture.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
    const auto *state = static_cast<RaceWavefrontState *>(wave->plugin_state(race->slot_index()));
    ASSERT_NE(state, nullptr);
    auto decoder = Decoder::create(wave->cu().arch());
    for (const auto &[incoming, dst] :
         std::array<std::pair<uint64_t, uint8_t>, 3>{{{6, 2}, {0, 2}, {2, 16}}}) {
      SCOPED_TRACE(testing::Message() << incoming << "/" << unsigned(dst));
      const bool allocated = dst < wave->num_vgprs();
      wave->pc += 0x20; // Race reports are deduplicated by instruction PC.
      wave->set_exec(incoming);
      const auto count =
          rdna2::build_ds(rdna2::kDsOrderedCount, {.gds = 1, .addr = 1, .vdst = dst});
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, count.data()));
      ASSERT_TRUE(inst);
      fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave->pc, *inst, *wave);
      ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
      ASSERT_EQ(test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), *wave),
                VmAccessOutcome::Complete);
      ASSERT_FALSE(observed->accesses.empty());
      const auto &access = observed->accesses.back();
      EXPECT_EQ(access.decoded_space, DecodedMemorySpace::GDS);
      EXPECT_EQ(access.architectural_exec_lane_mask, incoming);
      EXPECT_EQ(access.active_lane_mask, allocated ? incoming : 0);
      EXPECT_EQ(access.valid_lane_mask, allocated ? incoming : 0);
      EXPECT_EQ(access.request_lane_mask, allocated && incoming ? 2u : 0u);
      EXPECT_EQ(access.ordered_count_result_lane_mask,
                std::optional<uint64_t>(allocated ? ~uint64_t{0} : 0));
      ASSERT_EQ(access.issued_counters.size(), 2u);

      // Lane zero was inactive when GDS supplied the broadcast result.
      wave->set_exec(1);
      const auto mov =
          rdna2::build_vop1(rdna2::kVMovB32Vop1, {.src0 = uint16_t(256u + dst), .vdst = 3});
      inst.reset(decode_valid(*decoder, mov.data()));
      ASSERT_TRUE(inst);
      auto read_races = [&] {
        const auto before = sink.str().size();
        fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave->pc + 8, *inst, *wave);
        EXPECT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
        return sink.str().find("RACE ", before) != std::string::npos;
      };
      auto &race = *state->race_state;
      const auto &events = race.getDetector()->events();
      const auto pending_loads = [&] {
        return std::ranges::count_if(race.getVgprMemoryEvents(dst), [&](EventId id) {
          return events.type(id) == MemoryEventType::LDS_TO_VGPR && events.registerLanes(id, dst);
        });
      };
      const auto pending_source = [&](uint32_t lane) {
        return std::ranges::any_of(race.getVgprMemoryEvents(1), [&](EventId id) {
          return isFromVgpr(events.type(id)) &&
                 (events.registerLanes(id, 1) & (uint64_t{1} << lane));
        });
      };
      EXPECT_EQ(pending_loads(), allocated ? 1 : 0);
      EXPECT_EQ(read_races(), allocated);
      execute_wait(*fixture.plugin_group_, *wave, legacy_wait_word(wave->cu().arch(), 63, 0));
      EXPECT_EQ(pending_loads(), 0);
      EXPECT_FALSE(read_races());
      EXPECT_FALSE(pending_source(0));
      EXPECT_EQ(pending_source(1), allocated && incoming);
      EXPECT_FALSE(pending_source(2));
      execute_wait(*fixture.plugin_group_, *wave, 0xbf8c0000u);
      EXPECT_FALSE(pending_source(1));
    }
    wave->halt();
  }
}

TEST(RaceDetectorPluginTest, OrderedCountProtectsOnlyItsContributingVgprLaneUntilExpcnt) {
  for (std::string_view arch : {"rdna1", "rdna2"}) {
    SCOPED_TRACE(arch);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 32, 106, 32));
    PluginSinkConfig sinks;
    auto &sink = sinks.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    auto *wave = cu->dispatch_wf(0, 0, 106, 16, 32);
    ASSERT_NE(wave, nullptr);
    wave->set_m0(0);
    wave->set_lds_size(256);
    std::array<Wavefront *, 1> waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);

    const auto overwrite = [&](std::array<uint32_t, 2> producer, uint32_t issued_exec,
                               unsigned lane, uint8_t reg, uint32_t wait, bool expect_race) {
      SCOPED_TRACE(testing::Message()
                   << issued_exec << "/" << lane << "/v" << int(reg) << "/" << std::hex << wait);
      // Each producer has a different PC because the plugin deduplicates reports.
      const uint64_t pc = wave->pc + 0x100;
      wave->pc = pc;
      const auto mov = rdna2::build_vop1(rdna2::kVMovB32Vop1, {.src0 = 129, .vdst = reg})[0];
      const uint32_t code[] = {producer[0], producer[1], wait, mov, 0xbf8c0000u, S_ENDPGM};
      f.mem->load_image(reinterpret_cast<const uint8_t *>(code), sizeof(code), pc);
      const auto advance = [&](unsigned word) {
        for (unsigned tick = 0; tick < 32 && wave->pc < pc + 4 * word; ++tick)
          ASSERT_TRUE(cu->step());
        ASSERT_EQ(wave->pc, pc + 4 * word);
      };
      wave->set_exec(issued_exec);
      ASSERT_NO_FATAL_FAILURE(advance(3));
      wave->set_exec(uint32_t{1} << lane);
      const size_t before = sink.str().size();
      const auto diagnostics = cu->memory_wait_diagnostic_count();
      ASSERT_NO_FATAL_FAILURE(advance(4));
      EXPECT_EQ(cu->memory_wait_diagnostic_count() != diagnostics, expect_race);
      const auto report = sink.str().substr(before);
      EXPECT_EQ(report.find("RACE ") != std::string::npos, expect_race) << report;
      if (expect_race) {
        EXPECT_NE(report.find("type=VGPR access=write reg=" + std::to_string(reg)),
                  std::string::npos)
            << report;
      }
      EXPECT_EQ(cu->read_vgpr_storage(wave->vgpr_alloc().base + reg, lane), 1u);
      ASSERT_NO_FATAL_FAILURE(advance(5));
    };
    const uint32_t lgkm0 = legacy_wait_word(cu->arch(), 63, 0);
    const uint32_t exp0 = legacy_wait_word(cu->arch(), 63, 63) & ~(7u << 4);
    const auto count = rdna2::build_ds(rdna2::kDsOrderedCount, {.gds = 1, .addr = 1, .vdst = 2});
    ASSERT_NO_FATAL_FAILURE(overwrite(count, 6, 1, 1, S_NOP, true));
    ASSERT_NO_FATAL_FAILURE(overwrite(count, 6, 2, 1, S_NOP, false));
    ASSERT_NO_FATAL_FAILURE(overwrite(count, 6, 0, 1, S_NOP, false));
    ASSERT_NO_FATAL_FAILURE(overwrite(count, 6, 1, 4, S_NOP, false));
    ASSERT_NO_FATAL_FAILURE(overwrite(count, 0, 1, 1, S_NOP, false));
    ASSERT_NO_FATAL_FAILURE(overwrite(count, 6, 1, 1, lgkm0, true));
    ASSERT_NO_FATAL_FAILURE(overwrite(count, 6, 1, 1, exp0, false));
    const auto suppressed =
        rdna2::build_ds(rdna2::kDsOrderedCount, {.gds = 1, .addr = 1, .vdst = 16});
    ASSERT_NO_FATAL_FAILURE(overwrite(suppressed, 6, 1, 1, S_NOP, false));
    // The LDS store snapshots its input at issue.
    wave->set_m0(256);
    const auto vb = wave->vgpr_alloc().base;
    cu->write_vgpr(vb + 4, 1, 4);
    cu->write_vgpr(vb + 4, 2, 8);
    cu->write_vgpr(vb + 1, 1, 23);
    const auto store = rdna2::build_ds(rdna2::kDsWriteB32, {.addr = 4, .data0 = 1});
    ASSERT_NO_FATAL_FAILURE(overwrite(store, 6, 1, 1, S_NOP, false));
    EXPECT_EQ(wave->lds().read32(wave->lds_base() + 4), 23u);
    wave->halt();
  }
}

TEST(RaceDetectorPluginTest, Gfx10OrderedCountAndGdsLoadsHonorWritebackOrder) {
  enum class Operation { OrderedCount, LdsLoad, GdsLoad };
  const auto ds = [](Operation op, uint8_t src) {
    return rdna2::build_ds(op == Operation::OrderedCount ? rdna2::kDsOrderedCount
                                                         : rdna2::kDsReadB32,
                           {.gds = uint8_t(op != Operation::LdsLoad), .addr = src, .vdst = 2});
  };
  for (std::string_view arch : {"rdna1", "rdna2"}) {
    for (auto [first, second] : {std::pair{Operation::OrderedCount, Operation::OrderedCount},
                                 std::pair{Operation::OrderedCount, Operation::LdsLoad},
                                 std::pair{Operation::LdsLoad, Operation::OrderedCount},
                                 std::pair{Operation::OrderedCount, Operation::GdsLoad},
                                 std::pair{Operation::GdsLoad, Operation::OrderedCount}}) {
      // The ordinary GDS load executor is implemented on RDNA1.
      if (arch == "rdna2" && (first == Operation::GdsLoad || second == Operation::GdsLoad))
        continue;
      SCOPED_TRACE(testing::Message() << arch << "/" << int(first) << "/" << int(second));
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 32, 106, 32));
      PluginSinkConfig sinks;
      auto &sink = sinks.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      constexpr uint64_t pc = 0x100;
      auto *cu = f.cu();
      auto *wf = cu->dispatch_wf(0, pc, 106, 16, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_lds_size(256);
      std::array<Wavefront *, 1> waves{wf};
      f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
      const auto vb = wf->vgpr_alloc().base;
      cu->write_vgpr(vb + 4, 0, 4);
      cu->write_vgpr(vb + 5, 1, 4);
      const auto a = ds(first, 4);
      const auto b = ds(second, 5);
      const auto read = rdna2::build_vop1(rdna2::kVMovB32Vop1, {.src0 = 258, .vdst = 3})[0];
      const uint32_t exp0 = legacy_wait_word(cu->arch(), 63, 63) & ~(7u << 4);
      const uint32_t lgkm0 = legacy_wait_word(cu->arch(), 63, 0);
      const uint32_t program[] = {a[0], a[1], exp0, b[0], b[1], exp0, read, lgkm0, read, S_ENDPGM};
      f.mem->load_image(reinterpret_cast<const uint8_t *>(program), sizeof(program), pc);
      const auto run_to = [&](unsigned next_word) {
        for (unsigned tick = 0; tick < 32 && wf->pc < pc + 4 * next_word; ++tick)
          ASSERT_TRUE(cu->step());
        ASSERT_EQ(wf->pc, pc + 4 * next_word);
      };
      wf->set_m0(first == Operation::OrderedCount ? 0 : 8);
      wf->set_exec(1);
      ASSERT_NO_FATAL_FAILURE(run_to(3)); // Release the outgoing data before changing EXEC.
      ASSERT_EQ(cu->memory_wait_diagnostic_count(), 0u);
      ASSERT_EQ(sink.str().find("RACE "), std::string::npos);
      wf->set_m0(second == Operation::OrderedCount ? 0 : 8);
      wf->set_exec(2);
      ASSERT_NO_FATAL_FAILURE(run_to(6));
      const bool cross_type = first == Operation::LdsLoad || second == Operation::LdsLoad;
      EXPECT_EQ(cu->memory_wait_diagnostic_count(), unsigned(cross_type));
      EXPECT_EQ(sink.str().find("access=write") != std::string::npos, cross_type) << sink.str();
      if (first == Operation::OrderedCount && second == Operation::OrderedCount) {
        EXPECT_EQ(cu->read_vgpr_storage(vb + 2, 0), 4u);
        EXPECT_EQ(cu->read_vgpr_storage(vb + 2, 31), 4u);
      }
      // EXPCNT released the inputs. The result is still pending until LGKMCNT.
      ASSERT_NO_FATAL_FAILURE(run_to(7));
      if (!cross_type) {
        EXPECT_GT(cu->memory_wait_diagnostic_count(), 0u);
        EXPECT_NE(sink.str().find("access=read"), std::string::npos) << sink.str();
      }
      const auto reports = cu->memory_wait_diagnostic_count();
      const size_t output = sink.str().size();
      ASSERT_NO_FATAL_FAILURE(run_to(9));
      EXPECT_EQ(cu->memory_wait_diagnostic_count(), reports);
      EXPECT_EQ(sink.str().find("RACE ", output), std::string::npos);
      wf->halt();
    }
  }
}

TEST(RaceDetectorPluginTest, NamedVmcntWaitRetiresMonolithicAndSplitLoadEvents) {
  auto run = [](WaitCounterType event_counter, std::optional<uint32_t> addend) {
    PluginFixture f;
    f.initialize(/*num_wf_slots=*/1, /*arch=*/addend ? "rdna2" : "rdna3", /*wavefront_size=*/32,
                 /*sgprs_per_wf=*/128);
    if (testing::Test::HasFatalFailure())
      return false;
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    EXPECT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/128, /*vgprs=*/256, 32);
    EXPECT_NE(wf, nullptr);
    if (!wf)
      return false;
    wf->set_exec(1u);
    std::array<amdgpu::Wavefront *, 1> waves{wf};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(
        /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
        /*physical_sgpr_count=*/128, waves);

    auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
    state->is_load = true;
    state->num_elems = 1;
    state->dst_reg_base = wf->vgpr_alloc().base;
    state->exec_mask = 1;
    state->wait_counter_type = event_counter;
    TestMemoryInstruction load(std::move(state));
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, load, *wf);
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

    if (addend) {
      cu->write_sgpr(wf->sgpr_alloc().base + 2, *addend);
      execute_wait(*f.plugin_group_, *wf, 0xbc020001u); // s_waitcnt_vmcnt s2, 1.
    } else {
      execute_wait(*f.plugin_group_, *wf, 0xbcfc0000u); // s_waitcnt_vmcnt null, 0.
    }
    f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base, /*lane_mask=*/1);
    return sink.str().find("RACE ") == std::string::npos;
  };

  for (const auto counter : {WaitCounterType::VMCNT, WaitCounterType::LOADCNT}) {
    EXPECT_TRUE(run(counter, std::nullopt));
    EXPECT_TRUE(run(counter, 63));
    EXPECT_FALSE(run(counter, 0));
  }
}

TEST(RaceDetectorPluginTest, UnresolvedOrMixedFlatWholeResultRequiresBothWaitCounterDomains) {
  auto reports_race_after_wait = [](uint64_t shared_lanes, uint8_t vmcnt, uint8_t lgkmcnt) {
    PluginFixture f;
    f.initialize(/*num_wf_slots=*/1);
    if (testing::Test::HasFatalFailure())
      return false;
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    EXPECT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    auto *wf = f.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
    EXPECT_NE(wf, nullptr);
    if (!wf)
      return false;
    wf->set_exec(3u);
    std::array<amdgpu::Wavefront *, 1> waves{wf};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(
        /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
        /*physical_sgpr_count=*/104, waves);

    constexpr uint64_t shared_base = uint64_t{1} << 48;
    wf->set_lds_base(0x200);
    auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
    state->is_load = true;
    state->elem_size = 4;
    state->num_elems = 1;
    state->dst_reg_base = wf->vgpr_alloc().base;
    state->exec_mask = state->lane_mask = 3;
    state->wf_size = 64;
    state->flat_local_lane_mask = shared_lanes;
    std::array<uint64_t, 64> effective_addresses{};
    std::array<uint64_t, 64> original_addresses{};
    for (uint32_t lane = 0; lane < 2; ++lane) {
      state->per_lane_addr[lane] =
          (shared_lanes & (uint64_t{1} << lane) ? wf->lds_base() + 0x40 : 0x1000) + lane * 4;
      original_addresses[lane] =
          (shared_lanes & (uint64_t{1} << lane) ? shared_base + 0x40 : 0x1000) + lane * 4;
      effective_addresses[lane] =
          (shared_lanes & (uint64_t{1} << lane) ? wf->lds_base() + 0x40 : 0x1000) + lane * 4;
    }
    TestMemoryInstruction flat(std::move(state),
                               {{WaitCounterType::VMCNT, MemoryCompletionClass::UNORDERED},
                                {WaitCounterType::LGKMCNT, MemoryCompletionClass::UNORDERED}});
    // Zero denotes an unresolved observation. Both mixed lane orderings retain
    // GLOBAL execution while observations expose effective shared addresses.
    MemoryAccessObservation access;
    if (shared_lanes != 0) {
      access.route = MemoryRoute::GLOBAL;
      access.decoded_space = DecodedMemorySpace::FLAT;
      access.request_lane_mask = access.active_lane_mask = access.valid_lane_mask = 3;
      access.flat_local_lane_mask = shared_lanes;
      access.wavefront_size = wf->wf_size();
      access.addresses = {effective_addresses.data(), wf->wf_size()};
      access.pre_routing_addresses = {original_addresses.data(), wf->wf_size()};
    }
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, flat, *wf);
    f.plugin_group_->onAmdgpuMemoryAccessRouted(access, flat, *wf);

    execute_wait(*f.plugin_group_, *wf, legacy_wait_word(wf->cu().arch(), vmcnt, lgkmcnt));
    f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base, /*lane_mask=*/3);
    return sink.str().find("RACE ") != std::string::npos;
  };

  for (uint64_t shared_lanes : {0u, 1u, 2u}) {
    SCOPED_TRACE(shared_lanes);
    EXPECT_TRUE(reports_race_after_wait(shared_lanes, /*vmcnt=*/0, /*lgkmcnt=*/15));
    EXPECT_TRUE(reports_race_after_wait(shared_lanes, /*vmcnt=*/63, /*lgkmcnt=*/0));
    EXPECT_FALSE(reports_race_after_wait(shared_lanes, /*vmcnt=*/0, /*lgkmcnt=*/0));
  }
}

TEST(RaceDetectorPluginTest, FlatLoadWaitMakesOnlyTheCorrespondingLanesReady) {
  enum class Wait { None, Result, Both };
  for (const bool is_rdna4 : {false, true}) {
    for (const uint64_t shared_lanes : {0u, 1u, 2u, 3u}) {
      for (const bool wait_lds : {false, true}) {
        const uint64_t consumer_lanes = wait_lds ? shared_lanes : 3 & ~shared_lanes;
        if (!consumer_lanes)
          continue;
        for (const auto waited : {Wait::None, Wait::Result, Wait::Both}) {
          SCOPED_TRACE(is_rdna4 ? "rdna4" : "cdna4");
          SCOPED_TRACE(shared_lanes);
          SCOPED_TRACE(consumer_lanes);
          SCOPED_TRACE(static_cast<unsigned>(waited));
          const uint32_t wave_size = is_rdna4 ? 32 : 64;
          const uint32_t sgprs = is_rdna4 ? 128 : 104;
          PluginFixture f;
          ASSERT_NO_FATAL_FAILURE(f.initialize(1, is_rdna4 ? "rdna4" : "cdna4", wave_size, sgprs));
          PluginSinkConfig sink_config;
          auto &sink = sink_config.emplace<StringSink>();
          f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
          ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
          f.soc->set_plugin_group(f.plugin_group_);
          f.plugin_group_->onInit();
          auto *cu = f.cu();
          auto *wf = cu->dispatch_wf(0, 0x100, sgprs, 256, wave_size);
          ASSERT_NE(wf, nullptr);
          wf->set_exec(3);
          std::array<amdgpu::Wavefront *, 1> waves{wf};
          f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, sgprs, waves);

          auto decoder = Decoder::create(cu->arch());
          std::unique_ptr<Instruction> load;
          if (is_rdna4) {
            const auto words = rdna4::build_vflat(
                rdna4::kFlatLoadB32Vflat, {.saddr = amdgpu::kModernNullSelector, .vdst = 8});
            load.reset(decode_valid(*decoder, words.data()));
          } else {
            const auto words =
                cdna4::build_flat(cdna4::kFlatLoadDwordFlat, {.saddr = 0x7F, .vdst = 8});
            load.reset(decode_valid(*decoder, words.data()));
          }
          ASSERT_NE(load, nullptr);

          // Model routing after aperture resolution. Only uniform shared
          // requests use LOCAL; all shared addresses are rewritten into LDS.
          constexpr uint64_t shared_base = uint64_t{1} << 48;
          wf->set_lds_base(0x200);
          const bool local_route = shared_lanes == 3;
          auto data = std::make_unique<VectorMemState>(local_route ? LOCAL_MEM : GLOBAL_MEM);
          data->is_load = true;
          data->exec_mask = data->lane_mask = 3;
          data->wf_size = wave_size;
          data->elem_size = 4;
          data->num_elems = 1;
          data->dst_reg_base = wf->vgpr_alloc().base + 8;
          data->flat_local_lane_mask = local_route ? 0 : shared_lanes;
          std::array<uint64_t, 64> original_addresses{};
          std::array<uint64_t, 64> effective_addresses{};
          for (uint32_t lane = 0; lane < 2; ++lane) {
            original_addresses[lane] =
                (shared_lanes & (uint64_t{1} << lane) ? shared_base + 0x40 : 0x1000) + lane * 4;
            effective_addresses[lane] =
                (shared_lanes & (uint64_t{1} << lane) ? wf->lds_base() + 0x40 : 0x1000) + lane * 4;
            data->per_lane_addr[lane] = effective_addresses[lane];
          }
          cu->set_apertures(shared_base, shared_base + UINT32_MAX, 0, 0);
          for (unsigned lane = 0; lane < 2; ++lane) {
            cu->write_vgpr(wf->vgpr_alloc().base, lane, original_addresses[lane]);
            cu->write_vgpr(wf->vgpr_alloc().base + 1, lane, original_addresses[lane] >> 32);
          }
          cu->track_memory_wait(*load, *wf);
          load->set_data(std::move(data));
          MemoryAccessObservation access;
          access.route = local_route ? MemoryRoute::LOCAL : MemoryRoute::GLOBAL;
          access.decoded_space = DecodedMemorySpace::FLAT;
          access.request_lane_mask = access.active_lane_mask = access.valid_lane_mask = 3;
          access.flat_local_lane_mask = shared_lanes;
          access.wavefront_size = wave_size;
          access.normalized_to_local = local_route;
          access.addresses = {effective_addresses.data(), wave_size};
          if (shared_lanes != 0)
            access.pre_routing_addresses = {original_addresses.data(), wave_size};
          f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, *load, *wf);
          f.plugin_group_->onAmdgpuMemoryAccessRouted(access, *load, *wf);
          auto &core = wf->ensure_memory_wait_scoreboard();
          unsigned core_reports = 0;
          core.bind(0x200, &core_reports,
                    [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });

          const auto wait_zero = [&](bool lds) {
            const uint32_t word =
                is_rdna4 ? (lds ? 0xbfc60000u : 0xbfc00000u) : (lds ? 0xbf8cc07fu : 0xbf8c0f70u);
            execute_wait(*f.plugin_group_, *wf, word);
            core.wait(lds ? WaitCounterKind::Ds : WaitCounterKind::Load, 0);
          };
          if (waited != Wait::None)
            wait_zero(wait_lds);
          if (waited == Wait::Both)
            wait_zero(!wait_lds);
          core.access({RegClass::VGPR, 8, 1}, consumer_lanes, 0xF, false);
          f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + 8, consumer_lanes);

          const bool missing_wait = waited == Wait::None;
          EXPECT_EQ(core_reports != 0, missing_wait);
          EXPECT_EQ(sink.str().find("RACE ") != std::string::npos, missing_wait);
        }
      }
    }
  }
}

TEST(RaceDetectorPluginTest, ParameterLoadReadsThreeCoefficientsAndWritesFourLanes) {
  constexpr std::array coefficient_addresses{128, 144, 160};
  for (const auto arch : {"rdna3", "rdna3_5"}) {
    for (const int store_address : {0, 128, 132, 144, 160}) {
      SCOPED_TRACE(testing::Message() << arch << " store=" << store_address);
      PluginFixture fixture;
      ASSERT_NO_FATAL_FAILURE(fixture.initialize(2, arch));
      PluginSinkConfig sinks;
      auto &sink = sinks.emplace<StringSink>();
      auto plugin = std::make_unique<RaceDetectorPlugin>();
      auto *detector = plugin.get();
      fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
      ASSERT_TRUE(fixture.plugin_group_->add(std::move(plugin)));
      fixture.soc->set_plugin_group(fixture.plugin_group_);
      fixture.plugin_group_->onInit();
      auto *cu = fixture.cu();
      std::array waves{cu->dispatch_wf(0, 0x1000, 32, 32), cu->dispatch_wf(0, 0x2000, 32, 32)};
      for (auto *wave : waves) {
        ASSERT_NE(wave, nullptr);
        wave->set_lds_base(0);
        wave->set_lds_size(256);
        wave->set_exec(1);
      }
      fixture.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, 64, waves);
      const auto decoder = Decoder::create(cu->arch());
      const auto issue = [&](auto words, amdgpu::Wavefront &wave) {
        auto decoded = decoder->decode_window(words);
        ASSERT_TRUE(decoded.succeeded());
        auto instruction = std::move(decoded).value();
        fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave.pc, *instruction, wave);
        ASSERT_TRUE(cu->execute_instruction(instruction.get(), wave).succeeded());
        test::ComputeUnitTestAccess::route_memory_inst(*cu, instruction.release(), wave);
      };
      cu->write_vgpr(waves[0]->vgpr_alloc().base, 0, store_address);
      ASSERT_NO_FATAL_FAILURE(
          issue(rdna3::build_ds(rdna3::kDsStoreB32Ds, {.addr = 0, .data0 = 1}), *waves[0]));
      waves[1]->set_m0(128);
      ASSERT_NO_FATAL_FAILURE(
          issue(rdna3::build_ldsdir(rdna3::kLdsParamLoadLdsdir, {.vdst = 6}), *waves[1]));
      EXPECT_EQ(sink.str().find("Race on LDS byte ") != std::string::npos,
                std::ranges::binary_search(coefficient_addresses, store_address))
          << sink.str();
      auto *state =
          static_cast<RaceWavefrontState *>(waves[1]->plugin_state(detector->slot_index()));
      ASSERT_NE(state, nullptr);
      const auto &events = state->race_state->getDetector()->events();
      const EventId load{1};
      ASSERT_TRUE(events.contains(load));
      EXPECT_EQ(events.execMask(load), 15u);
      const auto &reads = events.ldsIntervals(load);
      EXPECT_TRUE(
          std::ranges::equal(reads, coefficient_addresses, [](const auto &range, int address) {
            return range.start == address && range.end == address + 4;
          }));
    }
  }
}

TEST(RaceDetectorPluginTest, Gfx11WritebacksRespectImageAndLdsClasses) {
  enum Op { Flat, Image, Sample, Bvh, Direct, Ds, Lod, Resinfo, Param };
  // Both GFX11 profiles encode these operations and their field layouts identically.
  const auto encode = [](Op op, uint8_t dst) -> std::vector<uint32_t> {
    if (op == Direct || op == Param) {
      const auto words = rdna3::build_ldsdir(op == Direct ? rdna3::kLdsDirectLoadLdsdir
                                                          : rdna3::kLdsParamLoadLdsdir,
                                             {.vdst = dst, .wait_vdst = 15});
      return {words.begin(), words.end()};
    }
    if (op == Ds) {
      const auto words = rdna3::build_ds(rdna3::kDsLoadB32Ds, {.vdst = dst});
      return {words.begin(), words.end()};
    }
    if (op == Flat) {
      const auto words = rdna3::build_flat(rdna3::kFlatLoadB32Flat, {.saddr = 124, .vdst = dst});
      return {words.begin(), words.end()};
    }
    const auto opcode = op == Image    ? rdna3::kImageLoadMimg
                        : op == Sample ? rdna3::kImageSampleLzMimg
                        : op == Bvh    ? rdna3::kImageBvhIntersectRayMimg
                        : op == Lod    ? rdna3::kImageGetLodMimg
                                       : rdna3::kImageGetResinfoMimg;
    const auto words =
        rdna3::build_mimg(opcode, {.dim = uint8_t(op != Bvh),
                                   .unorm = 1,
                                   .dmask = uint8_t(op == Bvh ? 15 : 1),
                                   .r128 = uint8_t(op == Bvh),
                                   .vaddr = 16,
                                   .vdata = dst,
                                   .srsrc = uint8_t(op == Bvh ? 3 : 0),
                                   .ssamp = uint8_t(op == Lod || op == Sample ? 2 : 0)});
    return {words.begin(), words.end()};
  };
  const auto pipeline = [](Op op, unsigned lane, unsigned shared_lanes) {
    switch (op) {
    case Flat:
      return (shared_lanes >> lane) & 1 ? Ds : Image;
    case Lod:
      return Sample;
    case Resinfo:
      return Image;
    case Param:
      return Direct;
    default:
      return op;
    }
  };
  constexpr Op ops[]{
      // FLAT writeback follows the address space selected by each lane.
      Flat, Direct, Ds, Image, Sample, Resinfo, Param,
  };
  constexpr uint64_t shared = uint64_t{2} << 32, global = uint64_t{8} << 32;
  for (const auto arch : {"rdna3", "rdna3.5"}) {
    for (const auto a : ops) {
      for (const auto b : ops) {
        for (const unsigned shared_lanes : {0u, 5u, 15u}) {
          if (shared_lanes && a != Flat && b != Flat)
            continue;
          enum Wait { None, Full, Partial };
          for (auto wait : {None, Full, Partial}) {
            const auto has_vmcnt_result = [](Op op) {
              return op == Image || op == Sample || op == Bvh || op == Resinfo || op == Lod;
            };
            if (wait == Partial && (!has_vmcnt_result(a) || !has_vmcnt_result(b)))
              continue;
            SCOPED_TRACE(testing::Message() << arch << " ops=" << a << ',' << b
                                            << " shared=" << shared_lanes << " wait=" << wait);
            PluginFixture f;
            ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 32, 128, 32));
            PluginSinkConfig sinks;
            auto &sink = sinks.emplace<StringSink>();
            f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
            ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
            auto observer = std::make_unique<MemoryObservationPlugin>();
            const auto *observed = observer.get();
            ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
            f.soc->set_plugin_group(f.plugin_group_);
            f.plugin_group_->onInit();
            auto *cu = f.cu();
            constexpr uint64_t pc = 0x8000;
            cu->set_apertures(shared, shared + UINT32_MAX, 0, 0);
            auto *wf = cu->dispatch_wf(0, pc, 128, 32, 32);
            ASSERT_NE(wf, nullptr);
            wf->set_lds_size(1024);
            wf->set_exec(15);
            wf->set_m0(0);
            wf->set_status_raw(wf->status_raw() |
                               (1u << 25)); // LDS_PARAM_RDY from the pixel stage.
            std::array<Wavefront *, 1> waves{wf};
            f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 128, waves);
            const auto sb = wf->sgpr_alloc().base;
            cu->write_sgpr(sb, 0x4000 >> 8);
            cu->write_sgpr(sb + 1, 20u << 20);
            cu->write_sgpr(sb + 3, (9u << 28) | 4); // 1x1 R32_UINT image, select X.
            cu->write_sgpr(sb + 4, 63);
            cu->write_sgpr(sb + 12, 0x6000 >> 8);
            cu->write_sgpr(sb + 15, 8u << 28); // One triangle node in the BVH resource.
            f.mem->write32(0x4000, 0x1234);
            for (unsigned lane = 0; lane < 4; ++lane)
              wf->debug_write_vgpr(1, lane, (shared_lanes & (1u << lane) ? shared : global) >> 32);
            wf->lds().write32(0, 0xabcd);
            f.mem->write32(global, 0x4321);
            auto words = encode(a, 8);
            if (wait == Partial) {
              // A second VMEM read establishes the older result under vmcnt(1).
              auto younger = encode(b, 12);
              words.insert(words.end(), younger.begin(), younger.end());
              words.push_back(0xbf890407u); // s_waitcnt vmcnt(1) lgkmcnt(0)
              words.push_back(rdna3::build_vop1(rdna3::kVMovB32Vop1, {.src0 = 264, .vdst = 6})[0]);
            } else {
              words.push_back(wait == Full ? rdna3::build_sopp(rdna3::kSWaitcntSopp)[0]
                                           : rdna3::build_sopp(rdna3::kSNopSopp)[0]);
              const auto second = encode(b, 8);
              words.insert(words.end(), second.begin(), second.end());
            }
            f.mem->load_image(reinterpret_cast<const uint8_t *>(words.data()), words.size() * 4,
                              pc);
            for (unsigned tick = 0; tick < 32 && wf->pc < pc + words.size() * 4; ++tick)
              ASSERT_TRUE(cu->step());
            ASSERT_EQ(wf->pc, pc + words.size() * 4);
            const auto routed = [](Op op) { return op != Resinfo && op != Lod; };
            ASSERT_EQ(observed->accesses.size(), unsigned(routed(a)) + unsigned(routed(b)));
            for (const auto &access : observed->accesses)
              EXPECT_NE(access.request_lane_mask, 0u);
            bool race = false;
            for (unsigned lane = 0; lane < 4; ++lane) {
              if ((a == Param || b == Param) && lane == 3)
                continue;
              race |= pipeline(a, lane, shared_lanes) != pipeline(b, lane, shared_lanes);
            }
            EXPECT_EQ(sink.str().find("Race on VGPR v8") != std::string::npos,
                      race && wait == None);
            EXPECT_EQ(cu->memory_wait_diagnostic_count() != 0, race && wait == None);
            wf->halt();
          }
        }
      }
    }
  }
}

TEST(RaceDetectorPluginTest, Gfx11FlatWriteOrderFollowsThePipelineOfEachLane) {
  constexpr uint64_t shared = uint64_t{2} << 32, global = uint64_t{8} << 32, pc = 0x8000;
  enum Op { Flat, Ds, Read };
  // The two masks select LDS for each FLAT access. Native DS always uses LDS.
  struct Case {
    Op first, second;
    uint32_t first_shared, second_shared;
    bool wait_vm, wait_ds;
    int race_lane;
    uint64_t second_exec = 15;
  };
  for (bool gfx115 : {false, true}) {
    for (const auto c :
         {Case{Flat, Flat, 15, 15, false, false, -1}, Case{Flat, Flat, 5, 5, false, false, -1},
          Case{Flat, Flat, 5, 7, false, false, 1}, Case{Flat, Flat, 5, 7, true, false, -1},
          Case{Flat, Flat, 5, 7, false, true, 1}, Case{Flat, Ds, 15, 0, false, false, -1},
          Case{Flat, Ds, 15, 0, false, false, -1, 5}, Case{Ds, Flat, 0, 15, false, false, -1},
          Case{Flat, Ds, 5, 0, false, false, 1}, Case{Ds, Flat, 0, 5, false, false, 1},
          Case{Flat, Read, 15, 0, false, true, 0}, Case{Flat, Read, 0, 0, true, false, 0},
          Case{Flat, Read, 5, 0, true, true, -1}}) {
      SCOPED_TRACE(testing::Message() << "gfx115=" << gfx115 << " ops=" << c.first << c.second
                                      << " LDS masks=" << c.first_shared << ',' << c.second_shared
                                      << " waits=" << c.wait_vm << c.wait_ds);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(1, gfx115 ? "rdna3.5" : "rdna3", 32, 128, 32));
      PluginSinkConfig sinks;
      auto &sink = sinks.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      auto *cu = f.cu();
      cu->set_apertures(shared, shared + UINT32_MAX, 0, 0);
      auto *wf = cu->dispatch_wf(0, pc, 128, 32, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_lds_size(1024);
      wf->set_exec(15);
      std::array<Wavefront *, 1> waves{wf};
      f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 128, waves);
      const uint32_t vgpr = wf->vgpr_alloc().base;
      const auto set_routes = [&](uint32_t mask) {
        for (unsigned lane = 0; lane < 4; ++lane)
          wf->debug_write_vgpr(1, lane, (mask & (1u << lane) ? shared : global) >> 32);
      };
      const auto flat =
          gfx115 ? rdna3_5::build_flat(rdna3_5::kFlatLoadB32Flat, {.saddr = 124, .vdst = 2})
                 : rdna3::build_flat(rdna3::kFlatLoadB32Flat, {.saddr = 124, .vdst = 2});
      const auto ds = gfx115 ? rdna3_5::build_ds(rdna3_5::kDsLoadB32Ds, {.vdst = 2})
                             : rdna3::build_ds(rdna3::kDsLoadB32Ds, {.vdst = 2});
      const auto read = gfx115
                            ? rdna3_5::build_vop1(rdna3_5::kVMovB32Vop1, {.src0 = 258, .vdst = 3})
                            : rdna3::build_vop1(rdna3::kVMovB32Vop1, {.src0 = 258, .vdst = 3});
      const auto first = c.first == Flat ? flat : ds;
      const auto second = c.second == Flat ? flat : ds;
      const uint32_t insns[] = {
          first[0],
          first[1],
          legacy_wait_word(cu->arch(), c.wait_vm ? 0 : 63, c.wait_ds ? 0 : 63),
          c.second == Read ? read[0] : second[0],
          second[1],
          legacy_wait_word(cu->arch(), 63, 0),
          read[0],
          read[0]};
      f.mem->load_image(reinterpret_cast<const uint8_t *>(insns), sizeof(insns), pc);
      wf->lds().write32(0, 0xabcd);
      f.mem->write32(global, 0x1234);
      set_routes(c.first_shared);
      ASSERT_TRUE(cu->step());
      EXPECT_EQ(wf->pc, pc + 8);
      ASSERT_TRUE(cu->step());
      set_routes(c.second_shared);
      wf->set_exec(c.second_exec);
      const bool check_replacement = c.first == Flat && c.second == Ds && c.first_shared == 15;
      if (check_replacement) {
        wf->lds().write32(4, 0x8765);
        for (unsigned lane = 0; lane < 4; ++lane)
          wf->debug_write_vgpr(0, lane, 4);
      }
      ASSERT_TRUE(cu->step());
      EXPECT_EQ(wf->pc, pc + (c.second == Read ? 16 : 20));
      EXPECT_EQ(sink.str().find("Race on VGPR v2") != std::string::npos, c.race_lane >= 0);
      if (c.race_lane >= 0) {
        EXPECT_NE(sink.str().find(std::format("wave 0, lane {}]", c.race_lane)), std::string::npos);
      }
      EXPECT_EQ(cu->memory_wait_diagnostic_count() != 0, c.race_lane >= 0);
      for (unsigned lane = 0; lane < 4; ++lane) {
        const bool from_lds =
            c.second == Ds || (c.second == Read ? c.first_shared : c.second_shared) & (1u << lane);
        EXPECT_EQ(cu->read_vgpr_storage(vgpr + (c.second == Read ? 3 : 2), lane),
                  check_replacement && (c.second_exec & (1u << lane)) ? 0x8765u
                  : from_lds                                          ? 0xabcdu
                                                                      : 0x1234u);
      }
      if (check_replacement) {
        ASSERT_TRUE(cu->step()); // Wait for LDS; the FLAT VM position remains outstanding.
        auto &scoreboard = *wf->memory_wait_scoreboard();
        EXPECT_EQ(scoreboard.outstanding(WaitCounterKind::Load), 1u);
        ASSERT_TRUE(cu->step()); // Consume the replacement on its lanes.
        EXPECT_EQ(cu->memory_wait_diagnostic_count(), 0u);
        EXPECT_EQ(sink.str().find("Race on VGPR v2"), std::string::npos);
        if (const uint64_t uncovered = 15 & ~c.second_exec) {
          wf->set_exec(uncovered);
          ASSERT_TRUE(cu->step());
          EXPECT_NE(cu->memory_wait_diagnostic_count(), 0u);
          EXPECT_NE(sink.str().find("Race on VGPR v2"), std::string::npos);
        }
      }
      wf->halt();
    }
  }
}

TEST(RaceDetectorPluginTest, FlatResultsRequireTargetCompletionCounters) {
  enum class Space { Global, Lds, Scratch, GlobalAndScratch };
  for (const std::string_view arch : {"cdna2", "cdna4", "rdna1", "rdna2", "rdna4", "cdna5"}) {
    const bool gfx10 = arch == "rdna1" || arch == "rdna2";
    const bool cdna_legacy = arch == "cdna2" || arch == "cdna4";
    const bool split = arch == "rdna4" || arch == "cdna5";
    const uint32_t wave_size = cdna_legacy ? 64 : 32;
    const uint32_t sgprs = cdna_legacy ? 104 : 128;
    for (const auto space : {Space::Global, Space::Lds, Space::Scratch, Space::GlobalAndScratch}) {
      for (const unsigned width : {1u, 4u}) {
        for (const uint64_t mask : {uint64_t{0}, uint64_t{0xA000'0000'A000'0008}, ~uint64_t{0}}) {
          const uint64_t exec = mask & (~uint64_t{0} >> (64 - wave_size));
          // None, VM zero, LGKM zero, both zero, and a nonzero wait on both.
          for (const unsigned waits : {0u, 1u, 2u, 3u, 4u}) {
            SCOPED_TRACE(static_cast<unsigned>(space));
            SCOPED_TRACE(arch);
            SCOPED_TRACE(width);
            SCOPED_TRACE(exec);
            SCOPED_TRACE(waits);
            PluginFixture f;
            ASSERT_NO_FATAL_FAILURE(f.initialize(1, std::string(arch), wave_size, sgprs));
            PluginSinkConfig sink_config;
            auto &sink = sink_config.emplace<StringSink>();
            auto plugin = std::make_unique<RaceDetectorPlugin>();
            auto *plugin_ptr = plugin.get();
            f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
            ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
            f.soc->set_plugin_group(f.plugin_group_);
            f.plugin_group_->onInit();

            constexpr uint64_t kShared = 0x0001'0000'0000'0000ULL;
            constexpr uint64_t kPrivate = 0x0002'0000'0000'0000ULL;
            constexpr uint64_t kGlobal = 0x400000;
            constexpr uint64_t kScratch = 0x800000;
            constexpr uint32_t kCanary = 0xBAD0BAD0;
            auto *cu = f.cu();
            cu->set_apertures(kShared, kShared + 0xFFFF, kPrivate, kPrivate + 0xFFFF);
            auto *wf = cu->dispatch_wf(0, 0x100, sgprs, 256, wave_size);
            ASSERT_NE(wf, nullptr);
            wf->set_exec(exec);
            const uint32_t lds_size = wave_size * 16;
            wf->set_lds_base(cu->allocate_lds(lds_size));
            wf->set_lds_size(lds_size);
            wf->set_scratch_base(kScratch);
            wf->set_scratch_lane_size(256);
            std::array<amdgpu::Wavefront *, 1> waves{wf};
            f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, sgprs, waves);
            const uint32_t vb = wf->vgpr_alloc().base;
            for (uint32_t lane = 0; lane < wave_size; ++lane) {
              const bool scratch =
                  space == Space::Scratch || (space == Space::GlobalAndScratch && (lane & 1));
              // CDNA5 embeds the lane in the pointer; older targets use the
              // private aperture and obtain the lane from the issuing thread.
              const uint64_t private_address =
                  arch == "cdna5" ? kScratch + (uint64_t{lane} << 52) + 0x20 : kPrivate + 0x20;
              const uint64_t address = space == Space::Lds ? kShared + lane * 16
                                       : scratch           ? private_address
                                                           : kGlobal + lane * 16;
              cu->write_vgpr(vb, lane, static_cast<uint32_t>(address));
              cu->write_vgpr(vb + 1, lane, static_cast<uint32_t>(address >> 32));
              for (unsigned part = 0; part < width; ++part) {
                const uint32_t value = 1000 + lane * 4 + part;
                cu->write_vgpr(vb + 8 + part, lane, kCanary);
                if (space == Space::Lds) {
                  cu->lds().write32(wf->lds_base() + lane * 16 + part * 4, value);
                } else {
                  const uint64_t physical = !scratch
                                                ? address + part * 4
                                                : kScratch + (8 + part) * wave_size * 4 + lane * 4;
                  f.mem->load_image(reinterpret_cast<const uint8_t *>(&value), 4, physical);
                }
              }
            }

            auto decoder = Decoder::create(cu->arch());
            std::unique_ptr<Instruction> load;
            const auto decode = [&](const auto &words) {
              load.reset(decode_valid(*decoder, words.data()));
            };
            if (arch == "cdna2")
              decode(cdna2::build_flat(width == 1 ? cdna2::kFlatLoadDwordFlat
                                                  : cdna2::kFlatLoadDwordx4Flat,
                                       {.addr = 0, .saddr = 0x7F, .vdst = 8}));
            else if (arch == "cdna4")
              decode(cdna4::build_flat(width == 1 ? cdna4::kFlatLoadDwordFlat
                                                  : cdna4::kFlatLoadDwordx4Flat,
                                       {.addr = 0, .saddr = 0x7F, .vdst = 8}));
            else if (gfx10)
              decode(rdna2::build_flat(width == 1 ? rdna2::kFlatLoadDwordFlat
                                                  : rdna2::kFlatLoadDwordx4Flat,
                                       {.addr = 0, .saddr = 0x7F, .vdst = 8}));
            else if (arch == "rdna4")
              decode(rdna4::build_vflat(
                  width == 1 ? rdna4::kFlatLoadB32Vflat : rdna4::kFlatLoadB128Vflat,
                  {.saddr = amdgpu::kModernNullSelector, .vdst = 8, .vaddr = 0}));
            else
              decode(cdna5::build_vflat(
                  width == 1 ? cdna5::kFlatLoadB32Vflat : cdna5::kFlatLoadB128Vflat,
                  {.saddr = amdgpu::kModernNullSelector, .vdst = 8, .vaddr = 0}));
            ASSERT_NE(load, nullptr);
            wf->ensure_memory_wait_scoreboard().check_instruction(*load, *wf);
            cu->track_memory_wait(*load, *wf);
            f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, *load, *wf);
            ASSERT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
            test::ComputeUnitTestAccess::route_memory_inst(*cu, load.release(), *wf);
            // Inspect eager values without making these assertions instruction consumers.
            for (unsigned part = 0; part < width; ++part)
              for (uint32_t lane = 0; lane < wave_size; ++lane)
                EXPECT_EQ(cu->read_vgpr_storage(vb + 8 + part, lane),
                          exec & (uint64_t{1} << lane) ? 1000 + lane * 4 + part : kCanary);

            auto *plugin_state =
                static_cast<RaceWavefrontState *>(wf->plugin_state(plugin_ptr->slot_index()));
            ASSERT_NE(plugin_state, nullptr);
            auto &race = *plugin_state->race_state;
            auto &core = wf->ensure_memory_wait_scoreboard();
            std::vector<MemoryWaitScoreboard::Hazard> hazards;
            core.bind(0x200, &hazards, [](void *p, const auto &hazard) {
              static_cast<decltype(hazards) *>(p)->push_back(hazard);
            });
            const uint8_t vmcnt = waits == 4 ? 1 : (waits & 1) ? 0 : 63;
            const uint8_t ds_max = cdna_legacy ? 15 : 63;
            const uint8_t lgkmcnt = waits == 4 ? 1 : (waits & 2) ? 0 : ds_max;
            if (split) {
              execute_wait(*f.plugin_group_, *wf, 0xbfc00000u | vmcnt);
              execute_wait(*f.plugin_group_, *wf, 0xbfc60000u | lgkmcnt);
            } else {
              execute_wait(*f.plugin_group_, *wf,
                           legacy_wait_word(wf->cu().arch(), vmcnt, lgkmcnt));
            }
            if (vmcnt != 63)
              core.wait(WaitCounterKind::Load, vmcnt);
            if (lgkmcnt != ds_max)
              core.wait(WaitCounterKind::Ds, lgkmcnt);

            // Inactive consumers do not acquire a dependency on active lanes.
            race.checkVgprReadLanes(8, ~exec, 0xF);
            core.access({RegClass::VGPR, 8, 1}, ~exec, 0xF, false);
            EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
            EXPECT_TRUE(hazards.empty());
            for (unsigned part = 0; part < width; ++part) {
              race.checkVgprReadLanes(8 + part, exec, 0xF);
              core.access({RegClass::VGPR, static_cast<uint16_t>(8 + part), 1}, exec, 0xF, false);
            }
            const bool missing = exec && (gfx10                 ? vmcnt != 0 || lgkmcnt != 0
                                          : space == Space::Lds ? lgkmcnt != 0
                                                                : vmcnt != 0);
            EXPECT_EQ(sink.str().find("RACE ") != std::string::npos, missing);
            EXPECT_EQ(!hazards.empty(), missing);
          }
        }
      }
    }
  }
}

TEST(RaceDetectorPluginTest, EmptyMemoryIssuesOrderOlderResultsOnGfx11AndGfx12) {
  enum class Route { Flat, Global, Lds };
  enum class Younger { None, ActiveLoad, EmptyLoad, EmptyStore };
  for (std::string_view arch : {"rdna3", "rdna3.5", "rdna4", "cdna5"}) {
    const bool split = arch == "rdna4" || arch == "cdna5";
    for (const auto route : {Route::Flat, Route::Global, Route::Lds}) {
      for (const auto younger :
           {Younger::None, Younger::ActiveLoad, Younger::EmptyLoad, Younger::EmptyStore}) {
        if (route != Route::Flat && younger == Younger::EmptyStore)
          continue;
        for (const uint16_t threshold : {0, 1}) {
          SCOPED_TRACE(testing::Message()
                       << arch << "/" << int(route) << "/" << int(younger) << "/" << threshold);
          PluginFixture f;
          ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 32, 128, 32));
          PluginSinkConfig sinks;
          auto &sink = sinks.emplace<StringSink>();
          f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
          auto plugin = std::make_unique<RaceDetectorPlugin>();
          const auto *race = plugin.get();
          ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
          f.soc->set_plugin_group(f.plugin_group_);
          f.plugin_group_->onInit();

          constexpr uint64_t global = 0x400000, shared = 0x0001'0000'0000'0000;
          auto *cu = f.cu();
          cu->set_apertures(shared, shared + UINT32_MAX, 0, 0);
          auto *wf = cu->dispatch_wf(0, 0x8000, 128, 32, 32);
          ASSERT_NE(wf, nullptr);
          wf->set_lds_size(256);
          std::array<Wavefront *, 1> waves{wf};
          f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 128, waves);
          wf->debug_write_vgpr(0, 0, global);
          wf->debug_write_vgpr(0, 1, route == Route::Global ? global + 4 : 0);
          wf->debug_write_vgpr(1, 1, route == Route::Flat ? shared >> 32 : 0);
          wf->debug_write_vgpr(2, 1, 4);
          f.mem->write32(global, 10);
          f.mem->write32(global + 4, 11);
          wf->lds().write32(wf->lds_base(), route == Route::Lds ? 10 : 11);
          wf->lds().write32(wf->lds_base() + 4, 11);

          std::vector<uint32_t> code;
          const auto append = [&](const auto &words) {
            code.insert(code.end(), words.begin(), words.end());
          };
          const auto load = [&](uint8_t destination) {
            if (route == Route::Lds) {
              if (split)
                append(rdna4::build_vds(rdna4::kDsLoadB32Vds, {.addr = 2, .vdst = destination}));
              else
                append(rdna3::build_ds(rdna3::kDsLoadB32Ds, {.addr = 2, .vdst = destination}));
            } else if (route == Route::Global) {
              if (split)
                append(rdna4::build_vglobal(
                    rdna4::kGlobalLoadB32Vglobal,
                    {.saddr = amdgpu::kModernNullSelector, .vdst = destination}));
              else
                append(rdna3::build_flat(rdna3::kFlatLoadB32Flat,
                                         {.seg = 2, .saddr = 124, .vdst = destination}));
            } else if (split)
              append(
                  rdna4::build_vflat(rdna4::kFlatLoadB32Vflat,
                                     {.saddr = amdgpu::kModernNullSelector, .vdst = destination}));
            else
              append(
                  rdna3::build_flat(rdna3::kFlatLoadB32Flat, {.saddr = 124, .vdst = destination}));
          };
          if (route != Route::Flat)
            load(7);
          const size_t empty_first = code.size();
          load(8);
          const size_t older_end = code.size();
          if (younger == Younger::ActiveLoad || younger == Younger::EmptyLoad)
            load(9);
          if (younger == Younger::EmptyStore) {
            if (split)
              append(rdna4::build_vflat(rdna4::kFlatStoreB32Vflat,
                                        {.saddr = amdgpu::kModernNullSelector, .vsrc = 9}));
            else
              append(rdna3::build_flat(rdna3::kFlatStoreB32Flat, {.data = 9, .saddr = 124}));
          }
          // A FLAT store also issues to DS, but uses the store counter instead of the load counter.
          const uint16_t load_wait = younger == Younger::EmptyStore ? 0 : threshold;
          if (split) {
            append(rdna4::build_sopp(rdna4::kSWaitLoadcntSopp, {.simm16 = load_wait}));
            append(rdna4::build_sopp(rdna4::kSWaitDscntSopp, {.simm16 = threshold}));
          } else {
            code.push_back(legacy_wait_word(cu->arch(), load_wait, threshold));
          }
          const size_t consumer = code.size();
          append(rdna3::build_vop1(rdna3::kVMovB32Vop1, {.src0 = 264, .vdst = 3}));
          const uint64_t pc = wf->pc;
          f.mem->load_image(reinterpret_cast<const uint8_t *>(code.data()), code.size() * 4, pc);
          const auto advance = [&](size_t words) {
            for (unsigned tick = 0; tick < 32 && wf->pc < pc + 4 * words; ++tick)
              ASSERT_TRUE(cu->step());
            ASSERT_EQ(wf->pc, pc + 4 * words);
          };
          if (empty_first) {
            wf->set_exec(0);
            ASSERT_NO_FATAL_FAILURE(advance(empty_first));
            const auto *state =
                static_cast<RaceWavefrontState *>(wf->plugin_state(race->slot_index()));
            EXPECT_TRUE(state->race_state->getWaveMemoryEvents().empty());
            const auto counter = route == Route::Lds ? WaitCounterKind::Ds : WaitCounterKind::Load;
            EXPECT_EQ(wf->ensure_memory_wait_scoreboard().outstanding(counter), 0u);
          }
          wf->set_exec(3);
          ASSERT_NO_FATAL_FAILURE(advance(older_end));
          wf->set_exec(younger == Younger::ActiveLoad ? 3 : 0);
          ASSERT_NO_FATAL_FAILURE(advance(consumer));
          wf->set_exec(3);
          const auto diagnostics = cu->memory_wait_diagnostic_count();
          const auto output = sink.str().size();
          ASSERT_NO_FATAL_FAILURE(advance(code.size()));
          const bool hazard = younger == Younger::None && threshold == 1;
          EXPECT_EQ(cu->memory_wait_diagnostic_count() != diagnostics, hazard);
          EXPECT_EQ(sink.str().find("Race on VGPR v8", output) != std::string::npos, hazard);
          EXPECT_EQ(wf->debug_read_vgpr(3, 0), 10u);
          EXPECT_EQ(wf->debug_read_vgpr(3, 1), 11u);
          wf->halt();
        }
      }
    }
  }
}

TEST(RaceDetectorPluginTest, Gfx10FlatReturnedBytesRequireBothCounterPositions) {
  struct Wait {
    uint8_t vm, lgkm;
    bool younger, pending;
  };
  constexpr std::array waits{Wait{63, 63, false, true}, Wait{0, 63, false, true},
                             Wait{63, 0, false, true},  Wait{0, 1, false, true},
                             Wait{1, 0, false, true},   Wait{0, 0, false, false},
                             Wait{1, 1, false, true},   Wait{1, 1, true, false},
                             Wait{1, 63, true, true},   Wait{63, 1, true, true}};
  for (const auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(
        f.initialize(1, arch == ROCJITSU_CODE_ARCH_RDNA1 ? "rdna1" : "rdna2", 32, 106, 32));
    PluginSinkConfig sinks;
    auto &sink = sinks.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    ASSERT_FALSE(cu->sram_ecc());
    constexpr uint64_t shared = 0x0001'0000'0000'0000ULL;
    constexpr uint64_t global = 0x400000;
    constexpr uint32_t value = 0x33442211;
    cu->set_apertures(shared, shared + 0xffff, 0, 0);
    auto *wave = cu->dispatch_wf(0, 0x8000, 106, 16, 32);
    ASSERT_NE(wave, nullptr);
    wave->set_lds_size(256);
    std::array<Wavefront *, 1> waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
    const uint32_t base = wave->vgpr_alloc().base;
    f.mem->write32(global, value);
    wave->lds().write32(wave->lds_base(), value);
    for (unsigned lane : {0u, 1u}) {
      const uint64_t address = lane ? shared : global;
      cu->write_vgpr(base, lane, static_cast<uint32_t>(address));
      cu->write_vgpr(base + 1, lane, address >> 32);
      cu->write_vgpr(base + 4, lane, 0); // The returning atomic adds zero to the initialized value.
    }
    for (const auto op : {rdna1::kFlatAtomicAddFlat, rdna1::kFlatLoadShortD16Flat,
                          rdna1::kFlatLoadShortD16HiFlat}) {
      const bool atomic = op == rdna1::kFlatAtomicAddFlat;
      const bool high_result = op == rdna1::kFlatLoadShortD16HiFlat;
      for (const auto wait : waits) {
        for (const unsigned reader : {0u, 1u, 2u}) {
          for (const bool read_high : {false, true}) {
            SCOPED_TRACE(testing::Message()
                         << int(arch) << "/" << op << "/" << int(wait.vm) << "/" << int(wait.lgkm)
                         << "/" << wait.younger << "/" << reader << "/" << read_high);
            const auto produce = rdna1::build_flat(
                op, {.glc = uint8_t(atomic), .addr = 0, .data = 4, .saddr = 125, .vdst = 8});
            const auto younger =
                rdna1::build_flat(rdna1::kFlatLoadDword, {.saddr = 125, .vdst = 9});
            // v_mov_b32_sdwa v3, v8 src0_sel:WORD_0/WORD_1.
            const uint32_t code[] = {produce[0],
                                     produce[1],
                                     wait.younger ? younger[0] : S_NOP,
                                     wait.younger ? younger[1] : S_NOP,
                                     legacy_wait_word(arch, wait.vm, wait.lgkm),
                                     0x7e0602f9u,
                                     read_high ? 0x00051608u : 0x00041608u,
                                     0xbf8c0000u,
                                     S_ENDPGM};
            const uint64_t pc = wave->pc + 0x40;
            wave->pc = pc;
            f.mem->load_image(reinterpret_cast<const uint8_t *>(code), sizeof(code), pc);
            const auto advance = [&](unsigned words) {
              for (unsigned tick = 0; tick < 32 && wave->pc < pc + 4 * words; ++tick)
                ASSERT_TRUE(cu->step());
              ASSERT_EQ(wave->pc, pc + 4 * words);
            };
            constexpr uint32_t canary = 0xaaaabbbb;
            for (unsigned lane : {0u, 1u, 2u})
              cu->write_vgpr(base + 8, lane, canary);
            wave->set_exec(3); // Global lane 0 and LDS lane 1 share the same FLAT result.
            ASSERT_NO_FATAL_FAILURE(advance(2));
            const uint32_t returned = atomic        ? value
                                      : high_result ? (value << 16) | (canary & 0xffff)
                                                    : (canary & 0xffff0000) | (value & 0xffff);
            EXPECT_EQ(cu->read_vgpr_storage(base + 8, 0), returned);
            EXPECT_EQ(cu->read_vgpr_storage(base + 8, 1), returned);
            EXPECT_EQ(cu->read_vgpr_storage(base + 8, 2), canary);
            wave->set_exec(0);
            ASSERT_NO_FATAL_FAILURE(advance(5));
            wave->set_exec(uint64_t{1} << reader);
            const auto warnings = cu->memory_wait_diagnostic_count();
            const auto output = sink.str().size();
            ASSERT_NO_FATAL_FAILURE(advance(7));
            const bool hazard = wait.pending && reader < 2 && (atomic || high_result == read_high);
            EXPECT_EQ(cu->memory_wait_diagnostic_count() != warnings, hazard);
            EXPECT_EQ(sink.str().find("RACE ", output) != std::string::npos, hazard);
            ASSERT_NO_FATAL_FAILURE(advance(8));
          }
        }
      }
    }
    wave->halt();
  }
}

TEST(RaceDetectorPluginTest, Gfx12TrapDrainsSplitMemoryCounters) {
  for (const auto *arch : {"rdna4", "cdna5"}) {
    SCOPED_TRACE(arch);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 32, 128));
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    const auto *observer = plugin.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
    f.soc->set_plugin_group(f.plugin_group_);
    auto *wave = f.cu()->dispatch_wf(0, 0x8000, 128, 16, 32);
    ASSERT_NE(wave, nullptr);
    std::array waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(0, 0, 16, 128, waves);
    auto &race =
        *static_cast<RaceWavefrontState *>(wave->plugin_state(observer->slot_index()))->race_state;
    register_pending_event(race, 0x7000, MemoryEventType::GLOBAL_TO_VGPR, {1},
                           WaitCounterType::LOADCNT, MemoryOrderClass::VMEM);
    register_pending_event(race, 0x7010, MemoryEventType::LDS_TO_VGPR, {2}, WaitCounterType::DSCNT,
                           MemoryOrderClass::LDS);
    const std::array km{
        MemoryCounterObligation{WaitCounterType::KMCNT, MemoryOrderClass::UNORDERED}};
    race.registerScalarLoad(testInstruction(0x7020), {RegClass::SGPR, 4, 1}, 1, km,
                            MemoryOrderClass::UNORDERED);
    ASSERT_EQ(race.getWaveMemoryEvents().size(), 3u);
    constexpr uint64_t handler = 0xa000;
    f.cu()->set_trap_handler_resolver(
        [](const Wavefront &) -> std::optional<ComputeUnitCore::TrapHandlerConfig> {
          return ComputeUnitCore::TrapHandlerConfig{handler, 0, true};
        });
    const auto code = cdna5::build_sopp(cdna5::kSTrapSopp, {.simm16 = 7});
    f.mem->load_image(reinterpret_cast<const uint8_t *>(code.data()), sizeof(code), wave->pc);
    ASSERT_TRUE(f.cu()->step());
    ASSERT_TRUE(wave->in_trap_handler());
    EXPECT_EQ(wave->pc, handler);
    EXPECT_TRUE(race.getWaveMemoryEvents().empty());
    wave->halt();
  }
}

TEST(RaceDetectorPluginTest, Gfx10IssuedTrapCompletesOnlyTheIssuingWavesMemory) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    for (bool retry : {false, true}) {
      SCOPED_TRACE(testing::Message() << "arch=" << arch << " retry=" << retry);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(
          f.initialize(2, arch == ROCJITSU_CODE_ARCH_RDNA1 ? "rdna1" : "rdna2", 32, 106, 32));
      PluginSinkConfig sinks;
      auto &sink = sinks.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
      auto plugin = std::make_unique<RaceDetectorPlugin>();
      auto *observer = plugin.get();
      ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      auto *cu = f.cu();
      constexpr uint64_t handler = 0xa000;
      std::array<Wavefront *, 2> waves{cu->dispatch_wf(0, 0x8000, 106, 16, 32),
                                       cu->dispatch_wf(0, 0x9000, 106, 16, 32)};
      auto *trapped = waves[0];
      const auto query =
          rdna1::build_mimg(rdna1::kImageGetResinfo, {.dim = 1, .dmask = 1, .vdata = 8});
      const auto read = rdna1::build_ds(rdna1::kDsReadB32, {.addr = 0, .vdst = 9});
      const auto use_query = rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = 264, .vdst = 10});
      const auto use_lds = rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = 265, .vdst = 11});
      const auto store = rdna1::build_ds(rdna1::kDsWriteB32, {.addr = 0, .data0 = 12});
      for (auto *wave : waves) {
        ASSERT_NE(wave, nullptr);
        wave->set_lds_size(512);
        wave->set_exec(1);
        cu->write_sgpr(wave->sgpr_alloc().base + 3, (9u << 28) | 4u); // 1x1 2D image.
        cu->write_vgpr(wave->vgpr_alloc().base + 12, 0, 99);
        const uint32_t code[] = {query[0],
                                 query[1],
                                 read[0],
                                 read[1],
                                 rdna1::build_sopp(rdna1::kSTrapSopp, {.simm16 = 7})[0],
                                 use_query[0],
                                 use_lds[0],
                                 store[0],
                                 store[1]};
        f.mem->load_image(reinterpret_cast<const uint8_t *>(code), sizeof(code), wave->pc);
      }
      const uint32_t body[] = {use_query[0], use_lds[0]};
      f.mem->load_image(reinterpret_cast<const uint8_t *>(body), sizeof(body), handler);
      cu->set_trap_handler_resolver(
          [=](const Wavefront &wave) -> std::optional<ComputeUnitCore::TrapHandlerConfig> {
            if (&wave == trapped)
              return ComputeUnitCore::TrapHandlerConfig{handler, 0, true};
            return std::nullopt;
          });
      cu->lds().write32(0, 41);
      cu->begin_workgroup(0, 0, 2);
      f.plugin_group_->onAmdgpuWorkgroupDispatched(0, 0, 32, 106, waves);
      const auto select = [&](Wavefront *wave) {
        for (auto *other : waves)
          other->set_runtime_suspended(other != wave);
      };
      const auto advance = [&](Wavefront *wave, uint64_t target) {
        for (unsigned tick = 0; tick < 32 && wave->pc != target; ++tick)
          cu->step();
        ASSERT_EQ(wave->pc, target);
      };
      for (auto *wave : waves) {
        select(wave);
        ASSERT_NO_FATAL_FAILURE(advance(wave, wave->pc + 16));
        ASSERT_TRUE(wave->wait_counters().empty());
      }
      select(trapped);
      if (retry) {
        trapped->wait_counters().increment(WaitCounterType::VSCNT);
        cu->step();
        ASSERT_EQ(trapped->pc, 0x8010u);
        ASSERT_FALSE(trapped->in_trap_handler());
        ASSERT_FALSE(trapped->memory_wait_scoreboard()->empty());
        auto *state =
            static_cast<RaceWavefrontState *>(trapped->plugin_state(observer->slot_index()));
        EXPECT_EQ(state->race_state->getWaveMemoryEvents().size(), 2u);
        trapped->release_wait_counter(WaitCounterType::VSCNT);
      }
      ASSERT_NO_FATAL_FAILURE(advance(trapped, handler));
      ASSERT_TRUE(trapped->in_trap_handler());
      EXPECT_EQ(trapped->ttmp(0), 0x8010u);
      EXPECT_EQ((trapped->ttmp(1) >> 16) & 0xffu, 7u);
      for (auto *wave : waves) {
        select(wave);
        const bool pending = wave != trapped;
        if (pending) {
          // The peer has no handler. Its trap retires even with a held counter.
          wave->wait_counters().increment(WaitCounterType::VMCNT);
          ASSERT_NO_FATAL_FAILURE(advance(wave, wave->pc + 4));
          EXPECT_FALSE(wave->in_trap_handler());
          wave->release_wait_counter(WaitCounterType::VMCNT);
        }
        for (unsigned reg : {8u, 9u}) {
          const auto warnings = cu->memory_wait_diagnostic_count();
          const auto output = sink.str().size();
          ASSERT_NO_FATAL_FAILURE(advance(wave, wave->pc + 4));
          EXPECT_EQ(cu->read_vgpr_storage(wave->vgpr_alloc().base + reg + 2, 0),
                    reg == 8 ? 1u : 41u);
          EXPECT_EQ(cu->memory_wait_diagnostic_count() != warnings, pending);
          EXPECT_EQ(sink.str().find("type=VGPR access=read reg=" + std::to_string(reg), output) !=
                        std::string::npos,
                    pending);
        }
      }
      // The trap completed the reader's LDS access but only a workgroup barrier publishes it.
      const auto output = sink.str().size();
      ASSERT_NO_FATAL_FAILURE(advance(waves[1], waves[1]->pc + 8));
      EXPECT_EQ(cu->lds().read32(0), 99u);
      EXPECT_NE(sink.str().find("type=LDS access=write", output), std::string::npos);
      for (auto *wave : waves)
        wave->halt();
    }
  }
}

TEST(RaceDetectorPluginTest, Gfx10BarrierPublishesCompletedLdsReadsFromExitedWaves) {
  enum class Read { Direct, Ds, Flat, FlatVmWait, FlatLgkmWait, DsLgkmWait };
  enum class Finish { Live, Endpgm, Saved, Ordered, DeferredEndpgm, Forced };
  struct Case {
    Read read;
    Finish finish;
    unsigned barriers;
    bool other_group = false;
  };
  for (const auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    for (const Case test : {Case{Read::Direct, Finish::Endpgm, 0},
                            {Read::Direct, Finish::Endpgm, 1},
                            {Read::Direct, Finish::Endpgm, 2, true},
                            {Read::Direct, Finish::Live, 0},
                            {Read::Direct, Finish::Live, 1},
                            {Read::FlatVmWait, Finish::Live, 1},
                            {Read::FlatLgkmWait, Finish::Live, 1},
                            {Read::Ds, Finish::Endpgm, 0},
                            {Read::Ds, Finish::Endpgm, 1},
                            {Read::DsLgkmWait, Finish::Endpgm, 1},
                            {Read::Flat, Finish::Endpgm, 0},
                            {Read::Flat, Finish::Endpgm, 1},
                            {Read::FlatVmWait, Finish::Endpgm, 1},
                            {Read::FlatLgkmWait, Finish::Endpgm, 1},
                            {Read::Ds, Finish::Saved, 1},
                            {Read::Ds, Finish::Ordered, 1},
                            {Read::Ds, Finish::DeferredEndpgm, 1},
                            {Read::Ds, Finish::Forced, 1}}) {
      SCOPED_TRACE(testing::Message()
                   << int(arch) << " read=" << int(test.read) << " finish=" << int(test.finish)
                   << " barriers=" << test.barriers << " other_group=" << test.other_group);
      PluginFixture f;
      const unsigned groups = test.other_group ? 2 : 1;
      ASSERT_NO_FATAL_FAILURE(f.initialize(
          2 * groups, arch == ROCJITSU_CODE_ARCH_RDNA1 ? "rdna1" : "rdna2", 32, 106, 32));
      PluginSinkConfig sinks;
      auto &sink = sinks.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
      auto race = std::make_unique<RaceDetectorPlugin>();
      auto *race_plugin = race.get();
      ASSERT_TRUE(f.plugin_group_->add(std::move(race)));
      auto hooks = std::make_unique<OrderingPlugin>();
      const auto &events = hooks->events;
      ASSERT_TRUE(f.plugin_group_->add(std::move(hooks)));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      auto *cu = f.cu();
      unsigned ordered_messages = 0;
      cu->set_sendmsg_handler([&](Wavefront &, uint32_t message) {
        if (message != 7)
          return false;
        ++ordered_messages;
        return true;
      });
      constexpr uint64_t shared = 0x0001'0000'0000'0000ULL;
      cu->set_apertures(shared, shared + 0xffff, 0, 0);
      std::vector<Wavefront *> waves;
      for (unsigned group = 0; group < groups; ++group) {
        for (unsigned i = 0; i < 2; ++i) {
          auto *wave = cu->dispatch_wf(group, 0x8000 + 0x4000 * waves.size(), 106, 16, 32);
          ASSERT_NE(wave, nullptr);
          waves.push_back(wave);
          wave->set_lds_base(256 * group);
          wave->set_lds_size(16);
          wave->set_m0(0x20004);
          wave->set_exec(1);
          const auto vb = wave->vgpr_alloc().base;
          cu->write_vgpr(vb, 0, 4);
          cu->write_vgpr(vb + 1, 0, uint32_t(shared >> 32));
          cu->write_vgpr(vb + 2, 0, 99);
        }
        cu->lds().write32(256 * group + 4, 41);
        cu->begin_workgroup(0, group, 2);
        f.plugin_group_->onAmdgpuWorkgroupDispatched(0, group, 32, 106, std::span(waves).last(2));
      }
      const auto issue = [&](unsigned index, const auto &words, bool end = false) {
        auto *wave = waves[index];
        for (auto *candidate : waves)
          if (!candidate->is_halted())
            candidate->set_runtime_suspended(candidate != wave);
        const uint64_t pc = wave->pc + 0x100;
        wave->pc = pc;
        f.mem->load_image(reinterpret_cast<const uint8_t *>(words.data()), sizeof(words), pc);
        for (unsigned tick = 0; tick < 32 && !wave->is_halted() && wave->pc == pc; ++tick)
          ASSERT_TRUE(cu->step() || wave->is_halted() || wave->pc != pc);
        ASSERT_EQ(wave->is_halted(), end);
        if (!end) {
          ASSERT_EQ(wave->pc, pc + sizeof(words));
        }
      };
      for (unsigned group = 0; group < groups; ++group) {
        const unsigned reader = 2 * group;
        if (test.read == Read::Direct) {
          ASSERT_NO_FATAL_FAILURE(
              issue(reader, rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = 254, .vdst = 9})));
        } else if (test.read == Read::Ds || test.read == Read::DsLgkmWait) {
          ASSERT_NO_FATAL_FAILURE(
              issue(reader, rdna1::build_ds(rdna1::kDsReadB32, {.addr = 0, .vdst = 9})));
          if (test.read == Read::DsLgkmWait) {
            ASSERT_NO_FATAL_FAILURE(issue(reader, std::array{legacy_wait_word(arch, 63, 0)}));
          }
        } else {
          ASSERT_NO_FATAL_FAILURE(
              issue(reader, rdna1::build_flat(rdna1::kFlatLoadDword,
                                              {.addr = 0, .saddr = 125, .vdst = 9})));
          if (test.read != Read::Flat) {
            const bool vm = test.read == Read::FlatVmWait;
            ASSERT_NO_FATAL_FAILURE(
                issue(reader, std::array{legacy_wait_word(arch, vm ? 0 : 63, vm ? 63 : 0)}));
          }
        }
        auto *wave = waves[reader];
        EXPECT_EQ(cu->read_vgpr_storage(wave->vgpr_alloc().base + 9, 0), 41u);
        if (test.finish == Finish::Forced) {
          ASSERT_NO_FATAL_FAILURE(issue(reader, std::array{S_NOP}));
          wave->halt();
        } else if (test.finish == Finish::DeferredEndpgm) {
          // Hold a physical counter while the decoded END_PGM enters ENDING.
          wave->wait_counters().increment(WaitCounterType::VSCNT);
          ASSERT_NO_FATAL_FAILURE(issue(reader, std::array{S_ENDPGM}));
          ASSERT_EQ(wave->state(), WfState::ENDING);
          ASSERT_EQ(wave->wait_counters().vscnt, 1u);
          EXPECT_EQ(cu->read_vgpr_storage(wave->vgpr_alloc().base + 9, 0), 41u);
          wave->release_wait_counter(WaitCounterType::VSCNT);
          ASSERT_TRUE(wave->is_halted());
        } else if (test.finish != Finish::Live) {
          const auto op = test.finish == Finish::Saved     ? rdna1::kSEndpgmSavedSopp
                          : test.finish == Finish::Ordered ? rdna1::kSEndpgmOrderedPsDoneSopp
                                                           : rdna1::kSEndpgmSopp;
          ASSERT_NO_FATAL_FAILURE(issue(reader, rdna1::build_sopp(op, {}), true));
        }
      }
      EXPECT_EQ(ordered_messages, unsigned(test.finish == Finish::Ordered));
      for (unsigned barrier = 0; barrier < test.barriers; ++barrier) {
        const auto words = rdna1::build_sopp(rdna1::kSBarrierSopp, {});
        if (test.finish == Finish::Live) {
          ASSERT_NO_FATAL_FAILURE(issue(0, words));
        }
        ASSERT_NO_FATAL_FAILURE(issue(1, words));
        // The following scheduler cycle publishes the legacy barrier before the next instruction.
        ASSERT_NO_FATAL_FAILURE(issue(1, std::array{S_NOP}));
      }
      EXPECT_EQ(std::ranges::count(events, HookEvent::BARRIER_RESOLVED, &HookEvent::kind),
                test.barriers);
      for (unsigned group = 0; group < groups; ++group) {
        const auto output = sink.str().size();
        ASSERT_NO_FATAL_FAILURE(
            issue(2 * group + 1, rdna1::build_ds(rdna1::kDsWriteB32, {.addr = 0, .data0 = 2})));
        EXPECT_EQ(cu->lds().read32(256 * group + 4), 99u);
        // Each workgroup publishes its own completed reads. Shader termination supplies all waits.
        const bool ended = test.finish != Finish::Live && test.finish != Finish::Forced;
        const bool lds_ready = ended || test.read == Read::Direct ||
                               test.read == Read::DsLgkmWait || test.read == Read::FlatLgkmWait;
        const bool hazard = group != 0 || test.barriers == 0 || !lds_ready;
        EXPECT_EQ(sink.str().find("type=LDS access=write", output) != std::string::npos, hazard)
            << sink.str().substr(output);
      }
      if (test.finish == Finish::Live && test.read == Read::FlatLgkmWait) {
        auto &pending =
            *static_cast<RaceWavefrontState *>(waves[0]->plugin_state(race_plugin->slot_index()))
                 ->race_state;
        const auto mov = rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = 265, .vdst = 3});
        ASSERT_FALSE(pending.getWaveMemoryEvents().empty());
        auto output = sink.str().size();
        ASSERT_NO_FATAL_FAILURE(issue(0, mov));
        EXPECT_NE(sink.str().find("type=VGPR access=read reg=9", output), std::string::npos);
        ASSERT_NO_FATAL_FAILURE(issue(0, std::array{legacy_wait_word(arch, 0, 63)}));
        EXPECT_TRUE(pending.getWaveMemoryEvents().empty());
        output = sink.str().size();
        ASSERT_NO_FATAL_FAILURE(issue(0, mov));
        EXPECT_EQ(sink.str().find("RACE ", output), std::string::npos);
      }
      for (auto *wave : waves)
        if (!wave->is_halted())
          wave->halt();
    }
  }
}

TEST(RaceDetectorPluginTest, Gfx10LdsDirectSourceChecksTheBroadcastBytesWithoutIssuingMemory) {
  enum class Store { Flat, Ds, Byte };
  enum class Read {
    Mov,
    First,
    Ds,
    Empty,
    SuppressedPair,
    Invalid,
    OutOfBounds,
    OtherGroup,
    SameWave,
    InterpP1,
    InterpP2,
    InterpMov
  };
  enum class Sync { None, Lgkm, LgkmBarrier, Barrier };
  constexpr uint32_t word = 0x1234ff80;
  constexpr uint32_t sentinel = 0xdecafbad;
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    const auto check = [&](Store store, uint32_t writer_offset, Read read, unsigned type, Sync sync,
                           bool hazard, uint32_t value, bool read_first = false) {
      SCOPED_TRACE(testing::Message() << int(arch) << " store=" << int(store) << " offset="
                                      << writer_offset << " read=" << int(read) << " type=" << type
                                      << " sync=" << int(sync) << " read_first=" << read_first);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(
          f.initialize(2, arch == ROCJITSU_CODE_ARCH_RDNA1 ? "rdna1" : "rdna2", 32, 106, 32));
      PluginSinkConfig sinks;
      auto &sink = sinks.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
      auto plugin = std::make_unique<RaceDetectorPlugin>();
      auto *observer = plugin.get();
      ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      auto *cu = f.cu();
      constexpr uint64_t shared = 0x0001'0000'0000'0000ULL;
      cu->set_apertures(shared, shared + 0xffff, 0, 0);
      const bool other = read == Read::OtherGroup;
      const bool interp =
          read == Read::InterpP1 || read == Read::InterpP2 || read == Read::InterpMov;
      std::array<Wavefront *, 2> waves{cu->dispatch_wf(0, 0x8000, 106, 16, 32),
                                       cu->dispatch_wf(other ? 1 : 0, 0xa000, 106, 16, 32)};
      auto *writer = waves[0];
      const unsigned reader_index = read == Read::SameWave ? 0 : 1;
      auto *reader = waves[reader_index];
      ASSERT_NE(writer, nullptr);
      ASSERT_NE(reader, nullptr);
      writer->set_lds_base(256);
      reader->set_lds_base(other ? 512 : 256);
      const unsigned allocation = interp ? 96 : 16;
      writer->set_lds_size(allocation);
      reader->set_lds_size(read == Read::OutOfBounds ? 4 : allocation);
      if (other) {
        f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, std::span(waves).first(1));
        f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 1, 32, 106, std::span(waves).last(1));
      } else {
        f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
      }
      cu->lds().write32(writer->lds_base() + 4, 0);
      cu->lds().write32(reader->lds_base() + 4, 0);
      const auto wv = writer->vgpr_alloc().base;
      cu->write_vgpr(wv, 0, writer_offset);
      cu->write_vgpr(wv + 1, 0, uint32_t(shared >> 32));
      cu->write_vgpr(wv + 2, 0, word);
      const auto rv = reader->vgpr_alloc().base;
      for (unsigned lane : {0u, 2u, 4u}) {
        cu->write_vgpr(rv + 1, lane, 4);
        cu->write_vgpr(rv + 9, lane, sentinel);
        cu->write_vgpr(rv + 15, lane, sentinel);
      }
      const auto issue = [&](unsigned who, const auto &words, uint64_t exec) {
        auto *wave = waves[who];
        waves[1 - who]->set_runtime_suspended(true);
        wave->set_runtime_suspended(false);
        wave->set_exec(exec);
        const uint64_t pc = wave->pc + 0x100;
        wave->pc = pc;
        f.mem->load_image(reinterpret_cast<const uint8_t *>(words.data()), sizeof(words), pc);
        for (unsigned tick = 0; tick < 32 && wave->pc == pc; ++tick)
          ASSERT_TRUE(cu->step());
        ASSERT_EQ(wave->pc, pc + sizeof(words));
      };
      const auto write = [&] {
        writer->set_m0(16);
        if (store == Store::Flat) {
          ASSERT_NO_FATAL_FAILURE(issue(
              0, rdna1::build_flat(rdna1::kFlatStoreDword, {.addr = 0, .data = 2, .saddr = 125}),
              1));
        } else {
          ASSERT_NO_FATAL_FAILURE(
              issue(0,
                    rdna1::build_ds(store == Store::Byte ? rdna1::kDsWriteB8 : rdna1::kDsWriteB32,
                                    {.addr = 0, .data0 = 2}),
                    1));
        }
      };
      const auto synchronize = [&](unsigned wave) {
        if (sync == Sync::Lgkm || sync == Sync::LgkmBarrier) {
          ASSERT_NO_FATAL_FAILURE(issue(wave, std::array{legacy_wait_word(arch, 63, 0)}, 1));
        }
        if (sync == Sync::Barrier || sync == Sync::LgkmBarrier)
          f.plugin_group_->onAmdgpuBarrierResolved(waves, BarrierKind::Workgroup);
      };
      if (!read_first) {
        ASSERT_NO_FATAL_FAILURE(write());
        ASSERT_NO_FATAL_FAILURE(synchronize(0));
      }
      auto &state = *static_cast<RaceWavefrontState *>(reader->plugin_state(observer->slot_index()))
                         ->race_state;
      const auto prior_events = state.getWaveMemoryEvents();
      const auto before = sink.str().size();
      const uint64_t exec = read == Read::Empty ? 0 : 0x14;
      reader->set_m0((type << 16) | 4);
      const auto mov = rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = 254, .vdst = 9});
      if (interp) {
        reader->set_m0(1u << 16); // The two active quads use separate primitive coefficient blocks.
        const uint8_t op = read == Read::InterpP1 ? 0 : read == Read::InterpP2 ? 1 : 2;
        ASSERT_NO_FATAL_FAILURE(
            issue(reader_index, rdna1::build_vintrp(op, {.vsrc = 2, .vdst = 9}), exec));
      } else if (read == Read::Ds) {
        ASSERT_NO_FATAL_FAILURE(
            issue(reader_index, rdna1::build_ds(rdna1::kDsReadB32, {.addr = 1, .vdst = 9}), exec));
      } else if (read == Read::Invalid) {
        auto inst = Decoder::create(arch)->decode_window(mov);
        ASSERT_TRUE(inst.succeeded());
        reader->set_exec(exec);
        f.plugin_group_->onAmdgpuBeforeExecuteInstruction(reader->pc, *inst.value(), *reader);
        ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *reader).failed());
        EXPECT_EQ(reader->instruction_execution_error(),
                  InstructionExecutionError::UnsupportedOperandValue);
      } else if (read == Read::SuppressedPair) {
        // v_cvt_f64_u32 v[15:16], lds_direct exceeds the 16 allocated VGPRs.
        ASSERT_NO_FATAL_FAILURE(
            issue(reader_index,
                  rdna1::build_vop1(rdna1::kVCvtF64U32Vop1, {.src0 = 254, .vdst = 15}), exec));
        EXPECT_EQ(cu->read_vgpr_storage(rv + 15, 2), sentinel);
      } else if (read == Read::First) {
        ASSERT_NO_FATAL_FAILURE(issue(
            reader_index,
            rdna1::build_vop1(rdna1::kVReadfirstlaneB32Vop1, {.src0 = 254, .vdst = 9}), exec));
        EXPECT_EQ(cu->read_sgpr(reader->sgpr_alloc().base + 9), value);
      } else {
        ASSERT_NO_FATAL_FAILURE(issue(reader_index, mov, exec));
      }
      const auto read_report = sink.str().substr(before);
      EXPECT_EQ(read_report.find("type=LDS access=read") != std::string::npos,
                !read_first && hazard)
          << read_report;
      if (read != Read::Ds) {
        EXPECT_EQ(state.getWaveMemoryEvents(), prior_events);
        EXPECT_EQ(reader->wait_counters().lgkmcnt, 0u);
        EXPECT_EQ(reader->wait_counters().vmcnt, 0u);
      }
      for (unsigned lane : {0u, 2u, 4u}) {
        if (interp || read == Read::Invalid)
          continue; // Interpolation values depend on the coefficients for each primitive.
        const bool writes_v9 = read != Read::First && read != Read::SuppressedPair;
        EXPECT_EQ(cu->read_vgpr_storage(rv + 9, lane),
                  writes_v9 && (exec & (1u << lane)) ? value : sentinel);
      }
      if (read_first) {
        ASSERT_NO_FATAL_FAILURE(synchronize(reader_index));
        const auto output = sink.str().size();
        ASSERT_NO_FATAL_FAILURE(write());
        const auto write_report = sink.str().substr(output);
        EXPECT_EQ(write_report.find("type=LDS access=write") != std::string::npos, hazard)
            << write_report;
      }
      for (auto *wave : waves)
        wave->halt();
    };
    for (const auto &[read, first, last] : {std::tuple{Read::InterpP1, 0u, 4u},
                                            {Read::InterpP2, 32u, 32u},
                                            {Read::InterpMov, 0u, 0u}}) {
      for (uint32_t offset : {first, last + 48}) {
        ASSERT_NO_FATAL_FAILURE(check(Store::Ds, offset, read, 0, Sync::None, true, 0));
        ASSERT_NO_FATAL_FAILURE(check(Store::Ds, offset, read, 0, Sync::LgkmBarrier, false, 0));
      }
      ASSERT_NO_FATAL_FAILURE(check(Store::Ds, 44, read, 0, Sync::None, false, 0));
    }
    for (auto store : {Store::Flat, Store::Ds}) {
      for (auto read : {Read::Mov, Read::Ds}) {
        ASSERT_NO_FATAL_FAILURE(check(store, 4, read, 2, Sync::None, true, word));
        ASSERT_NO_FATAL_FAILURE(check(store, 4, read, 2, Sync::Lgkm, true, word));
        ASSERT_NO_FATAL_FAILURE(check(store, 4, read, 2, Sync::LgkmBarrier, false, word));
        ASSERT_NO_FATAL_FAILURE(check(store, 4, read, 2, Sync::None, true, 0, true));
        ASSERT_NO_FATAL_FAILURE(check(store, 4, read, 2, Sync::Lgkm, true, 0, true));
        // A VALU read has completed when it returns; a native DS read still needs its wait.
        ASSERT_NO_FATAL_FAILURE(check(store, 4, read, 2, Sync::Barrier, read == Read::Ds, 0, true));
        ASSERT_NO_FATAL_FAILURE(check(store, 4, read, 2, Sync::LgkmBarrier, false, 0, true));
      }
    }
    ASSERT_NO_FATAL_FAILURE(check(Store::Ds, 4, Read::SameWave, 2, Sync::None, false, 0, true));
    ASSERT_NO_FATAL_FAILURE(check(Store::Ds, 0, Read::Mov, 2, Sync::None, false, 0, true));
    ASSERT_NO_FATAL_FAILURE(check(Store::Byte, 5, Read::Mov, 4, Sync::None, false, 0, true));
    ASSERT_NO_FATAL_FAILURE(check(Store::Byte, 5, Read::Mov, 5, Sync::None, true, 0, true));
    ASSERT_NO_FATAL_FAILURE(check(Store::Ds, 4, Read::Empty, 2, Sync::None, false, sentinel, true));
    ASSERT_NO_FATAL_FAILURE(check(Store::Ds, 4, Read::OutOfBounds, 2, Sync::None, false, 0, true));
    ASSERT_NO_FATAL_FAILURE(check(Store::Ds, 4, Read::SameWave, 2, Sync::None, true, word));
    ASSERT_NO_FATAL_FAILURE(check(Store::Ds, 4, Read::SameWave, 2, Sync::Lgkm, false, word));
    ASSERT_NO_FATAL_FAILURE(check(Store::Flat, 0, Read::Mov, 2, Sync::Lgkm, false, 0));
    ASSERT_NO_FATAL_FAILURE(check(Store::Flat, 4, Read::OtherGroup, 2, Sync::Lgkm, false, 0));
    ASSERT_NO_FATAL_FAILURE(check(Store::Flat, 4, Read::First, 2, Sync::Lgkm, true, word));
    ASSERT_NO_FATAL_FAILURE(check(Store::Ds, 4, Read::Empty, 2, Sync::None, false, sentinel));
    ASSERT_NO_FATAL_FAILURE(
        check(Store::Ds, 4, Read::SuppressedPair, 2, Sync::None, false, sentinel));
    ASSERT_NO_FATAL_FAILURE(check(Store::Ds, 4, Read::Invalid, 3, Sync::None, false, sentinel));
    ASSERT_NO_FATAL_FAILURE(check(Store::Ds, 4, Read::OutOfBounds, 2, Sync::None, false, 0));
    ASSERT_NO_FATAL_FAILURE(check(Store::Ds, 4, Read::Mov, 4, Sync::Lgkm, true, 0xffffff80u));
    // A writer at byte 5 overlaps a halfword beginning at 4, but not its low byte.
    ASSERT_NO_FATAL_FAILURE(check(Store::Byte, 5, Read::Mov, 0, Sync::Lgkm, false, 0));
    ASSERT_NO_FATAL_FAILURE(check(Store::Byte, 5, Read::Mov, 1, Sync::Lgkm, true, 0x8000));
    ASSERT_NO_FATAL_FAILURE(check(Store::Byte, 5, Read::Mov, 5, Sync::Lgkm, true, 0xffff8000u));
  }
}

TEST(RaceDetectorPluginTest, Gfx10SharedFlatClampsCompleteLaneTransfersAtTheWorkgroupEnd) {
  enum class Action { Load, Store, Add, AddReturn };
  struct Form {
    uint16_t opcode;
    Action action;
    unsigned words;
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    const bool rdna1 = arch == ROCJITSU_CODE_ARCH_RDNA1;
    const unsigned size = rdna1 ? 512 : 1024;
    const auto check = [&](Form form, unsigned offset, bool mixed) {
      SCOPED_TRACE(testing::Message() << int(arch) << "/" << form.opcode << "/" << int(form.action)
                                      << "/" << offset << "/" << mixed);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(1, rdna1 ? "rdna1" : "rdna2", 32, 106, 32));
      PluginSinkConfig sinks;
      auto &sink = sinks.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
      auto observer = std::make_unique<MemoryObservationPlugin>();
      const auto *observed = observer.get();
      ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
      auto race = std::make_unique<RaceDetectorPlugin>();
      const auto *plugin = race.get();
      ASSERT_TRUE(f.plugin_group_->add(std::move(race)));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      auto *cu = f.cu();
      constexpr uint64_t shared = uint64_t{1} << 48, global = 0x5000;
      cu->set_apertures(shared, shared + 0xffff, 0, 0);
      ASSERT_EQ(cu->allocate_lds(size), 0u);
      const unsigned base = cu->allocate_lds(size);
      ASSERT_EQ(base, size);
      auto *wave = cu->dispatch_wf(0, 0x8000, 106, 16, 32);
      ASSERT_NE(wave, nullptr);
      wave->set_lds_base(base);
      wave->set_lds_size(size);
      wave->set_m0(size);
      const uint64_t exec = mixed ? 5 : 1;
      wave->set_exec(exec);
      std::array<Wavefront *, 1> waves{wave};
      f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
      constexpr uint32_t initial = 0x11111111u, canary = 0xccccccccu;
      auto &lds = wave->lds();
      const auto vb = wave->vgpr_alloc().base;
      ASSERT_GT(lds.size_bytes(), base + size + 32);
      for (unsigned address = base + size - 32; address < base + size + 32; address += 4)
        lds.write32(address, initial);
      for (unsigned word = 0; word < 4; ++word)
        f.mem->write32(global + 4 * word, initial);
      const auto data = [](unsigned word) { return word == 1 ? 0u : 3u + word; };
      for (unsigned lane : {0u, 1u, 2u}) {
        // The masked lane points to an unrelated in-bounds LDS word.
        const uint64_t address = lane == 0 ? shared + offset : lane == 1 ? shared + 16 : global;
        cu->write_vgpr(vb, lane, uint32_t(address));
        cu->write_vgpr(vb + 1, lane, address >> 32);
        for (unsigned word = 0; word < form.words; ++word) {
          cu->write_vgpr(vb + 4 + word, lane, data(word));
          cu->write_vgpr(vb + 8 + word, lane, canary);
        }
      }
      const auto issue = [&](const auto &words) {
        auto pc = wave->pc + 0x100;
        wave->pc = pc;
        f.mem->load_image(reinterpret_cast<const uint8_t *>(words.data()), words.size() * 4, pc);
        for (unsigned tick = 0; tick < 32 && wave->pc == pc; ++tick)
          ASSERT_TRUE(cu->step());
        ASSERT_EQ(wave->pc, pc + 4 * words.size());
      };
      const bool atomic = form.action == Action::Add || form.action == Action::AddReturn;
      const bool returns = form.action == Action::Load || form.action == Action::AddReturn;
      ASSERT_NO_FATAL_FAILURE(
          issue(rdna1::build_flat(form.opcode, {.glc = uint8_t(form.action == Action::AddReturn),
                                                .addr = 0,
                                                .data = 4,
                                                .saddr = 125,
                                                .vdst = 8})));
      const unsigned width = 4 * form.words;
      const bool fits = offset + width <= size;
      ASSERT_EQ(observed->accesses.size(), 1u);
      const auto &access = observed->accesses.front();
      EXPECT_EQ(access.request_lane_mask, exec);
      EXPECT_EQ(access.flat_local_lane_mask, 1u);
      EXPECT_EQ(access.flat_local_oob_lane_mask, fits ? 0u : 1u);
      const auto vm = returns ? WaitCounterType::VMCNT : WaitCounterType::VSCNT;
      EXPECT_EQ(std::ranges::count(access.issued_counters, vm), 1);
      EXPECT_EQ(std::ranges::count(access.issued_counters, WaitCounterType::LGKMCNT), 1);
      auto &state =
          *static_cast<RaceWavefrontState *>(wave->plugin_state(plugin->slot_index()))->race_state;
      const auto &events = state.getWaveMemoryEvents();
      const auto &registry = state.getDetector()->events();
      unsigned lds_bytes = 0;
      for (auto id : events)
        for (const auto &interval : registry.ldsIntervals(id)) {
          EXPECT_EQ(interval.start, int(base + offset));
          EXPECT_EQ(interval.end, int(base + offset + width));
          lds_bytes += interval.end - interval.start;
        }
      EXPECT_EQ(lds_bytes, fits ? width : 0u);
      for (unsigned word = 0; word < form.words; ++word) {
        const uint32_t result = form.action == Action::Load ? initial
                                : atomic                    ? initial + data(word)
                                                            : data(word);
        EXPECT_EQ(lds.read32(base + offset + 4 * word), fits ? result : initial);
        EXPECT_EQ(f.mem->read32(global + 4 * word), mixed ? result : initial);
        EXPECT_EQ(cu->read_vgpr_storage(vb + 8 + word, 0),
                  returns ? (fits ? initial : 0u) : canary);
        EXPECT_EQ(cu->read_vgpr_storage(vb + 8 + word, 1), canary);
        EXPECT_EQ(cu->read_vgpr_storage(vb + 8 + word, 2), returns && mixed ? initial : canary);
      }
      if (returns) {
        wave->set_exec(1);
        // A suppressed LDS access still returns zero in an active lane; either single wait
        // leaves the other obligation pending.
        ASSERT_NO_FATAL_FAILURE(
            issue(std::array{legacy_wait_word(arch, mixed ? 0 : 63, mixed ? 63 : 0)}));
        for (bool waited : {false, true}) {
          if (waited) {
            ASSERT_NO_FATAL_FAILURE(issue(std::array{legacy_wait_word(arch, 0, 0)}));
          }
          const auto output = sink.str().size();
          const auto warnings = cu->memory_wait_diagnostic_count();
          ASSERT_NO_FATAL_FAILURE(
              issue(rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = 264, .vdst = 14})));
          EXPECT_EQ(cu->memory_wait_diagnostic_count() != warnings, !waited);
          EXPECT_EQ(sink.str().find("type=VGPR access=read reg=8 ", output) != std::string::npos,
                    !waited);
        }
        EXPECT_TRUE(state.getWaveMemoryEvents().empty());
      }
      wave->halt();
    };
    for (auto form : {Form{rdna1::kFlatLoadDword, Action::Load, 1},
                      Form{rdna1::kFlatStoreDword, Action::Store, 1},
                      Form{rdna1::kFlatLoadDwordx4Flat, Action::Load, 4},
                      Form{rdna1::kFlatStoreDwordx4Flat, Action::Store, 4},
                      Form{rdna1::kFlatAtomicAdd, Action::Add, 1},
                      Form{rdna1::kFlatAtomicAdd, Action::AddReturn, 1},
                      Form{rdna1::kFlatAtomicAddX2Flat, Action::Add, 2},
                      Form{rdna1::kFlatAtomicAddX2Flat, Action::AddReturn, 2}})
      for (bool mixed : {false, true})
        for (unsigned offset : {size - 4 * form.words, size - (form.words == 1 ? 2 : 4), size})
          ASSERT_NO_FATAL_FAILURE(check(form, offset, mixed));
  }
}

TEST(RaceDetectorPluginTest, Gfx10BufferToLdsChecksDestinationBeforeRequestingSource) {
  class SourcePages final : public AddressSpaceTranslator {
  public:
    explicit SourcePages(uint32_t readable) : readable_(readable) {}
    VmTranslationResult translate(uint64_t addr, size_t bytes, VmAccessKind access) const override {
      if (access == VmAccessKind::Read && addr >= 0x10000 && addr < 0x18000) {
        const auto bit = uint32_t{1} << ((addr - 0x10000) / 4096);
        seen |= bit;
        if (!(readable_ & bit))
          return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
      }
      return IdentityAddressSpaceTranslator{}.translate(addr, bytes, access);
    }
    mutable uint32_t seen = 0;

  private:
    uint32_t readable_;
  };
  struct Case {
    unsigned bytes_to_end;
    uint32_t exec, requests;
    bool bad_source = false;
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    const bool rdna1 = arch == ROCJITSU_CODE_ARCH_RDNA1;
    const unsigned size = rdna1 ? 512 : 1024;
    for (auto op : {rdna1::kBufferLoadDword, rdna1::kBufferLoadUbyte, rdna1::kBufferLoadSbyte,
                    rdna1::kBufferLoadFormatX}) {
      // Lane 1 is inactive in the sparse case. The last lane of each nonempty request mask
      // reaches the final DWORD, and a byte load still needs four bytes at its LDS destination.
      for (auto test : {Case{12, 0xd, 5}, Case{4, 3, 1}, Case{0, 3, 0}, Case{2, 3, 0},
                        Case{12, 0xd, 4, true}}) {
        SCOPED_TRACE(testing::Message() << int(arch) << "/" << op << "/" << test.bytes_to_end << "/"
                                        << test.exec << "/" << test.bad_source);
        PluginFixture f;
        ASSERT_NO_FATAL_FAILURE(f.initialize(1, rdna1 ? "rdna1" : "rdna2", 32, 106, 32));
        PluginSinkConfig sinks;
        sinks.emplace<StringSink>();
        f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
        auto observer = std::make_unique<MemoryObservationPlugin>();
        auto *observed = observer.get();
        ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
        auto race = std::make_unique<RaceDetectorPlugin>();
        auto *plugin = race.get();
        ASSERT_TRUE(f.plugin_group_->add(std::move(race)));
        f.soc->set_plugin_group(f.plugin_group_);
        f.plugin_group_->onInit();
        auto *cu = f.cu();
        ASSERT_EQ(cu->allocate_lds(size), 0u);
        const auto base = cu->allocate_lds(size);
        ASSERT_EQ(base, size);
        auto *wave = cu->dispatch_wf(0, 0x8000, 106, 16, 32);
        ASSERT_NE(wave, nullptr);
        GpuVm vm;
        auto pages = std::make_shared<SourcePages>(test.requests);
        auto space =
            vm.register_address_space(5, pages, std::make_shared<GpuMemoryPhysicalAccess>(*f.mem));
        cu->set_gpu_vm(&vm);
        wave->set_address_space(space);
        wave->set_process_id(5);
        const unsigned start = size - test.bytes_to_end;
        constexpr unsigned imm = 4;
        wave->set_lds_base(base);
        wave->set_lds_size(size);
        wave->set_m0(0xabcd0000 | (start - imm));
        wave->set_exec(test.exec);
        std::array<Wavefront *, 1> waves{wave};
        f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
        const auto sb = wave->sgpr_alloc().base, vb = wave->vgpr_alloc().base;
        cu->write_sgpr(sb, 0x10000);
        cu->write_sgpr(sb + 1, 0);
        cu->write_sgpr(sb + 2, 0x6000);
        cu->write_sgpr(sb + 3, (3u << 28) | (20u << 12) | 4); // 32_UINT, select X.
        std::array<uint32_t, 8> expected;
        expected.fill(0xcccccccc);
        for (unsigned i = 0; i < expected.size(); ++i)
          wave->lds().write32(base + size - 16 + 4 * i, expected[i]);
        for (unsigned lane = 0; lane < 4; ++lane) {
          const auto page = lane == 0 && test.bad_source ? 7 : lane;
          cu->write_vgpr(vb, lane, 4096 * page);
          const uint32_t value = 0x66554480u + lane;
          f.mem->write32(0x10000 + 4096 * page + imm, value);
          if (test.requests & (1u << lane)) {
            const uint32_t result = op == rdna1::kBufferLoadSbyte   ? (0xffffff00u | (value & 0xff))
                                    : op == rdna1::kBufferLoadUbyte ? (value & 0xff)
                                                                    : value;
            expected[(start + 4 * lane - (size - 16)) / 4] = result;
          }
        }
        const auto words = rdna1::build_mubuf(
            op, {.offset = imm, .offen = 1, .lds = 1, .vaddr = 0, .vdata = 8, .soffset = 128});
        const auto pc = wave->pc;
        f.mem->load_image(reinterpret_cast<const uint8_t *>(words.data()), sizeof(words), pc);
        ASSERT_TRUE(cu->step());
        EXPECT_EQ(wave->pc, pc + sizeof(words));
        EXPECT_EQ(pages->seen, test.requests);
        ASSERT_EQ(observed->accesses.size(), 1u);
        const auto &access = observed->accesses.front();
        EXPECT_TRUE(access.lds_destination);
        EXPECT_EQ(access.active_lane_mask, test.exec);
        EXPECT_EQ(access.request_lane_mask, test.requests);
        EXPECT_EQ(access.valid_lane_mask, test.requests);
        EXPECT_EQ(access.issued_counters, std::vector{WaitCounterType::VMCNT});
        auto &state = *static_cast<RaceWavefrontState *>(wave->plugin_state(plugin->slot_index()))
                           ->race_state;
        const auto &events = state.getWaveMemoryEvents();
        ASSERT_EQ(events.size(), test.requests ? 1u : 0u);
        const auto &registry = state.getDetector()->events();
        const IntervalSet empty;
        const auto &intervals = events.empty() ? empty : registry.ldsIntervals(events.front());
        if (test.requests) {
          EXPECT_TRUE(registry.registers(events.front()).empty());
          EXPECT_EQ(registry.pendingCounterIncrement(events.front(), WaitCounterType::VMCNT), 1);
        }
        for (unsigned i = 0; i < expected.size(); ++i) {
          const unsigned addr = base + size - 16 + 4 * i;
          EXPECT_EQ(wave->lds().read32(addr), expected[i]);
          for (unsigned byte = 0; byte < 4; ++byte)
            EXPECT_EQ(intervals.firstOverlap(addr + byte, addr + byte + 1).has_value(),
                      expected[i] != 0xcccccccc);
        }
        wave->set_address_space({});
        cu->set_gpu_vm(nullptr);
        wave->halt();
      }
    }
  }
}

TEST(RaceDetectorPluginTest, Gfx10PairedDsUsesOneAccessAndOnlyData0ForEqualOffsets) {
  enum class Op { Read, Write, Exchange };
  struct Form {
    uint16_t read, write, exchange;
    unsigned width, stride;
  };
  const std::array forms{
      Form{rdna1::kDsRead2B32, rdna1::kDsWrite2B32, rdna1::kDsWrxchg2RtnB32, 1, 4},
      Form{rdna1::kDsRead2B64, rdna1::kDsWrite2B64, rdna1::kDsWrxchg2RtnB64, 2, 8},
      Form{rdna1::kDsRead2st64B32, rdna1::kDsWrite2st64B32, rdna1::kDsWrxchg2st64RtnB32, 1, 256},
      Form{rdna1::kDsRead2st64B64, rdna1::kDsWrite2st64B64, rdna1::kDsWrxchg2st64RtnB64, 2, 512}};
  for (const auto arch : {"rdna1", "rdna2"}) {
    for (const auto &form : forms) {
      for (const auto op : {Op::Read, Op::Write, Op::Exchange}) {
        for (const auto &[equal, data1] :
             std::array<std::pair<bool, unsigned>, 3>{{{false, 4}, {true, 4}, {true, 16}}}) {
          if (op == Op::Read && data1 != 4)
            continue;
          SCOPED_TRACE(testing::Message() << arch << "/" << form.write << "/" << int(op)
                                          << " equal=" << equal << " data1=" << data1);
          PluginFixture f;
          ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 32, 106, 32));
          PluginSinkConfig sinks;
          auto &sink = sinks.emplace<StringSink>();
          f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
          auto memory = std::make_unique<MemoryObservationPlugin>();
          auto *observed = memory.get();
          ASSERT_TRUE(f.plugin_group_->add(std::move(memory)));
          ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
          f.soc->set_plugin_group(f.plugin_group_);
          f.plugin_group_->onInit();
          auto *cu = f.cu();
          auto *wave = cu->dispatch_wf(0, 0x8000, 106, 16, 32);
          ASSERT_NE(wave, nullptr);
          const uint32_t lds = 1024, base = 32, source = 832;
          wave->set_lds_base(lds);
          wave->set_lds_size(1024);
          wave->set_m0(1024);
          wave->set_exec(1);
          std::array<Wavefront *, 1> waves{wave};
          f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
          const auto vb = wave->vgpr_alloc().base;
          cu->write_vgpr(vb, 0, base);
          cu->write_vgpr(vb + 12, 0, source);
          for (unsigned i = 0; i < form.width; ++i) {
            wave->lds().write32(lds + base + form.stride + 4 * i, 11 + i);
            wave->lds().write32(lds + base + 4 * i, 21 + i);
            wave->lds().write32(lds + source + 4 * i, 41 + i);
            cu->write_vgpr(vb + 2 + i, 0, 31 + i);
          }
          const auto producer = rdna1::build_ds(
              form.width == 1 ? rdna1::kDsReadB32 : rdna1::kDsReadB64, {.addr = 12, .vdst = 4});
          const auto pair = rdna1::build_ds(op == Op::Read    ? form.read
                                            : op == Op::Write ? form.write
                                                              : form.exchange,
                                            {.offset0 = 1,
                                             .offset1 = uint8_t(equal),
                                             .addr = 0,
                                             .data0 = 2,
                                             .data1 = uint8_t(data1),
                                             .vdst = 8});
          const uint32_t code[] = {producer[0], producer[1], pair[0], pair[1], S_ENDPGM};
          f.mem->load_image(reinterpret_cast<const uint8_t *>(code), sizeof(code), wave->pc);
          ASSERT_TRUE(cu->step());
          ASSERT_EQ(wave->pc, 0x8008u);
          const auto warnings = cu->memory_wait_diagnostic_count();
          const auto output = sink.str().size();
          ASSERT_TRUE(cu->step());
          ASSERT_EQ(wave->pc, 0x8010u);
          const bool reads_pending_data1 = op != Op::Read && !equal;
          EXPECT_EQ(cu->memory_wait_diagnostic_count() != warnings, reads_pending_data1);
          EXPECT_EQ(sink.str().find("type=VGPR access=read reg=4 ", output) != std::string::npos,
                    reads_pending_data1);
          ASSERT_EQ(observed->accesses.size(), 2u);
          const auto &access = observed->accesses.back();
          EXPECT_EQ(access.request_lane_mask, 1u);
          ASSERT_EQ(access.issued_counters.size(), 1u);
          EXPECT_EQ(access.issued_counters.front(), WaitCounterType::LGKMCNT);
          EXPECT_EQ(access.secondary_addresses.empty(), equal);
          if (!access.secondary_addresses.empty()) {
            EXPECT_EQ(access.secondary_addresses[0], lds + base);
          }
          for (unsigned i = 0; i < form.width; ++i) {
            EXPECT_EQ(wave->lds().read32(lds + base + form.stride + 4 * i),
                      (op == Op::Read ? 11 : 31) + i);
            EXPECT_EQ(wave->lds().read32(lds + base + 4 * i),
                      (equal || op == Op::Read ? 21 : 41) + i);
            if (op != Op::Write) {
              EXPECT_EQ(cu->read_vgpr_storage(vb + 8 + i, 0), 11 + i);
              EXPECT_EQ(cu->read_vgpr_storage(vb + 8 + form.width + i, 0), (equal ? 11 : 21) + i);
            }
          }
          wave->halt();
        }
      }
    }
  }
}

TEST(RaceDetectorPluginTest, IndexedDsReportsOnlyExecutedWorkgroupLdsBytes) {
  enum class Access { Read, Write, Add, AddReturn };
  struct Form {
    uint16_t opcode;
    Access access;
    uint8_t words;
    bool dual = false;
  };
  for (const auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    const bool gfx1010 = arch == ROCJITSU_CODE_ARCH_RDNA1;
    const uint32_t size = gfx1010 ? 512 : 1024;
    const auto check = [&](Form form, uint32_t offset, bool reverse, unsigned probe) {
      SCOPED_TRACE(testing::Message() << int(arch) << "/" << form.opcode << "/" << offset << "/"
                                      << reverse << "/" << probe);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(2, gfx1010 ? "rdna1" : "rdna2", 32, 106, 32));
      PluginSinkConfig sinks;
      auto &sink = sinks.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
      auto race = std::make_unique<RaceDetectorPlugin>();
      const auto *plugin = race.get();
      ASSERT_TRUE(f.plugin_group_->add(std::move(race)));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      auto *cu = f.cu();
      const uint32_t base = size;
      std::array<Wavefront *, 2> waves{cu->dispatch_wf(0, 0x8000, 106, 16, 32),
                                       cu->dispatch_wf(0, 0xa000, 106, 16, 32)};
      const uint8_t first = form.dual && reverse ? 1 : 0;
      const uint8_t second = form.dual && !reverse ? 1 : 0;
      const unsigned width = form.words * 4;
      const std::array<uint32_t, 2> addresses{offset + first * width, offset + second * width};
      for (unsigned who = 0; who < waves.size(); ++who) {
        auto *wave = waves[who];
        ASSERT_NE(wave, nullptr);
        wave->set_lds_base(base);
        wave->set_lds_size(size);
        wave->set_m0(size);
        wave->set_exec(1);
        const auto vb = wave->vgpr_alloc().base;
        cu->write_vgpr(vb, 0, who ? addresses[probe] : offset);
        for (unsigned i = 2; i < 12; ++i)
          cu->write_vgpr(vb + i, 0, i < 8 ? i + 1 : 0xaaaaaaaa);
      }
      auto &lds = waves[0]->lds();
      ASSERT_GT(lds.size_bytes(), base + size + 32);
      for (uint32_t p = base + size - 32; p < base + size + 32; p += 4)
        lds.write32(p, 41);
      f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
      const auto issue = [&](unsigned who, const auto &words) {
        waves[1 - who]->set_runtime_suspended(true);
        auto *wave = waves[who];
        wave->set_runtime_suspended(false);
        const uint64_t pc = wave->pc + 0x100;
        wave->pc = pc;
        f.mem->load_image(reinterpret_cast<const uint8_t *>(words.data()), words.size() * 4, pc);
        for (unsigned tick = 0; tick < 32 && wave->pc == pc; ++tick)
          ASSERT_TRUE(cu->step());
        ASSERT_EQ(wave->pc, pc + words.size() * 4);
      };
      ASSERT_NO_FATAL_FAILURE(issue(0, rdna1::build_ds(form.opcode, {.offset0 = first,
                                                                     .offset1 = second,
                                                                     .addr = 0,
                                                                     .data0 = 2,
                                                                     .data1 = 6,
                                                                     .vdst = 8})));
      auto &state = *static_cast<RaceWavefrontState *>(waves[0]->plugin_state(plugin->slot_index()))
                         ->race_state;
      const auto &events = state.getWaveMemoryEvents();
      const bool write = form.access != Access::Read;
      const bool returns = form.access == Access::Read || form.access == Access::AddReturn;
      const auto executes = [&](unsigned part) { return addresses[part] + width <= size; };
      const unsigned parts = form.dual ? 2 : 1;
      const bool tracked = returns || executes(0) || (form.dual && executes(1));
      ASSERT_EQ(events.size(), unsigned(tracked));
      const auto &registry = state.getDetector()->events();
      const IntervalSet empty;
      const auto &intervals = events.empty() ? empty : registry.ldsIntervals(events.front());
      EXPECT_EQ(waves[0]->ensure_memory_wait_scoreboard().outstanding(WaitCounterKind::Ds), 1u);
      if (tracked) {
        EXPECT_EQ(registry.pendingCounterIncrement(events.front(), WaitCounterType::LGKMCNT), 1);
        EXPECT_EQ(registry.registers(events.front()).size(), returns ? parts * form.words : 0u);
      }
      for (unsigned part = 0; part < parts; ++part) {
        const uint32_t address = base + addresses[part];
        for (unsigned i = 0; i < width; ++i)
          EXPECT_EQ(intervals.firstOverlap(address + i, address + i + 1).has_value(),
                    executes(part));
        for (unsigned i = 0; i < form.words; ++i) {
          const uint32_t operand = (part ? 7u : 3u) + i;
          const uint32_t stored = form.access == Access::Write ? operand : 41 + operand;
          EXPECT_EQ(lds.read32(address + 4 * i), write && executes(part) ? stored : 41u);
          if (returns) {
            EXPECT_EQ(
                cu->read_vgpr_storage(waves[0]->vgpr_alloc().base + 8 + part * form.words + i, 0),
                executes(part) ? 41u : 0u);
          }
        }
      }
      const auto before = sink.str().size();
      ASSERT_NO_FATAL_FAILURE(
          issue(1, rdna1::build_ds(write ? rdna1::kDsReadB32 : rdna1::kDsWriteB32,
                                   {.addr = 0, .data0 = 2, .vdst = 12})));
      EXPECT_EQ(sink.str().find("type=LDS ", before) != std::string::npos, executes(probe));
      if (write) {
        EXPECT_EQ(cu->read_vgpr_storage(waves[1]->vgpr_alloc().base + 12, 0),
                  addresses[probe] + 4 > size ? 0u : lds.read32(base + addresses[probe]));
      }
      if (returns && !executes(probe)) {
        const auto read = rdna1::build_vop1(
            rdna1::kVMovB32Vop1, {.src0 = uint16_t(264 + probe * form.words), .vdst = 14});
        for (const uint8_t lgkm : {1u, 0u}) {
          ASSERT_NO_FATAL_FAILURE(issue(0, std::array{legacy_wait_word(arch, 63, lgkm)}));
          const auto warnings = cu->memory_wait_diagnostic_count();
          const auto output = sink.str().size();
          ASSERT_NO_FATAL_FAILURE(issue(0, read));
          EXPECT_EQ(cu->memory_wait_diagnostic_count() != warnings, lgkm == 1);
          EXPECT_EQ(sink.str().find("type=VGPR access=read", output) != std::string::npos,
                    lgkm == 1);
        }
      }
      for (auto *wave : waves)
        wave->halt();
    };
    for (const Form form :
         {Form{rdna1::kDsReadB32, Access::Read, 1}, Form{rdna1::kDsWriteB32, Access::Write, 1},
          Form{rdna1::kDsAddRtnU32, Access::AddReturn, 1}, Form{rdna1::kDsAddU32, Access::Add, 1},
          Form{rdna1::kDsReadB128, Access::Read, 4}, Form{rdna1::kDsWriteB128, Access::Write, 4}}) {
      ASSERT_NO_FATAL_FAILURE(check(form, size - 4 * form.words, false, 0));
      ASSERT_NO_FATAL_FAILURE(check(form, size, false, 0));
      if (form.words > 1) {
        ASSERT_NO_FATAL_FAILURE(check(form, size - 4, false, 0));
      }
    }
    for (const Form form : {Form{rdna1::kDsRead2B32, Access::Read, 1, true},
                            Form{rdna1::kDsWrite2B64, Access::Write, 2, true}}) {
      for (const bool reverse : {false, true})
        for (const unsigned probe : {0u, 1u}) {
          ASSERT_NO_FATAL_FAILURE(check(form, size - 8 * form.words, reverse, probe));
          ASSERT_NO_FATAL_FAILURE(check(form, size - (form.words == 1 ? 4 : 12), reverse, probe));
        }
    }
  }
}

TEST(RaceDetectorPluginTest, Gfx10FlatLdsBytesAndReturnedRegistersHaveIndependentWaits) {
  enum class Op { FlatRead, FlatWrite, FlatAdd, DsRead, DsWrite, DsAdd, DsAdd64, Direct };
  enum class Wait { None, Vm, Ds, Both, DsOne, DsOneWithYounger };
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    const auto check = [&](Op first, Op next, Wait wait, bool peer = false, bool barrier = false,
                           bool overlap = true, uint64_t flat_exec = 5) {
      SCOPED_TRACE(testing::Message() << int(arch) << "/" << int(first) << "/" << int(next) << "/"
                                      << int(wait) << " peer=" << peer << " barrier=" << barrier
                                      << " overlap=" << overlap << " exec=" << flat_exec);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(
          f.initialize(2, arch == ROCJITSU_CODE_ARCH_RDNA1 ? "rdna1" : "rdna2", 32, 106, 32));
      PluginSinkConfig sinks;
      auto &sink = sinks.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      auto *cu = f.cu();
      constexpr uint64_t shared = 0x0001'0000'0000'0000ULL;
      cu->set_apertures(shared, shared + 0xffff, 0, 0);
      std::array<Wavefront *, 2> waves{cu->dispatch_wf(0, 0x8000, 106, 16, 32),
                                       cu->dispatch_wf(0, 0xa000, 106, 16, 32)};
      for (auto *wave : waves) {
        ASSERT_NE(wave, nullptr);
        wave->set_lds_size(256);
        const auto vb = wave->vgpr_alloc().base;
        for (unsigned lane = 0; lane < 3; ++lane) {
          // Lane 0 accesses LDS byte 4. Lane 2 is global. Inactive lane 1 points to LDS byte 0.
          uint64_t address = lane == 2 ? 0x5000 : shared + (lane == 0 ? 4 : 0);
          cu->write_vgpr(vb, lane, uint32_t(address));
          cu->write_vgpr(vb + 1, lane, address >> 32);
          cu->write_vgpr(vb + 4, lane, overlap ? 4 : 0);
          cu->write_vgpr(vb + 5, lane, 99);
          cu->write_vgpr(vb + 6, lane, 0);
        }
        const auto sb = wave->sgpr_alloc().base;
        cu->write_sgpr(sb, 0x5000);
        cu->write_sgpr(sb + 1, 0);
        cu->write_sgpr(sb + 2, 256);
        cu->write_sgpr(sb + 3, (3u << 28) | (20u << 12));
      }
      f.mem->write32(0x5000, 99);
      waves[0]->lds().write32(waves[0]->lds_base() + 4, 41);
      ASSERT_EQ(waves[0]->lds_base(), waves[1]->lds_base());
      f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 106, waves);
      const auto issue = [&](unsigned who, const auto &words, uint64_t exec) {
        auto *wave = waves[who];
        waves[1 - who]->set_runtime_suspended(true);
        wave->set_runtime_suspended(false);
        wave->set_exec(exec);
        const uint64_t pc = wave->pc + 0x100;
        wave->pc = pc;
        f.mem->load_image(reinterpret_cast<const uint8_t *>(words.data()),
                          words.size() * sizeof(uint32_t), pc);
        for (unsigned tick = 0; tick < 32 && wave->pc == pc; ++tick)
          ASSERT_TRUE(cu->step());
        ASSERT_EQ(wave->pc, pc + 4 * words.size());
      };
      const auto is_flat = [](Op op) {
        return op == Op::FlatRead || op == Op::FlatWrite || op == Op::FlatAdd;
      };
      const auto memory = [&](unsigned who, Op op, uint8_t dest) {
        waves[who]->set_m0(op == Op::Direct ? (overlap ? 4 : 0) : 256);
        if (is_flat(op)) {
          const auto opcode = op == Op::FlatRead    ? rdna1::kFlatLoadDword
                              : op == Op::FlatWrite ? rdna1::kFlatStoreDword
                                                    : rdna1::kFlatAtomicAdd;
          issue(who,
                rdna1::build_flat(
                    opcode,
                    {.glc = uint8_t(op == Op::FlatAdd), .data = 5, .saddr = 125, .vdst = dest}),
                flat_exec);
        } else if (op == Op::Direct) {
          issue(who, rdna1::build_mubuf(rdna1::kBufferLoadDword, {.lds = 1, .soffset = 128}), 1);
        } else {
          const auto opcode = op == Op::DsRead    ? rdna1::kDsReadB32
                              : op == Op::DsWrite ? rdna1::kDsWriteB32
                              : op == Op::DsAdd64 ? rdna1::kDsAddRtnU64
                                                  : rdna1::kDsAddRtnU32;
          const uint8_t addr = op == Op::DsAdd64 ? 6 : 4;
          issue(who, rdna1::build_ds(opcode, {.addr = addr, .data0 = 5, .vdst = dest}), 1);
        }
      };
      ASSERT_NO_FATAL_FAILURE(memory(0, first, 8));
      const bool returns = first == Op::FlatRead || first == Op::FlatAdd;
      if (returns && (flat_exec & 1)) {
        EXPECT_EQ(cu->read_vgpr_storage(waves[0]->vgpr_alloc().base + 8, 0), 41u);
      }
      if (wait == Wait::DsOneWithYounger) {
        ASSERT_NO_FATAL_FAILURE(
            issue(0, rdna1::build_ds(rdna1::kDsReadB32, {.addr = 4, .vdst = 12}), 0));
      }
      const bool vm = wait == Wait::Vm || wait == Wait::Both;
      const bool lgkm = wait == Wait::Ds || wait == Wait::Both;
      const bool partial = wait == Wait::DsOne || wait == Wait::DsOneWithYounger;
      const bool ds_ready = lgkm || wait == Wait::DsOneWithYounger;
      ASSERT_NO_FATAL_FAILURE(issue(0,
                                    std::array{legacy_wait_word(arch, vm ? 0 : 63,
                                                                lgkm      ? 0
                                                                : partial ? 1
                                                                          : 63)},
                                    1));
      if (first == Op::FlatWrite && vm) {
        ASSERT_NO_FATAL_FAILURE(
            issue(0, rdna1::build_sopk(rdna1::kSWaitcntVscnt, {.sdst = 125}), 1));
      }
      if (barrier)
        f.plugin_group_->onAmdgpuBarrierResolved(waves, BarrierKind::Workgroup);
      const auto before = sink.str().size();
      ASSERT_NO_FATAL_FAILURE(memory(peer ? 1 : 0, next, 9));
      const bool atomic_pair = (first == Op::FlatAdd && next == Op::DsAdd) ||
                               (first == Op::DsAdd && next == Op::FlatAdd);
      const bool lds_conflict = !atomic_pair && overlap && (flat_exec & 1) &&
                                (peer ? !(ds_ready && barrier) : next == Op::Direct && !ds_ready);
      EXPECT_EQ(sink.str().find("type=LDS ", before) != std::string::npos, lds_conflict)
          << sink.str().substr(before);
      // The plugin suppresses further reports involving an instruction already reported as racing.
      if (returns && (flat_exec & 1) && !lds_conflict) {
        const auto output = sink.str().size();
        const auto warnings = cu->memory_wait_diagnostic_count();
        ASSERT_NO_FATAL_FAILURE(
            issue(0, rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = 264, .vdst = 10}), 1));
        EXPECT_EQ(sink.str().find("type=VGPR access=read reg=8 ", output) != std::string::npos,
                  !(vm && ds_ready));
        EXPECT_EQ(cu->memory_wait_diagnostic_count() != warnings, !(vm && ds_ready));
      }
      for (auto *wave : waves)
        wave->halt();
    };
    ASSERT_NO_FATAL_FAILURE(check(Op::FlatAdd, Op::DsAdd, Wait::None, true));
    ASSERT_NO_FATAL_FAILURE(check(Op::DsAdd, Op::FlatAdd, Wait::None, true));
    // A 64-bit atomic at byte zero overlaps the DWORD at byte four without
    // denoting the same atomic element.
    ASSERT_NO_FATAL_FAILURE(check(Op::FlatAdd, Op::DsAdd64, Wait::None, true));
    ASSERT_NO_FATAL_FAILURE(check(Op::DsAdd64, Op::FlatAdd, Wait::None, true));
    for (auto [first, next] : {std::pair{Op::FlatRead, Op::DsWrite},
                               {Op::DsWrite, Op::FlatRead},
                               {Op::FlatWrite, Op::DsRead},
                               {Op::DsRead, Op::FlatWrite},
                               {Op::FlatAdd, Op::DsRead},
                               {Op::DsRead, Op::FlatAdd},
                               {Op::FlatRead, Op::DsAdd},
                               {Op::DsAdd, Op::FlatRead},
                               {Op::FlatRead, Op::Direct}}) {
      for (auto wait :
           {Wait::None, Wait::Vm, Wait::Ds, Wait::Both, Wait::DsOne, Wait::DsOneWithYounger}) {
        ASSERT_NO_FATAL_FAILURE(check(first, next, wait));
      }
      ASSERT_NO_FATAL_FAILURE(check(first, next, Wait::Ds, true));
      ASSERT_NO_FATAL_FAILURE(check(first, next, Wait::Ds, true, true));
      ASSERT_NO_FATAL_FAILURE(check(first, next, Wait::Vm, true, true));
      ASSERT_NO_FATAL_FAILURE(check(first, next, Wait::None, true, false, false));
      ASSERT_NO_FATAL_FAILURE(check(first, next, Wait::None, true, false, true, 4));
    }
  }
}

TEST(RaceDetectorPluginTest, Gfx10FlatWritebackWithIndexedLdsAndOrdinaryVmem) {
  for (const auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    SCOPED_TRACE(static_cast<unsigned>(arch));
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(
        f.initialize(1, arch == ROCJITSU_CODE_ARCH_RDNA1 ? "rdna1" : "rdna2", 32, 106, 32));
    PluginSinkConfig sinks;
    auto &sink = sinks.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    auto *cu = f.cu();
    constexpr uint64_t shared = 0x0001'0000'0000'0000ULL;
    cu->set_apertures(shared, shared + 0xFFFF, 0, 0);
    auto *wave = cu->dispatch_wf(0, 0x8000, 106, 16, 32);
    ASSERT_NE(wave, nullptr);
    wave->set_lds_base(cu->allocate_lds(256));
    wave->set_lds_size(256);
    wave->set_m0(256);
    std::array<Wavefront *, 1> waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 16, 106, waves);
    const auto vb = wave->vgpr_alloc().base;
    for (unsigned lane = 0; lane < 2; ++lane) {
      const uint32_t offset = lane * 4;
      cu->write_vgpr(vb, lane, offset);
      cu->write_vgpr(vb + 1, lane, shared >> 32);
      cu->write_vgpr(vb + 4, lane, offset);
      wave->lds().write32(wave->lds_base() + offset, 41 + lane);
    }
    const auto sb = wave->sgpr_alloc().base;
    cu->write_sgpr(sb, 0x4000);
    cu->write_sgpr(sb + 1, 0);
    cu->write_sgpr(sb + 2, 256);
    cu->write_sgpr(sb + 3, (3u << 28) | (20u << 12));
    f.mem->write32(0x4000, 41);
    enum class Op { Lds, Flat, Buffer };
    const auto load = [](Op op, uint8_t dst) {
      if (op == Op::Buffer)
        return rdna2::build_mubuf(rdna2::kBufferLoadDword, {.vdata = dst, .soffset = 128});
      return op == Op::Flat ? rdna2::build_flat(rdna2::kFlatLoadDwordFlat,
                                                {.addr = 0, .saddr = 127, .vdst = dst})
                            : rdna2::build_ds(rdna2::kDsReadB32, {.addr = 4, .vdst = dst});
    };
    const auto check_waw = [&](Op older, Op younger, bool wait_vm, bool wait_ds,
                               uint64_t second_exec, uint8_t second_dst) {
      SCOPED_TRACE(testing::Message()
                   << int(older) << "/" << int(younger) << " wait_vm=" << wait_vm
                   << " wait_ds=" << wait_ds << " exec=" << second_exec << " dst=" << +second_dst);
      const auto first = load(older, 8);
      const auto second = load(younger, second_dst);
      const uint64_t pc = wave->pc + 0x100;
      wave->pc = pc;
      const uint32_t code[] = {
          first[0],  first[1],  legacy_wait_word(arch, wait_vm ? 0 : 63, wait_ds ? 0 : 63),
          second[0], second[1], legacy_wait_word(arch, 0, 0),
          S_ENDPGM};
      f.mem->load_image(reinterpret_cast<const uint8_t *>(code), sizeof(code), pc);
      const auto advance = [&](unsigned word) {
        for (unsigned tick = 0; tick < 32 && wave->pc < pc + word * 4; ++tick)
          ASSERT_TRUE(cu->step());
        ASSERT_EQ(wave->pc, pc + word * 4);
      };
      wave->set_exec(1);
      ASSERT_NO_FATAL_FAILURE(advance(3));
      EXPECT_EQ(cu->read_vgpr_storage(vb + 8, 0), 41u);
      wave->set_exec(second_exec);
      const auto before = sink.str().size();
      const auto warnings = cu->memory_wait_diagnostic_count();
      ASSERT_NO_FATAL_FAILURE(advance(5));
      const bool overlap = (second_exec & 1) && second_dst == 8;
      // Only an incoming ordinary VMEM load can supersede FLAT's VM
      // writeback after LGKM is proven complete. An incoming generic FLAT
      // has no writeback-order exemption.
      const bool pending_vm = older != Op::Lds && younger != Op::Buffer && !wait_vm;
      const bool pending_ds =
          older != Op::Buffer && !(older == Op::Lds && younger == Op::Lds) && !wait_ds;
      const bool needs_wait = overlap && (pending_vm || pending_ds);
      EXPECT_EQ(cu->memory_wait_diagnostic_count() != warnings, needs_wait);
      EXPECT_EQ(sink.str().find("access=write", before) != std::string::npos, needs_wait)
          << sink.str().substr(before);
      ASSERT_NO_FATAL_FAILURE(advance(6));
    };
    for (const Op older : {Op::Lds, Op::Flat, Op::Buffer})
      for (const Op younger : {Op::Lds, Op::Flat, Op::Buffer}) {
        for (const bool wait_vm : {false, true})
          for (const bool wait_ds : {false, true})
            ASSERT_NO_FATAL_FAILURE(check_waw(older, younger, wait_vm, wait_ds, 1, 8));
        ASSERT_NO_FATAL_FAILURE(check_waw(older, younger, false, false, 2, 8));
        ASSERT_NO_FATAL_FAILURE(check_waw(older, younger, false, false, 1, 9));
      }
    wave->halt();
  }
}

TEST(RaceDetectorPluginTest, Gfx950FlatEmptyPortionDoesNotOrderAnOlderResult) {
  for (const bool lds : {false, true}) {
    // Exercise both a partial wait and enough issues to trigger capacity
    // constraints if the checker retains empty FLAT portions.
    for (const int flat_count : {1, lds ? 64 : 16}) {
      SCOPED_TRACE(lds);
      SCOPED_TRACE(flat_count);
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
      PluginSinkConfig sink_config;
      auto &sink = sink_config.emplace<StringSink>();
      auto plugin = std::make_unique<RaceDetectorPlugin>();
      auto *plugin_ptr = plugin.get();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
      ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      auto *wf = f.cu()->dispatch_wf(0, 0x100, 104, 256);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(1);
      std::array<amdgpu::Wavefront *, 1> waves{wf};
      f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, 104, waves);
      auto *plugin_state =
          static_cast<RaceWavefrontState *>(wf->plugin_state(plugin_ptr->slot_index()));
      ASSERT_NE(plugin_state, nullptr);
      auto &race = *plugin_state->race_state;
      const auto empty_counter = lds ? WaitCounterType::VMCNT : WaitCounterType::LGKMCNT;

      // ds_read v9; flat_load(global) v8; vmcnt(0); lgkmcnt(1)
      // and the converse with global_load v9 followed by FLAT-to-LDS.
      // The empty FLAT portion cannot be used as a younger ordered operation
      // to prove v9 ready, even after many FLAT loads have completed their results.
      register_pending_event(
          race, 0x80, lds ? MemoryEventType::GLOBAL_TO_VGPR : MemoryEventType::LDS_TO_VGPR, {9},
          empty_counter, lds ? MemoryOrderClass::VMEM : MemoryOrderClass::LDS);
      auto state = std::make_unique<VectorMemState>(lds ? LOCAL_MEM : GLOBAL_MEM);
      state->is_load = true;
      state->elem_size = 4;
      state->num_elems = 1;
      state->dst_reg_base = wf->vgpr_alloc().base + 8;
      state->exec_mask = state->lane_mask = 1;
      state->wf_size = 64;
      TestMemoryInstruction flat(std::move(state),
                                 {{WaitCounterType::VMCNT, MemoryCompletionClass::UNORDERED},
                                  {WaitCounterType::LGKMCNT, MemoryCompletionClass::UNORDERED}});
      MemoryAccessObservation access;
      access.route = lds ? MemoryRoute::LOCAL : MemoryRoute::GLOBAL;
      access.decoded_space = DecodedMemorySpace::FLAT;
      access.request_lane_mask = access.active_lane_mask = 1;
      access.flat_local_lane_mask = lds ? 1 : 0;

      const auto apply_wait = [&](uint8_t vmcnt, uint8_t lgkmcnt) {
        execute_wait(*f.plugin_group_, *wf, legacy_wait_word(wf->cu().arch(), vmcnt, lgkmcnt));
      };
      for (int i = 0; i < flat_count; ++i) {
        f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, flat, *wf);
        f.plugin_group_->onAmdgpuMemoryAccessRouted(access, flat, *wf);
        apply_wait(lds ? 63 : 0, lds ? 0 : 15);
      }
      EXPECT_TRUE(race.getVgprMemoryEvents(8).empty());

      apply_wait(lds ? 1 : 63, lds ? 15 : 1);
      race.checkVgprReadLanes(8, 1, 0xF);
      EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
      race.checkVgprReadLanes(9, 1, 0xF);
      EXPECT_NE(sink.str().find("RACE "), std::string::npos);
      apply_wait(0, 0);
      EXPECT_TRUE(race.getWaveMemoryEvents().empty());
      EXPECT_TRUE(race.getVgprMemoryEvents(9).empty());
    }
  }
}

TEST(RaceDetectorPluginTest, FlatCounterCapacityRequiresResolvedDomain) {
  using namespace waitcheck_detail;
  enum class Phase { BeforeRouting, Routed, UnknownObservation };
  for (const bool requests : {false, true}) {
    for (const bool full_lds : {false, true}) {
      for (const bool flat_lds : {false, true}) {
        for (const auto phase : {Phase::BeforeRouting, Phase::Routed, Phase::UnknownObservation}) {
          const bool before_routing = phase == Phase::BeforeRouting;
          SCOPED_TRACE(full_lds);
          SCOPED_TRACE(requests);
          SCOPED_TRACE(flat_lds);
          SCOPED_TRACE(static_cast<int>(phase));
          PluginFixture f;
          ASSERT_NO_FATAL_FAILURE(f.initialize(1));
          PluginSinkConfig sink_config;
          auto &sink = sink_config.emplace<StringSink>();
          auto plugin = std::make_unique<RaceDetectorPlugin>();
          auto *plugin_ptr = plugin.get();
          f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
          ASSERT_TRUE(f.plugin_group_->add(std::move(plugin)));
          f.soc->set_plugin_group(f.plugin_group_);
          f.plugin_group_->onInit();
          auto *wf = f.cu()->dispatch_wf(0, 0x100, 104, 256);
          ASSERT_NE(wf, nullptr);
          wf->set_exec(1);
          std::array<amdgpu::Wavefront *, 1> waves{wf};
          f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, 104, waves);
          auto &race =
              *static_cast<RaceWavefrontState *>(wf->plugin_state(plugin_ptr->slot_index()))
                   ->race_state;
          auto &core = wf->ensure_memory_wait_scoreboard();
          unsigned core_reports = 0;
          core.bind(0x200, &core_reports,
                    [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
          const auto counter = full_lds ? WaitCounterKind::Ds : WaitCounterKind::Load;
          const ClassifiedEvent older{counter, full_lds ? WaitEventKind::Ds
                                                        : WaitEventKind::VmemNoSamplerLoad};
          const unsigned capacity = full_lds ? 15 : 63;
          for (unsigned i = 0; i < capacity; ++i) {
            const uint32_t reg = 16 + i;
            register_pending_event(
                race, 0x80 + i * 4,
                full_lds ? MemoryEventType::LDS_TO_VGPR : MemoryEventType::GLOBAL_TO_VGPR, {reg},
                full_lds ? WaitCounterType::LGKMCNT : WaitCounterType::VMCNT,
                full_lds ? MemoryOrderClass::LDS : MemoryOrderClass::VMEM);
            const auto sequence = core.issue(older, ROCJITSU_CODE_ARCH_CDNA4);
            core.add({sequence,
                      0x80 + i * 4,
                      1,
                      {RegClass::VGPR, static_cast<uint16_t>(reg), 1},
                      counter,
                      0xF});
          }

          auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
          const auto words =
              cdna4::build_flat(cdna4::kFlatLoadDwordFlat, {.addr = 0, .saddr = 0x7F, .vdst = 8});
          std::unique_ptr<Instruction> flat(decode_valid(*decoder, words.data()));
          ASSERT_NE(flat, nullptr);
          f.plugin_group_->onAmdgpuBeforeExecuteInstruction(0x100, *flat, *wf);
          core.before(*flat, ROCJITSU_CODE_ARCH_CDNA4);
          if (!before_routing) {
            auto data = std::make_unique<VectorMemState>(flat_lds ? LOCAL_MEM : GLOBAL_MEM);
            data->exec_mask = 1;
            data->lane_mask = requests ? 1 : 0;
            data->elem_size = 4;
            data->num_elems = 1;
            data->dst_reg_base = wf->vgpr_alloc().base + 8;
            flat->set_data(std::move(data));
            MemoryAccessObservation access;
            access.route = phase == Phase::UnknownObservation ? MemoryRoute::UNKNOWN
                           : flat_lds                         ? MemoryRoute::LOCAL
                                                              : MemoryRoute::GLOBAL;
            access.decoded_space = DecodedMemorySpace::FLAT;
            access.active_lane_mask = 1;
            access.request_lane_mask = requests ? 1 : 0;
            access.flat_local_lane_mask = flat_lds && requests ? 1 : 0;
            f.plugin_group_->onAmdgpuMemoryAccessRouted(access, *flat, *wf);
            constexpr uint64_t shared_base = uint64_t{1} << 32;
            f.cu()->set_apertures(shared_base, shared_base + UINT32_MAX, 0, 0);
            f.cu()->write_vgpr(wf->vgpr_alloc().base, 0, 0x100);
            f.cu()->write_vgpr(wf->vgpr_alloc().base + 1, 0, flat_lds ? 1 : 2);
            // The core derives requests from architectural state before execution.
            // Empty EXEC exercises the same no-admission case as a rejected
            // request in the plugin's synthetic routing observation.
            wf->set_exec(requests ? 1 : 0);
            f.cu()->track_memory_wait(*flat, *wf);
          }

          // Before routing, FLAT proves neither domain completed an old request.
          // After routing, only the domain actually used needs a free counter slot.
          core.bind(0x200, &core_reports,
                    [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
          if (phase == Phase::Routed && !requests) {
            // A rejected request can zero-fill an active destination without
            // adding an asynchronous memory dependency or capacity pressure.
            race.checkVgprReadLanes(8, 1, 0xF);
            core.access({RegClass::VGPR, 8, 1}, 1, 0xF, false);
            EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
            EXPECT_EQ(core_reports, 0u);
          }
          race.checkVgprReadLanes(16, 1, 0xF);
          core.access({RegClass::VGPR, 16, 1}, 1, 0xF, false);
          const bool missing_wait = before_routing || !requests || full_lds != flat_lds;
          // The core has the actual masks; an unresolved plugin observation
          // must not infer admission in either domain from decoded metadata.
          EXPECT_EQ(sink.str().find("RACE ") != std::string::npos,
                    missing_wait || phase == Phase::UnknownObservation);
          EXPECT_EQ(core_reports != 0, missing_wait);
        }
      }
    }
  }
}

TEST(RaceDetectorPluginTest, NamedLgkmcntWaitRetiresSplitScalarEvent) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1, /*arch=*/"rdna3", /*wavefront_size=*/32,
                                       /*sgprs_per_wf=*/128));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/128, /*vgprs=*/256, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/128, waves);

  auto state = std::make_unique<ScalarMemState>();
  state->dst_register = {ScalarRegisterStorage::TTMP, 0, 1};
  state->is_load = true;
  state->wait_counter_type = WaitCounterType::KMCNT;
  TestMemoryInstruction load(std::move(state));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, load, *wf);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

  execute_wait(*f.plugin_group_, *wf, 0xbdfc0000u); // s_waitcnt_lgkmcnt null, 0.
  static_cast<void>(RegisterAccess(*wf).read_ttmp(0));
  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

TEST(ExecutionPluginTest, MfmaReadObservationReportsRace) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto plugin = std::make_unique<MfmaRacePlugin>();
  auto *race_plugin = plugin.get();
  f.plugin_group_->add(std::move(plugin));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0,
                                               /*physical_vgpr_count=*/256,
                                               /*physical_sgpr_count=*/104, waves);

  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint32_t S0 = 0;

  race_plugin->registerOutstandingLoad(S0, /*exec_mask=*/1u);
  amdgpu::observe_mfma_input_reads(*cu, vb + S0, /*dim=*/16, /*K=*/32, /*B=*/1,
                                   /*data_bits=*/16, wf->wf_size());

  ASSERT_FALSE(race_plugin->violations.empty());
  const auto &violation = race_plugin->violations.front();
  EXPECT_EQ(violation.space, RaceViolation::Space::VGPR);
  EXPECT_EQ(violation.index, static_cast<int>(S0));
  EXPECT_EQ(violation.lane, 0);
}

TEST(ExecutionPluginTest, MfmaFastPathReadHookReportsRace) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "stdx SIMD is unavailable";
  } else {
    constexpr uint32_t width = static_cast<uint32_t>(util::native<float>::size());
    if (!amdgpu::mma_f32_native_width_supported(16, width))
      GTEST_SKIP() << "f16 MFMA shape is not divisible by the native SIMD width";

    struct ForceScalarGuard {
      bool old = util::force_scalar();
      ~ForceScalarGuard() { util::set_force_scalar_for_testing(old); }
    } force_scalar_guard;
    util::set_force_scalar_for_testing(false);

    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto plugin = std::make_unique<MfmaRacePlugin>();
    auto *race_plugin = plugin.get();
    f.plugin_group_->add(std::move(plugin));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    ASSERT_EQ(wf->wf_size(), 64u);
    std::array<amdgpu::Wavefront *, 1> waves{wf};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0,
                                                 /*physical_vgpr_count=*/256,
                                                 /*physical_sgpr_count=*/104, waves);

    const uint32_t vb = wf->vgpr_alloc().base;
    constexpr uint32_t S0 = 0, S1 = 16, ACC = 32, DST = 48;
    for (uint32_t reg = 0; reg < 64; ++reg)
      for (uint32_t lane = 0; lane < 64; ++lane)
        cu->write_vgpr(vb + reg, lane, 0x3c003c00u);

    race_plugin->registerOutstandingLoad(S0, /*exec_mask=*/1u);
    amdgpu::exec_f32_mfma_f16_spec<16, 16, 32>(*cu, vb + DST, vb + S0, vb + S1, vb + ACC,
                                               amdgpu::ACC_FROM_VGPR, /*cbsz=*/0, /*abid=*/0,
                                               /*blgp=*/0);

    ASSERT_FALSE(race_plugin->violations.empty());
    const auto &violation = race_plugin->violations.front();
    EXPECT_EQ(violation.space, RaceViolation::Space::VGPR);
    EXPECT_EQ(violation.index, static_cast<int>(S0));
    EXPECT_EQ(violation.lane, 0);
  }
}

namespace mfma_race {

constexpr uint32_t A = 0, B = 16, C = 32, D = 64;
using Run = std::function<void(ComputeUnitCore &, uint32_t vb, uint32_t acc)>;

/// Runs @p run after an outstanding load to one lane of @p reg and reports whether the race
/// detector flagged that register.
bool races_on(const Run &run, uint32_t reg, uint32_t lane, uint32_t acc, bool force_scalar) {
  PluginFixture f;
  f.initialize(/*num_wf_slots=*/1);
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto plugin = std::make_unique<MfmaRacePlugin>();
  auto *race_plugin = plugin.get();
  f.plugin_group_->add(std::move(plugin));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();
  auto *wf = f.cu()->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  EXPECT_NE(wf, nullptr);
  if (!wf)
    return false;
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0, 256, 104, waves);
  race_plugin->registerOutstandingLoad(reg, /*exec_mask=*/uint64_t{1} << lane);
  util::set_force_scalar_for_testing(force_scalar);
  run(*f.cu(), wf->vgpr_alloc().base, acc);
  util::set_force_scalar_for_testing(false);
  return race_plugin->raced_on(reg);
}

} // namespace mfma_race

// Each operand range is observed through its last register and lane, so a pending load into
// A, B or a VGPR accumulator, or under the destination, is a race on both MFMA paths.
TEST(ExecutionPluginTest, MfmaSnapshotsReportRacesOnEveryOperandRegister) {
  using namespace mfma_race;
  struct ForceScalarGuard {
    bool old = util::force_scalar();
    ~ForceScalarGuard() { util::set_force_scalar_for_testing(old); }
  } guard;
  struct Case {
    const char *name;
    uint32_t a_regs, b_regs, d_regs;
    mfma_race::Run run;
  };
  const Case cases[] = {
      {"f64 16x16x4", 2, 2, 8,
       [](ComputeUnitCore &cu, uint32_t vb, uint32_t acc) {
         exec_f64(cu, 16, 16, 4, 1, vb + D, vb + A, vb + B, vb + C, acc);
       }},
      {"i8 16x16x64", 4, 4, 4,
       [](ComputeUnitCore &cu, uint32_t vb, uint32_t acc) {
         exec_i32_i8(cu, 16, 16, 64, 1, vb + D, vb + A, vb + B, vb + C, acc);
       }},
      {"fp8 x fp4 16x16x128", 8, 4, 4,
       [](ComputeUnitCore &cu, uint32_t vb, uint32_t acc) {
         dispatch_matrix_fmt_pair(0, 4, [&](uint32_t a_bits, uint32_t b_bits, auto ea, auto eb) {
           exec_f32_mixed<16, 16, 128, 1>(cu, a_bits, b_bits, vb + D, vb + A, vb + B, vb + C, ea,
                                          eb, acc);
         });
       }},
      {"f32 4x4x1x16", 1, 1, 4,
       [](ComputeUnitCore &cu, uint32_t vb, uint32_t acc) {
         exec_f32<4, 4, 1, 16>(cu, 32, vb + D, vb + A, vb + B, vb + C, extract_f32, extract_f32,
                               acc);
       }},
  };
  for (const Case &test : cases)
    for (bool force_scalar : {false, true}) {
      SCOPED_TRACE(testing::Message() << test.name << " force_scalar=" << force_scalar);
      const auto races = [&](uint32_t reg, uint32_t acc = ACC_FROM_VGPR) {
        return races_on(test.run, reg, /*lane=*/63, acc, force_scalar);
      };
      EXPECT_TRUE(races(A + test.a_regs - 1)) << "A";
      EXPECT_TRUE(races(B + test.b_regs - 1)) << "B";
      EXPECT_TRUE(races(C + test.d_regs - 1)) << "accumulator";
      EXPECT_FALSE(races(C + test.d_regs - 1, /*acc=*/0u)) << "constant accumulator";
      EXPECT_TRUE(races(D + test.d_regs - 1)) << "destination";
      EXPECT_FALSE(races(A + test.a_regs)) << "register after A";
    }
}

// cbsz/abid and blgp broadcast part of a register; only the lanes they select are read.
TEST(ExecutionPluginTest, MfmaSnapshotsReportRacesOnlyOnBroadcastLanes) {
  using namespace mfma_race;
  struct ForceScalarGuard {
    bool old = util::force_scalar();
    ~ForceScalarGuard() { util::set_force_scalar_for_testing(old); }
  } guard;
  // cbsz=2 broadcasts A's 16-lane block abid=1 (lanes 16-31); blgp=4 broadcasts B lanes 0-15.
  const mfma_race::Run run = [](ComputeUnitCore &cu, uint32_t vb, uint32_t acc) {
    exec_i32_i8(cu, 4, 4, 4, 16, vb + D, vb + A, vb + B, vb + C, acc, /*cbsz=*/2, /*abid=*/1,
                /*blgp=*/4);
  };
  for (bool force_scalar : {false, true}) {
    SCOPED_TRACE(testing::Message() << "force_scalar=" << force_scalar);
    EXPECT_TRUE(races_on(run, A, /*lane=*/16, ACC_FROM_VGPR, force_scalar));
    EXPECT_FALSE(races_on(run, A, /*lane=*/0, ACC_FROM_VGPR, force_scalar));
    EXPECT_TRUE(races_on(run, B, /*lane=*/15, ACC_FROM_VGPR, force_scalar));
    EXPECT_FALSE(races_on(run, B, /*lane=*/63, ACC_FROM_VGPR, force_scalar));
  }
}

// -- Ordering tests ----------------------------------------------------------
//
// These tests use functional mode (the PluginFixture default). Tests that
// assert strictly sequential dispatch execution use num_wf_slots=1 so that
// only one wavefront can be active at a time, forcing the CP to complete each
// dispatch before starting the next.

// The immediate-halt branch frees a wave's registers the instant s_endpgm
// executes, so instruction hooks must not read the slot afterward. Concretely:
// the terminator must fire BEFORE_INSTRUCTION (it is fetched and decoded) but NOT
// AFTER_INSTRUCTION (there is no live slot to observe once it halts+frees), while
// every non-terminator retains a matched BEFORE/AFTER pair. This pins the guard
// that prevents hooks/logging from touching a freed register slot.

// -- Race trace tests --------------------------------------------------------

TEST(RaceDetectorPluginOutputTest, SwapPcDisassemblesAtTheCapturedInstructionAddress) {
  for (bool waited : {false, true}) {
    SCOPED_TRACE(waited);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1));
    PluginSinkConfig config;
    auto &sink = config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    const uint32_t target = waited ? 0x106c : 0x1068;
    const auto mov = [](uint8_t dst, uint8_t src) {
      return cdna4::build_sop1(cdna4::kSMovB32Sop1, {.ssrc0 = src, .sdst = dst})[0];
    };
    const auto load = cdna4::build_smem(cdna4::kSLoadDwordSmem, {.sdata = 8, .imm = 1});
    const auto swap = cdna4::build_sop1(cdna4::kSSwapPcB64Sop1, {.ssrc0 = 2, .sdst = 8});
    std::vector<uint32_t> code{mov(0, 255), 0x3000,      mov(1, 128), mov(2, 255),
                               target,      mov(3, 128), load[0],     load[1]};
    if (waited)
      code.push_back(legacy_wait_word(ROCJITSU_CODE_ARCH_CDNA4, 63, 0));
    code.insert(code.end(), {swap[0], S_NOP, S_ENDPGM});
    const uint32_t value = 7;
    f.mem->load_image(reinterpret_cast<const uint8_t *>(&value), sizeof(value), 0x3000);
    ASSERT_NO_THROW(f.run_kernel(code.data(), code.size()));
    if (waited) {
      EXPECT_EQ(sink.str().find("RACE "), std::string::npos) << sink.str();
    } else {
      EXPECT_NE(sink.str().find("  ==>  0x1060  s_swappc_b64 s[8:9], s[2:3]"), std::string::npos)
          << sink.str();
    }
  }
}

TEST(RaceDetectorPluginOutputTest, ExecutedInstructionsHaveExactPcsIncludingExpiredTrace) {
  for (unsigned padding : {1u, 270u}) {
    SCOPED_TRACE(padding);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1));
    PluginSinkConfig sink_config;
    auto &sink = sink_config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    const auto load = cdna4::build_ds(cdna4::kDsReadB32Ds, {.addr = 0, .vdst = 8});
    const auto read = cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 264, .vdst = 9});
    std::vector<uint32_t> code{S_NOP, load[0], load[1]};
    code.insert(code.end(), padding, S_NOP);
    code.push_back(read[0]);
    code.push_back(S_ENDPGM);
    f.run_kernel(code.data(), code.size());

    const auto &report = sink.str();
    EXPECT_NE(report.find("  ==>  0x1044  ds_read_b32 v8, v0"), std::string::npos) << report;
    EXPECT_NE(report.find(std::format("  ==>  0x{:x}  v_mov_b32_e32 v9, v8", 0x104c + 4 * padding)),
              std::string::npos)
        << report;
    EXPECT_EQ(report.find("before trace window") != std::string::npos, padding > 256);
    EXPECT_EQ(report.find("0x1040"), std::string::npos);
    EXPECT_NE(report.find("; <-- wave 0 lane 0"), std::string::npos);
  }
}

class RewriteRaceCode : public ExecutionPlugin {
public:
  RewriteRaceCode(GpuMemory &memory, uint64_t rewrite_pc)
      : ExecutionPlugin("rewrite_race_code"), memory_(memory), rewrite_pc_(rewrite_pc) {}
  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &, Wavefront &) override {
    if (pc == rewrite_pc_) {
      memory_.write32(0x1044, S_NOP | 7);
      memory_.write32(0x1048, S_NOP | 9);
    }
  }

private:
  GpuMemory &memory_;
  uint64_t rewrite_pc_;
};

TEST(RaceDetectorPluginOutputTest, PreservesExecutedEncodingAfterCodeRewrite) {
  for (unsigned padding : {1u, 270u}) {
    const uint64_t consumer_pc = 0x104c + 4 * padding;
    for (uint64_t rewrite_pc : {uint64_t{0x1040}, consumer_pc}) {
      SCOPED_TRACE(std::format("padding={} rewrite_pc={:x}", padding, rewrite_pc));
      PluginFixture f;
      ASSERT_NO_FATAL_FAILURE(f.initialize(1));
      PluginSinkConfig config;
      auto &sink = config.emplace<StringSink>();
      f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RewriteRaceCode>(*f.mem, rewrite_pc)));
      f.soc->set_plugin_group(f.plugin_group_);
      f.plugin_group_->onInit();
      const auto load = cdna4::build_ds(cdna4::kDsReadB32Ds, {.addr = 0, .vdst = 8});
      const auto read = cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 264, .vdst = 9});
      std::vector<uint32_t> code{S_NOP, load[0], load[1]};
      code.insert(code.end(), padding, S_NOP);
      code.push_back(read[0]);
      code.push_back(S_ENDPGM);
      f.run_kernel(code.data(), code.size());
      EXPECT_NE(sink.str().find("0x1044  ds_read_b32 v8, v0"), std::string::npos) << sink.str();
      EXPECT_NE(sink.str().find(std::format("0x{:x}  v_mov_b32_e32 v9, v8", consumer_pc)),
                std::string::npos)
          << sink.str();
      EXPECT_EQ(sink.str().find("0x1044  s_nop 7"), std::string::npos);
      EXPECT_EQ(sink.str().find("before trace window") != std::string::npos, padding > 256);
    }
  }
}

TEST(RaceDetectorPluginOutputTest, KeepsProducerBytesAfterWaitAndTraceOverwrite) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(2));
  PluginSinkConfig config;
  auto &sink = config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
  auto race = std::make_unique<RaceDetectorPlugin>();
  auto *plugin = race.get();
  ASSERT_TRUE(f.plugin_group_->add(std::move(race)));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();
  auto *producer = f.cu()->dispatch_wf(0, 0x1000, 104, 256);
  auto *consumer = f.cu()->dispatch_wf(1, 0x3000, 104, 256);
  ASSERT_NE(producer, nullptr);
  ASSERT_NE(consumer, nullptr);
  std::array<Wavefront *, 2> waves{producer, consumer};
  constexpr uint32_t lds_size = 1280;
  const uint32_t lds_base = f.cu()->allocate_lds(lds_size);
  for (auto *wave : waves) {
    wave->set_lds_base(lds_base);
    wave->set_lds_size(lds_size);
  }
  f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, 104, waves);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  const auto store_words = cdna4::build_ds(cdna4::kDsWriteB32Ds, {.addr = 0, .data0 = 1});
  auto store = decoder->decode_window(store_words);
  ASSERT_TRUE(store.succeeded());
  const auto expected_store = store.value()->disassemble();
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(producer->pc, *store.value(), *producer);
  ASSERT_TRUE(f.cu()->execute_instruction(store.value().get(), *producer).succeeded());
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, *store.value(), *producer);
  store.value().reset();
  auto *state = static_cast<RaceWavefrontState *>(producer->plugin_state(plugin->slot_index()));
  ASSERT_EQ(state->race_state->getWaveMemoryEvents().size(), 1u);
  const auto event = state->race_state->getWaveMemoryEvents().front();
  ASSERT_TRUE(state->race_state->getDetector()->events().ldsIntervals(event).contains(lds_base));
  execute_wait(*f.plugin_group_, *producer, legacy_wait_word(ROCJITSU_CODE_ARCH_CDNA4, 63, 0));
  EXPECT_FALSE(state->race_state->getDetector()->events().isTrimmable(event));
  const std::array<uint32_t, 1> nop_words{S_NOP};
  auto nop = decoder->decode_window(nop_words);
  ASSERT_TRUE(nop.succeeded());
  for (unsigned i = 0; i < 270; ++i) {
    producer->pc = 0x2000 + 4 * i;
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(producer->pc, *nop.value(), *producer);
  }
  const auto read_words = cdna4::build_ds(cdna4::kDsReadB32Ds, {.addr = 0, .vdst = 8});
  auto read = decoder->decode_window(read_words);
  ASSERT_TRUE(read.succeeded());
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(consumer->pc, *read.value(), *consumer);
  ASSERT_TRUE(f.cu()->execute_instruction(read.value().get(), *consumer).succeeded());
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, *read.value(), *consumer);
  EXPECT_NE(sink.str().find("0x1000  " + expected_store), std::string::npos) << sink.str();
  EXPECT_NE(sink.str().find("0x3000  ds_read_b32 v8, v0"), std::string::npos) << sink.str();
  EXPECT_NE(sink.str().find("; <-- wave 0"), std::string::npos);
  EXPECT_NE(sink.str().find("; <-- wave 1 lane 0"), std::string::npos);
  EXPECT_NE(sink.str().find("(before trace window)  ; <-- wave 0\n"
                            "       ... 0 instructions not recorded ...\n"),
            std::string::npos);
  state->race_state->flushBarrierPendingEvents();
  EXPECT_TRUE(state->race_state->getDetector()->events().isTrimmable(event));
}

TEST(FindConflictTest, UsesRecordedConflictingEvent) {
  RaceDetector detector(/*nWaves=*/
                        1, /*vgprCount=*/4, /*sgprCount=*/4, Dim3d(0), [](RaceViolation) {},
                        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4),
                        /*ldsSizeBytes=*/64 * 1024);
  const std::array obligations{
      MemoryCounterObligation{WaitCounterType::VMCNT, MemoryOrderClass::VMEM}};
  EventId first = detector.allocateEventId(
      WaveId{0}, testInstruction(/*pc=*/0x100), MemoryEventType::GLOBAL_TO_VGPR, {2},
      /*execMask=*/1, 0xf, {}, obligations, MemoryOrderClass::VMEM);
  EventId second = detector.allocateEventId(
      WaveId{0}, testInstruction(/*pc=*/0x200), MemoryEventType::GLOBAL_TO_VGPR, {2},
      /*execMask=*/1, 0xf, {}, obligations, MemoryOrderClass::VMEM);
  ASSERT_NE(first, second);

  RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, true, Dim3d(0), second};
  MarkedInstruction conflict = findConflict(violation, detector);

  EXPECT_EQ(conflict.instruction, testInstruction(0x200));
}

TEST(FindConflictTest, RejectsUnavailableConflictingEvent) {
  RaceDetector detector(/*nWaves=*/
                        1, /*vgprCount=*/4, /*sgprCount=*/4, Dim3d(0), [](RaceViolation) {},
                        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4),
                        /*ldsSizeBytes=*/64 * 1024);
  RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, true, Dim3d(0), EventId{}};

  EXPECT_THROW(findConflict(violation, detector), std::out_of_range);
}

auto make_trace(std::initializer_list<uint64_t> pcs) {
  RingBuffer<InstructionSnapshot, 256> rb;
  for (auto pc : pcs)
    rb.push(testInstruction(pc));
  return rb;
}

TEST(RingBufferTest, KeepsInsertionOrderBeforeAndAfterWrapping) {
  RingBuffer<int, 4> buffer;
  for (int value : {1, 2, 3})
    buffer.push(value);
  ASSERT_EQ(buffer.size(), 3u);
  EXPECT_EQ(buffer[0], 1);
  EXPECT_EQ(buffer[2], 3);
  for (int value : {4, 5, 6})
    buffer.push(value);
  ASSERT_EQ(buffer.size(), 4u);
  for (size_t i = 0; i < buffer.size(); ++i)
    EXPECT_EQ(buffer[i], static_cast<int>(i) + 3);
}

TEST(FormatTraceTest, WaveLaneAnnotations) {
  auto trace = make_trace({0x100, 0x108, 0x10c});
  std::unordered_map<uint64_t, std::string> disasm = {
      {0x100, "ds_write_b32 v9, v12"},
      {0x108, "s_nop 0"},
      {0x10c, "ds_read_b32 v8, v9"},
  };
  MarkedInstruction conflict{testInstruction(0x100), 3, -1};
  MarkedInstruction read{trace[2], 0, 5};
  auto result = formatTrace(
      trace, [&](const InstructionSnapshot &instruction) { return disasm.at(instruction.pc()); },
      conflict, read);
  EXPECT_NE(result.find("; <-- wave 3"), std::string::npos);
  EXPECT_NE(result.find("; <-- wave 0 lane 5"), std::string::npos);
  EXPECT_EQ(result, "  ==>  0x100  ds_write_b32 v9, v12  ; <-- wave 3\n"
                    "       0x108  s_nop 0\n"
                    "  ==>  0x10c  ds_read_b32 v8, v9  ; <-- wave 0 lane 5\n");
  EXPECT_NE(result.find("0x108  s_nop 0"), std::string::npos);
}

TEST(FormatTraceTest, NoLineBeforeFirstMarker) {
  auto trace = make_trace({0x100, 0x104, 0x108, 0x10c, 0x110});
  std::unordered_map<uint64_t, std::string> disasm = {
      {0x100, "s_nop 0"},
      {0x104, "s_nop 0"},
      {0x108, "ds_write_b32 v9, v12"},
      {0x10c, "s_nop 0"},
      {0x110, "ds_read_b32 v8, v9"},
  };
  MarkedInstruction conflict{trace[2], 1, -1};
  MarkedInstruction read{trace[4], 1, 3};
  auto result = formatTrace(
      trace, [&](const InstructionSnapshot &instruction) { return disasm.at(instruction.pc()); },
      conflict, read);
  EXPECT_EQ(result.substr(0, 5), "  ==>");
  EXPECT_EQ(result.find("0x100"), std::string::npos);
  EXPECT_EQ(result.find("0x104"), std::string::npos);
}

TEST(FormatTraceTest, ConflictBeforeTraceWindow) {
  auto trace = make_trace({0x200, 0x204, 0x208});
  std::unordered_map<uint64_t, std::string> disasm = {
      {0x100, "buffer_load_dwordx4 v[148:151], v0, s[8:11], 0"},
      {0x200, "s_nop 0"},
      {0x204, "s_nop 0"},
      {0x208, "ds_read_b32 v8, v9"},
  };
  MarkedInstruction conflict{testInstruction(0x100), 0, -1};
  MarkedInstruction read{trace[2], 0, 5};
  auto result = formatTrace(
      trace, [&](const InstructionSnapshot &instruction) { return disasm.at(instruction.pc()); },
      conflict, read);
  EXPECT_NE(result.find("before trace window"), std::string::npos);
  EXPECT_NE(result.find("buffer_load_dwordx4"), std::string::npos);
  EXPECT_NE(result.find("; <-- wave 0"), std::string::npos);
  EXPECT_EQ(result.find("0x200  s_nop 0"), std::string::npos);
  EXPECT_NE(result.find("not recorded"), std::string::npos);
  EXPECT_NE(result.find("; <-- wave 0 lane 5"), std::string::npos);
}

TEST(FormatTraceTest, ResolvesOnlyPrintedInstructions) {
  RingBuffer<InstructionSnapshot, 256> trace;
  for (uint64_t pc = 0x100; pc <= 0x300; pc += 4)
    trace.push(testInstruction(pc));
  std::vector<uint64_t> requested;
  auto resolve = [&](const InstructionSnapshot &instruction) {
    requested.push_back(instruction.pc());
    return "s_nop 0";
  };
  const auto report =
      formatTrace(trace, resolve, MarkedInstruction{trace[2], 0, -1}, {trace[128], 0, 0});
  ASSERT_EQ(requested.size(), 32u);
  EXPECT_EQ(requested.front(), 0x108u);
  EXPECT_EQ(requested.back(), 0x300u);
  EXPECT_NE(report.find("instructions elided"), std::string::npos);
}

TEST(FormatTraceTest, IdenticalEncodingsKeepPcBasedSelection) {
  auto trace = make_trace({0x100, 0x104, 0x108, 0x100, 0x104});
  auto resolve = [](const InstructionSnapshot &) { return "s_nop 0"; };
  const auto report =
      formatTrace(trace, resolve, MarkedInstruction{trace[3], 0, -1}, {trace[4], 0, 0});
  EXPECT_EQ(report, "  ==>  0x100  s_nop 0  ; <-- wave 0\n"
                    "  ==>  0x104  s_nop 0  ; <-- wave 0 lane 0\n");
}

TEST(FormatTraceTest, RepeatedPcKeepsEachCapturedEncoding) {
  const auto words = cdna4::build_ds(cdna4::kDsReadB32Ds, {.addr = 0, .vdst = 8});
  const InstructionSnapshot load{0x100, std::span(words.data(), 2)};
  const auto rewritten = testInstruction(0x100);
  RingBuffer<InstructionSnapshot, 256> trace;
  for (const auto &instruction : {load, testInstruction(0x104), rewritten, testInstruction(0x108)})
    trace.push(instruction);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  const auto report =
      formatTrace(trace,
                  [&](const InstructionSnapshot &instruction) {
                    auto decoded = decoder->decode_window(instruction.encoding(), instruction.pc());
                    EXPECT_TRUE(decoded.succeeded());
                    return decoded.succeeded() ? decoded.value()->disassemble() : std::string{};
                  },
                  MarkedInstruction{load, 0, -1}, {trace[3], 0, 0});
  EXPECT_EQ(report, "  ==>  0x100  ds_read_b32 v8, v0  ; <-- wave 0\n"
                    "       0x104  s_nop 0\n"
                    "       0x100  s_nop 0\n"
                    "  ==>  0x108  s_nop 0  ; <-- wave 0 lane 0\n");
}

TEST(FormatTraceTest, RewrittenReadPcDoesNotTrimAtAnEarlierEncoding) {
  const auto producer = testInstruction(0x100);
  const auto read = testInstruction(0x100, S_NOP | 7);
  RingBuffer<InstructionSnapshot, 256> trace;
  for (const auto &instruction : {producer, testInstruction(0x104), read})
    trace.push(instruction);
  const auto report =
      formatTrace(trace,
                  [](const InstructionSnapshot &instruction) {
                    return std::format("s_nop {}", instruction.encoding()[0] & 0xffff);
                  },
                  MarkedInstruction{producer, 0, -1}, {read, 0, 0});
  EXPECT_EQ(report, "  ==>  0x100  s_nop 0  ; <-- wave 0\n"
                    "       0x104  s_nop 0\n"
                    "  ==>  0x100  s_nop 7  ; <-- wave 0 lane 0\n");
}

TEST(FormatTraceTest, DifferentEncodingAtConflictPcIsNotTheProducer) {
  const auto producer = testInstruction(0x100, S_NOP | 7);
  RingBuffer<InstructionSnapshot, 256> trace;
  trace.push(testInstruction(0x100));
  trace.push(testInstruction(0x104));
  const auto report =
      formatTrace(trace,
                  [](const InstructionSnapshot &instruction) {
                    return std::format("s_nop {}", instruction.encoding()[0] & 0xffff);
                  },
                  MarkedInstruction{producer, 1, -1}, {trace[1], 0, 0});
  EXPECT_EQ(report, "  ==>  0x100  s_nop 7  (before trace window)  ; <-- wave 1\n"
                    "       ... 1 instructions not recorded ...\n"
                    "  ==>  0x104  s_nop 0  ; <-- wave 0 lane 0\n");
}

TEST(FormatTraceTest, PropagatesDisassemblyFailures) {
  auto trace = make_trace({0x100, 0x104});
  auto fail = [](const InstructionSnapshot &) -> std::string {
    throw std::logic_error("cannot disassemble");
  };
  EXPECT_THROW(formatTrace(trace, fail, MarkedInstruction{trace[0], 1, -1}, {trace[1], 0, 0}),
               std::logic_error);
}

TEST(RaceDisassemblyTest, SnapshotsOwnShortLiteralAndMaximumWidthEncodings) {
  const struct {
    const char *arch;
    rj_code_arch_t code_arch;
    uint32_t wave_size;
    std::array<uint32_t, 4> words;
  } cases[] = {
      {"cdna4", ROCJITSU_CODE_ARCH_CDNA4, 64, {S_NOP | 7}},
      {"rdna4", ROCJITSU_CODE_ARCH_RDNA4, 32, {0xC8D006FFu, 0x04020080u, 0x4F7FFFFEu}},
      {"cdna5", ROCJITSU_CODE_ARCH_CDNA5, 32, {0xCF455083u, 0x00000086u, 0x0A000001u}},
      {"cdna4", ROCJITSU_CODE_ARCH_CDNA4, 64,
       make_cdna4_mfma_scale_words(cdna4::kVMfmaF3216x16x128F8f6f4Vop3pMfma, 1, 448, 449)},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.arch);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, test.arch, test.wave_size, 128));
    PluginSinkConfig config;
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    auto race = std::make_unique<RaceDetectorPlugin>();
    auto *plugin = race.get();
    ASSERT_TRUE(f.plugin_group_->add(std::move(race)));
    f.soc->set_plugin_group(f.plugin_group_);
    auto *wave = f.cu()->dispatch_wf(0, 0x1000, 128, 256, test.wave_size);
    ASSERT_NE(wave, nullptr);
    std::array<Wavefront *, 1> waves{wave};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, 128, waves);
    std::string expected;
    {
      auto words = test.words;
      auto decoder = Decoder::create(test.code_arch);
      auto instruction = decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      expected = instruction.value()->disassemble();
      f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave->pc, *instruction.value(), *wave);
      words.fill(0);
    }
    auto *state = static_cast<RaceWavefrontState *>(wave->plugin_state(plugin->slot_index()));
    ASSERT_EQ(state->trace.size(), 1u);
    EXPECT_EQ(wave->disassemble_instruction(state->trace[0].encoding(), state->trace[0].pc()),
              expected);
  }
}

TEST(RaceDisassemblyTest, CapturedEncodingNeedsNoCodeMapping) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(1));
  auto *wave = f.cu()->dispatch_wf(0, 0x1000, 104, 256);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(7);
  const std::array<uint32_t, 1> nop{S_NOP | 7};
  EXPECT_EQ(wave->disassemble_instruction(nop, 0x1000), "s_nop 7");
  const auto load = cdna4::build_ds(cdna4::kDsReadB32Ds);
  EXPECT_EQ(wave->disassemble_instruction(load, 0x1000), "ds_read_b32 v0, v0");
}

TEST(RaceDisassemblyTest, RejectsMissingAndTruncatedEncodings) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(1));
  auto *wave = f.cu()->dispatch_wf(0, 0x1000, 104, 256);
  ASSERT_NE(wave, nullptr);
  const auto load = cdna4::build_ds(cdna4::kDsReadB32Ds);
  for (const auto encoding : {std::span<const uint32_t>{}, std::span(load).first(1)}) {
    try {
      wave->disassemble_instruction(encoding, 0x1000);
      FAIL() << "incomplete instruction encoding was accepted";
    } catch (const std::logic_error &error) {
      EXPECT_NE(std::string(error.what()).find("PC 0x1000"), std::string::npos);
      EXPECT_NE(std::string(error.what()).find(encoding.empty() ? "empty" : "truncated"),
                std::string::npos);
    }
  }
}

TEST(RaceDisassemblyTest, ConcurrentCusDecodeCapturedBytes) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(1, "cdna4", 64, 104, 256, 4));
  std::vector<std::thread> workers;
  for (unsigned i = 0; i < 4; ++i) {
    auto *wave = f.cu(i)->dispatch_wf(0, 0x1000, 104, 256);
    ASSERT_NE(wave, nullptr);
    workers.emplace_back([wave] {
      const std::array<uint32_t, 1> first{S_NOP | 7}, second{S_NOP | 9};
      for (unsigned repeat = 0; repeat < 100; ++repeat) {
        EXPECT_EQ(wave->disassemble_instruction(first, 0x540024b100), "s_nop 7");
        EXPECT_EQ(wave->disassemble_instruction(second, 0x100002a100), "s_nop 9");
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
}

TEST(RaceDetectorPluginOutputTest, DispatchLineUsesQuestionMarksForUnresolvedKernel) {
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  ExecutionPluginGroup plugin_group(std::move(sink_config));
  ASSERT_TRUE(plugin_group.add(std::make_unique<plugins::race_detector::RaceDetectorPlugin>()));

  KernelDispatchInfo info{};
  info.dispatch_id = 17;
  plugin_group.onAmdgpuDispatchPacketProcessed(info);

  EXPECT_NE(sink.str().find("[rocjitsu] Kernel dispatch: \"?\" symbol=\"?\"\n"), std::string::npos);
}

TEST(RaceDetectorPluginOutputTest, DispatchLineUsesReadableNameAndExactSymbol) {
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  ExecutionPluginGroup plugin_group(std::move(sink_config));
  ASSERT_TRUE(plugin_group.add(std::make_unique<plugins::race_detector::RaceDetectorPlugin>()));

  KernelDispatchInfo info{};
  info.dispatch_id = 18;
  info.kernel_name = "racy_kernel";
  info.kernel_symbol = "_Z11racy_kernelPKfPf";
  plugin_group.onAmdgpuDispatchPacketProcessed(info);

  EXPECT_NE(sink.str().find("[rocjitsu] Kernel dispatch: \"racy_kernel\" "
                            "symbol=\"_Z11racy_kernelPKfPf\"\n"),
            std::string::npos);
}

TEST(RaceDetectorPluginTest, DroppedAsyncLdsLaneDoesNotCreateLowAddressRace) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(
      f.initialize(/*num_wf_slots=*/2, /*arch=*/"cdna5", /*wavefront_size=*/32));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *writer = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  auto *reader = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(writer, nullptr);
  ASSERT_NE(reader, nullptr);
  writer->set_exec(0x1u);
  reader->set_exec(0x1u);
  std::array<amdgpu::Wavefront *, 2> waves{writer, reader};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0,
                                               /*physical_vgpr_count=*/512, /*sgpr_count=*/208,
                                               waves);

  auto dropped = std::make_unique<VectorMemState>(GLOBAL_MEM);
  dropped->elem_size = 4;
  dropped->num_elems = 1;
  dropped->is_load = true;
  dropped->lds_dst = true;
  dropped->lds_per_lane_addr = true;
  dropped->wf_size = writer->wf_size();
  dropped->exec_mask = 0x1u;
  dropped->lane_mask = 0x1u;
  dropped->per_lane_lds_addr[0] = amdgpu::kInvalidLdsAddress;
  TestMemoryInstruction dropped_inst(std::move(dropped));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(writer->pc, dropped_inst, *writer);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, dropped_inst, *writer);

  auto read = std::make_unique<VectorMemState>(LOCAL_MEM);
  read->elem_size = 4;
  read->num_elems = 1;
  read->is_load = true;
  read->wf_size = reader->wf_size();
  read->exec_mask = 0x1u;
  read->lane_mask = 0x1u;
  read->per_lane_addr[0] = 0;
  read->dst_reg_base = reader->vgpr_alloc().base;
  TestMemoryInstruction read_inst(std::move(read));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(reader->pc, read_inst, *reader);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, read_inst, *reader);

  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

TEST(RaceDetectorPluginTest, RejectedAsyncGlobalLaneDoesNotCreateLdsRace) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(
      f.initialize(/*num_wf_slots=*/2, /*arch=*/"cdna5", /*wavefront_size=*/32));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *writer = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  auto *reader = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(writer, nullptr);
  ASSERT_NE(reader, nullptr);
  writer->set_exec(0x1u);
  reader->set_exec(0x1u);
  std::array<amdgpu::Wavefront *, 2> waves{writer, reader};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0,
                                               /*physical_vgpr_count=*/512, /*sgpr_count=*/208,
                                               waves);

  auto dropped = std::make_unique<VectorMemState>(GLOBAL_MEM);
  dropped->elem_size = 4;
  dropped->num_elems = 1;
  dropped->is_load = true;
  dropped->lds_dst = true;
  dropped->lds_per_lane_addr = true;
  dropped->wf_size = writer->wf_size();
  dropped->exec_mask = 0x1u;
  dropped->lane_mask = 0;
  dropped->per_lane_lds_addr[0] = 0;
  TestMemoryInstruction dropped_inst(std::move(dropped));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(writer->pc, dropped_inst, *writer);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, dropped_inst, *writer);

  auto read = std::make_unique<VectorMemState>(LOCAL_MEM);
  read->elem_size = 4;
  read->num_elems = 1;
  read->is_load = true;
  read->wf_size = reader->wf_size();
  read->exec_mask = 0x1u;
  read->lane_mask = 0x1u;
  read->per_lane_addr[0] = 0;
  read->dst_reg_base = reader->vgpr_alloc().base;
  TestMemoryInstruction read_inst(std::move(read));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(reader->pc, read_inst, *reader);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, read_inst, *reader);

  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

class IgnoredGlobalMemoryRaceTest : public ::testing::TestWithParam<AtomicOp> {};

TEST_P(IgnoredGlobalMemoryRaceTest, DoesNotCreateDestinationRace) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(
      f.initialize(/*num_wf_slots=*/1, /*arch=*/"cdna5", /*wavefront_size=*/32));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *wf = f.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0,
                                               /*physical_vgpr_count=*/256, /*sgpr_count=*/104,
                                               waves);

  constexpr uint32_t kDestinationVgpr = 7;
  auto ignored = std::make_unique<VectorMemState>(GLOBAL_MEM);
  ignored->elem_size = 4;
  ignored->num_elems = 1;
  ignored->is_load = true;
  ignored->atomic_op = GetParam();
  ignored->exec_mask = 0;
  ignored->lane_mask = 0;
  ignored->wf_size = wf->wf_size();
  ignored->dst_reg_base = wf->vgpr_alloc().base + kDestinationVgpr;
  TestMemoryInstruction ignored_inst(std::move(ignored));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, ignored_inst, *wf);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, ignored_inst, *wf);

  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kDestinationVgpr,
                                         /*lane_mask=*/0x1u, /*byte_mask=*/0xFu);
  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

TEST_P(IgnoredGlobalMemoryRaceTest, DoesNotConsumeLoadcntOrderingSlot) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(
      f.initialize(/*num_wf_slots=*/1, /*arch=*/"cdna5", /*wavefront_size=*/32));
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *wf = f.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0,
                                               /*physical_vgpr_count=*/256, /*sgpr_count=*/104,
                                               waves);

  constexpr uint32_t kDestinationVgpr = 7;
  auto load = std::make_unique<VectorMemState>(GLOBAL_MEM);
  load->elem_size = 4;
  load->num_elems = 1;
  load->is_load = true;
  load->wait_counter_type = WaitCounterType::LOADCNT;
  load->exec_mask = 0x1u;
  load->lane_mask = 0x1u;
  load->wf_size = wf->wf_size();
  load->dst_reg_base = wf->vgpr_alloc().base + kDestinationVgpr;
  TestMemoryInstruction load_inst(std::move(load));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, load_inst, *wf);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load_inst, *wf);

  auto ignored = std::make_unique<VectorMemState>(GLOBAL_MEM);
  ignored->elem_size = 4;
  ignored->num_elems = 1;
  ignored->is_load = false;
  ignored->wait_counter_type = WaitCounterType::LOADCNT;
  ignored->atomic_op = GetParam();
  ignored->exec_mask = 0;
  ignored->lane_mask = 0;
  ignored->wf_size = wf->wf_size();
  TestMemoryInstruction ignored_inst(std::move(ignored));
  f.plugin_group_->onAmdgpuBeforeExecuteInstruction(wf->pc, ignored_inst, *wf);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, ignored_inst, *wf);

  execute_wait(*f.plugin_group_, *wf, 0xbfc00001u); // s_wait_loadcnt 1.

  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kDestinationVgpr,
                                         /*lane_mask=*/0x1u, /*byte_mask=*/0xFu);
  EXPECT_NE(sink.str().find("RACE "), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(LoadAndReturningAtomic, IgnoredGlobalMemoryRaceTest,
                         ::testing::Values(AtomicOp::NONE, AtomicOp::SWAP));

// Two CDNA4 waves share a 108-byte LDS allocation. The producer's M0 selects
// allocation byte 64, and its active lanes 0, 2 and 5 write 16-byte lane slots
// at bytes 64, 96 and 144. X3 writes the first 12 bytes of each slot and X4
// writes all 16. Lane 5's slot is outside the allocation, so it writes nothing.
class GlobalWideLdsRaceHarness {
public:
  void initialize() {
    ASSERT_NO_FATAL_FAILURE(f_.initialize(2));
    PluginSinkConfig config;
    sink_ = &config.emplace<StringSink>();
    f_.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    ASSERT_TRUE(f_.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f_.soc->set_plugin_group(f_.plugin_group_);
    f_.plugin_group_->onInit();
    producer_ = f_.cu()->dispatch_wf(0, 0x1000, 104, 256);
    peer_ = f_.cu()->dispatch_wf(0, 0x2000, 104, 256);
    ASSERT_NE(producer_, nullptr);
    ASSERT_NE(peer_, nullptr);
    waves_ = {producer_, peer_};
    for (auto *wave : waves_) {
      wave->set_exec(1);
      wave->set_lds_base(128);
      wave->set_lds_size(108);
    }
    producer_->set_m0(0xfffc0043);
    for (unsigned lane : {0u, 2u, 5u}) {
      f_.cu()->write_vgpr(producer_->vgpr_alloc().base + 2, lane, 0x20000 + lane * 32);
      f_.cu()->write_vgpr(producer_->vgpr_alloc().base + 3, lane, 0);
    }
    f_.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 512, 208, waves_);
  }

  Wavefront &producer() { return *producer_; }
  Wavefront &peer() { return *peer_; }
  bool raced() const { return sink_->str().find("RACE ") != std::string::npos; }
  const std::string &report() const { return sink_->str(); }

  void wide_load(unsigned count) {
    producer_->set_exec(0x25);
    execute(*producer_, cdna4::build_flat(count == 3 ? cdna4::kGlobalLoadLdsDwordx3Flat
                                                     : cdna4::kGlobalLoadLdsDwordx4Flat,
                                          {.seg = 2, .addr = 2, .saddr = 0x7f, .vdst = 0xff}));
  }

  void read_byte(Wavefront &wave, unsigned address) {
    wave.set_exec(1);
    wave.pc = 0x3000;
    f_.cu()->write_vgpr(wave.vgpr_alloc().base, 0, address);
    execute(wave, cdna4::build_ds(cdna4::kDsReadU8Ds, {.addr = 0, .vdst = 8}));
  }

  void wait(Wavefront &wave, uint8_t vmcnt, uint8_t lgkmcnt) {
    execute_wait(*f_.plugin_group_, wave,
                 legacy_wait_word(ROCJITSU_CODE_ARCH_CDNA4, vmcnt, lgkmcnt));
  }

  void barrier() { f_.plugin_group_->onAmdgpuBarrierResolved(waves_, BarrierKind::Workgroup); }

private:
  void execute(Wavefront &wave, std::array<uint32_t, 2> words) {
    auto decoded = decoder_->decode_window(words);
    ASSERT_TRUE(decoded.succeeded());
    auto &inst = *instructions_.emplace_back(std::move(decoded.value()));
    f_.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave.pc, inst, wave);
    ASSERT_TRUE(f_.cu()->execute_instruction(&inst, wave).succeeded());
    f_.plugin_group_->onAmdgpuMemoryAccessRouted({}, inst, wave);
  }

  PluginFixture f_;
  StringSink *sink_ = nullptr;
  Wavefront *producer_ = nullptr;
  Wavefront *peer_ = nullptr;
  std::array<Wavefront *, 2> waves_{};
  std::unique_ptr<Decoder> decoder_ = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<std::unique_ptr<Instruction>> instructions_;
};

enum class WideLdsWait {
  None,
  VmcntOne,         // Leaves the single outstanding load pending.
  LgkmcntZero,      // Waits on the wrong counter.
  VmcntZero,        // Completes the load for the producer only.
  VmcntZeroBarrier, // Completes the load and orders it for the peer.
};

TEST(RaceDetectorPluginTest, GlobalWideLdsLoadsTrackPayloadAndSynchronization) {
  using enum WideLdsWait;
  for (unsigned count : {3u, 4u})
    for (bool other_wave : {false, true})
      for (WideLdsWait wait : {None, VmcntOne, LgkmcntZero, VmcntZero, VmcntZeroBarrier})
        // Byte 75 is in lane 0's payload for both widths. Byte 79 is lane 0's
        // fourth DWORD, written only by X4. Byte 80 starts inactive lane 1's
        // slot. Lane 2's 96..107 payload fits only for X3, because X4 would
        // cross the 108-byte allocation, so byte 107 is written only by X3.
        // Bytes 108 and 144 are outside the allocation.
        for (unsigned address : {75u, 79u, 80u, 107u, 108u, 144u}) {
          SCOPED_TRACE(testing::Message() << count << ',' << other_wave << ','
                                          << static_cast<int>(wait) << ',' << address);
          GlobalWideLdsRaceHarness h;
          ASSERT_NO_FATAL_FAILURE(h.initialize());
          ASSERT_NO_FATAL_FAILURE(h.wide_load(count));
          EXPECT_FALSE(h.raced()) << h.report();
          if (wait == VmcntOne)
            h.wait(h.producer(), 1, 15);
          else if (wait == LgkmcntZero)
            h.wait(h.producer(), 63, 0);
          else if (wait == VmcntZero || wait == VmcntZeroBarrier)
            h.wait(h.producer(), 0, 15);
          if (wait == VmcntZeroBarrier)
            h.barrier();

          ASSERT_NO_FATAL_FAILURE(h.read_byte(other_wave ? h.peer() : h.producer(), address));
          const bool overlaps =
              address == 75 || (address == 79 && count == 4) || (address == 107 && count == 3);
          const bool synchronized = wait == VmcntZeroBarrier || (!other_wave && wait == VmcntZero);
          EXPECT_EQ(h.raced(), overlaps && !synchronized) << h.report();
        }
}

TEST(RaceDetectorPluginTest, GlobalWideLdsLoadsDetectPendingLdsReads) {
  enum class ReadSync { None, LgkmcntZero, LgkmcntZeroBarrier };
  using enum ReadSync;
  for (unsigned count : {3u, 4u})
    for (bool other_wave : {false, true})
      for (ReadSync sync : {None, LgkmcntZero, LgkmcntZeroBarrier})
        // Byte 75 is in lane 0's payload for both widths; byte 80 is in
        // inactive lane 1's slot.
        for (unsigned address : {75u, 80u}) {
          SCOPED_TRACE(testing::Message() << count << ',' << other_wave << ','
                                          << static_cast<int>(sync) << ',' << address);
          GlobalWideLdsRaceHarness h;
          ASSERT_NO_FATAL_FAILURE(h.initialize());
          Wavefront &reader = other_wave ? h.peer() : h.producer();
          ASSERT_NO_FATAL_FAILURE(h.read_byte(reader, address));
          if (sync != None)
            h.wait(reader, 63, 0);
          if (sync == LgkmcntZeroBarrier)
            h.barrier();
          EXPECT_FALSE(h.raced()) << h.report();
          ASSERT_NO_FATAL_FAILURE(h.wide_load(count));
          // The load's LDS write is not ordered after the producer's own DS
          // read, so that read also needs an lgkmcnt wait.
          const bool synchronized =
              sync == LgkmcntZeroBarrier || (!other_wave && sync == LgkmcntZero);
          EXPECT_EQ(h.raced(), address == 75 && !synchronized) << h.report();
        }
}

TEST(RaceDetectorPluginTest, Gfx11ImageCountersRetireInOrderAcrossWritebackClasses) {
  for (const char *arch : {"rdna3", "rdna3_5"}) {
    SCOPED_TRACE(arch);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, arch, 32, 96, 32));
    PluginSinkConfig config;
    auto &sink = config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(config));
    ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0x8000, 96, 32, 32);
    ASSERT_TRUE(wf);
    wf->set_exec(1);
    std::array waves{wf};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 32, 96, waves);
    const std::array<uint32_t, 12> descriptor{0x100,
                                              (22u << 20) | (3u << 30) | (3u << 16),
                                              15u | (63u << 14),
                                              (9u << 28) | (3u << 16) | 0x204,
                                              0,
                                              384u << 8,
                                              0,
                                              0,
                                              2u | (2u << 3),
                                              0xfffu << 12,
                                              1u << 26,
                                              0};
    for (unsigned i = 0; i < descriptor.size(); ++i)
      cu->write_sgpr(wf->sgpr_alloc().base + i, descriptor[i]);
    f.mem->write32(0x10000, 123);
    const auto issue = [&](const auto &words) {
      const auto pc = wf->pc + 0x100;
      wf->pc = pc;
      f.mem->load_image(reinterpret_cast<const uint8_t *>(words.data()), sizeof(words), pc);
      for (unsigned tick = 0; tick < 32 && wf->pc == pc; ++tick)
        ASSERT_TRUE(cu->step());
      ASSERT_EQ(wf->pc, pc + sizeof(words));
    };
    const auto wait = [&](unsigned count) {
      ASSERT_NO_FATAL_FAILURE(
          issue(std::array{legacy_wait_word(ROCJITSU_CODE_ARCH_RDNA3, count, 63)}));
    };
    const auto image = [](uint16_t op, uint8_t dst) {
      const bool load = op == rdna3_5::kImageLoadMimg;
      return rdna3_5::build_mimg(op, {.dim = 1,
                                      .unorm = load,
                                      .dmask = 1,
                                      .vdata = dst,
                                      .ssamp = uint8_t(op == rdna3_5::kImageSampleLzMimg ? 2 : 0)});
    };
    const auto query = rdna3_5::kImageGetResinfoMimg;
    for (const auto other : {rdna3_5::kImageLoadMimg, rdna3_5::kImageSampleLzMimg}) {
      for (bool query_first : {false, true}) {
        const auto first = query_first ? query : other, second = query_first ? other : query;
        const bool mixed = other == rdna3_5::kImageSampleLzMimg;
        ASSERT_NO_FATAL_FAILURE(issue(image(first, 8)));
        ASSERT_NO_FATAL_FAILURE(issue(image(second, 12)));
        ASSERT_NO_FATAL_FAILURE(wait(1));
        auto start = sink.str().size();
        ASSERT_NO_FATAL_FAILURE(
            issue(rdna3_5::build_vop1(rdna3_5::kVMovB32Vop1, {.src0 = 264, .vdst = 3})));
        EXPECT_EQ(sink.str().find("type=VGPR access=read reg=8", start), std::string::npos);
        ASSERT_NO_FATAL_FAILURE(wait(0));
        ASSERT_NO_FATAL_FAILURE(issue(image(first, 8)));
        start = sink.str().size();
        ASSERT_NO_FATAL_FAILURE(issue(image(second, 8)));
        EXPECT_EQ(sink.str().find("type=VGPR access=write reg=8", start) != std::string::npos,
                  mixed);
        ASSERT_NO_FATAL_FAILURE(wait(0));
      }
    }
    for (const bool warning : {false, true}) {
      for (const bool waited : {false, true}) {
        wf->debug_write_vgpr(2, 0, std::bit_cast<uint32_t>(warning ? 0.0f : 2.0f));
        ASSERT_NO_FATAL_FAILURE(issue(image(rdna3_5::kImageLoadMimg, 10)));
        const auto before = wf->debug_read_vgpr(10, 0);
        ASSERT_NE(before, 2u);
        if (waited) {
          ASSERT_NO_FATAL_FAILURE(wait(0));
        }
        const auto start = sink.str().size();
        ASSERT_NO_FATAL_FAILURE(issue(rdna3_5::build_mimg(
            rdna3_5::kImageSampleLMimg,
            {.dim = 1, .dmask = 7, .d16 = 1, .vdata = 8, .lwe = 1, .ssamp = 2})));
        EXPECT_EQ(wf->debug_read_vgpr(10, 0), warning ? 2u : before);
        EXPECT_EQ(sink.str().find("type=VGPR access=write reg=10", start) != std::string::npos,
                  warning && !waited);
        ASSERT_NO_FATAL_FAILURE(wait(0));
      }
    }
    wf->halt();
  }
}

TEST(RaceDetectorPluginTest, Cdna1LdsDirectSourceChecksPendingWritesWithoutCreatingLoad) {
  enum Order {
    WriterFirst,
    ReaderFirst,
    ReaderWaitWriter,
    ReaderBarrierWriter,
    WriterWaitBarrierReader
  };
  struct Case {
    Order order;
    bool returning_atomic;
    uint32_t writer_offset;
    uint64_t exec;
  };
  for (const auto &[order, returning_atomic, writer_offset, exec] :
       {Case{WriterFirst, false, 16, 0xff}, Case{WriterFirst, true, 16, 0xff},
        Case{ReaderFirst, false, 16, 0xff}, Case{ReaderFirst, true, 16, 0xff},
        Case{ReaderWaitWriter, false, 16, 0xff}, Case{ReaderBarrierWriter, false, 16, 0xff},
        Case{ReaderBarrierWriter, true, 16, 0xff}, Case{WriterWaitBarrierReader, true, 16, 0xff},
        Case{WriterFirst, false, 20, 0xff}, Case{ReaderFirst, true, 20, 0xff},
        Case{WriterFirst, false, 16, 0}, Case{ReaderFirst, false, 16, 0}}) {
    SCOPED_TRACE(testing::Message()
                 << order << ',' << returning_atomic << ',' << writer_offset << ',' << exec);
    PluginFixture fixture;
    ASSERT_NO_FATAL_FAILURE(fixture.initialize(2, "cdna1", 64));
    PluginSinkConfig sinks;
    auto &output = sinks.emplace<StringSink>();
    fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *race = plugin.get();
    ASSERT_TRUE(fixture.plugin_group_->add(std::move(plugin)));
    fixture.soc->set_plugin_group(fixture.plugin_group_);
    fixture.plugin_group_->onInit();
    auto &cu = *fixture.cu();
    auto *writer = cu.dispatch_wf(0, 0, 104, 256, 64);
    auto *reader = cu.dispatch_wf(0, 0x1000, 104, 256, 64);
    ASSERT_NE(writer, nullptr);
    ASSERT_NE(reader, nullptr);
    std::array<Wavefront *, 2> waves{writer, reader};
    fixture.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 512, 208, waves);
    auto *reader_state =
        static_cast<RaceWavefrontState *>(reader->plugin_state(race->slot_index()));
    ASSERT_NE(reader_state, nullptr);
    reader->set_lds_size(64);
    reader->set_exec(exec);
    reader->set_m0((2u << 16) | 16);
    reader->lds().write32(16, 0x12345678);

    auto decoder = Decoder::create(cu.arch());
    auto write = [&] {
      writer->set_lds_size(64);
      writer->set_exec(1);
      cu.write_vgpr(writer->vgpr_alloc().base, 0, writer_offset);
      cu.write_vgpr(writer->vgpr_alloc().base + 1, 0, 1);
      // ds_write_b32 v0, v1 or ds_add_rtn_u32 v2, v0, v1.
      const std::array<uint32_t, 2> words = returning_atomic ? std::array{0xd8400000u, 0x02000100u}
                                                             : std::array{0xd81a0000u, 0x00000100u};
      auto inst = std::unique_ptr<Instruction>(decode_valid(*decoder, words.data()));
      ASSERT_NE(inst, nullptr);
      fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(writer->pc, *inst, *writer);
      ASSERT_TRUE(cu.execute_instruction(inst.get(), *writer).succeeded());
      ASSERT_NE(inst->data(), nullptr);
      test::ComputeUnitTestAccess::route_memory_inst(cu, inst.release(), *writer);
    };
    auto read = [&] {
      const uint32_t expected = reader->lds().read32(16);
      const uint32_t word = 0x7e0202fe; // gfx908 v_mov_b32 v1, src_lds_direct
      auto mov = std::unique_ptr<Instruction>(decode_valid(*decoder, &word));
      ASSERT_NE(mov, nullptr);
      fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(reader->pc, *mov, *reader);
      ASSERT_TRUE(cu.execute_instruction(mov.get(), *reader).succeeded());
      fixture.plugin_group_->onAmdgpuAfterExecuteInstruction(reader->pc, *mov, *reader);
      auto &history = *reader_state->race_state;
      EXPECT_TRUE(history.getWaveMemoryEvents().empty());
      const auto &completed = history.getBarrierPendingEvents();
      ASSERT_EQ(completed.size(), exec ? 1u : 0u);
      if (exec) {
        EXPECT_EQ(cu.read_vgpr(reader->vgpr_alloc().base + 1, 0), expected);
        const auto &registry = history.getDetector()->events();
        EXPECT_EQ(registry.status(completed.front()), EventStatus::WAVE_COMPLETE);
        EXPECT_TRUE(registry.registers(completed.front()).empty());
        EXPECT_TRUE(registry.allWaitCountersSatisfied(completed.front()));
      }
    };
    auto wait = [&](Wavefront &wf) {
      execute_wait(*fixture.plugin_group_, wf, legacy_wait_word(cu.arch(), 63, 0));
    };
    if (order == WriterFirst || order == WriterWaitBarrierReader) {
      ASSERT_NO_FATAL_FAILURE(write());
      if (order == WriterWaitBarrierReader) {
        wait(*writer);
        fixture.plugin_group_->onAmdgpuBarrierResolved(waves, BarrierKind::Workgroup);
      }
      ASSERT_NO_FATAL_FAILURE(read());
    } else {
      ASSERT_NO_FATAL_FAILURE(read());
      if (order == ReaderWaitWriter)
        wait(*reader);
      if (order == ReaderBarrierWriter)
        fixture.plugin_group_->onAmdgpuBarrierResolved(waves, BarrierKind::Workgroup);
      ASSERT_NO_FATAL_FAILURE(write());
    }
    const bool barrier = order == ReaderBarrierWriter || order == WriterWaitBarrierReader;
    EXPECT_EQ(output.str().find("RACE ") != std::string::npos,
              writer_offset == 16 && exec && !barrier);
  }
}

TEST(RaceDetectorPluginTest, DecodedDirectLdsLoadsDetectConflictingWaveStores) {
  for (auto arch : {"rdna3", "rdna3_5", "rdna4"}) {
    for (uint32_t store_address : {16u, 20u}) {
      SCOPED_TRACE(testing::Message() << arch << ',' << store_address);
      PluginFixture fixture;
      ASSERT_NO_FATAL_FAILURE(fixture.initialize(2, arch, 32));
      PluginSinkConfig sink_config;
      auto &sink = sink_config.emplace<StringSink>();
      fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
      ASSERT_TRUE(fixture.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
      fixture.soc->set_plugin_group(fixture.plugin_group_);
      fixture.plugin_group_->onInit();
      auto *cu = fixture.cu();
      auto *writer = cu->dispatch_wf(0, 0, 104, 256, 32);
      auto *reader = cu->dispatch_wf(0, 0x1000, 104, 256, 32);
      ASSERT_NE(writer, nullptr);
      ASSERT_NE(reader, nullptr);
      writer->set_lds_size(1024);
      reader->set_lds_size(1024);
      writer->set_exec(1);
      reader->set_exec(1);
      reader->set_lds_size(1024);
      writer->set_lds_size(1024);
      reader->set_m0((2u << 16) | 16);
      std::array<amdgpu::Wavefront *, 2> waves{writer, reader};
      fixture.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 512, 208, waves);
      auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
      state->wf_size = 32;
      state->elem_size = 4;
      state->num_elems = 1;
      state->is_load = false;
      state->exec_mask = state->lane_mask = 1;
      state->per_lane_addr[0] = store_address;
      TestMemoryInstruction store(std::move(state));
      fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(writer->pc, store, *writer);
      fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, store, *writer);
      const uint32_t word = 0xce100006;
      auto decoder = Decoder::create(cu->arch());
      std::unique_ptr<Instruction> load(decode_valid(*decoder, &word));
      ASSERT_NE(load, nullptr);
      ASSERT_TRUE(load->is_memory_op());
      fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(reader->pc, *load, *reader);
      ASSERT_TRUE(cu->execute_instruction(load.get(), *reader).succeeded());
      ASSERT_NE(load->data(), nullptr);
      test::ComputeUnitTestAccess::route_memory_inst(*cu, load.release(), *reader);
      EXPECT_EQ(sink.str().find("RACE ") != std::string::npos, store_address == 16);
    }
  }
}

} // namespace
