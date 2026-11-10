// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin_race_test.cpp
/// @brief Execution plugin race detection and diagnostic output tests.

#include "execution_plugin_test_util.h"

#include "mma_test_util.h"
#include "race-detector/race_test_instruction.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/plugin.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand_types.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
using namespace rocjitsu::plugins::race_detector;
using namespace rocjitsu::test::execution_plugin;
using mma_test::make_cdna4_mfma_scale_words;

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
    wave->clear_instruction_execution_error();
    EXPECT_NO_THROW(
        fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(0x1000, instruction, *wave));
    EXPECT_EQ(wave->instruction_execution_error(), InstructionExecutionError::PluginFailure);
    EXPECT_EQ(plugin_state->trace.size(), 0u);
    wave->clear_instruction_execution_error();
    EXPECT_NO_THROW(fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, instruction, *wave));
    EXPECT_EQ(wave->instruction_execution_error(), InstructionExecutionError::PluginFailure);
    EXPECT_TRUE(plugin_state->race_state->getWaveMemoryEvents().empty());
  }
}

TEST_F(FormattedLoadRaceTest, RejectsMissingInstructionHistoryWhenReporting) {
  issue(rdna4::kBufferLoadFormatXVbuffer, 46, 8, 1);
  ASSERT_EQ(plugin_state->race_state->getWaveMemoryEvents().size(), 1u);
  plugin_state->trace = {};
  EXPECT_NO_THROW(
      fixture.plugin_group_->onAmdgpuReadVgprLanes(wave, wave->vgpr_alloc().base + 8, 1));
  EXPECT_EQ(wave->instruction_execution_error(), InstructionExecutionError::PluginFailure);
  EXPECT_EQ(sink->str().find("RACE "), std::string::npos);
}

TEST_F(FormattedLoadRaceTest, RejectsUndecodableInstructionHistoryWhenReporting) {
  issue(rdna4::kBufferLoadFormatXVbuffer, 46, 8, 1);
  ASSERT_EQ(plugin_state->race_state->getWaveMemoryEvents().size(), 1u);
  // A captured first word without the second word cannot be disassembled.
  const auto words = rdna4::build_vbuffer(rdna4::kBufferLoadFormatXVbuffer);
  const std::span<const uint32_t> truncated = std::span(words).first(1);
  plugin_state->trace = {};
  plugin_state->trace.push(InstructionSnapshot{0x1234, truncated});
  EXPECT_NO_THROW(
      fixture.plugin_group_->onAmdgpuReadVgprLanes(wave, wave->vgpr_alloc().base + 8, 1));
  EXPECT_EQ(wave->instruction_execution_error(), InstructionExecutionError::PluginFailure);
  EXPECT_NE(sink->str().find("cannot disassemble captured instruction at PC 0x1234"),
            std::string::npos);
  EXPECT_EQ(sink->str().find("RACE "), std::string::npos);
}

TEST_F(FormattedLoadRaceTest, ReportingFailurePreservesExistingExecutionFailure) {
  issue(rdna4::kBufferLoadFormatXVbuffer, 46, 8, 1);
  plugin_state->trace = {};
  wave->report_instruction_execution_error(InstructionExecutionError::InvalidRegisterAccess,
                                           "original failure");
  fixture.plugin_group_->onAmdgpuReadVgprLanes(wave, wave->vgpr_alloc().base + 8, 1);
  EXPECT_EQ(wave->instruction_execution_error(), InstructionExecutionError::InvalidRegisterAccess);
  EXPECT_STREQ(wave->instruction_execution_error_detail(), "original failure");
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
/// tensor store or, without any tensor instruction, a pending DS read. The
/// AsyncStore tests drive the routed-memory hook with gfx1250 async stores from
/// LDS, whose LDS source is an LDS read on ASYNCcnt. Race semantics are covered
/// in race_detector_tests.cpp; these tests cover the hooks' translation.
class TensorDmaRaceTest : public ::testing::Test {
protected:
  static constexpr uint64_t kTensorPc = 0x2010;

  PluginFixture fixture;
  StringSink *sink = nullptr;
  RaceDetectorPlugin *plugin = nullptr;
  std::array<Wavefront *, 2> waves{};
  std::unique_ptr<Decoder> decoder;

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(fixture.initialize(/*num_wf_slots=*/2, "cdna5",
                                               /*wavefront_size=*/32, /*sgprs_per_wf=*/128));
    PluginSinkConfig sink_config;
    sink = &sink_config.emplace<StringSink>();
    fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    auto race_plugin = std::make_unique<RaceDetectorPlugin>();
    plugin = race_plugin.get();
    ASSERT_TRUE(fixture.plugin_group_->add(std::move(race_plugin)));
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

  /// Issue a one-dword async load into LDS, as global_load_async_to_lds_b32
  /// does, and report whether it produced a race.
  bool async_load(int wave, uint32_t address) { return dtl_write(wave, address); }

  /// Issue an async store of `bytes` LDS bytes per lane to global memory, as
  /// global_store_async_from_lds does, and report whether it produced a race.
  /// Lane i reads from addresses[i]. kInvalidLdsAddress models a lane whose
  /// source lies outside the wave's LDS allocation; `in_range_bytes` models
  /// lanes that straddle its end and read only that many leading bytes. `exec`
  /// defaults to every lane in `addresses`; zero issues the store with no
  /// active lanes.
  bool async_store_lanes(int wave, std::span<const uint32_t> addresses,
                         uint32_t bytes = sizeof(uint32_t),
                         std::optional<uint32_t> in_range_bytes = std::nullopt,
                         std::optional<uint64_t> exec = std::nullopt) {
    waves[wave]->pc += 16;
    const size_t previous = sink->str().size();
    auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
    state->elem_size = std::min<uint32_t>(bytes, sizeof(uint32_t));
    state->num_elems = bytes / state->elem_size;
    state->is_load = false;
    state->lds_src = true;
    state->lds_per_lane_addr = true;
    state->per_lane_lds_addr.fill(kInvalidLdsAddress);
    for (size_t lane = 0; lane < addresses.size(); ++lane) {
      state->per_lane_lds_addr[lane] = addresses[lane];
      state->per_lane_lds_src_bytes[lane] = static_cast<uint8_t>(
          addresses[lane] == kInvalidLdsAddress ? 0 : in_range_bytes.value_or(bytes));
    }
    const uint64_t all_lanes =
        addresses.size() >= 64 ? ~uint64_t{0} : (uint64_t{1} << addresses.size()) - 1;
    const uint64_t lanes = exec.value_or(all_lanes);
    state->wf_size = waves[wave]->wf_size();
    state->exec_mask = lanes;
    state->lane_mask = lanes;
    TestMemoryInstruction inst(std::move(state),
                               {{WaitCounterType::ASYNCCNT, MemoryCompletionClass::ASYNC_STORE}});
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[wave]->pc, inst, *waves[wave]);
    fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, inst, *waves[wave]);
    return sink->str().find("RACE ", previous) != std::string::npos;
  }

  /// Issue a one-lane async store; see async_store_lanes().
  bool async_store(int wave, uint32_t address, uint32_t bytes = sizeof(uint32_t),
                   std::optional<uint32_t> in_range_bytes = std::nullopt, uint64_t exec = 1) {
    return async_store_lanes(wave, std::span<const uint32_t>(&address, 1), bytes, in_range_bytes,
                             exec);
  }

  /// Issue a one-lane ds_atomic_async_barrier_arrive_b64 on the barrier cell at
  /// `address`, as execution does. It completes on ASYNCcnt in the async-load
  /// class.
  void async_barrier_arrive(int wave, uint32_t address) {
    waves[wave]->pc += 16;
    auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
    state->elem_size = sizeof(uint64_t);
    state->num_elems = 1;
    state->is_load = false;
    state->wf_size = waves[wave]->wf_size();
    state->exec_mask = 1;
    state->lane_mask = 1;
    state->per_lane_addr[0] = address;
    TestMemoryInstruction inst(std::move(state),
                               {{WaitCounterType::ASYNCCNT, MemoryCompletionClass::ASYNC_LOAD}});
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[wave]->pc, inst, *waves[wave]);
    fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, inst, *waves[wave]);
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

// --- Async stores read their LDS source until ASYNCcnt retires them ---

TEST_F(TensorDmaRaceTest, AsyncStoreIsAnLdsReadUntilRetired) {
  async_store(0, 32);
  EXPECT_TRUE(ds_write(1, 32));
}

TEST_F(TensorDmaRaceTest, AsyncStoreIsRetiredByAsynccntAndBarrier) {
  async_store(0, 32);
  wait_asynccnt(0, 0);
  barrier();
  EXPECT_FALSE(ds_write(1, 32));
}

TEST_F(TensorDmaRaceTest, AsyncStoreBarrierWithoutAsynccntRaces) {
  async_store(0, 32);
  barrier();
  EXPECT_TRUE(ds_write(1, 32));
}

TEST_F(TensorDmaRaceTest, AsyncStoreAsynccntWithoutBarrierRacesInAnotherWave) {
  async_store(0, 32);
  wait_asynccnt(0, 0);
  EXPECT_TRUE(ds_write(1, 32));
}

// Async copies may access LDS out of order with DS instructions, so the storing
// wave's own DS write needs ASYNCcnt first.
TEST_F(TensorDmaRaceTest, SameWaveDsWriteOverAPendingAsyncStoreSourceRaces) {
  async_store(0, 32);
  EXPECT_TRUE(ds_write(0, 32));
}

TEST_F(TensorDmaRaceTest, SameWaveDsWriteAfterAsynccntIsSafe) {
  async_store(0, 32);
  wait_asynccnt(0, 0);
  EXPECT_FALSE(ds_write(0, 32));
}

TEST_F(TensorDmaRaceTest, AsyncStoreOfAPendingDsWriteRaces) {
  EXPECT_FALSE(ds_write(0, 4));
  EXPECT_TRUE(async_store(0, 4));
}

TEST_F(TensorDmaRaceTest, AsyncStoreAfterDscntIsSafe) {
  EXPECT_FALSE(ds_write(0, 4));
  wait_dscnt(0);
  EXPECT_FALSE(async_store(0, 4));
}

// Async loads and stores share ASYNCcnt but may access LDS out of order, even
// relative to each other (CDNA5 ISA 10.8).
TEST_F(TensorDmaRaceTest, AsyncStoreOfAPendingAsyncLoadRaces) {
  EXPECT_FALSE(async_load(0, 4));
  EXPECT_TRUE(async_store(0, 4));
}

TEST_F(TensorDmaRaceTest, AsyncStoreAfterAsynccntIsSafe) {
  EXPECT_FALSE(async_load(0, 4));
  wait_asynccnt(0, 0);
  EXPECT_FALSE(async_store(0, 4));
}

TEST_F(TensorDmaRaceTest, AsyncLoadOverAPendingAsyncStoreSourceRaces) {
  EXPECT_FALSE(async_store(0, 4));
  EXPECT_TRUE(async_load(0, 4));
}

TEST_F(TensorDmaRaceTest, AsyncStoresOfOneSourceDoNotRace) {
  EXPECT_FALSE(async_store(0, 4));
  EXPECT_FALSE(async_store(0, 4));
  EXPECT_FALSE(async_store(1, 4));
}

// Async loads complete in order with loads and stores with stores, but loads and
// stores complete out of order with each other (CDNA5 ISA 10.8). With a load and
// a store pending, waiting for all but one completes neither.
TEST_F(TensorDmaRaceTest, AsyncMixedPartialWaitLeavesTheStorePending) {
  async_load(0, 0);
  async_store(0, 64);
  wait_asynccnt(0, 1);
  EXPECT_TRUE(ds_write(0, 64));
}

TEST_F(TensorDmaRaceTest, AsyncMixedPartialWaitLeavesTheLoadPending) {
  async_load(0, 0);
  async_store(0, 64);
  wait_asynccnt(0, 1);
  EXPECT_TRUE(ds_read(0, 0));
}

TEST_F(TensorDmaRaceTest, AsyncStorePartialWaitRetiresTheOlderStore) {
  async_store(0, 0);
  async_store(0, 64);
  wait_asynccnt(0, 1);
  EXPECT_FALSE(ds_write(0, 0));
  EXPECT_TRUE(ds_write(0, 64));
}

// Load, store, load: waiting for all but one completes the older load, but the
// store may be the one still pending.
TEST_F(TensorDmaRaceTest, AsyncPartialWaitRetiresTheOlderLoadButNotTheStore) {
  async_load(0, 0);
  async_store(0, 64);
  async_load(0, 128);
  wait_asynccnt(0, 1);
  EXPECT_FALSE(ds_read(0, 0));
  EXPECT_TRUE(ds_write(0, 64));
  EXPECT_TRUE(ds_read(0, 128));
}

// Each store is one ASYNCcnt entry, whatever its number of lanes: of three
// stores, the middle one with two lanes, waiting for all but two completes only
// the first, and leaves both lanes of the middle store pending.
TEST_F(TensorDmaRaceTest, AsyncStoreTakesOneAsynccntSlot) {
  async_store(0, 0);
  const std::array<uint32_t, 2> lanes{64, 80};
  async_store_lanes(0, lanes);
  async_store(0, 128);
  wait_asynccnt(0, 2);
  EXPECT_FALSE(ds_write(0, 0));
  EXPECT_TRUE(ds_write(0, 64));
}

// Async copies and tensor DMA complete on different counters.
TEST_F(TensorDmaRaceTest, TensorLoadOverAPendingAsyncStoreSourceRaces) {
  async_store(0, 4);
  EXPECT_TRUE(tensor_load(0, {{.address = 0, .bytes = 16}}));
}

TEST_F(TensorDmaRaceTest, TensorLoadAfterOnlyTensorcntOverAnAsyncStoreSourceRaces) {
  async_store(0, 4);
  wait_tensorcnt(0, 0);
  EXPECT_TRUE(tensor_load(0, {{.address = 0, .bytes = 16}}));
}

TEST_F(TensorDmaRaceTest, TensorLoadAfterAsynccntOverAnAsyncStoreSourceIsSafe) {
  async_store(0, 4);
  wait_asynccnt(0, 0);
  EXPECT_FALSE(tensor_load(0, {{.address = 0, .bytes = 16}}));
}

TEST_F(TensorDmaRaceTest, AsyncStoreOfAPendingTensorLoadRaces) {
  tensor_load(0, {{.address = 0, .bytes = 16}});
  EXPECT_TRUE(async_store(0, 4));
}

TEST_F(TensorDmaRaceTest, AsyncStoreAfterOnlyAsynccntOfATensorLoadRaces) {
  tensor_load(0, {{.address = 0, .bytes = 16}});
  wait_asynccnt(0, 0);
  EXPECT_TRUE(async_store(0, 4));
}

TEST_F(TensorDmaRaceTest, AsyncStoreAfterTensorcntOfATensorLoadIsSafe) {
  tensor_load(0, {{.address = 0, .bytes = 16}});
  wait_tensorcnt(0, 0);
  EXPECT_FALSE(async_store(0, 4));
}

// A lane whose source lies outside the allocation reads no LDS bytes, so it
// does not race at address 0, but the store still takes its ASYNCcnt slot.
TEST_F(TensorDmaRaceTest, AsyncStoreLaneOutsideTheAllocationReadsNoLds) {
  async_store(0, kInvalidLdsAddress);
  EXPECT_FALSE(ds_write(1, 0));
}

TEST_F(TensorDmaRaceTest, AsyncStoreLaneOutsideTheAllocationKeepsItsAsynccntSlot) {
  async_store(0, 0);
  async_store(0, kInvalidLdsAddress);
  wait_asynccnt(0, 1);
  EXPECT_FALSE(ds_write(0, 0));
}

// A B128 lane reads 16 bytes; its last dword is part of the source.
TEST_F(TensorDmaRaceTest, AsyncStoreB128CoversTheLanesLastDword) {
  async_store(0, 0, /*bytes=*/16);
  EXPECT_FALSE(ds_write(1, 16));
  EXPECT_TRUE(ds_write(1, 12));
}

// A lane straddling the end of the allocation reads only its in-range bytes,
// so only those bytes are its source.
TEST_F(TensorDmaRaceTest, AsyncStoreLaneStraddlingTheAllocationTracksItsInRangeBytes) {
  async_store(0, 240, /*bytes=*/16, /*in_range_bytes=*/8);
  EXPECT_FALSE(ds_write(1, 248));
  EXPECT_TRUE(ds_write(1, 244));
}

// An async store with no active lanes takes one ASYNCcnt slot and no more: with
// it and an older store pending, waiting for all but two retires neither.
TEST_F(TensorDmaRaceTest, AsyncStoreWithoutActiveLanesLeavesTheOlderStorePendingUntilCovered) {
  async_store(0, 0);
  async_store(0, kInvalidLdsAddress, sizeof(uint32_t), std::nullopt, /*exec=*/0);
  wait_asynccnt(0, 2);
  EXPECT_TRUE(ds_write(0, 0));
}

// Lanes outside EXEC read no LDS, even with an address recorded: only lane 1
// of this store is active.
TEST_F(TensorDmaRaceTest, AsyncStoreLanesOutsideExecReadNoLds) {
  const std::array<uint32_t, 2> lanes{0, 16};
  async_store_lanes(0, lanes, sizeof(uint32_t), std::nullopt, /*exec=*/0b10);
  EXPECT_FALSE(ds_write(1, 0));
  EXPECT_TRUE(ds_write(1, 16));
}

// An async barrier arrive completes on ASYNCcnt in order with async loads
// (CDNA5 ISA 10.8), so it takes a slot that partial waits count with loads, and
// not with stores.
TEST_F(TensorDmaRaceTest, AsyncBarrierArriveCountsWithAsyncLoads) {
  async_load(0, 0);
  async_barrier_arrive(0, 192);
  wait_asynccnt(0, 1);
  EXPECT_FALSE(ds_read(0, 0));
}

TEST_F(TensorDmaRaceTest, AsyncBarrierArriveDoesNotCountWithAsyncStores) {
  async_store(0, 0);
  async_barrier_arrive(0, 192);
  wait_asynccnt(0, 1);
  EXPECT_TRUE(ds_write(0, 0));
}

// Each lane's source is tracked on its own: bytes between lanes are not read.
TEST_F(TensorDmaRaceTest, AsyncStoreLanesAreTrackedSeparately) {
  const std::array<uint32_t, 4> lanes{0, 16, 32, 48};
  async_store_lanes(0, lanes);
  EXPECT_FALSE(ds_write(1, 20));
  EXPECT_TRUE(ds_write(1, 32));
}

// A race found by the store names the lane whose source overlaps the pending
// write, and a byte both accesses touch.
TEST_F(TensorDmaRaceTest, AsyncStoreRaceIsReportedAtTheOverlappingLaneAndByte) {
  // Lane 2 reads bytes 112-127 but shares only bytes 120-123 with the write.
  EXPECT_FALSE(ds_write(0, 120));
  const std::array<uint32_t, 3> lanes{0, 64, 112};
  const size_t previous = sink->str().size();
  EXPECT_TRUE(async_store_lanes(0, lanes, /*bytes=*/16));
  const std::string report = sink->str().substr(previous);
  EXPECT_NE(report.find("type=LDS access=read"), std::string::npos) << report;
  EXPECT_NE(report.find("wave=0 lane=2"), std::string::npos) << report;
  EXPECT_NE(report.find("Race on LDS byte 120 ["), std::string::npos) << report;
}

// The store as the later access: another wave's async load into its source
// must be waited for by that wave and followed by a barrier.
TEST_F(TensorDmaRaceTest, AsyncStoreOfAnotherWavesAsyncLoadAfterWaitAndBarrierIsSafe) {
  async_load(1, 4);
  wait_asynccnt(1, 0);
  barrier();
  EXPECT_FALSE(async_store(0, 4));
}

TEST_F(TensorDmaRaceTest, AsyncStoreOfAnotherWavesAsyncLoadAfterOnlyItsWaitRaces) {
  async_load(1, 4);
  wait_asynccnt(1, 0);
  EXPECT_TRUE(async_store(0, 4));
}

// Reads of a pending store's source are safe: both accesses only read.
TEST_F(TensorDmaRaceTest, SameWaveDsReadOfAPendingAsyncStoreSourceIsSafe) {
  async_store(0, 4);
  EXPECT_FALSE(ds_read(0, 4));
}

TEST_F(TensorDmaRaceTest, TensorAndAsyncStoresOfOneTileAreSafe) {
  tensor_store(0, {{.address = 0, .bytes = 16}});
  EXPECT_FALSE(async_store(0, 4));
  EXPECT_FALSE(tensor_store(1, {{.address = 0, .bytes = 16}}));
}

// S_WAIT_IDLE waits for every counter, including ASYNCcnt.
TEST_F(TensorDmaRaceTest, AsyncStoreIsRetiredByWaitIdle) {
  async_store(0, 4);
  execute_wait(*fixture.plugin_group_, *waves[0], 0xbf8a0000u); // s_wait_idle
  EXPECT_FALSE(ds_write(0, 4));
}

/// Waits on counters other than ASYNCcnt, as encoded SOPP words.
class AsyncStoreOtherCounterWaitTest : public TensorDmaRaceTest,
                                       public ::testing::WithParamInterface<uint32_t> {};

TEST_P(AsyncStoreOtherCounterWaitTest, DoesNotRetireTheStore) {
  async_store(0, 4);
  execute_wait(*fixture.plugin_group_, *waves[0], GetParam());
  EXPECT_TRUE(ds_write(0, 4));
}

INSTANTIATE_TEST_SUITE_P(Waits, AsyncStoreOtherCounterWaitTest,
                         ::testing::Values(0xbfc90000u,   // s_wait_storecnt_dscnt 0
                                           0xbfc80000u,   // s_wait_loadcnt_dscnt 0
                                           0xbfc50000u)); // s_wait_xcnt 0

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

// Execute and route a decoded async store, so the race detector hears of it
// through the execution plugin group, as in a run.
TEST_F(TensorDmaRaceTest, ExecutedAsyncStoreRegistersItsLdsSource) {
  const uint32_t lds_base = fixture.cu()->allocate_lds(256);
  for (auto *wave : waves) {
    wave->set_lds_base(lds_base);
    wave->set_lds_size(256);
  }
  const uint32_t sgprs = waves[0]->sgpr_alloc().base;
  fixture.cu()->write_sgpr(sgprs, 0x10000u);
  fixture.cu()->write_sgpr(sgprs + 1, 0);
  const uint32_t vgprs = waves[0]->vgpr_alloc().base;
  fixture.cu()->write_vgpr(vgprs, 0, 0);      // v0: global offset.
  fixture.cu()->write_vgpr(vgprs + 1, 0, 32); // v1: LDS source.
  const auto words = cdna5::build_vglobal(cdna5::kGlobalStoreAsyncFromLdsB32Vglobal,
                                          {.saddr = 0, .vsrc = 1, .vaddr = 0});
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  waves[0]->pc = 0x3000;
  fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[0]->pc, *inst, *waves[0]);
  ASSERT_TRUE(fixture.cu()->execute_instruction(inst.get(), *waves[0]).succeeded());
  test::ComputeUnitTestAccess::route_memory_inst(*fixture.cu(), inst.release(), *waves[0]);

  EXPECT_FALSE(ds_write(1, lds_base + 36));
  const size_t previous = sink->str().size();
  EXPECT_TRUE(ds_write(1, lds_base + 32));
  const std::string report = sink->str().substr(previous);
  EXPECT_NE(report.find("type=LDS access=write"), std::string::npos) << report;
  EXPECT_NE(report.find("0x3000  global_store_async_from_lds_b32"), std::string::npos) << report;
}

// A lane whose LDS source lies outside the allocation reads zeros, so the store
// registers no LDS bytes for it.
TEST_F(TensorDmaRaceTest, ExecutedAsyncStoreOutsideTheAllocationRegistersNoLdsBytes) {
  const uint32_t lds_base = fixture.cu()->allocate_lds(256);
  for (auto *wave : waves) {
    wave->set_lds_base(lds_base);
    wave->set_lds_size(256);
  }
  const uint32_t sgprs = waves[0]->sgpr_alloc().base;
  fixture.cu()->write_sgpr(sgprs, 0x10000u);
  fixture.cu()->write_sgpr(sgprs + 1, 0);
  const uint32_t vgprs = waves[0]->vgpr_alloc().base;
  fixture.cu()->write_vgpr(vgprs, 0, 0);
  fixture.cu()->write_vgpr(vgprs + 1, 0, 256); // Just past the end of the allocation.
  const auto words = cdna5::build_vglobal(cdna5::kGlobalStoreAsyncFromLdsB32Vglobal,
                                          {.saddr = 0, .vsrc = 1, .vaddr = 0});
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[0]->pc, *inst, *waves[0]);
  ASSERT_TRUE(fixture.cu()->execute_instruction(inst.get(), *waves[0]).succeeded());
  test::ComputeUnitTestAccess::route_memory_inst(*fixture.cu(), inst.release(), *waves[0]);

  EXPECT_FALSE(ds_write(1, lds_base + 252));
  EXPECT_FALSE(ds_write(1, lds_base + 256));
  EXPECT_FALSE(ds_write(1, lds_base));
}

// A decoded B128 store at offset 250 of a 256-byte allocation reads bytes
// 250-255, so a write to its last two bytes races with it, and a write ending
// before byte 250 does not.
TEST_F(TensorDmaRaceTest, ExecutedAsyncStoreStraddlingTheAllocationTracksItsInRangeBytes) {
  const uint32_t lds_base = fixture.cu()->allocate_lds(256);
  for (auto *wave : waves) {
    wave->set_lds_base(lds_base);
    wave->set_lds_size(256);
  }
  const uint32_t sgprs = waves[0]->sgpr_alloc().base;
  fixture.cu()->write_sgpr(sgprs, 0x10000u);
  fixture.cu()->write_sgpr(sgprs + 1, 0);
  const uint32_t vgprs = waves[0]->vgpr_alloc().base;
  fixture.cu()->write_vgpr(vgprs, 0, 0);
  fixture.cu()->write_vgpr(vgprs + 1, 0, 250);
  const auto words = cdna5::build_vglobal(cdna5::kGlobalStoreAsyncFromLdsB128Vglobal,
                                          {.saddr = 0, .vsrc = 1, .vaddr = 0});
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[0]->pc, *inst, *waves[0]);
  ASSERT_TRUE(fixture.cu()->execute_instruction(inst.get(), *waves[0]).succeeded());
  test::ComputeUnitTestAccess::route_memory_inst(*fixture.cu(), inst.release(), *waves[0]);

  EXPECT_FALSE(ds_write(1, lds_base + 246));
  EXPECT_TRUE(ds_write(1, lds_base + 254));
}

// A decoded async store with EXEC zero still takes an ASYNCcnt slot, so waiting
// for all but one completes the store issued before it.
TEST_F(TensorDmaRaceTest, ExecutedAsyncStoreWithoutActiveLanesKeepsItsAsynccntSlot) {
  const uint32_t lds_base = fixture.cu()->allocate_lds(256);
  for (auto *wave : waves) {
    wave->set_lds_base(lds_base);
    wave->set_lds_size(256);
  }
  async_store(0, lds_base);
  const uint32_t sgprs = waves[0]->sgpr_alloc().base;
  fixture.cu()->write_sgpr(sgprs, 0x10000u);
  fixture.cu()->write_sgpr(sgprs + 1, 0);
  const auto words = cdna5::build_vglobal(cdna5::kGlobalStoreAsyncFromLdsB32Vglobal,
                                          {.saddr = 0, .vsrc = 1, .vaddr = 0});
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  waves[0]->set_exec(0);
  waves[0]->pc += 16;
  fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[0]->pc, *inst, *waves[0]);
  ASSERT_TRUE(fixture.cu()->execute_instruction(inst.get(), *waves[0]).succeeded());
  test::ComputeUnitTestAccess::route_memory_inst(*fixture.cu(), inst.release(), *waves[0]);
  waves[0]->set_exec(1);

  wait_asynccnt(0, 1);
  EXPECT_FALSE(ds_write(0, lds_base));
}

// A decoded async load with EXEC zero still takes an ASYNCcnt slot, so waiting
// for all but one completes the load issued before it.
TEST_F(TensorDmaRaceTest, ExecutedAsyncLoadWithoutActiveLanesKeepsItsAsynccntSlot) {
  const uint32_t lds_base = fixture.cu()->allocate_lds(256);
  for (auto *wave : waves) {
    wave->set_lds_base(lds_base);
    wave->set_lds_size(256);
  }
  async_load(0, lds_base);
  const uint32_t sgprs = waves[0]->sgpr_alloc().base;
  fixture.cu()->write_sgpr(sgprs, 0x10000u);
  fixture.cu()->write_sgpr(sgprs + 1, 0);
  const auto words = cdna5::build_vglobal(cdna5::kGlobalLoadAsyncToLdsB32Vglobal,
                                          {.saddr = 0, .vdst = 1, .vaddr = 0});
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  waves[0]->set_exec(0);
  waves[0]->pc += 16;
  fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[0]->pc, *inst, *waves[0]);
  ASSERT_TRUE(fixture.cu()->execute_instruction(inst.get(), *waves[0]).succeeded());
  test::ComputeUnitTestAccess::route_memory_inst(*fixture.cu(), inst.release(), *waves[0]);
  waves[0]->set_exec(1);

  wait_asynccnt(0, 1);
  EXPECT_FALSE(ds_read(0, lds_base));
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

/// Drives gfx1250 counter-capacity backpressure through the plugin's hooks.
/// Each split counter holds 63 tokens (CDNA5 ISA 5.7.1); issuing one more
/// stalls until the oldest completes, which proves it complete for its wave.
/// Each test issues an oldest operation and enough younger ones to fill or
/// overflow the counter, then probes the oldest operation's register or LDS
/// bytes. Probes are same-wave accesses in another completion class, so they
/// race unless the oldest operation is complete.
class Gfx1250CounterCapacityRaceTest : public TensorDmaRaceTest {
protected:
  // LOADcnt, DScnt, ASYNCcnt and TENSORcnt are 6-bit counters (CDNA5 ISA 5.7.1,
  // table "Counter Size Description"), and issue stalls rather than overflow
  // one, so each holds 2^6 - 1 tokens.
  static constexpr int kCapacity = 63;
  static constexpr uint32_t kOldestAddress = 0;
  static constexpr uint32_t kYoungerAddress = 64;
  static constexpr uint32_t kFlatLdsAddress = 128;

  /// Issue a one-lane, one-dword vector memory operation on the global route,
  /// as execution does: the issue hook, then the routed-memory hook.
  void vmem(int wave, bool is_load, uint32_t vgpr, uint64_t exec = 1) {
    waves[wave]->pc += 16;
    auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
    state->elem_size = sizeof(uint32_t);
    state->num_elems = 1;
    state->is_load = is_load;
    state->wf_size = waves[wave]->wf_size();
    state->exec_mask = exec;
    state->lane_mask = exec;
    state->dst_reg_base = waves[wave]->vgpr_alloc().base + vgpr;
    const auto counter = is_load ? WaitCounterType::LOADCNT : WaitCounterType::STORECNT;
    TestMemoryInstruction inst(std::move(state), {{counter, MemoryCompletionClass::VMEM}});
    issue(wave, inst, {}, exec);
  }

  void vmem_load(int wave, uint32_t vgpr, uint64_t exec = 1) { vmem(wave, true, vgpr, exec); }
  void vmem_store(int wave) { vmem(wave, false, /*vgpr=*/0); }

  enum class FlatRoute { Global, Lds, NoActiveLanes };

  /// Issue a one-lane, one-dword generic FLAT load whose addresses resolved as
  /// `route` says. It carries both decoded counter obligations, as gfx1250
  /// FLAT metadata does: LOADcnt in the VMEM class and DScnt in the LDS class.
  void flat_load(int wave, uint32_t vgpr, FlatRoute route) {
    waves[wave]->pc += 16;
    const bool lds = route == FlatRoute::Lds;
    const uint64_t exec = route == FlatRoute::NoActiveLanes ? 0 : 1;
    auto state = std::make_unique<VectorMemState>(lds ? LOCAL_MEM : GLOBAL_MEM);
    state->elem_size = sizeof(uint32_t);
    state->num_elems = 1;
    state->is_load = true;
    state->wf_size = waves[wave]->wf_size();
    state->exec_mask = exec;
    state->lane_mask = exec;
    state->dst_reg_base = waves[wave]->vgpr_alloc().base + vgpr;
    state->per_lane_addr[0] = kFlatLdsAddress;
    TestMemoryInstruction inst(std::move(state),
                               {{WaitCounterType::LOADCNT, MemoryCompletionClass::VMEM},
                                {WaitCounterType::DSCNT, MemoryCompletionClass::LDS}},
                               /*exec_masked=*/true, "flat_load_b32");
    MemoryAccessObservation access{};
    access.route = lds ? MemoryRoute::LOCAL : MemoryRoute::GLOBAL;
    access.decoded_space = DecodedMemorySpace::FLAT;
    access.request_lane_mask = exec;
    access.normalized_to_local = lds;
    access.flat_local_lane_mask = lds ? exec : 0;
    issue(wave, inst, access, exec);
  }

  /// Deliver a memory instruction to the issue hook, then the routed-memory
  /// hook, with the wave's EXEC set to `exec` for the issue.
  void issue(int wave, TestMemoryInstruction &inst, const MemoryAccessObservation &access,
             uint64_t exec) {
    const uint64_t wave_exec = waves[wave]->exec();
    waves[wave]->set_exec(exec);
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[wave]->pc, inst, *waves[wave]);
    fixture.plugin_group_->onAmdgpuMemoryAccessRouted(access, inst, *waves[wave]);
    waves[wave]->set_exec(wave_exec);
  }

  void wait_loadcnt(int wave, uint16_t count) {
    execute_wait(*fixture.plugin_group_, *waves[wave], 0xbfc00000u | count);
  }

  using TensorDmaRaceTest::wait_dscnt;
  void wait_dscnt(int wave, uint16_t count) {
    execute_wait(*fixture.plugin_group_, *waves[wave], 0xbfc60000u | count);
  }

  /// Issue an instruction decoded from `word` through the issue hook only, as
  /// execution does for one that is not a memory operation.
  void issue_word(int wave, uint32_t word) {
    waves[wave]->pc += 16;
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &word));
    ASSERT_NE(inst, nullptr);
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(waves[wave]->pc, *inst, *waves[wave]);
  }

  /// Read one lane of a VGPR and report whether it raced.
  bool vgpr_read(int wave, uint32_t vgpr) {
    waves[wave]->pc += 16;
    record_register_probe(*fixture.plugin_group_, *waves[wave]);
    const size_t previous = sink->str().size();
    fixture.plugin_group_->onAmdgpuReadVgprLanes(waves[wave], waves[wave]->vgpr_alloc().base + vgpr,
                                                 /*lane_mask=*/1);
    return sink->str().find("RACE ", previous) != std::string::npos;
  }

  bool tensor_load_oldest() { return tensor_load(0, {{.address = kOldestAddress, .bytes = 4}}); }

  WaveRaceState &race_state(int wave) {
    return *static_cast<RaceWavefrontState *>(waves[wave]->plugin_state(plugin->slot_index()))
                ->race_state;
  }
};

TEST_F(Gfx1250CounterCapacityRaceTest, FullLoadcntCompletesOldestLoad) {
  vmem_load(0, /*vgpr=*/0);
  for (int index = 0; index < kCapacity; ++index)
    vmem_load(0, /*vgpr=*/1);
  EXPECT_FALSE(vgpr_read(0, 0));
}

TEST_F(Gfx1250CounterCapacityRaceTest, LoadcntAtCapacityKeepsOldestLoadPending) {
  vmem_load(0, /*vgpr=*/0);
  for (int index = 0; index < kCapacity - 1; ++index)
    vmem_load(0, /*vgpr=*/1);
  EXPECT_TRUE(vgpr_read(0, 0));
}

// Loads and stores use separate counters on gfx1250 (CDNA5 ISA 5.7.1), so a
// full STOREcnt says nothing about a pending load.
TEST_F(Gfx1250CounterCapacityRaceTest, StorecntPressureDoesNotCompleteALoad) {
  vmem_load(0, /*vgpr=*/0);
  for (int index = 0; index < kCapacity; ++index)
    vmem_store(0);
  EXPECT_TRUE(vgpr_read(0, 0));
}

// A VMEM instruction with EXEC zero still counts unless every VMEM counter is
// zero (CDNA5 ISA 3.2.2.1), so it applies backpressure like any other load.
TEST_F(Gfx1250CounterCapacityRaceTest, ExecZeroLoadTakesALoadcntSlot) {
  vmem_load(0, /*vgpr=*/0);
  for (int index = 0; index < kCapacity - 1; ++index)
    vmem_load(0, /*vgpr=*/1);
  vmem_load(0, /*vgpr=*/2, /*exec=*/0);
  EXPECT_FALSE(vgpr_read(0, 0));
}

TEST_F(Gfx1250CounterCapacityRaceTest, ExecZeroLoadBelowCapacityKeepsOldestLoadPending) {
  vmem_load(0, /*vgpr=*/0);
  for (int index = 0; index < kCapacity - 2; ++index)
    vmem_load(0, /*vgpr=*/1);
  vmem_load(0, /*vgpr=*/2, /*exec=*/0);
  EXPECT_TRUE(vgpr_read(0, 0));
}

// Every FLAT instruction increments both DScnt and LOADcnt or STOREcnt when it
// issues, whichever memory its addresses resolve to (CDNA5 ISA 5.7.1.3), so it
// applies backpressure to both counters.
TEST_F(Gfx1250CounterCapacityRaceTest, FlatResolvedToGlobalAdvancesLoadcnt) {
  vmem_load(0, /*vgpr=*/0);
  for (int index = 0; index < kCapacity - 1; ++index)
    vmem_load(0, /*vgpr=*/1);
  flat_load(0, /*vgpr=*/2, FlatRoute::Global);
  EXPECT_FALSE(vgpr_read(0, 0));
}

TEST_F(Gfx1250CounterCapacityRaceTest, FlatResolvedToGlobalAdvancesDscnt) {
  ds_read(0, kOldestAddress);
  for (int index = 0; index < kCapacity - 1; ++index)
    ds_read(0, kYoungerAddress);
  flat_load(0, /*vgpr=*/2, FlatRoute::Global);
  EXPECT_FALSE(tensor_load_oldest());
}

TEST_F(Gfx1250CounterCapacityRaceTest, FlatResolvedToLdsAdvancesLoadcnt) {
  vmem_load(0, /*vgpr=*/0);
  for (int index = 0; index < kCapacity - 1; ++index)
    vmem_load(0, /*vgpr=*/1);
  flat_load(0, /*vgpr=*/2, FlatRoute::Lds);
  EXPECT_FALSE(vgpr_read(0, 0));
}

TEST_F(Gfx1250CounterCapacityRaceTest, FlatResolvedToLdsAdvancesDscnt) {
  ds_read(0, kOldestAddress);
  for (int index = 0; index < kCapacity - 1; ++index)
    ds_read(0, kYoungerAddress);
  flat_load(0, /*vgpr=*/2, FlatRoute::Lds);
  EXPECT_FALSE(tensor_load_oldest());
}

// A FLAT with EXEC zero still counts on both counters unless they are all zero
// (CDNA5 ISA 3.2.2.1).
TEST_F(Gfx1250CounterCapacityRaceTest, FlatWithNoActiveLanesAdvancesLoadcnt) {
  vmem_load(0, /*vgpr=*/0);
  for (int index = 0; index < kCapacity - 1; ++index)
    vmem_load(0, /*vgpr=*/1);
  flat_load(0, /*vgpr=*/2, FlatRoute::NoActiveLanes);
  EXPECT_FALSE(vgpr_read(0, 0));
}

TEST_F(Gfx1250CounterCapacityRaceTest, FlatWithNoActiveLanesAdvancesDscnt) {
  ds_read(0, kOldestAddress);
  for (int index = 0; index < kCapacity - 1; ++index)
    ds_read(0, kYoungerAddress);
  flat_load(0, /*vgpr=*/2, FlatRoute::NoActiveLanes);
  EXPECT_FALSE(tensor_load_oldest());
}

TEST_F(Gfx1250CounterCapacityRaceTest, FlatAtCapacityKeepsOldestDsReadPending) {
  ds_read(0, kOldestAddress);
  for (int index = 0; index < kCapacity - 2; ++index)
    ds_read(0, kYoungerAddress);
  flat_load(0, /*vgpr=*/2, FlatRoute::Global);
  EXPECT_TRUE(tensor_load_oldest());
}

// A FLAT keeps its slot in the counter of the domain it did not use, in order
// with that domain's other operations (CDNA5 ISA 5.7.1.3), so partial waits
// count it. A DS read, then a FLAT to global memory: DScnt holds two, and
// waiting for one to remain completes the DS read.
TEST_F(Gfx1250CounterCapacityRaceTest, PartialDscntWaitCountsGlobalFlatSlot) {
  ds_read(0, kOldestAddress);
  flat_load(0, /*vgpr=*/2, FlatRoute::Global);
  wait_dscnt(0, 1);
  EXPECT_FALSE(tensor_load_oldest());
}

TEST_F(Gfx1250CounterCapacityRaceTest, PartialDscntWaitAllowingBothKeepsDsReadPending) {
  ds_read(0, kOldestAddress);
  flat_load(0, /*vgpr=*/2, FlatRoute::Global);
  wait_dscnt(0, 2);
  EXPECT_TRUE(tensor_load_oldest());
}

TEST_F(Gfx1250CounterCapacityRaceTest, PartialLoadcntWaitCountsLdsFlatSlot) {
  vmem_load(0, /*vgpr=*/0);
  flat_load(0, /*vgpr=*/2, FlatRoute::Lds);
  wait_loadcnt(0, 1);
  EXPECT_FALSE(vgpr_read(0, 0));
}

TEST_F(Gfx1250CounterCapacityRaceTest, PartialLoadcntWaitAllowingBothKeepsLoadPending) {
  vmem_load(0, /*vgpr=*/0);
  flat_load(0, /*vgpr=*/2, FlatRoute::Lds);
  wait_loadcnt(0, 2);
  EXPECT_TRUE(vgpr_read(0, 0));
}

// The slot in the unused domain does not delay the FLAT's own result, which
// depends only on the domain its addresses used.
TEST_F(Gfx1250CounterCapacityRaceTest, GlobalFlatResultIsReadyAfterLoadcntWait) {
  flat_load(0, /*vgpr=*/2, FlatRoute::Global);
  wait_loadcnt(0, 0);
  EXPECT_FALSE(vgpr_read(0, 2));
}

TEST_F(Gfx1250CounterCapacityRaceTest, LdsFlatResultIsReadyAfterDscntWait) {
  flat_load(0, /*vgpr=*/2, FlatRoute::Lds);
  wait_dscnt(0, 0);
  EXPECT_FALSE(vgpr_read(0, 2));
}

TEST_F(Gfx1250CounterCapacityRaceTest, FullDscntCompletesOldestDsRead) {
  ds_read(0, kOldestAddress);
  for (int index = 0; index < kCapacity; ++index)
    ds_read(0, kYoungerAddress);
  EXPECT_FALSE(tensor_load_oldest());
}

TEST_F(Gfx1250CounterCapacityRaceTest, DscntAtCapacityKeepsOldestDsReadPending) {
  ds_read(0, kOldestAddress);
  for (int index = 0; index < kCapacity - 1; ++index)
    ds_read(0, kYoungerAddress);
  EXPECT_TRUE(tensor_load_oldest());
}

// Messages use KMcnt on gfx1250 (CDNA5 ISA 5.7.1.5), not a counter shared with
// DS operations.
TEST_F(Gfx1250CounterCapacityRaceTest, SendmsgDoesNotAdvanceDscnt) {
  ds_read(0, kOldestAddress);
  for (int index = 0; index < kCapacity - 1; ++index)
    ds_read(0, kYoungerAddress);
  ASSERT_NO_FATAL_FAILURE(issue_word(0, 0xBFB60000u)); // s_sendmsg 0.
  EXPECT_TRUE(tensor_load_oldest());
}

// S_SENDMSG_RTN_B* increments KMcnt by 2 (CDNA5 ISA 5.4), and other messages by
// 1 (CDNA5 ISA 5.7.1), on a 5-bit KMcnt that holds 2^5 - 1 tokens. No gfx1250
// instruction registers an event in an ordered KMcnt class, so these tests
// register synthetic ones to observe each message's increment.
class Gfx1250MessageCapacityRaceTest : public Gfx1250CounterCapacityRaceTest {
protected:
  static constexpr int kKmcntCapacity = 31;

  /// Leave one KMcnt slot free with ordered synthetic events.
  void fill_kmcnt_but_one() {
    for (int index = 0; index < kKmcntCapacity - 1; ++index)
      register_pending_event(race_state(0), /*pc=*/0x80 + 4 * index,
                             MemoryEventType::VGPR_TO_GLOBAL, /*registers=*/{},
                             WaitCounterType::KMCNT, MemoryOrderClass::VMEM);
  }

  EventStatus status(int event) {
    return race_state(0).getDetector()->events().status(EventId{event});
  }
};

TEST_F(Gfx1250MessageCapacityRaceTest, ReturningMessageTakesTwoKmcntSlots) {
  fill_kmcnt_but_one();
  ASSERT_NO_FATAL_FAILURE(issue_word(0, 0xBE804C00u)); // s_sendmsg_rtn_b32 s0, 0.
  EXPECT_EQ(status(0), EventStatus::WAVE_COMPLETE);
  EXPECT_EQ(status(1), EventStatus::ACTIVE);
}

TEST_F(Gfx1250MessageCapacityRaceTest, MessageTakesOneKmcntSlot) {
  fill_kmcnt_but_one();
  ASSERT_NO_FATAL_FAILURE(issue_word(0, 0xBFB60000u)); // s_sendmsg 0.
  EXPECT_EQ(status(0), EventStatus::ACTIVE);
}

TEST_F(Gfx1250CounterCapacityRaceTest, FullAsynccntCompletesOldestAsyncLoad) {
  dtl_write(0, kOldestAddress);
  for (int index = 0; index < kCapacity; ++index)
    dtl_write(0, kYoungerAddress);
  EXPECT_FALSE(ds_read(0, kOldestAddress));
}

TEST_F(Gfx1250CounterCapacityRaceTest, AsynccntAtCapacityKeepsOldestAsyncLoadPending) {
  dtl_write(0, kOldestAddress);
  for (int index = 0; index < kCapacity - 1; ++index)
    dtl_write(0, kYoungerAddress);
  EXPECT_TRUE(ds_read(0, kOldestAddress));
}

// Transfers that move no data still take a TENSORcnt slot.
TEST_F(Gfx1250CounterCapacityRaceTest, FullTensorcntCompletesOldestTensorLoad) {
  tensor_load_oldest();
  for (int index = 0; index < kCapacity; ++index)
    tensor_load(0, {});
  EXPECT_FALSE(ds_read(0, kOldestAddress));
}

TEST_F(Gfx1250CounterCapacityRaceTest, TensorcntAtCapacityKeepsOldestTensorLoadPending) {
  tensor_load_oldest();
  for (int index = 0; index < kCapacity - 1; ++index)
    tensor_load(0, {});
  EXPECT_TRUE(ds_read(0, kOldestAddress));
}

// A tensor instruction's admission completes the oldest transfer before its
// own LDS bytes are checked, so overwriting that transfer's source is safe.
TEST_F(Gfx1250CounterCapacityRaceTest, TensorAdmissionPrecedesItsLdsCheck) {
  tensor_store(0, {{.address = kOldestAddress, .bytes = 4}});
  for (int index = 0; index < kCapacity - 1; ++index)
    tensor_load(0, {});
  EXPECT_FALSE(tensor_load_oldest());
}

TEST_F(Gfx1250CounterCapacityRaceTest, TensorBelowCapacityOverwritingAPendingSourceRaces) {
  tensor_store(0, {{.address = kOldestAddress, .bytes = 4}});
  for (int index = 0; index < kCapacity - 2; ++index)
    tensor_load(0, {});
  EXPECT_TRUE(tensor_load_oldest());
}

// Capacity proves completion only for the issuing wave; another wave still
// needs a barrier.
TEST_F(Gfx1250CounterCapacityRaceTest, CapacityCompletedDsWriteIsReleasedByABarrier) {
  ds_write(0, kOldestAddress);
  for (int index = 0; index < kCapacity; ++index)
    ds_write(0, kYoungerAddress);
  barrier();
  EXPECT_FALSE(ds_read(1, kOldestAddress));
}

TEST_F(Gfx1250CounterCapacityRaceTest, DsWriteBelowCapacityIsNotReleasedByABarrier) {
  ds_write(0, kOldestAddress);
  for (int index = 0; index < kCapacity - 1; ++index)
    ds_write(0, kYoungerAddress);
  barrier();
  EXPECT_TRUE(ds_read(1, kOldestAddress));
}

// Wave 1 writes LDS and may end; wave 0 then passes a barrier and reads the
// bytes. The fixture's DS helpers are not specific to tensor DMA.
class ExitedWaveRaceTest : public TensorDmaRaceTest {
protected:
  void write_and_exit(bool wait) {
    EXPECT_FALSE(ds_write(1, 64));
    if (wait)
      wait_dscnt(1);
    fixture.plugin_group_->onAmdgpuWavefrontHalted(*waves[1]);
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
  fixture.plugin_group_->onAmdgpuWavefrontHalted(*waves[1]);
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
    if (arch == "cdna5")
      state->wait_counter_type = WaitCounterType::LOADCNT; // gfx1250 global loads.
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

TEST(RaceDetectorPluginTest, DualOffsetStoreTracksBothOffsetOrders) {
  constexpr uint8_t kFarOffset = 4;
  constexpr uint32_t kReadAddress = kFarOffset * sizeof(uint32_t);
  for (const std::array<uint8_t, 2> offsets :
       {std::array<uint8_t, 2>{0, kFarOffset}, std::array<uint8_t, 2>{kFarOffset, 0}}) {
    SCOPED_TRACE(std::format("offsets=({}, {})", offsets[0], offsets[1]));
    PluginFixture fixture;
    ASSERT_NO_FATAL_FAILURE(fixture.initialize(/*num_wf_slots=*/2, "cdna4"));
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    ASSERT_TRUE(fixture.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    fixture.soc->set_plugin_group(fixture.plugin_group_);
    fixture.plugin_group_->onInit();

    Wavefront *writer = fixture.cu()->dispatch_wf(0, 0x1000, 104, 256);
    Wavefront *reader = fixture.cu()->dispatch_wf(0, 0x2000, 104, 256);
    ASSERT_NE(writer, nullptr);
    ASSERT_NE(reader, nullptr);
    writer->set_lds_size(256);
    reader->set_lds_size(256);
    // One active lane leaves descending, disjoint offsets out of order until
    // interval finalization, without neighboring lanes merging the ranges.
    writer->set_exec(1);
    reader->set_exec(1);
    std::array<Wavefront *, 2> waves{writer, reader};
    fixture.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 512, 208, waves);
    fixture.cu()->write_vgpr(writer->vgpr_alloc().base, 0, 0);
    fixture.cu()->write_vgpr(writer->vgpr_alloc().base + 1, 0, 11);
    fixture.cu()->write_vgpr(writer->vgpr_alloc().base + 2, 0, 22);
    fixture.cu()->write_vgpr(reader->vgpr_alloc().base, 0, kReadAddress);

    std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_NE(decoder, nullptr);
    const std::array<uint32_t, 2> store_words = cdna4::build_ds(
        cdna4::kDsWrite2B32Ds,
        {.offset0 = offsets[0], .offset1 = offsets[1], .addr = 0, .data0 = 1, .data1 = 2});
    std::unique_ptr<Instruction> store(decode_valid(*decoder, store_words.data()));
    ASSERT_NE(store, nullptr);
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(writer->pc, *store, *writer);
    ASSERT_TRUE(fixture.cu()->execute_instruction(store.get(), *writer).succeeded());
    fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, *store, *writer);
    ASSERT_EQ(sink.str().find("RACE "), std::string::npos) << sink.str();

    const std::array<uint32_t, 2> load_words =
        cdna4::build_ds(cdna4::kDsReadB32Ds, {.addr = 0, .vdst = 8});
    std::unique_ptr<Instruction> load(decode_valid(*decoder, load_words.data()));
    ASSERT_NE(load, nullptr);
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(reader->pc, *load, *reader);
    ASSERT_TRUE(fixture.cu()->execute_instruction(load.get(), *reader).succeeded());
    fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, *load, *reader);

    const std::string report = sink.str();
    EXPECT_NE(report.find(std::format("Race on LDS byte {} [", kReadAddress)), std::string::npos)
        << report;
    EXPECT_NE(report.find("ds_write2_b32"), std::string::npos) << report;
    EXPECT_NE(report.find("ds_read_b32"), std::string::npos) << report;
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

TEST(RaceDetectorPluginTest, DualOffsetIntervalsPreserveUnionAndInstructionLifetime) {
  constexpr uint32_t kElementBytes = sizeof(uint32_t);
  struct AccessPattern {
    const char *name;
    uint32_t lane_stride;
    uint32_t second_base;
  };
  for (const AccessPattern pattern :
       {AccessPattern{"disjoint sets", kElementBytes, 4096},
        AccessPattern{"overlapping sets", kElementBytes, 64}, AccessPattern{"broadcast", 0, 0}}) {
    SCOPED_TRACE(pattern.name);
    for (bool is_load : {false, true}) {
      SCOPED_TRACE(is_load ? "read" : "write");
      PluginFixture fixture;
      ASSERT_NO_FATAL_FAILURE(fixture.initialize(/*num_wf_slots=*/1));
      auto plugin = std::make_unique<RaceDetectorPlugin>();
      RaceDetectorPlugin *plugin_ptr = plugin.get();
      fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
      ASSERT_TRUE(fixture.plugin_group_->add(std::move(plugin)));
      fixture.soc->set_plugin_group(fixture.plugin_group_);
      fixture.plugin_group_->onInit();

      Wavefront *wave = fixture.cu()->dispatch_wf(0, 0, 104, 256);
      ASSERT_NE(wave, nullptr);
      const uint32_t lanes = wave->wf_size();
      const uint64_t exec = lanes == 64 ? ~uint64_t{0} : (uint64_t{1} << lanes) - 1;
      wave->set_lds_size(pattern.second_base + pattern.lane_stride * (lanes - 1) + kElementBytes);
      wave->set_exec(exec);
      std::array<Wavefront *, 1> waves{wave};
      fixture.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, 104, waves);

      auto memory = std::make_unique<VectorMemState>(LOCAL_MEM);
      memory->elem_size = kElementBytes;
      memory->num_elems = 1;
      memory->is_load = is_load;
      memory->wf_size = lanes;
      memory->exec_mask = memory->lane_mask = exec;
      memory->ds2_active = true;
      memory->dst_reg_base = wave->vgpr_alloc().base;
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        memory->per_lane_addr[lane] = pattern.lane_stride * lane;
        memory->ds2_per_lane_addr[lane] = pattern.second_base + pattern.lane_stride * lane;
      }
      TestMemoryInstruction instruction(std::move(memory));
      for (uint64_t pc : {0x1000, 0x2000}) {
        wave->pc = pc;
        fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(pc, instruction, *wave);
        fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, instruction, *wave);
      }

      auto *state = static_cast<RaceWavefrontState *>(wave->plugin_state(plugin_ptr->slot_index()));
      WaveRaceState &race = *state->race_state;
      const RaceDetector &detector = *race.getDetector();
      const EventRegistry &events = detector.events();
      ASSERT_EQ(events.totalAllocated(), 2);
      const int first_end = static_cast<int>(pattern.lane_stride * (lanes - 1) + kElementBytes);
      const int second_start = static_cast<int>(pattern.second_base);
      for (EventId event : {EventId{0}, EventId{1}}) {
        const IntervalSet &intervals = events.ldsIntervals(event);
        ASSERT_EQ(intervals.size(), second_start > first_end ? 2u : 1u);
        EXPECT_EQ(intervals.begin()->start, 0);
        EXPECT_EQ(intervals.begin()->end,
                  second_start > first_end ? first_end : second_start + first_end);
        EXPECT_EQ(std::prev(intervals.end())->end, second_start + first_end);
        EXPECT_EQ(intervals.getTotalBytes(),
                  second_start > first_end ? 2 * first_end : second_start + first_end);
        EXPECT_EQ(events.execMask(event), exec);
        EXPECT_EQ(events.pendingCounterIncrement(event, WaitCounterType::LGKMCNT), 1);
      }

      // Completing one instruction must not remove the other instruction's
      // footprint, even when every lane and both address sets overlap.
      const IntervalSet &accessed = events.ldsIntervals(EventId{1});
      for (int remaining : {1, 0}) {
        PendingWaitCount wait;
        wait.add(WaitCounterType::LGKMCNT, remaining);
        race.dispatch(wait);
        race.flushBarrierPendingEvents();
        EXPECT_EQ(events.status(EventId{0}), EventStatus::RETIRED);
        EXPECT_EQ(events.status(EventId{1}),
                  remaining ? EventStatus::ACTIVE : EventStatus::RETIRED);
        EXPECT_EQ(detector.mayConflict(accessed, WaveId{0}, /*isWrite=*/is_load,
                                       MemoryOrderClass::UNORDERED),
                  remaining != 0);
      }
    }
  }
}

TEST(RaceDetectorPluginTest, LdsIntervalsRejectAddressesOutsideSignedIndexRange) {
  // Logical capacity does not materialize host storage. Exercise the interval
  // index boundary without allocating a multi-gigabyte LDS buffer.
  constexpr uint64_t kIndexLimit = std::numeric_limits<int>::max();
  Lds backing(static_cast<uint32_t>(kIndexLimit / 1024 + 2));
  PluginFixture fixture;
  ASSERT_NO_FATAL_FAILURE(fixture.initialize(/*num_wf_slots=*/1));
  auto plugin = std::make_unique<RaceDetectorPlugin>();
  RaceDetectorPlugin *plugin_ptr = plugin.get();
  fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(fixture.plugin_group_->add(std::move(plugin)));
  fixture.soc->set_plugin_group(fixture.plugin_group_);
  fixture.plugin_group_->onInit();

  Wavefront *wave = fixture.cu()->dispatch_wf(0, 0, 104, 256);
  ASSERT_NE(wave, nullptr);
  wave->set_lds(&backing);
  wave->set_lds_size(static_cast<uint32_t>(backing.size_bytes()));
  wave->set_exec(1);
  std::array<Wavefront *, 1> waves{wave};
  fixture.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, 256, 104, waves);
  auto *state = static_cast<RaceWavefrontState *>(wave->plugin_state(plugin_ptr->slot_index()));
  const EventRegistry &events = state->race_state->getDetector()->events();

  for (uint64_t address : {kIndexLimit - 1, kIndexLimit + 1}) {
    SCOPED_TRACE(address);
    const EventId event{events.totalAllocated()};
    auto memory = std::make_unique<VectorMemState>(LOCAL_MEM);
    memory->elem_size = sizeof(uint32_t);
    memory->num_elems = 1;
    memory->is_load = true;
    memory->wf_size = wave->wf_size();
    memory->exec_mask = memory->lane_mask = 1;
    memory->dst_reg_base = wave->vgpr_alloc().base;
    memory->per_lane_addr[0] = address;
    TestMemoryInstruction load(std::move(memory));
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave->pc, load, *wave);
    fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wave);

    ASSERT_EQ(events.totalAllocated(), event.value + 1);
    EXPECT_TRUE(events.ldsIntervals(event).empty());
    EXPECT_EQ(events.status(event), EventStatus::ACTIVE);
    ASSERT_EQ(events.registers(event).size(), 1u);
    EXPECT_EQ(events.registers(event).front(), wave->vgpr_alloc().base);
  }
}

// The pending stores lie inside the span of the strided load, so a check of
// the whole span cannot tell them apart. Only the store that a lane touches
// is a race.
TEST(RaceDetectorPluginTest, StridedLdsLoadReportsOnlyTheStoreALaneTouches) {
  // Separate the gap-only case so report deduplication cannot hide a false
  // positive behind the real conflict in the other case.
  for (bool include_racing_store : {false, true}) {
    SCOPED_TRACE(include_racing_store);
    PluginFixture fixture;
    ASSERT_NO_FATAL_FAILURE(fixture.initialize(/*num_wf_slots=*/2));
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    ASSERT_TRUE(fixture.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    fixture.soc->set_plugin_group(fixture.plugin_group_);
    fixture.plugin_group_->onInit();

    Wavefront *writer = fixture.cu()->dispatch_wf(0, 0x1000, 104, 256);
    Wavefront *reader = fixture.cu()->dispatch_wf(0, 0x2000, 104, 256);
    ASSERT_NE(writer, nullptr);
    ASSERT_NE(reader, nullptr);
    const uint32_t lanes = reader->wf_size();
    const uint64_t exec = lanes == 64 ? ~uint64_t{0} : (uint64_t{1} << lanes) - 1;
    constexpr uint32_t kStride = 128;
    writer->set_lds_size(kStride * lanes);
    reader->set_lds_size(kStride * lanes);
    writer->set_exec(1u);
    reader->set_exec(exec);
    std::array<amdgpu::Wavefront *, 2> waves{writer, reader};
    fixture.plugin_group_->onAmdgpuWorkgroupDispatched(
        /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/512,
        /*physical_sgpr_count=*/208, waves);

    constexpr uint32_t kRacingLane = 17;
    auto store_at = [&](uint32_t address) {
      auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
      state->elem_size = sizeof(uint32_t);
      state->num_elems = 1;
      state->is_load = false;
      state->wf_size = writer->wf_size();
      state->exec_mask = 1;
      state->lane_mask = 1;
      state->per_lane_addr[0] = address;
      TestMemoryInstruction store(std::move(state));
      writer->pc += 8;
      fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(writer->pc, store, *writer);
      fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, store, *writer);
    };
    store_at(kStride * kRacingLane - kStride / 2);
    if (include_racing_store)
      store_at(kStride * kRacingLane);
    ASSERT_EQ(sink.str().find("RACE "), std::string::npos) << sink.str();

    auto state = std::make_unique<VectorMemState>(LOCAL_MEM);
    state->elem_size = sizeof(uint32_t);
    state->num_elems = 1;
    state->is_load = true;
    state->wf_size = lanes;
    state->exec_mask = exec;
    state->lane_mask = exec;
    state->dst_reg_base = reader->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      state->per_lane_addr[lane] = kStride * lane;
    TestMemoryInstruction load(std::move(state));
    fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(reader->pc, load, *reader);
    fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *reader);

    const std::string output = sink.str();
    const std::string::size_type race = output.find("RACE ");
    if (!include_racing_store) {
      EXPECT_EQ(race, std::string::npos) << output;
      continue;
    }
    ASSERT_NE(race, std::string::npos);
    EXPECT_EQ(output.find("RACE ", race + 1), std::string::npos) << output;
    EXPECT_NE(output.find(std::format("access=read reg={} wave=1 lane={} ", kStride * kRacingLane,
                                      kRacingLane)),
              std::string::npos)
        << output;
  }
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
        ASSERT_EQ(test::ComputeUnitTestAccess::route_memory_inst(cu, std::move(inst.value()), wave),
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
      test::ComputeUnitTestAccess::route_memory_inst(*cu, std::move(store), *writer);
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
            state->wait_counter_type = WaitCounterType::DSCNT;
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
          test::ComputeUnitTestAccess::route_memory_inst(*cu, std::move(inst), *flat_wave);

          auto *plugin_state =
              static_cast<RaceWavefrontState *>(flat_wave->plugin_state(plugin_ptr->slot_index()));
          auto &events = plugin_state->race_state->getDetector()->events();
          const EventId flat_event{flat_first ? 0 : 1};
          ASSERT_EQ(events.totalAllocated(), flat_first ? 1 : 2);
          EXPECT_EQ(events.execMask(flat_event), 0x3u);
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
    const auto outcome =
        test::ComputeUnitTestAccess::route_memory_inst(*cu, std::move(inst), *wave);
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
    EXPECT_EQ(test::ComputeUnitTestAccess::route_memory_inst(*cu, std::move(inst), *flat_wave),
              dds ? VmAccessOutcome::Faulted : VmAccessOutcome::Complete);
    auto *state =
        static_cast<RaceWavefrontState *>(flat_wave->plugin_state(plugin_ptr->slot_index()));
    const auto &events = state->race_state->getDetector()->events();
    if (dds) {
      EXPECT_EQ(events.totalAllocated(), 0);
    } else {
      EXPECT_EQ(events.ldsIntervals(EventId{0}).getTotalBytes(), 8);
      EXPECT_TRUE(events.ldsIntervals(EventId{0}).contains(capacity - 1));
      EXPECT_FALSE(events.ldsIntervals(EventId{0}).contains(capacity));
    }
    auto probe = std::make_unique<VectorMemState>(LOCAL_MEM);
    probe->elem_size = 4;
    probe->num_elems = 1;
    probe->is_load = !load;
    probe->exec_mask = probe->lane_mask = 1;
    probe->wf_size = 32;
    probe->dst_reg_base = probe_wave->vgpr_alloc().base;
    probe->per_lane_addr[0] = capacity - 4;
    probe->wait_counter_type = WaitCounterType::DSCNT;
    TestMemoryInstruction probe_inst(std::move(probe));
    f.plugin_group_->onAmdgpuBeforeExecuteInstruction(probe_wave->pc, probe_inst, *probe_wave);
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, probe_inst, *probe_wave);
    EXPECT_EQ(sink.str().find("RACE ") != std::string::npos, !dds);
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
      test::ComputeUnitTestAccess::route_memory_inst(*cu, std::move(inst), *wave);
      auto *state = static_cast<RaceWavefrontState *>(wave->plugin_state(plugin_ptr->slot_index()));
      auto &race = *state->race_state;
      const auto &events = race.getDetector()->events();
      if (std::ranges::find(test.addresses, dds_address) != test.addresses.end()) {
        EXPECT_EQ(events.totalAllocated(), 0);
        continue;
      }
      // A gfx1250 FLAT takes a slot in both counters (CDNA5 ISA 5.7.1.3). When
      // its addresses use no global memory, that counter's slot is recorded
      // first, as an event with no registers or LDS bytes.
      const bool slot_event = !test.has_global;
      const EventId event{slot_event ? 1 : 0};
      ASSERT_EQ(events.totalAllocated(), slot_event ? 2 : 1);
      EXPECT_EQ(events.execMask(event), test.exec);
      EXPECT_EQ(events.ldsIntervals(event).getTotalBytes(), test.has_lds ? 4 : 0);
      EXPECT_EQ(events.ldsIntervals(event).contains(lds_base + 64), test.has_lds);
      EXPECT_EQ(events.registers(event).size(), load ? 1u : 0u);
      const auto global_counter = load ? WaitCounterType::LOADCNT : WaitCounterType::STORECNT;
      EXPECT_EQ(events.pendingCounterIncrement(event, global_counter), test.has_global ? 1 : 0);
      EXPECT_EQ(events.pendingCounterIncrement(event, WaitCounterType::DSCNT), 1);
      const EventId slot{0};
      if (slot_event) {
        EXPECT_EQ(events.registers(slot).size(), 0u);
        EXPECT_EQ(events.ldsIntervals(slot).getTotalBytes(), 0);
        EXPECT_EQ(events.pendingCounterIncrement(slot, global_counter), 1);
        EXPECT_EQ(events.pendingCounterIncrement(slot, WaitCounterType::DSCNT), 0);
      }
      EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
      PendingWaitCount wait;
      wait.add(global_counter, 0);
      race.dispatch(wait);
      if (slot_event) {
        EXPECT_EQ(events.status(slot), EventStatus::WAVE_COMPLETE);
      }
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

TEST(RaceDetectorPluginTest, NamedVmcntWaitRetiresMonolithicAndSplitLoadEvents) {
  auto run = [](WaitCounterType event_counter) {
    PluginFixture f;
    f.initialize(/*num_wf_slots=*/1, /*arch=*/"rdna3", /*wavefront_size=*/32,
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

    execute_wait(*f.plugin_group_, *wf, 0xbcfc0000u); // s_waitcnt_vmcnt null, 0.
    f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base, /*lane_mask=*/1);
    return sink.str().find("RACE ") == std::string::npos;
  };

  EXPECT_TRUE(run(WaitCounterType::VMCNT));
  EXPECT_TRUE(run(WaitCounterType::LOADCNT));
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

TEST(RaceDetectorPluginTest, FlatLoadReadyLanesKnownFalsePositives) {
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
          constexpr uint64_t shared_base = uint64_t{1} << 48;
          cu->set_apertures(shared_base, shared_base + UINT32_MAX, 0, 0);
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
          // TODO(newling): https://github.com/ROCm/rocm-systems/issues/12237
          // Expect only missing_wait here after the fix. A zero wait
          // suffices for the consumed lane group, but the plugin still reports
          // ready lanes for mixed CDNA4 and RDNA4 loads.
          // Track the producing counter separately for each consumed lane group.
          const bool known_false_positive =
              waited == Wait::Result && shared_lanes != 0 && shared_lanes != 3;
          EXPECT_EQ(sink.str().find("RACE ") != std::string::npos,
                    missing_wait || known_false_positive);
        }
      }
    }
  }
}

TEST(RaceDetectorPluginTest, FlatResultsFollowTheirResolvedCounter) {
  enum class Space { Global, Lds, Scratch, GlobalAndScratch };
  for (const std::string_view arch : {"cdna2", "cdna4", "rdna2", "rdna4", "cdna5"}) {
    const bool cdna_legacy = arch == "cdna2" || arch == "cdna4";
    const bool split = arch == "rdna4" || arch == "cdna5";
    const uint32_t wave_size = cdna_legacy ? 64 : 32;
    const uint32_t sgprs = cdna_legacy ? 104 : 128;
    for (const auto space : {Space::Global, Space::Lds, Space::Scratch, Space::GlobalAndScratch}) {
      // RDNA2's FLAT address calculator does not yet translate private pointers.
      if (arch == "rdna2" && (space == Space::Scratch || space == Space::GlobalAndScratch))
        continue;
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
            else if (arch == "rdna2")
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
            test::ComputeUnitTestAccess::route_memory_inst(*cu, std::move(load), *wf);
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
            const bool missing = exec && (space == Space::Lds ? lgkmcnt != 0 : vmcnt != 0);
            EXPECT_EQ(sink.str().find("RACE ") != std::string::npos, missing);
            EXPECT_EQ(!hazards.empty(), missing);
          }
        }
      }
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
    SameWave
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
      std::array<Wavefront *, 2> waves{cu->dispatch_wf(0, 0x8000, 106, 16, 32),
                                       cu->dispatch_wf(other ? 1 : 0, 0xa000, 106, 16, 32)};
      auto *writer = waves[0];
      const unsigned reader_index = read == Read::SameWave ? 0 : 1;
      auto *reader = waves[reader_index];
      ASSERT_NE(writer, nullptr);
      ASSERT_NE(reader, nullptr);
      writer->set_lds_base(256);
      reader->set_lds_base(other ? 512 : 256);
      writer->set_lds_size(16);
      reader->set_lds_size(read == Read::OutOfBounds ? 4 : 16);
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
      if (read == Read::Ds) {
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
        if (read == Read::Invalid)
          continue; // The reserved M0 type reports an execution error.
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
          constexpr uint64_t shared_base = uint64_t{1} << 32;
          f.cu()->set_apertures(shared_base, shared_base + UINT32_MAX, 0, 0);
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
  util::FailureOr<MarkedInstruction> conflict = findConflict(violation, detector);
  ASSERT_TRUE(conflict.succeeded());
  EXPECT_EQ(conflict.value().instruction, testInstruction(0x200));
}

TEST(FindConflictTest, RejectsUnavailableConflictingEvent) {
  RaceDetector detector(/*nWaves=*/
                        1, /*vgprCount=*/4, /*sgprCount=*/4, Dim3d(0), [](RaceViolation) {},
                        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4),
                        /*ldsSizeBytes=*/64 * 1024);
  RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, true, Dim3d(0), EventId{}};

  EXPECT_TRUE(findConflict(violation, detector).failed());
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
  ASSERT_TRUE(result.succeeded());
  EXPECT_NE(result.value().find("; <-- wave 3"), std::string::npos);
  EXPECT_NE(result.value().find("; <-- wave 0 lane 5"), std::string::npos);
  EXPECT_EQ(result.value(), "  ==>  0x100  ds_write_b32 v9, v12  ; <-- wave 3\n"
                            "       0x108  s_nop 0\n"
                            "  ==>  0x10c  ds_read_b32 v8, v9  ; <-- wave 0 lane 5\n");
  EXPECT_NE(result.value().find("0x108  s_nop 0"), std::string::npos);
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
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.value().substr(0, 5), "  ==>");
  EXPECT_EQ(result.value().find("0x100"), std::string::npos);
  EXPECT_EQ(result.value().find("0x104"), std::string::npos);
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
  ASSERT_TRUE(result.succeeded());
  EXPECT_NE(result.value().find("before trace window"), std::string::npos);
  EXPECT_NE(result.value().find("buffer_load_dwordx4"), std::string::npos);
  EXPECT_NE(result.value().find("; <-- wave 0"), std::string::npos);
  EXPECT_EQ(result.value().find("0x200  s_nop 0"), std::string::npos);
  EXPECT_NE(result.value().find("not recorded"), std::string::npos);
  EXPECT_NE(result.value().find("; <-- wave 0 lane 5"), std::string::npos);
}

TEST(FormatTraceTest, ResolvesOnlyPrintedInstructions) {
  RingBuffer<InstructionSnapshot, 256> trace;
  for (uint64_t pc = 0x100; pc <= 0x300; pc += 4)
    trace.push(testInstruction(pc));
  std::vector<uint64_t> requested;
  auto resolve = [&](const InstructionSnapshot &instruction) {
    requested.push_back(instruction.pc());
    return std::string("s_nop 0");
  };
  const auto report =
      formatTrace(trace, resolve, MarkedInstruction{trace[2], 0, -1}, {trace[128], 0, 0});
  ASSERT_TRUE(report.succeeded());
  ASSERT_EQ(requested.size(), 32u);
  EXPECT_EQ(requested.front(), 0x108u);
  EXPECT_EQ(requested.back(), 0x300u);
  EXPECT_NE(report.value().find("instructions elided"), std::string::npos);
}

TEST(FormatTraceTest, IdenticalEncodingsKeepPcBasedSelection) {
  auto trace = make_trace({0x100, 0x104, 0x108, 0x100, 0x104});
  auto resolve = [](const InstructionSnapshot &) { return std::string("s_nop 0"); };
  const auto report =
      formatTrace(trace, resolve, MarkedInstruction{trace[3], 0, -1}, {trace[4], 0, 0});
  ASSERT_TRUE(report.succeeded());
  EXPECT_EQ(report.value(), "  ==>  0x100  s_nop 0  ; <-- wave 0\n"
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
  ASSERT_TRUE(report.succeeded());
  EXPECT_EQ(report.value(), "  ==>  0x100  ds_read_b32 v8, v0  ; <-- wave 0\n"
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
  ASSERT_TRUE(report.succeeded());
  EXPECT_EQ(report.value(), "  ==>  0x100  s_nop 0  ; <-- wave 0\n"
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
  ASSERT_TRUE(report.succeeded());
  EXPECT_EQ(report.value(), "  ==>  0x100  s_nop 7  (before trace window)  ; <-- wave 1\n"
                            "       ... 1 instructions not recorded ...\n"
                            "  ==>  0x104  s_nop 0  ; <-- wave 0 lane 0\n");
}

TEST(FormatTraceTest, PropagatesDisassemblyFailures) {
  auto trace = make_trace({0x100, 0x104});
  auto fail = [](const InstructionSnapshot &) -> util::FailureOr<std::string> {
    return util::Result::failure();
  };
  EXPECT_TRUE(
      formatTrace(trace, fail, MarkedInstruction{trace[0], 1, -1}, {trace[1], 0, 0}).failed());
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
    util::FailureOr<std::string> text =
        wave->disassemble_instruction(state->trace[0].encoding(), state->trace[0].pc());
    ASSERT_TRUE(text.succeeded());
    EXPECT_EQ(text.value(), expected);
  }
}

TEST(RaceDisassemblyTest, CapturedEncodingNeedsNoCodeMapping) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(1));
  auto *wave = f.cu()->dispatch_wf(0, 0x1000, 104, 256);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(7);
  const std::array<uint32_t, 1> nop{S_NOP | 7};
  util::FailureOr<std::string> nop_text = wave->disassemble_instruction(nop, 0x1000);
  ASSERT_TRUE(nop_text.succeeded());
  EXPECT_EQ(nop_text.value(), "s_nop 7");
  const auto load = cdna4::build_ds(cdna4::kDsReadB32Ds);
  util::FailureOr<std::string> load_text = wave->disassemble_instruction(load, 0x1000);
  ASSERT_TRUE(load_text.succeeded());
  EXPECT_EQ(load_text.value(), "ds_read_b32 v0, v0");
}

TEST(RaceDisassemblyTest, RejectsMissingAndTruncatedEncodings) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(1));
  auto *wave = f.cu()->dispatch_wf(0, 0x1000, 104, 256);
  ASSERT_NE(wave, nullptr);
  const auto load = cdna4::build_ds(cdna4::kDsReadB32Ds);
  for (const auto encoding : {std::span<const uint32_t>{}, std::span(load).first(1)}) {
    util::StringDiagnostic diagnostic;
    util::FailureOr<std::string> text =
        wave->disassemble_instruction(encoding, 0x1000, diagnostic.emitter());
    EXPECT_TRUE(text.failed());
    EXPECT_NE(diagnostic.message().find("PC 0x1000"), std::string::npos);
    EXPECT_NE(diagnostic.message().find(encoding.empty() ? "empty" : "truncated"),
              std::string::npos);
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
        util::FailureOr<std::string> first_text =
            wave->disassemble_instruction(first, 0x540024b100);
        util::FailureOr<std::string> second_text =
            wave->disassemble_instruction(second, 0x100002a100);
        ASSERT_TRUE(first_text.succeeded());
        ASSERT_TRUE(second_text.succeeded());
        EXPECT_EQ(first_text.value(), "s_nop 7");
        EXPECT_EQ(second_text.value(), "s_nop 9");
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
  // A global load into LDS is an async load on gfx1250.
  TestMemoryInstruction dropped_inst(
      std::move(dropped), {{WaitCounterType::ASYNCCNT, MemoryCompletionClass::ASYNC_LOAD}});
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
  read->wait_counter_type = WaitCounterType::DSCNT;
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
  // A global load into LDS is an async load on gfx1250.
  TestMemoryInstruction dropped_inst(
      std::move(dropped), {{WaitCounterType::ASYNCCNT, MemoryCompletionClass::ASYNC_LOAD}});
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
  read->wait_counter_type = WaitCounterType::DSCNT;
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
  ignored->wait_counter_type = WaitCounterType::LOADCNT;
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
      test::ComputeUnitTestAccess::route_memory_inst(cu, std::move(inst), *writer);
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
      test::ComputeUnitTestAccess::route_memory_inst(*cu, std::move(load), *reader);
      EXPECT_EQ(sink.str().find("RACE ") != std::string::npos, store_address == 16);
    }
  }
}

} // namespace
