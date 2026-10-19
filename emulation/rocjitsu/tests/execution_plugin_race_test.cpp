// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin_race_test.cpp
/// @brief Execution plugin race detection and diagnostic output tests.

#include "execution_plugin_test_util.h"
#include "mma_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/vm/plugins/race_detector/plugin.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

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
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand_types.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
using namespace rocjitsu::plugins::race_detector;
using namespace rocjitsu::test::execution_plugin;
using mma_test::make_cdna4_mfma_scale_words;

class TestWaitcntInstruction : public Instruction {
public:
  explicit TestWaitcntInstruction(std::string_view mnemonic = "s_waitcnt")
      : Instruction(mnemonic, nullptr) {}
};

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
        counterCapacitiesForArch(wavefronts.front()->cu().arch()));
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

  void registerOutstandingLoad(uint32_t logical_reg, uint64_t exec_mask, uint8_t byte_mask = 0xF) {
    ASSERT_NE(state_, nullptr);
    state_->registerEvent(/*pc=*/0x1000, MemoryEventType::GLOBAL_TO_VGPR, {logical_reg}, exec_mask,
                          byte_mask);
  }

  std::vector<RaceViolation> violations;

private:
  std::unique_ptr<RaceDetector> detector_;
  amdgpu::Wavefront *wf_ = nullptr;
  WaveRaceState *state_ = nullptr;
};

class FormattedLoadRaceTest : public ::testing::Test {
protected:
  PluginFixture fixture{1, "rdna4", 32, 128};
  StringSink *sink = nullptr;
  Wavefront *wave = nullptr;
  RaceWavefrontState *plugin_state = nullptr;
  std::unique_ptr<Decoder> decoder;

  void SetUp() override {
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
    ASSERT_TRUE(cu->execute_instruction(load.get(), *wave).succeeded());
    ASSERT_NE(load->data(), nullptr);
    const auto &state = *load->data_as<VectorMemState>();
    EXPECT_EQ(state.buffer_components, components);
    EXPECT_EQ(state.num_elems, 1u);
    EXPECT_EQ(state.elem_size, format == 46 ? 4u : 16u);
    fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, *load, *wave);
  }

  void wait(uint32_t count) {
    wave->set_wait_target_loadcnt(count);
    TestWaitcntInstruction instruction("s_wait_loadcnt");
    fixture.plugin_group_->onAmdgpuAfterExecuteInstruction(wave->pc + 4, instruction, *wave);
  }

  bool probe(uint32_t destination, uint8_t byte_mask, bool write, uint64_t lanes = 1) {
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
    fixture = std::make_unique<PluginFixture>(/*num_wf_slots=*/1, GetParam().arch,
                                              /*wavefront_size=*/32, /*sgprs_per_wf=*/128);
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
    ASSERT_TRUE(fixture->cu()->execute_instruction(load.get(), *wave).succeeded());
    ASSERT_NE(load->data(), nullptr);
    wave->set_m0(~mask); // The plugin must use the issue-time snapshot.
    fixture->plugin_group_->onAmdgpuMemoryAccessRouted({}, *load, *wave);
  }

  void wait() {
    wave->set_wait_target_loadcnt(0);
    TestWaitcntInstruction load_wait("s_wait_loadcnt");
    fixture->plugin_group_->onAmdgpuAfterExecuteInstruction(wave->pc + 4, load_wait, *wave);
    wave->set_wait_target_dscnt(0);
    TestWaitcntInstruction ds_wait("s_wait_dscnt");
    fixture->plugin_group_->onAmdgpuAfterExecuteInstruction(wave->pc + 8, ds_wait, *wave);
  }

  bool probe(uint32_t destination, bool write, uint64_t lanes = 1) {
    wave->pc += 16;
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
      plugin_state->race_state->registerEvent(wave->pc, MemoryEventType::LDS_TO_VGPR, {32 + word},
                                              1, 0xf, WaitCounterType::DSCNT,
                                              MemoryOrderClass::LDS);
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
    plugin_state->race_state->registerEvent(wave->pc, MemoryEventType::LDS_TO_VGPR, {240}, 1, 0xf,
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

TEST(RaceDetectorPluginTest, D16LoadTracksFullDwordWhenSramEccEnabled) {
  auto opposite_half_read_reports_race = [](std::string_view arch, uint32_t wavefront_size,
                                            bool formatted) {
    PluginFixture f(/*num_wf_slots=*/1, arch, wavefront_size);
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

TEST(RaceDetectorPluginTest, InvalidVectorLoadDestinationIsRejectedBeforeWawChecks) {
  auto rejects_without_access = [](MemPipelineTag tag) {
    constexpr uint32_t kVgprsPerWave = 16;
    PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"rdna4", /*wavefront_size=*/32,
                    /*sgprs_per_wf=*/128, kVgprsPerWave);
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
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

    f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kVgprsPerWave - 1,
                                           /*lane_mask=*/1);
    return sink.str().find("RACE ") == std::string::npos;
  };

  EXPECT_TRUE(rejects_without_access(GLOBAL_MEM));
  EXPECT_TRUE(rejects_without_access(LOCAL_MEM));
}

TEST(RaceDetectorPluginTest, WideVectorLoadTracksEveryDestinationVgpr) {
  PluginFixture f(/*num_wf_slots=*/1);
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
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kDst + 1,
                                         /*lane_mask=*/1);
  EXPECT_NE(sink.str().find("RACE "), std::string::npos);
}

TEST(RaceDetectorPluginTest, DualOffsetLoadTracksBothDestinationRanges) {
  PluginFixture f(/*num_wf_slots=*/1);
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
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kSecondDst,
                                         /*lane_mask=*/1);
  EXPECT_NE(sink.str().find("RACE "), std::string::npos);
}

TEST(RaceDetectorPluginTest, DualOffsetStoreTracksSecondLdsRange) {
  PluginFixture f(/*num_wf_slots=*/2);
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
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *reader);

  EXPECT_NE(sink.str().find("RACE "), std::string::npos);
}

TEST(RaceDetectorPluginTest, DualOffsetAccessValidatesSecondLdsRange) {
  const auto second_range_reports_race = [](bool dual_is_load) {
    PluginFixture f(/*num_wf_slots=*/2);
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
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, dual_access, *second);

    return sink.str().find("RACE ") != std::string::npos;
  };

  EXPECT_TRUE(second_range_reports_race(/*dual_is_load=*/true));
  EXPECT_TRUE(second_range_reports_race(/*dual_is_load=*/false));
}

TEST(RaceDetectorPluginTest, LocalMemoryUsesEffectiveIssueMask) {
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"cdna5", /*wavefront_size=*/32,
                  /*sgprs_per_wf=*/128);
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
  PluginFixture f(/*num_wf_slots=*/1);
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
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

  auto *plugin_state =
      static_cast<RaceWavefrontState *>(wf->plugin_state(plugin_ptr->slot_index()));
  ASSERT_NE(plugin_state, nullptr);
  auto &events = plugin_state->race_state->getDetector()->events();
  ASSERT_EQ(events.totalAllocated(), 1);
  const EventId event{events.totalAllocated() - 1};
  EXPECT_EQ(events.memoryOrder(event), MemoryOrderClass::UNORDERED);
}

TEST(RaceDetectorPluginTest, LdsRoutedFlatStoreDwordx2TracksTrailingDword) {
  auto trailing_access_reports_race = [](bool flat_store_first) {
    PluginFixture f(/*num_wf_slots=*/2);
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

TEST(RaceDetectorPluginTest, Rdna4PartialLoadWaitRetiresOnlyOldestOrderedEvent) {
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"rdna4", /*wavefront_size=*/32,
                  /*sgprs_per_wf=*/128);
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
    EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, *load, *wf);
  }

  wf->set_wait_target_loadcnt(1);
  TestWaitcntInstruction wait("s_wait_loadcnt");
  f.plugin_group_->onAmdgpuAfterExecuteInstruction(/*pc=*/4, wait, *wf);

  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base, /*lane_mask=*/1);
  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + 1, /*lane_mask=*/1);
  EXPECT_NE(sink.str().find("RACE "), std::string::npos);
}

TEST(RaceDetectorPluginTest, Rdna4GenericFlatPartialWaitRetiresOldestEvent) {
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"rdna4", /*wavefront_size=*/32,
                  /*sgprs_per_wf=*/128);
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

  wf->set_wait_target_loadcnt_dscnt(/*loadcnt=*/1, /*dscnt=*/1);
  TestWaitcntInstruction wait("s_wait_loadcnt_dscnt");
  f.plugin_group_->onAmdgpuAfterExecuteInstruction(/*pc=*/4, wait, *wf);

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
    PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"rdna4", /*wavefront_size=*/32,
                    /*sgprs_per_wf=*/128);
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
        wf->set_wait_target_storecnt(0);
        TestWaitcntInstruction wait("s_wait_storecnt");
        f.plugin_group_->onAmdgpuAfterExecuteInstruction(/*pc=*/4, wait, *wf);
      } else {
        wf->set_wait_target_dscnt(0);
        TestWaitcntInstruction wait("s_wait_dscnt");
        f.plugin_group_->onAmdgpuAfterExecuteInstruction(/*pc=*/8, wait, *wf);
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

TEST(RaceDetectorPluginTest, ScalarLoadToTtmpReportsReadBeforeWait) {
  PluginFixture f(/*num_wf_slots=*/1);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
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
  EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
  ASSERT_NE(load->data(), nullptr);
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, *load, *wf);

  static_cast<void>(RegisterAccess(*wf).read_scalar(*load->dst_operand(0)));
  EXPECT_NE(sink.str().find("type=TTMP"), std::string::npos);
}

TEST(RaceDetectorPluginTest, ScalarLoadToTtmpReportsWriteBeforeWait) {
  PluginFixture f(/*num_wf_slots=*/1);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/104, waves);

  auto state = std::make_unique<ScalarMemState>();
  state->dst_register = {ScalarRegisterStorage::TTMP, 0, 1};
  state->is_load = true;
  TestMemoryInstruction load(std::move(state));
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

  RegisterAccess(*wf).write_ttmp(0, 0x12345678u);
  EXPECT_NE(sink.str().find("type=TTMP"), std::string::npos);
  EXPECT_NE(sink.str().find("access=write"), std::string::npos);
}

TEST(RaceDetectorPluginTest, ScalarLoadToTtmpHonorsSplitKmcntWait) {
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"rdna4", /*wavefront_size=*/32,
                  /*sgprs_per_wf=*/128);
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
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

  wf->set_wait_target_kmcnt(0);
  TestWaitcntInstruction wait("s_wait_kmcnt");
  f.plugin_group_->onAmdgpuAfterExecuteInstruction(/*pc=*/4, wait, *wf);
  static_cast<void>(RegisterAccess(*wf).read_ttmp(0));
  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

TEST(RaceDetectorPluginTest, NamedVmcntWaitRetiresMonolithicAndSplitLoadEvents) {
  auto run = [](WaitCounterType event_counter) {
    PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"rdna3", /*wavefront_size=*/32,
                    /*sgprs_per_wf=*/128);
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
    f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

    wf->set_wait_target_loadcnt(0);
    TestWaitcntInstruction wait("s_waitcnt_vmcnt");
    f.plugin_group_->onAmdgpuAfterExecuteInstruction(/*pc=*/4, wait, *wf);
    f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base, /*lane_mask=*/1);
    return sink.str().find("RACE ") == std::string::npos;
  };

  EXPECT_TRUE(run(WaitCounterType::VMCNT));
  EXPECT_TRUE(run(WaitCounterType::LOADCNT));
}

TEST(RaceDetectorPluginTest, UnresolvedOrMixedFlatWholeResultRequiresBothWaitCounterDomains) {
  auto reports_race_after_wait = [](MemoryRoute route, uint8_t vmcnt, uint8_t lgkmcnt) {
    PluginFixture f(/*num_wf_slots=*/1);
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

    auto state =
        std::make_unique<VectorMemState>(route == MemoryRoute::LOCAL ? LOCAL_MEM : GLOBAL_MEM);
    state->is_load = true;
    state->elem_size = 4;
    state->num_elems = 1;
    state->dst_reg_base = wf->vgpr_alloc().base;
    state->exec_mask = state->lane_mask = 3;
    state->wf_size = 64;
    TestMemoryInstruction flat(std::move(state),
                               {{WaitCounterType::VMCNT, MemoryCompletionClass::UNORDERED},
                                {WaitCounterType::LGKMCNT, MemoryCompletionClass::UNORDERED}});
    // Exercise the conservative fallback without executing unsupported mixed
    // LDS/global routing (#11456), including either choice of the first lane.
    MemoryAccessObservation access;
    if (route != MemoryRoute::UNKNOWN) {
      access.route = route;
      access.decoded_space = DecodedMemorySpace::FLAT;
      access.request_lane_mask = access.active_lane_mask = 3;
      access.flat_local_lane_mask = route == MemoryRoute::LOCAL ? 1 : 2;
    }
    f.plugin_group_->onAmdgpuMemoryAccessRouted(access, flat, *wf);

    wf->set_wait_target(vmcnt, lgkmcnt, WaitCounters::EXPCNT_MAX);
    TestWaitcntInstruction wait;
    f.plugin_group_->onAmdgpuAfterExecuteInstruction(/*pc=*/4, wait, *wf);
    f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base, /*lane_mask=*/3);
    return sink.str().find("RACE ") != std::string::npos;
  };

  for (const auto route : {MemoryRoute::UNKNOWN, MemoryRoute::GLOBAL, MemoryRoute::LOCAL}) {
    SCOPED_TRACE(static_cast<unsigned>(route));
    EXPECT_TRUE(reports_race_after_wait(route, /*vmcnt=*/0, /*lgkmcnt=*/15));
    EXPECT_TRUE(reports_race_after_wait(route, /*vmcnt=*/63, /*lgkmcnt=*/0));
    EXPECT_FALSE(reports_race_after_wait(route, /*vmcnt=*/0, /*lgkmcnt=*/0));
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
          PluginFixture f(1, is_rdna4 ? "rdna4" : "cdna4", wave_size, sgprs);
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

          // Feed the decoded load and resolved masks into both checkers without
          // executing the unsupported mixed LDS/global memory path:
          // https://github.com/ROCm/rocm-systems/issues/11456
          // The first requesting lane still selects the whole instruction's route.
          const bool local_route = shared_lanes & 1;
          auto data = std::make_unique<VectorMemState>(local_route ? LOCAL_MEM : GLOBAL_MEM);
          data->is_load = true;
          data->exec_mask = data->lane_mask = 3;
          data->wf_size = wave_size;
          data->elem_size = 4;
          data->num_elems = 1;
          data->dst_reg_base = wf->vgpr_alloc().base + 8;
          load->set_data(std::move(data));
          MemoryAccessObservation access;
          access.route = local_route ? MemoryRoute::LOCAL : MemoryRoute::GLOBAL;
          access.decoded_space = DecodedMemorySpace::FLAT;
          access.request_lane_mask = access.active_lane_mask = 3;
          access.flat_local_lane_mask = shared_lanes;
          f.plugin_group_->onAmdgpuMemoryAccessRouted(access, *load, *wf);
          constexpr uint64_t shared_base = uint64_t{1} << 32;
          cu->set_apertures(shared_base, shared_base + UINT32_MAX, 0, 0);
          for (unsigned lane = 0; lane < 2; ++lane) {
            cu->write_vgpr(wf->vgpr_alloc().base, lane, 0x100);
            cu->write_vgpr(wf->vgpr_alloc().base + 1, lane,
                           shared_lanes & (uint64_t{1} << lane) ? 1 : 2);
          }
          cu->track_memory_wait(*load, *wf);
          auto &core = wf->ensure_memory_wait_scoreboard();
          unsigned core_reports = 0;
          core.bind(0x200, &core_reports,
                    [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });

          const auto wait_zero = [&](bool lds) {
            if (is_rdna4) {
              if (lds)
                wf->set_wait_target_dscnt(0);
              else
                wf->set_wait_target_loadcnt(0);
              TestWaitcntInstruction wait(lds ? "s_wait_dscnt" : "s_wait_loadcnt");
              f.plugin_group_->onAmdgpuAfterExecuteInstruction(0x200, wait, *wf);
            } else {
              wf->set_wait_target(lds ? 63 : 0, lds ? 0 : 15, 7);
              TestWaitcntInstruction wait;
              f.plugin_group_->onAmdgpuAfterExecuteInstruction(0x200, wait, *wf);
            }
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
            PluginFixture f(1, std::string(arch), wave_size, sgprs);
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
                  const uint64_t physical = !scratch ? address + part * 4
                                            : arch == "rdna4"
                                                ? kScratch + lane * 256 + 0x20 + part * 4
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
              wf->set_wait_target_loadcnt(vmcnt);
              TestWaitcntInstruction vm_wait("s_wait_loadcnt");
              f.plugin_group_->onAmdgpuAfterExecuteInstruction(0x200, vm_wait, *wf);
              wf->set_wait_target_dscnt(lgkmcnt);
              TestWaitcntInstruction ds_wait("s_wait_dscnt");
              f.plugin_group_->onAmdgpuAfterExecuteInstruction(0x204, ds_wait, *wf);
            } else {
              wf->set_wait_target(vmcnt, lgkmcnt, 7);
              TestWaitcntInstruction wait;
              f.plugin_group_->onAmdgpuAfterExecuteInstruction(0x200, wait, *wf);
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

TEST(RaceDetectorPluginTest, Gfx950FlatEmptyPortionDoesNotOrderAnOlderResult) {
  for (const bool lds : {false, true}) {
    // Exercise both a partial wait and enough issues to trigger capacity
    // constraints if the checker retains empty FLAT portions.
    for (const int flat_count : {1, lds ? 64 : 16}) {
      SCOPED_TRACE(lds);
      SCOPED_TRACE(flat_count);
      PluginFixture f(/*num_wf_slots=*/1);
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
      race.registerEvent(0x80, lds ? MemoryEventType::GLOBAL_TO_VGPR : MemoryEventType::LDS_TO_VGPR,
                         {9}, 1, 0xF, empty_counter,
                         lds ? MemoryOrderClass::VMEM : MemoryOrderClass::LDS);
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
        wf->set_wait_target(vmcnt, lgkmcnt, 7);
        TestWaitcntInstruction wait;
        f.plugin_group_->onAmdgpuAfterExecuteInstruction(0x200, wait, *wf);
      };
      for (int i = 0; i < flat_count; ++i) {
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
          PluginFixture f(1);
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
            race.registerEvent(0x80 + i * 4,
                               full_lds ? MemoryEventType::LDS_TO_VGPR
                                        : MemoryEventType::GLOBAL_TO_VGPR,
                               {reg}, 1, 0xF);
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
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"rdna3", /*wavefront_size=*/32,
                  /*sgprs_per_wf=*/128);
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
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load, *wf);

  const auto current_wait = wf->wait_target();
  wf->set_wait_target(current_wait.vmcnt, 0, current_wait.expcnt);
  TestWaitcntInstruction wait("s_waitcnt_lgkmcnt");
  f.plugin_group_->onAmdgpuAfterExecuteInstruction(/*pc=*/4, wait, *wf);
  static_cast<void>(RegisterAccess(*wf).read_ttmp(0));
  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

TEST(ExecutionPluginTest, MfmaReadObservationReportsRace) {
  PluginFixture f(/*num_wf_slots=*/1);
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

    PluginFixture f(/*num_wf_slots=*/1);
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

TEST(FindConflictTest, UsesRecordedConflictingEvent) {
  RaceDetector detector(/*nWaves=*/
                        1, /*vgprCount=*/4, /*sgprCount=*/4, Dim3d(0), [](RaceViolation) {},
                        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4));
  EventId first = detector.allocateEventId(WaveId{0}, /*pc=*/0x100, MemoryEventType::GLOBAL_TO_VGPR,
                                           {2}, /*execMask=*/1);
  EventId second = detector.allocateEventId(WaveId{0}, /*pc=*/0x200,
                                            MemoryEventType::GLOBAL_TO_VGPR, {2}, /*execMask=*/1);
  ASSERT_NE(first, second);

  RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, true, Dim3d(0), second};
  MarkedPc conflict = findConflict(violation, detector);

  EXPECT_EQ(conflict.pc, 0x200u);
}

TEST(FindConflictTest, RejectsUnavailableConflictingEvent) {
  RaceDetector detector(/*nWaves=*/
                        1, /*vgprCount=*/4, /*sgprCount=*/4, Dim3d(0), [](RaceViolation) {},
                        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4));
  RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, true, Dim3d(0), EventId{}};

  EXPECT_THROW(findConflict(violation, detector), std::out_of_range);
}

TEST(DecorateExceptionTest, UsesRecordedConflictingEvent) {
  RaceDetector detector(/*nWaves=*/
                        1, /*vgprCount=*/4, /*sgprCount=*/4, Dim3d(0), [](RaceViolation) {},
                        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4));
  EventId first = detector.allocateEventId(WaveId{0}, /*pc=*/10, MemoryEventType::GLOBAL_TO_VGPR,
                                           {2}, /*execMask=*/1);
  EventId second = detector.allocateEventId(WaveId{0}, /*pc=*/20, MemoryEventType::GLOBAL_TO_VGPR,
                                            {2}, /*execMask=*/1);
  ASSERT_NE(first, second);

  RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, false, Dim3d(0), second};
  std::vector<std::string> source_lines(64, "instruction");
  std::string report = detector.decorateException(
      violation, /*wavePc=*/30, static_cast<int>(source_lines.size()),
      [&](int line) -> std::string_view { return source_lines.at(static_cast<size_t>(line)); });

  EXPECT_NE(report.find("20 --> |"), std::string::npos);
  EXPECT_NE(report.find("30 --> |"), std::string::npos);
  EXPECT_EQ(report.find("10 --> |"), std::string::npos);
}

auto make_trace(std::initializer_list<uint64_t> pcs) {
  plugins::race_detector::RingBuffer<uint64_t, 256> rb;
  for (auto pc : pcs)
    rb.push(pc);
  return rb;
}

TEST(FormatTraceTest, WaveLaneAnnotations) {
  auto trace = make_trace({0x100, 0x108, 0x10c});
  std::unordered_map<uint64_t, std::string> disasm = {
      {0x100, "ds_write_b32 v9, v12"},
      {0x108, "s_nop 0"},
      {0x10c, "ds_read_b32 v8, v9"},
  };
  plugins::race_detector::MarkedPc conflict{0x100, 3, -1};
  plugins::race_detector::MarkedPc read{0x10c, 0, 5};
  auto result = formatTrace(trace, disasm, conflict, read);
  EXPECT_NE(result.find("; <-- wave 3"), std::string::npos);
  EXPECT_NE(result.find("; <-- wave 0 lane 5"), std::string::npos);
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
  plugins::race_detector::MarkedPc conflict{0x108, 2, -1};
  plugins::race_detector::MarkedPc read{0x110, 1, 3};
  auto result = formatTrace(trace, disasm, conflict, read);
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
  plugins::race_detector::MarkedPc conflict{0x100, 3, -1};
  plugins::race_detector::MarkedPc read{0x208, 0, 5};
  auto result = formatTrace(trace, disasm, conflict, read);
  EXPECT_NE(result.find("before trace window"), std::string::npos);
  EXPECT_NE(result.find("buffer_load_dwordx4"), std::string::npos);
  EXPECT_NE(result.find("; <-- wave 3"), std::string::npos);
  EXPECT_NE(result.find("not recorded"), std::string::npos);
  EXPECT_NE(result.find("; <-- wave 0 lane 5"), std::string::npos);
}

TEST(DisasmCacheTest, HandlesNonMonotonicPcOrder) {
  plugins::race_detector::DisasmCache cache;
  Instruction high_instruction("s_nop 0", nullptr);
  Instruction low_instruction("s_endpgm", nullptr);
  cache.record(0x540024b100, high_instruction);
  cache.record(0x100002a100, low_instruction);

  auto disasm = cache.to_map();
  EXPECT_EQ(disasm.at(0x540024b100), "s_nop 0");
  EXPECT_EQ(disasm.at(0x100002a100), "s_endpgm");
}

TEST(DisasmCacheTest, DisassemblesOnlyFirstInstructionAtPc) {
  class ObservableInstruction final : public Instruction {
  public:
    ObservableInstruction() : Instruction("s_count", nullptr) {}
    bool was_disassembled() const { return !disassembly_.empty(); }
  };

  plugins::race_detector::DisasmCache cache;
  ObservableInstruction first;
  ObservableInstruction duplicate;

  // DisasmCache is keyed by the absolute instruction PC. The value itself is
  // arbitrary here; using the same synthetic PC proves that a newly decoded
  // instruction at an already-cached address is not disassembled again.
  constexpr uint64_t synthetic_pc = 0x100;
  cache.record(synthetic_pc, first);
  cache.record(synthetic_pc, duplicate);

  EXPECT_TRUE(first.was_disassembled());
  EXPECT_FALSE(duplicate.was_disassembled());
  EXPECT_EQ(cache.to_map().at(synthetic_pc), "s_count");
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
  PluginFixture f(/*num_wf_slots=*/2, /*arch=*/"cdna5", /*wavefront_size=*/32);
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
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, read_inst, *reader);

  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

TEST(RaceDetectorPluginTest, RejectedAsyncGlobalLaneDoesNotCreateLdsRace) {
  PluginFixture f(/*num_wf_slots=*/2, /*arch=*/"cdna5", /*wavefront_size=*/32);
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
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, read_inst, *reader);

  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

class IgnoredGlobalMemoryRaceTest : public ::testing::TestWithParam<AtomicOp> {};

TEST_P(IgnoredGlobalMemoryRaceTest, DoesNotCreateDestinationRace) {
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"cdna5", /*wavefront_size=*/32);
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
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, ignored_inst, *wf);

  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kDestinationVgpr,
                                         /*lane_mask=*/0x1u, /*byte_mask=*/0xFu);
  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

TEST_P(IgnoredGlobalMemoryRaceTest, DoesNotConsumeVmcntOrderingSlot) {
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"cdna5", /*wavefront_size=*/32);
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
  load->exec_mask = 0x1u;
  load->lane_mask = 0x1u;
  load->wf_size = wf->wf_size();
  load->dst_reg_base = wf->vgpr_alloc().base + kDestinationVgpr;
  TestMemoryInstruction load_inst(std::move(load));
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, load_inst, *wf);

  auto ignored = std::make_unique<VectorMemState>(GLOBAL_MEM);
  ignored->elem_size = 4;
  ignored->num_elems = 1;
  ignored->is_load = false;
  ignored->atomic_op = GetParam();
  ignored->exec_mask = 0;
  ignored->lane_mask = 0;
  ignored->wf_size = wf->wf_size();
  TestMemoryInstruction ignored_inst(std::move(ignored));
  f.plugin_group_->onAmdgpuMemoryAccessRouted({}, ignored_inst, *wf);

  wf->set_wait_target(/*vmcnt=*/1, /*lgkmcnt=*/0, /*expcnt=*/0);
  TestWaitcntInstruction waitcnt;
  f.plugin_group_->onAmdgpuAfterExecuteInstruction(wf->pc, waitcnt, *wf);

  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kDestinationVgpr,
                                         /*lane_mask=*/0x1u, /*byte_mask=*/0xFu);
  EXPECT_NE(sink.str().find("RACE "), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(LoadAndReturningAtomic, IgnoredGlobalMemoryRaceTest,
                         ::testing::Values(AtomicOp::NONE, AtomicOp::SWAP));

TEST(RaceDetectorPluginTest, DecodedDirectLdsLoadsDetectConflictingWaveStores) {
  for (auto arch : {"rdna3", "rdna3_5", "rdna4"}) {
    for (uint32_t store_address : {16u, 20u}) {
      SCOPED_TRACE(testing::Message() << arch << ',' << store_address);
      PluginFixture fixture(2, arch, 32);
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
      writer->set_exec(1);
      reader->set_exec(1);
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
      fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, store, *writer);
      const uint32_t word = 0xce100006;
      auto decoder = Decoder::create(cu->arch());
      std::unique_ptr<Instruction> load(decode_valid(*decoder, &word));
      ASSERT_NE(load, nullptr);
      ASSERT_TRUE(load->is_memory_op());
      ASSERT_TRUE(cu->execute_instruction(load.get(), *reader).succeeded());
      ASSERT_NE(load->data(), nullptr);
      test::ComputeUnitTestAccess::route_memory_inst(*cu, load.release(), *reader);
      EXPECT_EQ(sink.str().find("RACE ") != std::string::npos, store_address == 16);
    }
  }
}

} // namespace
