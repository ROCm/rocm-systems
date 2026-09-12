// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_access_test.cpp
/// @brief Tests for the AMDGPU instruction-facing register access facade.
///
/// @details These tests enforce the boundary beneath generated instructions:
/// observed read masks constrain returned data and observed write masks
/// constrain modified storage. End-to-end decoded instruction callbacks live
/// in execution_plugin_test.cpp.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_resolve.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/instruction_compute_unit_view.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "util/simd.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

template <typename T>
concept ExposesRawComputeUnit = requires(T &value) { value.raw_cu(); };

template <typename T>
concept ExposesRawVgprData = requires(T &value) { value.raw_vgpr_data(0); };

template <typename T>
concept ExposesReadRegionRegData = requires(const T &value) { value.reg_data(0); };

static_assert(!ExposesReadRegionRegData<RegisterAccess::VgprReadRegion>);

template <typename T>
concept ExposesUnobservedVgprWrite = requires(T &value) { value.write_vgpr_storage(0, 0, 0); };

template <typename T>
concept ExposesUnobservedSgprWrite = requires(T &value) { value.write_sgpr(0, 0); };

static_assert(!ExposesRawComputeUnit<InstructionComputeUnitView>);
static_assert(!ExposesRawComputeUnit<Wavefront>);
static_assert(!ExposesRawVgprData<InstructionComputeUnitView>);
static_assert(!ExposesUnobservedVgprWrite<InstructionComputeUnitView>);
static_assert(!ExposesUnobservedSgprWrite<InstructionComputeUnitView>);
static_assert(!std::is_move_constructible_v<ScopedOperandDelegate>);
static_assert(!std::is_move_assignable_v<ScopedOperandDelegate>);

constexpr uint32_t kSgprsPerWave = 104;
constexpr uint32_t kVgprsPerWave = 256;

struct ReadEvent {
  const Wavefront *wavefront = nullptr;
  uint32_t physical_reg = 0;
  uint64_t lane_mask = 0;
  uint8_t byte_mask = 0;
};

struct WriteEvent {
  const Wavefront *wavefront = nullptr;
  uint32_t physical_reg = 0;
  uint64_t lane_mask = 0;
  uint8_t byte_mask = 0;
};

class RecordingPlugin : public ExecutionPlugin {
public:
  RecordingPlugin() : ExecutionPlugin("register_access_recorder") {}

  void onAmdgpuReadVgprLanes(const Wavefront *wf, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask) override {
    reads.push_back({wf, physical_reg, lane_mask, byte_mask});
  }

  void onAmdgpuReadSgpr(const Wavefront *wf, uint32_t physical_reg) override {
    sgpr_reads.push_back(physical_reg);
    sgpr_read_wavefronts.push_back(wf);
  }

  void onAmdgpuReadScalarRegister(const Wavefront *wf, RegisterRef reg) override {
    ExecutionPlugin::onAmdgpuReadScalarRegister(wf, reg);
    scalar_reads.push_back(reg);
    scalar_read_wavefronts.push_back(wf);
  }

  void onAmdgpuWriteScalarRegister(const Wavefront *wf, RegisterRef reg) override {
    scalar_writes.push_back(reg);
    scalar_write_wavefronts.push_back(wf);
    if (wf && reg.cls == RegClass::SGPR) {
      for (uint32_t offset = 0; offset < reg.width; ++offset) {
        sgpr_writes.push_back(wf->sgpr_alloc().base + reg.index + offset);
        sgpr_write_wavefronts.push_back(wf);
      }
    }
  }

  void onAmdgpuWriteVgprLanes(const Wavefront *wf, uint32_t physical_reg, uint64_t lane_mask,
                              uint8_t byte_mask) override {
    writes.push_back({wf, physical_reg, lane_mask, byte_mask});
  }

  std::vector<ReadEvent> reads;
  std::vector<WriteEvent> writes;
  std::vector<uint32_t> sgpr_reads;
  std::vector<const Wavefront *> sgpr_read_wavefronts;
  std::vector<uint32_t> sgpr_writes;
  std::vector<const Wavefront *> sgpr_write_wavefronts;
  std::vector<RegisterRef> scalar_reads;
  std::vector<const Wavefront *> scalar_read_wavefronts;
  std::vector<RegisterRef> scalar_writes;
  std::vector<const Wavefront *> scalar_write_wavefronts;
};

struct Fixture {
  ScopedIsaExecutionBackend execution_backend_scope;
  GpuMemory gpu_mem{"register_access_mem"};
  L2Cache l2{"register_access_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  std::shared_ptr<ExecutionPluginGroup> plugin_group;
  RecordingPlugin *plugin = nullptr;
  Wavefront *wf = nullptr;

  explicit Fixture(rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA4,
                   uint32_t requested_sgprs = kSgprsPerWave, uint32_t wavefront_slots = 1,
                   uint32_t vgprs_per_wave = kVgprsPerWave, uint32_t wave_size = 0)
      : execution_backend_scope(arch == ROCJITSU_CODE_ARCH_CDNA5   ? &cdna5::execution_backend()
                                : arch == ROCJITSU_CODE_ARCH_RDNA4 ? &rdna4::execution_backend()
                                                                   : &cdna4::execution_backend()) {
    ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = wavefront_slots;
    cfg.sgprs_per_wf = kSgprsPerWave;
    cfg.vgprs_per_wf = vgprs_per_wave;
    cfg.lds_size_kb = 64;
    cu = ComputeUnitCore::create("register_access_cu", cfg, &gpu_mem, &l2);

    plugin_group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto recorder = std::make_unique<RecordingPlugin>();
    plugin = recorder.get();
    plugin_group->add(std::move(recorder));
    cu->set_plugin_group(plugin_group);

    wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, requested_sgprs, vgprs_per_wave, wave_size);
  }

  uint32_t sgpr_base() const { return wf->sgpr_alloc().base; }
  uint32_t vgpr_base() const { return wf->vgpr_alloc().base; }
};

TEST(RegisterAccessTest, ScopedOperandDelegateRestoresAfterException) {
  Operand source(32, 1);
  Operand previous(32, 2);
  Operand staged(32, 3);

  {
    ScopedOperandDelegate previous_binding(source, &previous);
    EXPECT_EQ(source.delegate(), &previous);

    EXPECT_THROW(
        {
          ScopedOperandDelegate binding(source, &staged);
          EXPECT_EQ(source.delegate(), &staged);
          throw std::runtime_error("test");
        },
        std::runtime_error);
    EXPECT_EQ(source.delegate(), &previous);
  }
  EXPECT_EQ(source.delegate(), nullptr);

  {
    ScopedOperandDelegate no_binding(source, nullptr);
    EXPECT_EQ(source.delegate(), nullptr);
  }
}

TEST(RegisterAccessTest, SdwaStageSourceObservesAndSelectsActiveBytes) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->set_exec(0b0101);

  constexpr uint32_t logical_vgpr = 6;
  const uint32_t physical_vgpr = fx.vgpr_base() + logical_vgpr;
  fx.cu->write_vgpr(physical_vgpr, 0, 0x11223380u);
  fx.cu->write_vgpr(physical_vgpr, 1, 0x44556681u);
  fx.cu->write_vgpr(physical_vgpr, 2, 0x7788997Fu);
  fx.cu->write_vgpr(physical_vgpr, 3, 0xAABBCC82u);

  cdna4::Operand source(32, cdna4::OperandType::OPR_SRC_VGPR, 256 + logical_vgpr);
  std::optional<StagedOperand> storage;
  sdwa::stage_source(source, sdwa::BYTE_0, /*sign_extend=*/true, /*negate=*/false,
                     /*absolute=*/false, sdwa::SourceModifierFormat::NONE, storage, *fx.wf);

  ASSERT_TRUE(storage.has_value());
  ASSERT_EQ(fx.plugin->reads.size(), 1u);
  EXPECT_EQ(fx.plugin->reads[0].physical_reg, physical_vgpr);
  EXPECT_EQ(fx.plugin->reads[0].lane_mask, 0b0101u);
  EXPECT_EQ(fx.plugin->reads[0].byte_mask, 0b0001u);

  auto staged = RegisterAccess(*fx.wf).read_operand(*storage, 0b1111);
  EXPECT_EQ(staged.lane(0), 0xFFFFFF80u);
  EXPECT_EQ(staged.lane(1), 0u);
  EXPECT_EQ(staged.lane(2), 0x0000007Fu);
  EXPECT_EQ(staged.lane(3), 0u);

  sdwa::stage_source(source, sdwa::DWORD, /*sign_extend=*/false, /*negate=*/false,
                     /*absolute=*/false, sdwa::SourceModifierFormat::NONE, storage, *fx.wf);
  EXPECT_FALSE(storage.has_value());
}

TEST(RegisterAccessTest, SdwaStageSourceAppliesModifiersInSemanticFloatWidth) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->set_exec(1);

  constexpr uint32_t logical_vgpr = 6;
  const uint32_t physical_vgpr = fx.vgpr_base() + logical_vgpr;
  cdna4::Operand source(32, cdna4::OperandType::OPR_SRC_VGPR, 256 + logical_vgpr);
  std::optional<StagedOperand> storage;

  fx.cu->write_vgpr(physical_vgpr, 0, std::bit_cast<uint32_t>(-2.0f));
  sdwa::stage_source(source, sdwa::DWORD, /*sign_extend=*/false, /*negate=*/false,
                     /*absolute=*/true, sdwa::SourceModifierFormat::F32, storage, *fx.wf);
  ASSERT_TRUE(storage.has_value());
  EXPECT_EQ(RegisterAccess(*fx.wf).read_operand(*storage, 1).lane(0),
            std::bit_cast<uint32_t>(2.0f));
  ASSERT_EQ(fx.plugin->reads.size(), 1u);
  EXPECT_EQ(fx.plugin->reads[0].byte_mask, ExecutionPlugin::kFullByteMask);

  fx.plugin->reads.clear();
  sdwa::stage_source(source, sdwa::DWORD, /*sign_extend=*/false, /*negate=*/true,
                     /*absolute=*/true, sdwa::SourceModifierFormat::F32, storage, *fx.wf);
  ASSERT_TRUE(storage.has_value());
  EXPECT_EQ(RegisterAccess(*fx.wf).read_operand(*storage, 1).lane(0),
            std::bit_cast<uint32_t>(-2.0f));

  fx.plugin->reads.clear();
  fx.cu->write_vgpr(physical_vgpr, 0, 0xC000'BEEFu);
  sdwa::stage_source(source, sdwa::WORD_1, /*sign_extend=*/false, /*negate=*/false,
                     /*absolute=*/true, sdwa::SourceModifierFormat::F16, storage, *fx.wf);
  ASSERT_TRUE(storage.has_value());
  EXPECT_EQ(RegisterAccess(*fx.wf).read_operand(*storage, 1).lane(0), 0x4000u);
  ASSERT_EQ(fx.plugin->reads.size(), 1u);
  EXPECT_EQ(fx.plugin->reads[0].byte_mask, ExecutionPlugin::kHighHalfByteMask);

  fx.plugin->reads.clear();
  fx.cu->write_vgpr(physical_vgpr, 0, 0x4000'BEEFu);
  sdwa::stage_source(source, sdwa::WORD_1, /*sign_extend=*/false, /*negate=*/true,
                     /*absolute=*/false, sdwa::SourceModifierFormat::F16, storage, *fx.wf);
  ASSERT_TRUE(storage.has_value());
  EXPECT_EQ(RegisterAccess(*fx.wf).read_operand(*storage, 1).lane(0), 0xC000u);

  fx.plugin->reads.clear();
  sdwa::stage_source(source, sdwa::WORD_1, /*sign_extend=*/false, /*negate=*/true,
                     /*absolute=*/true, sdwa::SourceModifierFormat::NONE, storage, *fx.wf);
  ASSERT_TRUE(storage.has_value());
  EXPECT_EQ(RegisterAccess(*fx.wf).read_operand(*storage, 1).lane(0), 0x4000u);
}

TEST(RegisterAccessTest, ReadRegionObservesAllRegistersAndReturnsLaneSpans) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t base = fx.vgpr_base();
  fx.cu->write_vgpr(base + 3, 0, 0x1111u);
  fx.cu->write_vgpr(base + 3, 5, 0x3333u);
  fx.cu->write_vgpr(base + 4, 0, 0x2222u);
  fx.cu->write_vgpr(base + 4, 5, 0x4444u);

  RegisterAccess regs(*fx.cu);
  auto region = regs.read_vgpr_region(base + 3, /*reg_count=*/2, /*lane_mask=*/0x21,
                                      /*byte_mask=*/0xF);

  ASSERT_EQ(fx.plugin->reads.size(), 2u);
  EXPECT_EQ(fx.plugin->reads[0].physical_reg, base + 3);
  EXPECT_EQ(fx.plugin->reads[0].lane_mask, 0x21u);
  EXPECT_EQ(fx.plugin->reads[0].byte_mask, 0xFu);
  EXPECT_EQ(fx.plugin->reads[1].physical_reg, base + 4);
  EXPECT_EQ(fx.plugin->reads[1].lane_mask, 0x21u);
  EXPECT_EQ(fx.plugin->reads[1].byte_mask, 0xFu);

  EXPECT_EQ(region.lanes(0)[0], 0x1111u);
  EXPECT_EQ(region.lanes(0)[5], 0x3333u);
  EXPECT_EQ(region.lanes(1)[0], 0x2222u);
  EXPECT_EQ(region.lanes(1)[5], 0x4444u);
}

TEST(RegisterAccessTest, ReadRegionCopiesDwordsToLaneMajorStorage) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  const uint32_t base = fx.vgpr_base() + 3;
  constexpr uint64_t lane_mask = (uint64_t{1} << 1) | (uint64_t{1} << 5);
  for (uint32_t reg = 0; reg < 4; ++reg) {
    fx.cu->write_vgpr(base + reg, 1, 0x1000u + reg);
    fx.cu->write_vgpr(base + reg, 5, 0x5000u + reg);
  }

  std::array<uint8_t, 64 * 4 * sizeof(uint32_t)> bytes{};
  bytes.fill(0xAA);
  auto region = RegisterAccess(*fx.wf).read_vgpr_region(base, 4, lane_mask);
  region.copy_dwords_lane_major(bytes.data(), lane_mask);

  ASSERT_EQ(fx.plugin->reads.size(), 4u);
  for (uint32_t reg = 0; reg < 4; ++reg) {
    EXPECT_EQ(fx.plugin->reads[reg].physical_reg, base + reg);
    EXPECT_EQ(fx.plugin->reads[reg].lane_mask, lane_mask);
    uint32_t lane1 = 0;
    uint32_t lane5 = 0;
    std::memcpy(&lane1, bytes.data() + (1 * 4 + reg) * sizeof(uint32_t), sizeof(lane1));
    std::memcpy(&lane5, bytes.data() + (5 * 4 + reg) * sizeof(uint32_t), sizeof(lane5));
    EXPECT_EQ(lane1, 0x1000u + reg);
    EXPECT_EQ(lane5, 0x5000u + reg);
  }
  EXPECT_EQ(bytes[0], 0xAA);
}

TEST(RegisterAccessTest, ReadRegionTraversesAndCopiesLogicalRegisterRange) {
  for (const auto arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    Fixture fx(arch);
    ASSERT_NE(fx.wf, nullptr);
    SCOPED_TRACE(arch);

    const uint32_t wf_size = fx.wf->wf_size();
    constexpr uint32_t reg_count = 64;
    const uint32_t physical_base = fx.vgpr_base() + 7;
    fx.cu->write_vgpr(physical_base, 3, 0x11112222u);
    fx.cu->write_vgpr(physical_base + reg_count / 2, 3, 0x33334444u);
    fx.cu->write_vgpr(physical_base + reg_count - 1, 3, 0x55556666u);

    RegisterAccess regs(*fx.cu);
    auto region = regs.read_vgpr_region(physical_base, reg_count,
                                        /*lane_mask=*/uint64_t{1} << 3);

    std::vector<uint32_t> copied(static_cast<size_t>(reg_count) * wf_size);
    region.copy_to(copied);
    EXPECT_EQ(copied[3], 0x11112222u);
    EXPECT_EQ(copied[(reg_count / 2) * wf_size + 3], 0x33334444u);
    EXPECT_EQ(copied[(reg_count - 1) * wf_size + 3], 0x55556666u);

    uint32_t relative_reg = 0;
    const auto visitor = [&](std::span<const uint32_t> lanes) {
      EXPECT_EQ(lanes[3], copied[static_cast<size_t>(relative_reg) * wf_size + 3]);
      ++relative_reg;
    };
    region.for_each(visitor);
    EXPECT_EQ(relative_reg, reg_count);
  }
}

TEST(RegisterAccessTest, PartialByteReadRegionMasksReturnedValues) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t reg = fx.vgpr_base() + 5;
  fx.cu->write_vgpr(reg, 3, 0xAABBCCDDu);

  RegisterAccess regs(*fx.cu);
  auto region = regs.read_vgpr_region(reg, /*reg_count=*/1, /*lane_mask=*/1u << 3,
                                      /*byte_mask=*/0b0110);

  ASSERT_EQ(fx.plugin->reads.size(), 1u);
  EXPECT_EQ(fx.plugin->reads[0].byte_mask, 0b0110);
  EXPECT_EQ(region.lane(/*relative_reg=*/0, /*lane=*/3), 0x00BBCC00u);
  EXPECT_THROW((void)region.lanes(), std::logic_error);
}

TEST(RegisterAccessTest, WriteRegionObservesWritesAndHonorsLaneMask) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t base = fx.vgpr_base();
  for (uint32_t lane = 0; lane < fx.wf->wf_size(); ++lane)
    fx.cu->write_vgpr(base + 7, lane, 0xAAAA0000u | lane);

  RegisterAccess regs(*fx.cu);
  auto region = regs.write_vgpr_region(base + 7, /*reg_count=*/1, /*lane_mask=*/0x5);
  region.set_lane(/*relative_reg=*/0, /*lane=*/0, 0x100u);
  region.set_lane(/*relative_reg=*/0, /*lane=*/1, 0x200u);
  region.set_lane(/*relative_reg=*/0, /*lane=*/2, 0x300u);

  EXPECT_TRUE(fx.plugin->reads.empty());
  ASSERT_EQ(fx.plugin->writes.size(), 1u);
  EXPECT_EQ(fx.plugin->writes[0].physical_reg, base + 7);
  EXPECT_EQ(fx.plugin->writes[0].lane_mask, 0x5u);
  EXPECT_EQ(fx.plugin->writes[0].byte_mask, 0xFu);
  EXPECT_EQ(fx.cu->read_vgpr(base + 7, 0), 0x100u);
  EXPECT_EQ(fx.cu->read_vgpr(base + 7, 1), 0xAAAA0001u);
  EXPECT_EQ(fx.cu->read_vgpr(base + 7, 2), 0x300u);
}

TEST(RegisterAccessTest, WriteRegionStoresOnlyObservedBytes) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t reg = fx.vgpr_base() + 8;
  fx.cu->write_vgpr(reg, 3, 0xAABBCCDDu);

  RegisterAccess regs(*fx.cu);
  auto region = regs.write_vgpr_region(reg, /*reg_count=*/1, /*lane_mask=*/1u << 3,
                                       /*byte_mask=*/0b0110);
  region.set_lane(/*relative_reg=*/0, /*lane=*/3, 0x11223344u);

  EXPECT_TRUE(fx.plugin->reads.empty());
  ASSERT_EQ(fx.plugin->writes.size(), 1u);
  EXPECT_EQ(fx.plugin->writes[0].lane_mask, 1u << 3);
  EXPECT_EQ(fx.plugin->writes[0].byte_mask, 0b0110);
  EXPECT_EQ(fx.cu->read_vgpr(reg, 3), 0xAA2233DDu);
}

TEST(RegisterAccessTest, ReadWriteRegionObservesThenAllowsWrites) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t base = fx.vgpr_base();
  fx.cu->write_vgpr(base + 9, 3, 0x1234u);

  RegisterAccess regs(*fx.cu);
  auto region = regs.readwrite_vgpr_region(base + 9, /*reg_count=*/1, /*lane_mask=*/0x8);

  ASSERT_EQ(fx.plugin->reads.size(), 1u);
  EXPECT_EQ(fx.plugin->reads[0].physical_reg, base + 9);
  EXPECT_EQ(fx.plugin->reads[0].lane_mask, 0x8u);
  ASSERT_EQ(fx.plugin->writes.size(), 1u);
  EXPECT_EQ(fx.plugin->writes[0].physical_reg, base + 9);
  EXPECT_EQ(fx.plugin->writes[0].lane_mask, 0x8u);
  EXPECT_EQ(region.read_lanes(0)[3], 0x1234u);

  region.write().set_lane(/*relative_reg=*/0, /*lane=*/3, 0x5678u);
  EXPECT_EQ(region.read_lanes(0)[3], 0x5678u);
}

TEST(RegisterAccessTest, MaskedLaneWritePreservesBytesWithoutSyntheticRead) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  const uint32_t reg = fx.vgpr_base() + 11;
  fx.cu->write_vgpr(reg, 3, 0xAABBCCDDu);

  cdna4::Operand destination(32, cdna4::OperandType::OPR_VGPR, 11);
  RegisterAccess(*fx.wf).write_lane_masked(destination, /*lane=*/3, /*value=*/0x00003300u,
                                           /*update_byte_mask=*/0b0010,
                                           /*observed_byte_mask=*/0b0010,
                                           /*post_transform=*/nullptr);

  EXPECT_TRUE(fx.plugin->reads.empty());
  ASSERT_EQ(fx.plugin->writes.size(), 1u);
  EXPECT_EQ(fx.plugin->writes[0].physical_reg, reg);
  EXPECT_EQ(fx.plugin->writes[0].lane_mask, 1u << 3);
  EXPECT_EQ(fx.plugin->writes[0].byte_mask, 0b0010);
  EXPECT_EQ(fx.cu->read_vgpr_storage(reg, 3), 0xAABB33DDu);
}

TEST(RegisterAccessTest, PartialByteReadWriteRegionUsesDeclaredReadAndWriteMasks) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t reg = fx.vgpr_base() + 10;
  fx.cu->write_vgpr(reg, 3, 0xAABBCCDDu);

  RegisterAccess regs(*fx.cu);
  auto symmetric =
      regs.readwrite_vgpr_region(reg, /*reg_count=*/1, /*lane_mask=*/1u << 3, /*byte_mask=*/0b0010);
  EXPECT_EQ(symmetric.read().lane(0, 3), 0x0000CC00u);
  symmetric.write().set_lane(0, 3, 0x11223344u);
  EXPECT_EQ(fx.cu->read_vgpr_storage(reg, 3), 0xAABB33DDu);
  ASSERT_EQ(fx.plugin->reads.size(), 1u);
  EXPECT_EQ(fx.plugin->reads[0].byte_mask, 0b0010);
  ASSERT_EQ(fx.plugin->writes.size(), 1u);
  EXPECT_EQ(fx.plugin->writes[0].byte_mask, 0b0010);

  fx.plugin->reads.clear();
  fx.plugin->writes.clear();
  fx.cu->write_vgpr(reg, 3, 0xAABBCCDDu);
  auto asymmetric = regs.readwrite_vgpr_region(reg, /*reg_count=*/1, /*lane_mask=*/1u << 3,
                                               /*read_byte_mask=*/0b0010,
                                               /*write_byte_mask=*/0b1100);
  EXPECT_EQ(asymmetric.read().lane(0, 3), 0x0000CC00u);
  asymmetric.write().set_lane(0, 3, 0x11223344u);
  EXPECT_EQ(fx.cu->read_vgpr_storage(reg, 3), 0x1122CCDDu);
  ASSERT_EQ(fx.plugin->reads.size(), 1u);
  EXPECT_EQ(fx.plugin->reads[0].byte_mask, 0b0010);
  ASSERT_EQ(fx.plugin->writes.size(), 1u);
  EXPECT_EQ(fx.plugin->writes[0].byte_mask, 0b1100);
}

TEST(RegisterAccessTest, Scalar64ReadObservesBothRegisters) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t base = fx.vgpr_base();
  fx.cu->write_vgpr(base + 11, 6, 0x89ABCDEFu);
  fx.cu->write_vgpr(base + 12, 6, 0x01234567u);

  RegisterAccess regs(*fx.cu);
  EXPECT_EQ(regs.read_vgpr64(base + 11, 6), 0x0123456789ABCDEFull);

  ASSERT_EQ(fx.plugin->reads.size(), 2u);
  EXPECT_EQ(fx.plugin->reads[0].wavefront, fx.wf);
  EXPECT_EQ(fx.plugin->reads[0].physical_reg, base + 11);
  EXPECT_EQ(fx.plugin->reads[0].lane_mask, 1ULL << 6);
  EXPECT_EQ(fx.plugin->reads[1].wavefront, fx.wf);
  EXPECT_EQ(fx.plugin->reads[1].physical_reg, base + 12);
  EXPECT_EQ(fx.plugin->reads[1].lane_mask, 1ULL << 6);
}

TEST(RegisterAccessTest, Sgpr64ReadObservesBothRegisters) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t base = fx.sgpr_base();
  fx.cu->write_sgpr(base + 17, 0x89ABCDEFu);
  fx.cu->write_sgpr(base + 18, 0x01234567u);

  RegisterAccess regs(*fx.wf);
  EXPECT_EQ(regs.read_sgpr64(base + 17), 0x0123456789ABCDEFull);

  ASSERT_EQ(fx.plugin->sgpr_reads.size(), 2u);
  EXPECT_EQ(fx.plugin->sgpr_reads[0], base + 17);
  EXPECT_EQ(fx.plugin->sgpr_reads[1], base + 18);
  EXPECT_EQ(fx.plugin->scalar_reads, (std::vector<RegisterRef>{{RegClass::SGPR, 17, 2}}));

  regs.write_sgpr64(base + 19, 0xABCDEF0123456789ull);
  ASSERT_EQ(fx.plugin->sgpr_writes.size(), 2u);
  EXPECT_EQ(fx.plugin->sgpr_writes[0], base + 19);
  EXPECT_EQ(fx.plugin->sgpr_writes[1], base + 20);
  EXPECT_EQ(fx.plugin->scalar_writes, (std::vector<RegisterRef>{{RegClass::SGPR, 19, 2}}));
  fx.plugin->sgpr_reads.clear();
  EXPECT_EQ(fx.cu->read_sgpr(base + 19), 0x23456789u);
  EXPECT_EQ(fx.cu->read_sgpr(base + 20), 0xABCDEF01u);
}

TEST(ScalarOperandSelectorsTest, ClassifiesRegisterPairLowWords) {
  EXPECT_TRUE(is_src_scalar_register_pair(0));
  EXPECT_TRUE(is_src_scalar_register_pair(kVccSelectorFirst));
  EXPECT_TRUE(is_src_scalar_register_pair(kTtmpSelectorFirst));
  EXPECT_TRUE(is_src_scalar_register_pair(kExecSelectorFirst));
  EXPECT_TRUE(is_src_scalar_register_pair(kFlatScratchBaseSelectorFirst));

  EXPECT_FALSE(is_src_scalar_register_pair(kVccSelectorLast));
  EXPECT_FALSE(is_src_scalar_register_pair(kLegacyM0Selector));
  EXPECT_FALSE(is_src_scalar_register_pair(128));
  EXPECT_FALSE(is_src_scalar_register_pair(kFlatScratchBaseSelectorLast));
  EXPECT_FALSE(is_src_scalar_register_pair(242));
}

TEST(ScalarOperandSelectorsTest, SelectsM0AndNullLayoutByArchitecture) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    EXPECT_EQ(scalar_m0_selector(arch), kLegacyM0Selector);
    EXPECT_FALSE(scalar_selector_is_null(arch, kLegacyM0Selector));
    EXPECT_FALSE(scalar_selector_is_null(arch, kGfx10NullSelector));
  }

  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    EXPECT_EQ(scalar_m0_selector(arch), kLegacyM0Selector);
    EXPECT_TRUE(scalar_selector_is_null(arch, kGfx10NullSelector));
  }

  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                              ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    EXPECT_EQ(scalar_m0_selector(arch), kModernM0Selector);
    EXPECT_TRUE(scalar_selector_is_null(arch, kModernNullSelector));
  }
}

TEST(ScalarOperandSelectorsTest, RejectsRangesThatCrossArchitecturalRegisterFiles) {
  Fixture fx(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(fx.wf, nullptr);

  auto ttmps = resolve_scalar_register_range(*fx.wf, kTtmpSelectorFirst, kTtmpRegisterCount);
  ASSERT_TRUE(ttmps.has_value());
  EXPECT_EQ(ttmps->storage, ScalarRegisterStorage::TTMP);
  EXPECT_EQ(ttmps->index, 0u);
  EXPECT_EQ(ttmps->width, kTtmpRegisterCount);

  EXPECT_FALSE(resolve_scalar_register_range(*fx.wf, kTtmpSelectorLast, 2).has_value());
  EXPECT_FALSE(resolve_scalar_register_range(*fx.wf, kScalarSgprSelectorLast, 2).has_value());

  auto vcc = resolve_scalar_register_range(*fx.wf, kVccSelectorFirst, 2);
  ASSERT_TRUE(vcc.has_value());
  EXPECT_EQ(vcc->storage, ScalarRegisterStorage::VCC);
  EXPECT_EQ(vcc->index, 0u);
  EXPECT_EQ(vcc->width, 2u);
}

TEST(ScalarOperandSelectorsTest, ResolvesSpecialScalarStorageByArchitecture) {
  Fixture cdna4_fx(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(cdna4_fx.wf, nullptr);
  cdna4_fx.wf->set_scratch_base(0x2222222211111111ull);
  cdna4_fx.wf->set_vcc_raw(0x4444444433333333ull);
  cdna4_fx.wf->set_m0(0x55555555u);

  EXPECT_EQ(read_scalar_selector64(*cdna4_fx.wf, kFlatScratchSelectorFirst), 0x2222222211111111ull);
  EXPECT_EQ(read_scalar_selector64(*cdna4_fx.wf, kVccSelectorFirst), 0x4444444433333333ull);
  EXPECT_EQ(read_scalar_selector(*cdna4_fx.wf, kLegacyM0Selector), 0x55555555u);

  Fixture rdna4_fx(ROCJITSU_CODE_ARCH_RDNA4, kSgprsPerWave, 1, kVgprsPerWave, 32);
  ASSERT_NE(rdna4_fx.wf, nullptr);
  auto null_range =
      resolve_scalar_register_range(*rdna4_fx.wf, kModernNullSelector, kTtmpRegisterCount);
  ASSERT_TRUE(null_range.has_value());
  EXPECT_EQ(null_range->storage, ScalarRegisterStorage::DISCARD);
  EXPECT_EQ(read_scalar_register(*rdna4_fx.wf, *null_range, 15), 0u);
}

TEST(RegisterAccessTest, ScalarSelectedLaneReadRejectsAnotherWaveBlock) {
  constexpr uint32_t kSmallVgprBlock = 16;
  Fixture fx(ROCJITSU_CODE_ARCH_RDNA4, kSgprsPerWave, /*wavefront_slots=*/2, kSmallVgprBlock,
             /*wave_size=*/32);
  ASSERT_NE(fx.wf, nullptr);
  auto *adjacent_wave =
      fx.cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0, kSgprsPerWave, kSmallVgprBlock, 32);
  ASSERT_NE(adjacent_wave, nullptr);
  ASSERT_EQ(adjacent_wave->vgpr_alloc().base, fx.vgpr_base() + kSmallVgprBlock);

  constexpr uint32_t kSentinel = 0x12345678u;
  fx.cu->write_vgpr(adjacent_wave->vgpr_alloc().base, 3, kSentinel);
  rdna4::Operand adjacent_vgpr(32, rdna4::OperandType::OPR_VGPR, kSmallVgprBlock);

  EXPECT_EQ(RegisterAccess(*fx.wf).read_scalar_selected_lane(adjacent_vgpr, 3), 0u);
  EXPECT_TRUE(fx.plugin->reads.empty());
  EXPECT_EQ(fx.cu->read_vgpr_storage(adjacent_wave->vgpr_alloc().base, 3), kSentinel);
}

TEST(RegisterAccessTest, LanePair32PreservesRegisterPairsAndSplatsSingleWords) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  RegisterAccess regs(*fx.wf);

  const uint32_t sgpr_base = fx.sgpr_base();
  fx.cu->write_sgpr(sgpr_base + 8, 0x11111111u);
  fx.cu->write_sgpr(sgpr_base + 9, 0x22222222u);
  cdna4::Operand sgpr_pair(64, cdna4::OperandType::OPR_SRC_SIMPLE, 8);
  OperandPair32 pair = regs.read_lane_pair32(sgpr_pair, 0);
  EXPECT_EQ(pair.lo, 0x11111111u);
  EXPECT_EQ(pair.hi, 0x22222222u);

  fx.wf->set_vcc_raw(0x4444444433333333ull);
  cdna4::Operand vcc_pair(64, cdna4::OperandType::OPR_SRC_SIMPLE, 106);
  pair = regs.read_lane_pair32(vcc_pair, 0);
  EXPECT_EQ(pair.lo, 0x33333333u);
  EXPECT_EQ(pair.hi, 0x44444444u);

  fx.wf->set_ttmp(0, 0x55555555u);
  fx.wf->set_ttmp(1, 0x66666666u);
  cdna4::Operand ttmp_pair(64, cdna4::OperandType::OPR_SRC_SIMPLE, 108);
  pair = regs.read_lane_pair32(ttmp_pair, 0);
  EXPECT_EQ(pair.lo, 0x55555555u);
  EXPECT_EQ(pair.hi, 0x66666666u);

  cdna4::Operand inline_one(64, cdna4::OperandType::OPR_SRC_SIMPLE, 242);
  pair = regs.read_lane_pair32(inline_one, 0);
  EXPECT_EQ(pair.lo, 0x3F800000u);
  EXPECT_EQ(pair.hi, 0x3F800000u);

  fx.wf->set_m0(0x77777777u);
  cdna4::Operand m0(64, cdna4::OperandType::OPR_SRC_SIMPLE, 124);
  pair = regs.read_lane_pair32(m0, 0);
  EXPECT_EQ(pair.lo, 0x77777777u);
  EXPECT_EQ(pair.hi, 0x77777777u);

  cdna4::Operand literal64(64, cdna4::OperandType::OPR_SRC_SIMPLE, 0x9999999988888888ull, true);
  pair = regs.read_lane_pair32(literal64, 0);
  EXPECT_EQ(pair.lo, 0x88888888u);
  EXPECT_EQ(pair.hi, 0x99999999u);
}

TEST(RegisterAccessTest, LanePair32ReadsGfx1250FlatScratchBase) {
  Fixture fx(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(fx.wf, nullptr);

  constexpr uint64_t kScratchBase = 0x7777777766666666ull;
  fx.wf->set_scratch_base(kScratchBase);
  cdna5::Operand flat_scratch_base(
      64, cdna5::OperandType::OPR_SRC,
      static_cast<int>(cdna5::OpSelSrc::OPR_SRC_SRC_FLAT_SCRATCH_BASE_LO));

  const OperandPair32 pair = RegisterAccess(*fx.wf).read_lane_pair32(flat_scratch_base, 0);
  EXPECT_EQ(pair.lo, 0x66666666u);
  EXPECT_EQ(pair.hi, 0x77777777u);
}

TEST(RegisterAccessTest, CuBoundSgprWritesCannotBypassObservation) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  const uint32_t reg = fx.sgpr_base() + 7;
  fx.cu->write_sgpr(reg, 0x11112222u);

  RegisterAccess registers(*fx.cu);
  EXPECT_THROW(registers.write_sgpr(reg, 0x33334444u), std::logic_error);
  EXPECT_THROW(registers.write_sgpr64(reg, 0x5555666677778888ull), std::logic_error);

  EXPECT_EQ(fx.cu->read_sgpr_storage(reg), 0x11112222u);
  EXPECT_TRUE(fx.plugin->sgpr_writes.empty());
}

TEST(RegisterAccessTest, Sgpr64AccessDoesNotCrossPhysicalBlockBoundary) {
  Fixture fx(ROCJITSU_CODE_ARCH_CDNA4, kSgprsPerWave, /*wavefront_slots=*/2);
  ASSERT_NE(fx.wf, nullptr);
  auto *adjacent_wave = fx.cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0, kSgprsPerWave, kVgprsPerWave);
  ASSERT_NE(adjacent_wave, nullptr);
  ASSERT_EQ(adjacent_wave->sgpr_alloc().base, fx.sgpr_base() + fx.cu->sgpr_allocation_block_size());

  const uint32_t boundary = adjacent_wave->sgpr_alloc().base - 1;
  constexpr uint32_t kLowSentinel = 0xA5A5A5A5u;
  constexpr uint32_t kAdjacentSentinel = 0x11112222u;
  fx.cu->write_sgpr(boundary, kLowSentinel);
  fx.cu->write_sgpr(adjacent_wave->sgpr_alloc().base, kAdjacentSentinel);

  RegisterAccess registers(*fx.wf);
  registers.write_sgpr64(boundary, 0x3333444455556666ull);

  EXPECT_EQ(fx.cu->read_sgpr_storage(boundary), kLowSentinel);
  EXPECT_EQ(fx.cu->read_sgpr_storage(adjacent_wave->sgpr_alloc().base), kAdjacentSentinel);
  EXPECT_TRUE(fx.plugin->sgpr_writes.empty());

  EXPECT_EQ(registers.read_sgpr64(boundary), 0u);
  EXPECT_TRUE(fx.plugin->sgpr_reads.empty());
}

TEST(RegisterAccessTest, SgprReadRegionRejectsWholeRangeBeforeCallbacks) {
  Fixture fx(ROCJITSU_CODE_ARCH_CDNA4, kSgprsPerWave, /*wavefront_slots=*/2);
  ASSERT_NE(fx.wf, nullptr);
  auto *adjacent_wave = fx.cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0, kSgprsPerWave, kVgprsPerWave);
  ASSERT_NE(adjacent_wave, nullptr);

  const uint32_t first = adjacent_wave->sgpr_alloc().base - 2;
  fx.cu->write_sgpr(first, 0x11223344u);
  fx.cu->write_sgpr(first + 1, 0x55667788u);
  fx.cu->write_sgpr(adjacent_wave->sgpr_alloc().base, 0xA5A5A5A5u);
  fx.cu->write_sgpr(adjacent_wave->sgpr_alloc().base + 1, 0x5A5A5A5Au);

  RegisterAccess registers(*fx.wf);
  auto denied = registers.read_sgpr_region(first, 4);
  EXPECT_FALSE(denied.valid());
  EXPECT_EQ(denied.qword(0), 0u);
  EXPECT_TRUE(fx.plugin->sgpr_reads.empty());

  auto owned = registers.read_sgpr_region(first, 2);
  ASSERT_TRUE(owned.valid());
  EXPECT_EQ(owned.qword(), 0x5566778811223344ull);
  ASSERT_EQ(fx.plugin->sgpr_reads.size(), 2u);
  EXPECT_EQ(fx.plugin->sgpr_reads[0], first);
  EXPECT_EQ(fx.plugin->sgpr_reads[1], first + 1);
  EXPECT_EQ(fx.plugin->sgpr_read_wavefronts[0], fx.wf);
  EXPECT_EQ(fx.plugin->sgpr_read_wavefronts[1], fx.wf);
  EXPECT_EQ(fx.plugin->scalar_reads,
            (std::vector<RegisterRef>{{RegClass::SGPR, kSgprsPerWave - 2, 2}}));
}

TEST(RegisterAccessTest, Scalar64OperandAndExplicitMaskWritesAreAtomic) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  constexpr int kLastSgprSelector = static_cast<int>(kSgprsPerWave - 1);
  const uint32_t last_sgpr = fx.sgpr_base() + kSgprsPerWave - 1;
  constexpr uint32_t kSentinel = 0xA5A5A5A5u;
  fx.cu->write_sgpr(last_sgpr, kSentinel);

  resolve_dst_write64(*fx.wf, kLastSgprSelector, 0x1122334455667788ull);

  EXPECT_EQ(fx.cu->read_sgpr_storage(last_sgpr), kSentinel);
  EXPECT_TRUE(fx.plugin->sgpr_writes.empty());
  EXPECT_EQ(resolve_src_scalar64(*fx.wf, kLastSgprSelector, /*m0_ev=*/124), 0u);
  EXPECT_TRUE(fx.plugin->sgpr_reads.empty());

  write_explicit_lane_mask(last_sgpr, *fx.wf, 0xFFEEDDCCBBAA9988ull);
  EXPECT_EQ(fx.cu->read_sgpr_storage(last_sgpr), kSentinel);
  EXPECT_TRUE(fx.plugin->sgpr_writes.empty());
}

TEST(RegisterAccessTest, TrapSelectorsUseDedicatedWaveStorage) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);

  constexpr int kTtmp0 = 108;
  fx.wf->set_ttmp(0, 0x11223344u);
  fx.wf->set_ttmp(1, 0x55667788u);

  EXPECT_EQ(resolve_src_scalar(*fx.wf, kTtmp0, /*m0_ev=*/124), 0x11223344u);
  EXPECT_EQ(resolve_src_scalar64(*fx.wf, kTtmp0, /*m0_ev=*/124), 0x5566778811223344ull);

  resolve_dst_write(*fx.wf, kTtmp0, 0xAABBCCDDu, /*m0_ev=*/124);
  EXPECT_EQ(fx.wf->ttmp(0), 0xAABBCCDDu);
  resolve_dst_write64(*fx.wf, kTtmp0, 0x0123456789ABCDEFull);
  EXPECT_EQ(fx.wf->ttmp(0), 0x89ABCDEFu);
  EXPECT_EQ(fx.wf->ttmp(1), 0x01234567u);

  EXPECT_TRUE(fx.plugin->sgpr_reads.empty());
  EXPECT_TRUE(fx.plugin->sgpr_writes.empty());
  EXPECT_EQ(fx.plugin->scalar_reads,
            (std::vector<RegisterRef>{{RegClass::TTMP, 0, 1}, {RegClass::TTMP, 0, 2}}));
  EXPECT_EQ(fx.plugin->scalar_writes,
            (std::vector<RegisterRef>{{RegClass::TTMP, 0, 1}, {RegClass::TTMP, 0, 2}}));
  EXPECT_TRUE(std::ranges::all_of(fx.plugin->scalar_read_wavefronts,
                                  [&](const Wavefront *wf) { return wf == fx.wf; }));
  EXPECT_TRUE(std::ranges::all_of(fx.plugin->scalar_write_wavefronts,
                                  [&](const Wavefront *wf) { return wf == fx.wf; }));
}

TEST(RegisterAccessTest, Vgpr64AccessRequiresOneOwnerForCompleteRange) {
  Fixture fx(ROCJITSU_CODE_ARCH_CDNA4, kSgprsPerWave, /*wavefront_slots=*/2);
  ASSERT_NE(fx.wf, nullptr);
  auto *adjacent_wave = fx.cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0, kSgprsPerWave, kVgprsPerWave);
  ASSERT_NE(adjacent_wave, nullptr);
  ASSERT_EQ(adjacent_wave->vgpr_alloc().base, fx.vgpr_base() + fx.cu->vgpr_allocation_block_size());

  const uint32_t boundary = adjacent_wave->vgpr_alloc().base - 1;
  constexpr uint32_t kLane = 0;
  constexpr uint32_t kLowSentinel = 0xA5A5A5A5u;
  constexpr uint32_t kAdjacentSentinel = 0x11112222u;
  fx.cu->write_vgpr(boundary, kLane, kLowSentinel);
  fx.cu->write_vgpr(adjacent_wave->vgpr_alloc().base, kLane, kAdjacentSentinel);

  RegisterAccess wave_registers(*fx.wf);
  wave_registers.write_vgpr64(boundary, kLane, 0x3333444455556666ull);
  EXPECT_EQ(wave_registers.read_vgpr64(boundary, kLane), 0u);

  RegisterAccess inferred_registers(*fx.cu);
  inferred_registers.write_vgpr64(boundary, kLane, 0x777788889999AAAAll);
  EXPECT_EQ(inferred_registers.read_vgpr64(boundary, kLane), 0u);

  EXPECT_EQ(fx.cu->read_vgpr_storage(boundary, kLane), kLowSentinel);
  EXPECT_EQ(fx.cu->read_vgpr_storage(adjacent_wave->vgpr_alloc().base, kLane), kAdjacentSentinel);
  EXPECT_TRUE(fx.plugin->writes.empty());
  EXPECT_TRUE(fx.plugin->reads.empty());
}

TEST(RegisterAccessTest, DeniedMultiRegisterReadRegionReturnsBoundedZeros) {
  Fixture fx(ROCJITSU_CODE_ARCH_CDNA4, kSgprsPerWave, /*wavefront_slots=*/2);
  ASSERT_NE(fx.wf, nullptr);
  auto *adjacent_wave = fx.cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0, kSgprsPerWave, kVgprsPerWave);
  ASSERT_NE(adjacent_wave, nullptr);

  const uint32_t boundary = adjacent_wave->vgpr_alloc().base - 1;
  auto region = RegisterAccess(*fx.wf).read_vgpr_region(boundary, /*reg_count=*/2, /*lane_mask=*/1);
  EXPECT_FALSE(region.valid());

  std::vector<uint32_t> copied(2 * fx.wf->wf_size(), 0xA5A5A5A5u);
  region.copy_to(copied);
  for (uint32_t value : copied)
    EXPECT_EQ(value, 0u);

  uint32_t visited = 0;
  region.for_each([&](std::span<const uint32_t> lanes) {
    ++visited;
    ASSERT_EQ(lanes.size(), fx.wf->wf_size());
    for (uint32_t value : lanes)
      EXPECT_EQ(value, 0u);
  });
  EXPECT_EQ(visited, 2u);
}

TEST(RegisterAccessTest, ZeroLaneMaskDoesNotMeanOwnershipDenial) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);

  auto region =
      RegisterAccess(*fx.wf).read_vgpr_region(fx.vgpr_base(), /*reg_count=*/1, /*lane_mask=*/0);
  EXPECT_TRUE(region.valid());
  EXPECT_TRUE(region.empty());
  EXPECT_EQ(region.lane(/*relative_reg=*/0, /*lane=*/0), 0u);
  EXPECT_TRUE(fx.plugin->reads.empty());
}

TEST(RegisterAccessTest, ExplicitWaveSgprAccessUsesReservedPhysicalBlock) {
  constexpr uint32_t kRequestedSgprs = 40;
  Fixture fx(ROCJITSU_CODE_ARCH_CDNA4, kRequestedSgprs);
  ASSERT_NE(fx.wf, nullptr);

  const uint32_t compiler_temporary = fx.sgpr_base() + kRequestedSgprs;
  ASSERT_LT(compiler_temporary, fx.sgpr_base() + fx.cu->sgpr_allocation_block_size());
  fx.cu->write_sgpr(compiler_temporary, 0x11223344u);
  EXPECT_TRUE(fx.plugin->sgpr_writes.empty());

  RegisterAccess regs(*fx.wf);
  EXPECT_EQ(regs.read_sgpr(compiler_temporary), 0x11223344u);
  ASSERT_EQ(fx.plugin->sgpr_reads.size(), 1u);
  EXPECT_EQ(fx.plugin->sgpr_reads[0], compiler_temporary);
  EXPECT_EQ(fx.plugin->sgpr_read_wavefronts[0], fx.wf);

  regs.write_sgpr(compiler_temporary, 0x55667788u);
  ASSERT_EQ(fx.plugin->sgpr_writes.size(), 1u);
  EXPECT_EQ(fx.plugin->sgpr_writes[0], compiler_temporary);
  EXPECT_EQ(fx.plugin->sgpr_write_wavefronts[0], fx.wf);
  EXPECT_EQ(fx.cu->read_sgpr_storage(compiler_temporary), 0x55667788u);

  fx.plugin->sgpr_reads.clear();
  fx.plugin->sgpr_read_wavefronts.clear();
  EXPECT_EQ(RegisterAccess(*fx.cu).read_sgpr(compiler_temporary), 0x55667788u);
  ASSERT_EQ(fx.plugin->sgpr_reads.size(), 1u);
  EXPECT_EQ(fx.plugin->sgpr_reads[0], compiler_temporary);
  EXPECT_EQ(fx.plugin->sgpr_read_wavefronts[0], fx.wf);
}

TEST(RegisterAccessTest, ReleasedWaveCannotOwnReallocatedRegisterBlock) {
  Fixture fx(ROCJITSU_CODE_ARCH_CDNA4, kSgprsPerWave, /*wavefront_slots=*/2);
  ASSERT_NE(fx.wf, nullptr);
  auto *released_wave = fx.cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0, kSgprsPerWave, kVgprsPerWave);
  ASSERT_NE(released_wave, nullptr);

  constexpr uint32_t kLane = 0;
  constexpr uint32_t kSgprSentinel = 0x11223344u;
  constexpr uint32_t kVgprSentinel = 0x55667788u;
  const uint32_t live_sgpr = fx.wf->sgpr_alloc().base;
  const uint32_t live_vgpr = fx.wf->vgpr_alloc().base;
  fx.cu->write_sgpr(live_sgpr, kSgprSentinel);
  fx.cu->write_vgpr(live_vgpr, kLane, kVgprSentinel);

  released_wave->reset();
  ASSERT_EQ(released_wave->sgpr_alloc().count, 0u);
  ASSERT_EQ(released_wave->vgpr_alloc().count, 0u);

  RegisterAccess released_access(*released_wave);
  released_access.write_sgpr(live_sgpr, 0xAABBCCDDu);
  released_access.write_vgpr(live_vgpr, kLane, 0xDDEEFF00u);

  EXPECT_EQ(fx.cu->read_sgpr_storage(live_sgpr), kSgprSentinel);
  EXPECT_EQ(fx.cu->read_vgpr_storage(live_vgpr, kLane), kVgprSentinel);
  EXPECT_TRUE(fx.plugin->sgpr_writes.empty());
  EXPECT_TRUE(fx.plugin->writes.empty());
}

TEST(RegisterAccessTest, FreeWavefrontClearsPhysicalOwnerMaps) {
  constexpr uint32_t kRequestedSgprs = 40;
  Fixture fx(ROCJITSU_CODE_ARCH_CDNA4, kRequestedSgprs);
  ASSERT_NE(fx.wf, nullptr);

  const uint32_t sgpr = fx.sgpr_base() + kRequestedSgprs;
  const uint32_t vgpr = fx.vgpr_base() + 3;
  fx.cu->write_sgpr(sgpr, 0x12345678u);
  fx.cu->write_vgpr(vgpr, /*lane=*/0, 0x87654321u);

  EXPECT_EQ(RegisterAccess(*fx.cu).read_sgpr(sgpr), 0x12345678u);
  EXPECT_EQ(RegisterAccess(*fx.cu).read_vgpr(vgpr, /*lane=*/0), 0x87654321u);
  ASSERT_EQ(fx.plugin->sgpr_reads.size(), 1u);
  ASSERT_EQ(fx.plugin->reads.size(), 1u);

  fx.cu->free_wavefront_resources(*fx.wf);
  fx.plugin->sgpr_reads.clear();
  fx.plugin->sgpr_read_wavefronts.clear();
  fx.plugin->reads.clear();

  (void)RegisterAccess(*fx.cu).read_sgpr(sgpr);
  (void)RegisterAccess(*fx.cu).read_vgpr(vgpr, /*lane=*/0);
  EXPECT_TRUE(fx.plugin->sgpr_reads.empty());
  EXPECT_TRUE(fx.plugin->reads.empty());
}

TEST(RegisterAccessTest, PublicOperandChunkReadObservesReadWindow) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t base = fx.vgpr_base();
  constexpr uint32_t logical_vgpr = 13;
  for (uint32_t lane = 0; lane < fx.wf->wf_size(); ++lane)
    fx.cu->write_vgpr(base + logical_vgpr, lane, 0xCAFE0000u | lane);

  cdna4::Operand src(32, cdna4::OperandType::OPR_SRC_VGPR, 256 + logical_vgpr);
  uint32_t out[3] = {};
  RegisterAccess regs(*fx.wf);
  regs.read_chunk(src, /*lane_base=*/4, /*count=*/3, out);

  ASSERT_EQ(fx.plugin->reads.size(), 1u);
  EXPECT_EQ(fx.plugin->reads[0].physical_reg, base + logical_vgpr);
  EXPECT_EQ(fx.plugin->reads[0].lane_mask, 0x70u);
  EXPECT_EQ(fx.plugin->reads[0].byte_mask, 0xFu);
  EXPECT_EQ(out[0], 0xCAFE0004u);
  EXPECT_EQ(out[1], 0xCAFE0005u);
  EXPECT_EQ(out[2], 0xCAFE0006u);
}

TEST(RegisterAccessTest, WriteChunkObservesWriteWindow) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t base = fx.vgpr_base();
  constexpr uint32_t logical_vgpr = 15;

  cdna4::Operand dst(32, cdna4::OperandType::OPR_VGPR, logical_vgpr);
  uint32_t values[4] = {0x10u, 0x11u, 0x12u, 0x13u};
  RegisterAccess regs(*fx.wf);
  regs.write_chunk(dst, /*lane_base=*/6, /*count=*/4, values, /*lane_mask=*/0b1011);

  ASSERT_EQ(fx.plugin->writes.size(), 1u);
  EXPECT_EQ(fx.plugin->writes[0].physical_reg, base + logical_vgpr);
  EXPECT_EQ(fx.plugin->writes[0].lane_mask, 0b1011u << 6);
  EXPECT_EQ(fx.cu->read_vgpr(base + logical_vgpr, 6), 0x10u);
  EXPECT_EQ(fx.cu->read_vgpr(base + logical_vgpr, 7), 0x11u);
  EXPECT_EQ(fx.cu->read_vgpr(base + logical_vgpr, 8), 0u);
  EXPECT_EQ(fx.cu->read_vgpr(base + logical_vgpr, 9), 0x13u);
}

TEST(RegisterAccessTest, ScopedVgprWriteMaskSuppressesScalarLaneCommits) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  constexpr uint32_t logical_vgpr = 21;
  const uint32_t reg = fx.vgpr_base() + logical_vgpr;
  cdna4::Operand dst(32, cdna4::OperandType::OPR_VGPR, logical_vgpr);

  {
    amdgpu::dpp::ScopedVgprWriteMask write_mask;
    write_mask.bind(*fx.wf, uint64_t{1} << 7);
    RegisterAccess(*fx.wf).write_lane(dst, 6, 0xAAAAu);
    RegisterAccess(*fx.wf).write_lane(dst, 7, 0xBBBBu);
  }

  ASSERT_EQ(fx.plugin->writes.size(), 1u);
  EXPECT_EQ(fx.plugin->writes[0].lane_mask, uint64_t{1} << 7);
  EXPECT_EQ(fx.cu->read_vgpr(reg, 6), 0u);
  EXPECT_EQ(fx.cu->read_vgpr(reg, 7), 0xBBBBu);

  RegisterAccess(*fx.wf).write_lane(dst, 6, 0xCCCCu);
  EXPECT_EQ(fx.cu->read_vgpr(reg, 6), 0xCCCCu);
}

TEST(RegisterAccessTest, ScopedVgprWriteMaskSuppressesNativeSimdCommits) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);
    constexpr uint32_t logical_vgpr = 22;
    constexpr uint64_t requested_lanes = 0b1111;
    constexpr uint64_t committed_lanes = 0b1010;
    const uint32_t reg = fx.vgpr_base() + logical_vgpr;
    cdna4::Operand dst(32, cdna4::OperandType::OPR_VGPR, logical_vgpr);

    amdgpu::dpp::ScopedVgprWriteMask write_mask;
    write_mask.bind(*fx.wf, committed_lanes);
    auto view = RegisterAccess(*fx.wf).write_operand(dst, requested_lanes);
    view.store_native<uint32_t>(0, util::native<uint32_t>(0xD00Du), requested_lanes);

    ASSERT_EQ(fx.plugin->writes.size(), 1u);
    EXPECT_EQ(fx.plugin->writes[0].lane_mask, committed_lanes);
    for (uint32_t lane = 0; lane < 4; ++lane) {
      const uint32_t expected = (committed_lanes & (uint64_t{1} << lane)) ? 0xD00Du : 0u;
      EXPECT_EQ(fx.cu->read_vgpr(reg, lane), expected);
    }
  }
}

TEST(RegisterAccessTest, ScopedVgprWriteMaskIntersectsAndRestoresNestedScopes) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);

  constexpr uint64_t outer_lanes = 0b111100;
  constexpr uint64_t inner_lanes = 0b101010;
  amdgpu::dpp::ScopedVgprWriteMask outer;
  outer.bind(*fx.wf, outer_lanes);
  EXPECT_EQ(fx.wf->vgpr_write_mask(), outer_lanes);

  {
    amdgpu::dpp::ScopedVgprWriteMask inner;
    inner.bind(*fx.wf, inner_lanes);
    EXPECT_EQ(fx.wf->vgpr_write_mask(), outer_lanes & inner_lanes);
  }

  EXPECT_EQ(fx.wf->vgpr_write_mask(), outer_lanes);
  outer.restore();
  EXPECT_EQ(fx.wf->vgpr_write_mask(), ~uint64_t{0});
  outer.restore();
  EXPECT_EQ(fx.wf->vgpr_write_mask(), ~uint64_t{0});
}

TEST(RegisterAccessTest, Wave32VgprWriteMaskRejectsNonExecutionLanes) {
  Fixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.wf, nullptr);
  ASSERT_EQ(fx.wf->wf_size(), 32u);

  constexpr uint64_t lane_mask = 0xFFFFFFFFULL;
  EXPECT_EQ(fx.wf->vgpr_write_mask(), lane_mask);
  fx.wf->set_vgpr_write_mask(~uint64_t{0});
  EXPECT_EQ(fx.wf->vgpr_write_mask(), lane_mask);

  amdgpu::dpp::ScopedVgprWriteMask scope;
  scope.bind(*fx.wf, uint64_t{1} << 43);
  EXPECT_EQ(fx.wf->vgpr_write_mask(), 0u);
  scope.restore();
  EXPECT_EQ(fx.wf->vgpr_write_mask(), lane_mask);
}

TEST(RegisterAccessTest, Wave64DispatchExpandsVgprWriteMaskToArchitecturalWidth) {
  Fixture fx(ROCJITSU_CODE_ARCH_RDNA4, kSgprsPerWave, /*wavefront_slots=*/1, kVgprsPerWave,
             /*wave_size=*/64);
  ASSERT_NE(fx.wf, nullptr);
  ASSERT_EQ(fx.wf->wf_size(), 64u);
  EXPECT_EQ(fx.wf->vgpr_write_mask(), ~uint64_t{0});

  rdna4::Operand destination(32, rdna4::OperandType::OPR_VGPR, 0);
  RegisterAccess(*fx.wf).write_lane(destination, 47, 0xCAFE0047u);
  EXPECT_EQ(fx.cu->read_vgpr(fx.vgpr_base(), 47), 0xCAFE0047u);
}

TEST(RegisterAccessTest, WriteViewsCannotOutliveTheirAcquiredWriteMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);
    RegisterAccess regs(*fx.wf);
    const uint32_t base = fx.vgpr_base();
    constexpr uint64_t requested_lanes = 0b1111;
    constexpr uint64_t acquired_lanes = 0b0101;
    constexpr uint64_t later_lanes = 0b0001;

    auto acquire = [&](auto make_view) {
      amdgpu::dpp::ScopedVgprWriteMask scope;
      scope.bind(*fx.wf, acquired_lanes);
      return make_view();
    };
    auto store_under_later_mask = [&](auto store) {
      amdgpu::dpp::ScopedVgprWriteMask scope;
      scope.bind(*fx.wf, later_lanes);
      store();
    };
    auto expect_captured_notifications = [&](size_t first_write) {
      ASSERT_GT(fx.plugin->writes.size(), first_write);
      for (size_t i = first_write; i < fx.plugin->writes.size(); ++i)
        EXPECT_EQ(fx.plugin->writes[i].lane_mask, acquired_lanes);
    };

    cdna4::Operand dst32(32, cdna4::OperandType::OPR_VGPR, 30);
    size_t first_write = fx.plugin->writes.size();
    auto write32 = acquire([&] { return regs.write_operand(dst32, requested_lanes); });
    expect_captured_notifications(first_write);
    store_under_later_mask([&] {
      write32.store_narrow<uint32_t>(0, util::broadcast_narrow<uint32_t>(0x1111u), acquired_lanes);
    });
    EXPECT_EQ(fx.cu->read_vgpr(base + 30, 0), 0x1111u);
    EXPECT_EQ(fx.cu->read_vgpr(base + 30, 2), 0u);
    EXPECT_THROW(write32.store_narrow<uint32_t>(0, util::broadcast_narrow<uint32_t>(0xAAAAu),
                                                uint64_t{1} << 1),
                 std::logic_error);

    cdna4::Operand dst64(64, cdna4::OperandType::OPR_VGPR, 32);
    first_write = fx.plugin->writes.size();
    auto write64 = acquire([&] { return regs.write_operand64(dst64, requested_lanes); });
    expect_captured_notifications(first_write);
    store_under_later_mask([&] {
      write64.store_native<uint64_t>(0, util::native<uint64_t>(0x2222u), acquired_lanes);
    });
    EXPECT_EQ(fx.cu->read_vgpr(base + 32, 0), 0x2222u);
    EXPECT_EQ(fx.cu->read_vgpr(base + 32, 2), 0u);
    EXPECT_THROW(
        write64.store_native<uint64_t>(0, util::native<uint64_t>(0xBBBBu), uint64_t{1} << 1),
        std::logic_error);

    cdna4::Operand dst_pair32(64, cdna4::OperandType::OPR_VGPR, 34);
    first_write = fx.plugin->writes.size();
    auto write_pair32 =
        acquire([&] { return regs.write_operand_pair32(dst_pair32, requested_lanes); });
    expect_captured_notifications(first_write);
    store_under_later_mask([&] {
      write_pair32.store_native_pair<uint32_t>(0, util::native<uint32_t>(0x3333u),
                                               util::native<uint32_t>(0x4444u), acquired_lanes);
    });
    EXPECT_EQ(fx.cu->read_vgpr(base + 34, 0), 0x3333u);
    EXPECT_EQ(fx.cu->read_vgpr(base + 35, 0), 0x4444u);
    EXPECT_EQ(fx.cu->read_vgpr(base + 34, 2), 0u);
    EXPECT_EQ(fx.cu->read_vgpr(base + 35, 2), 0u);
    EXPECT_THROW(write_pair32.store_native_pair<uint32_t>(0, util::native<uint32_t>(0xCCCCu),
                                                          util::native<uint32_t>(0xDDDDu),
                                                          uint64_t{1} << 1),
                 std::logic_error);

    cdna4::Operand acc32(32, cdna4::OperandType::OPR_VGPR, 36);
    const size_t first_read = fx.plugin->reads.size();
    first_write = fx.plugin->writes.size();
    auto readwrite32 = acquire([&] { return regs.readwrite_operand(acc32, requested_lanes); });
    ASSERT_GT(fx.plugin->reads.size(), first_read);
    EXPECT_EQ(fx.plugin->reads.back().lane_mask, requested_lanes);
    expect_captured_notifications(first_write);
    store_under_later_mask([&] {
      readwrite32.store_native<uint32_t>(0, util::native<uint32_t>(0x5555u), acquired_lanes);
    });
    EXPECT_EQ(fx.cu->read_vgpr(base + 36, 0), 0x5555u);
    EXPECT_EQ(fx.cu->read_vgpr(base + 36, 2), 0u);
    EXPECT_THROW(
        readwrite32.store_native<uint32_t>(0, util::native<uint32_t>(0xEEEEu), uint64_t{1} << 1),
        std::logic_error);

    cdna4::Operand acc64(64, cdna4::OperandType::OPR_VGPR, 38);
    const size_t first_read64 = fx.plugin->reads.size();
    first_write = fx.plugin->writes.size();
    auto readwrite64 = acquire([&] { return regs.readwrite_operand64(acc64, requested_lanes); });
    ASSERT_GT(fx.plugin->reads.size(), first_read64);
    for (size_t i = first_read64; i < fx.plugin->reads.size(); ++i)
      EXPECT_EQ(fx.plugin->reads[i].lane_mask, requested_lanes);
    expect_captured_notifications(first_write);
    store_under_later_mask([&] {
      readwrite64.store_native<uint64_t>(0, util::native<uint64_t>(0x6666u), acquired_lanes);
    });
    EXPECT_EQ(fx.cu->read_vgpr(base + 38, 0), 0x6666u);
    EXPECT_EQ(fx.cu->read_vgpr(base + 38, 2), 0u);
    EXPECT_THROW(
        readwrite64.store_native<uint64_t>(0, util::native<uint64_t>(0xFFFFu), uint64_t{1} << 1),
        std::logic_error);
  }
}

// Packed-half access is a single architectural operation on either bytes 0-1
// or bytes 2-3. Reading or preserving the other half inside the 32-bit
// register file must neither expose its value nor create another callback.
TEST(RegisterAccessTest, Packed16ReadsAndWritesObserveSelectedByteHalves) {
  Fixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.wf, nullptr);
  uint32_t reg = fx.vgpr_base() + 5;
  constexpr uint32_t lane = 3;
  fx.cu->write_vgpr(reg, lane, 0xAABBCCDDu);

  rdna4::Operand low_src(16, rdna4::OperandType::OPR_VGPR, 5, /*packed_16bit_source=*/true);
  rdna4::Operand high_src(16, rdna4::OperandType::OPR_VGPR, 128 + 5,
                          /*packed_16bit_source=*/true);
  RegisterAccess regs(*fx.wf);
  EXPECT_EQ(regs.read_lane(low_src, lane), 0xCCDDu);
  EXPECT_EQ(regs.read_lane(high_src, lane), 0xAABBu);

  ASSERT_EQ(fx.plugin->reads.size(), 2u);
  EXPECT_EQ(fx.plugin->reads[0].byte_mask, ExecutionPlugin::kLowHalfByteMask);
  EXPECT_EQ(fx.plugin->reads[1].byte_mask, ExecutionPlugin::kHighHalfByteMask);

  fx.plugin->reads.clear();
  rdna4::Operand low_dst(16, rdna4::OperandType::OPR_VGPR, 5,
                         /*packed_16bit_source=*/false, /*packed_16bit_dst=*/true);
  regs.write_lane(low_dst, lane, 0x1122u);
  EXPECT_TRUE(fx.plugin->reads.empty());
  ASSERT_EQ(fx.plugin->writes.size(), 1u);
  EXPECT_EQ(fx.plugin->writes[0].byte_mask, ExecutionPlugin::kLowHalfByteMask);
  EXPECT_EQ(fx.cu->read_vgpr(reg, lane), 0xAABB1122u);

  fx.plugin->reads.clear();
  fx.plugin->writes.clear();
  rdna4::Operand high_dst(16, rdna4::OperandType::OPR_VGPR, 128 + 5,
                          /*packed_16bit_source=*/false, /*packed_16bit_dst=*/true);
  regs.write_lane(high_dst, lane, 0x3344u);
  EXPECT_TRUE(fx.plugin->reads.empty());
  ASSERT_EQ(fx.plugin->writes.size(), 1u);
  EXPECT_EQ(fx.plugin->writes[0].byte_mask, ExecutionPlugin::kHighHalfByteMask);
  EXPECT_EQ(fx.cu->read_vgpr(reg, lane), 0x33441122u);
}

TEST(RegisterAccessTest, OperandWrite64ViewObservesBothPhysicalRegisters) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);
    constexpr uint32_t logical_vgpr = 16;
    constexpr uint32_t lane_base = 4;
    constexpr uint64_t lane_mask = 0b1011u << lane_base;
    const uint32_t base = fx.vgpr_base();

    cdna4::Operand dst(64, cdna4::OperandType::OPR_VGPR, logical_vgpr);
    RegisterAccess regs(*fx.wf);
    auto view = regs.write_operand64(dst, lane_mask);
    view.store_native<uint64_t>(lane_base, util::native<uint64_t>(0x1122334455667788ull),
                                /*lane_mask=*/0b1011);

    ASSERT_EQ(fx.plugin->writes.size(), 2u);
    EXPECT_EQ(fx.plugin->writes[0].physical_reg, base + logical_vgpr);
    EXPECT_EQ(fx.plugin->writes[1].physical_reg, base + logical_vgpr + 1);
    for (const WriteEvent &event : fx.plugin->writes) {
      EXPECT_EQ(event.lane_mask, lane_mask);
      EXPECT_EQ(event.byte_mask, ExecutionPlugin::kFullByteMask);
    }
    EXPECT_EQ(fx.cu->read_vgpr(base + logical_vgpr, lane_base), 0x55667788u);
    EXPECT_EQ(fx.cu->read_vgpr(base + logical_vgpr + 1, lane_base), 0x11223344u);
  }
}

TEST(RegisterAccessTest, OperandWriteViewStoreNarrowHonorsObservedLanes) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);
    constexpr uint32_t logical_vgpr = 19;
    constexpr uint32_t lane_base = 8;
    constexpr std::size_t width = util::native_width64;
    constexpr uint64_t chunk_mask = uint64_t{1} | (uint64_t{1} << (width - 1));
    constexpr uint64_t lane_mask = chunk_mask << lane_base;
    const uint32_t base = fx.vgpr_base();

    cdna4::Operand dst(32, cdna4::OperandType::OPR_VGPR, logical_vgpr);
    RegisterAccess regs(*fx.wf);
    auto view = regs.write_operand(dst, lane_mask);
    view.store_narrow<uint32_t>(lane_base, util::broadcast_narrow<uint32_t>(0xCAFEu), chunk_mask);

    ASSERT_EQ(fx.plugin->writes.size(), 1u);
    EXPECT_EQ(fx.plugin->writes[0].physical_reg, base + logical_vgpr);
    EXPECT_EQ(fx.plugin->writes[0].lane_mask, lane_mask);
    for (std::size_t lane = 0; lane < width; ++lane) {
      const uint32_t expected = (chunk_mask & (uint64_t{1} << lane)) ? 0xCAFEu : 0u;
      EXPECT_EQ(fx.cu->read_vgpr(base + logical_vgpr, lane_base + lane), expected);
    }
  }
}

TEST(RegisterAccessTest, OperandWriteViewRejectsStoreWindowPastWaveEnd) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);
    cdna4::Operand dst(32, cdna4::OperandType::OPR_VGPR, 20);
    RegisterAccess regs(*fx.wf);
    auto view = regs.write_operand(dst, ~uint64_t{0});

    EXPECT_THROW(view.store_narrow<uint32_t>(
                     /*lane_base=*/63, util::broadcast_narrow<uint32_t>(0x1234u),
                     /*lane_mask=*/1),
                 std::logic_error);
  }
}

TEST(RegisterAccessTest, OperandWriteViewsObserveActiveLanes) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);
    uint32_t base = fx.vgpr_base();
    constexpr uint32_t logical_vgpr = 17;
    cdna4::Operand dst(32, cdna4::OperandType::OPR_VGPR, logical_vgpr);

    RegisterAccess regs(*fx.wf);
    auto view = regs.write_operand(dst, /*lane_mask=*/0b1011u << 4);
    view.store_native<uint32_t>(/*lane_base=*/4, util::native<uint32_t>(0xABCDu),
                                /*lane_mask=*/0b1011);

    ASSERT_EQ(fx.plugin->writes.size(), 1u);
    EXPECT_EQ(fx.plugin->writes[0].physical_reg, base + logical_vgpr);
    EXPECT_EQ(fx.plugin->writes[0].lane_mask, 0b1011u << 4);
    EXPECT_EQ(fx.cu->read_vgpr(base + logical_vgpr, 4), 0xABCDu);
    EXPECT_EQ(fx.cu->read_vgpr(base + logical_vgpr, 5), 0xABCDu);
    EXPECT_EQ(fx.cu->read_vgpr(base + logical_vgpr, 6), 0u);
    EXPECT_EQ(fx.cu->read_vgpr(base + logical_vgpr, 7), 0xABCDu);
  }
}

TEST(RegisterAccessTest, OperandWriteViewStoresOnlySelectedBytes) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);
    constexpr uint32_t logical_vgpr = 18;
    constexpr uint32_t lane_base = 4;
    uint32_t reg = fx.vgpr_base() + logical_vgpr;
    fx.cu->write_vgpr(reg, lane_base, 0xAABBCCDDu);

    cdna4::Operand dst(32, cdna4::OperandType::OPR_VGPR, logical_vgpr);
    RegisterAccess regs(*fx.wf);
    auto view = regs.write_operand(dst, /*lane_mask=*/1u << lane_base,
                                   /*byte_mask=*/0b0010);
    view.store_native<uint32_t>(lane_base, util::native<uint32_t>(0x11223344u),
                                /*lane_mask=*/1);

    EXPECT_TRUE(fx.plugin->reads.empty());
    ASSERT_EQ(fx.plugin->writes.size(), 1u);
    EXPECT_EQ(fx.plugin->writes[0].lane_mask, 1u << lane_base);
    EXPECT_EQ(fx.plugin->writes[0].byte_mask, 0b0010);
    EXPECT_EQ(fx.cu->read_vgpr(reg, lane_base), 0xAABB33DDu);
  }
}

TEST(RegisterAccessTest, OperandWriteViewsRejectUnobservedLanes) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);
    uint32_t base = fx.vgpr_base();
    constexpr uint64_t observed_lane_mask = 1u << 4;
    constexpr uint32_t lane_base = 4;
    constexpr uint64_t unobserved_chunk_mask = 0b10;
    RegisterAccess regs(*fx.wf);

    cdna4::Operand dst32(32, cdna4::OperandType::OPR_VGPR, 18);
    auto write32 = regs.write_operand(dst32, observed_lane_mask);
    EXPECT_THROW(write32.store_native<uint32_t>(lane_base, util::native<uint32_t>(0x1111u),
                                                unobserved_chunk_mask),
                 std::logic_error);

    cdna4::Operand dst64(64, cdna4::OperandType::OPR_VGPR, 20);
    auto write64 = regs.write_operand64(dst64, observed_lane_mask);
    EXPECT_THROW(write64.store_native<uint64_t>(lane_base, util::native<uint64_t>(0x2222u),
                                                unobserved_chunk_mask),
                 std::logic_error);

    cdna4::Operand dst_pair32(64, cdna4::OperandType::OPR_VGPR, 22);
    auto write_pair32 = regs.write_operand_pair32(dst_pair32, observed_lane_mask);
    EXPECT_THROW(write_pair32.store_native_pair<uint32_t>(
                     lane_base, util::native<uint32_t>(0x3333u), util::native<uint32_t>(0x4444u),
                     unobserved_chunk_mask),
                 std::logic_error);

    cdna4::Operand acc32(32, cdna4::OperandType::OPR_VGPR, 24);
    auto readwrite32 = regs.readwrite_operand(acc32, observed_lane_mask);
    EXPECT_THROW(readwrite32.store_native<uint32_t>(lane_base, util::native<uint32_t>(0x5555u),
                                                    unobserved_chunk_mask),
                 std::logic_error);

    cdna4::Operand acc64(64, cdna4::OperandType::OPR_VGPR, 26);
    auto readwrite64 = regs.readwrite_operand64(acc64, observed_lane_mask);
    EXPECT_THROW(readwrite64.store_native<uint64_t>(lane_base, util::native<uint64_t>(0x6666u),
                                                    unobserved_chunk_mask),
                 std::logic_error);

    for (uint32_t reg : {18u, 20u, 21u, 22u, 23u, 24u, 26u, 27u})
      EXPECT_EQ(fx.cu->read_vgpr(base + reg, lane_base + 1), 0u);
  }
}

TEST(RegisterAccessTest, OperandReadViewFallbackUsesLaneSemantics) {
  Fixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.wf, nullptr);

  rdna4::Operand inline_one(16, rdna4::OperandType::OPR_SRC, 242);
  RegisterAccess regs(*fx.wf);
  auto view = regs.read_operand(inline_one, /*lane_mask=*/0x3);

  EXPECT_FALSE(view.has_storage());
  EXPECT_TRUE(fx.plugin->reads.empty());
  EXPECT_EQ(view.lane(0), 0x3C00u);
  const auto broadcast = view.load_native<uint32_t>(0);
  EXPECT_EQ(broadcast[0], 0x3C00u);
  EXPECT_EQ(broadcast[1], 0x3C00u);
}

TEST(RegisterAccessTest, OperandReadViewsMaskUnobservedBytes) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);
    constexpr uint32_t logical_vgpr = 21;
    uint32_t reg = fx.vgpr_base() + logical_vgpr;
    fx.cu->write_vgpr(reg, 0, 0xAABBCCDDu);
    fx.cu->write_vgpr(reg + 1, 0, 0x11223344u);

    cdna4::Operand src32(32, cdna4::OperandType::OPR_SRC_VGPR, 256 + logical_vgpr);
    cdna4::Operand src64(64, cdna4::OperandType::OPR_SRC_VGPR, 256 + logical_vgpr);
    RegisterAccess regs(*fx.wf);

    auto read32 = regs.read_operand(src32, /*lane_mask=*/1, /*byte_mask=*/0b0010);
    EXPECT_EQ(read32.lane(0), 0x0000CC00u);
    EXPECT_EQ(read32.load_native<uint32_t>(0)[0], 0x0000CC00u);

    auto read64 = regs.read_operand64(src64, /*lane_mask=*/1, /*byte_mask=*/0b0011);
    EXPECT_EQ(read64.load_native<uint64_t>(0)[0], 0x000033440000CCDDull);

    auto pair = regs.read_operand_pair32(src64, /*lane_mask=*/1, /*byte_mask=*/0b1100);
    EXPECT_EQ(pair.load_lo_native<uint32_t>(0)[0], 0xAABB0000u);
    EXPECT_EQ(pair.load_hi_native<uint32_t>(0)[0], 0x11220000u);
  }
}

} // namespace
