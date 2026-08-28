// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_access_test.cpp
/// @brief Tests for the AMDGPU instruction-facing register access facade.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/operand.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "util/simd.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

constexpr uint32_t kSgprsPerWave = 104;
constexpr uint32_t kVgprsPerWave = 256;

struct ReadEvent {
  uint32_t physical_reg = 0;
  uint64_t lane_mask = 0;
  uint8_t byte_mask = 0;
};

struct WriteEvent {
  uint32_t physical_reg = 0;
  uint64_t lane_mask = 0;
  uint8_t byte_mask = 0;
};

class RecordingPlugin : public ExecutionPlugin {
public:
  RecordingPlugin() : ExecutionPlugin("register_access_recorder") {}

  void onAmdgpuReadVgprLanes(const Wavefront *, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask) override {
    reads.push_back({physical_reg, lane_mask, byte_mask});
  }

  void onAmdgpuReadSgpr(const Wavefront *, uint32_t physical_reg) override {
    sgpr_reads.push_back(physical_reg);
  }

  void onAmdgpuWriteVgprLanes(const Wavefront *, uint32_t physical_reg, uint64_t lane_mask,
                              uint8_t byte_mask) override {
    writes.push_back({physical_reg, lane_mask, byte_mask});
  }

  std::vector<ReadEvent> reads;
  std::vector<WriteEvent> writes;
  std::vector<uint32_t> sgpr_reads;
};

struct Fixture {
  GpuMemory gpu_mem{"register_access_mem"};
  L2Cache l2{"register_access_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  std::shared_ptr<ExecutionPluginGroup> plugin_group;
  RecordingPlugin *plugin = nullptr;
  Wavefront *wf = nullptr;

  explicit Fixture(rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA4) {
    ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = kSgprsPerWave;
    cfg.vgprs_per_wf = kVgprsPerWave;
    cfg.lds_size_kb = 64;
    cu = ComputeUnitCore::create("register_access_cu", cfg, &gpu_mem, &l2);

    plugin_group = std::make_shared<ExecutionPluginGroup>();
    auto recorder = std::make_unique<RecordingPlugin>();
    plugin = recorder.get();
    plugin_group->add(std::move(recorder));
    cu->set_plugin_group(plugin_group);

    wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, kSgprsPerWave, kVgprsPerWave);
  }

  uint32_t sgpr_base() const { return wf->sgpr_alloc().base; }
  uint32_t vgpr_base() const { return wf->vgpr_alloc().base; }
};

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

TEST(RegisterAccessTest, Scalar64ReadObservesBothRegisters) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t base = fx.vgpr_base();
  fx.cu->write_vgpr(base + 11, 6, 0x89ABCDEFu);
  fx.cu->write_vgpr(base + 12, 6, 0x01234567u);

  RegisterAccess regs(*fx.cu);
  EXPECT_EQ(regs.read_vgpr64(base + 11, 6), 0x0123456789ABCDEFull);

  ASSERT_EQ(fx.plugin->reads.size(), 2u);
  EXPECT_EQ(fx.plugin->reads[0].physical_reg, base + 11);
  EXPECT_EQ(fx.plugin->reads[0].lane_mask, 1ULL << 6);
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

  regs.write_sgpr64(base + 19, 0xABCDEF0123456789ull);
  fx.plugin->sgpr_reads.clear();
  EXPECT_EQ(fx.cu->read_sgpr(base + 19), 0x23456789u);
  EXPECT_EQ(fx.cu->read_sgpr(base + 20), 0xABCDEF01u);
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
  Fixture fx;
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

} // namespace
