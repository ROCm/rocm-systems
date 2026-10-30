// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/sop2.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/sopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/sopk.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/sop2.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/sopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/sopk.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_sparse_exec.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include "util/data_types.h"
#include "util/simd.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

class WriteRecorder : public ExecutionPlugin {
public:
  WriteRecorder() : ExecutionPlugin("rj_arith_write_recorder") {}
  struct Write {
    uint32_t reg;
    uint64_t lanes;
    uint8_t bytes;
  };
  void onAmdgpuWriteVgprLanes(const Wavefront *, uint32_t reg, uint64_t lanes,
                              uint8_t bytes) override {
    writes.push_back({reg, lanes, bytes});
  }
  void onAmdgpuReadVgprLanes(const Wavefront *, uint32_t reg, uint64_t lanes,
                             uint8_t bytes) override {
    reads.push_back({reg, lanes, bytes});
  }
  void onAmdgpuReadSgpr(const Wavefront *, uint32_t reg) override { scalar_reads.push_back(reg); }
  void onAmdgpuWriteScalarRegister(const Wavefront *, RegisterRef reg) override {
    scalar_writes.push_back(reg);
  }
  std::vector<uint32_t> scalar_reads;
  std::vector<RegisterRef> scalar_writes;
  std::vector<Write> writes;
  std::vector<Write> reads;
};

// Use one architecture for decoding and execution, with observed register writes.
class BridgeFixture {
public:
  explicit BridgeFixture(bool rdna3, uint32_t lanes = 32, ArithmeticPolicy policy = {})
      : backend(rdna3 ? &rocjitsu::rdna3::execution_backend()
                      : &rocjitsu::rdna4::execution_backend()) {
    ComputeUnitCore::Config config{};
    config.arch = rdna3 ? ROCJITSU_CODE_ARCH_RDNA3 : ROCJITSU_CODE_ARCH_RDNA4;
    config.arithmetic = policy;
    decoder = Decoder::create(config.arch);
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cu = ComputeUnitCore::create("rj_arith_bridge", config, &memory, &l2);
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto observer = std::make_unique<WriteRecorder>();
    recorder = observer.get();
    group->add(std::move(observer));
    cu->set_plugin_group(group);
    wave = cu->dispatch_wf(0, 0, 104, 256, lanes);
    for (uint32_t reg = 0; reg < 24; ++reg)
      for (uint32_t lane = 0; lane < lanes; ++lane)
        cu->write_vgpr(base() + reg, lane, reg < 16 ? 0x3c003c00u : 0u);
  }
  uint32_t base() const { return wave->vgpr_alloc().base; }

  ScopedIsaExecutionBackend backend;
  std::unique_ptr<Decoder> decoder;
  GpuMemory memory{"rj_arith_bridge_memory"};
  L2Cache l2{"rj_arith_bridge_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  Wavefront *wave = nullptr;
  WriteRecorder *recorder = nullptr;
};

TEST(ArithBridgeTest, UnobservedMatrixWritesPreserveAliasesAndRejectedOutputs) {
  for (bool sparse : {false, true}) {
    BridgeFixture fixture(false);
    fixture.cu->set_plugin_group(nullptr);
    fixture.recorder = nullptr;
    fixture.wave->set_exec(0); // Matrix execution is independent of EXEC.
    const uint32_t destination = fixture.base() + (sparse ? 16 : 0);
    if (sparse) {
      for (uint32_t lane = 0; lane < 32; ++lane) {
        fixture.cu->write_vgpr(fixture.base() + 12, lane, 0x44444444u);
        for (uint32_t reg = 0; reg < 8; ++reg)
          fixture.cu->write_vgpr(destination + reg, lane, 0x3f800000u);
      }
    }
    auto invoke = [&](uint64_t flags) {
      if (sparse)
        arith::execute_swmmac(*fixture.wave, rj_arith_v_swmmac_f32_16x16x32_f16_w32, flags, 0,
                              destination, fixture.base(), fixture.base() + 4, fixture.base() + 12,
                              4, 8, 8, 32);
      else
        arith::execute_wmma(*fixture.wave, rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12, flags, 0,
                            destination, fixture.base(), fixture.base() + 8, fixture.base() + 16, 4,
                            8, std::nullopt);
    };
    const uint64_t flags =
        rj_arith_init_cpu_flags() | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
    invoke(flags | (1ULL << 63));
    EXPECT_TRUE(fixture.wave->instruction_execution_failed());
    for (uint32_t reg = 0; reg < 8; ++reg)
      for (uint32_t lane = 0; lane < 32; ++lane)
        EXPECT_EQ(fixture.cu->read_vgpr(destination + reg, lane),
                  sparse ? 0x3f800000u : 0x3c003c00u);
    fixture.wave->clear_instruction_execution_error();
    invoke(flags);
    ASSERT_FALSE(fixture.wave->instruction_execution_failed());
    for (uint32_t reg = 0; reg < 8; ++reg)
      for (uint32_t lane = 0; lane < 32; ++lane)
        EXPECT_EQ(fixture.cu->read_vgpr(destination + reg, lane),
                  sparse ? 0x41880000u : 0x41800000u);
  }
}

const uint64_t exact = RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;

TEST(ArithBridgeTest, MatrixIgnoresExecAndPreservesAliasedInputsUntilCommit) {
  BridgeFixture fixture(false);
  fixture.wave->set_exec(0);
  arith::execute_wmma(*fixture.wave, rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12, exact, 0,
                      fixture.base(), fixture.base(), fixture.base() + 8, fixture.base() + 16, 4, 8,
                      std::nullopt);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  for (uint32_t reg = 0; reg < 8; ++reg)
    for (uint32_t lane = 0; lane < 32; ++lane)
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + reg, lane), 0x41800000u);
  ASSERT_EQ(fixture.recorder->writes.size(), 8u);
  for (const auto &write : fixture.recorder->writes) {
    EXPECT_EQ(write.lanes, 0xffffffffu);
    EXPECT_EQ(write.bytes, 0xfu);
  }
}

TEST(ArithBridgeTest, FailedInvocationDoesNotReportOrCommitWrites) {
  BridgeFixture fixture(false);
  arith::execute_wmma(*fixture.wave, rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12,
                      exact | (1ULL << 63), 0, fixture.base(), fixture.base(), fixture.base() + 8,
                      fixture.base() + 16, 4, 8, std::nullopt);
  EXPECT_TRUE(fixture.wave->instruction_execution_failed());
  EXPECT_TRUE(fixture.recorder->writes.empty());
  for (uint32_t reg = 0; reg < 8; ++reg)
    for (uint32_t lane = 0; lane < 32; ++lane)
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + reg, lane), 0x3c003c00u);
}

TEST(ArithBridgeTest, Wave64Iu4ReadsOnlyTheLowerHalfOfItsFactorRegisters) {
  BridgeFixture fixture(false, 64);
  for (uint32_t lane = 0; lane < 64; ++lane) {
    const uint32_t factor = lane < 32 ? 0x11111111u : 0xffffffffu;
    fixture.cu->write_vgpr(fixture.base(), lane, factor);
    fixture.cu->write_vgpr(fixture.base() + 8, lane, factor);
  }
  arith::execute_wmma(*fixture.wave, rj_arith_v_wmma_i32_16x16x16_iu4_w64_gfx12, exact, 0,
                      fixture.base() + 24, fixture.base(), fixture.base() + 8, fixture.base() + 16,
                      1, 4, std::nullopt, 0xf, 32);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  ASSERT_EQ(fixture.recorder->reads.size(), 6u);
  for (const auto &read : fixture.recorder->reads) {
    const bool factor = read.reg == fixture.base() || read.reg == fixture.base() + 8;
    EXPECT_EQ(read.lanes, factor ? 0xffffffffu : ~uint64_t{0});
  }
  for (uint32_t reg = 0; reg < 4; ++reg)
    for (uint32_t lane = 0; lane < 64; ++lane)
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 24 + reg, lane), 16u);
}

TEST(ArithBridgeTest, PackedResultPreservesUnselectedHalfAndObservesOnlyWrittenBytes) {
  BridgeFixture fixture(true);
  arith::execute_wmma(*fixture.wave, rj_arith_v_wmma_f16_16x16x16_f16_w32, exact,
                      RJ_ARITH_WMMA_HIGH_C_D, fixture.base(), fixture.base(), fixture.base() + 8,
                      fixture.base() + 16, 8, 8, std::nullopt, 0xc);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  for (uint32_t reg = 0; reg < 8; ++reg)
    for (uint32_t lane = 0; lane < 32; ++lane)
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + reg, lane), 0x4c003c00u);
  ASSERT_EQ(fixture.recorder->writes.size(), 8u);
  for (const auto &write : fixture.recorder->writes)
    EXPECT_EQ(write.bytes, 0xcu);
}

TEST(ArithBridgeTest, SparseMatrixReadsItsTiedAccumulatorAndIgnoresExec) {
  BridgeFixture fixture(false);
  fixture.wave->set_exec(0);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    fixture.cu->write_vgpr(fixture.base() + 12, lane, 0x44444444u);
    for (uint32_t reg = 16; reg < 24; ++reg)
      fixture.cu->write_vgpr(fixture.base() + reg, lane, 0x3f800000u);
  }
  arith::execute_swmmac(*fixture.wave, rj_arith_v_swmmac_f32_16x16x32_f16_w32, exact, 0,
                        fixture.base() + 16, fixture.base(), fixture.base() + 4,
                        fixture.base() + 12, 4, 8, 8, 32);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  ASSERT_EQ(fixture.recorder->writes.size(), 8u);
  for (uint32_t reg = 16; reg < 24; ++reg)
    for (uint32_t lane = 0; lane < 32; ++lane)
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + reg, lane), 0x41880000u);
}

TEST(ArithBridgeTest, SparseWave64Iu4ObservesOnlyConsumedFactorAndIndexLanes) {
  BridgeFixture fixture(false, 64);
  for (uint32_t lane = 0; lane < 64; ++lane) {
    fixture.cu->write_vgpr(fixture.base(), lane, lane < 32 ? 0x11111111u : 0xffffffffu);
    fixture.cu->write_vgpr(fixture.base() + 8, lane, 0x11111111u);
    fixture.cu->write_vgpr(fixture.base() + 12, lane, lane < 32 ? 0x44444444u : 0xffffffffu);
  }
  arith::execute_swmmac(*fixture.wave, rj_arith_v_swmmac_i32_16x16x32_iu4_w64, exact, 0,
                        fixture.base() + 16, fixture.base(), fixture.base() + 8,
                        fixture.base() + 12, 1, 1, 4, 32);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  ASSERT_EQ(fixture.recorder->reads.size(), 7u);
  for (const auto &read : fixture.recorder->reads) {
    const bool partial = read.reg == fixture.base() || read.reg == fixture.base() + 12;
    EXPECT_EQ(read.lanes, partial ? 0xffffffffu : ~uint64_t{0});
  }
  for (uint32_t reg = 16; reg < 20; ++reg)
    for (uint32_t lane = 0; lane < 64; ++lane)
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + reg, lane), 16u);
}

TEST(ArithBridgeTest, FailedSparseInvocationPreservesItsTiedAccumulator) {
  BridgeFixture fixture(false);
  arith::execute_swmmac(*fixture.wave, rj_arith_v_swmmac_f32_16x16x32_f16_w32, exact | (1ULL << 63),
                        0, fixture.base(), fixture.base(), fixture.base() + 4, fixture.base() + 12,
                        4, 8, 8, 32);
  EXPECT_TRUE(fixture.wave->instruction_execution_failed());
  EXPECT_TRUE(fixture.recorder->writes.empty());
  for (uint32_t reg = 0; reg < 8; ++reg)
    for (uint32_t lane = 0; lane < 32; ++lane)
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + reg, lane), 0x3c003c00u);
}

} // namespace
} // namespace rocjitsu::amdgpu
