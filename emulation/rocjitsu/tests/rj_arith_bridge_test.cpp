// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_float_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

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
  std::vector<Write> writes;
  std::vector<Write> reads;
};

class BridgeFixture {
public:
  explicit BridgeFixture(bool rdna3, uint32_t lanes = 32, ArithmeticPolicy policy = {})
      : backend(rdna3 ? &rocjitsu::rdna3::execution_backend()
                      : &rocjitsu::rdna4::execution_backend()) {
    ComputeUnitCore::Config config{};
    config.arch = rdna3 ? ROCJITSU_CODE_ARCH_RDNA3 : ROCJITSU_CODE_ARCH_RDNA4;
    config.arithmetic = policy;
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
  GpuMemory memory{"rj_arith_bridge_memory"};
  L2Cache l2{"rj_arith_bridge_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  Wavefront *wave = nullptr;
  WriteRecorder *recorder = nullptr;
};

// Numeric 64-bit inline constants widen; packed pair inputs instead repeat a word.
TEST(ArithBridgeTest, NumericWideConstantDoesNotReplicateItsLowWord) {
  for (uint32_t lanes : {32u, 64u}) {
    BridgeFixture fixture(false, lanes);
    RegisterAccess registers(*fixture.wave);
    rdna4::Operand one(64, rdna4::OperandType::OPR_SRC, 129);
    arith::VectorInput64 input(registers, *fixture.wave, one, ~0ULL);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      EXPECT_EQ(input.data[0][lane], 1u);
      EXPECT_EQ(input.data[1][lane], 0u);
    }
    EXPECT_TRUE(fixture.recorder->reads.empty());
  }
}

TEST(ArithBridgeTest, FloatUnaryRespectsGuestDenormOmodAndMaskedWrites) {
  for (bool rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      for (uint64_t cpu : {RJ_ARITH_CPU_BASELINE, rj_arith_init_cpu_flags()}) {
        for (bool preserve : {false, true}) {
          BridgeFixture fixture(rdna3, lanes);
          fixture.wave->set_mode_raw(preserve ? 3u << 4 : 0u);
          fixture.wave->set_exec(5);
          fixture.cu->write_vgpr(fixture.base(), 0, 1u);
          fixture.cu->write_vgpr(fixture.base(), 2, 0x3fa00000u);
          rdna3::Operand d3(32, rdna3::OperandType::OPR_VGPR, 1);
          rdna4::Operand d4(32, rdna4::OperandType::OPR_VGPR, 1);
          const Operand &d = rdna3 ? static_cast<const Operand &>(d3) : d4;
          rdna3::Operand a3(32, rdna3::OperandType::OPR_SRC_VGPR, 256);
          rdna4::Operand a4(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
          const Operand &a = rdna3 ? static_cast<const Operand &>(a3) : a4;
          arith::execute_float_unary32(
              *fixture.wave, rdna3 ? rj_arith_v_ceil_f32_w32 : rj_arith_v_ceil_f32_w32_gfx12,
              rdna3 ? rj_arith_v_ceil_f32_w64 : rj_arith_v_ceil_f32_w64_gfx12,
              cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, RJ_ARITH_ALU_OMOD_2, d, a,
              rdna3, false);
          ASSERT_FALSE(fixture.wave->instruction_execution_failed());
          const bool scales = !rdna3 || !preserve;
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 0),
                    preserve ? (scales ? 0x40000000u : 0x3f800000u) : 0u);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 2),
                    scales ? 0x40800000u : 0x40000000u);
          for (uint32_t lane = 0; lane < lanes; ++lane)
            if (lane != 0 && lane != 2) {
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, lane), 0x3c003c00u);
            }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, LooseFloatUnarySkipsOptionalStateAndUsesDirectWrites) {
  for (bool rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      ArithmeticPolicy policy{};
      policy.semantics = ArithmeticSemantics::Loose;
      BridgeFixture fixture(rdna3, lanes, policy);
      fixture.cu->set_plugin_group(nullptr);
      fixture.recorder = nullptr;
      fixture.wave->set_exec(1);
      fixture.wave->set_trapsts(0x10);
      fixture.cu->write_vgpr(fixture.base(), 0, 0x3fa00000u);
      rdna3::Operand d3(32, rdna3::OperandType::OPR_VGPR, 1);
      rdna4::Operand d4(32, rdna4::OperandType::OPR_VGPR, 1);
      const Operand &d = rdna3 ? static_cast<const Operand &>(d3) : d4;
      rdna3::Operand a3(32, rdna3::OperandType::OPR_SRC_VGPR, 256);
      rdna4::Operand a4(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
      const Operand &a = rdna3 ? static_cast<const Operand &>(a3) : a4;
      EXPECT_FALSE(fixture.wave->cu().arithmetic_reports_fp_status());
      arith::execute_float_unary32(*fixture.wave,
                                   rdna3 ? rj_arith_v_ceil_f32_w32 : rj_arith_v_ceil_f32_w32_gfx12,
                                   rdna3 ? rj_arith_v_ceil_f32_w64 : rj_arith_v_ceil_f32_w64_gfx12,
                                   fixture.cu->arithmetic_flags(), 0, d, a, rdna3,
                                   fixture.wave->cu().arithmetic_reports_fp_status());
      EXPECT_FALSE(fixture.wave->instruction_execution_failed());
      EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 0), 0x40000000u);
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 1), 0x3c003c00u);
    }
  }
}

TEST(ArithBridgeTest, FloatUnaryRejectedStateReportingDoesNotCommit) {
  for (bool rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      BridgeFixture fixture(rdna3, lanes);
      fixture.wave->set_exec(1);
      fixture.wave->set_trapsts(0x10);
      fixture.recorder->writes.clear();
      rdna3::Operand d3(32, rdna3::OperandType::OPR_VGPR, 1);
      rdna4::Operand d4(32, rdna4::OperandType::OPR_VGPR, 1);
      const Operand &d = rdna3 ? static_cast<const Operand &>(d3) : d4;
      rdna3::Operand a3(32, rdna3::OperandType::OPR_SRC_VGPR, 256);
      rdna4::Operand a4(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
      const Operand &a = rdna3 ? static_cast<const Operand &>(a3) : a4;
      arith::execute_float_unary32(
          *fixture.wave, rdna3 ? rj_arith_v_ceil_f32_w32 : rj_arith_v_ceil_f32_w32_gfx12,
          rdna3 ? rj_arith_v_ceil_f32_w64 : rj_arith_v_ceil_f32_w64_gfx12,
          RJ_ARITH_CPU_BASELINE | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, 0, d, a, rdna3, true);
      EXPECT_TRUE(fixture.wave->instruction_execution_failed());
      EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
      EXPECT_TRUE(fixture.recorder->writes.empty());
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 0), 0x3c003c00u);
    }
  }
}

TEST(ArithBridgeTest, FloatBinaryPreservesAliasedInputsModifiersAndMasks) {
  for (bool rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      for (bool observed : {false, true}) {
        for (uint64_t cpu : {RJ_ARITH_CPU_BASELINE, rj_arith_init_cpu_flags()}) {
          BridgeFixture fixture(rdna3, lanes);
          if (!observed) {
            fixture.cu->set_plugin_group(nullptr);
            fixture.recorder = nullptr;
          }
          // Preserve inputs, flush outputs: both architectures apply OMOD.
          fixture.wave->set_mode_raw(1u << 4);
          const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000100000005ULL;
          fixture.wave->set_exec(exec_mask);
          fixture.wave->set_vgpr_write_mask(~4ULL);
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            fixture.cu->write_vgpr(fixture.base(), lane, 0xbfa00000u);     // -1.25
            fixture.cu->write_vgpr(fixture.base() + 1, lane, 0xbe800000u); // -0.25
          }
          rdna3::Operand d3(32, rdna3::OperandType::OPR_VGPR, 0);
          rdna4::Operand d4(32, rdna4::OperandType::OPR_VGPR, 0);
          rdna3::Operand a3(32, rdna3::OperandType::OPR_SRC_VGPR, 256);
          rdna4::Operand a4(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
          rdna3::Operand b3(32, rdna3::OperandType::OPR_SRC_VGPR, 257);
          rdna4::Operand b4(32, rdna4::OperandType::OPR_SRC_VGPR, 257);
          const Operand &d = rdna3 ? static_cast<const Operand &>(d3) : d4;
          const Operand &a = rdna3 ? static_cast<const Operand &>(a3) : a4;
          const Operand &b = rdna3 ? static_cast<const Operand &>(b3) : b4;
          arith::execute_float_binary32(
              *fixture.wave, rdna3 ? rj_arith_v_add_f32_w32 : rj_arith_v_add_f32_w32_gfx12,
              rdna3 ? rj_arith_v_add_f32_w64 : rj_arith_v_add_f32_w64_gfx12,
              cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
              RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_OMOD_2, d, a, b, false);
          ASSERT_FALSE(fixture.wave->instruction_execution_failed());
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            const bool written = (exec_mask & ~4ULL) & (1ULL << lane);
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                      written ? 0x40400000u : 0xbfa00000u); // (1.25 + 0.25) * 2
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, BulkWritesRequireUnobservedOwnedWholeRegisters) {
  BridgeFixture fixture(false);
  RegisterAccess registers(*fixture.wave);
  rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 0);
  unsigned calls = 0;
  auto invoke = [&](uint32_t *const *pointers, uint64_t exec_mask) {
    ++calls;
    EXPECT_EQ(exec_mask, 5u);
    auto source = registers.read_vgpr_region(fixture.base(), 1, exec_mask);
    EXPECT_EQ(pointers[0], source.lanes(0).data());
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  };
  EXPECT_FALSE(registers.try_bulk_write_operand(d, 7, invoke));
  EXPECT_EQ(calls, 0u);
  fixture.cu->set_plugin_group(nullptr);
  fixture.recorder = nullptr;
  fixture.wave->set_vgpr_write_mask(5);
  auto status = registers.try_bulk_write_operand(d, 7, invoke);
  ASSERT_TRUE(status);
  EXPECT_EQ(*status, RJ_ARITH_ERROR_INVALID_FLAGS);
  EXPECT_EQ(calls, 1u);
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), 0x3c003c00u);
  EXPECT_FALSE(registers.try_bulk_write_vgprs<2>(fixture.base() + 255, 2, 7, invoke));
  rdna4::Operand half(16, rdna4::OperandType::OPR_VGPR, 0);
  EXPECT_FALSE(registers.try_bulk_write_operand(half, 7, invoke));
  EXPECT_EQ(calls, 1u);
}

TEST(ArithBridgeTest, UnobservedVectorWritesPreserveAliasesMasksAndRejectedOutputs) {
  for (uint32_t lanes : {32u, 64u}) {
    for (uint64_t cpu : {RJ_ARITH_CPU_BASELINE, rj_arith_init_cpu_flags()}) {
      BridgeFixture fixture(false, lanes);
      fixture.cu->set_plugin_group(nullptr);
      fixture.recorder = nullptr;
      const uint64_t exec_mask = lanes == 32 ? 0x80000007ULL : 0x8000000100000007ULL;
      fixture.wave->set_exec(exec_mask);
      fixture.wave->set_vgpr_write_mask(~2ULL);
      rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 0);
      rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
      rdna4::Operand one(32, rdna4::OperandType::OPR_SRC, 129);
      arith::execute_binary32(*fixture.wave, rj_arith_v_add_nc_u32_w32, rj_arith_v_add_nc_u32_w64,
                              cpu, 0, d, a, one);
      arith::execute_unary32(*fixture.wave, rj_arith_v_not_b32_w32, rj_arith_v_not_b32_w64, cpu, 0,
                             d, a);
      arith::execute_ternary32(*fixture.wave, rj_arith_v_add3_u32_w32, rj_arith_v_add3_u32_w64, cpu,
                               0, d, a, one, one);
      ASSERT_FALSE(fixture.wave->instruction_execution_failed());
      arith::execute_binary32(*fixture.wave, rj_arith_v_add_nc_u32_w32, rj_arith_v_add_nc_u32_w64,
                              cpu | (1ULL << 63), 0, d, a, one);
      EXPECT_TRUE(fixture.wave->instruction_execution_failed());
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        const bool written = (exec_mask & ~2ULL) & (1ULL << lane);
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                  written ? ~(0x3c003c00u + 1u) + 2u : 0x3c003c00u);
      }
    }
  }
}

TEST(ArithBridgeTest, VectorCarryKeepsScalarLanesOutsideVgprWriteMask) {
  for (uint32_t lanes : {32u, 64u}) {
    for (bool observed : {false, true}) {
      for (uint64_t cpu : {RJ_ARITH_CPU_BASELINE, rj_arith_init_cpu_flags()}) {
        for (uint64_t write_mask : {~0ULL, 0ULL, 0x5555555555555555ULL}) {
          BridgeFixture fixture(false, lanes);
          if (!observed) {
            fixture.cu->set_plugin_group(nullptr);
            fixture.recorder = nullptr;
          }
          const uint64_t exec_mask = lanes == 32 ? 0x80000007ULL : 0x8000000100000007ULL;
          fixture.wave->set_exec(exec_mask);
          fixture.wave->set_vgpr_write_mask(write_mask);
          for (uint32_t lane = 0; lane < lanes; ++lane)
            fixture.cu->write_vgpr(fixture.base(), lane, 0xffffffffu);
          rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 0);
          rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
          rdna4::Operand zero(32, rdna4::OperandType::OPR_SRC, 128);
          uint64_t carry = ~0ULL;
          arith::execute_carry32<true>(*fixture.wave, rj_arith_v_add_co_ci_u32_w32,
                                       rj_arith_v_add_co_ci_u32_w64, cpu, 0, d, a, zero, exec_mask,
                                       [&](uint64_t value) { carry = value; });
          ASSERT_FALSE(fixture.wave->instruction_execution_failed());
          EXPECT_EQ(carry, exec_mask);
          for (uint32_t lane = 0; lane < lanes; ++lane)
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                      ((exec_mask & write_mask) >> lane) & 1 ? 0u : 0xffffffffu);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, VectorCarryZeroExecClearsScalarResultWithoutVectorAccess) {
  BridgeFixture fixture(false);
  fixture.wave->set_exec(0);
  rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 0);
  rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
  uint64_t carry = ~0ULL;
  arith::execute_carry32<false>(*fixture.wave, rj_arith_v_sub_co_u32_w32, rj_arith_v_sub_co_u32_w64,
                                0, 0, d, a, a, 0, [&](uint64_t value) { carry = value; });
  EXPECT_EQ(carry, 0u);
  EXPECT_TRUE(fixture.recorder->reads.empty());
  EXPECT_TRUE(fixture.recorder->writes.empty());
  EXPECT_FALSE(fixture.wave->instruction_execution_failed());
}

TEST(ArithBridgeTest, RejectedVectorCarryDoesNotCommitEitherOutput) {
  for (bool observed : {false, true}) {
    BridgeFixture fixture(false);
    if (!observed) {
      fixture.cu->set_plugin_group(nullptr);
      fixture.recorder = nullptr;
    }
    fixture.wave->set_exec(1);
    rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 0);
    rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
    bool committed = false;
    arith::execute_carry32<false>(*fixture.wave, rj_arith_v_subrev_co_u32_w32,
                                  rj_arith_v_subrev_co_u32_w64, 1ULL << 63, 0, d, a, a, 0,
                                  [&](uint64_t) { committed = true; });
    EXPECT_TRUE(fixture.wave->instruction_execution_failed());
    EXPECT_FALSE(committed);
    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), 0x3c003c00u);
    if (observed) {
      EXPECT_TRUE(fixture.recorder->writes.empty());
    }
  }
}

TEST(ArithBridgeTest, WideMadCommitsBothWordsAndCarry) {
  for (uint32_t lanes : {32u, 64u}) {
    for (bool observed : {false, true}) {
      BridgeFixture fixture(false, lanes);
      if (!observed) {
        fixture.cu->set_plugin_group(nullptr);
        fixture.recorder = nullptr;
      }
      const uint32_t lane = lanes - 1;
      fixture.wave->set_exec(1ULL << lane);
      fixture.cu->write_vgpr(fixture.base(), lane, UINT32_MAX);
      rdna4::Operand d(64, rdna4::OperandType::OPR_VGPR, 0);
      rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
      rdna4::Operand two(32, rdna4::OperandType::OPR_SRC, 130);
      rdna4::Operand zero(64, rdna4::OperandType::OPR_SRC, 128);
      uint64_t carry = ~0ULL;
      arith::execute_mad64(*fixture.wave, rj_arith_v_mad_co_u64_u32_w32,
                           rj_arith_v_mad_co_u64_u32_w64, 0, 0, d, a, two, zero,
                           [&](uint64_t value) { carry = value; });
      ASSERT_FALSE(fixture.wave->instruction_execution_failed());
      // (2^32-1)*2 crosses the destination-word boundary without a 65th-bit carry.
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), 0xfffffffeu);
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, lane), 1u);
      EXPECT_EQ(carry, 0u);
      for (uint32_t inactive = 0; inactive < lane; ++inactive) {
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), inactive), 0x3c003c00u);
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, inactive), 0x3c003c00u);
      }
    }
  }
}

TEST(ArithBridgeTest, RejectedWideOutputDoesNotWriteOrNotifyEitherWord) {
  for (bool observed : {false, true}) {
    BridgeFixture fixture(false);
    if (!observed) {
      fixture.cu->set_plugin_group(nullptr);
      fixture.recorder = nullptr;
    }
    fixture.wave->set_exec(1);
    rdna4::Operand d(64, rdna4::OperandType::OPR_VGPR, 0);
    rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
    rdna4::Operand c(64, rdna4::OperandType::OPR_SRC_VGPR, 256);
    bool committed = false;
    arith::execute_mad64(*fixture.wave, rj_arith_v_mad_co_u64_u32_w32,
                         rj_arith_v_mad_co_u64_u32_w64, 1ULL << 63, 0, d, a, a, c,
                         [&](uint64_t) { committed = true; });
    EXPECT_TRUE(fixture.wave->instruction_execution_failed());
    EXPECT_FALSE(committed);
    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), 0x3c003c00u);
    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 0), 0x3c003c00u);
    if (observed) {
      EXPECT_TRUE(fixture.recorder->writes.empty());
    }
  }
}

} // namespace
} // namespace rocjitsu::amdgpu
