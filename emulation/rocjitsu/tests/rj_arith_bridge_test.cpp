// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "fixtures/float_dot/gfx1100_cases.h"
#include "fixtures/float_dot/gfx1201_cases.h"
#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vopc.h"

#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vopc.h"

#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_compare_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_float_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_sparse_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
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
  std::vector<uint32_t> scalar_reads;
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

// Guest rounding is forwarded while host rounding remains at its default.
TEST(ArithBridgeTest, Fma64DirectedGuestRounding) {
  for (bool rdna3 : {false, true}) {
    for (unsigned lanes : {32u, 64u}) {
      for (unsigned rounding = 1; rounding < 4; ++rounding) {
        BridgeFixture fixture(rdna3, lanes);
        fixture.wave->set_mode_raw((rounding << 2) | (3u << 6));
        fixture.wave->set_exec(uint64_t{1} << (lanes - 1));
        const double inputs[] = {rounding == 2 ? -1.0 : 1.0, 0x1p-53, rounding == 2 ? -1.0 : 1.0};
        for (unsigned operand = 0; operand < 3; ++operand) {
          const auto bits = std::bit_cast<uint64_t>(inputs[operand]);
          fixture.cu->write_vgpr(fixture.base() + 2 * operand, lanes - 1, uint32_t(bits));
          fixture.cu->write_vgpr(fixture.base() + 2 * operand + 1, lanes - 1, uint32_t(bits >> 32));
        }
        const auto words =
            rdna3 ? rdna3::build_vop3(rdna3::kVFmaF64Vop3,
                                      {.vdst = 6, .src0 = 256, .src1 = 258, .src2 = 260})
                  : rdna4::build_vop3(rdna4::kVFmaF64Vop3,
                                      {.vdst = 6, .src0 = 256, .src1 = 258, .src2 = 260});
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        const uint64_t actual =
            fixture.cu->read_vgpr(fixture.base() + 6, lanes - 1) |
            (uint64_t(fixture.cu->read_vgpr(fixture.base() + 7, lanes - 1)) << 32);
        EXPECT_EQ(actual, rounding == 1   ? 0x3ff0000000000001ULL
                          : rounding == 2 ? 0xbff0000000000001ULL
                                          : 0x3ff0000000000000ULL);
      }
    }
  }
}

// Exercise guest rounding through decoded binary arithmetic and selection OMOD.
TEST(ArithBridgeTest, DoubleBinaryGuestRounding) {
  struct Sample {
    uint16_t gfx11_op, gfx12_op;
    uint64_t a, b, expected;
    uint8_t omod;
  };
  const Sample samples[] = {
      {rdna3::kVAddF64Vop3, rdna4::kVAddF64Vop3, 0x3ff0000000000000ULL, 0x3ca0000000000000ULL,
       0x3ff0000000000001ULL, 0},
      {rdna3::kVMulF64Vop3, rdna4::kVMulF64Vop3, 0x3ff0000000000001ULL, 0x3ff0000000000001ULL,
       0x3ff0000000000003ULL, 0},
      {rdna3::kVMinF64Vop3, rdna4::kVMinNumF64Vop3, 0xffefffffffffffffULL, 0xffefffffffffffffULL,
       0xffefffffffffffffULL, 1},
  };
  for (bool rdna3 : {false, true}) {
    for (unsigned lanes : {32u, 64u}) {
      for (const auto &sample : samples) {
        BridgeFixture fixture(rdna3, lanes);
        // Round up, flush output denormals; OMOD is enabled on both architectures.
        fixture.wave->set_mode_raw((1u << 2) | (1u << 6));
        fixture.wave->set_exec(1);
        for (unsigned operand = 0; operand < 2; ++operand) {
          const uint64_t bits = operand ? sample.b : sample.a;
          fixture.cu->write_vgpr(fixture.base() + 2 * operand, 0, uint32_t(bits));
          fixture.cu->write_vgpr(fixture.base() + 2 * operand + 1, 0, uint32_t(bits >> 32));
        }
        const auto words =
            rdna3 ? rdna3::build_vop3(sample.gfx11_op,
                                      {.vdst = 4, .src0 = 256, .src1 = 258, .omod = sample.omod})
                  : rdna4::build_vop3(sample.gfx12_op,
                                      {.vdst = 4, .src0 = 256, .src1 = 258, .omod = sample.omod});
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 4, 0) |
                      (uint64_t(fixture.cu->read_vgpr(fixture.base() + 5, 0)) << 32),
                  sample.expected);
      }
    }
  }
}

// RDNA3 DX10_CLAMP controls NaN handling for every migrated binary operation.
TEST(ArithBridgeTest, Binary32Dx10Clamp) {
  for (uint16_t op : {rdna3::kVAddF32Vop3, rdna3::kVMulF32Vop3}) {
    for (bool dx10 : {false, true}) {
      BridgeFixture fixture(true);
      fixture.wave->set_mode_raw(0xf0u | (dx10 ? Wavefront::DX10_CLAMP_BIT : 0));
      fixture.wave->set_exec(1);
      fixture.cu->write_vgpr(fixture.base(), 0, 0x7fc12345u);
      fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x3f800000u);
      const auto words = rdna3::build_vop3(op, {.vdst = 2, .clamp = 1, .src0 = 256, .src1 = 257});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      const uint32_t result = fixture.cu->read_vgpr(fixture.base() + 2, 0);
      if (dx10)
        EXPECT_EQ(result, 0u);
      else
        EXPECT_GT(result & 0x7fffffffu, 0x7f800000u);
    }
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
              rdna3);
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
      arith::execute_float_unary32(*fixture.wave,
                                   rdna3 ? rj_arith_v_ceil_f32_w32 : rj_arith_v_ceil_f32_w32_gfx12,
                                   rdna3 ? rj_arith_v_ceil_f32_w64 : rj_arith_v_ceil_f32_w64_gfx12,
                                   fixture.cu->arithmetic_flags(), 0, d, a, rdna3);
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
          RJ_ARITH_CPU_BASELINE | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, 0, d, a, rdna3);
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
              RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_OMOD_2, d, a, b);
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

TEST(ArithBridgeTest, DecodedFmaFormsPreserveAccumulatorLiteralOrderAndMasks) {
  for (bool rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      for (bool observed : {false, true}) {
        for (unsigned form = 0; form < 8; ++form) {
          if (!rdna3 && form >= 6)
            continue;
          const bool dx9 = form >= 5;
          ArithmeticPolicy policy{};
          if (dx9)
            policy.semantics = ArithmeticSemantics::BitExactEmpirical;
          BridgeFixture fixture(rdna3, lanes, policy);
          if (!observed) {
            fixture.cu->set_plugin_group(nullptr);
            fixture.recorder = nullptr;
          }
          // DX9 must flush its tiny input despite MODE preserving denormals.
          fixture.wave->set_mode_raw(3u << 4);
          const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000100000005ULL;
          fixture.wave->set_exec(exec_mask);
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            fixture.cu->write_vgpr(fixture.base(), lane, dx9 ? 1u : 0x40000000u);
            fixture.cu->write_vgpr(fixture.base() + 1, lane, dx9 ? 0x7f800000u : 0x40400000u);
            fixture.cu->write_vgpr(fixture.base() + 2, lane, 0x40800000u);
            fixture.cu->write_vgpr(fixture.base() + 4, lane, 0x40800000u);
          }
          std::array<uint32_t, 2> words{};
          const bool vop3 = form == 0 || form == 2 || form == 5 || form == 7;
          if (vop3) {
            const uint16_t opcode = form == 0   ? rdna4::kVFmaF32Vop3
                                    : form == 2 ? rdna4::kVFmacF32Vop3
                                    : form == 5 ? rdna4::kVFmaDx9ZeroF32Vop3
                                                : rdna3::kVFmacDx9ZeroF32Vop3;
            // Only ordinary FMA modifies C; FMAC ignores the C modifier bits.
            const uint8_t neg = dx9 ? 0 : 4;
            words =
                rdna3 ? rdna3::build_vop3(
                            opcode, {.vdst = 4, .src0 = 256, .src1 = 257, .src2 = 258, .neg = neg})
                      : rdna4::build_vop3(
                            opcode, {.vdst = 4, .src0 = 256, .src1 = 257, .src2 = 258, .neg = neg});
          } else {
            const uint16_t opcode = form == 1   ? rdna4::kVFmacF32Vop2
                                    : form == 3 ? rdna4::kVFmaakF32Vop2
                                    : form == 4 ? rdna4::kVFmamkF32Vop2
                                                : rdna3::kVFmacDx9ZeroF32Vop2;
            words[0] = rdna3 ? rdna3::build_vop2(opcode, {.src0 = 256, .vsrc1 = 1, .vdst = 4})[0]
                             : rdna4::build_vop2(opcode, {.src0 = 256, .vsrc1 = 1, .vdst = 4})[0];
            words[1] = 0x40800000u;
          }
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          const uint32_t expected = dx9         ? 0x40800000u
                                    : form == 0 ? 0x40000000u
                                    : form == 4 ? 0x41300000u
                                                : 0x41200000u;
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 4, lane),
                      exec_mask & (1ULL << lane) ? expected : 0x40800000u)
                << "arch=" << rdna3 << " form=" << form << " lane=" << lane;
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, DecodedFmaRequiredReportingRejectsWithoutPartialCommit) {
  for (bool rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      ArithmeticPolicy policy{};
      policy.semantics = ArithmeticSemantics::BitExactEmpirical;
      BridgeFixture fixture(rdna3, lanes, policy);
      fixture.wave->set_exec(1);
      fixture.wave->set_trapsts(0x10);
      fixture.recorder->writes.clear();
      const auto words =
          rdna3 ? rdna3::build_vop3(rdna3::kVFmaF32Vop3,
                                    {.vdst = 4, .src0 = 256, .src1 = 257, .src2 = 260})
                : rdna4::build_vop3(rdna4::kVFmaF32Vop3,
                                    {.vdst = 4, .src0 = 256, .src1 = 257, .src2 = 260});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      EXPECT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).failed());
      EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 4, 0), 0x3c003c00u);
      EXPECT_TRUE(fixture.recorder->writes.empty());
    }
  }
}

// Compatibility uses the legacy classifier even outside mode 3's reporting domain.
TEST(ArithBridgeTest, CompatibilityMulRetainsLegacyStatus) {
  for (bool rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      BridgeFixture fixture(rdna3, lanes);
      rdna3::Operand d3(32, rdna3::OperandType::OPR_VGPR, 0);
      rdna4::Operand d4(32, rdna4::OperandType::OPR_VGPR, 0);
      rdna3::Operand a3(32, rdna3::OperandType::OPR_SRC_VGPR, 256);
      rdna4::Operand a4(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
      rdna3::Operand b3(32, rdna3::OperandType::OPR_SRC_VGPR, 257);
      rdna4::Operand b4(32, rdna4::OperandType::OPR_SRC_VGPR, 257);
      const Operand &d = rdna3 ? static_cast<const Operand &>(d3) : d4;
      const Operand &a = rdna3 ? static_cast<const Operand &>(a3) : a4;
      const Operand &b = rdna3 ? static_cast<const Operand &>(b3) : b4;
      for (uint32_t guest : {0x30u, 0x31u, 0x20u}) {
        for (uint64_t modifiers : {uint64_t(0), uint64_t(RJ_ARITH_ALU_OMOD_2)}) {
          for (uint64_t semantics : {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_LOOSE}) {
            fixture.wave->set_mode_raw(guest);
            fixture.wave->set_exec(3);
            fixture.wave->set_vgpr_write_mask(~2ULL);
            fixture.wave->set_trapsts(0x80000000u);
            fixture.cu->write_vgpr(fixture.base(), 0, 0x7f7fffffu);
            fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x40000000u);
            arith::execute_float_binary32(
                *fixture.wave, rdna3 ? rj_arith_v_mul_f32_w32 : rj_arith_v_mul_f32_w32_gfx12,
                rdna3 ? rj_arith_v_mul_f32_w64 : rj_arith_v_mul_f32_w64_gfx12,
                rj_arith_init_cpu_flags() | semantics, modifiers, d, a, b,
                /*rdna3_dx10_clamp=*/false, /*compatibility_mul=*/true);
            ASSERT_FALSE(fixture.wave->instruction_execution_failed());
            const uint32_t expected = semantics == RJ_ARITH_SEMANTICS_LOOSE ? 0
                                      : !rdna3 && modifiers                 ? 0x08u
                                                                            : 0x28u;
            EXPECT_EQ(fixture.wave->trapsts(), 0x80000000u | expected);
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), 0x7f800000u);
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

TEST(ArithBridgeTest, DecodedWideShiftsPreserveOverlappingInputsAndInactiveWords) {
  struct Operation {
    uint16_t rdna3_opcode;
    uint16_t rdna4_opcode;
  };
  const Operation operations[] = {
      {rdna3::kVLshlrevB64Vop3, rdna4::kVLshlrevB64Vop3},
      {rdna3::kVLshrrevB64Vop3, rdna4::kVLshrrevB64Vop3},
      {rdna3::kVAshrrevI64Vop3, rdna4::kVAshrrevI64Vop3},
  };
  for (bool is_rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      for (bool observed : {false, true}) {
        BridgeFixture fixture(is_rdna3, lanes);
        if (!observed) {
          fixture.cu->set_plugin_group(nullptr);
          fixture.recorder = nullptr;
        }
        for (unsigned op = 0; op < 3; ++op) {
          for (bool clamp : {false, true}) {
            for (uint64_t exec_mask : {0ULL, 0x8000000180000005ULL}) {
              fixture.wave->set_exec(exec_mask);
              const uint64_t active = fixture.wave->exec();
              std::array<uint64_t, 64> values;
              std::array<uint32_t, 64> counts;
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                counts[lane] = lane * 31u + 64u;
                values[lane] = 0x8123456789abcdefULL ^ (0x123456789abcdefULL * lane);
                fixture.cu->write_vgpr(fixture.base(), lane, counts[lane]);
                fixture.cu->write_vgpr(fixture.base() + 1, lane, uint32_t(values[lane]));
                fixture.cu->write_vgpr(fixture.base() + 2, lane, uint32_t(values[lane] >> 32));
              }
              // D overlaps both the count and the low word of the value.
              const auto words =
                  is_rdna3 ? rdna3::build_vop3(
                                 operations[op].rdna3_opcode,
                                 {.vdst = 0, .clamp = uint8_t(clamp), .src0 = 256, .src1 = 257})
                           : rdna4::build_vop3(
                                 operations[op].rdna4_opcode,
                                 {.vdst = 0, .clamp = uint8_t(clamp), .src0 = 256, .src1 = 257});
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                const unsigned shift = counts[lane] & 63;
                uint64_t expected = op == 0 ? values[lane] << shift : values[lane] >> shift;
                if (op == 2 && shift && (values[lane] >> 63))
                  expected |= ~0ULL << (64 - shift);
                const bool enabled = (active >> lane) & 1;
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                          enabled ? uint32_t(expected) : counts[lane]);
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, lane),
                          enabled ? uint32_t(expected >> 32) : uint32_t(values[lane]));
              }
            }
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, Rdna4WideShiftVop2UsesOneWordCountAndTwoWordValue) {
  BridgeFixture fixture(false);
  fixture.wave->set_exec(1);
  fixture.cu->write_vgpr(fixture.base(), 0, 0x89abcdefu);
  fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x12345678u);
  const auto words =
      rdna4::build_vop2(rdna4::kVLshlrevB64Vop2, {.src0 = 160, .vsrc1 = 0, .vdst = 0});
  auto instruction = fixture.decoder->decode_window(words);
  ASSERT_TRUE(instruction.succeeded());
  ASSERT_TRUE(
      fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), 0u);
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 0), 0x89abcdefu);
}

TEST(ArithBridgeTest, DecodedWideMadCommitsPairAndExtendedCarryAcrossWaveWidths) {
  struct Case {
    bool is_signed;
    uint32_t a, b;
    uint64_t c, wrapped, saturated;
    bool carry;
  };
  // Signed cases inherit the existing rj-arith Mad64.SignedScalarOutputIsTheExtendedSign
  // fixtures, validated against its gfx1201 hardware corpus: bit 64, not overflow.
  const Case cases[] = {
      {true, 0, 0, ~0ULL, ~0ULL, ~0ULL, true},
      {true, 0x80000000u, 0x80000000u, 0x7fffffffffffffffULL, 0xbfffffffffffffffULL,
       0x7fffffffffffffffULL, false},
      {true, 0x80000000u, 0x7fffffffu, 0x8000000000000000ULL, 0x4000000080000000ULL,
       0x8000000000000000ULL, true},
      {false, 0xffffffffu, 0xffffffffu, ~0ULL, 0xfffffffe00000000ULL, ~0ULL, true},
      {false, 1, 2, 3, 5, 5, false},
  };
  for (bool is_rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      for (bool observed : {false, true}) {
        BridgeFixture fixture(is_rdna3, lanes);
        if (!observed) {
          fixture.cu->set_plugin_group(nullptr);
          fixture.recorder = nullptr;
        }
        for (const auto &test : cases) {
          for (bool clamp : {false, true}) {
            for (uint64_t exec_mask : {0ULL, 0x8000000180000005ULL}) {
              fixture.wave->set_exec(exec_mask);
              const uint64_t active = fixture.wave->exec();
              const uint32_t sb = fixture.wave->sgpr_alloc().base;
              fixture.cu->write_sgpr(sb, 0xffffffffu);
              fixture.cu->write_sgpr(sb + 1, 0x12345678u);
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                fixture.cu->write_vgpr(fixture.base(), lane, test.a);
                fixture.cu->write_vgpr(fixture.base() + 1, lane, test.b);
                fixture.cu->write_vgpr(fixture.base() + 2, lane, uint32_t(test.c));
                fixture.cu->write_vgpr(fixture.base() + 3, lane, uint32_t(test.c >> 32));
              }
              const auto words =
                  is_rdna3
                      ? rdna3::build_vop3_sdst_enc(test.is_signed ? rdna3::kVMadI64I32Vop3SdstEnc
                                                                  : rdna3::kVMadU64U32Vop3SdstEnc,
                                                   {.vdst = 2,
                                                    .sdst = 0,
                                                    .clamp = uint8_t(clamp),
                                                    .src0 = 256,
                                                    .src1 = 257,
                                                    .src2 = 258})
                      : rdna4::build_vop3_sdst_enc(test.is_signed ? rdna4::kVMadCoI64I32Vop3SdstEnc
                                                                  : rdna4::kVMadCoU64U32Vop3SdstEnc,
                                                   {.vdst = 2,
                                                    .sdst = 0,
                                                    .clamp = uint8_t(clamp),
                                                    .src0 = 256,
                                                    .src1 = 257,
                                                    .src2 = 258});
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              const uint64_t carry = test.carry ? active : 0;
              EXPECT_EQ(fixture.cu->read_sgpr(sb), uint32_t(carry));
              EXPECT_EQ(fixture.cu->read_sgpr(sb + 1),
                        lanes == 64 ? uint32_t(carry >> 32) : 0x12345678u);
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                const uint64_t expected =
                    ((active >> lane) & 1) ? (clamp ? test.saturated : test.wrapped) : test.c;
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lane), uint32_t(expected));
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 3, lane),
                          uint32_t(expected >> 32));
              }
            }
          }
        }
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

TEST(ArithBridgeTest, HalfBinarySelectorsModifiersAliasesAndObservations) {
  const uint16_t rdna3_ops[] = {rdna3::kVAddF16Vop3, rdna3::kVSubF16Vop3, rdna3::kVSubrevF16Vop3,
                                rdna3::kVMulF16Vop3};
  const uint16_t rdna4_ops[] = {rdna4::kVAddF16Vop3, rdna4::kVSubF16Vop3, rdna4::kVSubrevF16Vop3,
                                rdna4::kVMulF16Vop3};
  const uint32_t inputs[] = {0x4000bc00, 0xc0003800, 0xdeadbeef};
  for (bool is_rdna3 : {false, true})
    for (uint32_t lanes : {32u, 64u}) {
      BridgeFixture fixture(is_rdna3, lanes);
      const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000180000005ULL;
      fixture.wave->set_exec(exec_mask);
      fixture.wave->set_mode_raw(0x3c0); // Preserve FP16 denormals; IEEE and DX10_CLAMP enabled.
      for (unsigned op = 0; op < 4; ++op)
        for (unsigned controls = 0; controls < 32; ++controls)
          for (unsigned halves = 0; halves < 8; ++halves)
            for (uint8_t dest = 0; dest < 3; ++dest) {
              for (unsigned source = 0; source < 3; ++source)
                for (uint32_t lane = 0; lane < lanes; ++lane)
                  fixture.cu->write_vgpr(fixture.base() + source, lane, inputs[source]);
              const uint8_t opsel = uint8_t((halves & 3) | ((halves & 4) << 1));
              const uint8_t omod = uint8_t(controls & 3);
              const auto words =
                  is_rdna3
                      ? rdna3::build_vop3(rdna3_ops[op], {.vdst = dest,
                                                          .abs = uint8_t(controls & 3),
                                                          .op_sel = opsel,
                                                          .clamp = bool(controls & 16),
                                                          .src0 = 256,
                                                          .src1 = 257,
                                                          .omod = omod,
                                                          .neg = uint8_t((controls >> 2) & 3)})
                      : rdna4::build_vop3(rdna4_ops[op], {.vdst = dest,
                                                          .abs = uint8_t(controls & 3),
                                                          .opsel = opsel,
                                                          .clamp = bool(controls & 16),
                                                          .src0 = 256,
                                                          .src1 = 257,
                                                          .omod = omod,
                                                          .neg = uint8_t((controls >> 2) & 3)});
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              ASSERT_EQ(fixture.recorder->reads.size(), 2u);
              EXPECT_EQ(fixture.recorder->reads[0].bytes, halves & 1 ? 0xc : 0x3);
              EXPECT_EQ(fixture.recorder->reads[1].bytes, halves & 2 ? 0xc : 0x3);
              ASSERT_EQ(fixture.recorder->writes.size(), 1u);
              EXPECT_EQ(fixture.recorder->writes[0].bytes, halves & 4 ? 0xc : 0x3);
              float values[2];
              for (unsigned i = 0; i < 2; ++i) {
                values[i] =
                    util::f16_to_f32(uint16_t(inputs[i] >> ((halves & (1u << i)) ? 16 : 0)));
                if (controls & (1u << i))
                  values[i] = std::abs(values[i]);
                if (controls & (4u << i))
                  values[i] = -values[i];
              }
              float expected = op == 0   ? values[0] + values[1]
                               : op == 1 ? values[0] - values[1]
                               : op == 2 ? values[1] - values[0]
                                         : values[0] * values[1];
              if (!is_rdna3) {
                const float scales[] = {1, 2, 4, 0.5f};
                expected *= scales[omod];
              }
              if (controls & 16)
                expected = std::clamp(expected, 0.0f, 1.0f);
              const unsigned shift = halves & 4 ? 16 : 0;
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                const uint32_t actual = fixture.cu->read_vgpr(fixture.base() + dest, lane);
                EXPECT_EQ(actual & ~(0xffffu << shift), inputs[dest] & ~(0xffffu << shift));
                if ((exec_mask >> lane) & 1)
                  EXPECT_NEAR(util::f16_to_f32(uint16_t(actual >> shift)), expected,
                              1e-6f + 1e-6f * std::abs(expected));
                else
                  EXPECT_EQ(actual, inputs[dest]);
              }
              fixture.recorder->reads.clear();
              fixture.recorder->writes.clear();
            }
    }
}

TEST(ArithBridgeTest, HalfBinaryVop2InlineGuestRoundingAndRejection) {
  for (bool is_rdna3 : {false, true})
    for (uint32_t lanes : {32u, 64u})
      for (unsigned policy_index = 0; policy_index < 3; ++policy_index) {
        ArithmeticPolicy policy{};
        if (policy_index == 1) {
          policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
        } else if (policy_index == 2) {
          policy.semantics = ArithmeticSemantics::BitExactEmpirical;
        }
        BridgeFixture fixture(is_rdna3, lanes, policy);
        fixture.wave->set_exec(1);
        fixture.wave->set_mode_raw(0x3c0 | (1u << 2)); // Guest toward +infinity.
        fixture.wave->set_trapsts(0x10);
        fixture.cu->write_vgpr(fixture.base() + 1, 0, 1);
        fixture.cu->write_vgpr(fixture.base() + 2, 0, 0xdeadbeef);
        const auto words =
            is_rdna3 ? rdna3::build_vop2(rdna3::kVAddF16Vop2, {.src0 = 242, .vsrc1 = 1, .vdst = 2})
                     : rdna4::build_vop2(rdna4::kVAddF16Vop2, {.src0 = 242, .vsrc1 = 1, .vdst = 2});
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        const auto status =
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave);
        EXPECT_EQ(status.succeeded(), policy_index == 0);
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0),
                  policy_index == 0 ? 0xdead3c01U : 0xdeadbeefU);
        EXPECT_EQ(fixture.wave->trapsts(), 0x10U);
      }
}

// Finite outcomes select literal half inputs; NaN payloads remain unspecified.
TEST(ArithBridgeTest, HalfSelectionNanZeroDenormModesAndPartialWrites) {
  struct Operation {
    bool rdna3, maximum, propagate;
    uint16_t opcode;
  };
  const Operation operations[] = {
      {true, false, false, rdna3::kVMinF16Vop3},     {true, true, false, rdna3::kVMaxF16Vop3},
      {false, false, false, rdna4::kVMinNumF16Vop3}, {false, true, false, rdna4::kVMaxNumF16Vop3},
      {false, false, true, rdna4::kVMinimumF16Vop3}, {false, true, true, rdna4::kVMaximumF16Vop3},
  };
  struct Case {
    uint16_t a, b, minimum, maximum, flushed_minimum, flushed_maximum;
  };
  const Case cases[] = {
      {0x3c00, 0x4000, 0x3c00, 0x4000, 0x3c00, 0x4000},
      {0, 0x8000, 0x8000, 0, 0x8000, 0},
      {0x8000, 0, 0x8000, 0, 0x8000, 0},
      {0x7e55, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00},
      {0x7c55, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00},
      {0x3c00, 0x7e55, 0x3c00, 0x3c00, 0x3c00, 0x3c00},
      {0x7e55, 0x7c11, 0x7e00, 0x7e00, 0x7e00, 0x7e00},
      {1, 0, 0, 1, 0, 0},
      {0x8001, 0, 0x8001, 0, 0x8000, 0},
      {0x7c00, 0xfc00, 0xfc00, 0x7c00, 0xfc00, 0x7c00},
  };
  for (const auto &op : operations)
    for (uint32_t lanes : {32u, 64u}) {
      BridgeFixture fixture(op.rdna3, lanes);
      const auto words =
          op.rdna3
              ? rdna3::build_vop3(op.opcode, {.vdst = 0, .op_sel = 11, .src0 = 256, .src1 = 257})
              : rdna4::build_vop3(op.opcode, {.vdst = 0, .opsel = 11, .src0 = 256, .src1 = 257});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      for (unsigned denorm = 0; denorm < 4; ++denorm)
        for (bool ieee : {false, true})
          for (unsigned rounding = 0; rounding < 4; ++rounding) {
            fixture.wave->set_mode_raw((denorm << 6) | (rounding << 2) | (ieee ? 0x200 : 0));
            fixture.wave->set_trapsts(0x10);
            const uint64_t exec_mask = lanes == 32 ? 0xa55a5aa5ULL : 0xa55a5aa5a55a5aa5ULL;
            fixture.wave->set_exec(exec_mask);
            for (const auto &c : cases) {
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                fixture.cu->write_vgpr(fixture.base(), lane, (uint32_t(c.a) << 16) | 0x7c01U);
                fixture.cu->write_vgpr(fixture.base() + 1, lane, (uint32_t(c.b) << 16) | 0xdeadU);
              }
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              ASSERT_EQ(fixture.recorder->reads.size(), 2u);
              EXPECT_EQ(fixture.recorder->reads[0].bytes, 0xc);
              EXPECT_EQ(fixture.recorder->reads[1].bytes, 0xc);
              ASSERT_EQ(fixture.recorder->writes.size(), 1u);
              EXPECT_EQ(fixture.recorder->writes[0].bytes, 0xc);
              uint16_t expected = (denorm & 1)
                                      ? (op.maximum ? c.maximum : c.minimum)
                                      : (op.maximum ? c.flushed_maximum : c.flushed_minimum);
              const bool any_nan = (c.a & 0x7fff) > 0x7c00 || (c.b & 0x7fff) > 0x7c00;
              const bool signaling = ((c.a & 0x7fff) > 0x7c00 && !(c.a & 0x200)) ||
                                     ((c.b & 0x7fff) > 0x7c00 && !(c.b & 0x200));
              if ((op.propagate && any_nan) || (op.rdna3 && ieee && signaling))
                expected = 0x7e00;
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                const uint32_t actual = fixture.cu->read_vgpr(fixture.base(), lane);
                EXPECT_EQ(actual & 0xffffU, 0x7c01U);
                if (!((exec_mask >> lane) & 1))
                  EXPECT_EQ(actual >> 16, c.a);
                else if ((expected & 0x7fff) > 0x7c00)
                  EXPECT_TRUE(std::isnan(util::f16_to_f32(uint16_t(actual >> 16))));
                else
                  EXPECT_EQ(actual >> 16, expected);
              }
              EXPECT_EQ(fixture.wave->trapsts(), 0x10U);
              fixture.recorder->reads.clear();
              fixture.recorder->writes.clear();
            }
          }
    }
}

TEST(ArithBridgeTest, TrigEncodingsNanClampMasksAndOptionalState) {
  struct Case {
    uint16_t rdna3[2], rdna4[2];
    bool half, cosine;
  };
  const Case cases[] = {
      {{rdna3::kVSinF16Vop1, rdna3::kVSinF16Vop3},
       {rdna4::kVSinF16Vop1, rdna4::kVSinF16Vop3},
       true,
       false},
      {{rdna3::kVCosF16Vop1, rdna3::kVCosF16Vop3},
       {rdna4::kVCosF16Vop1, rdna4::kVCosF16Vop3},
       true,
       true},
      {{rdna3::kVSinF32Vop1, rdna3::kVSinF32Vop3},
       {rdna4::kVSinF32Vop1, rdna4::kVSinF32Vop3},
       false,
       false},
      {{rdna3::kVCosF32Vop1, rdna3::kVCosF32Vop3},
       {rdna4::kVCosF32Vop1, rdna4::kVCosF32Vop3},
       false,
       true},
  };
  for (const auto &c : cases)
    for (bool rdna3 : {false, true})
      for (uint32_t lanes : {32u, 64u})
        for (unsigned encoding = 0; encoding < 2; ++encoding)
          for (unsigned policy_index = 0; policy_index < 3; ++policy_index)
            for (bool ieee : {false, true})
              for (bool dx10 : {false, true})
                for (bool clamp : {false, true}) {
                  if (clamp && !encoding)
                    continue;
                  ArithmeticPolicy policy{};
                  if (policy_index == 1)
                    policy.semantics = ArithmeticSemantics::Loose;
                  else if (policy_index == 2)
                    if (policy.semantics != ArithmeticSemantics::Loose)
                      policy.semantics = ArithmeticSemantics::BitExactEmpirical;
                  BridgeFixture fixture(rdna3, lanes, policy);
                  fixture.wave->set_mode_raw(0xfc | (ieee ? 0x200 : 0) | (dx10 ? 0x100 : 0));
                  const unsigned shift = c.half && encoding ? 16 : 0;
                  auto packed = [&](uint32_t raw) {
                    return !c.half ? raw : shift ? (raw << 16) | 0xdeadU : 0xdead0000U | raw;
                  };
                  const uint32_t infinity = c.half ? 0x7c00 : 0x7f800000;
                  const uint32_t sign = c.half ? 0x8000 : 0x80000000;
                  const uint32_t quiet = c.half ? 0x200 : 0x400000;
                  const uint32_t one = c.half ? 0x3c00 : 0x3f800000;
                  const uint16_t opcode = rdna3 ? c.rdna3[encoding] : c.rdna4[encoding];
                  std::array<uint32_t, 2> words{};
                  if (encoding) {
                    words = rdna3 ? rdna3::build_vop3(opcode, {.vdst = 0,
                                                               .op_sel = uint8_t(c.half ? 9 : 0),
                                                               .clamp = uint8_t(clamp),
                                                               .src0 = 256})
                                  : rdna4::build_vop3(opcode, {.vdst = 0,
                                                               .opsel = uint8_t(c.half ? 9 : 0),
                                                               .clamp = uint8_t(clamp),
                                                               .src0 = 256});
                  } else {
                    words[0] = rdna3 ? rdna3::build_vop1(opcode, {.src0 = 256, .vdst = 0})[0]
                                     : rdna4::build_vop1(opcode, {.src0 = 256, .vdst = 0})[0];
                  }
                  auto instruction = fixture.decoder->decode_window(words);
                  ASSERT_TRUE(instruction.succeeded());
                  for (unsigned kind = 0; kind < 3; ++kind)
                    for (bool empty : {false, true}) {
                      SCOPED_TRACE(::testing::Message()
                                   << c.half << '/' << c.cosine << '/' << rdna3 << '/' << lanes
                                   << '/' << encoding << '/' << policy_index << '/' << ieee << '/'
                                   << dx10 << '/' << clamp << '/' << kind << '/' << empty);
                      const uint64_t exec_mask = empty         ? 0
                                                 : lanes == 32 ? 0x80000001ULL
                                                               : 0x8000000180000001ULL;
                      fixture.wave->set_exec(exec_mask);
                      fixture.wave->set_trapsts(0x10);
                      const uint32_t raw = kind == 0   ? 0
                                           : kind == 1 ? sign | infinity | 1
                                                       : infinity;
                      for (uint32_t lane = 0; lane < lanes; ++lane)
                        fixture.cu->write_vgpr(fixture.base(), lane, packed(raw));
                      fixture.recorder->writes.clear();
                      const bool rejected = !empty && policy_index == 2 && rdna3;
                      EXPECT_EQ(
                          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .failed(),
                          rejected);
                      uint64_t written = 0;
                      for (const auto &write : fixture.recorder->writes) {
                        EXPECT_EQ(write.reg, 0U);
                        EXPECT_EQ(write.bytes, c.half ? (shift ? 0xc : 3) : 0xf);
                        EXPECT_EQ(written & write.lanes, 0U);
                        written |= write.lanes;
                      }
                      EXPECT_EQ(written, rejected ? 0 : exec_mask);
                      uint32_t expected = kind == 0   ? (c.cosine ? one : 0)
                                          : kind == 1 ? raw | ((!rdna3 || ieee) ? quiet : 0)
                                                      : sign | infinity | quiet;
                      if (clamp && (!rdna3 || dx10) && kind)
                        expected = 0;
                      for (uint32_t lane = 0; lane < lanes; ++lane) {
                        const uint32_t actual = fixture.cu->read_vgpr(fixture.base(), lane);
                        if (rejected || !((exec_mask >> lane) & 1)) {
                          EXPECT_EQ(actual, packed(raw));
                        } else if (policy_index == 1 && (expected & ~sign) > infinity) {
                          EXPECT_GT((actual >> shift) & ~sign & (c.half ? 0xffffU : ~0U), infinity);
                          if (c.half) {
                            EXPECT_EQ(actual & (shift ? 0xffffU : 0xffff0000U),
                                      packed(raw) & (shift ? 0xffffU : 0xffff0000U));
                          }
                        } else {
                          EXPECT_EQ(actual, packed(expected));
                        }
                      }
                      const uint32_t causes =
                          !empty && !rejected && policy_index == 2 && kind && !clamp
                              ? RJ_ARITH_EXCEPTION_INVALID
                              : 0;
                      EXPECT_EQ(fixture.wave->trapsts(), 0x10U | causes);
                    }
                }
}

TEST(ArithBridgeTest, TrigDppFilteredLanesDoNotReportExceptions) {
  BridgeFixture fixture(false, 32, parse_arithmetic_policy("bit_exact_empirical"));
  fixture.wave->set_exec((1u << 0) | (1u << 16) | (1u << 17));
  fixture.wave->set_mode_raw(0xf0);
  fixture.cu->write_vgpr(fixture.base(), 0, 0x7f800001);  // Disabled row.
  fixture.cu->write_vgpr(fixture.base(), 15, 0x7f800001); // Invalid shift source.
  fixture.cu->write_vgpr(fixture.base(), 16, 0);
  rdna4::Vop1VopDpp16MachineInst raw{};
  raw.src0 = SRC_DPP;
  raw.vsrc0 = 0;
  raw.vdst = 1;
  raw.dpp_ctrl = dpp::ROW_SHR1;
  raw.fi = 1;
  raw.bound_ctrl = 0;
  raw.bank_mask = 0xf;
  raw.row_mask = 2;
  rdna4::VSinF32Vop1 instruction(reinterpret_cast<const rdna4::MachineInst *>(&raw));
  instruction.execute_impl(*fixture.wave);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  EXPECT_EQ(fixture.wave->trapsts(), 0U);
  ASSERT_EQ(fixture.recorder->writes.size(), 1U);
  EXPECT_EQ(fixture.recorder->writes[0].lanes, 1ULL << 17);
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 17), 0U);
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 0), 0x3c003c00U);
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 16), 0x3c003c00U);
}

// Exact binary fractions exercise the decoded adapter without introducing a new
// numerical model. Existing rj-arith and rocjitsu numerical suites retain the full oracles.
TEST(ArithBridgeTest, TernarySelectionWidthsModifiersAndState) {
  struct Case {
    bool rdna3;
    unsigned width;
    uint16_t opcode;
    unsigned operation;
  };
  const Case cases[] = {
      {true, 16, rdna3::kVMin3F16Vop3, 0},
      {true, 32, rdna3::kVMin3F32Vop3, 0},
      {true, 16, rdna3::kVMax3F16Vop3, 1},
      {true, 32, rdna3::kVMax3F32Vop3, 1},
      {true, 16, rdna3::kVMinmaxF16Vop3, 2},
      {true, 32, rdna3::kVMinmaxF32Vop3, 2},
      {true, 16, rdna3::kVMaxminF16Vop3, 3},
      {true, 32, rdna3::kVMaxminF32Vop3, 3},
      {true, 16, rdna3::kVMed3F16Vop3, 4},
      {true, 32, rdna3::kVMed3F32Vop3, 4},
      {false, 16, rdna4::kVMin3NumF16Vop3, 0},
      {false, 32, rdna4::kVMin3NumF32Vop3, 0},
      {false, 16, rdna4::kVMax3NumF16Vop3, 1},
      {false, 32, rdna4::kVMax3NumF32Vop3, 1},
      {false, 16, rdna4::kVMinmaxNumF16Vop3, 2},
      {false, 32, rdna4::kVMinmaxNumF32Vop3, 2},
      {false, 16, rdna4::kVMaxminNumF16Vop3, 3},
      {false, 32, rdna4::kVMaxminNumF32Vop3, 3},
      {false, 16, rdna4::kVMinimum3F16Vop3, 0},
      {false, 32, rdna4::kVMinimum3F32Vop3, 0},
      {false, 16, rdna4::kVMaximum3F16Vop3, 1},
      {false, 32, rdna4::kVMaximum3F32Vop3, 1},
      {false, 16, rdna4::kVMinimummaximumF16Vop3, 2},
      {false, 32, rdna4::kVMinimummaximumF32Vop3, 2},
      {false, 16, rdna4::kVMaximumminimumF16Vop3, 3},
      {false, 32, rdna4::kVMaximumminimumF32Vop3, 3},
      {false, 16, rdna4::kVMed3NumF16Vop3, 4},
      {false, 32, rdna4::kVMed3NumF32Vop3, 4},
  };
  // Binary fractions select/scale exactly. Tight atol/rtol expresses the loose
  // contract while catching selector, sign, and output-scaling mistakes.
  for (const auto &c : cases)
    for (unsigned lanes : {32u, 64u})
      for (unsigned halves = 0; halves < (c.width == 16 ? 16u : 1u); ++halves)
        for (unsigned settings = 0; settings < 5; ++settings)
          for (unsigned policy_index = 0; policy_index < 3; ++policy_index) {
            SCOPED_TRACE(::testing::Message() << c.rdna3 << '/' << c.width << '/' << c.opcode << '/'
                                              << halves << '/' << settings << '/' << policy_index);
            ArithmeticPolicy policy{};
            if (policy_index == 1)
              policy.semantics = ArithmeticSemantics::Loose;
            if (policy_index == 2)
              if (policy.semantics != ArithmeticSemantics::Loose)
                policy.semantics = ArithmeticSemantics::BitExactEmpirical;
            BridgeFixture fixture(c.rdna3, lanes, policy);
            const bool nan = settings == 2 || settings == 3;
            const bool negate = settings == 1;
            fixture.wave->set_mode_raw((negate ? 0u : 0x2f0u) | 0xfu |
                                       (settings == 2 ? 0x100u : 0));
            fixture.wave->set_trapsts(0x10);
            const uint64_t exec_mask = settings == 4 ? 0
                                       : lanes == 32 ? 0x80000001ULL
                                                     : 0x8000000180000001ULL;
            fixture.wave->set_exec(exec_mask);
            const float inputs[] = {-0.75f, -0.25f, -1.0f};
            uint32_t original = 0;
            for (unsigned reg = 0; reg < 3; ++reg) {
              uint32_t word = nan ? 0xffc12345U : std::bit_cast<uint32_t>(inputs[reg]);
              if (c.width == 16) {
                const uint16_t half = nan ? 0xfe15 : util::f32_to_f16(inputs[reg]);
                word = halves & (1u << reg) ? (uint32_t(half) << 16) | 0xdead : 0xdead0000U | half;
              }
              for (unsigned lane = 0; lane < lanes; ++lane)
                fixture.cu->write_vgpr(fixture.base() + reg, lane, word);
              if (reg == 2)
                original = word;
            }
            auto words = c.rdna3 ? rdna3::build_vop3(c.opcode, {.vdst = 2,
                                                                .abs = 7,
                                                                .op_sel = uint8_t(halves),
                                                                .clamp = uint8_t(nan),
                                                                .src0 = 256,
                                                                .src1 = 257,
                                                                .src2 = 258,
                                                                .omod = uint8_t(negate),
                                                                .neg = uint8_t(negate ? 7 : 0)})
                                 : rdna4::build_vop3(c.opcode, {.vdst = 2,
                                                                .abs = 7,
                                                                .opsel = uint8_t(halves),
                                                                .clamp = uint8_t(nan),
                                                                .src0 = 256,
                                                                .src1 = 257,
                                                                .src2 = 258,
                                                                .omod = uint8_t(negate),
                                                                .neg = uint8_t(negate ? 7 : 0)});
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            const bool rejected = policy_index == 2 && exec_mask;
            EXPECT_EQ(
                fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).failed(),
                rejected);
            const unsigned d_shift = halves & 8 ? 16 : 0;
            uint64_t written = 0;
            for (const auto &write : fixture.recorder->writes) {
              EXPECT_EQ(write.reg, 2u);
              EXPECT_EQ(write.bytes, c.width == 32 ? 0xf : d_shift ? 0xc : 0x3);
              EXPECT_EQ(written & write.lanes, 0u);
              written |= write.lanes;
            }
            EXPECT_EQ(written, rejected ? 0 : exec_mask);
            const float positive[] = {0.25f, 1.0f, 1.0f, 0.75f, 0.75f};
            const float negative[] = {-1.0f, -0.25f, -0.75f, -1.0f, -0.75f};
            const float expected = negate ? 2 * negative[c.operation] : positive[c.operation];
            for (unsigned lane = 0; lane < lanes; ++lane) {
              const uint32_t actual = fixture.cu->read_vgpr(fixture.base() + 2, lane);
              if (rejected || !((exec_mask >> lane) & 1)) {
                EXPECT_EQ(actual, original);
                continue;
              }
              if (c.width == 16) {
                EXPECT_EQ((actual ^ original) & ~(0xffffU << d_shift), 0u);
              }
              const float value = c.width == 16 ? util::f16_to_f32(uint16_t(actual >> d_shift))
                                                : std::bit_cast<float>(actual);
              if (nan && c.rdna3 && settings == 3)
                EXPECT_TRUE(std::isnan(value));
              else if (nan)
                EXPECT_EQ(value, 0.0f);
              else
                EXPECT_NEAR(value, expected, 1e-6f + 1e-6f * std::abs(expected));
            }
            EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
          }
}

TEST(ArithBridgeTest, HalfLdexpGuestRoundingSelectorsAndIntegerExponent) {
  struct Case {
    uint16_t input, exponent, expected[4];
    uint8_t omod;
  };
  // Directed ties and overflow before OMOD: nearest/upward overflow stays
  // infinite; downward/toward-zero first rounds to max-finite, then halves it.
  const Case cases[] = {
      {0x0401, 0xffff, {0x0200, 0x0201, 0x0200, 0x0200}, 0},
      {0x8401, 0xffff, {0x8200, 0x8200, 0x8201, 0x8200}, 0},
      {0x7bff, 1, {0x7c00, 0x7c00, 0x77ff, 0x77ff}, 3},
  };
  for (bool rdna3 : {false, true})
    for (unsigned lanes : {32u, 64u})
      for (bool vop3 : {false, true})
        for (unsigned halves = 0; halves < (vop3 ? 8u : 1u); ++halves)
          for (unsigned rounding = 0; rounding < 4; ++rounding)
            for (unsigned policy_index = 0; policy_index < 3; ++policy_index)
              for (const auto &c : cases) {
                if (!vop3 && c.omod)
                  continue;
                ArithmeticPolicy policy{};
                if (policy_index == 1)
                  policy.semantics = ArithmeticSemantics::Loose;
                if (policy_index == 2)
                  if (policy.semantics != ArithmeticSemantics::Loose)
                    policy.semantics = ArithmeticSemantics::BitExactEmpirical;
                BridgeFixture fixture(rdna3, lanes, policy);
                // Output flush with IEEE=0 enables RDNA3 OMOD; otherwise
                // preserve denormals to observe directed halfway rounding.
                fixture.wave->set_mode_raw((c.omod ? 0u : 0xc0u) | (rounding << 2));
                fixture.wave->set_trapsts(0x10);
                const uint64_t exec_mask = lanes == 32 ? 0x80000001ULL : 0x8000000180000001ULL;
                fixture.wave->set_exec(exec_mask);
                const unsigned a_shift = halves & 1 ? 16 : 0;
                const unsigned b_shift = halves & 2 ? 16 : 0;
                const unsigned d_shift = halves & 4 ? 16 : 0;
                const uint32_t input =
                    a_shift ? (uint32_t(c.input) << 16) | 0xbeef : 0xbeef0000U | c.input;
                const uint32_t original =
                    b_shift ? (uint32_t(c.exponent) << 16) | 0xdead : 0xdead0000U | c.exponent;
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  fixture.cu->write_vgpr(fixture.base(), lane, input);
                  fixture.cu->write_vgpr(fixture.base() + 1, lane, original);
                }
                std::array<uint32_t, 2> words{};
                if (vop3) {
                  const uint8_t opsel = uint8_t((halves & 3) | ((halves & 4) << 1));
                  // B is a signed integer: its ABS/NEG encoding bits are ignored.
                  words = rdna3 ? rdna3::build_vop3(rdna3::kVLdexpF16Vop3, {.vdst = 1,
                                                                            .abs = 2,
                                                                            .op_sel = opsel,
                                                                            .src0 = 256,
                                                                            .src1 = 257,
                                                                            .omod = uint8_t(c.omod),
                                                                            .neg = 2})
                                : rdna4::build_vop3(rdna4::kVLdexpF16Vop3, {.vdst = 1,
                                                                            .abs = 2,
                                                                            .opsel = opsel,
                                                                            .src0 = 256,
                                                                            .src1 = 257,
                                                                            .omod = uint8_t(c.omod),
                                                                            .neg = 2});
                } else {
                  words[0] = rdna3 ? rdna3::build_vop2(rdna3::kVLdexpF16Vop2,
                                                       {.src0 = 256, .vsrc1 = 1, .vdst = 1})[0]
                                   : rdna4::build_vop2(rdna4::kVLdexpF16Vop2,
                                                       {.src0 = 256, .vsrc1 = 1, .vdst = 1})[0];
                }
                auto instruction = fixture.decoder->decode_window(words);
                ASSERT_TRUE(instruction.succeeded());
                const bool rejected = policy_index == 2;
                EXPECT_EQ(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .failed(),
                          rejected);
                uint64_t written = 0;
                for (const auto &write : fixture.recorder->writes) {
                  EXPECT_EQ(write.reg, 1u);
                  EXPECT_EQ(write.bytes, d_shift ? 0xc : 0x3);
                  EXPECT_EQ(written & write.lanes, 0u);
                  written |= write.lanes;
                }
                EXPECT_EQ(written, rejected ? 0 : exec_mask);
                const uint16_t result = c.expected[rounding];
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  const uint32_t expected =
                      !rejected && ((exec_mask >> lane) & 1)
                          ? (original & ~(0xffffU << d_shift)) | (uint32_t(result) << d_shift)
                          : original;
                  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, lane), expected);
                }
                EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
              }
}

// The ISA destination is I16, not a floating-point encoding of the exponent.
TEST(ArithBridgeTest, DoubleFrexpExponentWritesOneIntegerWord) {
  const uint64_t inputs[] = {0x3ff0000000000000ULL, 0x3fd0000000000000ULL, 1,
                             0xfffffffffffffULL,    0x7fefffffffffffffULL, 0x8000000000000000ULL,
                             0x7ff0000000000000ULL, 0x7ff0000000000001ULL};
  const int32_t expected[] = {1, -1, 0, 0, 1024, 0, 0, 0};
  for (bool rdna3 : {false, true})
    for (unsigned lanes : {32u, 64u})
      for (bool vop3 : {false, true})
        for (unsigned policy_index = 0; policy_index < 3; ++policy_index)
          for (bool empty : {false, true}) {
            ArithmeticPolicy policy{};
            if (policy_index == 1)
              policy.semantics = ArithmeticSemantics::Loose;
            if (policy_index == 2)
              if (policy.semantics != ArithmeticSemantics::Loose)
                policy.semantics = ArithmeticSemantics::BitExactEmpirical;
            BridgeFixture fixture(rdna3, lanes, policy);
            // Input flushing makes subnormal exponents zero.
            fixture.wave->set_mode_raw(3u << 2);
            fixture.wave->set_trapsts(0x10);
            const uint64_t exec_mask = empty         ? 0
                                       : lanes == 32 ? 0x800000ffULL
                                                     : 0x800000ff800000ffULL;
            fixture.wave->set_exec(exec_mask);
            for (unsigned lane = 0; lane < lanes; ++lane) {
              fixture.cu->write_vgpr(fixture.base(), lane, uint32_t(inputs[lane % 8]));
              fixture.cu->write_vgpr(fixture.base() + 1, lane, uint32_t(inputs[lane % 8] >> 32));
            }
            std::array<uint32_t, 2> words{};
            if (vop3)
              words =
                  rdna3 ? rdna3::build_vop3(
                              rdna3::kVFrexpExpI32F64Vop3,
                              {.vdst = 0, .abs = 1, .clamp = 1, .src0 = 256, .omod = 3, .neg = 1})
                        : rdna4::build_vop3(
                              rdna4::kVFrexpExpI32F64Vop3,
                              {.vdst = 0, .abs = 1, .clamp = 1, .src0 = 256, .omod = 3, .neg = 1});
            else
              words[0] =
                  rdna3
                      ? rdna3::build_vop1(rdna3::kVFrexpExpI32F64Vop1, {.src0 = 256, .vdst = 0})[0]
                      : rdna4::build_vop1(rdna4::kVFrexpExpI32F64Vop1, {.src0 = 256, .vdst = 0})[0];
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            fixture.recorder->writes.clear();
            const bool rejected = policy_index == 2 && !empty;
            EXPECT_EQ(
                fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).failed(),
                rejected);
            uint64_t written = 0;
            for (const auto &write : fixture.recorder->writes) {
              EXPECT_EQ(write.reg, 0u);
              EXPECT_EQ(write.bytes, 0xf);
              written |= write.lanes;
            }
            EXPECT_EQ(written, rejected ? 0 : exec_mask);
            for (unsigned lane = 0; lane < lanes; ++lane) {
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                        !rejected && ((exec_mask >> lane) & 1) ? uint32_t(expected[lane % 8])
                                                               : uint32_t(inputs[lane % 8]));
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, lane),
                        uint32_t(inputs[lane % 8] >> 32));
            }
            EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
          }
}

TEST(ArithBridgeTest, DoubleUnaryEncodingsPairAliasesAndState) {
  struct Case {
    uint16_t op3[2], op4[2];
    double result;
  };
  const Case cases[] = {
      {{rdna3::kVTruncF64Vop1, rdna3::kVTruncF64Vop3},
       {rdna4::kVTruncF64Vop1, rdna4::kVTruncF64Vop3},
       1.0},
      {{rdna3::kVCeilF64Vop1, rdna3::kVCeilF64Vop3},
       {rdna4::kVCeilF64Vop1, rdna4::kVCeilF64Vop3},
       2.0},
      {{rdna3::kVRndneF64Vop1, rdna3::kVRndneF64Vop3},
       {rdna4::kVRndneF64Vop1, rdna4::kVRndneF64Vop3},
       2.0},
      {{rdna3::kVFloorF64Vop1, rdna3::kVFloorF64Vop3},
       {rdna4::kVFloorF64Vop1, rdna4::kVFloorF64Vop3},
       1.0},
      {{rdna3::kVFractF64Vop1, rdna3::kVFractF64Vop3},
       {rdna4::kVFractF64Vop1, rdna4::kVFractF64Vop3},
       0.5},
      {{rdna3::kVFrexpMantF64Vop1, rdna3::kVFrexpMantF64Vop3},
       {rdna4::kVFrexpMantF64Vop1, rdna4::kVFrexpMantF64Vop3},
       0.75},
      {{rdna3::kVRcpF64Vop1, rdna3::kVRcpF64Vop3},
       {rdna4::kVRcpF64Vop1, rdna4::kVRcpF64Vop3},
       1.0 / 1.5},
      {{rdna3::kVRsqF64Vop1, rdna3::kVRsqF64Vop3},
       {rdna4::kVRsqF64Vop1, rdna4::kVRsqF64Vop3},
       1.0 / std::sqrt(1.5)},
      {{rdna3::kVSqrtF64Vop1, rdna3::kVSqrtF64Vop3},
       {rdna4::kVSqrtF64Vop1, rdna4::kVSqrtF64Vop3},
       std::sqrt(1.5)},
  };
  for (bool rdna3 : {false, true})
    for (unsigned lanes : {32u, 64u})
      for (bool vop3 : {false, true})
        for (unsigned policy_index = 0; policy_index < 3; ++policy_index)
          for (unsigned guest = 0; guest < 4; ++guest)
            for (const auto &test : cases) {
              ArithmeticPolicy policy{};
              if (policy_index == 1)
                policy.semantics = ArithmeticSemantics::Loose;
              if (policy_index == 2)
                if (policy.semantics != ArithmeticSemantics::Loose)
                  policy.semantics = ArithmeticSemantics::BitExactEmpirical;
              BridgeFixture fixture(rdna3, lanes, policy);
              // These unary operations ignore guest rounding mode.
              // Alternate denormal preservation to exercise RDNA3 OMOD gating.
              fixture.wave->set_mode_raw((guest << 2) | ((guest & 1) ? 3u << 6 : 0));
              fixture.wave->set_trapsts(0x10);
              const uint64_t exec_mask = lanes == 32 ? 0x80000003ULL : 0x8000000380000003ULL;
              fixture.wave->set_exec(exec_mask);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                fixture.cu->write_vgpr(fixture.base(), lane, 0);
                fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x3ff80000); // 1.5
              }
              std::array<uint32_t, 2> words{};
              const auto *op = rdna3 ? test.op3 : test.op4;
              if (vop3)
                words = rdna3 ? rdna3::build_vop3(op[1], {.vdst = 0, .src0 = 256, .omod = 1})
                              : rdna4::build_vop3(op[1], {.vdst = 0, .src0 = 256, .omod = 1});
              else
                words[0] = rdna3 ? rdna3::build_vop1(op[0], {.src0 = 256, .vdst = 0})[0]
                                 : rdna4::build_vop1(op[0], {.src0 = 256, .vdst = 0})[0];
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              fixture.recorder->writes.clear();
              const bool rejected = policy_index == 2;
              EXPECT_EQ(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .failed(),
                        rejected);
              uint64_t written[2] = {};
              for (const auto &write : fixture.recorder->writes) {
                ASSERT_LT(write.reg, 2u);
                EXPECT_EQ(write.bytes, 0xf);
                written[write.reg] |= write.lanes;
              }
              EXPECT_EQ(written[0], rejected ? 0 : exec_mask);
              EXPECT_EQ(written[1], rejected ? 0 : exec_mask);
              const double result = test.result * (vop3 && (!rdna3 || !(guest & 1)) ? 2 : 1);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                const uint64_t actual =
                    fixture.cu->read_vgpr(fixture.base(), lane) |
                    (uint64_t(fixture.cu->read_vgpr(fixture.base() + 1, lane)) << 32);
                const double expected = !rejected && ((exec_mask >> lane) & 1) ? result : 1.5;
                EXPECT_NEAR(std::bit_cast<double>(actual), expected,
                            1e-12 + 1e-12 * std::abs(expected));
              }
              EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
            }
}

TEST(ArithBridgeTest, DoubleLdexpGuestRoundingIntegerExponentAndState) {
  const uint64_t inputs[] = {1, 0x8000000000000001ULL, 0x7fefffffffffffffULL,
                             0x001fffffffffffffULL};
  const int32_t exponents[] = {-1, -1, 1, -1};
  const uint64_t results[4][4] = {
      {0, 1, 0, 0},
      {0x8000000000000000ULL, 0x8000000000000000ULL, 0x8000000000000001ULL, 0x8000000000000000ULL},
      {0x7ff0000000000000ULL, 0x7ff0000000000000ULL, 0x7fefffffffffffffULL, 0x7fefffffffffffffULL},
      {0x0010000000000000ULL, 0x0010000000000000ULL, 0x000fffffffffffffULL, 0x000fffffffffffffULL},
  };
  for (bool rdna3 : {false, true})
    for (unsigned lanes : {32u, 64u})
      for (unsigned rounding = 0; rounding < 4; ++rounding)
        for (unsigned policy_index = 0; policy_index < 3; ++policy_index)
          for (bool empty : {false, true}) {
            ArithmeticPolicy policy{};
            if (policy_index == 1)
              policy.semantics = ArithmeticSemantics::Loose;
            if (policy_index == 2)
              if (policy.semantics != ArithmeticSemantics::Loose)
                policy.semantics = ArithmeticSemantics::BitExactEmpirical;
            BridgeFixture fixture(rdna3, lanes, policy);
            fixture.wave->set_mode_raw((3u << 6) | (rounding << 2));
            fixture.wave->set_trapsts(0x10);
            const uint64_t exec_mask = empty         ? 0
                                       : lanes == 32 ? 0x800000ffULL
                                                     : 0x800000ff800000ffULL;
            fixture.wave->set_exec(exec_mask);
            for (unsigned lane = 0; lane < lanes; ++lane) {
              fixture.cu->write_vgpr(fixture.base(), lane, uint32_t(inputs[lane % 4]));
              fixture.cu->write_vgpr(fixture.base() + 1, lane, uint32_t(inputs[lane % 4] >> 32));
              fixture.cu->write_vgpr(fixture.base() + 2, lane, uint32_t(exponents[lane % 4]));
            }
            // B is one integer word and aliases the low destination word.
            const auto words =
                rdna3
                    ? rdna3::build_vop3(rdna3::kVLdexpF64Vop3,
                                        {.vdst = 2, .abs = 2, .src0 = 256, .src1 = 258, .neg = 2})
                    : rdna4::build_vop3(rdna4::kVLdexpF64Vop3,
                                        {.vdst = 2, .abs = 2, .src0 = 256, .src1 = 258, .neg = 2});
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            fixture.recorder->writes.clear();
            const bool rejected = policy_index == 2 && !empty;
            EXPECT_EQ(
                fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).failed(),
                rejected);
            uint64_t written[2] = {};
            for (const auto &write : fixture.recorder->writes) {
              ASSERT_GE(write.reg, 2u);
              ASSERT_LE(write.reg, 3u);
              EXPECT_EQ(write.bytes, 0xf);
              written[write.reg - 2] |= write.lanes;
            }
            EXPECT_EQ(written[0], rejected ? 0 : exec_mask);
            EXPECT_EQ(written[1], rejected ? 0 : exec_mask);
            for (unsigned lane = 0; lane < lanes; ++lane) {
              const uint64_t expected = !rejected && ((exec_mask >> lane) & 1)
                                            ? results[lane % 4][rounding]
                                            : (0x3c003c00ULL << 32) | uint32_t(exponents[lane % 4]);
              const uint64_t actual =
                  fixture.cu->read_vgpr(fixture.base() + 2, lane) |
                  (uint64_t(fixture.cu->read_vgpr(fixture.base() + 3, lane)) << 32);
              EXPECT_EQ(actual, expected)
                  << rdna3 << '/' << lanes << '/' << rounding << '/' << policy_index;
            }
            EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
          }
}

TEST(ArithBridgeTest, FloatLdexpGuestRoundingIntegerExponentAndState) {
  const uint32_t inputs[] = {1, 0x80000001, 0x7f7fffff, 0x00ffffff};
  const int32_t exponents[] = {-1, -1, 1, -1};
  const uint32_t results[4][4] = {
      {0, 1, 0, 0},
      {0x80000000, 0x80000000, 0x80000001, 0x80000000},
      {0x7f800000, 0x7f800000, 0x7f7fffff, 0x7f7fffff},
      {0x00800000, 0x00800000, 0x007fffff, 0x007fffff},
  };
  for (bool rdna3 : {false, true})
    for (unsigned lanes : {32u, 64u})
      for (unsigned rounding = 0; rounding < 4; ++rounding)
        for (unsigned policy_index = 0; policy_index < 3; ++policy_index)
          for (bool empty : {false, true}) {
            ArithmeticPolicy policy{};
            if (policy_index == 1)
              policy.semantics = ArithmeticSemantics::Loose;
            if (policy_index == 2)
              if (policy.semantics != ArithmeticSemantics::Loose)
                policy.semantics = ArithmeticSemantics::BitExactEmpirical;
            BridgeFixture fixture(rdna3, lanes, policy);
            fixture.wave->set_mode_raw((3u << 4) | rounding);
            fixture.wave->set_trapsts(0x10);
            const uint64_t exec_mask = empty         ? 0
                                       : lanes == 32 ? 0x800000ffULL
                                                     : 0x800000ff800000ffULL;
            fixture.wave->set_exec(exec_mask);
            for (unsigned lane = 0; lane < lanes; ++lane) {
              fixture.cu->write_vgpr(fixture.base(), lane, inputs[lane % 4]);
              fixture.cu->write_vgpr(fixture.base() + 1, lane, uint32_t(exponents[lane % 4]));
            }
            // B is signed integer data; its ABS/NEG bits are ignored. Alias D/B.
            const auto words =
                rdna3
                    ? rdna3::build_vop3(rdna3::kVLdexpF32Vop3,
                                        {.vdst = 1, .abs = 2, .src0 = 256, .src1 = 257, .neg = 2})
                    : rdna4::build_vop3(rdna4::kVLdexpF32Vop3,
                                        {.vdst = 1, .abs = 2, .src0 = 256, .src1 = 257, .neg = 2});
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            fixture.recorder->writes.clear();
            const bool rejected = policy_index == 2 && !empty;
            EXPECT_EQ(
                fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).failed(),
                rejected);
            uint64_t written = 0;
            for (const auto &write : fixture.recorder->writes) {
              EXPECT_EQ(write.reg, 1u);
              EXPECT_EQ(write.bytes, 0xf);
              written |= write.lanes;
            }
            EXPECT_EQ(written, rejected ? 0 : exec_mask);
            for (unsigned lane = 0; lane < lanes; ++lane) {
              const uint32_t expected = !rejected && ((exec_mask >> lane) & 1)
                                            ? results[lane % 4][rounding]
                                            : uint32_t(exponents[lane % 4]);
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, lane), expected)
                  << rdna3 << '/' << lanes << '/' << rounding << '/' << policy_index << '/' << lane;
            }
            EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
          }
}

TEST(ArithBridgeTest, FloatFrexpEncodingsGuestDenormsAliasesAndState) {
  // Values inherited from the integer frexp_f32 model in shared/fp_mode.h.
  const uint32_t inputs[] = {0x3f800000, 0xbe800000, 1,          0x007fffff,
                             0x7f7fffff, 0x80000000, 0x7f800000, 0x7f800001};
  const int32_t exponents[] = {1, -1, -148, -126, 128, 0, 0, 0};
  const uint32_t mantissas[] = {0x3f000000, 0xbf000000, 0x3f000000, 0x3f7ffffe,
                                0x3f7fffff, 0x80000000, 0x7f800000, 0x7fc00001};
  for (bool rdna3 : {false, true})
    for (unsigned lanes : {32u, 64u})
      for (bool exponent : {false, true})
        for (bool vop3 : {false, true})
          for (bool preserve : {false, true})
            for (unsigned policy_index = 0; policy_index < 3; ++policy_index) {
              ArithmeticPolicy policy{};
              if (policy_index == 1)
                policy.semantics = ArithmeticSemantics::Loose;
              if (policy_index == 2)
                if (policy.semantics != ArithmeticSemantics::Loose)
                  policy.semantics = ArithmeticSemantics::BitExactEmpirical;
              BridgeFixture fixture(rdna3, lanes, policy);
              fixture.wave->set_mode_raw(preserve ? 3u << 4 : 0u);
              fixture.wave->set_trapsts(0x10);
              const uint64_t exec_mask = lanes == 32 ? 0x800000ffULL : 0x800000ff800000ffULL;
              fixture.wave->set_exec(exec_mask);
              for (unsigned lane = 0; lane < lanes; ++lane)
                fixture.cu->write_vgpr(fixture.base(), lane, inputs[lane % 8]);
              fixture.recorder->writes.clear();
              std::array<uint32_t, 2> words{};
              if (vop3) {
                words = rdna3 ? rdna3::build_vop3(exponent ? rdna3::kVFrexpExpI32F32Vop3
                                                           : rdna3::kVFrexpMantF32Vop3,
                                                  {.vdst = 0, .src0 = 256})
                              : rdna4::build_vop3(exponent ? rdna4::kVFrexpExpI32F32Vop3
                                                           : rdna4::kVFrexpMantF32Vop3,
                                                  {.vdst = 0, .src0 = 256});
              } else {
                words[0] = rdna3 ? rdna3::build_vop1(exponent ? rdna3::kVFrexpExpI32F32Vop1
                                                              : rdna3::kVFrexpMantF32Vop1,
                                                     {.src0 = 256, .vdst = 0})[0]
                                 : rdna4::build_vop1(exponent ? rdna4::kVFrexpExpI32F32Vop1
                                                              : rdna4::kVFrexpMantF32Vop1,
                                                     {.src0 = 256, .vdst = 0})[0];
              }
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              const bool rejected = policy_index == 2;
              EXPECT_EQ(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .failed(),
                        rejected);
              uint64_t written = 0;
              for (const auto &write : fixture.recorder->writes) {
                EXPECT_EQ(write.reg, 0u);
                EXPECT_EQ(write.bytes, 0xf);
                written |= write.lanes;
              }
              EXPECT_EQ(written, rejected ? 0 : exec_mask);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                unsigned index = lane % 8;
                uint32_t expected = inputs[index];
                if (!rejected && ((exec_mask >> lane) & 1)) {
                  expected = exponent ? uint32_t(exponents[index]) : mantissas[index];
                  if (!preserve && (index == 2 || index == 3))
                    expected = 0;
                }
                const uint32_t actual = fixture.cu->read_vgpr(fixture.base(), lane);
                // Loose mantissas need not preserve NaN payloads or quiet bits.
                // Finite references here are exactly representable; use the
                // ordinary atol/rtol criterion for the loose floating result.
                if (policy_index == 1 && !exponent && ((exec_mask >> lane) & 1)) {
                  const float value = std::bit_cast<float>(actual);
                  const float reference = std::bit_cast<float>(expected);
                  if (std::isnan(reference)) {
                    EXPECT_TRUE(std::isnan(value));
                  } else if (std::isinf(reference)) {
                    EXPECT_EQ(value, reference);
                  } else {
                    EXPECT_NEAR(value, reference, 1e-6f + 1e-6f * std::abs(reference));
                  }
                  continue;
                }
                EXPECT_EQ(actual, expected)
                    << rdna3 << '/' << lanes << '/' << exponent << '/' << vop3 << '/' << preserve
                    << '/' << policy_index << '/' << lane;
              }
              EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
            }
}

TEST(ArithBridgeTest, HalfFrexpExponentIntegerOutputAndState) {
  const uint16_t inputs[] = {0x3c00, 0x3400, 1, 0x3ff, 0x7bff, 0x8000, 0x7c00, 0x7c01};
  const uint16_t results[] = {1, uint16_t(-1), 0, 0, 16, 0, 0, 0};
  for (bool rdna3 : {false, true})
    for (unsigned lanes : {32u, 64u})
      for (bool vop3 : {false, true})
        for (unsigned halves = 0; halves < (vop3 ? 4u : 1u); ++halves)
          for (unsigned policy_index = 0; policy_index < 3; ++policy_index)
            for (bool empty : {false, true}) {
              ArithmeticPolicy policy{};
              if (policy_index == 1)
                policy.semantics = ArithmeticSemantics::Loose;
              if (policy_index == 2)
                if (policy.semantics != ArithmeticSemantics::Loose)
                  policy.semantics = ArithmeticSemantics::BitExactEmpirical;
              BridgeFixture fixture(rdna3, lanes, policy);
              // Input flushing makes subnormal exponents zero. Other FP controls
              // do not modify the integer result.
              fixture.wave->set_mode_raw(3u << 2);
              fixture.wave->set_trapsts(0x10);
              const uint64_t exec_mask = empty         ? 0
                                         : lanes == 32 ? 0x800000ffULL
                                                       : 0x800000ff800000ffULL;
              fixture.wave->set_exec(exec_mask);
              const unsigned a_shift = halves & 1 ? 16 : 0;
              const unsigned d_shift = halves & 2 ? 16 : 0;
              uint32_t original[64];
              for (unsigned lane = 0; lane < lanes; ++lane) {
                original[lane] = a_shift ? (uint32_t(inputs[lane % 8]) << 16) | 0xbeef
                                         : 0xbeef0000U | inputs[lane % 8];
                fixture.cu->write_vgpr(fixture.base(), lane, original[lane]);
              }
              std::array<uint32_t, 2> words{};
              if (vop3) {
                const uint8_t opsel = uint8_t((halves & 1) | ((halves & 2) << 2));
                words = rdna3 ? rdna3::build_vop3(rdna3::kVFrexpExpI16F16Vop3, {.vdst = 0,
                                                                                .abs = 1,
                                                                                .op_sel = opsel,
                                                                                .clamp = 1,
                                                                                .src0 = 256,
                                                                                .omod = 3,
                                                                                .neg = 1})
                              : rdna4::build_vop3(rdna4::kVFrexpExpI16F16Vop3, {.vdst = 0,
                                                                                .abs = 1,
                                                                                .opsel = opsel,
                                                                                .clamp = 1,
                                                                                .src0 = 256,
                                                                                .omod = 3,
                                                                                .neg = 1});
              } else {
                words[0] = rdna3 ? rdna3::build_vop1(rdna3::kVFrexpExpI16F16Vop1,
                                                     {.src0 = 256, .vdst = 0})[0]
                                 : rdna4::build_vop1(rdna4::kVFrexpExpI16F16Vop1,
                                                     {.src0 = 256, .vdst = 0})[0];
              }
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              const bool rejected = policy_index == 2 && !empty;
              EXPECT_EQ(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .failed(),
                        rejected);
              uint64_t written = 0;
              for (const auto &write : fixture.recorder->writes) {
                EXPECT_EQ(write.reg, 0u);
                EXPECT_EQ(write.bytes, d_shift ? 0xc : 0x3);
                EXPECT_EQ(written & write.lanes, 0u);
                written |= write.lanes;
              }
              EXPECT_EQ(written, rejected ? 0 : exec_mask);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                uint32_t expected = original[lane];
                if (!rejected && ((exec_mask >> lane) & 1))
                  expected =
                      (expected & ~(0xffffU << d_shift)) | (uint32_t(results[lane % 8]) << d_shift);
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), expected);
              }
              EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
            }
}

TEST(ArithBridgeTest, HalfUnaryEncodingsSelectorsFixedRoundingAndState) {
  struct Case {
    uint16_t rdna3[2], rdna4[2];
    uint16_t input;
    float result;
  };
  const Case cases[] = {
      {{rdna3::kVTruncF16Vop1, rdna3::kVTruncF16Vop3},
       {rdna4::kVTruncF16Vop1, rdna4::kVTruncF16Vop3},
       0x3e00,
       1.0f},
      {{rdna3::kVCeilF16Vop1, rdna3::kVCeilF16Vop3},
       {rdna4::kVCeilF16Vop1, rdna4::kVCeilF16Vop3},
       0x3e00,
       2.0f},
      {{rdna3::kVRndneF16Vop1, rdna3::kVRndneF16Vop3},
       {rdna4::kVRndneF16Vop1, rdna4::kVRndneF16Vop3},
       0x3e00,
       2.0f},
      {{rdna3::kVFloorF16Vop1, rdna3::kVFloorF16Vop3},
       {rdna4::kVFloorF16Vop1, rdna4::kVFloorF16Vop3},
       0x3e00,
       1.0f},
      {{rdna3::kVRcpF16Vop1, rdna3::kVRcpF16Vop3},
       {rdna4::kVRcpF16Vop1, rdna4::kVRcpF16Vop3},
       0x4400,
       0.25f},
      {{rdna3::kVRsqF16Vop1, rdna3::kVRsqF16Vop3},
       {rdna4::kVRsqF16Vop1, rdna4::kVRsqF16Vop3},
       0x4400,
       0.5f},
      {{rdna3::kVSqrtF16Vop1, rdna3::kVSqrtF16Vop3},
       {rdna4::kVSqrtF16Vop1, rdna4::kVSqrtF16Vop3},
       0x4400,
       2.0f},
      {{rdna3::kVExpF16Vop1, rdna3::kVExpF16Vop3},
       {rdna4::kVExpF16Vop1, rdna4::kVExpF16Vop3},
       0x4000,
       4.0f},
      {{rdna3::kVLogF16Vop1, rdna3::kVLogF16Vop3},
       {rdna4::kVLogF16Vop1, rdna4::kVLogF16Vop3},
       0x4400,
       2.0f},
      {{rdna3::kVFractF16Vop1, rdna3::kVFractF16Vop3},
       {rdna4::kVFractF16Vop1, rdna4::kVFractF16Vop3},
       0x3e00,
       0.5f},
      {{rdna3::kVFrexpMantF16Vop1, rdna3::kVFrexpMantF16Vop3},
       {rdna4::kVFrexpMantF16Vop1, rdna4::kVFrexpMantF16Vop3},
       0x4600,
       0.75f},
  };
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32u, 64u})
      for (const auto &c : cases)
        for (unsigned encoding = 0; encoding < 2; ++encoding)
          for (unsigned halves = 0; halves < (encoding ? 4u : 1u); ++halves)
            for (unsigned rounding = 0; rounding < 4; ++rounding)
              for (unsigned policy_index = 0; policy_index < 3; ++policy_index) {
                SCOPED_TRACE(::testing::Message()
                             << rdna3 << '/' << lanes << '/' << c.rdna4[0] << '/' << encoding << '/'
                             << halves << '/' << rounding << '/' << policy_index);
                ArithmeticPolicy policy{};
                if (policy_index == 1)
                  policy.semantics = ArithmeticSemantics::Loose;
                else if (policy_index == 2)
                  if (policy.semantics != ArithmeticSemantics::Loose)
                    policy.semantics = ArithmeticSemantics::BitExactEmpirical;
                BridgeFixture fixture(rdna3, lanes, policy);
                // Preserve denormals and enable IEEE: RDNA3 suppresses OMOD,
                // while RDNA4 honors it. Guest rounding must not reject SFUs.
                fixture.wave->set_mode_raw(0x3c0 | (rounding << 2));
                fixture.wave->set_trapsts(0x10);
                const uint64_t exec_mask = lanes == 32 ? 0x80000001ULL : 0x8000000180000001ULL;
                fixture.wave->set_exec(exec_mask);
                const unsigned a_shift = halves & 1 ? 16 : 0;
                const unsigned d_shift = halves & 2 ? 16 : 0;
                // VOP3 ABS must remove this sign before evaluating the function.
                const uint32_t raw = c.input | (encoding ? 0x8000 : 0);
                const uint32_t initial = a_shift ? (raw << 16) | 0xdeadU : 0xdead0000U | raw;
                for (uint32_t lane = 0; lane < lanes; ++lane)
                  fixture.cu->write_vgpr(fixture.base(), lane, initial);
                const uint16_t opcode = rdna3 ? c.rdna3[encoding] : c.rdna4[encoding];
                std::array<uint32_t, 2> words{};
                if (encoding) {
                  const uint8_t opsel = uint8_t((halves & 1) | ((halves & 2) << 2));
                  words = rdna3
                              ? rdna3::build_vop3(
                                    opcode,
                                    {.vdst = 0, .abs = 1, .op_sel = opsel, .src0 = 256, .omod = 1})
                              : rdna4::build_vop3(
                                    opcode,
                                    {.vdst = 0, .abs = 1, .opsel = opsel, .src0 = 256, .omod = 1});
                } else {
                  words[0] = rdna3 ? rdna3::build_vop1(opcode, {.src0 = 256, .vdst = 0})[0]
                                   : rdna4::build_vop1(opcode, {.src0 = 256, .vdst = 0})[0];
                }
                auto instruction = fixture.decoder->decode_window(words);
                ASSERT_TRUE(instruction.succeeded());
                const bool rejected = policy_index == 2;
                EXPECT_EQ(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .failed(),
                          rejected);
                uint64_t written = 0;
                for (const auto &write : fixture.recorder->writes) {
                  EXPECT_EQ(write.reg, 0U);
                  EXPECT_EQ(write.bytes, d_shift ? 0xc : 0x3);
                  EXPECT_EQ(written & write.lanes, 0U);
                  written |= write.lanes;
                }
                EXPECT_EQ(written, rejected ? 0 : exec_mask);
                const float expected = c.result * (encoding && !rdna3 ? 2.0f : 1.0f);
                const bool exact = policy_index == 0 && c.rdna4[0] != rdna4::kVFractF16Vop1;
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                  const uint32_t actual = fixture.cu->read_vgpr(fixture.base(), lane);
                  if (rejected || !((exec_mask >> lane) & 1)) {
                    EXPECT_EQ(actual, initial);
                  } else {
                    EXPECT_EQ(actual & ~(0xffffU << d_shift), initial & ~(0xffffU << d_shift));
                    if (exact) {
                      EXPECT_EQ(uint16_t(actual >> d_shift), util::f32_to_f16(expected));
                    } else {
                      EXPECT_NEAR(util::f16_to_f32(uint16_t(actual >> d_shift)), expected,
                                  1e-6f + 1e-3f * std::abs(expected));
                    }
                  }
                }
                EXPECT_EQ(fixture.wave->trapsts(), 0x10U);
              }
}

TEST(ArithBridgeTest, HalfFmaFormsRoundingPartialAccumulatorsAndState) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32u, 64u})
      for (unsigned form = 0; form < 5; ++form)
        for (unsigned rounding = 0; rounding < 4; ++rounding)
          for (unsigned policy_index = 0; policy_index < 3; ++policy_index)
            for (bool high : {false, true}) {
              const bool vop3 = form == 0 || form == 2;
              if (high && !vop3)
                continue;
              SCOPED_TRACE(::testing::Message() << rdna3 << '/' << lanes << '/' << form << '/'
                                                << rounding << '/' << policy_index << '/' << high);
              ArithmeticPolicy policy{};
              if (policy_index == 1)
                policy.semantics = ArithmeticSemantics::Loose;
              else if (policy_index == 2)
                policy.semantics = ArithmeticSemantics::BitExactEmpirical;
              BridgeFixture fixture(rdna3, lanes, policy);
              fixture.wave->set_mode_raw(0x3c0 | (rounding << 2));
              fixture.wave->set_trapsts(0x10);
              const uint64_t exec_mask = lanes == 32 ? 0x80000001ULL : 0x8000000180000001ULL;
              fixture.wave->set_exec(exec_mask);
              const unsigned shift = high ? 16 : 0;
              auto packed = [high](uint32_t value) {
                return high ? (value << 16) | 0xdeadU : 0xdead0000U | value;
              };
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                fixture.cu->write_vgpr(fixture.base(), lane, packed(0x3c00));
                fixture.cu->write_vgpr(fixture.base() + 1, lane, packed(form == 4 ? 1 : 0x3c00));
                fixture.cu->write_vgpr(fixture.base() + 2, lane, packed(1));
              }
              std::array<uint32_t, 2> words{};
              if (vop3) {
                const uint16_t opcode =
                    rdna3 ? (form == 0 ? rdna3::kVFmaF16Vop3 : rdna3::kVFmacF16Vop3)
                          : (form == 0 ? rdna4::kVFmaF16Vop3 : rdna4::kVFmacF16Vop3);
                // FMAC ignores C's selector and ABS/NEG bits, reading D's half.
                const uint8_t opsel =
                    uint8_t((high ? 11 : 0) | ((form == 2 ? !high : high) ? 4 : 0));
                words = rdna3 ? rdna3::build_vop3(opcode, {.vdst = 2,
                                                           .abs = uint8_t(form == 2 ? 4 : 0),
                                                           .op_sel = opsel,
                                                           .src0 = 256,
                                                           .src1 = 257,
                                                           .src2 = 258,
                                                           .neg = uint8_t(form == 2 ? 4 : 0)})
                              : rdna4::build_vop3(opcode, {.vdst = 2,
                                                           .abs = uint8_t(form == 2 ? 4 : 0),
                                                           .opsel = opsel,
                                                           .src0 = 256,
                                                           .src1 = 257,
                                                           .src2 = 258,
                                                           .neg = uint8_t(form == 2 ? 4 : 0)});
              } else {
                const uint16_t opcode = rdna3 ? (form == 1   ? rdna3::kVFmacF16Vop2
                                                 : form == 3 ? rdna3::kVFmaakF16Vop2
                                                             : rdna3::kVFmamkF16Vop2)
                                              : (form == 1   ? rdna4::kVFmacF16Vop2
                                                 : form == 3 ? rdna4::kVFmaakF16Vop2
                                                             : rdna4::kVFmamkF16Vop2);
                words[0] = rdna3
                               ? rdna3::build_vop2(opcode, {.src0 = 256, .vsrc1 = 1, .vdst = 2})[0]
                               : rdna4::build_vop2(opcode, {.src0 = 256, .vsrc1 = 1, .vdst = 2})[0];
                words[1] = form == 3 ? 1 : 0x3c00;
              }
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              const bool rejected = policy_index == 2 && (rdna3 || rounding != 0);
              EXPECT_EQ(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .failed(),
                        rejected);
              // VOP2's delegated half operands report individual lane writes;
              // native VOP3 operands report the combined lane mask.
              uint64_t written = 0;
              for (const auto &write : fixture.recorder->writes) {
                EXPECT_EQ(write.reg, 2U);
                EXPECT_EQ(write.bytes, high ? 0xc : 0x3);
                EXPECT_EQ(written & write.lanes, 0U);
                written |= write.lanes;
              }
              EXPECT_EQ(written, rejected ? 0 : exec_mask);
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                const uint32_t half =
                    !rejected && ((exec_mask >> lane) & 1) ? (rounding == 1 ? 0x3c01 : 0x3c00) : 1;
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lane), packed(half))
                    << rdna3 << '/' << form << '/' << rounding << '/' << policy_index << '/'
                    << shift;
              }
              const uint32_t causes =
                  policy_index == 2 && !rejected
                      ? RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_INEXACT
                      : 0;
              EXPECT_EQ(fixture.wave->trapsts(), 0x10U | causes);
            }
}

TEST(ArithBridgeTest, HalfDotModifiersSelectorsAliasesAndByteObservations) {
  // Small binary fractions make the FP32 intermediate and half result exact;
  // a tight atol/rtol still expresses the API's loose numerical contract.
  for (bool is_rdna3 : {false, true}) {
    for (bool bf16 : {false, true}) {
      auto narrow = [bf16](float value) {
        return bf16 ? util::f32_to_bf16(value) : util::f32_to_f16(value);
      };
      auto widen = [bf16](uint16_t value) {
        return bf16 ? util::bf16_to_f32(value) : util::f16_to_f32(value);
      };
      const uint32_t inputs[] = {uint32_t(narrow(-1)) | (uint32_t(narrow(2)) << 16),
                                 uint32_t(narrow(0.5f)) | (uint32_t(narrow(-2)) << 16),
                                 uint32_t(narrow(1)) | (uint32_t(narrow(-0.5f)) << 16)};
      for (uint32_t lanes : {32u, 64u}) {
        ArithmeticPolicy policy{};
        policy.semantics = ArithmeticSemantics::Loose;
        BridgeFixture fixture(is_rdna3, lanes, policy);
        const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000180000005ULL;
        fixture.wave->set_exec(exec_mask);
        for (unsigned controls = 0; controls < 64; ++controls) {
          for (unsigned halves = 0; halves < 4; ++halves) {
            for (uint8_t dest = 0; dest < 3; ++dest) {
              fixture.wave->set_mode_raw(controls
                                         << 2); // Guest rounding/denorm controls are ignored.
              for (unsigned source = 0; source < 3; ++source)
                for (uint32_t lane = 0; lane < lanes; ++lane)
                  fixture.cu->write_vgpr(fixture.base() + source, lane, inputs[source]);
              const auto words = is_rdna3 ? rdna3::build_vop3(bf16 ? rdna3::kVDot2Bf16Bf16Vop3
                                                                   : rdna3::kVDot2F16F16Vop3,
                                                              {.vdst = dest,
                                                               .abs = uint8_t(controls & 7),
                                                               .op_sel = uint8_t(halves << 2),
                                                               .src0 = 256,
                                                               .src1 = 257,
                                                               .src2 = 258,
                                                               .neg = uint8_t(controls >> 3)})
                                          : rdna4::build_vop3(bf16 ? rdna4::kVDot2Bf16Bf16Vop3
                                                                   : rdna4::kVDot2F16F16Vop3,
                                                              {.vdst = dest,
                                                               .abs = uint8_t(controls & 7),
                                                               .opsel = uint8_t(halves << 2),
                                                               .src0 = 256,
                                                               .src1 = 257,
                                                               .src2 = 258,
                                                               .neg = uint8_t(controls >> 3)});
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              ASSERT_EQ(fixture.recorder->reads.size(), 3u);
              EXPECT_EQ(fixture.recorder->reads[0].bytes, 0xf);
              EXPECT_EQ(fixture.recorder->reads[1].bytes, 0xf);
              EXPECT_EQ(fixture.recorder->reads[2].bytes, halves & 1 ? 0xc : 0x3);
              ASSERT_EQ(fixture.recorder->writes.size(), 1u);
              EXPECT_EQ(fixture.recorder->writes[0].bytes, halves & 2 ? 0xc : 0x3);
              auto input = [&](unsigned source, unsigned half) {
                float value = widen(uint16_t(inputs[source] >> (half * 16)));
                if (controls & (1u << source))
                  value = std::abs(value);
                if (controls & (8u << source))
                  value = -value;
                return value;
              };
              const float expected =
                  input(0, 0) * input(1, 0) + input(0, 1) * input(1, 1) + input(2, halves & 1);
              const unsigned shift = halves & 2 ? 16 : 0;
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                const uint32_t actual = fixture.cu->read_vgpr(fixture.base() + dest, lane);
                EXPECT_EQ(actual & ~(0xffffu << shift), inputs[dest] & ~(0xffffu << shift));
                if ((exec_mask >> lane) & 1) {
                  EXPECT_NEAR(widen(uint16_t(actual >> shift)), expected,
                              1e-6f + 1e-6f * std::abs(expected));
                } else {
                  EXPECT_EQ(actual, inputs[dest]);
                }
              }
              fixture.recorder->reads.clear();
              fixture.recorder->writes.clear();
            }
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, HalfDotInlineReplicationAndRejectedPartialWrites) {
  for (bool is_rdna3 : {false, true}) {
    for (bool bf16 : {false, true}) {
      for (uint32_t lanes : {32u, 64u}) {
        for (unsigned policy_index = 0; policy_index < 3; ++policy_index) {
          ArithmeticPolicy policy{};
          if (policy_index == 1) {
            policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
          } else if (policy_index == 2) {
            policy.semantics = ArithmeticSemantics::BitExactEmpirical;
          }
          BridgeFixture fixture(is_rdna3, lanes, policy);
          fixture.wave->set_exec(lanes == 32 ? 0x80000001ULL : 0x8000000000000001ULL);
          fixture.wave->set_trapsts(0x10);
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            fixture.cu->write_vgpr(fixture.base() + 1, lane, bf16 ? 0x40003f80u : 0x40003c00u);
            fixture.cu->write_vgpr(fixture.base() + 2, lane, 0xdead0000);
          }
          const auto words =
              is_rdna3 ? rdna3::build_vop3(
                             bf16 ? rdna3::kVDot2Bf16Bf16Vop3 : rdna3::kVDot2F16F16Vop3,
                             {.vdst = 2, .op_sel = 8, .src0 = 242, .src1 = 257, .src2 = 258})
                       : rdna4::build_vop3(
                             bf16 ? rdna4::kVDot2Bf16Bf16Vop3 : rdna4::kVDot2F16F16Vop3,
                             {.vdst = 2, .opsel = 8, .src0 = 242, .src1 = 257, .src2 = 258});
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          const bool rejected =
              policy_index == 2; // Exact arithmetic is supported; FP status is not.
          EXPECT_EQ(
              fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).failed(),
              rejected);
          const uint32_t expected = rejected ? 0xdead0000u : bf16 ? 0x40400000u : 0x42000000u;
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0), expected);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lanes - 1), expected);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 1), 0xdead0000u);
          EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
          EXPECT_EQ(fixture.recorder->writes.size(), rejected ? 0u : 1u);
          if (!rejected) {
            for (bool saturate : {false, true}) {
              fixture.wave->set_mode_raw(saturate ? Wavefront::FP16_OVFL_BIT : 0);
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                fixture.cu->write_vgpr(fixture.base() + 1, lane, bf16 ? 0x7f7f7f7fu : 0x7bff7bffu);
                fixture.cu->write_vgpr(fixture.base() + 2, lane, 0xdead0000u);
              }
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              // Two largest finite inputs overflow. Only FP16 honors saturation.
              const uint32_t overflow = bf16 ? 0x7f800000u : saturate ? 0x7bff0000u : 0x7c000000u;
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0), overflow);
            }
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, Dot2HardwareBitsWaveWidthsAliasesAndStatePolicy) {
  // Existing rocjitsu hardware captures validate both architecture models.
  auto check = [](bool is_rdna3, bool bf16, const auto &cases) {
    for (uint32_t lanes : {32u, 64u}) {
      for (bool observed : {false, true}) {
        for (bool report : {false, true}) {
          ArithmeticPolicy policy{ArithmeticSemantics::BitExactEmpiricalNoFpStatus};
          if (report)
            if (policy.semantics != ArithmeticSemantics::Loose)
              policy.semantics = ArithmeticSemantics::BitExactEmpirical;
          BridgeFixture fixture(is_rdna3, lanes, policy);
          if (!observed) {
            fixture.cu->set_plugin_group(nullptr);
            fixture.recorder = nullptr;
          }
          fixture.wave->set_mode_raw(0x3ff); // DOT2 accumulation ignores guest MODE.
          fixture.wave->set_trapsts(0x10);
          const uint64_t exec_mask = lanes == 32 ? 0xa55a5aa5ULL : 0xa55a5aa5a55a5aa5ULL;
          fixture.wave->set_exec(exec_mask);
          const auto words =
              is_rdna3 ? rdna3::build_vop3p(
                             bf16 ? rdna3::kVDot2F32Bf16Vop3p : rdna3::kVDot2F32F16Vop3p,
                             {.vdst = 2, .src0 = 256, .src1 = 257, .src2 = 258, .op_sel_hi = 3})
                       : rdna4::build_vop3p(
                             bf16 ? rdna4::kVDot2F32Bf16Vop3p : rdna4::kVDot2F32F16Vop3p,
                             {.vdst = 2, .src0 = 256, .src1 = 257, .src2 = 258, .opsel_hi = 3});
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          for (size_t offset = 0; offset < cases.size(); offset += lanes) {
            for (uint32_t lane = 0; lane < lanes; ++lane) {
              const auto &c = cases[(offset + lane) % cases.size()];
              fixture.cu->write_vgpr(fixture.base(), lane, c.a);
              fixture.cu->write_vgpr(fixture.base() + 1, lane, c.b);
              fixture.cu->write_vgpr(fixture.base() + 2, lane, c.c);
            }
            const bool rejected = report && is_rdna3;
            EXPECT_EQ(
                fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).failed(),
                rejected);
            for (uint32_t lane = 0; lane < lanes; ++lane) {
              const auto &c = cases[(offset + lane) % cases.size()];
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lane),
                        !rejected && ((exec_mask >> lane) & 1) ? c.expected : c.c);
            }
            EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
            if (observed) {
              EXPECT_EQ(fixture.recorder->writes.size(), rejected ? 0u : 1u);
              fixture.recorder->reads.clear();
              fixture.recorder->writes.clear();
            }
          }
        }
      }
    }
  };
  check(true, false, kDot2F16Cases);
  check(true, true, kDot2BF16Cases);
  check(false, false, kGfx12DotF16Cases);
  check(false, true, kGfx12DotBF16Cases);
}

TEST(ArithBridgeTest, Dot2DppLooseRoutingAndWriteFilter) {
  for (uint32_t lanes : {32u, 64u}) {
    BridgeFixture fixture(true, lanes, {ArithmeticSemantics::Loose});
    fixture.wave->set_exec(UINT64_MAX);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      fixture.cu->write_vgpr(fixture.base(), lane, uint32_t(util::f32_to_f16(float(lane + 1))));
      fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x3c00);
      fixture.cu->write_vgpr(fixture.base() + 2, lane, 0);
    }
    rdna3::Vop3pVopDpp16MachineInst raw{};
    raw.encoding = 0xcc;
    raw.op = rdna3::kVDot2F32F16Vop3p;
    raw.vdst = 2;
    raw.src0 = amdgpu::SRC_DPP;
    raw.src1 = 257;
    raw.src2 = 258;
    raw.op_sel_hi = 3;
    raw.vsrc0 = 0;
    raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
    raw.fi = 1;
    raw.row_mask = 0xf;
    raw.bank_mask = 0xe;
    rdna3::VDot2F32F16Vop3p instruction(reinterpret_cast<const rdna3::MachineInst *>(&raw));
    ASSERT_TRUE(fixture.cu->execute_instruction(&instruction, *fixture.wave).succeeded());
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const float expected = lane % 16 >= 4 ? float(lane) : 0.0f;
      const float actual = std::bit_cast<float>(fixture.cu->read_vgpr(fixture.base() + 2, lane));
      EXPECT_NEAR(actual, expected, 1e-6f + 1e-6f * std::abs(expected));
    }
  }
}

// DPP row/bank filtering limits B/C observations as well as destination stores.
TEST(ArithBridgeTest, Dot4DppObservesOnlyWritableLanes) {
  for (uint32_t lanes : {32u, 64u}) {
    BridgeFixture fixture(false, lanes, {ArithmeticSemantics::Loose});
    fixture.wave->set_exec(UINT64_MAX);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      fixture.cu->write_vgpr(fixture.base(), lane, 0x38383838u);
      fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x38383838u);
      fixture.cu->write_vgpr(fixture.base() + 2, lane, 0);
    }
    rdna4::Vop3pVopDpp16MachineInst raw{};
    raw.encoding = 0xcc;
    raw.op = rdna4::kVDot4F32Fp8Fp8Vop3p;
    raw.vdst = 2;
    raw.src0 = amdgpu::SRC_DPP;
    raw.src1 = 257;
    raw.src2 = 258;
    raw.opsel_hi = 3;
    raw.vsrc0 = 0;
    raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
    raw.fi = 1;
    raw.row_mask = 1;
    raw.bank_mask = 0xe;
    fixture.recorder->reads.clear();
    rdna4::VDot4F32Fp8Fp8Vop3p instruction(reinterpret_cast<const rdna4::MachineInst *>(&raw));
    ASSERT_TRUE(fixture.cu->execute_instruction(&instruction, *fixture.wave).succeeded());
    uint64_t observed_b = 0, observed_c = 0;
    for (const auto &read : fixture.recorder->reads) {
      if (read.reg == fixture.base() + 1)
        observed_b |= read.lanes;
      if (read.reg == fixture.base() + 2)
        observed_c |= read.lanes;
    }
    EXPECT_EQ(observed_b, 0xfff0u);
    EXPECT_EQ(observed_c, 0xfff0u);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lane),
                lane >= 4 && lane < 16 ? 0x40800000u : 0u);
    }
  }
}

TEST(ArithBridgeTest, Dot4FormatsAccumulatorModifiersAndExactPolicy) {
  const uint16_t ops[] = {rdna4::kVDot4F32Fp8Fp8Vop3p, rdna4::kVDot4F32Fp8Bf8Vop3p,
                          rdna4::kVDot4F32Bf8Fp8Vop3p, rdna4::kVDot4F32Bf8Bf8Vop3p};
  for (uint32_t lanes : {32u, 64u}) {
    for (unsigned op = 0; op < 4; ++op) {
      for (unsigned modifiers = 0; modifiers < 4; ++modifiers) {
        for (bool strict : {false, true}) {
          ArithmeticPolicy policy{};
          if (strict) {
            policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
          }
          BridgeFixture fixture(false, lanes, policy);
          fixture.wave->set_exec(lanes == 32 ? 0x80000001ULL : 0x8000000000000001ULL);
          fixture.wave->set_trapsts(0x10);
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            // 0x3c is 1.5 in E4M3 and 1 in E5M2: the format combinations
            // have exact finite dot products (9, 6, 6, 4).
            fixture.cu->write_vgpr(fixture.base(), lane, 0x3c3c3c3c);
            fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x3c3c3c3c);
            fixture.cu->write_vgpr(fixture.base() + 2, lane, 0xbf800000);
          }
          const auto words = rdna4::build_vop3p(ops[op], {.vdst = 2,
                                                          .neg_hi = uint8_t(modifiers & 2 ? 4 : 0),
                                                          .opsel_hi_2 = 1,
                                                          .src0 = 256,
                                                          .src1 = 257,
                                                          .src2 = 258,
                                                          .opsel_hi = 3,
                                                          .neg = uint8_t(modifiers & 1 ? 4 : 0)});
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          EXPECT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          const float products[] = {9, 6, 6, 4};
          float acc = modifiers & 2 ? 1.0f : -1.0f;
          if (modifiers & 1)
            acc = -acc;
          const uint32_t expected = std::bit_cast<uint32_t>(products[op] + acc);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0), expected);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lanes - 1), expected);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 1), 0xbf800000u);
          EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
          EXPECT_EQ(fixture.recorder->writes.size(), 1u);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, DecodedPackedSelectionGuestModesAndAliases) {
  const uint16_t values[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x3c00, 0xbc00};
  const uint16_t rdna3_ops[] = {rdna3::kVPkMinF16Vop3p, rdna3::kVPkMaxF16Vop3p};
  const uint16_t rdna4_ops[] = {rdna4::kVPkMinNumF16Vop3p, rdna4::kVPkMaxNumF16Vop3p,
                                rdna4::kVPkMinimumF16Vop3p, rdna4::kVPkMaximumF16Vop3p};
  for (bool is_rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      BridgeFixture fixture(is_rdna3, lanes);
      fixture.wave->set_trapsts(0x10);
      const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000180000005ULL;
      fixture.wave->set_exec(exec_mask);
      for (unsigned op = 0; op < (is_rdna3 ? 2u : 4u); ++op) {
        for (unsigned denorm = 0; denorm < 4; ++denorm) {
          for (unsigned selectors = 0; selectors < 16; ++selectors) {
            for (unsigned signs : {0u, 5u, 10u, 15u}) {
              for (bool clamp : {false, true}) {
                fixture.wave->set_mode_raw(0x300 | (denorm << 6) | ((selectors & 3) << 2));
                uint32_t inputs[2][64];
                for (unsigned source = 0; source < 2; ++source)
                  for (uint32_t lane = 0; lane < lanes; ++lane) {
                    inputs[source][lane] = values[(lane + 3 * source) % 8] |
                                           (uint32_t(values[(lane + 3 * source + 1) % 8]) << 16);
                    fixture.cu->write_vgpr(fixture.base() + source, lane, inputs[source][lane]);
                  }
                const auto words =
                    is_rdna3
                        ? rdna3::build_vop3p(rdna3_ops[op], {.vdst = 0,
                                                             .neg_hi = uint8_t(signs >> 2),
                                                             .op_sel = uint8_t(selectors & 3),
                                                             .clamp = uint8_t(clamp),
                                                             .src0 = 256,
                                                             .src1 = 257,
                                                             .op_sel_hi = uint8_t(selectors >> 2),
                                                             .neg = uint8_t(signs & 3)})
                        : rdna4::build_vop3p(rdna4_ops[op], {.vdst = 0,
                                                             .neg_hi = uint8_t(signs >> 2),
                                                             .opsel = uint8_t(selectors & 3),
                                                             .clamp = uint8_t(clamp),
                                                             .src0 = 256,
                                                             .src1 = 257,
                                                             .opsel_hi = uint8_t(selectors >> 2),
                                                             .neg = uint8_t(signs & 3)});
                auto instruction = fixture.decoder->decode_window(words);
                ASSERT_TRUE(instruction.succeeded());
                ASSERT_TRUE(
                    fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                        .succeeded());
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                  uint32_t expected = inputs[0][lane];
                  if ((exec_mask >> lane) & 1) {
                    expected = 0;
                    for (unsigned half = 0; half < 2; ++half) {
                      uint16_t operand[2];
                      for (unsigned source = 0; source < 2; ++source) {
                        unsigned bit = source + 2 * half;
                        operand[source] =
                            uint16_t(inputs[source][lane] >> ((selectors & (1u << bit)) ? 16 : 0));
                        if (signs & (1u << bit))
                          operand[source] ^= 0x8000;
                        if (!(denorm & 1) && (operand[source] & 0x7fff) < 0x400)
                          operand[source] &= 0x8000;
                      }
                      // Numeric selection returns an input encoding; OR/AND
                      // gives the required -0/+0 ordering for equal inputs.
                      float a = util::f16_to_f32(operand[0]), b = util::f16_to_f32(operand[1]);
                      uint16_t chosen =
                          a == b ? (op & 1 ? operand[0] & operand[1] : operand[0] | operand[1])
                                 : ((op & 1 ? a > b : a < b) ? operand[0] : operand[1]);
                      if (clamp)
                        chosen = chosen & 0x8000 ? 0 : std::min(chosen, uint16_t(0x3c00));
                      // Packed selection preserves denormal outputs (upstream #13898).
                      expected |= uint32_t(chosen) << (16 * half);
                    }
                  }
                  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), expected);
                }
                EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
                fixture.recorder->reads.clear();
                fixture.recorder->writes.clear();
              }
            }
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, PackedSelectionRdna3NanClampPolicy) {
  for (uint32_t lanes : {32u, 64u}) {
    for (bool maximum : {false, true}) {
      for (bool ieee : {false, true}) {
        for (bool dx10 : {false, true}) {
          BridgeFixture fixture(true, lanes);
          fixture.wave->set_mode_raw(0xc0 | (ieee ? 0x200 : 0) | (dx10 ? 0x100 : 0));
          fixture.wave->set_exec(1);
          fixture.cu->write_vgpr(fixture.base(), 0, 0x7d017d01u);
          fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x7e027e02u);
          const auto words =
              rdna3::build_vop3p(maximum ? rdna3::kVPkMaxF16Vop3p : rdna3::kVPkMinF16Vop3p,
                                 {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .op_sel_hi = 3});
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), dx10   ? 0u
                                                              : ieee ? 0x7f017f01u
                                                                     : 0x7d017d01u);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 1), 0x3c003c00u);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, DecodedPackedBinarySelectorsSignsClampAndAliases) {
  // Small binary fractions have exactly representable ADD/MUL results.
  const uint32_t inputs[] = {0x3c004000u, 0x38003c00u};
  for (bool is_rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      for (bool multiply : {false, true}) {
        for (bool observed : {false, true}) {
          BridgeFixture fixture(is_rdna3, lanes);
          if (!observed) {
            fixture.cu->set_plugin_group(nullptr);
            fixture.recorder = nullptr;
          }
          fixture.wave->set_mode_raw(0x3c0);
          const uint16_t opcode =
              is_rdna3 ? (multiply ? rdna3::kVPkMulF16Vop3p : rdna3::kVPkAddF16Vop3p)
                       : (multiply ? rdna4::kVPkMulF16Vop3p : rdna4::kVPkAddF16Vop3p);
          for (unsigned selectors = 0; selectors < 16; ++selectors) {
            for (unsigned signs : {0u, 5u, 10u, 15u}) {
              for (bool clamp : {false, true}) {
                const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000180000005ULL;
                fixture.wave->set_exec(exec_mask);
                for (unsigned source = 0; source < 2; ++source)
                  for (uint32_t lane = 0; lane < lanes; ++lane)
                    fixture.cu->write_vgpr(fixture.base() + source, lane, inputs[source]);
                const auto words =
                    is_rdna3 ? rdna3::build_vop3p(opcode, {.vdst = 0,
                                                           .neg_hi = uint8_t(signs >> 2),
                                                           .op_sel = uint8_t(selectors & 3),
                                                           .clamp = uint8_t(clamp),
                                                           .src0 = 256,
                                                           .src1 = 257,
                                                           .op_sel_hi = uint8_t(selectors >> 2),
                                                           .neg = uint8_t(signs & 3)})
                             : rdna4::build_vop3p(opcode, {.vdst = 0,
                                                           .neg_hi = uint8_t(signs >> 2),
                                                           .opsel = uint8_t(selectors & 3),
                                                           .clamp = uint8_t(clamp),
                                                           .src0 = 256,
                                                           .src1 = 257,
                                                           .opsel_hi = uint8_t(selectors >> 2),
                                                           .neg = uint8_t(signs & 3)});
                auto instruction = fixture.decoder->decode_window(words);
                ASSERT_TRUE(instruction.succeeded());
                ASSERT_TRUE(
                    fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                        .succeeded());
                uint32_t expected = 0;
                for (unsigned half = 0; half < 2; ++half) {
                  float values[2];
                  for (unsigned source = 0; source < 2; ++source) {
                    const unsigned bit = source + 2 * half;
                    uint16_t value =
                        uint16_t(inputs[source] >> ((selectors & (1u << bit)) ? 16 : 0));
                    if (signs & (1u << bit))
                      value ^= 0x8000;
                    values[source] = util::f16_to_f32(value);
                  }
                  float value = multiply ? values[0] * values[1] : values[0] + values[1];
                  if (clamp)
                    value = std::clamp(value, 0.0f, 1.0f);
                  expected |= uint32_t(util::f32_to_f16(value)) << (16 * half);
                }
                for (uint32_t lane = 0; lane < lanes; ++lane)
                  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                            (exec_mask & (1ULL << lane)) ? expected : inputs[0]);
                if (observed) {
                  EXPECT_EQ(fixture.recorder->writes.size(), 1u);
                  fixture.recorder->reads.clear();
                  fixture.recorder->writes.clear();
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, PackedBinaryInlineRoundingAndStrictPolicy) {
  // Low = 1 + 2^-12; high = 0 + 2^-12: inline 1 narrows only into the low half.
  const uint32_t expected[] = {0x0c003c00u, 0x0c003c01u, 0x0c003c00u, 0x0c003c00u};
  for (bool is_rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      for (unsigned rounding = 0; rounding < 4; ++rounding) {
        for (unsigned policy_index = 0; policy_index < 4; ++policy_index) {
          ArithmeticPolicy policy{};
          if (policy_index == 1) {
            policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
          } else if (policy_index == 2) {
            if (policy.semantics != ArithmeticSemantics::Loose)
              policy.semantics = ArithmeticSemantics::BitExactEmpirical;
          } else if (policy_index == 3) {
            policy.semantics = ArithmeticSemantics::Loose;
          }
          BridgeFixture fixture(is_rdna3, lanes, policy);
          fixture.wave->set_mode_raw(0x3c0 | (rounding << 2));
          fixture.wave->set_exec(lanes == 32 ? 0x80000001ULL : 0x8000000000000001ULL);
          fixture.wave->set_trapsts(0x10);
          for (uint32_t lane = 0; lane < lanes; ++lane)
            fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x0c000c00u);
          const auto words =
              is_rdna3 ? rdna3::build_vop3p(rdna3::kVPkAddF16Vop3p,
                                            {.vdst = 1, .src0 = 242, .src1 = 257, .op_sel_hi = 3})
                       : rdna4::build_vop3p(rdna4::kVPkAddF16Vop3p,
                                            {.vdst = 1, .src0 = 242, .src1 = 257, .opsel_hi = 3});
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          const bool rejected = policy_index == 1 || policy_index == 2;
          EXPECT_EQ(
              fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).failed(),
              rejected);
          for (uint32_t lane = 0; lane < lanes; ++lane)
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, lane),
                      !rejected && (lane == 0 || lane == lanes - 1) ? expected[rounding]
                                                                    : 0x0c000c00u);
          EXPECT_EQ(fixture.wave->trapsts(), 0x10u);
          EXPECT_EQ(fixture.recorder->writes.size(), rejected ? 0u : 1u);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, DecodedPackedFmaSelectorsSignsClampAndAliases) {
  // Every reference operation uses small binary fractions with an exactly
  // representable FP16 fused result; no host rounding decision is involved.
  const uint32_t inputs[] = {0x3c004000u, 0x38003c00u, 0xbc003800u};
  for (bool is_rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      for (auto semantics :
           {ArithmeticSemantics::Loose, ArithmeticSemantics::BitExactEmpiricalAllowLoose}) {
        BridgeFixture fixture(is_rdna3, lanes,
                              {semantics == ArithmeticSemantics::Loose
                                   ? semantics
                                   : ArithmeticSemantics::BitExactEmpiricalNoFpStatus});
        fixture.wave->set_mode_raw(0x3c0);
        for (unsigned selectors = 0; selectors < 64; ++selectors) {
          for (unsigned signs : {0u, 0x15u, 0x2au, 0x3fu}) {
            for (bool clamp : {false, true}) {
              const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000180000005ULL;
              fixture.wave->set_exec(exec_mask);
              for (unsigned source = 0; source < 3; ++source)
                for (uint32_t lane = 0; lane < lanes; ++lane)
                  fixture.cu->write_vgpr(fixture.base() + source, lane, inputs[source]);
              const auto words =
                  is_rdna3 ? rdna3::build_vop3p(rdna3::kVPkFmaF16Vop3p,
                                                {.vdst = 0,
                                                 .neg_hi = uint8_t(signs >> 3),
                                                 .op_sel = uint8_t(selectors & 7),
                                                 .op_sel_hi_2 = uint8_t(selectors >> 5),
                                                 .clamp = uint8_t(clamp),
                                                 .src0 = 256,
                                                 .src1 = 257,
                                                 .src2 = 258,
                                                 .op_sel_hi = uint8_t((selectors >> 3) & 3),
                                                 .neg = uint8_t(signs & 7)})
                           : rdna4::build_vop3p(rdna4::kVPkFmaF16Vop3p,
                                                {.vdst = 0,
                                                 .neg_hi = uint8_t(signs >> 3),
                                                 .opsel = uint8_t(selectors & 7),
                                                 .opsel_hi_2 = uint8_t(selectors >> 5),
                                                 .clamp = uint8_t(clamp),
                                                 .src0 = 256,
                                                 .src1 = 257,
                                                 .src2 = 258,
                                                 .opsel_hi = uint8_t((selectors >> 3) & 3),
                                                 .neg = uint8_t(signs & 7)});
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              uint32_t expected = 0;
              for (unsigned half = 0; half < 2; ++half) {
                float values[3];
                for (unsigned source = 0; source < 3; ++source) {
                  const unsigned bit = source + 3 * half;
                  uint16_t value = uint16_t(inputs[source] >> ((selectors & (1u << bit)) ? 16 : 0));
                  if (signs & (1u << bit))
                    value ^= 0x8000;
                  values[source] = util::f16_to_f32(value);
                }
                float value = values[0] * values[1] + values[2];
                if (clamp)
                  value = std::clamp(value, 0.0f, 1.0f);
                expected |= uint32_t(util::f32_to_f16(value)) << (16 * half);
              }
              for (uint32_t lane = 0; lane < lanes; ++lane)
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                          (exec_mask & (1ULL << lane)) ? expected : inputs[0]);
              fixture.recorder->reads.clear();
              fixture.recorder->writes.clear();
            }
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, PackedFmaInlineConstantsRoundingAndFmacReplication) {
  const uint16_t expected[] = {0x3c00, 0x3c01, 0x3c00, 0x3c00};
  for (bool is_rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      for (unsigned rounding = 0; rounding < 4; ++rounding) {
        for (bool observed : {false, true}) {
          BridgeFixture fixture(is_rdna3, lanes);
          if (!observed) {
            fixture.cu->set_plugin_group(nullptr);
            fixture.recorder = nullptr;
          }
          fixture.wave->set_mode_raw(0x3c0 | (rounding << 2));
          fixture.wave->set_exec(lanes == 32 ? 0x80000001ULL : 0x8000000000000001ULL);
          for (uint32_t lane = 0; lane < lanes; ++lane)
            fixture.cu->write_vgpr(fixture.base() + 2, lane, 0x10001000u);
          const auto words =
              is_rdna3 ? rdna3::build_vop3p(
                             rdna3::kVPkFmaF16Vop3p,
                             {.vdst = 2, .op_sel_hi_2 = 1, .src0 = 242, .src1 = 242, .src2 = 258})
                       : rdna4::build_vop3p(
                             rdna4::kVPkFmaF16Vop3p,
                             {.vdst = 2, .opsel_hi_2 = 1, .src0 = 242, .src1 = 242, .src2 = 258});
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0),
                    uint32_t(expected[rounding]) * 0x10001u);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lanes - 1),
                    uint32_t(expected[rounding]) * 0x10001u);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 1), 0x10001000u);
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x40004000u);
            fixture.cu->write_vgpr(fixture.base() + 2, lane, 0x3c003c00u);
          }
          const auto fmac_words =
              is_rdna3
                  ? rdna3::build_vop2(rdna3::kVPkFmacF16Vop2, {.src0 = 241, .vsrc1 = 1, .vdst = 2})
                  : rdna4::build_vop2(rdna4::kVPkFmacF16Vop2, {.src0 = 241, .vsrc1 = 1, .vdst = 2});
          instruction = fixture.decoder->decode_window(fmac_words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0), rounding == 2 ? 0x80008000u : 0u);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lanes - 1),
                    rounding == 2 ? 0x80008000u : 0u);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 1), 0x3c003c00u);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, PackedFmaRequiredStateAndRejectedCommit) {
  for (bool is_rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      for (bool accumulate : {false, true}) {
        for (bool unsupported_rounding : {false, true}) {
          ArithmeticPolicy policy{};
          policy.semantics = ArithmeticSemantics::BitExactEmpirical;
          BridgeFixture fixture(is_rdna3, lanes, policy);
          fixture.wave->set_mode_raw(0x3c0 | (unsupported_rounding ? 4u : 0u));
          fixture.wave->set_exec(1);
          fixture.wave->set_trapsts(0x10);
          fixture.cu->write_vgpr(fixture.base(), 0, 0);
          fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x7c007c00u);
          auto fma = is_rdna3 ? rdna3::build_vop3p(rdna3::kVPkFmaF16Vop3p, {.vdst = 2,
                                                                            .op_sel_hi_2 = 1,
                                                                            .src0 = 256,
                                                                            .src1 = 257,
                                                                            .src2 = 258,
                                                                            .op_sel_hi = 3})
                              : rdna4::build_vop3p(rdna4::kVPkFmaF16Vop3p, {.vdst = 2,
                                                                            .opsel_hi_2 = 1,
                                                                            .src0 = 256,
                                                                            .src1 = 257,
                                                                            .src2 = 258,
                                                                            .opsel_hi = 3});
          auto fmac =
              is_rdna3
                  ? rdna3::build_vop2(rdna3::kVPkFmacF16Vop2, {.src0 = 256, .vsrc1 = 1, .vdst = 2})
                  : rdna4::build_vop2(rdna4::kVPkFmacF16Vop2, {.src0 = 256, .vsrc1 = 1, .vdst = 2});
          auto instruction = accumulate ? fixture.decoder->decode_window(fmac)
                                        : fixture.decoder->decode_window(fma);
          ASSERT_TRUE(instruction.succeeded());
          const bool rejected = is_rdna3 || unsupported_rounding;
          EXPECT_EQ(
              fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).failed(),
              rejected);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0),
                    rejected ? 0x3c003c00u : 0xfe00fe00u);
          EXPECT_EQ(fixture.wave->trapsts(), rejected ? 0x10u : 0x10u | RJ_ARITH_EXCEPTION_INVALID);
          EXPECT_EQ(fixture.recorder->writes.size(), rejected ? 0u : 1u);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, DecodedIntegerDotSignednessSaturationAndAliases) {
  const uint16_t rdna3_opcodes[] = {rdna3::kVDot4I32Iu8Vop3p, rdna3::kVDot4U32U8Vop3p,
                                    rdna3::kVDot8I32Iu4Vop3p, rdna3::kVDot8U32U4Vop3p};
  const uint16_t rdna4_opcodes[] = {rdna4::kVDot4I32Iu8Vop3p, rdna4::kVDot4U32U8Vop3p,
                                    rdna4::kVDot8I32Iu4Vop3p, rdna4::kVDot8U32U4Vop3p};
  for (bool is_rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      for (unsigned operation = 0; operation < 4; ++operation) {
        for (unsigned signs = 0; signs < 4; ++signs) {
          for (bool clamp : {false, true}) {
            for (bool observed : {false, true}) {
              for (bool empty : {false, true}) {
                BridgeFixture fixture(is_rdna3, lanes);
                if (!observed) {
                  fixture.cu->set_plugin_group(nullptr);
                  fixture.recorder = nullptr;
                }
                const uint64_t exec_mask = empty         ? 0
                                           : lanes == 32 ? 0x80000007ULL
                                                         : 0x8000000180000007ULL;
                fixture.wave->set_exec(exec_mask);
                std::array<std::array<uint32_t, 64>, 3> inputs;
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                  inputs[0][lane] = 0x80ff78ffu ^ (lane * 0x13571357u);
                  inputs[1][lane] = 0x7fff017fu ^ (lane * 0x12345679u);
                  inputs[2][lane] = lane % 3 == 0   ? 0x7fffffffu
                                    : lane % 3 == 1 ? 0x80000000u
                                                    : 0xffffffffu;
                  for (unsigned source = 0; source < 3; ++source)
                    fixture.cu->write_vgpr(fixture.base() + source, lane, inputs[source][lane]);
                }
                const auto words =
                    is_rdna3
                        ? rdna3::build_vop3p(rdna3_opcodes[operation], {.vdst = 2,
                                                                        .clamp = uint8_t(clamp),
                                                                        .src0 = 256,
                                                                        .src1 = 257,
                                                                        .src2 = 258,
                                                                        .op_sel_hi = 3,
                                                                        .neg = uint8_t(signs)})
                        : rdna4::build_vop3p(rdna4_opcodes[operation], {.vdst = 2,
                                                                        .clamp = uint8_t(clamp),
                                                                        .src0 = 256,
                                                                        .src1 = 257,
                                                                        .src2 = 258,
                                                                        .opsel_hi = 3,
                                                                        .neg = uint8_t(signs)});
                auto instruction = fixture.decoder->decode_window(words);
                ASSERT_TRUE(instruction.succeeded());
                ASSERT_TRUE(
                    fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                        .succeeded());
                if (observed) {
                  EXPECT_EQ(fixture.recorder->reads.size(), empty ? 0u : 3u);
                  EXPECT_EQ(fixture.recorder->writes.size(), empty ? 0u : 1u);
                  if (!empty) {
                    EXPECT_EQ(fixture.recorder->writes[0].bytes, 0xF);
                    EXPECT_EQ(fixture.recorder->writes[0].lanes, exec_mask);
                  }
                }
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                  uint32_t expected = inputs[2][lane];
                  if (exec_mask & (1ULL << lane)) {
                    const bool mixed = (operation & 1) == 0;
                    const unsigned bits = operation < 2 ? 8 : 4;
                    const uint32_t field_mask = (1u << bits) - 1;
                    int64_t sum = int64_t(inputs[2][lane]) -
                                  ((mixed && (inputs[2][lane] & 0x80000000u)) ? 0x100000000LL : 0);
                    for (unsigned component = 0; component < 32 / bits; ++component) {
                      int64_t a = (inputs[0][lane] >> (bits * component)) & field_mask;
                      int64_t b = (inputs[1][lane] >> (bits * component)) & field_mask;
                      if (mixed && (signs & 1) && (a & (1u << (bits - 1))))
                        a -= 1u << bits;
                      if (mixed && (signs & 2) && (b & (1u << (bits - 1))))
                        b -= 1u << bits;
                      sum += a * b;
                    }
                    if (clamp)
                      sum = mixed ? std::clamp<int64_t>(sum, -0x80000000LL, 0x7fffffffLL)
                                  : std::clamp<int64_t>(sum, 0, 0xffffffffLL);
                    expected = uint32_t(sum);
                  }
                  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lane), expected)
                      << operation << " " << signs << " " << clamp << " " << lane;
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, PackedShiftsSupportAllArithmeticPolicies) {
  for (bool is_rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      for (auto semantics :
           {ArithmeticSemantics::Loose, ArithmeticSemantics::BitExactEmpiricalAllowLoose,
            ArithmeticSemantics::BitExactEmpiricalNoFpStatus,
            ArithmeticSemantics::BitExactEmpirical}) {
        BridgeFixture fixture(is_rdna3, lanes, {semantics});
        fixture.cu->set_plugin_group(nullptr);
        fixture.recorder = nullptr;
        fixture.wave->set_exec(1);
        fixture.cu->write_vgpr(fixture.base(), 0, 0x00010002u);
        fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x00030004u);
        const auto words =
            is_rdna3 ? rdna3::build_vop3p(rdna3::kVPkLshlrevB16Vop3p,
                                          {.vdst = 1, .src0 = 256, .src1 = 257, .op_sel_hi = 3})
                     : rdna4::build_vop3p(rdna4::kVPkLshlrevB16Vop3p,
                                          {.vdst = 1, .src0 = 256, .src1 = 257, .opsel_hi = 3});
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        EXPECT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 0), 0x00060010u);
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 1), 0x3c003c00u);
      }
    }
  }
}

TEST(ArithBridgeTest, DecodedPackedIntegerSelectorsClampAndAliases) {
  const uint16_t rdna3_opcodes[] = {
      rdna3::kVPkAddI16Vop3p,     rdna3::kVPkAddU16Vop3p,     rdna3::kVPkSubI16Vop3p,
      rdna3::kVPkSubU16Vop3p,     rdna3::kVPkMinI16Vop3p,     rdna3::kVPkMinU16Vop3p,
      rdna3::kVPkMaxI16Vop3p,     rdna3::kVPkMaxU16Vop3p,     rdna3::kVPkMulLoU16Vop3p,
      rdna3::kVPkLshlrevB16Vop3p, rdna3::kVPkLshrrevB16Vop3p, rdna3::kVPkAshrrevI16Vop3p,
      rdna3::kVPkMadI16Vop3p,     rdna3::kVPkMadU16Vop3p};
  const uint16_t rdna4_opcodes[] = {
      rdna4::kVPkAddI16Vop3p,     rdna4::kVPkAddU16Vop3p,     rdna4::kVPkSubI16Vop3p,
      rdna4::kVPkSubU16Vop3p,     rdna4::kVPkMinI16Vop3p,     rdna4::kVPkMinU16Vop3p,
      rdna4::kVPkMaxI16Vop3p,     rdna4::kVPkMaxU16Vop3p,     rdna4::kVPkMulLoU16Vop3p,
      rdna4::kVPkLshlrevB16Vop3p, rdna4::kVPkLshrrevB16Vop3p, rdna4::kVPkAshrrevI16Vop3p,
      rdna4::kVPkMadI16Vop3p,     rdna4::kVPkMadU16Vop3p};
  for (bool is_rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      BridgeFixture fixture(is_rdna3, lanes);
      for (unsigned operation = 0; operation < 14; ++operation) {
        const unsigned sources = operation >= 12 ? 3 : 2;
        for (unsigned selection = 0; selection < (1u << (2 * sources)); ++selection) {
          const unsigned low = selection & ((1u << sources) - 1);
          const unsigned high = selection >> sources;
          for (bool clamp : {false, true}) {
            const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000180000005ULL;
            fixture.wave->set_exec(exec_mask);
            std::array<std::array<uint32_t, 64>, 3> inputs;
            for (uint32_t lane = 0; lane < lanes; ++lane) {
              inputs[0][lane] = 0x8000ffffu ^ (lane * 0x13571357u);
              inputs[1][lane] = 0x7fff0002u ^ (lane * 0x12345679u);
              inputs[2][lane] = 0xfffe0003u ^ (lane * 0x01010101u);
              for (unsigned source = 0; source < 3; ++source)
                fixture.cu->write_vgpr(fixture.base() + source, lane, inputs[source][lane]);
            }
            fixture.recorder->reads.clear();
            fixture.recorder->writes.clear();
            const auto words = is_rdna3
                                   ? rdna3::build_vop3p(rdna3_opcodes[operation],
                                                        {.vdst = 0,
                                                         .op_sel = uint8_t(low),
                                                         .op_sel_hi_2 = uint8_t(high >> 2),
                                                         .clamp = uint8_t(clamp),
                                                         .src0 = 256,
                                                         .src1 = 257,
                                                         .src2 = uint16_t(sources == 3 ? 258 : 0),
                                                         .op_sel_hi = uint8_t(high & 3)})
                                   : rdna4::build_vop3p(rdna4_opcodes[operation],
                                                        {.vdst = 0,
                                                         .opsel = uint8_t(low),
                                                         .opsel_hi_2 = uint8_t(high >> 2),
                                                         .clamp = uint8_t(clamp),
                                                         .src0 = 256,
                                                         .src1 = 257,
                                                         .src2 = uint16_t(sources == 3 ? 258 : 0),
                                                         .opsel_hi = uint8_t(high & 3)});
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .succeeded())
                << operation;
            ASSERT_EQ(fixture.recorder->reads.size(), sources);
            ASSERT_EQ(fixture.recorder->writes.size(), 1u);
            EXPECT_EQ(fixture.recorder->writes[0].bytes, 0xF);
            EXPECT_EQ(fixture.recorder->writes[0].lanes, exec_mask);
            for (uint32_t lane = 0; lane < lanes; ++lane) {
              uint32_t expected = inputs[0][lane];
              if (exec_mask & (1ULL << lane)) {
                expected = 0;
                for (unsigned half = 0; half < 2; ++half) {
                  const unsigned selectors = half ? high : low;
                  int64_t values[3];
                  const bool signed_operation = operation == 0 || operation == 2 ||
                                                operation == 4 || operation == 6 ||
                                                operation == 11 || operation == 12;
                  for (unsigned source = 0; source < 3; ++source) {
                    uint32_t raw =
                        (inputs[source][lane] >> ((selectors & (1u << source)) ? 16 : 0)) & 0xffff;
                    values[source] =
                        int64_t(raw) - ((signed_operation && (raw & 0x8000)) ? 65536 : 0);
                  }
                  const int64_t a = values[0], b = values[1], c = values[2];
                  int64_t result;
                  switch (operation) {
                  case 0:
                  case 1:
                    result = a + b;
                    break;
                  case 2:
                  case 3:
                    result = a - b;
                    break;
                  case 4:
                  case 5:
                    result = std::min(a, b);
                    break;
                  case 6:
                  case 7:
                    result = std::max(a, b);
                    break;
                  case 8:
                    result = a * b;
                    break;
                  case 9:
                    result = uint32_t(b) << (a & 15);
                    break;
                  case 10:
                    result = uint32_t(b) >> (a & 15);
                    break;
                  case 11: {
                    const int64_t divisor = 1LL << (a & 15);
                    result = b >= 0 ? b / divisor : -((-b + divisor - 1) / divisor);
                    break;
                  }
                  default:
                    result = a * b + c;
                    break;
                  }
                  if (clamp && (operation < 4 || operation >= 12))
                    result = signed_operation ? std::clamp<int64_t>(result, -32768, 32767)
                                              : std::clamp<int64_t>(result, 0, 65535);
                  expected |= (uint32_t(result) & 0xffff) << (16 * half);
                }
              }
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), expected)
                  << operation << " " << selection << " " << clamp << " " << lane;
            }
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, DecodedInteger16SelectorsClampAliasesAndInactiveHalves) {
  struct Operation {
    uint16_t rdna3_opcode, rdna4_opcode;
    unsigned sources;
  };
  const Operation operations[] = {
      {rdna3::kVNotB16Vop3, rdna4::kVNotB16Vop3, 1},
      {rdna3::kVAddNcU16Vop3, rdna4::kVAddNcU16Vop3, 2},
      {rdna3::kVAshrrevI16Vop3, rdna4::kVAshrrevI16Vop3, 2},
      {rdna3::kVMadI16Vop3, rdna4::kVMadI16Vop3, 3},
  };
  for (bool is_rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      BridgeFixture fixture(is_rdna3, lanes);
      for (unsigned op = 0; op < 4; ++op) {
        const auto &test = operations[op];
        for (unsigned opsel = 0; opsel < 16; ++opsel) {
          if (opsel & (7u & ~((1u << test.sources) - 1)))
            continue;
          for (bool clamp : {false, true}) {
            const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000180000005ULL;
            fixture.wave->set_exec(exec_mask);
            std::array<std::array<uint32_t, 64>, 3> inputs;
            for (uint32_t lane = 0; lane < lanes; ++lane) {
              inputs[0][lane] = 0x8000ffffu ^ (lane * 0x13571357u);
              inputs[1][lane] = 0x7fff0002u ^ (lane * 0x12345679u);
              inputs[2][lane] = 0xfffe0003u ^ (lane * 0x01010101u);
              for (unsigned src = 0; src < 3; ++src)
                fixture.cu->write_vgpr(fixture.base() + src, lane, inputs[src][lane]);
            }
            const auto words =
                is_rdna3 ? rdna3::build_vop3(test.rdna3_opcode,
                                             {.vdst = 0,
                                              .op_sel = uint8_t(opsel),
                                              .clamp = uint8_t(clamp),
                                              .src0 = 256,
                                              .src1 = uint16_t(test.sources > 1 ? 257 : 0),
                                              .src2 = uint16_t(test.sources > 2 ? 258 : 0)})
                         : rdna4::build_vop3(test.rdna4_opcode,
                                             {.vdst = 0,
                                              .opsel = uint8_t(opsel),
                                              .clamp = uint8_t(clamp),
                                              .src0 = 256,
                                              .src1 = uint16_t(test.sources > 1 ? 257 : 0),
                                              .src2 = uint16_t(test.sources > 2 ? 258 : 0)});
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .succeeded());
            for (uint32_t lane = 0; lane < lanes; ++lane) {
              const uint32_t a = (inputs[0][lane] >> ((opsel & 1) ? 16 : 0)) & 0xffff;
              const uint32_t b = (inputs[1][lane] >> ((opsel & 2) ? 16 : 0)) & 0xffff;
              const uint32_t c = (inputs[2][lane] >> ((opsel & 4) ? 16 : 0)) & 0xffff;
              auto signed16 = [](uint32_t value) {
                return int64_t(value) - ((value & 0x8000) ? 65536 : 0);
              };
              uint32_t result;
              if (op == 0)
                result = ~a;
              else if (op == 1)
                result = clamp ? std::min(a + b, 65535u) : a + b;
              else if (op == 2) {
                const unsigned shift = a & 15;
                const uint32_t sign = (b & 0x8000) ? 0xffffu : 0u;
                result = ((b ^ sign) >> shift) ^ sign;
              } else {
                const int64_t value = signed16(a) * signed16(b) + signed16(c);
                result = uint32_t(clamp ? std::clamp<int64_t>(value, -32768, 32767) : value);
              }
              const uint32_t shift = (opsel & 8) ? 16 : 0;
              const uint32_t bits = 0xffffu << shift;
              const uint32_t expected =
                  ((exec_mask >> lane) & 1)
                      ? (inputs[0][lane] & ~bits) | ((result & 0xffffu) << shift)
                      : inputs[0][lane];
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), expected)
                  << "op=" << op << " opsel=" << opsel << " clamp=" << clamp << " lane=" << lane;
            }
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, Integer16RecordsOnlySelectedBytesAndNoPreservationRead) {
  for (uint32_t lanes : {32u, 64u}) {
    for (uint64_t cpu : {RJ_ARITH_CPU_BASELINE, rj_arith_init_cpu_flags()}) {
      for (uint64_t exec_mask : {0ULL, 0x8000000180000005ULL}) {
        BridgeFixture fixture(false, lanes);
        fixture.wave->set_exec(exec_mask);
        fixture.wave->set_vgpr_write_mask(~4ULL);
        rdna4::Operand d(16, rdna4::OperandType::OPR_VGPR, 3);
        rdna4::Operand a(16, rdna4::OperandType::OPR_SRC_VGPR, 256);
        rdna4::Operand b(16, rdna4::OperandType::OPR_SRC_VGPR, 257);
        rdna4::Operand c(16, rdna4::OperandType::OPR_SRC_VGPR, 258);
        const uint64_t active = fixture.wave->exec();
        arith::execute_integer16<3>(
            *fixture.wave, rj_arith_v_mad_u16_w32, rj_arith_v_mad_u16_w64, cpu,
            RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_C | RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_CLAMP, d,
            a, &b, &c);
        ASSERT_FALSE(fixture.wave->instruction_execution_failed());
        if (!active) {
          EXPECT_TRUE(fixture.recorder->reads.empty());
          EXPECT_TRUE(fixture.recorder->writes.empty());
          continue;
        }
        ASSERT_EQ(fixture.recorder->reads.size(), 3u);
        for (unsigned src = 0; src < 3; ++src) {
          EXPECT_EQ(fixture.recorder->reads[src].reg, fixture.base() + src);
          EXPECT_EQ(fixture.recorder->reads[src].lanes, active);
          EXPECT_EQ(fixture.recorder->reads[src].bytes, src == 1 ? 0x3 : 0xc);
        }
        ASSERT_EQ(fixture.recorder->writes.size(), 1u);
        EXPECT_EQ(fixture.recorder->writes[0].reg, fixture.base() + 3);
        EXPECT_EQ(fixture.recorder->writes[0].lanes, active & ~4ULL);
        EXPECT_EQ(fixture.recorder->writes[0].bytes, 0xc);
        for (uint32_t lane = 0; lane < lanes; ++lane)
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 3, lane),
                    ((active & ~4ULL) >> lane) & 1 ? 0xffff3c00u : 0x3c003c00u);
      }
    }
  }
}

TEST(ArithBridgeTest, Integer16RejectsWithoutPartialWrites) {
  BridgeFixture fixture(false);
  fixture.wave->set_exec(1);
  rdna4::Operand d(16, rdna4::OperandType::OPR_VGPR, 0);
  rdna4::Operand a(16, rdna4::OperandType::OPR_SRC_VGPR, 256);
  arith::execute_integer16<1>(*fixture.wave, rj_arith_v_not_b16_w32, rj_arith_v_not_b16_w64,
                              1ULL << 63, RJ_ARITH_ALU_HIGH_D, d, a);
  EXPECT_TRUE(fixture.wave->instruction_execution_failed());
  EXPECT_TRUE(fixture.recorder->writes.empty());
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), 0x3c003c00u);
}

TEST(ArithBridgeTest, DecodedNot16Vop1ResolvesPackedSourceAndDestinationHalves) {
  for (bool is_rdna3 : {false, true}) {
    BridgeFixture fixture(is_rdna3);
    fixture.wave->set_exec(1);
    for (unsigned source_high = 0; source_high < 2; ++source_high) {
      for (unsigned destination_high = 0; destination_high < 2; ++destination_high) {
        fixture.cu->write_vgpr(fixture.base(), 0, 0x1234abcdu);
        fixture.cu->write_vgpr(fixture.base() + 1, 0, 0xdeadbeefu);
        const auto words = is_rdna3
                               ? rdna3::build_vop1(rdna3::kVNotB16Vop1,
                                                   {.src0 = uint16_t(256 + 128 * source_high),
                                                    .vdst = uint8_t(1 + 128 * destination_high)})
                               : rdna4::build_vop1(rdna4::kVNotB16Vop1,
                                                   {.src0 = uint16_t(256 + 128 * source_high),
                                                    .vdst = uint8_t(1 + 128 * destination_high)});
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        const uint32_t value = source_high ? 0xedcbu : 0x5432u;
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 0),
                  destination_high ? (value << 16) | 0xbeefu : 0xdead0000u | value);
      }
    }
  }
}

TEST(ArithBridgeTest, Integer16DppRetainsSourceSuppressionAndDestinationHalf) {
  BridgeFixture fixture(false);
  fixture.wave->set_exec(0xffffffffu);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    fixture.cu->write_vgpr(fixture.base(), lane, ((100u + lane) << 16) | 7u);
    fixture.cu->write_vgpr(fixture.base() + 1, lane, (2u << 16) | 1u);
    fixture.cu->write_vgpr(fixture.base() + 2, lane, 0xdeadbeefu);
  }
  rdna4::Vop3VopDpp16MachineInst raw{};
  raw.encoding = 0x35;
  raw.op = rdna4::kVAddNcU16Vop3;
  raw.vdst = 2;
  raw.opsel = 0xb;
  raw.clamp = 1;
  raw.src0 = amdgpu::SRC_DPP;
  raw.src1 = 257;
  raw.vsrc0 = 0;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  raw.fi = 1;
  raw.bound_ctrl = 0;
  raw.row_mask = 0xf;
  raw.bank_mask = 0xe;
  rdna4::VAddNcU16Vop3 instruction(reinterpret_cast<const rdna4::MachineInst *>(&raw));
  ASSERT_TRUE(fixture.cu->execute_instruction(&instruction, *fixture.wave).succeeded());
  for (uint32_t lane = 0; lane < 32; ++lane) {
    const bool written = (lane % 16) >= 4;
    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lane),
              written ? ((101u + lane) << 16) | 0xbeefu : 0xdeadbeefu);
  }
}

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
        arith::exec_swmmac(*fixture.wave, rj_arith_v_swmmac_f32_16x16x32_f16_w32, flags, 0,
                           destination, fixture.base(), fixture.base() + 4, fixture.base() + 12, 4,
                           8, 8, 32);
      else
        arith::exec_wmma(*fixture.wave, rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12, flags, 0,
                         destination, fixture.base(), fixture.base() + 8, fixture.base() + 16, 4, 8,
                         std::nullopt);
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

TEST(ArithBridgeTest, IntegerMultiplyClampIsIgnoredOrSaturatingPerInstruction) {
  struct Case {
    uint16_t rdna3_opcode, rdna4_opcode;
    uint32_t a, b, expected;
  };
  const Case cases[] = {
      {rocjitsu::rdna3::kVMulLoU32Vop3, rdna4::kVMulLoU32Vop3, 0xffffffffu, 2, 0xfffffffeu},
      {rocjitsu::rdna3::kVMulHiU32Vop3, rdna4::kVMulHiU32Vop3, 0xffffffffu, 2, 1},
      {rocjitsu::rdna3::kVMulHiI32Vop3, rdna4::kVMulHiI32Vop3, 0x80000000u, 2, 0xffffffffu},
      {rocjitsu::rdna3::kVMulHiI32I24Vop3, rdna4::kVMulHiI32I24Vop3, 0x00800000u, 0x200,
       0xffffffffu},
      {rocjitsu::rdna3::kVMulHiU32U24Vop3, rdna4::kVMulHiU32U24Vop3, 0x00ffffffu, 0x00ffffffu,
       0xffffu},
      {rocjitsu::rdna3::kVMulI32I24Vop3, rdna4::kVMulI32I24Vop3, 0x007fffffu, 0x007fffffu,
       0x7fffffffu},
      {rocjitsu::rdna3::kVMulU32U24Vop3, rdna4::kVMulU32U24Vop3, 0x00ffffffu, 0x00ffffffu,
       0xffffffffu},
  };
  for (bool rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      BridgeFixture fixture(rdna3, lanes);
      ASSERT_NE(fixture.decoder, nullptr);
      fixture.wave->set_exec(5);
      for (const auto &test : cases) {
        SCOPED_TRACE(test.rdna4_opcode);
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          fixture.cu->write_vgpr(fixture.base(), lane, test.a);
          fixture.cu->write_vgpr(fixture.base() + 1, lane, test.b);
          fixture.cu->write_vgpr(fixture.base() + 2, lane, 0xdeadbeefu);
        }
        const auto words =
            rdna3 ? rocjitsu::rdna3::build_vop3(test.rdna3_opcode,
                                                {.vdst = 2, .clamp = 1, .src0 = 256, .src1 = 257})
                  : rdna4::build_vop3(test.rdna4_opcode,
                                      {.vdst = 2, .clamp = 1, .src0 = 256, .src1 = 257});
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        for (uint32_t lane = 0; lane < lanes; ++lane)
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lane),
                    lane == 0 || lane == 2 ? test.expected : 0xdeadbeefu);
      }
    }
  }
}

TEST(ArithBridgeTest, IntegerVectorMasksAliasingAndBroadcast) {
  for (uint32_t lanes : {32u, 64u})
    for (uint64_t cpu : {RJ_ARITH_CPU_BASELINE, rj_arith_init_cpu_flags()}) {
      BridgeFixture fixture(false, lanes);
      const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000100000005ULL;
      fixture.wave->set_exec(exec_mask);
      rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 0);
      rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
      rdna4::Operand b(32, rdna4::OperandType::OPR_SRC, 129); // Inline integer 1.
      arith::execute_binary32(*fixture.wave, rj_arith_v_add_nc_u32_w32, rj_arith_v_add_nc_u32_w64,
                              cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, d, a,
                              b);
      ASSERT_FALSE(fixture.wave->instruction_execution_failed());
      ASSERT_EQ(fixture.recorder->reads.size(), 1u);
      EXPECT_EQ(fixture.recorder->reads[0].lanes, exec_mask);
      ASSERT_EQ(fixture.recorder->writes.size(), 1u);
      EXPECT_EQ(fixture.recorder->writes[0].lanes, exec_mask);
      for (uint32_t lane = 0; lane < lanes; ++lane)
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                  0x3c003c00u + ((exec_mask >> lane) & 1));
    }
}

TEST(ArithBridgeTest, VectorRejectedFlagsDoNotWriteOrNotify) {
  BridgeFixture fixture(false);
  rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 0);
  rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
  arith::execute_binary32(*fixture.wave, rj_arith_v_and_b32_w32, rj_arith_v_and_b32_w64, 1ULL << 63,
                          0, d, a, a);
  EXPECT_TRUE(fixture.wave->instruction_execution_failed());
  EXPECT_TRUE(fixture.recorder->writes.empty());
  for (uint32_t lane = 0; lane < 32; ++lane)
    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), 0x3c003c00u);
}

TEST(ArithBridgeTest, UnaryVectorPreservesInactiveAliasedLanesWithClamp) {
  for (uint32_t lanes : {32u, 64u}) {
    for (uint64_t cpu : {RJ_ARITH_CPU_BASELINE, rj_arith_init_cpu_flags()}) {
      BridgeFixture fixture(false, lanes);
      const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000100000005ULL;
      fixture.wave->set_exec(exec_mask);
      rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 0);
      rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
      arith::execute_unary32(*fixture.wave, rj_arith_v_not_b32_w32, rj_arith_v_not_b32_w64, cpu,
                             RJ_ARITH_ALU_CLAMP, d, a);
      ASSERT_FALSE(fixture.wave->instruction_execution_failed());
      ASSERT_EQ(fixture.recorder->reads.size(), 1u);
      EXPECT_EQ(fixture.recorder->reads[0].lanes, exec_mask);
      ASSERT_EQ(fixture.recorder->writes.size(), 1u);
      EXPECT_EQ(fixture.recorder->writes[0].lanes, exec_mask);
      for (uint32_t lane = 0; lane < lanes; ++lane)
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                  (exec_mask >> lane) & 1 ? ~0x3c003c00u : 0x3c003c00u);
    }
  }
}

TEST(ArithBridgeTest, TernaryVectorReadsAllInputsBeforeMaskedAliasedCommit) {
  for (uint32_t lanes : {32u, 64u}) {
    for (uint64_t cpu : {RJ_ARITH_CPU_BASELINE, rj_arith_init_cpu_flags()}) {
      BridgeFixture fixture(false, lanes);
      const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000100000005ULL;
      fixture.wave->set_exec(exec_mask);
      const uint64_t write_mask = exec_mask & ~4ULL;
      fixture.wave->set_vgpr_write_mask(write_mask);
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        fixture.cu->write_vgpr(fixture.base(), lane, 0xff00ff00u);
        fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x12345678u);
        fixture.cu->write_vgpr(fixture.base() + 2, lane, 0xabcdef01u);
      }
      rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 2);
      rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
      rdna4::Operand b(32, rdna4::OperandType::OPR_SRC_VGPR, 257);
      rdna4::Operand c(32, rdna4::OperandType::OPR_SRC_VGPR, 258);
      arith::execute_ternary32(*fixture.wave, rj_arith_v_bfi_b32_w32, rj_arith_v_bfi_b32_w64,
                               cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                               RJ_ARITH_ALU_CLAMP, d, a, b, c);
      ASSERT_FALSE(fixture.wave->instruction_execution_failed());
      ASSERT_EQ(fixture.recorder->reads.size(), 3u);
      for (const auto &read : fixture.recorder->reads)
        EXPECT_EQ(read.lanes, write_mask);
      ASSERT_EQ(fixture.recorder->writes.size(), 1u);
      EXPECT_EQ(fixture.recorder->writes[0].reg, fixture.base() + 2);
      EXPECT_EQ(fixture.recorder->writes[0].lanes, write_mask);
      for (uint32_t lane = 0; lane < lanes; ++lane)
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lane),
                  (write_mask >> lane) & 1 ? 0x12cd5601u : 0xabcdef01u);
    }
  }
}

TEST(ArithBridgeTest, DelegatedVectorInputIsNotReadAsItsOriginalVgpr) {
  BridgeFixture fixture(false);
  fixture.wave->set_exec(0x5);
  rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 0);
  rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
  rdna4::Operand b(32, rdna4::OperandType::OPR_SRC, 129);
  uint32_t values[32];
  for (unsigned i = 0; i < 32; ++i)
    values[i] = 100 + i;
  StagedOperand staged(a, values, 32);
  ScopedOperandDelegate delegate(a, &staged);
  arith::execute_binary32(*fixture.wave, rj_arith_v_add_nc_u32_w32, rj_arith_v_add_nc_u32_w64, 0, 0,
                          d, a, b);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  EXPECT_TRUE(fixture.recorder->reads.empty());
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), 101u);
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 2), 103u);
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 1), 0x3c003c00u);
}

TEST(ArithBridgeTest, VectorInvalidSourceRetainsZeroReadBehavior) {
  BridgeFixture fixture(false);
  fixture.wave->set_exec(1);
  rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 0);
  rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256 + 256);
  rdna4::Operand b(32, rdna4::OperandType::OPR_SRC, 129);
  arith::execute_binary32(*fixture.wave, rj_arith_v_add_nc_u32_w32, rj_arith_v_add_nc_u32_w64, 0, 0,
                          d, a, b);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  EXPECT_TRUE(fixture.recorder->reads.empty());
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), 1u);
}

TEST(ArithBridgeTest, VectorZeroExecDoesNotReadOrWrite) {
  BridgeFixture fixture(false);
  fixture.wave->set_exec(0);
  rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 0);
  rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
  arith::execute_binary32(*fixture.wave, rj_arith_v_and_b32_w32, rj_arith_v_and_b32_w64, 0, 0, d, a,
                          a);
  EXPECT_TRUE(fixture.recorder->reads.empty());
  EXPECT_TRUE(fixture.recorder->writes.empty());
  EXPECT_FALSE(fixture.wave->instruction_execution_failed());
}

TEST(ArithBridgeTest, IntegerCompareUsesHalfSelectorsAndClearsInactiveMaskBits) {
  for (uint32_t lanes : {32u, 64u}) {
    for (uint64_t cpu : {RJ_ARITH_CPU_BASELINE, rj_arith_init_cpu_flags()}) {
      BridgeFixture fixture(false, lanes);
      rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
      rdna4::Operand b(32, rdna4::OperandType::OPR_SRC_VGPR, 257);
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        fixture.cu->write_vgpr(fixture.base(), lane, 0x00020001);
        fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x00010002);
      }
      const uint64_t active = lanes == 32 ? 0x80000005ULL : 0x8000000100000005ULL;
      for (uint64_t exec_mask : {uint64_t{0}, active}) {
        fixture.wave->set_exec(exec_mask);
        for (uint64_t mode :
             {uint64_t{0}, uint64_t(RJ_ARITH_ALU_HIGH_A), uint64_t(RJ_ARITH_ALU_HIGH_B),
              uint64_t(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B)}) {
          fixture.recorder->reads.clear();
          uint64_t result = ~0ULL;
          arith::execute_compare<16>(*fixture.wave, rj_arith_v_cmp_eq_i16_w32,
                                     rj_arith_v_cmp_eq_i16_w64, cpu, mode, a, b, false,
                                     [&](uint64_t value) { result = value; });
          ASSERT_FALSE(fixture.wave->instruction_execution_failed());
          const bool equal = mode == RJ_ARITH_ALU_HIGH_A || mode == RJ_ARITH_ALU_HIGH_B;
          EXPECT_EQ(result, equal ? exec_mask : 0);
          EXPECT_EQ(fixture.recorder->reads.size(), exec_mask ? 2u : 0u);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, IntegerCompareReadsPackedVopcHighHalf) {
  BridgeFixture fixture(false);
  fixture.wave->set_exec(5);
  // True16 encoding bit 7 selects the high half of v0; v1 uses its low half.
  rdna4::Operand a(16, rdna4::OperandType::OPR_SRC, 256 + 128, true);
  rdna4::Operand b(16, rdna4::OperandType::OPR_VGPR, 1, true);
  fixture.cu->write_vgpr(fixture.base(), 0, 0x00070003);
  fixture.cu->write_vgpr(fixture.base(), 2, 0x00080003);
  fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x00090007);
  fixture.cu->write_vgpr(fixture.base() + 1, 2, 0x00090007);
  uint64_t result = 0;
  arith::execute_compare<16>(*fixture.wave, rj_arith_v_cmp_eq_i16_w32, rj_arith_v_cmp_eq_i16_w64,
                             RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, a, b, false,
                             [&](uint64_t value) { result = value; });
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  EXPECT_EQ(result, 1u);
  ASSERT_EQ(fixture.recorder->reads.size(), 4u);
  for (const auto &read : fixture.recorder->reads) {
    EXPECT_EQ(read.bytes, read.reg == fixture.base() ? 0xcu : 0x3u);
    EXPECT_EQ(read.lanes & ~5ULL, 0u);
  }
}

TEST(ArithBridgeTest, IntegerCompare64ReadsBothWordsAndSnapshotsAliasedSgprInput) {
  for (uint32_t lanes : {32u, 64u}) {
    BridgeFixture fixture(false, lanes);
    const auto sgpr = fixture.wave->sgpr_alloc().base;
    fixture.cu->write_sgpr(sgpr, 7);
    fixture.cu->write_sgpr(sgpr + 1, 1);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      fixture.cu->write_vgpr(fixture.base(), lane, 7);
      fixture.cu->write_vgpr(fixture.base() + 1, lane, lane & 1 ? 2 : 1);
    }
    rdna4::Operand a(64, rdna4::OperandType::OPR_SRC, 0);
    rdna4::Operand b(64, rdna4::OperandType::OPR_SRC_VGPR, 256);
    const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000100000005ULL;
    fixture.wave->set_exec(exec_mask);
    fixture.recorder->scalar_reads.clear();
    arith::execute_compare<64>(*fixture.wave, rj_arith_v_cmp_eq_u64_w32, rj_arith_v_cmp_eq_u64_w64,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, a, b, false,
                               [&](uint64_t result) {
                                 RegisterAccess registers(*fixture.wave);
                                 registers.write_sgpr(sgpr, uint32_t(result));
                                 if (lanes == 64)
                                   registers.write_sgpr(sgpr + 1, uint32_t(result >> 32));
                               });
    ASSERT_FALSE(fixture.wave->instruction_execution_failed());
    EXPECT_EQ(fixture.recorder->scalar_reads, (std::vector<uint32_t>{sgpr, sgpr + 1}));
    ASSERT_EQ(fixture.recorder->reads.size(), 2u);
    EXPECT_EQ(fixture.recorder->reads[0].reg, fixture.base());
    EXPECT_EQ(fixture.recorder->reads[1].reg, fixture.base() + 1);
    EXPECT_EQ(fixture.cu->read_sgpr(sgpr), 5u);
    EXPECT_EQ(fixture.cu->read_sgpr(sgpr + 1), 1u);
  }
}

TEST(ArithBridgeTest, ConstantCompareSkipsSourceReadsAndRejectedCallsDoNotCommit) {
  BridgeFixture fixture(false, 64);
  fixture.wave->set_exec(0x8000000100000005ULL);
  rdna4::Operand a(64, rdna4::OperandType::OPR_SRC_VGPR, 256);
  uint64_t result = 0;
  arith::execute_compare<64>(*fixture.wave, rj_arith_v_cmp_t_u64_w32, rj_arith_v_cmp_t_u64_w64, 0,
                             0, a, a, true, [&](uint64_t value) { result = value; });
  EXPECT_EQ(result, fixture.wave->exec());
  EXPECT_TRUE(fixture.recorder->reads.empty());
  bool committed = false;
  arith::execute_compare<64>(*fixture.wave, rj_arith_v_cmp_eq_u64_w32, rj_arith_v_cmp_eq_u64_w64,
                             1ULL << 63, 0, a, a, false, [&](uint64_t) { committed = true; });
  EXPECT_TRUE(fixture.wave->instruction_execution_failed());
  EXPECT_FALSE(committed);
}

TEST(ArithBridgeTest, ClassCompare64ReadsOneClassMaskWordAtTheAllocationBoundary) {
  for (uint32_t lanes : {32u, 64u}) {
    BridgeFixture fixture(false, lanes);
    const uint64_t exec_mask = lanes == 64 ? (1ULL << 32) | 5 : 5;
    fixture.wave->set_exec(exec_mask);
    fixture.cu->write_vgpr(fixture.base(), 0, 1);
    fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x7ff80000); // Quiet NaN.
    fixture.cu->write_vgpr(fixture.base(), 2, 0);
    fixture.cu->write_vgpr(fixture.base() + 1, 2, 0x80000000); // Negative zero.
    if (lanes == 64) {
      fixture.cu->write_vgpr(fixture.base(), 32, 0);
      fixture.cu->write_vgpr(fixture.base() + 1, 32, 0x7ff00000); // Positive infinity.
    }
    for (uint32_t lane = 0; lane < lanes; ++lane)
      fixture.cu->write_vgpr(fixture.base() + 255, lane,
                             RJ_ARITH_CLASS_QNAN | RJ_ARITH_CLASS_NEG_ZERO);
    rdna4::Operand a(64, rdna4::OperandType::OPR_SRC_VGPR, 256);
    rdna4::Operand b(32, rdna4::OperandType::OPR_SRC_VGPR, 511);
    for (uint64_t cpu : {RJ_ARITH_CPU_BASELINE, rj_arith_init_cpu_flags()}) {
      fixture.recorder->reads.clear();
      uint64_t result = ~0ULL;
      arith::execute_compare<64, 32>(*fixture.wave, rj_arith_v_cmp_class_f64_w32,
                                     rj_arith_v_cmp_class_f64_w64,
                                     cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0,
                                     a, b, false, [&](uint64_t value) { result = value; });
      ASSERT_FALSE(fixture.wave->instruction_execution_failed());
      EXPECT_EQ(result, 5u);
      ASSERT_EQ(fixture.recorder->reads.size(), 3u);
      EXPECT_EQ(fixture.recorder->reads[2].reg, fixture.base() + 255);
      EXPECT_EQ(fixture.recorder->reads[2].lanes, exec_mask);
    }
  }
}

TEST(ArithBridgeTest, ClassCompareUsesRawHalfClassesAndIndependentHalfSelectors) {
  BridgeFixture fixture(false);
  fixture.wave->set_exec(1);
  fixture.wave->set_mode_raw(0); // Classification must not flush subnormals.
  fixture.cu->write_vgpr(fixture.base(), 0, 0x80013c00); // High half: negative subnormal.
  fixture.cu->write_vgpr(fixture.base() + 1, 0, RJ_ARITH_CLASS_NEG_SUBNORMAL << 16);
  rdna4::Operand a(16, rdna4::OperandType::OPR_SRC, 256);
  rdna4::Operand b(16, rdna4::OperandType::OPR_SRC, 257);
  for (uint64_t mode :
       {uint64_t(RJ_ARITH_ALU_HIGH_A), uint64_t(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B),
        uint64_t(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_ABS_A)}) {
    uint64_t result = ~0ULL;
    arith::execute_compare<16, 32>(*fixture.wave, rj_arith_v_cmp_class_f16_w32,
                                   rj_arith_v_cmp_class_f16_w64,
                                   fixture.wave->cu().arithmetic_flags(), mode, a, b, false,
                                   [&](uint64_t value) { result = value; });
    ASSERT_FALSE(fixture.wave->instruction_execution_failed());
    EXPECT_EQ(result, mode == (RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B) ? 1u : 0u);
  }
}

TEST(ArithBridgeTest, FloatCompareReportsRepeatedCausesAndPreservesUnrelatedState) {
  for (uint32_t lanes : {32u, 64u}) {
    BridgeFixture fixture(false, lanes, parse_arithmetic_policy("bit_exact_empirical"));
    fixture.wave->set_exec(5);
    fixture.wave->set_mode_raw(0xf0);
    fixture.wave->set_trapsts(0x80000001u);
    rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
    rdna4::Operand b(32, rdna4::OperandType::OPR_SRC_VGPR, 257);
    fixture.cu->write_vgpr(fixture.base(), 0, 0x7f800001); // Signaling NaN.
    fixture.cu->write_vgpr(fixture.base(), 1, 1);          // Inactive subnormal.
    fixture.cu->write_vgpr(fixture.base(), 2, 0x3f800000);
    for (uint32_t lane = 0; lane < lanes; ++lane)
      fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x3f800000);
    for (int repeat = 0; repeat < 2; ++repeat) {
      fixture.wave->clear_pending_alu_causes();
      uint64_t result = ~0ULL;
      arith::execute_float_compare<32>(*fixture.wave, rj_arith_v_cmp_eq_f32_w32_gfx12,
                                       rj_arith_v_cmp_eq_f32_w64_gfx12,
                                       fixture.wave->cu().arithmetic_flags(), 0, a, b, false,
                                       [&](uint64_t value) { result = value; });
      ASSERT_FALSE(fixture.wave->instruction_execution_failed());
      EXPECT_EQ(result, 4u);
      EXPECT_EQ(fixture.wave->trapsts(), 0x80000001u);
      EXPECT_EQ(fixture.wave->pending_alu_causes(), RJ_ARITH_EXCEPTION_INVALID);
    }
  }
}

TEST(ArithBridgeTest, FloatCompareHonorsGuestDenormalsAndExplicitLoosePolicy) {
  for (bool loose : {false, true})
    for (bool flush : {false, true}) {
      BridgeFixture fixture(false, 32,
                            parse_arithmetic_policy(loose ? "loose" : "bit_exact_empirical"));
      fixture.wave->set_exec(1);
      fixture.wave->set_mode_raw(flush ? 0 : 0xf0);
      fixture.wave->set_trapsts(0x80000000u);
      fixture.cu->write_vgpr(fixture.base(), 0, 1);
      rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
      rdna4::Operand b(32, rdna4::OperandType::OPR_SRC, 128); // Inline zero.
      uint64_t result = ~0ULL;
      arith::execute_float_compare<32>(*fixture.wave, rj_arith_v_cmp_eq_f32_w32_gfx12,
                                       rj_arith_v_cmp_eq_f32_w64_gfx12,
                                       fixture.wave->cu().arithmetic_flags(), 0, a, b, false,
                                       [&](uint64_t value) { result = value; });
      ASSERT_FALSE(fixture.wave->instruction_execution_failed());
      EXPECT_EQ(result, flush ? 1u : 0u);
      const uint32_t causes = !loose && !flush ? RJ_ARITH_EXCEPTION_INPUT_DENORM : 0;
      EXPECT_EQ(fixture.wave->trapsts(), 0x80000000u | causes);
      EXPECT_EQ(fixture.wave->pending_alu_causes(), causes);
    }
}

TEST(ArithBridgeTest, FloatCompareHalfSelectorAndInlineFloatKeepTheirFormat) {
  BridgeFixture fixture(false);
  fixture.wave->set_exec(1);
  fixture.wave->set_mode_raw(0xf0);
  fixture.cu->write_vgpr(fixture.base(), 0, 0x3c008000);
  rdna4::Operand a(16, rdna4::OperandType::OPR_SRC, 256);
  rdna4::Operand b(16, rdna4::OperandType::OPR_SRC, 242); // Inline half 1.0.
  for (uint64_t mode :
       {uint64_t(RJ_ARITH_ALU_HIGH_A), uint64_t(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_NEG_A)}) {
    uint64_t result = ~0ULL;
    arith::execute_float_compare<16>(*fixture.wave, rj_arith_v_cmp_eq_f16_w32_gfx12,
                                     rj_arith_v_cmp_eq_f16_w64_gfx12,
                                     fixture.wave->cu().arithmetic_flags(), mode, a, b, false,
                                     [&](uint64_t value) { result = value; });
    ASSERT_FALSE(fixture.wave->instruction_execution_failed());
    EXPECT_EQ(result, mode & RJ_ARITH_ALU_NEG_A ? 0u : 1u);
  }
}

TEST(ArithBridgeTest, FloatCompareDppFilteredLanesCannotRaiseExceptions) {
  BridgeFixture fixture(false, 32, parse_arithmetic_policy("bit_exact_empirical"));
  fixture.wave->set_exec((1u << 0) | (1u << 16) | (1u << 17));
  fixture.wave->set_mode_raw(0xf0);
  fixture.cu->write_vgpr(fixture.base(), 0, 0x7f800001); // Disabled row.
  fixture.cu->write_vgpr(fixture.base(), 16, 0x3f800000);
  fixture.cu->write_vgpr(fixture.base() + 1, 16, 0x7f800001); // Invalid DPP source lane.
  fixture.cu->write_vgpr(fixture.base() + 1, 17, 0x3f800000);
  rdna4::VopcVopDpp16MachineInst raw{};
  raw.src0 = SRC_DPP;
  raw.vsrc0 = 0;
  raw.vsrc1 = 1;
  raw.dpp_ctrl = dpp::ROW_SHR1;
  raw.fi = 1;
  raw.bound_ctrl = 0;
  raw.bank_mask = 0xf;
  raw.row_mask = 2;
  rdna4::VCmpEqF32Vopc instruction(reinterpret_cast<const rdna4::MachineInst *>(&raw));
  instruction.execute_impl(*fixture.wave);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  EXPECT_EQ(fixture.wave->vcc(), 1u << 17);
  EXPECT_EQ(fixture.wave->trapsts(), 0u);
  EXPECT_EQ(fixture.wave->pending_alu_causes(), 0u);
}

TEST(ArithBridgeTest, RejectedFloatCompareDppDoesNotChangeExecOrExceptionState) {
  BridgeFixture fixture(true, 32, parse_arithmetic_policy("bit_exact_empirical"));
  constexpr uint64_t old_exec = 0xa5a55a5affff005bULL;
  fixture.wave->set_exec_raw(old_exec);
  fixture.wave->set_vcc_raw(0x123456789abcdef0ULL);
  fixture.wave->set_trapsts(0x80000001u);
  rdna3::VopcVopDpp16MachineInst raw{};
  raw.src0 = SRC_DPP;
  raw.vsrc0 = 0;
  raw.vsrc1 = 1;
  raw.dpp_ctrl = dpp::ROW_SHR1;
  raw.fi = 1;
  raw.bound_ctrl = 0;
  raw.bank_mask = 0xf;
  raw.row_mask = 0xf;
  rdna3::VCmpxEqF32Vopc instruction(reinterpret_cast<const rdna3::MachineInst *>(&raw));
  instruction.execute_impl(*fixture.wave);
  EXPECT_TRUE(fixture.wave->instruction_execution_failed());
  EXPECT_EQ(fixture.wave->exec_raw(), old_exec);
  EXPECT_EQ(fixture.wave->vcc(), 0x123456789abcdef0ULL);
  EXPECT_EQ(fixture.wave->trapsts(), 0x80000001u);
  EXPECT_EQ(fixture.wave->pending_alu_causes(), 0u);
}

const uint64_t kExactEmpiricalNoFpStatus = RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;

TEST(ArithBridgeTest, MatrixIgnoresExecAndPreservesAliasedInputsUntilCommit) {
  BridgeFixture fixture(false);
  fixture.wave->set_exec(0);
  arith::exec_wmma(*fixture.wave, rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12,
                   kExactEmpiricalNoFpStatus, 0, fixture.base(), fixture.base(), fixture.base() + 8,
                   fixture.base() + 16, 4, 8, std::nullopt);
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
  arith::exec_wmma(*fixture.wave, rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12,
                   kExactEmpiricalNoFpStatus | (1ULL << 63), 0, fixture.base(), fixture.base(),
                   fixture.base() + 8, fixture.base() + 16, 4, 8, std::nullopt);
  EXPECT_TRUE(fixture.wave->instruction_execution_failed());
  EXPECT_TRUE(fixture.recorder->writes.empty());
  for (uint32_t reg = 0; reg < 8; ++reg)
    for (uint32_t lane = 0; lane < 32; ++lane)
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + reg, lane), 0x3c003c00u);
}

TEST(ArithBridgeTest, DecodedIu4UsesOnlyConsumedFactorLanes) {
  for (uint32_t lanes : {32u, 64u}) {
    BridgeFixture fixture(false, lanes);
    fixture.wave->set_exec(0); // Matrix instructions ignore EXEC.
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t factor = lane < 32 ? 0x11111111u : 0xffffffffu;
      fixture.cu->write_vgpr(fixture.base(), lane, factor);
      fixture.cu->write_vgpr(fixture.base() + 8, lane, factor);
    }
    // V_WMMA_I32_16X16X16_IU4: A=v0, B=v8, C=v16, D=v24, unsigned inputs.
    const uint32_t words[] = {0xcc454018u, 0x04421100u, 0, 0};
    auto decoded = fixture.decoder->decode(words);
    ASSERT_TRUE(decoded.succeeded());
    auto instruction = std::move(decoded).value();
    ASSERT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wave).succeeded());
    ASSERT_EQ(fixture.recorder->reads.size(), 2u + 256u / lanes);
    for (const auto &read : fixture.recorder->reads) {
      const bool factor = read.reg == fixture.base() || read.reg == fixture.base() + 8;
      EXPECT_EQ(read.lanes, factor || lanes == 32 ? 0xffffffffu : ~uint64_t{0});
    }
    for (uint32_t reg = 0; reg < 256 / lanes; ++reg)
      for (uint32_t lane = 0; lane < lanes; ++lane)
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 24 + reg, lane), 16u);
  }
}

TEST(ArithBridgeTest, PackedResultPreservesUnselectedHalfAndObservesOnlyWrittenBytes) {
  BridgeFixture fixture(true);
  arith::exec_wmma(*fixture.wave, rj_arith_v_wmma_f16_16x16x16_f16_w32, kExactEmpiricalNoFpStatus,
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
  arith::exec_swmmac(*fixture.wave, rj_arith_v_swmmac_f32_16x16x32_f16_w32,
                     kExactEmpiricalNoFpStatus, 0, fixture.base() + 16, fixture.base(),
                     fixture.base() + 4, fixture.base() + 12, 4, 8, 8, 32);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  ASSERT_EQ(fixture.recorder->writes.size(), 8u);
  for (uint32_t reg = 16; reg < 24; ++reg)
    for (uint32_t lane = 0; lane < 32; ++lane)
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + reg, lane), 0x41880000u);
}

TEST(ArithBridgeTest, DecodedSparseIu4NormalizesSelectorAndObservesConsumedLanes) {
  for (uint32_t selector : {1u, 3u}) {
    BridgeFixture fixture(false, 64);
    for (uint32_t lane = 0; lane < 64; ++lane) {
      fixture.cu->write_vgpr(fixture.base(), lane, lane < 32 ? 0x11111111u : 0xffffffffu);
      fixture.cu->write_vgpr(fixture.base() + 8, lane, 0x11111111u);
      fixture.cu->write_vgpr(fixture.base() + 12, lane, 0x44444444u);
    }
    // V_SWMMAC_I32_16X16X32_IU4: A=v0, B=v8, index=v12, C/D=v16.
    // Only selector bit zero is meaningful; the higher bit aliases the same set.
    const uint32_t words[] = {0xcc554010u | (selector << 11), 0x04321100u, 0, 0};
    auto decoded = fixture.decoder->decode(words);
    ASSERT_TRUE(decoded.succeeded());
    auto instruction = std::move(decoded).value();
    ASSERT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wave).succeeded());
    ASSERT_EQ(fixture.recorder->reads.size(), 7u);
    for (const auto &read : fixture.recorder->reads) {
      const bool partial = read.reg == fixture.base() || read.reg == fixture.base() + 12;
      EXPECT_EQ(read.lanes, partial ? 0xffffffffu : ~uint64_t{0});
    }
    for (uint32_t reg = 16; reg < 20; ++reg)
      for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + reg, lane), 16u);
  }
}

TEST(ArithBridgeTest, FailedSparseInvocationPreservesItsTiedAccumulator) {
  BridgeFixture fixture(false);
  arith::exec_swmmac(*fixture.wave, rj_arith_v_swmmac_f32_16x16x32_f16_w32,
                     kExactEmpiricalNoFpStatus | (1ULL << 63), 0, fixture.base(), fixture.base(),
                     fixture.base() + 4, fixture.base() + 12, 4, 8, 8, 32);
  EXPECT_TRUE(fixture.wave->instruction_execution_failed());
  EXPECT_TRUE(fixture.recorder->writes.empty());
  for (uint32_t reg = 0; reg < 8; ++reg)
    for (uint32_t lane = 0; lane < 32; ++lane)
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + reg, lane), 0x3c003c00u);
}

TEST(ArithBridgeTest, DoubleBinaryPairAliasesAndPolicies) {
  for (bool rdna3 : {false, true})
    for (unsigned lanes : {32u, 64u})
      for (unsigned operation = 0; operation < (rdna3 ? 4u : 6u); ++operation)
        for (bool required : {false, true})
          for (bool preserve_denorm : {false, true})
            for (bool vop2 : {false, true}) {
              if (vop2 && (rdna3 || operation == 2 || operation == 3))
                continue;
              ArithmeticPolicy policy{};
              if (required)
                policy.semantics = ArithmeticSemantics::BitExactEmpirical;
              BridgeFixture fixture(rdna3, lanes, policy);
              fixture.wave->set_mode_raw(preserve_denorm ? 3u << 6 : 0);
              const uint64_t exec_mask = lanes == 32 ? 0x80000003ULL : 0x8000000380000003ULL;
              fixture.wave->set_exec(exec_mask);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                fixture.cu->write_vgpr(fixture.base(), lane, 0);
                fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x3ff80000); // 1.5
                fixture.cu->write_vgpr(fixture.base() + 2, lane, 0);
                fixture.cu->write_vgpr(fixture.base() + 3, lane, 0x40040000); // 2.5
              }
              const uint16_t ops3[] = {rdna3::kVMinF64Vop3, rdna3::kVMaxF64Vop3,
                                       rdna3::kVAddF64Vop3, rdna3::kVMulF64Vop3};
              const uint16_t ops4[] = {rdna4::kVMinNumF64Vop3,  rdna4::kVMaxNumF64Vop3,
                                       rdna4::kVMinimumF64Vop3, rdna4::kVMaximumF64Vop3,
                                       rdna4::kVAddF64Vop3,     rdna4::kVMulF64Vop3};
              auto words =
                  rdna3 ? rdna3::build_vop3(ops3[operation],
                                            {.vdst = 2, .src0 = 256, .src1 = 258, .omod = 1})
                        : rdna4::build_vop3(ops4[operation],
                                            {.vdst = 2, .src0 = 256, .src1 = 258, .omod = 1});
              if (vop2) {
                const uint16_t ops2[] = {rdna4::kVMinNumF64Vop2, rdna4::kVMaxNumF64Vop2, 0, 0,
                                         rdna4::kVAddF64Vop2,    rdna4::kVMulF64Vop2};
                words = {};
                words[0] =
                    rdna4::build_vop2(ops2[operation], {.src0 = 256, .vsrc1 = 2, .vdst = 2})[0];
              }
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              fixture.recorder->writes.clear();
              EXPECT_EQ(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .failed(),
                        required);
              uint64_t written[2] = {};
              for (const auto &write : fixture.recorder->writes) {
                ASSERT_GE(write.reg, 2u);
                ASSERT_LE(write.reg, 3u);
                EXPECT_EQ(write.bytes, 0xf);
                written[write.reg - 2] |= write.lanes;
              }
              EXPECT_EQ(written[0], required ? 0 : exec_mask);
              EXPECT_EQ(written[1], required ? 0 : exec_mask);
              const double core = operation < (rdna3 ? 2u : 4u) ? (operation % 2 ? 2.5 : 1.5)
                                                                : (operation % 2 ? 3.75 : 4.0);
              const double result = core * (vop2 || (rdna3 && preserve_denorm) ? 1 : 2);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                const uint64_t bits =
                    fixture.cu->read_vgpr(fixture.base() + 2, lane) |
                    (uint64_t(fixture.cu->read_vgpr(fixture.base() + 3, lane)) << 32);
                const double actual = std::bit_cast<double>(bits);
                EXPECT_NEAR(actual, !required && ((exec_mask >> lane) & 1) ? result : 2.5, 1e-12);
              }
            }
}

TEST(ArithBridgeTest, DoubleFmaPairAliasesModifiersAndPolicies) {
  // Loose FMA uses a computed std::fma reference. Dyadic inputs make the
  // modifier/packing witnesses exact; lane zero exercises input denormals.
  for (bool rdna3 : {false, true})
    for (unsigned lanes : {32u, 64u})
      for (unsigned mode = 0; mode < 4; ++mode)
        for (unsigned modifier = 0; modifier < 8; ++modifier)
          for (unsigned destination : {0u, 1u, 2u, 3u, 4u, 5u, 8u})
            for (unsigned policy_kind = 0; policy_kind < 4; ++policy_kind) {
              ArithmeticPolicy policy{};
              if (policy_kind == 1 || policy_kind == 2)
                policy.semantics = ArithmeticSemantics::BitExactEmpiricalAllowLoose;
              if (policy_kind == 2)
                policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
              if (policy_kind == 3)
                policy.semantics = ArithmeticSemantics::BitExactEmpirical;
              BridgeFixture fixture(rdna3, lanes, policy);
              const bool preserve_denorm = mode & 1;
              const bool ieee = mode & 2;
              fixture.wave->set_mode_raw((preserve_denorm ? 3u << 6 : 0) | (ieee ? 0x200 : 0));
              const uint64_t exec_mask = lanes == 32 ? 0x8000000fULL : 0x8000000f8000000fULL;
              fixture.wave->set_exec(exec_mask);
              uint32_t initial[10][64]{};
              double inputs[3][64];
              for (unsigned lane = 0; lane < lanes; ++lane) {
                inputs[0][lane] = lane ? -(double(lane) + 1) / 8 : 0x1p-1074;
                inputs[1][lane] = lane ? -0.5 : 2.0;
                inputs[2][lane] = lane ? -0.25 : 0.0;
                if (lane == 2)
                  inputs[2][lane] = std::bit_cast<double>(0x7ff8000000000000ULL);
                for (unsigned operand = 0; operand < 3; ++operand) {
                  const uint64_t bits = std::bit_cast<uint64_t>(inputs[operand][lane]);
                  initial[2 * operand][lane] = uint32_t(bits);
                  initial[2 * operand + 1][lane] = uint32_t(bits >> 32);
                }
                for (unsigned reg = 0; reg < 10; ++reg)
                  fixture.cu->write_vgpr(fixture.base() + reg, lane, initial[reg][lane]);
              }
              const unsigned abs = modifier >= 1 && modifier <= 3 ? 1u << (modifier - 1) : 0;
              const unsigned neg = modifier >= 4 && modifier <= 6 ? 1u << (modifier - 4) : 0;
              auto words =
                  rdna3 ? rdna3::build_vop3(rdna3::kVFmaF64Vop3, {.vdst = uint8_t(destination),
                                                                  .abs = uint8_t(abs),
                                                                  .clamp = modifier == 7,
                                                                  .src0 = 256,
                                                                  .src1 = 258,
                                                                  .src2 = 260,
                                                                  .omod = 1,
                                                                  .neg = uint8_t(neg)})
                        : rdna4::build_vop3(rdna4::kVFmaF64Vop3, {.vdst = uint8_t(destination),
                                                                  .abs = uint8_t(abs),
                                                                  .clamp = modifier == 7,
                                                                  .src0 = 256,
                                                                  .src1 = 258,
                                                                  .src2 = 260,
                                                                  .omod = 1,
                                                                  .neg = uint8_t(neg)});
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              fixture.recorder->reads.clear();
              fixture.recorder->writes.clear();
              const bool rejected = policy_kind >= 2;
              EXPECT_EQ(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .failed(),
                        rejected);
              uint64_t written[2]{};
              for (const auto &write : fixture.recorder->writes) {
                ASSERT_GE(write.reg, destination);
                ASSERT_LE(write.reg, destination + 1);
                EXPECT_EQ(write.bytes, 0xf);
                written[write.reg - destination] |= write.lanes;
              }
              EXPECT_EQ(written[0], rejected ? 0 : exec_mask);
              EXPECT_EQ(written[1], rejected ? 0 : exec_mask);
              EXPECT_EQ(fixture.wave->trapsts(), 0u);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                uint64_t expected =
                    initial[destination][lane] | (uint64_t(initial[destination + 1][lane]) << 32);
                if (!rejected && ((exec_mask >> lane) & 1)) {
                  double operands[3];
                  for (unsigned i = 0; i < 3; ++i) {
                    operands[i] = inputs[i][lane];
                    if (!preserve_denorm && std::fpclassify(operands[i]) == FP_SUBNORMAL)
                      operands[i] = std::copysign(0.0, operands[i]);
                    if (abs & (1u << i))
                      operands[i] = std::fabs(operands[i]);
                    if (neg & (1u << i))
                      operands[i] = -operands[i];
                  }
                  double result = std::fma(operands[0], operands[1], operands[2]);
                  if (!(rdna3 && (preserve_denorm || ieee))) {
                    // The existing rj-arith OMOD reference flushes zero/subnormal
                    // intermediates to +0 before scaling, independently of MODE.
                    if (std::fabs(result) < 0x1p-1022)
                      result = 0;
                    result *= 2;
                  }
                  if (!preserve_denorm && std::fpclassify(result) == FP_SUBNORMAL)
                    result = std::copysign(0.0, result);
                  if (modifier == 7)
                    result = !rdna3 && std::isnan(result) ? 0.0 : std::clamp(result, 0.0, 1.0);
                  expected = std::bit_cast<uint64_t>(result);
                }
                const uint64_t actual =
                    fixture.cu->read_vgpr(fixture.base() + destination, lane) |
                    (uint64_t(fixture.cu->read_vgpr(fixture.base() + destination + 1, lane)) << 32);
                if (std::isnan(std::bit_cast<double>(expected))) {
                  EXPECT_TRUE(std::isnan(std::bit_cast<double>(actual)));
                  continue;
                }
                EXPECT_EQ(actual, expected)
                    << rdna3 << '/' << mode << '/' << modifier << '/' << destination << '/' << lane;
              }
            }
}

TEST(ArithBridgeTest, TrigPreopArchitectureLiteralsPairsAndPolicies) {
  // Inherited from rj-arith TrigPreopArchitecture.LiteralArchitectureDifferences:
  // borrowed RDNA3 table versus the existing RDNA4 hardware-captured table.
  const unsigned exponent[] = {1968, 1091, 1091, 1997};
  const unsigned selector[] = {5, 19, 19, 4};
  const uint64_t golden[2][4] = {
      {0x180cc11bf1ecULL, 0x19e839cfbc5294ULL, 0x29e839cfbc5294ULL, 0x11aba10ac06608dULL},
      {0x180cc1180000ULL, 0, 0, 0x11aba10ac06608cULL}};
  for (bool rdna3 : {true, false})
    for (unsigned lanes : {32u, 64u})
      for (unsigned omod = 0; omod < 4; ++omod)
        for (unsigned modifiers = 0; modifiers < 4; ++modifiers)
          for (unsigned destination : {0u, 1u, 2u, 3u, 6u})
            for (unsigned policy_kind = 0; policy_kind < 4; ++policy_kind) {
              ArithmeticPolicy policy{};
              if (policy_kind == 1 || policy_kind == 2)
                policy.semantics = ArithmeticSemantics::BitExactEmpiricalAllowLoose;
              if (policy_kind == 2)
                policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
              if (policy_kind == 3)
                policy.semantics = ArithmeticSemantics::BitExactEmpirical;
              BridgeFixture fixture(rdna3, lanes, policy);
              // Preserve inputs, flush outputs, enable OMOD. Directed guest
              // rounding and DX10_CLAMP-off must not reject this fixed lookup.
              fixture.wave->set_mode_raw((1u << 6) | (3u << 2));
              const uint64_t exec_mask = lanes == 32 ? 0x8000000fULL : 0x8000000f8000000fULL;
              fixture.wave->set_exec(exec_mask);
              uint32_t initial[8][64];
              for (unsigned lane = 0; lane < lanes; ++lane) {
                for (unsigned reg = 0; reg < 8; ++reg)
                  initial[reg][lane] = 0xdeadbeef;
                initial[0][lane] = lane * 0x9e3779b9u;
                initial[1][lane] = (exponent[omod] << 20) | ((lane & 1) << 31) | lane;
                initial[2][lane] = selector[omod] | 0xffffffe0u;
                for (unsigned reg = 0; reg < 8; ++reg)
                  fixture.cu->write_vgpr(fixture.base() + reg, lane, initial[reg][lane]);
              }
              auto words = rdna3 ? rdna3::build_vop3(rdna3::kVTrigPreopF64Vop3,
                                                     {.vdst = uint8_t(destination),
                                                      .abs = uint8_t(modifiers & 1),
                                                      .clamp = uint8_t(modifiers >> 1),
                                                      .src0 = 256,
                                                      .src1 = 258,
                                                      .omod = uint8_t(omod),
                                                      .neg = uint8_t(modifiers & 1)})
                                 : rdna4::build_vop3(rdna4::kVTrigPreopF64Vop3,
                                                     {.vdst = uint8_t(destination),
                                                      .abs = uint8_t(modifiers & 1),
                                                      .clamp = uint8_t(modifiers >> 1),
                                                      .src0 = 256,
                                                      .src1 = 258,
                                                      .omod = uint8_t(omod),
                                                      .neg = uint8_t(modifiers & 1)});
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              fixture.recorder->reads.clear();
              fixture.recorder->writes.clear();
              const bool rejected = policy_kind == 3;
              EXPECT_EQ(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .failed(),
                        rejected);
              // A is a pair; B is one full word. No adjacent selector word or
              // old destination storage is an architectural source.
              uint64_t read_lanes[3]{};
              for (const auto &read : fixture.recorder->reads) {
                ASSERT_LT(read.reg, 3u);
                EXPECT_EQ(read.bytes, 0xf);
                read_lanes[read.reg] |= read.lanes;
              }
              for (uint64_t observed : read_lanes)
                EXPECT_EQ(observed, exec_mask);
              uint64_t written[2]{};
              for (const auto &write : fixture.recorder->writes) {
                ASSERT_GE(write.reg, destination);
                ASSERT_LE(write.reg, destination + 1);
                EXPECT_EQ(write.bytes, 0xf);
                written[write.reg - destination] |= write.lanes;
              }
              EXPECT_EQ(written[0], rejected ? 0 : exec_mask);
              EXPECT_EQ(written[1], rejected ? 0 : exec_mask);
              EXPECT_EQ(fixture.wave->trapsts(), 0u);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                const uint64_t before =
                    initial[destination][lane] | (uint64_t(initial[destination + 1][lane]) << 32);
                const uint64_t expected =
                    !rejected && ((exec_mask >> lane) & 1) ? golden[rdna3 ? 0 : 1][omod] : before;
                const uint64_t actual =
                    fixture.cu->read_vgpr(fixture.base() + destination, lane) |
                    (uint64_t(fixture.cu->read_vgpr(fixture.base() + destination + 1, lane)) << 32);
                EXPECT_EQ(actual, expected) << rdna3 << '/' << omod << '/' << modifiers << '/'
                                            << destination << '/' << lane;
              }
            }
}

TEST(ArithBridgeTest, QuadSadWindowsAccumulatorsAliasesAndMasks) {
  // Integer references follow the legacy four-window QSAD/MQSAD definition:
  // independently accumulate byte differences, skipping zero B bytes for MQSAD.
  const uint16_t ops3[] = {rdna3::kVQsadPkU16U8Vop3, rdna3::kVMqsadPkU16U8Vop3,
                           rdna3::kVMqsadU32U8Vop3};
  const uint16_t ops4[] = {rdna4::kVQsadPkU16U8Vop3, rdna4::kVMqsadPkU16U8Vop3,
                           rdna4::kVMqsadU32U8Vop3};
  for (bool rdna3 : {true, false})
    for (unsigned lanes : {32u, 64u})
      for (unsigned operation = 0; operation < 3; ++operation)
        for (bool clamp : {false, true})
          for (unsigned constant = 0; constant < 3; ++constant)
            for (unsigned destination : {0u, 1u, 2u, 5u, 12u})
              for (bool observed : {false, true})
                for (unsigned mask_kind = 0; mask_kind < 3; ++mask_kind)
                  for (bool required : {false, true}) {
                    ArithmeticPolicy policy{};
                    policy.semantics = ArithmeticSemantics::Loose;
                    if (required) {
                      policy.semantics = ArithmeticSemantics::BitExactEmpirical;
                    }
                    BridgeFixture fixture(rdna3, lanes, policy);
                    if (!observed)
                      fixture.cu->set_plugin_group(nullptr);
                    const uint64_t full = lanes == 32 ? 0xffffffffULL : ~0ULL;
                    const uint64_t exec_mask = mask_kind == 0   ? 0
                                               : mask_kind == 1 ? (full & 0x8000000f8000000fULL)
                                                                : full;
                    fixture.wave->set_exec(exec_mask);
                    uint32_t initial[16][64];
                    for (unsigned lane = 0; lane < lanes; ++lane) {
                      for (unsigned reg = 0; reg < 16; ++reg)
                        initial[reg][lane] = (lane + 7 * reg + 1) * 0x9e3779b9u;
                      initial[2][lane] &= 0xff00ff00u;
                      for (unsigned reg = 4; reg < 8; ++reg)
                        initial[reg][lane] = 0xffffff00u + lane + reg;
                      for (unsigned reg = 0; reg < 16; ++reg)
                        fixture.cu->write_vgpr(fixture.base() + reg, lane, initial[reg][lane]);
                    }
                    const uint16_t src_c = constant == 0 ? 260 : constant == 1 ? 129 : 193;
                    auto words =
                        rdna3 ? rdna3::build_vop3(ops3[operation], {.vdst = uint8_t(destination),
                                                                    .clamp = clamp,
                                                                    .src0 = 256,
                                                                    .src1 = 258,
                                                                    .src2 = src_c})
                              : rdna4::build_vop3(ops4[operation], {.vdst = uint8_t(destination),
                                                                    .clamp = clamp,
                                                                    .src0 = 256,
                                                                    .src1 = 258,
                                                                    .src2 = src_c});
                    auto instruction = fixture.decoder->decode_window(words);
                    if (operation == 2 && constant) {
                      EXPECT_TRUE(instruction.failed());
                      continue;
                    }
                    ASSERT_TRUE(instruction.succeeded());
                    if (observed) {
                      fixture.recorder->reads.clear();
                      fixture.recorder->writes.clear();
                    }
                    ASSERT_FALSE(
                        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .failed());
                    const unsigned output_words = operation == 2 ? 4 : 2;
                    if (observed) {
                      uint64_t read_lanes[8]{};
                      for (const auto &read : fixture.recorder->reads) {
                        ASSERT_LT(read.reg, 8u);
                        EXPECT_EQ(read.bytes, 0xf);
                        read_lanes[read.reg] |= read.lanes;
                      }
                      for (unsigned reg = 0; reg < 8; ++reg)
                        EXPECT_EQ(read_lanes[reg],
                                  (reg < 3 || (!constant && reg >= 4 && reg < 4 + output_words))
                                      ? exec_mask
                                      : 0);
                      uint64_t written[4]{};
                      for (const auto &write : fixture.recorder->writes) {
                        ASSERT_GE(write.reg, destination);
                        ASSERT_LT(write.reg, destination + output_words);
                        EXPECT_EQ(write.bytes, 0xf);
                        written[write.reg - destination] |= write.lanes;
                      }
                      for (unsigned reg = 0; reg < output_words; ++reg)
                        EXPECT_EQ(written[reg], exec_mask);
                    }
                    for (unsigned lane = 0; lane < lanes; ++lane) {
                      uint32_t expected[4]{};
                      const uint64_t source = initial[0][lane] | (uint64_t(initial[1][lane]) << 32);
                      const uint32_t reference = initial[2][lane];
                      for (unsigned window = 0; window < 4; ++window) {
                        uint32_t sum = 0;
                        for (unsigned byte = 0; byte < 4; ++byte) {
                          const unsigned a = (source >> (8 * (window + byte))) & 255;
                          const unsigned b = (reference >> (8 * byte)) & 255;
                          if (operation == 0 || b)
                            sum += a > b ? a - b : b - a;
                        }
                        const unsigned reg = operation == 2 ? window : window / 2;
                        const uint32_t accum_word = !constant       ? initial[4 + reg][lane]
                                                    : reg >= 2      ? 0
                                                    : constant == 2 ? 0xffffffffu
                                                    : reg == 0      ? 1
                                                                    : 0;
                        const unsigned shift = operation == 2 ? 0 : 16 * (window % 2);
                        const uint32_t limit = operation == 2 ? 0xffffffffu : 0xffff;
                        uint64_t value = uint64_t((accum_word >> shift) & limit) + sum;
                        value = clamp ? std::min(value, uint64_t(limit)) : value & limit;
                        expected[reg] |= uint32_t(value) << shift;
                      }
                      for (unsigned reg = 0; reg < output_words; ++reg)
                        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination + reg, lane),
                                  ((exec_mask >> lane) & 1) ? expected[reg]
                                                            : initial[destination + reg][lane])
                            << rdna3 << '/' << operation << '/' << clamp << '/' << constant << '/'
                            << destination << '/' << lane;
                    }
                  }
}

TEST(ArithBridgeTest, MixedFmaFormatsInlineConstantsAndPartialAliases) {
  const uint16_t ops3[] = {rdna3::kVFmaMixF32Vop3p, rdna3::kVFmaMixloF16Vop3p,
                           rdna3::kVFmaMixhiF16Vop3p};
  const uint16_t ops4[] = {rdna4::kVFmaMixF32Vop3p, rdna4::kVFmaMixloF16Vop3p,
                           rdna4::kVFmaMixhiF16Vop3p};
  // Dyadic inputs/results are exactly representable in FP16 and FP32. Thus
  // the computed reference does not assume arbitrary FMA hardware equivalence.
  const uint32_t halves[] = {0xc000bc00, 0x44004000, 0x38003400};
  const float floats[] = {-1.5f, 2.0f, 0.25f};
  for (bool rdna3 : {true, false})
    for (unsigned lanes : {32u, 64u})
      for (unsigned op = 0; op < 3; ++op)
        for (unsigned formats = 0; formats < 8; ++formats)
          for (bool high : {false, true})
            for (bool inline_a : {false, true})
              for (unsigned destination = 0; destination < 4; ++destination)
                for (unsigned policy_kind = 0; policy_kind < 4; ++policy_kind) {
                  ArithmeticPolicy policy{};
                  if (policy_kind == 1)
                    policy.semantics = ArithmeticSemantics::Loose;
                  if (policy_kind == 2) {
                    policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
                  }
                  if (policy_kind == 3)
                    policy.semantics = ArithmeticSemantics::BitExactEmpirical;
                  BridgeFixture fixture(rdna3, lanes, policy);
                  fixture.wave->set_mode_raw(0x1f0);
                  const uint64_t exec_mask = lanes == 32 ? 0x8000000fULL : 0x8000000f8000000fULL;
                  fixture.wave->set_exec(exec_mask);
                  uint32_t initial[4] = {0, 0, 0, 0xa55aa55a};
                  float inputs[3];
                  for (unsigned i = 0; i < 3; ++i) {
                    initial[i] =
                        formats & (1u << i) ? halves[i] : std::bit_cast<uint32_t>(floats[i]);
                    inputs[i] = formats & (1u << i)
                                    ? util::f16_to_f32(uint16_t(halves[i] >> (high ? 16 : 0)))
                                    : floats[i];
                  }
                  if (inline_a)
                    inputs[0] = 0.5f;
                  for (unsigned reg = 0; reg < 4; ++reg)
                    for (unsigned lane = 0; lane < lanes; ++lane)
                      fixture.cu->write_vgpr(fixture.base() + reg, lane, initial[reg]);
                  const unsigned abs = formats;
                  const unsigned neg = high ? 5 : 2;
                  const bool clamp = destination == 3;
                  auto words =
                      rdna3 ? rdna3::build_vop3p(ops3[op], {.vdst = uint8_t(destination),
                                                            .neg_hi = uint8_t(abs),
                                                            .op_sel = uint8_t(high ? 7 : 0),
                                                            .op_sel_hi_2 = uint8_t(formats >> 2),
                                                            .clamp = clamp,
                                                            .src0 = uint16_t(inline_a ? 240 : 256),
                                                            .src1 = 257,
                                                            .src2 = 258,
                                                            .op_sel_hi = uint8_t(formats & 3),
                                                            .neg = uint8_t(neg)})
                            : rdna4::build_vop3p(ops4[op], {.vdst = uint8_t(destination),
                                                            .neg_hi = uint8_t(abs),
                                                            .opsel = uint8_t(high ? 7 : 0),
                                                            .opsel_hi_2 = uint8_t(formats >> 2),
                                                            .clamp = clamp,
                                                            .src0 = uint16_t(inline_a ? 240 : 256),
                                                            .src1 = 257,
                                                            .src2 = 258,
                                                            .opsel_hi = uint8_t(formats & 3),
                                                            .neg = uint8_t(neg)});
                  auto instruction = fixture.decoder->decode_window(words);
                  ASSERT_TRUE(instruction.succeeded());
                  fixture.recorder->reads.clear();
                  fixture.recorder->writes.clear();
                  const bool rejected = policy_kind == 3 && (rdna3 || op == 0);
                  EXPECT_EQ(
                      fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .failed(),
                      rejected);
                  uint64_t written = 0;
                  const uint8_t bytes = op == 0 ? 0xf : op == 1 ? 3 : 0xc;
                  for (const auto &write : fixture.recorder->writes) {
                    EXPECT_EQ(write.reg, destination);
                    EXPECT_EQ(write.bytes, bytes);
                    written |= write.lanes;
                  }
                  EXPECT_EQ(written, rejected ? 0 : exec_mask);
                  for (const auto &read : fixture.recorder->reads) {
                    EXPECT_LT(read.reg, 3u);
                    EXPECT_EQ(read.bytes, (formats & (1u << read.reg)) ? (high ? 0xc : 3) : 0xf);
                    EXPECT_EQ(read.lanes, exec_mask);
                    if (inline_a) {
                      EXPECT_NE(read.reg, 0u);
                    }
                  }
                  for (unsigned i = 0; i < 3; ++i) {
                    if (abs & (1u << i))
                      inputs[i] = std::fabs(inputs[i]);
                    if (neg & (1u << i))
                      inputs[i] = -inputs[i];
                  }
                  float result = std::fma(inputs[0], inputs[1], inputs[2]);
                  if (clamp)
                    result = std::clamp(result, 0.0f, 1.0f);
                  const unsigned shift = op == 2 ? 16 : 0;
                  const uint32_t expected = op == 0
                                                ? std::bit_cast<uint32_t>(result)
                                                : (initial[destination] & ~(0xffffu << shift)) |
                                                      (uint32_t(util::f32_to_f16(result)) << shift);
                  for (unsigned lane = 0; lane < lanes; ++lane)
                    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, lane),
                              !rejected && ((exec_mask >> lane) & 1) ? expected
                                                                     : initial[destination])
                        << rdna3 << '/' << op << '/' << formats << '/' << high << '/' << inline_a
                        << '/' << destination;
                }
}

TEST(ArithBridgeTest, MixedFmaGuestControlsAndReportingPolicy) {
  for (bool rdna3 : {true, false})
    for (unsigned control = 0; control < 3; ++control)
      for (unsigned policy_kind = 0; policy_kind < 5; ++policy_kind) {
        ArithmeticPolicy policy{};
        if (policy_kind == 1)
          policy.semantics = ArithmeticSemantics::Loose;
        if (policy_kind == 2 || policy_kind == 3)
          policy.semantics = ArithmeticSemantics::BitExactEmpiricalAllowLoose;
        if (policy_kind == 2)
          policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
        if (policy_kind == 4)
          policy.semantics = ArithmeticSemantics::BitExactEmpirical;
        BridgeFixture fixture(rdna3, 32, policy);
        // Directed rounding; mixed-format input denorm settings; DX10_CLAMP-off.
        const uint32_t modes[] = {0x1f1, 0x1b0, 0xf0};
        fixture.wave->set_mode_raw(modes[control]);
        fixture.wave->set_exec(1);
        fixture.cu->write_vgpr(fixture.base(), 0, 0x3c00);
        fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x3f800000);
        fixture.cu->write_vgpr(fixture.base() + 2, 0, 0);
        auto words = rdna3 ? rdna3::build_vop3p(rdna3::kVFmaMixF32Vop3p, {.vdst = 3,
                                                                          .clamp = 1,
                                                                          .src0 = 256,
                                                                          .src1 = 257,
                                                                          .src2 = 258,
                                                                          .op_sel_hi = 1})
                           : rdna4::build_vop3p(rdna4::kVFmaMixF32Vop3p, {.vdst = 3,
                                                                          .clamp = 1,
                                                                          .src0 = 256,
                                                                          .src1 = 257,
                                                                          .src2 = 258,
                                                                          .opsel_hi = 1});
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        fixture.recorder->writes.clear();
        const bool rejected = policy_kind == 4;
        EXPECT_EQ(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).failed(),
            rejected);
        EXPECT_EQ(fixture.recorder->writes.empty(), rejected);
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 3, 0),
                  rejected ? 0x3c003c00u : 0x3f800000u);
      }
}

TEST(ArithBridgeTest, RcpIflagAccumulatesArchitectureStatusAfterAliasedInputs) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32u, 64u})
      for (unsigned policy_kind = 0; policy_kind < 4; ++policy_kind)
        for (bool clamp : {false, true})
          for (uint32_t raw : {0u, 1u, 0x3f800000u, 0x7fc00000u}) {
            ArithmeticPolicy policy{};
            if (policy_kind == 1)
              policy.semantics = ArithmeticSemantics::Loose;
            if (policy_kind == 2) {
              policy.semantics = ArithmeticSemantics::BitExactEmpirical;
            }
            if (policy_kind == 3)
              policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
            BridgeFixture fixture(rdna3, lanes, policy);
            const unsigned last = lanes - 1;
            const uint64_t exec_mask = 1ULL << last;
            fixture.wave->set_exec(exec_mask);
            // DX10_CLAMP off and guest denormals preserved; RCP still flushes.
            fixture.wave->set_mode_raw(0xf0);
            fixture.wave->set_trapsts(0x80000001);
            fixture.cu->write_vgpr(fixture.base(), 0, 0);
            fixture.cu->write_vgpr(fixture.base(), last, raw);
            const auto words =
                rdna3 ? rdna3::build_vop3(rdna3::kVRcpIflagF32Vop3,
                                          {.vdst = 0, .clamp = uint8_t(clamp), .src0 = 256})
                      : rdna4::build_vop3(rdna4::kVRcpIflagF32Vop3,
                                          {.vdst = 0, .clamp = uint8_t(clamp), .src0 = 256});
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            fixture.recorder->writes.clear();
            ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .succeeded());
            const bool report = policy_kind == 0 || policy_kind == 2;
            const bool raises = rdna3 ? raw == 0 : raw <= 1 && !clamp;
            EXPECT_EQ(fixture.wave->trapsts(), 0x80000001u | (report && raises ? 64u : 0u));
            uint32_t expected = raw <= 1 ? 0x7f800000u : raw;
            if (clamp && !(rdna3 && raw == 0x7fc00000u))
              expected = raw == 0x7fc00000u ? 0 : 0x3f800000u;
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), last), expected);
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), 0u);
            uint64_t written = 0;
            for (const auto &write : fixture.recorder->writes) {
              EXPECT_EQ(write.reg, 0u);
              EXPECT_EQ(write.bytes, 0xf);
              written |= write.lanes;
            }
            EXPECT_EQ(written, exec_mask);
          }
}

TEST(ArithBridgeTest, SqrtClassifiesModifiedInputsBeforeAliasedWrites) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32u, 64u})
      for (unsigned policy_kind = 0; policy_kind < 4; ++policy_kind)
        for (bool negate : {false, true})
          for (bool clamp : {false, true})
            for (uint32_t raw : {0u, 0x80000000u, 1u, 0x80000001u, 0x40800000u, 0xc0800000u,
                                 0x7f800000u, 0xff800000u, 0x7fc00000u}) {
              ArithmeticPolicy policy{};
              if (policy_kind == 1)
                policy.semantics = ArithmeticSemantics::Loose;
              if (policy_kind == 2) {
                policy.semantics = ArithmeticSemantics::BitExactEmpirical;
              }
              if (policy_kind == 3)
                policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
              BridgeFixture fixture(rdna3, lanes, policy);
              const unsigned last = lanes - 1;
              const uint64_t exec_mask = 1ULL << last;
              fixture.wave->set_exec(exec_mask);
              // Preserve guest denormals, with DX10_CLAMP enabled. SQRT still flushes.
              fixture.wave->set_mode_raw(0x1f0);
              fixture.wave->set_trapsts(0x80000000);
              fixture.cu->write_vgpr(fixture.base(), 0, 0);
              fixture.cu->write_vgpr(fixture.base(), last, raw);
              const auto words =
                  rdna3 ? rdna3::build_vop3(rdna3::kVSqrtF32Vop3, {.vdst = 0,
                                                                   .clamp = uint8_t(clamp),
                                                                   .src0 = 256,
                                                                   .neg = uint8_t(negate)})
                        : rdna4::build_vop3(rdna4::kVSqrtF32Vop3, {.vdst = 0,
                                                                   .clamp = uint8_t(clamp),
                                                                   .src0 = 256,
                                                                   .neg = uint8_t(negate)});
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              fixture.recorder->writes.clear();
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              const bool report = policy_kind == 0 || policy_kind == 2;
              const uint32_t modified = raw ^ (negate ? 0x80000000u : 0u);
              const uint32_t magnitude = modified & 0x7fffffffu;
              const bool raises = (modified >> 31) && magnitude && magnitude <= 0x7f800000u;
              EXPECT_EQ(fixture.wave->trapsts(), 0x80000000u | (report && raises ? 1u : 0u));
              uint32_t expected;
              if (magnitude < 0x00800000u)
                expected = modified & 0x80000000u;
              else if (raises || magnitude > 0x7f800000u)
                expected = 0xffc00000u;
              else
                expected = magnitude == 0x40800000u ? 0x40000000u : magnitude;
              if (clamp)
                expected =
                    raises || magnitude > 0x7f800000u || magnitude < 0x00800000u ? 0u : 0x3f800000u;
              const uint32_t actual = fixture.cu->read_vgpr(fixture.base(), last);
              if ((expected & 0x7fffffffu) > 0x7f800000u) {
                EXPECT_GT(actual & 0x7fffffffu, 0x7f800000u);
              } else {
                EXPECT_EQ(actual, expected);
              }
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), 0u);
              uint64_t written = 0;
              for (const auto &write : fixture.recorder->writes) {
                EXPECT_EQ(write.reg, 0u);
                EXPECT_EQ(write.bytes, 0xf);
                written |= write.lanes;
              }
              EXPECT_EQ(written, exec_mask);
            }
}

TEST(ArithBridgeTest, MulReportingPolicyPreservesDirectedNumericalExecution) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32u, 64u})
      for (unsigned policy_kind = 0; policy_kind < 4; ++policy_kind)
        for (unsigned rounding = 0; rounding < 4; ++rounding)
          for (bool clamp : {false, true}) {
            ArithmeticPolicy policy{};
            if (policy_kind == 1)
              policy.semantics = ArithmeticSemantics::Loose;
            if (policy_kind == 2) {
              policy.semantics = ArithmeticSemantics::BitExactEmpirical;
            }
            if (policy_kind == 3)
              policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
            BridgeFixture fixture(rdna3, lanes, policy);
            const unsigned last = lanes - 1;
            const uint64_t exec_mask = 1ULL << last;
            fixture.wave->set_exec(exec_mask);
            fixture.wave->set_mode_raw(0x1f0 | rounding);
            fixture.wave->set_trapsts(0x80000000);
            fixture.cu->write_vgpr(fixture.base(), last, 0x7f7fffff);
            fixture.cu->write_vgpr(fixture.base() + 1, last, 0x40000000);
            const auto words =
                rdna3 ? rdna3::build_vop3(
                            rdna3::kVMulF32Vop3,
                            {.vdst = 0, .clamp = uint8_t(clamp), .src0 = 256, .src1 = 257})
                      : rdna4::build_vop3(
                            rdna4::kVMulF32Vop3,
                            {.vdst = 0, .clamp = uint8_t(clamp), .src0 = 256, .src1 = 257});
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            fixture.recorder->writes.clear();
            // Only gfx12 nearest-even without CLAMP has qualified reporting.
            // Compatibility retains the legacy classifier in every configuration.
            const bool reporting_supported = !rdna3 && !rounding && !clamp;
            const bool rejected = policy_kind == 2 && !reporting_supported;
            EXPECT_EQ(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded(),
                      !rejected);
            const bool report = policy_kind == 0 || (reporting_supported && policy_kind == 2);
            EXPECT_EQ(fixture.wave->trapsts(), 0x80000000u | (report ? 40u : 0u));
            const uint32_t expected = rejected                           ? 0x7f7fffffu
                                      : clamp                            ? 0x3f800000u
                                      : policy_kind == 1 || rounding < 2 ? 0x7f800000u
                                                                         : 0x7f7fffffu;
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), last), expected);
            uint64_t written = 0;
            for (const auto &write : fixture.recorder->writes) {
              EXPECT_EQ(write.reg, 0u);
              EXPECT_EQ(write.bytes, 0xf);
              written |= write.lanes;
            }
            EXPECT_EQ(written, rejected ? 0 : exec_mask);
          }
}

TEST(ArithBridgeTest, DivisionMasksConditionAndDirectedRounding) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32u, 64u})
      for (bool wide : {false, true})
        for (bool scale : {false, true})
          for (unsigned rounding = 0; rounding < 4; ++rounding)
            for (bool empty : {false, true}) {
              BridgeFixture fixture(rdna3, lanes);
              const unsigned last = lanes - 1;
              const uint64_t exec_mask = empty ? 0 : 1ULL << last;
              const uint64_t old_condition = lanes == 32 ? UINT32_MAX : UINT64_MAX;
              fixture.wave->set_exec(exec_mask);
              fixture.wave->set_vcc_mask(scale ? old_condition : exec_mask);
              fixture.wave->set_mode_raw(0x1f0 | (rounding << (wide ? 2 : 0)));
              const uint64_t one = wide ? 0x3ff0000000000000ULL : 0x3f800000ULL;
              const uint64_t tiny = wide ? 0x04c0000000000000ULL : 0x14800000ULL;
              const uint64_t values[] = {scale ? one : 0, scale ? one : 0, scale ? one : tiny};
              for (unsigned input = 0; input < 3; ++input) {
                fixture.cu->write_vgpr(fixture.base() + 2 * input, last, uint32_t(values[input]));
                if (wide)
                  fixture.cu->write_vgpr(fixture.base() + 2 * input + 1, last,
                                         uint32_t(values[input] >> 32));
              }
              const auto sb = fixture.wave->sgpr_alloc().base;
              fixture.cu->write_sgpr(sb + 4, 0x12345678);
              fixture.cu->write_sgpr(sb + 5, 0xabcdef01);
              std::array<uint32_t, 2> words;
              if (scale) {
                words = rdna3 ? rdna3::build_vop3_sdst_enc(
                                    wide ? rdna3::kVDivScaleF64Vop3SdstEnc
                                         : rdna3::kVDivScaleF32Vop3SdstEnc,
                                    {.vdst = 4, .sdst = 4, .src0 = 256, .src1 = 258, .src2 = 260})
                              : rdna4::build_vop3_sdst_enc(
                                    wide ? rdna4::kVDivScaleF64Vop3SdstEnc
                                         : rdna4::kVDivScaleF32Vop3SdstEnc,
                                    {.vdst = 4, .sdst = 4, .src0 = 256, .src1 = 258, .src2 = 260});
              } else {
                words = rdna3
                            ? rdna3::build_vop3(wide ? rdna3::kVDivFmasF64Vop3
                                                     : rdna3::kVDivFmasF32Vop3,
                                                {.vdst = 4, .src0 = 256, .src1 = 258, .src2 = 260})
                            : rdna4::build_vop3(wide ? rdna4::kVDivFmasF64Vop3
                                                     : rdna4::kVDivFmasF32Vop3,
                                                {.vdst = 4, .src0 = 256, .src1 = 258, .src2 = 260});
              }
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              const uint64_t expected = scale ? one : empty ? tiny : rounding == 1 ? 1 : 0;
              const uint64_t actual =
                  fixture.cu->read_vgpr(fixture.base() + 4, last) |
                  (wide ? uint64_t(fixture.cu->read_vgpr(fixture.base() + 5, last)) << 32 : 0);
              EXPECT_EQ(actual, expected);
              if (scale) {
                // gfx1201 DIV_SCALE writes zero outside EXEC, including EXEC=0.
                const uint64_t condition = 0;
                EXPECT_EQ(fixture.cu->read_sgpr(sb + 4), uint32_t(condition));
                EXPECT_EQ(fixture.cu->read_sgpr(sb + 5),
                          lanes == 64 ? uint32_t(condition >> 32) : 0xabcdef01U);
                EXPECT_EQ(fixture.wave->vcc(), old_condition);
              }
            }
}

// All three division operations forward RDNA3 DX10_CLAMP in both FP widths.
TEST(ArithBridgeTest, DivisionDx10ClampAndFixupStatus) {
  for (bool wide : {false, true}) {
    for (bool dx10 : {false, true}) {
      for (int operation = 0; operation < 3; ++operation) {
        for (bool divide_by_zero : {false, true}) {
          if (divide_by_zero && operation != 2)
            continue;
          BridgeFixture fixture(true);
          fixture.wave->set_mode_raw(0xf0u | (dx10 ? Wavefront::DX10_CLAMP_BIT : 0));
          fixture.wave->set_exec(1);
          fixture.wave->set_trapsts(0x20);
          const uint64_t nan = wide ? 0x7ff8123456789abcULL : 0x7fc12345ULL;
          const uint64_t inf = wide ? 0x7ff0000000000000ULL : 0x7f800000ULL;
          const uint64_t one = wide ? 0x3ff0000000000000ULL : 0x3f800000ULL;
          const uint64_t values[] = {divide_by_zero ? inf : nan, divide_by_zero ? 0 : nan,
                                     divide_by_zero ? one : nan};
          for (unsigned input = 0; input < 3; ++input) {
            fixture.cu->write_vgpr(fixture.base() + input * 2, 0, uint32_t(values[input]));
            if (wide)
              fixture.cu->write_vgpr(fixture.base() + input * 2 + 1, 0,
                                     uint32_t(values[input] >> 32));
          }
          std::array<uint32_t, 2> words;
          if (operation == 0) {
            words = rdna3::build_vop3_sdst_enc(
                wide ? rdna3::kVDivScaleF64Vop3SdstEnc : rdna3::kVDivScaleF32Vop3SdstEnc,
                {.vdst = 6, .sdst = 4, .clamp = 1, .src0 = 256, .src1 = 258, .src2 = 260});
          } else {
            uint16_t op = operation == 1
                              ? (wide ? rdna3::kVDivFmasF64Vop3 : rdna3::kVDivFmasF32Vop3)
                              : (wide ? rdna3::kVDivFixupF64Vop3 : rdna3::kVDivFixupF32Vop3);
            words = rdna3::build_vop3(op, {.vdst = 6,
                                           .clamp = uint8_t(!divide_by_zero),
                                           .src0 = 256,
                                           .src1 = 258,
                                           .src2 = 260});
          }
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          const uint64_t result =
              fixture.cu->read_vgpr(fixture.base() + 6, 0) |
              (wide ? uint64_t(fixture.cu->read_vgpr(fixture.base() + 7, 0)) << 32 : 0);
          EXPECT_EQ(result, divide_by_zero ? inf : dx10 ? 0 : nan);
          // Preserve the inherited classifier; gfx1100 review captures also show
          // DIV0 with DX10_CLAMP clear and suppression with it set.
          EXPECT_EQ(fixture.wave->trapsts(), 0x20u | (divide_by_zero && !dx10 ? 4u : 0u));
        }
      }
    }
  }
}

TEST(ArithBridgeTest, HalfFixupGuestRoundingNanClampAndPartialWrites) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32u, 64u})
      for (unsigned rounding = 0; rounding < 4; ++rounding)
        for (bool dx10 : {false, true})
          for (bool clamp : {false, true})
            for (bool nan : {false, true}) {
              BridgeFixture fixture(rdna3, lanes);
              const unsigned last = lanes - 1;
              fixture.wave->set_exec(1ULL << last);
              fixture.wave->set_mode_raw(0xf0 | (rounding << 2) | (dx10 ? 0x100 : 0));
              fixture.wave->set_trapsts(0x20);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                fixture.cu->write_vgpr(fixture.base(), lane, 0x7c00a55a);
                fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x3c000000);
                fixture.cu->write_vgpr(fixture.base() + 2, lane, nan ? 0xfc150000 : 0x3c000000);
              }
              const auto words = rdna3
                                     ? rdna3::build_vop3(rdna3::kVDivFixupF16Vop3, {.vdst = 0,
                                                                                    .op_sel = 15,
                                                                                    .clamp = clamp,
                                                                                    .src0 = 256,
                                                                                    .src1 = 257,
                                                                                    .src2 = 258})
                                     : rdna4::build_vop3(rdna4::kVDivFixupF16Vop3, {.vdst = 0,
                                                                                    .opsel = 15,
                                                                                    .clamp = clamp,
                                                                                    .src0 = 256,
                                                                                    .src1 = 257,
                                                                                    .src2 = 258});
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              ASSERT_EQ(fixture.recorder->reads.size(), 3U);
              for (const auto &read : fixture.recorder->reads) {
                EXPECT_EQ(read.bytes, 0xc);
                EXPECT_EQ(read.lanes, 1ULL << last);
              }
              ASSERT_EQ(fixture.recorder->writes.size(), 1U);
              EXPECT_EQ(fixture.recorder->writes[0].bytes, 0xc);
              EXPECT_EQ(fixture.recorder->writes[0].lanes, 1ULL << last);
              uint32_t expected = nan ? 0xfe15 : rounding < 2 ? 0x7c00 : 0x7bff;
              if (clamp)
                expected = nan ? (rdna3 && !dx10 ? 0xfe15 : 0) : 0x3c00;
              for (unsigned lane = 0; lane < lanes; ++lane)
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                          lane == last ? (expected << 16) | 0xa55a : 0x7c00a55a);
              EXPECT_EQ(fixture.wave->trapsts(), 0x20U);
            }
}
} // namespace
} // namespace rocjitsu::amdgpu
