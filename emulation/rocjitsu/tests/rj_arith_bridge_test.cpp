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
                rj_arith_init_cpu_flags() | semantics, modifiers, d, a, b, true,
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
    unsigned omod;
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
                if (policy.semantics != ArithmeticSemantics::Loose)
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
                    if (policy.semantics != ArithmeticSemantics::Loose)
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
          if (policy.semantics != ArithmeticSemantics::Loose)
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
                const uint64_t condition = old_condition & ~exec_mask;
                EXPECT_EQ(fixture.cu->read_sgpr(sb + 4), uint32_t(condition));
                EXPECT_EQ(fixture.cu->read_sgpr(sb + 5),
                          lanes == 64 ? uint32_t(condition >> 32) : 0xabcdef01U);
                EXPECT_EQ(fixture.wave->vcc(), old_condition);
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
