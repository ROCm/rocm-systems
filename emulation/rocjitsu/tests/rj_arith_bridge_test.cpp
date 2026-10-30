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

} // namespace
} // namespace rocjitsu::amdgpu
