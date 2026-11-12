// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "fixtures/float_dot/gfx1100_cases.h"
#include "fixtures/float_dot/gfx1201_cases.h"
#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/sop2.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/sopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/sopk.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vop2.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop2.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/conversion.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_compare_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_lane_exec.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/sop2.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/sopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/sopk.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop3p.h"

#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_division_exec.h"
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

#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <optional>

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
  std::vector<RegisterRef> scalar_writes;
  std::vector<uint32_t> scalar_reads;
  std::vector<Write> writes;
  std::vector<Write> reads;
};

// Use one architecture for decoding and execution, with observed register writes.
class BridgeFixture {
public:
  explicit BridgeFixture(bool rdna3, uint32_t lanes = 32, ArithmeticPolicy policy = {},
                         uint32_t vgprs = 256)
      : BridgeFixture(rdna3 ? ROCJITSU_CODE_ARCH_RDNA3 : ROCJITSU_CODE_ARCH_RDNA4, lanes, policy,
                      vgprs) {}

  explicit BridgeFixture(rj_code_arch_t arch, uint32_t lanes = 32, ArithmeticPolicy policy = {},
                         uint32_t vgprs = 256)
      : backend(arch == ROCJITSU_CODE_ARCH_RDNA3   ? &rocjitsu::rdna3::execution_backend()
                : arch == ROCJITSU_CODE_ARCH_CDNA3 ? &rocjitsu::cdna3::execution_backend()
                : arch == ROCJITSU_CODE_ARCH_CDNA4 ? &rocjitsu::cdna4::execution_backend()
                : arch == ROCJITSU_CODE_ARCH_CDNA5 ? &rocjitsu::cdna5::execution_backend()
                                                   : &rocjitsu::rdna4::execution_backend()) {
    ComputeUnitCore::Config config{};
    config.arch = arch;
    config.arithmetic = policy;
    decoder = Decoder::create(config.arch);
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    rocjitsu::test::initialize_compute_unit(cu, "rj_arith_bridge", config, &memory, &l2);
    if (!cu)
      return;
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto observer = std::make_unique<WriteRecorder>();
    recorder = observer.get();
    group->add(std::move(observer));
    cu->set_plugin_group(group);
    wave = cu->dispatch_wf(0, 0, 104, vgprs, lanes);
    if (!wave) {
      ADD_FAILURE() << "Could not dispatch fixture wave";
      return;
    }
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

// Compare migrated narrowing with the inherited format model, including partial
// destination preservation, source aliasing, overflow and seed-dependent rounding.

// gfx1250 half sources share FP8 rounding but ignore FP16_OVFL and preserve other D bytes.

// Widening retains inherited format/NaN rules, selectors, aliases and masked stores.

// FP8/BF8-to-half bindings retain selectors, NaN signs, finite saturation and inactive lanes.
// Mixed gfx1250 forms preserve narrow denormals, inline conversion, aliases and partial writes.
// PACK retains half bits/NaN policy; byte conversions distinguish MODE rounding from legacy
// truncation.

// Selected bytes and signed nibbles convert exactly; OMOD gating differs by generation.

// FP16/BF16 migration preserves inherited narrowing and special encodings,
// selected halves, SDWA placement, aliased operands and EXEC masking.

void check_cdna_float16_conversions(bool bf16) {
  const uint32_t inputs[] = {0x7f800001U, 0xff800001U, 0x3f801000U, 0x3f803000U,
                             0x00000001U, 0x80000001U, 0x7f7fffffU, 0xff7fffffU,
                             0x7f800000U, 0xff800000U, 0x387fffffU, 0xb87fffffU,
                             0x7fc01234U, 0x80000000U, 0,           0x3f800000U};
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      for (int op = bf16 ? 0 : 1; op < 3; ++op) {
        for (int variant = 0; variant < 4; ++variant) {
          SCOPED_TRACE(::testing::Message() << arch << ":" << op << ":" << variant);
          ArithmeticPolicy policy{};
          policy.semantics = semantics;
          int lanes = arch == ROCJITSU_CODE_ARCH_CDNA4 ? 64 : 32;
          BridgeFixture fixture(arch, lanes, policy);
          fixture.wave->set_exec((lanes == 32 ? 0xffffffffULL : ~0ULL) & ~2ULL);
          bool sat = variant & 1;
          fixture.wave->set_mode_raw(0xff | (sat ? Wavefront::FP16_OVFL_BIT : 0));
          bool sdwa = op == 0 && variant == 2 && arch == ROCJITSU_CODE_ARCH_CDNA4;
          for (int lane = 0; lane < lanes; ++lane) {
            fixture.cu->write_vgpr(fixture.base(), lane, inputs[lane % 16]);
            fixture.cu->write_vgpr(fixture.base() + 1, lane,
                                   op == 2 && arch == ROCJITSU_CODE_ARCH_CDNA4
                                       ? (lane & 1 ? ~0U : 0U)
                                       : inputs[(lane + 3) % 16]);
            fixture.cu->write_vgpr(fixture.base() + 2, lane, lane & 1 ? ~0U : 0U);
          }
          auto nearest = bf16 ? util::f32_to_bf16_rne_mode : util::f32_to_f16_mode;
          auto stochastic = bf16 ? util::f32_to_bf16_sr_mode : util::f32_to_f16_sr_mode;
          uint8_t selector = op == 0 ? variant & 1 : op == 2 ? (variant & 1) << 3 : 0;
          std::vector<uint32_t> words;
          if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
            if (sdwa) {
              auto encoded =
                  cdna4::build_vop1(cdna4::kVCvtF32Bf16Vop1, {.src0 = amdgpu::SRC_SDWA, .vdst = 0});
              words.assign(encoded.begin(), encoded.end());
              words.push_back((5U << 8) | (2U << 11) | (6U << 16));
            } else {
              const uint16_t ops[] = {
                  cdna4::kVCvtF32Bf16Vop3,
                  (bf16 ? cdna4::kVCvtPkBf16F32Vop3 : cdna4::kVCvtPkF16F32Vop3),
                  (bf16 ? cdna4::kVCvtSrBf16F32Vop3 : cdna4::kVCvtSrF16F32Vop3)};
              auto encoded = cdna4::build_vop3(ops[op], {.vdst = 0,
                                                         .abs = 7,
                                                         .op_sel = selector,
                                                         .clamp = 1,
                                                         .src0 = 256,
                                                         .src1 = 257,
                                                         .src2 = 258,
                                                         .omod = 3,
                                                         .neg = 7});
              words.assign(encoded.begin(), encoded.end());
            }
          } else {
            const uint16_t ops[] = {
                cdna5::kVCvtF32Bf16Vop3,
                (bf16 ? cdna5::kVCvtPkBf16F32Vop3 : cdna5::kVCvtPkF16F32Vop3),
                (bf16 ? cdna5::kVCvtSrPkBf16F32Vop3 : cdna5::kVCvtSrPkF16F32Vop3)};
            auto encoded = cdna5::build_vop3(ops[op], {.vdst = 0,
                                                       .abs = 7,
                                                       .opsel = selector,
                                                       .clamp = 1,
                                                       .src0 = 256,
                                                       .src1 = 257,
                                                       .src2 = 258,
                                                       .omod = 3,
                                                       .neg = 7});
            words.assign(encoded.begin(), encoded.end());
          }
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          for (int lane = 0; lane < lanes; ++lane) {
            uint32_t raw = inputs[lane % 16], other = inputs[(lane + 3) % 16],
                     seed = lane & 1 ? ~0U : 0U;
            uint32_t expected;
            if (op == 0) {
              expected = (raw >> (selector ? 16 : 0)) << 16;
              if (sdwa) {
                expected = (expected << 16) | (raw & 0xffff);
              }
            } else if (op == 1) {
              // These encodings request ABS followed by NEG on both float sources.
              expected = nearest(std::bit_cast<float>(raw | 0x80000000U), sat) |
                         (uint32_t(nearest(std::bit_cast<float>(other | 0x80000000U), sat)) << 16);
            } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
              uint32_t lo = stochastic(std::bit_cast<float>(raw | 0x80000000U), seed, sat);
              int shift = selector ? 16 : 0;
              expected = (raw & ~(0xffffU << shift)) | (lo << shift);
            } else {
              // Packed SR ignores sign modifiers and uses two independent seed halves.
              auto packed = bf16 ? fp_mode::cvt_sr_pk_bf16_f32 : fp_mode::cvt_sr_pk_f16_f32;
              expected = packed(raw, uint16_t(seed), sat) |
                         (uint32_t(packed(other, uint16_t(seed >> 16), sat)) << 16);
            }
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), lane == 1 ? raw : expected)
                << "lane=" << lane;
          }
        }
      }
    }
  }
}

// Check guest rounding, fixed rounding, denormals and SDWA destination placement.
// Preserve old CDNA zero-extension, newer partial writes, and SDWA prescaling.

// Check guest rounding, fixed rounding, denormals and SDWA destination placement.

// FP64 conversions preserve guest rounding, NaN quieting and split-VGPR aliases.

// CDNA packs saturate full words; gfx1250 extension honors true16 source halves.

// Normalized conversions retain nearest-even rounding, aliases and destination layout.

// Packed conversion ignores guest rounding and output modifiers, preserving masked aliases.

// Older CDNA zero-extends saturated bytes; gfx1250 preserves the other true16 half.

// Integer conversion encodings ignore numerical modifiers except widening OPSEL.
// Compare whole words so partial source reads, inactive lanes and aliases stay visible.

const uint64_t kExactEmpiricalNoFpStatus = RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;

// Pair selectors must observe pre-instruction words even when both destination
// registers alias A. The FMA inline addend must replicate rather than widen.
// Guest rounding is forwarded while host rounding remains at its default.
TEST(ArithBridgeTest, WideMadCommitsBothWordsAndCarry) {
  for (uint32_t lanes : {32u, 64u}) {
    for (bool observed : {false, true}) {
      BridgeFixture fixture(false, lanes);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
      // (2^32-1)*2 crosses the destination-word boundary without a 65th-bit
      // carry.
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
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
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

TEST(ArithBridgeTest, Fma64DirectedGuestRounding) {
  for (bool rdna3 : {false, true}) {
    for (unsigned lanes : {32u, 64u}) {
      for (unsigned rounding = 1; rounding < 4; ++rounding) {
        BridgeFixture fixture(rdna3, lanes);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
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

TEST(ArithBridgeTest, Fma64WidensSccWithoutReplicatingHighWord) {
  for (bool rdna3 : {false, true}) {
    for (uint32_t lanes : {32U, 64U}) {
      BridgeFixture fixture(rdna3, lanes);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_exec(1ULL << (lanes - 1));
      fixture.wave->write_scc(true);
      fixture.wave->set_mode_raw(0xf0); // Preserve the subnormal numeric value of SCC=1.
      const uint32_t lane = lanes - 1;
      for (uint32_t reg = 0; reg < 6; ++reg)
        fixture.cu->write_vgpr(fixture.base() + reg, lane, reg == 3 ? 0x3ff00000 : 0);
      const auto words =
          rdna3 ? rdna3::build_vop3(rdna3::kVFmaF64Vop3,
                                    {.vdst = 0, .src0 = 253, .src1 = 258, .src2 = 260})
                : rdna4::build_vop3(rdna4::kVFmaF64Vop3,
                                    {.vdst = 0, .src0 = 253, .src1 = 258, .src2 = 260});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), 1U);
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, lane), 0U);
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
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        // Round up, flush output denormals; OMOD is enabled on both
        // architectures.
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
            fixture.cu->write_vgpr(fixture.base(), lane, 0xbfa00000u); // -1.25
            fixture.cu->write_vgpr(fixture.base() + 1, lane,
                                   0xbe800000u); // -0.25
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
              RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_OMOD_2, d, a, b, rdna3);
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
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
          if (!observed) {
            fixture.cu->set_plugin_group(nullptr);
            fixture.recorder = nullptr;
          }
          // DX9 must flush its tiny input despite MODE preserving denormals.
          fixture.wave->set_mode_raw(3u << 4);
          const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000100000005ULL;
          fixture.wave->set_exec(exec_mask);
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            fixture.cu->write_vgpr(fixture.base(), lane, dx9 && lane != 2 ? 1u : 0x40000000u);
            fixture.cu->write_vgpr(fixture.base() + 1, lane,
                                   dx9 && lane != 2 ? 0x7f800000u : 0x40400000u);
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
                      exec_mask & (1ULL << lane) ? (dx9 && lane == 2 ? 0x41200000u : expected)
                                                 : 0x40800000u)
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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

TEST(ArithBridgeTest, PendingRegisterErrorPreventsVectorEvaluationAndWrites) {
  for (bool observed : {false, true}) {
    BridgeFixture fixture(false);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    if (!observed)
      fixture.cu->set_plugin_group(nullptr);
    RegisterAccess registers(*fixture.wave);
    fixture.wave->report_instruction_execution_error(
        InstructionExecutionError::InvalidRegisterAccess);
    rdna4::Operand destination(32, rdna4::OperandType::OPR_VGPR, 16);
    bool invoked = false;
    auto invoke = [&](uint32_t *const *, uint64_t) {
      invoked = true;
      return RJ_ARITH_SUCCESS;
    };
    EXPECT_FALSE(arith::write_vector32_result(*fixture.wave, registers, destination, 1, invoke));
    destination.size_bits_ = 64;
    EXPECT_FALSE(arith::write_vector64_result(*fixture.wave, registers, destination, 1, invoke));
    EXPECT_FALSE(invoked);
    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 16, 0), 0u);
    if (observed) {
      EXPECT_TRUE(fixture.recorder->writes.empty());
    }
  }
}

TEST(ArithBridgeTest, DppFloatReadsExecButWritesOnlyEnabledRows) {
  for (bool rdna3 : {false, true}) {
    for (unsigned kind = 0; kind < 4; ++kind) {
      for (uint64_t exec_mask : {0x10001ULL, 1ULL}) {
        SCOPED_TRACE(::testing::Message()
                     << "rdna3=" << rdna3 << " kind=" << kind << " exec=" << exec_mask);
        BridgeFixture fixture(rdna3);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        fixture.wave->set_exec(exec_mask);
        std::array<uint32_t, 3> words{};
        if (kind < 2) {
          const uint16_t op = kind ? rdna4::kVAddF16Vop2 : rdna4::kVAddF32Vop2;
          words[0] = rdna3 ? rdna3::build_vop2(op, {.src0 = 250, .vsrc1 = 1, .vdst = 3})[0]
                           : rdna4::build_vop2(op, {.src0 = 250, .vsrc1 = 1, .vdst = 3})[0];
          words[1] = 0x2f00e400;
        } else {
          const auto base =
              rdna3
                  ? rdna3::build_vop3(kind == 2 ? rdna3::kVMin3F32Vop3 : rdna3::kVMin3F16Vop3,
                                      {.vdst = 3, .src0 = 250, .src1 = 257, .src2 = 258})
                  : rdna4::build_vop3(kind == 2 ? rdna4::kVMin3NumF32Vop3 : rdna4::kVMin3NumF16Vop3,
                                      {.vdst = 3, .src0 = 250, .src1 = 257, .src2 = 258});
          words = {base[0], base[1], 0x2f00e400};
        }
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        uint64_t read = 0, written = 0;
        for (const auto &event : fixture.recorder->reads)
          if (event.reg == fixture.base() + 1)
            read |= event.lanes;
        for (const auto &event : fixture.recorder->writes)
          written |= event.lanes;
        EXPECT_EQ(read, exec_mask);
        EXPECT_EQ(written, exec_mask & 0x10000ULL);
      }
    }
  }
}

TEST(ArithBridgeTest, ExactMulStatusRespectsDppRows) {
  for (bool enabled : {false, true}) {
    BridgeFixture fixture(false, 32, parse_arithmetic_policy("bit_exact_empirical"));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_exec(1);
    fixture.wave->set_mode_raw(0x1f0);
    fixture.cu->write_vgpr(fixture.base(), 0, 0x7f7fffff);
    fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x40000000);
    const std::array<uint32_t, 2> words{0x100402fa, enabled ? 0xff00e400U : 0x2f00e400U};
    auto instruction = fixture.decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    ASSERT_TRUE(
        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
    EXPECT_EQ(fixture.wave->trapsts(), enabled ? 0x28U : 0U);
  }
}

TEST(ArithBridgeTest, FloatFrexpExponentIgnoresGuestRounding) {
  BridgeFixture fixture(false);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.wave->set_exec(1);
  fixture.wave->set_mode_raw(0x1f3); // Directed guest rounding, denormals preserved.
  fixture.cu->write_vgpr(fixture.base(), 0, 0x3fc00000); // 1.5 = 0.75 * 2^1.
  const auto words = rdna4::build_vop1(rdna4::kVFrexpExpI32F32Vop1, {.src0 = 256, .vdst = 1});
  auto instruction = fixture.decoder->decode_window(words);
  ASSERT_TRUE(instruction.succeeded());
  ASSERT_TRUE(
      fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 0), 1U);
}

// Compatibility uses the legacy classifier even outside mode 3's reporting domain.
TEST(ArithBridgeTest, CompatibilityMulDppStatusIncludesWriteSuppressedLanes) {
  // v_mul_f32 v2, v0, v1 with DPP row_mask:2. Lane 0 overflows but cannot
  // write.
  for (uint64_t exec_mask : {0x10001ULL, 1ULL}) {
    BridgeFixture fixture(false);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_exec(exec_mask);
    fixture.wave->set_trapsts(0x80000000);
    for (unsigned lane = 0; lane < 32; ++lane) {
      fixture.cu->write_vgpr(fixture.base(), lane, lane == 0 ? 0x7f7fffff : 0x3f800000);
      fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x40000000);
      fixture.cu->write_vgpr(fixture.base() + 2, lane, 0xdeadbeef);
    }
    const std::array<uint32_t, 2> words{0x100402fa, 0x2f00e400};
    auto instruction = fixture.decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    fixture.recorder->writes.clear();
    ASSERT_TRUE(
        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
    EXPECT_EQ(fixture.wave->trapsts(), 0x80000028u);
    for (unsigned lane = 0; lane < 32; ++lane)
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lane),
                lane == 16 && (exec_mask & 0x10000) ? 0x40000000u : 0xdeadbeefu)
          << lane;
    if (exec_mask == 1)
      EXPECT_TRUE(fixture.recorder->writes.empty());
    else {
      ASSERT_EQ(fixture.recorder->writes.size(), 1u);
      EXPECT_EQ(fixture.recorder->writes[0].lanes, 0x10000u);
    }
  }
}

TEST(ArithBridgeTest, Gfx12ExceptionalOperandsUpdateUserFlags) {
  for (const bool report : {true, false}) {
    BridgeFixture fixture(
        false, 32,
        parse_arithmetic_policy(report ? "compatibility" : "bit_exact_empirical_no_fp_status"));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_mode_raw(0xf0);
    auto check = [&](std::array<uint32_t, 2> words, uint32_t lhs, uint32_t rhs, uint32_t causes) {
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      fixture.cu->write_vgpr(fixture.base() + 1, 0, lhs);
      fixture.cu->write_vgpr(fixture.base() + 2, 0, rhs);
      fixture.cu->write_vgpr(fixture.base() + 1, 1, 0);
      fixture.cu->write_vgpr(fixture.base() + 2, 1, 0);
      for (uint64_t exec : {1u, 2u}) {
        fixture.wave->set_exec(exec);
        fixture.wave->set_trapsts(0x40);
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        EXPECT_EQ(fixture.wave->trapsts(), 0x40u | (report && exec == 1 ? causes : 0u))
            << instruction.value()->mnemonic();
      }
    };
    check({0xd5080000, 0x00020501}, 0, 0x7f800000, 1);             // MUL of zero and infinity
    check({0xd5030000, 0x00020501}, 0x7f7fffff, 0x7f7fffff, 0x28); // ADD overflows
    check({0xd5150000, 0x00020501}, 0x7f800010, 0, 1);             // MIN_NUM with sNaN
    check({0xd7650000, 0x00020501}, 0x7fc12345, 0xff800010, 1);    // MINIMUM with mixed NaNs
    check({0xd4128008, 0x00020501}, 0x7fc12345, 0, 1);             // signaling CMP
    for (uint32_t mode : {0u, 0x10u, 0x20u, 0x30u, 0x31u, 0x32u, 0x33u}) {
      fixture.wave->set_mode_raw(mode);
      check({0xc8d00501, 0x00020104}, 0x7f7fffff, 0x40000000, 0x28); // dual MUL in slot X
      check({0xca060104, 0x00020501}, 0x7f7fffff, 0x40000000, 0x28); // dual MUL in slot Y
    }
  }
}

TEST(ArithBridgeTest, CompatibilityMulRetainsLegacyStatus) {
  for (bool rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      BridgeFixture fixture(rdna3, lanes);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
                /*rdna3=*/rdna3, /*compatibility_mul=*/true);
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  RegisterAccess registers(*fixture.wave);
  rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 0);
  unsigned calls = 0;
  auto invoke = [&](uint32_t *const *pointers, uint64_t exec_mask) {
    ++calls;
    EXPECT_EQ(exec_mask, 5u);
    auto source = registers.read_vgpr_region(fixture.base(), 1, exec_mask);
    auto lanes = source.lanes(0);
    EXPECT_TRUE(lanes.succeeded());
    if (lanes.succeeded()) {
      EXPECT_EQ(pointers[0], lanes.value().data());
    }
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
  auto whole_wave = [&](uint32_t *const *) {
    ++calls;
    return RJ_ARITH_SUCCESS;
  };
  EXPECT_FALSE(registers.try_bulk_write_vgprs<2>(fixture.base(), 2, whole_wave));
  fixture.wave->set_vgpr_write_mask(~uint64_t{0});
  EXPECT_FALSE(registers.try_bulk_write_vgprs<2>(fixture.base() + 255, 2, whole_wave));
  rdna4::Operand half(16, rdna4::OperandType::OPR_VGPR, 0);
  EXPECT_FALSE(registers.try_bulk_write_operand(half, 7, invoke));
  EXPECT_EQ(calls, 1u);
}

TEST(ArithBridgeTest, UnobservedVectorWritesPreserveAliasesMasksAndRejectedOutputs) {
  for (uint32_t lanes : {32u, 64u}) {
    for (uint64_t cpu : {RJ_ARITH_CPU_BASELINE, rj_arith_init_cpu_flags()}) {
      BridgeFixture fixture(false, lanes);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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

// SUBREV routes its minuend (VSRC1), including the carry form, on both ISAs.
TEST(ArithBridgeTest, DecodedReverseSubtractRoutesSecondSource) {
  for (bool gfx11 : {false, true}) {
    for (bool carry : {false, true}) {
      for (bool dpp8 : {false, true}) {
        BridgeFixture fixture(gfx11);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        fixture.wave->set_exec(3);
        fixture.wave->set_vcc_mask(0);
        fixture.cu->write_vgpr(fixture.base(), 0, 10);
        fixture.cu->write_vgpr(fixture.base(), 1, 20);
        fixture.cu->write_vgpr(fixture.base() + 1, 0, 100);
        fixture.cu->write_vgpr(fixture.base() + 1, 1, 200);
        const auto opcode = carry ? rdna4::kVSubrevCoCiU32Vop2 : rdna4::kVSubrevNcU32Vop2;
        std::array<uint32_t, 2> words;
        if (dpp8) {
          rdna4::Vop2VopDpp8MachineInst raw{};
          raw.src0 = amdgpu::SRC_DPP8_FI_0;
          raw.vsrc1 = 1;
          raw.vdst = 2;
          raw.op = opcode;
          raw.lane_sel_0 = 1;
          words = std::bit_cast<decltype(words)>(raw);
        } else {
          rdna4::Vop2VopDpp16MachineInst raw{};
          raw.src0 = amdgpu::SRC_DPP;
          raw.vsrc1 = 1;
          raw.vdst = 2;
          raw.op = opcode;
          raw.dpp_ctrl = 1; // quad_perm lane 0 selects lane 1.
          raw.row_mask = raw.bank_mask = 0xf;
          words = std::bit_cast<decltype(words)>(raw);
        }
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0), 190u);
      }
    }
  }
}

// gfx1201 DPP16 scalar/inline SRC1 is zero, unlike ordinary SUBREV.
TEST(ArithBridgeTest, ReverseSubtractDppScalarSourceIsZero) {
  for (unsigned lanes : {32u, 64u}) {
    for (bool dpp8 : {false, true}) {
      for (unsigned source : {24u, 133u, 257u}) {
        SCOPED_TRACE(lanes);
        SCOPED_TRACE(dpp8);
        SCOPED_TRACE(source);
        const bool vector = source == 257;
        BridgeFixture fixture(false, lanes);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        fixture.wave->set_exec(vector ? 3 : ~0ULL);
        fixture.cu->write_sgpr(fixture.wave->sgpr_alloc().base + 24, 100);
        for (unsigned lane = 0; lane < lanes; ++lane)
          fixture.cu->write_vgpr(fixture.base(), lane, 10 * (lane + 1));
        fixture.cu->write_vgpr(fixture.base() + 1, 0, 100);
        fixture.cu->write_vgpr(fixture.base() + 1, 1, 5);
        const auto initialize = [&](auto &raw) {
          raw.encoding = 0x35;
          raw.op = rdna4::kVSubrevCoU32Vop3SdstEnc;
          raw.vdst = 4;
          raw.src0 = dpp8 ? SRC_DPP8_FI_0 : SRC_DPP;
          raw.src1 = source;
        };
        std::array<uint32_t, 3> words;
        if (dpp8) {
          rdna4::Vop3SdstEncVopDpp8MachineInst raw{};
          initialize(raw);
          raw.lane_sel_0 = raw.lane_sel_1 = raw.lane_sel_2 = raw.lane_sel_3 = 1;
          raw.lane_sel_4 = raw.lane_sel_5 = raw.lane_sel_6 = raw.lane_sel_7 = 1;
          words = std::bit_cast<decltype(words)>(raw);
        } else {
          rdna4::Vop3SdstEncVopDpp16MachineInst raw{};
          initialize(raw);
          raw.dpp_ctrl = vector ? 0xb1u : uint32_t(dpp::ROW_SHL1);
          raw.row_mask = raw.bank_mask = 0xf;
          words = std::bit_cast<decltype(words)>(raw);
        }
        auto inst = fixture.decoder->decode_window(words);
        ASSERT_TRUE(inst.succeeded());
        fixture.recorder->scalar_reads.clear();
        ASSERT_TRUE(fixture.cu->execute_instruction(inst.value().get(), *fixture.wave).succeeded());
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 4, 0), uint32_t(vector ? -5 : -10));
        if (vector) {
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 4, 1), uint32_t(dpp8 ? -15 : 80));
        }
        const uint32_t borrow = vector ? (dpp8 ? 3 : 1) : dpp8 ? UINT32_MAX : 0x7fff7fffU;
        EXPECT_EQ(fixture.cu->read_sgpr(fixture.wave->sgpr_alloc().base), borrow);
        if (lanes == 64) {
          EXPECT_EQ(fixture.cu->read_sgpr(fixture.wave->sgpr_alloc().base + 1),
                    vector ? 0u : borrow);
        }
        if (source == 24) {
          for (auto read : fixture.recorder->scalar_reads) {
            EXPECT_NE(read, fixture.wave->sgpr_alloc().base + 24);
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, EmptyMad64SkipsScalarAccumulatorAndClearsCarry) {
  class ScalarReads : public ExecutionPlugin {
  public:
    ScalarReads() : ExecutionPlugin("scalar_reads") {}
    void onAmdgpuReadSgpr(const Wavefront *, uint32_t) override { ++count; }
    int count = 0;
  };
  BridgeFixture fixture(false, 64);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<ScalarReads>();
  auto *reads = observer.get();
  group->add(std::move(observer));
  fixture.cu->set_plugin_group(group);
  fixture.wave->set_exec(0);
  auto base = fixture.wave->sgpr_alloc().base;
  fixture.cu->write_sgpr(base, ~0U);
  fixture.cu->write_sgpr(base + 1, ~0U);
  auto words =
      rdna4::build_vop3_sdst_enc(rdna4::kVMadCoU64U32Vop3SdstEnc,
                                 {.vdst = 4, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = 10});
  auto instruction = fixture.decoder->decode_window(words);
  ASSERT_TRUE(instruction.succeeded());
  ASSERT_TRUE(
      fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
  EXPECT_EQ(reads->count, 0);
  words = rdna4::build_vop3_sdst_enc(
      rdna4::kVAddCoCiU32Vop3SdstEnc,
      {.vdst = 4, .sdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = 10});
  instruction = fixture.decoder->decode_window(words);
  ASSERT_TRUE(instruction.succeeded());
  ASSERT_TRUE(
      fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
  EXPECT_EQ(reads->count, 0);
  EXPECT_EQ(fixture.cu->read_sgpr(base), 0U);
  EXPECT_EQ(fixture.cu->read_sgpr(base + 1), 0U);
}

TEST(ArithBridgeTest, DppCarryDoesNotRestoreSuppressedDestinationBits) {
  for (bool gfx11 : {false, true}) {
    BridgeFixture fixture(gfx11);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_exec(0xffffffffU);
    auto base = fixture.wave->sgpr_alloc().base;
    for (unsigned lane = 0; lane < 32; ++lane) {
      fixture.cu->write_vgpr(fixture.base(), lane, ~0U);
      fixture.cu->write_vgpr(fixture.base() + 1, lane, 1);
    }
    rdna4::Vop3SdstEncVopDpp16MachineInst raw{};
    raw.encoding = 0x35;
    raw.op = gfx11 ? rdna3::kVAddCoU32Vop3SdstEnc : rdna4::kVAddCoU32Vop3SdstEnc;
    raw.vdst = 2;
    raw.sdst = 4;
    raw.src0 = SRC_DPP;
    raw.dpp_ctrl = dpp::ROW_SHL1;
    raw.fi = 1;
    raw.row_mask = raw.bank_mask = 0xf;
    for (auto src1 : {257, 129}) { // v1 and inline constant 1
      raw.src1 = src1;
      auto words = std::bit_cast<std::array<uint32_t, 3>>(raw);
      auto instruction = fixture.decoder->decode_window(words);
      if (gfx11 && src1 == 129) {
        EXPECT_FALSE(instruction.succeeded());
        continue;
      }
      ASSERT_TRUE(instruction.succeeded());
      fixture.cu->write_sgpr(base + 4, ~0U);
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      EXPECT_EQ(fixture.cu->read_sgpr(base + 4), 0x7fff7fffU);
    }
  }
}

TEST(ArithBridgeTest, VectorCarryMasksScalarAndVectorResultsTogether) {
  for (uint32_t lanes : {32u, 64u}) {
    for (bool observed : {false, true}) {
      for (uint64_t cpu : {RJ_ARITH_CPU_BASELINE, rj_arith_init_cpu_flags()}) {
        for (uint64_t write_mask : {~0ULL, 0ULL, 0x5555555555555555ULL}) {
          BridgeFixture fixture(false, lanes);
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
          EXPECT_EQ(carry, exec_mask & write_mask);
          for (uint32_t lane = 0; lane < lanes; ++lane)
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                      ((exec_mask & write_mask) >> lane) & 1 ? 0u : 0xffffffffu);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, VectorCarryZeroExecClearsScalarResultWithoutVectorAccess) {
  for (bool rdna3 : {true, false}) {
    for (uint32_t lanes : {32u, 64u}) {
      BridgeFixture fixture(rdna3, lanes);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_exec(0);
      const auto base = fixture.wave->sgpr_alloc().base;
      fixture.cu->write_sgpr(base + 12, ~0u);
      fixture.cu->write_sgpr(base + 13, ~0u);
      // Both data sources are scalar, including one overlapping the carry
      // result.
      const auto words = rdna3 ? rdna3::build_vop3_sdst_enc(
                                     rdna3::kVAddCoU32Vop3SdstEnc,
                                     {.vdst = 4, .sdst = 12, .clamp = 1, .src0 = 12, .src1 = 8})
                               : rdna4::build_vop3_sdst_enc(
                                     rdna4::kVAddCoU32Vop3SdstEnc,
                                     {.vdst = 4, .sdst = 12, .clamp = 1, .src0 = 12, .src1 = 8});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      fixture.recorder->scalar_reads.clear();
      fixture.recorder->reads.clear();
      fixture.recorder->writes.clear();
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      EXPECT_TRUE(fixture.recorder->scalar_reads.empty());
      EXPECT_TRUE(fixture.recorder->reads.empty());
      EXPECT_TRUE(fixture.recorder->writes.empty());
      EXPECT_EQ(fixture.cu->read_sgpr(base + 12), 0u);
      EXPECT_EQ(fixture.cu->read_sgpr(base + 13), lanes == 64 ? 0u : ~0u);
    }
  }
}

TEST(ArithBridgeTest, RejectedVectorCarryDoesNotCommitEitherOutput) {
  for (uint64_t exec_mask : {0ULL, 1ULL}) {
    for (bool observed : {false, true}) {
      BridgeFixture fixture(false);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      if (!observed) {
        fixture.cu->set_plugin_group(nullptr);
        fixture.recorder = nullptr;
      }
      fixture.wave->set_exec(exec_mask);
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
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
  // Signed cases inherit the existing rj-arith
  // Mad64.SignedScalarOutputIsTheExtendedSign fixtures, validated against its
  // gfx1201 hardware corpus: bit 64, not overflow.
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
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
                  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.wave->set_exec((1u << 0) | (1u << 16) | (1u << 17));
  fixture.wave->set_mode_raw(0xf0);
  fixture.cu->write_vgpr(fixture.base(), 0, 0x7f800001); // Disabled row.
  fixture.cu->write_vgpr(fixture.base(), 15,
                         0x7f800001); // Invalid shift source.
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
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
                ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
                  // B is a signed integer: its ABS/NEG encoding bits are
                  // ignored.
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
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
    // Compatibility selects the TRANS-unit model for RCP/RSQ/SQRT; nonzero
    // encodings are the gfx1201/gfx1100 results for 1.5.
    uint64_t compatibility = 0;
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
       1.0 / 1.5,
       0x3fe5555554000000ULL},
      {{rdna3::kVRsqF64Vop1, rdna3::kVRsqF64Vop3},
       {rdna4::kVRsqF64Vop1, rdna4::kVRsqF64Vop3},
       1.0 / std::sqrt(1.5),
       0x3fea20bd74000000ULL},
      {{rdna3::kVSqrtF64Vop1, rdna3::kVSqrtF64Vop3},
       {rdna4::kVSqrtF64Vop1, rdna4::kVSqrtF64Vop3},
       std::sqrt(1.5),
       0x3ff3988e14000000ULL},
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
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
              // Integral rounding is named; FRACT subtraction and OMOD use
              // guest rounding. Alternate denormal preservation to exercise
              // RDNA3 OMOD gating.
              fixture.wave->set_mode_raw((guest << 2) | ((guest & 1) ? 3u << 6 : 0));
              fixture.wave->set_trapsts(0x10);
              const uint64_t exec_mask = lanes == 32 ? 0x80000003ULL : 0x8000000380000003ULL;
              fixture.wave->set_exec(exec_mask);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                fixture.cu->write_vgpr(fixture.base(), lane, 0);
                fixture.cu->write_vgpr(fixture.base() + 1, lane,
                                       0x3ff80000); // 1.5
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
              const bool doubled = vop3 && (!rdna3 || !(guest & 1));
              const double result = test.result * (doubled ? 2 : 1);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                const uint64_t actual =
                    fixture.cu->read_vgpr(fixture.base(), lane) |
                    (uint64_t(fixture.cu->read_vgpr(fixture.base() + 1, lane)) << 32);
                const bool written = !rejected && ((exec_mask >> lane) & 1);
                if (written && test.compatibility && policy_index == 0) {
                  EXPECT_EQ(actual, test.compatibility + (doubled ? 1ULL << 52 : 0));
                  continue;
                }
                const double expected = written ? result : 1.5;
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
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
            // B is signed integer data; its ABS/NEG bits are ignored. Alias
            // D/B.
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
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
                  if (rdna3 && !exponent && index == 7) {
                    expected = inputs[index]; // gfx11 preserves sNaN with IEEE clear.
                  }
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
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
              // Input flushing makes subnormal exponents zero. Other FP
              // controls do not modify the integer result.
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
                ASSERT_FALSE(::testing::Test::HasFatalFailure());
                // Preserve denormals and enable IEEE: RDNA3 suppresses OMOD,
                // while RDNA4 honors it. Guest rounding must not reject SFUs.
                fixture.wave->set_mode_raw(0x3c0 | (rounding << 2));
                fixture.wave->set_trapsts(0x10);
                const uint64_t exec_mask = lanes == 32 ? 0x80000001ULL : 0x8000000180000001ULL;
                fixture.wave->set_exec(exec_mask);
                const unsigned a_shift = halves & 1 ? 16 : 0;
                const unsigned d_shift = halves & 2 ? 16 : 0;
                // VOP3 ABS must remove this sign before evaluating the
                // function.
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
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
              // Two largest finite inputs overflow. Only FP16 honors
              // saturation.
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
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
                      // Packed selection preserves denormal outputs (upstream
                      // #13898).
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
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
          // RDNA3 and RDNA4 support exact semantics without FP status, not with it.
          const bool rejected = policy_index == 2;
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
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
                ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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

TEST(ArithBridgeTest, DecodedWideningMadAndIntegerPack) {
  const uint16_t rdna3_opcodes[] = {rdna3::kVMadU32U16Vop3, rdna3::kVMadI32I16Vop3,
                                    rdna3::kVCvtPkU16U32Vop3, rdna3::kVCvtPkI16I32Vop3};
  const uint16_t rdna4_opcodes[] = {rdna4::kVMadU32U16Vop3, rdna4::kVMadI32I16Vop3,
                                    rdna4::kVCvtPkU16U32Vop3, rdna4::kVCvtPkI16I32Vop3};
  auto signed_value = [](uint32_t value, unsigned bits) -> int64_t {
    return value & (1u << (bits - 1)) ? int64_t(value) - (1LL << bits) : value;
  };
  for (bool is_rdna3 : {false, true}) {
    for (uint32_t lanes : {32u, 64u}) {
      for (unsigned operation = 0; operation < 4; ++operation) {
        for (unsigned opsel = 0; opsel < (operation < 2 ? 4u : 1u); ++opsel) {
          for (bool clamp : {false, true}) {
            for (bool observed : {false, true}) {
              BridgeFixture fixture(is_rdna3, lanes);
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
              if (!observed) {
                fixture.cu->set_plugin_group(nullptr);
                fixture.recorder = nullptr;
              }
              const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000180000005ULL;
              fixture.wave->set_exec(exec_mask);
              std::array<std::array<uint32_t, 64>, 3> inputs;
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                inputs[0][lane] = 0x8000ffffu ^ (lane * 0x13571357u);
                inputs[1][lane] = 0x7fffffffu ^ (lane * 0x12345679u);
                inputs[2][lane] = 0x7fffffffu ^ (lane * 0x01010101u);
                for (unsigned source = 0; source < 3; ++source)
                  fixture.cu->write_vgpr(fixture.base() + source, lane, inputs[source][lane]);
              }
              const auto words =
                  is_rdna3 ? rdna3::build_vop3(rdna3_opcodes[operation],
                                               {.vdst = 0,
                                                .op_sel = uint8_t(opsel),
                                                .clamp = uint8_t(clamp),
                                                .src0 = 256,
                                                .src1 = 257,
                                                .src2 = uint16_t(operation < 2 ? 258 : 0)})
                           : rdna4::build_vop3(rdna4_opcodes[operation],
                                               {.vdst = 0,
                                                .opsel = uint8_t(opsel),
                                                .clamp = uint8_t(clamp),
                                                .src0 = 256,
                                                .src1 = 257,
                                                .src2 = uint16_t(operation < 2 ? 258 : 0)});
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              if (observed) {
                ASSERT_EQ(fixture.recorder->reads.size(), operation < 2 ? 3u : 2u);
                for (unsigned source = 0; source < fixture.recorder->reads.size(); ++source) {
                  EXPECT_EQ(fixture.recorder->reads[source].bytes,
                            operation < 2 && source < 2 ? ((opsel & (1u << source)) ? 0xC : 0x3)
                                                        : 0xF);
                  EXPECT_EQ(fixture.recorder->reads[source].lanes, exec_mask);
                }
                ASSERT_EQ(fixture.recorder->writes.size(), 1u);
                EXPECT_EQ(fixture.recorder->writes[0].bytes, 0xF);
                EXPECT_EQ(fixture.recorder->writes[0].lanes, exec_mask);
              }
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                uint32_t expected = inputs[0][lane];
                if (exec_mask & (1ULL << lane)) {
                  if (operation < 2) {
                    const uint32_t a = (inputs[0][lane] >> ((opsel & 1) ? 16 : 0)) & 0xffff;
                    const uint32_t b = (inputs[1][lane] >> ((opsel & 2) ? 16 : 0)) & 0xffff;
                    int64_t value = operation == 0 ? int64_t(a) * b + inputs[2][lane]
                                                   : signed_value(a, 16) * signed_value(b, 16) +
                                                         signed_value(inputs[2][lane], 32);
                    if (clamp)
                      value = operation == 0
                                  ? std::clamp<int64_t>(value, 0, 0xffffffffLL)
                                  : std::clamp<int64_t>(value, -0x80000000LL, 0x7fffffffLL);
                    expected = uint32_t(value);
                  } else {
                    auto pack = [&](uint32_t value) {
                      return operation == 2 ? std::min(value, 65535u)
                                            : uint32_t(std::clamp<int64_t>(signed_value(value, 32),
                                                                           -32768, 32767)) &
                                                  0xffff;
                    };
                    expected = pack(inputs[0][lane]) | (pack(inputs[1][lane]) << 16);
                  }
                }
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), expected)
                    << operation << " " << opsel << " " << clamp << " " << lane;
              }
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
    for (bool partial : {false, true}) {
      BridgeFixture fixture(false);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.cu->set_plugin_group(nullptr);
      fixture.recorder = nullptr;
      fixture.wave->set_exec(0); // Matrix execution is independent of EXEC.
      fixture.wave->set_vgpr_write_mask(partial ? 0x55555555ULL : ~0ULL);
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
                             destination, fixture.base(), fixture.base() + 4, fixture.base() + 12,
                             4, 8, 8, 32);
        else
          arith::exec_wmma(*fixture.wave, rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12, flags, 0,
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
                    partial && (lane & 1) ? (sparse ? 0x3f800000u : 0x3c003c00u)
                                          : (sparse ? 0x41880000u : 0x41800000u));
    }
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000100000005ULL;
      fixture.wave->set_exec(exec_mask);
      rdna4::Operand d(32, rdna4::OperandType::OPR_VGPR, 0);
      rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
      rdna4::Operand b(32, rdna4::OperandType::OPR_SRC,
                       129); // Inline integer 1.
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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

// DPP suppresses whole destination banks; unpermuted inputs must use that same mask.
TEST(ArithBridgeTest, CubeDppReadsOnlyWritingLanes) {
  BridgeFixture fixture(true, 32);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.wave->set_exec(0xffffffffu);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x3f800000u);
    fixture.cu->write_vgpr(fixture.base() + 2, lane, 0x40000000u);
    fixture.cu->write_vgpr(fixture.base() + 3, lane, 0x3f800000u);
  }
  rdna3::Vop3VopDpp16MachineInst raw{};
  raw.encoding = 0x35;
  raw.op = rdna3::kVCubeidF32Vop3;
  raw.vdst = 5;
  raw.src0 = SRC_DPP;
  raw.src1 = 257;
  raw.src2 = 259;
  raw.vsrc0 = 2;
  raw.dpp_ctrl = dpp::ROW_SHR1;
  raw.fi = 1;
  raw.row_mask = 1;
  raw.bank_mask = 0xe;
  rdna3::VCubeidF32Vop3 instruction(reinterpret_cast<const rdna3::MachineInst *>(&raw));
  ASSERT_TRUE(fixture.cu->execute_instruction(&instruction, *fixture.wave).succeeded());
  unsigned unpermuted_reads = 0;
  for (const auto &read : fixture.recorder->reads) {
    if (read.reg == fixture.base() + 1 || read.reg == fixture.base() + 3) {
      EXPECT_EQ(read.lanes, 0xfff0u);
      ++unpermuted_reads;
    }
  }
  EXPECT_EQ(unpermuted_reads, 2u);
  ASSERT_EQ(fixture.recorder->writes.size(), 1u);
  EXPECT_EQ(fixture.recorder->writes[0].lanes, 0xfff0u);
}

TEST(ArithBridgeTest, TernaryVectorReadsAllInputsBeforeMaskedAliasedCommit) {
  for (uint32_t lanes : {32u, 64u}) {
    for (uint64_t cpu : {RJ_ARITH_CPU_BASELINE, rj_arith_init_cpu_flags()}) {
      BridgeFixture fixture(false, lanes);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
        EXPECT_EQ(read.lanes, exec_mask);
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
          for (const auto &read : fixture.recorder->reads) {
            const uint64_t high =
                read.reg == fixture.base() ? RJ_ARITH_ALU_HIGH_A : RJ_ARITH_ALU_HIGH_B;
            EXPECT_EQ(read.bytes, mode & high ? 0xCu : 0x3u);
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, IntegerCompareReadsPackedVopcHighHalf) {
  BridgeFixture fixture(false);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
    arith::execute_compare<64>(
        *fixture.wave, rj_arith_v_cmp_eq_u64_w32, rj_arith_v_cmp_eq_u64_w64,
        RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, a, b, false, [&](uint64_t result) {
          RegisterAccess registers(*fixture.wave);
          EXPECT_TRUE(registers.write_sgpr(sgpr, uint32_t(result)).succeeded());
          if (lanes == 64) {
            EXPECT_TRUE(registers.write_sgpr(sgpr + 1, uint32_t(result >> 32)).succeeded());
          }
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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

TEST(ArithBridgeTest, ConstantCompareDppDoesNotObserveSources) {
  BridgeFixture fixture(true);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.wave->set_exec(0x10001);
  rdna3::VopcVopDpp16MachineInst raw{};
  raw.src0 = SRC_DPP;
  raw.vsrc0 = 0;
  raw.vsrc1 = 1;
  raw.dpp_ctrl = 0xe4; // Identity quad permutation.
  raw.bank_mask = 0xf;
  raw.row_mask = 1;
  rdna3::VCmpTI32Vopc instruction(reinterpret_cast<const rdna3::MachineInst *>(&raw));
  instruction.execute_impl(*fixture.wave);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  EXPECT_EQ(fixture.wave->vcc(), 1u);
  EXPECT_TRUE(fixture.recorder->reads.empty());
  raw.src0_neg = 1;
  raw.src1_abs = 1;
  rdna3::VCmpTF32Vopc modified(reinterpret_cast<const rdna3::MachineInst *>(&raw));
  modified.execute_impl(*fixture.wave);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  EXPECT_EQ(fixture.wave->vcc(), 1u);
  EXPECT_TRUE(fixture.recorder->reads.empty());
  rdna3::Vop3VopDpp16MachineInst vop3{};
  vop3.vdst = 4;
  fixture.cu->write_sgpr(fixture.wave->sgpr_alloc().base + 4, 0x80);
  fixture.wave->set_vcc_mask(0x40);
  vop3.src0 = SRC_DPP;
  vop3.src1 = 257;
  vop3.vsrc0 = 0;
  vop3.neg = 1;
  vop3.abs = 2;
  vop3.dpp_ctrl = 0xe4;
  vop3.bank_mask = 0xf;
  vop3.row_mask = 1;
  rdna3::VCmpTF32Vop3 explicit_destination(reinterpret_cast<const rdna3::MachineInst *>(&vop3));
  explicit_destination.execute_impl(*fixture.wave);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  EXPECT_EQ(fixture.cu->read_sgpr(fixture.wave->sgpr_alloc().base + 4), 1u);
  EXPECT_EQ(fixture.wave->vcc(), 0x40u);
  EXPECT_TRUE(fixture.recorder->reads.empty());
}

TEST(ArithBridgeTest, PendingRegisterErrorPreventsCompareMaskAndStatusCommit) {
  BridgeFixture fixture(false, 32);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.wave->set_exec(1);
  fixture.wave->set_trapsts(0x80000000U);
  fixture.cu->write_vgpr(fixture.base(), 0,
                         0x7f800001U); // Would raise invalid.
  rdna4::Operand a(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
  fixture.wave->report_instruction_execution_error(
      InstructionExecutionError::InvalidRegisterAccess);
  bool committed = false;
  arith::execute_compare<32>(*fixture.wave, rj_arith_v_cmp_eq_u32_w32, rj_arith_v_cmp_eq_u32_w64, 0,
                             0, a, a, false, [&](uint64_t) { committed = true; });
  EXPECT_FALSE(committed);
  committed = false;
  arith::execute_float_compare<32>(
      *fixture.wave, rj_arith_v_cmp_eq_f32_w32_gfx12, rj_arith_v_cmp_eq_f32_w64_gfx12,
      RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, 0, a, a, false, [&](uint64_t) { committed = true; });
  EXPECT_FALSE(committed);
  EXPECT_EQ(fixture.wave->trapsts(), 0x80000000U);
  EXPECT_EQ(fixture.wave->pending_alu_causes(), 0U);
}

TEST(ArithBridgeTest, ClassCompare64ReadsOneClassMaskWordAtTheAllocationBoundary) {
  for (uint32_t lanes : {32u, 64u}) {
    BridgeFixture fixture(false, lanes);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const uint64_t exec_mask = lanes == 64 ? (1ULL << 32) | 5 : 5;
    fixture.wave->set_exec(exec_mask);
    fixture.cu->write_vgpr(fixture.base(), 0, 1);
    fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x7ff80000); // Quiet NaN.
    fixture.cu->write_vgpr(fixture.base(), 2, 0);
    fixture.cu->write_vgpr(fixture.base() + 1, 2, 0x80000000); // Negative zero.
    if (lanes == 64) {
      fixture.cu->write_vgpr(fixture.base(), 32, 0);
      fixture.cu->write_vgpr(fixture.base() + 1, 32,
                             0x7ff00000); // Positive infinity.
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.wave->set_exec(1);
  fixture.wave->set_mode_raw(0); // Classification must not flush subnormals.
  fixture.cu->write_vgpr(fixture.base(), 0,
                         0x80013c00); // High half: negative subnormal.
  fixture.cu->write_vgpr(fixture.base() + 1, 0, RJ_ARITH_CLASS_NEG_SUBNORMAL << 16);
  rdna4::Operand a(16, rdna4::OperandType::OPR_SRC, 256);
  rdna4::Operand b(16, rdna4::OperandType::OPR_SRC, 257);
  for (uint64_t mode :
       {uint64_t(RJ_ARITH_ALU_HIGH_A), uint64_t(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B),
        uint64_t(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_ABS_A)}) {
    fixture.recorder->reads.clear();
    uint64_t result = ~0ULL;
    arith::execute_compare<16, 16>(*fixture.wave, rj_arith_v_cmp_class_f16_w32,
                                   rj_arith_v_cmp_class_f16_w64,
                                   fixture.wave->cu().arithmetic_flags(), mode, a, b, false,
                                   [&](uint64_t value) { result = value; });
    ASSERT_FALSE(fixture.wave->instruction_execution_failed());
    ASSERT_EQ(fixture.recorder->reads.size(), 2u);
    EXPECT_EQ(fixture.recorder->reads[0].bytes, 0xc);
    EXPECT_EQ(fixture.recorder->reads[1].bytes, mode & RJ_ARITH_ALU_HIGH_B ? 0xc : 0x3);
    EXPECT_EQ(result, mode == (RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B) ? 1u : 0u);
  }
}

TEST(ArithBridgeTest, FloatCompareReportsRepeatedCausesAndPreservesUnrelatedState) {
  for (uint32_t lanes : {32u, 64u}) {
    BridgeFixture fixture(false, lanes, parse_arithmetic_policy("bit_exact_empirical"));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.wave->set_exec((1u << 0) | (1u << 16) | (1u << 17));
  fixture.wave->set_mode_raw(0xf0);
  fixture.cu->write_vgpr(fixture.base(), 0, 0x7f800001); // Disabled row.
  fixture.cu->write_vgpr(fixture.base(), 16, 0x3f800000);
  fixture.cu->write_vgpr(fixture.base() + 1, 16,
                         0x7f800001); // Invalid DPP source lane.
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
  uint64_t observed_rhs = 0;
  for (const auto &read : fixture.recorder->reads) {
    if (read.reg == fixture.base() + 1)
      observed_rhs |= read.lanes;
  }
  EXPECT_EQ(observed_rhs, fixture.wave->exec());
}

TEST(ArithBridgeTest, RejectedFloatCompareDppDoesNotChangeExecOrExceptionState) {
  BridgeFixture fixture(true, 32, parse_arithmetic_policy("bit_exact_empirical"));
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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

TEST(ArithBridgeTest, ScalarAddIgnoresExecAndCommitsCarryAfterAliasedInputs) {
  for (bool rdna3 : {false, true}) {
    BridgeFixture fixture(rdna3);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_exec(0);
    fixture.wave->write_scc(false);
    const auto base = fixture.wave->sgpr_alloc().base;
    fixture.cu->write_sgpr(base, 0xffffffffu);
    fixture.cu->write_sgpr(base + 1, 1u);
    fixture.recorder->scalar_reads.clear();
    fixture.recorder->scalar_writes.clear();
    if (rdna3) {
      const auto words = rocjitsu::rdna3::build_sop2(rocjitsu::rdna3::kSAddU32Sop2,
                                                     {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0});
      rocjitsu::rdna3::SAddU32Sop2 instruction(words.data());
      instruction.execute_impl(*fixture.wave);
    } else {
      const auto words =
          rdna4::build_sop2(rdna4::kSAddCoU32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0});
      rdna4::SAddCoU32Sop2 instruction(words.data());
      instruction.execute_impl(*fixture.wave);
    }
    ASSERT_FALSE(fixture.wave->instruction_execution_failed());
    EXPECT_EQ(fixture.recorder->scalar_reads, (std::vector<uint32_t>{base, base + 1}));
    ASSERT_EQ(fixture.recorder->scalar_writes.size(), 1u);
    EXPECT_EQ(fixture.recorder->scalar_writes[0].index, 0u);
    EXPECT_EQ(fixture.recorder->scalar_writes[0].width, 1u);
    EXPECT_EQ(fixture.cu->read_sgpr(base), 0u);
    EXPECT_TRUE(fixture.wave->read_scc());
    EXPECT_TRUE(fixture.recorder->writes.empty());
  }
}

TEST(ArithBridgeTest, ScalarPairReadAndWritePreserveWidthsAndScc) {
  for (bool rdna3 : {false, true}) {
    BridgeFixture fixture(rdna3);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_exec(0);
    const auto base = fixture.wave->sgpr_alloc().base;
    fixture.cu->write_sgpr(base, 0x12345678u);
    fixture.cu->write_sgpr(base + 1, 0xabcdef01u);
    fixture.cu->write_sgpr(base + 2, 0xffff0000u);
    fixture.cu->write_sgpr(base + 3, 0x0000ffffu);
    fixture.recorder->scalar_reads.clear();
    fixture.recorder->scalar_writes.clear();
    if (rdna3) {
      const auto words = rocjitsu::rdna3::build_sop2(rocjitsu::rdna3::kSAndB64Sop2,
                                                     {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0});
      rocjitsu::rdna3::SAndB64Sop2 instruction(words.data());
      instruction.execute_impl(*fixture.wave);
    } else {
      const auto words =
          rdna4::build_sop2(rdna4::kSAndB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0});
      rdna4::SAndB64Sop2 instruction(words.data());
      instruction.execute_impl(*fixture.wave);
    }
    ASSERT_FALSE(fixture.wave->instruction_execution_failed());
    EXPECT_EQ(fixture.recorder->scalar_reads,
              (std::vector<uint32_t>{base, base + 1, base + 2, base + 3}));
    ASSERT_EQ(fixture.recorder->scalar_writes.size(), 1u);
    EXPECT_EQ(fixture.recorder->scalar_writes[0].index, 0u);
    EXPECT_EQ(fixture.recorder->scalar_writes[0].width, 2u);
    EXPECT_EQ(fixture.cu->read_sgpr(base), 0x12340000u);
    EXPECT_EQ(fixture.cu->read_sgpr(base + 1), 0x0000ef01u);
    EXPECT_TRUE(fixture.wave->read_scc());
  }
}

TEST(ArithBridgeTest, ScalarSelectReadsOnlyChosenSourceAndPreservesScc) {
  for (bool rdna3 : {false, true})
    for (bool wide : {false, true})
      for (bool scc : {false, true}) {
        BridgeFixture fixture(rdna3);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        fixture.wave->set_exec(0);
        fixture.wave->write_scc(scc);
        const auto base = fixture.wave->sgpr_alloc().base;
        const uint32_t values[] = {0x12345678u, 0xabcdef01u, 0x98765432u, 0x10203040u};
        for (unsigned i = 0; i < 4; ++i)
          fixture.cu->write_sgpr(base + i, values[i]);
        fixture.recorder->scalar_reads.clear();
        fixture.recorder->scalar_writes.clear();
        const auto words =
            rdna3 ? rdna3::build_sop2(wide ? rdna3::kSCselectB64Sop2 : rdna3::kSCselectB32Sop2,
                                      {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0})
                  : rdna4::build_sop2(wide ? rdna4::kSCselectB64Sop2 : rdna4::kSCselectB32Sop2,
                                      {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0});
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        const unsigned selected = scc ? 0 : 2;
        std::vector<uint32_t> reads{base + selected};
        if (wide)
          reads.push_back(base + selected + 1);
        EXPECT_EQ(fixture.recorder->scalar_reads, reads);
        EXPECT_EQ(fixture.cu->read_sgpr(base), values[selected]);
        EXPECT_EQ(fixture.cu->read_sgpr(base + 1), wide ? values[selected + 1] : values[1]);
        EXPECT_EQ(fixture.wave->read_scc(), scc);
        ASSERT_EQ(fixture.recorder->scalar_writes.size(), 1u);
        EXPECT_EQ(fixture.recorder->scalar_writes[0].width, wide ? 2u : 1u);
      }
}

TEST(ArithBridgeTest, ScalarBitsetCapturesAliasedDestinationAndMasksIndex) {
  struct Case {
    uint16_t rdna3_op, rdna4_op;
    bool wide, set;
  };
  const Case cases[] = {{rdna3::kSBitset0B32Sop1, rdna4::kSBitset0B32Sop1, false, false},
                        {rdna3::kSBitset0B64Sop1, rdna4::kSBitset0B64Sop1, true, false},
                        {rdna3::kSBitset1B32Sop1, rdna4::kSBitset1B32Sop1, false, true},
                        {rdna3::kSBitset1B64Sop1, rdna4::kSBitset1B64Sop1, true, true}};
  for (bool rdna3 : {false, true})
    for (const auto &test : cases)
      for (bool scc : {false, true}) {
        BridgeFixture fixture(rdna3);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        fixture.wave->set_exec(0);
        fixture.wave->write_scc(scc);
        const auto base = fixture.wave->sgpr_alloc().base;
        const uint64_t initial = test.set ? 0x012345670000007fULL : 0xffffffffffffffffULL;
        fixture.cu->write_sgpr(base, uint32_t(initial));
        fixture.cu->write_sgpr(base + 1, uint32_t(initial >> 32));
        fixture.recorder->scalar_reads.clear();
        fixture.recorder->scalar_writes.clear();
        const auto words = rdna3 ? rdna3::build_sop1(test.rdna3_op, {.ssrc0 = 0, .sdst = 0})
                                 : rdna4::build_sop1(test.rdna4_op, {.ssrc0 = 0, .sdst = 0});
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        const uint64_t bit = test.wide ? 0x8000000000000000ULL : 0x80000000ULL;
        const uint64_t expected = test.set ? initial | bit : initial & ~bit;
        const std::vector<uint32_t> reads = test.wide ? std::vector<uint32_t>{base, base + 1, base}
                                                      : std::vector<uint32_t>{base, base};
        EXPECT_EQ(fixture.recorder->scalar_reads, reads);
        EXPECT_EQ(fixture.cu->read_sgpr(base), uint32_t(expected));
        EXPECT_EQ(fixture.cu->read_sgpr(base + 1), uint32_t(expected >> 32));
        EXPECT_EQ(fixture.wave->read_scc(), scc);
        ASSERT_EQ(fixture.recorder->scalar_writes.size(), 1u);
        EXPECT_EQ(fixture.recorder->scalar_writes[0].width, test.wide ? 2u : 1u);
      }
}

TEST(ArithBridgeTest, ScalarQuadMasksCoverUpperWordAndCommitScc) {
  struct Case {
    uint16_t rdna3_op, rdna4_op;
    bool wide;
    uint64_t expected;
  };
  const Case cases[] = {{rdna3::kSQuadmaskB32Sop1, rdna4::kSQuadmaskB32Sop1, false, 5},
                        {rdna3::kSQuadmaskB64Sop1, rdna4::kSQuadmaskB64Sop1, true, 0x8005},
                        {rdna3::kSWqmB32Sop1, rdna4::kSWqmB32Sop1, false, 0xf0f},
                        {rdna3::kSWqmB64Sop1, rdna4::kSWqmB64Sop1, true, 0xf000000000000f0fULL}};
  for (bool rdna3 : {false, true})
    for (const auto &test : cases)
      for (bool nonzero : {false, true}) {
        BridgeFixture fixture(rdna3);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        fixture.wave->set_exec(0);
        fixture.wave->write_scc(!nonzero);
        const auto base = fixture.wave->sgpr_alloc().base;
        fixture.cu->write_sgpr(base, nonzero ? 0x101 : 0);
        fixture.cu->write_sgpr(base + 1, nonzero ? 0x80000000 : 0);
        const auto words = rdna3 ? rdna3::build_sop1(test.rdna3_op, {.ssrc0 = 0, .sdst = 0})
                                 : rdna4::build_sop1(test.rdna4_op, {.ssrc0 = 0, .sdst = 0});
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        const uint64_t expected = nonzero ? test.expected : 0;
        EXPECT_EQ(fixture.cu->read_sgpr(base), uint32_t(expected));
        EXPECT_EQ(fixture.cu->read_sgpr(base + 1),
                  test.wide ? uint32_t(expected >> 32) : (nonzero ? 0x80000000u : 0));
        EXPECT_EQ(fixture.wave->read_scc(), nonzero);
      }
}

TEST(ArithBridgeTest, ScalarMultiply64ReadsFullPairsBeforeAliasedWrite) {
  for (bool scc : {false, true}) {
    BridgeFixture fixture(false);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_exec(0);
    fixture.wave->write_scc(scc);
    const auto base = fixture.wave->sgpr_alloc().base;
    fixture.cu->write_sgpr(base, 3);
    fixture.cu->write_sgpr(base + 1, 2);
    fixture.cu->write_sgpr(base + 2, 5);
    fixture.cu->write_sgpr(base + 3, 7);
    fixture.recorder->scalar_reads.clear();
    const auto words = rdna4::build_sop2(rdna4::kSMulU64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0});
    rdna4::SMulU64Sop2 instruction(words.data());
    instruction.execute_impl(*fixture.wave);
    ASSERT_FALSE(fixture.wave->instruction_execution_failed());
    EXPECT_EQ(fixture.recorder->scalar_reads,
              (std::vector<uint32_t>{base, base + 1, base + 2, base + 3}));
    EXPECT_EQ(fixture.cu->read_sgpr(base), 15u);
    EXPECT_EQ(fixture.cu->read_sgpr(base + 1), 31u);
    EXPECT_EQ(fixture.wave->read_scc(), scc);
  }
}

TEST(ArithBridgeTest, ScalarCarryInputIsCapturedBeforeOutput) {
  for (bool rdna3 : {false, true}) {
    for (bool carry : {false, true}) {
      BridgeFixture fixture(rdna3);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_exec(0);
      fixture.wave->write_scc(carry);
      const auto base = fixture.wave->sgpr_alloc().base;
      fixture.cu->write_sgpr(base, 0xffffffffu);
      fixture.cu->write_sgpr(base + 1, 0u);
      if (rdna3) {
        const auto words = rocjitsu::rdna3::build_sop2(rocjitsu::rdna3::kSAddcU32Sop2,
                                                       {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0});
        rocjitsu::rdna3::SAddcU32Sop2 instruction(words.data());
        instruction.execute_impl(*fixture.wave);
      } else {
        const auto words =
            rdna4::build_sop2(rdna4::kSAddCoCiU32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0});
        rdna4::SAddCoCiU32Sop2 instruction(words.data());
        instruction.execute_impl(*fixture.wave);
      }
      ASSERT_FALSE(fixture.wave->instruction_execution_failed());
      EXPECT_EQ(fixture.cu->read_sgpr(base), carry ? 0u : 0xffffffffu);
      EXPECT_EQ(fixture.wave->read_scc(), carry);
    }
  }
}

TEST(ArithBridgeTest, ScalarImmediateAddUsesSignedOverflow) {
  struct Case {
    uint32_t input;
    uint16_t immediate;
    uint32_t result;
    bool overflow;
  };
  constexpr Case cases[] = {{0x7fffffffu, 1, 0x80000000u, true},
                            {0x80000000u, 0xffff, 0x7fffffffu, true},
                            {0xffffffffu, 1, 0, false},
                            {0, 0xffff, 0xffffffffu, false}};
  for (bool rdna3 : {false, true}) {
    for (const auto &test : cases) {
      BridgeFixture fixture(rdna3);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_exec(0);
      const auto base = fixture.wave->sgpr_alloc().base;
      fixture.cu->write_sgpr(base, test.input);
      if (rdna3) {
        const auto words = rocjitsu::rdna3::build_sopk(rocjitsu::rdna3::kSAddkI32Sopk,
                                                       {.simm16 = test.immediate, .sdst = 0});
        rocjitsu::rdna3::SAddkI32Sopk instruction(words.data());
        instruction.execute_impl(*fixture.wave);
      } else {
        const auto words =
            rdna4::build_sopk(rdna4::kSAddkCoI32Sopk, {.simm16 = test.immediate, .sdst = 0});
        rdna4::SAddkCoI32Sopk instruction(words.data());
        instruction.execute_impl(*fixture.wave);
      }
      ASSERT_FALSE(fixture.wave->instruction_execution_failed());
      EXPECT_EQ(fixture.cu->read_sgpr(base), test.result);
      EXPECT_EQ(fixture.wave->read_scc(), test.overflow);
    }
  }
}

TEST(ArithBridgeTest, ScalarImmediateCompareReadsButDoesNotWriteSdst) {
  BridgeFixture fixture(true);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.wave->set_exec(0);
  const auto base = fixture.wave->sgpr_alloc().base;
  fixture.cu->write_sgpr(base, 0xffffffffu);
  fixture.recorder->scalar_reads.clear();
  fixture.recorder->scalar_writes.clear();
  const auto words =
      rocjitsu::rdna3::build_sopk(rocjitsu::rdna3::kSCmpkEqI32Sopk, {.simm16 = 0xffff, .sdst = 0});
  rocjitsu::rdna3::SCmpkEqI32Sopk instruction(words.data());
  instruction.execute_impl(*fixture.wave);
  ASSERT_FALSE(fixture.wave->instruction_execution_failed());
  EXPECT_EQ(fixture.recorder->scalar_reads, (std::vector<uint32_t>{base}));
  EXPECT_TRUE(fixture.recorder->scalar_writes.empty());
  EXPECT_TRUE(fixture.wave->read_scc());
  EXPECT_EQ(fixture.cu->read_sgpr(base), 0xffffffffu);
}

TEST(ArithBridgeTest, ScalarFloatBinaryOperationsHonorSemanticsAndFailureBoundaries) {
  struct Case {
    uint16_t opcode;
    uint32_t a, b, expected;
  };
  const Case cases[] = {{rdna4::kSAddF32Sop2, 0x40000000, 0x40400000, 0x40a00000},
                        {rdna4::kSSubF32Sop2, 0x40000000, 0x40400000, 0xbf800000},
                        {rdna4::kSMulF32Sop2, 0x40000000, 0x40400000, 0x40c00000},
                        {rdna4::kSMinNumF32Sop2, 0x40000000, 0x40400000, 0x40000000},
                        {rdna4::kSMaxNumF32Sop2, 0x40000000, 0x40400000, 0x40400000},
                        {rdna4::kSMinimumF32Sop2, 0x40000000, 0x40400000, 0x40000000},
                        {rdna4::kSMaximumF32Sop2, 0x40000000, 0x40400000, 0x40400000},
                        {rdna4::kSAddF16Sop2, 0xdead4000, 0xbeef4200, 0x4500},
                        {rdna4::kSSubF16Sop2, 0xdead4000, 0xbeef4200, 0xbc00},
                        {rdna4::kSMulF16Sop2, 0xdead4000, 0xbeef4200, 0x4600},
                        {rdna4::kSMinNumF16Sop2, 0xdead4000, 0xbeef4200, 0x4000},
                        {rdna4::kSMaxNumF16Sop2, 0xdead4000, 0xbeef4200, 0x4200},
                        {rdna4::kSMinimumF16Sop2, 0xdead4000, 0xbeef4200, 0x4000},
                        {rdna4::kSMaximumF16Sop2, 0xdead4000, 0xbeef4200, 0x4200}};
  for (const auto &test : cases)
    for (auto semantics :
         {ArithmeticSemantics::Loose, ArithmeticSemantics::BitExactEmpiricalAllowLoose,
          ArithmeticSemantics::BitExactEmpiricalNoFpStatus, ArithmeticSemantics::BitExactEmpirical,
          ArithmeticSemantics::Compatibility}) {
      ArithmeticPolicy policy{semantics};
      BridgeFixture fixture(false, 32, policy);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_exec(0);
      fixture.wave->set_mode_raw(0xf0);
      fixture.wave->write_scc(true);
      fixture.wave->set_trapsts(0x80000010u);
      const auto base = fixture.wave->sgpr_alloc().base;
      fixture.cu->write_sgpr(base, test.a);
      fixture.cu->write_sgpr(base + 1, test.b);
      fixture.recorder->scalar_reads.clear();
      fixture.recorder->scalar_writes.clear();
      const auto words = rdna4::build_sop2(test.opcode, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      // Every form supports exact semantics but not FP status.
      const bool supported = semantics != ArithmeticSemantics::BitExactEmpirical;
      EXPECT_EQ(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded(),
          supported);
      EXPECT_EQ(fixture.recorder->scalar_reads, (std::vector<uint32_t>{base, base + 1}));
      EXPECT_EQ(fixture.recorder->scalar_writes.size(), supported ? 1u : 0u);
      EXPECT_EQ(fixture.cu->read_sgpr(base), supported ? test.expected : test.a);
      EXPECT_TRUE(fixture.wave->read_scc());
      EXPECT_EQ(fixture.wave->trapsts(), 0x80000010u);
    }
}

TEST(ArithBridgeTest, ScalarConversionsPreserveOperandFormatsAndRejectedOutputs) {
  struct Case {
    uint16_t opcode;
    uint32_t a, b, expected;
    bool packed;
  };
  const Case cases[] = {{rdna4::kSCvtF32I32Sop1, 0xfffffffd, 0, 0xc0400000, false},
                        {rdna4::kSCvtF32U32Sop1, 3, 0, 0x40400000, false},
                        {rdna4::kSCvtI32F32Sop1, 0xc0700000, 0, 0xfffffffd, false},
                        {rdna4::kSCvtU32F32Sop1, 0x40700000, 0, 3, false},
                        {rdna4::kSCvtF16F32Sop1, 0x3fc00000, 0, 0x3e00, false},
                        {rdna4::kSCvtF32F16Sop1, 0xdead3e00, 0, 0x3fc00000, false},
                        {rdna4::kSCvtHiF32F16Sop1, 0x3e00beef, 0, 0x3fc00000, false},
                        {rdna4::kSCvtPkRtzF16F32Sop2, 0x3fc00000, 0xc0200000, 0xc1003e00, true}};
  for (const auto &test : cases)
    for (auto semantics :
         {ArithmeticSemantics::Loose, ArithmeticSemantics::BitExactEmpiricalAllowLoose,
          ArithmeticSemantics::BitExactEmpiricalNoFpStatus, ArithmeticSemantics::BitExactEmpirical,
          ArithmeticSemantics::Compatibility}) {
      ArithmeticPolicy policy{semantics};
      // Every conversion supports exact semantics but not FP status.
      const bool reporting = semantics == ArithmeticSemantics::BitExactEmpirical;
      BridgeFixture fixture(false, 64, policy);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_exec(0);
      fixture.wave->set_mode_raw(0xf0);
      fixture.wave->write_scc(true);
      fixture.wave->set_trapsts(0x80000010u);
      const auto base = fixture.wave->sgpr_alloc().base;
      fixture.cu->write_sgpr(base, test.a);
      fixture.cu->write_sgpr(base + 1, test.b);
      fixture.recorder->scalar_writes.clear();
      const auto words = test.packed
                             ? rdna4::build_sop2(test.opcode, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0})
                             : rdna4::build_sop1(test.opcode, {.ssrc0 = 0, .sdst = 0});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      EXPECT_EQ(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded(),
          !reporting);
      EXPECT_EQ(fixture.recorder->scalar_writes.size(), reporting ? 0u : 1u);
      EXPECT_EQ(fixture.cu->read_sgpr(base), reporting ? test.a : test.expected);
      EXPECT_TRUE(fixture.wave->read_scc());
      EXPECT_EQ(fixture.wave->trapsts(), 0x80000010u);
    }
}

TEST(ArithBridgeTest, ScalarConversionUsesDifferentInputAndOutputDenormFields) {
  for (bool to_half : {false, true})
    for (bool preserve : {false, true}) {
      BridgeFixture fixture(false);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_exec(0);
      // Widening consumes FP16 input mode, narrowing consumes FP16 output mode.
      // FP32's corresponding policy is deliberately opposite.
      const unsigned half_bit = to_half ? 7 : 6;
      const unsigned float_bit = to_half ? 5 : 4;
      fixture.wave->set_mode_raw(preserve ? 1u << half_bit : 1u << float_bit);
      const auto base = fixture.wave->sgpr_alloc().base;
      fixture.cu->write_sgpr(base, to_half ? 0x33800000u : 1u);
      const auto words = rdna4::build_sop1(
          to_half ? rdna4::kSCvtF16F32Sop1 : rdna4::kSCvtF32F16Sop1, {.ssrc0 = 0, .sdst = 0});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      EXPECT_EQ(fixture.cu->read_sgpr(base), preserve ? (to_half ? 1u : 0x33800000u) : 0u);
    }
}

TEST(ArithBridgeTest, ScalarFmaFormsPreserveAccumulatorLiteralOrderAndGuestRounding) {
  const uint16_t opcodes[] = {rdna4::kSFmacF32Sop2, rdna4::kSFmacF16Sop2, rdna4::kSFmaakF32Sop2,
                              rdna4::kSFmamkF32Sop2};
  for (unsigned op = 0; op < 4; ++op)
    for (bool loose : {false, true})
      for (bool reporting : {false, true}) {
        ArithmeticPolicy policy{};
        policy.semantics =
            loose ? ArithmeticSemantics::Loose : ArithmeticSemantics::BitExactEmpiricalAllowLoose;
        if (reporting && !loose)
          policy.semantics = ArithmeticSemantics::BitExactEmpirical;
        BridgeFixture fixture(false, 32, policy);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        fixture.wave->set_exec(0);
        fixture.wave->set_mode_raw(0xf0);
        fixture.wave->write_scc(true);
        fixture.wave->set_trapsts(0x80000010u);
        const auto base = fixture.wave->sgpr_alloc().base;
        fixture.cu->write_sgpr(base, op == 1 ? 0xdead4000u : 0x40000000u);
        fixture.cu->write_sgpr(base + 1, op == 1 ? 0xbeef4200u : 0x40400000u);
        fixture.cu->write_sgpr(base + 2, op == 1 ? 0xabcd4400u : 0x40800000u);
        fixture.recorder->scalar_writes.clear();
        const uint32_t words[] = {
            rdna4::build_sop2(opcodes[op], {.ssrc0 = 0, .ssrc1 = 1, .sdst = 2})[0], 0x40800000u};
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        const bool supported = loose || !reporting;
        EXPECT_EQ(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded(),
            supported);
        const uint32_t initial = op == 1 ? 0xabcd4400u : 0x40800000u;
        const uint32_t expected = op == 1 ? 0x4900u : op == 3 ? 0x41300000u : 0x41200000u;
        EXPECT_EQ(fixture.cu->read_sgpr(base + 2), supported ? expected : initial);
        EXPECT_EQ(fixture.recorder->scalar_writes.size(), supported ? 1u : 0u);
        EXPECT_TRUE(fixture.wave->read_scc());
        EXPECT_EQ(fixture.wave->trapsts(), 0x80000010u);
      }
  for (unsigned rounding = 0; rounding < 4; ++rounding) {
    BridgeFixture fixture(false);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_exec(0);
    fixture.wave->set_mode_raw(0xf0 | rounding);
    const auto base = fixture.wave->sgpr_alloc().base;
    fixture.cu->write_sgpr(base, 0x3f800000);
    fixture.cu->write_sgpr(base + 1, 0x3f800000);
    fixture.cu->write_sgpr(base + 2, 0x33800000);
    const auto words = rdna4::build_sop2(rdna4::kSFmacF32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 2});
    auto instruction = fixture.decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    ASSERT_TRUE(
        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
    EXPECT_EQ(fixture.cu->read_sgpr(base + 2), rounding == 1 ? 0x3f800001u : 0x3f800000u);
  }
}

// gfx1201: FP32 results round by MODE[1:0], FP16 results (S_FMAC_F16 and S_CVT_F16_F32 too)
// by MODE[3:2], in decoded Compatibility and through rj-arith. Expected values are captures;
// S_SUB negates a NaN B.
TEST(ArithBridgeTest, ScalarFloatArithmeticUsesDestinationRoundingField) {
  struct Case {
    uint16_t opcode;
    bool sop1;
    uint32_t mode, a, b, d, expected;
  };
  // d is the S_FMAC_F16 accumulator; other forms overwrite it.
  const Case cases[] = {{rdna4::kSAddF32Sop2, false, 0xf1, 0x3f800000, 0x1, 0, 0x3f800001},
                        {rdna4::kSAddF32Sop2, false, 0xf4, 0x3f800000, 0x1, 0, 0x3f800000},
                        {rdna4::kSSubF16Sop2, false, 0xfc, 0x3c00, 0x1, 0, 0x3bff},
                        {rdna4::kSSubF16Sop2, false, 0xf3, 0x3c00, 0x1, 0, 0x3c00},
                        {rdna4::kSSubF32Sop2, false, 0xf0, 0x0, 0x7fc00000, 0, 0xffc00000},
                        {rdna4::kSFmacF16Sop2, false, 0xf4, 0x3c00, 0x3c00, 0x0c00, 0x3c01},
                        {rdna4::kSFmacF16Sop2, false, 0xf0, 0x3c00, 0x3c00, 0x0c00, 0x3c00},
                        {rdna4::kSCvtF32I32Sop1, true, 0xf1, 0x80000001, 0, 0, 0xceffffff},
                        {rdna4::kSCvtF16F32Sop1, true, 0xf4, 0x40490fdb, 0, 0, 0x4249},
                        {rdna4::kSCvtF16F32Sop1, true, 0xf1, 0x40490fdb, 0, 0, 0x4248}};
  for (const auto &test : cases)
    for (auto semantics :
         {ArithmeticSemantics::Compatibility, ArithmeticSemantics::BitExactEmpiricalAllowLoose,
          ArithmeticSemantics::BitExactEmpiricalNoFpStatus}) {
      ArithmeticPolicy policy{semantics};
      BridgeFixture fixture(false, 32, policy);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_exec(0);
      fixture.wave->set_mode_raw(test.mode);
      fixture.wave->set_trapsts(0x80000000u);
      const auto base = fixture.wave->sgpr_alloc().base;
      fixture.cu->write_sgpr(base, test.a);
      fixture.cu->write_sgpr(base + 1, test.b);
      fixture.cu->write_sgpr(base + 2, test.d);
      const auto words = test.sop1
                             ? rdna4::build_sop1(test.opcode, {.ssrc0 = 0, .sdst = 2})
                             : rdna4::build_sop2(test.opcode, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 2});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      EXPECT_EQ(fixture.cu->read_sgpr(base + 2), test.expected)
          << std::hex << test.opcode << " mode 0x" << test.mode;
      // Only Compatibility (decoded RDNA4 scalar path) records FP causes.
      if (semantics != ArithmeticSemantics::Compatibility) {
        EXPECT_EQ(fixture.wave->trapsts(), 0x80000000u);
      }
    }
}

TEST(ArithBridgeTest, ScalarFloatComparisonsCoverEveryPredicateAndReportingPolicy) {
  const uint16_t opcodes[2][14] = {
      {rdna4::kSCmpLtF32Sopc, rdna4::kSCmpEqF32Sopc, rdna4::kSCmpLeF32Sopc, rdna4::kSCmpGtF32Sopc,
       rdna4::kSCmpLgF32Sopc, rdna4::kSCmpGeF32Sopc, rdna4::kSCmpOF32Sopc, rdna4::kSCmpUF32Sopc,
       rdna4::kSCmpNgeF32Sopc, rdna4::kSCmpNlgF32Sopc, rdna4::kSCmpNgtF32Sopc,
       rdna4::kSCmpNleF32Sopc, rdna4::kSCmpNeqF32Sopc, rdna4::kSCmpNltF32Sopc},
      {rdna4::kSCmpLtF16Sopc, rdna4::kSCmpEqF16Sopc, rdna4::kSCmpLeF16Sopc, rdna4::kSCmpGtF16Sopc,
       rdna4::kSCmpLgF16Sopc, rdna4::kSCmpGeF16Sopc, rdna4::kSCmpOF16Sopc, rdna4::kSCmpUF16Sopc,
       rdna4::kSCmpNgeF16Sopc, rdna4::kSCmpNlgF16Sopc, rdna4::kSCmpNgtF16Sopc,
       rdna4::kSCmpNleF16Sopc, rdna4::kSCmpNeqF16Sopc, rdna4::kSCmpNltF16Sopc}};
  // Raw IEEE witnesses; relation bits are the ISA truth table's
  // LT/EQ/GT/UNORDERED.
  struct Sample {
    uint32_t a32, b32, a16, b16, relation, causes;
    bool denormal;
  };
  const Sample samples[] = {{0xbf800000, 0x40000000, 0xbc00, 0x4000, 1, 0, false},
                            {0x80000000, 0, 0x8000, 0, 2, 0, false},
                            {0x7f800000, 0x3f800000, 0x7c00, 0x3c00, 4, 0, false},
                            {0x7fc01234, 0, 0x7e34, 0, 8, 0, false},
                            {0x7f801234, 0, 0x7c34, 0, 8, 1, false},
                            {1, 0, 1, 0, 4, 2, true},
                            {0x80000001, 0, 0x8001, 0, 1, 2, true}};
  for (bool half : {false, true})
    for (unsigned predicate = 1; predicate < 15; ++predicate)
      for (bool loose : {false, true})
        for (bool reporting : {false, true}) {
          ArithmeticPolicy policy{};
          policy.semantics =
              loose ? ArithmeticSemantics::Loose : ArithmeticSemantics::BitExactEmpiricalNoFpStatus;

          if (reporting && !loose)
            policy.semantics = ArithmeticSemantics::BitExactEmpirical;
          BridgeFixture fixture(false, 32, policy);
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
          fixture.wave->set_exec(0);
          const auto base = fixture.wave->sgpr_alloc().base;
          const auto words =
              rdna4::build_sopc(opcodes[half][predicate - 1], {.ssrc0 = 0, .ssrc1 = 1});
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          for (bool flush : {false, true})
            for (const auto &sample : samples) {
              SCOPED_TRACE(testing::Message() << "half=" << half << " predicate=" << predicate
                                              << " loose=" << loose << " flush=" << flush);
              // Opposite policies for FP16 and FP32 detect use of the wrong
              // MODE field.
              fixture.wave->set_mode_raw((flush ? 0u : 1u) << (half ? 6 : 4) |
                                         (flush ? 1u : 0u) << (half ? 4 : 6));
              fixture.wave->set_trapsts(0x80000010u);
              fixture.wave->write_scc(true);
              fixture.cu->write_sgpr(base, half ? 0xdead0000u | sample.a16 : sample.a32);
              fixture.cu->write_sgpr(base + 1, half ? 0xbeef0000u | sample.b16 : sample.b32);
              fixture.recorder->scalar_reads.clear();
              fixture.recorder->scalar_writes.clear();
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              EXPECT_EQ(fixture.recorder->scalar_reads, (std::vector<uint32_t>{base, base + 1}));
              EXPECT_TRUE(fixture.recorder->scalar_writes.empty());
              const uint32_t relation = flush && sample.denormal ? 2 : sample.relation;
              EXPECT_EQ(fixture.wave->read_scc(), (predicate & relation) != 0);
              const uint32_t causes =
                  !loose && reporting && !(flush && sample.denormal) ? sample.causes : 0;
              EXPECT_EQ(fixture.wave->trapsts(), 0x80000010u | causes);
              EXPECT_EQ(fixture.wave->exec(), 0u);
            }
        }
}

TEST(ArithBridgeTest, ScalarFloatRoundingUsesPrecisionModeAndPreservesScc) {
  // Goldens for -1.5: ceil=-1, floor=-2, trunc=-1, nearest-even=-2.
  const uint16_t opcodes[2][4] = {
      {rdna4::kSCeilF32Sop1, rdna4::kSFloorF32Sop1, rdna4::kSTruncF32Sop1, rdna4::kSRndneF32Sop1},
      {rdna4::kSCeilF16Sop1, rdna4::kSFloorF16Sop1, rdna4::kSTruncF16Sop1, rdna4::kSRndneF16Sop1}};
  for (bool half : {false, true})
    for (unsigned op = 0; op < 4; ++op)
      for (bool loose : {false, true})
        for (bool reporting : {false, true}) {
          ArithmeticPolicy policy{};
          policy.semantics =
              loose ? ArithmeticSemantics::Loose : ArithmeticSemantics::BitExactEmpiricalNoFpStatus;

          if (reporting && !loose)
            policy.semantics = ArithmeticSemantics::BitExactEmpirical;
          BridgeFixture fixture(false, 64, policy);
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
          fixture.wave->set_exec(0);
          const auto base = fixture.wave->sgpr_alloc().base;
          const auto words = rdna4::build_sop1(opcodes[half][op], {.ssrc0 = 0, .sdst = 0});
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          for (bool flush : {false, true})
            for (unsigned sample = 0; sample < 3; ++sample) {
              SCOPED_TRACE(testing::Message()
                           << "half=" << half << " op=" << op << " sample=" << sample);
              uint32_t input, expected, causes;
              if (sample == 0) {
                input = half ? 0xbe00 : 0xbfc00000;
                expected = op == 0 || op == 2 ? (half ? 0xbc00 : 0xbf800000)
                                              : (half ? 0xc000 : 0xc0000000);
                causes = 0;
              } else if (sample == 1) {
                input = half ? 0x8001 : 0x80000001;
                expected =
                    !flush && op == 1 ? (half ? 0xbc00 : 0xbf800000) : (half ? 0x8000 : 0x80000000);
                causes = flush ? 0 : 2;
              } else {
                input = half ? 0x7c34 : 0x7f801234;
                expected = half ? 0x7e34 : 0x7fc01234;
                causes = 1;
              }
              fixture.wave->set_mode_raw((flush ? 0u : 1u) << (half ? 6 : 4) |
                                         (flush ? 1u : 0u) << (half ? 4 : 6));
              fixture.wave->write_scc(sample != 0);
              fixture.wave->set_trapsts(0x80000010u);
              fixture.cu->write_sgpr(base, half ? 0xdead0000u | input : input);
              fixture.recorder->scalar_reads.clear();
              fixture.recorder->scalar_writes.clear();
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              EXPECT_EQ(fixture.recorder->scalar_reads, (std::vector<uint32_t>{base}));
              ASSERT_EQ(fixture.recorder->scalar_writes.size(), 1u);
              EXPECT_EQ(fixture.recorder->scalar_writes[0].width, 1u);
              EXPECT_EQ(fixture.cu->read_sgpr(base), expected);
              EXPECT_EQ(fixture.wave->read_scc(), sample != 0);
              EXPECT_EQ(fixture.wave->trapsts(), 0x80000010u | (!loose && reporting ? causes : 0));
            }
        }
}

TEST(ArithBridgeTest, ScalarCompareObservesBothRegisterPairsWithoutWritingThem) {
  for (bool rdna3 : {false, true}) {
    BridgeFixture fixture(rdna3);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_exec(0);
    const auto base = fixture.wave->sgpr_alloc().base;
    fixture.cu->write_sgpr(base, 0x1234u);
    fixture.cu->write_sgpr(base + 1, 1u);
    fixture.cu->write_sgpr(base + 2, 0x1234u);
    fixture.cu->write_sgpr(base + 3, 2u);
    fixture.recorder->scalar_reads.clear();
    fixture.recorder->scalar_writes.clear();
    if (rdna3) {
      const auto words =
          rocjitsu::rdna3::build_sopc(rocjitsu::rdna3::kSCmpEqU64Sopc, {.ssrc0 = 0, .ssrc1 = 2});
      rocjitsu::rdna3::SCmpEqU64Sopc instruction(words.data());
      instruction.execute_impl(*fixture.wave);
    } else {
      const auto words = rdna4::build_sopc(rdna4::kSCmpEqU64Sopc, {.ssrc0 = 0, .ssrc1 = 2});
      rdna4::SCmpEqU64Sopc instruction(words.data());
      instruction.execute_impl(*fixture.wave);
    }
    ASSERT_FALSE(fixture.wave->instruction_execution_failed());
    EXPECT_EQ(fixture.recorder->scalar_reads,
              (std::vector<uint32_t>{base, base + 1, base + 2, base + 3}));
    EXPECT_TRUE(fixture.recorder->scalar_writes.empty());
    EXPECT_FALSE(fixture.wave->read_scc());
  }
}

TEST(ArithBridgeTest, MatrixIgnoresExecAndPreservesAliasedInputsUntilCommit) {
  BridgeFixture fixture(false);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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

TEST(ArithBridgeTest, CuOnlyAccessorDeclinesBulkMatrixWrites) {
  BridgeFixture fixture(false);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.cu->set_plugin_group(nullptr);
  RegisterAccess registers(*fixture.cu);
  bool invoked = false;
  EXPECT_FALSE(registers
                   .try_bulk_write_vgprs<1>(fixture.base(), 1,
                                            [&](uint32_t *const *) {
                                              invoked = true;
                                              return RJ_ARITH_SUCCESS;
                                            })
                   .has_value());
  EXPECT_FALSE(invoked);
}

TEST(ArithBridgeTest, OutOfRangeMatrixSourceTupleReadsV0) {
  for (bool rdna3 : {false, true}) {
    for (uint32_t vgprs : {32U, 256U}) {
      for (uint32_t a : {0U, vgprs - 2}) {
        BridgeFixture fixture(rdna3, 32, {}, vgprs);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        const uint32_t words[] = {0xcc404018U, 0x04421100U | a, 0, 0};
        auto decoded = fixture.decoder->decode(words);
        ASSERT_TRUE(decoded.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(decoded.value().get(), *fixture.wave).succeeded());
        for (uint32_t reg = 0; reg < 8; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 24 + reg, lane), 0x41800000U);
        const uint32_t count = rdna3 ? 8 : 4;
        ASSERT_GE(fixture.recorder->reads.size(), count);
        for (uint32_t reg = 0; reg < count; ++reg) {
          EXPECT_EQ(fixture.recorder->reads[reg].reg, fixture.base() + reg);
          EXPECT_EQ(fixture.recorder->reads[reg].lanes, 0xffffffffULL);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, OutOfRangeMatrixDestinationSuppressesReadsAndWrites) {
  for (bool sparse : {false, true}) {
    for (uint32_t vgprs : {32U, 256U}) {
      BridgeFixture fixture(false, 32, {}, vgprs);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      const uint32_t destination = vgprs - 4;
      for (uint32_t reg = 0; reg < 4; ++reg)
        fixture.cu->write_vgpr(fixture.base() + destination + reg, 0, 0xdeadbeef);
      const uint32_t words[] = {(sparse ? 0xcc504000U : 0xcc404000U) | destination,
                                sparse ? 0x04321100U : (0x00021100U | ((256 + destination) << 18)),
                                0, 0};
      auto decoded = fixture.decoder->decode(words);
      ASSERT_TRUE(decoded.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(decoded.value().get(), *fixture.wave).succeeded());
      EXPECT_TRUE(fixture.recorder->reads.empty());
      EXPECT_TRUE(fixture.recorder->writes.empty());
      for (uint32_t reg = 0; reg < 4; ++reg)
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination + reg, 0), 0xdeadbeefU);
    }
  }
}

TEST(ArithBridgeTest, FailedRegisterReadPreventsMatrixCommit) {
  BridgeFixture fixture(false);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  RegisterAccess registers(*fixture.wave);
  auto source = registers.read_vgpr_region(fixture.base(), 1, ~0ULL, 0x3);
  const uint32_t *data[1];
  EXPECT_FALSE(arith::read_matrix_pointers(source, data));
  ASSERT_TRUE(fixture.wave->instruction_execution_failed());
  bool invoked = false;
  arith::commit_matrix_result(*fixture.wave, registers, fixture.base() + 16, 1, 0xf,
                              [&](uint32_t *const *) {
                                invoked = true;
                                return RJ_ARITH_SUCCESS;
                              });
  EXPECT_FALSE(invoked);
  EXPECT_TRUE(fixture.recorder->writes.empty());
}

TEST(ArithBridgeTest, FailedInvocationDoesNotReportOrCommitWrites) {
  BridgeFixture fixture(false);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
  std::array<std::array<uint32_t, 256>, 4> results{};
  for (uint32_t sample = 0; sample < 4; ++sample) {
    const uint32_t selector = sample == 2 ? 3 : sample == 3 ? 1 : sample;
    BridgeFixture fixture(false, 64);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    for (uint32_t lane = 0; lane < 64; ++lane) {
      fixture.cu->write_vgpr(fixture.base(), lane,
                             lane < 32     ? 0x11111111u
                             : sample == 3 ? 0x22222222u
                                           : 0xffffffffu);
      fixture.cu->write_vgpr(fixture.base() + 8, lane, 0x87654321u + lane);
      fixture.cu->write_vgpr(fixture.base() + 12, lane,
                             sample == 3 && lane >= 32 ? 0x4444eeeeu : 0xeeee4444u);
    }
    // V_SWMMAC_I32_16X16X32_IU4: A=v0, B=v8, index=v12, C/D=v16.
    // Only selector bit zero is meaningful; the higher bit aliases the same
    // set.
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
        results[sample][(reg - 16) * 64 + lane] = fixture.cu->read_vgpr(fixture.base() + reg, lane);
  }
  EXPECT_EQ(results[1], results[2]);
  EXPECT_EQ(results[1],
            results[3]); // Unconsumed A/index lanes cannot affect arithmetic.
  EXPECT_NE(results[0], results[1]);
}

TEST(ArithBridgeTest, DecodedMatrixUsesConfiguredSemantics) {
  for (int kind = 0; kind < 3; ++kind) {
    for (auto semantics :
         {ArithmeticSemantics::Loose, ArithmeticSemantics::BitExactEmpiricalNoFpStatus}) {
      const bool sparse = kind == 2;
      BridgeFixture fixture(kind == 0, 32, {semantics});
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      for (uint32_t lane = 0; lane < 32; ++lane) {
        for (uint32_t reg = 16; reg < 24; ++reg)
          fixture.cu->write_vgpr(fixture.base() + reg, lane, 0x4b800000u);
        fixture.cu->write_vgpr(fixture.base() + 32, lane, 0x44444444u);
      }
      // FP16 factors are all one; accumulation at 2^24 distinguishes grouping.
      const uint32_t words[] = {sparse ? 0xcc504010u : 0xcc404018u,
                                sparse ? 0x04821100u : 0x04421100u, 0, 0};
      auto decoded = fixture.decoder->decode(words);
      ASSERT_TRUE(decoded.succeeded());
      auto instruction = std::move(decoded).value();
      ASSERT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wave).succeeded());
      if (sparse) {
        // B is the full dense operand regardless of sparse A metadata.
        for (uint32_t reg = 8; reg < 16; ++reg) {
          uint64_t read_lanes = 0;
          for (const auto &read : fixture.recorder->reads)
            if (read.reg == fixture.base() + reg)
              read_lanes |= read.lanes;
          EXPECT_EQ(read_lanes, 0xffffffffULL);
        }
      }
      for (uint32_t reg = 0; reg < 8; ++reg)
        for (uint32_t lane = 0; lane < 32; ++lane)
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + (sparse ? 16 : 24) + reg, lane),
                    semantics == ArithmeticSemantics::Loose ? 0x4b800000u : 0x4b800008u);
    }
  }
}

TEST(ArithBridgeTest, FailedSparseInvocationPreservesItsTiedAccumulator) {
  BridgeFixture fixture(false);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
              fixture.wave->set_mode_raw(preserve_denorm ? 3u << 6 : 0);
              const uint64_t exec_mask = lanes == 32 ? 0x80000003ULL : 0x8000000380000003ULL;
              fixture.wave->set_exec(exec_mask);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                fixture.cu->write_vgpr(fixture.base(), lane, 0);
                fixture.cu->write_vgpr(fixture.base() + 1, lane,
                                       0x3ff80000); // 1.5
                fixture.cu->write_vgpr(fixture.base() + 2, lane, 0);
                fixture.cu->write_vgpr(fixture.base() + 3, lane,
                                       0x40040000); // 2.5
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

TEST(ArithBridgeTest, IntegerWideningSelectedHalfAndAliases) {
  for (bool rdna3 : {false, true})
    for (unsigned lanes : {32u, 64u})
      for (bool is_signed : {false, true})
        for (unsigned encoding = 0; encoding < 4; ++encoding) {
          ArithmeticPolicy policy{};
          policy.semantics = ArithmeticSemantics::BitExactEmpirical;
          BridgeFixture fixture(rdna3, lanes, policy);
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
          const uint64_t exec_mask = lanes == 32 ? 0x80000007ULL : 0x8000000780000007ULL;
          fixture.wave->set_exec(exec_mask);
          for (unsigned lane = 0; lane < lanes; ++lane)
            fixture.cu->write_vgpr(fixture.base(), lane, 0x80017fffu);
          std::array<uint32_t, 2> words{};
          if (encoding == 1 || encoding == 2) {
            const uint8_t high = encoding == 2;
            words =
                rdna3
                    ? rdna3::build_vop3(is_signed ? rdna3::kVCvtI32I16Vop3 : rdna3::kVCvtU32U16Vop3,
                                        {.vdst = 0,
                                         .abs = 1,
                                         .op_sel = high,
                                         .clamp = 1,
                                         .src0 = 256,
                                         .omod = 3,
                                         .neg = 1})
                    : rdna4::build_vop3(is_signed ? rdna4::kVCvtI32I16Vop3 : rdna4::kVCvtU32U16Vop3,
                                        {.vdst = 0,
                                         .abs = 1,
                                         .opsel = high,
                                         .clamp = 1,
                                         .src0 = 256,
                                         .omod = 3,
                                         .neg = 1});
          } else {
            words[0] =
                rdna3
                    ? rdna3::build_vop1(is_signed ? rdna3::kVCvtI32I16Vop1 : rdna3::kVCvtU32U16Vop1,
                                        {.src0 = uint16_t(encoding == 3 ? 384 : 256), .vdst = 0})[0]
                    : rdna4::build_vop1(
                          is_signed ? rdna4::kVCvtI32I16Vop1 : rdna4::kVCvtU32U16Vop1,
                          {.src0 = uint16_t(encoding == 3 ? 384 : 256), .vdst = 0})[0];
          }
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          fixture.recorder->writes.clear();
          fixture.recorder->reads.clear();
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          EXPECT_EQ(fixture.recorder->reads.size(), exec_mask ? 1U : 0U);
          uint64_t read_lanes = 0;
          for (const auto &read : fixture.recorder->reads) {
            EXPECT_EQ(read.reg, 0u);
            EXPECT_EQ(read.bytes, encoding >= 2 ? 0xc : 0x3);
            read_lanes |= read.lanes;
          }
          EXPECT_EQ(read_lanes, exec_mask);
          const uint32_t want = encoding >= 2 ? (is_signed ? 0xffff8001u : 0x8001u) : 0x7fffu;
          for (unsigned lane = 0; lane < lanes; ++lane)
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                      ((exec_mask >> lane) & 1) ? want : 0x80017fffu);
          uint64_t written = 0;
          for (const auto &write : fixture.recorder->writes) {
            EXPECT_EQ(write.reg, 0u);
            EXPECT_EQ(write.bytes, 0xf);
            written |= write.lanes;
          }
          EXPECT_EQ(written, exec_mask);
        }
}

TEST(ArithBridgeTest, PackedSaturationPreservesOtherHalf) {
  const uint32_t inputs[] = {0xffff012c, 0x007f0001, 0x01008000, 0x00ff00ff};
  const uint32_t packed[] = {0x00ff, 0x7f01, 0xff00, 0xffff};
  for (bool rdna3 : {false, true})
    for (unsigned lanes : {32u, 64u})
      for (bool vop3 : {false, true})
        for (bool high : {false, true})
          for (bool alias : {false, true}) {
            ArithmeticPolicy policy{};
            policy.semantics = ArithmeticSemantics::BitExactEmpirical;
            BridgeFixture fixture(rdna3, lanes, policy);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            const unsigned destination = alias ? 0 : 1;
            const uint64_t exec_mask = lanes == 32 ? 0x8000000fULL : 0x8000000f8000000fULL;
            fixture.wave->set_exec(exec_mask);
            for (unsigned lane = 0; lane < lanes; ++lane) {
              fixture.cu->write_vgpr(fixture.base(), lane, inputs[lane % 4]);
              fixture.cu->write_vgpr(fixture.base() + 1, lane, 0xdeadbeef);
            }
            std::array<uint32_t, 2> words{};
            if (vop3) {
              words =
                  rdna3
                      ? rdna3::build_vop3(rdna3::kVSatPkU8I16Vop3, {.vdst = uint8_t(destination),
                                                                    .abs = 1,
                                                                    .op_sel = uint8_t(high ? 8 : 0),
                                                                    .clamp = 1,
                                                                    .src0 = 256,
                                                                    .omod = 3,
                                                                    .neg = 1})
                      : rdna4::build_vop3(rdna4::kVSatPkU8I16Vop3, {.vdst = uint8_t(destination),
                                                                    .abs = 1,
                                                                    .opsel = uint8_t(high ? 8 : 0),
                                                                    .clamp = 1,
                                                                    .src0 = 256,
                                                                    .omod = 3,
                                                                    .neg = 1});
            } else {
              const uint8_t encoded_destination = uint8_t(destination + (high ? 128 : 0));
              words[0] = rdna3 ? rdna3::build_vop1(rdna3::kVSatPkU8I16Vop1,
                                                   {.src0 = 256, .vdst = encoded_destination})[0]
                               : rdna4::build_vop1(rdna4::kVSatPkU8I16Vop1,
                                                   {.src0 = 256, .vdst = encoded_destination})[0];
            }
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            fixture.recorder->writes.clear();
            ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .succeeded());
            EXPECT_EQ(fixture.recorder->writes.size(), exec_mask ? 1U : 0U);
            const unsigned shift = high ? 16 : 0;
            for (unsigned lane = 0; lane < lanes; ++lane) {
              uint32_t want = alias ? inputs[lane % 4] : 0xdeadbeef;
              if ((exec_mask >> lane) & 1)
                want = (want & ~(0xffffu << shift)) | (packed[lane % 4] << shift);
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, lane), want);
            }
            uint64_t written = 0;
            for (const auto &write : fixture.recorder->writes) {
              EXPECT_EQ(write.reg, destination);
              EXPECT_EQ(write.bytes, high ? 0xc : 0x3);
              written |= write.lanes;
            }
            EXPECT_EQ(written, exec_mask);
          }
}

TEST(ArithBridgeTest, ByteAndNibbleConversionsModesAndObservedReads) {
  const uint16_t op3[2][5] = {
      {rdna3::kVCvtF32Ubyte0Vop1, rdna3::kVCvtF32Ubyte1Vop1, rdna3::kVCvtF32Ubyte2Vop1,
       rdna3::kVCvtF32Ubyte3Vop1, rdna3::kVCvtOffF32I4Vop1},
      {rdna3::kVCvtF32Ubyte0Vop3, rdna3::kVCvtF32Ubyte1Vop3, rdna3::kVCvtF32Ubyte2Vop3,
       rdna3::kVCvtF32Ubyte3Vop3, rdna3::kVCvtOffF32I4Vop3}};
  const uint16_t op4[2][5] = {
      {rdna4::kVCvtF32Ubyte0Vop1, rdna4::kVCvtF32Ubyte1Vop1, rdna4::kVCvtF32Ubyte2Vop1,
       rdna4::kVCvtF32Ubyte3Vop1, rdna4::kVCvtOffF32I4Vop1},
      {rdna4::kVCvtF32Ubyte0Vop3, rdna4::kVCvtF32Ubyte1Vop3, rdna4::kVCvtF32Ubyte2Vop3,
       rdna4::kVCvtF32Ubyte3Vop3, rdna4::kVCvtOffF32I4Vop3}};
  for (bool rdna3 : {false, true})
    for (unsigned lanes : {32u, 64u})
      for (bool vop3 : {false, true})
        for (unsigned byte = 0; byte < 5; ++byte)
          for (unsigned guest = 0; guest < 4; ++guest)
            for (bool clamp : {false, true}) {
              ArithmeticPolicy policy{};
              policy.semantics = ArithmeticSemantics::BitExactEmpirical;
              BridgeFixture fixture(rdna3, lanes, policy);
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
              fixture.wave->set_mode_raw((guest & 1 ? 0x200 : 0) | (guest & 2 ? 3u << 4 : 0));
              const uint64_t exec_mask = lanes == 32 ? 0x8000ffffULL : 0x8000ffff8000ffffULL;
              fixture.wave->set_exec(exec_mask);
              const uint32_t inputs[] = {0xff800100, 0x0080ff01, 0x0100ff80, 0x800100ff};
              for (unsigned lane = 0; lane < lanes; ++lane)
                fixture.cu->write_vgpr(fixture.base(), lane,
                                       byte == 4 ? 0xcafe1200u | (lane % 16) : inputs[lane % 4]);
              const auto opcode = rdna3 ? op3[vop3][byte] : op4[vop3][byte];
              std::array<uint32_t, 2> words{};
              if (vop3)
                words = rdna3 ? rdna3::build_vop3(opcode, {.vdst = 0,
                                                           .abs = 1,
                                                           .clamp = uint8_t(clamp),
                                                           .src0 = 256,
                                                           .omod = 3,
                                                           .neg = 1})
                              : rdna4::build_vop3(opcode, {.vdst = 0,
                                                           .abs = 1,
                                                           .clamp = uint8_t(clamp),
                                                           .src0 = 256,
                                                           .omod = 3,
                                                           .neg = 1});
              else
                words[0] = rdna3 ? rdna3::build_vop1(opcode, {.src0 = 256, .vdst = 0})[0]
                                 : rdna4::build_vop1(opcode, {.src0 = 256, .vdst = 0})[0];
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              fixture.recorder->reads.clear();
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              uint64_t read_lanes = 0;
              for (const auto &read : fixture.recorder->reads) {
                EXPECT_EQ(read.reg, 0u);
                EXPECT_EQ(read.bytes, 1u << (byte % 4));
                read_lanes |= read.lanes;
              }
              EXPECT_EQ(read_lanes, exec_mask);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                const uint32_t raw = byte == 4 ? 0xcafe1200u | (lane % 16) : inputs[lane % 4];
                float result = byte == 4 ? float(int(raw & 15) - ((raw & 8) ? 16 : 0)) / 16
                                         : float((raw >> (8 * byte)) & 255);
                if (vop3 && (!rdna3 || guest == 0))
                  result *= 0.5f;
                if (vop3 && clamp)
                  result = std::clamp(result, 0.0f, 1.0f);
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                          ((exec_mask >> lane) & 1) ? std::bit_cast<uint32_t>(result) : raw);
              }
            }
}

TEST(ArithBridgeTest, Fp8WidenSelectorsAndPairAliases) {
  // Literal OCP encodings: +0, +1, negative zero/infinity, and NaN.
  const uint32_t inputs[] = {0x7f803800u, 0x7dfc3c00u};
  const uint32_t expected[2][4] = {{0, 0x3f800000, 0x80000000, 0xffc00000},
                                   {0, 0x3f800000, 0xff800000, 0xffc00000}};
  const uint16_t opcodes[2][2][2] = {{{rdna4::kVCvtF32Fp8Vop1, rdna4::kVCvtF32Bf8Vop1},
                                      {rdna4::kVCvtPkF32Fp8Vop1, rdna4::kVCvtPkF32Bf8Vop1}},
                                     {{rdna4::kVCvtF32Fp8Vop3, rdna4::kVCvtF32Bf8Vop3},
                                      {rdna4::kVCvtPkF32Fp8Vop3, rdna4::kVCvtPkF32Bf8Vop3}}};
  for (unsigned lanes : {32u, 64u})
    for (bool vop3 : {false, true})
      for (bool packed : {false, true})
        for (unsigned kind = 0; kind < 2; ++kind)
          for (unsigned byte = 0; byte < (vop3 ? 4u : 1u); byte += packed ? 2 : 1)
            for (unsigned source : {0u, 1u, 4u}) {
              ArithmeticPolicy policy{};
              policy.semantics = ArithmeticSemantics::BitExactEmpirical;
              BridgeFixture fixture(false, lanes, policy);
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
              fixture.wave->set_mode_raw(0); // These conversions ignore guest FP controls.
              const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000580000005ULL;
              fixture.wave->set_exec(exec_mask);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                fixture.cu->write_vgpr(fixture.base(), lane, 0xdeadbeef);
                fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x12345678);
                fixture.cu->write_vgpr(fixture.base() + source, lane, inputs[kind]);
              }
              const unsigned selector = packed ? byte / 2 : ((byte & 1) << 1) | (byte >> 1);
              std::array<uint32_t, 2> words{};
              if (vop3)
                words =
                    rdna4::build_vop3(opcodes[vop3][packed][kind], {.vdst = 0,
                                                                    .abs = 1,
                                                                    .opsel = uint8_t(selector),
                                                                    .clamp = 1,
                                                                    .src0 = uint16_t(256 + source),
                                                                    .omod = 3,
                                                                    .neg = 1});
              else
                words[0] = rdna4::build_vop1(opcodes[vop3][packed][kind],
                                             {.src0 = uint16_t(256 + source), .vdst = 0})[0];
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              fixture.recorder->reads.clear();
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              uint64_t read_lanes = 0;
              for (const auto &read : fixture.recorder->reads) {
                EXPECT_EQ(read.reg, source);
                EXPECT_EQ(read.bytes, (packed ? 3u : 1u) << byte);
                read_lanes |= read.lanes;
              }
              EXPECT_EQ(read_lanes, exec_mask);
              for (unsigned lane = 0; lane < lanes; ++lane)
                for (unsigned word = 0; word < 2; ++word) {
                  uint32_t want = source == word ? inputs[kind] : (word ? 0x12345678 : 0xdeadbeef);
                  if (((exec_mask >> lane) & 1) && (word == 0 || packed))
                    want = expected[kind][byte + word];
                  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + word, lane), want);
                }
            }
}

TEST(ArithBridgeTest, Fp8NarrowPartialWritesModifiersAndSeeds) {
  const uint16_t opcodes[2][2] = {{rdna4::kVCvtPkFp8F32Vop3, rdna4::kVCvtPkBf8F32Vop3},
                                  {rdna4::kVCvtSrFp8F32Vop3, rdna4::kVCvtSrBf8F32Vop3}};
  for (unsigned lanes : {32u, 64u})
    for (bool stochastic : {false, true})
      for (unsigned kind = 0; kind < 2; ++kind)
        for (unsigned byte = 0; byte < 4; byte += stochastic ? 1 : 2)
          for (bool modifiers : {false, true})
            for (bool seed_high : {false, true})
              for (unsigned destination : {0u, 1u, 2u}) {
                ArithmeticPolicy policy{};
                policy.semantics = ArithmeticSemantics::BitExactEmpirical;
                BridgeFixture fixture(false, lanes, policy);
                ASSERT_FALSE(::testing::Test::HasFatalFailure());
                fixture.wave->set_mode_raw(Wavefront::FP16_OVFL_BIT);
                const uint64_t exec_mask = lanes == 32 ? 0x80000005ULL : 0x8000000580000005ULL;
                fixture.wave->set_exec(exec_mask);
                // Midpoint above +1: RNE rounds to even, SR distinguishes seed
                // extremes.
                const uint32_t a = std::bit_cast<uint32_t>(kind ? 1.125f : 1.0625f);
                const uint32_t b = stochastic ? (seed_high ? 0xffffffffu : 0u) : a | 0x80000000u;
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  fixture.cu->write_vgpr(fixture.base(), lane, a);
                  fixture.cu->write_vgpr(fixture.base() + 1, lane, b);
                  fixture.cu->write_vgpr(fixture.base() + 2, lane, 0x12345678);
                }
                const auto words =
                    rdna4::build_vop3(opcodes[stochastic][kind],
                                      {.vdst = uint8_t(destination),
                                       .abs = uint8_t(modifiers ? 3 : 0),
                                       .opsel = uint8_t(byte << 2),
                                       .clamp = 1,
                                       .src0 = 256,
                                       .src1 = 257,
                                       .omod = 3,
                                       .neg = uint8_t(modifiers ? (stochastic ? 3 : 1) : 0)});
                auto instruction = fixture.decoder->decode_window(words);
                ASSERT_TRUE(instruction.succeeded());
                fixture.recorder->reads.clear();
                fixture.recorder->writes.clear();
                ASSERT_TRUE(
                    fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                        .succeeded());
                uint64_t read_lanes[2] = {}, written = 0;
                for (const auto &read : fixture.recorder->reads) {
                  ASSERT_LT(read.reg,
                            2u); // No preservation read of old destination.
                  EXPECT_EQ(read.bytes, 0xf);
                  read_lanes[read.reg] |= read.lanes;
                }
                EXPECT_EQ(read_lanes[0], exec_mask);
                EXPECT_EQ(read_lanes[1], exec_mask);
                for (const auto &write : fixture.recorder->writes) {
                  EXPECT_EQ(write.reg, destination);
                  EXPECT_EQ(write.bytes, (stochastic ? 1u : 3u) << byte);
                  written |= write.lanes;
                }
                EXPECT_EQ(written, exec_mask);
                const uint32_t code = kind ? 0x3c : 0x38;
                const uint32_t low =
                    code + (stochastic && seed_high ? 1 : 0) + (modifiers ? 0x80 : 0);
                const uint32_t high = code + (modifiers ? 0 : 0x80);
                const uint32_t converted = stochastic ? low : low | (high << 8);
                const uint32_t selected = (stochastic ? 0xffu : 0xffffu) << (byte * 8);
                const uint32_t old = destination == 0 ? a : destination == 1 ? b : 0x12345678;
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  const uint32_t want = ((exec_mask >> lane) & 1)
                                            ? (old & ~selected) | (converted << (byte * 8))
                                            : old;
                  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, lane), want);
                }
              }
}

TEST(ArithBridgeTest, DoubleConversionsLayoutsModesAndPolicies) {
  const uint16_t opcodes[2][2][6] = {
      {{rdna3::kVCvtF64I32Vop1, rdna3::kVCvtF64U32Vop1, rdna3::kVCvtI32F64Vop1,
        rdna3::kVCvtU32F64Vop1, rdna3::kVCvtF64F32Vop1, rdna3::kVCvtF32F64Vop1},
       {rdna3::kVCvtF64I32Vop3, rdna3::kVCvtF64U32Vop3, rdna3::kVCvtI32F64Vop3,
        rdna3::kVCvtU32F64Vop3, rdna3::kVCvtF64F32Vop3, rdna3::kVCvtF32F64Vop3}},
      {{rdna4::kVCvtF64I32Vop1, rdna4::kVCvtF64U32Vop1, rdna4::kVCvtI32F64Vop1,
        rdna4::kVCvtU32F64Vop1, rdna4::kVCvtF64F32Vop1, rdna4::kVCvtF32F64Vop1},
       {rdna4::kVCvtF64I32Vop3, rdna4::kVCvtF64U32Vop3, rdna4::kVCvtI32F64Vop3,
        rdna4::kVCvtU32F64Vop3, rdna4::kVCvtF64F32Vop3, rdna4::kVCvtF32F64Vop3}}};
  for (bool rdna3 : {true, false})
    for (unsigned lanes : {32u, 64u})
      for (bool vop3 : {false, true})
        for (unsigned op = 0; op < 6; ++op)
          for (bool modifiers : {false, true})
            for (bool preserve : {false, true})
              for (unsigned destination : {1u, 2u, 3u, 5u})
                for (unsigned policy_kind = 0; policy_kind < 4; ++policy_kind) {
                  ArithmeticPolicy policy{};
                  if (policy_kind == 1 || policy_kind == 2)
                    policy.semantics = ArithmeticSemantics::BitExactEmpiricalAllowLoose;
                  if (policy_kind == 2)
                    policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
                  if (policy_kind == 3)
                    policy.semantics = ArithmeticSemantics::BitExactEmpirical;
                  BridgeFixture fixture(rdna3, lanes, policy);
                  ASSERT_FALSE(::testing::Test::HasFatalFailure());
                  fixture.wave->set_mode_raw(preserve ? 0x2f0 : 0);
                  const uint64_t exec_mask = lanes == 32 ? 0x8000000fULL : 0x8000000f8000000fULL;
                  fixture.wave->set_exec(exec_mask);
                  const bool from_integer = op < 2, to_double = op < 2 || op == 4;
                  std::array<uint64_t, 4> inputs{}, expected{};
                  for (unsigned i = 0; i < 4; ++i) {
                    double value = std::array{-3.75, 1.25, 3.75, 65504.0}[i];
                    if (from_integer) {
                      inputs[i] = std::array{0u, 1u, 0xffffffffu, 0x80000000u}[i];
                      value = op == 0 ? double(std::bit_cast<int32_t>(uint32_t(inputs[i])))
                                      : double(inputs[i]);
                    } else if (op == 4) {
                      inputs[i] = std::bit_cast<uint32_t>(float(value));
                    } else {
                      if ((op == 2 || op == 3) && i == 2)
                        value = 4294967296.0;
                      if ((op == 2 || op == 3) && i == 3)
                        value = std::bit_cast<double>(0x7ff8000000000001ULL);
                      inputs[i] = std::bit_cast<uint64_t>(value);
                    }
                    if (vop3 && modifiers && !from_integer)
                      value = -std::abs(value);
                    if (op == 2)
                      expected[i] = std::isnan(value)        ? 0u
                                    : value >= 2147483648.0  ? 0x7fffffffu
                                    : value <= -2147483648.0 ? 0x80000000u
                                                             : uint32_t(int32_t(value));
                    else if (op == 3)
                      expected[i] = !(value > 0)            ? 0u
                                    : value >= 4294967296.0 ? 0xffffffffu
                                                            : uint32_t(value);
                    else {
                      if (op == 5)
                        value = float(value);
                      if (vop3 && modifiers && (!rdna3 || !preserve))
                        value *= 0.5;
                      expected[i] = to_double ? std::bit_cast<uint64_t>(value)
                                              : uint64_t(std::bit_cast<uint32_t>(float(value)));
                    }
                  }
                  for (unsigned lane = 0; lane < lanes; ++lane) {
                    for (unsigned reg = 0; reg < 8; ++reg)
                      fixture.cu->write_vgpr(fixture.base() + reg, lane, 0xdead0000u + reg);
                    fixture.cu->write_vgpr(fixture.base() + 2, lane, uint32_t(inputs[lane % 4]));
                    if (!to_double)
                      fixture.cu->write_vgpr(fixture.base() + 3, lane,
                                             uint32_t(inputs[lane % 4] >> 32));
                  }
                  std::array<uint32_t, 2> words{};
                  const auto opcode = opcodes[!rdna3][vop3][op];
                  if (vop3)
                    words = rdna3 ? rdna3::build_vop3(opcode, {.vdst = uint8_t(destination),
                                                               .abs = uint8_t(modifiers),
                                                               .src0 = 258,
                                                               .omod = uint8_t(modifiers ? 3 : 0),
                                                               .neg = uint8_t(modifiers)})
                                  : rdna4::build_vop3(opcode, {.vdst = uint8_t(destination),
                                                               .abs = uint8_t(modifiers),
                                                               .src0 = 258,
                                                               .omod = uint8_t(modifiers ? 3 : 0),
                                                               .neg = uint8_t(modifiers)});
                  else
                    words[0] = rdna3 ? rdna3::build_vop1(
                                           opcode, {.src0 = 258, .vdst = uint8_t(destination)})[0]
                                     : rdna4::build_vop1(
                                           opcode, {.src0 = 258, .vdst = uint8_t(destination)})[0];
                  auto instruction = fixture.decoder->decode_window(words);
                  ASSERT_TRUE(instruction.succeeded());
                  fixture.recorder->reads.clear();
                  fixture.recorder->writes.clear();
                  const bool success =
                      fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded();
                  ASSERT_EQ(success, policy_kind < 2);
                  uint64_t read_lanes[2] = {}, written[2] = {};
                  for (const auto &read : fixture.recorder->reads) {
                    ASSERT_GE(read.reg, 2u);
                    ASSERT_LE(read.reg, to_double ? 2u : 3u);
                    EXPECT_EQ(read.bytes, 0xf);
                    read_lanes[read.reg - 2] |= read.lanes;
                  }
                  EXPECT_EQ(read_lanes[0], exec_mask);
                  EXPECT_EQ(read_lanes[1], to_double ? 0u : exec_mask);
                  for (const auto &write : fixture.recorder->writes) {
                    ASSERT_GE(write.reg, destination);
                    ASSERT_LT(write.reg, destination + (to_double ? 2u : 1u));
                    EXPECT_EQ(write.bytes, 0xf);
                    written[write.reg - destination] |= write.lanes;
                  }
                  EXPECT_EQ(written[0], success ? exec_mask : 0u);
                  EXPECT_EQ(written[1], success && to_double ? exec_mask : 0u);
                  for (unsigned lane = 0; lane < lanes; ++lane)
                    for (unsigned reg = 0; reg < 8; ++reg) {
                      uint32_t want = 0xdead0000u + reg;
                      if (reg == 2)
                        want = uint32_t(inputs[lane % 4]);
                      if (reg == 3 && !to_double)
                        want = uint32_t(inputs[lane % 4] >> 32);
                      if (success && ((exec_mask >> lane) & 1) && reg >= destination &&
                          reg < destination + (to_double ? 2u : 1u))
                        want = uint32_t(expected[lane % 4] >> ((reg - destination) * 32));
                      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + reg, lane), want);
                    }
                }
}

TEST(ArithBridgeTest, WordConversionsRoundingSaturationAndPolicies) {
  const uint16_t opcodes[2][2][6] = {
      {{rdna3::kVCvtF32I32Vop1, rdna3::kVCvtF32U32Vop1, rdna3::kVCvtI32F32Vop1,
        rdna3::kVCvtU32F32Vop1, rdna3::kVCvtNearestI32F32Vop1, rdna3::kVCvtFloorI32F32Vop1},
       {rdna3::kVCvtF32I32Vop3, rdna3::kVCvtF32U32Vop3, rdna3::kVCvtI32F32Vop3,
        rdna3::kVCvtU32F32Vop3, rdna3::kVCvtNearestI32F32Vop3, rdna3::kVCvtFloorI32F32Vop3}},
      {{rdna4::kVCvtF32I32Vop1, rdna4::kVCvtF32U32Vop1, rdna4::kVCvtI32F32Vop1,
        rdna4::kVCvtU32F32Vop1, rdna4::kVCvtNearestI32F32Vop1, rdna4::kVCvtFloorI32F32Vop1},
       {rdna4::kVCvtF32I32Vop3, rdna4::kVCvtF32U32Vop3, rdna4::kVCvtI32F32Vop3,
        rdna4::kVCvtU32F32Vop3, rdna4::kVCvtNearestI32F32Vop3, rdna4::kVCvtFloorI32F32Vop3}}};
  const uint32_t integer_inputs[] = {0,          1,          0xffffffff, 0x80000000,
                                     0x7fffffff, 0x01000001, 0xfffffffe, 0x80000001};
  const uint32_t float_inputs[] = {0x3fc00000, 0xbfc00000, 0x40200000, 0xc0200000,
                                   0x7f800000, 0xff800000, 0x7fc00123, 0xffc00123};
  for (bool rdna3 : {true, false})
    for (unsigned lanes : {32u, 64u})
      for (bool vop3 : {false, true})
        for (unsigned op = 0; op < 6; ++op)
          for (bool modifiers : {false, true})
            for (bool preserve : {false, true})
              for (unsigned destination : {0u, 2u})
                for (unsigned policy_kind = 0; policy_kind < 4; ++policy_kind) {
                  ArithmeticPolicy policy{};
                  if (policy_kind == 1 || policy_kind == 2)
                    policy.semantics = ArithmeticSemantics::BitExactEmpiricalAllowLoose;
                  if (policy_kind == 2)
                    policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
                  if (policy_kind == 3)
                    policy.semantics = ArithmeticSemantics::BitExactEmpirical;
                  BridgeFixture fixture(rdna3, lanes, policy);
                  ASSERT_FALSE(::testing::Test::HasFatalFailure());
                  fixture.wave->set_mode_raw(preserve ? 0x2f0 : 0);
                  const uint64_t exec_mask = lanes == 32 ? 0x800000ffULL : 0x800000ff800000ffULL;
                  fixture.wave->set_exec(exec_mask);
                  const uint32_t *inputs = op < 2 ? integer_inputs : float_inputs;
                  uint32_t expected[8];
                  for (unsigned i = 0; i < 8; ++i) {
                    uint32_t raw = inputs[i];
                    if (op < 2) {
                      float value = op == 0 ? float(std::bit_cast<int32_t>(raw)) : float(raw);
                      if (vop3 && modifiers) {
                        if (!rdna3 || !preserve)
                          value *= 0.5f;
                        value = std::clamp(value, 0.0f, 1.0f);
                      }
                      expected[i] = std::bit_cast<uint32_t>(value);
                    } else {
                      if (vop3 && modifiers)
                        raw = (raw & 0x7fffffffu) | 0x80000000u;
                      float value = std::bit_cast<float>(raw);
                      if (std::isnan(value)) {
                        expected[i] = op < 4 ? 0u : raw >> 31 ? 0x80000000u : 0x7fffffffu;
                        continue;
                      }
                      if (op >= 4) {
                        float rounded = std::floor(value);
                        if (op == 4 && value - rounded >= 0.5f)
                          rounded += 1;
                        value = rounded;
                      }
                      if (op == 3)
                        expected[i] = !(value > 0)             ? 0u
                                      : value >= 4294967296.0f ? 0xffffffffu
                                                               : uint32_t(value);
                      else
                        expected[i] = value >= 2147483648.0f    ? 0x7fffffffu
                                      : value <= -2147483648.0f ? 0x80000000u
                                                                : uint32_t(int32_t(value));
                    }
                  }
                  for (unsigned lane = 0; lane < lanes; ++lane) {
                    fixture.cu->write_vgpr(fixture.base() + 2, lane, 0xdeadbeef);
                    fixture.cu->write_vgpr(fixture.base(), lane, inputs[lane % 8]);
                  }
                  const auto opcode = opcodes[!rdna3][vop3][op];
                  std::array<uint32_t, 2> words{};
                  if (vop3)
                    words = rdna3 ? rdna3::build_vop3(opcode, {.vdst = uint8_t(destination),
                                                               .abs = uint8_t(modifiers),
                                                               .clamp = uint8_t(modifiers),
                                                               .src0 = 256,
                                                               .omod = uint8_t(modifiers ? 3 : 0),
                                                               .neg = uint8_t(modifiers)})
                                  : rdna4::build_vop3(opcode, {.vdst = uint8_t(destination),
                                                               .abs = uint8_t(modifiers),
                                                               .clamp = uint8_t(modifiers),
                                                               .src0 = 256,
                                                               .omod = uint8_t(modifiers ? 3 : 0),
                                                               .neg = uint8_t(modifiers)});
                  else
                    words[0] = rdna3 ? rdna3::build_vop1(
                                           opcode, {.src0 = 256, .vdst = uint8_t(destination)})[0]
                                     : rdna4::build_vop1(
                                           opcode, {.src0 = 256, .vdst = uint8_t(destination)})[0];
                  auto instruction = fixture.decoder->decode_window(words);
                  ASSERT_TRUE(instruction.succeeded());
                  fixture.recorder->reads.clear();
                  fixture.recorder->writes.clear();
                  const bool success =
                      fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded();
                  ASSERT_EQ(success, policy_kind < 2);
                  uint64_t read_lanes = 0, written = 0;
                  for (const auto &read : fixture.recorder->reads) {
                    EXPECT_EQ(read.reg, 0u);
                    EXPECT_EQ(read.bytes, 0xf);
                    read_lanes |= read.lanes;
                  }
                  EXPECT_EQ(read_lanes, exec_mask);
                  for (const auto &write : fixture.recorder->writes) {
                    EXPECT_EQ(write.reg, destination);
                    EXPECT_EQ(write.bytes, 0xf);
                    written |= write.lanes;
                  }
                  EXPECT_EQ(written, success ? exec_mask : 0u);
                  for (unsigned lane = 0; lane < lanes; ++lane) {
                    const uint32_t old = destination == 0 ? inputs[lane % 8] : 0xdeadbeef;
                    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, lane),
                              success && ((exec_mask >> lane) & 1) ? expected[lane % 8] : old);
                  }
                }
}

TEST(ArithBridgeTest, HalfConversionsSelectedBytesModesAndPolicies) {
  const uint16_t opcodes[2][2][6] = {
      {{rdna3::kVCvtF16I16Vop1, rdna3::kVCvtF16U16Vop1, rdna3::kVCvtI16F16Vop1,
        rdna3::kVCvtU16F16Vop1, rdna3::kVCvtF16F32Vop1, rdna3::kVCvtF32F16Vop1},
       {rdna3::kVCvtF16I16Vop3, rdna3::kVCvtF16U16Vop3, rdna3::kVCvtI16F16Vop3,
        rdna3::kVCvtU16F16Vop3, rdna3::kVCvtF16F32Vop3, rdna3::kVCvtF32F16Vop3}},
      {{rdna4::kVCvtF16I16Vop1, rdna4::kVCvtF16U16Vop1, rdna4::kVCvtI16F16Vop1,
        rdna4::kVCvtU16F16Vop1, rdna4::kVCvtF16F32Vop1, rdna4::kVCvtF32F16Vop1},
       {rdna4::kVCvtF16I16Vop3, rdna4::kVCvtF16U16Vop3, rdna4::kVCvtI16F16Vop3,
        rdna4::kVCvtU16F16Vop3, rdna4::kVCvtF16F32Vop3, rdna4::kVCvtF32F16Vop3}}};
  const uint16_t signed_inputs[] = {0, 1, 0xfffd, 0x7fff};
  const uint16_t unsigned_inputs[] = {0, 1, 0xffff, 2049};
  const uint16_t half_inputs[] = {0xbc00, 0x3e00, 0x7c00, 0x7e00};
  const float float_inputs[] = {1.5f, -1.5f, 70000.0f, -70000.0f};
  for (bool rdna3 : {true, false})
    for (unsigned lanes : {32u, 64u})
      for (bool vop3 : {false, true})
        for (unsigned op = 0; op < 6; ++op)
          for (unsigned selection = 0; selection < 4; ++selection)
            for (bool modifiers : {false, true})
              for (bool preserve : {false, true})
                for (unsigned destination : {0u, 2u})
                  for (bool required : {false, true}) {
                    ArithmeticPolicy policy{};
                    if (required)
                      policy.semantics = ArithmeticSemantics::BitExactEmpirical;
                    BridgeFixture fixture(rdna3, lanes, policy);
                    ASSERT_FALSE(::testing::Test::HasFatalFailure());
                    fixture.wave->set_mode_raw(Wavefront::FP16_OVFL_BIT |
                                               Wavefront::DX10_CLAMP_BIT | (preserve ? 0x2f0 : 0));
                    const uint64_t exec_mask = lanes == 32 ? 0x8000000fULL : 0x8000000f8000000fULL;
                    fixture.wave->set_exec(exec_mask);
                    const bool source_half = op != 4, destination_half = op != 5;
                    const bool high_a = source_half && (selection & 1);
                    const bool high_d = destination_half && (selection & 2);
                    uint32_t inputs[4], expected[4];
                    for (unsigned i = 0; i < 4; ++i) {
                      uint16_t half = op == 0   ? signed_inputs[i]
                                      : op == 1 ? unsigned_inputs[i]
                                                : half_inputs[i];
                      inputs[i] = source_half ? (high_a ? (uint32_t(half) << 16) | 0xdeadu
                                                        : 0xdead0000u | half)
                                              : std::bit_cast<uint32_t>(float_inputs[i]);
                      float value = op == 0   ? float(std::bit_cast<int16_t>(half))
                                    : op == 1 ? float(half)
                                    : op == 4 ? float_inputs[i]
                                              : util::f16_to_f32(half);
                      if (vop3 && modifiers && op >= 2)
                        value = -std::abs(value);
                      if (op == 2)
                        expected[i] = std::isnan(value) ? 0u
                                      : value >= 32768  ? 32767u
                                      : value <= -32768 ? 32768u
                                                        : uint16_t(int16_t(value));
                      else if (op == 3)
                        expected[i] = !(value > 0) ? 0u : value >= 65536 ? 65535u : uint16_t(value);
                      else {
                        if (destination_half)
                          value = util::f16_to_f32(util::f32_to_f16_mode(value, true));
                        if (vop3 && modifiers) {
                          if (!rdna3 || !preserve)
                            value *= 0.5f;
                          value = std::isnan(value) ? 0.0f : std::clamp(value, 0.0f, 1.0f);
                        }
                        expected[i] = destination_half ? util::f32_to_f16_mode(value, true)
                                                       : std::bit_cast<uint32_t>(value);
                      }
                    }
                    for (unsigned lane = 0; lane < lanes; ++lane) {
                      fixture.cu->write_vgpr(fixture.base() + 2, lane, 0x12345678);
                      fixture.cu->write_vgpr(fixture.base(), lane, inputs[lane % 4]);
                    }
                    std::array<uint32_t, 2> words{};
                    const auto opcode = opcodes[!rdna3][vop3][op];
                    const uint8_t opsel = (high_a ? 1 : 0) | (high_d ? 8 : 0);
                    if (vop3)
                      words = rdna3 ? rdna3::build_vop3(opcode, {.vdst = uint8_t(destination),
                                                                 .abs = uint8_t(modifiers),
                                                                 .op_sel = opsel,
                                                                 .clamp = uint8_t(modifiers),
                                                                 .src0 = 256,
                                                                 .omod = uint8_t(modifiers ? 3 : 0),
                                                                 .neg = uint8_t(modifiers)})
                                    : rdna4::build_vop3(opcode, {.vdst = uint8_t(destination),
                                                                 .abs = uint8_t(modifiers),
                                                                 .opsel = opsel,
                                                                 .clamp = uint8_t(modifiers),
                                                                 .src0 = 256,
                                                                 .omod = uint8_t(modifiers ? 3 : 0),
                                                                 .neg = uint8_t(modifiers)});
                    else {
                      const uint16_t encoded_source = high_a ? 384 : 256;
                      const uint8_t encoded_destination = uint8_t(destination + (high_d ? 128 : 0));
                      words[0] = rdna3
                                     ? rdna3::build_vop1(opcode, {.src0 = encoded_source,
                                                                  .vdst = encoded_destination})[0]
                                     : rdna4::build_vop1(opcode, {.src0 = encoded_source,
                                                                  .vdst = encoded_destination})[0];
                    }
                    auto instruction = fixture.decoder->decode_window(words);
                    ASSERT_TRUE(instruction.succeeded());
                    fixture.recorder->reads.clear();
                    fixture.recorder->writes.clear();
                    ASSERT_EQ(
                        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .succeeded(),
                        !required);
                    // Packed E32 halves must use one bulk read, not a per-lane
                    // fallback.
                    EXPECT_EQ(fixture.recorder->reads.size(), 1u);
                    uint64_t read_lanes = 0, written = 0;
                    for (const auto &read : fixture.recorder->reads) {
                      EXPECT_EQ(read.reg, 0u);
                      EXPECT_EQ(read.bytes, source_half ? (high_a ? 0xc : 0x3) : 0xf);
                      read_lanes |= read.lanes;
                    }
                    EXPECT_EQ(read_lanes, exec_mask);
                    for (const auto &write : fixture.recorder->writes) {
                      EXPECT_EQ(write.reg, destination);
                      EXPECT_EQ(write.bytes, destination_half ? (high_d ? 0xc : 0x3) : 0xf);
                      written |= write.lanes;
                    }
                    EXPECT_EQ(written, required ? 0u : exec_mask);
                    for (unsigned lane = 0; lane < lanes; ++lane) {
                      uint32_t want = destination == 0 ? inputs[lane % 4] : 0x12345678;
                      if (!required && ((exec_mask >> lane) & 1)) {
                        const unsigned shift = high_d ? 16 : 0;
                        want = destination_half
                                   ? (want & ~(0xffffu << shift)) | (expected[lane % 4] << shift)
                                   : expected[lane % 4];
                      }
                      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, lane), want);
                    }
                  }
}

TEST(ArithBridgeTest, PackedConversionsFixedRoundingModifiersAliasesAndPolicies) {
  const uint16_t opcodes[2][4] = {{rdna3::kVCvtPkRtzF16F32Vop2, rdna3::kVCvtPkRtzF16F32Vop3,
                                   rdna3::kVCvtPkI16F32Vop3, rdna3::kVCvtPkU16F32Vop3},
                                  {rdna4::kVCvtPkRtzF16F32Vop2, rdna4::kVCvtPkRtzF16F32Vop3,
                                   rdna4::kVCvtPkI16F32Vop3, rdna4::kVCvtPkU16F32Vop3}};
  // The inherited RTZ helper is independent of rj-arith. The fractional first
  // input distinguishes truncation from nearest-even at the half boundary.
  const uint32_t inputs[2][4] = {{0x3f801800, 0xc0700000, 0x4788b800, 0x7fc00000},
                                 {0x40200000, 0xbf801800, 0xc788b800, 0x7f800000}};
  for (bool rdna3 : {true, false})
    for (unsigned lanes : {32u, 64u})
      for (unsigned encoding = 0; encoding < 4; ++encoding)
        for (unsigned rounding = 0; rounding < 4; ++rounding)
          for (bool modifiers : {false, true})
            for (unsigned destination : {0u, 1u, 2u})
              for (unsigned policy_kind = 0; policy_kind < 4; ++policy_kind) {
                ArithmeticPolicy policy{};
                if (policy_kind == 1 || policy_kind == 2)
                  policy.semantics = ArithmeticSemantics::BitExactEmpiricalAllowLoose;
                if (policy_kind == 2)
                  policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
                if (policy_kind == 3)
                  policy.semantics = ArithmeticSemantics::BitExactEmpirical;
                BridgeFixture fixture(rdna3, lanes, policy);
                ASSERT_FALSE(::testing::Test::HasFatalFailure());
                fixture.wave->set_mode_raw(0x3f0 | rounding | (rounding << 2) |
                                           Wavefront::FP16_OVFL_BIT);
                const uint64_t exec_mask = lanes == 32 ? 0x8000000fULL : 0x8000000f8000000fULL;
                fixture.wave->set_exec(exec_mask);
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  fixture.cu->write_vgpr(fixture.base(), lane, inputs[0][lane % 4]);
                  fixture.cu->write_vgpr(fixture.base() + 1, lane, inputs[1][lane % 4]);
                  fixture.cu->write_vgpr(fixture.base() + 2, lane, 0x12345678);
                }
                std::array<uint32_t, 2> words{};
                const auto opcode = opcodes[!rdna3][encoding];
                if (encoding)
                  words = rdna3 ? rdna3::build_vop3(opcode, {.vdst = uint8_t(destination),
                                                             .abs = uint8_t(modifiers ? 1 : 0),
                                                             .clamp = uint8_t(modifiers),
                                                             .src0 = 256,
                                                             .src1 = 257,
                                                             .omod = uint8_t(modifiers ? 3 : 0),
                                                             .neg = uint8_t(modifiers ? 2 : 0)})
                                : rdna4::build_vop3(opcode, {.vdst = uint8_t(destination),
                                                             .abs = uint8_t(modifiers ? 1 : 0),
                                                             .clamp = uint8_t(modifiers),
                                                             .src0 = 256,
                                                             .src1 = 257,
                                                             .omod = uint8_t(modifiers ? 3 : 0),
                                                             .neg = uint8_t(modifiers ? 2 : 0)});
                else
                  words[0] =
                      rdna3
                          ? rdna3::build_vop2(
                                opcode, {.src0 = 256, .vsrc1 = 1, .vdst = uint8_t(destination)})[0]
                          : rdna4::build_vop2(
                                opcode, {.src0 = 256, .vsrc1 = 1, .vdst = uint8_t(destination)})[0];
                auto instruction = fixture.decoder->decode_window(words);
                ASSERT_TRUE(instruction.succeeded());
                fixture.recorder->reads.clear();
                fixture.recorder->writes.clear();
                const bool success = policy_kind < 2;
                ASSERT_EQ(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded(),
                          success);
                uint64_t read_lanes[2] = {}, written = 0;
                for (const auto &read : fixture.recorder->reads) {
                  ASSERT_LT(read.reg, 2u);
                  EXPECT_EQ(read.bytes, 0xf);
                  read_lanes[read.reg] |= read.lanes;
                }
                EXPECT_EQ(read_lanes[0], exec_mask);
                EXPECT_EQ(read_lanes[1], exec_mask);
                for (const auto &write : fixture.recorder->writes) {
                  EXPECT_EQ(write.reg, destination);
                  EXPECT_EQ(write.bytes, 0xf);
                  written |= write.lanes;
                }
                EXPECT_EQ(written, success ? exec_mask : 0);
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  uint32_t expected = destination < 2 ? inputs[destination][lane % 4] : 0x12345678;
                  if (success && ((exec_mask >> lane) & 1)) {
                    expected = 0;
                    for (unsigned operand = 0; operand < 2; ++operand) {
                      float value = std::bit_cast<float>(inputs[operand][lane % 4]);
                      if (encoding && modifiers)
                        value = operand ? -value : std::abs(value);
                      uint16_t half;
                      if (encoding < 2)
                        half = util::f32_to_f16_rtz(value);
                      else if (encoding == 2)
                        half = std::isnan(value) ? 0
                               : value >= 32767  ? 32767
                               : value <= -32768 ? 32768
                                                 : uint16_t(int16_t(value));
                      else
                        half = !(value > 0) ? 0 : value >= 65535 ? 65535 : uint16_t(value);
                      expected |= uint32_t(half) << (operand * 16);
                    }
                  }
                  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, lane), expected)
                      << rdna3 << '/' << encoding << '/' << rounding << '/' << modifiers;
                }
              }
}

TEST(ArithBridgeTest, NormalizedConversionsLayoutsRoundingAndPolicies) {
  const uint16_t opcodes[2][4][2] = {
      {{rdna3::kVCvtNormI16F16Vop1, rdna3::kVCvtNormU16F16Vop1},
       {rdna3::kVCvtNormI16F16Vop3, rdna3::kVCvtNormU16F16Vop3},
       {rdna3::kVCvtPkNormI16F16Vop3, rdna3::kVCvtPkNormU16F16Vop3},
       {rdna3::kVCvtPkNormI16F32Vop3, rdna3::kVCvtPkNormU16F32Vop3}},
      {{rdna4::kVCvtNormI16F16Vop1, rdna4::kVCvtNormU16F16Vop1},
       {rdna4::kVCvtNormI16F16Vop3, rdna4::kVCvtNormU16F16Vop3},
       {rdna4::kVCvtPkNormI16F16Vop3, rdna4::kVCvtPkNormU16F16Vop3},
       {rdna4::kVCvtPkNormI16F32Vop3, rdna4::kVCvtPkNormU16F32Vop3}}};
  // Includes false FP32 ties from the existing physical normalized witnesses.
  const uint32_t values[2][4] = {{0x3f000000, 0xbf800000, 0x7fc00000, 0x37c000c0},
                                 {0x3f800000, 0xbf000000, 0x7f800000, 0x38b000b0}};
  for (bool rdna3 : {true, false})
    for (unsigned lanes : {32u, 64u})
      for (unsigned form = 0; form < 4; ++form)
        for (bool is_unsigned : {false, true})
          for (unsigned selection = 0; selection < 4; ++selection)
            for (bool modifiers : {false, true})
              for (unsigned rounding : {0u, 3u})
                for (unsigned destination : {0u, 1u, 2u})
                  for (unsigned policy_kind = 0; policy_kind < 4; ++policy_kind) {
                    const bool packed = form >= 2, half_sources = form != 3;
                    const bool high_a = half_sources && (selection & 1);
                    const bool high_b = half_sources && packed && (selection & 2);
                    const bool high_d = !packed && (selection & 2);
                    ArithmeticPolicy policy{};
                    if (policy_kind == 1 || policy_kind == 2)
                      policy.semantics = ArithmeticSemantics::BitExactEmpiricalAllowLoose;
                    if (policy_kind == 2)
                      policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
                    if (policy_kind == 3)
                      policy.semantics = ArithmeticSemantics::BitExactEmpirical;
                    BridgeFixture fixture(rdna3, lanes, policy);
                    ASSERT_FALSE(::testing::Test::HasFatalFailure());
                    fixture.wave->set_mode_raw(0x3f0 | rounding | (rounding << 2));
                    const uint64_t exec_mask = lanes == 32 ? 0x8000000fULL : 0x8000000f8000000fULL;
                    fixture.wave->set_exec(exec_mask);
                    uint32_t inputs[2][4];
                    for (unsigned operand = 0; operand < 2; ++operand)
                      for (unsigned i = 0; i < 4; ++i) {
                        const auto half =
                            util::f32_to_f16(std::bit_cast<float>(values[operand][i]));
                        inputs[operand][i] = !half_sources ? values[operand][i]
                                             : (operand ? high_b : high_a)
                                                 ? (uint32_t(half) << 16) | 0xdead
                                                 : 0xdead0000 | half;
                      }
                    for (unsigned lane = 0; lane < lanes; ++lane) {
                      fixture.cu->write_vgpr(fixture.base(), lane, inputs[0][lane % 4]);
                      fixture.cu->write_vgpr(fixture.base() + 1, lane, inputs[1][lane % 4]);
                      fixture.cu->write_vgpr(fixture.base() + 2, lane, 0x12345678);
                    }
                    std::array<uint32_t, 2> words{};
                    const auto opcode = opcodes[!rdna3][form][is_unsigned];
                    const uint8_t opsel = (high_a ? 1 : 0) | (high_b ? 2 : 0) | (high_d ? 8 : 0);
                    if (form)
                      words = rdna3
                                  ? rdna3::build_vop3(
                                        opcode, {.vdst = uint8_t(destination),
                                                 .abs = uint8_t(modifiers ? 1 : 0),
                                                 .op_sel = opsel,
                                                 .clamp = uint8_t(modifiers),
                                                 .src0 = 256,
                                                 .src1 = 257,
                                                 .omod = uint8_t(modifiers ? 3 : 0),
                                                 .neg = uint8_t(modifiers ? (packed ? 3 : 1) : 0)})
                                  : rdna4::build_vop3(
                                        opcode, {.vdst = uint8_t(destination),
                                                 .abs = uint8_t(modifiers ? 1 : 0),
                                                 .opsel = opsel,
                                                 .clamp = uint8_t(modifiers),
                                                 .src0 = 256,
                                                 .src1 = 257,
                                                 .omod = uint8_t(modifiers ? 3 : 0),
                                                 .neg = uint8_t(modifiers ? (packed ? 3 : 1) : 0)});
                    else {
                      const uint16_t src = high_a ? 384 : 256;
                      const uint8_t dst = uint8_t(destination + (high_d ? 128 : 0));
                      words[0] = rdna3 ? rdna3::build_vop1(opcode, {.src0 = src, .vdst = dst})[0]
                                       : rdna4::build_vop1(opcode, {.src0 = src, .vdst = dst})[0];
                    }
                    auto instruction = fixture.decoder->decode_window(words);
                    ASSERT_TRUE(instruction.succeeded());
                    fixture.recorder->reads.clear();
                    fixture.recorder->writes.clear();
                    const bool success = policy_kind < 2;
                    ASSERT_EQ(
                        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .succeeded(),
                        success);
                    if (success && form == 0) {
                      EXPECT_EQ(fixture.recorder->reads.size(), 1u);
                      EXPECT_EQ(fixture.recorder->writes.size(), 1u);
                    }
                    uint64_t read_lanes[2] = {}, written = 0;
                    for (const auto &read : fixture.recorder->reads) {
                      ASSERT_LT(read.reg, packed ? 2u : 1u);
                      EXPECT_EQ(read.bytes,
                                half_sources ? ((read.reg ? high_b : high_a) ? 0xc : 3) : 0xf);
                      read_lanes[read.reg] |= read.lanes;
                    }
                    EXPECT_EQ(read_lanes[0], exec_mask);
                    EXPECT_EQ(read_lanes[1], packed ? exec_mask : 0);
                    for (const auto &write : fixture.recorder->writes) {
                      EXPECT_EQ(write.reg, destination);
                      EXPECT_EQ(write.bytes, packed ? 0xf : high_d ? 0xc : 3);
                      written |= write.lanes;
                    }
                    EXPECT_EQ(written, success ? exec_mask : 0);
                    for (unsigned lane = 0; lane < lanes; ++lane) {
                      uint32_t expected =
                          destination < 2 ? inputs[destination][lane % 4] : 0x12345678;
                      if (success && ((exec_mask >> lane) & 1)) {
                        uint32_t result = 0;
                        for (unsigned operand = 0; operand < (packed ? 2u : 1u); ++operand) {
                          float value = std::bit_cast<float>(values[operand][lane % 4]);
                          if (half_sources)
                            value = util::f16_to_f32(util::f32_to_f16(value));
                          if (form && modifiers)
                            value = operand ? -value : -std::abs(value);
                          // Inherited scalar model: multiply in double, then
                          // round once.
                          double scaled =
                              std::isnan(value)
                                  ? 0.0
                                  : std::clamp(double(value) * (is_unsigned ? 65535.0 : 32767.0),
                                               is_unsigned ? 0.0 : -32767.0,
                                               is_unsigned ? 65535.0 : 32767.0);
                          result |= uint32_t(uint16_t(int32_t(util::rndne_scalar(scaled))))
                                    << (operand * 16);
                        }
                        const unsigned shift = high_d ? 16 : 0;
                        expected =
                            packed ? result : (expected & ~(0xffffu << shift)) | (result << shift);
                      }
                      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, lane), expected)
                          << rdna3 << '/' << form << '/' << selection << '/' << modifiers;
                    }
                  }
}

TEST(ArithBridgeTest, BitPackingSelectorsAliasingAndPolicies) {
  const uint32_t half_inputs[2][4] = {{0x7c013c00, 0xfc018001, 0x7e55bc00, 0x00010000},
                                      {0x3e00fc01, 0x80017c01, 0xfe550400, 0x80003c00}};
  const uint32_t float_inputs[] = {0x3f000000, 0x3fc00000, 0x437e8000, 0x437f8000,
                                   0xbfc00000, 0xff800000, 0x7fc00000, 0x7f800000};
  for (bool rdna3 : {true, false})
    for (unsigned lanes : {32u, 64u})
      for (bool half : {false, true})
        for (unsigned selection = 0; selection < 4; ++selection)
          for (unsigned modifiers = 0; modifiers < 4; ++modifiers)
            for (unsigned destination = 0; destination < 4; ++destination)
              for (unsigned policy_kind = 0; policy_kind < 4; ++policy_kind) {
                ArithmeticPolicy policy{};
                if (policy_kind == 1 || policy_kind == 2)
                  policy.semantics = ArithmeticSemantics::BitExactEmpiricalAllowLoose;
                if (policy_kind == 2)
                  policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
                if (policy_kind == 3)
                  policy.semantics = ArithmeticSemantics::BitExactEmpirical;
                BridgeFixture fixture(rdna3, lanes, policy);
                ASSERT_FALSE(::testing::Test::HasFatalFailure());
                fixture.wave->set_mode_raw(0x3ff); // Byte conversion uses RTZ; halfword packing
                                                   // ignores rounding.
                const uint64_t exec_mask = lanes == 32 ? 0x800000ffULL : 0x800000ff800000ffULL;
                fixture.wave->set_exec(exec_mask);
                uint32_t inputs[3][64];
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  inputs[0][lane] = half ? half_inputs[0][lane % 4] : float_inputs[lane % 8];
                  inputs[1][lane] =
                      half ? half_inputs[1][lane % 4] : 0xa5a50000 | ((lane + selection) & 3);
                  inputs[2][lane] = 0x12345678 ^ (lane * 0x01010101);
                  for (unsigned reg = 0; reg < 3; ++reg)
                    fixture.cu->write_vgpr(fixture.base() + reg, lane, inputs[reg][lane]);
                  fixture.cu->write_vgpr(fixture.base() + 3, lane, 0xdeadbeef);
                }
                const uint16_t opcode =
                    rdna3 ? (half ? rdna3::kVPackB32F16Vop3 : rdna3::kVCvtPkU8F32Vop3)
                          : (half ? rdna4::kVPackB32F16Vop3 : rdna4::kVCvtPkU8F32Vop3);
                const uint8_t abs = (modifiers & 1) ? (half ? 3 : 1) : 0;
                const uint8_t neg = (modifiers & 2) ? (half ? 3 : 1) : 0;
                const uint8_t opsel = half ? selection : 0;
                const uint8_t clamp = !half && modifiers ? 1 : 0;
                auto words = rdna3 ? rdna3::build_vop3(opcode, {.vdst = uint8_t(destination),
                                                                .abs = abs,
                                                                .op_sel = opsel,
                                                                .clamp = clamp,
                                                                .src0 = 256,
                                                                .src1 = 257,
                                                                .src2 = 258,
                                                                .neg = neg})
                                   : rdna4::build_vop3(opcode, {.vdst = uint8_t(destination),
                                                                .abs = abs,
                                                                .opsel = opsel,
                                                                .clamp = clamp,
                                                                .src0 = 256,
                                                                .src1 = 257,
                                                                .src2 = 258,
                                                                .neg = neg});
                auto instruction = fixture.decoder->decode_window(words);
                ASSERT_TRUE(instruction.succeeded());
                fixture.recorder->reads.clear();
                fixture.recorder->writes.clear();
                const bool success = policy_kind < 2 || (policy_kind == 2 && half && !rdna3);
                ASSERT_EQ(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded(),
                          success);
                uint64_t read_lanes[3] = {}, written = 0;
                for (const auto &read : fixture.recorder->reads) {
                  ASSERT_LT(read.reg, half ? 2u : 3u);
                  EXPECT_EQ(read.bytes, half ? ((selection & (1u << read.reg)) ? 0xc : 3) : 0xf);
                  read_lanes[read.reg] |= read.lanes;
                }
                for (unsigned reg = 0; reg < (half ? 2u : 3u); ++reg)
                  EXPECT_EQ(read_lanes[reg], exec_mask);
                for (const auto &write : fixture.recorder->writes) {
                  EXPECT_EQ(write.reg, destination);
                  EXPECT_EQ(write.bytes, 0xf);
                  written |= write.lanes;
                }
                EXPECT_EQ(written, success ? exec_mask : 0);
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  uint32_t expected = destination < 3 ? inputs[destination][lane] : 0xdeadbeef;
                  if (success && ((exec_mask >> lane) & 1)) {
                    if (half) {
                      // rj-arith's captured RDNA4 PACK rule quiets signaling
                      // NaNs, preserving payload/sign after selected-half
                      // ABS/NEG.
                      expected = 0;
                      for (unsigned operand = 0; operand < 2; ++operand) {
                        uint16_t value = uint16_t(inputs[operand][lane] >>
                                                  ((selection & (1u << operand)) ? 16 : 0));
                        if (abs & (1u << operand))
                          value &= 0x7fff;
                        if (neg & (1u << operand))
                          value ^= 0x8000;
                        if ((value & 0x7fff) > 0x7c00)
                          value |= 0x200;
                        expected |= uint32_t(value) << (operand * 16);
                      }
                    } else {
                      float value = std::bit_cast<float>(inputs[0][lane]);
                      if (abs)
                        value = std::abs(value);
                      if (neg)
                        value = -value;
                      value = std::isnan(value) ? 0.0f : std::clamp(value, 0.0f, 255.0f);
                      const uint32_t byte = uint32_t(value); // MODE.FP_ROUND is toward zero.
                      const unsigned shift = (inputs[1][lane] & 3) * 8;
                      expected = (inputs[2][lane] & ~(0xffu << shift)) | (byte << shift);
                    }
                  }
                  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, lane), expected)
                      << rdna3 << '/' << half << '/' << selection << '/' << modifiers;
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
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
                    // The existing rj-arith OMOD reference flushes
                    // zero/subnormal intermediates to +0 before scaling,
                    // independently of MODE.
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
  // gfx1100/gfx1201 use the same table tail and flush it before OMOD (hw1100-noieee, MODE 0x00).
  const unsigned exponent[] = {1968, 1091, 1091, 1997};
  const unsigned selector[] = {5, 19, 19, 4};
  const uint64_t golden[4] = {0, 0, 0, 0x11aba10ac06608cULL};
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
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
                    !rejected && ((exec_mask >> lane) & 1) ? golden[omod] : before;
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
                    ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
                  ASSERT_FALSE(::testing::Test::HasFatalFailure());
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

TEST(ArithBridgeTest, MixedFmaDppPreservesSuppressedDestinations) {
  BridgeFixture fixture(false);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.wave->set_exec(0x38003);
  for (unsigned lane = 0; lane < 32; ++lane) {
    fixture.cu->write_vgpr(fixture.base() + 1, lane, std::bit_cast<uint32_t>(float(lane + 1)));
    fixture.cu->write_vgpr(fixture.base() + 2, lane, std::bit_cast<uint32_t>(2.0f));
    fixture.cu->write_vgpr(fixture.base() + 3, lane, std::bit_cast<uint32_t>(1.0f));
  }
  // v_fma_mix_f32 v1, v1, v2, v3 row_shl:1 row_mask:1 bank_mask:f.
  // Only lane 0 has both an enabled destination row and an active shifted
  // source.
  const std::array<uint32_t, 3> words{0xcc200001, 0x040e04fa, 0x1f010101};
  auto instruction = fixture.decoder->decode_window(words);
  ASSERT_TRUE(instruction.succeeded());
  fixture.recorder->writes.clear();
  ASSERT_TRUE(
      fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
  for (unsigned lane = 0; lane < 32; ++lane)
    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, lane),
              std::bit_cast<uint32_t>(lane == 0 ? 5.0f : float(lane + 1)))
        << lane;
  ASSERT_EQ(fixture.recorder->writes.size(), 1u);
  EXPECT_EQ(fixture.recorder->writes[0].lanes, 1u);
}

TEST(ArithBridgeTest, FmaGuestControlsChangeResults) {
  // Distinct FP32 and FP16 rounding fields must reach the selected result
  // format.
  for (bool rdna3 : {false, true}) {
    for (int kind = 0; kind < 3; ++kind) {
      for (uint32_t mode : {0xf1u, 0xf4u}) {
        BridgeFixture fixture(rdna3);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        fixture.wave->set_mode_raw(mode);
        fixture.wave->set_exec(1);
        fixture.cu->write_vgpr(fixture.base(), 0, 0x3f800000u);
        fixture.cu->write_vgpr(fixture.base() + 1, 0, kind == 2 ? 0x3a000000u : 0x33800000u);
        fixture.cu->write_vgpr(fixture.base() + 2, 0, 0x3f800000u);
        auto operand = [&](unsigned reg) -> std::unique_ptr<Operand> {
          if (rdna3)
            return std::make_unique<rdna3::Operand>(32, rdna3::OperandType::OPR_VGPR, reg);
          return std::make_unique<rdna4::Operand>(32, rdna4::OperandType::OPR_VGPR, reg);
        };
        const auto dst = operand(3);
        const auto a = operand(0);
        const auto b = operand(1);
        const auto c = operand(2);
        if (kind == 0)
          arith::execute_float_fma32<false>(
              *fixture.wave, rdna3 ? rj_arith_v_fma_f32_w32 : rj_arith_v_fma_f32_w32_gfx12,
              rdna3 ? rj_arith_v_fma_f32_w64 : rj_arith_v_fma_f32_w64_gfx12,
              fixture.cu->arithmetic_flags(), 0, *dst, *a, *b, *c, rdna3);
        else
          arith::execute_mixed_fma(
              *fixture.wave,
              kind == 2
                  ? (rdna3 ? rj_arith_v_fma_mixlo_f16_w32 : rj_arith_v_fma_mixlo_f16_w32_gfx12)
                  : (rdna3 ? rj_arith_v_fma_mix_f32_w32 : rj_arith_v_fma_mix_f32_w32_gfx12),
              kind == 2
                  ? (rdna3 ? rj_arith_v_fma_mixlo_f16_w64 : rj_arith_v_fma_mixlo_f16_w64_gfx12)
                  : (rdna3 ? rj_arith_v_fma_mix_f32_w64 : rj_arith_v_fma_mix_f32_w64_gfx12),
              fixture.cu->arithmetic_flags(), 0, *dst, *a, *b, *c, 0, kind == 2 ? 3 : 15, rdna3);
        ASSERT_FALSE(fixture.wave->instruction_execution_failed());
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 3, 0),
                  kind == 2 ? (mode == 0xf4 ? 0x3c003c01u : 0x3c003c00u)
                            : (mode == 0xf1 ? 0x3f800001u : 0x3f800000u));
      }
    }
  }
  // Distinguish FP32 input/output denormal fields and RDNA3 DX10 clamp.
  BridgeFixture fixture(true);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.wave->set_exec(1);
  struct Case {
    uint32_t mode, a, b, expected;
    bool clamp;
  };
  for (const Case &test : {Case{0x30, 1, 0x7e800000, 0x34000000, false},
                           Case{0xf0, 0x7fc00000, 0x3f800000, 0x7fc00000, true},
                           Case{0x70, 0x1f000000, 0x1f000000, 0x00080000, false}}) {
    for (bool mixed : {false, true}) {
      fixture.cu->write_vgpr(fixture.base() + 3, 0, 0xdeadbeef);
      fixture.wave->set_mode_raw(test.mode);
      fixture.cu->write_vgpr(fixture.base(), 0, test.a);
      fixture.cu->write_vgpr(fixture.base() + 1, 0, test.b);
      fixture.cu->write_vgpr(fixture.base() + 2, 0, 0);
      const rdna3::Operand dst(32, rdna3::OperandType::OPR_VGPR, 3);
      const rdna3::Operand a(32, rdna3::OperandType::OPR_VGPR, 0);
      const rdna3::Operand b(32, rdna3::OperandType::OPR_VGPR, 1);
      const rdna3::Operand c(32, rdna3::OperandType::OPR_VGPR, 2);
      const uint64_t modifiers = test.clamp ? RJ_ARITH_ALU_CLAMP : 0;
      if (mixed)
        arith::execute_mixed_fma(*fixture.wave, rj_arith_v_fma_mix_f32_w32,
                                 rj_arith_v_fma_mix_f32_w64, fixture.cu->arithmetic_flags(),
                                 modifiers, dst, a, b, c, 0, 15, true);
      else
        arith::execute_float_fma32<false>(*fixture.wave, rj_arith_v_fma_f32_w32,
                                          rj_arith_v_fma_f32_w64, fixture.cu->arithmetic_flags(),
                                          modifiers, dst, a, b, c, true);
      ASSERT_FALSE(fixture.wave->instruction_execution_failed());
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 3, 0), test.expected);
    }
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
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        // Directed rounding; mixed-format input denorm settings;
        // DX10_CLAMP-off.
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

// Legacy classifiers use EXEC; DPP row masks only suppress destination writes.
TEST(ArithBridgeTest, CompatibilityUnaryStatusIncludesDppSuppressedLanes) {
  for (bool sqrt : {false, true}) {
    for (bool status_only : {false, true}) {
      for (const char *policy : {"compatibility", "loose"}) {
        BridgeFixture fixture(false, 32, parse_arithmetic_policy(policy));
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        fixture.wave->set_exec(status_only ? 1U : 1U | (1U << 16));
        fixture.wave->set_mode_raw(0xf0);
        fixture.cu->write_vgpr(fixture.base(), 0, sqrt ? 0xbf800000U : 0U);
        fixture.cu->write_vgpr(fixture.base(), 16, 0x3f800000U);
        rdna4::Vop1VopDpp16MachineInst raw{};
        raw.src0 = SRC_DPP;
        raw.vsrc0 = 0;
        raw.vdst = 1;
        raw.dpp_ctrl = 0xe4; // Identity quad permutation.
        raw.bank_mask = 0xf;
        raw.row_mask = 2;
        const auto *encoding = reinterpret_cast<const rdna4::MachineInst *>(&raw);
        if (sqrt) {
          rdna4::VSqrtF32Vop1(encoding).execute_impl(*fixture.wave);
        } else {
          rdna4::VRcpIflagF32Vop1(encoding).execute_impl(*fixture.wave);
        }
        ASSERT_FALSE(fixture.wave->instruction_execution_failed());
        const bool report = (fixture.cu->arithmetic_flags() & RJ_ARITH_SEMANTICS_MASK) ==
                            RJ_ARITH_SEMANTICS_COMPATIBILITY;
        EXPECT_EQ(fixture.wave->trapsts(), report ? (sqrt ? 1U : 64U) : 0U);
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 0), 0x3c003c00U);
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 16),
                  status_only ? 0x3c003c00U : 0x3f800000U);
      }
    }
  }
}

TEST(ArithBridgeTest, PendingRegisterErrorPreventsScalarCommit) {
  for (bool pseudo_scalar : {false, true}) {
    BridgeFixture fixture(false, 32);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const uint32_t base = fixture.wave->sgpr_alloc().base;
    fixture.cu->write_sgpr(base, 1U);
    fixture.cu->write_sgpr(base + 1, 2U);
    fixture.cu->write_sgpr(base + 2, 0xdeadbeefU);
    fixture.wave->write_scc(true);
    fixture.wave->set_trapsts(0x80000000U);
    fixture.wave->report_instruction_execution_error(
        InstructionExecutionError::InvalidRegisterAccess);
    if (pseudo_scalar) {
      const auto words = rdna4::build_vop3(rdna4::kVSSqrtF32Vop3, {.vdst = 2, .src0 = 0});
      rdna4::VSSqrtF32Vop3(words.data()).execute_impl(*fixture.wave);
    } else {
      const auto words =
          rdna4::build_sop2(rdna4::kSAddCoU32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 2});
      rdna4::SAddCoU32Sop2(words.data()).execute_impl(*fixture.wave);
    }
    EXPECT_EQ(fixture.cu->read_sgpr(base + 2), 0xdeadbeefU);
    EXPECT_TRUE(fixture.wave->read_scc());
    EXPECT_EQ(fixture.wave->trapsts(), 0x80000000U);
    EXPECT_TRUE(fixture.recorder->scalar_writes.empty());
  }
}

TEST(ArithBridgeTest, PseudoScalarIgnoresExecAndStagesScalarOutputs) {
  const uint16_t ops[] = {rdna4::kVSExpF32Vop3, rdna4::kVSExpF16Vop3, rdna4::kVSLogF32Vop3,
                          rdna4::kVSLogF16Vop3, rdna4::kVSRcpF32Vop3, rdna4::kVSRcpF16Vop3,
                          rdna4::kVSRsqF32Vop3, rdna4::kVSRsqF16Vop3, rdna4::kVSSqrtF32Vop3,
                          rdna4::kVSSqrtF16Vop3};
  // Exactly representable exp2(0), log2(1), reciprocal(2), rsqrt(4), sqrt(4).
  const float inputs[] = {0, 1, 2, 4, 4}, outputs[] = {1, 0, 0.5f, 0.5f, 2};
  for (unsigned op = 0; op < 10; ++op)
    for (uint32_t lanes : {32u, 64u})
      for (auto semantics :
           {ArithmeticSemantics::Loose, ArithmeticSemantics::BitExactEmpiricalAllowLoose,
            ArithmeticSemantics::BitExactEmpiricalNoFpStatus,
            ArithmeticSemantics::BitExactEmpirical, ArithmeticSemantics::Compatibility})
        for (bool alias : {false, true})
          for (bool modifiers : {false, true}) {
            const bool half = op & 1;
            ArithmeticPolicy policy{};
            policy.semantics = semantics;
            BridgeFixture fixture(false, lanes, policy);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            fixture.wave->set_exec(0);
            fixture.wave->set_mode_raw(0xff);
            fixture.wave->set_trapsts(0x80000001);
            fixture.wave->write_scc(true);
            const uint32_t base = fixture.wave->sgpr_alloc().base;
            const uint32_t destination = alias ? 0 : 4;
            const float input = modifiers ? -inputs[op / 2] : inputs[op / 2];
            const uint32_t raw =
                half ? 0xa5a50000u | util::f32_to_f16(input) : std::bit_cast<uint32_t>(input);
            fixture.cu->write_sgpr(base, raw);
            if (!alias)
              fixture.cu->write_sgpr(base + destination, 0xdeadbeef);
            const auto words = rdna4::build_vop3(ops[op], {.vdst = uint8_t(destination),
                                                           .abs = uint8_t(modifiers),
                                                           .opsel = uint8_t(half),
                                                           .clamp = uint8_t(modifiers),
                                                           .src0 = 0,
                                                           .omod = uint8_t(modifiers)});
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            fixture.recorder->scalar_reads.clear();
            fixture.recorder->scalar_writes.clear();
            const bool rejected = semantics == ArithmeticSemantics::BitExactEmpirical ||
                                  (semantics == ArithmeticSemantics::BitExactEmpiricalNoFpStatus &&
                                   (op == 1 || op == 3));
            EXPECT_EQ(
                fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).failed(),
                rejected);
            EXPECT_EQ(fixture.recorder->scalar_reads, (std::vector<uint32_t>{base}));
            ASSERT_EQ(fixture.recorder->scalar_writes.size(), rejected ? 0u : 1u);
            if (!rejected) {
              EXPECT_EQ(fixture.recorder->scalar_writes[0].index, destination);
              EXPECT_EQ(fixture.recorder->scalar_writes[0].width, 1u);
            }
            const float value = modifiers ? std::min(outputs[op / 2] * 2, 1.0f) : outputs[op / 2];
            const uint32_t expected =
                half ? util::f32_to_f16(value) : std::bit_cast<uint32_t>(value);
            EXPECT_EQ(fixture.cu->read_sgpr(base + destination),
                      rejected ? (alias ? raw : 0xdeadbeef) : expected);
            EXPECT_EQ(fixture.wave->exec(), 0u);
            EXPECT_TRUE(fixture.wave->read_scc());
            EXPECT_EQ(fixture.wave->trapsts(), 0x80000001u);
            EXPECT_EQ(fixture.wave->pending_alu_causes(), 0u);
            EXPECT_TRUE(fixture.recorder->writes.empty());
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
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
              const unsigned last = lanes - 1;
              const uint64_t exec_mask = 1ULL << last;
              fixture.wave->set_exec(exec_mask);
              // Preserve guest denormals, with DX10_CLAMP enabled. SQRT still
              // flushes.
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
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
            // Compatibility retains the legacy classifier in every
            // configuration.
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

TEST(ArithBridgeTest, DivisionScaleWritesExplicitHighConditionBit) {
  for (bool rdna3 : {false, true}) {
    for (uint32_t lanes : {32U, 64U}) {
      for (bool wide : {false, true}) {
        BridgeFixture fixture(rdna3, lanes);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        fixture.wave->set_exec(1ULL | (1ULL << (lanes - 1)));
        fixture.wave->set_mode_raw(0x1f0);
        fixture.wave->set_vcc(0x55);
        const uint64_t one = wide ? 0x3ff0000000000000ULL : 0x3f800000ULL;
        const uint64_t tiny = wide ? 0x0010000000000000ULL : 0x00800000ULL;
        for (uint32_t lane : {0U, lanes - 1}) {
          for (uint32_t operand = 0; operand < 3; ++operand) {
            const uint64_t value = operand == 2 && lane != 0 ? tiny : one;
            fixture.cu->write_vgpr(fixture.base() + 2 * operand, lane, uint32_t(value));
            if (wide)
              fixture.cu->write_vgpr(fixture.base() + 2 * operand + 1, lane, uint32_t(value >> 32));
          }
        }
        const auto words =
            rdna3 ? rdna3::build_vop3_sdst_enc(
                        wide ? rdna3::kVDivScaleF64Vop3SdstEnc : rdna3::kVDivScaleF32Vop3SdstEnc,
                        {.vdst = 4, .sdst = 4, .src0 = 256, .src1 = 258, .src2 = 260})
                  : rdna4::build_vop3_sdst_enc(
                        wide ? rdna4::kVDivScaleF64Vop3SdstEnc : rdna4::kVDivScaleF32Vop3SdstEnc,
                        {.vdst = 4, .sdst = 4, .src0 = 256, .src1 = 258, .src2 = 260});
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        const uint32_t sb = fixture.wave->sgpr_alloc().base;
        EXPECT_EQ(fixture.cu->read_sgpr(sb + 4), lanes == 32 ? 0x80000000U : 0U);
        if (lanes == 64) {
          EXPECT_EQ(fixture.cu->read_sgpr(sb + 5), 0x80000000U);
        }
        EXPECT_EQ(fixture.wave->vcc(), 0x55U);
      }
    }
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
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
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
          // Preserve the inherited classifier; gfx1100 review captures also
          // show DIV0 with DX10_CLAMP clear and suppression with it set.
          EXPECT_EQ(fixture.wave->trapsts(), 0x20u | (divide_by_zero && !dx10 ? 4u : 0u));
        }
      }
    }
  }
}

TEST(ArithBridgeTest, EmptyScaleValidatesBeforeCommittingCondition) {
  for (uint32_t lanes : {32u, 64u}) {
    for (bool unsupported : {false, true}) {
      BridgeFixture fixture(true, lanes, {ArithmeticSemantics::BitExactEmpiricalNoFpStatus});
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_mode_raw(0xd0); // OMOD active: FP32 output flushing, IEEE clear.
      fixture.wave->set_exec(0);
      uint64_t condition = 0x1234;
      rdna3::Operand operand(32, rdna3::OperandType::OPR_VGPR, 0);
      arith::execute_division<32, arith::DivisionOperation::Scale>(
          *fixture.wave, rj_arith_v_div_scale_f32_w32, rj_arith_v_div_scale_f32_w64,
          fixture.cu->arithmetic_flags(), unsupported ? RJ_ARITH_ALU_OMOD_2 : 0, operand, operand,
          operand, operand, true, [&](uint64_t value) { condition = value; });
      EXPECT_EQ(fixture.wave->instruction_execution_failed(), unsupported);
      EXPECT_EQ(condition, unsupported ? 0x1234u : 0u);
      EXPECT_TRUE(fixture.recorder->writes.empty());
    }
  }
}

TEST(ArithBridgeTest, HalfFixupGuestRoundingNanClampAndPartialWrites) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32u, 64u})
      for (unsigned rounding = 0; rounding < 4; ++rounding)
        for (bool dx10 : {false, true})
          for (bool clamp : {false, true})
            for (bool nan : {false, true})
              for (bool ieee : {false, true}) {
                BridgeFixture fixture(rdna3, lanes);
                ASSERT_FALSE(::testing::Test::HasFatalFailure());
                const unsigned last = lanes - 1;
                fixture.wave->set_exec(1ULL << last);
                fixture.wave->set_mode_raw(0xf0 | (rounding << 2) | (dx10 ? 0x100 : 0) |
                                           (ieee ? 0x200 : 0));
                fixture.wave->set_trapsts(0x20);
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  fixture.cu->write_vgpr(fixture.base(), lane, 0x7c00a55a);
                  fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x3c000000);
                  fixture.cu->write_vgpr(fixture.base() + 2, lane, nan ? 0xfc150000 : 0x3c000000);
                }
                const auto words =
                    rdna3 ? rdna3::build_vop3(rdna3::kVDivFixupF16Vop3, {.vdst = 0,
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
                ASSERT_TRUE(
                    fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                        .succeeded());
                ASSERT_EQ(fixture.recorder->reads.size(), 3U);
                for (const auto &read : fixture.recorder->reads) {
                  EXPECT_EQ(read.bytes, 0xc);
                  EXPECT_EQ(read.lanes, 1ULL << last);
                }
                ASSERT_EQ(fixture.recorder->writes.size(), 1U);
                EXPECT_EQ(fixture.recorder->writes[0].bytes, 0xc);
                EXPECT_EQ(fixture.recorder->writes[0].lanes, 1ULL << last);
                const uint32_t nan_bits = rdna3 && !ieee ? 0xfc15 : 0xfe15;
                uint32_t expected = nan ? nan_bits : rounding < 2 ? 0x7c00 : 0x7bff;
                if (clamp)
                  expected = nan ? (rdna3 && !dx10 ? nan_bits : 0) : 0x3c00;
                for (unsigned lane = 0; lane < lanes; ++lane)
                  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                            lane == last ? (expected << 16) | 0xa55a : 0x7c00a55a);
                EXPECT_EQ(fixture.wave->trapsts(), 0x20U);
              }
}

TEST(ArithBridgeTest, GuestModeInstructionsUpdateOnlyTheirNibbleRegardlessOfExecOrReporting) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5})
    for (uint32_t lanes : {32U, 64U})
      for (bool denorm : {false, true})
        for (bool loose : {false, true})
          for (bool report : {false, true})
            for (uint16_t immediate : {0, 1, 5, 10, 15, 0xfff2, 0xffff}) {
              ArithmeticPolicy policy{};
              policy.semantics = loose ? ArithmeticSemantics::Loose
                                       : ArithmeticSemantics::BitExactEmpiricalNoFpStatus;

              if (report && !loose)
                policy.semantics = ArithmeticSemantics::BitExactEmpirical;
              const bool rdna3 = arch == ROCJITSU_CODE_ARCH_RDNA3;
              if (arch == ROCJITSU_CODE_ARCH_CDNA5 && lanes != 32)
                continue;
              BridgeFixture fixture(arch, lanes, policy);
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
              constexpr uint32_t initial = 0x1234abcfU;
              fixture.wave->set_mode_raw(initial);
              fixture.wave->set_exec(0);
              fixture.wave->write_scc(1);
              const auto words =
                  arch == ROCJITSU_CODE_ARCH_CDNA5
                      ? cdna5::build_sopp(denorm ? cdna5::kSDenormModeSopp : cdna5::kSRoundModeSopp,
                                          {.simm16 = immediate})
                  : rdna3
                      ? rdna3::build_sopp(denorm ? rdna3::kSDenormModeSopp : rdna3::kSRoundModeSopp,
                                          {.simm16 = immediate})
                      : rdna4::build_sopp(denorm ? rdna4::kSDenormModeSopp : rdna4::kSRoundModeSopp,
                                          {.simm16 = immediate});
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              const unsigned shift = denorm ? 4 : 0;
              EXPECT_EQ(fixture.wave->mode_raw(),
                        (initial & ~(15U << shift)) | ((immediate & 15U) << shift));
              EXPECT_EQ(fixture.wave->exec(), 0U);
              EXPECT_EQ(fixture.wave->read_scc(), 1U);
            }
}

TEST(ArithBridgeTest, GuestDenormInstructionControlsFollowingArithmetic) {
  for (bool rdna3 : {false, true})
    for (bool preserve : {false, true}) {
      BridgeFixture fixture(rdna3);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_mode_raw(0);
      fixture.wave->set_exec(0);
      const auto mode_words =
          rdna3
              ? rdna3::build_sopp(rdna3::kSDenormModeSopp, {.simm16 = uint16_t(preserve ? 15 : 0)})
              : rdna4::build_sopp(rdna4::kSDenormModeSopp, {.simm16 = uint16_t(preserve ? 15 : 0)});
      auto mode = fixture.decoder->decode_window(mode_words);
      ASSERT_TRUE(mode.succeeded());
      ASSERT_TRUE(fixture.cu->execute_instruction(mode.value().get(), *fixture.wave).succeeded());
      fixture.wave->set_exec(1);
      fixture.cu->write_vgpr(fixture.base(), 0, 1U);
      fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x3f800000U);
      const auto multiply_words =
          rdna3 ? rdna3::build_vop3(rdna3::kVMulF32Vop3, {.vdst = 2, .src0 = 256, .src1 = 257})
                : rdna4::build_vop3(rdna4::kVMulF32Vop3, {.vdst = 2, .src0 = 256, .src1 = 257});
      auto multiply = fixture.decoder->decode_window(multiply_words);
      ASSERT_TRUE(multiply.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(multiply.value().get(), *fixture.wave).succeeded());
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0), preserve ? 1U : 0U);
    }
}

TEST(ArithBridgeTest, RelativeScalarMovesKeepResolvedSelectorsAndIgnoreExec) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA3,
        ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5})
    for (unsigned operation = 0; operation < 5; ++operation)
      for (uint32_t m0 : {0U, 3U, 0x0403U, 0x00040003U})
        for (bool alias : {false, true}) {
          if (operation == 4 &&
              (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4))
            continue;
          const bool wide = operation == 1 || operation == 3;
          const unsigned width = wide ? 2 : 1;
          const unsigned source_base = 4, destination_base = alias ? 4 : 24;
          // Upstream MOVREL uses word offsets (B64 clears bit 0), ten-bit
          // split offsets, SGPR0 for missing sources and no write for missing
          // destinations.
          const unsigned source_offset = operation == 4 ? m0 & 1023 : wide ? m0 & ~1U : m0;
          const unsigned destination_offset = operation == 4 ? (m0 >> 16) & 1023 : source_offset;
          const unsigned indexed_source =
              source_base + (operation < 2 || operation == 4 ? source_offset : 0);
          const unsigned source = indexed_source + width <= 104 ? indexed_source : 0;
          const unsigned destination = destination_base + (operation >= 2 ? destination_offset : 0);
          const bool write = destination + width <= 104;
          ArithmeticPolicy policy{};
          policy.semantics =
              alias ? ArithmeticSemantics::Loose : ArithmeticSemantics::BitExactEmpiricalNoFpStatus;

          const bool rdna3 = arch == ROCJITSU_CODE_ARCH_RDNA3;
          BridgeFixture fixture(
              arch, arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ? 64 : 32,
              policy);
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
          fixture.wave->set_exec(0);
          fixture.wave->set_m0(m0);
          fixture.wave->write_scc(1);
          const uint32_t base = fixture.wave->sgpr_alloc().base;
          for (unsigned reg = 0; reg < 64; ++reg)
            fixture.cu->write_sgpr(base + reg, 0xabc00000U + reg);
          const auto words = arch == ROCJITSU_CODE_ARCH_CDNA3
                                 ? cdna3::build_sop1(cdna3::kSMovrelsB32Sop1 + operation,
                                                     {.ssrc0 = uint8_t(source_base),
                                                      .sdst = uint8_t(destination_base)})
                             : arch == ROCJITSU_CODE_ARCH_CDNA4
                                 ? cdna4::build_sop1(cdna4::kSMovrelsB32Sop1 + operation,
                                                     {.ssrc0 = uint8_t(source_base),
                                                      .sdst = uint8_t(destination_base)})
                             : arch == ROCJITSU_CODE_ARCH_CDNA5
                                 ? cdna5::build_sop1(cdna5::kSMovrelsB32Sop1 + operation,
                                                     {.ssrc0 = uint8_t(source_base),
                                                      .sdst = uint8_t(destination_base)})
                             : rdna3 ? rdna3::build_sop1(rdna3::kSMovrelsB32Sop1 + operation,
                                                         {.ssrc0 = uint8_t(source_base),
                                                          .sdst = uint8_t(destination_base)})
                                     : rdna4::build_sop1(rdna4::kSMovrelsB32Sop1 + operation,
                                                         {.ssrc0 = uint8_t(source_base),
                                                          .sdst = uint8_t(destination_base)});
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          for (unsigned reg = 0; reg < 64; ++reg)
            EXPECT_EQ(fixture.cu->read_sgpr(base + reg),
                      0xabc00000U + (write && reg >= destination && reg < destination + width
                                         ? source + reg - destination
                                         : reg));
          EXPECT_EQ(fixture.wave->exec(), 0U);
          EXPECT_EQ(fixture.wave->read_scc(), 1U);
        }
}

TEST(ArithBridgeTest, RelativeVectorMovesAndSwapsKeepAddressingMasksAndAliases) {
  const uint16_t vop1_ops[2][5] = {
      {rdna4::kVMovrelsB32Vop1, rdna4::kVMovreldB32Vop1, rdna4::kVMovrelsdB32Vop1,
       rdna4::kVMovrelsd2B32Vop1, rdna4::kVSwaprelB32Vop1},
      {rdna3::kVMovrelsB32Vop1, rdna3::kVMovreldB32Vop1, rdna3::kVMovrelsdB32Vop1,
       rdna3::kVMovrelsd2B32Vop1, rdna3::kVSwaprelB32Vop1}};
  const uint16_t vop3_ops[2][4] = {{rdna4::kVMovrelsB32Vop3, rdna4::kVMovreldB32Vop3,
                                    rdna4::kVMovrelsdB32Vop3, rdna4::kVMovrelsd2B32Vop3},
                                   {rdna3::kVMovrelsB32Vop3, rdna3::kVMovreldB32Vop3,
                                    rdna3::kVMovrelsdB32Vop3, rdna3::kVMovrelsd2B32Vop3}};
  const auto value = [](unsigned reg, unsigned lane) {
    return 0x80000000U + reg * 0x10000U + lane;
  };
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32U, 64U})
      for (bool e64 : {false, true})
        for (unsigned operation = 0; operation < (e64 ? 4U : 5U); ++operation)
          for (uint32_t m0 : {0U, 3U, 0x00040003U, 255U, 1023U, 1024U})
            for (bool alias : {false, true}) {
              SCOPED_TRACE(::testing::Message()
                           << "rdna3=" << rdna3 << " lanes=" << lanes << " e64=" << e64
                           << " operation=" << operation << " m0=" << m0 << " alias=" << alias);
              BridgeFixture fixture(rdna3, lanes);
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
              const unsigned source_base = 1, destination_base = alias ? 1 : 8;
              const unsigned source_offset = operation == 1 ? 0 : operation >= 3 ? m0 & 1023 : m0;
              const unsigned destination_offset = operation == 0   ? 0
                                                  : operation >= 3 ? (m0 >> 16) & 1023
                                                                   : m0;
              const bool source_valid =
                  source_offset <= 1023 && source_base + uint64_t(source_offset) < 256;
              const bool destination_valid = destination_offset <= 1023 &&
                                             destination_base + uint64_t(destination_offset) < 256;
              const unsigned source = source_valid ? source_base + source_offset : 0;
              const unsigned destination =
                  destination_valid ? destination_base + destination_offset : 0;
              const uint64_t exec_mask =
                  0xa55af00f0f0f00f3ULL & (lanes == 32 ? 0xffffffffULL : ~0ULL);
              const uint64_t write_mask = exec_mask & ~2ULL;
              fixture.wave->set_exec(exec_mask);
              fixture.wave->set_vgpr_write_mask(~2ULL);
              fixture.wave->set_m0(m0);
              for (unsigned reg : {0U, source_base, destination_base, source, destination})
                for (unsigned lane = 0; lane < lanes; ++lane)
                  fixture.cu->write_vgpr(fixture.base() + reg, lane, value(reg, lane));
              std::vector<uint32_t> words;
              const auto set_words = [&](const auto &encoding) {
                words.assign(encoding.begin(), encoding.end());
              };
              if (e64)
                set_words(rdna3 ? rdna3::build_vop3(vop3_ops[1][operation],
                                                    {.vdst = uint8_t(destination_base),
                                                     .src0 = uint16_t(256 + source_base)})
                                : rdna4::build_vop3(vop3_ops[0][operation],
                                                    {.vdst = uint8_t(destination_base),
                                                     .src0 = uint16_t(256 + source_base)}));
              else
                set_words(rdna3 ? rdna3::build_vop1(vop1_ops[1][operation],
                                                    {.src0 = uint16_t(256 + source_base),
                                                     .vdst = uint8_t(destination_base)})
                                : rdna4::build_vop1(vop1_ops[0][operation],
                                                    {.src0 = uint16_t(256 + source_base),
                                                     .vdst = uint8_t(destination_base)}));
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              fixture.recorder->writes.clear();
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              const bool writes = destination_valid && (operation != 4 || source_valid);
              uint64_t actual_write_mask = 0;
              for (const auto &event : fixture.recorder->writes)
                actual_write_mask |= event.lanes;
              EXPECT_EQ(actual_write_mask, writes ? write_mask : 0);
              for (unsigned reg : {0U, source_base, destination_base, source, destination})
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  uint32_t expected = value(reg, lane);
                  if (writes && ((write_mask >> lane) & 1)) {
                    if (reg == destination)
                      expected = value(source, lane);
                    if (operation == 4 && reg == source)
                      expected = value(destination, lane);
                  }
                  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + reg, lane), expected);
                }
            }
}

TEST(ArithBridgeTest, DppMoveHalfNegatesSelectedHalfAndPreservesTheOther) {
  for (bool gfx11 : {false, true}) {
    for (bool high : {false, true}) {
      BridgeFixture fixture(gfx11);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_exec(1);
      fixture.cu->write_vgpr(fixture.base(), 0, high ? 0x3c001234U : 0x12343c00U);
      fixture.cu->write_vgpr(fixture.base() + 2, 0, 0xdeadbeefU);
      const auto first =
          gfx11 ? rdna3::build_vop1(rdna3::kVMovB16Vop1,
                                    {.src0 = SRC_DPP, .vdst = uint8_t(2 + 128 * high)})
                : rdna4::build_vop1(rdna4::kVMovB16Vop1,
                                    {.src0 = SRC_DPP, .vdst = uint8_t(2 + 128 * high)});
      // Identity permutation, NEG, all rows/banks; vsrc0 selects the source
      // half.
      const std::array<uint32_t, 2> words{first[0], 0xff10e400U | (128U * high)};
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0), high ? 0xbc00beefU : 0xdeadbc00U);
    }
  }
}

TEST(ArithBridgeTest, RelativeSourceSelectionPrecedesDppRouting) {
  for (bool rdna3 : {false, true}) {
    BridgeFixture fixture(rdna3);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_m0(1);
    fixture.wave->set_exec(0xffffffffU);
    for (unsigned lane = 0; lane < 32; ++lane) {
      fixture.cu->write_vgpr(fixture.base() + 7, lane, 0x7000U + lane);
      fixture.cu->write_vgpr(fixture.base() + 8, lane, 0x8000U + lane);
    }
    const auto first = rdna3 ? rdna3::build_vop1(rdna3::kVMovrelsB32Vop1, {.src0 = 250, .vdst = 5})
                             : rdna4::build_vop1(rdna4::kVMovrelsB32Vop1, {.src0 = 250, .vdst = 5});
    // row_shr:1, NEG, all rows/banks, bound_ctrl=1, encoded source v7.
    const std::array<uint32_t, 2> words{first[0], 0xff191107U};
    auto instruction = fixture.decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    ASSERT_TRUE(
        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
    for (const auto &read : fixture.recorder->reads)
      EXPECT_NE(read.reg, fixture.base() + 7);
    for (unsigned lane = 0; lane < 32; ++lane)
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 5, lane),
                (lane % 16 ? 0x8000U + lane - 1 : 0) ^ 0x80000000U);
  }
}

TEST(ArithBridgeTest, DsPermutationsRespectGroupingCollisionsAliasesAndSourceObservations) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32U, 64U})
      for (bool observed : {false, true})
        for (bool suppress : {false, true})
          for (unsigned operation = 0; operation < (rdna3 ? 2U : 3U); ++operation)
            for (unsigned destination : {0U, 1U, 2U})
              for (uint16_t offset : {uint16_t(0), uint16_t(1), uint16_t(4), uint16_t(0xfffc)})
                for (uint64_t requested_exec : {0ULL, 0x80ff00015355500fULL, ~0ULL}) {
                  SCOPED_TRACE(::testing::Message()
                               << "rdna3=" << rdna3 << " lanes=" << lanes
                               << " operation=" << operation << " destination=" << destination
                               << " offset=" << offset << " observed=" << observed
                               << " suppress=" << suppress << " exec=" << requested_exec);
                  ArithmeticPolicy policy{};
                  policy.semantics = suppress ? ArithmeticSemantics::Loose
                                              : ArithmeticSemantics::BitExactEmpiricalNoFpStatus;

                  BridgeFixture fixture(rdna3, lanes, policy);
                  ASSERT_FALSE(::testing::Test::HasFatalFailure());
                  if (!observed)
                    fixture.cu->set_plugin_group(nullptr);
                  const uint64_t exec_mask = requested_exec & (lanes == 32 ? 0xffffffffULL : ~0ULL);
                  const uint64_t write_mask =
                      exec_mask & (suppress ? 0x3333333333333333ULL : ~0ULL);
                  fixture.wave->set_exec(exec_mask);
                  fixture.wave->set_vgpr_write_mask(write_mask);
                  std::array<uint32_t, 64> address{}, values{}, expected{};
                  for (unsigned lane = 0; lane < lanes; ++lane) {
                    address[lane] =
                        lane == 0 ? 0xfffffffcU
                                  : 4 * (operation == 0 ? lane % 5 : (lane * 13 + 37) % lanes) +
                                        (lane & 3);
                    values[lane] = 0xa0000000U + lane;
                    expected[lane] = destination == 0   ? address[lane]
                                     : destination == 1 ? values[lane]
                                                        : 0xdeadbeefU;
                    fixture.cu->write_vgpr(fixture.base(), lane, address[lane]);
                    fixture.cu->write_vgpr(fixture.base() + 1, lane, values[lane]);
                    fixture.cu->write_vgpr(fixture.base() + 2, lane, 0xdeadbeefU);
                  }
                  std::array<uint32_t, 2> words;
                  if (rdna3)
                    words = rdna3::build_ds(operation == 0 ? rdna3::kDsPermuteB32Ds
                                                           : rdna3::kDsBpermuteB32Ds,
                                            {.offset0 = uint8_t(offset),
                                             .offset1 = uint8_t(offset >> 8),
                                             .addr = 0,
                                             .data0 = 1,
                                             .vdst = uint8_t(destination)});
                  else {
                    const uint16_t opcodes[] = {rdna4::kDsPermuteB32Vds, rdna4::kDsBpermuteB32Vds,
                                                rdna4::kDsBpermuteFiB32Vds};
                    words = rdna4::build_vds(opcodes[operation], {.offset0 = uint8_t(offset),
                                                                  .offset1 = uint8_t(offset >> 8),
                                                                  .addr = 0,
                                                                  .data0 = 1,
                                                                  .vdst = uint8_t(destination)});
                  }
                  auto instruction = fixture.decoder->decode_window(words);
                  ASSERT_TRUE(instruction.succeeded());
                  if (observed) {
                    fixture.recorder->reads.clear();
                    fixture.recorder->writes.clear();
                  }
                  ASSERT_TRUE(
                      fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
                  std::array<uint32_t, 64> result{};
                  uint64_t consumed = 0;
                  const unsigned group = rdna3 ? 32 : lanes;
                  // Ordered scatter matches the existing collision rule. Gather
                  // uses original EXEC even when destination stores are
                  // suppressed.
                  for (unsigned lane = 0; lane < lanes; ++lane) {
                    if (!((exec_mask >> lane) & 1))
                      continue;
                    const unsigned selected =
                        lane / group * group + (uint32_t(address[lane] + offset) / 4) % group;
                    if (operation == 0) {
                      result[selected] = values[lane];
                      consumed |= 1ULL << lane;
                    } else if (operation == 2 || ((exec_mask >> selected) & 1)) {
                      result[lane] = values[selected];
                      consumed |= 1ULL << selected;
                    }
                  }
                  if (observed) {
                    uint64_t address_reads = 0, data_reads = 0, written = 0;
                    for (const auto &read : fixture.recorder->reads) {
                      EXPECT_EQ(read.bytes, 0xf);
                      if (read.reg == fixture.base())
                        address_reads |= read.lanes;
                      else if (read.reg == fixture.base() + 1)
                        data_reads |= read.lanes;
                      else
                        ADD_FAILURE() << "Unexpected source register";
                    }
                    for (const auto &write : fixture.recorder->writes) {
                      EXPECT_EQ(write.reg, fixture.base() + destination);
                      EXPECT_EQ(write.bytes, 0xf);
                      written |= write.lanes;
                    }
                    EXPECT_EQ(address_reads, exec_mask);
                    EXPECT_EQ(data_reads, consumed);
                    EXPECT_EQ(written, write_mask);
                  }
                  for (unsigned lane = 0; lane < lanes; ++lane) {
                    if ((write_mask >> lane) & 1)
                      expected[lane] = result[lane];
                    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, lane),
                              expected[lane]);
                  }
                }
}

TEST(ArithBridgeTest, DsSwizzleModesUseMatchingExecutionAndReadFootprints) {
  // Published basic patterns and FFT order, plus one-step masked rotations.
  // rj-arith tests these rules using ISA-derived reference lane mappings.
  constexpr unsigned fft[] = {0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
                              1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31};
  constexpr uint16_t offsets[] = {0x801b, 0x401f, 0xc020, 0xc420, 0xc02f, 0xe000, 0xe00f};
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32U, 64U})
      for (bool observed : {false, true})
        for (bool alias : {false, true})
          for (unsigned mode = 0; mode < std::size(offsets); ++mode)
            for (uint64_t requested_exec : {0ULL, 0x55550003f003ff01ULL, ~0ULL}) {
              SCOPED_TRACE(::testing::Message()
                           << "rdna3=" << rdna3 << " lanes=" << lanes << " mode=" << mode
                           << " observed=" << observed << " alias=" << alias);
              BridgeFixture fixture(rdna3, lanes);
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
              if (!observed)
                fixture.cu->set_plugin_group(nullptr);
              const uint64_t exec_mask = requested_exec & (lanes == 32 ? 0xffffffffULL : ~0ULL);
              const uint64_t write_mask = exec_mask & (alias ? ~2ULL : ~0ULL);
              fixture.wave->set_exec(exec_mask);
              fixture.wave->set_vgpr_write_mask(write_mask);
              for (unsigned lane = 0; lane < lanes; ++lane)
                fixture.cu->write_vgpr(fixture.base(), lane, 0xab000000U + lane);
              const uint8_t destination = alias ? 0 : 1;
              const uint16_t offset = offsets[mode];
              const auto words =
                  rdna3
                      ? rdna3::build_ds(rdna3::kDsSwizzleB32Ds, {.offset0 = uint8_t(offset),
                                                                 .offset1 = uint8_t(offset >> 8),
                                                                 .addr = 0,
                                                                 .vdst = destination})
                      : rdna4::build_vds(rdna4::kDsSwizzleB32Vds, {.offset0 = uint8_t(offset),
                                                                   .offset1 = uint8_t(offset >> 8),
                                                                   .addr = 0,
                                                                   .vdst = destination});
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              if (observed) {
                fixture.recorder->reads.clear();
                fixture.recorder->writes.clear();
              }
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              uint64_t consumed = 0;
              std::array<uint32_t, 64> expected{};
              for (unsigned lane = 0; lane < lanes; ++lane) {
                const unsigned local = lane % 32;
                const unsigned source =
                    lane / 32 * 32 + (mode == 0   ? (local / 4 * 4 + 3 - local % 4)
                                      : mode == 1 ? local ^ 16
                                      : mode == 2 ? (local + 1) % 32
                                      : mode == 3 ? (local + 31) % 32
                                      : mode == 4 ? (local & 15) | ((local + 1) & 16)
                                      : mode == 5 ? fft[local]
                                                  : (fft[local] / 16) | (local & 15));
                const bool readable = (exec_mask >> source) & 1;
                if (((exec_mask >> lane) & 1) && readable)
                  consumed |= 1ULL << source;
                expected[lane] = (write_mask >> lane) & 1 ? (readable ? 0xab000000U + source : 0)
                                 : alias                  ? 0xab000000U + lane
                                                          : 0x3c003c00U;
              }
              if (observed) {
                uint64_t read = 0, written = 0;
                for (const auto &event : fixture.recorder->reads) {
                  EXPECT_EQ(event.reg, fixture.base());
                  read |= event.lanes;
                }
                for (const auto &event : fixture.recorder->writes)
                  written |= event.lanes;
                EXPECT_EQ(read, consumed);
                EXPECT_EQ(written, write_mask);
              }
              for (unsigned lane = 0; lane < lanes; ++lane)
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, lane),
                          expected[lane]);
            }
}

TEST(ArithBridgeTest, PendingRegisterErrorPreventsLaneTransferCommit) {
  for (int operation = 0; operation < 3; ++operation) {
    BridgeFixture fixture(false, 32);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_exec(1);
    const uint32_t sgpr = fixture.wave->sgpr_alloc().base;
    fixture.cu->write_sgpr(sgpr, 0xdeadbeefU);
    fixture.cu->write_vgpr(fixture.base(), 0, 0x12345678U);
    rdna4::Operand scalar(32, rdna4::OperandType::OPR_SRC, 0);
    rdna4::Operand vector(32, rdna4::OperandType::OPR_SRC_VGPR, 256);
    rdna4::Operand other(32, rdna4::OperandType::OPR_SRC_VGPR, 257);
    fixture.wave->report_instruction_execution_error(
        InstructionExecutionError::InvalidRegisterAccess);
    if (operation == 0) {
      arith::execute_read_lane<false>(*fixture.wave, 0, scalar, vector);
    } else if (operation == 1) {
      arith::execute_write_lane(*fixture.wave, 0, vector, scalar, scalar);
    } else {
      arith::execute_swap<false>(*fixture.wave, 0, vector, other);
    }
    EXPECT_EQ(fixture.cu->read_sgpr(sgpr), 0xdeadbeefU);
    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), 0x12345678U);
    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 0), 0x3c003c00U);
    EXPECT_TRUE(fixture.recorder->writes.empty());
  }
}

TEST(ArithBridgeTest, LaneTransfersIgnoreVectorMasksAndObserveSelectedSource) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32U, 64U})
      for (unsigned operation = 0; operation < 4; ++operation)
        for (bool empty : {false, true}) {
          BridgeFixture fixture(rdna3, lanes);
          ASSERT_FALSE(::testing::Test::HasFatalFailure());
          fixture.wave->set_exec(empty ? 0 : 1ULL << (lanes - 2));
          fixture.wave->set_vgpr_write_mask(0);
          for (unsigned lane = 0; lane < lanes; ++lane)
            fixture.cu->write_vgpr(fixture.base(), lane, 0xabc00000U + lane);
          const uint32_t scalar = fixture.wave->sgpr_alloc().base;
          fixture.cu->write_sgpr(scalar, 0x12345678U);
          fixture.cu->write_sgpr(scalar + 1, lanes + 7);
          std::vector<uint32_t> words;
          const auto set_words = [&](const auto &value) {
            words.assign(value.begin(), value.end());
          };
          if (operation == 0)
            set_words(
                rdna3 ? rdna3::build_vop1(rdna3::kVReadfirstlaneB32Vop1, {.src0 = 256, .vdst = 4})
                      : rdna4::build_vop1(rdna4::kVReadfirstlaneB32Vop1, {.src0 = 256, .vdst = 4}));
          else if (operation == 1)
            set_words(
                rdna3 ? rdna3::build_vop3(rdna3::kVReadfirstlaneB32Vop3, {.vdst = 4, .src0 = 256})
                      : rdna4::build_vop3(rdna4::kVReadfirstlaneB32Vop3, {.vdst = 4, .src0 = 256}));
          else if (operation == 2)
            set_words(rdna3 ? rdna3::build_vop3(rdna3::kVReadlaneB32Vop3,
                                                {.vdst = 4, .src0 = 256, .src1 = 1})
                            : rdna4::build_vop3(rdna4::kVReadlaneB32Vop3,
                                                {.vdst = 4, .src0 = 256, .src1 = 1}));
          else
            set_words(rdna3 ? rdna3::build_vop3(rdna3::kVWritelaneB32Vop3,
                                                {.vdst = 0, .src0 = 0, .src1 = 1})
                            : rdna4::build_vop3(rdna4::kVWritelaneB32Vop3,
                                                {.vdst = 0, .src0 = 0, .src1 = 1}));
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          fixture.recorder->reads.clear();
          fixture.recorder->writes.clear();
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          if (operation < 3) {
            const unsigned selected = operation == 2 ? 7 : empty ? 0 : lanes - 2;
            ASSERT_EQ(fixture.recorder->reads.size(), 1U);
            EXPECT_EQ(fixture.recorder->reads[0].lanes, 1ULL << selected);
            EXPECT_TRUE(fixture.recorder->writes.empty());
            EXPECT_EQ(fixture.cu->read_sgpr(scalar + 4), 0xabc00000U + selected);
          } else {
            EXPECT_TRUE(fixture.recorder->reads.empty());
            ASSERT_EQ(fixture.recorder->writes.size(), 1U);
            EXPECT_EQ(fixture.recorder->writes[0].lanes, 1ULL << 7);
            for (unsigned lane = 0; lane < lanes; ++lane)
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                        lane == 7 ? 0x12345678U : 0xabc00000U + lane);
          }
        }
}

TEST(ArithBridgeTest, LaneSwapsPreserveOtherHalvesAndSnapshotAliases) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32U, 64U})
      for (bool half : {false, true})
        for (bool alias : {false, true})
          for (unsigned halves = 0; halves < (half ? 4U : 1U); ++halves) {
            BridgeFixture fixture(rdna3, lanes);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            const uint64_t exec_mask =
                0x9249249249249249ULL & (lanes == 32 ? 0xffffffffULL : ~0ULL);
            const uint64_t write_mask = exec_mask & ~1ULL;
            fixture.wave->set_exec(exec_mask);
            fixture.wave->set_vgpr_write_mask(~1ULL);
            for (unsigned lane = 0; lane < lanes; ++lane) {
              fixture.cu->write_vgpr(fixture.base(), lane, 0xa0001000U + 0x10001U * lane);
              fixture.cu->write_vgpr(fixture.base() + 1, lane, 0xb0002000U + 0x10001U * lane);
            }
            const uint16_t src = 256 + (alias ? 0 : 1) + ((halves & 1) ? 128 : 0);
            const uint8_t dst = halves & 2 ? 128 : 0;
            const auto words(
                rdna3 ? rdna3::build_vop1(half ? rdna3::kVSwapB16Vop1 : rdna3::kVSwapB32Vop1,
                                          {.src0 = src, .vdst = dst})
                      : rdna4::build_vop1(half ? rdna4::kVSwapB16Vop1 : rdna4::kVSwapB32Vop1,
                                          {.src0 = src, .vdst = dst}));
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .succeeded());
            for (unsigned lane = 0; lane < lanes; ++lane) {
              uint32_t d = 0xa0001000U + 0x10001U * lane;
              uint32_t a = alias ? d : 0xb0002000U + 0x10001U * lane;
              if ((write_mask >> lane) & 1) {
                if (!half) {
                  std::swap(d, a);
                } else {
                  const unsigned ds = halves & 2 ? 16 : 0, as = halves & 1 ? 16 : 0;
                  const uint32_t old_d = (d >> ds) & 0xffff, old_a = (a >> as) & 0xffff;
                  d = (d & ~(0xffffU << ds)) | (old_a << ds);
                  if (alias)
                    d = (d & ~(0xffffU << as)) | (old_d << as);
                  else
                    a = (a & ~(0xffffU << as)) | (old_d << as);
                }
              }
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), d);
              if (!alias) {
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, lane), a);
              }
            }
          }
}

TEST(ArithBridgeTest, LanePermutationsPreserveInactiveSourcesAndDestinationMasks) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32U, 64U})
      for (bool observed : {false, true})
        for (bool cross : {false, true})
          for (bool variable : {false, true})
            for (unsigned modifiers = 0; modifiers < 4; ++modifiers)
              for (unsigned destination : {0U, 1U, 2U}) {
                if (rdna3 && variable)
                  continue;
                BridgeFixture fixture(rdna3, lanes);
                ASSERT_FALSE(::testing::Test::HasFatalFailure());
                if (!observed)
                  fixture.cu->set_plugin_group(nullptr);
                const uint64_t exec_mask =
                    0xf0ff00f30ff000f3ULL & (lanes == 32 ? 0xffffffffULL : ~0ULL);
                const uint64_t write_mask = exec_mask & (observed ? ~2ULL : ~0ULL);
                fixture.wave->set_exec(exec_mask);
                fixture.wave->set_vgpr_write_mask(write_mask);
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  fixture.cu->write_vgpr(fixture.base(), lane, 0xabc00000U + lane);
                  fixture.cu->write_vgpr(fixture.base() + 1, lane, 15 - (lane & 15));
                  fixture.cu->write_vgpr(fixture.base() + 2, lane, 0xdef00000U + lane);
                }
                fixture.cu->write_sgpr(fixture.wave->sgpr_alloc().base, 0x89abcdefU);
                fixture.cu->write_sgpr(fixture.wave->sgpr_alloc().base + 1, 0x01234567U);
                std::vector<uint32_t> words;
                const auto set_words = [&](const auto &value) {
                  words.assign(value.begin(), value.end());
                };
                if (rdna3)
                  set_words(rdna3::build_vop3(cross ? rdna3::kVPermlanex16B32Vop3
                                                    : rdna3::kVPermlane16B32Vop3,
                                              {.vdst = static_cast<uint8_t>(destination),
                                               .op_sel = static_cast<uint8_t>(modifiers),
                                               .src0 = 256,
                                               .src1 = 0,
                                               .src2 = 1}));
                else
                  set_words(rdna4::build_vop3(
                      variable
                          ? (cross ? rdna4::kVPermlanex16VarB32Vop3 : rdna4::kVPermlane16VarB32Vop3)
                          : (cross ? rdna4::kVPermlanex16B32Vop3 : rdna4::kVPermlane16B32Vop3),
                      {.vdst = static_cast<uint8_t>(destination),
                       .opsel = static_cast<uint8_t>(modifiers),
                       .src0 = 256,
                       .src1 = static_cast<uint16_t>(variable ? 257 : 0),
                       .src2 = static_cast<uint16_t>(variable ? 0 : 1)}));
                auto instruction = fixture.decoder->decode_window(words);
                ASSERT_TRUE(instruction.succeeded());
                if (observed)
                  fixture.recorder->writes.clear();
                ASSERT_TRUE(
                    fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                        .succeeded());
                uint64_t expected_writes = 0;
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  uint32_t expected = destination == 0   ? 0xabc00000U + lane
                                      : destination == 1 ? 15 - (lane & 15)
                                                         : 0xdef00000U + lane;
                  const unsigned from = ((lane & ~15U) ^ (cross ? 16 : 0)) | (15 - (lane & 15));
                  if ((write_mask >> lane) & 1) {
                    if ((modifiers & 1) || ((exec_mask >> from) & 1)) {
                      expected = 0xabc00000U + from;
                      expected_writes |= 1ULL << lane;
                    } else if (modifiers & 2) {
                      expected = 0;
                      expected_writes |= 1ULL << lane;
                    }
                  }
                  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, lane), expected);
                }
                if (observed) {
                  uint64_t written = 0;
                  for (const auto &write : fixture.recorder->writes)
                    written |= write.lanes;
                  EXPECT_EQ(written, expected_writes);
                }
              }
}

TEST(ArithBridgeTest, ConditionalObservesOnlySelectedSources) {
  BridgeFixture fixture(false);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.wave->set_exec(3);
  fixture.wave->set_vcc_mask(2);
  auto words = rdna4::build_vop2(rdna4::kVCndmaskB32Vop2, {.src0 = 256, .vsrc1 = 1, .vdst = 2});
  auto instruction = fixture.decoder->decode_window(words);
  ASSERT_TRUE(instruction.succeeded());
  ASSERT_TRUE(
      fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
  ASSERT_EQ(fixture.recorder->reads.size(), 2U);
  EXPECT_EQ(fixture.recorder->reads[0].lanes, 1U);
  EXPECT_EQ(fixture.recorder->reads[1].lanes, 2U);
  // A uniform source is observed only when at least one active lane selects it.
  for (uint32_t lanes : {32U, 64U}) {
    BridgeFixture scalar_fixture(false, lanes);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    scalar_fixture.wave->set_exec(1);
    auto scalar_words =
        rdna4::build_vop2(rdna4::kVCndmaskB32Vop2, {.src0 = 8, .vsrc1 = 1, .vdst = 2});
    auto scalar_instruction = scalar_fixture.decoder->decode_window(scalar_words);
    ASSERT_TRUE(scalar_instruction.succeeded());
    for (uint64_t vcc : {0ULL, 1ULL}) {
      scalar_fixture.wave->set_vcc_mask(vcc);
      scalar_fixture.recorder->scalar_reads.clear();
      ASSERT_TRUE(scalar_fixture.cu
                      ->execute_instruction(scalar_instruction.value().get(), *scalar_fixture.wave)
                      .succeeded());
      EXPECT_EQ(scalar_fixture.recorder->scalar_reads,
                vcc ? std::vector<uint32_t>{} : std::vector<uint32_t>{8});
    }
  }
}

TEST(ArithBridgeTest, PermlaneObservesOnlySelectedSourceLanes) {
  BridgeFixture fixture(false);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.wave->set_exec(1);
  auto words = rdna4::build_vop3(rdna4::kVPermlane16B32Vop3,
                                 {.vdst = 2, .src0 = 256, .src1 = 128, .src2 = 128});
  auto instruction = fixture.decoder->decode_window(words);
  ASSERT_TRUE(instruction.succeeded());
  ASSERT_TRUE(
      fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
  ASSERT_EQ(fixture.recorder->reads.size(), 1U);
  EXPECT_EQ(fixture.recorder->reads[0].lanes, 1U);
  ASSERT_EQ(fixture.recorder->writes.size(), 1U);
  EXPECT_EQ(fixture.recorder->writes[0].lanes, 1U);
}

TEST(ArithBridgeTest, Permlane64ReadsOppositeHalfEvenWhenInactive) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32U, 64U})
      for (bool alias : {false, true}) {
        BridgeFixture fixture(rdna3, lanes);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        fixture.wave->set_exec(5);
        for (unsigned lane = 0; lane < lanes; ++lane)
          fixture.cu->write_vgpr(fixture.base(), lane, 0xabc00000U + lane);
        const uint8_t destination = alias ? 0 : 1;
        const auto words(rdna3 ? rdna3::build_vop1(rdna3::kVPermlane64B32Vop1,
                                                   {.src0 = 256, .vdst = destination})
                               : rdna4::build_vop1(rdna4::kVPermlane64B32Vop1,
                                                   {.src0 = 256, .vdst = destination}));
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        fixture.recorder->reads.clear();
        fixture.recorder->writes.clear();
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        if (lanes == 32) {
          EXPECT_TRUE(fixture.recorder->reads.empty());
          EXPECT_TRUE(fixture.recorder->writes.empty());
        } else {
          ASSERT_EQ(fixture.recorder->reads.size(), 1U);
          EXPECT_EQ(fixture.recorder->reads[0].lanes, 5ULL << 32);
        }
        for (unsigned lane = 0; lane < lanes; ++lane)
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, lane),
                    lanes == 64 && (lane == 0 || lane == 2) ? 0xabc00000U + (lane ^ 32)
                    : alias                                 ? 0xabc00000U + lane
                                                            : 0x3c003c00U);
      }
}

TEST(ArithBridgeTest, DppConditionalAppliesIndependentSignsAfterRouting) {
  for (bool rdna3 : {false, true}) {
    for (uint32_t lanes : {32U, 64U}) {
      BridgeFixture fixture(rdna3, lanes);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_exec(3);
      fixture.wave->set_vcc_mask(2);
      fixture.cu->write_vgpr(fixture.base(), 0, 0x3f000000);
      fixture.cu->write_vgpr(fixture.base() + 1, 1, 0xbf000000);
      // Decode through the architecture-specific machine layout.
      if (rdna3) {
        rdna3::Vop2VopDpp16MachineInst raw{};
        raw.op = rdna3::kVCndmaskB32Vop2;
        raw.src0 = SRC_DPP;
        raw.vsrc0 = 0;
        raw.vsrc1 = 1;
        raw.vdst = 2;
        raw.dpp_ctrl = 0xe4;
        raw.row_mask = raw.bank_mask = 0xf;
        raw.src0_neg = 1;
        raw.src1_abs = 1;
        rdna3::VCndmaskB32Vop2 instruction(reinterpret_cast<const rdna3::MachineInst *>(&raw));
        instruction.execute_impl(*fixture.wave);
      } else {
        rdna4::Vop2VopDpp16MachineInst raw{};
        raw.op = rdna4::kVCndmaskB32Vop2;
        raw.src0 = SRC_DPP;
        raw.vsrc0 = 0;
        raw.vsrc1 = 1;
        raw.vdst = 2;
        raw.dpp_ctrl = 0xe4;
        raw.row_mask = raw.bank_mask = 0xf;
        raw.src0_neg = 1;
        raw.src1_abs = 1;
        rdna4::VCndmaskB32Vop2 instruction(reinterpret_cast<const rdna4::MachineInst *>(&raw));
        instruction.execute_impl(*fixture.wave);
      }
      EXPECT_FALSE(fixture.wave->instruction_execution_failed());
      uint64_t source1_lanes = 0;
      for (const auto &read : fixture.recorder->reads) {
        if (read.reg == 1) {
          source1_lanes |= read.lanes;
        }
      }
      EXPECT_EQ(source1_lanes, 2u);

      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0), 0xbf000000U);
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 1), 0x3f000000U);
    }
  }
}

TEST(ArithBridgeTest, ExecBooleanUpdatesPreserveWidthsOrderingAndAliasedDestinations) {
  const uint16_t opcodes[5][2][12] = {
      {{rdna4::kSAndSaveExecB32Sop1, rdna4::kSOrSaveExecB32Sop1, rdna4::kSXorSaveExecB32Sop1,
        rdna4::kSNandSaveExecB32Sop1, rdna4::kSNorSaveExecB32Sop1, rdna4::kSXnorSaveExecB32Sop1,
        rdna4::kSAndNot0SaveExecB32Sop1, rdna4::kSOrNot0SaveExecB32Sop1,
        rdna4::kSAndNot1SaveExecB32Sop1, rdna4::kSOrNot1SaveExecB32Sop1,
        rdna4::kSAndNot0WrExecB32Sop1, rdna4::kSAndNot1WrExecB32Sop1},
       {rdna4::kSAndSaveExecB64Sop1, rdna4::kSOrSaveExecB64Sop1, rdna4::kSXorSaveExecB64Sop1,
        rdna4::kSNandSaveExecB64Sop1, rdna4::kSNorSaveExecB64Sop1, rdna4::kSXnorSaveExecB64Sop1,
        rdna4::kSAndNot0SaveExecB64Sop1, rdna4::kSOrNot0SaveExecB64Sop1,
        rdna4::kSAndNot1SaveExecB64Sop1, rdna4::kSOrNot1SaveExecB64Sop1,
        rdna4::kSAndNot0WrExecB64Sop1, rdna4::kSAndNot1WrExecB64Sop1}},
      {{rdna3::kSAndSaveExecB32Sop1, rdna3::kSOrSaveExecB32Sop1, rdna3::kSXorSaveExecB32Sop1,
        rdna3::kSNandSaveExecB32Sop1, rdna3::kSNorSaveExecB32Sop1, rdna3::kSXnorSaveExecB32Sop1,
        rdna3::kSAndNot0SaveExecB32Sop1, rdna3::kSOrNot0SaveExecB32Sop1,
        rdna3::kSAndNot1SaveExecB32Sop1, rdna3::kSOrNot1SaveExecB32Sop1,
        rdna3::kSAndNot0WrExecB32Sop1, rdna3::kSAndNot1WrExecB32Sop1},
       {rdna3::kSAndSaveExecB64Sop1, rdna3::kSOrSaveExecB64Sop1, rdna3::kSXorSaveExecB64Sop1,
        rdna3::kSNandSaveExecB64Sop1, rdna3::kSNorSaveExecB64Sop1, rdna3::kSXnorSaveExecB64Sop1,
        rdna3::kSAndNot0SaveExecB64Sop1, rdna3::kSOrNot0SaveExecB64Sop1,
        rdna3::kSAndNot1SaveExecB64Sop1, rdna3::kSOrNot1SaveExecB64Sop1,
        rdna3::kSAndNot0WrExecB64Sop1, rdna3::kSAndNot1WrExecB64Sop1}},
      {{},
       {cdna3::kSAndSaveExecB64Sop1, cdna3::kSOrSaveExecB64Sop1, cdna3::kSXorSaveExecB64Sop1,
        cdna3::kSNandSaveExecB64Sop1, cdna3::kSNorSaveExecB64Sop1, cdna3::kSXnorSaveExecB64Sop1,
        cdna3::kSAndn1SaveExecB64Sop1, cdna3::kSOrn1SaveExecB64Sop1, cdna3::kSAndn2SaveExecB64Sop1,
        cdna3::kSOrn2SaveExecB64Sop1, cdna3::kSAndn1WrExecB64Sop1, cdna3::kSAndn2WrExecB64Sop1}},
      {{},
       {cdna4::kSAndSaveExecB64Sop1, cdna4::kSOrSaveExecB64Sop1, cdna4::kSXorSaveExecB64Sop1,
        cdna4::kSNandSaveExecB64Sop1, cdna4::kSNorSaveExecB64Sop1, cdna4::kSXnorSaveExecB64Sop1,
        cdna4::kSAndn1SaveExecB64Sop1, cdna4::kSOrn1SaveExecB64Sop1, cdna4::kSAndn2SaveExecB64Sop1,
        cdna4::kSOrn2SaveExecB64Sop1, cdna4::kSAndn1WrExecB64Sop1, cdna4::kSAndn2WrExecB64Sop1}},
      {{cdna5::kSAndSaveExecB32Sop1, cdna5::kSOrSaveExecB32Sop1, cdna5::kSXorSaveExecB32Sop1,
        cdna5::kSNandSaveExecB32Sop1, cdna5::kSNorSaveExecB32Sop1, cdna5::kSXnorSaveExecB32Sop1,
        cdna5::kSAndNot0SaveExecB32Sop1, cdna5::kSOrNot0SaveExecB32Sop1,
        cdna5::kSAndNot1SaveExecB32Sop1, cdna5::kSOrNot1SaveExecB32Sop1,
        cdna5::kSAndNot0WrExecB32Sop1, cdna5::kSAndNot1WrExecB32Sop1},
       {cdna5::kSAndSaveExecB64Sop1, cdna5::kSOrSaveExecB64Sop1, cdna5::kSXorSaveExecB64Sop1,
        cdna5::kSNandSaveExecB64Sop1, cdna5::kSNorSaveExecB64Sop1, cdna5::kSXnorSaveExecB64Sop1,
        cdna5::kSAndNot0SaveExecB64Sop1, cdna5::kSOrNot0SaveExecB64Sop1,
        cdna5::kSAndNot1SaveExecB64Sop1, cdna5::kSOrNot1SaveExecB64Sop1,
        cdna5::kSAndNot0WrExecB64Sop1, cdna5::kSAndNot1WrExecB64Sop1}}};
  const rj_code_arch_t architectures[] = {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                                          ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4,
                                          ROCJITSU_CODE_ARCH_CDNA5};
  for (int architecture = 0; architecture < 5; ++architecture)
    for (uint32_t lanes : {32U, 64U})
      for (bool wide : {false, true})
        for (bool loose : {false, true})
          for (unsigned op = 0; op < 12; ++op)
            for (uint8_t destination : {uint8_t(0), uint8_t(2), uint8_t(126), uint8_t(127)}) {
              const auto arch = architectures[architecture];
              const bool rdna3 = arch == ROCJITSU_CODE_ARCH_RDNA3;
              if ((architecture == 2 || architecture == 3) && (!wide || lanes != 64))
                continue;
              if (architecture == 4 && lanes != 32)
                continue;
              if (wide && destination == 127)
                continue;
              SCOPED_TRACE(::testing::Message() << "rdna3=" << rdna3 << " lanes=" << lanes
                                                << " wide=" << wide << " loose=" << loose << " op="
                                                << op << " destination=" << unsigned(destination));
              ArithmeticPolicy policy{};
              policy.semantics = loose ? ArithmeticSemantics::Loose
                                       : ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
              BridgeFixture fixture(arch, lanes, policy);
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
              const uint32_t base = fixture.wave->sgpr_alloc().base;
              // Alternate zero, high-only and mixed EXEC to exercise the width of SCC.
              const uint64_t old_exec = op % 3 == 0   ? 0
                                        : op % 3 == 1 ? 0xa55a123400000000ULL
                                                      : 0x876543210f0ff0f0ULL;
              const uint64_t input = 0x02468ace55aa0ff0ULL;
              fixture.wave->set_exec_raw(old_exec);
              fixture.wave->write_scc(1);
              fixture.cu->write_sgpr(base, uint32_t(input));
              fixture.cu->write_sgpr(base + 1, uint32_t(input >> 32));
              const auto opcode = opcodes[architecture][wide][op];
              const auto words = arch == ROCJITSU_CODE_ARCH_CDNA3
                                     ? cdna3::build_sop1(opcode, {.ssrc0 = 0, .sdst = destination})
                                 : arch == ROCJITSU_CODE_ARCH_CDNA4
                                     ? cdna4::build_sop1(opcode, {.ssrc0 = 0, .sdst = destination})
                                 : arch == ROCJITSU_CODE_ARCH_CDNA5
                                     ? cdna5::build_sop1(opcode, {.ssrc0 = 0, .sdst = destination})
                                 : rdna3
                                     ? rdna3::build_sop1(opcode, {.ssrc0 = 0, .sdst = destination})
                                     : rdna4::build_sop1(opcode, {.ssrc0 = 0, .sdst = destination});
              auto instruction = fixture.decoder->decode_window(words);
              // The SREG destination class excludes EXEC; preserve that decoder
              // rejection instead of manufacturing an unsupported register
              // alias.
              if (destination >= 126) {
                EXPECT_FALSE(instruction.succeeded());
                continue;
              }
              ASSERT_TRUE(instruction.succeeded());
              fixture.recorder->scalar_reads.clear();
              fixture.recorder->scalar_writes.clear();
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              uint64_t expected = 0;
              switch (op) {
              case 0:
                expected = old_exec & input;
                break;
              case 1:
                expected = old_exec | input;
                break;
              case 2:
                expected = old_exec ^ input;
                break;
              case 3:
                expected = ~(old_exec & input);
                break;
              case 4:
                expected = ~(old_exec | input);
                break;
              case 5:
                expected = ~(old_exec ^ input);
                break;
              case 6:
              case 10:
                expected = old_exec & ~input;
                break;
              case 7:
                expected = old_exec | ~input;
                break;
              case 8:
              case 11:
                expected = input & ~old_exec;
                break;
              case 9:
                expected = input | ~old_exec;
                break;
              }
              if (!wide)
                expected = uint32_t(expected);
              const uint64_t saved = op >= 10 ? expected : wide ? old_exec : uint32_t(old_exec);
              uint64_t expected_raw = expected;
              if (!wide)
                expected_raw |= old_exec & 0xffffffff00000000ULL;
              EXPECT_EQ(fixture.wave->exec_raw(), expected_raw);
              EXPECT_EQ(fixture.wave->read_scc(), expected != 0);
              std::vector<uint32_t> reads{base};
              if (wide)
                reads.push_back(base + 1);
              EXPECT_EQ(fixture.recorder->scalar_reads, reads);
              ASSERT_EQ(fixture.recorder->scalar_writes.size(), 1U);
              if (destination < 126) {
                EXPECT_EQ(fixture.cu->read_sgpr(base + destination), uint32_t(saved));
                if (wide) {
                  EXPECT_EQ(fixture.cu->read_sgpr(base + destination + 1), uint32_t(saved >> 32));
                }
              }
            }
}

TEST(ArithBridgeTest, ScalarMovesIgnoreExecAndUntakenMovesHaveNoRegisterAccesses) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA3,
        ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5})
    for (unsigned kind = 0; kind < 6; ++kind)
      for (bool scc : {false, true}) {
        const bool rdna3 = arch == ROCJITSU_CODE_ARCH_RDNA3;
        BridgeFixture fixture(
            arch, arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ? 64 : 32,
            {});
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        const bool conditional = kind & 1;
        const bool wide = kind == 2 || kind == 3;
        const bool immediate = kind >= 4;
        const uint32_t base = fixture.wave->sgpr_alloc().base;
        fixture.wave->set_exec(0);
        fixture.wave->write_scc(scc);
        fixture.cu->write_sgpr(base, 0x12345678);
        fixture.cu->write_sgpr(base + 1, 0x87654321);
        fixture.cu->write_sgpr(base + 2, 0xabcdef01);
        fixture.cu->write_sgpr(base + 3, 0x98765432);
        const uint16_t opcodes[5][6] = {
            {rdna4::kSMovB32Sop1, rdna4::kSCmovB32Sop1, rdna4::kSMovB64Sop1, rdna4::kSCmovB64Sop1,
             rdna4::kSMovkI32Sopk, rdna4::kSCmovkI32Sopk},
            {rdna3::kSMovB32Sop1, rdna3::kSCmovB32Sop1, rdna3::kSMovB64Sop1, rdna3::kSCmovB64Sop1,
             rdna3::kSMovkI32Sopk, rdna3::kSCmovkI32Sopk},
            {cdna3::kSMovB32Sop1, cdna3::kSCmovB32Sop1, cdna3::kSMovB64Sop1, cdna3::kSCmovB64Sop1,
             cdna3::kSMovkI32Sopk, cdna3::kSCmovkI32Sopk},
            {cdna4::kSMovB32Sop1, cdna4::kSCmovB32Sop1, cdna4::kSMovB64Sop1, cdna4::kSCmovB64Sop1,
             cdna4::kSMovkI32Sopk, cdna4::kSCmovkI32Sopk},
            {cdna5::kSMovB32Sop1, cdna5::kSCmovB32Sop1, cdna5::kSMovB64Sop1, cdna5::kSCmovB64Sop1,
             cdna5::kSMovkI32Sopk, cdna5::kSCmovkI32Sopk}};
        const uint16_t opcode = opcodes[arch == ROCJITSU_CODE_ARCH_CDNA3   ? 2
                                        : arch == ROCJITSU_CODE_ARCH_CDNA4 ? 3
                                        : arch == ROCJITSU_CODE_ARCH_CDNA5 ? 4
                                                                           : int(rdna3)][kind];
        const auto words =
            arch == ROCJITSU_CODE_ARCH_CDNA3
                ? (immediate ? cdna3::build_sopk(opcode, {.simm16 = 0x8123, .sdst = 0})
                             : cdna3::build_sop1(opcode, {.ssrc0 = 2, .sdst = 0}))
            : arch == ROCJITSU_CODE_ARCH_CDNA4
                ? (immediate ? cdna4::build_sopk(opcode, {.simm16 = 0x8123, .sdst = 0})
                             : cdna4::build_sop1(opcode, {.ssrc0 = 2, .sdst = 0}))
            : arch == ROCJITSU_CODE_ARCH_CDNA5
                ? (immediate ? cdna5::build_sopk(opcode, {.simm16 = 0x8123, .sdst = 0})
                             : cdna5::build_sop1(opcode, {.ssrc0 = 2, .sdst = 0}))
            : immediate ? (rdna3 ? rdna3::build_sopk(opcode, {.simm16 = 0x8123, .sdst = 0})
                                 : rdna4::build_sopk(opcode, {.simm16 = 0x8123, .sdst = 0}))
                        : (rdna3 ? rdna3::build_sop1(opcode, {.ssrc0 = 2, .sdst = 0})
                                 : rdna4::build_sop1(opcode, {.ssrc0 = 2, .sdst = 0}));
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        fixture.recorder->scalar_reads.clear();
        fixture.recorder->scalar_writes.clear();
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        const bool writes = !conditional || scc;
        std::vector<uint32_t> reads;
        if (writes && !immediate) {
          reads.push_back(base + 2);
          if (wide)
            reads.push_back(base + 3);
        }
        EXPECT_EQ(fixture.recorder->scalar_reads, reads);
        EXPECT_EQ(fixture.recorder->scalar_writes.size(), writes ? 1U : 0U);
        EXPECT_EQ(fixture.cu->read_sgpr(base),
                  writes ? (immediate ? 0xffff8123U : 0xabcdef01U) : 0x12345678U);
        EXPECT_EQ(fixture.cu->read_sgpr(base + 1), writes && wide ? 0x98765432U : 0x87654321U);
        EXPECT_EQ(fixture.wave->read_scc(), scc);
      }
}

TEST(ArithBridgeTest, VectorMovesAndSelectionsPreservePayloadsHalvesAndMaskedAliases) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32U, 64U})
      for (bool half : {false, true})
        for (bool conditional : {false, true})
          for (bool high : {false, true})
            for (unsigned modifiers : {0U, 1U, 2U, 3U})
              for (bool inactive : {false, true}) {
                BridgeFixture fixture(rdna3, lanes);
                ASSERT_FALSE(::testing::Test::HasFatalFailure());
                const uint64_t exec_mask = inactive ? 0 : 5ULL | (1ULL << (lanes - 1));
                const uint64_t condition = 1ULL << (lanes - 1);
                fixture.wave->set_exec(exec_mask);
                fixture.wave->set_vcc_mask(condition);
                const uint32_t sb = fixture.wave->sgpr_alloc().base;
                fixture.cu->write_sgpr(sb, uint32_t(condition));
                fixture.cu->write_sgpr(sb + 1, uint32_t(condition >> 32));
                std::array<uint32_t, 64> a{}, b{};
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  // Fixed integer mixing retains NaN payloads and arbitrary raw
                  // halves.
                  a[lane] = 0xff800001U | ((lane * 0x19660dU) & 0x7fffffU);
                  b[lane] = 0x80001234U ^ (lane * 0x41c64e6dU);
                  fixture.cu->write_vgpr(fixture.base(), lane, a[lane]);
                  fixture.cu->write_vgpr(fixture.base() + 1, lane, b[lane]);
                }
                const uint16_t opcode3 =
                    conditional ? (half ? rdna3::kVCndmaskB16Vop3 : rdna3::kVCndmaskB32Vop3)
                                : (half ? rdna3::kVMovB16Vop3 : rdna3::kVMovB32Vop3);
                const uint16_t opcode4 =
                    conditional ? (half ? rdna4::kVCndmaskB16Vop3 : rdna4::kVCndmaskB32Vop3)
                                : (half ? rdna4::kVMovB16Vop3 : rdna4::kVMovB32Vop3);
                const uint8_t opsel = half ? (high ? 9 : 2) : 0;
                const uint8_t absolute = modifiers & 1 ? 3 : 0;
                const uint8_t negate = modifiers & 2 ? 3 : 0;
                const auto words = rdna3 ? rdna3::build_vop3(opcode3, {.vdst = 0,
                                                                       .abs = absolute,
                                                                       .op_sel = opsel,
                                                                       .clamp = 1,
                                                                       .src0 = 256,
                                                                       .src1 = 257,
                                                                       .src2 = 0,
                                                                       .omod = 3,
                                                                       .neg = negate})
                                         : rdna4::build_vop3(opcode4, {.vdst = 0,
                                                                       .abs = absolute,
                                                                       .opsel = opsel,
                                                                       .clamp = 1,
                                                                       .src0 = 256,
                                                                       .src1 = 257,
                                                                       .src2 = 0,
                                                                       .omod = 3,
                                                                       .neg = negate});
                auto instruction = fixture.decoder->decode_window(words);
                ASSERT_TRUE(instruction.succeeded());
                fixture.recorder->writes.clear();
                fixture.recorder->reads.clear();
                fixture.recorder->scalar_reads.clear();
                ASSERT_TRUE(
                    fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                        .succeeded());
                if (inactive) {
                  EXPECT_TRUE(fixture.recorder->reads.empty());
                  EXPECT_TRUE(fixture.recorder->scalar_reads.empty());
                  EXPECT_TRUE(fixture.recorder->writes.empty());
                } else {
                  ASSERT_EQ(fixture.recorder->writes.size(), 1U);
                  EXPECT_EQ(fixture.recorder->writes[0].lanes, exec_mask);
                  EXPECT_EQ(fixture.recorder->writes[0].bytes, half ? (high ? 0xc : 0x3) : 0xf);
                }
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  const bool choose_b = conditional && ((condition >> lane) & 1);
                  uint32_t value = choose_b ? b[lane] : a[lane];
                  if (half) {
                    const unsigned shift = choose_b ? (high ? 0 : 16) : (high ? 16 : 0);
                    value = (value >> shift) & 0xffff;
                    if (modifiers) {
                      // MOV/CNDMASK apply ABS/NEG to the selected half's sign
                      // bit.
                      if (absolute)
                        value &= 0x7fff;
                      if (negate)
                        value ^= 0x8000;
                    }
                    value =
                        high ? (a[lane] & 0xffff) | (value << 16) : (a[lane] & 0xffff0000) | value;
                  } else if (modifiers) {
                    if (absolute)
                      value &= 0x7fffffff;
                    if (negate)
                      value ^= 0x80000000;
                  }
                  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                            ((exec_mask >> lane) & 1) ? value : a[lane]);
                }
              }
}

// Wave32 VOPD: X moves old v7 into v6, Y reads old v6 and writes v7.
// Values are exactly representable; the existing rj-arith dual/dot suites cover
// numerical boundaries. This table checks every decoded slot and operand role.

TEST(ArithBridgeTest, MullitPoliciesNanClampAndMaskedAlias) {
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32u, 64u})
      for (bool dx10 : {false, true})
        for (unsigned rounding = 0; rounding < 4; ++rounding)
          for (unsigned policy_kind = 0; policy_kind < 3; ++policy_kind) {
            ArithmeticPolicy policy{};
            if (policy_kind == 1) {
              policy.semantics = ArithmeticSemantics::Loose;
            }
            if (policy_kind == 2) {
              policy.semantics = ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
            }
            BridgeFixture fixture(rdna3, lanes, policy);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            const uint64_t exec_mask = 1ULL << (lanes - 1);
            fixture.wave->set_exec(exec_mask);
            fixture.wave->set_mode_raw(0xf0 | rounding | (dx10 ? 0x100 : 0));
            fixture.wave->set_trapsts(0x20);
            fixture.cu->write_vgpr(fixture.base(), lanes - 1, 0x7fc12345);
            fixture.cu->write_vgpr(fixture.base() + 1, lanes - 1, 0x3f800000);
            fixture.cu->write_vgpr(fixture.base() + 2, lanes - 1, 0x3f800000);
            const auto words =
                rdna3 ? rdna3::build_vop3(
                            rdna3::kVMullitF32Vop3,
                            {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = 258})
                      : rdna4::build_vop3(
                            rdna4::kVMullitF32Vop3,
                            {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = 258});
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            fixture.recorder->writes.clear();
            const bool success =
                fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                    .succeeded();
            ASSERT_EQ(success, policy_kind != 2);
            if (success) {
              ASSERT_EQ(fixture.recorder->writes.size(), 1U);
              EXPECT_EQ(fixture.recorder->writes[0].lanes, exec_mask);
            } else {
              EXPECT_TRUE(fixture.recorder->writes.empty());
            }
            const uint32_t result = fixture.cu->read_vgpr(fixture.base(), lanes - 1);
            if (!success) {
              EXPECT_EQ(result, 0x7fc12345U);
            } else if (rdna3 && !dx10) {
              EXPECT_GT(result & 0x7fffffffU, 0x7f800000U);
            } else {
              EXPECT_EQ(result, 0U);
            }
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), 0x3c003c00U);
            EXPECT_EQ(fixture.wave->trapsts(), 0x20U);
          }
}

TEST(ArithBridgeTest, InterpolationDistinguishesHalfAndFloatDenormalControls) {
  for (bool gfx11 : {false, true}) {
    for (bool p2 : {false, true}) {
      for (bool preserve : {false, true}) {
        BridgeFixture fixture(gfx11);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        fixture.wave->set_exec(1);
        fixture.wave->set_mode_raw(p2 ? (preserve ? 0x80 : 0x20) : (preserve ? 0x40 : 0x10));
        fixture.cu->write_vgpr(fixture.base(), p2 ? 2 : 1, p2 ? 0x3c00 : 0x0001);
        fixture.cu->write_vgpr(fixture.base() + 1, 0, p2 ? 0x33800000 : 0x3f800000);
        fixture.cu->write_vgpr(fixture.base() + 2, 0, 0);
        const auto words =
            gfx11 ? rdna3::build_vinterp(p2 ? rdna3::kVInterpP2F16F32Vinterp
                                            : rdna3::kVInterpP10F16F32Vinterp,
                                         {.vdst = 3, .src0 = 256, .src1 = 257, .src2 = 258})
                  : rdna4::build_vinterp(p2 ? rdna4::kVInterpP2F16F32Vinterp
                                            : rdna4::kVInterpP10F16F32Vinterp,
                                         {.vdst = 3, .src0 = 256, .src1 = 257, .src2 = 258});
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 3, 0) & (p2 ? 0xffffU : ~0U),
                  preserve ? (p2 ? 1U : 0x33800000U) : 0U);
      }
    }
  }
}

// Captured on gfx1100 and gfx1201. The first NaN wins after source modifiers;
// narrowing P2 to FP16 always sets the quiet bit.
TEST(ArithBridgeTest, InterpolationNanPayloadsFollowGuestModes) {
  for (auto semantics :
       {ArithmeticSemantics::Compatibility, ArithmeticSemantics::BitExactEmpiricalNoFpStatus,
        ArithmeticSemantics::BitExactEmpirical})
    for (bool rdna3 : {true, false}) {
      BridgeFixture fixture(rdna3, 32, ArithmeticPolicy{semantics});
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_exec(1);
      for (unsigned op = 0; op < 6; ++op) {
        const bool half = op >= 2, p2 = op & 1;
        const unsigned coefficient = p2 ? 2 : 1;
        fixture.cu->write_vgpr(fixture.base(), coefficient, half ? 0x7c15U : 0x7f801abcU);
        fixture.cu->write_vgpr(fixture.base() + 1, 0, 0xffc00456U);
        fixture.cu->write_vgpr(fixture.base() + 2, 0, 0x3f803c00U);
        for (uint32_t denorm : {0U, 0xf0U})
          for (bool ieee : {false, true})
            for (bool dx10 : {false, true})
              for (uint8_t clamp : {0, 1}) {
                SCOPED_TRACE(::testing::Message()
                             << int(semantics) << '/' << rdna3 << '/' << op << '/' << denorm << '/'
                             << ieee << '/' << dx10 << '/' << +clamp);
                fixture.wave->set_mode_raw(denorm | (uint32_t(ieee) << 9) | (uint32_t(dx10) << 8));
                fixture.cu->write_vgpr(fixture.base() + 3, 0, 0xabcd1234U);
                const auto words = rdna3 ? rdna3::build_vinterp(op, {.vdst = 3,
                                                                     .clamp = clamp,
                                                                     .src0 = 256,
                                                                     .src1 = 257,
                                                                     .src2 = 258,
                                                                     .neg = 1})
                                         : rdna4::build_vinterp(op, {.vdst = 3,
                                                                     .clamp = clamp,
                                                                     .src0 = 256,
                                                                     .src1 = 257,
                                                                     .src2 = 258,
                                                                     .neg = 1});
                auto inst = fixture.decoder->decode_window(words);
                ASSERT_TRUE(inst.succeeded());
                ASSERT_TRUE(
                    fixture.cu->execute_instruction(inst.value().get(), *fixture.wave).succeeded());
                uint32_t expected = half ? 0xff82a000U : 0xff801abcU;
                if (ieee || !rdna3)
                  expected |= 0x00400000U;
                if (half && p2)
                  expected = 0xfe15U;
                if (clamp && (dx10 || !rdna3))
                  expected = 0;
                if (half && p2)
                  expected |= 0xabcd0000U;
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 3, 0), expected);
              }
      }
    }
}

TEST(ArithBridgeTest, InterpolationCarriesPackedSignalingNanIntoP2) {
  BridgeFixture fixture(/*rdna3=*/true, /*lanes=*/64);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fixture.cu->set_plugin_group(nullptr); // Exercise the direct, aliased VGPR write.
  fixture.wave->set_mode_raw(0x1f0);     // Preserve denormals, IEEE clear, DX10_CLAMP set.
  fixture.wave->set_exec((uint64_t{1} << 32) | (uint64_t{1} << 51));
  const uint32_t base = fixture.base();
  auto run = [&](uint16_t opcode, rdna3::VinterpBuilderFields fields) {
    auto inst = fixture.decoder->decode_window(rdna3::build_vinterp(opcode, fields));
    ASSERT_TRUE(inst.succeeded());
    ASSERT_TRUE(fixture.cu->execute_instruction(inst.value().get(), *fixture.wave).succeeded());
  };
  for (bool nan_in_a : {true, false}) {
    SCOPED_TRACE(nan_in_a);
    for (uint32_t quad : {32u, 48u}) {
      fixture.cu->write_vgpr(base, quad + 1, 0x3c003c00);
      fixture.cu->write_vgpr(base + 2, quad, 0);
      fixture.cu->write_vgpr(base + 3, quad + 2, 0x3f800000);
    }
    fixture.cu->write_vgpr(base + 1, 32, 0x3f800000);
    fixture.cu->write_vgpr(base + 1, 51, 0x3f800000);
    if (nan_in_a)
      fixture.cu->write_vgpr(base, 49, 0x7c253c00);
    else
      fixture.cu->write_vgpr(base + 2, 48, 0xfc150000);

    // Only the high half of the selected coefficient is a NaN. P2 then consumes
    // it solely through the aliased accumulator, after a finite quad.
    ASSERT_NO_FATAL_FAILURE(
        run(rdna3::kVInterpP10F16F32Vinterp, {.vdst = 2,
                                              .op_sel = uint8_t(nan_in_a ? 1 : 4),
                                              .src0 = 256,
                                              .src1 = 257,
                                              .src2 = 258,
                                              .neg = uint8_t(nan_in_a)}));
    const uint32_t nan = nan_in_a ? 0xff84a000 : 0xff82a000;
    EXPECT_EQ(fixture.cu->read_vgpr(base + 2, 51), nan);
    ASSERT_NO_FATAL_FAILURE(
        run(rdna3::kVInterpP2F32Vinterp, {.vdst = 2, .src0 = 259, .src1 = 257, .src2 = 258}));
    EXPECT_EQ(fixture.cu->read_vgpr(base + 2, 51), nan);
    EXPECT_EQ(fixture.cu->read_vgpr(base + 2, 32), nan_in_a ? 0u : 0x40000000u);
  }
}

TEST(ArithBridgeTest, InterpolationUsesGuestRoundingAndInputFlushing) {
  for (bool rdna3 : {false, true}) {
    BridgeFixture fixture(rdna3);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_exec(1);
    for (bool half : {false, true}) {
      fixture.wave->set_mode_raw(half ? 0xfc : 0);
      fixture.cu->write_vgpr(fixture.base(), half ? 2 : 1, half ? 0x3c00 : 1);
      fixture.cu->write_vgpr(fixture.base() + 1, 0, half ? 0x3a400000 : 0x7f000000);
      fixture.cu->write_vgpr(fixture.base() + 2, 0, half ? 0x3f800000 : 0);
      auto words = rdna3 ? rdna3::build_vinterp(half ? rdna3::kVInterpP2F16F32Vinterp
                                                     : rdna3::kVInterpP10F32Vinterp,
                                                {.vdst = 3, .src0 = 256, .src1 = 257, .src2 = 258})
                         : rdna4::build_vinterp(half ? rdna4::kVInterpP2F16F32Vinterp
                                                     : rdna4::kVInterpP10F32Vinterp,
                                                {.vdst = 3, .src0 = 256, .src1 = 257, .src2 = 258});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 3, 0) & (half ? 0xffffU : ~0U),
                half ? 0x3c00U : 0U);
    }
  }
}

TEST(ArithBridgeTest, InterpolationQuadReadsModesAliasesAndPartialWrites) {
  const uint16_t opcodes[2][6] = {
      {rdna3::kVInterpP10F32Vinterp, rdna3::kVInterpP2F32Vinterp, rdna3::kVInterpP10F16F32Vinterp,
       rdna3::kVInterpP2F16F32Vinterp, rdna3::kVInterpP10RtzF16F32Vinterp,
       rdna3::kVInterpP2RtzF16F32Vinterp},
      {rdna4::kVInterpP10F32Vinterp, rdna4::kVInterpP2F32Vinterp, rdna4::kVInterpP10F16F32Vinterp,
       rdna4::kVInterpP2F16F32Vinterp, rdna4::kVInterpP10RtzF16F32Vinterp,
       rdna4::kVInterpP2RtzF16F32Vinterp}};
  for (bool rdna3 : {false, true})
    for (uint32_t lanes : {32u, 64u})
      for (unsigned op = 0; op < 6; ++op)
        for (unsigned halves : {0u, 1u, 2u, 3u})
          for (bool alias : {false, true})
            for (unsigned controls = 0; controls < 4; ++controls) {
              BridgeFixture fixture(rdna3, lanes);
              ASSERT_FALSE(::testing::Test::HasFatalFailure());
              bool half = op >= 2, p2 = op & 1;
              bool high = halves & 1, other_high = halves & 2;
              uint8_t dest = alias ? 0 : 3;
              const uint64_t exec_mask = lanes == 32 ? 0x88888888ULL : 0x8888888888888888ULL;
              fixture.wave->set_exec(exec_mask);
              // These exactly representable operands are unchanged by directed
              // guest rounding.
              fixture.wave->set_mode_raw(0xf);
              fixture.wave->set_trapsts(0x20);
              uint32_t initial[4][64] = {};
              for (unsigned lane = 0; lane < lanes; ++lane) {
                uint32_t value = std::bit_cast<uint32_t>(float(lane + 1));
                uint32_t packed = high ? uint32_t(util::f32_to_f16(float(lane + 1))) << 16 | 0x3555
                                       : 0x35550000 | util::f32_to_f16(float(lane + 1));
                initial[0][lane] = half ? packed : value;
                initial[1][lane] = 0x3f000000;
                initial[2][lane] =
                    half && !p2
                        ? (other_high ? uint32_t(util::f32_to_f16(float(lane + 1))) << 16 | 0x3555
                                      : 0x35550000 | util::f32_to_f16(float(lane + 1)))
                        : value;
                initial[3][lane] = 0xa55a5aa5;
                for (unsigned reg = 0; reg < 4; ++reg)
                  fixture.cu->write_vgpr(fixture.base() + reg, lane, initial[reg][lane]);
              }
              const uint8_t opsel = half ? (high ? 1 : 0) | (other_high ? (p2 ? 8 : 4) : 0) : 0;
              const uint8_t neg = controls & 1 ? 7 : 0;
              const auto words =
                  rdna3
                      ? rdna3::build_vinterp(opcodes[0][op], {.vdst = dest,
                                                              .wait_exp = 7,
                                                              .op_sel = opsel,
                                                              .clamp = uint8_t((controls >> 1) & 1),
                                                              .src0 = 256,
                                                              .src1 = 257,
                                                              .src2 = 258,
                                                              .neg = neg})
                      : rdna4::build_vinterp(opcodes[1][op], {.vdst = dest,
                                                              .wait_exp = 7,
                                                              .opsel = opsel,
                                                              .clamp = uint8_t((controls >> 1) & 1),
                                                              .src0 = 256,
                                                              .src1 = 257,
                                                              .src2 = 258,
                                                              .neg = neg});
              auto instruction = fixture.decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                              .succeeded());
              const uint64_t quads = exec_mask >> 3;
              ASSERT_EQ(fixture.recorder->reads.size(), 3U);
              EXPECT_EQ(fixture.recorder->reads[0].lanes, quads << (p2 ? 2 : 1));
              EXPECT_EQ(fixture.recorder->reads[0].bytes, half ? (high ? 0xc : 3) : 0xf);
              EXPECT_EQ(fixture.recorder->reads[1].lanes, exec_mask);
              EXPECT_EQ(fixture.recorder->reads[2].lanes, p2 ? exec_mask : quads);
              EXPECT_EQ(fixture.recorder->reads[2].bytes,
                        half && !p2 ? (other_high ? 0xc : 3) : 0xf);
              ASSERT_EQ(fixture.recorder->writes.size(), 1U);
              EXPECT_EQ(fixture.recorder->writes[0].lanes, exec_mask);
              EXPECT_EQ(fixture.recorder->writes[0].bytes,
                        half && p2 ? (other_high ? 0xc : 3) : 0xf);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                uint32_t expected = initial[dest][lane];
                if ((exec_mask >> lane) & 1) {
                  unsigned quad = lane & ~3U;
                  float value = float(quad + (p2 ? 3 : 2)) * 0.5f +
                                (neg ? -1 : 1) * float((p2 ? lane : quad) + 1);
                  if (controls & 2)
                    value = std::clamp(value, 0.0f, 1.0f);
                  if (half && p2) {
                    unsigned shift = other_high ? 16 : 0;
                    expected = (expected & ~(0xffffU << shift)) | uint32_t(util::f32_to_f16(value))
                                                                      << shift;
                  } else {
                    expected = std::bit_cast<uint32_t>(value);
                  }
                }
                EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + dest, lane), expected)
                    << rdna3 << '/' << lanes << '/' << op << '/' << high << '/' << alias << '/'
                    << controls << '/' << lane;
              }
              EXPECT_EQ(fixture.wave->trapsts(), 0x20U);
            }
}
TEST(ArithBridgeTest, Dot4DppObservesOnlyWritableLanes) {
  for (uint32_t lanes : {32u, 64u}) {
    BridgeFixture fixture(false, lanes, {ArithmeticSemantics::Loose});
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
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

TEST(ArithBridgeTest, Fp8SdwaRejectsPendingInputFailure) {
  BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA4, 64);
  fixture.wave->set_exec(1);
  fixture.wave->report_instruction_execution_error(
      InstructionExecutionError::InvalidRegisterAccess);
  cdna4::Operand a(32, cdna4::OperandType::OPR_VGPR, 0);
  cdna4::Operand d(32, cdna4::OperandType::OPR_VGPR, 16);
  int inst = 0;
  arith::execute_fp8_widen_sdwa(inst, *fixture.wave, rj_arith_v_cvt_f32_fp8_w32_gfx950,
                                rj_arith_v_cvt_f32_fp8_w64_gfx950, d, a);
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 16, 0), 0U);
  EXPECT_TRUE(fixture.recorder->writes.empty());
}

TEST(ArithBridgeTest, CdnaFp8Widening) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      for (int op = 0; op < 4; ++op) {
        for (int form = 0; form < 4; ++form) {
          if (form == 2 && arch == ROCJITSU_CODE_ARCH_CDNA5) {
            continue;
          }
          SCOPED_TRACE(::testing::Message() << arch << ":" << op << ":" << form);
          ArithmeticPolicy policy{};
          policy.semantics = semantics;
          int lanes = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32 : 64;
          BridgeFixture fixture(arch, lanes, policy);
          fixture.wave->set_exec(1ULL | (1ULL << (lanes - 1)));
          fixture.wave->set_mode_raw(0xff); // Guest rounding and flushing do not affect widening.
          constexpr uint32_t input = 0xff7f0180U;
          for (int lane = 0; lane < lanes; ++lane) {
            fixture.cu->write_vgpr(fixture.base(), lane, input);
            fixture.cu->write_vgpr(fixture.base() + 1, lane, 0xdeadbeefU);
          }
          bool vop3 = form == 1 || form == 3;
          bool packed = op >= 2;
          uint8_t selector = form == 3 ? (packed ? 1 : 3) : 0;
          uint8_t clamp = arch == ROCJITSU_CODE_ARCH_CDNA5 && op == 0 && form == 3;
          std::vector<uint32_t> words;
          if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
            constexpr uint16_t vop1[] = {cdna3::kVCvtF32Fp8Vop1, cdna3::kVCvtF32Bf8Vop1,
                                         cdna3::kVCvtPkF32Fp8Vop1, cdna3::kVCvtPkF32Bf8Vop1};
            constexpr uint16_t vop3_ops[] = {cdna3::kVCvtF32Fp8Vop3, cdna3::kVCvtF32Bf8Vop3,
                                             cdna3::kVCvtPkF32Fp8Vop3, cdna3::kVCvtPkF32Bf8Vop3};
            if (vop3) {
              auto encoded = cdna3::build_vop3(
                  vop3_ops[op], {.vdst = 0, .op_sel = selector, .clamp = clamp, .src0 = 256});
              words.assign(encoded.begin(), encoded.end());
            } else {
              auto encoded = cdna3::build_vop1(
                  vop1[op], {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
              words.assign(encoded.begin(), encoded.end());
            }
          } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
            constexpr uint16_t vop1[] = {cdna4::kVCvtF32Fp8Vop1, cdna4::kVCvtF32Bf8Vop1,
                                         cdna4::kVCvtPkF32Fp8Vop1, cdna4::kVCvtPkF32Bf8Vop1};
            constexpr uint16_t vop3_ops[] = {cdna4::kVCvtF32Fp8Vop3, cdna4::kVCvtF32Bf8Vop3,
                                             cdna4::kVCvtPkF32Fp8Vop3, cdna4::kVCvtPkF32Bf8Vop3};
            if (vop3) {
              auto encoded = cdna4::build_vop3(
                  vop3_ops[op], {.vdst = 0, .op_sel = selector, .clamp = clamp, .src0 = 256});
              words.assign(encoded.begin(), encoded.end());
            } else {
              auto encoded = cdna4::build_vop1(
                  vop1[op], {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
              words.assign(encoded.begin(), encoded.end());
            }
          } else if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
            constexpr uint16_t vop1[] = {cdna5::kVCvtF32Fp8Vop1, cdna5::kVCvtF32Bf8Vop1,
                                         cdna5::kVCvtPkF32Fp8Vop1, cdna5::kVCvtPkF32Bf8Vop1};
            constexpr uint16_t vop3_ops[] = {cdna5::kVCvtF32Fp8Vop3, cdna5::kVCvtF32Bf8Vop3,
                                             cdna5::kVCvtPkF32Fp8Vop3, cdna5::kVCvtPkF32Bf8Vop3};
            if (vop3) {
              auto encoded = cdna5::build_vop3(
                  vop3_ops[op], {.vdst = 0, .opsel = selector, .clamp = clamp, .src0 = 256});
              words.assign(encoded.begin(), encoded.end());
            } else {
              auto encoded = cdna5::build_vop1(
                  vop1[op], {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
              words.assign(encoded.begin(), encoded.end());
            }
          }
          if (form == 2) {
            words.push_back((5U << 8) | (2U << 11) | (6U << 16));
          }
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          auto expected = [&](int byte) -> uint32_t {
            uint8_t code = uint8_t(input >> (8 * byte));
            if (clamp) {
              return 0x7fc00000U;
            } // Selected 0xff is the sole E5M3 NaN.
            float value =
                arch == ROCJITSU_CODE_ARCH_CDNA3
                    ? (op & 1 ? util::bf8_e5m2_fnuz_to_f32(code) : util::fp8_e4m3_fnuz_to_f32(code))
                    : (op & 1 ? util::bf8_e5m2_to_f32(code) : util::fp8_e4m3_to_f32(code));
            return std::bit_cast<uint32_t>(value);
          };
          int byte = form == 3 ? (packed ? 2 : 3) : 0;
          uint32_t low = expected(byte);
          // CDNA3 FP8/BF8 widening uses SDWA only for source selection;
          // destination selection is ignored, as in the decoder on develop.
          if (form == 2 && !packed && arch != ROCJITSU_CODE_ARCH_CDNA3) {
            low = (low << 16) | (input & 0xffff);
          }
          for (int lane : {0, lanes - 1}) {
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), low);
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, lane),
                      packed ? expected(byte + 1) : 0xdeadbeefU);
          }
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 1), input);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 1), 0xdeadbeefU);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, Gfx1250Fp8ToF16) {
  const uint16_t ops[] = {cdna5::kVCvtF16Fp8Vop3, cdna5::kVCvtF16Bf8Vop3, cdna5::kVCvtPkF16Fp8Vop3,
                          cdna5::kVCvtPkF16Bf8Vop3};
  const uint16_t vop1_ops[] = {cdna5::kVCvtF16Fp8Vop1, cdna5::kVCvtF16Bf8Vop1,
                               cdna5::kVCvtPkF16Fp8Vop1, cdna5::kVCvtPkF16Bf8Vop1};
  for (bool vop1 : {false, true}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      for (int op = 0; op < 4; ++op) {
        for (int variant = 0; variant < 4; ++variant) {
          ArithmeticPolicy policy{};
          policy.semantics = semantics;
          BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, 32, policy);
          fixture.wave->set_exec(0xfffffffdU);
          bool saturate = variant & 1, clamp = !vop1 && (variant & 2), packed = op >= 2;
          fixture.wave->set_mode_raw(0xff | (saturate ? Wavefront::FP16_OVFL_BIT : 0));
          int selector = (variant & 1 ? 8 : 0) | variant;
          if (vop1)
            selector = (variant & 1) ? (packed ? 1 : 8) : 0;
          std::vector<uint32_t> words;
          if (vop1) {
            auto encoded = cdna5::build_vop1(
                vop1_ops[op], {.src0 = uint16_t(256 + (packed && (selector & 1) ? 128 : 0)),
                               .vdst = uint8_t(!packed && (selector & 8) ? 128 : 0)});
            words.assign(encoded.begin(), encoded.end());
          } else {
            auto encoded = cdna5::build_vop3(ops[op], {.vdst = 0,
                                                       .abs = 1,
                                                       .opsel = uint8_t(selector),
                                                       .clamp = uint8_t(clamp),
                                                       .src0 = 256,
                                                       .omod = 3,
                                                       .neg = 1});
            words.assign(encoded.begin(), encoded.end());
          }
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          for (int batch = 0; batch < 8; ++batch) {
            for (int lane = 0; lane < 32; ++lane) {
              uint32_t code = batch * 32 + lane;
              fixture.cu->write_vgpr(fixture.base(), lane, (code * 0x01010101U) ^ 0xaa558000U);
            }
            ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .succeeded());
            for (int lane = 0; lane < 32; ++lane) {
              uint32_t input = (uint32_t(batch * 32 + lane) * 0x01010101U) ^ 0xaa558000U;
              int byte = packed ? 2 * (selector & 1) : 2 * (selector & 1) + ((selector >> 1) & 1);
              auto half = [&](int index) {
                uint8_t code = uint8_t(input >> (8 * index));
                float value = op & 1             ? util::bf8_e5m2_to_f32(code)
                              : !packed && clamp ? util::fp8_e5m3_to_f32(code)
                                                 : util::fp8_e4m3_to_f32(code);
                return uint32_t(util::f32_to_f16_mode(value, saturate));
              };
              int shift = selector & 8 ? 16 : 0;
              uint32_t expected = packed ? half(byte) | (half(byte + 1) << 16)
                                         : (input & ~(0xffffU << shift)) | (half(byte) << shift);
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), lane == 1 ? input : expected)
                  << "op=" << op << " variant=" << variant << " code=" << (batch * 32 + lane);
            }
          }
        }
      }
    }
  }
  // DPP stages the selected high half right-aligned: do not select it twice.
  BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, 32);
  fixture.wave->set_exec(3);
  fixture.cu->write_vgpr(fixture.base(), 0, 0x40303800);
  fixture.cu->write_vgpr(fixture.base(), 1, 0xb840b030);
  const auto first = cdna5::build_vop1(cdna5::kVCvtPkF16Fp8Vop1, {.src0 = SRC_DPP, .vdst = 1});
  // quad_perm:[1,0,2,3], all rows/banks, source VGPR 0 high half.
  const std::array<uint32_t, 2> words{first[0], 0xff00e180U};
  auto instruction = fixture.decoder->decode_window(words);
  ASSERT_TRUE(instruction.succeeded());
  ASSERT_TRUE(
      fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 0), 0xbc004000U);
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 1), 0x40003800U);
}

TEST(ArithBridgeTest, CdnaFp8Narrowing) {
  const uint32_t inputs[] = {0,           0x80000000U, 0x7f800000U, 0xff800000U,
                             0x7fc12345U, 0xff800001U, 0x3f880001U, 0xbf980000U,
                             0x3b000001U, 0x37800001U, 0x43e80000U, 0x47700000U,
                             0x43780000U, 0x80000001U, 0x3f800000U, 0x7f7fffffU};
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      for (int op = 0; op < 4; ++op) {
        for (int variant = 0; variant < 4; ++variant) {
          SCOPED_TRACE(::testing::Message() << arch << ":" << op << ":" << variant);
          ArithmeticPolicy policy{};
          policy.semantics = semantics;
          int lanes = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32 : 64;
          BridgeFixture fixture(arch, lanes, policy);
          fixture.wave->set_exec((lanes == 32 ? 0xffffffffULL : ~0ULL) & ~(1ULL << 1));
          bool saturate = variant & 1;
          fixture.wave->set_mode_raw(0xff | (saturate ? Wavefront::FP16_OVFL_BIT : 0));
          bool sr = op >= 2, bf8 = op & 1;
          for (int lane = 0; lane < lanes; ++lane) {
            fixture.cu->write_vgpr(fixture.base(), lane, inputs[lane % 16]);
            fixture.cu->write_vgpr(fixture.base() + 1, lane,
                                   sr ? (lane & 1 ? ~0U : 0U) : inputs[(lane + 5) % 16]);
          }
          uint8_t selector = sr ? variant << 2 : (variant & 1) << 3;
          uint8_t clamp = arch == ROCJITSU_CODE_ARCH_CDNA5 && !bf8 && variant >= 2;
          std::vector<uint32_t> words;
          if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
            constexpr uint16_t ops[] = {cdna3::kVCvtPkFp8F32Vop3, cdna3::kVCvtPkBf8F32Vop3,
                                        cdna3::kVCvtSrFp8F32Vop3, cdna3::kVCvtSrBf8F32Vop3};
            auto encoded = cdna3::build_vop3(ops[op], {.vdst = 0,
                                                       .abs = 3,
                                                       .op_sel = selector,
                                                       .clamp = clamp,
                                                       .src0 = 256,
                                                       .src1 = 257,
                                                       .omod = 3,
                                                       .neg = 3});
            words.assign(encoded.begin(), encoded.end());
          } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
            constexpr uint16_t ops[] = {cdna4::kVCvtPkFp8F32Vop3, cdna4::kVCvtPkBf8F32Vop3,
                                        cdna4::kVCvtSrFp8F32Vop3, cdna4::kVCvtSrBf8F32Vop3};
            auto encoded = cdna4::build_vop3(ops[op], {.vdst = 0,
                                                       .abs = 3,
                                                       .op_sel = selector,
                                                       .clamp = clamp,
                                                       .src0 = 256,
                                                       .src1 = 257,
                                                       .omod = 3,
                                                       .neg = 3});
            words.assign(encoded.begin(), encoded.end());
          } else if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
            constexpr uint16_t ops[] = {cdna5::kVCvtPkFp8F32Vop3, cdna5::kVCvtPkBf8F32Vop3,
                                        cdna5::kVCvtSrFp8F32Vop3, cdna5::kVCvtSrBf8F32Vop3};
            auto encoded = cdna5::build_vop3(ops[op], {.vdst = 0,
                                                       .abs = 3,
                                                       .opsel = selector,
                                                       .clamp = clamp,
                                                       .src0 = 256,
                                                       .src1 = 257,
                                                       .omod = 3,
                                                       .neg = 3});
            words.assign(encoded.begin(), encoded.end());
          }
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          auto narrow = [&](uint32_t raw, uint32_t seed) -> uint32_t {
            float value = std::bit_cast<float>(raw);
            if (clamp) {
              return sr ? util::f32_to_fp8_e5m3_sr_mode(value, seed, saturate)
                        : util::f32_to_fp8_e5m3_rne_mode(value, saturate);
            }
            raw = (raw & 0x7fffffffU) | 0x80000000U; // ABS then NEG on each FP source.
            value = std::bit_cast<float>(raw);
            if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
              if (bf8) {
                return sr ? conversion::encode_fp8_stochastic<conversion::Bf8>(raw, seed, saturate)
                          : conversion::encode_fp8<conversion::Bf8>(raw, saturate);
              }
              return sr ? conversion::encode_fp8_stochastic<conversion::Fp8>(raw, seed, saturate)
                        : conversion::encode_fp8<conversion::Fp8>(raw, saturate);
            }
            if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
              if (bf8) {
                return sr ? util::f32_to_bf8_e5m2_fnuz_sr_mode(value, seed, saturate)
                          : util::f32_to_bf8_e5m2_fnuz_rne_mode(value, saturate);
              }
              return sr ? util::f32_to_fp8_e4m3_fnuz_sr_mode(value, seed, saturate)
                        : util::f32_to_fp8_e4m3_fnuz_rne_mode(value, saturate);
            }
            if (bf8) {
              return sr ? util::f32_to_bf8_e5m2_sr_mode(value, seed, saturate)
                        : util::f32_to_bf8_e5m2_rne_mode(value, saturate);
            }
            return sr ? util::f32_to_fp8_e4m3_sr_mode(value, seed, saturate)
                      : util::f32_to_fp8_e4m3_rne_mode(value, saturate);
          };
          for (int lane = 0; lane < lanes; ++lane) {
            uint32_t a = inputs[lane % 16];
            uint32_t b = sr ? (lane & 1 ? ~0U : 0U) : inputs[(lane + 5) % 16];
            uint32_t value = narrow(a, b);
            if (!sr) {
              value |= narrow(b, 0) << 8;
            }
            int shift = sr ? variant * 8 : (variant & 1) * 16;
            uint32_t selected = (sr ? 255U : 65535U) << shift;
            uint32_t expected = lane == 1 ? a : (a & ~selected) | (value << shift);
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), expected) << "lane=" << lane;
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, Gfx1250F16ToFp8) {
  const uint32_t inputs[] = {0x7c017c00U, 0xfc01fc00U, 0x3c403c41U, 0x3c3fbc40U,
                             0x7bfffbffU, 0x00010000U, 0x03ff8001U, 0x80003c00U};
  const uint16_t ops[] = {cdna5::kVCvtPkFp8F16Vop3, cdna5::kVCvtPkBf8F16Vop3,
                          cdna5::kVCvtSrFp8F16Vop3, cdna5::kVCvtSrBf8F16Vop3};
  for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
    for (int op = 0; op < 4; ++op) {
      for (int variant = 0; variant < 8; ++variant) {
        ArithmeticPolicy policy{};
        policy.semantics = semantics;
        BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, 32, policy);
        fixture.wave->set_exec(0xfffffffdU);
        fixture.wave->set_mode_raw(0xff | (variant & 1 ? Wavefront::FP16_OVFL_BIT : 0));
        bool sr = op >= 2, bf8 = op & 1, clamp = variant & 4;
        int selector = sr ? ((variant & 3) << 2) | (variant & 1) : (variant & 1) << 3;
        for (int lane = 0; lane < 32; ++lane) {
          fixture.cu->write_vgpr(fixture.base(), lane, inputs[lane % 8]);
          fixture.cu->write_vgpr(fixture.base() + 1, lane, lane & 1 ? ~0U : 0);
        }
        auto words = cdna5::build_vop3(ops[op], {.vdst = 0,
                                                 .abs = 3,
                                                 .opsel = uint8_t(selector),
                                                 .clamp = uint8_t(clamp),
                                                 .src0 = 256,
                                                 .src1 = 257,
                                                 .omod = 3,
                                                 .neg = 3});
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        for (int lane = 0; lane < 32; ++lane) {
          uint32_t input = inputs[lane % 8], seed = lane & 1 ? ~0U : 0;
          auto narrow = [&](int half) -> uint32_t {
            float value = util::f16_to_f32(uint16_t(input >> (16 * half)));
            // Packed forms apply SRC0 modifiers to both halves; SR consumes the raw half.
            if (!sr)
              value = -std::abs(value);
            if (clamp && !bf8) {
              return sr ? util::f32_to_fp8_e5m3_sr(value, seed) : util::f32_to_fp8_e5m3_rne(value);
            }
            if (bf8) {
              return sr ? util::f32_to_bf8_e5m2_sr(value, seed) : util::f32_to_bf8_e5m2_rne(value);
            }
            return sr ? util::f32_to_fp8_e4m3_sr(value, seed) : util::f32_to_fp8_e4m3_rne(value);
          };
          uint32_t value = sr ? narrow(selector & 1) : narrow(0) | (narrow(1) << 8);
          int shift = sr ? ((selector >> 2) & 3) * 8 : selector & 8 ? 16 : 0;
          uint32_t expected = (input & ~((sr ? 0xffU : 0xffffU) << shift)) | (value << shift);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), lane == 1 ? input : expected)
              << "op=" << op << " variant=" << variant << " lane=" << lane;
        }
      }
    }
  }
}

TEST(ArithBridgeTest, DoubleReciprocalFlushesSubnormalInput) {
  for (bool rdna3 : {false, true}) {
    BridgeFixture fixture(rdna3);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_mode_raw(0x80); // FP64 input flush, output preservation.
    fixture.wave->set_exec(1);
    fixture.cu->write_vgpr(fixture.base(), 0, 0xffffffff);
    fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x000fffff);
    const auto words = rdna3 ? rdna3::build_vop3(rdna3::kVRcpF64Vop3, {.vdst = 2, .src0 = 256})
                             : rdna4::build_vop3(rdna4::kVRcpF64Vop3, {.vdst = 2, .src0 = 256});
    auto instruction = fixture.decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    ASSERT_TRUE(
        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0), 0U);
    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 3, 0), 0x7ff00000U);
  }
}

TEST(ArithBridgeTest, InterpolationRtzOverridesGuestNearest) {
  for (bool rdna3 : {false, true}) {
    for (bool p2 : {false, true}) {
      BridgeFixture fixture(rdna3);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      fixture.wave->set_exec(1);
      fixture.wave->set_mode_raw(0);
      fixture.cu->write_vgpr(fixture.base(), p2 ? 2 : 1, 0x3c00);
      fixture.cu->write_vgpr(fixture.base() + 1, 0, p2 ? 0x3a400000 : 0x33c00000);
      fixture.cu->write_vgpr(fixture.base() + 2, 0, p2 ? 0x3f800000 : 0x3c00);
      auto words = rdna3 ? rdna3::build_vinterp(p2 ? rdna3::kVInterpP2RtzF16F32Vinterp
                                                   : rdna3::kVInterpP10RtzF16F32Vinterp,
                                                {.vdst = 3, .src0 = 256, .src1 = 257, .src2 = 258})
                         : rdna4::build_vinterp(p2 ? rdna4::kVInterpP2RtzF16F32Vinterp
                                                   : rdna4::kVInterpP10RtzF16F32Vinterp,
                                                {.vdst = 3, .src0 = 256, .src1 = 257, .src2 = 258});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 3, 0) & (p2 ? 0xffffU : ~0U),
                p2 ? 0x3c00U : 0x3f800000U);
    }
  }
}

TEST(ArithBridgeTest, DecodedClassCompareStrictStatusIsArchitectureSpecific) {
  for (bool gfx11 : {false, true}) {
    BridgeFixture fixture(gfx11, 32, parse_arithmetic_policy("bit_exact_empirical"));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_exec(1);
    fixture.wave->set_vcc_mask(0x80);
    fixture.cu->write_vgpr(fixture.base(), 0, 0x7f800001);
    fixture.cu->write_vgpr(fixture.base() + 1, 0, 1);
    auto words = gfx11 ? rdna3::build_vopc(rdna3::kVCmpClassF32Vopc, {.src0 = 256, .vsrc1 = 1})
                       : rdna4::build_vopc(rdna4::kVCmpClassF32Vopc, {.src0 = 256, .vsrc1 = 1});
    auto instruction = fixture.decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    EXPECT_EQ(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).failed(),
              gfx11);
    EXPECT_EQ(fixture.wave->vcc(), gfx11 ? 0x80U : 1U);
  }
}

TEST(ArithBridgeTest, PseudoScalarReciprocalDistinguishesLooseAndEmpirical) {
  for (auto semantics :
       {ArithmeticSemantics::Loose, ArithmeticSemantics::BitExactEmpiricalAllowLoose,
        ArithmeticSemantics::Compatibility}) {
    ArithmeticPolicy policy{};
    policy.semantics = semantics;
    BridgeFixture fixture(false, 32, policy);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_exec(0);
    fixture.wave->set_mode_raw(0xff);
    const auto base = fixture.wave->sgpr_alloc().base;
    fixture.cu->write_sgpr(base, 0x3f803686);
    auto words = rdna4::build_vop3(rdna4::kVSRcpF32Vop3, {.vdst = 1, .src0 = 0});
    auto instruction = fixture.decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    ASSERT_TRUE(
        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
    EXPECT_EQ(fixture.cu->read_sgpr(base + 1),
              semantics == ArithmeticSemantics::Loose ? 0x3f7f9322U : 0x3f7f9323U);
  }
}

TEST(ArithBridgeTest, DualSlotsMasksAliasesLiteralsAndDots) {
  for (bool rdna3 : {false, true})
    for (bool loose : {false, true})
      for (uint32_t exec_mask : {0U, 0x80000005U, 0xffffffffU})
        for (unsigned op :
             {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U, 13U, 16U, 17U, 18U})
          for (unsigned variant = 0; variant < (op == 12 || op == 13 ? 2U : 1U); ++variant) {
            SCOPED_TRACE(::testing::Message()
                         << rdna3 << ":" << loose << ":" << exec_mask << ":" << op);
            ArithmeticPolicy policy{};
            if (loose) {
              policy.semantics = ArithmeticSemantics::Loose;
            }
            BridgeFixture fixture(rdna3, 32, policy);
            fixture.wave->set_mode_raw(0xf0);
            fixture.wave->set_exec(exec_mask);
            fixture.wave->set_vcc_mask(0xaaaaaaaa);
            fixture.wave->set_trapsts(0x20);
            uint32_t a = 0x40000000, b = 0x40400000, c = 0x40a00000;
            if (op == 12) {
              a = 0x40003c00;
              b = 0x44004200;
            }
            if (op == 13) {
              a = 0x40003f80;
              b = 0x40804040;
            }
            if (op >= 16) {
              a = 2;
              b = 3;
            }
            for (unsigned lane = 0; lane < 32; ++lane) {
              fixture.cu->write_vgpr(fixture.base() + 6, lane, a);
              fixture.cu->write_vgpr(fixture.base() + 3, lane, b);
              fixture.cu->write_vgpr(fixture.base() + 7, lane, c);
            }
            const std::array<uint32_t, 3> words = {
                (0x32U << 26) | (8U << 22) | (op << 17) | (5U << 9) | 263U,
                (6U << 24) | (3U << 17) | (3U << 9) | (variant ? 242U : 262U), 0x40800000U};
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            fixture.recorder->reads.clear();
            fixture.recorder->writes.clear();
            ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .succeeded());
            // MOV's unused VSRCX1=5 must not become an observed source read.
            for (const auto &read : fixture.recorder->reads) {
              EXPECT_NE(read.reg, fixture.base() + 5);
              EXPECT_EQ(read.lanes, exec_mask);
            }
            EXPECT_EQ(fixture.recorder->writes.size(), exec_mask ? 2U : 0U);
            for (unsigned lane = 0; lane < 32; ++lane) {
              uint32_t expected = 0;
              switch (op) {
              case 0:
                expected = 0x41300000;
                break; // 2*3+5 = 11
              case 1:
                expected = 0x41200000;
                break; // 2*3+4 = 10
              case 2:
                expected = 0x41300000;
                break; // 2*4+3 = 11
              case 3:
              case 7:
                expected = 0x40c00000;
                break;
              case 4:
                expected = 0x40a00000;
                break;
              case 5:
                expected = 0xbf800000;
                break;
              case 6:
                expected = 0x3f800000;
                break;
              case 8:
              case 11:
                expected = a;
                break;
              case 9:
                expected = lane & 1 ? b : a;
                break;
              case 10:
                expected = b;
                break;
              case 12:
              case 13:
                expected = variant ? 0x41400000 : 0x41800000;
                break; // 1*3+2*4+5 = 16
              case 16:
                expected = 5;
                break;
              case 17:
                expected = 12;
                break;
              case 18:
                expected = 2;
                break;
              }
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 6, lane),
                        (exec_mask >> lane) & 1 ? c : a);
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 7, lane),
                        (exec_mask >> lane) & 1 ? expected : c);
            }
            EXPECT_EQ(fixture.wave->trapsts(), 0x20U);
          }
}

TEST(ArithBridgeTest, DualObservesExecBeforeWriteMask) {
  for (uint32_t write_mask : {0U, 1U}) {
    BridgeFixture fixture(false);
    fixture.wave->set_exec(3);
    fixture.wave->set_vgpr_write_mask(write_mask);
    const std::array<uint32_t, 2> words = {(0x32U << 26) | (8U << 22) | (4U << 17) | (1U << 9) |
                                               256U,
                                           (6U << 24) | (3U << 17) | (3U << 9) | 258U};
    auto instruction = fixture.decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    ASSERT_TRUE(
        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
    ASSERT_FALSE(fixture.recorder->reads.empty());
    for (const auto &read : fixture.recorder->reads) {
      EXPECT_EQ(read.lanes, 3U);
    }
    for (const auto &write : fixture.recorder->writes) {
      EXPECT_EQ(write.lanes, write_mask);
    }
  }
}

TEST(ArithBridgeTest, DualCompatibilityUsesGuestIeeeForGfx11SignalingNan) {
  for (bool ieee : {false, true}) {
    BridgeFixture fixture(true);
    fixture.wave->set_mode_raw(0x30 | (ieee ? (1u << 9) : 0));
    fixture.wave->set_exec(1);
    fixture.cu->write_vgpr(fixture.base(), 0, 0x3f800000);
    fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x7f800001);
    const std::array<uint32_t, 2> words = {(0x32U << 26) | (7U << 22) | (8U << 17) | (1U << 9) |
                                               256U,
                                           (6U << 24) | (3U << 17) | (3U << 9) | 258U};
    auto instruction = fixture.decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    ASSERT_TRUE(
        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
    EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 6, 0), ieee ? 0x7fc00001U : 0x7f800001U);
  }
}

TEST(ArithBridgeTest, DualCommitsBothSlotsFpCauses) {
  ArithmeticPolicy policy{ArithmeticSemantics::BitExactEmpirical};
  BridgeFixture fixture(false, 32, policy);
  fixture.wave->set_mode_raw(0xf0);
  fixture.wave->set_exec(1);
  fixture.wave->set_trapsts(4);
  for (unsigned i = 0; auto value : {0x7f7fffffU, 0x40000000U, 0x7f800001U, 0x3f800000U}) {
    fixture.cu->write_vgpr(fixture.base() + i++, 0, value);
  }
  const std::array<uint32_t, 2> words = {(0x32U << 26) | (3U << 22) | (3U << 17) | (1U << 9) | 256U,
                                         (6U << 24) | (3U << 17) | (3U << 9) | 258U};
  auto instruction = fixture.decoder->decode_window(words);
  ASSERT_TRUE(instruction.succeeded());
  ASSERT_TRUE(
      fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
  EXPECT_EQ(fixture.wave->pending_alu_causes(), 0x29U);
  EXPECT_EQ(fixture.wave->trapsts(), 0x2dU);
}

TEST(ArithBridgeTest, DualSecondSlotRejectionCommitsNeitherResultNorState) {
  for (bool rdna3 : {false, true}) {
    ArithmeticPolicy policy{};
    policy.semantics = ArithmeticSemantics::BitExactEmpirical;
    BridgeFixture fixture(rdna3, 32, policy);
    fixture.wave->set_mode_raw(0xf0);
    fixture.wave->set_exec(0xffffffff);
    fixture.wave->set_trapsts(0x20);
    // MOV succeeds on gfx11; gfx12 MUL can report overflow/inexact.
    // MIN rejects mode 3 because it has no FP-status implementation.
    const std::array<uint32_t, 2> words = {(0x32U << 26) | ((rdna3 ? 8U : 3U) << 22) | (11U << 17) |
                                               (1U << 9) | 256U,
                                           (6U << 24) | (3U << 17) | (3U << 9) | 258U};
    for (unsigned lane = 0; lane < 32; ++lane) {
      fixture.cu->write_vgpr(fixture.base(), lane, 0x7f7fffff);
      fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x40000000);
    }
    auto instruction = fixture.decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    fixture.recorder->writes.clear();
    EXPECT_FALSE(
        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
    EXPECT_TRUE(fixture.recorder->writes.empty());
    for (unsigned lane = 0; lane < 32; ++lane) {
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 6, lane), 0x3c003c00U);
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 7, lane), 0x3c003c00U);
    }
    EXPECT_EQ(fixture.wave->trapsts(), 0x20U);
    EXPECT_EQ(fixture.wave->pending_alu_causes(), 0U);
  }
}

TEST(ArithBridgeTest, DualDx9CompatibilityUnderflowBoundary) {
  // gfx1201 flushes before rounding at MODE=0x10; the legacy host multiply
  // rounded up to minimum-normal first. Compatibility follows the GPU here.
  // gfx11 inherits the ordinary MUL model; this is not a gfx11 capture.
  for (bool rdna3 : {false, true}) {
    for (uint32_t mode : {0x10U, 0x30U}) {
      BridgeFixture fixture(rdna3);
      fixture.wave->set_mode_raw(mode);
      fixture.wave->set_exec(1);
      fixture.cu->write_vgpr(fixture.base(), 0, 0x00800000);
      fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x3f7fffff);
      const std::array<uint32_t, 2> words = {(0x32U << 26) | (7U << 22) | (8U << 17) | (1U << 9) |
                                                 256U,
                                             (6U << 24) | (3U << 17) | (3U << 9) | 258U};
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 6, 0), mode == 0x10 ? 0U : 0x00800000U);
    }
  }
}

TEST(ArithBridgeTest, CdnaBf16MfmaFusedCancellation) {
  for (bool scalar : {false, true}) {
    struct RestoreScalar {
      bool previous = util::force_scalar();
      ~RestoreScalar() { util::set_force_scalar_for_testing(previous); }
    } restore;
    util::set_force_scalar_for_testing(scalar);
    ScopedIsaExecutionBackend backend(&rocjitsu::cdna4::execution_backend());
    ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_CDNA4;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    GpuMemory memory("bf16_cancel");
    L2Cache l2("bf16_cancel_l2");
    auto cu = ComputeUnitCore::create("bf16_cancel", config, &memory, &l2);
    auto *wave = cu->dispatch_wf(0, 0, 104, 256, 64);
    ASSERT_NE(wave, nullptr);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    const auto base = wave->vgpr_alloc().base;
    for (int reg = 0; reg < 12; ++reg)
      for (int lane = 0; lane < 64; ++lane)
        cu->write_vgpr(base + reg, lane, reg >= 8 ? 0xff7f0000U : 0U);
    cu->write_vgpr(base, 0, 0x7f7fU);
    cu->write_vgpr(base + 4, 0, 0x4000U);
    auto words = rocjitsu::cdna4::build_vop3p_mfma(
        rocjitsu::cdna4::kVMfmaF3216x16x32Bf16Vop3pMfma,
        {.vdst = 8, .acc_cd = 0, .src0 = 256, .src1 = 260, .src2 = 264});
    auto instruction = decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    ASSERT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
    EXPECT_EQ(cu->read_vgpr(base + 8, 0), 0x7f7f0000U) << scalar;
  }
}

TEST(ArithBridgeTest, Gfx1250HalfWmmaUsesLibraryPolicyAndInlineAccumulator) {
  struct Case {
    uint16_t opcode;
    bool bf16;
    bool packed;
    bool packed_c;
  };
  for (auto test : {Case{cdna5::kVWmmaF3216x16x32F16Vop3p, false, false, false},
                    Case{cdna5::kVWmmaF3216x16x32Bf16Vop3p, true, false, false},
                    Case{cdna5::kVWmmaF1616x16x32F16Vop3p, false, true, true},
                    Case{cdna5::kVWmmaBf1616x16x32Bf16Vop3p, true, true, true},
                    Case{cdna5::kVWmmaBf16f3216x16x32Bf16Vop3p, true, true, false}}) {
    for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility,
                           ArithmeticSemantics::BitExactEmpiricalNoFpStatus}) {
      ScopedIsaExecutionBackend backend(&cdna5::execution_backend());
      ComputeUnitCore::Config config{};
      config.arch = ROCJITSU_CODE_ARCH_CDNA5;
      config.arithmetic = {semantics};
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 104;
      config.vgprs_per_wf = 32;
      config.lds_size_kb = 64;
      GpuMemory memory("wmma_bridge_memory");
      L2Cache l2("wmma_bridge_l2");
      auto cu = ComputeUnitCore::create("wmma_bridge", config, &memory, &l2);
      auto *wave = cu->dispatch_wf(0, 0, 104, 32, 32);
      ASSERT_NE(wave, nullptr);
      wave->set_exec(1); // WMMA still writes the entire wave.
      const auto base = wave->vgpr_alloc().base;
      for (uint32_t reg = 0; reg < 24; ++reg) {
        for (uint32_t lane = 0; lane < 32; ++lane) {
          cu->write_vgpr(base + reg, lane,
                         reg < 16 ? (test.bf16 ? 0x3f803f80U : 0x3c003c00U) : 0xdeadbeefU);
        }
      }
      auto words = cdna5::build_vop3p(
          test.opcode, {.vdst = 16, .neg_hi = 4, .src0 = 256, .src1 = 264, .src2 = 244, .neg = 4});
      auto decoder = Decoder::create(config.arch);
      auto instruction = decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      const bool exact = semantics == ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
      EXPECT_EQ(cu->execute_instruction(instruction.value().get(), *wave).succeeded(), !exact);
      float value =
          test.packed_c && semantics == ArithmeticSemantics::Compatibility ? 34.0f : 30.0f;
      uint32_t expected = std::bit_cast<uint32_t>(value);
      if (test.packed) {
        uint32_t half = test.bf16 ? util::f32_to_bf16(value) : util::f32_to_f16(value);
        expected = half | (half << 16);
      }
      for (uint32_t reg = 0; reg < 8; ++reg) {
        for (uint32_t lane = 0; lane < 32; ++lane) {
          EXPECT_EQ(cu->read_vgpr(base + 16 + reg, lane),
                    exact || (test.packed && reg >= 4) ? 0xdeadbeefU : expected);
        }
      }
      if (semantics == ArithmeticSemantics::Loose) {
        // Inline -1 has the sentinel's bits but is still C, never the old
        // destination.
        for (uint32_t reg = 0; reg < 16; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            cu->write_vgpr(base + reg, lane, 0);
        for (uint32_t reg = 24; reg < 32; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            cu->write_vgpr(base + reg, lane, 0xffffffffU);
        uint32_t inline_result = 0;
        for (uint16_t selector : {193, 280}) {
          for (uint32_t reg = 16; reg < 24; ++reg)
            for (uint32_t lane = 0; lane < 32; ++lane)
              cu->write_vgpr(base + reg, lane, 0x40004000U);
          auto code = cdna5::build_vop3p(test.opcode,
                                         {.vdst = 16, .src0 = 256, .src1 = 264, .src2 = selector});
          auto decoded = decoder->decode_window(code);
          ASSERT_TRUE(decoded.succeeded());
          ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
          uint32_t actual = cu->read_vgpr(base + 16, 0);
          EXPECT_NE(actual, 0x40004000U);
          if (selector == 193)
            inline_result = actual;
          else
            EXPECT_EQ(actual, inline_result);
        }
        if (test.packed && !test.bf16) {
          // CDNA5 §7.12's WMMA-specific saturation rule, despite §7.7's general
          // wording.
          for (uint32_t reg = 0; reg < 16; ++reg)
            for (uint32_t lane = 0; lane < 32; ++lane)
              cu->write_vgpr(base + reg, lane,
                             0x5c005c00U); // 256 in both halves.
          for (bool saturate : {false, true}) {
            wave->set_mode_raw(saturate ? Wavefront::FP16_OVFL_BIT : 0);
            auto code = cdna5::build_vop3p(test.opcode,
                                           {.vdst = 16, .src0 = 256, .src1 = 264, .src2 = 128});
            auto decoded = decoder->decode_window(code);
            ASSERT_TRUE(decoded.succeeded());
            ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
            EXPECT_EQ(cu->read_vgpr(base + 16, 0), saturate ? 0x7bff7bffU : 0x7c007c00U);
          }
        }
      }
      if (test.bf16 && test.packed && semantics == ArithmeticSemantics::Loose) {
        // CDNA5 §7.12: finite BF16 overflow saturates when FP16_OVFL is set.
        for (uint32_t reg = 0; reg < 24; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            cu->write_vgpr(base + reg, lane, 0);
        cu->write_vgpr(base, 0, 0x7b00U);     // 2^119.
        cu->write_vgpr(base + 8, 0, 0x3f80U); // 1.
        cu->write_vgpr(base + 24, 0, test.packed_c ? 0x7f7fU : 0x7f7f0000U);
        for (bool saturate : {false, true}) {
          wave->set_mode_raw(saturate ? Wavefront::FP16_OVFL_BIT : 0);
          auto code =
              cdna5::build_vop3p(test.opcode, {.vdst = 16, .src0 = 256, .src1 = 264, .src2 = 280});
          auto decoded = decoder->decode_window(code);
          ASSERT_TRUE(decoded.succeeded());
          ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
          EXPECT_EQ(cu->read_vgpr(base + 16, 0) & 0xffffU, saturate ? 0x7f7fU : 0x7f80U);
        }
        wave->set_mode_raw(0);
      }
      if (test.packed_c && semantics == ArithmeticSemantics::Loose) {
        // Integer selectors are raw half encodings; float selectors are
        // narrowed to C's element type before matrix arithmetic (CDNA5
        // section 7.7.2).
        for (uint32_t reg = 0; reg < 16; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            cu->write_vgpr(base + reg, lane, 0);
        for (uint16_t selector : {129, 192, 248}) {
          auto inline_words = cdna5::build_vop3p(
              test.opcode, {.vdst = 16, .src0 = 256, .src1 = 264, .src2 = selector});
          auto inline_instruction = decoder->decode_window(inline_words);
          ASSERT_TRUE(inline_instruction.succeeded());
          ASSERT_TRUE(cu->execute_instruction(inline_instruction.value().get(), *wave).succeeded());
          const uint32_t half = selector == 248 ? (test.bf16 ? 0x3e22U : 0x3118U) : selector - 128;
          EXPECT_EQ(cu->read_vgpr(base + 16, 0), half | (half << 16));
        }
      }
    }
  }
}

// Inline -1 and a register with identical bits must name the same accumulator
// value.
TEST(ArithBridgeTest, CdnaMfmaInlineMinusOneIsNotDestination) {
  for (bool cdna4 : {false, true}) {
    auto arch = cdna4 ? ROCJITSU_CODE_ARCH_CDNA4 : ROCJITSU_CODE_ARCH_CDNA3;
    ScopedIsaExecutionBackend backend(cdna4 ? &rocjitsu::cdna4::execution_backend()
                                            : &rocjitsu::cdna3::execution_backend());
    ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    GpuMemory memory("mfma_inline_memory");
    L2Cache l2("mfma_inline_l2");
    auto cu = ComputeUnitCore::create("mfma_inline", config, &memory, &l2);
    auto *wave = cu->dispatch_wf(0, 0, 104, 256, 64);
    ASSERT_NE(wave, nullptr);
    auto decoder = Decoder::create(arch);
    const auto base = wave->vgpr_alloc().base;
    for (int kind : {0, 1, 2}) {
      const uint32_t outputs = kind == 2 ? 8 : 4;
      for (uint32_t reg = 0; reg < 12; ++reg)
        for (uint32_t lane = 0; lane < 64; ++lane)
          cu->write_vgpr(base + reg, lane,
                         reg < 4                     ? 0
                         : kind == 2 && reg % 2 == 0 ? 0xe0000000U
                                                     : 0xffffffffU);
      uint64_t expected = 0;
      for (uint16_t selector : {260, 193}) {
        for (uint32_t reg = 8; reg < 8 + outputs; ++reg)
          for (uint32_t lane = 0; lane < 64; ++lane)
            cu->write_vgpr(base + reg, lane, 0x40004000U);
        auto words =
            cdna4 ? rocjitsu::cdna4::build_vop3p_mfma(
                        kind == 0   ? rocjitsu::cdna4::kVMfmaF3216x16x16F16Vop3pMfma
                        : kind == 1 ? rocjitsu::cdna4::kVMfmaI3216x16x32I8Vop3pMfma
                                    : rocjitsu::cdna4::kVMfmaF6416x16x4F64Vop3pMfma,
                        {.vdst = 8, .acc_cd = 0, .src0 = 256, .src1 = 258, .src2 = selector})
                  : rocjitsu::cdna3::build_vop3p_mfma(
                        kind == 0   ? rocjitsu::cdna3::kVMfmaF3216x16x16F16Vop3pMfma
                        : kind == 1 ? rocjitsu::cdna3::kVMfmaI3216x16x32I8Vop3pMfma
                                    : rocjitsu::cdna3::kVMfmaF6416x16x4F64Vop3pMfma,
                        {.vdst = 8, .acc_cd = 0, .src0 = 256, .src1 = 258, .src2 = selector});
        auto instruction = decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
        uint64_t actual = cu->read_vgpr(base + 8, 0);
        if (kind == 2)
          actual |= uint64_t(cu->read_vgpr(base + 9, 0)) << 32;
        EXPECT_NE(actual, 0x40004000U);
        if (selector == 260)
          expected = actual;
        else
          EXPECT_EQ(actual, expected);
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaHalfMfmaUsesLibraryPolicyAndAccumulatorBank) {
  for (bool cdna4 : {false, true}) {
    for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility,
                           ArithmeticSemantics::BitExactEmpiricalNoFpStatus}) {
      for (uint8_t acc_cd : {uint8_t{0}, uint8_t{1}}) {
        ScopedIsaExecutionBackend backend(cdna4 ? &rocjitsu::cdna4::execution_backend()
                                                : &rocjitsu::cdna3::execution_backend());
        ComputeUnitCore::Config config{};
        config.arch = cdna4 ? ROCJITSU_CODE_ARCH_CDNA4 : ROCJITSU_CODE_ARCH_CDNA3;
        config.arithmetic = {semantics};
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 104;
        config.vgprs_per_wf = 512;
        config.lds_size_kb = 64;
        GpuMemory memory("mfma_bridge_memory");
        L2Cache l2("mfma_bridge_l2");
        auto cu = ComputeUnitCore::create("mfma_bridge", config, &memory, &l2);
        auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
        auto observer = std::make_unique<WriteRecorder>();
        auto *recorder = observer.get();
        group->add(std::move(observer));
        cu->set_plugin_group(group);
        auto *wave = cu->dispatch_wf(0, 0, 104, 512, 64);
        ASSERT_NE(wave, nullptr);
        wave->set_exec(1);
        const auto base = wave->vgpr_alloc().base;
        const auto dst = base + 16 + (acc_cd ? 256 : 0);
        for (uint32_t lane = 0; lane < 64; ++lane) {
          uint32_t half = util::f32_to_f16(float(lane + 1));
          for (uint32_t reg = 0; reg < 2; ++reg) {
            cu->write_vgpr(base + reg, lane, half | (half << 16));
            cu->write_vgpr(base + 4 + reg, lane, 0x3c003c00);
          }
          for (uint32_t reg = 0; reg < 4; ++reg) {
            cu->write_vgpr(dst + reg, lane, 0xdeadbeef);
          }
        }
        const auto words =
            cdna4
                ? rocjitsu::cdna4::build_vop3p_mfma(rocjitsu::cdna4::kVMfmaF324x4x416bF16Vop3pMfma,
                                                    {.vdst = 16,
                                                     .cbsz = 4,
                                                     .abid = 13,
                                                     .acc_cd = acc_cd,
                                                     .src0 = 256,
                                                     .src1 = 260,
                                                     .src2 = 128})
                : rocjitsu::cdna3::build_vop3p_mfma(rocjitsu::cdna3::kVMfmaF324x4x416bF16Vop3pMfma,
                                                    {.vdst = 16,
                                                     .cbsz = 4,
                                                     .abid = 13,
                                                     .acc_cd = acc_cd,
                                                     .src0 = 256,
                                                     .src1 = 260,
                                                     .src2 = 128});
        auto decoder = Decoder::create(config.arch);
        auto instruction = decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        const bool exact = semantics == ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
        recorder->writes.clear();
        EXPECT_EQ(cu->execute_instruction(instruction.value().get(), *wave).succeeded(), !exact);
        if (exact) {
          EXPECT_TRUE(recorder->writes.empty());
        }
        for (uint32_t reg = 0; reg < 4; ++reg) {
          for (uint32_t lane = 0; lane < 64; ++lane) {
            EXPECT_EQ(cu->read_vgpr(dst + reg, lane),
                      exact ? 0xdeadbeefU : std::bit_cast<uint32_t>(4.0f * float(53 + reg)));
          }
        }
        if (cdna4 && !acc_cd && semantics == ArithmeticSemantics::Loose) {
          // Broadcast controls must not observe lanes that do not contribute.
          auto masked_words = rocjitsu::cdna4::build_vop3p_mfma(
              rocjitsu::cdna4::kVMfmaF3216x16x44bF16Vop3pMfma,
              {.vdst = 16, .cbsz = 2, .abid = 1, .src0 = 256, .src1 = 260, .src2 = 128, .blgp = 4});
          auto masked = decoder->decode_window(masked_words);
          ASSERT_TRUE(masked.succeeded());
          recorder->reads.clear();
          ASSERT_TRUE(cu->execute_instruction(masked.value().get(), *wave).succeeded());
          uint64_t a_lanes = 0, b_lanes = 0;
          for (auto read : recorder->reads) {
            if (read.reg == base)
              a_lanes |= read.lanes;
            if (read.reg == base + 4)
              b_lanes |= read.lanes;
          }
          EXPECT_NE(a_lanes & (1ULL << 16), 0U);
          EXPECT_EQ(a_lanes & 1, 0U);
          EXPECT_NE(b_lanes & (1ULL << 15), 0U);
          EXPECT_EQ(b_lanes & (1ULL << 63), 0U);
        }
      }
    }
  }
}

// Dense gfx1250 decoding preserves inline C, whole-wave writes and the selected
// semantics.
TEST(ArithBridgeTest, Gfx1250MatrixUsesLibraryPolicyAndInlineAccumulator) {
  struct Case {
    uint16_t opcode;
    bool bf16;
    bool packed;
    bool packed_c;
    bool integer = false;
  };
  for (auto test : {Case{cdna5::kVWmmaI3216x16x64Iu8Vop3p, false, false, false, true},
                    Case{cdna5::kVWmmaF3216x16x32F16Vop3p, false, false, false},
                    Case{cdna5::kVWmmaF3216x16x32Bf16Vop3p, true, false, false},
                    Case{cdna5::kVWmmaF1616x16x32F16Vop3p, false, true, true},
                    Case{cdna5::kVWmmaBf1616x16x32Bf16Vop3p, true, true, true},
                    Case{cdna5::kVWmmaBf16f3216x16x32Bf16Vop3p, true, true, false}}) {
    for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility,
                           ArithmeticSemantics::BitExactEmpiricalNoFpStatus}) {
      ScopedIsaExecutionBackend backend(&cdna5::execution_backend());
      ComputeUnitCore::Config config{};
      config.arch = ROCJITSU_CODE_ARCH_CDNA5;
      config.arithmetic = {semantics};
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 104;
      config.vgprs_per_wf = 32;
      config.lds_size_kb = 64;
      GpuMemory memory("wmma_bridge_memory");
      L2Cache l2("wmma_bridge_l2");
      auto cu = ComputeUnitCore::create("wmma_bridge", config, &memory, &l2);
      auto *wave = cu->dispatch_wf(0, 0, 104, 32, 32);
      ASSERT_NE(wave, nullptr);
      wave->set_exec(1); // WMMA still writes the entire wave.
      const auto base = wave->vgpr_alloc().base;
      for (uint32_t reg = 0; reg < 24; ++reg) {
        for (uint32_t lane = 0; lane < 32; ++lane) {
          cu->write_vgpr(base + reg, lane,
                         reg < 16 ? (test.integer ? 0x01010101U
                                     : test.bf16  ? 0x3f803f80U
                                                  : 0x3c003c00U)
                                  : 0xdeadbeefU);
        }
      }
      auto words = cdna5::build_vop3p(
          test.opcode, {.vdst = 16, .neg_hi = 4, .src0 = 256, .src1 = 264, .src2 = 244, .neg = 4});
      auto decoder = Decoder::create(config.arch);
      auto instruction = decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      const bool exact =
          !test.integer && semantics == ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
      EXPECT_EQ(cu->execute_instruction(instruction.value().get(), *wave).succeeded(), !exact);
      float value =
          test.packed_c && semantics == ArithmeticSemantics::Compatibility ? 34.0f : 30.0f;
      uint32_t expected = test.integer ? 0x40000040U : std::bit_cast<uint32_t>(value);
      if (test.packed) {
        uint32_t half = test.bf16 ? util::f32_to_bf16(value) : util::f32_to_f16(value);
        expected = half | (half << 16);
      }
      for (uint32_t reg = 0; reg < 8; ++reg) {
        for (uint32_t lane = 0; lane < 32; ++lane) {
          EXPECT_EQ(cu->read_vgpr(base + 16 + reg, lane),
                    exact || (test.packed && reg >= 4) ? 0xdeadbeefU : expected);
        }
      }
      if (semantics == ArithmeticSemantics::Loose) {
        // Inline -1 has the sentinel's bits but is still C, never the old
        // destination.
        for (uint32_t reg = 0; reg < 16; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            cu->write_vgpr(base + reg, lane, 0);
        for (uint32_t reg = 24; reg < 32; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            cu->write_vgpr(base + reg, lane, 0xffffffffU);
        uint32_t inline_result = 0;
        for (uint16_t selector : {193, 280}) {
          for (uint32_t reg = 16; reg < 24; ++reg)
            for (uint32_t lane = 0; lane < 32; ++lane)
              cu->write_vgpr(base + reg, lane, 0x40004000U);
          auto code = cdna5::build_vop3p(test.opcode,
                                         {.vdst = 16, .src0 = 256, .src1 = 264, .src2 = selector});
          auto decoded = decoder->decode_window(code);
          ASSERT_TRUE(decoded.succeeded());
          ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
          uint32_t actual = cu->read_vgpr(base + 16, 0);
          EXPECT_NE(actual, 0x40004000U);
          if (selector == 193)
            inline_result = actual;
          else
            EXPECT_EQ(actual, inline_result);
        }
        if (test.packed && !test.bf16) {
          // CDNA5 §7.12's WMMA-specific saturation rule, despite §7.7's general
          // wording.
          for (uint32_t reg = 0; reg < 16; ++reg)
            for (uint32_t lane = 0; lane < 32; ++lane)
              cu->write_vgpr(base + reg, lane,
                             0x5c005c00U); // 256 in both halves.
          for (bool saturate : {false, true}) {
            wave->set_mode_raw(saturate ? Wavefront::FP16_OVFL_BIT : 0);
            auto code = cdna5::build_vop3p(test.opcode,
                                           {.vdst = 16, .src0 = 256, .src1 = 264, .src2 = 128});
            auto decoded = decoder->decode_window(code);
            ASSERT_TRUE(decoded.succeeded());
            ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
            EXPECT_EQ(cu->read_vgpr(base + 16, 0), saturate ? 0x7bff7bffU : 0x7c007c00U);
          }
        }
      }
      if (test.bf16 && test.packed && semantics == ArithmeticSemantics::Loose) {
        // CDNA5 §7.12: finite BF16 overflow saturates when FP16_OVFL is set.
        for (uint32_t reg = 0; reg < 24; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            cu->write_vgpr(base + reg, lane, 0);
        cu->write_vgpr(base, 0, 0x7b00U);     // 2^119.
        cu->write_vgpr(base + 8, 0, 0x3f80U); // 1.
        cu->write_vgpr(base + 24, 0, test.packed_c ? 0x7f7fU : 0x7f7f0000U);
        for (bool saturate : {false, true}) {
          wave->set_mode_raw(saturate ? Wavefront::FP16_OVFL_BIT : 0);
          auto code =
              cdna5::build_vop3p(test.opcode, {.vdst = 16, .src0 = 256, .src1 = 264, .src2 = 280});
          auto decoded = decoder->decode_window(code);
          ASSERT_TRUE(decoded.succeeded());
          ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
          EXPECT_EQ(cu->read_vgpr(base + 16, 0) & 0xffffU, saturate ? 0x7f7fU : 0x7f80U);
        }
        wave->set_mode_raw(0);
      }
      if (test.packed_c && semantics == ArithmeticSemantics::Loose) {
        // Integer selectors are raw half encodings; float selectors are
        // narrowed to C's element type before matrix arithmetic (CDNA5
        // section 7.7.2).
        for (uint32_t reg = 0; reg < 16; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            cu->write_vgpr(base + reg, lane, 0);
        for (uint16_t selector : {129, 192, 248}) {
          auto inline_words = cdna5::build_vop3p(
              test.opcode, {.vdst = 16, .src0 = 256, .src1 = 264, .src2 = selector});
          auto inline_instruction = decoder->decode_window(inline_words);
          ASSERT_TRUE(inline_instruction.succeeded());
          ASSERT_TRUE(cu->execute_instruction(inline_instruction.value().get(), *wave).succeeded());
          const uint32_t half = selector == 248 ? (test.bf16 ? 0x3e22U : 0x3118U) : selector - 128;
          EXPECT_EQ(cu->read_vgpr(base + 16, 0), half | (half << 16));
        }
      }
    }
  }
}

// The decoded CDNA bridge forwards all ABID bits and accumulator-bank
// selection.
TEST(ArithBridgeTest, CdnaMatrixUsesLibraryPolicyAndAccumulatorBank) {
  for (bool cdna4 : {false, true}) {
    for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility,
                           ArithmeticSemantics::BitExactEmpiricalNoFpStatus}) {
      for (uint8_t acc_cd : {uint8_t{0}, uint8_t{1}}) {
        ScopedIsaExecutionBackend backend(cdna4 ? &rocjitsu::cdna4::execution_backend()
                                                : &rocjitsu::cdna3::execution_backend());
        ComputeUnitCore::Config config{};
        config.arch = cdna4 ? ROCJITSU_CODE_ARCH_CDNA4 : ROCJITSU_CODE_ARCH_CDNA3;
        config.arithmetic = {semantics};
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 104;
        config.vgprs_per_wf = 512;
        config.lds_size_kb = 64;
        GpuMemory memory("mfma_bridge_memory");
        L2Cache l2("mfma_bridge_l2");
        auto cu = ComputeUnitCore::create("mfma_bridge", config, &memory, &l2);
        auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
        auto observer = std::make_unique<WriteRecorder>();
        auto *recorder = observer.get();
        group->add(std::move(observer));
        cu->set_plugin_group(group);
        auto *wave = cu->dispatch_wf(0, 0, 104, 512, 64);
        ASSERT_NE(wave, nullptr);
        wave->set_exec(1);
        const auto base = wave->vgpr_alloc().base;
        const auto dst = base + 16 + (acc_cd ? 256 : 0);
        for (uint32_t lane = 0; lane < 64; ++lane) {
          uint32_t half = util::f32_to_f16(float(lane + 1));
          for (uint32_t reg = 0; reg < 2; ++reg) {
            cu->write_vgpr(base + reg, lane, half | (half << 16));
            cu->write_vgpr(base + 4 + reg, lane, 0x3c003c00);
          }
          for (uint32_t reg = 0; reg < 4; ++reg) {
            cu->write_vgpr(dst + reg, lane, 0xdeadbeef);
          }
        }
        const auto words =
            cdna4
                ? rocjitsu::cdna4::build_vop3p_mfma(rocjitsu::cdna4::kVMfmaF324x4x416bF16Vop3pMfma,
                                                    {.vdst = 16,
                                                     .cbsz = 4,
                                                     .abid = 13,
                                                     .acc_cd = acc_cd,
                                                     .src0 = 256,
                                                     .src1 = 260,
                                                     .src2 = 128})
                : rocjitsu::cdna3::build_vop3p_mfma(rocjitsu::cdna3::kVMfmaF324x4x416bF16Vop3pMfma,
                                                    {.vdst = 16,
                                                     .cbsz = 4,
                                                     .abid = 13,
                                                     .acc_cd = acc_cd,
                                                     .src0 = 256,
                                                     .src1 = 260,
                                                     .src2 = 128});
        auto decoder = Decoder::create(config.arch);
        auto instruction = decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        const bool exact = semantics == ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
        recorder->writes.clear();
        EXPECT_EQ(cu->execute_instruction(instruction.value().get(), *wave).succeeded(), !exact);
        if (exact) {
          EXPECT_TRUE(recorder->writes.empty());
        }
        for (uint32_t reg = 0; reg < 4; ++reg) {
          for (uint32_t lane = 0; lane < 64; ++lane) {
            EXPECT_EQ(cu->read_vgpr(dst + reg, lane),
                      exact ? 0xdeadbeefU : std::bit_cast<uint32_t>(4.0f * float(53 + reg)));
          }
        }
        if (cdna4 && !acc_cd && semantics == ArithmeticSemantics::Loose) {
          // Broadcast controls must not observe lanes that do not contribute.
          for (auto opcode : {rocjitsu::cdna4::kVMfmaF3216x16x44bF16Vop3pMfma,
                              rocjitsu::cdna4::kVMfmaI3216x16x44bI8Vop3pMfma}) {
            auto masked_words = rocjitsu::cdna4::build_vop3p_mfma(opcode, {.vdst = 16,
                                                                           .cbsz = 2,
                                                                           .abid = 1,
                                                                           .src0 = 256,
                                                                           .src1 = 260,
                                                                           .src2 = 128,
                                                                           .blgp = 4});
            auto masked = decoder->decode_window(masked_words);
            ASSERT_TRUE(masked.succeeded());
            recorder->reads.clear();
            ASSERT_TRUE(cu->execute_instruction(masked.value().get(), *wave).succeeded());
            uint64_t a_lanes = 0, b_lanes = 0;
            for (auto read : recorder->reads) {
              if (read.reg == base)
                a_lanes |= read.lanes;
              if (read.reg == base + 4)
                b_lanes |= read.lanes;
            }
            EXPECT_NE(a_lanes & (1ULL << 16), 0U);
            EXPECT_EQ(a_lanes & 1, 0U);
            EXPECT_NE(b_lanes & (1ULL << 15), 0U);
            EXPECT_EQ(b_lanes & (1ULL << 63), 0U);
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaF64MfmaInlineAccumulatorPrecision) {
  for (bool cdna4 : {false, true}) {
    for (auto semantics :
         {ArithmeticSemantics::Loose, ArithmeticSemantics::BitExactEmpiricalAllowLoose,
          ArithmeticSemantics::Compatibility, ArithmeticSemantics::BitExactEmpiricalNoFpStatus}) {
      for (uint8_t acc_cd : {uint8_t{0}, uint8_t{1}}) {
        ScopedIsaExecutionBackend backend(cdna4 ? &rocjitsu::cdna4::execution_backend()
                                                : &rocjitsu::cdna3::execution_backend());
        ComputeUnitCore::Config config{};
        config.arch = cdna4 ? ROCJITSU_CODE_ARCH_CDNA4 : ROCJITSU_CODE_ARCH_CDNA3;
        config.arithmetic = {semantics};
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 104;
        config.vgprs_per_wf = 512;
        config.lds_size_kb = 64;
        GpuMemory memory("mfma_bridge_memory");
        L2Cache l2("mfma_bridge_l2");
        auto cu = ComputeUnitCore::create("mfma_bridge", config, &memory, &l2);
        auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
        auto observer = std::make_unique<WriteRecorder>();
        auto *recorder = observer.get();
        group->add(std::move(observer));
        cu->set_plugin_group(group);
        auto *wave = cu->dispatch_wf(0, 0, 104, 512, 64);
        ASSERT_NE(wave, nullptr);
        wave->set_exec(1);
        const auto base = wave->vgpr_alloc().base;
        const auto dst = base + 16 + (acc_cd ? 256 : 0);
        for (uint16_t selector : {244, 248, 288}) {
          for (uint32_t lane = 0; lane < 64; ++lane) {

            for (uint32_t reg = 0; reg < 2; ++reg) {
              cu->write_vgpr(base + reg, lane, selector != 244 || reg == 0 ? 0U : 0x3ff00000U);
              cu->write_vgpr(base + 4 + reg, lane, reg == 0 ? 0U : 0x40000000U);
            }
            for (uint32_t reg = 0; reg < 8; ++reg) {
              cu->write_vgpr(dst + reg, lane, 0xdeadbeef);
              // The ISA's FP64 encoding of selector 248 (1/(2*pi)).
              cu->write_vgpr(base + 32 + (acc_cd ? 256 : 0) + reg, lane,
                             reg % 2 ? 0x3fc45f30U : 0x6dc9c882U);
            }
          }
          const auto words =
              cdna4
                  ? rocjitsu::cdna4::build_vop3p_mfma(rocjitsu::cdna4::kVMfmaF6416x16x4F64Vop3pMfma,
                                                      {.vdst = 16,
                                                       .cbsz = 0,
                                                       .abid = 0,
                                                       .acc_cd = acc_cd,
                                                       .src0 = 256,
                                                       .src1 = 260,
                                                       .src2 = selector,
                                                       .blgp = uint8_t(selector == 244 ? 7 : 0)})
                  : rocjitsu::cdna3::build_vop3p_mfma(rocjitsu::cdna3::kVMfmaF6416x16x4F64Vop3pMfma,
                                                      {.vdst = 16,
                                                       .cbsz = 0,
                                                       .abid = 0,
                                                       .acc_cd = acc_cd,
                                                       .src0 = 256,
                                                       .src1 = 260,
                                                       .src2 = selector,
                                                       .blgp = uint8_t(selector == 244 ? 7 : 0)});
          auto decoder = Decoder::create(config.arch);
          auto instruction = decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          const bool exact = semantics == ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
          recorder->writes.clear();
          EXPECT_EQ(cu->execute_instruction(instruction.value().get(), *wave).succeeded(), !exact);
          if (exact) {
            EXPECT_TRUE(recorder->writes.empty());
          }
          const uint64_t expected =
              selector == 244 ? 0x4018000000000000ULL
              : selector == 248 && semantics == ArithmeticSemantics::Compatibility
                  ? 0x3fc45f3060000000ULL
                  : 0x3fc45f306dc9c882ULL;
          for (uint32_t reg = 0; reg < 8; ++reg) {
            for (uint32_t lane = 0; lane < 64; ++lane) {
              EXPECT_EQ(cu->read_vgpr(dst + reg, lane),
                        exact ? 0xdeadbeefU : uint32_t(expected >> (32 * (reg % 2))));
            }
          }
        }
      }
    }
  }
}
TEST(ArithBridgeTest, Fp32MatrixFusedCancellation) {
  for (bool scalar : {false, true}) {
    struct RestoreScalar {
      bool previous = util::force_scalar();
      ~RestoreScalar() { util::set_force_scalar_for_testing(previous); }
    } restore;
    util::set_force_scalar_for_testing(scalar);
    for (bool gfx1250 : {false, true}) {
      auto arch = gfx1250 ? ROCJITSU_CODE_ARCH_CDNA5 : ROCJITSU_CODE_ARCH_CDNA4;
      ScopedIsaExecutionBackend backend(gfx1250 ? &rocjitsu::cdna5::execution_backend()
                                                : &rocjitsu::cdna4::execution_backend());
      ComputeUnitCore::Config config{};
      config.arch = arch;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 104;
      config.vgprs_per_wf = 256;
      config.lds_size_kb = 64;
      GpuMemory memory("fp32_cancel");
      L2Cache l2("fp32_cancel_l2");
      auto cu = ComputeUnitCore::create("fp32_cancel", config, &memory, &l2);
      auto *wave = cu->dispatch_wf(0, 0, 104, 256, gfx1250 ? 32 : 64);
      ASSERT_NE(wave, nullptr);
      auto decoder = Decoder::create(arch);
      const auto base = wave->vgpr_alloc().base;
      for (int reg = 0; reg < 16; ++reg)
        for (int lane = 0; lane < (gfx1250 ? 32 : 64); ++lane)
          cu->write_vgpr(base + reg, lane, reg >= 8 ? 0xff7fffffU : 0U);
      cu->write_vgpr(base, 0, 0x7f7fffffU);
      cu->write_vgpr(base + 4, 0, 0x40000000U);
      auto words =
          gfx1250 ? rocjitsu::cdna5::build_vop3p(rocjitsu::cdna5::kVWmmaF3216x16x4F32Vop3p,
                                                 {.vdst = 8, .src0 = 256, .src1 = 260, .src2 = 264})
                  : rocjitsu::cdna4::build_vop3p_mfma(
                        rocjitsu::cdna4::kVMfmaF324x4x116bF32Vop3pMfma,
                        {.vdst = 8, .acc_cd = 0, .src0 = 256, .src1 = 260, .src2 = 264});
      auto instruction = decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
      EXPECT_EQ(cu->read_vgpr(base + 8, 0), 0x7f7fffffU) << scalar << ":" << gfx1250;
    }
  }
}

TEST(ArithBridgeTest, Gfx1250FloatingWmmaUsesLibraryPolicyAndInlineAccumulator) {
  struct Case {
    uint16_t opcode;
    bool bf16;
    bool packed;
    bool packed_c;
    int k = 32;
    bool fp8 = false;
    bool fp32 = false;
  };
  for (auto test : {Case{cdna5::kVWmmaF3216x16x4F32Vop3p, false, false, false, 4, false, true},
                    Case{cdna5::kVWmmaF3216x16x32F16Vop3p, false, false, false},
                    Case{cdna5::kVWmmaF3216x16x32Bf16Vop3p, true, false, false},
                    Case{cdna5::kVWmmaF1616x16x32F16Vop3p, false, true, true},
                    Case{cdna5::kVWmmaBf1616x16x32Bf16Vop3p, true, true, true},
                    Case{cdna5::kVWmmaBf16f3216x16x32Bf16Vop3p, true, true, false},
                    Case{cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p, false, false, false, 64, true},
                    Case{cdna5::kVWmmaF1616x16x128Fp8Fp8Vop3p, false, true, true, 128, true}}) {
    for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility,
                           ArithmeticSemantics::BitExactEmpiricalNoFpStatus}) {
      ScopedIsaExecutionBackend backend(&cdna5::execution_backend());
      ComputeUnitCore::Config config{};
      config.arch = ROCJITSU_CODE_ARCH_CDNA5;
      config.arithmetic = {semantics};
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 104;
      config.vgprs_per_wf = 64;
      config.lds_size_kb = 64;
      GpuMemory memory("wmma_bridge_memory");
      L2Cache l2("wmma_bridge_l2");
      auto cu = ComputeUnitCore::create("wmma_bridge", config, &memory, &l2);
      auto *wave = cu->dispatch_wf(0, 0, 104, 64, 32);
      ASSERT_NE(wave, nullptr);
      wave->set_exec(1); // WMMA still writes the entire wave.
      const auto base = wave->vgpr_alloc().base;
      uint32_t factors = test.fp32 ? 2 : test.fp8 ? test.k / 8 : 8;
      for (uint32_t reg = 0; reg < 2 * factors + 8; ++reg) {
        for (uint32_t lane = 0; lane < 32; ++lane) {
          cu->write_vgpr(base + reg, lane,
                         reg < 2 * factors ? (test.fp32   ? 0x3f800000U
                                              : test.fp8  ? 0x38383838U
                                              : test.bf16 ? 0x3f803f80U
                                                          : 0x3c003c00U)
                                           : 0xdeadbeefU);
        }
      }
      auto words = cdna5::build_vop3p(test.opcode, {.vdst = uint8_t(2 * factors),
                                                    .neg_hi = 4,
                                                    .src0 = 256,
                                                    .src1 = uint16_t(256 + factors),
                                                    .src2 = 244,
                                                    .neg = 4});
      auto decoder = Decoder::create(config.arch);
      auto instruction = decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      const bool exact = semantics == ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
      EXPECT_EQ(cu->execute_instruction(instruction.value().get(), *wave).succeeded(), !exact);
      float value =
          test.k +
          (test.packed_c && semantics == ArithmeticSemantics::Compatibility ? 2.0f : -2.0f);
      uint32_t expected = std::bit_cast<uint32_t>(value);
      if (test.packed) {
        uint32_t half = test.bf16 ? util::f32_to_bf16(value) : util::f32_to_f16(value);
        expected = half | (half << 16);
      }
      for (uint32_t reg = 0; reg < 8; ++reg) {
        for (uint32_t lane = 0; lane < 32; ++lane) {
          EXPECT_EQ(cu->read_vgpr(base + 2 * factors + reg, lane),
                    exact || (test.packed && reg >= 4) ? 0xdeadbeefU : expected);
        }
      }
      if (semantics == ArithmeticSemantics::Loose) {
        // Inline -1 has the sentinel's bits but is still C, never the old
        // destination.
        for (uint32_t reg = 0; reg < 16; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            cu->write_vgpr(base + reg, lane, 0);
        for (uint32_t reg = 24; reg < 32; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            cu->write_vgpr(base + reg, lane, 0xffffffffU);
        uint32_t inline_result = 0;
        for (uint16_t selector : {193, 280}) {
          for (uint32_t reg = 16; reg < 24; ++reg)
            for (uint32_t lane = 0; lane < 32; ++lane)
              cu->write_vgpr(base + reg, lane, 0x40004000U);
          auto code = cdna5::build_vop3p(test.opcode,
                                         {.vdst = 16, .src0 = 256, .src1 = 264, .src2 = selector});
          auto decoded = decoder->decode_window(code);
          ASSERT_TRUE(decoded.succeeded());
          ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
          uint32_t actual = cu->read_vgpr(base + 16, 0);
          EXPECT_NE(actual, 0x40004000U);
          if (selector == 193)
            inline_result = actual;
          else
            EXPECT_EQ(actual, inline_result);
        }
        if (test.packed && !test.bf16 && !test.fp8) {
          // CDNA5 §7.12's WMMA-specific saturation rule, despite §7.7's general
          // wording.
          for (uint32_t reg = 0; reg < 16; ++reg)
            for (uint32_t lane = 0; lane < 32; ++lane)
              cu->write_vgpr(base + reg, lane,
                             0x5c005c00U); // 256 in both halves.
          for (bool saturate : {false, true}) {
            wave->set_mode_raw(saturate ? Wavefront::FP16_OVFL_BIT : 0);
            auto code = cdna5::build_vop3p(test.opcode,
                                           {.vdst = 16, .src0 = 256, .src1 = 264, .src2 = 128});
            auto decoded = decoder->decode_window(code);
            ASSERT_TRUE(decoded.succeeded());
            ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
            EXPECT_EQ(cu->read_vgpr(base + 16, 0), saturate ? 0x7bff7bffU : 0x7c007c00U);
          }
        }
      }
      if (test.bf16 && test.packed && semantics == ArithmeticSemantics::Loose) {
        // CDNA5 §7.12: finite BF16 overflow saturates when FP16_OVFL is set.
        for (uint32_t reg = 0; reg < 24; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            cu->write_vgpr(base + reg, lane, 0);
        cu->write_vgpr(base, 0, 0x7b00U);     // 2^119.
        cu->write_vgpr(base + 8, 0, 0x3f80U); // 1.
        cu->write_vgpr(base + 24, 0, test.packed_c ? 0x7f7fU : 0x7f7f0000U);
        for (bool saturate : {false, true}) {
          wave->set_mode_raw(saturate ? Wavefront::FP16_OVFL_BIT : 0);
          auto code =
              cdna5::build_vop3p(test.opcode, {.vdst = 16, .src0 = 256, .src1 = 264, .src2 = 280});
          auto decoded = decoder->decode_window(code);
          ASSERT_TRUE(decoded.succeeded());
          ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
          EXPECT_EQ(cu->read_vgpr(base + 16, 0) & 0xffffU, saturate ? 0x7f7fU : 0x7f80U);
        }
        wave->set_mode_raw(0);
      }
      if (test.packed_c && semantics == ArithmeticSemantics::Loose) {
        // Integer selectors are raw half encodings; float selectors are
        // narrowed to C's element type before matrix arithmetic (CDNA5
        // section 7.7.2).
        for (uint32_t reg = 0; reg < 2 * factors; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            cu->write_vgpr(base + reg, lane, 0);
        for (uint16_t selector : {129, 192, 248}) {
          auto inline_words = cdna5::build_vop3p(test.opcode, {.vdst = uint8_t(2 * factors),
                                                               .src0 = 256,
                                                               .src1 = uint16_t(256 + factors),
                                                               .src2 = selector});
          auto inline_instruction = decoder->decode_window(inline_words);
          ASSERT_TRUE(inline_instruction.succeeded());
          ASSERT_TRUE(cu->execute_instruction(inline_instruction.value().get(), *wave).succeeded());
          const uint32_t half = selector == 248 ? (test.bf16 ? 0x3e22U : 0x3118U) : selector - 128;
          EXPECT_EQ(cu->read_vgpr(base + 2 * factors, 0), half | (half << 16));
        }
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaFloatingMfmaInlineMinusOneIsNotDestination) {
  for (bool cdna4 : {false, true}) {
    auto arch = cdna4 ? ROCJITSU_CODE_ARCH_CDNA4 : ROCJITSU_CODE_ARCH_CDNA3;
    ScopedIsaExecutionBackend backend(cdna4 ? &rocjitsu::cdna4::execution_backend()
                                            : &rocjitsu::cdna3::execution_backend());
    ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    GpuMemory memory("mfma_inline_memory");
    L2Cache l2("mfma_inline_l2");
    auto cu = ComputeUnitCore::create("mfma_inline", config, &memory, &l2);
    auto *wave = cu->dispatch_wf(0, 0, 104, 256, 64);
    ASSERT_NE(wave, nullptr);
    auto decoder = Decoder::create(arch);
    const auto base = wave->vgpr_alloc().base;
    for (uint32_t reg = 0; reg < 8; ++reg)
      for (uint32_t lane = 0; lane < 64; ++lane)
        cu->write_vgpr(base + reg, lane, reg < 4 ? 0 : 0xffffffffU);
    const uint32_t opcodes[] = {cdna4 ? rocjitsu::cdna4::kVMfmaF3216x16x16F16Vop3pMfma
                                      : rocjitsu::cdna3::kVMfmaF3216x16x16F16Vop3pMfma,
                                cdna4 ? rocjitsu::cdna4::kVMfmaF3216x16x4F32Vop3pMfma
                                      : rocjitsu::cdna3::kVMfmaF3216x16x4F32Vop3pMfma,
                                cdna4 ? rocjitsu::cdna4::kVMfmaF3216x16x32Fp8Fp8Vop3pMfma
                                      : rocjitsu::cdna3::kVMfmaF3216x16x32Fp8Fp8Vop3pMfma,
                                cdna4 ? rocjitsu::cdna4::kVMfmaF3216x16x32Bf8Bf8Vop3pMfma
                                      : rocjitsu::cdna3::kVMfmaF3216x16x32Bf8Bf8Vop3pMfma,
                                cdna4 ? rocjitsu::cdna4::kVMfmaF3216x16x32Fp8Bf8Vop3pMfma
                                      : rocjitsu::cdna3::kVMfmaF3216x16x32Fp8Bf8Vop3pMfma,
                                cdna4 ? rocjitsu::cdna4::kVMfmaF3216x16x32Bf8Fp8Vop3pMfma
                                      : rocjitsu::cdna3::kVMfmaF3216x16x32Bf8Fp8Vop3pMfma,
                                cdna4 ? rocjitsu::cdna4::kVMfmaF3216x16x4F32Vop3pMfma
                                      : rocjitsu::cdna3::kVMfmaF3216x16x8Xf32Vop3pMfma};
    for (auto opcode : opcodes) {
      SCOPED_TRACE(opcode);
      uint32_t expected = 0;
      for (uint16_t selector : {260, 193}) {
        for (uint32_t reg = 8; reg < 12; ++reg)
          for (uint32_t lane = 0; lane < 64; ++lane)
            cu->write_vgpr(base + reg, lane, 0x40004000U);
        auto words =
            cdna4
                ? rocjitsu::cdna4::build_vop3p_mfma(
                      opcode, {.vdst = 8, .acc_cd = 0, .src0 = 256, .src1 = 258, .src2 = selector})
                : rocjitsu::cdna3::build_vop3p_mfma(
                      opcode, {.vdst = 8, .acc_cd = 0, .src0 = 256, .src1 = 258, .src2 = selector});
        auto instruction = decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
        uint32_t actual = cu->read_vgpr(base + 8, 0);
        EXPECT_NE(actual, 0x40004000U);
        if (selector == 260)
          expected = actual;
        else
          EXPECT_EQ(actual, expected);
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaFloatingMfmaUsesLibraryPolicyAndAccumulatorBank) {
  struct Case {
    bool cdna4, fp8;
    bool fp32 = false;
  };
  for (auto test : {Case{false, false}, Case{true, false}, Case{false, true}, Case{true, true},
                    Case{false, false, true}, Case{true, false, true}}) {
    bool cdna4 = test.cdna4;
    for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility,
                           ArithmeticSemantics::BitExactEmpiricalNoFpStatus}) {
      for (uint8_t acc_cd : {uint8_t{0}, uint8_t{1}}) {
        ScopedIsaExecutionBackend backend(cdna4 ? &rocjitsu::cdna4::execution_backend()
                                                : &rocjitsu::cdna3::execution_backend());
        ComputeUnitCore::Config config{};
        config.arch = cdna4 ? ROCJITSU_CODE_ARCH_CDNA4 : ROCJITSU_CODE_ARCH_CDNA3;
        config.arithmetic = {semantics};
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 104;
        config.vgprs_per_wf = 512;
        config.lds_size_kb = 64;
        GpuMemory memory("mfma_bridge_memory");
        L2Cache l2("mfma_bridge_l2");
        auto cu = ComputeUnitCore::create("mfma_bridge", config, &memory, &l2);
        auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
        auto observer = std::make_unique<WriteRecorder>();
        auto *recorder = observer.get();
        group->add(std::move(observer));
        cu->set_plugin_group(group);
        auto *wave = cu->dispatch_wf(0, 0, 104, 512, 64);
        ASSERT_NE(wave, nullptr);
        wave->set_exec(1);
        const auto base = wave->vgpr_alloc().base;
        const auto dst = base + 16 + (acc_cd ? 256 : 0);
        for (uint32_t lane = 0; lane < 64; ++lane) {
          uint32_t half = util::f32_to_f16(float(lane + 1));
          for (uint32_t reg = 0; reg < 2; ++reg) {
            cu->write_vgpr(base + reg, lane,
                           test.fp32  ? std::bit_cast<uint32_t>(float(lane + 1))
                           : test.fp8 ? 0x40404040U
                                      : half | (half << 16));
            cu->write_vgpr(base + 4 + reg, lane,
                           test.fp32  ? 0x3f800000U
                           : test.fp8 ? 0x40404040U
                                      : 0x3c003c00U);
          }
          for (uint32_t reg = 0; reg < 4; ++reg) {
            cu->write_vgpr(dst + reg, lane, 0xdeadbeef);
          }
        }
        const auto words = cdna4
                               ? rocjitsu::cdna4::build_vop3p_mfma(
                                     test.fp32  ? rocjitsu::cdna4::kVMfmaF324x4x116bF32Vop3pMfma
                                     : test.fp8 ? rocjitsu::cdna4::kVMfmaF3216x16x32Fp8Fp8Vop3pMfma
                                                : rocjitsu::cdna4::kVMfmaF324x4x416bF16Vop3pMfma,
                                     {.vdst = 16,
                                      .cbsz = uint8_t(test.fp8 ? 0 : 4),
                                      .abid = uint8_t(test.fp8 ? 0 : 13),
                                      .acc_cd = acc_cd,
                                      .src0 = 256,
                                      .src1 = 260,
                                      .src2 = 128})
                               : rocjitsu::cdna3::build_vop3p_mfma(
                                     test.fp32  ? rocjitsu::cdna3::kVMfmaF324x4x116bF32Vop3pMfma
                                     : test.fp8 ? rocjitsu::cdna3::kVMfmaF3216x16x32Fp8Fp8Vop3pMfma
                                                : rocjitsu::cdna3::kVMfmaF324x4x416bF16Vop3pMfma,
                                     {.vdst = 16,
                                      .cbsz = uint8_t(test.fp8 ? 0 : 4),
                                      .abid = uint8_t(test.fp8 ? 0 : 13),
                                      .acc_cd = acc_cd,
                                      .src0 = 256,
                                      .src1 = 260,
                                      .src2 = 128});
        auto decoder = Decoder::create(config.arch);
        auto instruction = decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        const bool exact = semantics == ArithmeticSemantics::BitExactEmpiricalNoFpStatus;
        recorder->writes.clear();
        EXPECT_EQ(cu->execute_instruction(instruction.value().get(), *wave).succeeded(), !exact);
        if (exact) {
          EXPECT_TRUE(recorder->writes.empty());
        }
        for (uint32_t reg = 0; reg < 4; ++reg) {
          for (uint32_t lane = 0; lane < 64; ++lane) {
            EXPECT_EQ(cu->read_vgpr(dst + reg, lane),
                      exact ? 0xdeadbeefU
                            : std::bit_cast<uint32_t>(test.fp8 ? (cdna4 ? 128.0f : 32.0f)
                                                               : (test.fp32 ? 1.0f : 4.0f) *
                                                                     float(53 + reg)));
          }
        }
        if (cdna4 && !acc_cd && semantics == ArithmeticSemantics::Loose) {
          // Broadcast controls must not observe lanes that do not contribute.
          auto masked_words = rocjitsu::cdna4::build_vop3p_mfma(
              rocjitsu::cdna4::kVMfmaF3216x16x44bF16Vop3pMfma,
              {.vdst = 16, .cbsz = 2, .abid = 1, .src0 = 256, .src1 = 260, .src2 = 128, .blgp = 4});
          auto masked = decoder->decode_window(masked_words);
          ASSERT_TRUE(masked.succeeded());
          recorder->reads.clear();
          ASSERT_TRUE(cu->execute_instruction(masked.value().get(), *wave).succeeded());
          uint64_t a_lanes = 0, b_lanes = 0;
          for (auto read : recorder->reads) {
            if (read.reg == base)
              a_lanes |= read.lanes;
            if (read.reg == base + 4)
              b_lanes |= read.lanes;
          }
          EXPECT_NE(a_lanes & (1ULL << 16), 0U);
          EXPECT_EQ(a_lanes & 1, 0U);
          EXPECT_NE(b_lanes & (1ULL << 15), 0U);
          EXPECT_EQ(b_lanes & (1ULL << 63), 0U);
        }
      }
    }
  }
}

// Compact SMFMAC selects a metadata set; CBSZ low bits select metadata mode.
TEST(ArithBridgeTest, SparseFloatIndexSelection) {
  for (bool cdna4 : {false, true}) {
    const auto arch = cdna4 ? ROCJITSU_CODE_ARCH_CDNA4 : ROCJITSU_CODE_ARCH_CDNA3;
    ScopedIsaExecutionBackend backend(cdna4 ? &rocjitsu::cdna4::execution_backend()
                                            : &rocjitsu::cdna3::execution_backend());
    ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    GpuMemory memory("sparse_index_memory");
    L2Cache l2("sparse_index_l2");
    auto cu = ComputeUnitCore::create("sparse_index", config, &memory, &l2);
    auto *wave = cu->dispatch_wf(0, 0, 104, 256, 64);
    ASSERT_NE(wave, nullptr);
    const auto base = wave->vgpr_alloc().base;
    auto decoder = Decoder::create(arch);
    for (int kind = 0; kind < (cdna4 ? 11 : 3); ++kind) {
      const bool wide = kind >= 3;
      const bool large = kind == 5 || kind == 6 || kind >= 9;
      const uint32_t output_regs = large ? 16 : 4;
      const uint32_t value = (kind == 2 || kind >= 7)                ? 0x403c3834U
                             : (kind == 1 || kind == 4 || kind == 6) ? 0x40003f80U
                                                                     : 0x40003c00U;
      for (uint32_t lane = 0; lane < 64; ++lane)
        for (uint32_t reg = 0; reg < 16; ++reg)
          cu->write_vgpr(base + reg, lane, value);
      for (uint8_t cbsz : {0, 1, 4})
        for (uint8_t abid : {0, 1, 2, 3, 5}) {
          uint32_t expected[16][64];
          for (bool selected : {false, true}) {
            const int shift = (cbsz & 3) || kind >= 7 ? 0
                              : (kind == 2 || wide)   ? 16 * (abid & 1)
                                                      : 8 * (abid & 3);
            for (uint32_t lane = 0; lane < 64; ++lane) {
              cu->write_vgpr(base + 64, lane, selected ? 0xdd449988U : 0xdd449988U >> shift);
              for (uint32_t reg = 32; reg < 32 + output_regs; ++reg)
                cu->write_vgpr(base + reg, lane, 0);
            }
            const auto words =
                cdna4 ? rocjitsu::cdna4::build_vop3p_mfma(
                            kind == 7    ? rocjitsu::cdna4::kVSmfmacF3216x16x128Fp8Fp8Vop3pMfma
                            : kind == 8  ? rocjitsu::cdna4::kVSmfmacF3216x16x128Bf8Bf8Vop3pMfma
                            : kind == 9  ? rocjitsu::cdna4::kVSmfmacF3232x32x64Fp8Fp8Vop3pMfma
                            : kind == 10 ? rocjitsu::cdna4::kVSmfmacF3232x32x64Bf8Bf8Vop3pMfma
                            : kind == 3  ? rocjitsu::cdna4::kVSmfmacF3216x16x64F16Vop3pMfma
                            : kind == 4  ? rocjitsu::cdna4::kVSmfmacF3216x16x64Bf16Vop3pMfma
                            : kind == 5  ? rocjitsu::cdna4::kVSmfmacF3232x32x32F16Vop3pMfma
                            : kind == 6  ? rocjitsu::cdna4::kVSmfmacF3232x32x32Bf16Vop3pMfma
                            : kind == 0  ? rocjitsu::cdna4::kVSmfmacF3216x16x32F16Vop3pMfma
                            : kind == 1  ? rocjitsu::cdna4::kVSmfmacF3216x16x32Bf16Vop3pMfma
                                         : rocjitsu::cdna4::kVSmfmacF3216x16x64Fp8Fp8Vop3pMfma,
                            {.vdst = 32,
                             .cbsz = uint8_t(selected ? cbsz : 0),
                             .abid = uint8_t(selected ? abid : 0),
                             .acc_cd = 0,
                             .src0 = 256,
                             .src1 = 264,
                             .src2 = 320})
                      : rocjitsu::cdna3::build_vop3p_mfma(
                            kind == 0   ? rocjitsu::cdna3::kVSmfmacF3216x16x32F16Vop3pMfma
                            : kind == 1 ? rocjitsu::cdna3::kVSmfmacF3216x16x32Bf16Vop3pMfma
                                        : rocjitsu::cdna3::kVSmfmacF3216x16x64Fp8Fp8Vop3pMfma,
                            {.vdst = 32,
                             .cbsz = uint8_t(selected ? cbsz : 0),
                             .abid = uint8_t(selected ? abid : 0),
                             .acc_cd = 0,
                             .src0 = 256,
                             .src1 = 264,
                             .src2 = 320});
            auto inst = decoder->decode_window(words);
            ASSERT_TRUE(inst.succeeded());
            ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *wave).succeeded());
            for (uint32_t reg = 0; reg < output_regs; ++reg)
              for (uint32_t lane = 0; lane < 64; ++lane) {
                const auto value = cu->read_vgpr(base + 32 + reg, lane);
                if (selected)
                  EXPECT_EQ(value, expected[reg][lane]);
                else
                  expected[reg][lane] = value;
              }
          }
        }
    }
  }
}

// gfx950 forwards overflow saturation to both integer and floating SMFMAC.
TEST(ArithBridgeTest, SparseCdnaOverflowModeSaturatesResults) {
  ScopedIsaExecutionBackend backend(&rocjitsu::cdna4::execution_backend());
  for (bool integer : {false, true}) {
    ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_CDNA4;
    config.arithmetic.semantics =
        integer ? ArithmeticSemantics::BitExactEmpirical : ArithmeticSemantics::Loose;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    GpuMemory memory("sparse_overflow_memory");
    L2Cache l2("sparse_overflow_l2");
    auto cu = ComputeUnitCore::create("sparse_overflow", config, &memory, &l2);
    auto *wave = cu->dispatch_wf(0, 0, 104, 256, 64);
    ASSERT_NE(wave, nullptr);
    auto decoder = Decoder::create(config.arch);
    const auto base = wave->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < 64; ++lane) {
      for (uint32_t reg = 0; reg < 24; ++reg)
        cu->write_vgpr(base + reg, lane,
                       integer   ? 0x01010101U
                       : reg < 8 ? 0x7d007d00U
                                 : 0x3f803f80U);
      cu->write_vgpr(base + 32, lane, 0x44444444U);
    }
    auto words = cdna4::build_vop3p_mfma(
        integer ? cdna4::kVSmfmacI3216x16x64I8Vop3pMfma : cdna4::kVSmfmacF3216x16x64Bf16Vop3pMfma,
        {.vdst = 24, .acc_cd = 0, .src0 = 256, .src1 = 264, .src2 = 288});
    auto instruction = decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    for (bool saturate : {false, true}) {
      wave->clear_instruction_execution_error();
      wave->set_mode_raw(saturate ? Wavefront::FP16_OVFL_BIT : 0);
      const uint32_t initial = integer ? 0x7fffffffU : 0U;
      for (uint32_t reg = 24; reg < 28; ++reg)
        for (uint32_t lane = 0; lane < 64; ++lane)
          cu->write_vgpr(base + reg, lane, initial);
      ASSERT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
      for (uint32_t reg = 24; reg < 28; ++reg)
        for (uint32_t lane = 0; lane < 64; ++lane)
          EXPECT_EQ(cu->read_vgpr(base + reg, lane), integer
                                                         ? (saturate ? 0x7fffffffU : 0x8000001fU)
                                                         : (saturate ? 0x7f7fffffU : 0x7f800000U));
    }
  }
}

// Reject unsupported CDNA5 half-precision source modifiers without writing outputs.
TEST(ArithBridgeTest, SparseRejectsUnsupportedControls) {
  ScopedIsaExecutionBackend backend(&rocjitsu::cdna5::execution_backend());
  ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 104;
  config.vgprs_per_wf = 256;
  config.lds_size_kb = 64;
  GpuMemory memory("sparse_neg_memory");
  L2Cache l2("sparse_neg_l2");
  auto cu = ComputeUnitCore::create("sparse_neg", config, &memory, &l2);
  auto *wave = cu->dispatch_wf(0, 0, 104, 256, 32);
  ASSERT_NE(wave, nullptr);
  auto decoder = Decoder::create(config.arch);
  const auto base = wave->vgpr_alloc().base;
  for (uint32_t reg = 24; reg < 32; ++reg)
    for (uint32_t lane = 0; lane < 32; ++lane)
      cu->write_vgpr(base + reg, lane, 0xdeadbeefU);
  for (uint16_t opcode : {cdna5::kVSwmmacF3216x16x64F16Vop3p, cdna5::kVSwmmacF3216x16x64Bf16Vop3p,
                          cdna5::kVSwmmacF1616x16x64F16Vop3p, cdna5::kVSwmmacBf1616x16x64Bf16Vop3p,
                          cdna5::kVSwmmacBf16f3216x16x64Bf16Vop3p}) {
    for (int modifier : {1, 2, 4, 8}) {
      wave->clear_instruction_execution_error();
      auto words = cdna5::build_vop3p(opcode, {.vdst = 24,
                                               .neg_hi = uint8_t((modifier >> 2) & 3),
                                               .src0 = 256,
                                               .src1 = 264,
                                               .src2 = 288,
                                               .neg = uint8_t(modifier & 3)});
      auto instruction = decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      EXPECT_FALSE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
      EXPECT_EQ(wave->instruction_execution_error(),
                InstructionExecutionError::UnsupportedOperandValue);
      for (uint32_t reg = 24; reg < 32; ++reg)
        for (uint32_t lane = 0; lane < 32; ++lane)
          EXPECT_EQ(cu->read_vgpr(base + reg, lane), 0xdeadbeefU);
    }
  }
}

TEST(ArithBridgeTest, CdnaPackedFp32SelectorsAndAliases) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      ArithmeticPolicy policy{};
      policy.semantics = semantics;
      const unsigned lanes = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32 : 64;
      BridgeFixture fixture(arch, lanes, policy);
      ASSERT_NE(fixture.wave, nullptr);
      fixture.wave->set_mode_raw(0xf0);
      fixture.wave->set_exec(1ULL | (1ULL << (lanes - 1)));
      for (int operation = 0; operation < 3; ++operation) {
        for (unsigned lane = 0; lane < lanes; ++lane) {
          fixture.cu->write_vgpr(fixture.base(), lane, 0x40000000U);
          fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x40400000U);
          fixture.cu->write_vgpr(fixture.base() + 2, lane, 0x40800000U);
          fixture.cu->write_vgpr(fixture.base() + 3, lane, 0x40a00000U);
        }
        std::array<uint32_t, 2> words;
        if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
          const uint16_t opcodes[] = {cdna3::kVPkAddF32Vop3p, cdna3::kVPkMulF32Vop3p,
                                      cdna3::kVPkFmaF32Vop3p};
          words = cdna3::build_vop3p(opcodes[operation], {.vdst = 0,
                                                          .neg_hi = 0,
                                                          .op_sel = 1,
                                                          .op_sel_hi_2 = 1,
                                                          .src0 = 256,
                                                          .src1 = 258,
                                                          .src2 = 242,
                                                          .op_sel_hi = 1,
                                                          .neg = 2});
        } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
          const uint16_t opcodes[] = {cdna4::kVPkAddF32Vop3p, cdna4::kVPkMulF32Vop3p,
                                      cdna4::kVPkFmaF32Vop3p};
          words = cdna4::build_vop3p(opcodes[operation], {.vdst = 0,
                                                          .neg_hi = 0,
                                                          .op_sel = 1,
                                                          .op_sel_hi_2 = 1,
                                                          .src0 = 256,
                                                          .src1 = 258,
                                                          .src2 = 242,
                                                          .op_sel_hi = 1,
                                                          .neg = 2});
        } else if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
          const uint16_t opcodes[] = {cdna5::kVPkAddF32Vop3p, cdna5::kVPkMulF32Vop3p,
                                      cdna5::kVPkFmaF32Vop3p};
          words = cdna5::build_vop3p(opcodes[operation], {.vdst = 0,
                                                          .neg_hi = 0,
                                                          .opsel = 1,
                                                          .opsel_hi_2 = 1,
                                                          .src0 = 256,
                                                          .src1 = 258,
                                                          .src2 = 242,
                                                          .opsel_hi = 1,
                                                          .neg = 2});
        }
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_FALSE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).failed());
        const uint32_t low[] = {0xbf800000U, 0xc1400000U, 0xc1300000U};
        const uint32_t high[] = {0x40e00000U, 0x41400000U, 0x41500000U};
        for (unsigned lane = 0; lane < lanes; ++lane) {
          const bool active = lane == 0 || lane == lanes - 1;
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                    active ? low[operation] : 0x40000000U);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, lane),
                    active ? high[operation] : 0x40400000U);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaFloatDotPairSumAndAccumulation) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    for (bool bf16 : {false, true}) {
      if (bf16 && arch == ROCJITSU_CODE_ARCH_CDNA3) {
        continue;
      }
      for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
        for (int form = 0; form < 3; ++form) {
          ArithmeticPolicy policy{};
          policy.semantics = semantics;
          BridgeFixture fixture(arch, 64, policy);
          fixture.wave->set_exec(0x8000000000000001ULL);
          fixture.wave->set_mode_raw(0); // Guest denormal controls do not affect DOT.
          fixture.wave->set_trapsts(0x10);
          for (int lane = 0; lane < 64; ++lane) {
            fixture.cu->write_vgpr(fixture.base(), lane, bf16 ? 0x3f803f80U : 0x3c003c00U);
            fixture.cu->write_vgpr(fixture.base() + 1, lane, bf16 ? 0x3f803f80U : 0x3c003c00U);
            fixture.cu->write_vgpr(fixture.base() + 2, lane, 0x4b800000U);
          }
          std::vector<uint32_t> words;
          if (form == 0) {
            const auto encoded =
                arch == ROCJITSU_CODE_ARCH_CDNA3
                    ? cdna3::build_vop3p(cdna3::kVDot2F32F16Vop3p, {.vdst = 2,
                                                                    .clamp = 1,
                                                                    .src0 = 256,
                                                                    .src1 = 257,
                                                                    .src2 = 258,
                                                                    .op_sel_hi = 3})
                    : cdna4::build_vop3p(bf16 ? cdna4::kVDot2F32Bf16Vop3p
                                              : cdna4::kVDot2F32F16Vop3p,
                                         {.vdst = 2,
                                          .clamp = 1,
                                          .src0 = 256,
                                          .src1 = 257,
                                          .src2 = 258,
                                          .op_sel_hi = 3});
            words.assign(encoded.begin(), encoded.end());
          } else if (form == 1) {
            const auto encoded =
                arch == ROCJITSU_CODE_ARCH_CDNA3
                    ? cdna3::build_vop2(cdna3::kVDot2cF32F16Vop2,
                                        {.src0 = 256, .vsrc1 = 1, .vdst = 2})
                    : cdna4::build_vop2(bf16 ? cdna4::kVDot2cF32Bf16Vop2 : cdna4::kVDot2cF32F16Vop2,
                                        {.src0 = 256, .vsrc1 = 1, .vdst = 2});
            words.assign(encoded.begin(), encoded.end());
          } else {
            const auto encoded =
                arch == ROCJITSU_CODE_ARCH_CDNA3
                    ? cdna3::build_vop3(cdna3::kVDot2cF32F16Vop3,
                                        {.vdst = 2, .src0 = 256, .src1 = 257})
                    : cdna4::build_vop3(bf16 ? cdna4::kVDot2cF32Bf16Vop3 : cdna4::kVDot2cF32F16Vop3,
                                        {.vdst = 2, .src0 = 256, .src1 = 257});
            words.assign(encoded.begin(), encoded.end());
          }
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0), 0x4b800001U);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 63), 0x4b800001U);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 1), 0x4b800000U);
          EXPECT_EQ(fixture.wave->trapsts(), 0x10U);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaMixedMadRoundingAndAliasing) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
      for (unsigned rounding = 0; rounding < 4; ++rounding) {
        for (unsigned output = 0; output < 3; ++output) {
          ArithmeticPolicy policy{};
          policy.semantics = semantics;
          BridgeFixture fixture(arch, 64, policy);
          ASSERT_NE(fixture.wave, nullptr);
          fixture.wave->set_mode_raw(rounding << 2);
          fixture.wave->set_exec(0x8000000000000001ULL);
          for (unsigned lane = 0; lane < 64; ++lane) {
            fixture.cu->write_vgpr(fixture.base(), lane, 0x3f800000);
            fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x3f800000);
            fixture.cu->write_vgpr(fixture.base() + 2, lane, 0x3a000000);
          }
          const auto words =
              arch == ROCJITSU_CODE_ARCH_CDNA3
                  ? cdna3::build_vop3p(cdna3::kVMadMixF32Vop3p + output, {.vdst = 2,
                                                                          .op_sel_hi_2 = 0,
                                                                          .src0 = 256,
                                                                          .src1 = 257,
                                                                          .src2 = 258,
                                                                          .op_sel_hi = 0})
                  : cdna4::build_vop3p(cdna4::kVMadMixF32Vop3p + output, {.vdst = 2,
                                                                          .op_sel_hi_2 = 0,
                                                                          .src0 = 256,
                                                                          .src1 = 257,
                                                                          .src2 = 258,
                                                                          .op_sel_hi = 0});
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          uint32_t expected = output == 0 ? 0x3f801000U : rounding == 1 ? 0x3c01U : 0x3c00U;
          if (output == 1) {
            expected |= 0x3a000000U;
          } else if (output == 2) {
            expected <<= 16;
          }
          for (unsigned lane = 0; lane < 64; ++lane) {
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lane),
                      lane == 0 || lane == 63 ? expected : 0x3a000000U);
          }
        }
      }
    }
  }
}

// CDNA3 specifies fused arithmetic even where legacy SIMD rounded the product first.
TEST(ArithBridgeTest, MixedHalfFusedCancellationAndBf16Payload) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (unsigned half = 0; half < 2; ++half) {
      BridgeFixture fixture(arch, arch == ROCJITSU_CODE_ARCH_CDNA3 ? 64 : 32);
      ASSERT_NE(fixture.wave, nullptr);
      fixture.wave->set_mode_raw(0);
      fixture.wave->set_exec(1);
      const bool bf16 = arch == ROCJITSU_CODE_ARCH_CDNA5;
      fixture.cu->write_vgpr(fixture.base(), 0, bf16 ? 0x7fc00001 : 0x47000400);
      fixture.cu->write_vgpr(fixture.base() + 1, 0, bf16 ? 0x3f800000 : 0x3f7ff800);
      fixture.cu->write_vgpr(fixture.base() + 2, 0, bf16 ? 0 : 0xc7000000);
      fixture.cu->write_vgpr(fixture.base() + 3, 0, 0xabcdefff);
      const auto words =
          bf16 ? cdna5::build_vop3p(
                     cdna5::kVFmaMixloBf16Vop3p + half,
                     {.vdst = 3, .src0 = 256, .src1 = 257, .src2 = 258, .opsel_hi = 0})
               : cdna3::build_vop3p(cdna3::kVMadMixloF16Vop3p + half, {.vdst = 3,
                                                                       .op_sel_hi_2 = 0,
                                                                       .src0 = 256,
                                                                       .src1 = 257,
                                                                       .src2 = 258,
                                                                       .op_sel_hi = 0});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      const uint32_t result = bf16 ? 0x7fc1 : 0x9000;
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 3, 0),
                half ? (result << 16) | 0xefffU : 0xabcd0000U | result);
    }
  }
}

TEST(ArithBridgeTest, CdnaMixedFma) {
  const uint16_t opcodes[] = {cdna5::kVFmaMixF32Vop3p, cdna5::kVFmaMixloF16Vop3p,
                              cdna5::kVFmaMixhiF16Vop3p};
  struct Sample {
    uint32_t a, b, result32;
    uint16_t result16;
    bool half_a, inline_a;
  };
  const Sample samples[] = {{0x00010001, 0x3f800000, 0x33800000, 1, true, false},
                            {1, 0x7f000000, 0x34800000, 4, false, false},
                            {0xdeadbeef, 0x3f800000, 0x40000000, 0x4000, true, true}};
  for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
    const uint32_t lanes = 32; // gfx1250 executes Wave32.

    for (unsigned operation = 0; operation < 3; ++operation) {
      for (uint32_t mode : {0u, 0xf0u}) {
        for (const auto &sample : samples) {
          ArithmeticPolicy policy{};
          policy.semantics = semantics;
          BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, lanes, policy);
          ASSERT_NE(fixture.wave, nullptr);
          fixture.wave->set_mode_raw(mode);
          fixture.wave->set_exec((1ULL << (lanes - 1)) | 1);
          for (unsigned lane = 0; lane < lanes; ++lane) {
            fixture.cu->write_vgpr(fixture.base(), lane, sample.a);
            fixture.cu->write_vgpr(fixture.base() + 1, lane, sample.b);
            fixture.cu->write_vgpr(fixture.base() + 2, lane, 0);
          }
          const auto words =
              cdna5::build_vop3p(opcodes[operation], {.vdst = 0,
                                                      .src0 = uint16_t(sample.inline_a ? 244 : 256),
                                                      .src1 = 257,
                                                      .src2 = 258,
                                                      .opsel_hi = uint8_t(sample.half_a)});
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          const uint32_t expected =
              operation == 0   ? (!sample.half_a && mode == 0 ? 0 : sample.result32)
              : operation == 1 ? (sample.a & 0xffff0000) | sample.result16
                               : (sample.a & 0xffff) | (uint32_t(sample.result16) << 16);
          for (unsigned lane = 0; lane < lanes; ++lane) {
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane),
                      lane == 0 || lane == lanes - 1 ? expected : sample.a)
                << operation << '/' << mode << '/' << sample.a << '/' << lane;
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, Bf16SdwaRejectsPendingInputFailure) {
  BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA4, 64);
  fixture.wave->set_exec(1);
  fixture.wave->report_instruction_execution_error(
      InstructionExecutionError::InvalidRegisterAccess);
  cdna4::Operand a(32, cdna4::OperandType::OPR_VGPR, 0);
  cdna4::Operand d(32, cdna4::OperandType::OPR_VGPR, 16);
  int inst = 0;
  arith::execute_bf16_widen_sdwa(inst, *fixture.wave, rj_arith_v_cvt_f32_bf16_w32,
                                 rj_arith_v_cvt_f32_bf16_w64, d, a);
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 16, 0), 0U);
  EXPECT_TRUE(fixture.recorder->writes.empty());
}

TEST(ArithBridgeTest, Gfx1250IntegerConversions) {
  const uint16_t ops[] = {cdna5::kVCvtI32I16Vop3, cdna5::kVCvtU32U16Vop3, cdna5::kVCvtPkI16I32Vop3,
                          cdna5::kVCvtPkU16U32Vop3};
  const uint32_t values[] = {0x80017fffU, 0xffffffffU, 0x7fffffffU, 0x10000U, 0xffff8000U, 0x8000U};
  for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose,
                         ArithmeticSemantics::BitExactEmpirical}) {
    for (int op = 0; op < 4; ++op) {
      for (int variant = 0; variant < 4; ++variant) {
        ArithmeticPolicy policy{};
        policy.semantics = semantics;
        BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, 32, policy);
        bool observed = variant < 2;
        if (!observed) {
          fixture.cu->set_plugin_group(nullptr);
          fixture.recorder = nullptr;
        }
        uint32_t exec_mask = 0x80000015U;
        fixture.wave->set_exec(exec_mask);
        for (int lane = 0; lane < 32; ++lane) {
          fixture.cu->write_vgpr(fixture.base(), lane, values[lane % 6]);
          fixture.cu->write_vgpr(fixture.base() + 1, lane, values[(lane + 1) % 6]);
        }
        auto words = cdna5::build_vop3(ops[op], {.vdst = 0,
                                                 .abs = 7,
                                                 .opsel = uint8_t(variant & 1),
                                                 .clamp = 1,
                                                 .src0 = 256,
                                                 .src1 = uint16_t(op < 2 ? 0 : 257),
                                                 .omod = 3,
                                                 .neg = 7});
        auto inst = fixture.decoder->decode_window(words);
        ASSERT_TRUE(inst.succeeded());
        if (observed) {
          fixture.recorder->reads.clear();
          fixture.recorder->writes.clear();
        }
        ASSERT_TRUE(fixture.cu->execute_instruction(inst.value().get(), *fixture.wave).succeeded());
        if (observed) {
          ASSERT_EQ(fixture.recorder->writes.size(), 1U);
          EXPECT_EQ(fixture.recorder->writes[0].lanes, exec_mask);
          ASSERT_EQ(fixture.recorder->reads.size(), op < 2 ? 1U : 2U);
          EXPECT_EQ(fixture.recorder->reads[0].bytes, op < 2 ? (variant & 1 ? 12 : 3) : 15);
        }
        for (int lane = 0; lane < 32; ++lane) {
          uint32_t a = values[lane % 6], b = values[(lane + 1) % 6], expected = a;
          if ((exec_mask >> lane) & 1) {
            if (op < 2) {
              expected = (a >> ((variant & 1) * 16)) & 0xffffU;
              if (op == 0 && (expected & 0x8000))
                expected |= 0xffff0000U;
            } else {
              auto pack = [&](uint32_t v) {
                int64_t value =
                    op == 2 && (v & 0x80000000U) ? int64_t(v) - (1LL << 32) : int64_t(v);
                return uint32_t(std::clamp<int64_t>(value, op == 2 ? -32768LL : 0LL,
                                                    op == 2 ? 32767LL : 65535LL)) &
                       0xffffU;
              };
              expected = pack(a) | (pack(b) << 16);
            }
          }
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), expected);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaSaturatingPackDestinationWidth) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      for (int form = 0; form < 4; ++form) {
        if (form == 2 && arch == ROCJITSU_CODE_ARCH_CDNA5) {
          continue;
        }
        SCOPED_TRACE(::testing::Message() << arch << ":" << form);
        ArithmeticPolicy policy{};
        policy.semantics = semantics;
        const int lanes = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32 : 64;
        BridgeFixture fixture(arch, lanes, policy);
        fixture.wave->set_exec(1ULL | (1ULL << (lanes - 1)));
        for (int lane = 0; lane < lanes; ++lane) {
          fixture.cu->write_vgpr(fixture.base(), lane, 0x0101ffffU);
        }
        std::vector<uint32_t> words;
        if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
          if (form == 1 || form == 3) {
            const auto encoded =
                cdna3::build_vop3(cdna3::kVSatPkU8I16Vop3,
                                  {.vdst = 0, .op_sel = uint8_t(form == 3 ? 8 : 0), .src0 = 256});
            words.assign(encoded.begin(), encoded.end());
          } else {
            const auto encoded = cdna3::build_vop1(
                cdna3::kVSatPkU8I16Vop1,
                {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
            words.assign(encoded.begin(), encoded.end());
          }
        } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
          if (form == 1 || form == 3) {
            const auto encoded =
                cdna4::build_vop3(cdna4::kVSatPkU8I16Vop3,
                                  {.vdst = 0, .op_sel = uint8_t(form == 3 ? 8 : 0), .src0 = 256});
            words.assign(encoded.begin(), encoded.end());
          } else {
            const auto encoded = cdna4::build_vop1(
                cdna4::kVSatPkU8I16Vop1,
                {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
            words.assign(encoded.begin(), encoded.end());
          }
        } else if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
          if (form == 1 || form == 3) {
            const auto encoded =
                cdna5::build_vop3(cdna5::kVSatPkU8I16Vop3,
                                  {.vdst = 0, .opsel = uint8_t(form == 3 ? 8 : 0), .src0 = 256});
            words.assign(encoded.begin(), encoded.end());
          } else {
            const auto encoded = cdna5::build_vop1(
                cdna5::kVSatPkU8I16Vop1,
                {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
            words.assign(encoded.begin(), encoded.end());
          }
        }
        if (form == 2) {
          // Preserve the old low half and place the packed bytes in WORD1.
          words.push_back((5U << 8) | (2U << 11) | (6U << 16));
        }
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        const uint32_t expected = (form == 2 || form == 3)           ? 0xff00ffffU
                                  : arch == ROCJITSU_CODE_ARCH_CDNA5 ? 0x0101ff00U
                                                                     : 0x0000ff00U;
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), expected);
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lanes - 1), expected);
        EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 1), 0x0101ffffU);
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaBf16Conversions) { check_cdna_float16_conversions(true); }

TEST(ArithBridgeTest, CdnaF16PackedConversions) { check_cdna_float16_conversions(false); }

TEST(ArithBridgeTest, CdnaPackedConversions) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      for (int op = 0; op < 3; ++op) {
        if (op && arch != ROCJITSU_CODE_ARCH_CDNA5) {
          continue;
        }
        for (int destination : {0, 1, 2}) {
          for (bool vop3 : {false, true}) {
            if (!vop3 && (op || arch != ROCJITSU_CODE_ARCH_CDNA5)) {
              continue;
            }
            SCOPED_TRACE(::testing::Message()
                         << arch << ":" << op << ":" << destination << ":" << vop3);
            ArithmeticPolicy policy{};
            policy.semantics = semantics;
            const int lanes = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32 : 64;
            BridgeFixture fixture(arch, lanes, policy);
            fixture.wave->set_exec(1ULL | (1ULL << (lanes - 1)));
            fixture.wave->set_mode_raw(0xff); // directed guest rounding, preserved denormals
            fixture.wave->set_trapsts(0x10);
            for (int lane = 0; lane < lanes; ++lane) {
              fixture.cu->write_vgpr(fixture.base(), lane, 0x3f801800);
              fixture.cu->write_vgpr(fixture.base() + 1, lane, 0xc0700000);
              fixture.cu->write_vgpr(fixture.base() + 2, lane, 0xdeadbeef);
            }
            std::vector<uint32_t> words;
            if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
              const auto encoded =
                  cdna3::build_vop3(cdna3::kVCvtPkrtzF16F32Vop3, {.vdst = uint8_t(destination),
                                                                  .clamp = 1,
                                                                  .src0 = 256,
                                                                  .src1 = 257,
                                                                  .omod = 1,
                                                                  .neg = 1});
              words.assign(encoded.begin(), encoded.end());
            } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
              const auto encoded =
                  cdna4::build_vop3(cdna4::kVCvtPkrtzF16F32Vop3, {.vdst = uint8_t(destination),
                                                                  .clamp = 1,
                                                                  .src0 = 256,
                                                                  .src1 = 257,
                                                                  .omod = 1,
                                                                  .neg = 1});
              words.assign(encoded.begin(), encoded.end());
            } else if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
              constexpr uint16_t ops[] = {cdna5::kVCvtPkRtzF16F32Vop3, cdna5::kVCvtPkI16F32Vop3,
                                          cdna5::kVCvtPkU16F32Vop3};
              if (!vop3) {
                const auto encoded =
                    cdna5::build_vop2(cdna5::kVCvtPkRtzF16F32Vop2,
                                      {.src0 = 256, .vsrc1 = 1, .vdst = uint8_t(destination)});
                words.assign(encoded.begin(), encoded.end());
              } else {
                const auto encoded = cdna5::build_vop3(ops[op], {.vdst = uint8_t(destination),
                                                                 .clamp = 1,
                                                                 .src0 = 256,
                                                                 .src1 = 257,
                                                                 .omod = 1,
                                                                 .neg = 1});
                words.assign(encoded.begin(), encoded.end());
              }
            }
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .succeeded());
            const uint32_t expected = op == 0   ? (vop3 ? 0xc380bc00U : 0xc3803c00U)
                                      : op == 1 ? 0xfffdffffU
                                                : 0;
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, 0), expected);
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, lanes - 1), expected);
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + destination, 1),
                      destination == 0   ? 0x3f801800U
                      : destination == 1 ? 0xc0700000U
                                         : 0xdeadbeefU);
            EXPECT_EQ(fixture.wave->trapsts(), 0x10U);
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaNormalizedConversions) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      for (int op = 0; op < 6; ++op) {
        for (int form = 0; form < 3; ++form) {
          if ((op >= 2 && form != 1) || (form == 2 && arch == ROCJITSU_CODE_ARCH_CDNA5)) {
            continue;
          }
          SCOPED_TRACE(::testing::Message() << arch << ":" << op << ":" << form);
          ArithmeticPolicy policy{};
          policy.semantics = semantics;
          const int lanes = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32 : 64;
          BridgeFixture fixture(arch, lanes, policy);
          fixture.wave->set_exec(1ULL | (1ULL << (lanes - 1)));
          fixture.wave->set_mode_raw(0xff); // directed guest rounding must not affect NORM
          const uint32_t input = op < 4 ? 0xdead3800U : 0x3f000000U;
          const uint32_t other = op < 4 ? 0xdeadbc00U : 0xbf800000U;
          for (int lane = 0; lane < lanes; ++lane) {
            fixture.cu->write_vgpr(fixture.base(), lane, input);
            fixture.cu->write_vgpr(fixture.base() + 1, lane, other);
          }
          std::vector<uint32_t> words;
          if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
            constexpr uint16_t vop3[] = {
                cdna3::kVCvtNormI16F16Vop3,   cdna3::kVCvtNormU16F16Vop3,
                cdna3::kVCvtPknormI16F16Vop3, cdna3::kVCvtPknormU16F16Vop3,
                cdna3::kVCvtPknormI16F32Vop3, cdna3::kVCvtPknormU16F32Vop3};
            if (form == 1) {
              const auto encoded =
                  cdna3::build_vop3(vop3[op], {.vdst = 0, .src0 = 256, .src1 = 257});
              words.assign(encoded.begin(), encoded.end());
            } else {
              const auto encoded = cdna3::build_vop1(
                  op == 0 ? cdna3::kVCvtNormI16F16Vop1 : cdna3::kVCvtNormU16F16Vop1,
                  {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
              words.assign(encoded.begin(), encoded.end());
            }
          } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
            constexpr uint16_t vop3[] = {cdna4::kVCvtNormI16F16Vop3, cdna4::kVCvtNormU16F16Vop3,
                                         cdna4::kVCvtPknormI16F16,   cdna4::kVCvtPknormU16F16,
                                         cdna4::kVCvtPknormI16F32,   cdna4::kVCvtPknormU16F32};
            if (form == 1) {
              const auto encoded =
                  cdna4::build_vop3(vop3[op], {.vdst = 0, .src0 = 256, .src1 = 257});
              words.assign(encoded.begin(), encoded.end());
            } else {
              const auto encoded = cdna4::build_vop1(
                  op == 0 ? cdna4::kVCvtNormI16F16Vop1 : cdna4::kVCvtNormU16F16Vop1,
                  {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
              words.assign(encoded.begin(), encoded.end());
            }
          } else if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
            constexpr uint16_t vop3[] = {
                cdna5::kVCvtNormI16F16Vop3,   cdna5::kVCvtNormU16F16Vop3,
                cdna5::kVCvtPkNormI16F16Vop3, cdna5::kVCvtPkNormU16F16Vop3,
                cdna5::kVCvtPkNormI16F32Vop3, cdna5::kVCvtPkNormU16F32Vop3};
            if (form == 1) {
              const auto encoded =
                  cdna5::build_vop3(vop3[op], {.vdst = 0, .src0 = 256, .src1 = 257});
              words.assign(encoded.begin(), encoded.end());
            } else {
              const auto encoded = cdna5::build_vop1(
                  op == 0 ? cdna5::kVCvtNormI16F16Vop1 : cdna5::kVCvtNormU16F16Vop1,
                  {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
              words.assign(encoded.begin(), encoded.end());
            }
          }
          if (form == 2) {
            words.push_back((5U << 8) | (2U << 11) | (6U << 16));
          }
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          uint32_t expected = op & 1 ? 0x8000U : 0x4000U;
          if (op >= 2) {
            expected |= op & 1 ? 0 : 0x80010000U;
          } else if (form == 2) {
            expected = (expected << 16) | (input & 0xffff);
          } else if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
            expected |= input & 0xffff0000U;
          }
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), expected);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lanes - 1), expected);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 1), input);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaIntegerConversionPackingAndHalves) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (bool is_signed : {false, true}) {
      for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
        for (bool high : {false, true}) {
          ArithmeticPolicy policy{};
          policy.semantics = semantics;
          const bool extend = arch == ROCJITSU_CODE_ARCH_CDNA5;
          const int lanes = extend ? 32 : 64;
          BridgeFixture fixture(arch, lanes, policy);
          fixture.wave->set_exec(1ULL | (1ULL << (lanes - 1)));
          fixture.wave->set_trapsts(0x10);
          for (int lane = 0; lane < lanes; ++lane) {
            fixture.cu->write_vgpr(fixture.base(), lane, extend ? 0x80017fffU : 0xffff0000U);
            fixture.cu->write_vgpr(fixture.base() + 1, lane, 0x10000U);
          }
          std::vector<uint32_t> words;
          if (extend) {
            const auto encoded =
                cdna5::build_vop1(is_signed ? cdna5::kVCvtI32I16Vop1 : cdna5::kVCvtU32U16Vop1,
                                  {.src0 = uint16_t(256 + (high ? 128 : 0)), .vdst = 0});
            words.assign(encoded.begin(), encoded.end());
          } else {
            const auto encoded = arch == ROCJITSU_CODE_ARCH_CDNA3
                                     ? cdna3::build_vop3(is_signed ? cdna3::kVCvtPkI16I32Vop3
                                                                   : cdna3::kVCvtPkU16U32Vop3,
                                                         {.vdst = 0,
                                                          .abs = uint8_t(high ? 3 : 0),
                                                          .clamp = uint8_t(high),
                                                          .src0 = 256,
                                                          .src1 = 257,
                                                          .omod = uint8_t(high ? 3 : 0),
                                                          .neg = uint8_t(high ? 3 : 0)})
                                     : cdna4::build_vop3(is_signed ? cdna4::kVCvtPkI16I32Vop3
                                                                   : cdna4::kVCvtPkU16U32Vop3,
                                                         {.vdst = 0,
                                                          .abs = uint8_t(high ? 3 : 0),
                                                          .clamp = uint8_t(high),
                                                          .src0 = 256,
                                                          .src1 = 257,
                                                          .omod = uint8_t(high ? 3 : 0),
                                                          .neg = uint8_t(high ? 3 : 0)});
            words.assign(encoded.begin(), encoded.end());
          }
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          const uint32_t expected = extend ? (high ? (is_signed ? 0xffff8001U : 0x8001U) : 0x7fffU)
                                           : (is_signed ? 0x7fff8000U : 0xffffffffU);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), expected);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lanes - 1), expected);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 1), extend ? 0x80017fffU : 0xffff0000U);
          EXPECT_EQ(fixture.wave->trapsts(), 0x10U);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaBitPacking) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      for (int op = 0; op < 3; ++op) {
        if (op == 2 && arch == ROCJITSU_CODE_ARCH_CDNA5) {
          continue;
        }
        for (bool ieee : {false, true}) {
          for (bool modifiers : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << arch << ":" << op << ":" << ieee << ":" << modifiers);
            ArithmeticPolicy policy{};
            policy.semantics = semantics;
            const int lanes = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32 : 64;
            BridgeFixture fixture(arch, lanes, policy);
            fixture.wave->set_exec(1ULL | (1ULL << (lanes - 1)));
            fixture.wave->set_mode_raw(0xf1 | (ieee ? 0x200U : 0));
            fixture.wave->set_trapsts(0x10);
            for (int lane = 0; lane < lanes; ++lane) {
              fixture.cu->write_vgpr(fixture.base(), lane, op == 0 ? 0x7c013800 : 0x3fc00000);
              fixture.cu->write_vgpr(fixture.base() + 1, lane,
                                     op == 0 ? 0xbc003c00 : uint32_t(lane));
              fixture.cu->write_vgpr(fixture.base() + 2, lane, 0xabcdef12);
            }
            std::vector<uint32_t> words;
            if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
              constexpr uint16_t ops[] = {cdna3::kVPackB32F16Vop3, cdna3::kVCvtPkU8F32Vop3,
                                          cdna3::kVCvtPkaccumU8F32Vop3};
              const auto encoded =
                  cdna3::build_vop3(ops[op], {.vdst = 2,
                                              .op_sel = uint8_t(op == 0 ? 1 : 0),
                                              .clamp = 1,
                                              .src0 = 256,
                                              .src1 = 257,
                                              .src2 = 258,
                                              .omod = 1,
                                              .neg = uint8_t(modifiers ? (op == 0 ? 2 : 1) : 0)});
              words.assign(encoded.begin(), encoded.end());
            } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
              constexpr uint16_t ops[] = {cdna4::kVPackB32F16Vop3, cdna4::kVCvtPkU8F32Vop3,
                                          cdna4::kVCvtPkaccumU8F32Vop3};
              const auto encoded =
                  cdna4::build_vop3(ops[op], {.vdst = 2,
                                              .op_sel = uint8_t(op == 0 ? 1 : 0),
                                              .clamp = 1,
                                              .src0 = 256,
                                              .src1 = 257,
                                              .src2 = 258,
                                              .omod = 1,
                                              .neg = uint8_t(modifiers ? (op == 0 ? 2 : 1) : 0)});
              words.assign(encoded.begin(), encoded.end());
            } else if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
              constexpr uint16_t ops[] = {cdna5::kVPackB32F16Vop3, cdna5::kVCvtPkU8F32Vop3};
              const auto encoded =
                  cdna5::build_vop3(ops[op], {.vdst = 2,
                                              .opsel = uint8_t(op == 0 ? 1 : 0),
                                              .clamp = 1,
                                              .src0 = 256,
                                              .src1 = 257,
                                              .src2 = 258,
                                              .omod = 1,
                                              .neg = uint8_t(modifiers ? (op == 0 ? 2 : 1) : 0)});
              words.assign(encoded.begin(), encoded.end());
            }
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .succeeded());
            for (int lane : {0, lanes - 1}) {
              uint32_t expected;
              if (op == 0) {
                expected = (modifiers ? 0xbc000000U : 0x3c000000U) |
                           ((ieee || arch == ROCJITSU_CODE_ARCH_CDNA5) ? 0x7e01U : 0x7c01U);
              } else {
                const int shift = (lane & 3) * 8;
                const uint32_t byte = op == 2 ? 1 : modifiers ? 0 : 2;
                expected = (0xabcdef12U & ~(0xffU << shift)) | (byte << shift);
              }
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, lane), expected);
            }
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 1), 0xabcdef12U);
            EXPECT_EQ(fixture.wave->trapsts(), 0x10U);
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaByteAndNibbleConversions) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      for (int operation = 0; operation < 5; ++operation) {
        for (int form = 0; form < 5; ++form) {
          if (form >= 3 && arch == ROCJITSU_CODE_ARCH_CDNA5) {
            continue; // SDWA is absent on gfx1250.
          }
          SCOPED_TRACE(::testing::Message() << arch << ":" << operation << ":" << form);
          ArithmeticPolicy policy{};
          policy.semantics = semantics;
          const int lanes = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32 : 64;
          BridgeFixture fixture(arch, lanes, policy);
          fixture.wave->set_exec(1ULL | (1ULL << (lanes - 1)));
          fixture.wave->set_mode_raw(form == 2 ? Wavefront::IEEE_BIT : 0);
          fixture.wave->set_trapsts(0x10);
          constexpr uint32_t input = 0x7f8001f9U;
          for (int lane = 0; lane < lanes; ++lane) {
            fixture.cu->write_vgpr(fixture.base(), lane, input);
          }
          std::vector<uint32_t> words;
          if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
            constexpr uint16_t vop1[] = {cdna3::kVCvtF32Ubyte0Vop1, cdna3::kVCvtF32Ubyte1Vop1,
                                         cdna3::kVCvtF32Ubyte2Vop1, cdna3::kVCvtF32Ubyte3Vop1,
                                         cdna3::kVCvtOffF32I4Vop1};
            constexpr uint16_t vop3[] = {cdna3::kVCvtF32Ubyte0Vop3, cdna3::kVCvtF32Ubyte1Vop3,
                                         cdna3::kVCvtF32Ubyte2Vop3, cdna3::kVCvtF32Ubyte3Vop3,
                                         cdna3::kVCvtOffF32I4Vop3};
            if (form == 1 || form == 2) {
              const auto encoded =
                  cdna3::build_vop3(vop3[operation], {.vdst = 0, .src0 = 256, .omod = 1});
              words.assign(encoded.begin(), encoded.end());
            } else {
              const auto encoded = cdna3::build_vop1(
                  vop1[operation],
                  {.src0 = uint16_t(form >= 3 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
              words.assign(encoded.begin(), encoded.end());
            }
          } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
            constexpr uint16_t vop1[] = {cdna4::kVCvtF32Ubyte0Vop1, cdna4::kVCvtF32Ubyte1Vop1,
                                         cdna4::kVCvtF32Ubyte2Vop1, cdna4::kVCvtF32Ubyte3Vop1,
                                         cdna4::kVCvtOffF32I4Vop1};
            constexpr uint16_t vop3[] = {cdna4::kVCvtF32Ubyte0Vop3, cdna4::kVCvtF32Ubyte1Vop3,
                                         cdna4::kVCvtF32Ubyte2Vop3, cdna4::kVCvtF32Ubyte3Vop3,
                                         cdna4::kVCvtOffF32I4Vop3};
            if (form == 1 || form == 2) {
              const auto encoded =
                  cdna4::build_vop3(vop3[operation], {.vdst = 0, .src0 = 256, .omod = 1});
              words.assign(encoded.begin(), encoded.end());
            } else {
              const auto encoded = cdna4::build_vop1(
                  vop1[operation],
                  {.src0 = uint16_t(form >= 3 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
              words.assign(encoded.begin(), encoded.end());
            }
          } else if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
            constexpr uint16_t vop1[] = {cdna5::kVCvtF32Ubyte0Vop1, cdna5::kVCvtF32Ubyte1Vop1,
                                         cdna5::kVCvtF32Ubyte2Vop1, cdna5::kVCvtF32Ubyte3Vop1,
                                         cdna5::kVCvtOffF32I4Vop1};
            constexpr uint16_t vop3[] = {cdna5::kVCvtF32Ubyte0Vop3, cdna5::kVCvtF32Ubyte1Vop3,
                                         cdna5::kVCvtF32Ubyte2Vop3, cdna5::kVCvtF32Ubyte3Vop3,
                                         cdna5::kVCvtOffF32I4Vop3};
            if (form == 1 || form == 2) {
              const auto encoded =
                  cdna5::build_vop3(vop3[operation], {.vdst = 0, .src0 = 256, .omod = 1});
              words.assign(encoded.begin(), encoded.end());
            } else {
              const auto encoded = cdna5::build_vop1(
                  vop1[operation],
                  {.src0 = uint16_t(form >= 3 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
              words.assign(encoded.begin(), encoded.end());
            }
          }
          if (form >= 3) {
            // Exercise partial destination preservation and source WORD1 selection.
            const uint32_t destination_selection = form == 3 ? 5 : 6;
            const uint32_t source_selection = form == 3 ? 6 : 5;
            words.push_back((destination_selection << 8) | (2U << 11) | (1U << 14) |
                            (source_selection << 16));
          }
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          const uint32_t selected = form == 4 ? input >> 16 : input;
          const int nibble = int(selected & 7) - int(selected & 8);
          float value =
              operation == 4 ? float(nibble) / 16.0f : float((selected >> (8 * operation)) & 0xff);
          if (form == 1 || form >= 3 || (form == 2 && arch == ROCJITSU_CODE_ARCH_CDNA5)) {
            value *= 2;
          }
          uint32_t expected = std::bit_cast<uint32_t>(value);
          if (form == 3) {
            expected = (expected << 16) | (input & 0xffff);
          }
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), expected);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lanes - 1), expected);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 1), input);
          EXPECT_EQ(fixture.wave->trapsts(), 0x10U);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaFp64Conversions) {
  struct Case {
    uint64_t input;
    uint64_t expected;
    bool wide_output;
  };
  const Case cases[] = {
      {0x80000001U, 0xc1dfffffffc00000ULL, true},  // I32 -> F64, exact.
      {0xffffffffU, 0x41efffffffe00000ULL, true},  // U32 -> F64, exact.
      {0x41e0000000000000ULL, 0x7fffffffU, false}, // I32 saturation.
      {0xfff0000000000000ULL, 0, false},           // Negative infinity -> U32 zero.
      {0x7f800123U, 0x7ff0002460000000ULL, true},  // Widened signaling NaN.
      {0x3ff0000010000000ULL, 0x3f800000U, false}, // F32 rounding tie at 1.
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      for (int operation = 0; operation < 6; ++operation) {
        for (int rounding = 0; rounding < 4; ++rounding) {
          for (bool vop3 : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << arch << ":" << operation << ":" << rounding << ":" << vop3);
            ArithmeticPolicy policy{};
            policy.semantics = semantics;
            const int lanes = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32 : 64;
            BridgeFixture fixture(arch, lanes, policy);
            const auto &c = cases[operation];
            fixture.wave->set_exec(1ULL | (1ULL << (lanes - 1)));
            fixture.wave->set_mode_raw(rounding | (rounding << 2) | (3U << 4) | (3U << 6) |
                                       (rounding & 1 ? Wavefront::IEEE_BIT : 0));
            fixture.wave->set_trapsts(0x10);
            for (int lane = 0; lane < lanes; ++lane) {
              fixture.cu->write_vgpr(fixture.base(), lane, uint32_t(c.input));
              fixture.cu->write_vgpr(fixture.base() + 1, lane, uint32_t(c.input >> 32));
            }
            std::vector<uint32_t> words;
            if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
              constexpr uint16_t vop1_ops[] = {cdna3::kVCvtF64I32Vop1, cdna3::kVCvtF64U32Vop1,
                                               cdna3::kVCvtI32F64Vop1, cdna3::kVCvtU32F64Vop1,
                                               cdna3::kVCvtF64F32Vop1, cdna3::kVCvtF32F64Vop1};
              constexpr uint16_t vop3_ops[] = {cdna3::kVCvtF64I32Vop3, cdna3::kVCvtF64U32Vop3,
                                               cdna3::kVCvtI32F64Vop3, cdna3::kVCvtU32F64Vop3,
                                               cdna3::kVCvtF64F32Vop3, cdna3::kVCvtF32F64Vop3};
              if (vop3) {
                const auto encoded =
                    cdna3::build_vop3(vop3_ops[operation], {.vdst = 0, .src0 = 256, .omod = 1});
                words.assign(encoded.begin(), encoded.end());
              } else {
                const auto encoded =
                    cdna3::build_vop1(vop1_ops[operation], {.src0 = 256, .vdst = 0});
                words.assign(encoded.begin(), encoded.end());
              }
            } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
              constexpr uint16_t vop1_ops[] = {cdna4::kVCvtF64I32Vop1, cdna4::kVCvtF64U32Vop1,
                                               cdna4::kVCvtI32F64Vop1, cdna4::kVCvtU32F64Vop1,
                                               cdna4::kVCvtF64F32Vop1, cdna4::kVCvtF32F64Vop1};
              constexpr uint16_t vop3_ops[] = {cdna4::kVCvtF64I32Vop3, cdna4::kVCvtF64U32Vop3,
                                               cdna4::kVCvtI32F64Vop3, cdna4::kVCvtU32F64Vop3,
                                               cdna4::kVCvtF64F32Vop3, cdna4::kVCvtF32F64Vop3};
              if (vop3) {
                const auto encoded =
                    cdna4::build_vop3(vop3_ops[operation], {.vdst = 0, .src0 = 256, .omod = 1});
                words.assign(encoded.begin(), encoded.end());
              } else {
                const auto encoded =
                    cdna4::build_vop1(vop1_ops[operation], {.src0 = 256, .vdst = 0});
                words.assign(encoded.begin(), encoded.end());
              }
            } else if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
              constexpr uint16_t vop1_ops[] = {cdna5::kVCvtF64I32Vop1, cdna5::kVCvtF64U32Vop1,
                                               cdna5::kVCvtI32F64Vop1, cdna5::kVCvtU32F64Vop1,
                                               cdna5::kVCvtF64F32Vop1, cdna5::kVCvtF32F64Vop1};
              constexpr uint16_t vop3_ops[] = {cdna5::kVCvtF64I32Vop3, cdna5::kVCvtF64U32Vop3,
                                               cdna5::kVCvtI32F64Vop3, cdna5::kVCvtU32F64Vop3,
                                               cdna5::kVCvtF64F32Vop3, cdna5::kVCvtF32F64Vop3};
              if (vop3) {
                const auto encoded =
                    cdna5::build_vop3(vop3_ops[operation], {.vdst = 0, .src0 = 256, .omod = 1});
                words.assign(encoded.begin(), encoded.end());
              } else {
                const auto encoded =
                    cdna5::build_vop1(vop1_ops[operation], {.src0 = 256, .vdst = 0});
                words.assign(encoded.begin(), encoded.end());
              }
            }
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .succeeded());
            uint64_t expected = c.expected;
            if (operation == 4 && ((rounding & 1) || arch == ROCJITSU_CODE_ARCH_CDNA5)) {
              expected |= 0x0008000000000000ULL;
            }
            if (operation == 5 && rounding == 1) {
              ++expected;
            }
            // Preserved output denormals gate OMOD off on older CDNA only.
            if (vop3 && arch == ROCJITSU_CODE_ARCH_CDNA5) {
              if (operation < 2) {
                expected += 1ULL << 52;
              } else if (operation == 5) {
                expected += 1U << 23;
              }
            }
            for (int lane : {0, lanes - 1}) {
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lane), uint32_t(expected));
              EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, lane),
                        uint32_t(c.wide_output ? expected >> 32 : c.input >> 32));
            }
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 1), uint32_t(c.input));
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 1, 1), uint32_t(c.input >> 32));
            EXPECT_EQ(fixture.wave->trapsts(), 0x10U);
          }
        }
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaE32HalfConversionsKeepFullRegisterIndices) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    BridgeFixture fixture(arch, 64);
    fixture.wave->set_exec(1);
    fixture.wave->set_mode_raw(0xf0);
    fixture.cu->write_vgpr(fixture.base(), 0, 0xdead0000);
    fixture.cu->write_vgpr(fixture.base() + 128, 0, 0x12343c00);
    for (bool normalized : {false, true}) {
      const auto words =
          cdna3::build_vop1(normalized ? cdna3::kVCvtNormU16F16Vop1 : cdna3::kVCvtU16F16Vop1,
                            {.src0 = 384, .vdst = 129});
      auto inst = fixture.decoder->decode_window(words);
      ASSERT_TRUE(inst.succeeded());
      ASSERT_TRUE(fixture.cu->execute_instruction(inst.value().get(), *fixture.wave).succeeded());
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 129, 0), normalized ? 65535U : 1U);
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), 0xdead0000U);
    }
  }
}

TEST(ArithBridgeTest, Gfx1250Bf16E32UsesOneBulkHighHalfRead) {
  BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, 32);
  fixture.wave->set_exec(0x5);
  for (int lane : {0, 2})
    fixture.cu->write_vgpr(fixture.base(), lane, 0x3f804000);
  fixture.recorder->reads.clear();
  const auto words = cdna5::build_vop1(cdna5::kVCvtF32Bf16Vop1, {.src0 = 384, .vdst = 2});
  auto inst = fixture.decoder->decode_window(words);
  ASSERT_TRUE(inst.succeeded());
  ASSERT_TRUE(fixture.cu->execute_instruction(inst.value().get(), *fixture.wave).succeeded());
  ASSERT_EQ(fixture.recorder->reads.size(), 1U);
  EXPECT_EQ(fixture.recorder->reads[0].bytes, 0xc);
  EXPECT_EQ(fixture.recorder->reads[0].lanes, 0x5U);
  EXPECT_EQ(fixture.cu->read_vgpr(fixture.base() + 2, 0), 0x3f800000U);
}

TEST(ArithBridgeTest, CdnaHalfConversions) {
  constexpr uint32_t inputs[] = {0xdead0003U, 0xdead0003U, 0xdead3e00U,
                                 0xdead3e00U, 0x477fff00U, 0xdead3e00U};
  constexpr uint32_t outputs[] = {0x4200U, 0x4200U, 1U, 1U, 0x7c00U, 0x3fc00000U};
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      for (int op = 0; op < 6; ++op) {
        for (int form = 0; form < 3; ++form) {
          if (form == 2 && arch == ROCJITSU_CODE_ARCH_CDNA5) {
            continue;
          }
          SCOPED_TRACE(::testing::Message() << arch << ":" << op << ":" << form);
          ArithmeticPolicy policy{};
          policy.semantics = semantics;
          const int lanes = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32 : 64;
          BridgeFixture fixture(arch, lanes, policy);
          fixture.wave->set_exec(1ULL | (1ULL << (lanes - 1)));
          fixture.wave->set_mode_raw(0); // nearest, flush outputs, OMOD enabled
          fixture.wave->set_trapsts(0x10);
          for (int lane = 0; lane < lanes; ++lane) {
            fixture.cu->write_vgpr(fixture.base(), lane, inputs[op]);
          }
          std::vector<uint32_t> words;
          if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
            constexpr uint16_t vop1[] = {cdna3::kVCvtF16I16Vop1, cdna3::kVCvtF16U16Vop1,
                                         cdna3::kVCvtI16F16Vop1, cdna3::kVCvtU16F16Vop1,
                                         cdna3::kVCvtF16F32Vop1, cdna3::kVCvtF32F16Vop1};
            constexpr uint16_t vop3[] = {cdna3::kVCvtF16I16Vop3, cdna3::kVCvtF16U16Vop3,
                                         cdna3::kVCvtI16F16Vop3, cdna3::kVCvtU16F16Vop3,
                                         cdna3::kVCvtF16F32Vop3, cdna3::kVCvtF32F16Vop3};
            if (form == 1) {
              const auto encoded = cdna3::build_vop3(vop3[op], {.vdst = 0, .src0 = 256});
              words.assign(encoded.begin(), encoded.end());
            } else {
              const auto encoded = cdna3::build_vop1(
                  vop1[op], {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
              words.assign(encoded.begin(), encoded.end());
            }
          } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
            constexpr uint16_t vop1[] = {cdna4::kVCvtF16I16Vop1, cdna4::kVCvtF16U16Vop1,
                                         cdna4::kVCvtI16F16Vop1, cdna4::kVCvtU16F16Vop1,
                                         cdna4::kVCvtF16F32Vop1, cdna4::kVCvtF32F16Vop1};
            constexpr uint16_t vop3[] = {cdna4::kVCvtF16I16Vop3, cdna4::kVCvtF16U16Vop3,
                                         cdna4::kVCvtI16F16Vop3, cdna4::kVCvtU16F16Vop3,
                                         cdna4::kVCvtF16F32Vop3, cdna4::kVCvtF32F16Vop3};
            if (form == 1) {
              const auto encoded = cdna4::build_vop3(vop3[op], {.vdst = 0, .src0 = 256});
              words.assign(encoded.begin(), encoded.end());
            } else {
              const auto encoded = cdna4::build_vop1(
                  vop1[op], {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
              words.assign(encoded.begin(), encoded.end());
            }
          } else if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
            constexpr uint16_t vop1[] = {cdna5::kVCvtF16I16Vop1, cdna5::kVCvtF16U16Vop1,
                                         cdna5::kVCvtI16F16Vop1, cdna5::kVCvtU16F16Vop1,
                                         cdna5::kVCvtF16F32Vop1, cdna5::kVCvtF32F16Vop1};
            constexpr uint16_t vop3[] = {cdna5::kVCvtF16I16Vop3, cdna5::kVCvtF16U16Vop3,
                                         cdna5::kVCvtI16F16Vop3, cdna5::kVCvtU16F16Vop3,
                                         cdna5::kVCvtF16F32Vop3, cdna5::kVCvtF32F16Vop3};
            if (form == 1) {
              const auto encoded = cdna5::build_vop3(vop3[op], {.vdst = 0, .src0 = 256});
              words.assign(encoded.begin(), encoded.end());
            } else {
              const auto encoded = cdna5::build_vop1(
                  vop1[op], {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
              words.assign(encoded.begin(), encoded.end());
            }
          }
          if (form == 2) {
            // Full source, OMOD /2, WORD1 destination preserving WORD0.
            words.push_back((5U << 8) | (2U << 11) | (3U << 14) | (6U << 16));
          }
          auto instruction = fixture.decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                          .succeeded());
          uint32_t expected = outputs[op];
          if (form == 2) {
            if (op < 2) {
              expected = 0x3e00;
            }
            if (op == 4) {
              expected = 0x7800;
            }
            if (op == 5) {
              expected = 0x3f400000;
            }
            expected = (expected << 16) | (inputs[op] & 0xffff);
          } else if (op < 5 && arch == ROCJITSU_CODE_ARCH_CDNA5) {
            expected |= inputs[op] & 0xffff0000U;
          }
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), expected);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lanes - 1), expected);
          EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 1), inputs[op]);
          EXPECT_EQ(fixture.wave->trapsts(), 0x10U);
        }
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaFp32IntegerConversions) {
  constexpr uint32_t inputs[] = {0xfeffffffU, 0xffffffffU, 0xbfc00000U,
                                 0x3fc00000U, 0xbfc00000U, 0x80000001U};
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      for (int operation = 0; operation < 6; ++operation) {
        for (int rounding = 0; rounding < 4; ++rounding) {
          for (int form = 0; form < 3; ++form) {
            if (form == 2 && arch == ROCJITSU_CODE_ARCH_CDNA5) {
              continue;
            }
            SCOPED_TRACE(::testing::Message()
                         << arch << ":" << operation << ":" << rounding << ":" << form);
            ArithmeticPolicy policy{};
            policy.semantics = semantics;
            const int lanes = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32 : 64;
            BridgeFixture fixture(arch, lanes, policy);
            fixture.wave->set_exec(1ULL | (1ULL << (lanes - 1)));
            fixture.wave->set_mode_raw(rounding | ((rounding & 1) ? 3U << 4 : 0));
            fixture.wave->set_trapsts(0x10);
            for (int lane = 0; lane < lanes; ++lane) {
              fixture.cu->write_vgpr(fixture.base(), lane, inputs[operation]);
            }
            std::vector<uint32_t> words;
            if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
              constexpr uint16_t vop1_ops[] = {
                  cdna3::kVCvtF32I32Vop1, cdna3::kVCvtF32U32Vop1,    cdna3::kVCvtI32F32Vop1,
                  cdna3::kVCvtU32F32Vop1, cdna3::kVCvtRpiI32F32Vop1, cdna3::kVCvtFlrI32F32Vop1};
              constexpr uint16_t vop3_ops[] = {
                  cdna3::kVCvtF32I32Vop3, cdna3::kVCvtF32U32Vop3,    cdna3::kVCvtI32F32Vop3,
                  cdna3::kVCvtU32F32Vop3, cdna3::kVCvtRpiI32F32Vop3, cdna3::kVCvtFlrI32F32Vop3};
              if (form == 1) {
                const auto encoded =
                    cdna3::build_vop3(vop3_ops[operation], {.vdst = 0, .src0 = 256, .omod = 1});
                words.assign(encoded.begin(), encoded.end());
              } else {
                const auto encoded = cdna3::build_vop1(
                    vop1_ops[operation],
                    {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
                words.assign(encoded.begin(), encoded.end());
              }
            } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
              constexpr uint16_t vop1_ops[] = {
                  cdna4::kVCvtF32I32Vop1, cdna4::kVCvtF32U32Vop1,    cdna4::kVCvtI32F32Vop1,
                  cdna4::kVCvtU32F32Vop1, cdna4::kVCvtRpiI32F32Vop1, cdna4::kVCvtFlrI32F32Vop1};
              constexpr uint16_t vop3_ops[] = {
                  cdna4::kVCvtF32I32Vop3, cdna4::kVCvtF32U32Vop3,    cdna4::kVCvtI32F32Vop3,
                  cdna4::kVCvtU32F32Vop3, cdna4::kVCvtRpiI32F32Vop3, cdna4::kVCvtFlrI32F32Vop3};
              if (form == 1) {
                const auto encoded =
                    cdna4::build_vop3(vop3_ops[operation], {.vdst = 0, .src0 = 256, .omod = 1});
                words.assign(encoded.begin(), encoded.end());
              } else {
                const auto encoded = cdna4::build_vop1(
                    vop1_ops[operation],
                    {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
                words.assign(encoded.begin(), encoded.end());
              }
            } else if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
              constexpr uint16_t vop1_ops[] = {
                  cdna5::kVCvtF32I32Vop1,        cdna5::kVCvtF32U32Vop1,
                  cdna5::kVCvtI32F32Vop1,        cdna5::kVCvtU32F32Vop1,
                  cdna5::kVCvtNearestI32F32Vop1, cdna5::kVCvtFloorI32F32Vop1};
              constexpr uint16_t vop3_ops[] = {
                  cdna5::kVCvtF32I32Vop3,        cdna5::kVCvtF32U32Vop3,
                  cdna5::kVCvtI32F32Vop3,        cdna5::kVCvtU32F32Vop3,
                  cdna5::kVCvtNearestI32F32Vop3, cdna5::kVCvtFloorI32F32Vop3};
              if (form == 1) {
                const auto encoded =
                    cdna5::build_vop3(vop3_ops[operation], {.vdst = 0, .src0 = 256, .omod = 1});
                words.assign(encoded.begin(), encoded.end());
              } else {
                const auto encoded = cdna5::build_vop1(
                    vop1_ops[operation],
                    {.src0 = uint16_t(form == 2 ? amdgpu::SRC_SDWA : 256), .vdst = 0});
                words.assign(encoded.begin(), encoded.end());
              }
            }
            if (form == 2) {
              // Read a full source; place the result in WORD1 and preserve WORD0.
              words.push_back((5U << 8) | (2U << 11) | (1U << 14) | (6U << 16));
            }
            auto instruction = fixture.decoder->decode_window(words);
            ASSERT_TRUE(instruction.succeeded());
            ASSERT_TRUE(fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave)
                            .succeeded());
            uint32_t expected = operation == 0   ? 0xcb800000U + (rounding == 2)
                                : operation == 1 ? (rounding < 2 ? 0x4f800000U : 0x4f7fffffU)
                                : operation == 3 ? 1U
                                : operation == 5 ? (rounding & 1 ? 0xffffffffU : 0U)
                                                 : 0xffffffffU;
            if (operation < 2 && form != 0 &&
                (arch == ROCJITSU_CODE_ARCH_CDNA5 || !(rounding & 1))) {
              expected += 1U << 23;
            }
            if (form == 2) {
              expected = (expected << 16) | (inputs[operation] & 0xffff);
            }
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0), expected);
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), lanes - 1), expected);
            EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 1), inputs[operation]);
            EXPECT_EQ(fixture.wave->trapsts(), 0x10U);
          }
        }
      }
    }
  }
}

// Nibble packs select a true16 destination; the other half must survive.
TEST(ArithBridgeTest, Gfx1250NibblePackDestinationHalf) {
  BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, 32);
  fixture.wave->set_exec(1);
  fixture.cu->write_vgpr(fixture.base() + 1, 0, 0x7f80ff01U);
  for (bool is_signed : {false, true}) {
    for (bool high : {false, true}) {
      fixture.cu->write_vgpr(fixture.base(), 0, 0xaaaa5555U);
      auto words = cdna5::build_vop3(is_signed ? cdna5::kVSatPk4I4I8Vop3 : cdna5::kVSatPk4U4U8Vop3,
                                     {.vdst = 0, .opsel = uint8_t(high ? 8 : 0), .src0 = 257});
      auto inst = fixture.decoder->decode_window(words);
      ASSERT_TRUE(inst.succeeded());
      ASSERT_TRUE(fixture.cu->execute_instruction(inst.value().get(), *fixture.wave).succeeded());
      const uint32_t packed = is_signed ? 0x78f1U : 0xfff1U;
      EXPECT_EQ(fixture.cu->read_vgpr(fixture.base(), 0),
                high ? (packed << 16) | 0x5555U : 0xaaaa0000U | packed);
    }
  }
}

TEST(ArithBridgeTest, Gfx1250ScalarConversionCompatibility) {
  for (bool packed : {false, true}) {
    BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, 32, {});
    fixture.wave->set_exec(0);
    fixture.wave->set_mode_raw(0);
    fixture.wave->set_trapsts(0x80000000U);
    fixture.wave->write_scc(true);
    auto base = fixture.wave->sgpr_alloc().base;
    fixture.cu->write_sgpr(base, packed ? 0x7f800001U : 0x00010000U);
    fixture.cu->write_sgpr(base + 1, 0xb3800000U);
    const auto words =
        packed ? cdna5::build_sop2(cdna5::kSCvtPkRtzF16F32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0})
               : cdna5::build_sop1(cdna5::kSCvtHiF32F16Sop1, {.ssrc0 = 0, .sdst = 0});
    auto instruction = fixture.decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    ASSERT_TRUE(
        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
    EXPECT_EQ(fixture.cu->read_sgpr(base), packed ? 0x80017c01U : 0x33800000U);
    EXPECT_TRUE(fixture.wave->read_scc());
    EXPECT_EQ(fixture.wave->exec(), 0U);
    EXPECT_EQ(fixture.wave->trapsts(), 0x80000000U);
  }
}

// Loose honors overflow control; Compatibility retains gfx1250's legacy fixed narrowing.
TEST(ArithBridgeTest, Gfx1250ScalarHalfFmacOverflow) {
  for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
    BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, 32, ArithmeticPolicy{semantics});
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fixture.wave->set_exec(0);
    const auto base = fixture.wave->sgpr_alloc().base;
    fixture.cu->write_sgpr(base, 0x7bffu);
    fixture.cu->write_sgpr(base + 1, 0x4000u);
    const auto words = cdna5::build_sop2(cdna5::kSFmacF16Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 2});
    auto instruction = fixture.decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    for (bool saturate : {false, true}) {
      // Reset old D: an infinity left by the first call would conceal a missing flag.
      fixture.cu->write_sgpr(base + 2, 0);
      fixture.wave->set_mode_raw(saturate ? Wavefront::FP16_OVFL_BIT : 0);
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      const bool clamp = saturate && semantics == ArithmeticSemantics::Loose;
      EXPECT_EQ(fixture.cu->read_sgpr(base + 2) & 0xffffu, clamp ? 0x7bffu : 0x7c00u);
    }
  }
}

TEST(ArithBridgeTest, Gfx1250ScalarFmaCompatibility) {
  for (uint32_t rounding = 0; rounding < 4; ++rounding) {
    BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, 32, {});
    fixture.wave->set_exec(0);
    fixture.wave->set_mode_raw(rounding | (3U << 4));
    fixture.wave->set_trapsts(0x80000000U);
    fixture.wave->write_scc(true);
    auto base = fixture.wave->sgpr_alloc().base;
    fixture.cu->write_sgpr(base, 0x33800000U);
    fixture.cu->write_sgpr(base + 1, 0x3f800000U);
    const auto words = cdna5::build_sop2(cdna5::kSFmacF32Sop2, {.ssrc0 = 1, .ssrc1 = 1, .sdst = 0});
    auto instruction = fixture.decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    ASSERT_TRUE(
        fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
    EXPECT_EQ(fixture.cu->read_sgpr(base), rounding == 1 ? 0x3f800001U : 0x3f800000U);
    EXPECT_TRUE(fixture.wave->read_scc());
    EXPECT_EQ(fixture.wave->exec(), 0U);
    EXPECT_EQ(fixture.wave->trapsts(), 0x80000000U);
  }
}

TEST(ArithBridgeTest, Gfx1250ScalarBinaryCompatibility) {
  for (bool half : {false, true}) {
    for (bool select : {false, true}) {
      BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, 32, {});
      fixture.wave->set_exec(0);
      fixture.wave->set_mode_raw(0);
      fixture.wave->set_trapsts(0x80000000U);
      fixture.wave->write_scc(true);
      const auto base = fixture.wave->sgpr_alloc().base;
      fixture.cu->write_sgpr(base, 1);
      fixture.cu->write_sgpr(base + 1, 1);
      const auto opcode = half ? (select ? cdna5::kSMaxNumF16Sop2 : cdna5::kSAddF16Sop2)
                               : (select ? cdna5::kSMaxNumF32Sop2 : cdna5::kSAddF32Sop2);
      const auto words = cdna5::build_sop2(opcode, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      EXPECT_EQ(fixture.cu->read_sgpr(base), select ? 0U : 2U);
      EXPECT_TRUE(fixture.wave->read_scc());
      EXPECT_EQ(fixture.wave->trapsts(), 0x80000000U);
      EXPECT_EQ(fixture.wave->exec(), 0U);
    }
  }
}

TEST(ArithBridgeTest, Gfx1250ScalarRoundingCompatibility) {
  for (bool half : {false, true}) {
    for (uint32_t input : {1U, half ? 0x7c01U : 0x7f800001U}) {
      BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, 32, {});
      fixture.wave->set_exec(0);
      fixture.wave->set_mode_raw(0);
      fixture.wave->set_trapsts(0x80000000U);
      fixture.wave->write_scc(true);
      const auto base = fixture.wave->sgpr_alloc().base;
      fixture.cu->write_sgpr(base, input);
      const auto words = cdna5::build_sop1(half ? cdna5::kSCeilF16Sop1 : cdna5::kSCeilF32Sop1,
                                           {.ssrc0 = 0, .sdst = 0});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      EXPECT_EQ(fixture.cu->read_sgpr(base),
                input == 1 ? (half ? 0x3c00U : 0x3f800000U) : input | (half ? 0x200U : 0x400000U));
      EXPECT_TRUE(fixture.wave->read_scc());
      EXPECT_EQ(fixture.wave->trapsts(), 0x80000000U);
      EXPECT_EQ(fixture.wave->exec(), 0U);
    }
  }
}

TEST(ArithBridgeTest, CdnaScalarIntegerCarryFieldsAndCounts) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (bool carry : {false, true}) {
      for (int operation = 0; operation < 4; ++operation) {
        BridgeFixture fixture(arch, 32, {});
        fixture.wave->set_exec(0);
        fixture.wave->write_scc(carry);
        const auto base = fixture.wave->sgpr_alloc().base;
        fixture.cu->write_sgpr(base, operation == 0   ? 0xffffffffU
                                     : operation == 3 ? 0x80000000U
                                                      : 0xfU);
        fixture.cu->write_sgpr(base + 1, 0xffffffffU);
        fixture.cu->write_sgpr(base + 2, operation == 0 ? 0 : operation == 3 ? 1 : 0x00200010U);
        std::array<uint32_t, 1> words;
        if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
          if (operation == 2) {
            words = cdna3::build_sop1(cdna3::kSBcnt1I32B64Sop1, {.ssrc0 = 0, .sdst = 0});
          } else {
            words = cdna3::build_sop2(operation == 0   ? cdna3::kSAddcU32Sop2
                                      : operation == 1 ? cdna3::kSBfeU64Sop2
                                                       : cdna3::kSAbsdiffI32Sop2,
                                      {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0});
          }
        } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
          if (operation == 2) {
            words = cdna4::build_sop1(cdna4::kSBcnt1I32B64Sop1, {.ssrc0 = 0, .sdst = 0});
          } else {
            words = cdna4::build_sop2(operation == 0   ? cdna4::kSAddcU32Sop2
                                      : operation == 1 ? cdna4::kSBfeU64Sop2
                                                       : cdna4::kSAbsdiffI32Sop2,
                                      {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0});
          }
        } else if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
          if (operation == 2) {
            words = cdna5::build_sop1(cdna5::kSBcnt1I32B64Sop1, {.ssrc0 = 0, .sdst = 0});
          } else {
            words = cdna5::build_sop2(operation == 0   ? cdna5::kSAddCoCiU32Sop2
                                      : operation == 1 ? cdna5::kSBfeU64Sop2
                                                       : cdna5::kSAbsdiffI32Sop2,
                                      {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0});
          }
        }
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        const uint32_t expected = operation == 0   ? (carry ? 0 : 0xffffffffU)
                                  : operation == 1 ? 0xffff0000U
                                  : operation == 2 ? 36
                                                   : 0x7fffffffU;
        EXPECT_EQ(fixture.cu->read_sgpr(base), expected);
        EXPECT_EQ(fixture.cu->read_sgpr(base + 1), operation == 1 ? 0 : 0xffffffffU);
        EXPECT_EQ(fixture.wave->read_scc(), operation == 0 ? carry : true);
        EXPECT_EQ(fixture.wave->exec(), 0U);
      }
    }
  }
}

TEST(ArithBridgeTest, CdnaScalarScansPreserveCondition) {
  struct Case {
    uint64_t input;
    uint32_t first, sign;
  };
  constexpr Case cases[] = {{0, UINT32_MAX, UINT32_MAX},
                            {1, 0, 63},
                            {0x8000000000000000ULL, 63, 1},
                            {UINT64_MAX, 0, UINT32_MAX}};
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (bool signed_scan : {false, true}) {
      for (const auto &test : cases) {
        BridgeFixture fixture(arch, 32, {});
        fixture.wave->set_exec(0);
        fixture.wave->write_scc(signed_scan);
        const auto base = fixture.wave->sgpr_alloc().base;
        fixture.cu->write_sgpr(base, uint32_t(test.input));
        fixture.cu->write_sgpr(base + 1, uint32_t(test.input >> 32));
        std::array<uint32_t, 1> words;
        if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
          words = cdna3::build_sop1(signed_scan ? cdna3::kSFlbitI32I64Sop1 : cdna3::kSFf1I32B64Sop1,
                                    {.ssrc0 = 0, .sdst = 0});
        } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
          words = cdna4::build_sop1(signed_scan ? cdna4::kSFlbitI32I64Sop1 : cdna4::kSFf1I32B64Sop1,
                                    {.ssrc0 = 0, .sdst = 0});
        } else {
          words = cdna5::build_sop1(signed_scan ? cdna5::kSClsI32I64Sop1 : cdna5::kSCtzI32B64Sop1,
                                    {.ssrc0 = 0, .sdst = 0});
        }
        auto instruction = fixture.decoder->decode_window(words);
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(
            fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
        EXPECT_EQ(fixture.cu->read_sgpr(base), signed_scan ? test.sign : test.first);
        EXPECT_EQ(fixture.cu->read_sgpr(base + 1), uint32_t(test.input >> 32));
        EXPECT_EQ(fixture.wave->read_scc(), signed_scan);
        EXPECT_EQ(fixture.wave->exec(), 0U);
      }
    }
  }
}

TEST(ArithBridgeTest, Gfx1250ScalarCompareCompatibility) {
  for (bool half : {false, true}) {
    for (uint32_t input : {1U, half ? 0x7c01U : 0x7f800001U}) {
      BridgeFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, 32, {});
      fixture.wave->set_exec(0);
      fixture.wave->set_mode_raw(0); // Request guest input/output flushing.
      fixture.wave->set_trapsts(0x80000000U);
      fixture.cu->write_sgpr(fixture.wave->sgpr_alloc().base, input);
      fixture.cu->write_sgpr(fixture.wave->sgpr_alloc().base + 1, 0);
      const auto words = cdna5::build_sopc(half ? cdna5::kSCmpEqF16Sopc : cdna5::kSCmpEqF32Sopc,
                                           {.ssrc0 = 0, .ssrc1 = 1});
      auto instruction = fixture.decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_TRUE(
          fixture.cu->execute_instruction(instruction.value().get(), *fixture.wave).succeeded());
      EXPECT_FALSE(fixture.wave->read_scc());
      EXPECT_EQ(fixture.wave->trapsts(), 0x80000000U);
      EXPECT_EQ(fixture.wave->exec(), 0U);
    }
  }
}

} // namespace
} // namespace rocjitsu::amdgpu
