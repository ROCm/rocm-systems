// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded older-target FP16 conversions: inherited guest policies, full-word
// versus preserved-half destinations, source selection, aliases and EXEC.

#include "compute_unit_test_util.h"
#include "conversion_binding.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/execution_backend.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <random>

namespace rocjitsu::amdgpu {
namespace {

// Floating destinations check decoder/API binding against the separately tested
// scalar conversion APIs; integer destinations use independent bounded arithmetic.
uint32_t reference(int operation, uint32_t raw, const Wavefront &wave, unsigned omod, bool clamp,
                   bool absolute, bool negate, bool sdwa) {
  if (operation < 2 || operation >= 4) {
    const bool wide = operation == 5;
    const unsigned source_denorm =
        operation == 4 ? wave.fp_denorm_mode_f32() : wave.fp_denorm_mode_f16_f64();
    const unsigned destination_denorm =
        wide ? wave.fp_denorm_mode_f32() : wave.fp_denorm_mode_f16_f64();
    const unsigned rounding = wide ? wave.fp_round_mode_f32() : wave.fp_round_mode_f16_f64();
    uint64_t flags = RJ_ARITH_SEMANTICS_COMPATIBILITY | (uint64_t(rounding) << 24);
    flags |= wave.ieee_mode() ? RJ_ARITH_FP_IEEE_MODE : 0;
    flags |= wave.dx10_clamp() ? 0 : RJ_ARITH_FP_DX10_CLAMP_OFF;
    flags |= !wide && wave.fp16_ovfl() ? RJ_ARITH_FP16_OVFL : 0;
    flags |= operation >= 4 && !(source_denorm & 1) ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0;
    flags |= (destination_denorm & 2) ? 0 : RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
    uint64_t modifiers = (uint64_t(omod) << 6) | (clamp ? RJ_ARITH_ALU_CLAMP : 0) |
                         (operation >= 4 && absolute ? RJ_ARITH_ALU_ABS_A : 0) |
                         (operation >= 4 && negate ? RJ_ARITH_ALU_NEG_A : 0);
    modifiers |= sdwa ? RJ_ARITH_CVT_SDWA : 0;
    const auto function = operation == 0   ? rj_arith_v_cvt_f16_i16_w32_gfx942
                          : operation == 1 ? rj_arith_v_cvt_f16_u16_w32_gfx942
                          : operation == 4 ? rj_arith_v_cvt_f16_f32_w32_gfx942
                                           : rj_arith_v_cvt_f32_f16_w32;
    return uint32_t(rocjitsu::test::conversion_binding(function, flags, modifiers, raw));
  }
  raw = (absolute ? raw & 0x7fffU : raw) ^ (negate ? 0x8000U : 0);
  const unsigned magnitude = raw & 0x7fffU;
  const unsigned exponent = magnitude >> 10;
  // Decode the half exactly in FP32; only the bounded integer cast rounds.
  const float significand = float((raw & 0x3ffU) | (exponent ? 0x400U : 0));
  float x = exponent == 31 ? std::numeric_limits<float>::infinity()
                           : std::ldexp(significand, exponent ? int(exponent) - 25 : -24);
  if (exponent == 31 && (raw & 0x3ffU)) {
    x = std::numeric_limits<float>::quiet_NaN();
  }
  if (raw & 0x8000U) {
    x = -x;
  }
  // Integer conversions truncate after saturation; the bounded cast is defined for every input.
  const float bounded = std::isnan(x) ? 0.0f
                                      : std::clamp(x, operation == 2 ? -32768.0f : 0.0f,
                                                   operation == 2 ? 32767.0f : 65535.0f);
  return uint16_t(int32_t(bounded));
}

// Restore the caller's test override even when an assertion exits the test.
struct ScalarGuard {
  bool saved = util::force_scalar();
  ~ScalarGuard() { util::set_force_scalar_for_testing(saved); }
};

// Exercise the same models through each real target's decoder and operand wrappers.
TEST(OlderConversion16, Arithmetic) {
  ScalarGuard guard;
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<std::array<uint16_t, 2>, 6> opcodes;
  };
  // Order: I16/U16 to F16, F16 to I16/U16, F32 to F16, F16 to F32.
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {{
           {cdna1::kVCvtF16I16Vop1, cdna1::kVCvtF16I16Vop3},
           {cdna1::kVCvtF16U16Vop1, cdna1::kVCvtF16U16Vop3},
           {cdna1::kVCvtI16F16Vop1, cdna1::kVCvtI16F16Vop3},
           {cdna1::kVCvtU16F16Vop1, cdna1::kVCvtU16F16Vop3},
           {cdna1::kVCvtF16F32Vop1, cdna1::kVCvtF16F32Vop3},
           {cdna1::kVCvtF32F16Vop1, cdna1::kVCvtF32F16Vop3},
       }}},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {{
           {cdna2::kVCvtF16I16Vop1, cdna2::kVCvtF16I16Vop3},
           {cdna2::kVCvtF16U16Vop1, cdna2::kVCvtF16U16Vop3},
           {cdna2::kVCvtI16F16Vop1, cdna2::kVCvtI16F16Vop3},
           {cdna2::kVCvtU16F16Vop1, cdna2::kVCvtU16F16Vop3},
           {cdna2::kVCvtF16F32Vop1, cdna2::kVCvtF16F32Vop3},
           {cdna2::kVCvtF32F16Vop1, cdna2::kVCvtF32F16Vop3},
       }}},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {{
           {rdna1::kVCvtF16I16Vop1, rdna1::kVCvtF16I16Vop3},
           {rdna1::kVCvtF16U16Vop1, rdna1::kVCvtF16U16Vop3},
           {rdna1::kVCvtI16F16Vop1, rdna1::kVCvtI16F16Vop3},
           {rdna1::kVCvtU16F16Vop1, rdna1::kVCvtU16F16Vop3},
           {rdna1::kVCvtF16F32Vop1, rdna1::kVCvtF16F32Vop3},
           {rdna1::kVCvtF32F16Vop1, rdna1::kVCvtF32F16Vop3},
       }}},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {{
           {rdna2::kVCvtF16I16Vop1, rdna2::kVCvtF16I16Vop3},
           {rdna2::kVCvtF16U16Vop1, rdna2::kVCvtF16U16Vop3},
           {rdna2::kVCvtI16F16Vop1, rdna2::kVCvtI16F16Vop3},
           {rdna2::kVCvtU16F16Vop1, rdna2::kVCvtU16F16Vop3},
           {rdna2::kVCvtF16F32Vop1, rdna2::kVCvtF16F32Vop3},
           {rdna2::kVCvtF32F16Vop1, rdna2::kVCvtF32F16Vop3},
       }}},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {{
           {rdna3_5::kVCvtF16I16Vop1, rdna3_5::kVCvtF16I16Vop3},
           {rdna3_5::kVCvtF16U16Vop1, rdna3_5::kVCvtF16U16Vop3},
           {rdna3_5::kVCvtI16F16Vop1, rdna3_5::kVCvtI16F16Vop3},
           {rdna3_5::kVCvtU16F16Vop1, rdna3_5::kVCvtU16F16Vop3},
           {rdna3_5::kVCvtF16F32Vop1, rdna3_5::kVCvtF16F32Vop3},
           {rdna3_5::kVCvtF32F16Vop1, rdna3_5::kVCvtF32F16Vop3},
       }}},
  };
  enum class Form {
    Vop1,
    Vop3,
    Sdwa,
    SdwaScale,
    Dpp,
    SuppressedDpp,
    ScalarSource,
    HighRegister,
    HighHalf,
    E32HighHalf
  };
  struct Scenario {
    Form form;
    uint32_t mode = 0xf0;
    unsigned omod = 0;
    bool clamp = false, absolute = false, negate = false, alias = false;
  };
  const Scenario scenarios[] = {
      {Form::Vop1},
      {.form = Form::Vop3, .mode = 0xf0},
      {.form = Form::Vop3, .mode = 0xf4, .alias = true},
      {.form = Form::Vop3, .mode = 0xf8},
      {.form = Form::Vop3, .mode = 0xfc},
      {.form = Form::Vop3, .mode = 0, .omod = 1, .absolute = true, .negate = true},
      {.form = Form::Vop3, .mode = 0x2f0, .omod = 3, .clamp = true},
      {.form = Form::Vop3, .mode = 0xf0 | Wavefront::FP16_OVFL_BIT},
      {Form::Sdwa},
      {.form = Form::Sdwa, .mode = 0, .omod = 2},
      {.form = Form::Sdwa, .mode = 3, .omod = 3},
      {.form = Form::Sdwa, .mode = 0x200, .omod = 1},
      {.form = Form::Sdwa, .mode = 0x100, .clamp = true},
      {.form = Form::SdwaScale, .mode = 0},
      {.form = Form::SdwaScale, .mode = 0, .clamp = true, .alias = true},
      {Form::HighRegister},
      {Form::HighHalf},
      {Form::E32HighHalf},
      {.form = Form::Dpp, .alias = true},
      {Form::SuppressedDpp},
      {.form = Form::ScalarSource, .mode = 0xf1},
  };
  constexpr uint32_t special32[] = {0x7fc12345, 0x7f800000, 0xff800000, 0x80000000,
                                    1,          0x80000001, 0x477fff00, 0x47800000,
                                    0x38800000, 0x387fe000, 0x3f801000, 0xbf801000,
                                    0x33800000, 0x33000000, 0x7f800001, 0};
  constexpr uint16_t special16[] = {0x7e45, 0x7c00, 0xfc00, 0x8000, 1,      0x8001, 0x7bff, 0xfbff,
                                    0x0400, 0x03ff, 0x3e00, 0xbe00, 0x7fff, 0xffff, 0x7c01, 0};
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    const bool cdna =
        family.arch == rocjitsu::IsaFamily::Cdna1 || family.arch == rocjitsu::IsaFamily::Cdna2;
    for (int lanes : {32, 64}) {
      if (cdna && lanes != 64) {
        continue;
      }
      for (bool scalar : {false, true}) {
        util::set_force_scalar_for_testing(scalar);
        for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
          ComputeUnitCore::Config config{};
          config.arch = family.arch;
          config.arithmetic = {semantics};
          config.num_wf_slots = 1;
          config.sgprs_per_wf = 104;
          config.vgprs_per_wf = 144;
          config.lds_size_kb = 64;
          GpuMemory memory("conversion_memory");
          L2Cache cache("conversion_cache");
          cache.set_backing_memory(&memory);
          std::unique_ptr<ComputeUnitCore> cu;
          ASSERT_NO_FATAL_FAILURE(
              rocjitsu::test::initialize_compute_unit(cu, "conversion", config, &memory, &cache));
          auto *wave = cu->dispatch_wf(0, 0, 104, 144, lanes);
          ASSERT_NE(wave, nullptr);
          auto decoder = Decoder::create(family.arch);
          const auto base = wave->vgpr_alloc().base;
          std::minstd_rand random;
          const auto first_draw = random();
          const auto word = [&]() {
            return (uint32_t(first_draw >> 15) << 16) | uint32_t(random() >> 15);
          };
          std::array<uint32_t, 64> half_input{}, float_input{};
          for (int lane = 0; lane < lanes; ++lane) {
            half_input[lane] =
                lane < 16 ? uint32_t(special16[lane]) | (uint32_t(special16[15 - lane]) << 16)
                          : word();
            float_input[lane] = lane < 16 ? special32[lane] : word();
          }
          for (int operation = 0; operation < 6; ++operation) {
            for (const auto &sample : scenarios) {
              const bool vop3 = sample.form == Form::Vop3 || sample.form == Form::HighHalf;
              const bool sdwa_scale = sample.form == Form::SdwaScale;
              const bool sdwa = sample.form == Form::Sdwa || sdwa_scale;
              const bool dpp = sample.form == Form::Dpp || sample.form == Form::SuppressedDpp;
              const bool sgpr = sample.form == Form::ScalarSource;
              const bool high_register = sample.form == Form::HighRegister;
              const bool encoded_high_half = sample.form == Form::HighHalf;
              const bool e32_high_half = sample.form == Form::E32HighHalf;
              const bool true16 = family.arch == rocjitsu::IsaFamily::Rdna3_5;
              const bool high_half = e32_high_half || (encoded_high_half && (true16 || cdna));
              if (e32_high_half && !true16) {
                continue;
              }
              // Widening exercises every output modifier; narrowing retains its prescale witness.
              if (sdwa && sample.omod && operation != 5) {
                continue;
              }
              if (sdwa_scale && operation != 4) {
                continue;
              }
              if ((sdwa || high_register) && true16) {
                continue;
              }
              SCOPED_TRACE(::testing::Message()
                           << family.arch << ':' << lanes << ':' << scalar << ':' << int(semantics)
                           << ':' << operation << ':' << int(sample.form) << ':' << sample.mode);
              const uint32_t source_register = high_register ? 128 : 0;
              const uint32_t destination = high_register ? 130 : sample.alias ? 0 : 2;
              const uint64_t all = lanes == 64 ? ~0ULL : 0xffffffffULL;
              const uint64_t exec_mask = all & 0x800000038000ffffULL;
              wave->set_exec(dpp ? all : exec_mask);
              wave->set_vgpr_write_mask(dpp ? exec_mask : all);
              wave->set_mode_raw(sample.mode);
              wave->set_trapsts(0x100);
              cu->write_sgpr(wave->sgpr_alloc().base, 0x3fc03e00U);
              const auto &input = operation == 4 ? float_input : half_input;
              for (int lane = 0; lane < lanes; ++lane) {
                cu->write_vgpr(base + destination, lane, 0xdeadbeefU);
                cu->write_vgpr(base + source_register, lane,
                               sdwa_scale ? 0x477fff00U : input[lane]);
              }
              const uint32_t source =
                  sgpr ? 0U
                       : 256U + source_register + (e32_high_half && operation != 4 ? 128U : 0U);
              const uint32_t encoded_destination =
                  destination + (e32_high_half && operation != 5 ? 128U : 0U);
              // CDNA and true16 VOP3 honor OPSEL; GFX10 retains ignored-bit behavior.
              const uint32_t opsel = encoded_high_half ? (operation == 4   ? 8U
                                                          : operation == 5 ? 1U
                                                                           : 9U)
                                                       : 0;
              std::array<uint32_t, 2> words =
                  vop3
                      ? std::array<uint32_t, 2>{(cdna ? 0xd0000000U : 0xd4000000U) |
                                                    (uint32_t(family.opcodes[operation][1]) << 16) |
                                                    destination | (opsel << 11) |
                                                    (sample.absolute ? 1U << 8 : 0U) |
                                                    (sample.clamp ? 1U << 15 : 0U),
                                                source | (sample.omod << 27) |
                                                    (sample.negate ? 1U << 29 : 0U)}
                      : std::array<uint32_t, 2>{0x7e000000U | (encoded_destination << 17) |
                                                    (uint32_t(family.opcodes[operation][0]) << 9) |
                                                    source,
                                                0};
              if (sdwa) {
                // Select the source word/half; preserve WORD_0 while writing WORD_1.
                words[0] = (words[0] & ~0x1ffU) | 249U;
                words[1] = (operation == 4 ? 0x00061500U : 0x00041500U) | (sample.omod << 14) |
                           (sample.clamp ? 1U << 13 : 0U);
                if (sdwa_scale) {
                  words[1] |= 3U << 14;
                }
              } else if (dpp) {
                words[0] = (words[0] & ~0x1ffU) | 250U;
                words[1] = sample.form == Form::SuppressedDpp ? 0x0f00b100U : 0xff00b100U;
              }
              std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
              ASSERT_NE(instruction, nullptr);
              ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
              EXPECT_EQ(wave->trapsts(), 0x100U);
              for (int lane = 0; lane < lanes; ++lane) {
                const uint32_t old =
                    sample.alias ? (sdwa_scale ? 0x477fff00U : input[lane]) : 0xdeadbeefU;
                uint32_t expected = old;
                if ((exec_mask >> lane & 1) && sample.form != Form::SuppressedDpp) {
                  uint32_t raw = sgpr ? 0x3fc03e00U : input[dpp ? lane ^ 1 : lane];
                  if (operation != 4) {
                    raw = (raw >> (high_half ? 16 : 0)) & 0xffffU;
                  }
                  if (sdwa_scale) {
                    // Inherited SDWA producer order: 65535 / 2 rounds to half 32768.
                    expected = sample.clamp ? 0x3c00U : 0x7800U;
                  } else {
                    expected = reference(operation, raw, *wave, vop3 || sdwa ? sample.omod : 0,
                                         (vop3 || sdwa) && sample.clamp, vop3 && sample.absolute,
                                         vop3 && sample.negate, sdwa);
                  }
                  // Legacy SDWA scaling quiets a widened signaling NaN even with IEEE clear.
                  if (sdwa && operation == 5 && sample.omod && !wave->ieee_mode() &&
                      !(wave->fp_denorm_mode_f32() & 2) && (expected & 0x7fffffffU) > 0x7f800000U) {
                    expected |= 0x00400000U;
                  }
                  if (sdwa || (high_half && operation != 5)) {
                    expected = (old & 0xffffU) | (expected << 16);
                  } else if (operation != 5 && !cdna) {
                    expected = (old & 0xffff0000U) | (expected & 0xffffU);
                  }
                }
                EXPECT_EQ(cu->read_vgpr(base + destination, lane), expected) << "lane=" << lane;
              }
            }
          }
          wave->halt();
        }
      }
    }
  }
}

} // namespace
} // namespace rocjitsu::amdgpu
