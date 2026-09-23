// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mxfp_cvt_simd_test.cpp
/// @brief Bit-exact correctness and microbenchmarks for gfx1250 scaled MXFP
/// conversions. Each case decodes a real VOP3 instruction and executes it on a
/// wave32 through ComputeUnitCore.

#include "decode_test_util.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>

#if (defined(__x86_64__) || defined(__i386__)) && defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace {

using namespace rocjitsu;
using Clock = std::chrono::steady_clock;

constexpr uint32_t kWaveSize = 32;
constexpr uint32_t kSgprsPerWave = 128;
constexpr uint32_t kVgprsPerWave = 1024;
constexpr uint32_t kSourceBase = 0;
constexpr uint32_t kSeedReg = 20;
constexpr uint32_t kScaleReg = 21;
constexpr uint32_t kDestinationBase = 32;
constexpr uint32_t kMaxSourceRegs = 16;
constexpr uint32_t kMaxDestinationRegs = 16;
constexpr uint32_t kUnpackScaleByte = 2;
constexpr uint64_t kSparseExec = 0xD6B5A39Du;
constexpr uint64_t kFullExec = 0xFFFF'FFFFu;
constexpr uint64_t kRestrictedWriteMask = 0x0F0F'F0F0u;
constexpr int kWarmupIterations = 50;
constexpr int kTimedIterations = 4000;
constexpr std::size_t kBenchmarkSamples = 5;

enum class ConversionKind { PackRne, PackSr, Unpack };
enum class WideFormat { F32, F16, Bf16 };
enum class InputProfile { Finite, Exceptional };

struct ConversionCase {
  std::string_view mnemonic;
  uint16_t opcode;
  uint32_t count;
  uint32_t low_bits;
  ConversionKind kind;
  WideFormat wide_format;
};

constexpr std::array<ConversionCase, 45> kConversionCases = {{
    {"v_cvt_scalef32_pk8_fp4_f32", cdna5::kVCvtScalef32Pk8Fp4F32Vop3, 8, 4, ConversionKind::PackRne,
     WideFormat::F32},
    {"v_cvt_scalef32_pk8_fp4_f16", cdna5::kVCvtScalef32Pk8Fp4F16Vop3, 8, 4, ConversionKind::PackRne,
     WideFormat::F16},
    {"v_cvt_scalef32_pk8_fp4_bf16", cdna5::kVCvtScalef32Pk8Fp4Bf16Vop3, 8, 4,
     ConversionKind::PackRne, WideFormat::Bf16},
    {"v_cvt_scalef32_sr_pk8_fp4_f32", cdna5::kVCvtScalef32SrPk8Fp4F32Vop3, 8, 4,
     ConversionKind::PackSr, WideFormat::F32},
    {"v_cvt_scalef32_sr_pk8_fp4_f16", cdna5::kVCvtScalef32SrPk8Fp4F16Vop3, 8, 4,
     ConversionKind::PackSr, WideFormat::F16},
    {"v_cvt_scalef32_sr_pk8_fp4_bf16", cdna5::kVCvtScalef32SrPk8Fp4Bf16Vop3, 8, 4,
     ConversionKind::PackSr, WideFormat::Bf16},
    {"v_cvt_scale_pk8_f32_fp4", cdna5::kVCvtScalePk8F32Fp4Vop3, 8, 4, ConversionKind::Unpack,
     WideFormat::F32},
    {"v_cvt_scale_pk8_f16_fp4", cdna5::kVCvtScalePk8F16Fp4Vop3, 8, 4, ConversionKind::Unpack,
     WideFormat::F16},
    {"v_cvt_scale_pk8_bf16_fp4", cdna5::kVCvtScalePk8Bf16Fp4Vop3, 8, 4, ConversionKind::Unpack,
     WideFormat::Bf16},

    {"v_cvt_scalef32_pk16_fp6_f32", cdna5::kVCvtScalef32Pk16Fp6F32Vop3, 16, 6,
     ConversionKind::PackRne, WideFormat::F32},
    {"v_cvt_scalef32_pk16_fp6_f16", cdna5::kVCvtScalef32Pk16Fp6F16Vop3, 16, 6,
     ConversionKind::PackRne, WideFormat::F16},
    {"v_cvt_scalef32_pk16_fp6_bf16", cdna5::kVCvtScalef32Pk16Fp6Bf16Vop3, 16, 6,
     ConversionKind::PackRne, WideFormat::Bf16},
    {"v_cvt_scalef32_sr_pk16_fp6_f32", cdna5::kVCvtScalef32SrPk16Fp6F32Vop3, 16, 6,
     ConversionKind::PackSr, WideFormat::F32},
    {"v_cvt_scalef32_sr_pk16_fp6_f16", cdna5::kVCvtScalef32SrPk16Fp6F16Vop3, 16, 6,
     ConversionKind::PackSr, WideFormat::F16},
    {"v_cvt_scalef32_sr_pk16_fp6_bf16", cdna5::kVCvtScalef32SrPk16Fp6Bf16Vop3, 16, 6,
     ConversionKind::PackSr, WideFormat::Bf16},
    {"v_cvt_scale_pk16_f32_fp6", cdna5::kVCvtScalePk16F32Fp6Vop3, 16, 6, ConversionKind::Unpack,
     WideFormat::F32},
    {"v_cvt_scale_pk16_f16_fp6", cdna5::kVCvtScalePk16F16Fp6Vop3, 16, 6, ConversionKind::Unpack,
     WideFormat::F16},
    {"v_cvt_scale_pk16_bf16_fp6", cdna5::kVCvtScalePk16Bf16Fp6Vop3, 16, 6, ConversionKind::Unpack,
     WideFormat::Bf16},

    {"v_cvt_scalef32_pk16_bf6_f32", cdna5::kVCvtScalef32Pk16Bf6F32Vop3, 16, 6,
     ConversionKind::PackRne, WideFormat::F32},
    {"v_cvt_scalef32_pk16_bf6_f16", cdna5::kVCvtScalef32Pk16Bf6F16Vop3, 16, 6,
     ConversionKind::PackRne, WideFormat::F16},
    {"v_cvt_scalef32_pk16_bf6_bf16", cdna5::kVCvtScalef32Pk16Bf6Bf16Vop3, 16, 6,
     ConversionKind::PackRne, WideFormat::Bf16},
    {"v_cvt_scalef32_sr_pk16_bf6_f32", cdna5::kVCvtScalef32SrPk16Bf6F32Vop3, 16, 6,
     ConversionKind::PackSr, WideFormat::F32},
    {"v_cvt_scalef32_sr_pk16_bf6_f16", cdna5::kVCvtScalef32SrPk16Bf6F16Vop3, 16, 6,
     ConversionKind::PackSr, WideFormat::F16},
    {"v_cvt_scalef32_sr_pk16_bf6_bf16", cdna5::kVCvtScalef32SrPk16Bf6Bf16Vop3, 16, 6,
     ConversionKind::PackSr, WideFormat::Bf16},
    {"v_cvt_scale_pk16_f32_bf6", cdna5::kVCvtScalePk16F32Bf6Vop3, 16, 6, ConversionKind::Unpack,
     WideFormat::F32},
    {"v_cvt_scale_pk16_f16_bf6", cdna5::kVCvtScalePk16F16Bf6Vop3, 16, 6, ConversionKind::Unpack,
     WideFormat::F16},
    {"v_cvt_scale_pk16_bf16_bf6", cdna5::kVCvtScalePk16Bf16Bf6Vop3, 16, 6, ConversionKind::Unpack,
     WideFormat::Bf16},

    {"v_cvt_scalef32_pk8_fp8_f32", cdna5::kVCvtScalef32Pk8Fp8F32Vop3, 8, 8, ConversionKind::PackRne,
     WideFormat::F32},
    {"v_cvt_scalef32_pk8_fp8_f16", cdna5::kVCvtScalef32Pk8Fp8F16Vop3, 8, 8, ConversionKind::PackRne,
     WideFormat::F16},
    {"v_cvt_scalef32_pk8_fp8_bf16", cdna5::kVCvtScalef32Pk8Fp8Bf16Vop3, 8, 8,
     ConversionKind::PackRne, WideFormat::Bf16},
    {"v_cvt_scalef32_sr_pk8_fp8_f32", cdna5::kVCvtScalef32SrPk8Fp8F32Vop3, 8, 8,
     ConversionKind::PackSr, WideFormat::F32},
    {"v_cvt_scalef32_sr_pk8_fp8_f16", cdna5::kVCvtScalef32SrPk8Fp8F16Vop3, 8, 8,
     ConversionKind::PackSr, WideFormat::F16},
    {"v_cvt_scalef32_sr_pk8_fp8_bf16", cdna5::kVCvtScalef32SrPk8Fp8Bf16Vop3, 8, 8,
     ConversionKind::PackSr, WideFormat::Bf16},
    {"v_cvt_scale_pk8_f32_fp8", cdna5::kVCvtScalePk8F32Fp8Vop3, 8, 8, ConversionKind::Unpack,
     WideFormat::F32},
    {"v_cvt_scale_pk8_f16_fp8", cdna5::kVCvtScalePk8F16Fp8Vop3, 8, 8, ConversionKind::Unpack,
     WideFormat::F16},
    {"v_cvt_scale_pk8_bf16_fp8", cdna5::kVCvtScalePk8Bf16Fp8Vop3, 8, 8, ConversionKind::Unpack,
     WideFormat::Bf16},

    {"v_cvt_scalef32_pk8_bf8_f32", cdna5::kVCvtScalef32Pk8Bf8F32Vop3, 8, 8, ConversionKind::PackRne,
     WideFormat::F32},
    {"v_cvt_scalef32_pk8_bf8_f16", cdna5::kVCvtScalef32Pk8Bf8F16Vop3, 8, 8, ConversionKind::PackRne,
     WideFormat::F16},
    {"v_cvt_scalef32_pk8_bf8_bf16", cdna5::kVCvtScalef32Pk8Bf8Bf16Vop3, 8, 8,
     ConversionKind::PackRne, WideFormat::Bf16},
    {"v_cvt_scalef32_sr_pk8_bf8_f32", cdna5::kVCvtScalef32SrPk8Bf8F32Vop3, 8, 8,
     ConversionKind::PackSr, WideFormat::F32},
    {"v_cvt_scalef32_sr_pk8_bf8_f16", cdna5::kVCvtScalef32SrPk8Bf8F16Vop3, 8, 8,
     ConversionKind::PackSr, WideFormat::F16},
    {"v_cvt_scalef32_sr_pk8_bf8_bf16", cdna5::kVCvtScalef32SrPk8Bf8Bf16Vop3, 8, 8,
     ConversionKind::PackSr, WideFormat::Bf16},
    {"v_cvt_scale_pk8_f32_bf8", cdna5::kVCvtScalePk8F32Bf8Vop3, 8, 8, ConversionKind::Unpack,
     WideFormat::F32},
    {"v_cvt_scale_pk8_f16_bf8", cdna5::kVCvtScalePk8F16Bf8Vop3, 8, 8, ConversionKind::Unpack,
     WideFormat::F16},
    {"v_cvt_scale_pk8_bf16_bf8", cdna5::kVCvtScalePk8Bf16Bf8Vop3, 8, 8, ConversionKind::Unpack,
     WideFormat::Bf16},
}};

constexpr uint16_t vgpr_source(uint32_t relative_reg) {
  return static_cast<uint16_t>(cdna5::OPR_SRC_VGPR_MIN + relative_reg);
}

constexpr uint32_t packed_word_count(const ConversionCase &test_case) {
  return (test_case.count * test_case.low_bits + 31u) / 32u;
}

constexpr uint32_t wide_word_count(const ConversionCase &test_case) {
  return test_case.wide_format == WideFormat::F32 ? test_case.count : test_case.count / 2u;
}

constexpr uint32_t source_reg_count(const ConversionCase &test_case) {
  return test_case.kind == ConversionKind::Unpack ? packed_word_count(test_case)
                                                  : wide_word_count(test_case);
}

std::array<uint32_t, 2> build_conversion(const ConversionCase &test_case,
                                         uint32_t destination_base = kDestinationBase,
                                         uint32_t source_base = kSourceBase,
                                         uint16_t seed_operand = vgpr_source(kSeedReg),
                                         uint16_t scale_operand = vgpr_source(kScaleReg),
                                         uint32_t scale_byte = kUnpackScaleByte) {
  cdna5::Vop3BuilderFields fields{};
  fields.vdst = destination_base;
  fields.src0 = vgpr_source(source_base);
  if (test_case.kind == ConversionKind::PackSr) {
    fields.src1 = seed_operand;
    fields.src2 = scale_operand;
  } else {
    fields.src1 = scale_operand;
  }
  if (test_case.kind == ConversionKind::Unpack)
    fields.opsel = scale_byte;
  return cdna5::build_vop3(test_case.opcode, fields);
}

class ForceScalarGuard {
public:
  explicit ForceScalarGuard(bool force_scalar) : saved_(util::force_scalar()) {
    util::set_force_scalar_for_testing(force_scalar);
  }
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(saved_); }

private:
  bool saved_;
};

struct ConversionFixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;
  uint32_t vgpr_base = 0;

  explicit ConversionFixture(std::string name) : gpu_mem(name + "_mem"), l2(name + "_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = kSgprsPerWave;
    cfg.vgprs_per_wf = kVgprsPerWave;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create(name, cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    if (cu)
      wf = cu->dispatch_wf(0, 0, kSgprsPerWave, kVgprsPerWave, kWaveSize);
    if (wf)
      vgpr_base = wf->vgpr_alloc().base;
  }

  ~ConversionFixture() {
    if (wf && !wf->is_halted())
      wf->halt();
  }

  void seed(const ConversionCase &test_case, uint64_t exec,
            InputProfile profile = InputProfile::Finite) {
    static constexpr std::array<float, 16> normalized_values = {
        -7.5f, -6.0f, -4.5f, -3.0f, -2.5f, -1.5f, -0.75f, -0.25f,
        -0.0f, 0.0f,  0.25f, 0.75f, 1.25f, 1.75f, 4.5f,   6.0f,
    };
    static constexpr std::array<uint32_t, 8> exceptional_bits = {
        0x0000'0000u, 0x8000'0000u, 0x7F80'0000u, 0xFF80'0000u,
        0x7FC1'2345u, 0x3F80'0000u, 0xBF80'0000u, 0x0080'0000u,
    };

    wf->set_exec(exec);
    wf->set_mode_raw(0);
    wf->set_vgpr_msb_mode(0);
    wf->set_vgpr_write_mask(~uint64_t{0});

    for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
      for (uint32_t reg = 0; reg < kMaxSourceRegs; ++reg)
        cu->write_vgpr(vgpr_base + kSourceBase + reg, lane, 0u);
      for (uint32_t reg = 0; reg < kMaxDestinationRegs; ++reg) {
        const uint32_t sentinel = 0xA5A5'0000u ^ (reg * 0x101u) ^ lane;
        cu->write_vgpr(vgpr_base + kDestinationBase + reg, lane, sentinel);
      }

      const int scale_exp = static_cast<int>(lane % 5u) - 2;
      const float lane_scale = std::ldexp(1.0f, scale_exp);
      const uint32_t lane_seed = 0x243F'6A88u ^ ((lane + 1u) * 0x9E37'79B9u);
      cu->write_vgpr(vgpr_base + kSeedReg, lane, lane_seed);

      if (test_case.kind == ConversionKind::Unpack) {
        const uint32_t selected_scale = 125u + lane % 5u;
        const uint32_t scale_word = 0x7Du | (0x7Fu << 8) | (selected_scale << 16) | (0x80u << 24);
        cu->write_vgpr(vgpr_base + kScaleReg, lane, scale_word);
        for (uint32_t word = 0; word < source_reg_count(test_case); ++word) {
          uint32_t packed = 0x9E37'79B9u * (lane + 1u) ^ 0x85EB'CA6Bu * (word + 3u);
          packed ^= packed >> 13;
          packed *= 0xC2B2'AE35u;
          if (profile == InputProfile::Finite && test_case.low_bits == 8u) {
            // E4M3 reserves magnitude 0x7f for NaN; E5M2 reserves
            // 0x7c..0x7f for infinity and NaNs. Keep benchmark and finite
            // correctness profiles on the common SIMD path for both formats.
            uint32_t finite_packed = 0u;
            for (uint32_t byte = 0; byte < 4u; ++byte) {
              const uint32_t shift = byte * 8u;
              const uint32_t code = (packed >> shift) & 0xffu;
              const uint32_t magnitude = std::min(code & 0x7fu, 0x7bu);
              finite_packed |= ((code & 0x80u) | magnitude) << shift;
            }
            packed = finite_packed;
          }
          cu->write_vgpr(vgpr_base + kSourceBase + word, lane, packed);
        }
      } else {
        const uint32_t scale_bits =
            profile == InputProfile::Exceptional
                ? exceptional_bits[(lane * 3u + 1u) % exceptional_bits.size()]
                : std::bit_cast<uint32_t>(lane_scale);
        cu->write_vgpr(vgpr_base + kScaleReg, lane, scale_bits);
        const auto source_value = [&](uint32_t element) {
          if (profile == InputProfile::Exceptional) {
            const std::size_t index = (lane * 5u + element * 3u) % exceptional_bits.size();
            return std::bit_cast<float>(exceptional_bits[index]);
          }
          const std::size_t index = (lane * 5u + element * 3u) % normalized_values.size();
          return normalized_values[index] * lane_scale;
        };
        if (test_case.wide_format == WideFormat::F32) {
          for (uint32_t element = 0; element < test_case.count; ++element) {
            cu->write_vgpr(vgpr_base + kSourceBase + element, lane,
                           std::bit_cast<uint32_t>(source_value(element)));
          }
        } else {
          for (uint32_t word = 0; word < wide_word_count(test_case); ++word) {
            const uint32_t lo = test_case.wide_format == WideFormat::F16
                                    ? util::f32_to_f16(source_value(word * 2u))
                                    : util::f32_to_bf16_rne(source_value(word * 2u));
            const uint32_t hi = test_case.wide_format == WideFormat::F16
                                    ? util::f32_to_f16(source_value(word * 2u + 1u))
                                    : util::f32_to_bf16_rne(source_value(word * 2u + 1u));
            cu->write_vgpr(vgpr_base + kSourceBase + word, lane, lo | (hi << 16));
          }
        }
      }
    }
  }

  [[nodiscard]] std::array<uint32_t, kMaxDestinationRegs * kWaveSize>
  snapshot_vgpr_window(uint32_t base = kDestinationBase) const {
    std::array<uint32_t, kMaxDestinationRegs * kWaveSize> snapshot{};
    for (uint32_t reg = 0; reg < kMaxDestinationRegs; ++reg)
      for (uint32_t lane = 0; lane < kWaveSize; ++lane)
        snapshot[reg * kWaveSize + lane] = cu->read_vgpr(vgpr_base + base + reg, lane);
    return snapshot;
  }

  void fill_packed_source(const ConversionCase &test_case, uint32_t source_base, uint32_t code) {
    ASSERT_EQ(test_case.kind, ConversionKind::Unpack);
    const uint32_t code_mask = (1u << test_case.low_bits) - 1u;
    for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
      std::array<uint32_t, 4> words{};
      for (uint32_t index = 0; index < test_case.count; ++index) {
        const uint32_t bit = index * test_case.low_bits;
        const uint32_t word = bit / 32u;
        const uint32_t shift = bit & 31u;
        words[word] |= (code & code_mask) << shift;
        if (shift + test_case.low_bits > 32u)
          words[word + 1u] |= (code & code_mask) >> (32u - shift);
      }
      for (uint32_t word = 0; word < source_reg_count(test_case); ++word)
        cu->write_vgpr(vgpr_base + source_base + word, lane, words[word]);
    }
  }

  template <typename ValueFn>
  void fill_wide_source(const ConversionCase &test_case, uint32_t source_base, ValueFn value_for) {
    ASSERT_NE(test_case.kind, ConversionKind::Unpack);
    for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
      if (test_case.wide_format == WideFormat::F32) {
        for (uint32_t element = 0; element < test_case.count; ++element) {
          const float value = value_for(lane, element);
          cu->write_vgpr(vgpr_base + source_base + element, lane, std::bit_cast<uint32_t>(value));
        }
      } else {
        for (uint32_t word = 0; word < wide_word_count(test_case); ++word) {
          const float lo_value = value_for(lane, word * 2u);
          const float hi_value = value_for(lane, word * 2u + 1u);
          const uint32_t lo = test_case.wide_format == WideFormat::F16
                                  ? util::f32_to_f16(lo_value)
                                  : util::f32_to_bf16_rne(lo_value);
          const uint32_t hi = test_case.wide_format == WideFormat::F16
                                  ? util::f32_to_f16(hi_value)
                                  : util::f32_to_bf16_rne(hi_value);
          cu->write_vgpr(vgpr_base + source_base + word, lane, lo | (hi << 16));
        }
      }
    }
  }

  [[nodiscard]] std::array<uint32_t, kWaveSize> snapshot_seed() const {
    std::array<uint32_t, kWaveSize> snapshot{};
    for (uint32_t lane = 0; lane < kWaveSize; ++lane)
      snapshot[lane] = cu->read_vgpr(vgpr_base + kSeedReg, lane);
    return snapshot;
  }
};

double time_sample(ConversionFixture &fixture, const ConversionCase &test_case,
                   Instruction &instruction, bool force_scalar) {
  fixture.seed(test_case, kFullExec);
  ForceScalarGuard guard(force_scalar);
  bool succeeded = true;
  for (int iteration = 0; iteration < kWarmupIterations; ++iteration)
    succeeded &= fixture.cu->execute_instruction(&instruction, *fixture.wf).succeeded();
  const auto begin = Clock::now();
  for (int iteration = 0; iteration < kTimedIterations; ++iteration)
    succeeded &= fixture.cu->execute_instruction(&instruction, *fixture.wf).succeeded();
  const auto end = Clock::now();
  EXPECT_TRUE(succeeded);
  const double total_us = std::chrono::duration<double, std::micro>(end - begin).count();
  return total_us / kTimedIterations;
}

template <std::size_t N> double median(std::array<double, N> samples) {
  static_assert(N % 2u == 1u);
  std::sort(samples.begin(), samples.end());
  return samples[N / 2u];
}

const ConversionCase *find_conversion(std::string_view mnemonic) {
  const auto it =
      std::find_if(kConversionCases.begin(), kConversionCases.end(),
                   [&](const ConversionCase &test_case) { return test_case.mnemonic == mnemonic; });
  return it == kConversionCases.end() ? nullptr : &*it;
}

class HeldFloatingEnvironment {
public:
  HeldFloatingEnvironment() : valid_(std::feholdexcept(&saved_) == 0) {}
  ~HeldFloatingEnvironment() {
    if (valid_)
      std::fesetenv(&saved_);
  }

  [[nodiscard]] bool valid() const { return valid_; }

private:
  std::fenv_t saved_{};
  bool valid_ = false;
};

void seed_inactive_lane_fp_hazard(ConversionFixture &fixture, const ConversionCase &test_case) {
  fixture.seed(test_case, 1u);
  if (test_case.kind == ConversionKind::Unpack) {
    // Lane 0 executes FP4 1.0 * E8M0(0x7f). Lane 1 is inactive but would
    // overflow by evaluating FP4 6.0 * E8M0(0xfe) in an unmasked host lane.
    fixture.fill_packed_source(test_case, kSourceBase, 0x2u);
    for (uint32_t lane = 0; lane < kWaveSize; ++lane)
      fixture.cu->write_vgpr(fixture.vgpr_base + kScaleReg, lane, 0x007F'0000u);
    fixture.cu->write_vgpr(fixture.vgpr_base + kSourceBase, 1, 0x7777'7777u);
    fixture.cu->write_vgpr(fixture.vgpr_base + kScaleReg, 1, 0x00FE'0000u);
    return;
  }

  // Lane 0 executes 1.0 / 1.0. Lane 1 is inactive but would divide by zero
  // if the SIMD host operation evaluated its unsanitized operands.
  fixture.fill_wide_source(test_case, kSourceBase, [](uint32_t, uint32_t) { return 1.0f; });
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    fixture.cu->write_vgpr(fixture.vgpr_base + kScaleReg, lane, std::bit_cast<uint32_t>(1.0f));
  fixture.cu->write_vgpr(fixture.vgpr_base + kScaleReg, 1, 0u);
}

#if GTEST_HAS_DEATH_TEST && defined(__linux__) && defined(__SSE__) &&                              \
    (defined(__x86_64__) || defined(__i386__))
int run_inactive_lane_unmasked_exception_test() {
  std::fenv_t environment{};
  if (std::feholdexcept(&environment) != 0)
    return 1;
  if constexpr (!util::has_stdx_simd || util::native_width_v<uint32_t> < 2u) {
    return 0;
  } else {
    ConversionFixture fixture("gfx1250_mxfp_cvt_inactive_unmasked");
    if (!fixture.cu || !fixture.decoder || !fixture.wf)
      return 2;

    for (std::string_view mnemonic : {"v_cvt_scalef32_pk8_fp4_f32", "v_cvt_scale_pk8_f32_fp4"}) {
      const ConversionCase *test_case = find_conversion(mnemonic);
      if (test_case == nullptr)
        return 3;
      const auto words = build_conversion(*test_case);
      std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
      if (!instruction)
        return 4;

      seed_inactive_lane_fp_hazard(fixture, *test_case);
      ForceScalarGuard enable_simd(false);
      if (std::feclearexcept(FE_ALL_EXCEPT) != 0)
        return 5;
      uint32_t mxcsr = _mm_getcsr();
      mxcsr &= ~static_cast<uint32_t>(_MM_EXCEPT_MASK);
      mxcsr &= ~static_cast<uint32_t>(_MM_MASK_DIV_ZERO | _MM_MASK_OVERFLOW);
      _mm_setcsr(mxcsr);
      const bool succeeded =
          fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded();
      _mm_setcsr(_mm_getcsr() | _MM_MASK_MASK);
      if (!succeeded)
        return 6;
    }
    return 0;
  }
}
#endif

} // namespace

TEST(Gfx1250MxfpCvtSimdCorrectness, AllWideFormatsAndExceptionalPackInputsAreBitExact) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    ConversionFixture fixture("gfx1250_mxfp_cvt_correctness");
    ASSERT_NE(fixture.cu, nullptr);
    ASSERT_NE(fixture.decoder, nullptr);
    ASSERT_NE(fixture.wf, nullptr);
    ASSERT_EQ(fixture.wf->wf_size(), kWaveSize);

    for (const ConversionCase &test_case : kConversionCases) {
      SCOPED_TRACE(test_case.mnemonic.data());
      const auto words = build_conversion(test_case);
      std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
      ASSERT_NE(instruction, nullptr);
      ASSERT_EQ(std::string_view(instruction->mnemonic()), test_case.mnemonic);

      const auto compare_modes = [&](InputProfile profile, uint64_t exec, uint64_t write_mask) {
        SCOPED_TRACE(profile == InputProfile::Finite ? "finite" : "zero_inf_nan");
        fixture.seed(test_case, exec, profile);
        fixture.wf->set_vgpr_write_mask(write_mask);
        const auto scalar_seed_before = fixture.snapshot_seed();
        {
          ForceScalarGuard force_scalar(true);
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
        }
        const auto scalar_result = fixture.snapshot_vgpr_window();
        const auto scalar_seed_after = fixture.snapshot_seed();

        fixture.seed(test_case, exec, profile);
        fixture.wf->set_vgpr_write_mask(write_mask);
        const auto simd_seed_before = fixture.snapshot_seed();
        {
          ForceScalarGuard enable_simd(false);
          ASSERT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
        }
        const auto simd_result = fixture.snapshot_vgpr_window();
        const auto simd_seed_after = fixture.snapshot_seed();

        EXPECT_EQ(simd_result, scalar_result);
        if (test_case.kind == ConversionKind::PackSr) {
          EXPECT_EQ(scalar_seed_after, scalar_seed_before);
          EXPECT_EQ(simd_seed_after, simd_seed_before);
          EXPECT_EQ(simd_seed_before, scalar_seed_before);
        }
      };

      compare_modes(InputProfile::Finite, kSparseExec, ~uint64_t{0});
      if (test_case.kind != ConversionKind::Unpack) {
        compare_modes(InputProfile::Exceptional, kFullExec, ~uint64_t{0});
        compare_modes(InputProfile::Finite, kFullExec, kRestrictedWriteMask);
      }
    }
  }
}

TEST(Gfx1250MxfpCvtSimdCorrectness, PackNaNAndInvalidDivisionPreservesScalarBaseline) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    HeldFloatingEnvironment environment;
    ASSERT_TRUE(environment.valid());
    struct DivisionCase {
      enum class Kind { QuietNaN, SignalingNaN, InvalidDivision };

      std::string_view name;
      uint32_t value_bits;
      uint32_t scale_bits;
      Kind kind;
    };
    constexpr std::array<DivisionCase, 16> division_cases = {{
        {"source_negative_qnan", 0xFFC1'2345u, 0x3F80'0000u, DivisionCase::Kind::QuietNaN},
        {"scale_negative_qnan", 0x3F80'0000u, 0xFFC1'2345u, DivisionCase::Kind::QuietNaN},
        {"source_positive_qnan_precedes_negative", 0x7FC1'2345u, 0xFFC1'2345u,
         DivisionCase::Kind::QuietNaN},
        {"source_negative_qnan_precedes_positive", 0xFFC1'2345u, 0x7FC1'2345u,
         DivisionCase::Kind::QuietNaN},
        {"source_positive_snan", 0x7F80'0001u, 0x3F80'0000u, DivisionCase::Kind::SignalingNaN},
        {"source_negative_snan", 0xFF80'0001u, 0x3F80'0000u, DivisionCase::Kind::SignalingNaN},
        {"scale_positive_snan", 0x3F80'0000u, 0x7F80'0001u, DivisionCase::Kind::SignalingNaN},
        {"scale_negative_snan", 0x3F80'0000u, 0xFF80'0001u, DivisionCase::Kind::SignalingNaN},
        {"positive_zero_over_positive_zero", 0x0000'0000u, 0x0000'0000u,
         DivisionCase::Kind::InvalidDivision},
        {"negative_zero_over_positive_zero", 0x8000'0000u, 0x0000'0000u,
         DivisionCase::Kind::InvalidDivision},
        {"positive_zero_over_negative_zero", 0x0000'0000u, 0x8000'0000u,
         DivisionCase::Kind::InvalidDivision},
        {"negative_zero_over_negative_zero", 0x8000'0000u, 0x8000'0000u,
         DivisionCase::Kind::InvalidDivision},
        {"positive_inf_over_positive_inf", 0x7F80'0000u, 0x7F80'0000u,
         DivisionCase::Kind::InvalidDivision},
        {"negative_inf_over_positive_inf", 0xFF80'0000u, 0x7F80'0000u,
         DivisionCase::Kind::InvalidDivision},
        {"positive_inf_over_negative_inf", 0x7F80'0000u, 0xFF80'0000u,
         DivisionCase::Kind::InvalidDivision},
        {"negative_inf_over_negative_inf", 0xFF80'0000u, 0xFF80'0000u,
         DivisionCase::Kind::InvalidDivision},
    }};

    ConversionFixture fixture("gfx1250_mxfp_cvt_pack_nan_policy");
    ASSERT_NE(fixture.cu, nullptr);
    ASSERT_NE(fixture.decoder, nullptr);
    ASSERT_NE(fixture.wf, nullptr);

    for (const ConversionCase &test_case : kConversionCases) {
      if (test_case.kind == ConversionKind::Unpack)
        continue;
      const auto words = build_conversion(test_case);
      std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
      ASSERT_NE(instruction, nullptr);

      for (const DivisionCase &division_case : division_cases) {
        SCOPED_TRACE(test_case.mnemonic.data());
        SCOPED_TRACE(division_case.name.data());
        struct RunResult {
          int clear_result;
          int invalid_exceptions;
          std::array<uint32_t, kMaxDestinationRegs * kWaveSize> destination;
        };
        const auto run = [&](bool force_scalar) {
          fixture.seed(test_case, kFullExec);
          const float value = std::bit_cast<float>(division_case.value_bits);
          uint32_t packed_value = division_case.value_bits;
          if (test_case.wide_format == WideFormat::F16) {
            const uint32_t half = util::f32_to_f16(value);
            packed_value = half | (half << 16);
          } else if (test_case.wide_format == WideFormat::Bf16) {
            const uint32_t half = util::f32_to_bf16_rne(value);
            packed_value = half | (half << 16);
          }
          for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
            fixture.cu->write_vgpr(fixture.vgpr_base + kScaleReg, lane, division_case.scale_bits);
            for (uint32_t word = 0; word < wide_word_count(test_case); ++word)
              fixture.cu->write_vgpr(fixture.vgpr_base + kSourceBase + word, lane, packed_value);
          }
          ForceScalarGuard guard(force_scalar);
          const int clear_result = std::feclearexcept(FE_ALL_EXCEPT);
          EXPECT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
          return RunResult{clear_result, std::fetestexcept(FE_INVALID),
                           fixture.snapshot_vgpr_window()};
        };

        const auto scalar_result = run(true);
        const auto simd_result = run(false);
        ASSERT_EQ(scalar_result.clear_result, 0);
        ASSERT_EQ(simd_result.clear_result, 0);
        // A quiet-NaN comparison inside an encoder may or may not raise
        // FE_INVALID after inlining and optimization. Its result bits remain
        // part of the scalar contract; its host exception side effect does not.
        if (division_case.kind != DivisionCase::Kind::QuietNaN) {
          EXPECT_EQ(simd_result.invalid_exceptions, scalar_result.invalid_exceptions);
        }
        EXPECT_EQ(simd_result.destination, scalar_result.destination);
        if (division_case.kind == DivisionCase::Kind::InvalidDivision) {
#if defined(__x86_64__) || defined(__i386__)
          // Independent regression for the original direct host division on
          // the configured x86 build: both 0/0 and Inf/Inf produce the x86
          // indefinite negative NaN and therefore pack as 0xff in FP8/BF8.
          // The C++ abstract machine does not specify this NaN sign on other
          // hosts, where the untouched generated scalar result stays the oracle.
          EXPECT_NE(scalar_result.invalid_exceptions, 0);
          EXPECT_NE(simd_result.invalid_exceptions, 0);
          const uint32_t expected_code = test_case.low_bits == 8u ? 0xFFu : 0u;
          const uint32_t expected_word = 0x0101'0101u * expected_code;
          for (uint32_t word = 0; word < packed_word_count(test_case); ++word) {
            EXPECT_EQ(scalar_result.destination[word * kWaveSize], expected_word);
            EXPECT_EQ(simd_result.destination[word * kWaveSize], expected_word);
          }
#endif
        }
      }
    }
  }
}

TEST(Gfx1250MxfpCvtSimdCorrectness, NonPowerOfTwoPackScalesAreBitExact) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    // The regular finite profile deliberately cancels powers of two exactly.
    // These values force ordinary rounded division around low-format
    // subnormal, midpoint, maximum-finite, and overflow boundaries.
    static constexpr std::array<float, 18> quotients = {
        -61440.0f, -464.0f, -29.0f, -7.75f, -6.25f, -0.75f, -0.25f, -0.0625f, -0.03125f,
        0.03125f,  0.0625f, 0.25f,  0.75f,  1.25f,  6.25f,  7.75f,  29.0f,    464.0f,
    };
    ConversionFixture fixture("gfx1250_mxfp_cvt_nonbinary_scale");
    ASSERT_NE(fixture.cu, nullptr);
    ASSERT_NE(fixture.decoder, nullptr);
    ASSERT_NE(fixture.wf, nullptr);

    for (const ConversionCase &test_case : kConversionCases) {
      if (test_case.kind == ConversionKind::Unpack)
        continue;
      SCOPED_TRACE(test_case.mnemonic.data());
      const auto words = build_conversion(test_case);
      std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
      ASSERT_NE(instruction, nullptr);

      const auto run = [&](bool force_scalar) {
        fixture.seed(test_case, kFullExec);
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          const float scale = (lane & 1u) == 0u ? 3.0f : 0.1f;
          fixture.cu->write_vgpr(fixture.vgpr_base + kScaleReg, lane,
                                 std::bit_cast<uint32_t>(scale));
        }
        fixture.fill_wide_source(test_case, kSourceBase, [&](uint32_t lane, uint32_t element) {
          const float scale = (lane & 1u) == 0u ? 3.0f : 0.1f;
          float quotient = quotients[(lane * 5u + element * 7u) % quotients.size()];
          switch ((lane + element) % 3u) {
          case 0:
            quotient = std::nextafter(quotient, 0.0f);
            break;
          case 2:
            quotient = std::nextafter(quotient, quotient < 0.0f ? -1.0e30f : 1.0e30f);
            break;
          default:
            break;
          }
          return quotient * scale;
        });
        ForceScalarGuard guard(force_scalar);
        EXPECT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
        return fixture.snapshot_vgpr_window();
      };

      EXPECT_EQ(run(false), run(true));
    }
  }
}

TEST(Gfx1250MxfpCvtSimdCorrectness, Fp8AndBf8PackOverflowModeIsBitExact) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    ConversionFixture fixture("gfx1250_mxfp_cvt_pack_overflow_mode");
    ASSERT_NE(fixture.cu, nullptr);
    ASSERT_NE(fixture.decoder, nullptr);
    ASSERT_NE(fixture.wf, nullptr);

    for (const ConversionCase &test_case : kConversionCases) {
      if (test_case.kind == ConversionKind::Unpack || test_case.low_bits != 8u)
        continue;
      SCOPED_TRACE(test_case.mnemonic.data());
      const bool bf8 = test_case.mnemonic.find("_bf8_") != std::string_view::npos;
      const auto words = build_conversion(test_case);
      std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
      ASSERT_NE(instruction, nullptr);

      const auto run = [&](bool force_scalar, bool fp16_ovfl) {
        fixture.seed(test_case, kFullExec);
        fixture.wf->set_mode_raw(fp16_ovfl ? amdgpu::Wavefront::FP16_OVFL_BIT : 0u);
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          fixture.cu->write_vgpr(fixture.vgpr_base + kScaleReg, lane,
                                 std::bit_cast<uint32_t>(1.0f));
          fixture.cu->write_vgpr(fixture.vgpr_base + kSeedReg, lane,
                                 (lane & 1u) == 0u ? 0u : 0xFFFF'FFFFu);
        }
        fixture.fill_wide_source(test_case, kSourceBase, [](uint32_t lane, uint32_t) {
          switch (lane % 6u) {
          case 0:
            return std::bit_cast<float>(0x7F80'0000u);
          case 1:
            return std::bit_cast<float>(0xFF80'0000u);
          case 2:
            return std::bit_cast<float>(0x7FC1'2345u);
          case 3:
            return std::bit_cast<float>(0xFFC1'2345u);
          case 4:
            return 70000.0f;
          default:
            return -70000.0f;
          }
        });
        ForceScalarGuard guard(force_scalar);
        EXPECT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
        return fixture.snapshot_vgpr_window();
      };

      for (bool fp16_ovfl : {false, true}) {
        SCOPED_TRACE(fp16_ovfl ? "fp16_ovfl" : "default_mode");
        const auto scalar_result = run(true, fp16_ovfl);
        const auto simd_result = run(false, fp16_ovfl);
        EXPECT_EQ(simd_result, scalar_result);
        const uint32_t positive_code =
            bf8 ? (fp16_ovfl ? 0x7Bu : 0x7Cu) : (fp16_ovfl ? 0x7Eu : 0x7Fu);
        for (uint32_t word = 0; word < packed_word_count(test_case); ++word) {
          for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
            const bool nan = lane % 6u == 2u || lane % 6u == 3u;
            const bool negative = lane % 6u == 1u || lane % 6u == 3u || lane % 6u == 5u;
            const uint32_t code = (nan ? 0x7Fu : positive_code) | (negative ? 0x80u : 0u);
            EXPECT_EQ(scalar_result[word * kWaveSize + lane], 0x0101'0101u * code);
          }
        }
      }
    }
  }
}

TEST(Gfx1250MxfpCvtSimdCorrectness, UnpackSpecialScalesMasksAndOverflowAreBitExact) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    ConversionFixture fixture("gfx1250_mxfp_cvt_unpack_edges");
    ASSERT_NE(fixture.cu, nullptr);
    ASSERT_NE(fixture.decoder, nullptr);
    ASSERT_NE(fixture.wf, nullptr);

    // Bytes selected by OPSEL are respectively E8M0 underflow, NaN, maximum
    // finite, and unity. EXEC leaves lanes 16-23 inactive, while the write mask
    // suppresses a different subset of active lanes.
    constexpr uint32_t kScaleWord = 0x7FFE'FF00u;
    constexpr uint64_t kExecWithInactiveChunk = 0xFF00'FF0Fu;
    for (const ConversionCase &test_case : kConversionCases) {
      if (test_case.kind != ConversionKind::Unpack)
        continue;
      for (uint32_t scale_byte = 0; scale_byte < 4; ++scale_byte) {
        SCOPED_TRACE(test_case.mnemonic.data());
        SCOPED_TRACE(scale_byte);
        const auto words =
            build_conversion(test_case, kDestinationBase, kSourceBase, vgpr_source(kSeedReg),
                             vgpr_source(kScaleReg), scale_byte);
        std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
        ASSERT_NE(instruction, nullptr);

        const auto run = [&](bool force_scalar) {
          fixture.seed(test_case, kExecWithInactiveChunk);
          fixture.wf->set_mode_raw(amdgpu::Wavefront::FP16_OVFL_BIT);
          fixture.wf->set_vgpr_write_mask(kRestrictedWriteMask);
          for (uint32_t lane = 0; lane < kWaveSize; ++lane)
            fixture.cu->write_vgpr(fixture.vgpr_base + kScaleReg, lane, kScaleWord);
          fixture.fill_packed_source(test_case, kSourceBase, (1u << test_case.low_bits) - 1u);
          ForceScalarGuard guard(force_scalar);
          EXPECT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
          return fixture.snapshot_vgpr_window();
        };

        const auto simd_result = run(false);
        const auto scalar_result = run(true);
        EXPECT_EQ(simd_result, scalar_result);

        // NaN operand selection for multiplication is compiler- and
        // consumer-sensitive. The untouched generated scalar expression is
        // deliberately the oracle for those bit patterns.
      }

      // One exceptional active lane scalarizes its native chunk. Keep the
      // neighboring lanes ordinary to verify that the complete chunk still
      // matches the generated scalar pipeline.
      SCOPED_TRACE("mixed_exceptional_chunk");
      const auto words = build_conversion(test_case, kDestinationBase, kSourceBase,
                                          vgpr_source(kSeedReg), vgpr_source(kScaleReg), 0u);
      std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
      ASSERT_NE(instruction, nullptr);
      const auto run_mixed_chunk = [&](bool force_scalar) {
        fixture.seed(test_case, kFullExec);
        fixture.fill_packed_source(test_case, kSourceBase, 1u);
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          const uint32_t scale_code = lane == 0u ? 0xffu : 0x7fu;
          fixture.cu->write_vgpr(fixture.vgpr_base + kScaleReg, lane, scale_code);
        }
        ForceScalarGuard guard(force_scalar);
        EXPECT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
        return fixture.snapshot_vgpr_window();
      };
      EXPECT_EQ(run_mixed_chunk(false), run_mixed_chunk(true));
    }
  }
}

TEST(Gfx1250MxfpCvtSimdCorrectness, InactiveExecLanesPreserveHostFloatingPointExceptions) {
  if constexpr (!util::has_stdx_simd || util::native_width_v<uint32_t> < 2u) {
    GTEST_SKIP() << "native SIMD with at least two lanes unavailable";
  } else {
    HeldFloatingEnvironment environment;
    ASSERT_TRUE(environment.valid());
    ConversionFixture fixture("gfx1250_mxfp_cvt_inactive_fenv");
    ASSERT_NE(fixture.cu, nullptr);
    ASSERT_NE(fixture.decoder, nullptr);
    ASSERT_NE(fixture.wf, nullptr);

    struct RunResult {
      int clear_result;
      int raise_result;
      int baseline_exceptions;
      int result_exceptions;
      bool succeeded;
      std::array<uint32_t, kMaxDestinationRegs * kWaveSize> destination;
    };

    for (std::string_view mnemonic : {"v_cvt_scalef32_pk8_fp4_f32", "v_cvt_scale_pk8_f32_fp4"}) {
      SCOPED_TRACE(mnemonic);
      const ConversionCase *test_case = find_conversion(mnemonic);
      ASSERT_NE(test_case, nullptr);
      const auto words = build_conversion(*test_case);
      std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
      ASSERT_NE(instruction, nullptr);

      const auto run = [&](bool force_scalar) {
        seed_inactive_lane_fp_hazard(fixture, *test_case);
        ForceScalarGuard guard(force_scalar);
        const int clear_result = std::feclearexcept(FE_ALL_EXCEPT);
        const int raise_result = std::feraiseexcept(FE_INVALID);
        const int baseline_exceptions = std::fetestexcept(FE_ALL_EXCEPT);
        const bool succeeded =
            fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded();
        const int result_exceptions = std::fetestexcept(FE_ALL_EXCEPT);
        return RunResult{clear_result,      raise_result, baseline_exceptions,
                         result_exceptions, succeeded,    fixture.snapshot_vgpr_window()};
      };

      const RunResult scalar = run(true);
      const RunResult simd = run(false);
      ASSERT_EQ(scalar.clear_result, 0);
      ASSERT_EQ(scalar.raise_result, 0);
      ASSERT_EQ(simd.clear_result, 0);
      ASSERT_EQ(simd.raise_result, 0);
      ASSERT_TRUE(scalar.succeeded);
      ASSERT_TRUE(simd.succeeded);
      EXPECT_EQ(scalar.baseline_exceptions, FE_INVALID);
      EXPECT_EQ(scalar.result_exceptions, scalar.baseline_exceptions);
      EXPECT_EQ(simd.baseline_exceptions, scalar.baseline_exceptions);
      EXPECT_EQ(simd.result_exceptions, scalar.result_exceptions);
      EXPECT_EQ(simd.destination, scalar.destination);

      const uint32_t destination_words = test_case->kind == ConversionKind::Unpack
                                             ? wide_word_count(*test_case)
                                             : packed_word_count(*test_case);
      for (uint32_t word = 0; word < destination_words; ++word) {
        const uint32_t sentinel = 0xA5A5'0000u ^ (word * 0x101u) ^ 1u;
        EXPECT_EQ(simd.destination[word * kWaveSize + 1u], sentinel);
      }
    }
  }
}

#if GTEST_HAS_DEATH_TEST && defined(__linux__) && defined(__SSE__) &&                              \
    (defined(__x86_64__) || defined(__i386__))
TEST(Gfx1250MxfpCvtSimdDeathTest, InactiveExecLanesDoNotTrapWithUnmaskedHostExceptions) {
  if constexpr (!util::has_stdx_simd || util::native_width_v<uint32_t> < 2u) {
    GTEST_SKIP() << "native SIMD with at least two lanes unavailable";
  } else {
    ASSERT_EXIT(std::_Exit(run_inactive_lane_unmasked_exception_test()),
                ::testing::ExitedWithCode(0), "");
  }
}
#endif

TEST(Gfx1250MxfpCvtSimdCorrectness, NativeSimdHelperHandlesSupportedOperands) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    const ConversionCase *test_case = find_conversion("v_cvt_scalef32_sr_pk8_fp4_f32");
    ASSERT_NE(test_case, nullptr);
    ConversionFixture fixture("gfx1250_mxfp_cvt_simd_gate");
    ASSERT_NE(fixture.cu, nullptr);
    ASSERT_NE(fixture.decoder, nullptr);
    ASSERT_NE(fixture.wf, nullptr);

    const auto words = build_conversion(*test_case);
    std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
    ASSERT_NE(instruction, nullptr);
    auto *typed = dynamic_cast<cdna5::VCvtScalef32SrPk8Fp4F32Vop3 *>(instruction.get());
    ASSERT_NE(typed, nullptr);

    fixture.seed(*test_case, kSparseExec);
    {
      ForceScalarGuard force_scalar(true);
      ASSERT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
    }
    const auto scalar_result = fixture.snapshot_vgpr_window();

    fixture.seed(*test_case, kSparseExec);
    {
      ForceScalarGuard enable_simd(false);
      ASSERT_TRUE((amdgpu::try_execute_mxfp_cvt_scale_simd<amdgpu::MxfpFormat::Fp4E2m1,
                                                           amdgpu::MxfpWideFormat::F32, 8u,
                                                           amdgpu::MxfpDirection::Pack, true>(
          *fixture.wf, fixture.vgpr_base + kDestinationBase, fixture.vgpr_base + kSourceBase,
          typed->src2, typed->src1, 0u)));
    }
    EXPECT_EQ(fixture.snapshot_vgpr_window(), scalar_result);
  }
}

TEST(Gfx1250MxfpCvtSimdCorrectness, ZeroExecLeavesEveryDestinationUntouched) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    ConversionFixture fixture("gfx1250_mxfp_cvt_zero_exec");
    ASSERT_NE(fixture.cu, nullptr);
    ASSERT_NE(fixture.decoder, nullptr);
    ASSERT_NE(fixture.wf, nullptr);

    for (const ConversionCase &test_case : kConversionCases) {
      SCOPED_TRACE(test_case.mnemonic.data());
      const auto words = build_conversion(test_case);
      std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
      ASSERT_NE(instruction, nullptr);
      fixture.seed(test_case, 0);
      const auto before = fixture.snapshot_vgpr_window();
      {
        ForceScalarGuard enable_simd(false);
        ASSERT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
      }
      EXPECT_EQ(fixture.snapshot_vgpr_window(), before);
    }
  }
}

TEST(Gfx1250MxfpCvtSimdCorrectness, Fp6PackSnapshotsOverlappingSourceScaleAndSeed) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    const ConversionCase *test_case = find_conversion("v_cvt_scalef32_sr_pk16_fp6_f32");
    ASSERT_NE(test_case, nullptr);
    ConversionFixture fixture("gfx1250_mxfp_cvt_overlap");
    ASSERT_NE(fixture.cu, nullptr);
    ASSERT_NE(fixture.decoder, nullptr);
    ASSERT_NE(fixture.wf, nullptr);

    // Destination v[0:2] aliases the F32 source, with v1 also supplying the SR
    // seed and v2 the scale. FP6 elements 5 and 10 cross DWORD boundaries.
    const auto words = build_conversion(*test_case, 0, 0, vgpr_source(1), vgpr_source(2));
    std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
    ASSERT_NE(instruction, nullptr);
    const auto run = [&](bool force_scalar) {
      fixture.seed(*test_case, kFullExec);
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        fixture.cu->write_vgpr(fixture.vgpr_base + 1, lane, 0x9E37'79B9u ^ (lane * 0x85EB'CA6Bu));
        fixture.cu->write_vgpr(fixture.vgpr_base + 2, lane, std::bit_cast<uint32_t>(1.0f));
      }
      ForceScalarGuard guard(force_scalar);
      EXPECT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
      return fixture.snapshot_vgpr_window(0);
    };
    EXPECT_EQ(run(false), run(true));
  }
}

TEST(Gfx1250MxfpCvtSimdCorrectness, Fp6UnpackSnapshotsOverlappingSourceDestinationAndScale) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    const ConversionCase *test_case = find_conversion("v_cvt_scale_pk16_f32_fp6");
    ASSERT_NE(test_case, nullptr);
    ConversionFixture fixture("gfx1250_mxfp_cvt_unpack_overlap");
    ASSERT_NE(fixture.cu, nullptr);
    ASSERT_NE(fixture.decoder, nullptr);
    ASSERT_NE(fixture.wf, nullptr);

    // Destination v[0:15] aliases the packed source v[0:2], and v4 supplies
    // the scale while also being overwritten by destination element 4.
    const auto words = build_conversion(*test_case, 0, 0, vgpr_source(kSeedReg), vgpr_source(4), 0);
    std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
    ASSERT_NE(instruction, nullptr);
    const auto run = [&](bool force_scalar) {
      fixture.seed(*test_case, kFullExec);
      fixture.fill_packed_source(*test_case, 0, 0x15u);
      for (uint32_t lane = 0; lane < kWaveSize; ++lane)
        fixture.cu->write_vgpr(fixture.vgpr_base + 4, lane, 0x7F7F'7F7Fu);
      ForceScalarGuard guard(force_scalar);
      EXPECT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
      return fixture.snapshot_vgpr_window(0);
    };
    EXPECT_EQ(run(false), run(true));
  }
}

TEST(Gfx1250MxfpCvtSimdCorrectness, ScalarAndInlineScaleSeedOperandsAreBitExact) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    ConversionFixture fixture("gfx1250_mxfp_cvt_scalar_operands");
    ASSERT_NE(fixture.cu, nullptr);
    ASSERT_NE(fixture.decoder, nullptr);
    ASSERT_NE(fixture.wf, nullptr);

    const auto compare = [&](const ConversionCase &test_case, const std::array<uint32_t, 2> &words,
                             auto prepare) {
      std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
      ASSERT_NE(instruction, nullptr);
      const auto run = [&](bool force_scalar) {
        fixture.seed(test_case, kSparseExec);
        prepare(fixture);
        ForceScalarGuard guard(force_scalar);
        EXPECT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
        return fixture.snapshot_vgpr_window();
      };
      EXPECT_EQ(run(false), run(true));
    };

    const ConversionCase *rne = find_conversion("v_cvt_scalef32_pk8_fp4_f32");
    ASSERT_NE(rne, nullptr);
    compare(*rne,
            build_conversion(*rne, kDestinationBase, kSourceBase, cdna5::OPR_SRC_POS_INT_MIN,
                             cdna5::OPR_SRC_FLOAT_ONE),
            [](ConversionFixture &) {});

    const ConversionCase *sr = find_conversion("v_cvt_scalef32_sr_pk16_fp6_f32");
    ASSERT_NE(sr, nullptr);
    compare(*sr,
            build_conversion(*sr, kDestinationBase, kSourceBase, cdna5::OPR_SRC_POS_INT_MIN + 17,
                             cdna5::OPR_SRC_SGPR_MIN + 8),
            [](ConversionFixture &prepared) {
              prepared.cu->write_sgpr(prepared.wf->sgpr_alloc().base + 8,
                                      std::bit_cast<uint32_t>(1.0f));
            });

    const ConversionCase *unpack = find_conversion("v_cvt_scale_pk8_f16_fp8");
    ASSERT_NE(unpack, nullptr);
    compare(*unpack,
            build_conversion(*unpack, kDestinationBase, kSourceBase, cdna5::OPR_SRC_POS_INT_MIN,
                             cdna5::OPR_SRC_SGPR_MIN + 9, 1),
            [](ConversionFixture &prepared) {
              prepared.cu->write_sgpr(prepared.wf->sgpr_alloc().base + 9, 0x0080'7FFFu);
            });
  }
}

TEST(Gfx1250MxfpCvtSimdCorrectness, VgprMsbBanksAreResolvedBeforeSimdExecution) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    const ConversionCase *test_case = find_conversion("v_cvt_scalef32_pk8_fp4_f32");
    ASSERT_NE(test_case, nullptr);
    ConversionFixture fixture("gfx1250_mxfp_cvt_vgpr_msb");
    ASSERT_NE(fixture.cu, nullptr);
    ASSERT_NE(fixture.decoder, nullptr);
    ASSERT_NE(fixture.wf, nullptr);
    const auto words = build_conversion(*test_case);
    std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
    ASSERT_NE(instruction, nullptr);

    constexpr uint32_t kSrc0Bank = 1;
    constexpr uint32_t kScaleBank = 2;
    constexpr uint32_t kDstBank = 3;
    constexpr uint8_t kVgprMsb = kSrc0Bank | (kScaleBank << 2) | (kDstBank << 6);
    const auto run = [&](bool force_scalar) {
      fixture.seed(*test_case, kFullExec);
      fixture.wf->set_vgpr_msb_mode(kVgprMsb);
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        for (uint32_t reg = 0; reg < source_reg_count(*test_case); ++reg) {
          const float value =
              static_cast<float>(static_cast<int>(lane % 7u) - 3) + static_cast<float>(reg) * 0.25f;
          fixture.cu->write_vgpr(fixture.vgpr_base + kSrc0Bank * 256u + reg, lane,
                                 std::bit_cast<uint32_t>(value));
        }
        fixture.cu->write_vgpr(fixture.vgpr_base + kScaleBank * 256u + kScaleReg, lane,
                               std::bit_cast<uint32_t>(1.0f));
        for (uint32_t reg = 0; reg < kMaxDestinationRegs; ++reg)
          fixture.cu->write_vgpr(fixture.vgpr_base + kDstBank * 256u + kDestinationBase + reg, lane,
                                 0xC5C5'0000u ^ (reg * 0x101u) ^ lane);
      }
      ForceScalarGuard guard(force_scalar);
      EXPECT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
      return fixture.snapshot_vgpr_window(kDstBank * 256u + kDestinationBase);
    };
    EXPECT_EQ(run(false), run(true));
  }
}

TEST(Gfx1250MxfpCvtBenchmark, ScalarVsSimdWave32) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
    ConversionFixture fixture("gfx1250_mxfp_cvt_benchmark");
    ASSERT_NE(fixture.cu, nullptr);
    ASSERT_NE(fixture.decoder, nullptr);
    ASSERT_NE(fixture.wf, nullptr);
    ASSERT_EQ(fixture.wf->wf_size(), kWaveSize);

    std::printf("\n  === gfx1250 scaled MXFP conversions (wave32) ===\n"
                "  native uint32 lanes: %zu\n"
                "  samples: %zu  warmup/sample: %d  timed iterations/sample: %d\n",
                util::native_width_v<uint32_t>, kBenchmarkSamples, kWarmupIterations,
                kTimedIterations);
    for (const ConversionCase &test_case : kConversionCases) {
      SCOPED_TRACE(test_case.mnemonic.data());
      const auto words = build_conversion(test_case);
      std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
      ASSERT_NE(instruction, nullptr);
      ASSERT_EQ(std::string_view(instruction->mnemonic()), test_case.mnemonic);

      fixture.seed(test_case, kFullExec);
      {
        ForceScalarGuard force_scalar(true);
        ASSERT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
      }
      const auto scalar_result = fixture.snapshot_vgpr_window();
      fixture.seed(test_case, kFullExec);
      {
        ForceScalarGuard enable_simd(false);
        ASSERT_TRUE(fixture.cu->execute_instruction(instruction.get(), *fixture.wf).succeeded());
      }
      EXPECT_EQ(fixture.snapshot_vgpr_window(), scalar_result);

      std::array<double, kBenchmarkSamples> scalar_samples{};
      std::array<double, kBenchmarkSamples> simd_samples{};
      for (std::size_t sample = 0; sample < kBenchmarkSamples; ++sample) {
        if ((sample & 1u) == 0u) {
          scalar_samples[sample] = time_sample(fixture, test_case, *instruction, true);
          simd_samples[sample] = time_sample(fixture, test_case, *instruction, false);
        } else {
          simd_samples[sample] = time_sample(fixture, test_case, *instruction, false);
          scalar_samples[sample] = time_sample(fixture, test_case, *instruction, true);
        }
      }

      const double scalar_us = median(scalar_samples);
      const double simd_us = median(simd_samples);
      const double speedup = scalar_us / simd_us;
      std::printf("  %-38s scalar %8.3f us  SIMD %8.3f us  speedup %6.2fx\n",
                  test_case.mnemonic.data(), scalar_us, simd_us, speedup);
      EXPECT_GT(scalar_us, 0.0);
      EXPECT_GT(simd_us, 0.0);
    }
  }
}
