// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded normalized conversions: inherited rounding, guest denormals,
// halfword publication, modifiers, source staging, aliases and masked lanes.

#include "compute_unit_test_util.h"
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
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/execution_backend.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include "util/simd.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <random>

namespace rocjitsu::amdgpu {
namespace {

// Evaluate the inherited double-product/RNE model independently of API dispatch.
uint16_t reference(uint32_t raw, bool half, bool is_unsigned, bool absolute, bool negate,
                   const Wavefront &wave) {
  if (half) {
    raw &= 0xffff;
    if (!(wave.fp_denorm_mode_f16_f64() & 1) && (raw & 0x7fff) < 0x400) {
      raw &= 0x8000;
    }
    raw = std::bit_cast<uint32_t>(util::f16_to_f32(uint16_t(raw)));
  }
  if (absolute) {
    raw &= 0x7fffffff;
  }
  if (negate) {
    raw ^= 0x80000000;
  }
  double value = std::bit_cast<float>(raw);
  if (value != value) {
    return 0;
  }
  const double scale = is_unsigned ? 65535 : 32767;
  return uint16_t(
      int32_t(util::rndne_scalar(std::clamp(value * scale, is_unsigned ? 0.0 : -scale, scale))));
}

// Restore the test dispatch override on every exit path.
struct ScalarGuard {
  bool saved = util::force_scalar();
  ~ScalarGuard() { util::set_force_scalar_for_testing(saved); }
};

// Check all six conversions against the inherited model on every older profile.
TEST(OlderNormalizedConversion, Arithmetic) {
  ScalarGuard guard;
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    // Signed/unsigned packed F32, packed F16 and unary F16 forms.
    std::array<uint16_t, 6> vop3;
    std::array<uint16_t, 2> vop1;
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {cdna1::kVCvtPknormI16F32Vop3, cdna1::kVCvtPknormU16F32Vop3, cdna1::kVCvtPknormI16F16Vop3,
        cdna1::kVCvtPknormU16F16Vop3, cdna1::kVCvtNormI16F16Vop3, cdna1::kVCvtNormU16F16Vop3},
       {cdna1::kVCvtNormI16F16Vop1, cdna1::kVCvtNormU16F16Vop1}},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {cdna2::kVCvtPknormI16F32Vop3, cdna2::kVCvtPknormU16F32Vop3, cdna2::kVCvtPknormI16F16Vop3,
        cdna2::kVCvtPknormU16F16Vop3, cdna2::kVCvtNormI16F16Vop3, cdna2::kVCvtNormU16F16Vop3},
       {cdna2::kVCvtNormI16F16Vop1, cdna2::kVCvtNormU16F16Vop1}},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {rdna1::kVCvtPknormI16F32Vop3, rdna1::kVCvtPknormU16F32Vop3, rdna1::kVCvtPknormI16F16Vop3,
        rdna1::kVCvtPknormU16F16Vop3, rdna1::kVCvtNormI16F16Vop3, rdna1::kVCvtNormU16F16Vop3},
       {rdna1::kVCvtNormI16F16Vop1, rdna1::kVCvtNormU16F16Vop1}},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {rdna2::kVCvtPknormI16F32Vop3, rdna2::kVCvtPknormU16F32Vop3, rdna2::kVCvtPknormI16F16Vop3,
        rdna2::kVCvtPknormU16F16Vop3, rdna2::kVCvtNormI16F16Vop3, rdna2::kVCvtNormU16F16Vop3},
       {rdna2::kVCvtNormI16F16Vop1, rdna2::kVCvtNormU16F16Vop1}},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {rdna3_5::kVCvtPkNormI16F32Vop3, rdna3_5::kVCvtPkNormU16F32Vop3,
        rdna3_5::kVCvtPkNormI16F16Vop3, rdna3_5::kVCvtPkNormU16F16Vop3,
        rdna3_5::kVCvtNormI16F16Vop3, rdna3_5::kVCvtNormU16F16Vop3},
       {rdna3_5::kVCvtNormI16F16Vop1, rdna3_5::kVCvtNormU16F16Vop1}},
  };
  enum class Form {
    Vop3,
    Vop1,
    Sdwa,
    Dpp,
    SuppressedDpp,
    ScalarSource,
    HighRegister,
    HighHalf,
    E32HighHalf
  };
  struct Scenario {
    Form form = Form::Vop3;
    uint32_t mode = 0xf0;
    unsigned absolute = 0, negate = 0;
  };
  const Scenario scenarios[] = {
      {},
      {.mode = 0xff},
      {.mode = 0},
      {.absolute = 2, .negate = 1},
      {.absolute = 1, .negate = 2},
      {Form::Vop1},
      {Form::Sdwa},
      {Form::Dpp},
      {Form::SuppressedDpp},
      {Form::ScalarSource},
      {Form::HighRegister},
      {Form::HighHalf},
      {Form::E32HighHalf},
  };
  constexpr uint32_t special[] = {0x7fc12345, 0x7f800000, 0xff800000, 0x80000000,
                                  0x3effffff, 0xbeffffff, 0x3f800000, 0xbf800000,
                                  0x38000100, 0xb8000100, 0x3f000001, 0xbf000001,
                                  0x3f7fffff, 0x33800000, 0x7f800001, 0};
  constexpr uint32_t special_half[] = {0x7e12, 0x7c00, 0xfc00, 0x8000, 0x0001, 0x8001,
                                       0x3c00, 0xbc00, 0x3801, 0xb801, 0x3bff, 0xbbff,
                                       0x0400, 0x03ff, 0x7c01, 0};
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    const bool cdna =
        family.arch == rocjitsu::IsaFamily::Cdna1 || family.arch == rocjitsu::IsaFamily::Cdna2;
    const bool packed_e32 = family.arch == rocjitsu::IsaFamily::Rdna3_5;
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
          GpuMemory memory("normalized_memory");
          L2Cache cache("normalized_cache");
          cache.set_backing_memory(&memory);
          std::unique_ptr<ComputeUnitCore> cu;
          ASSERT_NO_FATAL_FAILURE(
              rocjitsu::test::initialize_compute_unit(cu, "normalized", config, &memory, &cache));
          auto *wave = cu->dispatch_wf(0, 0, 104, 144, lanes);
          ASSERT_NE(wave, nullptr);
          auto decoder = Decoder::create(family.arch);
          const auto base = wave->vgpr_alloc().base;
          std::minstd_rand random;
          // Sequence high-bit draws so word construction is toolchain-independent.
          const auto random_word = [&] {
            const uint32_t high = uint32_t(random() >> 15);
            return (high << 16) | uint32_t(random() >> 15);
          };
          std::array<uint32_t, 64> random_a{}, random_b{};
          for (int lane = 0; lane < lanes; ++lane) {
            random_a[lane] = random_word();
            random_b[lane] = random_word();
          }
          for (int operation = 0; operation < 6; ++operation) {
            const bool packed = operation < 4, half = operation >= 2;
            auto a = random_a, b = random_b;
            for (int lane = 0; lane < 16; ++lane) {
              a[lane] = half ? special_half[lane] | (special_half[15 - lane] << 16) : special[lane];
              b[lane] =
                  half ? special_half[15 - lane] | (special_half[lane] << 16) : special[15 - lane];
            }
            for (const auto &sample : scenarios) {
              const bool sdwa = sample.form == Form::Sdwa;
              const bool dpp = sample.form == Form::Dpp || sample.form == Form::SuppressedDpp;
              const bool sgpr = sample.form == Form::ScalarSource;
              const bool high = sample.form == Form::HighHalf;
              const bool e32_high = sample.form == Form::E32HighHalf;
              const bool vop3 = packed || sample.form == Form::Vop3 || high;
              if ((packed && (sdwa || dpp || sample.form == Form::Vop1 || e32_high)) ||
                  (high && !half) || (sdwa && packed_e32) || (e32_high && !packed_e32) ||
                  (sample.form == Form::HighRegister && packed_e32)) {
                continue;
              }
              const uint32_t source_a = sample.form == Form::HighRegister ? 128 : 0;
              const uint32_t source_b = source_a + 1;
              for (int aliasing_offset : {0, 1, 2}) {
                SCOPED_TRACE(::testing::Message()
                             << family.arch << ':' << lanes << ':' << scalar << ':'
                             << int(semantics) << ':' << operation << ':' << int(sample.form) << ':'
                             << sample.mode << ':' << aliasing_offset);
                const uint32_t destination = source_a + aliasing_offset;
                const uint64_t exec_mask =
                    (lanes == 64 ? ~0ULL : 0xffffffffULL) & 0x800000038000ffffULL;
                wave->set_exec(dpp ? (lanes == 64 ? ~0ULL : 0xffffffffULL) : exec_mask);
                wave->set_vgpr_write_mask(dpp ? exec_mask : ~0ULL);
                wave->set_mode_raw(sample.mode);
                wave->set_trapsts(0x100);
                cu->write_sgpr(wave->sgpr_alloc().base, 0x3f801fff);
                for (int lane = 0; lane < lanes; ++lane) {
                  cu->write_vgpr(base + destination, lane, 0xdeadbeef);
                  cu->write_vgpr(base + source_a, lane, a[lane]);
                  cu->write_vgpr(base + source_b, lane, b[lane]);
                }
                const uint32_t source = sgpr ? 0 : 256 + source_a + (e32_high ? 128 : 0);
                const uint32_t opsel = high ? (packed ? 3 : 9) : 0;
                std::array<uint32_t, 2> words =
                    vop3 ? std::array<uint32_t, 2>{(cdna ? 0xd0000000U : 0xd4000000U) |
                                                       (uint32_t(family.vop3[operation]) << 16) |
                                                       destination | (opsel << 11) |
                                                       (sample.absolute << 8),
                                                   source | (packed ? ((256 + source_b) << 9) : 0) |
                                                       (sample.negate << 29)}
                         : std::array<uint32_t, 2>{
                               0x7e000000U | ((destination + (e32_high ? 128 : 0)) << 17) |
                                   (uint32_t(family.vop1[operation - 4]) << 9) | source,
                               0};
                if (sdwa) {
                  // Select WORD_1 of A and publish to WORD_1, preserving WORD_0.
                  words[0] = (words[0] & ~0x1ffU) | 249;
                  words[1] = 0x00051500;
                } else if (dpp) {
                  words[0] = (words[0] & ~0x1ffU) | 250;
                  words[1] = sample.form == Form::SuppressedDpp ? 0x0f00b100 : 0xff00b100;
                }
                std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
                ASSERT_NE(instruction, nullptr);
                ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
                EXPECT_EQ(wave->trapsts(), 0x100U);
                for (int lane = 0; lane < lanes; ++lane) {
                  const uint32_t old = aliasing_offset == 0   ? a[lane]
                                       : aliasing_offset == 1 ? b[lane]
                                                              : 0xdeadbeef;
                  uint32_t expected = old;
                  if ((exec_mask >> lane & 1) && sample.form != Form::SuppressedDpp) {
                    uint32_t raw_a = sgpr ? 0x3f801fff : a[dpp ? lane ^ 1 : lane], raw_b = b[lane];
                    if (high || e32_high || sdwa) {
                      raw_a >>= 16;
                    }
                    if (packed && high) {
                      raw_b >>= 16;
                    }
                    const uint32_t lo =
                        reference(raw_a, half, operation & 1, vop3 && (sample.absolute & 1),
                                  vop3 && (sample.negate & 1), *wave);
                    if (packed) {
                      expected =
                          lo | (uint32_t(reference(raw_b, half, operation & 1, sample.absolute & 2,
                                                   sample.negate & 2, *wave))
                                << 16);
                    } else if (sdwa || high || e32_high) {
                      expected = (old & 0xffff) | (lo << 16);
                    } else {
                      expected = (cdna ? 0 : old & 0xffff0000) | lo;
                    }
                  }
                  EXPECT_EQ(cu->read_vgpr(base + destination, lane), expected) << "lane=" << lane;
                }
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
