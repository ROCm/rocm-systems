// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded older-profile byte/halfword packing: inherited arithmetic, guest
// controls, zero-extension/halfword preservation, aliases and masked stores.

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

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <random>

namespace rocjitsu::amdgpu {
namespace {

enum class Packing { Halves, Byte, Accumulate, Saturate };

// Return packed bits under guest controls, preserving unselected bytes of C.
uint32_t reference(Packing operation, uint32_t a, uint32_t b, uint32_t c, const Wavefront &wave,
                   unsigned absolute, unsigned negate) {
  if (operation == Packing::Saturate) {
    return uint32_t(std::clamp(int(int16_t(a)), 0, 255)) |
           (uint32_t(std::clamp(int(int16_t(a >> 16)), 0, 255)) << 8);
  }
  if (operation == Packing::Halves) {
    const auto prepare = [&](uint32_t raw, int source) {
      raw &= 0xffff;
      if (absolute & (1U << source)) {
        raw &= 0x7fff;
      }
      if (negate & (1U << source)) {
        raw ^= 0x8000;
      }
      if (!(wave.fp_denorm_mode_f16_f64() & 1) && (raw & 0x7fff) < 0x400) {
        raw &= 0x8000;
      }
      if (wave.ieee_mode() && (raw & 0x7fff) > 0x7c00) {
        raw |= 0x200;
      }
      return raw;
    };
    return prepare(a, 0) | (prepare(b, 1) << 16);
  }
  // PKACCUM ignores modifiers and guest rounding/input flushing.
  if (operation == Packing::Byte) {
    if (absolute & 1) {
      a &= 0x7fffffff;
    }
    if (negate & 1) {
      a ^= 0x80000000;
    }
    if (!(wave.fp_denorm_mode_f32() & 1) && (a & 0x7fffffff) < 0x800000) {
      a &= 0x80000000;
    }
  }
  const float input = std::bit_cast<float>(a);
  double value = input != input ? 0 : std::clamp(double(input), 0.0, 255.0);
  switch (operation == Packing::Accumulate ? 3 : wave.fp_round_mode_f32()) {
  case 0:
    // The default host rounding mode supplies guest RNE for this reference.
    value = std::nearbyint(value);
    break;
  case 1:
    value = std::ceil(value);
    break;
  case 2:
    value = std::floor(value);
    break;
  default:
    break;
  }
  const uint32_t shift = (b & 3) * 8;
  return (c & ~(0xffU << shift)) | (uint32_t(value) << shift);
}

// Restore the test dispatch override on every exit path.
struct ScalarGuard {
  bool saved = util::force_scalar();
  ~ScalarGuard() { util::set_force_scalar_for_testing(saved); }
};

// Check all older packing forms against the inherited models and write footprints.
TEST(OlderPacking, Arithmetic) {
  ScalarGuard guard;
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    // Halfword packing, byte packing, byte accumulation, and saturating narrowing.
    std::array<int, 4> vop3;
    uint16_t vop1;
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {cdna1::kVPackB32F16Vop3, cdna1::kVCvtPkU8F32Vop3, cdna1::kVCvtPkaccumU8F32Vop3,
        cdna1::kVSatPkU8I16Vop3},
       cdna1::kVSatPkU8I16Vop1},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {cdna2::kVPackB32F16Vop3, cdna2::kVCvtPkU8F32Vop3, cdna2::kVCvtPkaccumU8F32Vop3,
        cdna2::kVSatPkU8I16Vop3},
       cdna2::kVSatPkU8I16Vop1},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {rdna1::kVPackB32F16Vop3, rdna1::kVCvtPkU8F32Vop3, -1, rdna1::kVSatPkU8I16Vop3},
       rdna1::kVSatPkU8I16Vop1},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {rdna2::kVPackB32F16Vop3, rdna2::kVCvtPkU8F32Vop3, -1, rdna2::kVSatPkU8I16Vop3},
       rdna2::kVSatPkU8I16Vop1},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {rdna3_5::kVPackB32F16Vop3, rdna3_5::kVCvtPkU8F32Vop3, -1, rdna3_5::kVSatPkU8I16Vop3},
       rdna3_5::kVSatPkU8I16Vop1},
  };
  enum class Form {
    Vop3,
    Vop1,
    HighHalf,
    E32HighHalf,
    HighRegister,
    E32HighRegister,
    ScalarSource,
    Sdwa,
    Dpp,
    SuppressedDpp,
    Vop3Dpp,
    Vop3HighHalfDpp
  };
  struct Scenario {
    Form form = Form::Vop3;
    uint32_t mode = 0xf0;
    unsigned absolute = 0, negate = 0, omod = 0;
    bool clamp = false;
  };
  const Scenario scenarios[] = {
      {},
      {.mode = 0xf1},
      {.mode = 0xf2},
      {.mode = 0xf3},
      {.mode = 0xc1},
      {.mode = 0x2f0},
      {.mode = 0},
      {.absolute = 1, .negate = 2},
      {.absolute = 2, .negate = 1, .omod = 3, .clamp = true},
      {Form::Vop1},
      {Form::HighHalf},
      {Form::E32HighHalf},
      {Form::HighRegister},
      {Form::E32HighRegister},
      {Form::ScalarSource},
      {Form::Sdwa},
      {Form::Dpp},
      {Form::SuppressedDpp},
      {Form::Vop3Dpp},
      {Form::Vop3HighHalfDpp},
  };
  constexpr uint32_t special[] = {0x7fc12345, 0x7f800000, 0xff800000, 0x80000000,
                                  0x3fc00000, 0x40200000, 0x3f000000, 0xbf000000,
                                  0x437f8000, 0x43800000, 1,          0x80000001,
                                  0x40400000, 0xc0400000, 0x7f800001, 0};
  constexpr uint32_t halves[] = {0x7e12, 0x7c00, 0xfc00, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00,
                                 0x3801, 0xb801, 0x3bff, 0xbbff, 0x0400, 0x03ff, 0x7c01, 0};
  constexpr uint32_t integers[] = {0xffff0001, 0x0001ffff, 0x00ff0100, 0x010000ff,
                                   0x80007fff, 0x7fff8000, 0x00000000, 0xffffffff,
                                   0x007f0080, 0x00fe00ff, 0x01000101, 0xffff8000,
                                   0x8000ffff, 0x7fff0000, 0x00007fff, 0x11223344};
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    const bool cdna =
        family.arch == rocjitsu::IsaFamily::Cdna1 || family.arch == rocjitsu::IsaFamily::Cdna2;
    const bool gfx10 =
        family.arch == rocjitsu::IsaFamily::Rdna1 || family.arch == rocjitsu::IsaFamily::Rdna2;
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
          GpuMemory memory("packing_memory");
          L2Cache cache("packing_cache");
          cache.set_backing_memory(&memory);
          std::unique_ptr<ComputeUnitCore> cu;
          ASSERT_NO_FATAL_FAILURE(
              rocjitsu::test::initialize_compute_unit(cu, "packing", config, &memory, &cache));
          auto *wave = cu->dispatch_wf(0, 0, 104, 144, lanes);
          ASSERT_NE(wave, nullptr);
          auto decoder = Decoder::create(family.arch);
          const auto base = wave->vgpr_alloc().base;
          std::minstd_rand random;
          // Sequence high-bit draws to specify words independently of evaluation order.
          const auto random_word = [&] {
            const uint32_t high = uint32_t(random() >> 15);
            return (high << 16) | uint32_t(random() >> 15);
          };
          std::array<std::array<uint32_t, 64>, 3> raw{};
          for (auto &operand : raw) {
            for (int lane = 0; lane < lanes; ++lane) {
              operand[lane] = random_word();
            }
          }
          for (Packing operation :
               {Packing::Halves, Packing::Byte, Packing::Accumulate, Packing::Saturate}) {
            if (family.vop3[int(operation)] < 0) {
              continue;
            }
            const bool sat = operation == Packing::Saturate;
            auto input = raw;
            for (int lane = 0; lane < 16; ++lane) {
              input[0][lane] = sat ? integers[lane]
                               : operation == Packing::Halves
                                   ? halves[lane] | (halves[15 - lane] << 16)
                                   : special[lane];
              input[1][lane] = operation == Packing::Halves
                                   ? halves[15 - lane] | (halves[lane] << 16)
                                   : 0xa5a50000U | uint32_t(lane);
              input[2][lane] = 0x11223344U ^ (uint32_t(lane) << 8);
            }
            for (const auto &sample : scenarios) {
              const bool sdwa = sample.form == Form::Sdwa;
              const bool vop3_dpp =
                  sample.form == Form::Vop3Dpp || sample.form == Form::Vop3HighHalfDpp;
              const bool dpp =
                  sample.form == Form::Dpp || sample.form == Form::SuppressedDpp || vop3_dpp;
              const bool e32_high = sample.form == Form::E32HighHalf;
              const bool high =
                  sample.form == Form::HighHalf || sample.form == Form::Vop3HighHalfDpp;
              const bool e32_high_register = sample.form == Form::E32HighRegister;
              const bool vop3 = sample.form != Form::Vop1 && !sdwa && !(dpp && !vop3_dpp) &&
                                !e32_high && !e32_high_register;
              if ((vop3_dpp && family.arch != rocjitsu::IsaFamily::Rdna3_5) || (!sat && !vop3) ||
                  (high && operation != Packing::Halves && !sat) ||
                  (e32_high && family.arch != rocjitsu::IsaFamily::Rdna3_5) ||
                  ((sdwa || e32_high_register) && family.arch == rocjitsu::IsaFamily::Rdna3_5)) {
                continue;
              }
              const bool sgpr = sample.form == Form::ScalarSource;
              const uint32_t first =
                  sample.form == Form::HighRegister || e32_high_register ? 128 : 0;
              for (int aliasing_offset : {0, 1, 2, 3}) {
                SCOPED_TRACE(::testing::Message()
                             << family.arch << ':' << lanes << ':' << scalar << ':'
                             << int(semantics) << ':' << int(operation) << ':' << int(sample.form)
                             << ':' << sample.mode << ':' << aliasing_offset);
                const uint32_t destination = first + aliasing_offset;
                const uint64_t all = lanes == 64 ? ~0ULL : 0xffffffffULL;
                const uint64_t exec_mask = all & 0x800000038000ffffULL;
                wave->set_exec(dpp ? all : exec_mask);
                wave->set_vgpr_write_mask(dpp ? exec_mask : all);
                wave->set_mode_raw(sample.mode);
                wave->set_trapsts(0x100);
                cu->write_sgpr(wave->sgpr_alloc().base, 0x3fc00000);
                for (int lane = 0; lane < lanes; ++lane) {
                  cu->write_vgpr(base + destination, lane, 0xdeadbeef);
                  for (int operand = 0; operand < 3; ++operand) {
                    cu->write_vgpr(base + first + operand, lane, input[operand][lane]);
                  }
                }
                const uint32_t source = sgpr ? 0 : 256 + first;
                const uint32_t opsel = high ? (sat ? 8 : 3) : 0;
                const uint32_t source_b =
                    operation != Packing::Saturate ? (256 + first + 1) << 9 : 0;
                const uint32_t source_c = operation == Packing::Byte ? (256 + first + 2) << 18 : 0;
                std::array<uint32_t, 3> words =
                    vop3
                        ? std::array<uint32_t, 3>{(cdna ? 0xd0000000U : 0xd4000000U) |
                                                      (uint32_t(family.vop3[int(operation)])
                                                       << 16) |
                                                      destination | (opsel << 11) |
                                                      (sample.absolute << 8) |
                                                      (sample.clamp ? 1U << 15 : 0),
                                                  source | source_b | source_c |
                                                      (sample.omod << 27) | (sample.negate << 29)}
                        : std::array<uint32_t, 3>{0x7e000000U |
                                                      ((destination + (e32_high ? 128 : 0)) << 17) |
                                                      (uint32_t(family.vop1) << 9) | source,
                                                  0};
                if (sdwa) {
                  // Narrow the selected upper half, then place the packed bytes in WORD_1.
                  words[0] = (words[0] & ~0x1ffU) | 249;
                  words[1] = 0x00051500;
                } else if (dpp) {
                  words[vop3 ? 1 : 0] = (words[vop3 ? 1 : 0] & ~0x1ffU) | 250;
                  words[vop3 ? 2 : 1] =
                      sample.form == Form::SuppressedDpp ? 0x0f00b100 : 0xff00b100;
                }
                std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
                ASSERT_NE(instruction, nullptr);
                ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
                EXPECT_EQ(wave->trapsts(), 0x100U);
                for (int lane = 0; lane < lanes; ++lane) {
                  const uint32_t old =
                      aliasing_offset < 3 ? input[aliasing_offset][lane] : 0xdeadbeef;
                  uint32_t expected = old;
                  if ((exec_mask >> lane & 1) && sample.form != Form::SuppressedDpp) {
                    uint32_t a = sgpr ? 0x3fc00000 : input[0][dpp ? lane ^ 1 : lane],
                             b = input[1][lane];
                    if (high && operation == Packing::Halves) {
                      a >>= 16;
                      b >>= 16;
                    }
                    if (sdwa) {
                      a >>= 16;
                    }
                    const uint32_t value = reference(
                        operation, a, b, operation == Packing::Accumulate ? old : input[2][lane],
                        *wave, vop3 ? sample.absolute : 0, vop3 ? sample.negate : 0);
                    if (sat && (sdwa || e32_high || (high && !gfx10))) {
                      expected = (old & 0xffff) | (value << 16);
                    } else if (sat && !cdna && !gfx10) {
                      expected = (old & 0xffff0000) | value;
                    } else {
                      expected = value;
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
