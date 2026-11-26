// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded F32/F64 exponent extraction: integer results, guest modes and routed operands.
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
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
#include "rocjitsu/isa/target_registry.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <random>

namespace rocjitsu::amdgpu {
namespace {

// Restore test-only dispatch overrides even when an assertion exits early.
struct ScalarGuard {
  bool saved = util::force_scalar();
  ~ScalarGuard() { util::set_force_scalar_for_testing(saved); }
};

// The inherited model is frexp's exponent for finite nonzero inputs; specials yield zero.
uint32_t reference(uint64_t bits, bool wide, bool flush) {
  const uint64_t magnitude = bits & (wide ? 0x7fffffffffffffffULL : 0x7fffffffULL);
  if (flush && magnitude < (wide ? 0x0010000000000000ULL : 0x00800000ULL)) {
    return 0;
  }
  const double value =
      wide ? std::bit_cast<double>(bits) : double(std::bit_cast<float>(uint32_t(bits)));
  int exponent = 0;
  if (std::isfinite(value) && value != 0) {
    std::frexp(value, &exponent);
  }
  return uint32_t(exponent);
}

// Check integer outputs, status preservation, overlap, SDWA and admitted DPP forms.
TEST(FrexpExponent, Arithmetic) {
  ScalarGuard guard;
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<uint16_t, 2> vop1, vop3;
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {cdna1::kVFrexpExpI32F32Vop1, cdna1::kVFrexpExpI32F64Vop1},
       {cdna1::kVFrexpExpI32F32Vop3, cdna1::kVFrexpExpI32F64Vop3}},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {cdna2::kVFrexpExpI32F32Vop1, cdna2::kVFrexpExpI32F64Vop1},
       {cdna2::kVFrexpExpI32F32Vop3, cdna2::kVFrexpExpI32F64Vop3}},
      {rocjitsu::IsaFamily::Cdna3,
       &cdna3::execution_backend(),
       {cdna3::kVFrexpExpI32F32Vop1, cdna3::kVFrexpExpI32F64Vop1},
       {cdna3::kVFrexpExpI32F32Vop3, cdna3::kVFrexpExpI32F64Vop3}},
      {rocjitsu::IsaFamily::Cdna4,
       &cdna4::execution_backend(),
       {cdna4::kVFrexpExpI32F32Vop1, cdna4::kVFrexpExpI32F64Vop1},
       {cdna4::kVFrexpExpI32F32Vop3, cdna4::kVFrexpExpI32F64Vop3}},
      {rocjitsu::IsaFamily::Cdna5,
       &cdna5::execution_backend(),
       {cdna5::kVFrexpExpI32F32Vop1, cdna5::kVFrexpExpI32F64Vop1},
       {cdna5::kVFrexpExpI32F32Vop3, cdna5::kVFrexpExpI32F64Vop3}},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {rdna1::kVFrexpExpI32F32Vop1, rdna1::kVFrexpExpI32F64Vop1},
       {rdna1::kVFrexpExpI32F32Vop3, rdna1::kVFrexpExpI32F64Vop3}},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {rdna2::kVFrexpExpI32F32Vop1, rdna2::kVFrexpExpI32F64Vop1},
       {rdna2::kVFrexpExpI32F32Vop3, rdna2::kVFrexpExpI32F64Vop3}},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {rdna3_5::kVFrexpExpI32F32Vop1, rdna3_5::kVFrexpExpI32F64Vop1},
       {rdna3_5::kVFrexpExpI32F32Vop3, rdna3_5::kVFrexpExpI32F64Vop3}},
  };
  enum class Form { Vop1, Vop3, Scalar, Sdwa, Dpp, SuppressedDpp, Vop3Dpp, SuppressedVop3Dpp };
  struct Scenario {
    Form form = Form::Vop3;
    uint32_t mode = 0;
    bool modifiers = false;
  };
  const Scenario scenarios[] = {
      {Form::Vop1},
      {Form::Vop1, 0xf0},
      {},
      {.mode = 0xff},
      {.mode = 0x200},
      {.mode = 0xf0, .modifiers = true},
      {Form::Scalar},
      {Form::Sdwa, 0xf0},
      {Form::Dpp, 0xf0},
      {Form::SuppressedDpp, 0xf0},
      {Form::Vop3Dpp, 0xf0},
      {Form::SuppressedVop3Dpp, 0xf0},
  };
  constexpr uint32_t special32[] = {0,          0x80000000, 1,          0x80000001,
                                    0x007fffff, 0x00800000, 0x3f800000, 0x3f7fffff,
                                    0xbf800000, 0x3fc00000, 0x7f7fffff, 0xff7fffff,
                                    0x7f800000, 0xff800000, 0x7f812345, 0xffc12345};
  constexpr uint64_t special64[] = {0,
                                    0x8000000000000000ULL,
                                    1,
                                    0x8000000000000001ULL,
                                    0x000fffffffffffffULL,
                                    0x0010000000000000ULL,
                                    0x3ff0000000000000ULL,
                                    0x3fefffffffffffffULL,
                                    0xbff0000000000000ULL,
                                    0x3ff8000000000000ULL,
                                    0x7fefffffffffffffULL,
                                    0xffefffffffffffffULL,
                                    0x7ff0000000000000ULL,
                                    0xfff0000000000000ULL,
                                    0x7ff0123456789abcULL,
                                    0xfff8abcdef012345ULL};
  std::minstd_rand random;
  const auto word = [&] {
    const uint32_t high = uint32_t(random() >> 15);
    return (high << 16) | uint32_t(random() >> 15);
  };
  std::array<uint64_t, 64> raw;
  for (auto &bits : raw) {
    const uint64_t high = word();
    bits = (high << 32) | word();
  }
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    const bool cdna =
        (family.arch <= rocjitsu::IsaFamily::Cdna4) || family.arch == rocjitsu::IsaFamily::Cdna5;
    const bool cdna5_family = family.arch == rocjitsu::IsaFamily::Cdna5;
    const bool gfx1150 = family.arch == rocjitsu::IsaFamily::Rdna3_5;
    for (int lanes : {32, 64}) {
      if ((cdna && !cdna5_family && lanes == 32) || (cdna5_family && lanes == 64)) {
        continue;
      }
      for (bool scalar : {true, false}) {
        util::set_force_scalar_for_testing(scalar);
        for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
          ComputeUnitCore::Config config{};
          config.arch = family.arch;
          config.arithmetic = {semantics};
          config.num_wf_slots = 1;
          if (cdna5_family) {
            config.target = ROCJITSU_CODE_TARGET_GFX1251;
          }
          config.sgprs_per_wf = 104;
          config.vgprs_per_wf = 8;
          config.lds_size_kb = 64;
          GpuMemory memory("exponent_memory");
          L2Cache cache("exponent_cache");
          auto cu = ComputeUnitCore::create("exponent", config, &memory, &cache);
          auto *wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
          ASSERT_NE(wave, nullptr);
          auto decoder = Decoder::create(family.arch);
          auto wide_decoder = cdna5_family ? Decoder::create(default_isa_target_registry(),
                                                             ROCJITSU_CODE_TARGET_GFX1251)
                                           : nullptr;
          const auto base = wave->vgpr_alloc().base;
          for (int width : {32, 64}) {
            const bool wide = width == 64;
            auto input = raw;
            for (int lane = 0; lane < 16; ++lane) {
              input[lane] = wide ? special64[lane] : special32[lane];
            }
            for (const auto &sample : scenarios) {
              const bool sdwa = sample.form == Form::Sdwa;
              const bool vop3_dpp =
                  sample.form == Form::Vop3Dpp || sample.form == Form::SuppressedVop3Dpp;
              const bool dpp =
                  sample.form == Form::Dpp || sample.form == Form::SuppressedDpp || vop3_dpp;
              const bool suppressed =
                  sample.form == Form::SuppressedDpp || sample.form == Form::SuppressedVop3Dpp;
              const bool vop3 =
                  sample.form == Form::Vop3 || sample.form == Form::Scalar || vop3_dpp;
              if ((sdwa && (wide || cdna5_family || gfx1150)) || (dpp && wide && !cdna5_family) ||
                  (vop3_dpp && !cdna5_family && !gfx1150)) {
                continue;
              }
              for (int aliasing_offset : {0, 1, 3}) {
                SCOPED_TRACE(::testing::Message()
                             << family.arch << ':' << width << ':' << lanes << ':' << scalar << ':'
                             << int(semantics) << ':' << int(sample.form) << ':' << sample.mode
                             << ':' << aliasing_offset);
                const uint64_t all = lanes == 64 ? ~0ULL : 0xffffffffULL;
                const uint64_t exec_mask = all & 0x800000038000ffffULL;
                wave->set_exec(dpp ? all : exec_mask);
                wave->set_vgpr_write_mask(dpp ? exec_mask : all);
                wave->set_mode_raw(sample.mode);
                wave->set_trapsts(0x100);
                cu->write_sgpr(wave->sgpr_alloc().base, wide ? 0 : 0x3fc00000);
                cu->write_sgpr(wave->sgpr_alloc().base + 1, 0x3ff80000);
                for (int lane = 0; lane < lanes; ++lane) {
                  cu->write_vgpr(base + aliasing_offset, lane, 0xdeadbeef);
                  cu->write_vgpr(base, lane, uint32_t(input[lane]));
                  cu->write_vgpr(base + 1, lane, uint32_t(input[lane] >> 32));
                }
                const uint32_t source = sample.form == Form::Scalar ? 0 : 256;
                std::array<uint32_t, 3> words =
                    vop3 ? std::array<uint32_t, 3>{(cdna && !cdna5_family ? 0xd0000000U
                                                                          : 0xd4000000U) |
                                                       (uint32_t(family.vop3[wide]) << 16) |
                                                       uint32_t(aliasing_offset) |
                                                       (sample.modifiers ? 0x8100U : 0),
                                                   source | (sample.modifiers ? 0x38000000U : 0)}
                         : std::array<uint32_t, 3>{0x7e000000U | (uint32_t(aliasing_offset) << 17) |
                                                       (uint32_t(family.vop1[wide]) << 9) | source,
                                                   0};
                if (sdwa) {
                  // WORD_1 input/output, preserving other destination bytes; OMOD is ignored.
                  words[0] = (words[0] & ~0x1ffU) | 249;
                  words[1] = 0x00055500;
                } else if (dpp) {
                  words[vop3 ? 1 : 0] = (words[vop3 ? 1 : 0] & ~0x1ffU) | 250;
                  words[vop3 ? 2 : 1] =
                      (suppressed ? 0x0f000000U : 0xff000000U) | (wide ? 0x15100U : 0xb100U);
                }
                if (dpp && wide) {
                  EXPECT_TRUE(decode_fails(*decoder, words.data()));
                }
                std::unique_ptr<Instruction> instruction(
                    decode_valid(dpp && wide ? *wide_decoder : *decoder, words.data()));
                ASSERT_NE(instruction, nullptr);
                ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
                EXPECT_EQ(wave->trapsts(), 0x100U);
                for (int lane = 0; lane < lanes; ++lane) {
                  uint32_t expected = aliasing_offset == 0   ? uint32_t(input[lane])
                                      : aliasing_offset == 1 ? uint32_t(input[lane] >> 32)
                                                             : 0xdeadbeef;
                  if ((exec_mask >> lane & 1) && !suppressed) {
                    const int source_lane = dpp ? (wide ? (lane & ~15) + 1 : lane ^ 1) : lane;
                    uint64_t bits = sample.form == Form::Scalar
                                        ? (wide ? 0x3ff8000000000000ULL : 0x3fc00000ULL)
                                        : input[source_lane];
                    if (sdwa) {
                      bits = uint32_t(bits) >> 16;
                    }
                    if (!wide) {
                      bits = uint32_t(bits);
                    }
                    const bool flush =
                        !((wide ? wave->fp_denorm_mode_f16_f64() : wave->fp_denorm_mode_f32()) & 1);
                    const uint32_t value = reference(bits, wide, flush);
                    expected = sdwa ? (expected & 0xffff) | (value << 16) : value;
                  }
                  EXPECT_EQ(cu->read_vgpr(base + aliasing_offset, lane), expected)
                      << "lane=" << lane;
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
