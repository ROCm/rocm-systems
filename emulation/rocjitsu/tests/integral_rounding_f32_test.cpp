// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded FP32 integral rounding: inherited NaNs, guest controls, aliases and stores.
#include "decode_test_util.h"
#include "fp_reference.h"
#include "integral_rounding_reference.h"
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
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/execution_backend.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
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

// Restore the test-only CPU dispatch override even after an assertion fails.
struct ScalarGuard {
  bool saved = util::force_scalar();
  ~ScalarGuard() { util::set_force_scalar_for_testing(saved); }
};

// Check inherited arithmetic, guest policies and write footprints through decoded instructions.
TEST(IntegralRoundingF32, Arithmetic) {
  ScalarGuard guard;
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<uint16_t, 4> vop1, vop3;
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {cdna1::kVTruncF32Vop1, cdna1::kVCeilF32Vop1, cdna1::kVRndneF32Vop1, cdna1::kVFloorF32Vop1},
       {cdna1::kVTruncF32Vop3, cdna1::kVCeilF32Vop3, cdna1::kVRndneF32Vop3, cdna1::kVFloorF32Vop3}},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {cdna2::kVTruncF32Vop1, cdna2::kVCeilF32Vop1, cdna2::kVRndneF32Vop1, cdna2::kVFloorF32Vop1},
       {cdna2::kVTruncF32Vop3, cdna2::kVCeilF32Vop3, cdna2::kVRndneF32Vop3, cdna2::kVFloorF32Vop3}},
      {rocjitsu::IsaFamily::Cdna3,
       &cdna3::execution_backend(),
       {cdna3::kVTruncF32Vop1, cdna3::kVCeilF32Vop1, cdna3::kVRndneF32Vop1, cdna3::kVFloorF32Vop1},
       {cdna3::kVTruncF32Vop3, cdna3::kVCeilF32Vop3, cdna3::kVRndneF32Vop3, cdna3::kVFloorF32Vop3}},
      {rocjitsu::IsaFamily::Cdna4,
       &cdna4::execution_backend(),
       {cdna4::kVTruncF32Vop1, cdna4::kVCeilF32Vop1, cdna4::kVRndneF32Vop1, cdna4::kVFloorF32Vop1},
       {cdna4::kVTruncF32Vop3, cdna4::kVCeilF32Vop3, cdna4::kVRndneF32Vop3, cdna4::kVFloorF32Vop3}},
      {rocjitsu::IsaFamily::Cdna5,
       &cdna5::execution_backend(),
       {cdna5::kVTruncF32Vop1, cdna5::kVCeilF32Vop1, cdna5::kVRndneF32Vop1, cdna5::kVFloorF32Vop1},
       {cdna5::kVTruncF32Vop3, cdna5::kVCeilF32Vop3, cdna5::kVRndneF32Vop3, cdna5::kVFloorF32Vop3}},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {rdna1::kVTruncF32Vop1, rdna1::kVCeilF32Vop1, rdna1::kVRndneF32Vop1, rdna1::kVFloorF32Vop1},
       {rdna1::kVTruncF32Vop3, rdna1::kVCeilF32Vop3, rdna1::kVRndneF32Vop3, rdna1::kVFloorF32Vop3}},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {rdna2::kVTruncF32Vop1, rdna2::kVCeilF32Vop1, rdna2::kVRndneF32Vop1, rdna2::kVFloorF32Vop1},
       {rdna2::kVTruncF32Vop3, rdna2::kVCeilF32Vop3, rdna2::kVRndneF32Vop3, rdna2::kVFloorF32Vop3}},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {rdna3_5::kVTruncF32Vop1, rdna3_5::kVCeilF32Vop1, rdna3_5::kVRndneF32Vop1,
        rdna3_5::kVFloorF32Vop1},
       {rdna3_5::kVTruncF32Vop3, rdna3_5::kVCeilF32Vop3, rdna3_5::kVRndneF32Vop3,
        rdna3_5::kVFloorF32Vop3}},
  };
  enum class Form { Vop1, Vop3, Scalar, Sdwa, Dpp, SuppressedDpp };
  struct Scenario {
    Form form = Form::Vop3;
    uint32_t mode = 0;
    unsigned omod = 0;
    bool clamp = false, absolute = false, negate = false;
  };
  const Scenario scenarios[] = {
      {Form::Vop1},
      {Form::Vop1, 0xf0},
      {},
      {.omod = 1},
      {.mode = 1, .omod = 2},
      {.mode = 3, .omod = 1},
      {.mode = 2, .omod = 1},
      {.omod = 3},
      {.mode = 0x200, .omod = 1},
      {.mode = 0x20, .omod = 2},
      {.clamp = true},
      {.mode = 0x100, .clamp = true},
      {.absolute = true, .negate = true},
      {Form::Scalar},
      {Form::Sdwa},
      {.form = Form::Sdwa, .mode = 0, .omod = 1},
      {.form = Form::Sdwa, .mode = 1, .omod = 2},
      {.form = Form::Sdwa, .mode = 2, .omod = 3},
      {.form = Form::Sdwa, .mode = 3, .omod = 2, .clamp = true},
      {.form = Form::Sdwa, .mode = 0x200, .omod = 1},
      {.form = Form::Sdwa, .mode = 0x20, .omod = 2},
      {Form::Dpp},
      {Form::SuppressedDpp},
  };
  constexpr uint32_t special[] = {
      0x3fc00000, 0xbfc00000, 0x40200000, 0xc0200000, 1, 0x80000001, 0x7f812345, 0xff812345,
      0x7fcabcde, 0x7f800000, 0xff800000, 0x80000000, 0, 0x3f7fffff, 0xbf7fffff, 0x7f7fffff};
  std::minstd_rand random;
  std::array<uint32_t, 64> input;
  for (auto &word : input) {
    const uint32_t high = uint32_t(random() >> 15);
    word = (high << 16) | uint32_t(random() >> 15);
  }
  for (int i = 0; i < 16; ++i) {
    input[i] = special[i];
  }
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    const bool cdna =
        family.arch >= rocjitsu::IsaFamily::Cdna1 &&
        (family.arch <= rocjitsu::IsaFamily::Cdna4 || family.arch == rocjitsu::IsaFamily::Cdna5);
    const bool gfx1250 = family.arch == rocjitsu::IsaFamily::Cdna5;
    const bool gfx1150 = family.arch == rocjitsu::IsaFamily::Rdna3_5;
    for (int lanes : {32, 64}) {
      if ((cdna && !gfx1250 && lanes == 32) || (gfx1250 && lanes == 64)) {
        continue;
      }
      for (bool scalar : {true, false}) {
        util::set_force_scalar_for_testing(scalar);
        for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
          ComputeUnitCore::Config config{};
          config.arch = family.arch;
          config.arithmetic = {semantics};
          config.num_wf_slots = 1;
          config.sgprs_per_wf = 104;
          config.vgprs_per_wf = 8;
          config.lds_size_kb = 64;
          GpuMemory memory("rounding_memory");
          L2Cache cache("rounding_cache");
          auto cu = ComputeUnitCore::create("rounding", config, &memory, &cache);
          auto *wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
          ASSERT_NE(wave, nullptr);
          auto decoder = Decoder::create(family.arch);
          const auto base = wave->vgpr_alloc().base;
          for (int op = 0; op < 4; ++op) {
            for (const auto &sample : scenarios) {
              const bool sdwa = sample.form == Form::Sdwa;
              if (sdwa && (gfx1150 || gfx1250)) {
                continue;
              }
              const bool dpp = sample.form == Form::Dpp || sample.form == Form::SuppressedDpp;
              const bool vop3 = sample.form == Form::Vop3 || sample.form == Form::Scalar;
              for (int aliasing_offset : {0, 2}) {
                SCOPED_TRACE(::testing::Message()
                             << family.arch << ':' << lanes << ':' << scalar << ':'
                             << int(semantics) << ':' << op << ':' << int(sample.form) << ':'
                             << sample.mode << ':' << aliasing_offset);
                const uint64_t all = lanes == 64 ? ~0ULL : 0xffffffffULL;
                const uint64_t exec_mask = all & 0x800000038000ffffULL;
                wave->set_exec(dpp ? all : exec_mask);
                wave->set_vgpr_write_mask(dpp ? exec_mask : all);
                wave->set_mode_raw(sample.mode);
                wave->set_trapsts(0x100);
                cu->write_sgpr(wave->sgpr_alloc().base, 0xbfc00000);
                for (int lane = 0; lane < lanes; ++lane) {
                  cu->write_vgpr(base + aliasing_offset, lane, 0xdeadbeef);
                  cu->write_vgpr(base, lane, input[lane]);
                }
                const uint32_t source = sample.form == Form::Scalar ? 0 : 256;
                std::array<uint32_t, 2> words =
                    vop3 ? std::array<uint32_t, 2>{(cdna && !gfx1250 ? 0xd0000000U : 0xd4000000U) |
                                                       (uint32_t(family.vop3[op]) << 16) |
                                                       uint32_t(aliasing_offset) |
                                                       (sample.absolute ? 1U << 8 : 0U) |
                                                       (sample.clamp ? 1U << 15 : 0U),
                                                   source | (sample.omod << 27) |
                                                       (sample.negate ? 1U << 29 : 0U)}
                         : std::array<uint32_t, 2>{0x7e000000U | (uint32_t(aliasing_offset) << 17) |
                                                       (uint32_t(family.vop1[op]) << 9) | source,
                                                   0};
                if (sdwa) {
                  // DWORD input, WORD_1 output with other bytes preserved.
                  words[0] = (words[0] & ~0x1ffU) | 249;
                  words[1] = 0x00061500 | (sample.omod << 14) | (sample.clamp ? 1U << 13 : 0U);
                } else if (dpp) {
                  words[0] = (words[0] & ~0x1ffU) | 250;
                  // quad_perm:[1,0,3,2], with an optional all-row write suppression.
                  words[1] = sample.form == Form::SuppressedDpp ? 0x0f00b100 : 0xff00b100;
                }
                std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
                ASSERT_NE(instruction, nullptr);
                ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
                EXPECT_EQ(wave->trapsts(), 0x100U);
                for (int lane = 0; lane < lanes; ++lane) {
                  uint32_t expected = aliasing_offset ? 0xdeadbeef : input[lane];
                  const bool written =
                      (exec_mask >> lane & 1) && sample.form != Form::SuppressedDpp;
                  if (written) {
                    uint32_t raw =
                        sample.form == Form::Scalar ? 0xbfc00000 : input[dpp ? lane ^ 1 : lane];
                    if (vop3 && sample.absolute) {
                      raw &= 0x7fffffff;
                    }
                    if (vop3 && sample.negate) {
                      raw ^= 0x80000000;
                    }
                    if ((op == 1 || op == 3) && !(wave->fp_denorm_mode_f32() & 1) &&
                        (raw & 0x7fffffff) < 0x800000) {
                      raw &= 0x80000000;
                    }
                    const bool modern = family.arch == rocjitsu::IsaFamily::Rdna4 ||
                                        family.arch == rocjitsu::IsaFamily::Cdna5;
                    const bool scale =
                        modern || (!wave->ieee_mode() && !(wave->fp_denorm_mode_f32() & 2));
                    const rocjitsu::test::FloatingOutput output{
                        scale ? sample.omod : 0, wave->fp_round_mode_f32(), sample.clamp,
                        modern || wave->dx10_clamp()};
                    const uint32_t result = std::bit_cast<uint32_t>(rocjitsu::test::finish_floating(
                        integral_reference::evaluate(op, std::bit_cast<float>(raw)), output));
                    expected = sdwa ? (expected & 0xffff) | (result << 16) : result;
                  }
                  const uint32_t actual = cu->read_vgpr(base + aliasing_offset, lane);
                  if (semantics == ArithmeticSemantics::Loose && written && !sdwa) {
                    const float got = std::bit_cast<float>(actual),
                                want = std::bit_cast<float>(expected);
                    if (std::isnan(want)) {
                      EXPECT_TRUE(std::isnan(got));
                    } else if (std::isinf(want)) {
                      EXPECT_EQ(got, want);
                    } else {
                      EXPECT_LE(std::abs(double(got) - want), 1e-6 + 1e-6 * std::abs(want));
                    }
                  } else {
                    EXPECT_EQ(actual, expected) << "lane=" << lane;
                  }
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
