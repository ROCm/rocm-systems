// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded FP64 integral rounding: inherited NaNs, guest controls, aliases and stores.
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

// Restore the test-only CPU dispatch override even after an assertion fails.
struct ScalarGuard {
  bool saved = util::force_scalar();
  ~ScalarGuard() { util::set_force_scalar_for_testing(saved); }
};

// Check inherited arithmetic, guest policies and write footprints through decoded instructions.
TEST(IntegralRoundingF64, Arithmetic) {
  ScalarGuard guard;
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<uint16_t, 4> vop1, vop3;
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {cdna1::kVTruncF64Vop1, cdna1::kVCeilF64Vop1, cdna1::kVRndneF64Vop1, cdna1::kVFloorF64Vop1},
       {cdna1::kVTruncF64Vop3, cdna1::kVCeilF64Vop3, cdna1::kVRndneF64Vop3, cdna1::kVFloorF64Vop3}},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {cdna2::kVTruncF64Vop1, cdna2::kVCeilF64Vop1, cdna2::kVRndneF64Vop1, cdna2::kVFloorF64Vop1},
       {cdna2::kVTruncF64Vop3, cdna2::kVCeilF64Vop3, cdna2::kVRndneF64Vop3, cdna2::kVFloorF64Vop3}},
      {rocjitsu::IsaFamily::Cdna3,
       &cdna3::execution_backend(),
       {cdna3::kVTruncF64Vop1, cdna3::kVCeilF64Vop1, cdna3::kVRndneF64Vop1, cdna3::kVFloorF64Vop1},
       {cdna3::kVTruncF64Vop3, cdna3::kVCeilF64Vop3, cdna3::kVRndneF64Vop3, cdna3::kVFloorF64Vop3}},
      {rocjitsu::IsaFamily::Cdna4,
       &cdna4::execution_backend(),
       {cdna4::kVTruncF64Vop1, cdna4::kVCeilF64Vop1, cdna4::kVRndneF64Vop1, cdna4::kVFloorF64Vop1},
       {cdna4::kVTruncF64Vop3, cdna4::kVCeilF64Vop3, cdna4::kVRndneF64Vop3, cdna4::kVFloorF64Vop3}},
      {rocjitsu::IsaFamily::Cdna5,
       &cdna5::execution_backend(),
       {cdna5::kVTruncF64Vop1, cdna5::kVCeilF64Vop1, cdna5::kVRndneF64Vop1, cdna5::kVFloorF64Vop1},
       {cdna5::kVTruncF64Vop3, cdna5::kVCeilF64Vop3, cdna5::kVRndneF64Vop3, cdna5::kVFloorF64Vop3}},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {rdna1::kVTruncF64Vop1, rdna1::kVCeilF64Vop1, rdna1::kVRndneF64Vop1, rdna1::kVFloorF64Vop1},
       {rdna1::kVTruncF64Vop3, rdna1::kVCeilF64Vop3, rdna1::kVRndneF64Vop3, rdna1::kVFloorF64Vop3}},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {rdna2::kVTruncF64Vop1, rdna2::kVCeilF64Vop1, rdna2::kVRndneF64Vop1, rdna2::kVFloorF64Vop1},
       {rdna2::kVTruncF64Vop3, rdna2::kVCeilF64Vop3, rdna2::kVRndneF64Vop3, rdna2::kVFloorF64Vop3}},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {rdna3_5::kVTruncF64Vop1, rdna3_5::kVCeilF64Vop1, rdna3_5::kVRndneF64Vop1,
        rdna3_5::kVFloorF64Vop1},
       {rdna3_5::kVTruncF64Vop3, rdna3_5::kVCeilF64Vop3, rdna3_5::kVRndneF64Vop3,
        rdna3_5::kVFloorF64Vop3}},
  };
  enum class Form { Vop1, Vop3, Scalar, Dpp, SuppressedDpp, Vop3Dpp, SuppressedVop3Dpp };
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
      {.mode = 4, .omod = 2},
      {.mode = 12, .omod = 1},
      {.mode = 8, .omod = 1},
      {.omod = 3},
      {.mode = 0x200, .omod = 1},
      {.mode = 0x80, .omod = 2},
      {.clamp = true},
      {.mode = 0x100, .clamp = true},
      {.absolute = true, .negate = true},
      {Form::Scalar},
      {Form::Dpp},
      {Form::SuppressedDpp},
      {Form::Vop3Dpp},
      {Form::SuppressedVop3Dpp},
  };
  constexpr uint64_t special[] = {0x3ff8000000000000ULL,
                                  0xbff8000000000000ULL,
                                  0x4004000000000000ULL,
                                  0xc004000000000000ULL,
                                  1,
                                  0x8000000000000001ULL,
                                  0x7ff0123456789abcULL,
                                  0xfff0123456789abcULL,
                                  0x7ff8abcdef012345ULL,
                                  0x7ff0000000000000ULL,
                                  0xfff0000000000000ULL,
                                  0x8000000000000000ULL,
                                  0,
                                  0x3fefffffffffffffULL,
                                  0xbfefffffffffffffULL,
                                  0x7fefffffffffffffULL};
  std::minstd_rand random;
  const auto random_word = [&] {
    const uint32_t high = uint32_t(random() >> 15);
    return (high << 16) | uint32_t(random() >> 15);
  };
  std::array<uint64_t, 64> input;
  for (auto &bits : input) {
    const uint64_t high = random_word();
    bits = (high << 32) | random_word();
  }
  for (int lane = 0; lane < 16; ++lane) {
    input[lane] = special[lane];
  }
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    const bool cdna =
        family.arch >= rocjitsu::IsaFamily::Cdna1 &&
        (family.arch <= rocjitsu::IsaFamily::Cdna4 || family.arch == rocjitsu::IsaFamily::Cdna5);
    const bool cdna5_family = family.arch == rocjitsu::IsaFamily::Cdna5;
    for (int lanes : {32, 64}) {
      if ((cdna && !cdna5_family && lanes == 32) || (cdna5_family && lanes == 64)) {
        continue;
      }
      for (bool scalar : {true, false}) {
        util::set_force_scalar_for_testing(scalar);
        for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
          ComputeUnitCore::Config config{};
          config.arch = family.arch;
          if (cdna5_family) {
            config.target = ROCJITSU_CODE_TARGET_GFX1251;
          }
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
          auto wide_decoder = cdna5_family ? Decoder::create(default_isa_target_registry(),
                                                             ROCJITSU_CODE_TARGET_GFX1251)
                                           : nullptr;
          const auto base = wave->vgpr_alloc().base;
          for (int op = 0; op < 4; ++op) {
            for (const auto &sample : scenarios) {
              const bool vop3_dpp =
                  sample.form == Form::Vop3Dpp || sample.form == Form::SuppressedVop3Dpp;
              const bool dpp =
                  sample.form == Form::Dpp || sample.form == Form::SuppressedDpp || vop3_dpp;
              const bool suppressed =
                  sample.form == Form::SuppressedDpp || sample.form == Form::SuppressedVop3Dpp;
              if (dpp && !cdna5_family) {
                continue;
              }
              const bool vop3 =
                  sample.form == Form::Vop3 || sample.form == Form::Scalar || vop3_dpp;
              for (int aliasing_offset : {0, 1, 3}) {
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
                cu->write_sgpr(wave->sgpr_alloc().base, 0);
                cu->write_sgpr(wave->sgpr_alloc().base + 1, 0xbff80000);
                for (int lane = 0; lane < lanes; ++lane) {
                  for (int word = 0; word < 2; ++word) {
                    cu->write_vgpr(base + aliasing_offset + word, lane, 0xdeadbeef);
                  }
                  for (int word = 0; word < 2; ++word) {
                    cu->write_vgpr(base + word, lane, uint32_t(input[lane] >> (word * 32)));
                  }
                }
                const uint32_t source = sample.form == Form::Scalar ? 0 : 256;
                std::array<uint32_t, 3> words =
                    vop3 ? std::array<uint32_t, 3>{(cdna && !cdna5_family ? 0xd0000000U
                                                                          : 0xd4000000U) |
                                                       (uint32_t(family.vop3[op]) << 16) |
                                                       uint32_t(aliasing_offset) |
                                                       (sample.absolute ? 1U << 8 : 0U) |
                                                       (sample.clamp ? 1U << 15 : 0U),
                                                   source | (sample.omod << 27) |
                                                       (sample.negate ? 1U << 29 : 0U)}
                         : std::array<uint32_t, 3>{0x7e000000U | (uint32_t(aliasing_offset) << 17) |
                                                       (uint32_t(family.vop1[op]) << 9) | source,
                                                   0};
                if (dpp) {
                  words[vop3 ? 1 : 0] = (words[vop3 ? 1 : 0] & ~0x1ffU) | 250;
                  // gfx1251 row-select lane1, transporting both source VGPR words.
                  words[vop3 ? 2 : 1] = suppressed ? 0x0f015100 : 0xff015100;
                }
                if (dpp) {
                  // Wide DPP requires gfx1251; ordinary gfx1250 decoding must reject it.
                  EXPECT_TRUE(decode_fails(*decoder, words.data()));
                }
                std::unique_ptr<Instruction> instruction(
                    decode_valid(dpp ? *wide_decoder : *decoder, words.data()));
                ASSERT_NE(instruction, nullptr);
                ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
                EXPECT_EQ(wave->trapsts(), 0x100U);
                for (int lane = 0; lane < lanes; ++lane) {
                  uint64_t expected = aliasing_offset == 0 ? input[lane]
                                      : aliasing_offset == 1
                                          ? (0xdeadbeef00000000ULL | uint32_t(input[lane] >> 32))
                                          : 0xdeadbeefdeadbeefULL;
                  const bool written = (exec_mask >> lane & 1) && !suppressed;
                  if (written) {
                    uint64_t raw = sample.form == Form::Scalar
                                       ? 0xbff8000000000000ULL
                                       : input[dpp ? (lane & ~15) + 1 : lane];
                    if (vop3 && sample.absolute) {
                      raw &= 0x7fffffffffffffffULL;
                    }
                    if (vop3 && sample.negate) {
                      raw ^= 0x8000000000000000ULL;
                    }
                    if ((op == 1 || op == 3) && !(wave->fp_denorm_mode_f16_f64() & 1) &&
                        (raw & 0x7fffffffffffffffULL) < 0x0010000000000000ULL) {
                      raw &= 0x8000000000000000ULL;
                    }
                    const bool modern = family.arch == rocjitsu::IsaFamily::Rdna4 ||
                                        family.arch == rocjitsu::IsaFamily::Cdna5;
                    const bool scale =
                        modern || (!wave->ieee_mode() && !(wave->fp_denorm_mode_f16_f64() & 2));
                    const rocjitsu::test::FloatingOutput output{
                        scale ? sample.omod : 0, wave->fp_round_mode_f16_f64(), sample.clamp,
                        modern || wave->dx10_clamp()};
                    expected = std::bit_cast<uint64_t>(rocjitsu::test::finish_floating(
                        integral_reference::evaluate(op, std::bit_cast<double>(raw)), output));
                  }
                  const uint64_t actual =
                      cu->read_vgpr(base + aliasing_offset, lane) |
                      (uint64_t(cu->read_vgpr(base + aliasing_offset + 1, lane)) << 32);
                  if (semantics == ArithmeticSemantics::Loose && written) {
                    const double got = std::bit_cast<double>(actual),
                                 want = std::bit_cast<double>(expected);
                    if (std::isnan(want)) {
                      EXPECT_TRUE(std::isnan(got));
                    } else if (std::isinf(want)) {
                      EXPECT_EQ(got, want);
                    } else {
                      EXPECT_LE(std::abs(got - want), 1e-14 + 1e-14 * std::abs(want));
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
