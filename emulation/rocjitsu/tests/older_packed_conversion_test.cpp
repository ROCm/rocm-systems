// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded older-target FP32 packing: inherited RTZ/saturating narrowing, guest
// controls, source staging, destination placement, aliases and inactive lanes.

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

#include <array>
#include <bit>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <random>

namespace rocjitsu::amdgpu {
namespace {

// Check RTZ packing against separately tested scalar narrowing; integer outputs
// retain independent bounded references.
uint32_t reference(int operation, uint32_t a, uint32_t b, const Wavefront &wave, unsigned absolute,
                   unsigned negate) {
  if (operation) {
    const auto narrow = [operation](uint32_t raw, bool absolute, bool negate) -> uint32_t {
      if (absolute) {
        raw &= 0x7fffffff;
      }
      if (negate) {
        raw ^= 0x80000000;
      }
      float value = std::bit_cast<float>(raw);
      if (value != value) {
        return 0;
      }
      if (operation == 2) {
        return value < 0 ? 0 : value >= 65536 ? 65535 : uint16_t(value);
      }
      return value < -32768 ? 0x8000 : value >= 32768 ? 0x7fff : uint16_t(int16_t(value));
    };
    return narrow(a, absolute & 1, negate & 1) | (narrow(b, absolute & 2, negate & 2) << 16);
  }
  uint64_t flags = RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_ROUND_ZERO;
  flags |= (wave.fp_denorm_mode_f32() & 1) ? 0 : RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  flags |= (wave.fp_denorm_mode_f16_f64() & 2) ? 0 : RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  flags |= wave.fp16_ovfl() ? RJ_ARITH_FP16_OVFL : 0;
  const auto narrow = [&](uint32_t raw, unsigned operand) {
    const uint64_t modifiers = ((absolute >> operand) & 1 ? RJ_ARITH_ALU_ABS_A : 0) |
                               ((negate >> operand) & 1 ? RJ_ARITH_ALU_NEG_A : 0);
    return uint32_t(rocjitsu::test::conversion_binding(rj_arith_v_cvt_f16_f32_w32_gfx942, flags,
                                                       modifiers, raw));
  };
  return narrow(a, 0) | (narrow(b, 1) << 16);
}

// Restore the test dispatch override on every exit path.
struct ScalarGuard {
  bool saved = util::force_scalar();
  ~ScalarGuard() { util::set_force_scalar_for_testing(saved); }
};

// Exercise each available encoding, including source-before-destination aliasing.
TEST(OlderPackedConversion, Arithmetic) {
  ScalarGuard guard;
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    int vop2;
    uint16_t vop3;
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1, &cdna1::execution_backend(), -1, cdna1::kVCvtPkrtzF16F32Vop3},
      {rocjitsu::IsaFamily::Cdna2, &cdna2::execution_backend(), -1, cdna2::kVCvtPkrtzF16F32Vop3},
      {rocjitsu::IsaFamily::Rdna1, &rdna1::execution_backend(), rdna1::kVCvtPkrtzF16F32Vop2,
       rdna1::kVCvtPkrtzF16F32Vop3},
      {rocjitsu::IsaFamily::Rdna2, &rdna2::execution_backend(), rdna2::kVCvtPkrtzF16F32Vop2,
       rdna2::kVCvtPkrtzF16F32Vop3},
      {rocjitsu::IsaFamily::Rdna3_5, &rdna3_5::execution_backend(), rdna3_5::kVCvtPkRtzF16F32Vop2,
       rdna3_5::kVCvtPkRtzF16F32Vop3},
  };
  enum class Form { Vop3, Vop2, Sdwa, Dpp, SuppressedDpp, ScalarSource, HighRegister };
  struct Scenario {
    Form form = Form::Vop3;
    uint32_t mode = 0xf0;
    unsigned absolute = 0, negate = 0, omod = 0;
    bool clamp = false;
  };
  const Scenario scenarios[] = {
      {},
      {.mode = 0xf3},
      {.mode = 0},
      {.mode = 0xf0 | Wavefront::FP16_OVFL_BIT},
      {.absolute = 2, .negate = 1},
      {.absolute = 1, .negate = 2, .omod = 3, .clamp = true},
      {.form = Form::Vop2},
      {.form = Form::Sdwa},
      {.form = Form::Dpp},
      {.form = Form::SuppressedDpp},
      {.form = Form::ScalarSource},
      {.form = Form::HighRegister},
  };
  constexpr uint32_t special[] = {0x7fc12345, 0x7f800000, 0xff800000, 0x80000000,
                                  1,          0x80000001, 0x477fff00, 0x47800000,
                                  0x38800000, 0x387fe000, 0x3f801000, 0xbf801000,
                                  0x33800000, 0x33000000, 0x7f800001, 0};
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
          GpuMemory memory("rtz_memory");
          L2Cache cache("rtz_cache");
          cache.set_backing_memory(&memory);
          std::unique_ptr<ComputeUnitCore> cu;
          ASSERT_NO_FATAL_FAILURE(
              rocjitsu::test::initialize_compute_unit(cu, "rtz", config, &memory, &cache));
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
          std::array<uint32_t, 64> a{}, b{};
          for (int lane = 0; lane < lanes; ++lane) {
            a[lane] = lane < 16 ? special[lane] : random_word();
            b[lane] = lane < 16 ? special[15 - lane] : random_word();
          }
          // gfx1150 also has packed saturating I16/U16 conversions; older profiles have RTZ only.
          for (int operation : {0, 1, 2}) {
            if (operation && family.arch != rocjitsu::IsaFamily::Rdna3_5) {
              continue;
            }
            const uint16_t opcode = operation == 1   ? rdna3_5::kVCvtPkI16F32Vop3
                                    : operation == 2 ? rdna3_5::kVCvtPkU16F32Vop3
                                                     : family.vop3;
            for (const auto &sample : scenarios) {
              const bool vop2 = sample.form == Form::Vop2 || sample.form == Form::Sdwa ||
                                sample.form == Form::Dpp || sample.form == Form::SuppressedDpp;
              const bool sdwa = sample.form == Form::Sdwa;
              const bool dpp = sample.form == Form::Dpp || sample.form == Form::SuppressedDpp;
              const bool sgpr = sample.form == Form::ScalarSource;
              const uint32_t source_a = sample.form == Form::HighRegister ? 128 : 0;
              const uint32_t source_b = source_a + 1;
              if (vop2 && (operation || family.vop2 < 0 ||
                           (sdwa && family.arch == rocjitsu::IsaFamily::Rdna3_5))) {
                continue;
              }
              for (int aliasing_offset : {0, 1, 2}) {
                SCOPED_TRACE(::testing::Message()
                             << operation << ':' << family.arch << ':' << lanes << ':' << scalar
                             << ':' << int(semantics) << ':' << int(sample.form) << ':'
                             << sample.mode << ':' << aliasing_offset);
                const uint32_t destination = source_a + aliasing_offset;
                const uint64_t all = lanes == 64 ? ~0ULL : 0xffffffffULL;
                const uint64_t exec_mask = all & 0x800000038000ffffULL;
                wave->set_exec(dpp ? all : exec_mask);
                wave->set_vgpr_write_mask(dpp ? exec_mask : all);
                wave->set_mode_raw(sample.mode);
                wave->set_trapsts(0x100);
                cu->write_sgpr(wave->sgpr_alloc().base, 0x3f801fff);
                for (int lane = 0; lane < lanes; ++lane) {
                  cu->write_vgpr(base + destination, lane, 0xdeadbeef);
                  cu->write_vgpr(base + source_a, lane, a[lane]);
                  cu->write_vgpr(base + source_b, lane, b[lane]);
                }
                const uint32_t source = sgpr ? 0 : 256 + source_a;
                std::array<uint32_t, 2> words =
                    vop2 ? std::array<uint32_t, 2>{(uint32_t(family.vop2) << 25) |
                                                       (destination << 17) | (source_b << 9) |
                                                       source,
                                                   0}
                         : std::array<uint32_t, 2>{(cdna ? 0xd0000000U : 0xd4000000U) |
                                                       (uint32_t(opcode) << 16) | destination |
                                                       (sample.absolute << 8) |
                                                       (sample.clamp ? 1U << 15 : 0U),
                                                   source | ((256 + source_b) << 9) |
                                                       (sample.omod << 27) | (sample.negate << 29)};
                if (sdwa) {
                  // DWORD sources; place the low result half in WORD_1, preserving WORD_0.
                  words[0] = (words[0] & ~0x1ffU) | 249;
                  words[1] = 0x06061500;
                } else if (dpp) {
                  // Swap neighboring lanes of A; B remains lane-local.
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
                    expected =
                        reference(operation, sgpr ? 0x3f801fff : a[dpp ? lane ^ 1 : lane], b[lane],
                                  *wave, vop2 ? 0 : sample.absolute, vop2 ? 0 : sample.negate);
                    if (sdwa) {
                      expected = (old & 0xffff) | (expected << 16);
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
