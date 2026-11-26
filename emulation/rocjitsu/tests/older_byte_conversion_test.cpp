// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded older-target byte/nibble conversions: exact inherited arithmetic, guest
// controls, source staging, destination placement, aliases and inactive lanes.

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

// Byte extraction and a signed-nibble offset are exactly representable in FP32.
uint32_t reference(int operation, uint32_t raw, const Wavefront &wave, unsigned omod, bool clamp) {
  int value = operation < 4 ? int((raw >> (8 * operation)) & 0xffU) : int(raw & 0xfU);
  if (operation == 4 && value >= 8) {
    value -= 16;
  }
  float result = float(value) * (operation == 4 ? 0x1p-4f : 1.0f);
  constexpr float scale[] = {1, 2, 4, 0.5f};
  // These older targets ignore OMOD under IEEE or preserved output denormals.
  if (!wave.ieee_mode() && !(wave.fp_denorm_mode_f32() & 2)) {
    result *= scale[omod];
  }
  if (clamp) {
    result = result < 0 ? 0 : result > 1 ? 1 : result;
  }
  return std::bit_cast<uint32_t>(result);
}

// Restore the caller's test override even when an assertion exits the test.
struct ScalarGuard {
  bool saved = util::force_scalar();
  ~ScalarGuard() { util::set_force_scalar_for_testing(saved); }
};

// Exercise the same models through each real target's decoder and operand wrappers.
TEST(OlderByteConversion, Arithmetic) {
  ScalarGuard guard;
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<std::array<uint16_t, 2>, 5> opcodes;
  };
  // Four selected bytes, then the signed low-nibble offset.
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {{
           {cdna1::kVCvtF32Ubyte0Vop1, cdna1::kVCvtF32Ubyte0Vop3},
           {cdna1::kVCvtF32Ubyte1Vop1, cdna1::kVCvtF32Ubyte1Vop3},
           {cdna1::kVCvtF32Ubyte2Vop1, cdna1::kVCvtF32Ubyte2Vop3},
           {cdna1::kVCvtF32Ubyte3Vop1, cdna1::kVCvtF32Ubyte3Vop3},
           {cdna1::kVCvtOffF32I4Vop1, cdna1::kVCvtOffF32I4Vop3},
       }}},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {{
           {cdna2::kVCvtF32Ubyte0Vop1, cdna2::kVCvtF32Ubyte0Vop3},
           {cdna2::kVCvtF32Ubyte1Vop1, cdna2::kVCvtF32Ubyte1Vop3},
           {cdna2::kVCvtF32Ubyte2Vop1, cdna2::kVCvtF32Ubyte2Vop3},
           {cdna2::kVCvtF32Ubyte3Vop1, cdna2::kVCvtF32Ubyte3Vop3},
           {cdna2::kVCvtOffF32I4Vop1, cdna2::kVCvtOffF32I4Vop3},
       }}},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {{
           {rdna1::kVCvtF32Ubyte0Vop1, rdna1::kVCvtF32Ubyte0Vop3},
           {rdna1::kVCvtF32Ubyte1Vop1, rdna1::kVCvtF32Ubyte1Vop3},
           {rdna1::kVCvtF32Ubyte2Vop1, rdna1::kVCvtF32Ubyte2Vop3},
           {rdna1::kVCvtF32Ubyte3Vop1, rdna1::kVCvtF32Ubyte3Vop3},
           {rdna1::kVCvtOffF32I4Vop1, rdna1::kVCvtOffF32I4Vop3},
       }}},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {{
           {rdna2::kVCvtF32Ubyte0Vop1, rdna2::kVCvtF32Ubyte0Vop3},
           {rdna2::kVCvtF32Ubyte1Vop1, rdna2::kVCvtF32Ubyte1Vop3},
           {rdna2::kVCvtF32Ubyte2Vop1, rdna2::kVCvtF32Ubyte2Vop3},
           {rdna2::kVCvtF32Ubyte3Vop1, rdna2::kVCvtF32Ubyte3Vop3},
           {rdna2::kVCvtOffF32I4Vop1, rdna2::kVCvtOffF32I4Vop3},
       }}},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {{
           {rdna3_5::kVCvtF32Ubyte0Vop1, rdna3_5::kVCvtF32Ubyte0Vop3},
           {rdna3_5::kVCvtF32Ubyte1Vop1, rdna3_5::kVCvtF32Ubyte1Vop3},
           {rdna3_5::kVCvtF32Ubyte2Vop1, rdna3_5::kVCvtF32Ubyte2Vop3},
           {rdna3_5::kVCvtF32Ubyte3Vop1, rdna3_5::kVCvtF32Ubyte3Vop3},
           {rdna3_5::kVCvtOffF32I4Vop1, rdna3_5::kVCvtOffF32I4Vop3},
       }}},
  };
  enum class Form { Vop1, Vop3, Sdwa, Dpp, SuppressedDpp, ScalarSource, HighRegister };
  struct Scenario {
    Form form;
    uint32_t mode = 0xf0;
    unsigned omod = 0;
    bool clamp = false, absolute = false, negate = false, alias = false;
  };
  const Scenario scenarios[] = {
      {Form::Vop1},
      {.form = Form::Vop3, .mode = 0xf0},
      {.form = Form::Vop3, .mode = 0xf1, .alias = true},
      {.form = Form::Vop3, .mode = 0xf2},
      {.form = Form::Vop3, .mode = 0xf3},
      {.form = Form::Vop3, .mode = 0, .omod = 1, .absolute = true, .negate = true},
      {.form = Form::Vop3, .mode = 0x2f0, .omod = 3, .clamp = true},
      {.form = Form::Vop3, .mode = 0, .omod = 3},
      {.form = Form::Vop3, .mode = 0x2f0, .omod = 2},
      {.form = Form::Vop3, .mode = 0x200, .omod = 1},
      {Form::HighRegister},
      {Form::Sdwa},
      {.form = Form::Sdwa, .mode = 0, .omod = 1},
      {.form = Form::Sdwa, .mode = 0, .omod = 2, .clamp = true, .alias = true},
      {.form = Form::Sdwa, .mode = 3, .omod = 3},
      {.form = Form::Sdwa, .mode = 0x200, .omod = 1},
      {.form = Form::Sdwa, .mode = 0xf0, .omod = 2},
      {.form = Form::Dpp, .alias = true},
      {Form::SuppressedDpp},
      {.form = Form::ScalarSource, .mode = 0xf1},
  };
  constexpr uint32_t special[] = {0,          0xffffffff, 0x00010203, 0xfffefdfc,
                                  0x807f00ff, 0x0100807f, 0x12345678, 0x87654321,
                                  0x00000008, 0x0000000f, 0x00000007, 0x00000010,
                                  0x00000100, 0x00010000, 0x01000000, 0x80000000};
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
          // Sequence high-bit draws so word construction is toolchain-independent.
          const auto random_word = [&] {
            const uint32_t high = uint32_t(random() >> 15);
            return (high << 16) | uint32_t(random() >> 15);
          };
          std::array<uint32_t, 64> input{};
          for (int lane = 0; lane < lanes; ++lane) {
            input[lane] = lane < 16 ? special[lane] : random_word();
          }
          for (int operation = 0; operation < 5; ++operation) {
            for (const auto &sample : scenarios) {
              const bool vop3 = sample.form == Form::Vop3;
              const bool sdwa = sample.form == Form::Sdwa;
              const bool dpp = sample.form == Form::Dpp || sample.form == Form::SuppressedDpp;
              const bool sgpr = sample.form == Form::ScalarSource;
              const bool high_register = sample.form == Form::HighRegister;
              const uint32_t source_register = high_register ? 128 : 0;
              if (sdwa && family.arch == rocjitsu::IsaFamily::Rdna3_5) {
                continue;
              }
              SCOPED_TRACE(::testing::Message()
                           << family.arch << ':' << lanes << ':' << scalar << ':' << int(semantics)
                           << ':' << operation << ':' << int(sample.form) << ':' << sample.mode);
              const uint32_t destination = high_register ? 130U : sample.alias ? 0U : 2U;
              const uint64_t all = lanes == 64 ? ~0ULL : 0xffffffffULL;
              const uint64_t selected = all & 0x800000038000ffffULL;
              wave->set_exec(dpp ? all : selected);
              wave->set_vgpr_write_mask(dpp ? selected : all);
              wave->set_mode_raw(sample.mode);
              wave->set_trapsts(0x100);
              cu->write_sgpr(wave->sgpr_alloc().base, 0x4b800001U);
              for (int lane = 0; lane < lanes; ++lane) {
                cu->write_vgpr(base + destination, lane, 0xdeadbeefU);
                cu->write_vgpr(base + source_register, lane, input[lane]);
              }
              const uint32_t source = sgpr ? 0U : 256U + source_register;
              std::array<uint32_t, 2> words =
                  vop3
                      ? std::array<uint32_t, 2>{(cdna ? 0xd0000000U : 0xd4000000U) |
                                                    (uint32_t(family.opcodes[operation][1]) << 16) |
                                                    destination | (sample.absolute ? 1U << 8 : 0U) |
                                                    (sample.clamp ? 1U << 15 : 0U),
                                                source | (sample.omod << 27) |
                                                    (sample.negate ? 1U << 29 : 0U)}
                      : std::array<uint32_t, 2>{0x7e000000U | (destination << 17) |
                                                    (uint32_t(family.opcodes[operation][0]) << 9) |
                                                    source,
                                                0};
              if (sdwa) {
                // Select BYTE_2 before conversion; retain the full FP32 result.
                words[0] = (words[0] & ~0x1ffU) | 249U;
                words[1] = 0x00020600U | (sample.omod << 14) | (sample.clamp ? 1U << 13 : 0U);
              } else if (dpp) {
                // quad_perm:[1,0,3,2], optionally suppressing every destination row.
                words[0] = (words[0] & ~0x1ffU) | 250U;
                words[1] = sample.form == Form::SuppressedDpp ? 0x0f00b100U : 0xff00b100U;
              }
              std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
              ASSERT_NE(instruction, nullptr);
              ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
              EXPECT_EQ(wave->trapsts(), 0x100U);
              for (int lane = 0; lane < lanes; ++lane) {
                const uint32_t old = sample.alias ? input[lane] : 0xdeadbeefU;
                uint32_t expected = old;
                if ((selected >> lane & 1) && sample.form != Form::SuppressedDpp) {
                  uint32_t raw = sgpr ? 0x4b800001U : input[dpp ? lane ^ 1 : lane];
                  if (sdwa) {
                    raw = (raw >> 16) & 0xffU;
                  }
                  expected = reference(operation, raw, *wave, vop3 || sdwa ? sample.omod : 0,
                                       (vop3 || sdwa) && sample.clamp);
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
