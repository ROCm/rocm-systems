// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded older-target FP32/integer conversions: independent arithmetic, guest
// controls, source staging, destination placement, aliases and inactive lanes.

#include "compute_unit_test_util.h"
#include "conversion32_reference.h"
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
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <random>

namespace rocjitsu::amdgpu {
namespace {

// Resolve the older profiles' MODE fields for the independent scalar reference.
uint32_t reference(int operation, uint32_t raw, const Wavefront &wave, unsigned omod, bool clamp,
                   bool absolute, bool negate) {
  if (wave.ieee_mode() || (wave.fp_denorm_mode_f32() & 2)) {
    omod = 0;
  }
  return rocjitsu::test::convert32(operation, raw,
                                   {omod, wave.fp_round_mode_f32(), clamp, wave.dx10_clamp()},
                                   (wave.fp_denorm_mode_f32() & 1) == 0, absolute, negate);
}

// Restore the caller's test override even when an assertion exits the test.
struct ScalarGuard {
  bool saved = util::force_scalar();
  ~ScalarGuard() { util::set_force_scalar_for_testing(saved); }
};

// Exercise the same models through each real target's decoder and operand wrappers.
TEST(OlderConversion32, Arithmetic) {
  ScalarGuard guard;
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<std::array<uint16_t, 2>, 6> opcodes;
  };
  // Order: I32/U32 to F32, F32 to I32/U32, nearest-up, floor.
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {{
           {cdna1::kVCvtF32I32Vop1, cdna1::kVCvtF32I32Vop3},
           {cdna1::kVCvtF32U32Vop1, cdna1::kVCvtF32U32Vop3},
           {cdna1::kVCvtI32F32Vop1, cdna1::kVCvtI32F32Vop3},
           {cdna1::kVCvtU32F32Vop1, cdna1::kVCvtU32F32Vop3},
           {cdna1::kVCvtRpiI32F32Vop1, cdna1::kVCvtRpiI32F32Vop3},
           {cdna1::kVCvtFlrI32F32Vop1, cdna1::kVCvtFlrI32F32Vop3},
       }}},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {{
           {cdna2::kVCvtF32I32Vop1, cdna2::kVCvtF32I32Vop3},
           {cdna2::kVCvtF32U32Vop1, cdna2::kVCvtF32U32Vop3},
           {cdna2::kVCvtI32F32Vop1, cdna2::kVCvtI32F32Vop3},
           {cdna2::kVCvtU32F32Vop1, cdna2::kVCvtU32F32Vop3},
           {cdna2::kVCvtRpiI32F32Vop1, cdna2::kVCvtRpiI32F32Vop3},
           {cdna2::kVCvtFlrI32F32Vop1, cdna2::kVCvtFlrI32F32Vop3},
       }}},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {{
           {rdna1::kVCvtF32I32Vop1, rdna1::kVCvtF32I32Vop3},
           {rdna1::kVCvtF32U32Vop1, rdna1::kVCvtF32U32Vop3},
           {rdna1::kVCvtI32F32Vop1, rdna1::kVCvtI32F32Vop3},
           {rdna1::kVCvtU32F32Vop1, rdna1::kVCvtU32F32Vop3},
           {rdna1::kVCvtRpiI32F32Vop1, rdna1::kVCvtRpiI32F32Vop3},
           {rdna1::kVCvtFlrI32F32Vop1, rdna1::kVCvtFlrI32F32Vop3},
       }}},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {{
           {rdna2::kVCvtF32I32Vop1, rdna2::kVCvtF32I32Vop3},
           {rdna2::kVCvtF32U32Vop1, rdna2::kVCvtF32U32Vop3},
           {rdna2::kVCvtI32F32Vop1, rdna2::kVCvtI32F32Vop3},
           {rdna2::kVCvtU32F32Vop1, rdna2::kVCvtU32F32Vop3},
           {rdna2::kVCvtRpiI32F32Vop1, rdna2::kVCvtRpiI32F32Vop3},
           {rdna2::kVCvtFlrI32F32Vop1, rdna2::kVCvtFlrI32F32Vop3},
       }}},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {{
           {rdna3_5::kVCvtF32I32Vop1, rdna3_5::kVCvtF32I32Vop3},
           {rdna3_5::kVCvtF32U32Vop1, rdna3_5::kVCvtF32U32Vop3},
           {rdna3_5::kVCvtI32F32Vop1, rdna3_5::kVCvtI32F32Vop3},
           {rdna3_5::kVCvtU32F32Vop1, rdna3_5::kVCvtU32F32Vop3},
           {rdna3_5::kVCvtNearestI32F32Vop1, rdna3_5::kVCvtNearestI32F32Vop3},
           {rdna3_5::kVCvtFloorI32F32Vop1, rdna3_5::kVCvtFloorI32F32Vop3},
       }}},
  };
  enum class Form { Vop1, Vop3, Sdwa, Dpp, SuppressedDpp, ScalarSource };
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
      {Form::Sdwa},
      {.form = Form::Dpp, .alias = true},
      {Form::SuppressedDpp},
      {.form = Form::ScalarSource, .mode = 0xf1},
  };
  constexpr uint32_t special[] = {0x7fc12345, 0x7f800000, 0xff800000, 0x80000000,
                                  0x00000001, 0x80000001, 0x4f000000, 0xcf000000,
                                  0x4f800000, 0x4effffff, 0x3fc00000, 0xbfc00000,
                                  0x01000001, 0xfeffffff, 0x7f800001, 0};
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
          config.vgprs_per_wf = 8;
          config.lds_size_kb = 64;
          GpuMemory memory("conversion_memory");
          L2Cache cache("conversion_cache");
          cache.set_backing_memory(&memory);
          std::unique_ptr<ComputeUnitCore> cu;
          ASSERT_NO_FATAL_FAILURE(
              rocjitsu::test::initialize_compute_unit(cu, "conversion", config, &memory, &cache));
          auto *wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
          ASSERT_NE(wave, nullptr);
          auto decoder = Decoder::create(family.arch);
          const auto base = wave->vgpr_alloc().base;
          std::minstd_rand random;
          const auto next_word = [&] {
            const uint32_t high = uint32_t(random() >> 15);
            return (high << 16) | uint32_t(random() >> 15);
          };
          std::array<uint32_t, 64> input{};
          for (int lane = 0; lane < lanes; ++lane) {
            input[lane] = lane < 16 ? special[lane] : next_word();
          }
          for (int operation = 0; operation < 6; ++operation) {
            for (const auto &sample : scenarios) {
              const bool vop3 = sample.form == Form::Vop3;
              const bool sdwa = sample.form == Form::Sdwa;
              const bool dpp = sample.form == Form::Dpp || sample.form == Form::SuppressedDpp;
              const bool sgpr = sample.form == Form::ScalarSource;
              if (sdwa && family.arch == rocjitsu::IsaFamily::Rdna3_5) {
                continue;
              }
              SCOPED_TRACE(::testing::Message()
                           << family.arch << ':' << lanes << ':' << scalar << ':' << int(semantics)
                           << ':' << operation << ':' << int(sample.form) << ':' << sample.mode);
              const uint32_t destination = sample.alias ? 0U : 2U;
              const uint64_t all = lanes == 64 ? ~0ULL : 0xffffffffULL;
              const uint64_t selected = all & 0x800000038000ffffULL;
              wave->set_exec(dpp ? all : selected);
              wave->set_vgpr_write_mask(dpp ? selected : all);
              wave->set_mode_raw(sample.mode);
              wave->set_trapsts(0x100);
              cu->write_sgpr(wave->sgpr_alloc().base, 0x4b800001U);
              for (int lane = 0; lane < lanes; ++lane) {
                cu->write_vgpr(base + 2, lane, 0xdeadbeefU);
                cu->write_vgpr(base, lane, input[lane]);
              }
              const uint32_t source = sgpr ? 0U : 256U;
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
                // DWORD source, WORD_1 destination, preserve the other destination half.
                words[0] = (words[0] & ~0x1ffU) | 249U;
                words[1] = 0x00061500U;
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
                  expected = reference(operation, sgpr ? 0x4b800001U : input[dpp ? lane ^ 1 : lane],
                                       *wave, vop3 ? sample.omod : 0, vop3 && sample.clamp,
                                       vop3 && sample.absolute, vop3 && sample.negate);
                  if (sdwa) {
                    expected = (old & 0xffffU) | (expected << 16);
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
