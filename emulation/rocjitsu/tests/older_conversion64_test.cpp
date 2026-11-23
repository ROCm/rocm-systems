// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded older-target FP64 conversions: inherited rounding/NaN policies,
// split VGPR layout, overlapping operands and inactive destination preservation.

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

// Compare against the inherited raw conversion model, independently of dispatch.
uint64_t reference(int operation, uint64_t raw, const Wavefront &wave, unsigned omod, bool clamp,
                   bool absolute, bool negate) {
  if (operation == 0) {
    return conversion_to_float<conversion::I32, fp_format::F64>(wave, 0, 0, omod, clamp)(raw);
  }
  if (operation == 1) {
    return conversion_to_float<conversion::U32, fp_format::F64>(wave, 0, 0, omod, clamp)(raw);
  }
  if (operation == 2) {
    return conversion_to_float<fp_format::F32, fp_format::F64>(wave, absolute, negate, omod,
                                                               clamp)(raw);
  }
  if (operation == 3) {
    return conversion_to_float<fp_format::F64, fp_format::F32>(wave, absolute, negate, omod,
                                                               clamp)(raw);
  }
  raw = (absolute ? raw & 0x7fffffffffffffffULL : raw) ^ (negate ? 0x8000000000000000ULL : 0);
  const double x = std::bit_cast<double>(raw);
  if (std::isnan(x)) {
    return 0;
  }
  if (operation == 4) {
    return x >= 0x1p31    ? uint32_t(INT32_MAX)
           : x <= -0x1p31 ? uint32_t(INT32_MIN)
                          : uint32_t(int32_t(x));
  }
  return !(x > 0) ? 0 : x >= 0x1p32 ? UINT32_MAX : uint32_t(x);
}

// Restore the caller's test override even when an assertion exits the test.
struct ScalarGuard {
  bool saved = util::force_scalar();
  ~ScalarGuard() { util::set_force_scalar_for_testing(saved); }
};

// Exercise the same models through each real target's decoder and operand wrappers.
TEST(OlderConversion64, Arithmetic) {
  ScalarGuard guard;
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<std::array<uint16_t, 2>, 6> opcodes;
  };
  // Order: I32/U32/F32 to F64, then F64 to F32/I32/U32.
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {{
           {cdna1::kVCvtF64I32Vop1, cdna1::kVCvtF64I32Vop3},
           {cdna1::kVCvtF64U32Vop1, cdna1::kVCvtF64U32Vop3},
           {cdna1::kVCvtF64F32Vop1, cdna1::kVCvtF64F32Vop3},
           {cdna1::kVCvtF32F64Vop1, cdna1::kVCvtF32F64Vop3},
           {cdna1::kVCvtI32F64Vop1, cdna1::kVCvtI32F64Vop3},
           {cdna1::kVCvtU32F64Vop1, cdna1::kVCvtU32F64Vop3},
       }}},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {{
           {cdna2::kVCvtF64I32Vop1, cdna2::kVCvtF64I32Vop3},
           {cdna2::kVCvtF64U32Vop1, cdna2::kVCvtF64U32Vop3},
           {cdna2::kVCvtF64F32Vop1, cdna2::kVCvtF64F32Vop3},
           {cdna2::kVCvtF32F64Vop1, cdna2::kVCvtF32F64Vop3},
           {cdna2::kVCvtI32F64Vop1, cdna2::kVCvtI32F64Vop3},
           {cdna2::kVCvtU32F64Vop1, cdna2::kVCvtU32F64Vop3},
       }}},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {{
           {rdna1::kVCvtF64I32Vop1, rdna1::kVCvtF64I32Vop3},
           {rdna1::kVCvtF64U32Vop1, rdna1::kVCvtF64U32Vop3},
           {rdna1::kVCvtF64F32Vop1, rdna1::kVCvtF64F32Vop3},
           {rdna1::kVCvtF32F64Vop1, rdna1::kVCvtF32F64Vop3},
           {rdna1::kVCvtI32F64Vop1, rdna1::kVCvtI32F64Vop3},
           {rdna1::kVCvtU32F64Vop1, rdna1::kVCvtU32F64Vop3},
       }}},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {{
           {rdna2::kVCvtF64I32Vop1, rdna2::kVCvtF64I32Vop3},
           {rdna2::kVCvtF64U32Vop1, rdna2::kVCvtF64U32Vop3},
           {rdna2::kVCvtF64F32Vop1, rdna2::kVCvtF64F32Vop3},
           {rdna2::kVCvtF32F64Vop1, rdna2::kVCvtF32F64Vop3},
           {rdna2::kVCvtI32F64Vop1, rdna2::kVCvtI32F64Vop3},
           {rdna2::kVCvtU32F64Vop1, rdna2::kVCvtU32F64Vop3},
       }}},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {{
           {rdna3_5::kVCvtF64I32Vop1, rdna3_5::kVCvtF64I32Vop3},
           {rdna3_5::kVCvtF64U32Vop1, rdna3_5::kVCvtF64U32Vop3},
           {rdna3_5::kVCvtF64F32Vop1, rdna3_5::kVCvtF64F32Vop3},
           {rdna3_5::kVCvtF32F64Vop1, rdna3_5::kVCvtF32F64Vop3},
           {rdna3_5::kVCvtI32F64Vop1, rdna3_5::kVCvtI32F64Vop3},
           {rdna3_5::kVCvtU32F64Vop1, rdna3_5::kVCvtU32F64Vop3},
       }}},
  };
  struct Scenario {
    bool vop3 = false;
    uint32_t mode = 0xf0;
    unsigned omod = 0;
    bool clamp = false, absolute = false, negate = false;
    int aliasing_offset = 4;
  };
  const Scenario scenarios[] = {
      {},
      {.mode = 0x61}, // Distinct FP32 and FP16/FP64 guest controls.

      {.vop3 = true, .mode = 0xf0, .aliasing_offset = 0},
      {.vop3 = true, .mode = 0xf5, .aliasing_offset = 1},
      {.vop3 = true, .mode = 0xfa},
      {.vop3 = true, .mode = 0xff},
      {.vop3 = true, .mode = 0, .omod = 1, .absolute = true, .negate = true},
      {.vop3 = true, .mode = 0x2f0, .omod = 3, .clamp = true},
      {.vop3 = true, .mode = 0x3f0, .omod = 2, .clamp = true},
  };
  constexpr uint64_t special64[] = {0x7ff8123456789abcULL,
                                    0x7ff0000000000000ULL,
                                    0xfff0000000000000ULL,
                                    0x8000000000000000ULL,
                                    1,
                                    0x8000000000000001ULL,
                                    0x41e0000000000000ULL,
                                    0xc1e0000000000000ULL,
                                    0x41f0000000000000ULL,
                                    0x41dfffffffffffffULL,
                                    0x3ff0000010000000ULL,
                                    0xbff0000010000000ULL,
                                    0x47efffffe0000000ULL,
                                    0x3690000000000000ULL,
                                    0x7ff0000000000001ULL,
                                    0};
  constexpr uint32_t special32[] = {0x7fc12345, 0x7f800000, 0xff800000, 0x80000000,
                                    1,          0x80000001, 0x4f000000, 0xcf000000,
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
          const auto word = [&]() {
            return (uint32_t(random() >> 15) << 16) | uint32_t(random() >> 15);
          };
          std::array<uint64_t, 64> input64{};
          std::array<uint32_t, 64> input32{};
          for (int lane = 0; lane < lanes; ++lane) {
            input64[lane] = lane < 16 ? special64[lane] : (uint64_t(word()) << 32) | word();
            input32[lane] = lane < 16 ? special32[lane] : word();
          }
          for (int operation = 0; operation < 6; ++operation) {
            const bool source_pair = operation >= 3;
            const bool dest_pair = operation < 3;
            for (const auto &sample : scenarios) {
              SCOPED_TRACE(::testing::Message() << family.arch << ':' << lanes << ':' << scalar
                                                << ':' << int(semantics) << ':' << operation << ':'
                                                << sample.mode << ':' << sample.aliasing_offset);
              const uint32_t destination = sample.aliasing_offset;
              const uint64_t all = lanes == 64 ? ~0ULL : 0xffffffffULL;
              const uint64_t exec_mask = all & 0x800000038000ffffULL;
              wave->set_exec(exec_mask);
              wave->set_vgpr_write_mask(all);
              wave->set_mode_raw(sample.mode);
              wave->set_trapsts(0x100);
              std::array<std::array<uint32_t, 64>, 8> before{};
              for (int lane = 0; lane < lanes; ++lane) {
                for (int reg = 0; reg < 8; ++reg) {
                  before[reg][lane] = 0xdeadbeefU;
                }
                before[0][lane] = source_pair ? uint32_t(input64[lane]) : input32[lane];
                before[1][lane] = source_pair ? uint32_t(input64[lane] >> 32) : 0xdeadbeefU;
                for (int reg = 0; reg < 8; ++reg) {
                  cu->write_vgpr(base + reg, lane, before[reg][lane]);
                }
              }
              std::array<uint32_t, 2> words =
                  sample.vop3
                      ? std::array<uint32_t, 2>{(cdna ? 0xd0000000U : 0xd4000000U) |
                                                    (uint32_t(family.opcodes[operation][1]) << 16) |
                                                    destination | (sample.absolute ? 1U << 8 : 0U) |
                                                    (sample.clamp ? 1U << 15 : 0U),
                                                256U | (sample.omod << 27) |
                                                    (sample.negate ? 1U << 29 : 0U)}
                      : std::array<uint32_t, 2>{0x7e000000U | (destination << 17) |
                                                    (uint32_t(family.opcodes[operation][0]) << 9) |
                                                    256U,
                                                0};
              std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
              ASSERT_NE(instruction, nullptr);
              ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
              EXPECT_EQ(wave->trapsts(), 0x100U);
              for (int lane = 0; lane < lanes; ++lane) {
                uint64_t expected = before[destination][lane];
                if (dest_pair) {
                  expected |= uint64_t(before[destination + 1][lane]) << 32;
                }
                if ((exec_mask >> lane) & 1) {
                  expected =
                      reference(operation, source_pair ? input64[lane] : input32[lane], *wave,
                                sample.vop3 ? sample.omod : 0, sample.vop3 && sample.clamp,
                                sample.vop3 && sample.absolute, sample.vop3 && sample.negate);
                }
                EXPECT_EQ(cu->read_vgpr(base + destination, lane), uint32_t(expected))
                    << "lane=" << lane;
                if (dest_pair) {
                  EXPECT_EQ(cu->read_vgpr(base + destination + 1, lane), uint32_t(expected >> 32))
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
