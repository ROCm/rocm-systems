// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older division routing: guest rounding, conditions, exception causes and aliased half writes.
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/execution_backend.h"
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

namespace rocjitsu::amdgpu {
namespace {

// Restore the test-only CPU dispatch override after each test.
struct ScalarGuard {
  bool saved = util::force_scalar();
  ~ScalarGuard() { util::set_force_scalar_for_testing(saved); }
};

// Encode an exactly representable test value in either register format.
uint64_t encoded(double value, bool wide) {
  return wide ? std::bit_cast<uint64_t>(value) : std::bit_cast<uint32_t>(float(value));
}

// Exercise the same decoded contract on each older family's real encoding/backend.
template <typename Build, typename BuildScale>
void check_family(rocjitsu::IsaFamily arch, const IsaExecutionBackend &backend, Build build,
                  BuildScale build_scale, std::array<uint16_t, 7> opcodes,
                  uint16_t legacy_half = 0) {
  ScopedIsaExecutionBackend scoped_backend(&backend);
  ScalarGuard scalar_guard;
  const bool cdna = arch == rocjitsu::IsaFamily::Cdna1 || arch == rocjitsu::IsaFamily::Cdna2;
  for (int lanes : {32, 64}) {
    if (cdna && lanes == 32) {
      continue;
    }
    for (bool scalar : {true, false}) {
      util::set_force_scalar_for_testing(scalar);
      for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
        ComputeUnitCore::Config config{};
        config.arch = arch;
        config.arithmetic = {semantics};
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 104;
        config.vgprs_per_wf = 8;
        config.lds_size_kb = 64;
        GpuMemory memory("division_memory");
        L2Cache cache("division_cache");
        auto cu = ComputeUnitCore::create("division", config, &memory, &cache);
        ASSERT_NE(cu, nullptr);
        auto *wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
        ASSERT_NE(wave, nullptr);
        auto decoder = Decoder::create(arch);
        ASSERT_NE(decoder, nullptr);
        const auto base = wave->vgpr_alloc().base;
        const auto put = [&](int reg, int lane, uint64_t bits, bool wide) {
          cu->write_vgpr(base + reg, lane, uint32_t(bits));
          if (wide) {
            cu->write_vgpr(base + reg + 1, lane, uint32_t(bits >> 32));
          }
        };
        const auto get = [&](int reg, int lane, bool wide) {
          return uint64_t(cu->read_vgpr(base + reg, lane)) |
                 (wide ? uint64_t(cu->read_vgpr(base + reg + 1, lane)) << 32 : 0);
        };
        const uint64_t active = 1ULL | (1ULL << (lanes - 1));
        for (bool wide : {false, true}) {
          for (int operation = 0; operation < 3; ++operation) {
            for (unsigned rounding = 0; rounding < 4; ++rounding) {
              for (int aliasing_offset : {0, 2, 4, 6}) {
                SCOPED_TRACE(::testing::Message()
                             << arch << ':' << lanes << ':' << scalar << ':' << int(semantics)
                             << ':' << wide << ':' << operation << ':' << rounding << ':'
                             << aliasing_offset);
                wave->set_exec(rounding == 3 ? 0 : active);
                wave->set_vcc_mask(operation == 1 ? 0 : 0x55);
                wave->set_mode_raw(0xf0 | rounding | (rounding << 2));
                wave->set_trapsts(0x100);
                const double midpoint = wide ? 0x1p-53 : 0x1p-24;
                for (int lane = 0; lane < lanes; ++lane) {
                  put(6, lane, 0xdeadbeefdeadbeefULL, wide);
                  // SCALE's source zero is the numerator. FMAS straddles a rounding
                  // midpoint; FIXUP corrects quotient sign and reports division by zero.
                  put(0, lane,
                      encoded(operation == 0   ? (wide ? 0x1p-1022 : 0x1p-126)
                              : operation == 1 ? 1.
                                               : -3.,
                              wide),
                      wide);
                  put(2, lane, encoded(operation == 2 && lane == lanes - 1 ? 0. : 1., wide), wide);
                  put(4, lane,
                      encoded(operation == 0   ? (wide ? 0x1p-1022 : 0x1p-126)
                              : operation == 1 ? midpoint
                                               : 1.,
                              wide),
                      wide);
                }
                std::array<uint64_t, 64> before;
                for (int lane = 0; lane < lanes; ++lane) {
                  before[lane] = get(aliasing_offset, lane, wide);
                }
                // SCALE ignores OMOD and CLAMP. The other operations use no output
                // modifiers here, so the selected goldens isolate their core behavior.
                auto words = operation == 0 ? build_scale(opcodes[wide ? 1 : 0],
                                                          {.vdst = uint8_t(aliasing_offset),
                                                           .sdst = 4,
                                                           .clamp = 1,
                                                           .src0 = 256,
                                                           .src1 = 258,
                                                           .src2 = 260,
                                                           .omod = 1})
                                            : build(opcodes[(operation == 1 ? 2 : 4) + int(wide)],
                                                    {.vdst = uint8_t(aliasing_offset),
                                                     .src0 = 256,
                                                     .src1 = 258,
                                                     .src2 = 260});
                auto instruction = decoder->decode_window(words);
                ASSERT_TRUE(instruction.succeeded());
                ASSERT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
                const uint64_t exec_mask = wave->exec();
                for (int lane = 0; lane < lanes; ++lane) {
                  const double expected =
                      operation == 0 ? (wide ? 0x1p-894 : 0x1p-62)
                      : operation == 1
                          ? (rounding == 1 ? (wide ? 0x1.0000000000001p0 : 0x1.000002p0) : 1.)
                      : lane == lanes - 1 ? INFINITY
                                          : 3.;
                  EXPECT_EQ(get(aliasing_offset, lane, wide),
                            (exec_mask & (1ULL << lane)) ? encoded(expected, wide) : before[lane]);
                }
                if (operation == 0) {
                  const uint64_t expected = (0x55 & ~exec_mask) | exec_mask;
                  const auto sb = wave->sgpr_alloc().base;
                  EXPECT_EQ(cu->read_sgpr(sb + 4), uint32_t(expected));
                  if (lanes == 64) {
                    EXPECT_EQ(cu->read_sgpr(sb + 5), uint32_t(expected >> 32));
                  }
                }
                EXPECT_EQ(wave->vcc(), operation == 1 ? 0U : 0x55U);
                if (semantics == ArithmeticSemantics::Compatibility) {
                  EXPECT_EQ(wave->trapsts(), 0x100U | ((operation == 2 && exec_mask) ? 4U : 0U));
                }
              }
            }
          }
        }
        // OP_SEL chooses each half independently; aliasing must retain unread inputs.
        // Only the CDNA legacy low-half form clears the other half of the destination.
        for (bool legacy : {false, true}) {
          if (legacy && !legacy_half) {
            continue;
          }
          for (bool high : {false, true}) {
            if (legacy && high) {
              continue;
            }
            for (int aliasing_offset : {0, 2, 4, 6}) {
              wave->set_exec(active);
              wave->set_vcc_mask(0x55);
              wave->set_mode_raw(0xf0);
              for (int lane = 0; lane < lanes; ++lane) {
                put(6, lane, 0xa55aa55a, false);
                put(0, lane, high ? 0xc200a55a : 0xa55ac200, false); // -3
                put(2, lane, high ? 0x3c00a55a : 0xa55a3c00, false); // +1
                put(4, lane, high ? 0x3c00a55a : 0xa55a3c00, false);
              }
              std::array<uint32_t, 64> before;
              for (int lane = 0; lane < lanes; ++lane) {
                before[lane] = uint32_t(get(aliasing_offset, lane, false));
              }
              auto words =
                  build(legacy ? legacy_half : opcodes[6], {.vdst = uint8_t(aliasing_offset),
                                                            .op_sel = uint8_t(high ? 15 : 0),
                                                            .src0 = 256,
                                                            .src1 = 258,
                                                            .src2 = 260});
              auto instruction = decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              ASSERT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
              for (int lane = 0; lane < lanes; ++lane) {
                const uint32_t expected = high     ? (before[lane] & 0xffff) | 0x42000000
                                          : legacy ? 0x4200
                                                   : (before[lane] & 0xffff0000) | 0x4200;
                EXPECT_EQ(get(aliasing_offset, lane, false),
                          (active & (1ULL << lane)) ? expected : before[lane]);
              }
            }
          }
        }
      }
    }
  }
}

TEST(LegacyDivision, Arithmetic) {
  check_family(
      rocjitsu::IsaFamily::Cdna1, cdna1::execution_backend(),
      [](uint16_t op, cdna1::Vop3BuilderFields fields) { return cdna1::build_vop3(op, fields); },
      [](uint16_t op, cdna1::Vop3SdstEncBuilderFields fields) {
        return cdna1::build_vop3_sdst_enc(op, fields);
      },
      {cdna1::kVDivScaleF32Vop3SdstEnc, cdna1::kVDivScaleF64Vop3SdstEnc, cdna1::kVDivFmasF32Vop3,
       cdna1::kVDivFmasF64Vop3, cdna1::kVDivFixupF32Vop3, cdna1::kVDivFixupF64Vop3,
       cdna1::kVDivFixupF16Vop3},
      cdna1::kVDivFixupLegacyF16Vop3);
  check_family(
      rocjitsu::IsaFamily::Cdna2, cdna2::execution_backend(),
      [](uint16_t op, cdna2::Vop3BuilderFields fields) { return cdna2::build_vop3(op, fields); },
      [](uint16_t op, cdna2::Vop3SdstEncBuilderFields fields) {
        return cdna2::build_vop3_sdst_enc(op, fields);
      },
      {cdna2::kVDivScaleF32Vop3SdstEnc, cdna2::kVDivScaleF64Vop3SdstEnc, cdna2::kVDivFmasF32Vop3,
       cdna2::kVDivFmasF64Vop3, cdna2::kVDivFixupF32Vop3, cdna2::kVDivFixupF64Vop3,
       cdna2::kVDivFixupF16Vop3},
      cdna2::kVDivFixupLegacyF16Vop3);
  check_family(
      rocjitsu::IsaFamily::Rdna1, rdna1::execution_backend(),
      [](uint16_t op, rdna1::Vop3BuilderFields fields) { return rdna1::build_vop3(op, fields); },
      [](uint16_t op, rdna1::Vop3SdstEncBuilderFields fields) {
        return rdna1::build_vop3_sdst_enc(op, fields);
      },
      {rdna1::kVDivScaleF32Vop3SdstEnc, rdna1::kVDivScaleF64Vop3SdstEnc, rdna1::kVDivFmasF32Vop3,
       rdna1::kVDivFmasF64Vop3, rdna1::kVDivFixupF32Vop3, rdna1::kVDivFixupF64Vop3,
       rdna1::kVDivFixupF16Vop3});
  check_family(
      rocjitsu::IsaFamily::Rdna2, rdna2::execution_backend(),
      [](uint16_t op, rdna2::Vop3BuilderFields fields) { return rdna2::build_vop3(op, fields); },
      [](uint16_t op, rdna2::Vop3SdstEncBuilderFields fields) {
        return rdna2::build_vop3_sdst_enc(op, fields);
      },
      {rdna2::kVDivScaleF32Vop3SdstEnc, rdna2::kVDivScaleF64Vop3SdstEnc, rdna2::kVDivFmasF32Vop3,
       rdna2::kVDivFmasF64Vop3, rdna2::kVDivFixupF32Vop3, rdna2::kVDivFixupF64Vop3,
       rdna2::kVDivFixupF16Vop3});
  check_family(
      rocjitsu::IsaFamily::Rdna3_5, rdna3_5::execution_backend(),
      [](uint16_t op, rdna3_5::Vop3BuilderFields fields) {
        return rdna3_5::build_vop3(op, fields);
      },
      [](uint16_t op, rdna3_5::Vop3SdstEncBuilderFields fields) {
        return rdna3_5::build_vop3_sdst_enc(op, fields);
      },
      {rdna3_5::kVDivScaleF32Vop3SdstEnc, rdna3_5::kVDivScaleF64Vop3SdstEnc,
       rdna3_5::kVDivFmasF32Vop3, rdna3_5::kVDivFmasF64Vop3, rdna3_5::kVDivFixupF32Vop3,
       rdna3_5::kVDivFixupF64Vop3, rdna3_5::kVDivFixupF16Vop3});
}

} // namespace
} // namespace rocjitsu::amdgpu
