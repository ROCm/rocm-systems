// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded CDNA integer migration: aliases, saturation, masks and aliased scalar results.
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <bit>
#include <gtest/gtest.h>

namespace rocjitsu::amdgpu {
namespace {
// The instruction layouts share fields; opcodes remain architecture-specific.
std::array<uint32_t, 2> build32(int arch, uint16_t opcode, uint16_t src2, uint8_t opsel = 0) {
  if (arch == 0) {
    return cdna3::build_vop3(
        opcode, {.vdst = 0, .op_sel = opsel, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = src2});
  }
  if (arch == 1) {
    return cdna4::build_vop3(
        opcode, {.vdst = 0, .op_sel = opsel, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = src2});
  }
  return cdna5::build_vop3(
      opcode, {.vdst = 0, .opsel = opsel, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = src2});
}
// Wide values use adjacent VGPR pairs; shift-add uses the low word of B as its count.
std::array<uint32_t, 2> build_wide(int arch, uint16_t opcode) {
  if (arch == 0) {
    return cdna3::build_vop3(opcode,
                             {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 258, .src2 = 260});
  }
  if (arch == 1) {
    return cdna4::build_vop3(opcode,
                             {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 258, .src2 = 260});
  }
  return cdna5::build_vop3(opcode, {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 258, .src2 = 260});
}

// Carry-in aliases its scalar destination; wide MAD aliases its vector accumulator.
std::array<uint32_t, 2> build_carry(int arch, bool wide) {
  uint8_t destination = wide ? 2 : 0;
  uint16_t source = wide ? 258 : 0;
  if (arch == 0) {
    return cdna3::build_vop3_sdst_enc(
        wide ? cdna3::kVMadU64U32Vop3SdstEnc : cdna3::kVAddcCoU32Vop3SdstEnc,
        {.vdst = destination, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = source});
  }
  if (arch == 1) {
    return cdna4::build_vop3_sdst_enc(
        wide ? cdna4::kVMadU64U32Vop3SdstEnc : cdna4::kVAddcCoU32Vop3SdstEnc,
        {.vdst = destination, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = source});
  }
  return cdna5::build_vop3_sdst_enc(
      wide ? cdna5::kVMadCoU64U32Vop3SdstEnc : cdna5::kVAddCoCiU32Vop3SdstEnc,
      {.vdst = destination, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = source});
}

TEST(CdnaIntegerArith, DecodedMasksAliasesAndCarry) {
  struct Case {
    std::array<uint16_t, 3> opcode;
    uint32_t a, b, c, expected;
    bool lane_count = false;
  };
  const Case cases[] = {
      {{cdna3::kVAndB32Vop3, cdna4::kVAndB32Vop3, cdna5::kVAndB32Vop3},
       0xff00ff00U,
       0x12345678U,
       0,
       0x12005600U},
      {{cdna3::kVAddU32Vop3, cdna4::kVAddU32Vop3, cdna5::kVAddNcU32Vop3},
       0xffffffffU,
       2,
       0,
       0xffffffffU},
      {{cdna3::kVFfbhU32Vop3, cdna4::kVFfbhU32Vop3, cdna5::kVClzI32U32Vop3}, 0, 0, 0, 0xffffffffU},
      {{cdna3::kVFfbhI32Vop3, cdna4::kVFfbhI32Vop3, cdna5::kVClsI32Vop3},
       0xffffffffU,
       0,
       0,
       0xffffffffU},
      {{cdna3::kVBfeI32Vop3, cdna4::kVBfeI32Vop3, cdna5::kVBfeI32Vop3},
       0x00800000U,
       8,
       16,
       0xffff8000U},
      {{cdna3::kVMulHiU32Vop3, cdna4::kVMulHiU32Vop3, cdna5::kVMulHiU32Vop3}, 0xffffffffU, 2, 0, 1},
      {{cdna3::kVMbcntHiU32B32Vop3, cdna4::kVMbcntHiU32B32Vop3, cdna5::kVMbcntHiU32B32Vop3},
       0xffffffffU,
       3,
       0,
       3,
       true},
  };
  for (int arch = 0; arch < 3; ++arch) {
    for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::BitExactEmpirical,
                           ArithmeticSemantics::Compatibility}) {
      ScopedIsaExecutionBackend backend(arch == 0   ? &cdna3::execution_backend()
                                        : arch == 1 ? &cdna4::execution_backend()
                                                    : &cdna5::execution_backend());
      ComputeUnitCore::Config config{};
      config.arch = arch == 0   ? ROCJITSU_CODE_ARCH_CDNA3
                    : arch == 1 ? ROCJITSU_CODE_ARCH_CDNA4
                                : ROCJITSU_CODE_ARCH_CDNA5;
      config.arithmetic = {semantics};
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 104;
      config.vgprs_per_wf = 32;
      config.lds_size_kb = 64;
      GpuMemory memory("cdna_integer_memory");
      L2Cache l2("cdna_integer_l2");
      auto cu = ComputeUnitCore::create("cdna_integer", config, &memory, &l2);
      unsigned lanes = arch == 2 ? 32 : 64;
      auto *wave = cu->dispatch_wf(0, 0, 104, 32, lanes);
      ASSERT_NE(wave, nullptr);
      auto decoder = Decoder::create(config.arch);
      uint32_t base = wave->vgpr_alloc().base, sb = wave->sgpr_alloc().base;
      wave->set_exec(0x8000000380000005ULL);
      uint64_t active = wave->exec();
      if (arch < 2) {
        // FPSan packs high-half results with SDWA subtraction on gfx950.
        for (unsigned lane = 0; lane < lanes; ++lane) {
          cu->write_vgpr(base + 1, lane, 5);
          cu->write_vgpr(base + 3, lane, 2);
        }
        cdna4::Vop2VopSdwaMachineInst raw{};
        raw.op = arch == 0 ? cdna3::kVSubU32Vop2 : cdna4::kVSubU32Vop2;
        raw.src0 = SRC_SDWA;
        raw.vsrc0 = 1;
        raw.vsrc1 = raw.vdst = 3;
        raw.src0_sel = raw.src1_sel = sdwa::DWORD;
        raw.dst_sel = sdwa::WORD_1;
        raw.dst_unused = sdwa::UNUSED_PAD;
        auto inst = decoder->decode_window(std::bit_cast<std::array<uint32_t, 2>>(raw));
        ASSERT_TRUE(inst.succeeded());
        ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *wave).succeeded());
        for (unsigned lane = 0; lane < lanes; ++lane) {
          EXPECT_EQ(cu->read_vgpr(base + 3, lane), (active >> lane) & 1 ? 0x30000U : 2U);
        }
      }
      for (const auto &test : cases) {
        SCOPED_TRACE(test.opcode[arch]);
        for (unsigned lane = 0; lane < lanes; ++lane) {
          cu->write_vgpr(base, lane, test.a);
          cu->write_vgpr(base + 1, lane, test.b);
          cu->write_vgpr(base + 2, lane, test.c);
        }
        auto words = build32(arch, test.opcode[arch], 258);
        auto inst = decoder->decode_window(words);
        ASSERT_TRUE(inst.succeeded());
        ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *wave).succeeded());
        for (unsigned lane = 0; lane < lanes; ++lane) {
          uint32_t expected = test.expected + (test.lane_count && lane > 32 ? lane - 32 : 0);
          EXPECT_EQ(cu->read_vgpr(base, lane), (active >> lane) & 1 ? expected : test.a);
        }
      }
      // All architectures select input halves; only gfx942/gfx950 low writes clear high.
      for (bool high : {false, true}) {
        for (unsigned lane = 0; lane < lanes; ++lane) {
          cu->write_vgpr(base, lane, 0x1234000bU);
          cu->write_vgpr(base + 1, lane, 0x0007fffeU);
        }
        uint16_t opcode = arch == 0   ? cdna3::kVMinU16Vop3
                          : arch == 1 ? cdna4::kVMinU16Vop3
                                      : cdna5::kVMinU16Vop3;
        auto words = build32(arch, opcode, 258, 3 | (high ? 8 : 0));
        auto inst = decoder->decode_window(words);
        ASSERT_TRUE(inst.succeeded());
        ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *wave).succeeded());
        for (unsigned lane = 0; lane < lanes; ++lane) {
          uint32_t expected = high ? 0x0007000bU : arch == 2 ? 0x12340007U : 7U;
          EXPECT_EQ(cu->read_vgpr(base, lane), (active >> lane) & 1 ? expected : 0x1234000bU);
        }
      }
      if (arch != 2) {
        // Legacy MAD ignores CLAMP: overflow wraps, with normal OP_SEL and destination rules.
        for (bool is_signed : {false, true}) {
          const uint32_t a = is_signed ? 0x7fff000bU : 0xffff000bU;
          for (bool high : {false, true}) {
            for (unsigned lane = 0; lane < lanes; ++lane) {
              cu->write_vgpr(base, lane, a);
              cu->write_vgpr(base + 1, lane, 0x00020007U);
              cu->write_vgpr(base + 2, lane, 0x00010009U);
            }
            uint16_t opcode =
                arch == 0 ? (is_signed ? cdna3::kVMadLegacyI16Vop3 : cdna3::kVMadLegacyU16Vop3)
                          : (is_signed ? cdna4::kVMadLegacyI16Vop3 : cdna4::kVMadLegacyU16Vop3);
            auto words = build32(arch, opcode, 258, 7 | (high ? 8 : 0));
            auto inst = decoder->decode_window(words);
            ASSERT_TRUE(inst.succeeded());
            ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *wave).succeeded());
            for (unsigned lane = 0; lane < lanes; ++lane) {
              uint32_t expected = high ? 0xffff000bU : 0x0000ffffU;
              EXPECT_EQ(cu->read_vgpr(base, lane), (active >> lane) & 1 ? expected : a);
            }
          }
        }
        // Reverse subtraction must swap source selectors along with source pointers.
        for (unsigned lane = 0; lane < lanes; ++lane) {
          cu->write_vgpr(base, lane, 0x000b0002U);
          cu->write_vgpr(base + 1, lane, 0x00070008U);
        }
        auto words =
            build32(arch, arch == 0 ? cdna3::kVSubrevU16Vop3 : cdna4::kVSubrevU16Vop3, 258, 1);
        auto inst = decoder->decode_window(words);
        ASSERT_TRUE(inst.succeeded());
        ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *wave).succeeded());
        for (unsigned lane = 0; lane < lanes; ++lane) {
          EXPECT_EQ(cu->read_vgpr(base, lane), (active >> lane) & 1 ? 0U : 0x000b0002U);
        }
      }
      if (arch != 0) {
        // Inputs are full signed words; only the destination uses OP_SEL.
        for (bool is_signed : {false, true}) {
          for (bool high : {false, true}) {
            for (unsigned lane = 0; lane < lanes; ++lane) {
              cu->write_vgpr(base, lane, 0xfffffe00U);
              cu->write_vgpr(base + 1, lane, 0x00000800U);
              cu->write_vgpr(base + 2, lane, 2);
            }
            uint16_t opcode =
                arch == 1 ? (is_signed ? cdna4::kVAshrPkI8I32Vop3 : cdna4::kVAshrPkU8I32Vop3)
                          : (is_signed ? cdna5::kVAshrPkI8I32Vop3 : cdna5::kVAshrPkU8I32Vop3);
            auto words = build32(arch, opcode, 258, 7 | (high ? 8 : 0));
            auto inst = decoder->decode_window(words);
            ASSERT_TRUE(inst.succeeded());
            ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *wave).succeeded());
            for (unsigned lane = 0; lane < lanes; ++lane) {
              uint32_t packed = is_signed ? 0x7f80U : 0xff00U;
              uint32_t expected =
                  high ? (packed << 16) | 0xfe00U : packed | (arch == 2 ? 0xffff0000U : 0);
              EXPECT_EQ(cu->read_vgpr(base, lane), (active >> lane) & 1 ? expected : 0xfffffe00U);
            }
          }
        }
      }
      if (arch == 2) {
        // Non-carry MAD saturates the wide result without changing VCC.
        for (bool is_signed : {false, true}) {
          wave->set_vcc(0x12345678U);
          for (unsigned lane = 0; lane < lanes; ++lane) {
            cu->write_vgpr(base, lane, is_signed ? 0x80000000U : 0xffffffffU);
            cu->write_vgpr(base + 1, lane, is_signed ? 0x80000000U : 0xffffffffU);
            cu->write_vgpr(base + 2, lane, 0xffffffffU);
            cu->write_vgpr(base + 3, lane, is_signed ? 0x7fffffffU : 0xffffffffU);
          }
          auto words =
              build32(arch, is_signed ? cdna5::kVMadNcI64I32Vop3 : cdna5::kVMadNcU64U32Vop3, 258);
          auto inst = decoder->decode_window(words);
          ASSERT_TRUE(inst.succeeded());
          ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *wave).succeeded());
          EXPECT_EQ(wave->vcc(), 0x12345678U);
          for (unsigned lane = 0; lane < lanes; ++lane) {
            bool written = (active >> lane) & 1;
            EXPECT_EQ(cu->read_vgpr(base, lane), written || !is_signed ? 0xffffffffU : 0x80000000U);
            EXPECT_EQ(cu->read_vgpr(base + 1, lane),
                      is_signed ? (written ? 0x7fffffffU : 0x80000000U) : 0xffffffffU);
          }
        }
      }
      // Wide operations retain whole-pair aliases, saturation and masked shift counts.
      struct WideCase {
        uint16_t opcode;
        uint64_t a, b, c, expected;
      };
      const WideCase wide_cases[] = {
          {arch == 0   ? cdna3::kVLshlAddU64Vop3
           : arch == 1 ? cdna4::kVLshlAddU64Vop3
                       : cdna5::kVLshlAddU64Vop3,
           0x8000000000000001ULL, 65, UINT64_MAX, 1},
          {cdna5::kVAddNcU64Vop3, UINT64_MAX, 2, 0, UINT64_MAX},
          {cdna5::kVSubNcU64Vop3, 1, 2, 0, 0},
          {cdna5::kVMulU64Vop3, UINT64_MAX, 2, 0, UINT64_MAX - 1},
          {cdna5::kVMinU64Vop3, 1ULL << 63, 1, 0, 1},
          {cdna5::kVMaxU64Vop3, 1ULL << 63, 1, 0, 1ULL << 63},
          {cdna5::kVMinI64Vop3, 1ULL << 63, 1, 0, 1ULL << 63},
          {cdna5::kVMaxI64Vop3, 1ULL << 63, 1, 0, 1},
      };
      for (int index = 0; index < (arch == 2 ? 8 : 1); ++index) {
        const auto &test = wide_cases[index];
        SCOPED_TRACE(test.opcode);
        for (unsigned lane = 0; lane < lanes; ++lane) {
          cu->write_vgpr(base, lane, uint32_t(test.a));
          cu->write_vgpr(base + 1, lane, uint32_t(test.a >> 32));
          cu->write_vgpr(base + 2, lane, uint32_t(test.b));
          cu->write_vgpr(base + 3, lane, uint32_t(test.b >> 32));
          cu->write_vgpr(base + 4, lane, uint32_t(test.c));
          cu->write_vgpr(base + 5, lane, uint32_t(test.c >> 32));
        }
        auto inst = decoder->decode_window(build_wide(arch, test.opcode));
        ASSERT_TRUE(inst.succeeded());
        ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *wave).succeeded());
        for (unsigned lane = 0; lane < lanes; ++lane) {
          uint64_t expected = (active >> lane) & 1 ? test.expected : test.a;
          EXPECT_EQ(cu->read_vgpr(base, lane), uint32_t(expected));
          EXPECT_EQ(cu->read_vgpr(base + 1, lane), uint32_t(expected >> 32));
        }
      }
      // Packed addition selects opposite input halves and saturates both output halves.
      for (unsigned lane = 0; lane < lanes; ++lane) {
        cu->write_vgpr(base, lane, 0xfffe0002U);
        cu->write_vgpr(base + 1, lane, 0xffff0003U);
      }
      auto packed_words =
          arch == 0
              ? cdna3::build_vop3p(
                    cdna3::kVPkAddU16Vop3p,
                    {.vdst = 0, .op_sel = 1, .clamp = 1, .src0 = 256, .src1 = 257, .op_sel_hi = 2})
          : arch == 1
              ? cdna4::build_vop3p(
                    cdna4::kVPkAddU16Vop3p,
                    {.vdst = 0, .op_sel = 1, .clamp = 1, .src0 = 256, .src1 = 257, .op_sel_hi = 2})
              : cdna5::build_vop3p(
                    cdna5::kVPkAddU16Vop3p,
                    {.vdst = 0, .opsel = 1, .clamp = 1, .src0 = 256, .src1 = 257, .opsel_hi = 2});
      auto packed_inst = decoder->decode_window(packed_words);
      ASSERT_TRUE(packed_inst.succeeded());
      ASSERT_TRUE(cu->execute_instruction(packed_inst.value().get(), *wave).succeeded());
      for (unsigned lane = 0; lane < lanes; ++lane) {
        EXPECT_EQ(cu->read_vgpr(base, lane), (active >> lane) & 1 ? 0xffffffffU : 0xfffe0002U);
      }
      for (bool wide : {false, true}) {
        constexpr uint64_t input_carry = 0x5555555555555555ULL;
        cu->write_sgpr(sb, uint32_t(input_carry));
        cu->write_sgpr(sb + 1, uint32_t(input_carry >> 32));
        for (unsigned lane = 0; lane < lanes; ++lane) {
          cu->write_vgpr(base, lane, 0xffffffffU);
          cu->write_vgpr(base + 1, lane, wide ? 0xffffffffU : 0);
          cu->write_vgpr(base + 2, lane, 0xffffffffU);
          cu->write_vgpr(base + 3, lane, 0xffffffffU);
        }
        auto words = build_carry(arch, wide);
        auto inst = decoder->decode_window(words);
        ASSERT_TRUE(inst.succeeded());
        ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *wave).succeeded());
        uint64_t expected_carry = active & (wide ? ~0ULL : input_carry);
        EXPECT_EQ(cu->read_sgpr(sb), uint32_t(expected_carry));
        EXPECT_EQ(cu->read_sgpr(sb + 1),
                  lanes == 64 ? uint32_t(expected_carry >> 32) : uint32_t(input_carry >> 32));
        for (unsigned lane = 0; lane < lanes; ++lane) {
          bool written = (active >> lane) & 1;
          EXPECT_EQ(cu->read_vgpr(base + (wide ? 2 : 0), lane),
                    written && (wide || ((input_carry >> lane) & 1)) ? 0U : 0xffffffffU);
          if (wide) {
            EXPECT_EQ(cu->read_vgpr(base + 3, lane), written ? 0xfffffffeU : 0xffffffffU);
          }
        }
      }
    }
  }
}
} // namespace
} // namespace rocjitsu::amdgpu
