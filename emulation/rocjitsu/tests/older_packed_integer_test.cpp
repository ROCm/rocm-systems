// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older-profile halfword and packed integer arithmetic: selection, saturation and overlap.
#include "rocjitsu/code/rj_code.h"
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
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <gtest/gtest.h>

namespace rocjitsu::amdgpu {
namespace {
// All operations alias the destination with A; inactive lanes retain its whole word.
TEST(OlderPackedInteger, Arithmetic) {
  struct Family {
    rj_code_arch_t arch;
    const IsaExecutionBackend *backend;
    bool multiply_clamps;
    std::array<std::array<uint32_t, 2>, 8> instructions;
    std::array<uint16_t, 6> selections; // Signed/unsigned MIN3, MAX3, MED3.
  };
  const Family families[] = {
      {ROCJITSU_CODE_ARCH_CDNA1,
       &cdna1::execution_backend(),
       false,
       {cdna1::build_vop3(cdna1::kVAddU16Vop3, {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257}),
        cdna1::build_vop3(cdna1::kVMadU16Vop3,
                          {.vdst = 0, .op_sel = 1, .src0 = 256, .src1 = 257, .src2 = 258}),
        cdna1::build_vop3(cdna1::kVMadU16Vop3,
                          {.vdst = 0, .op_sel = 9, .src0 = 256, .src1 = 257, .src2 = 258}),
        cdna1::build_vop3p(cdna1::kVPkMulLoU16Vop3p,
                           {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .op_sel_hi = 3}),
        cdna1::build_vop3(cdna1::kVCvtPkU16U32Vop3, {.vdst = 0, .src0 = 256, .src1 = 257}),
        cdna1::build_vop3(cdna1::kVMadU32U16Vop3,
                          {.vdst = 0, .op_sel = 3, .src0 = 256, .src1 = 257, .src2 = 258}),
        cdna1::build_vop3(cdna1::kVSubrevU16Vop3, {.vdst = 0, .src0 = 256, .src1 = 257}),
        cdna1::build_vop3(cdna1::kVMadLegacyU16Vop3,
                          {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = 258})},
       {cdna1::kVMin3I16Vop3, cdna1::kVMin3U16Vop3, cdna1::kVMax3I16Vop3, cdna1::kVMax3U16Vop3,
        cdna1::kVMed3I16Vop3, cdna1::kVMed3U16Vop3}},
      {ROCJITSU_CODE_ARCH_CDNA2,
       &cdna2::execution_backend(),
       true,
       {cdna2::build_vop3(cdna2::kVAddU16Vop3, {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257}),
        cdna2::build_vop3(cdna2::kVMadU16Vop3,
                          {.vdst = 0, .op_sel = 1, .src0 = 256, .src1 = 257, .src2 = 258}),
        cdna2::build_vop3(cdna2::kVMadU16Vop3,
                          {.vdst = 0, .op_sel = 9, .src0 = 256, .src1 = 257, .src2 = 258}),
        cdna2::build_vop3p(cdna2::kVPkMulLoU16Vop3p,
                           {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .op_sel_hi = 3}),
        cdna2::build_vop3(cdna2::kVCvtPkU16U32Vop3, {.vdst = 0, .src0 = 256, .src1 = 257}),
        cdna2::build_vop3(cdna2::kVMadU32U16Vop3,
                          {.vdst = 0, .op_sel = 3, .src0 = 256, .src1 = 257, .src2 = 258}),
        cdna2::build_vop3(cdna2::kVSubrevU16Vop3, {.vdst = 0, .src0 = 256, .src1 = 257}),
        cdna2::build_vop3(cdna2::kVMadLegacyU16Vop3,
                          {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = 258})},
       {cdna2::kVMin3I16Vop3, cdna2::kVMin3U16Vop3, cdna2::kVMax3I16Vop3, cdna2::kVMax3U16Vop3,
        cdna2::kVMed3I16Vop3, cdna2::kVMed3U16Vop3}},
      {ROCJITSU_CODE_ARCH_RDNA1,
       &rdna1::execution_backend(),
       false,
       {rdna1::build_vop3(rdna1::kVAddNcU16Vop3, {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257}),
        rdna1::build_vop3(rdna1::kVMadU16Vop3,
                          {.vdst = 0, .op_sel = 1, .src0 = 256, .src1 = 257, .src2 = 258}),
        rdna1::build_vop3(rdna1::kVMadU16Vop3,
                          {.vdst = 0, .op_sel = 9, .src0 = 256, .src1 = 257, .src2 = 258}),
        rdna1::build_vop3p(rdna1::kVPkMulLoU16Vop3p,
                           {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .op_sel_hi = 3}),
        rdna1::build_vop3(rdna1::kVCvtPkU16U32Vop3, {.vdst = 0, .src0 = 256, .src1 = 257}),
        rdna1::build_vop3(rdna1::kVMadU32U16Vop3,
                          {.vdst = 0, .op_sel = 3, .src0 = 256, .src1 = 257, .src2 = 258}),
        rdna1::build_vop3(rdna1::kVSubNcU16Vop3, {.vdst = 0, .src0 = 257, .src1 = 256}),
        {}},
       {rdna1::kVMin3I16Vop3, rdna1::kVMin3U16Vop3, rdna1::kVMax3I16Vop3, rdna1::kVMax3U16Vop3,
        rdna1::kVMed3I16Vop3, rdna1::kVMed3U16Vop3}},
      {ROCJITSU_CODE_ARCH_RDNA2,
       &rdna2::execution_backend(),
       true,
       {rdna2::build_vop3(rdna2::kVAddNcU16Vop3, {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257}),
        rdna2::build_vop3(rdna2::kVMadU16Vop3,
                          {.vdst = 0, .op_sel = 1, .src0 = 256, .src1 = 257, .src2 = 258}),
        rdna2::build_vop3(rdna2::kVMadU16Vop3,
                          {.vdst = 0, .op_sel = 9, .src0 = 256, .src1 = 257, .src2 = 258}),
        rdna2::build_vop3p(rdna2::kVPkMulLoU16Vop3p,
                           {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .op_sel_hi = 3}),
        rdna2::build_vop3(rdna2::kVCvtPkU16U32Vop3, {.vdst = 0, .src0 = 256, .src1 = 257}),
        rdna2::build_vop3(rdna2::kVMadU32U16Vop3,
                          {.vdst = 0, .op_sel = 3, .src0 = 256, .src1 = 257, .src2 = 258}),
        rdna2::build_vop3(rdna2::kVSubNcU16Vop3, {.vdst = 0, .src0 = 257, .src1 = 256}),
        {}},
       {rdna2::kVMin3I16Vop3, rdna2::kVMin3U16Vop3, rdna2::kVMax3I16Vop3, rdna2::kVMax3U16Vop3,
        rdna2::kVMed3I16Vop3, rdna2::kVMed3U16Vop3}},
      {ROCJITSU_CODE_ARCH_RDNA3_5,
       &rdna3_5::execution_backend(),
       false,
       {rdna3_5::build_vop3(rdna3_5::kVAddNcU16Vop3,
                            {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257}),
        rdna3_5::build_vop3(rdna3_5::kVMadU16Vop3,
                            {.vdst = 0, .op_sel = 1, .src0 = 256, .src1 = 257, .src2 = 258}),
        rdna3_5::build_vop3(rdna3_5::kVMadU16Vop3,
                            {.vdst = 0, .op_sel = 9, .src0 = 256, .src1 = 257, .src2 = 258}),
        rdna3_5::build_vop3p(rdna3_5::kVPkMulLoU16Vop3p,
                             {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .op_sel_hi = 3}),
        rdna3_5::build_vop3(rdna3_5::kVCvtPkU16U32Vop3, {.vdst = 0, .src0 = 256, .src1 = 257}),
        rdna3_5::build_vop3(rdna3_5::kVMadU32U16Vop3,
                            {.vdst = 0, .op_sel = 3, .src0 = 256, .src1 = 257, .src2 = 258}),
        rdna3_5::build_vop3(rdna3_5::kVSubNcU16Vop3, {.vdst = 0, .src0 = 257, .src1 = 256}),
        {}},
       {rdna3_5::kVMin3I16Vop3, rdna3_5::kVMin3U16Vop3, rdna3_5::kVMax3I16Vop3,
        rdna3_5::kVMax3U16Vop3, rdna3_5::kVMed3I16Vop3, rdna3_5::kVMed3U16Vop3}},
  };
  struct Case {
    uint32_t a, b, c, expected;
  };
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    for (unsigned lanes : {32U, 64U}) {
      if (lanes == 32 &&
          (family.arch == ROCJITSU_CODE_ARCH_CDNA1 || family.arch == ROCJITSU_CODE_ARCH_CDNA2)) {
        continue;
      }
      for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
        ComputeUnitCore::Config config{};
        config.arch = family.arch;
        config.arithmetic = {semantics};
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 104;
        config.vgprs_per_wf = 16;
        config.lds_size_kb = 64;
        GpuMemory memory("older_packed_memory");
        L2Cache l2("older_packed_l2");
        auto cu = ComputeUnitCore::create("older_packed", config, &memory, &l2);
        auto *wave = cu->dispatch_wf(0, 0, 104, 16, lanes);
        ASSERT_NE(wave, nullptr);
        auto decoder = Decoder::create(family.arch);
        const auto base = wave->vgpr_alloc().base;
        const Case cases[] = {
            {0xabcdfffeU, 3, 0,
             (family.arch == ROCJITSU_CODE_ARCH_CDNA1 || family.arch == ROCJITSU_CODE_ARCH_CDNA2)
                 ? 0xffffU
                 : 0xabcdffffU},
            {0x0003fffeU, 2, 5, 0x0003000bU},
            {0x0003fffeU, 2, 5, 0x000bfffeU},
            {0x0002ffffU, 0x00030002U, 0, family.multiply_clamps ? 0x0006ffffU : 0x0006fffeU},
            {0xfffffffeU, 123, 0, 0x007bffffU},
            {0x0003ffffU, 0x00040002U, 17, 29},
            {0x00090003U, 0x00020005U, 0,
             (family.arch == ROCJITSU_CODE_ARCH_CDNA1 || family.arch == ROCJITSU_CODE_ARCH_CDNA2)
                 ? 2U
                 : 0x00090002U},
            {0xffffU, 2, 0, family.arch == ROCJITSU_CODE_ARCH_CDNA2 ? 0xffffU : 0xfffeU},
        };
        for (uint64_t exec_mask : {0x8000000380000005ULL, 0ULL}) {
          wave->set_exec(exec_mask);
          for (int operation = 0; operation < 8; ++operation) {
            if (operation == 7 && family.arch != ROCJITSU_CODE_ARCH_CDNA1 &&
                family.arch != ROCJITSU_CODE_ARCH_CDNA2) {
              continue;
            }
            SCOPED_TRACE(family.arch);
            SCOPED_TRACE(lanes);
            SCOPED_TRACE(operation);
            const auto &test = cases[operation];
            for (unsigned lane = 0; lane < lanes; ++lane) {
              cu->write_vgpr(base, lane, test.a);
              cu->write_vgpr(base + 1, lane, test.b);
              cu->write_vgpr(base + 2, lane, test.c);
            }
            auto decoded = decoder->decode_window(family.instructions[operation]);
            ASSERT_TRUE(decoded.succeeded());
            ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
            for (unsigned lane = 0; lane < lanes; ++lane) {
              EXPECT_EQ(cu->read_vgpr(base, lane),
                        (wave->exec() >> lane) & 1 ? test.expected : test.a);
              EXPECT_EQ(cu->read_vgpr(base + 1, lane), test.b);
              EXPECT_EQ(cu->read_vgpr(base + 2, lane), test.c);
            }
          }
          if (family.arch == ROCJITSU_CODE_ARCH_RDNA2) {
            // Native binary OPSEL chooses A/B independently and preserves the other destination
            // half.
            for (uint8_t opsel : {0, 1, 2, 3, 8, 9, 10, 11}) {
              for (unsigned lane = 0; lane < lanes; ++lane) {
                cu->write_vgpr(base, lane, 0x00030009);
                cu->write_vgpr(base + 1, lane, 0x00040002);
                cu->write_vgpr(base + 4, lane, 0xdeadbeef);
              }
              const auto words = rdna2::build_vop3(
                  rdna2::kVAddNcU16Vop3, {.vdst = 4, .op_sel = opsel, .src0 = 256, .src1 = 257});
              auto decoded = decoder->decode_window(words);
              ASSERT_TRUE(decoded.succeeded());
              ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
              const uint32_t sum = (opsel & 1 ? 3 : 9) + (opsel & 2 ? 4 : 2);
              const uint32_t expected = opsel & 8 ? (sum << 16) | 0xbeef : 0xdead0000 | sum;
              for (unsigned lane = 0; lane < lanes; ++lane)
                EXPECT_EQ(cu->read_vgpr(base + 4, lane),
                          (wave->exec() >> lane) & 1 ? expected : 0xdeadbeefU);
            }
          }
          // Distinct high/low triples catch lost selectors and destination-byte mismatches.
          const bool cdna =
              family.arch == ROCJITSU_CODE_ARCH_CDNA1 || family.arch == ROCJITSU_CODE_ARCH_CDNA2;
          const uint32_t prefix = cdna ? 0xd0000000U : 0xd4000000U;
          const uint16_t low[] = {1, 1, 9, 9, 2, 2};
          const uint16_t high[] = {0x8000, 4, 4, 0xffff, 0xffff, 0x8000};
          for (int operation = 0; operation < 6; ++operation) {
            for (uint8_t opsel : {0, 15}) {
              // gfx1010's legacy executor does not yet honor these selectors;
              // do not turn that known omission into a required numeric result.
              if (opsel && family.arch == ROCJITSU_CODE_ARCH_RDNA1)
                continue;
              const bool high_half = opsel && family.arch != ROCJITSU_CODE_ARCH_RDNA1;
              const uint32_t value = high_half ? high[operation] : low[operation];
              const uint32_t expected =
                  high_half ? (value << 16) | 9U : (cdna ? 0U : 0xffff0000U) | value;
              for (unsigned lane = 0; lane < lanes; ++lane) {
                cu->write_vgpr(base, lane, 0xffff0009);
                cu->write_vgpr(base + 1, lane, 0x00040002);
                cu->write_vgpr(base + 2, lane, 0x80000001);
              }
              const std::array words{prefix | (uint32_t(family.selections[operation]) << 16) |
                                         (uint32_t(opsel) << 11),
                                     256U | (257U << 9) | (258U << 18)};
              auto decoded = decoder->decode_window(words);
              ASSERT_TRUE(decoded.succeeded());
              ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
              for (unsigned lane = 0; lane < lanes; ++lane) {
                EXPECT_EQ(cu->read_vgpr(base, lane),
                          (wave->exec() >> lane) & 1 ? expected : 0xffff0009U)
                    << family.arch << ',' << operation << ',' << unsigned(opsel) << ',' << lane;
              }
            }
          }
        }
      }
    }
  }
}
// gfx1150's inherited MOV copies selected raw halves, ignoring encoded ABS/NEG.
TEST(OlderPackedInteger, Gfx1150MovePreservesRawHalves) {
  ScopedIsaExecutionBackend backend(&rdna3_5::execution_backend());
  for (unsigned lanes : {32U, 64U}) {
    for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
      ComputeUnitCore::Config config{};
      config.arch = ROCJITSU_CODE_ARCH_RDNA3_5;
      config.arithmetic = {semantics};
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 104;
      config.vgprs_per_wf = 8;
      config.lds_size_kb = 64;
      GpuMemory memory("half_move_memory");
      L2Cache l2("half_move_l2");
      auto cu = ComputeUnitCore::create("half_move", config, &memory, &l2);
      auto *wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
      ASSERT_NE(wave, nullptr);
      auto decoder = Decoder::create(config.arch);
      const auto base = wave->vgpr_alloc().base;
      for (uint8_t opsel : {0, 1, 8, 9}) {
        for (uint8_t signs : {0, 1, 2, 3}) {
          auto words = rdna3_5::build_vop3(rdna3_5::kVMovB16Vop3, {.vdst = 0,
                                                                   .abs = uint8_t(signs & 1),
                                                                   .op_sel = opsel,
                                                                   .src0 = 256,
                                                                   .neg = uint8_t(signs >> 1)});
          auto decoded = decoder->decode_window(words);
          ASSERT_TRUE(decoded.succeeded());
          const uint64_t exec_mask = 0x8000000380000005ULL;
          wave->set_exec(exec_mask);
          wave->set_vgpr_write_mask(~4ULL);
          for (unsigned lane = 0; lane < lanes; ++lane) {
            cu->write_vgpr(base, lane, 0x81237fedU);
          }
          ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
          for (unsigned lane = 0; lane < lanes; ++lane) {
            const uint32_t half = opsel & 1 ? 0x8123U : 0x7fedU;
            const uint32_t expected = opsel & 8 ? (half << 16) | 0x7fedU : 0x81230000U | half;
            EXPECT_EQ(cu->read_vgpr(base, lane),
                      ((exec_mask & ~4ULL) >> lane) & 1 ? expected : 0x81237fedU)
                << unsigned(opsel) << ',' << unsigned(signs) << ',' << lane;
          }
        }
      }
    }
  }
}
} // namespace
} // namespace rocjitsu::amdgpu
