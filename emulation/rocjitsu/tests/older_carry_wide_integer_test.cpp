// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older-profile carry and multiword arithmetic with sparse EXEC and overlapping registers.
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
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/execution_backend.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <bit>
#include <cstdint>
#include <gtest/gtest.h>

namespace rocjitsu::amdgpu {
namespace {

// Carry input/output aliases SGPR0; wide results overlap vector inputs.
TEST(OlderCarryWideInteger, ArithmeticAndAliases) {
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<std::array<uint32_t, 2>, 4> instructions;
    std::array<uint32_t, 2> sdwa;
    std::array<uint32_t, 2> dpp{};
  };
  const auto sdwa_words = []<typename Raw>(uint16_t opcode) {
    Raw raw{};
    raw.op = opcode;
    raw.src0 = SRC_SDWA;
    raw.vsrc0 = 0;
    raw.vsrc1 = 1;
    raw.vdst = 0;
    raw.src0_sel = raw.src1_sel = sdwa::DWORD;
    raw.dst_sel = sdwa::WORD_1;
    raw.dst_unused = sdwa::UNUSED_PAD;
    return std::bit_cast<std::array<uint32_t, 2>>(raw);
  };
  const auto dpp_words = []<typename Raw>(uint16_t opcode) {
    Raw raw{};
    raw.op = opcode;
    raw.src0 = SRC_DPP;
    raw.vsrc0 = 0;
    raw.vsrc1 = 1;
    raw.vdst = 2;
    raw.dpp_ctrl = 0xe4; // Identity quad permutation.
    raw.row_mask = 0xf;
    raw.bank_mask = 0xe;
    return std::bit_cast<std::array<uint32_t, 2>>(raw);
  };
  const Family families[] = {
    {rocjitsu::IsaFamily::Cdna1,
     &cdna1::execution_backend(),
     {cdna1::build_vop3_sdst_enc(cdna1::kVAddcCoU32Vop3SdstEnc,
                                 {.vdst = 0, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = 0}),
      cdna1::build_vop3_sdst_enc(cdna1::kVMadU64U32Vop3SdstEnc,
                                 {.vdst = 2, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = 258}),
      cdna1::build_vop3(cdna1::kVLshrrevB64Vop3, {.vdst = 0, .src0 = 260, .src1 = 256}),
      cdna1::build_vop3(cdna1::kVMqsadU32U8Vop3,
                        {.vdst = 3, .src0 = 256, .src1 = 262, .src2 = 258})},
     sdwa_words.template operator()<cdna1::Vop2VopSdwaMachineInst>(cdna1::kVAddcCoU32Vop2)},
    {rocjitsu::IsaFamily::Cdna2,
     &cdna2::execution_backend(),
     {cdna2::build_vop3_sdst_enc(cdna2::kVAddcCoU32Vop3SdstEnc,
                                 {.vdst = 0, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = 0}),
      cdna2::build_vop3_sdst_enc(cdna2::kVMadU64U32Vop3SdstEnc,
                                 {.vdst = 2, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = 258}),
      cdna2::build_vop3(cdna2::kVLshrrevB64Vop3, {.vdst = 0, .src0 = 260, .src1 = 256}),
      cdna2::build_vop3(cdna2::kVMqsadU32U8Vop3,
                        {.vdst = 3, .src0 = 256, .src1 = 262, .src2 = 258})},
     sdwa_words.template operator()<cdna2::Vop2VopSdwaMachineInst>(cdna2::kVAddcCoU32Vop2)},
    {rocjitsu::IsaFamily::Rdna1,
     &rdna1::execution_backend(),
     {rdna1::build_vop3_sdst_enc(rdna1::kVAddCoCiU32Vop3SdstEnc,
                                 {.vdst = 0, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = 0}),
      rdna1::build_vop3_sdst_enc(rdna1::kVMadU64U32Vop3SdstEnc,
                                 {.vdst = 2, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = 258}),
      rdna1::build_vop3(rdna1::kVLshrrevB64Vop3, {.vdst = 0, .src0 = 260, .src1 = 256}),
      rdna1::build_vop3(rdna1::kVMqsadU32U8Vop3,
                        {.vdst = 3, .src0 = 256, .src1 = 262, .src2 = 258})},
     sdwa_words.template operator()<rdna1::Vop2VopSdwaMachineInst>(rdna1::kVAddCoCiU32Vop2),
     dpp_words.template operator()<rdna1::Vop2VopDpp16MachineInst>(rdna1::kVAddCoCiU32Vop2)},
    {rocjitsu::IsaFamily::Rdna2,
     &rdna2::execution_backend(),
     {rdna2::build_vop3_sdst_enc(rdna2::kVAddCoCiU32Vop3SdstEnc,
                                 {.vdst = 0, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = 0}),
      rdna2::build_vop3_sdst_enc(rdna2::kVMadU64U32Vop3SdstEnc,
                                 {.vdst = 2, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = 258}),
      rdna2::build_vop3(rdna2::kVLshrrevB64Vop3, {.vdst = 0, .src0 = 260, .src1 = 256}),
      rdna2::build_vop3(rdna2::kVMqsadU32U8Vop3,
                        {.vdst = 3, .src0 = 256, .src1 = 262, .src2 = 258})},
     sdwa_words.template operator()<rdna2::Vop2VopSdwaMachineInst>(rdna2::kVAddCoCiU32Vop2),
     dpp_words.template operator()<rdna2::Vop2VopDpp16MachineInst>(rdna2::kVAddCoCiU32Vop2)},
    {rocjitsu::IsaFamily::Rdna3_5,
     &rdna3_5::execution_backend(),
     {rdna3_5::build_vop3_sdst_enc(rdna3_5::kVAddCoCiU32Vop3SdstEnc,
                                   {.vdst = 0, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = 0}),
      rdna3_5::build_vop3_sdst_enc(rdna3_5::kVMadU64U32Vop3SdstEnc,
                                   {.vdst = 2, .sdst = 0, .src0 = 256, .src1 = 257, .src2 = 258}),
      rdna3_5::build_vop3(rdna3_5::kVLshrrevB64Vop3, {.vdst = 0, .src0 = 260, .src1 = 256}),
      rdna3_5::build_vop3(rdna3_5::kVMqsadU32U8Vop3,
                          {.vdst = 3, .src0 = 256, .src1 = 262, .src2 = 258})},
     {}},
  };
  struct Case {
    int destination, words;
    std::array<uint32_t, 8> input;
    std::array<uint32_t, 4> result;
  };
  const Case cases[] = {
      {0, 1, {0xffffffffU, 0}, {}},
      {2, 2, {0xffffffffU, 2, 0xffffffffU, 0xffffffffU}, {0xfffffffdU, 1}},
      {0, 2, {1, 0x80000000U, 0, 0, 65}, {0, 0x40000000U}},
      {3, 4, {0x03020100U, 0x07060504U, 100, 200, 300, 400, 0x04000300U}, {103, 201, 301, 403}},
  };
  for (const auto &family : families) {
    SCOPED_TRACE(family.arch);
    ScopedIsaExecutionBackend backend(family.backend);
    for (unsigned lanes : {32U, 64U}) {
      if (lanes == 32 && (family.arch == rocjitsu::IsaFamily::Cdna1 ||
                          family.arch == rocjitsu::IsaFamily::Cdna2)) {
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
        GpuMemory memory("older_wide_memory");
        L2Cache l2("older_wide_l2");
        auto cu = ComputeUnitCore::create("older_wide", config, &memory, &l2);
        auto *wave = cu->dispatch_wf(0, 0, 104, 16, lanes);
        ASSERT_NE(wave, nullptr);
        auto decoder = Decoder::create(family.arch);
        const uint32_t vb = wave->vgpr_alloc().base, sb = wave->sgpr_alloc().base;
        constexpr uint64_t input_carry = 0xaaaaaaaaaaaaaaaaULL;
        for (uint64_t exec_mask : {0x8000000380000005ULL, 0ULL}) {
          wave->set_exec(exec_mask);
          const uint64_t active = wave->exec();
          for (int operation = 0; operation < 4; ++operation) {
            SCOPED_TRACE(operation);
            const auto &test = cases[operation];
            for (unsigned lane = 0; lane < lanes; ++lane) {
              for (int reg = 0; reg < 8; ++reg) {
                cu->write_vgpr(vb + reg, lane, test.input[reg]);
              }
            }
            cu->write_sgpr(sb, uint32_t(input_carry));
            cu->write_sgpr(sb + 1, uint32_t(input_carry >> 32));
            auto decoded = decoder->decode_window(family.instructions[operation]);
            ASSERT_TRUE(decoded.succeeded());
            ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
            for (unsigned lane = 0; lane < lanes; ++lane) {
              for (int word = 0; word < test.words; ++word) {
                uint32_t expected = operation == 0 ? ((input_carry >> lane) & 1 ? 0U : 0xffffffffU)
                                                   : test.result[word];
                if (!((active >> lane) & 1)) {
                  expected = test.input[test.destination + word];
                }
                EXPECT_EQ(cu->read_vgpr(vb + test.destination + word, lane), expected);
              }
            }
            const uint64_t carry = operation == 0 ? active & input_carry : active;
            EXPECT_EQ(cu->read_sgpr(sb), uint32_t(operation < 2 ? carry : input_carry));
            EXPECT_EQ(cu->read_sgpr(sb + 1),
                      uint32_t(operation < 2 && lanes == 64 ? carry >> 32 : input_carry >> 32));
          }
          if (family.arch == rocjitsu::IsaFamily::Rdna1 ||
              family.arch == rocjitsu::IsaFamily::Rdna2) {
            // ROW/BANK masks suppress VGPR writes, not the carry computation.
            wave->set_exec(~uint64_t{0});
            wave->set_vcc_mask(0);
            for (unsigned lane = 0; lane < lanes; ++lane) {
              cu->write_vgpr(vb, lane, 0xffffffffU);
              cu->write_vgpr(vb + 1, lane, 1);
              cu->write_vgpr(vb + 2, lane, 0xfeed);
            }
            auto decoded = decoder->decode_window(family.dpp);
            ASSERT_TRUE(decoded.succeeded());
            ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
            EXPECT_EQ(wave->vcc_mask(), wave->exec());
            for (unsigned lane = 0; lane < lanes; ++lane) {
              EXPECT_EQ(cu->read_vgpr(vb + 2, lane), lane % 16 < 4 ? 0xfeedU : 0U);
            }
            wave->set_exec(exec_mask);
          }
          if (family.arch == rocjitsu::IsaFamily::Rdna3_5) {
            continue; // This profile has no SDWA encodings.
          }
          for (unsigned lane = 0; lane < lanes; ++lane) {
            cu->write_vgpr(vb, lane, 0xffffffffU);
            cu->write_vgpr(vb + 1, lane, 0);
          }
          wave->set_vcc_mask(input_carry);
          auto decoded = decoder->decode_window(family.sdwa);
          ASSERT_TRUE(decoded.succeeded());
          ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
          for (unsigned lane = 0; lane < lanes; ++lane) {
            EXPECT_EQ(cu->read_vgpr(vb, lane), ((active >> lane) & 1)
                                                   ? ((input_carry >> lane) & 1 ? 0U : 0xffff0000U)
                                                   : 0xffffffffU);
          }
          EXPECT_EQ(wave->vcc_mask(), active & input_carry);
        }
      }
    }
  }
}

// Missing CDNA2 accumulator words independently read V0, with or without indexing.
TEST(OlderCarryWideInteger, Cdna2MqsadAccumulatorFallback) {
  struct Case {
    bool indexed;
    uint16_t source;
    std::array<uint32_t, 4> expected;
    bool split = false;
  };
  const Case cases[] = {{false, 260, {40, 50, 60, 70}},
                        {false, 262, {60, 70, 10, 10}},
                        {false, 264, {10, 10, 10, 10}},
                        {true, 260, {10, 10, 10, 10}},
                        {false, 262, {60, 70, 10, 10}, true}};
  for (const auto &c : cases) {
    ScopedIsaExecutionBackend backend(&cdna2::execution_backend());
    ComputeUnitCore::Config config{};
    config.arch = rocjitsu::IsaFamily::Cdna2;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = c.split ? 16 : 8;
    config.lds_size_kb = 64;
    GpuMemory memory("indexed_qsad_memory");
    L2Cache l2("indexed_qsad_l2");
    auto cu = ComputeUnitCore::create("indexed_qsad", config, &memory, &l2);
    auto *wave = cu->dispatch_wf(0, 0, 104, config.vgprs_per_wf, 64);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(1);
    if (c.split) {
      wave->set_accum_offset(8);
    }
    const uint32_t vb = wave->vgpr_alloc().base;
    if (c.split) {
      // Physically resident AccVGPRs must not satisfy ordinary VGPR operands.
      cu->write_vgpr(vb + 8, 0, 0x108);
      cu->write_vgpr(vb + 9, 0, 0x109);
    }
    cu->write_vgpr(vb, 0, 10);
    cu->write_vgpr(vb + 1, 0, 20);
    cu->write_vgpr(vb + 2, 0, 30);
    cu->write_vgpr(vb + 3, 0, 0);
    cu->write_vgpr(vb + 4, 0, 40);
    cu->write_vgpr(vb + 5, 0, 50);
    cu->write_vgpr(vb + 6, 0, 60);
    cu->write_vgpr(vb + 7, 0, 70);
    cu->write_sgpr(wave->sgpr_alloc().base, 4);
    auto decoder = Decoder::create(config.arch);
    if (c.indexed) {
      const auto enable = cdna2::build_sopc(cdna2::kSSetGprIdxOnSopc, {.ssrc0 = 0, .ssrc1 = 4});
      auto decoded = decoder->decode_window(enable);
      ASSERT_TRUE(decoded.succeeded());
      ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
    }
    const auto words = cdna2::build_vop3(cdna2::kVMqsadU32U8Vop3,
                                         {.vdst = 4, .src0 = 256, .src1 = 259, .src2 = c.source});
    auto decoded = decoder->decode_window(words);
    ASSERT_TRUE(decoded.succeeded());
    ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
    for (int reg = 0; reg < 4; ++reg) {
      EXPECT_EQ(cu->read_vgpr(vb + 4 + reg, 0), c.expected[reg]);
    }
  }
}

} // namespace
} // namespace rocjitsu::amdgpu
