// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older word copies and conditional selection: encoded signs, masks and destination aliases.
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
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {
// Computed raw-bit references preserve NaN payloads and distinguish explicit masks from VCC.
TEST(OlderWordRouting, CopiesAndConditionalSelection) {
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    bool cdna;
    std::vector<uint32_t> (*encode)(bool, bool, uint8_t);
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1, &cdna1::execution_backend(), true,
       [](bool conditional, bool e64, uint8_t destination) {
         if (e64) {
           auto words =
               cdna1::build_vop3(conditional ? cdna1::kVCndmaskB32Vop3 : cdna1::kVMovB32Vop3,
                                 {.vdst = destination,
                                  .abs = 1,
                                  .src0 = 256,
                                  .src1 = uint16_t(conditional ? 257 : 0),
                                  .src2 = uint16_t(conditional ? 4 : 0),
                                  .neg = uint8_t(conditional ? 2 : 0)});
           return std::vector<uint32_t>(words.begin(), words.end());
         }
         if (conditional) {
           auto words = cdna1::build_vop2(cdna1::kVCndmaskB32Vop2,
                                          {.src0 = 256, .vsrc1 = 1, .vdst = destination});
           return std::vector<uint32_t>(words.begin(), words.end());
         }
         auto words = cdna1::build_vop1(cdna1::kVMovB32Vop1, {.src0 = 256, .vdst = destination});
         return std::vector<uint32_t>(words.begin(), words.end());
       }},
      {rocjitsu::IsaFamily::Cdna2, &cdna2::execution_backend(), true,
       [](bool conditional, bool e64, uint8_t destination) {
         if (e64) {
           auto words =
               cdna2::build_vop3(conditional ? cdna2::kVCndmaskB32Vop3 : cdna2::kVMovB32Vop3,
                                 {.vdst = destination,
                                  .abs = 1,
                                  .src0 = 256,
                                  .src1 = uint16_t(conditional ? 257 : 0),
                                  .src2 = uint16_t(conditional ? 4 : 0),
                                  .neg = uint8_t(conditional ? 2 : 0)});
           return std::vector<uint32_t>(words.begin(), words.end());
         }
         if (conditional) {
           auto words = cdna2::build_vop2(cdna2::kVCndmaskB32Vop2,
                                          {.src0 = 256, .vsrc1 = 1, .vdst = destination});
           return std::vector<uint32_t>(words.begin(), words.end());
         }
         auto words = cdna2::build_vop1(cdna2::kVMovB32Vop1, {.src0 = 256, .vdst = destination});
         return std::vector<uint32_t>(words.begin(), words.end());
       }},
      {rocjitsu::IsaFamily::Rdna1, &rdna1::execution_backend(), false,
       [](bool conditional, bool e64, uint8_t destination) {
         if (e64) {
           auto words =
               rdna1::build_vop3(conditional ? rdna1::kVCndmaskB32Vop3 : rdna1::kVMovB32Vop3,
                                 {.vdst = destination,
                                  .abs = 1,
                                  .src0 = 256,
                                  .src1 = uint16_t(conditional ? 257 : 0),
                                  .src2 = uint16_t(conditional ? 4 : 0),
                                  .neg = uint8_t(conditional ? 2 : 0)});
           return std::vector<uint32_t>(words.begin(), words.end());
         }
         if (conditional) {
           auto words = rdna1::build_vop2(rdna1::kVCndmaskB32Vop2,
                                          {.src0 = 256, .vsrc1 = 1, .vdst = destination});
           return std::vector<uint32_t>(words.begin(), words.end());
         }
         auto words = rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = 256, .vdst = destination});
         return std::vector<uint32_t>(words.begin(), words.end());
       }},
      {rocjitsu::IsaFamily::Rdna2, &rdna2::execution_backend(), false,
       [](bool conditional, bool e64, uint8_t destination) {
         if (e64) {
           auto words =
               rdna2::build_vop3(conditional ? rdna2::kVCndmaskB32Vop3 : rdna2::kVMovB32Vop3,
                                 {.vdst = destination,
                                  .abs = 1,
                                  .src0 = 256,
                                  .src1 = uint16_t(conditional ? 257 : 0),
                                  .src2 = uint16_t(conditional ? 4 : 0),
                                  .neg = uint8_t(conditional ? 2 : 0)});
           return std::vector<uint32_t>(words.begin(), words.end());
         }
         if (conditional) {
           auto words = rdna2::build_vop2(rdna2::kVCndmaskB32Vop2,
                                          {.src0 = 256, .vsrc1 = 1, .vdst = destination});
           return std::vector<uint32_t>(words.begin(), words.end());
         }
         auto words = rdna2::build_vop1(rdna2::kVMovB32Vop1, {.src0 = 256, .vdst = destination});
         return std::vector<uint32_t>(words.begin(), words.end());
       }},
      {rocjitsu::IsaFamily::Rdna3_5, &rdna3_5::execution_backend(), false,
       [](bool conditional, bool e64, uint8_t destination) {
         if (e64) {
           auto words =
               rdna3_5::build_vop3(conditional ? rdna3_5::kVCndmaskB32Vop3 : rdna3_5::kVMovB32Vop3,
                                   {.vdst = destination,
                                    .abs = 1,
                                    .src0 = 256,
                                    .src1 = uint16_t(conditional ? 257 : 0),
                                    .src2 = uint16_t(conditional ? 4 : 0),
                                    .neg = uint8_t(conditional ? 2 : 0)});
           return std::vector<uint32_t>(words.begin(), words.end());
         }
         if (conditional) {
           auto words = rdna3_5::build_vop2(rdna3_5::kVCndmaskB32Vop2,
                                            {.src0 = 256, .vsrc1 = 1, .vdst = destination});
           return std::vector<uint32_t>(words.begin(), words.end());
         }
         auto words =
             rdna3_5::build_vop1(rdna3_5::kVMovB32Vop1, {.src0 = 256, .vdst = destination});
         return std::vector<uint32_t>(words.begin(), words.end());
       }},
  };
  // This named engine and high-bit extraction are specified independently of the C++ library.
  std::minstd_rand random;
  uint32_t input[3][64];
  for (auto &reg : input) {
    for (auto &word : reg) {
      const auto first_draw = random();
      word = (uint32_t(first_draw >> 15) << 16) | uint32_t(random() >> 15);
    }
  }
  input[0][0] = 0x80000000U;
  input[1][1] = 0xffc12345U;
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    for (unsigned lanes : {32U, 64U}) {
      if (family.cdna && lanes != 64) {
        continue;
      }
      for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
        ComputeUnitCore::Config config{};
        config.arch = family.arch;
        config.arithmetic = {semantics};
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 104;
        config.vgprs_per_wf = 8;
        config.lds_size_kb = 64;
        GpuMemory memory("word_routing_memory");
        L2Cache l2("word_routing_l2");
        auto cu = ComputeUnitCore::create("word_routing", config, &memory, &l2);
        auto *wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
        ASSERT_NE(wave, nullptr);
        auto decoder = Decoder::create(family.arch);
        const auto vb = wave->vgpr_alloc().base, sb = wave->sgpr_alloc().base;
        for (bool conditional : {false, true}) {
          for (bool e64 : {false, true}) {
            for (uint8_t destination : {0, 1, 2}) {
              for (uint64_t exec_mask : {~0ULL, 0xa55af00f0f0f00f3ULL, 0ULL}) {
                SCOPED_TRACE(family.arch);
                SCOPED_TRACE(conditional);
                SCOPED_TRACE(e64);
                SCOPED_TRACE(destination);
                exec_mask &= lanes == 32 ? 0xffffffffULL : ~0ULL;
                constexpr uint64_t condition = 0xcccc33335555aaaaULL;
                const uint64_t vcc = e64 ? ~condition : condition;
                wave->set_exec(exec_mask);
                wave->set_vgpr_write_mask(~2ULL);
                wave->set_vcc_raw(vcc);
                wave->write_scc(true);
                cu->write_sgpr(sb + 4, uint32_t(condition));
                cu->write_sgpr(sb + 5, uint32_t(condition >> 32));
                for (unsigned reg = 0; reg < 3; ++reg) {
                  for (unsigned lane = 0; lane < lanes; ++lane) {
                    cu->write_vgpr(vb + reg, lane, input[reg][lane]);
                  }
                }
                auto decoded = decoder->decode_window(family.encode(conditional, e64, destination));
                ASSERT_TRUE(decoded.succeeded());
                const auto diagnostics = cu->isa_diagnostic_count();
                ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
                EXPECT_EQ(cu->isa_diagnostic_count(), diagnostics);
                for (unsigned reg = 0; reg < 3; ++reg) {
                  for (unsigned lane = 0; lane < lanes; ++lane) {
                    uint32_t expected = input[reg][lane];
                    if (reg == destination && (((exec_mask & ~2ULL) >> lane) & 1)) {
                      bool b = conditional && ((condition >> lane) & 1);
                      expected = input[b ? 1 : 0][lane];
                      // Radeon MOV applies ABS; CDNA keeps its raw-bit copy policy.
                      // Conditional moves apply the selected source's modifier on both.
                      if (e64 && (conditional || !family.cdna)) {
                        expected = b ? expected ^ 0x80000000U : expected & 0x7fffffffU;
                      }
                    }
                    EXPECT_EQ(cu->read_vgpr(vb + reg, lane), expected) << reg << ',' << lane;
                  }
                }
                EXPECT_EQ(wave->exec(), exec_mask);
                EXPECT_EQ(wave->vcc(), vcc);
                EXPECT_TRUE(wave->read_scc());
              }
            }
          }
        }
      }
    }
  }
}
} // namespace
} // namespace rocjitsu::amdgpu
