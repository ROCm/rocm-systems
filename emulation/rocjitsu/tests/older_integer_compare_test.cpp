// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded older-target 32-bit comparisons: signedness, wave masks and aliased scalar output.
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
#include <array>
#include <gtest/gtest.h>

namespace rocjitsu::amdgpu {
namespace {
// VOPC targets VCC; VOP3 writes SGPR0 over its scalar input. CMPX additionally changes EXEC.
TEST(OlderIntegerCompare, WordMasksAndAliasing) {
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<uint32_t, 2> instructions[5];
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {{cdna1::build_vopc(cdna1::kVCmpLtU32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        {cdna1::build_vopc(cdna1::kVCmpLtI32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        {cdna1::build_vopc(cdna1::kVCmpxEqU32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        cdna1::build_vop3(cdna1::kVCmpNeU32Vop3, {.vdst = 0, .src0 = 0, .src1 = 257}),
        {cdna1::build_vopc(cdna1::kVCmpTU32Vopc, {.src0 = 511, .vsrc1 = 255})[0], 0}}},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {{cdna2::build_vopc(cdna2::kVCmpLtU32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        {cdna2::build_vopc(cdna2::kVCmpLtI32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        {cdna2::build_vopc(cdna2::kVCmpxEqU32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        cdna2::build_vop3(cdna2::kVCmpNeU32Vop3, {.vdst = 0, .src0 = 0, .src1 = 257}),
        {cdna2::build_vopc(cdna2::kVCmpTU32Vopc, {.src0 = 511, .vsrc1 = 255})[0], 0}}},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {{rdna1::build_vopc(rdna1::kVCmpLtU32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        {rdna1::build_vopc(rdna1::kVCmpLtI32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        {rdna1::build_vopc(rdna1::kVCmpxEqU32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        rdna1::build_vop3(rdna1::kVCmpNeU32Vop3, {.vdst = 0, .src0 = 0, .src1 = 257}),
        {rdna1::build_vopc(rdna1::kVCmpTU32Vopc, {.src0 = 511, .vsrc1 = 255})[0], 0}}},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {{rdna2::build_vopc(rdna2::kVCmpLtU32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        {rdna2::build_vopc(rdna2::kVCmpLtI32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        {rdna2::build_vopc(rdna2::kVCmpxEqU32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        rdna2::build_vop3(rdna2::kVCmpNeU32Vop3, {.vdst = 0, .src0 = 0, .src1 = 257}),
        {rdna2::build_vopc(rdna2::kVCmpTU32Vopc, {.src0 = 511, .vsrc1 = 255})[0], 0}}},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {{rdna3_5::build_vopc(rdna3_5::kVCmpLtU32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        {rdna3_5::build_vopc(rdna3_5::kVCmpLtI32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        {rdna3_5::build_vopc(rdna3_5::kVCmpxEqU32Vopc, {.src0 = 256, .vsrc1 = 1})[0], 0},
        rdna3_5::build_vop3(rdna3_5::kVCmpNeU32Vop3, {.vdst = 0, .src0 = 0, .src1 = 257}),
        {rdna3_5::build_vopc(rdna3_5::kVCmpTU32Vopc, {.src0 = 511, .vsrc1 = 255})[0], 0}}},
  };
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    const bool cdna =
        family.arch == rocjitsu::IsaFamily::Cdna1 || family.arch == rocjitsu::IsaFamily::Cdna2;
    for (unsigned lanes : {32U, 64U}) {
      if (lanes == 32 && cdna) {
        continue;
      }
      const uint64_t lane_mask = lanes == 64 ? ~0ULL : 0xffffffffULL;
      for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
        ComputeUnitCore::Config config{};
        config.arch = family.arch;
        config.arithmetic = {semantics};
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 104;
        config.vgprs_per_wf = 16;
        config.lds_size_kb = 64;
        GpuMemory memory("older_compare_memory");
        L2Cache l2("older_compare_l2");
        auto cu = ComputeUnitCore::create("older_compare", config, &memory, &l2);
        auto *wave = cu->dispatch_wf(0, 0, 104, 16, lanes);
        ASSERT_NE(wave, nullptr);
        auto decoder = Decoder::create(family.arch);
        const auto vb = wave->vgpr_alloc().base, sb = wave->sgpr_alloc().base;
        constexpr uint64_t old_vcc = 0x55aac0de12345678ULL;
        constexpr uint32_t untouched = 0xa5a5c3c3U;
        // SDWA BYTE_0 with sign extension must compare -128, not the positive raw word.
        if (family.arch != rocjitsu::IsaFamily::Rdna3_5) {
          const std::array<uint32_t, 2> words = {(family.instructions[1][0] & ~0x1ffU) | 249U,
                                                 0x06080000U};
          wave->set_exec(1);
          cu->write_vgpr(vb, 0, 0x12345680U);
          cu->write_vgpr(vb + 1, 0, 0U);
          auto instruction = decoder->decode_window(words);
          ASSERT_TRUE(instruction.succeeded());
          ASSERT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
          EXPECT_EQ(wave->vcc(), 1U);
        }
        for (uint64_t exec_mask : {0x8000000380000005ULL, 0ULL}) {
          const uint64_t active = exec_mask & lane_mask;
          for (int operation = 0; operation < 5; ++operation) {
            SCOPED_TRACE(family.arch);
            SCOPED_TRACE(lanes);
            SCOPED_TRACE(operation);
            wave->set_exec(active);
            wave->set_vcc_raw(old_vcc);
            wave->write_scc(true);
            cu->write_sgpr(sb, 5);
            cu->write_sgpr(sb + 1, untouched);
            for (unsigned lane = 0; lane < lanes; ++lane) {
              const uint32_t values[] = {0x80000000U, 0, 0xffffffffU, 1};
              cu->write_vgpr(vb, lane, values[lane % 4]);
              cu->write_vgpr(vb + 1, lane, operation == 3 ? (lane & 1 ? 7 : 5) : 1);
            }
            uint64_t expected = 0;
            for (unsigned lane = 0; lane < lanes; ++lane) {
              const bool selected = operation == 0   ? lane % 4 == 1
                                    : operation == 1 ? lane % 4 != 3
                                    : operation == 2 ? lane % 4 == 3
                                    : operation == 3 ? bool(lane & 1)
                                                     : true;
              if (((active >> lane) & 1) && selected) {
                expected |= 1ULL << lane;
              }
            }
            auto decoded = decoder->decode_window(family.instructions[operation]);
            ASSERT_TRUE(decoded.succeeded());
            ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
            const bool writes_vcc = operation != 3 && (operation != 2 || cdna);
            EXPECT_EQ(wave->vcc(), writes_vcc ? (old_vcc & ~lane_mask) | expected : old_vcc);
            EXPECT_EQ(wave->exec(), operation == 2 ? expected : active);
            EXPECT_TRUE(wave->read_scc());
            if (operation == 3) {
              EXPECT_EQ(cu->read_sgpr(sb), uint32_t(expected));
              EXPECT_EQ(cu->read_sgpr(sb + 1), lanes == 64 ? uint32_t(expected >> 32) : untouched);
            }
          }
        }
      }
    }
  }
}
} // namespace
} // namespace rocjitsu::amdgpu
