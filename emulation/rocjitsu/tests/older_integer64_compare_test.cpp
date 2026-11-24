// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded older-target 64-bit comparisons: signedness, wave masks and aliased scalar output.
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
TEST(OlderIntegerCompare, DoublewordMasksAndAliasing) {
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<uint32_t, 2> instructions[5];
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {{cdna1::build_vopc(cdna1::kVCmpLtU64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        {cdna1::build_vopc(cdna1::kVCmpLtI64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        {cdna1::build_vopc(cdna1::kVCmpxEqU64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        cdna1::build_vop3(cdna1::kVCmpNeU64Vop3, {.vdst = 0, .src0 = 0, .src1 = 258}),
        {cdna1::build_vopc(cdna1::kVCmpTU64Vopc, {.src0 = 511, .vsrc1 = 255})[0], 0}}},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {{cdna2::build_vopc(cdna2::kVCmpLtU64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        {cdna2::build_vopc(cdna2::kVCmpLtI64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        {cdna2::build_vopc(cdna2::kVCmpxEqU64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        cdna2::build_vop3(cdna2::kVCmpNeU64Vop3, {.vdst = 0, .src0 = 0, .src1 = 258}),
        {cdna2::build_vopc(cdna2::kVCmpTU64Vopc, {.src0 = 511, .vsrc1 = 255})[0], 0}}},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {{rdna1::build_vopc(rdna1::kVCmpLtU64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        {rdna1::build_vopc(rdna1::kVCmpLtI64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        {rdna1::build_vopc(rdna1::kVCmpxEqU64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        rdna1::build_vop3(rdna1::kVCmpNeU64Vop3, {.vdst = 0, .src0 = 0, .src1 = 258}),
        {rdna1::build_vopc(rdna1::kVCmpTU64Vopc, {.src0 = 511, .vsrc1 = 255})[0], 0}}},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {{rdna2::build_vopc(rdna2::kVCmpLtU64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        {rdna2::build_vopc(rdna2::kVCmpLtI64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        {rdna2::build_vopc(rdna2::kVCmpxEqU64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        rdna2::build_vop3(rdna2::kVCmpNeU64Vop3, {.vdst = 0, .src0 = 0, .src1 = 258}),
        {rdna2::build_vopc(rdna2::kVCmpTU64Vopc, {.src0 = 511, .vsrc1 = 255})[0], 0}}},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {{rdna3_5::build_vopc(rdna3_5::kVCmpLtU64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        {rdna3_5::build_vopc(rdna3_5::kVCmpLtI64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        {rdna3_5::build_vopc(rdna3_5::kVCmpxEqU64Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        rdna3_5::build_vop3(rdna3_5::kVCmpNeU64Vop3, {.vdst = 0, .src0 = 0, .src1 = 258}),
        {rdna3_5::build_vopc(rdna3_5::kVCmpTU64Vopc, {.src0 = 511, .vsrc1 = 255})[0], 0}}},
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
        constexpr uint32_t untouched = 0x12345678U;
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
            cu->write_sgpr(sb + 1, 0x12345678U);
            for (unsigned lane = 0; lane < lanes; ++lane) {
              const uint64_t values[] = {0x8000000000000000ULL, 0, ~0ULL, 1};
              const uint64_t a = values[lane % 4];
              // Both words participate: the aliased SGPR source has a nonzero high word.
              const uint64_t b = operation == 3 ? 0x1234567800000000ULL + (lane & 1 ? 7 : 5) : 1;
              cu->write_vgpr(vb, lane, uint32_t(a));
              cu->write_vgpr(vb + 1, lane, uint32_t(a >> 32));
              cu->write_vgpr(vb + 2, lane, uint32_t(b));
              cu->write_vgpr(vb + 3, lane, uint32_t(b >> 32));
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
