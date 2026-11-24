// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded class masks across older targets, including input halves and aliased SGPR output.
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
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/execution_backend.h"
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
// Raw encodings distinguish all ten classes without host floating-point arithmetic.
TEST(OlderClassCompare, MasksAndAliasing) {
  struct Family {
    rj_code_arch_t arch;
    const IsaExecutionBackend *backend;
    std::array<uint32_t, 2> instructions[3];
  };
  const Family families[] = {
      {ROCJITSU_CODE_ARCH_CDNA1,
       &cdna1::execution_backend(),
       {{cdna1::build_vopc(cdna1::kVCmpClassF32Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        cdna1::build_vop3(cdna1::kVCmpClassF64Vop3, {.vdst = 0, .src0 = 0, .src1 = 258}),
        cdna1::build_vop3(cdna1::kVCmpxClassF16Vop3,
                          {.vdst = 0, .op_sel = 3, .src0 = 256, .src1 = 258, .neg = 1})}},
      {ROCJITSU_CODE_ARCH_CDNA2,
       &cdna2::execution_backend(),
       {{cdna2::build_vopc(cdna2::kVCmpClassF32Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        cdna2::build_vop3(cdna2::kVCmpClassF64Vop3, {.vdst = 0, .src0 = 0, .src1 = 258}),
        cdna2::build_vop3(cdna2::kVCmpxClassF16Vop3,
                          {.vdst = 0, .op_sel = 3, .src0 = 256, .src1 = 258, .neg = 1})}},
      {ROCJITSU_CODE_ARCH_RDNA1,
       &rdna1::execution_backend(),
       {{rdna1::build_vopc(rdna1::kVCmpClassF32Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        rdna1::build_vop3(rdna1::kVCmpClassF64Vop3, {.vdst = 0, .src0 = 0, .src1 = 258}),
        rdna1::build_vop3(rdna1::kVCmpxClassF16Vop3,
                          {.vdst = 126, .op_sel = 0, .src0 = 256, .src1 = 258, .neg = 1})}},
      {ROCJITSU_CODE_ARCH_RDNA2,
       &rdna2::execution_backend(),
       {{rdna2::build_vopc(rdna2::kVCmpClassF32Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        rdna2::build_vop3(rdna2::kVCmpClassF64Vop3, {.vdst = 0, .src0 = 0, .src1 = 258}),
        rdna2::build_vop3(rdna2::kVCmpxClassF16Vop3,
                          {.vdst = 126, .op_sel = 0, .src0 = 256, .src1 = 258, .neg = 1})}},
      {ROCJITSU_CODE_ARCH_RDNA3_5,
       &rdna3_5::execution_backend(),
       {{rdna3_5::build_vopc(rdna3_5::kVCmpClassF32Vopc, {.src0 = 256, .vsrc1 = 2})[0], 0},
        rdna3_5::build_vop3(rdna3_5::kVCmpClassF64Vop3, {.vdst = 0, .src0 = 0, .src1 = 258}),
        rdna3_5::build_vop3(rdna3_5::kVCmpxClassF16Vop3,
                            {.vdst = 126, .op_sel = 3, .src0 = 256, .src1 = 258, .neg = 1})}},
  };
  // ISA class-mask order: sNaN, qNaN, -Inf, -normal, -denormal, -0,
  // +0, +denormal, +normal, +Inf. NEG leaves either NaN class unchanged.
  constexpr uint32_t f32_bits[] = {0x7f800001U, 0x7fc00001U, 0xff800000U, 0xbf800000U, 0x80000001U,
                                   0x80000000U, 0,           1,           0x3f800000U, 0x7f800000U};
  constexpr uint16_t f16_bits[] = {0x7c01, 0x7e01, 0xfc00, 0xbc00, 0x8001,
                                   0x8000, 0,      1,      0x3c00, 0x7c00};
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    const bool cdna =
        family.arch == ROCJITSU_CODE_ARCH_CDNA1 || family.arch == ROCJITSU_CODE_ARCH_CDNA2;
    const bool high_half = cdna || family.arch == ROCJITSU_CODE_ARCH_RDNA3_5;
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
        GpuMemory memory("older_class_memory");
        L2Cache l2("older_class_l2");
        auto cu = ComputeUnitCore::create("older_class", config, &memory, &l2);
        auto *wave = cu->dispatch_wf(0, 0, 104, 16, lanes);
        ASSERT_NE(wave, nullptr);
        auto decoder = Decoder::create(family.arch);
        const auto vb = wave->vgpr_alloc().base, sb = wave->sgpr_alloc().base;
        constexpr uint64_t old_vcc = 0x55aac0de12345678ULL;
        for (uint64_t exec_mask : {~0ULL, 0x8000000380000005ULL, 0ULL}) {
          const uint64_t active = exec_mask & lane_mask;
          for (int operation = 0; operation < 3; ++operation) {
            SCOPED_TRACE(family.arch);
            SCOPED_TRACE(lanes);
            SCOPED_TRACE(operation);
            wave->set_exec(active);
            wave->set_vcc_raw(old_vcc);
            wave->write_scc(true);
            // An sNaN whose low word distinguishes it from negative infinity.
            // F64 writes its mask over this same SGPR pair after reading it.
            cu->write_sgpr(sb, 1);
            cu->write_sgpr(sb + 1, 0xfff00000U);
            for (unsigned lane = 0; lane < lanes; ++lane) {
              const unsigned index = lane % 10;
              const unsigned cls = operation == 1                 ? 0
                                   : operation == 2 && index >= 2 ? 11 - index
                                                                  : index;
              // Alternate ten-lane blocks: every class both matches and misses.
              uint32_t classes = 1U << ((lane / 10) & 1 ? cls : (cls + 1) % 10);
              uint32_t bits = f32_bits[index];
              if (operation == 2) {
                bits = uint32_t(f16_bits[index]) << (high_half ? 16 : 0);
                classes <<= high_half ? 16 : 0;
              }
              cu->write_vgpr(vb, lane, bits);
              cu->write_vgpr(vb + 2, lane, classes);
            }
            auto decoded = decoder->decode_window(family.instructions[operation]);
            ASSERT_TRUE(decoded.succeeded());
            ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
            const uint64_t expected = active & 0x0ffc00ffc00ffc00ULL;
            EXPECT_EQ(wave->vcc(), operation == 0 ? (old_vcc & ~lane_mask) | expected : old_vcc);
            EXPECT_EQ(wave->exec(), operation == 2 ? expected : active);
            EXPECT_TRUE(wave->read_scc());
            if (operation == 1 || (operation == 2 && cdna)) {
              EXPECT_EQ(cu->read_sgpr(sb), uint32_t(expected));
              EXPECT_EQ(cu->read_sgpr(sb + 1),
                        lanes == 64 ? uint32_t(expected >> 32) : 0xfff00000U);
            }
          }
        }
      }
    }
  }
}
// A CMPX row mask filters EXEC independently of source availability in both rows.
TEST(OlderClassCompare, Gfx1150DppRowMask) {
  ScopedIsaExecutionBackend backend(&rdna3_5::execution_backend());
  ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_RDNA3_5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 104;
  config.vgprs_per_wf = 8;
  config.lds_size_kb = 64;
  GpuMemory memory("class_dpp_memory");
  L2Cache l2("class_dpp_l2");
  auto cu = ComputeUnitCore::create("class_dpp", config, &memory, &l2);
  auto *wave = cu->dispatch_wf(0, 0, 104, 8, 32);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0x30007);
  const uint32_t vb = wave->vgpr_alloc().base;
  for (int lane = 0; lane < 32; ++lane) {
    cu->write_vgpr(vb, lane, lane == 0 || lane == 16 ? 0x8001 : 0x3c00);
    cu->write_vgpr(vb + 2, lane, 0x10);
  }
  rdna3_5::VopcVopDpp16MachineInst raw{};
  raw.encoding = rdna3_5::encoding::kVopc >> 2;
  raw.op = rdna3_5::kVCmpxClassF16Vopc;
  raw.src0 = SRC_DPP;
  raw.vsrc0 = 0;
  raw.vsrc1 = 2;
  raw.dpp_ctrl = dpp::ROW_SHR1;
  raw.row_mask = 1;
  raw.bank_mask = 0xf;
  auto decoder = Decoder::create(config.arch);
  auto decoded = decoder->decode_window(std::bit_cast<std::array<uint32_t, 2>>(raw));
  ASSERT_TRUE(decoded.succeeded());
  ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
  EXPECT_EQ(wave->exec(), 2U);
}

} // namespace
} // namespace rocjitsu::amdgpu
