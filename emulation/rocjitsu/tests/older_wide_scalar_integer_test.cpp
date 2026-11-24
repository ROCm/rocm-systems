// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded paired-SGPR arithmetic: counts, SCC and overlapping sources with empty EXEC.
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
#include <cstdint>
#include <gtest/gtest.h>

namespace rocjitsu::amdgpu {
namespace {

// Scalar results and SCC are independent of EXEC; destination aliases must preserve inputs.
TEST(OlderWideScalarInteger, ArithmeticAndCondition) {
  struct Family {
    rj_code_arch_t arch;
    const IsaExecutionBackend *backend;
    std::array<std::array<uint32_t, 1>, 7> instructions;
  };
  const Family families[] = {
      {ROCJITSU_CODE_ARCH_CDNA1,
       &cdna1::execution_backend(),
       {cdna1::build_sop2(cdna1::kSAndB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        cdna1::build_sop2(cdna1::kSLshlB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        cdna1::build_sop1(cdna1::kSBcnt1I32B64Sop1, {.ssrc0 = 0, .sdst = 0}),
        cdna1::build_sop1(cdna1::kSFf1I32B64Sop1, {.ssrc0 = 0, .sdst = 0}),
        cdna1::build_sop2(cdna1::kSBfmB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        cdna1::build_sop1(cdna1::kSBitset1B64Sop1, {.ssrc0 = 2, .sdst = 0}),
        cdna1::build_sopc(cdna1::kSCmpEqU64Sopc, {.ssrc0 = 0, .ssrc1 = 2})}},
      {ROCJITSU_CODE_ARCH_CDNA2,
       &cdna2::execution_backend(),
       {cdna2::build_sop2(cdna2::kSAndB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        cdna2::build_sop2(cdna2::kSLshlB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        cdna2::build_sop1(cdna2::kSBcnt1I32B64Sop1, {.ssrc0 = 0, .sdst = 0}),
        cdna2::build_sop1(cdna2::kSFf1I32B64Sop1, {.ssrc0 = 0, .sdst = 0}),
        cdna2::build_sop2(cdna2::kSBfmB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        cdna2::build_sop1(cdna2::kSBitset1B64Sop1, {.ssrc0 = 2, .sdst = 0}),
        cdna2::build_sopc(cdna2::kSCmpEqU64Sopc, {.ssrc0 = 0, .ssrc1 = 2})}},
      {ROCJITSU_CODE_ARCH_RDNA1,
       &rdna1::execution_backend(),
       {rdna1::build_sop2(rdna1::kSAndB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        rdna1::build_sop2(rdna1::kSLshlB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        rdna1::build_sop1(rdna1::kSBcnt1I32B64Sop1, {.ssrc0 = 0, .sdst = 0}),
        rdna1::build_sop1(rdna1::kSFf1I32B64Sop1, {.ssrc0 = 0, .sdst = 0}),
        rdna1::build_sop2(rdna1::kSBfmB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        rdna1::build_sop1(rdna1::kSBitset1B64Sop1, {.ssrc0 = 2, .sdst = 0}),
        rdna1::build_sopc(rdna1::kSCmpEqU64Sopc, {.ssrc0 = 0, .ssrc1 = 2})}},
      {ROCJITSU_CODE_ARCH_RDNA2,
       &rdna2::execution_backend(),
       {rdna2::build_sop2(rdna2::kSAndB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        rdna2::build_sop2(rdna2::kSLshlB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        rdna2::build_sop1(rdna2::kSBcnt1I32B64Sop1, {.ssrc0 = 0, .sdst = 0}),
        rdna2::build_sop1(rdna2::kSFf1I32B64Sop1, {.ssrc0 = 0, .sdst = 0}),
        rdna2::build_sop2(rdna2::kSBfmB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        rdna2::build_sop1(rdna2::kSBitset1B64Sop1, {.ssrc0 = 2, .sdst = 0}),
        rdna2::build_sopc(rdna2::kSCmpEqU64Sopc, {.ssrc0 = 0, .ssrc1 = 2})}},
      {ROCJITSU_CODE_ARCH_RDNA3_5,
       &rdna3_5::execution_backend(),
       {rdna3_5::build_sop2(rdna3_5::kSAndB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        rdna3_5::build_sop2(rdna3_5::kSLshlB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        rdna3_5::build_sop1(rdna3_5::kSBcnt1I32B64Sop1, {.ssrc0 = 0, .sdst = 0}),
        rdna3_5::build_sop1(rdna3_5::kSCtzI32B64Sop1, {.ssrc0 = 0, .sdst = 0}),
        rdna3_5::build_sop2(rdna3_5::kSBfmB64Sop2, {.ssrc0 = 0, .ssrc1 = 2, .sdst = 0}),
        rdna3_5::build_sop1(rdna3_5::kSBitset1B64Sop1, {.ssrc0 = 2, .sdst = 0}),
        rdna3_5::build_sopc(rdna3_5::kSCmpEqU64Sopc, {.ssrc0 = 0, .ssrc1 = 2})}},
  };
  struct Case {
    int operation;
    uint64_t a, b, result;
    uint32_t scc;
  };
  const Case cases[] = {
      {0, 0x8000000100000001ULL, 0x80000000ffffffffULL, 0x8000000000000001ULL, 1},
      {0, 0xffffffffffffffffULL, 0, 0, 0},
      {1, 1, 63, 0x8000000000000000ULL, 1},
      {2, 0x8000000100000001ULL, 0, 0x8000000100000003ULL, 1}, // Count writes one SGPR.
      {2, 0, 0, 0, 0},
      {3, 0, 0, 0xffffffffULL, 1},           // Zero search sentinel; SCC is unchanged.
      {4, 65, 63, 0x8000000000000000ULL, 1}, // Width and offset are masked to six bits.
      {5, 0, 65, 2, 1},
      {6, 0x8000000100000001ULL, 0x8000000100000001ULL, 0x8000000100000001ULL, 1},
  };
  for (const auto &family : families) {
    SCOPED_TRACE(family.arch);
    ScopedIsaExecutionBackend backend(family.backend);
    for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
      ComputeUnitCore::Config config{};
      config.arch = family.arch;
      config.arithmetic = {semantics};
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 104;
      config.vgprs_per_wf = 16;
      config.lds_size_kb = 64;
      GpuMemory memory("older_scalar_memory");
      L2Cache l2("older_scalar_l2");
      auto cu = ComputeUnitCore::create("older_scalar", config, &memory, &l2);
      auto *wave = cu->dispatch_wf(0, 0, 104, 16, 64);
      ASSERT_NE(wave, nullptr);
      auto decoder = Decoder::create(family.arch);
      const uint32_t base = wave->sgpr_alloc().base;
      // Scalar instructions execute even when no vector lane is active.
      wave->set_exec(0);
      for (const auto &test : cases) {
        SCOPED_TRACE(test.operation);
        cu->write_sgpr(base, uint32_t(test.a));
        cu->write_sgpr(base + 1, uint32_t(test.a >> 32));
        cu->write_sgpr(base + 2, uint32_t(test.b));
        cu->write_sgpr(base + 3, uint32_t(test.b >> 32));
        wave->write_scc(1);
        auto decoded = decoder->decode_window(family.instructions[test.operation]);
        ASSERT_TRUE(decoded.succeeded());
        ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
        EXPECT_EQ(cu->read_sgpr(base), uint32_t(test.result));
        EXPECT_EQ(cu->read_sgpr(base + 1), uint32_t(test.result >> 32));
        EXPECT_EQ(cu->read_sgpr(base + 2), uint32_t(test.b));
        EXPECT_EQ(cu->read_sgpr(base + 3), uint32_t(test.b >> 32));
        EXPECT_EQ(wave->read_scc(), test.scc);
        EXPECT_EQ(wave->exec(), 0U);
      }
    }
  }
}

} // namespace
} // namespace rocjitsu::amdgpu
