// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded older-profile scalar arithmetic: SCC, immediates and aliased SGPRs with empty EXEC.
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
#include <cstdint>
#include <gtest/gtest.h>

namespace rocjitsu::amdgpu {
namespace {

// Scalar results and SCC are independent of EXEC; destination aliases must preserve inputs.
TEST(OlderScalarInteger, ArithmeticAndCondition) {
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<std::array<uint32_t, 1>, 5> instructions;
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {cdna1::build_sop2(cdna1::kSAddcU32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0}),
        cdna1::build_sopk(cdna1::kSCmpkGtU32Sopk, {.simm16 = 0xffff, .sdst = 0}),
        cdna1::build_sopk(cdna1::kSCmpkGtI32Sopk, {.simm16 = 0xffff, .sdst = 0}),
        cdna1::build_sop1(cdna1::kSFf1I32B32Sop1, {.ssrc0 = 0, .sdst = 0}),
        cdna1::build_sop2(cdna1::kSCselectB32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0})}},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {cdna2::build_sop2(cdna2::kSAddcU32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0}),
        cdna2::build_sopk(cdna2::kSCmpkGtU32Sopk, {.simm16 = 0xffff, .sdst = 0}),
        cdna2::build_sopk(cdna2::kSCmpkGtI32Sopk, {.simm16 = 0xffff, .sdst = 0}),
        cdna2::build_sop1(cdna2::kSFf1I32B32Sop1, {.ssrc0 = 0, .sdst = 0}),
        cdna2::build_sop2(cdna2::kSCselectB32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0})}},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {rdna1::build_sop2(rdna1::kSAddcU32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0}),
        rdna1::build_sopk(rdna1::kSCmpkGtU32Sopk, {.simm16 = 0xffff, .sdst = 0}),
        rdna1::build_sopk(rdna1::kSCmpkGtI32Sopk, {.simm16 = 0xffff, .sdst = 0}),
        rdna1::build_sop1(rdna1::kSFf1I32B32Sop1, {.ssrc0 = 0, .sdst = 0}),
        rdna1::build_sop2(rdna1::kSCselectB32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0})}},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {rdna2::build_sop2(rdna2::kSAddcU32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0}),
        rdna2::build_sopk(rdna2::kSCmpkGtU32Sopk, {.simm16 = 0xffff, .sdst = 0}),
        rdna2::build_sopk(rdna2::kSCmpkGtI32Sopk, {.simm16 = 0xffff, .sdst = 0}),
        rdna2::build_sop1(rdna2::kSFf1I32B32Sop1, {.ssrc0 = 0, .sdst = 0}),
        rdna2::build_sop2(rdna2::kSCselectB32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0})}},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {rdna3_5::build_sop2(rdna3_5::kSAddcU32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0}),
        rdna3_5::build_sopk(rdna3_5::kSCmpkGtU32Sopk, {.simm16 = 0xffff, .sdst = 0}),
        rdna3_5::build_sopk(rdna3_5::kSCmpkGtI32Sopk, {.simm16 = 0xffff, .sdst = 0}),
        rdna3_5::build_sop1(rdna3_5::kSCtzI32B32Sop1, {.ssrc0 = 0, .sdst = 0}),
        rdna3_5::build_sop2(rdna3_5::kSCselectB32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 0})}},
  };
  struct Case {
    int operation;
    uint32_t a, b, input_scc, result, scc;
  };
  const Case cases[] = {
      {0, 0xffffffffU, 0, 1, 0, 1},
      {1, 0x10000U, 0, 0, 0x10000U, 1}, // Unsigned immediate is zero-extended.
      {2, 0, 0, 0, 0, 1},               // Signed immediate is sign-extended.
      {3, 0, 0, 1, 0xffffffffU, 1},
      {4, 11, 22, 0, 22, 0},
      {4, 11, 22, 1, 11, 1},
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
        cu->write_sgpr(base, test.a);
        cu->write_sgpr(base + 1, test.b);
        wave->write_scc(test.input_scc);
        auto decoded = decoder->decode_window(family.instructions[test.operation]);
        ASSERT_TRUE(decoded.succeeded());
        ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
        EXPECT_EQ(cu->read_sgpr(base), test.result);
        EXPECT_EQ(cu->read_sgpr(base + 1), test.b);
        EXPECT_EQ(wave->read_scc(), test.scc);
        EXPECT_EQ(wave->exec(), 0U);
      }
    }
  }
}

} // namespace
} // namespace rocjitsu::amdgpu
