// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Register-only DS routing on older profiles: group boundaries, collisions and aliased writes.
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
// Read original operands before stores; forward collisions select the last active source.
TEST(OlderDsRouting, GroupsMasksAndAliasing) {
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    bool cdna;
    std::array<uint32_t, 2> (*encode)(int, uint8_t, uint16_t, bool);
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1, &cdna1::execution_backend(), true,
       [](int operation, uint8_t destination, uint16_t offset, bool gds) {
         const uint16_t opcodes[] = {cdna1::kDsBpermuteB32, cdna1::kDsPermuteB32,
                                     cdna1::kDsSwizzleB32};
         return cdna1::build_ds(opcodes[operation], {.offset0 = uint8_t(offset),
                                                     .offset1 = uint8_t(offset >> 8),
                                                     .gds = uint8_t(gds),
                                                     .addr = 0,
                                                     .data0 = uint8_t(operation == 2 ? 0 : 1),
                                                     .vdst = destination});
       }},
      {rocjitsu::IsaFamily::Cdna2, &cdna2::execution_backend(), true,
       [](int operation, uint8_t destination, uint16_t offset, bool gds) {
         const uint16_t opcodes[] = {cdna2::kDsBpermuteB32, cdna2::kDsPermuteB32,
                                     cdna2::kDsSwizzleB32};
         return cdna2::build_ds(opcodes[operation], {.offset0 = uint8_t(offset),
                                                     .offset1 = uint8_t(offset >> 8),
                                                     .gds = uint8_t(gds),
                                                     .addr = 0,
                                                     .data0 = uint8_t(operation == 2 ? 0 : 1),
                                                     .vdst = destination});
       }},
      {rocjitsu::IsaFamily::Rdna1, &rdna1::execution_backend(), false,
       [](int operation, uint8_t destination, uint16_t offset, bool gds) {
         const uint16_t opcodes[] = {rdna1::kDsBpermuteB32, rdna1::kDsPermuteB32,
                                     rdna1::kDsSwizzleB32};
         return rdna1::build_ds(opcodes[operation], {.offset0 = uint8_t(offset),
                                                     .offset1 = uint8_t(offset >> 8),
                                                     .gds = uint8_t(gds),
                                                     .addr = 0,
                                                     .data0 = uint8_t(operation == 2 ? 0 : 1),
                                                     .vdst = destination});
       }},
      {rocjitsu::IsaFamily::Rdna2, &rdna2::execution_backend(), false,
       [](int operation, uint8_t destination, uint16_t offset, bool gds) {
         const uint16_t opcodes[] = {rdna2::kDsBpermuteB32, rdna2::kDsPermuteB32,
                                     rdna2::kDsSwizzleB32};
         return rdna2::build_ds(opcodes[operation], {.offset0 = uint8_t(offset),
                                                     .offset1 = uint8_t(offset >> 8),
                                                     .gds = uint8_t(gds),
                                                     .addr = 0,
                                                     .data0 = uint8_t(operation == 2 ? 0 : 1),
                                                     .vdst = destination});
       }},
      {rocjitsu::IsaFamily::Rdna3_5, &rdna3_5::execution_backend(), false,
       [](int operation, uint8_t destination, uint16_t offset, bool gds) {
         const uint16_t opcodes[] = {rdna3_5::kDsBpermuteB32, rdna3_5::kDsPermuteB32,
                                     rdna3_5::kDsSwizzleB32};
         return rdna3_5::build_ds(opcodes[operation], {.offset0 = uint8_t(offset),
                                                       .offset1 = uint8_t(offset >> 8),
                                                       .gds = uint8_t(gds),
                                                       .addr = 0,
                                                       .data0 = uint8_t(operation == 2 ? 0 : 1),
                                                       .vdst = destination});
       }},
  };
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
        GpuMemory memory("older_ds_memory");
        L2Cache l2("older_ds_l2");
        auto cu = ComputeUnitCore::create("older_ds", config, &memory, &l2);
        auto *wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
        ASSERT_NE(wave, nullptr);
        auto decoder = Decoder::create(family.arch);
        const auto vb = wave->vgpr_alloc().base;
        for (int operation = 0; operation < 3; ++operation) {
          for (uint8_t destination : {0, 1, 2}) {
            for (uint64_t exec_mask : {~0ULL, 0x00ff0f0f0000ffffULL, 0ULL}) {
              exec_mask &= lanes == 32 ? 0xffffffffULL : ~0ULL;
              SCOPED_TRACE(family.arch);
              SCOPED_TRACE(operation);
              SCOPED_TRACE(destination);
              wave->set_exec(exec_mask);
              wave->set_vgpr_write_mask(~2ULL);
              wave->set_vcc_raw(0x123456789abcdef0ULL);
              wave->write_scc(true);
              uint32_t input[3][64]{};
              for (unsigned lane = 0; lane < lanes; ++lane) {
                // Gather crosses half-wave boundaries on CDNA only. Scatter deliberately collides.
                input[0][lane] = operation == 0   ? 4 * (lane ^ (lanes == 64 ? 32U : 16U)) + 3
                                 : operation == 1 ? 4 * (lane & 3)
                                                  : 0xaaa00000U + lane;
                input[1][lane] = 0xbbb00000U + lane;
                input[2][lane] = 0xccc00000U + lane;
                for (unsigned reg = 0; reg < 3; ++reg) {
                  cu->write_vgpr(vb + reg, lane, input[reg][lane]);
                }
              }
              const uint16_t offset = operation == 0 ? 5 : operation == 1 ? 0 : 0x401f;
              auto decoded =
                  decoder->decode_window(family.encode(operation, destination, offset, false));
              ASSERT_TRUE(decoded.succeeded());
              const auto diagnostics = cu->isa_diagnostic_count();
              ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
              EXPECT_EQ(cu->isa_diagnostic_count(), diagnostics);
              uint32_t result[64]{};
              for (unsigned lane = 0; lane < lanes; ++lane) {
                if (!((exec_mask >> lane) & 1)) {
                  continue;
                }
                const unsigned group = family.cdna ? lanes : 32;
                const unsigned selected =
                    operation == 2
                        ? lane ^ 16
                        : (lane / group) * group + ((input[0][lane] + offset) / 4) % group;
                if (operation == 1) {
                  result[selected] = input[1][lane];
                } else if ((exec_mask >> selected) & 1) {
                  result[lane] = input[operation == 2 ? 0 : 1][selected];
                }
              }
              for (unsigned reg = 0; reg < 3; ++reg) {
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  const auto expected = reg == destination && (((exec_mask & ~2ULL) >> lane) & 1)
                                            ? result[lane]
                                            : input[reg][lane];
                  EXPECT_EQ(cu->read_vgpr(vb + reg, lane), expected) << reg << ',' << lane;
                }
              }
              EXPECT_EQ(wave->exec(), exec_mask);
              EXPECT_EQ(wave->vcc(), 0x123456789abcdef0ULL);
              EXPECT_TRUE(wave->read_scc());
            }
          }
        }
        if (family.arch == rocjitsu::IsaFamily::Rdna3_5) {
          // gfx1150 rejects GDS even when EXEC is empty, before any register commit.
          wave->set_exec(0);
          auto decoded = decoder->decode_window(family.encode(0, 2, 0, true));
          ASSERT_TRUE(decoded.succeeded());
          EXPECT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).failed());
          EXPECT_EQ(wave->instruction_execution_error(),
                    InstructionExecutionError::UnimplementedInstruction);
          wave->clear_instruction_execution_error();
        }
      }
    }
  }
}
} // namespace
} // namespace rocjitsu::amdgpu
