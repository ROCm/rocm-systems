// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older CDNA accumulator transfers retain separate banks, raw bits and masked writes.
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
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
// A different value in each bank makes a mistaken VGPR-only round trip visibly wrong.
TEST(OlderAccumulatorTransfers, BanksAndMasks) {
  for (auto arch : {rocjitsu::IsaFamily::Cdna1, rocjitsu::IsaFamily::Cdna2}) {
    ScopedIsaExecutionBackend backend(arch == rocjitsu::IsaFamily::Cdna1
                                          ? &cdna1::execution_backend()
                                          : &cdna2::execution_backend());
    for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
      ComputeUnitCore::Config config{};
      config.arch = arch;
      config.arithmetic = {semantics};
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 104;
      config.vgprs_per_wf = 256;
      config.lds_size_kb = 64;
      GpuMemory memory("older_accumulator_memory");
      L2Cache l2("older_accumulator_l2");
      auto cu = ComputeUnitCore::create("older_accumulator", config, &memory, &l2);
      auto *wave = cu->dispatch_wf(0, 0, 104, 256);
      ASSERT_NE(wave, nullptr);
      auto decoder = Decoder::create(arch);
      const auto base = wave->vgpr_alloc().base;
      const std::array<std::array<uint32_t, 2>, 2> transfers =
          arch == rocjitsu::IsaFamily::Cdna1
              ? std::array{cdna1::build_vop3p(cdna1::kVAccvgprWriteVop3p, {.vdst = 0, .src0 = 256}),
                           cdna1::build_vop3p(cdna1::kVAccvgprReadVop3p, {.vdst = 2, .src0 = 256})}
              : std::array{cdna2::build_vop3p(cdna2::kVAccvgprWriteVop3p, {.vdst = 0, .src0 = 256}),
                           cdna2::build_vop3p(cdna2::kVAccvgprReadVop3p, {.vdst = 2, .src0 = 256})};
      for (uint64_t exec_mask : {~0ULL, 0x8000000380000005ULL, 0ULL}) {
        wave->set_exec(exec_mask);
        wave->set_vgpr_write_mask(~4ULL);
        wave->set_vcc_raw(0x123456789abcdef0ULL);
        wave->write_scc(true);
        for (unsigned lane = 0; lane < 64; ++lane) {
          cu->write_vgpr(base, lane, 0x7f800001U + lane);
          cu->write_vgpr(base + 256, lane, 0xdeadbeefU);
          cu->write_vgpr(base + 257, lane, 0xbaadf00dU);
          cu->write_vgpr(base + 2, lane, 0xbeef1234U);
        }
        const auto diagnostics = cu->isa_diagnostic_count();
        for (const auto &words : transfers) {
          auto decoded = decoder->decode_window(words);
          ASSERT_TRUE(decoded.succeeded());
          ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
        }
        if (arch == rocjitsu::IsaFamily::Cdna2) {
          // VOP1 and VOP3 MOV copy within the accumulator bank; sign bits are ignored.
          const auto encoding =
              cdna2::build_vop1(cdna2::kVAccvgprMovB32Vop1, {.src0 = 256, .vdst = 1});
          const std::array<uint32_t, 2> move1{encoding[0], 0};
          const auto move3 = cdna2::build_vop3(cdna2::kVAccvgprMovB32Vop3,
                                               {.vdst = 1, .abs = 1, .src0 = 256, .neg = 1});
          for (const auto &words : {move1, move3}) {
            for (unsigned lane = 0; lane < 64; ++lane) {
              cu->write_vgpr(base + 257, lane, 0xbaadf00dU);
            }
            auto decoded = decoder->decode_window(words);
            ASSERT_TRUE(decoded.succeeded());
            ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
            for (unsigned lane = 0; lane < 64; ++lane) {
              EXPECT_EQ(cu->read_vgpr(base + 257, lane),
                        ((exec_mask & ~4ULL) >> lane) & 1 ? 0x7f800001U + lane : 0xbaadf00dU);
            }
          }
        }
        EXPECT_EQ(cu->isa_diagnostic_count(), diagnostics);
        for (unsigned lane = 0; lane < 64; ++lane) {
          const bool written = ((exec_mask & ~4ULL) >> lane) & 1;
          const uint32_t value = 0x7f800001U + lane;
          EXPECT_EQ(cu->read_vgpr(base, lane), value);
          EXPECT_EQ(cu->read_vgpr(base + 256, lane), written ? value : 0xdeadbeefU);
          EXPECT_EQ(cu->read_vgpr(base + 2, lane), written ? value : 0xbeef1234U);
          EXPECT_EQ(cu->read_vgpr(base + 257, lane),
                    written && arch == rocjitsu::IsaFamily::Cdna2 ? value : 0xbaadf00dU);
        }
        EXPECT_EQ(wave->exec(), exec_mask);
        EXPECT_EQ(wave->vcc(), 0x123456789abcdef0ULL);
        EXPECT_TRUE(wave->read_scc());
      }
    }
  }
}
} // namespace
} // namespace rocjitsu::amdgpu
