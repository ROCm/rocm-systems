// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older CDNA integer MFMA broadcasts, accumulator banks and whole-wave execution.
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
#include <random>

namespace rocjitsu::amdgpu {
namespace {
// The 4x4 batched form distinguishes adjacent-block and full-wave A broadcasts.
TEST(OlderIntegerMfma, BroadcastBanksAndInlineAccumulator) {
  std::array<std::array<std::array<uint32_t, 64>, 40>, 2> input;
  std::minstd_rand random;
  for (auto &bank : input) {
    for (auto &reg : bank) {
      for (auto &word : reg) {
        const auto first_draw = random();
        word = uint32_t(first_draw >> 15) << 16 | uint32_t(random() >> 15);
      }
    }
  }
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
      GpuMemory memory("integer_mfma_memory");
      L2Cache l2("integer_mfma_l2");
      auto cu = ComputeUnitCore::create("integer_mfma", config, &memory, &l2);
      auto *wave = cu->dispatch_wf(0, 0, 104, 256);
      ASSERT_NE(wave, nullptr);
      auto decoder = Decoder::create(arch);
      const auto base = wave->vgpr_alloc().base;
      for (uint8_t destination : {16, 32}) {
        for (bool inline_c : {false, true}) {
          for (uint64_t exec_mask : {~0ULL, 0x800000010000000bULL, 0ULL}) {
            wave->set_exec(exec_mask);
            wave->set_mode_raw(Wavefront::FP16_OVFL_BIT); // Ignored by these older integer forms.
            wave->set_vcc_raw(0x123456789abcdef0ULL);
            wave->write_scc(true);
            for (unsigned bank = 0; bank < 2; ++bank) {
              for (unsigned reg = 0; reg < 40; ++reg) {
                for (unsigned lane = 0; lane < 64; ++lane) {
                  cu->write_vgpr(base + 256 * bank + reg, lane, input[bank][reg][lane]);
                }
              }
            }
            // Source encoding193 is inline -1, whose value also equals the old sentinel.
            const auto words =
                arch == rocjitsu::IsaFamily::Cdna1
                    ? cdna1::build_vop3p_mfma(cdna1::kVMfmaI324x4x4i8Vop3pMfma,
                                              {.vdst = destination,
                                               .cbsz = 2,
                                               .abid = 1,
                                               .src0 = 256,
                                               .src1 = 264,
                                               .src2 = uint16_t(inline_c ? 193 : 272),
                                               .blgp = 4})
                    : cdna2::build_vop3p_mfma(cdna2::kVMfmaI324x4x4i8Vop3pMfma,
                                              {.vdst = destination,
                                               .cbsz = 2,
                                               .abid = 1,
                                               .acc_cd = 1,
                                               .src0 = 256,
                                               .src1 = 264,
                                               .src2 = uint16_t(inline_c ? 193 : 272),
                                               .blgp = 4});
            auto decoded = decoder->decode_window(words);
            ASSERT_TRUE(decoded.succeeded());
            const auto diagnostics = cu->isa_diagnostic_count();
            ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
            EXPECT_EQ(cu->isa_diagnostic_count(), diagnostics);
            auto expected = input;
            for (unsigned batch = 0; batch < 16; ++batch) {
              for (unsigned row = 0; row < 4; ++row) {
                for (unsigned col = 0; col < 4; ++col) {
                  const unsigned a_lane = arch == rocjitsu::IsaFamily::Cdna2
                                              ? batch / 4 * 16 + 4 + row
                                              : (batch * 4 + row) % 16 + 16;
                  const unsigned b_lane = (batch * 4 + col) % 16;
                  const unsigned output_lane = batch * 4 + col;
                  uint32_t sum = inline_c ? ~0U : input[1][16 + row][output_lane];
                  for (unsigned k = 0; k < 4; ++k) {
                    auto byte = [](uint32_t word, unsigned i) {
                      return int(((word >> (8 * i)) & 0xffU) ^ 0x80U) - 0x80;
                    };
                    sum += uint32_t(byte(input[0][0][a_lane], k) * byte(input[0][8][b_lane], k));
                  }
                  expected[1][destination + row][output_lane] = sum;
                }
              }
            }
            for (unsigned bank = 0; bank < 2; ++bank) {
              for (unsigned reg = 0; reg < 40; ++reg) {
                for (unsigned lane = 0; lane < 64; ++lane) {
                  ASSERT_EQ(cu->read_vgpr(base + 256 * bank + reg, lane), expected[bank][reg][lane])
                      << arch << ',' << bank << ',' << reg << ',' << lane;
                }
              }
            }
            EXPECT_EQ(wave->exec(), exec_mask);
            EXPECT_EQ(wave->vcc(), 0x123456789abcdef0ULL);
            EXPECT_TRUE(wave->read_scc());
          }
        }
      }
    }
  }
}
} // namespace
} // namespace rocjitsu::amdgpu
