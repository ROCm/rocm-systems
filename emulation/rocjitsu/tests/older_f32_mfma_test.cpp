// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older floating MFMA broadcast mapping, guest/fixed denormals and accumulator banks.
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
#include "util/data_types.h"
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <random>

namespace rocjitsu::amdgpu {
namespace {
// Input and output flush cases distinguish both guest mode bits independently.
void broadcast_banks_and_denormals(int format) {
  std::array<std::array<std::array<uint32_t, 64>, 40>, 2> seed;
  std::minstd_rand random;
  for (auto &bank : seed) {
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
      GpuMemory memory("older_f32_mfma_memory");
      L2Cache l2("older_f32_mfma_l2");
      auto cu = ComputeUnitCore::create("older_f32_mfma", config, &memory, &l2);
      auto *wave = cu->dispatch_wf(0, 0, 104, 256);
      ASSERT_NE(wave, nullptr);
      auto decoder = Decoder::create(arch);
      const auto base = wave->vgpr_alloc().base;
      for (bool tiny_output : {false, true}) {
        auto input = seed;
        for (unsigned lane = 0; lane < 64; ++lane) {
          input[0][0][lane] = tiny_output ? 0x1f000000U + (lane << 14) : 0x10000U * (lane + 1);
          input[0][8][lane] = tiny_output ? 0x9f000000U + (lane << 14) : 0x7b800000U + (lane << 16);
          if (format != 0) {
            // Half operands contain two elements per lane and two FP16 registers.
            // Tiny A times normal B distinguishes input-format flushing; BF16
            // normal factors can separately produce an FP32 subnormal result.
            const uint16_t av =
                format == 1 ? (tiny_output ? 0 : 1 + lane) : (tiny_output ? 0x0080 : 0x0040 + lane);
            const uint16_t bv = format == 1 ? 0x3c00 : (tiny_output ? 0x3d80 : 0x4080);
            for (unsigned reg = 0; reg < 2; ++reg) {
              input[0][reg][lane] = uint32_t(av) * 0x10001U;
              input[0][8 + reg][lane] = uint32_t(bv) * 0x10001U;
            }
          }
          for (unsigned row = 0; row < 4; ++row) {
            input[1][16 + row][lane] = format == 1 && tiny_output ? 0x00400000U
                                       : tiny_output              ? 0
                                                     : 0x3f000000U + ((row * 64 + lane) << 14);
          }
        }
        for (unsigned mode : {0, 1, 2, 3}) {
          for (uint8_t destination : {16, 32}) {
            for (bool inline_c : {false, true}) {
              for (uint64_t exec_mask : {~0ULL, 0x800000010000000bULL, 0ULL}) {
                wave->set_exec(exec_mask);
                wave->set_mode_raw(Wavefront::FP16_OVFL_BIT);
                wave->set_fp_denorm_mode(mode);
                wave->set_vcc_raw(0x123456789abcdef0ULL);
                wave->write_scc(true);
                for (unsigned bank = 0; bank < 2; ++bank) {
                  for (unsigned reg = 0; reg < 40; ++reg) {
                    for (unsigned lane = 0; lane < 64; ++lane) {
                      cu->write_vgpr(base + bank * 256 + reg, lane, input[bank][reg][lane]);
                    }
                  }
                }
                // Inline zero omits C reads; otherwise C shares the accumulator bank with D.
                const auto words =
                    arch == rocjitsu::IsaFamily::Cdna1
                        ? cdna1::build_vop3p_mfma(format == 0 ? cdna1::kVMfmaF324x4x1f32Vop3pMfma
                                                  : format == 1
                                                      ? cdna1::kVMfmaF324x4x4f16Vop3pMfma
                                                      : cdna1::kVMfmaF324x4x2bf16Vop3pMfma,
                                                  {.vdst = destination,
                                                   .cbsz = 2,
                                                   .abid = 1,
                                                   .src0 = 256,
                                                   .src1 = 264,
                                                   .src2 = uint16_t(inline_c ? 128 : 272),
                                                   .blgp = 3})
                        : cdna2::build_vop3p_mfma(format == 0 ? cdna2::kVMfmaF324x4x1f32Vop3pMfma
                                                  : format == 1
                                                      ? cdna2::kVMfmaF324x4x4f16Vop3pMfma
                                                      : cdna2::kVMfmaF324x4x2bf16Vop3pMfma,
                                                  {.vdst = destination,
                                                   .cbsz = 2,
                                                   .abid = 1,
                                                   .acc_cd = 1,
                                                   .src0 = 256,
                                                   .src1 = 264,
                                                   .src2 = uint16_t(inline_c ? 128 : 272),
                                                   .blgp = 3});
                auto decoded = decoder->decode_window(words);
                ASSERT_TRUE(decoded.succeeded());
                const auto diagnostics = cu->isa_diagnostic_count();
                ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
                EXPECT_EQ(cu->isa_diagnostic_count(), diagnostics);
                auto expected = input;
                auto read = [&](uint32_t bits) {
                  if (arch == rocjitsu::IsaFamily::Cdna2 && (format != 0 || !(mode & 1)) &&
                      !(bits & 0x7f800000U)) {
                    bits &= 0x80000000U;
                  }
                  return std::bit_cast<float>(bits);
                };
                for (unsigned batch = 0; batch < 16; ++batch) {
                  for (unsigned row = 0; row < 4; ++row) {
                    for (unsigned col = 0; col < 4; ++col) {
                      unsigned a_lane = arch == rocjitsu::IsaFamily::Cdna2
                                            ? batch / 4 * 16 + 4 + row
                                            : (batch * 4 + row) % 16 + 16;
                      unsigned b_lane = (batch * 4 + col + 16) % 64, output_lane = batch * 4 + col;
                      float value = inline_c ? 0.0f : read(input[1][16 + row][output_lane]);
                      const unsigned depth = format == 0 ? 1 : format == 1 ? 4 : 2;
                      for (unsigned k = 0; k < depth; ++k) {
                        auto factor = [&](unsigned source, unsigned lane) {
                          if (format == 0) {
                            return read(input[0][source][lane]);
                          }
                          uint16_t bits =
                              uint16_t(input[0][source + k / 2][lane] >> (16 * (k % 2)));
                          const uint16_t exponent = format == 1 ? 0x7c00 : 0x7f80;
                          if (arch == rocjitsu::IsaFamily::Cdna2 && !(bits & exponent)) {
                            bits &= 0x8000;
                          }
                          return format == 1 ? util::f16_to_f32(bits)
                                             : std::bit_cast<float>(uint32_t(bits) << 16);
                        };
                        value = std::fma(factor(0, a_lane), factor(8, b_lane), value);
                      }
                      uint32_t bits = std::bit_cast<uint32_t>(value);
                      if (arch == rocjitsu::IsaFamily::Cdna2 && (format != 0 || !(mode & 2)) &&
                          !(bits & 0x7f800000U)) {
                        bits &= 0x80000000U;
                      }
                      expected[1][destination + row][output_lane] = bits;
                    }
                  }
                }
                for (unsigned bank = 0; bank < 2; ++bank) {
                  for (unsigned reg = 0; reg < 40; ++reg) {
                    for (unsigned lane = 0; lane < 64; ++lane) {
                      ASSERT_EQ(cu->read_vgpr(base + bank * 256 + reg, lane),
                                expected[bank][reg][lane])
                          << arch << ',' << mode << ',' << tiny_output << ',' << bank << ',' << reg
                          << ',' << lane;
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
  }
}
TEST(OlderF32Mfma, BroadcastBanksAndDenormals) { broadcast_banks_and_denormals(0); }
TEST(OlderHalfMfma, F16BroadcastBanksAndFixedDenormals) { broadcast_banks_and_denormals(1); }
TEST(OlderHalfMfma, Bf16BroadcastBanksAndFixedDenormals) { broadcast_banks_and_denormals(2); }
} // namespace
} // namespace rocjitsu::amdgpu
