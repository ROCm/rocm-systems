// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx90a FP64 MFMA decoding, accumulator banks, overlap and inherited fused arithmetic.
#include "rocjitsu/code/rj_code.h"
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
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>

namespace rocjitsu::amdgpu {
namespace {
using Registers = std::array<std::array<std::array<uint32_t, 64>, 40>, 2>;

// Read or write one double without depending on host byte order or pointer aliasing.
double read_pair(const Registers &v, int bank, int reg, int lane) {
  return std::bit_cast<double>(uint64_t(v[bank][reg][lane]) |
                               (uint64_t(v[bank][reg + 1][lane]) << 32));
}
void write_pair(Registers &v, int bank, int reg, int lane, double value) {
  const auto bits = std::bit_cast<uint64_t>(value);
  v[bank][reg][lane] = uint32_t(bits);
  v[bank][reg + 1][lane] = uint32_t(bits >> 32);
}

// Exercise each real shape with a few independent layout/control and fused-result witnesses.
template <int M, int Batch> void check_f64_mfma() {
  constexpr int output_words = M * M * Batch / 32;
  ScopedIsaExecutionBackend backend(&cdna2::execution_backend());
  Registers seed;
  std::minstd_rand random;
  for (auto &bank : seed) {
    for (auto &reg : bank) {
      for (auto &word : reg) {
        const auto first_draw = random();
        word = uint32_t(first_draw >> 15) << 16 | uint32_t(random() >> 15);
      }
    }
  }
  for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
    ComputeUnitCore::Config config{};
    config.arch = rocjitsu::IsaFamily::Cdna2;
    config.arithmetic = {semantics};
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    GpuMemory memory("gfx90a_f64_memory");
    L2Cache l2("gfx90a_f64_l2");
    auto cu = ComputeUnitCore::create("gfx90a_f64", config, &memory, &l2);
    auto *wave = cu->dispatch_wf(0, 0, 104, 256);
    ASSERT_NE(wave, nullptr);
    auto decoder = Decoder::create(config.arch);
    const auto base = wave->vgpr_alloc().base;
    // Cases 0..5 cover separate/aliased outputs, both banks, inline C and ignored EXEC/MODE.
    // Cases 6..8 isolate fused rounding, overflow cancellation and preserved subnormals.
    for (int test = 0; test < 9; ++test) {
      if (test >= 6 && semantics == ArithmeticSemantics::Loose) {
        continue;
      }
      SCOPED_TRACE(test);
      SCOPED_TRACE(int(semantics));
      const int bank = test == 0 || test == 4 ? 1 : 0;
      const int destination = test == 1 ? 0 : test == 2 ? 16 : 32;
      const bool inline_c = test == 3 || test == 4;
      const uint64_t exec_mask = test == 4 ? 0 : test == 3 ? 0x800000010000000bULL : ~0ULL;
      auto input = seed;
      for (int lane = 0; lane < 64; ++lane) {
        for (int reg : {0, 8}) {
          // High engine bits construct ordinary fractional values, not just exact integers.
          const auto first_draw = random();
          const uint64_t bits =
              0x3fe0000000000000ULL | (uint64_t(first_draw >> 3) << 24) | (random() >> 7);
          write_pair(input, 0, reg, lane, std::bit_cast<double>(bits));
        }
        for (int reg = 0; reg < output_words; reg += 2) {
          write_pair(input, bank, 16 + reg, lane, -0.25 + double(lane + reg) / 257.0);
        }
      }
      if (test >= 6) {
        for (int lane = 0; lane < 64; ++lane) {
          write_pair(input, 0, 0, lane, 0);
          write_pair(input, 0, 8, lane, 0);
          for (int reg = 0; reg < output_words; reg += 2) {
            write_pair(input, bank, 16 + reg, lane, 0);
          }
        }
        const double a = test == 6 ? 0x1.0000002p+0 : std::numeric_limits<double>::max();
        const double b = test == 6 ? 0x1.ffffffcp-1 : 2;
        write_pair(input, 0, 0, 0, test == 8 ? 0 : a);
        write_pair(input, 0, 8, 0, test == 8 ? 0 : b);
        write_pair(input, bank, 16, 0, test == 6 ? -1 : test == 7 ? -a : 0x1p-1074);
      }
      wave->set_exec(exec_mask);
      wave->set_mode_raw(test % 2 ? Wavefront::FP16_OVFL_BIT | Wavefront::FP_ROUND_MASK |
                                        Wavefront::FP_DENORM_MASK
                                  : 0);
      wave->set_vcc_raw(0x123456789abcdef0ULL);
      wave->write_scc(true);
      for (int b = 0; b < 2; ++b) {
        for (int reg = 0; reg < 40; ++reg) {
          for (int lane = 0; lane < 64; ++lane) {
            cu->write_vgpr(base + b * 256 + reg, lane, input[b][reg][lane]);
          }
        }
      }
      const auto opcode =
          M == 16 ? cdna2::kVMfmaF6416x16x4f64Vop3pMfma : cdna2::kVMfmaF644x4x4f64Vop3pMfma;
      const auto words = cdna2::build_vop3p_mfma(opcode, {.vdst = uint8_t(destination),
                                                          .acc_cd = uint8_t(bank),
                                                          .src0 = 256,
                                                          .src1 = 264,
                                                          .src2 = uint16_t(inline_c ? 248 : 272)});
      auto decoded = decoder->decode_window(words);
      ASSERT_TRUE(decoded.succeeded());
      const auto diagnostics = cu->isa_diagnostic_count();
      ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
      EXPECT_EQ(cu->isa_diagnostic_count(), diagnostics);
      auto expected = input;
      for (int batch = 0; batch < Batch; ++batch) {
        for (int row = 0; row < M; ++row) {
          for (int col = 0; col < M; ++col) {
            // Legacy FP64 mapping: each lane owns a contiguous group of four rows.
            const int reg = 2 * (batch * (M * M / 64) + row / (64 / M));
            const int batch_groups = (64 + M * M - 1) / (M * M);
            const int lane = batch % batch_groups * M + row % (64 / M) * batch_groups * M + col;
            double value = inline_c ? (semantics == ArithmeticSemantics::Compatibility
                                           ? double(std::bit_cast<float>(0x3e22f983U))
                                           : 0x1.45f306dc9c882p-3)
                                    : read_pair(input, bank, 16 + reg, lane);
            for (int k = 0; k < 4; ++k) {
              value = std::fma(read_pair(input, 0, 0, batch * M + k * M * Batch + row),
                               read_pair(input, 0, 8, batch * M + k * M * Batch + col), value);
            }
            write_pair(expected, bank, destination + reg, lane, value);
          }
        }
      }
      if (test >= 6) {
        // The inherited fused model avoids an intermediate rounded product or infinity.
        EXPECT_EQ(read_pair(expected, bank, destination, 0),
                  test == 6   ? -0x1p-54
                  : test == 7 ? std::numeric_limits<double>::max()
                              : 0x1p-1074);
      }
      for (int b = 0; b < 2; ++b) {
        for (int reg = 0; reg < 40; ++reg) {
          for (int lane = 0; lane < 64; ++lane) {
            if (b == bank && reg >= destination && reg < destination + output_words && test < 6) {
              if (reg % 2 == 0) {
                const auto low = cu->read_vgpr(base + b * 256 + reg, lane);
                const auto high = cu->read_vgpr(base + b * 256 + reg + 1, lane);
                const double actual = std::bit_cast<double>(uint64_t(low) | (uint64_t(high) << 32));
                const double reference = read_pair(expected, b, reg, lane);
                ASSERT_NEAR(actual, reference, 1e-12 + 1e-12 * std::abs(reference))
                    << b << ',' << reg << ',' << lane;
              }
            } else {
              ASSERT_EQ(cu->read_vgpr(base + b * 256 + reg, lane), expected[b][reg][lane])
                  << b << ',' << reg << ',' << lane;
            }
          }
        }
      }
      EXPECT_EQ(wave->exec(), exec_mask);
      EXPECT_EQ(wave->vcc(), 0x123456789abcdef0ULL);
      EXPECT_TRUE(wave->read_scc());
    }
  }
}
TEST(Gfx90aF64Mfma, Matrix16) { check_f64_mfma<16, 1>(); }
TEST(Gfx90aF64Mfma, Matrix4Batched) { check_f64_mfma<4, 4>(); }
} // namespace
} // namespace rocjitsu::amdgpu
