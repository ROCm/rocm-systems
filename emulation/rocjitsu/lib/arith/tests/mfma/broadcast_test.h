// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/cpu.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <array>
#include <bit>
#include <cstdint>
#include <gtest/gtest.h>

namespace rocjitsu::arith::test {

// MI300X captures from tests/kernels/mfma_broadcast_gfx942.hip: A[lane] = lane + 1,
// B = 1, C = 0. All legal CBSZ/ABID combinations were captured for these batched forms.
template <int M, int K, int Bits, bool Integer = false>
void expect_gfx942_broadcast(decltype(&rj_arith_v_mfma_f32_4x4x1_16b_f32_w64) entry, auto encode) {
  constexpr int inputs = K * Bits / 32;
  constexpr int outputs = M;
  std::array<std::array<uint32_t, 64>, inputs> a{}, b{};
  std::array<std::array<uint32_t, 64>, outputs> c{}, d{};
  std::array<const uint32_t *, inputs> a_ptr{}, b_ptr{};
  std::array<const uint32_t *, outputs> c_ptr{};
  std::array<uint32_t *, outputs> d_ptr{};
  for (int reg = 0; reg < inputs; ++reg) {
    a_ptr[reg] = a[reg].data();
    b_ptr[reg] = b[reg].data();
    for (int lane = 0; lane < 64; ++lane) {
      a[reg][lane] = encode(lane + 1);
      b[reg][lane] = encode(1);
    }
  }
  for (int reg = 0; reg < outputs; ++reg) {
    c_ptr[reg] = c[reg].data();
    d_ptr[reg] = d[reg].data();
  }
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int cbsz = 0; (1 << cbsz) <= 64 / M; ++cbsz) {
      for (int abid = 0; abid < (1 << cbsz); ++abid) {
        SCOPED_TRACE(::testing::Message() << "M=" << M << " K=" << K << " cpu=" << cpu
                                          << " cbsz=" << cbsz << " abid=" << abid);
        for (auto &reg : d) {
          reg.fill(0xdeadbeef);
        }
        const uint64_t controls = RJ_ARITH_MFMA_A_ADJACENT_BLOCKS | uint64_t(cbsz | (abid << 3));
        ASSERT_EQ(entry(cpu | RJ_ARITH_SEMANTICS_LOOSE, controls, d_ptr.data(), a_ptr.data(),
                        b_ptr.data(), c_ptr.data()),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 64; ++lane) {
          for (int reg = 0; reg < outputs; ++reg) {
            const int batch = M == 4 ? lane / 4 : reg / (M * M / 64);
            const int row =
                M == 4 ? reg : (reg % (M * M / 64) / 4) * (256 / M) + 4 * (lane / M) + reg % 4;
            const int source_batch = (batch >> cbsz << cbsz) + abid;
            const int expected = K * (source_batch * M + row + 1);
            const uint32_t word =
                Integer ? uint32_t(expected) : std::bit_cast<uint32_t>(float(expected));
            ASSERT_EQ(d[reg][lane], word) << "lane=" << lane << " reg=" << reg;
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
