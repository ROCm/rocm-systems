// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for bitfield extraction/insertion and word alignment.

#include "integer/bitfield_test.h"
#include "common/cpu.h"
#include "rj_arith/detail/rj_arith_integer.h"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>

// Check bitfield operations with routing and aliases.
TEST(IntegerBfeU32W32, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<32, 3>(rj_arith_v_bfe_u32_w32,
                                               rocjitsu::arith::test::Bitfield::BfeU32);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerBfeU32W64, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<64, 3>(rj_arith_v_bfe_u32_w64,
                                               rocjitsu::arith::test::Bitfield::BfeU32);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerBfeI32W32, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<32, 3>(rj_arith_v_bfe_i32_w32,
                                               rocjitsu::arith::test::Bitfield::BfeI32);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerBfeI32W64, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<64, 3>(rj_arith_v_bfe_i32_w64,
                                               rocjitsu::arith::test::Bitfield::BfeI32);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerBfiB32W32, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<32, 3>(rj_arith_v_bfi_b32_w32,
                                               rocjitsu::arith::test::Bitfield::Bfi);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerBfiB32W64, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<64, 3>(rj_arith_v_bfi_b32_w64,
                                               rocjitsu::arith::test::Bitfield::Bfi);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerBfmB32W32, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<32, 2>(rj_arith_v_bfm_b32_w32,
                                               rocjitsu::arith::test::Bitfield::Bfm);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerBfmB32W64, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<64, 2>(rj_arith_v_bfm_b32_w64,
                                               rocjitsu::arith::test::Bitfield::Bfm);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerBfrevB32W32, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<32, 1>(rj_arith_v_bfrev_b32_w32,
                                               rocjitsu::arith::test::Bitfield::Bfrev);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerBfrevB32W64, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<64, 1>(rj_arith_v_bfrev_b32_w64,
                                               rocjitsu::arith::test::Bitfield::Bfrev);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerAlignbitB32W32, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<32, 3>(rj_arith_v_alignbit_b32_w32,
                                               rocjitsu::arith::test::Bitfield::Alignbit);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerAlignbitB32W64, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<64, 3>(rj_arith_v_alignbit_b32_w64,
                                               rocjitsu::arith::test::Bitfield::Alignbit);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerAlignbyteB32W32, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<32, 3>(rj_arith_v_alignbyte_b32_w32,
                                               rocjitsu::arith::test::Bitfield::Alignbyte);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerAlignbyteB32W64, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<64, 3>(rj_arith_v_alignbyte_b32_w64,
                                               rocjitsu::arith::test::Bitfield::Alignbyte);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerPermB32W32, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<32, 3>(rj_arith_v_perm_b32_w32,
                                               rocjitsu::arith::test::Bitfield::Perm);
}

// Check bitfield operations with routing and aliases.
TEST(IntegerPermB32W64, Arithmetic) {
  rocjitsu::arith::test::check_bitfield<64, 3>(rj_arith_v_perm_b32_w64,
                                               rocjitsu::arith::test::Bitfield::Perm);
}

// Check the scalar-selector API against an independent bit-level reference,
// including both halves of Wave64 and an in-place destination.
TEST(IntegerPermB32UniformW64, SelectorsMasksAndAliasing) {
  using namespace rocjitsu::arith::test;
  std::array<uint32_t, 64> a, b;
  for (int lane = 0; lane < 64; ++lane) {
    a[lane] = 0x9e3779b9u * (lane + 1);
  }
  for (int lane = 0; lane < 64; ++lane) {
    b[lane] = ~a[63 - lane];
  }
  for (uint32_t selector : {0x05040100u, 0x07060302u, 0x0d0c0b08u, 0xff09060au}) {
    for (uint64_t mask : {~uint64_t{0}, uint64_t{0x8421842118421842}, uint64_t{0}}) {
      auto out = a;
      ASSERT_EQ(rj_arith_v_perm_b32_uniform_w64(RJ_ARITH_SEMANTICS_COMPATIBILITY, mask, out.data(),
                                                out.data(), b.data(), selector),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < 64; ++lane) {
        EXPECT_EQ(out[lane], (mask >> lane) & 1
                                 ? bitfield_reference(Bitfield::Perm, a[lane], b[lane], selector)
                                 : a[lane]);
      }
    }
  }
  auto out = b;
  EXPECT_EQ(rj_arith_v_perm_b32_uniform_w64(UINT64_MAX, UINT64_MAX, out.data(), a.data(),
                                            out.data(), 0x05040100u),
            RJ_ARITH_ERROR_INVALID_FLAGS);
  EXPECT_EQ(out, b);
}

// Check the half-packing selectors at every CPU level the host supports. The destination is
// separate from, or identical to, either source, and the operands sit at each 16-byte
// alignment a VGPR can have.
TEST(IntegerPermB32UniformW64, HalfPackingAtEveryCpuLevel) {
  using namespace rocjitsu::arith::test;
  enum class Destination { Separate, IsA, IsB };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int offset : {0, 4, 8, 12}) {
      alignas(64) std::array<uint32_t, 80> a_block, b_block, d_block;
      uint32_t *a = a_block.data() + offset, *b = b_block.data() + offset;
      std::array<uint32_t, 64> a_values, b_values;
      for (int lane = 0; lane < 64; ++lane) {
        a_values[lane] = 0x9e3779b9u * (lane + 1) + offset;
        b_values[lane] = ~(0x85ebca6bu * (lane + 3));
      }
      for (uint32_t selector : {0x05040100u, 0x07060302u}) {
        for (auto destination : {Destination::Separate, Destination::IsA, Destination::IsB}) {
          a_block.fill(0x11111111u);
          b_block.fill(0x22222222u);
          d_block.fill(0x33333333u);
          std::copy(a_values.begin(), a_values.end(), a);
          std::copy(b_values.begin(), b_values.end(), b);
          auto &block = destination == Destination::Separate ? d_block
                        : destination == Destination::IsA    ? a_block
                                                             : b_block;
          const auto guard = block;
          uint32_t *d = block.data() + offset;
          ASSERT_EQ(rj_arith_v_perm_b32_uniform_w64(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY,
                                                    ~uint64_t{0}, d, a, b, selector),
                    RJ_ARITH_SUCCESS);
          for (int lane = 0; lane < 64; ++lane) {
            ASSERT_EQ(d[lane],
                      bitfield_reference(Bitfield::Perm, a_values[lane], b_values[lane], selector))
                << "cpu " << cpu << " offset " << offset << " selector " << selector << " lane "
                << lane;
          }
          // Nothing outside the 64 destination lanes changes.
          for (int word = 0; word < 80; ++word) {
            if (word < offset || word >= offset + 64) {
              ASSERT_EQ(block[word], guard[word]) << "cpu " << cpu << " word " << word;
            }
          }
        }
      }
    }
  }
}
