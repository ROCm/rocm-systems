// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for bitfield extraction/insertion and word alignment.

#include "integer/bitfield_test.h"
#include "rj_arith/rj_arith.h"

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
