// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>

namespace rocjitsu::test_support {

inline bool only_ds_address_changed(const std::array<uint32_t, 2> &source,
                                    const std::array<uint32_t, 2> &replacement, uint32_t address) {
  return source[0] == replacement[0] &&
         (source[1] & 0xffffff00u) == (replacement[1] & 0xffffff00u) &&
         (replacement[1] & 0xffu) == address;
}

inline void expect_same_ds_operand(const Operand *before, const Operand *after) {
  ASSERT_NE(before, nullptr);
  ASSERT_NE(after, nullptr);
  EXPECT_EQ(after->encoding_value(), before->encoding_value());
  EXPECT_EQ(after->size_bits(), before->size_bits());
  EXPECT_EQ(after->name(), before->name());
  EXPECT_EQ(after->decoded_vgpr(), before->decoded_vgpr());
  EXPECT_EQ(after->register_byte_mask(), before->register_byte_mask());
  EXPECT_EQ(after->is_fieldless(), before->is_fieldless());
  EXPECT_EQ(after->vgpr_msb_role(), before->vgpr_msb_role());
}

/// @brief Check decoded DS instructions before and after replacing an address selector.
inline void expect_ds_address_replacement(const Instruction &before, const Instruction &after,
                                          uint32_t source_address, uint32_t replacement_address) {
  EXPECT_EQ(after.mnemonic(), before.mnemonic());
  EXPECT_EQ(after.opcode(), before.opcode());
  EXPECT_EQ(before.size(), 8u);
  EXPECT_EQ(after.size(), 8u);
  ASSERT_EQ(after.num_src_operands(), before.num_src_operands());
  ASSERT_EQ(after.num_dst_operands(), before.num_dst_operands());
  ASSERT_NE(before.src_operand(0), nullptr);
  ASSERT_NE(after.src_operand(0), nullptr);
  const std::optional<RegisterRef> before_address = before.src_operand(0)->decoded_vgpr();
  const std::optional<RegisterRef> after_address = after.src_operand(0)->decoded_vgpr();
  ASSERT_TRUE(before_address);
  ASSERT_TRUE(after_address);
  EXPECT_EQ(before_address->cls, RegClass::VGPR);
  EXPECT_EQ(after_address->cls, RegClass::VGPR);
  EXPECT_EQ(before_address->index, source_address);
  EXPECT_EQ(after_address->index, replacement_address);
  EXPECT_EQ(before_address->width, 1u);
  EXPECT_EQ(after_address->width, 1u);
  EXPECT_EQ(before.src_operand(0)->encoding_value(), source_address);
  EXPECT_EQ(after.src_operand(0)->encoding_value(), replacement_address);
  EXPECT_EQ(after.src_operand(0)->size_bits(), before.src_operand(0)->size_bits());
  EXPECT_EQ(after.src_operand(0)->vgpr_msb_role(), before.src_operand(0)->vgpr_msb_role());
  for (int index = 1; index < before.num_src_operands(); ++index) {
    SCOPED_TRACE(index);
    ASSERT_NO_FATAL_FAILURE(
        expect_same_ds_operand(before.src_operand(index), after.src_operand(index)));
  }
  for (int index = 0; index < before.num_dst_operands(); ++index) {
    SCOPED_TRACE(index);
    ASSERT_NO_FATAL_FAILURE(
        expect_same_ds_operand(before.dst_operand(index), after.dst_operand(index)));
  }
}

} // namespace rocjitsu::test_support
