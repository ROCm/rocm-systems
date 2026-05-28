// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/mem_descriptor.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {
namespace {

TEST(MemDescriptor, ScalarCollapsesToSingleLane) {
  ScalarMemState s;
  s.addr = 0x1000;
  s.num_dwords = 4;
  s.mtype = Mtype::RW;
  s.is_load = true;
  auto d = mem_descriptor(&s);
  ASSERT_TRUE(d.has_value());
  ASSERT_EQ(d->per_lane_addr.size(), 1u);
  EXPECT_EQ(d->per_lane_addr[0], 0x1000u);
  EXPECT_EQ(d->lane_mask, 1u);
  EXPECT_EQ(d->elem_bytes, 16u);
  EXPECT_EQ(d->mtype, static_cast<uint8_t>(Mtype::RW));
  EXPECT_FALSE(d->non_temporal);
}

TEST(MemDescriptor, VectorPreservesSparseLaneMaskAndAddresses) {
  VectorMemState v(GLOBAL_MEM);
  v.lane_mask = 0b1010;
  v.per_lane_addr[1] = 0x2000;
  v.per_lane_addr[3] = 0x3000;
  v.elem_size = 4;
  v.num_elems = 1;
  auto d = mem_descriptor(&v);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->lane_mask, 0b1010u);
  ASSERT_EQ(d->per_lane_addr.size(), 64u);
  EXPECT_EQ(d->per_lane_addr[1], 0x2000u);
  EXPECT_EQ(d->per_lane_addr[3], 0x3000u);
  EXPECT_EQ(d->per_lane_addr[0], 0u);
  EXPECT_EQ(d->elem_bytes, 4u);
}

TEST(MemDescriptor, NonTemporalFlagSurvives) {
  VectorMemState v(GLOBAL_MEM);
  v.non_temporal = true;
  v.mtype = Mtype::NT;
  v.elem_size = 4;
  v.num_elems = 1;
  auto d = mem_descriptor(&v);
  ASSERT_TRUE(d.has_value());
  EXPECT_TRUE(d->non_temporal);
  EXPECT_EQ(d->mtype, static_cast<uint8_t>(Mtype::NT));
}

TEST(MemDescriptor, LocalMemDecodesAsVector) {
  VectorMemState v(LOCAL_MEM);
  v.lane_mask = 0x1;
  v.per_lane_addr[0] = 0x40;
  v.elem_size = 4;
  v.num_elems = 2;
  auto d = mem_descriptor(&v);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->per_lane_addr[0], 0x40u);
  EXPECT_EQ(d->elem_bytes, 8u);
}

TEST(MemDescriptor, NullStateReturnsNullopt) {
  EXPECT_FALSE(
      mem_descriptor(static_cast<const DynamicInstState *>(nullptr)).has_value());
}

}  // namespace
}  // namespace amdgpu
}  // namespace rocjitsu
