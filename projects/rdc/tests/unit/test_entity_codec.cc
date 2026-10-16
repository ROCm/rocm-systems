/*
Copyright (c) 2025 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Pure-logic unit tests for the entity-index codec that SmiUtils relies on to
// decode a gpu_id into (device_index, instance_index, role). These require no
// hardware and run on any machine. The CPX handle-dispatch logic in
// get_processor_handle_from_id() branches on exactly these decoded fields, so
// exercising the codec directly locks down the contract that dispatch depends on.

#include <gtest/gtest.h>

#include <cstdint>

#include "rdc/rdc.h"
#include "rdc_lib/RdcEntityCodec.h"

namespace {

// A plain physical GPU (role PHYSICAL, instance 0) is what SmiUtils treats as a
// flat index into the GPU table. Encoding then decoding must be lossless.
TEST(EntityCodec, PhysicalRoundTrip) {
  for (uint32_t dev = 0; dev < RDC_MAX_NUM_DEVICES; dev++) {
    rdc_entity_info_t in{};
    in.device_index = dev;
    in.instance_index = 0;
    in.entity_role = RDC_DEVICE_ROLE_PHYSICAL;
    in.device_type = RDC_DEVICE_TYPE_GPU;

    rdc_entity_info_t out = rdc_get_info_from_entity_index(rdc_get_entity_index_from_info(in));
    EXPECT_EQ(out.device_index, dev);
    EXPECT_EQ(out.instance_index, 0u);
    EXPECT_EQ(out.entity_role, RDC_DEVICE_ROLE_PHYSICAL);
    EXPECT_EQ(out.device_type, RDC_DEVICE_TYPE_GPU);
  }
}

// Partition instances carry a socket in device_index and a per-socket proc in
// instance_index. This is the CPX case that diverges from the flat-index path.
TEST(EntityCodec, PartitionInstanceRoundTrip) {
  for (uint32_t sock = 0; sock < 8; sock++) {
    for (uint32_t part = 0; part < RDC_MAX_NUM_PARTITIONS; part++) {
      rdc_entity_info_t in{};
      in.device_index = sock;
      in.instance_index = part;
      in.entity_role = RDC_DEVICE_ROLE_PARTITION_INSTANCE;
      in.device_type = RDC_DEVICE_TYPE_GPU;

      rdc_entity_info_t out = rdc_get_info_from_entity_index(rdc_get_entity_index_from_info(in));
      EXPECT_EQ(out.device_index, sock);
      EXPECT_EQ(out.instance_index, part);
      EXPECT_EQ(out.entity_role, RDC_DEVICE_ROLE_PARTITION_INSTANCE);
      EXPECT_EQ(out.device_type, RDC_DEVICE_TYPE_GPU);
    }
  }
}

// A bare flat index (small integer, as passed by rdc_group_gpu_add and the CLI)
// must decode to PHYSICAL/instance-0 so it takes the flat-table branch, not the
// socket/proc branch. This is the exact predicate in get_processor_handle_from_id.
TEST(EntityCodec, BareIndexIsPhysicalInstanceZero) {
  for (uint32_t idx = 0; idx < 64; idx++) {
    rdc_entity_info_t info = rdc_get_info_from_entity_index(idx);
    EXPECT_EQ(info.device_index, idx);
    EXPECT_EQ(info.instance_index, 0u);
    EXPECT_EQ(info.entity_role, RDC_DEVICE_ROLE_PHYSICAL);
    EXPECT_EQ(info.device_type, RDC_DEVICE_TYPE_GPU);
  }
}

// The bitfield layout must not let a large field bleed into an adjacent one.
TEST(EntityCodec, FieldsDoNotOverlap) {
  rdc_entity_info_t in{};
  in.device_index = RDC_ENTITY_DEVICE_MASK;      // all device bits set
  in.instance_index = RDC_ENTITY_INSTANCE_MASK;  // all instance bits set
  in.entity_role = RDC_DEVICE_ROLE_PARTITION_INSTANCE;
  in.device_type = RDC_DEVICE_TYPE_CPU;

  rdc_entity_info_t out = rdc_get_info_from_entity_index(rdc_get_entity_index_from_info(in));
  EXPECT_EQ(out.device_index, RDC_ENTITY_DEVICE_MASK);
  EXPECT_EQ(out.instance_index, RDC_ENTITY_INSTANCE_MASK);
  EXPECT_EQ(out.entity_role, RDC_DEVICE_ROLE_PARTITION_INSTANCE);
  EXPECT_EQ(out.device_type, RDC_DEVICE_TYPE_CPU);
}

// Partition-string parsing feeds the same socket/partition indices. Spot-check
// the accept/reject contract since CPX enumeration surfaces "gS.P" strings.
TEST(EntityCodec, PartitionStringParsing) {
  uint32_t socket = 0, partition = 0;

  EXPECT_TRUE(rdc_is_partition_string("g0.0"));
  EXPECT_TRUE(rdc_parse_partition_string("g3.7", &socket, &partition));
  EXPECT_EQ(socket, 3u);
  EXPECT_EQ(partition, 7u);

  // Non-partition plain indices and malformed strings must be rejected.
  EXPECT_FALSE(rdc_is_partition_string("5"));
  EXPECT_FALSE(rdc_is_partition_string("g0"));
  EXPECT_FALSE(rdc_is_partition_string("gx.y"));
  EXPECT_FALSE(rdc_is_partition_string(""));
  EXPECT_FALSE(rdc_is_partition_string(nullptr));

  // Out-of-range partition index (>= RDC_MAX_NUM_PARTITIONS) must be rejected.
  EXPECT_FALSE(rdc_is_partition_string("g0.99"));
}

}  // namespace
