// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file virtual_lds_dispatch_rewrite_test.cpp
/// @brief CPU-only tests for virtual-LDS AQL resource planning.

#include "rocjitsu/code/dbt/virtual_lds.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>

namespace rocjitsu {
namespace {

VirtualLdsDispatchPacketFields base_packet() {
  return VirtualLdsDispatchPacketFields{
      .group_segment_size = 71024,
      .private_segment_size = 40,
      .workgroup_size_x = 64,
      .workgroup_size_y = 1,
      .workgroup_size_z = 1,
      .grid_size_x = 256,
      .grid_size_y = 1,
      .grid_size_z = 1,
  };
}

VirtualLdsDispatchKernelFacts base_kernel() {
  return VirtualLdsDispatchKernelFacts{
      .static_lds_bytes = 70000,
      .normal_private_segment_size = 40,
      .virtual_private_segment_size = 96,
      .has_workgroup_id_x = true,
  };
}

TEST(VirtualLdsDispatchRewrite, KeepsDispatchWhoseTotalLdsFitsHost) {
  auto packet = base_packet();
  packet.group_segment_size = 64 * 1024;
  auto kernel = base_kernel();
  kernel.static_lds_bytes = 32 * 1024;

  const auto result = plan_virtual_lds_dispatch(packet, kernel, 64 * 1024);

  EXPECT_EQ(result.decision, VirtualLdsDispatchDecision::KeepNormal);
  EXPECT_EQ(result.requested_lds, 64u * 1024u);
  EXPECT_EQ(result.private_segment_size, packet.private_segment_size);
}

TEST(VirtualLdsDispatchRewrite, PreservesDynamicPrivateBytesWhenSwitchingDescriptors) {
  struct PrivateCase {
    uint32_t packet_private;
    uint32_t expected_virtual_private;
  };
  constexpr std::array cases = {
      PrivateCase{12, 96},   // An undersized packet is normalized to the normal fixed size.
      PrivateCase{40, 96},   // No dynamic request.
      PrivateCase{120, 176}, // Preserve 80 dynamic bytes above the normal fixed size.
  };

  for (const PrivateCase &test : cases) {
    SCOPED_TRACE(test.packet_private);
    auto packet = base_packet();
    packet.private_segment_size = test.packet_private;
    const auto result = plan_virtual_lds_dispatch(packet, base_kernel(), 64 * 1024);
    ASSERT_EQ(result.decision, VirtualLdsDispatchDecision::UseSidecar);
    EXPECT_EQ(result.private_segment_size, test.expected_virtual_private);
  }
}

TEST(VirtualLdsDispatchRewrite, ComputesDenseBackingAndSingletonRuntimeStrides) {
  const auto result = plan_virtual_lds_dispatch(base_packet(), base_kernel(), 64 * 1024);

  ASSERT_EQ(result.decision, VirtualLdsDispatchDecision::UseSidecar);
  EXPECT_EQ(result.geometry.groups_x, 4u);
  EXPECT_EQ(result.geometry.groups_y, 1u);
  EXPECT_EQ(result.geometry.groups_z, 1u);
  EXPECT_EQ(result.geometry.stride_x, 71024u);
  EXPECT_EQ(result.geometry.stride_y, 0u);
  EXPECT_EQ(result.geometry.stride_z, 0u);
  EXPECT_EQ(result.geometry.backing_bytes, 4u * 71024u);
}

TEST(VirtualLdsDispatchRewrite, ComputesThreeDimensionalDenseStrides) {
  auto packet = base_packet();
  packet.group_segment_size = 70000;
  packet.grid_size_x = 128;
  packet.grid_size_y = 3;
  packet.grid_size_z = 4;
  auto kernel = base_kernel();
  kernel.has_workgroup_id_y = true;
  kernel.has_workgroup_id_z = true;

  const auto result = plan_virtual_lds_dispatch(packet, kernel, 64 * 1024);

  ASSERT_EQ(result.decision, VirtualLdsDispatchDecision::UseSidecar);
  EXPECT_EQ(result.geometry.groups_x, 2u);
  EXPECT_EQ(result.geometry.groups_y, 3u);
  EXPECT_EQ(result.geometry.groups_z, 4u);
  EXPECT_EQ(result.geometry.stride_x, 70000u);
  EXPECT_EQ(result.geometry.stride_y, 140000u);
  EXPECT_EQ(result.geometry.stride_z, 420000u);
  EXPECT_EQ(result.geometry.backing_bytes, 1680000u);
}

TEST(VirtualLdsDispatchRewrite, HandlesMaximumGridWithoutCeilingDivisionOverflow) {
  auto packet = base_packet();
  packet.group_segment_size = 2;
  packet.grid_size_x = std::numeric_limits<uint32_t>::max();
  packet.workgroup_size_x = std::numeric_limits<uint16_t>::max();
  auto kernel = base_kernel();
  kernel.static_lds_bytes = 2;

  const auto result = plan_virtual_lds_dispatch(packet, kernel, 1);

  ASSERT_EQ(result.decision, VirtualLdsDispatchDecision::UseSidecar);
  EXPECT_EQ(result.geometry.groups_x, 65537u);
  EXPECT_EQ(result.geometry.backing_bytes, 131074u);
}

TEST(VirtualLdsDispatchRewrite, RejectsGeometryWhoseRuntimeStrideOverflows) {
  auto packet = base_packet();
  packet.grid_size_x = std::numeric_limits<uint32_t>::max();
  packet.workgroup_size_x = 2;

  const auto result = plan_virtual_lds_dispatch(packet, base_kernel(), 64 * 1024);

  EXPECT_EQ(result.decision, VirtualLdsDispatchDecision::Reject);
  EXPECT_NE(result.diagnostic.find("stride"), std::string::npos);
}

TEST(VirtualLdsDispatchRewrite, RejectsMissingWorkgroupIdForMultiGroupDimension) {
  auto kernel = base_kernel();
  kernel.has_workgroup_id_x = false;

  const auto result = plan_virtual_lds_dispatch(base_packet(), kernel, 64 * 1024);

  EXPECT_EQ(result.decision, VirtualLdsDispatchDecision::Reject);
  EXPECT_NE(result.diagnostic.find("workgroup-id"), std::string::npos);
}

TEST(VirtualLdsDispatchRewrite, RejectsZeroWorkgroupDimension) {
  auto packet = base_packet();
  packet.workgroup_size_x = 0;

  const auto result = plan_virtual_lds_dispatch(packet, base_kernel(), 64 * 1024);

  EXPECT_EQ(result.decision, VirtualLdsDispatchDecision::Reject);
  EXPECT_NE(result.diagnostic.find("zero workgroup"), std::string::npos);
}

TEST(VirtualLdsDispatchRewrite, RejectsPrivateSegmentOverflow) {
  auto packet = base_packet();
  packet.private_segment_size = std::numeric_limits<uint32_t>::max();

  const auto result = plan_virtual_lds_dispatch(packet, base_kernel(), 64 * 1024);

  EXPECT_EQ(result.decision, VirtualLdsDispatchDecision::Reject);
  EXPECT_NE(result.diagnostic.find("private-segment"), std::string::npos);
}

TEST(VirtualLdsDispatchRewrite, ZeroGridStillSelectsSidecarWithoutUsingIds) {
  auto packet = base_packet();
  packet.grid_size_x = 0;
  auto kernel = base_kernel();
  kernel.has_workgroup_id_x = false;

  const auto result = plan_virtual_lds_dispatch(packet, kernel, 64 * 1024);

  ASSERT_EQ(result.decision, VirtualLdsDispatchDecision::UseSidecar);
  EXPECT_EQ(result.geometry.groups_x, 0u);
  EXPECT_EQ(result.geometry.stride_x, 0u);
  EXPECT_EQ(result.geometry.backing_bytes, 71024u);
}

} // namespace
} // namespace rocjitsu
