// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/amd_ext_aql_packet.h"
#include "rocjitsu/vm/amdgpu/aql_packet_normalizer.h"

#include <gtest/gtest.h>

#include <cstring>
#include <limits>

namespace rocjitsu::amdgpu {
namespace {

template <typename Packet> Packet load_packet(const AqlPacketSlot &slot) {
  static_assert(sizeof(Packet) == sizeof(AqlPacketSlot));
  Packet packet{};
  std::memcpy(&packet, slot.bytes.data(), sizeof(packet));
  return packet;
}

uint16_t packet_header(hsa_packet_type_t type) {
  return static_cast<uint16_t>((type << HSA_PACKET_HEADER_TYPE) |
                               (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                               (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
}

AmdExtKernelDispatchPacket make_extended_packet() {
  AmdExtKernelDispatchPacket packet{};
  packet.header = packet_header(HSA_PACKET_TYPE_VENDOR_SPECIFIC);
  packet.amd_format = kHsaAmdPacketTypeExtKernelDispatch;
  packet.setup = 3;
  packet.workgroup_size_x = 8;
  packet.workgroup_size_y = 4;
  packet.workgroup_size_z = 2;
  packet.cluster_count_x = 5;
  packet.cluster_count_y = 3;
  packet.cluster_count_z = 2;
  packet.cluster_size_x = 1;
  packet.cluster_size_y = 1;
  packet.cluster_size_z = 1;
  packet.perf_hint = 0x5a;
  packet.private_segment_size = 96;
  packet.group_segment_size = 4096;
  packet.kernel_object = 0x123456789abcdef0ULL;
  packet.kernarg_address = reinterpret_cast<void *>(0x98765000ULL);
  packet.completion_signal.handle = 0x4444;
  return packet;
}

TEST(AqlPacketNormalizer, PreservesOrdinaryPacket) {
  hsa_kernel_dispatch_packet_t packet{};
  packet.header = packet_header(HSA_PACKET_TYPE_KERNEL_DISPATCH);
  packet.grid_size_x = 99;
  packet.kernel_object = 0x1234;

  const auto result = normalize_aql_packet(&packet, AqlExecutionBackend::Hardware);
  EXPECT_EQ(result.disposition, AqlPacketDisposition::Forwarded);
  ASSERT_EQ(result.packet_count, 1u);
  EXPECT_EQ(std::memcmp(result.packets[0].bytes.data(), &packet, sizeof(packet)), 0);
}

TEST(AqlPacketNormalizer, PreservesClusteredDispatchForSimulator) {
  auto packet = make_extended_packet();
  packet.cluster_size_x = 2;
  packet.cluster_size_y = 3;

  const auto result = normalize_aql_packet(&packet, AqlExecutionBackend::Simulator);
  EXPECT_EQ(result.disposition, AqlPacketDisposition::Forwarded);
  ASSERT_EQ(result.packet_count, 1u);
  EXPECT_EQ(std::memcmp(result.packets[0].bytes.data(), &packet, sizeof(packet)), 0);
}

TEST(AqlPacketNormalizer, ConvertsClusterSizeOneDispatchForHardware) {
  const auto packet = make_extended_packet();
  const AqlKernelResourceRequirements resources{.private_segment_size = 192,
                                                .group_segment_fixed_size = 4096};
  const auto result = normalize_aql_packet(&packet, AqlExecutionBackend::Hardware, resources);

  EXPECT_EQ(result.disposition, AqlPacketDisposition::NormalizedExtendedDispatch);
  ASSERT_EQ(result.packet_count, 1u);
  const auto dispatch = load_packet<hsa_kernel_dispatch_packet_t>(result.packets[0]);
  EXPECT_EQ((dispatch.header >> HSA_PACKET_HEADER_TYPE) & 0xffu, HSA_PACKET_TYPE_KERNEL_DISPATCH);
  EXPECT_EQ(dispatch.setup, packet.setup);
  EXPECT_EQ(dispatch.workgroup_size_x, packet.workgroup_size_x);
  EXPECT_EQ(dispatch.workgroup_size_y, packet.workgroup_size_y);
  EXPECT_EQ(dispatch.workgroup_size_z, packet.workgroup_size_z);
  EXPECT_EQ(dispatch.grid_size_x, 40u);
  EXPECT_EQ(dispatch.grid_size_y, 12u);
  EXPECT_EQ(dispatch.grid_size_z, 4u);
  EXPECT_EQ(dispatch.private_segment_size, resources.private_segment_size);
  EXPECT_EQ(dispatch.group_segment_size, packet.group_segment_size);
  EXPECT_EQ(dispatch.kernel_object, packet.kernel_object);
  EXPECT_EQ(dispatch.kernarg_address, packet.kernarg_address);
  EXPECT_EQ(dispatch.completion_signal.handle, packet.completion_signal.handle);
}

TEST(AqlPacketNormalizer, PreservesLargeFixedGroupSegmentRequestForHardwareValidation) {
  auto packet = make_extended_packet();
  packet.group_segment_size = 128u * 1024u;
  const AqlKernelResourceRequirements resources{.private_segment_size = 0,
                                                .group_segment_fixed_size = 128u * 1024u};

  const auto result = normalize_aql_packet(&packet, AqlExecutionBackend::Hardware, resources);

  ASSERT_EQ(result.packet_count, 1u);
  const auto dispatch = load_packet<hsa_kernel_dispatch_packet_t>(result.packets[0]);
  EXPECT_EQ(dispatch.group_segment_size, packet.group_segment_size);
}

TEST(AqlPacketNormalizer, PreservesDependencyWithBarrierPacket) {
  auto packet = make_extended_packet();
  packet.dep_signal.handle = 0x2222;

  const auto result = normalize_aql_packet(&packet, AqlExecutionBackend::Hardware);
  EXPECT_EQ(result.disposition, AqlPacketDisposition::NormalizedExtendedDispatch);
  ASSERT_EQ(result.packet_count, 2u);

  const auto barrier = load_packet<hsa_barrier_and_packet_t>(result.packets[0]);
  EXPECT_EQ((barrier.header >> HSA_PACKET_HEADER_TYPE) & 0xffu, HSA_PACKET_TYPE_BARRIER_AND);
  EXPECT_EQ(barrier.dep_signal[0].handle, packet.dep_signal.handle);
  EXPECT_EQ((barrier.header >> HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE) & 0x3u,
            HSA_FENCE_SCOPE_NONE);

  const auto dispatch = load_packet<hsa_kernel_dispatch_packet_t>(result.packets[1]);
  EXPECT_EQ((dispatch.header >> HSA_PACKET_HEADER_TYPE) & 0xffu, HSA_PACKET_TYPE_KERNEL_DISPATCH);
  EXPECT_NE(dispatch.header & (1u << HSA_PACKET_HEADER_BARRIER), 0u);
  EXPECT_EQ(dispatch.completion_signal.handle, packet.completion_signal.handle);
}

TEST(AqlPacketNormalizer, ReportsUnsupportedHardwareClusterAndKeepsRuntimeErrorPacket) {
  auto packet = make_extended_packet();
  packet.cluster_size_x = 2;

  const auto result = normalize_aql_packet(&packet, AqlExecutionBackend::Hardware);
  EXPECT_EQ(result.disposition, AqlPacketDisposition::UnsupportedClusterDispatch);
  ASSERT_EQ(result.packet_count, 1u);
  EXPECT_EQ(std::memcmp(result.packets[0].bytes.data(), &packet, sizeof(packet)), 0);
}

TEST(AqlPacketNormalizer, ReportsMalformedGridAndKeepsRuntimeErrorPacket) {
  auto packet = make_extended_packet();
  packet.cluster_count_x = std::numeric_limits<uint32_t>::max();
  packet.workgroup_size_x = 2;

  const auto result = normalize_aql_packet(&packet, AqlExecutionBackend::Hardware);
  EXPECT_EQ(result.disposition, AqlPacketDisposition::MalformedExtendedDispatch);
  ASSERT_EQ(result.packet_count, 1u);
  EXPECT_EQ(std::memcmp(result.packets[0].bytes.data(), &packet, sizeof(packet)), 0);
}

TEST(AqlPacketNormalizer, EnrichesOrdinaryHardwareDispatchFromExecutableResources) {
  hsa_kernel_dispatch_packet_t packet{};
  packet.header = packet_header(HSA_PACKET_TYPE_KERNEL_DISPATCH);
  packet.private_segment_size = 0;
  packet.group_segment_size = 1024;
  packet.kernel_object = 0x1234;

  const AqlKernelResourceRequirements resources{.private_segment_size = 140,
                                                .group_segment_fixed_size = 8192};
  const auto result = normalize_aql_packet(&packet, AqlExecutionBackend::Hardware, resources);
  EXPECT_EQ(result.disposition, AqlPacketDisposition::NormalizedKernelResources);
  ASSERT_EQ(result.packet_count, 1u);
  const auto dispatch = load_packet<hsa_kernel_dispatch_packet_t>(result.packets[0]);
  EXPECT_EQ(dispatch.private_segment_size, resources.private_segment_size);
  EXPECT_EQ(dispatch.group_segment_size, packet.group_segment_size);
  EXPECT_EQ(aql_packet_kernel_object(&packet), packet.kernel_object);
}

} // namespace
} // namespace rocjitsu::amdgpu
