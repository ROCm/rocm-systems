// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/aql_packet_processor.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_ext_aql_packet.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>

namespace rocjitsu::amdgpu {
namespace {

template <typename Packet>
std::array<std::byte, kAqlPacketBytes> packet_bytes(const Packet &packet) {
  static_assert(sizeof(Packet) == kAqlPacketBytes);
  std::array<std::byte, kAqlPacketBytes> bytes{};
  std::memcpy(bytes.data(), &packet, bytes.size());
  return bytes;
}

const GpuVmAccess &flat_access() {
  class Fixture {
  public:
    Fixture() {
      auto translator = std::make_shared<IdentityAddressSpaceTranslator>();
      auto backing = std::make_shared<GpuMemoryPhysicalAccess>(memory);
      const AddressSpaceHandle handle =
          vm.register_unrouted_address_space(23, std::move(translator), std::move(backing));
      access = vm.snapshot(handle);
    }

    GpuMemory memory{"memory"};
    GpuVm vm;
    std::optional<GpuVmAccess> access;
  };
  static Fixture fixture;
  return *fixture.access;
}

AqlPacketProcessRequest request_for(const std::array<std::byte, kAqlPacketBytes> &packet) {
  return {
      .packet = packet,
      .access = flat_access(),
      .registration_id = 17,
      .process_id = 23,
      .queue_id = 29,
      .ring_slot = 3,
      .packet_index = 37,
      .packet_address = 0x4000,
  };
}

TEST(AqlPacketProcessorTest, RetiresKernelDispatchOnlyAfterDurableAdmission) {
  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 2;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 4;
  packet.grid_size_z = 1;
  packet.kernel_object = 0x8000;
  packet.completion_signal.handle = 0x9000;
  const auto bytes = packet_bytes(packet);

  AqlAdmissionStatus admission_status = AqlAdmissionStatus::Blocked;
  uint32_t admissions = 0;
  AqlPreparedPacket admitted{};
  AqlPacketProcessor processor({
      .load_signal = {},
      .admit =
          [&](const AqlPacketProcessRequest &request, AqlPreparedPacket prepared) {
            EXPECT_EQ(request.registration_id, 17u);
            EXPECT_EQ(request.packet_index, 37u);
            ++admissions;
            admitted = prepared;
            return AqlAdmissionResult{.status = admission_status};
          },
  });

  const AqlPacketProcessResult blocked = processor.process(request_for(bytes));
  EXPECT_EQ(blocked.packet.status, PacketProcessStatus::Blocked);
  EXPECT_EQ(blocked.packet.retirement, PacketRetirement::Hold);
  EXPECT_EQ(blocked.packet.retirement_bytes, kAqlPacketBytes);
  EXPECT_EQ(blocked.blocked_reason, AqlBlockedReason::AdmissionUnavailable);
  EXPECT_EQ(admissions, 1u);

  admission_status = AqlAdmissionStatus::Complete;
  const AqlPacketProcessResult complete = processor.process(request_for(bytes));
  EXPECT_EQ(complete.packet.status, PacketProcessStatus::Complete);
  EXPECT_EQ(complete.packet.retirement, PacketRetirement::Retire);
  EXPECT_EQ(complete.packet.retirement_bytes, kAqlPacketBytes);
  EXPECT_EQ(admissions, 2u);
  EXPECT_EQ(admitted.kind, AqlPreparedPacketKind::KernelDispatch);
  EXPECT_EQ(admitted.kernel_dispatch.kernel_object, packet.kernel_object);
  EXPECT_EQ(admitted.kernel_dispatch.grid_size_x, packet.grid_size_x);
  EXPECT_EQ(admitted.kernel_dispatch.completion_signal.handle, packet.completion_signal.handle);
}

TEST(AqlPacketProcessorTest, PreservesTypedAdmissionFailureWithoutRetiringThePacket) {
  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  const auto bytes = packet_bytes(packet);
  AqlPacketProcessor processor({
      .load_signal = {},
      .admit =
          [](const AqlPacketProcessRequest &, AqlPreparedPacket) {
            return AqlAdmissionResult{
                .status = AqlAdmissionStatus::Unsupported,
                .diagnostic = AqlPacketDiagnostic::UnsupportedKernelWaveSize,
            };
          },
  });

  const AqlPacketProcessResult result = processor.process(request_for(bytes));

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Unsupported);
  EXPECT_EQ(result.packet.retirement, PacketRetirement::Hold);
  EXPECT_EQ(result.diagnostic, AqlPacketDiagnostic::UnsupportedKernelWaveSize);
}

TEST(AqlPacketProcessorTest, InvalidHeaderBlocksUntilTheProducerPublishesIt) {
  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_INVALID;
  const auto bytes = packet_bytes(packet);
  uint32_t admissions = 0;
  AqlPacketProcessor processor({
      .load_signal = {},
      .admit =
          [&](const AqlPacketProcessRequest &, AqlPreparedPacket) {
            ++admissions;
            return AqlAdmissionResult{.status = AqlAdmissionStatus::Complete};
          },
  });

  const AqlPacketProcessResult result = processor.process(request_for(bytes));

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Blocked);
  EXPECT_EQ(result.packet.retirement, PacketRetirement::Hold);
  EXPECT_EQ(result.packet.retirement_bytes, kAqlPacketBytes);
  EXPECT_EQ(result.packet.required_bytes, 0u);
  EXPECT_EQ(result.blocked_reason, AqlBlockedReason::HeaderInvalid);
  EXPECT_EQ(admissions, 0u);
}

TEST(AqlPacketProcessorTest, BarrierDependencyMustResolveBeforeAdmission) {
  hsa_barrier_and_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_BARRIER_AND | (1 << HSA_PACKET_HEADER_BARRIER);
  packet.dep_signal[0].handle = 0x1000;
  packet.completion_signal.handle = 0x2000;
  const auto bytes = packet_bytes(packet);

  int64_t dependency_value = 1;
  uint32_t admissions = 0;
  AqlPreparedPacket admitted{};
  AqlPacketProcessor processor({
      .load_signal =
          [&](const AqlPacketProcessRequest &request, uint64_t address) {
            EXPECT_EQ(address, 0x1008u);
            EXPECT_EQ(request.process_id, 23u);
            return AtomicLoadResult{.outcome = VmAccessOutcome::Complete,
                                    .value = static_cast<uint64_t>(dependency_value)};
          },
      .admit =
          [&](const AqlPacketProcessRequest &, AqlPreparedPacket prepared) {
            ++admissions;
            admitted = prepared;
            return AqlAdmissionResult{.status = AqlAdmissionStatus::Complete};
          },
  });

  const AqlPacketProcessResult blocked = processor.process(request_for(bytes));
  EXPECT_EQ(blocked.packet.status, PacketProcessStatus::Blocked);
  EXPECT_EQ(blocked.blocked_reason, AqlBlockedReason::DependencyUnsatisfied);
  EXPECT_EQ(admissions, 0u);

  dependency_value = 0;
  const AqlPacketProcessResult complete = processor.process(request_for(bytes));
  EXPECT_EQ(complete.packet.status, PacketProcessStatus::Complete);
  EXPECT_EQ(complete.packet.retirement, PacketRetirement::Retire);
  ASSERT_EQ(admissions, 1u);
  EXPECT_EQ(admitted.kind, AqlPreparedPacketKind::NonKernel);
  EXPECT_EQ(admitted.completion_signal, 0x2000u);
  EXPECT_TRUE(admitted.barrier_bit);
}

TEST(AqlPacketProcessorTest, DecodesExtendedDispatchIntoNormalizedKernelPacket) {
  AmdExtKernelDispatchPacket packet{};
  packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
  packet.amd_format = kHsaAmdPacketTypeExtKernelDispatch;
  packet.setup = 3;
  packet.workgroup_size_x = 32;
  packet.workgroup_size_y = 2;
  packet.workgroup_size_z = 1;
  packet.cluster_count_x = 4;
  packet.cluster_count_y = 3;
  packet.cluster_count_z = 2;
  packet.cluster_size_x = 2;
  packet.cluster_size_y = 1;
  packet.cluster_size_z = 1;
  packet.kernel_object = 0x7000;
  packet.kernarg_address = reinterpret_cast<void *>(0x7100);
  packet.completion_signal.handle = 0x7200;
  const auto bytes = packet_bytes(packet);

  AqlPreparedPacket admitted{};
  AqlPacketProcessor processor({
      .load_signal = {},
      .admit =
          [&](const AqlPacketProcessRequest &, AqlPreparedPacket prepared) {
            admitted = prepared;
            return AqlAdmissionResult{.status = AqlAdmissionStatus::Complete};
          },
  });

  const AqlPacketProcessResult result = processor.process(request_for(bytes));

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Complete);
  EXPECT_EQ(result.packet.retirement, PacketRetirement::Retire);
  EXPECT_EQ(admitted.kind, AqlPreparedPacketKind::KernelDispatch);
  EXPECT_EQ(admitted.kernel_dispatch.grid_size_x, 256u);
  EXPECT_EQ(admitted.kernel_dispatch.grid_size_y, 6u);
  EXPECT_EQ(admitted.kernel_dispatch.grid_size_z, 2u);
  EXPECT_EQ(admitted.kernel_dispatch.kernel_object, packet.kernel_object);
  EXPECT_EQ(admitted.cluster_shape.count_x, packet.cluster_count_x);
  EXPECT_EQ(admitted.cluster_shape.size_x, packet.cluster_size_x);
}

TEST(AqlPacketProcessorTest, ReportsUnsupportedVendorFormatWithoutAdmission) {
  AmdExtKernelDispatchPacket packet{};
  packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
  packet.amd_format = kHsaAmdPacketTypeReserved200;
  const auto bytes = packet_bytes(packet);
  uint32_t admissions = 0;
  AqlPacketProcessor processor({
      .load_signal = {},
      .admit =
          [&](const AqlPacketProcessRequest &, AqlPreparedPacket) {
            ++admissions;
            return AqlAdmissionResult{.status = AqlAdmissionStatus::Complete};
          },
  });

  const AqlPacketProcessResult result = processor.process(request_for(bytes));

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Unsupported);
  EXPECT_EQ(result.packet.retirement, PacketRetirement::Hold);
  EXPECT_EQ(result.diagnostic, AqlPacketDiagnostic::UnsupportedVendorFormat);
  EXPECT_EQ(admissions, 0u);
}

TEST(AqlPacketProcessorTest, ReportsUnsupportedBarrierConditionWithoutAdmission) {
  AmdBarrierValuePacket packet{};
  packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC | (1 << HSA_PACKET_HEADER_BARRIER);
  packet.amd_format = kHsaAmdPacketTypeBarrierValue;
  packet.signal.handle = 0x1000;
  packet.mask = std::numeric_limits<int64_t>::max();
  packet.condition = 99;
  const auto bytes = packet_bytes(packet);

  uint32_t admissions = 0;
  AqlPacketProcessor processor({
      .load_signal =
          [](const AqlPacketProcessRequest &, uint64_t address) {
            EXPECT_EQ(address, 0x1008u);
            return AtomicLoadResult{.outcome = VmAccessOutcome::Complete, .value = 1};
          },
      .admit =
          [&](const AqlPacketProcessRequest &, AqlPreparedPacket) {
            ++admissions;
            return AqlAdmissionResult{.status = AqlAdmissionStatus::Complete};
          },
  });

  const AqlPacketProcessResult result = processor.process(request_for(bytes));

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Unsupported);
  EXPECT_EQ(result.packet.retirement, PacketRetirement::Hold);
  EXPECT_EQ(result.diagnostic, AqlPacketDiagnostic::UnsupportedBarrierCondition);
  EXPECT_EQ(admissions, 0u);
}

TEST(AqlPacketProcessorTest, ReportsMalformedExtendedDispatchDimensionsWithoutThrowing) {
  AmdExtKernelDispatchPacket packet{};
  packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
  packet.amd_format = kHsaAmdPacketTypeExtKernelDispatch;
  packet.workgroup_size_x = 32;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.cluster_count_x = 1;
  packet.cluster_count_y = 1;
  packet.cluster_count_z = 1;
  packet.cluster_size_x = 0;
  packet.cluster_size_y = 1;
  packet.cluster_size_z = 1;

  uint32_t admissions = 0;
  AqlPacketProcessor processor({
      .load_signal = {},
      .admit =
          [&](const AqlPacketProcessRequest &, AqlPreparedPacket) {
            ++admissions;
            return AqlAdmissionResult{.status = AqlAdmissionStatus::Complete};
          },
  });

  auto bytes = packet_bytes(packet);
  AqlPacketProcessResult result{};
  EXPECT_NO_THROW(result = processor.process(request_for(bytes)));
  EXPECT_EQ(result.packet.status, PacketProcessStatus::Malformed);
  EXPECT_EQ(result.packet.retirement, PacketRetirement::Hold);
  EXPECT_EQ(result.diagnostic, AqlPacketDiagnostic::MalformedExtendedDispatchDimensions);
  EXPECT_EQ(admissions, 0u);

  packet.cluster_size_x = std::numeric_limits<decltype(packet.cluster_size_x)>::max();
  packet.cluster_count_x = std::numeric_limits<decltype(packet.cluster_count_x)>::max();
  packet.workgroup_size_x = std::numeric_limits<decltype(packet.workgroup_size_x)>::max();
  bytes = packet_bytes(packet);
  EXPECT_NO_THROW(result = processor.process(request_for(bytes)));
  EXPECT_EQ(result.packet.status, PacketProcessStatus::Malformed);
  EXPECT_EQ(result.diagnostic, AqlPacketDiagnostic::MalformedExtendedDispatchDimensions);
  EXPECT_EQ(admissions, 0u);
}

} // namespace
} // namespace rocjitsu::amdgpu
