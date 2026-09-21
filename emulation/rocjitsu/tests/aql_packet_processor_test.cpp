// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/aql/aql_packet_processor.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_ext_aql_packet.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

class IdentityTranslator final : public AddressSpaceTranslator {
public:
  VmTranslationResult translate(uint64_t address, std::size_t size, VmAccessKind) const override {
    if (size == 0)
      return {.outcome = VmAccessOutcome::Malformed, .translation = {}};
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation = {.domain = VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = size,
                        .mtype = Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }
};

class NullPhysicalMemory final : public PhysicalMemoryAccess {
public:
  VmAccessOutcome read(VmMemoryDomain, uint64_t, std::span<std::byte> bytes) override {
    std::ranges::fill(bytes, std::byte{0});
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain, uint64_t, std::span<const std::byte>) override {
    return VmAccessOutcome::Complete;
  }
};

template <typename Packet>
std::array<std::byte, kAqlPacketBytes> packet_bytes(const Packet &packet) {
  static_assert(sizeof(Packet) == kAqlPacketBytes);
  std::array<std::byte, kAqlPacketBytes> bytes{};
  std::memcpy(bytes.data(), &packet, sizeof(packet));
  return bytes;
}

class AqlPacketProcessorTest : public ::testing::Test {
protected:
  AqlPacketProcessorTest() {
    auto translator = std::make_shared<IdentityTranslator>();
    auto memory = std::make_shared<NullPhysicalMemory>();
    handle_ = vm_.register_translated(1, std::move(translator), std::move(memory));
    access_ = vm_.snapshot(handle_);
  }

  AqlPacketProcessRequest request(const std::array<std::byte, kAqlPacketBytes> &packet) const {
    return {.packet = std::span<const std::byte, kAqlPacketBytes>(packet), .access = *access_};
  }

  GpuVm vm_;
  AddressSpaceHandle handle_;
  std::optional<GpuVmAccess> access_;
};

TEST_F(AqlPacketProcessorTest, RetriesBarrierAdmissionAndPreservesQueueOrdering) {
  ASSERT_TRUE(access_);
  hsa_barrier_and_packet_t barrier{};
  barrier.header = HSA_PACKET_TYPE_BARRIER_AND;
  barrier.completion_signal.handle = 0x2000;
  const auto bytes = packet_bytes(barrier);
  uint32_t attempts = 0;
  std::vector<AqlPreparedPacket> admitted;
  AqlPacketProcessor processor({
      .load_signal = {},
      .admit =
          [&](const AqlPacketProcessRequest &, AqlPreparedPacket prepared) {
            ++attempts;
            admitted.push_back(std::move(prepared));
            return AqlAdmissionResult{.status = attempts == 1 ? AqlAdmissionStatus::Blocked
                                                              : AqlAdmissionStatus::Complete};
          },
  });

  const AqlPacketProcessResult blocked = processor.process(request(bytes));
  const AqlPacketProcessResult complete = processor.process(request(bytes));

  EXPECT_EQ(blocked.packet.status, PacketProcessStatus::Blocked);
  EXPECT_EQ(blocked.blocked_reason, AqlBlockedReason::AdmissionUnavailable);
  EXPECT_EQ(complete.packet.status, PacketProcessStatus::Complete);
  ASSERT_EQ(admitted.size(), 2u);
  EXPECT_EQ(admitted[0].kind, AqlPreparedPacketKind::NonKernel);
  EXPECT_TRUE(admitted[0].blocks_following);
  EXPECT_EQ(admitted[0].completion_signal, 0x2000u);
  EXPECT_TRUE(admitted[1].blocks_following);
}

TEST_F(AqlPacketProcessorTest, DistinguishesBarrierValueFromPm4IbOrdering) {
  ASSERT_TRUE(access_);
  std::vector<AqlPreparedPacket> admitted;
  AqlPacketProcessor processor({
      .load_signal = {},
      .admit =
          [&](const AqlPacketProcessRequest &, AqlPreparedPacket prepared) {
            admitted.push_back(std::move(prepared));
            return AqlAdmissionResult{.status = AqlAdmissionStatus::Complete};
          },
  });

  AmdBarrierValuePacket barrier{};
  barrier.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
  barrier.amd_format = kHsaAmdPacketTypeBarrierValue;
  const auto barrier_bytes = packet_bytes(barrier);
  EXPECT_EQ(processor.process(request(barrier_bytes)).packet.status, PacketProcessStatus::Complete);

  AmdExtKernelDispatchPacket pm4_ib{};
  pm4_ib.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
  pm4_ib.amd_format = kAmdAqlFormatPm4Ib;
  const auto pm4_bytes = packet_bytes(pm4_ib);
  EXPECT_EQ(processor.process(request(pm4_bytes)).packet.status, PacketProcessStatus::Complete);

  ASSERT_EQ(admitted.size(), 2u);
  EXPECT_TRUE(admitted[0].blocks_following);
  EXPECT_FALSE(admitted[1].blocks_following);
}

TEST_F(AqlPacketProcessorTest, RejectsMalformedExtendedDispatchDimensionsBeforeAdmission) {
  ASSERT_TRUE(access_);
  AmdExtKernelDispatchPacket dispatch{};
  dispatch.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
  dispatch.amd_format = kHsaAmdPacketTypeExtKernelDispatch;
  dispatch.workgroup_size_x = 2;
  dispatch.workgroup_size_y = 1;
  dispatch.workgroup_size_z = 1;
  dispatch.cluster_count_x = std::numeric_limits<uint32_t>::max();
  dispatch.cluster_count_y = 1;
  dispatch.cluster_count_z = 1;
  dispatch.cluster_size_x = 2;
  dispatch.cluster_size_y = 1;
  dispatch.cluster_size_z = 1;
  const auto bytes = packet_bytes(dispatch);
  uint32_t admissions = 0;
  AqlPacketProcessor processor({
      .load_signal = {},
      .admit =
          [&](const AqlPacketProcessRequest &, AqlPreparedPacket) {
            ++admissions;
            return AqlAdmissionResult{.status = AqlAdmissionStatus::Complete};
          },
  });

  const AqlPacketProcessResult result = processor.process(request(bytes));

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Malformed);
  EXPECT_EQ(result.diagnostic, AqlPacketDiagnostic::MalformedExtendedDispatchDimensions);
  EXPECT_EQ(admissions, 0u);
}

TEST_F(AqlPacketProcessorTest, RejectsOverflowInEverySignalValueAddress) {
  ASSERT_TRUE(access_);
  constexpr uint64_t kOverflowingHandle = std::numeric_limits<uint64_t>::max() - 7;
  uint32_t signal_loads = 0;
  uint32_t admissions = 0;
  AqlPacketProcessor processor({
      .load_signal =
          [&](const AqlPacketProcessRequest &, uint64_t) {
            ++signal_loads;
            return AtomicLoadResult{.outcome = VmAccessOutcome::Complete};
          },
      .admit =
          [&](const AqlPacketProcessRequest &, AqlPreparedPacket) {
            ++admissions;
            return AqlAdmissionResult{.status = AqlAdmissionStatus::Complete};
          },
  });

  hsa_barrier_and_packet_t barrier{};
  barrier.header = HSA_PACKET_TYPE_BARRIER_AND;
  barrier.dep_signal[0].handle = kOverflowingHandle;
  const auto barrier_bytes = packet_bytes(barrier);
  const AqlPacketProcessResult barrier_result = processor.process(request(barrier_bytes));

  AmdBarrierValuePacket barrier_value{};
  barrier_value.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
  barrier_value.amd_format = kHsaAmdPacketTypeBarrierValue;
  barrier_value.signal.handle = kOverflowingHandle;
  const auto barrier_value_bytes = packet_bytes(barrier_value);
  const AqlPacketProcessResult barrier_value_result =
      processor.process(request(barrier_value_bytes));

  AmdExtKernelDispatchPacket dispatch{};
  dispatch.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
  dispatch.amd_format = kHsaAmdPacketTypeExtKernelDispatch;
  dispatch.dep_signal.handle = kOverflowingHandle;
  const auto dispatch_bytes = packet_bytes(dispatch);
  const AqlPacketProcessResult dispatch_result = processor.process(request(dispatch_bytes));

  for (const AqlPacketProcessResult *result :
       {&barrier_result, &barrier_value_result, &dispatch_result}) {
    EXPECT_EQ(result->packet.status, PacketProcessStatus::Malformed);
    EXPECT_EQ(result->diagnostic, AqlPacketDiagnostic::InvalidSignalAddress);
  }
  EXPECT_EQ(signal_loads, 0u);
  EXPECT_EQ(admissions, 0u);
}

} // namespace
} // namespace rocjitsu::amdgpu
