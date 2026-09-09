// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pm4_packet_processor.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu {
namespace {

TEST(PacketProcessorContractTest, AcceptsLegalCommonResults) {
  EXPECT_TRUE(valid_packet_process_result({.status = PacketProcessStatus::Complete,
                                           .retirement = PacketRetirement::Retire,
                                           .retirement_bytes = 4},
                                          4, 4));
  EXPECT_TRUE(valid_packet_process_result(
      {.status = PacketProcessStatus::NeedInput, .required_bytes = 12}, 4, 4));
  EXPECT_TRUE(valid_packet_process_result(
      {.status = PacketProcessStatus::Blocked, .retirement_bytes = 12}, 12, 4));
  EXPECT_TRUE(valid_packet_process_result({.status = PacketProcessStatus::Faulted,
                                           .retirement = PacketRetirement::Retire,
                                           .retirement_bytes = 12},
                                          12, 4));
  EXPECT_TRUE(valid_packet_process_result({.status = PacketProcessStatus::Complete,
                                           .retirement = PacketRetirement::Retire,
                                           .retirement_bytes = 28},
                                          20, 4));
}

TEST(PacketProcessorContractTest, RejectsContradictoryCommonResults) {
  EXPECT_FALSE(valid_packet_process_result(
      {.status = PacketProcessStatus::Complete, .retirement_bytes = 4}, 4, 4));
  EXPECT_FALSE(valid_packet_process_result({.status = PacketProcessStatus::NeedInput,
                                            .retirement = PacketRetirement::Retire,
                                            .retirement_bytes = 4,
                                            .required_bytes = 8},
                                           4, 4));
  EXPECT_FALSE(valid_packet_process_result(
      {.status = PacketProcessStatus::NeedInput, .required_bytes = 4}, 4, 4));
  EXPECT_FALSE(valid_packet_process_result(
      {.status = PacketProcessStatus::Malformed, .retirement = PacketRetirement::Retire}, 4, 4));
  EXPECT_FALSE(valid_packet_process_result(
      {.status = PacketProcessStatus::Blocked, .required_bytes = 8}, 4, 4));
}

TEST(Pm4PacketProcessorTest, RetiresSupportedOneDwordNop) {
  Pm4PacketProcessor processor;
  const std::array<uint32_t, 1> packet = {0xffff1000};

  const Pm4PacketProcessResult result = processor.process({.available_dwords = packet});

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Complete);
  EXPECT_EQ(result.packet.retirement, PacketRetirement::Retire);
  EXPECT_EQ(result.packet.retirement_bytes, sizeof(uint32_t));
}

TEST(Pm4PacketProcessorTest, RequestsTheCompleteRegisterPacketBeforeApplyingItsEffect) {
  uint32_t writes = 0;
  const Pm4PacketCallbacks callbacks{.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  }};
  Pm4PacketProcessor processor(callbacks);
  const std::array<uint32_t, 1> header = {0xc0017900};

  const Pm4PacketProcessResult result = processor.process({.available_dwords = header});

  EXPECT_EQ(result.packet.status, PacketProcessStatus::NeedInput);
  EXPECT_EQ(result.packet.retirement, PacketRetirement::Hold);
  EXPECT_EQ(result.packet.required_bytes, 3 * sizeof(uint32_t));
  EXPECT_EQ(writes, 0u);
}

TEST(Pm4PacketProcessorTest, AppliesSupportedRegisterWriteExactlyOnce) {
  uint32_t writes = 0;
  uint64_t register_dword = 0;
  uint32_t register_value = 0;
  const Pm4PacketCallbacks callbacks{.write_uconfig_register = [&](uint64_t reg, uint32_t value) {
    ++writes;
    register_dword = reg;
    register_value = value;
    return Pm4RegisterWriteStatus::Complete;
  }};
  Pm4PacketProcessor processor(callbacks);
  const std::array<uint32_t, 3> packet = {0xc0017900, 0x40, 0xdeadbeef};

  const Pm4PacketProcessResult result = processor.process({.available_dwords = packet});

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Complete);
  EXPECT_EQ(result.packet.retirement, PacketRetirement::Retire);
  EXPECT_EQ(result.packet.retirement_bytes, packet.size() * sizeof(uint32_t));
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(register_dword, 0xc040u);
  EXPECT_EQ(register_value, 0xdeadbeefu);
}

TEST(Pm4PacketProcessorTest, RejectsUnsupportedPacketWithoutRetiringIt) {
  Pm4PacketProcessor processor;
  const std::array<uint32_t, 1> packet = {0xc0002000};

  const Pm4PacketProcessResult result = processor.process({.available_dwords = packet});

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Unsupported);
  EXPECT_EQ(result.packet.retirement, PacketRetirement::Hold);
  EXPECT_EQ(result.diagnostic, Pm4PacketDiagnostic::UnsupportedPacket);
}

TEST(Pm4PacketProcessorTest, PreservesRetryableRegisterBackpressureWithoutRetirement) {
  uint32_t attempts = 0;
  const Pm4PacketCallbacks callbacks{.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++attempts;
    return Pm4RegisterWriteStatus::Blocked;
  }};
  Pm4PacketProcessor processor(callbacks);
  const std::array<uint32_t, 3> packet = {0xc0017900, 0x40, 0xdeadbeef};

  const Pm4PacketProcessResult result = processor.process({.available_dwords = packet});

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Blocked);
  EXPECT_EQ(result.packet.retirement, PacketRetirement::Hold);
  EXPECT_EQ(attempts, 1u);
}

} // namespace
} // namespace rocjitsu::amdgpu
