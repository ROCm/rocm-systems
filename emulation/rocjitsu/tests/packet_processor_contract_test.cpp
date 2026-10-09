// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/packet_processor.h"

#include <gtest/gtest.h>

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

} // namespace
} // namespace rocjitsu::amdgpu
